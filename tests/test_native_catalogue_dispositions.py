"""Every pinned native class/role/type has an honest catalogue disposition."""

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import (
    RegistryError,
    load_registry,
    metadata_digest,
)
from ruinedfooocus.features.models.native_catalogue import dispositions
from tools.audit_native_model_coverage import assignments, derive_sources, folder_role_declarations
from test_image_chat_resource_scope import (
    REMOVED_NATIVE_CLASSES,
    REMOVED_CLIP_TYPES,
    RETAINED_TE_MODELS,
    REMOVED_FOLDER_ROLES,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
GGUF = Path(os.environ.get("RF_TEST_GGUF_SOURCE", ROOT / "repositories/molbal_comfyui_gguf"))


class NativeCatalogueTests(unittest.TestCase):
    def test_denominators_match_exact_pinned_native_registrations_without_imports(self):
        registry = load_registry()
        source = ast.parse((NATIVE / "comfy/supported_models.py").read_text(encoding="utf-8"))
        classes = {item.id for item in assignments(source.body)["models"].elts}
        rows = registry.native_dispositions()
        self.assertEqual({row["id"] for row in rows}, classes - REMOVED_NATIVE_CLASSES)
        self.assertEqual(len(rows), 62)
        source = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        enum = next(
            item
            for item in source.body
            if isinstance(item, ast.ClassDef) and item.name == "CLIPType"
        )
        self.assertEqual(
            {row["id"] for row in registry.native_dispositions(kind="clip_types")},
            set(assignments(enum.body)) - REMOVED_CLIP_TYPES,
        )
        native_roles = folder_role_declarations(
            {
                "native/folder_paths.py": ast.parse(
                    (NATIVE / "folder_paths.py").read_text(encoding="utf-8")
                ),
                "gguf/nodes.py": ast.parse((GGUF / "nodes.py").read_text(encoding="utf-8")),
            }
        )
        self.assertEqual(
            {row["id"] for row in registry.native_dispositions(kind="roles")},
            {row["role"] for row in native_roles} - REMOVED_FOLDER_ROLES,
        )
        self.assertEqual(len(native_roles), 32)
        index = json.loads(
            resource_path("model_metadata", "native_coverage", "index.json").read_text(
                encoding="utf-8"
            )
        )
        bootstrap = ast.parse((ROOT / "ruinedfooocus/app/bootstrap.py").read_text(encoding="utf-8"))
        repositories = ast.literal_eval(assignments(bootstrap.body)["git_repos"])
        self.assertEqual(
            index["source"]["native_pin"],
            next(row["hash"] for row in repositories if row["name"] == "ComfyUI"),
        )

    def test_image_rows_exclude_other_media_and_never_claim_installed_or_qualified(self):
        registry = load_registry()
        image = registry.native_dispositions(media="image")
        self.assertEqual(len(image), 56)
        self.assertFalse({"LTXAV", "StableAudio", "Trellis2"} & {row["id"] for row in image})
        for row in registry.native_dispositions():
            self.assertEqual(row["installed_status"], "runtime-inventory-not-examined")
            self.assertEqual(row["qualification"], "not-inherited-from-source-support")
            self.assertTrue(row["reason"])
            self.assertTrue(row["download_reason"])
            self.assertIn("989d93d0d968e704c589ce0146bb18c77f6b50cc", row["native_source_url"])
            if not row["available_artifact_ids"] and row["media"] in ("image", "utility"):
                self.assertIn(
                    row["download_disposition"],
                    (
                        "no-reviewed-exact-checkpoint-tuple",
                        "publisher-access-gated",
                        "native-source-conversion-gap",
                        "native-sharded-source-ingestion-gap",
                    ),
                )
                self.assertIn(row["id"], row["download_reason"])

    def test_registered_payloads_keep_variants_and_prefer_smaller_published_quantization(self):
        registry = load_registry()
        rows = {row["id"]: row for row in registry.native_dispositions()}
        choices = {
            item["behavioral_variant"]: item["artifact_ids"]
            for item in rows["QwenImage21"]["artifact_choices"]
        }
        self.assertEqual(set(choices), {"base", "turbo"})
        self.assertEqual(len(choices["base"]), 2)
        self.assertIn("Q4_K_M.gguf", choices["base"][0])
        for variant, identities in choices.items():
            for identity in identities:
                self.assertEqual(registry.artifact(identity)["behavioral_variant"], variant)
        self.assertEqual(rows["Stable_Cascade_B"]["operation"], "stage-b-companion")
        self.assertEqual(rows["SAM31"]["app_route"], "specialized-source-route")
        self.assertEqual(rows["DepthAnything3"]["download_disposition"], "registered-artifacts")
        with self.assertRaises(TypeError):
            rows["QwenImage21"]["artifact_choices"][0]["selection_basis"] = "poisoned"

    def test_native_lookup_never_opens_qualification_history(self):
        registry = load_registry()
        original = Path.read_text

        def read(path, *args, **kwargs):
            self.assertNotIn("qualification", path.parts)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            self.assertEqual(len(registry.native_dispositions()), 62)

    def test_detected_text_encoder_models_are_separate_from_logical_clip_types(self):
        registry = load_registry()
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        enum = next(
            item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "TEModel"
        )
        native = {key: ast.literal_eval(value) for key, value in assignments(enum.body).items()}
        rows = {row["id"]: row for row in registry.native_dispositions(kind="te_models")}
        self.assertEqual(
            {key: row["native_value"] for key, row in rows.items()},
            {key: value for key, value in native.items() if key in RETAINED_TE_MODELS},
        )
        self.assertEqual(len(rows), 26)
        self.assertEqual(len(registry.native_dispositions(kind="clip_types")), 28)
        for key in ("WEVISONE_2B", "WEVISONE_4B"):
            self.assertEqual(rows[key]["media"], "utility")
            self.assertEqual(rows[key]["native_role"], "image-utility-text-and-mask-decoder")
            self.assertEqual(rows[key]["app_route"], "native-wevisone-image-utility")
            self.assertEqual(rows[key]["native_execution_node"], "WeVisOneDetect")
            self.assertEqual(rows[key]["component_roles"]["wevisone-clip"], key)
            self.assertEqual(rows[key]["evidence"]["ui_api"], "guarded-cpu-callers")
            self.assertEqual(rows[key]["evidence"]["real_runtime"], "pending")
            self.assertEqual(rows[key]["evidence"]["visual_quality"], "pending")
            expected = (
                ("existing:clip/wevisone_vl_2b_int8_convrot.safetensors",)
                if key == "WEVISONE_2B"
                else ()
            )
            self.assertEqual(rows[key]["available_artifact_ids"], expected)
            for identity in expected:
                artifact = registry.artifact(identity)
                self.assertEqual(artifact["native_text_type"], key)
                self.assertEqual(artifact["component_role"], "wevisone-clip")
                self.assertFalse(artifact["chat_weight"] or artifact["auto_recipe_supported"])
        self.assertNotIn("QWEN3VL_2B", rows)

    def test_image_configuration_records_connected_pdd_graph_without_hardware_claim(self):
        rows = {row["id"]: row for row in load_registry().native_dispositions()}
        row = rows["MiniMaxH3Image"]
        self.assertEqual(row["app_route"], "ImagePipeline MiniMaxH3Image")
        self.assertEqual(row["native_execution_node"], "MiniMaxH3ImageConditioning")
        self.assertEqual((row["native_layout"]["channels"], row["native_layout"]["rank"]), (24, 5))
        self.assertEqual(row["native_layout"]["spatial_scale"], 16)
        self.assertEqual(len(row["available_artifact_ids"]), 1)
        self.assertEqual(len(row["artifact_choices"]), 1)
        self.assertNotIn("MiniMaxH3", rows)

    def test_refreshed_configuration_provenance_matches_the_actual_native_contracts(self):
        paths = {
            "native/supported_models.py": NATIVE / "comfy/supported_models.py",
            "native/supported_models_base.py": NATIVE / "comfy/supported_models_base.py",
            "native/sd.py": NATIVE / "comfy/sd.py",
            "native/utils.py": NATIVE / "comfy/utils.py",
            "rf/image_components.py": ROOT / "ruinedfooocus/features/models/image_components.py",
            "rf/image_pipeline.py": ROOT / "ruinedfooocus/features/generation/image_pipeline.py",
            "rf/model_architecture.py": ROOT
            / "ruinedfooocus/features/models/model_architecture.py",
            "rf/model_handler.py": ROOT / "ruinedfooocus/features/models/model_handler.py",
            "rf/model_formats.py": ROOT / "ruinedfooocus/core/model_formats.py",
        }
        source = derive_sources({key: path.read_bytes() for key, path in paths.items()})
        expected = {row["class"]: row for row in source["native_config_classes"]}
        for row in load_registry().native_dispositions():
            with self.subTest(native_class=row["id"]):
                self.assertEqual(
                    row["native_contract_sha256"], expected[row["id"]]["contract_sha256"]
                )
                self.assertTrue(
                    row["native_source_url"].endswith(f"#L{expected[row['id']]['line']}")
                )

    def test_valid_digest_invalid_references_and_escaping_paths_raise_registry_error(self):
        example = json.loads(
            resource_path(
                "model_metadata", "native_coverage", "configurations-image.json"
            ).read_text(encoding="utf-8")
        )["rows"][0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native = root / "native_coverage"
            native.mkdir()
            example["available_artifact_ids"] = ["existing:missing"]
            shard = dict(schema_version=1, kind="configurations", rows=[example])
            (native / "rows.json").write_text(json.dumps(shard), encoding="utf-8")
            index = dict(
                schema_version=1,
                groups={"configurations": ["rows.json"]},
                shards={
                    "rows.json": dict(kind="configurations", rows=1, sha256=metadata_digest(shard))
                },
            )
            (native / "index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "Unknown native catalogue artifact"):
                dispositions(load_registry(), root)
            index["groups"]["configurations"] = ["../outside.json"]
            index["shards"]["../outside.json"] = index["shards"]["rows.json"]
            (native / "index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "escapes"):
                dispositions(load_registry(), root)


if __name__ == "__main__":
    unittest.main()
