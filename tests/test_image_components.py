from ruinedfooocus.core.paths import asset_path, resource_path

"""Image component status/download policy without model loading or network access."""
import ast
import builtins
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
import ruinedfooocus.app
from unittest.mock import Mock
from unittest.mock import patch

from ruinedfooocus.features.models.image_components import (
    IMAGE_COMPONENTS,
    inventory,
    download_missing,
)
from ruinedfooocus.features.models.model_identity import file_signature

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path, entries=None):
    raw = json.dumps(entries or {"weight": {"shape": [1]}}).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)
    return path


class ImageComponentTests(unittest.TestCase):
    def test_native_config_aliases_keep_existing_runtime_component_contracts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / "local.safetensors")
            for label in ("SSD1B", "KOALA_700M", "KOALA_1B", "Segmind_Vega"):
                with self.subTest(label=label):
                    state = inventory(model, label, {}, paths)
                    self.assertEqual(state["architecture"], "SDXL")
                    self.assertEqual(
                        [row["key"] for row in state["rows"]], ["clip_l", "clip_g", "vae_sdxl"]
                    )
            state = inventory(model, "CosmosT2IPredict2", {}, paths)
            self.assertEqual(state["architecture"], "CosmosPredict2")
            self.assertEqual([row["key"] for row in state["rows"]], ["clip_oldt5", "vae_wan"])
            # A native detection result is authoritative over stale cached family text.
            with (
                patch.dict(sys.modules, {"comfy.model_detection": SimpleNamespace()}),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_tensor_model",
                    return_value=dict(
                        architecture="KOALA_1B", text_encoder_prefixes=(), vae_prefixes=()
                    ),
                ),
            ):
                state = inventory(model, "Unknown", {}, paths)
            self.assertEqual(state["architecture"], "SDXL")
            paths.download_file.assert_not_called()

    def test_cached_flux3_klein_alias_uses_native_slots_without_backend_import(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(
                root / "arbitrary.safetensors",
                {
                    "content_mode_blocks.image.0.attn_out.weight": {"shape": [3072, 3072]},
                    "single_stream_modulations.txt.lin.weight": {"shape": [9216, 3072]},
                    "ref_time_in.in_layer.weight": {"shape": [3072, 256]},
                },
            )
            original_import = builtins.__import__

            def local_import(name, *args, **kwargs):
                if name.split(".")[0] in ("torch", "comfy", "comfy_extras"):
                    raise AssertionError("Passive inventory must not initialize native modules")
                return original_import(name, *args, **kwargs)

            with patch.dict(sys.modules), patch("builtins.__import__", side_effect=local_import):
                sys.modules.pop("comfy.model_detection", None)
                state = inventory(model, "Flux.3 Klein", {}, paths)
            self.assertEqual(state["architecture"], "Flux3")
            self.assertEqual(
                [row["key"] for row in state["rows"]], ["clip_qwen3vl_4b", "vae_flux2"]
            )
            paths.download_file.assert_not_called()

    def test_request_local_component_choices_match_native_precedence_and_automatic(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / "image.safetensors")
            defaults = {"clip_qwen3vl_4b": "legacy.safetensors", "vae_mage_flow": "legacy-vae"}
            state = inventory(
                model,
                "MageFlow",
                defaults,
                paths,
                component_overrides={
                    "clip_qwen3vl_4b": "selected.safetensors",
                    "vae_mage_flow": None,
                },
            )
            self.assertEqual(
                [row["filename"] for row in state["rows"]],
                ["selected.safetensors", "mage_flow_vae_bf16.safetensors"],
            )
            self.assertEqual(defaults["vae_mage_flow"], "legacy-vae")
            checkpoint(model, {"cond_stage_model.weight": {"shape": [1]}})
            with self.assertRaisesRegex(ValueError, "built into this checkpoint"):
                inventory(
                    model,
                    "MageFlow",
                    defaults,
                    paths,
                    component_overrides={"clip_qwen3vl_4b": "selected.safetensors"},
                )
            state = inventory(
                model,
                "MageFlow",
                defaults,
                paths,
                component_overrides={"clip_qwen3vl_4b": None},
            )
            self.assertEqual(state["rows"][0]["status"], "Embedded in checkpoint")
            with self.assertRaisesRegex(ValueError, "not a slot"):
                inventory(model, "MageFlow", defaults, paths, component_overrides={"clip_l": None})
            paths.download_file.assert_not_called()

    def test_required_payload_totals_exclude_embedded_components_and_keep_manual_unknowns(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / "image.safetensors")
            registry = SimpleNamespace(
                annotations_for=lambda key, entry: {
                    "id": key,
                    "artifact": {"sha256": "same-payload", "bytes": 4},
                }
            )
            with patch(
                "ruinedfooocus.features.models.image_components.load_registry",
                return_value=registry,
            ):
                state = inventory(model, "MageFlow", {}, paths)
                self.assertEqual(state["missing_payload_bytes"], 4)
                self.assertEqual([row["published_bytes"] for row in state["rows"]], [4, 4])
                state = inventory(
                    model, "MageFlow", {"clip_qwen3vl_4b": "custom.safetensors"}, paths
                )
                self.assertIsNone(state["missing_payload_bytes"])
                checkpoint(
                    model,
                    {
                        "cond_stage_model.weight": {"shape": [1]},
                        "first_stage_model.weight": {"shape": [1]},
                    },
                )
                state = inventory(model, "MageFlow", {}, paths)
                self.assertEqual(state["missing_payload_bytes"], 0)
            paths.download_file.assert_not_called()

    def test_friendly_cached_moss_name_resolves_with_real_header_shapes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(
                root / "moss.safetensors",
                {
                    "shared_block_modulation.weight": {"shape": [18432, 3072]},
                    "layers.0.proj.ffn_up_gate.weight": {"shape": [16384, 3072]},
                },
            )
            cache = root / "cache"
            cache.mkdir()
            (cache / "moss.json").write_text(json.dumps({"baseModel": "MOSS Image", "id": 123}))
            models = SimpleNamespace(
                get_names=lambda category: ["moss.safetensors"],
                get_file_from_name=lambda category, name: model,
                cache_paths={"checkpoints": cache},
            )
            # Execute the real cache-selection function without importing Gradio/shared runtime.
            tree = ast.parse(
                (ROOT / "ruinedfooocus/features/models/ui_model_components.py").read_text(
                    encoding="utf-8"
                )
            )
            selected = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "selected"
            )
            namespace = dict(
                asset_path=asset_path,
                resource_path=resource_path,
                Path=Path,
                json=json,
                file_signature=file_signature,
            )
            shared = SimpleNamespace(models=models)
            with (
                patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
                patch.object(ruinedfooocus.app, "state", shared, create=True),
            ):
                exec(
                    compile(
                        ast.Module(body=[selected], type_ignores=[]), "component_selection", "exec"
                    ),
                    namespace,
                )
                path, family = namespace["selected"]("moss.safetensors")
            self.assertEqual(family, "MOSS Image")
            state = inventory(path, family, {}, paths)
            self.assertEqual(state["architecture"], "MOSSImage")
            self.assertEqual([row["key"] for row in state["rows"]], ["clip_qwen35_9b", "vae_flux2"])
            # An already-initialized backend's shape detection wins over stale/friendly metadata.
            with (
                patch.dict("sys.modules", {"comfy.model_detection": SimpleNamespace()}),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_tensor_model",
                    return_value=dict(
                        architecture="MOSSImage", text_encoder_prefixes=(), vae_prefixes=()
                    ),
                ) as detect,
            ):
                self.assertEqual(
                    inventory(path, "Qwen Image", {}, paths)["architecture"], "MOSSImage"
                )
            detect.assert_called_once()
            paths.download_file.assert_not_called()

    def test_friendly_image_family_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / "image.safetensors")
            for family, canonical in [
                ("Qwen Image", "QwenImage"),
                ("Mage-Flow", "MageFlow"),
                ("MOSS Image", "MOSSImage"),
                ("Z-Image Turbo", "ZImage"),
                ("Krea 2", "Krea2"),
            ]:
                with self.subTest(family=family):
                    self.assertEqual(inventory(model, family, {}, paths)["architecture"], canonical)

    def test_registry_matches_every_production_image_architecture(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        table = next(
            node.value
            for node in cls.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "known_model_info"
                for target in node.targets
            )
        )
        architectures = set()
        for name, values in zip(table.keys, table.values):
            architectures.add(ast.literal_eval(name))
            keys = {ast.literal_eval(key) for key in values.keys}
            self.assertFalse(
                {"clip_names", "vae_name"} & keys, "Component declarations must have one source"
            )
        self.assertEqual(set(IMAGE_COMPONENTS), architectures)
        resolver = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_clip_and_vae"
        )
        updates = [
            node
            for node in ast.walk(resolver)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "info.update"
        ]
        self.assertTrue(
            any(ast.unparse(node.args[0]) == "IMAGE_COMPONENTS[unet_type]" for node in updates)
        )

    def paths(self, root):
        for folder in ("clip", "vae"):
            (root / folder).mkdir()
        catalog = {
            "clip/qwen3vl_4b_bf16.safetensors": {
                "filename": "qwen3vl_4b_bf16.safetensors",
                "path": "path_clip",
                "url": "https://example.invalid/encoder",
            },
            "vae/mage_flow_vae_bf16.safetensors": {
                "filename": "mage_flow_vae_bf16.safetensors",
                "path": "path_vae",
                "url": "https://example.invalid/vae",
            },
            "checkpoints/video": {
                "filename": "video.safetensors",
                "path": "path_checkpoints",
                "url": "https://example.invalid/video",
            },
        }
        return SimpleNamespace(
            paths={"path_clip": str(root / "clip"), "path_vae": str(root / "vae")},
            get_abspath=Path,
            DOWNLOADABLE_FILES=catalog,
            download_file=Mock(),
        )

    def test_status_and_explicit_download_only_selected_missing_components(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(root / "image.safetensors")
            (root / "vae/mage_flow_vae_bf16.safetensors").write_bytes(b"installed")
            state = inventory(model, "MageFlow", {}, paths)
            self.assertEqual([row["status"] for row in state["rows"]], ["Missing", "Installed"])
            paths.download_file.assert_not_called()
            self.assertEqual(
                download_missing(model, "MageFlow", {}, paths), ["qwen3vl_4b_bf16.safetensors"]
            )
            paths.download_file.assert_called_once()
            self.assertEqual(
                paths.download_file.call_args.args[0], "clip/qwen3vl_4b_bf16.safetensors"
            )
            paths.download_file.reset_mock()
            with self.assertRaisesRegex(ValueError, "Offline"):
                download_missing(model, "MageFlow", {}, paths, offline=True)
            paths.download_file.assert_not_called()

    def test_embedded_components_and_custom_overrides_do_not_trigger_downloads(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(
                root / "image.safetensors",
                {
                    "cond_stage_model.weight": {"shape": [1]},
                    "first_stage_model.weight": {"shape": [1]},
                },
            )
            rows = inventory(model, "MageFlow", {}, paths)["rows"]
            self.assertTrue(all(row["status"] == "Embedded in checkpoint" for row in rows))
            self.assertEqual(download_missing(model, "MageFlow", {}, paths), [])
            external = root / "outside.safetensors"
            external.write_bytes(b"unused external override")
            overrides = {"clip_qwen3vl_4b": str(external), "vae_mage_flow": str(external)}
            rows = inventory(model, "MageFlow", overrides, paths)["rows"]
            for row in rows:
                self.assertEqual(row["filename"], "Built into checkpoint")
                self.assertEqual(row["location"], str(model))
                self.assertFalse(row["override"])
                self.assertIsNone(row["bytes"])
                self.assertIsNone(row["published_bytes"])
                self.assertFalse(row["download_available"])
            self.assertEqual(overrides["clip_qwen3vl_4b"], str(external))
            checkpoint(model)
            rows = inventory(
                model, "MageFlow", {"clip_qwen3vl_4b": "../outside.safetensors"}, paths
            )["rows"]
            self.assertTrue(rows[0]["override"])
            self.assertFalse(rows[0]["download_available"])

    def test_video_headers_and_video_families_are_refused_even_with_image_label(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = checkpoint(
                root / "video.safetensors",
                {"head.modulation": {"shape": [1]}, "patch_embedding.weight": {"shape": [1]}},
            )
            with self.assertRaisesRegex(ValueError, "image checkpoints only"):
                download_missing(model, "MageFlow", {}, paths)
            checkpoint(model)
            with self.assertRaisesRegex(ValueError, "image checkpoints only"):
                download_missing(model, "Wan Video", {}, paths)
            paths.download_file.assert_not_called()

    def test_gguf_video_metadata_is_refused_without_tensor_runtime(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = root / "video.gguf"
            model.write_bytes(b"GGUF")
            with patch(
                "ruinedfooocus.features.models.gguf_metadata.read_gguf_header",
                return_value=({}, "wan"),
            ):
                with self.assertRaisesRegex(ValueError, "image checkpoints only"):
                    download_missing(model, "MageFlow", {}, paths)
            paths.download_file.assert_not_called()

    def test_gguf_header_identity_overrides_stale_image_family(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            paths = self.paths(root)
            model = root / "image.gguf"
            model.write_bytes(b"GGUF")
            header = {"shared_block_modulation.weight": {"shape": [18432, 3072]}}
            with patch(
                "ruinedfooocus.features.models.gguf_metadata.read_gguf_header",
                return_value=(header, "moss_image"),
            ):
                state = inventory(model, "MageFlow", {}, paths)
            self.assertEqual(state["architecture"], "MOSSImage")
            self.assertEqual(state["rows"][0]["key"], "clip_qwen35_9b")
            # Initialized backend shapes are authoritative even over an inaccurate GGUF tag.
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "comfy.model_detection": SimpleNamespace(),
                    },
                ),
                patch(
                    "ruinedfooocus.features.models.gguf_metadata.read_gguf_header",
                    return_value=(header, "moss_image"),
                ),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_tensor_model",
                    return_value=dict(
                        architecture="QwenImage", text_encoder_prefixes=(), vae_prefixes=()
                    ),
                ) as detect,
            ):
                self.assertEqual(
                    inventory(model, "MageFlow", {}, paths)["architecture"], "QwenImage"
                )
            self.assertEqual(
                detect.call_args.args[0]["shared_block_modulation.weight"]["shape"], [18432, 3072]
            )
            with patch(
                "ruinedfooocus.features.models.gguf_metadata.read_gguf_header",
                return_value=(header, "lumina2"),
            ):
                with self.assertRaisesRegex(ValueError, "cannot be identified safely"):
                    download_missing(model, "MageFlow", {}, paths)
            paths.download_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
