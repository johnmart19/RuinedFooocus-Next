"""Source/task decisions, trusted content and UI state are independent of GPUs."""

import hashlib
import ast
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models import model_capabilities as capabilities
from ruinedfooocus.features.models.model_registry import load_registry, RegistryError


class ModelCapabilityTests(unittest.TestCase):
    def test_additive_native_source_review_preserves_old_artifact_and_task_limits(self):
        registry = load_registry()
        old = capabilities.RuntimeIdentity("llama.cpp", "b12110")
        for build in ("b12327", "b12366"):
            reviewed = capabilities.RuntimeIdentity("llama.cpp", build)
            unknown = capabilities.RuntimeIdentity("llama.cpp", "b12367")
            for row in registry.entries(include_proposed=True):
                if not row.get("chat_weight"):
                    continue
                for task in ("text", "image"):
                    with self.subTest(build=build, artifact=row["id"], task=task):
                        before = capabilities.task_eligibility(row, old, task)
                        after = capabilities.task_eligibility(row, reviewed, task)
                        self.assertEqual(after.status, before.status)
                        if before.status == "source_supported":
                            self.assertIn(build, after.reason)
                            self.assertIn("unverified", after.reason)
                            self.assertEqual(
                                capabilities.task_eligibility(row, unknown, task).status, "unknown"
                            )
                self.assertEqual(
                    row["runtime"]["native_llama_cpp"]["tested_source_build"], "b12110"
                )

    def test_source_review_never_applies_to_custom_build_or_new_architecture(self):
        row = dict(load_registry().artifact("existing:llm/LFM3-1.7B-Q4_K_M.gguf"))
        row["architecture"] = "future-unreviewed"
        reviewed = capabilities.RuntimeIdentity("llama.cpp", "b12327")
        self.assertEqual(capabilities.task_eligibility(row, reviewed).status, "unknown")
        row["architecture"] = "lfm3"
        row["runtime"] = {
            "native_llama_cpp": {"tested_source_build": "b9999", "chat": "source_supported"}
        }
        self.assertEqual(capabilities.task_eligibility(row, reviewed).status, "unknown")

    def test_wrong_automatic_projector_is_rejected_before_replacing_working_model(self):
        from unittest.mock import Mock
        from ruinedfooocus.features.chat.llama_vision import projector_setting

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse(
            (root / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
        )
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        method_name = (
            "_initialize_model"
            if any(
                isinstance(node, ast.FunctionDef) and node.name == "_initialize_model"
                for node in cls.body
            )
            else "load_base_model"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        with tempfile.TemporaryDirectory() as folder:
            weight, projector = (Path(folder) / name for name in ("model.gguf", "mmproj.gguf"))
            weight.write_bytes(b"fixture")
            projector.write_bytes(b"wrong projector fixture")
            settings = SimpleNamespace(
                default_settings={"llama_localfile": str(weight), "llama_mmproj": " Auto "}
            )
            state = SimpleNamespace(path_manager=SimpleNamespace(DOWNLOADABLE_FILES={}))
            scope = {
                "Path": Path,
                "settings": settings,
                "DEFAULT_MODEL": str(weight),
                "resolve_model": Mock(return_value=weight),
                "projector_path": Mock(return_value=projector),
                "projector_setting": projector_setting,
            }
            exec(
                compile(ast.Module(body=[method], type_ignores=[]), "load_base_model", "exec"),
                scope,
            )
            owner, release = SimpleNamespace(unload=Mock()), Mock()
            annotation = load_registry().artifact("existing:vision/SmolVLM-256M-Instruct-Q8_0.gguf")
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "ruinedfooocus.app.state": state,
                        "ruinedfooocus.features.image_processing.auxiliary_resources": SimpleNamespace(
                            release_except=release
                        ),
                    },
                ),
                patch.object(
                    capabilities,
                    "configured_runtime",
                    return_value=capabilities.RuntimeIdentity("llama.cpp", "b12110"),
                ),
                patch.object(capabilities, "local_architecture", return_value="llama"),
                patch.object(
                    capabilities, "verified_local_artifact", side_effect=[annotation, None]
                ),
            ):
                with self.assertRaisesRegex(ValueError, "automatic projector"):
                    scope[method_name](
                        owner,
                        vision=True,
                        **({"runtime_load": Mock()} if method_name == "_initialize_model" else {}),
                    )
            owner.unload.assert_not_called()
            release.assert_not_called()

    def test_worker_rejects_known_incompatibility_before_download_or_unloading(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse(
            (root / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
        )
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        method_name = (
            "_initialize_model"
            if any(
                isinstance(node, ast.FunctionDef) and node.name == "_initialize_model"
                for node in cls.body
            )
            else "load_base_model"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        from unittest.mock import Mock

        selected = "LFM3-1.7B-Q4_K_M.gguf"
        settings = SimpleNamespace(
            default_settings={"llama_localfile": selected, "llm_runtime": "xllamacpp"}
        )
        state = SimpleNamespace(
            settings=settings,
            path_manager=SimpleNamespace(
                DOWNLOADABLE_FILES=json.loads(
                    resource_path("model_catalogue", "llm.json").read_text()
                )
            ),
        )
        resolve = Mock(side_effect=AssertionError("Must not download an unsupported model"))
        owner = SimpleNamespace(unload=Mock())
        scope = {
            "Path": Path,
            "settings": settings,
            "DEFAULT_MODEL": selected,
            "resolve_model": resolve,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "load_base_model", "exec"), scope)
        release = Mock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.app.state": state,
                    "ruinedfooocus.features.image_processing.auxiliary_resources": SimpleNamespace(
                        release_except=release
                    ),
                },
            ),
            patch.object(
                capabilities,
                "configured_runtime",
                return_value=capabilities.RuntimeIdentity("xllamacpp", "b11870"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "LFM3"):
                scope[method_name](
                    owner,
                    **({"runtime_load": Mock()} if method_name == "_initialize_model" else {}),
                )
        resolve.assert_not_called()
        owner.unload.assert_not_called()
        release.assert_not_called()

    def test_native_and_wrapper_support_are_source_and_task_specific(self):
        registry = load_registry()
        lfm = registry.artifact("existing:llm/LFM3-1.7B-Q4_K_M.gguf")
        native = capabilities.RuntimeIdentity("llama.cpp", "b12110")
        wrapper = capabilities.RuntimeIdentity("xllamacpp", "b11870")
        self.assertEqual(capabilities.task_eligibility(lfm, native).status, "source_supported")
        self.assertFalse(capabilities.task_eligibility(lfm, wrapper).allowed)
        affected = next(
            row
            for row in registry.entries(include_proposed=True)
            if row["runtime"].get("xllamacpp", {}).get("image")
            == "blocked_by_default_known_correctness_issue"
        )
        self.assertTrue(capabilities.task_eligibility(affected, wrapper, "text").allowed)
        self.assertFalse(capabilities.task_eligibility(affected, wrapper, "image").allowed)
        projector = registry.artifact(affected["companions"]["image"][0])
        self.assertFalse(capabilities.task_eligibility(projector, native).allowed)

    def test_unknown_custom_builds_are_not_falsely_certified_or_substituted(self):
        row = load_registry().artifact("existing:llm/LFM3-1.7B-Q4_K_M.gguf")
        custom = capabilities.RuntimeIdentity("llama.cpp", None, origin="custom")
        result = capabilities.task_eligibility(row, custom)
        self.assertTrue(result.allowed)
        self.assertEqual(result.status, "unknown")
        with patch.dict(os.environ, LLAMA_SERVER="C:/custom/server.exe"):
            self.assertIsNone(capabilities.configured_runtime("llama.cpp").build)

    def test_missing_registry_does_not_disable_custom_loads(self):
        with patch.object(
            capabilities, "load_registry", side_effect=RegistryError("missing sidecar")
        ):
            self.assertIsNone(capabilities.catalogue_artifact("custom.gguf", {}))
            self.assertEqual(
                capabilities.architecture_eligibility(
                    "llama", capabilities.RuntimeIdentity("llama.cpp", "b12110")
                ).status,
                "unknown",
            )
            self.assertIn(
                "missing sidecar",
                capabilities.selection_eligibility(
                    "custom.gguf", {}, capabilities.RuntimeIdentity("llama.cpp", "b12110")
                ).reason,
            )

    def test_actual_architecture_blocks_renamed_unsupported_weights(self):
        from gguf import GGUFWriter

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ordinary-model.gguf"
            writer = GGUFWriter(path, "lfm3")
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            architecture = capabilities.local_architecture(path)
            self.assertEqual(architecture, "lfm3")
            self.assertFalse(
                capabilities.architecture_eligibility(
                    architecture, capabilities.RuntimeIdentity("xllamacpp", "b11870")
                ).allowed
            )

    def test_matching_name_size_and_architecture_do_not_prove_weight_identity(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            path = Path(folder) / "local.gguf"
            path.write_bytes(b"published")
            row = {"artifact": {"bytes": 9, "sha256": hashlib.sha256(b"published").hexdigest()}}
            with patch.object(capabilities, "catalogue_artifact", return_value=row):
                self.assertIs(capabilities.verified_local_artifact(path, {}), row)
                path.write_bytes(b"different")
                self.assertIsNone(capabilities.verified_local_artifact(path, {}))

    def test_selection_keeps_missing_quant_and_ambiguous_paths(self):
        with patch.dict(
            "sys.modules",
            {"ruinedfooocus.app.state": SimpleNamespace(path_manager=None, settings=None)},
        ):
            module = runpy.run_path(
                str(
                    Path(__file__).resolve().parents[1]
                    / "ruinedfooocus/features/chat/llama_models.py"
                )
            )
        catalogue = {
            "Family": [
                ("Q4 · Local", "C:/one/model-Q4.gguf"),
                ("Q4 · Local", "D:/two/model-Q4.gguf"),
            ]
        }
        model, value = module["saved_selection"](catalogue, "D:/two/model-Q4.gguf")
        self.assertEqual((model, value), ("Family", "D:/two/model-Q4.gguf"))
        self.assertEqual(module["saved_selection"](catalogue, "model-Q4.gguf")[1], "model-Q4.gguf")
        choices, selected, available = module["selection_choices"]({}, model, value)
        self.assertEqual(selected, value)
        self.assertFalse(available)
        self.assertEqual(choices[0][1], value)
        self.assertEqual(module["selection_choices"](catalogue, model, value)[1], value)
        self.assertEqual(
            module["selection_choices"]({}, model, value, preserve=False), ([], None, False)
        )
        choices, selected, available = module["selection_choices"](
            catalogue, model, "model-Q4.gguf"
        )
        self.assertEqual(selected, "model-Q4.gguf")
        self.assertFalse(available)
        self.assertEqual([item[1] for item in choices[:2]], [item[1] for item in catalogue[model]])


if __name__ == "__main__":
    unittest.main()
