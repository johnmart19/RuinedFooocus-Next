"""Exercise the worker dispatch boundary without loading a model or starting it."""

import ast
from pathlib import Path
import unittest
import tempfile
from unittest.mock import Mock, patch
from types import SimpleNamespace

from ruinedfooocus.features.models.component_profiles import SCOPE_KEY, checkpoint_identity
from ruinedfooocus.features.models.effective_image_plan import COMPONENTS_KEY, ImagePlanComponent


class WorkerRoutingTests(unittest.TestCase):
    def test_explicit_local_chat_load_uses_io_admission_without_changing_ready_messages(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        gate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "needs_io" for target in node.targets
            )
        )
        scope = dict(
            api_job=None,
            chat_artifacts=lambda data: (),
            needs_preparation=lambda data: False,
            needs_prompt_preparation=lambda data: False,
            needs_image_preparation=lambda data: False,
        )
        for task, load_only, unload, expected in (
            ("llama", True, False, True),
            ("llama", False, False, False),
            ("llama", True, True, False),
            ("process", False, False, False),
        ):
            scope["gen_data"] = dict(task_type=task, load_only=load_only, unload=unload)
            exec(compile(ast.Module(body=[gate], type_ignores=[]), str(source), "exec"), scope)
            self.assertEqual(bool(scope["needs_io"]), expected)

    def test_startup_loads_weights_only_when_explicitly_enabled(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preload_startup_model"
        )
        registry = SimpleNamespace(update=Mock())
        settings = SimpleNamespace(default_settings={"base_model": "selected.safetensors"})
        shared = SimpleNamespace(
            models=SimpleNamespace(
                get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
                get_model_base=lambda value: value["baseModel"],
                get_file=lambda *args: checkpoint,
            ),
            path_manager=SimpleNamespace(paths={}, DOWNLOADABLE_FILES={}, get_abspath=Path),
        )
        scope = {"settings": settings, "pipelines": registry, "shared": shared}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        scope["preload_startup_model"]()
        registry.update.assert_not_called()
        settings.default_settings["preload_image_model"] = True
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "fixture.safetensors"
            checkpoint.write_bytes((2).to_bytes(8, "little") + b"{}")
            scope["preload_startup_model"]()
            identity = checkpoint_identity(checkpoint)
        choices = {"clip_l": None, "clip_g": None, "vae_sdxl": None}
        registry.update.assert_called_once()
        captured = registry.update.call_args.args[0]
        self.assertEqual(captured["base_model_name"], "selected.safetensors")
        self.assertEqual(captured["image_component_overrides"], choices)
        self.assertEqual(captured[SCOPE_KEY], identity)
        self.assertIsInstance(captured[COMPONENTS_KEY], tuple)
        self.assertTrue(
            all(isinstance(row, ImagePlanComponent) for row in captured[COMPONENTS_KEY])
        )
        self.assertEqual({row.key for row in captured[COMPONENTS_KEY]}, set(choices))
        registry.update.return_value.load_base_model.assert_called_once_with(
            "selected.safetensors", component_overrides=choices
        )

    def test_image_dispatch_uses_explicit_controls_and_preserves_queued_request(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        handler = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "handler"
        )
        for kind, expected in (("img2img", None), ("rembg", "rembg")):
            with self.subTest(kind=kind):
                process, release = Mock(), Mock()
                controls = SimpleNamespace(get_settings=Mock(return_value={"type": kind}))
                scope = {"control_tools": controls, "process": process}
                exec(
                    compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), scope
                )
                request = {
                    "task_type": "process",
                    "prompt": "Keep two red apples.",
                    "image_number": 2,
                }
                original = dict(request)
                with patch(
                    "ruinedfooocus.features.image_processing.auxiliary_resources.release_except",
                    release,
                ):
                    scope["handler"](request)
                release.assert_not_called()
                process.assert_called_once_with(request)
                controls.get_settings.assert_not_called()
                self.assertEqual(request, original)


if __name__ == "__main__":
    unittest.main()
