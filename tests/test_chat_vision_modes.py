"""Text/vision transitions and reviewer restoration never reuse the wrong load."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ruinedfooocus.features.chat.llama_vision import requested_vision, projector_setting

ROOT = Path(__file__).resolve().parents[1]


def method(name, scope):
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
    )
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
    )
    node = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "llama_pipeline", "exec"), scope)
    return scope[name]


class ChatVisionModeTests(unittest.TestCase):
    def test_sharing_loaded_model_does_not_reload_but_default_changes_do(self):
        settings = SimpleNamespace(default_settings={})
        identity = method(
            "current_model_settings",
            {"json": json, "settings": settings, "DEFAULT_MODEL": "default.gguf"},
        )
        owner = SimpleNamespace(loaded_paths=())
        explicit = identity(owner, "selected.gguf")
        settings.default_settings["llama_localfile"] = "selected.gguf"
        self.assertEqual(explicit, identity(owner, "selected.gguf"))
        self.assertEqual(explicit, identity(owner))
        settings.default_settings["llama_localfile"] = "other.gguf"
        self.assertEqual(explicit, identity(owner, "selected.gguf"))
        self.assertNotEqual(explicit, identity(owner))

    def test_text_does_not_request_an_explicit_projector_without_image_intent(self):
        settings = {"llama_mmproj": "matching-projector.gguf"}
        self.assertFalse(requested_vision("custom.gguf", settings, False))
        self.assertTrue(requested_vision("custom.gguf", settings, True))
        self.assertFalse(requested_vision("custom.gguf", {"llama_mmproj": "None"}, True))
        self.assertEqual(projector_setting({"llama_mmproj": " Auto "}), "Auto")
        self.assertEqual(projector_setting({"llama_mmproj": "  "}), "Auto")
        self.assertFalse(requested_vision("custom.gguf", {"llama_mmproj": " None "}, True))

    def test_mode_file_changes_and_mutable_sampler_overrides_invalidate_cache_identity(self):
        settings = SimpleNamespace(default_settings={"llm_sampling": {"temperature": 0}})
        identity = method("current_model_settings", {"json": json, "settings": settings})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.gguf"
            path.write_bytes(b"first")
            owner = SimpleNamespace(loaded_paths=(path,))
            original = identity(owner, "model.gguf", vision=False)
            loaded_values = {"llm_sampling": {"temperature": 0}}
            self.assertNotEqual(original, identity(owner, "model.gguf", vision=True))
            self.assertEqual(original, identity(owner, "model.gguf", vision=False))
            settings.default_settings["llm_sampling"]["temperature"] = 0.7
            self.assertNotEqual(original, identity(owner, "model.gguf", vision=False))
            self.assertEqual(original, identity(owner, "model.gguf", values=loaded_values))
            replaced = identity(owner, "model.gguf")
            path.write_bytes(b"replacement")
            self.assertNotEqual(replaced, identity(owner, "model.gguf"))
            path.unlink()
            self.assertNotEqual(replaced, identity(owner, "model.gguf"))

    def test_load_only_and_persistent_vision_switch_modes_once_each(self):
        settings = SimpleNamespace(
            default_settings={"llama_localfile": "custom.gguf", "llama_mmproj": "projector.gguf"}
        )
        process = method(
            "process",
            {
                "settings": settings,
                "DEFAULT_MODEL": "custom.gguf",
                "model_has_vision": lambda name: False,
                "requested_vision": requested_vision,
                "worker": SimpleNamespace(add_result=Mock()),
            },
        )
        owner = SimpleNamespace(
            llm=None,
            model_settings=None,
            vision_projector=None,
            runtime_name="llama.cpp",
            runtime_label="ROCm",
            fallback_reason="",
        )
        owner.current_model_settings = lambda model, vision=False, values=None: (model, vision)

        def load(model, progress=None, *, vision=False, **load_options):
            owner.llm = object()
            owner.model_settings = owner.current_model_settings(model, vision=vision)
            owner.vision_projector = "projector.gguf" if vision else None

        owner.load_base_model = Mock(side_effect=load)
        for enabled in (False, True, True, False):
            self.assertEqual(
                process(
                    owner, {"load_only": True, "model": "custom.gguf", "vision_enabled": enabled}
                ),
                {
                    "ready": True,
                    "runtime_name": "llama.cpp",
                    "runtime_label": "ROCm",
                    "fallback_reason": "",
                    "runtime_snapshot": None,
                    "load_state": None,
                    "reasoning_capability": None,
                    "sampling_state": None,
                },
            )
        self.assertEqual(
            [call.kwargs["vision"] for call in owner.load_base_model.call_args_list],
            [False, True, False],
        )

    def test_temporary_review_restores_success_and_defers_failed_runtime_restoration(self):
        inspect = method(
            "inspect_image",
            {"worker": SimpleNamespace(check_interrupt=Mock(), current_job=lambda: None)},
        )
        for fail in (False, True):
            with self.subTest(fail=fail):
                owner = SimpleNamespace(vision_projector=None, llm=object())

                def load(model, progress=None, *, vision=False, **load_options):
                    owner.vision_projector = "projector.gguf" if vision else None

                owner.load_base_model = Mock(side_effect=load)
                owner.unload = Mock(side_effect=lambda: setattr(owner, "vision_projector", None))
                owner.complete_text = Mock(
                    side_effect=RuntimeError("review failed") if fail else None,
                    return_value="caption",
                )
                if fail:
                    data = {}
                    with self.assertRaisesRegex(RuntimeError, "review failed"):
                        inspect(owner, [], "review.gguf", "chat.gguf", data)
                    self.assertTrue(data["_chat_restoration_pending"])
                    owner.unload.assert_called_once()
                else:
                    self.assertEqual(inspect(owner, [], "review.gguf", "chat.gguf", {}), "caption")
                self.assertEqual(
                    [call.kwargs["vision"] for call in owner.load_base_model.call_args_list],
                    [True] if fail else [True, False],
                )
                self.assertIsNone(owner.vision_projector)


if __name__ == "__main__":
    unittest.main()
