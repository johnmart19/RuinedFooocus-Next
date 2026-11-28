"""Actual callbacks for UI actions the event runner cannot fire with initial values.

They need a real upload, a page event payload, a typed wildcard or the API, so each
is copied from its source and run with stand-ins for the services it calls.
"""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ui_test_helpers import function_from

GR = SimpleNamespace(
    update=lambda **values: values,
    skip=lambda: "skip",
    Info=Mock(),
    Request=object,
    EventData=object,
)


class SpellcheckTests(unittest.TestCase):
    def setUp(self):
        self.select = function_from(
            "ruinedfooocus/ui/app.py", "select_spellcheck", {"gr": GR, "prompt": "prompt"}
        )

    def test_choice_completes_the_last_wildcard(self):
        self.assertEqual(self.select("a __col", "colors")["prompt"], {"value": "a __colors__"})
        self.assertEqual(
            self.select("__a__ and __an", "animals")["prompt"],
            {"value": "__a__ and __animals__"},
        )

    def test_a_prompt_without_a_wildcard_or_a_choice_is_left_unchanged(self):
        for text, choice in (("edited away", "colors"), (None, "colors"), ("a __x", None)):
            with self.subTest(text=text, choice=choice):
                self.assertEqual(self.select(text, choice)["prompt"], {})


class ApiImageTests(unittest.TestCase):
    def test_generate_image_returns_the_first_file_as_gradio_file_data(self):
        runtime = SimpleNamespace(
            legacy_images=Mock(return_value=["first.png", "second.png"]),
            wait_for_result=lambda result: result,
        )
        generate = function_from(
            "ruinedfooocus/app/api.py", "api_generate_image", {"api_runtime": runtime}
        )
        self.assertEqual(
            generate("a red cup"),
            {"path": "first.png", "meta": {"_type": "gradio.FileData"}},
        )
        runtime.legacy_images.assert_called_once_with("a red cup")


class ImageAddedTests(unittest.TestCase):
    def callback(self, tool):
        GR.Info.reset_mock()
        return function_from(
            "ruinedfooocus/features/image_editing/ui.py",
            "image_added",
            {
                "gr": GR,
                "edit_tool": Mock(return_value=tool),
                "checkpoint_features": Mock(return_value={}),
            },
        )

    def test_an_upload_without_a_tool_selects_the_best_edit_tool(self):
        self.assertEqual(self.callback("Img2Img")("image", "None", "model"), {"value": "Img2Img"})
        GR.Info.assert_called_once_with("Image added: using Image to image.")

    def test_an_existing_tool_a_cleared_upload_or_no_tool_keep_the_selection(self):
        for image, selection, tool in (
            ("image", "Canny", "Img2Img"),
            (None, "None", "Img2Img"),
            ("image", "None", None),
        ):
            with self.subTest(image=image, selection=selection, tool=tool):
                self.assertEqual(self.callback(tool)(image, selection, "model"), {})
                GR.Info.assert_not_called()


class AdoptMaskTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.store.bind_mask.return_value = SimpleNamespace(artifact_id="mask-1")
        staged = Mock()
        staged.return_value.__enter__ = Mock(return_value="owned.png")
        staged.return_value.__exit__ = Mock(return_value=False)
        owner = SimpleNamespace(selected_store=lambda: self.store, _staged_input=staged)
        self.adopt = function_from(
            "ruinedfooocus/features/gallery/ui_workspace.py",
            "adopt_mask",
            {"self": owner, "gr": GR, "ui_request_owner": lambda request: ("ui", "", "local")},
        )
        self.context = dict(session_id="s", source_id="src", revision=3)

    def test_mask_is_bound_to_the_current_owned_image_revision(self):
        result = self.adopt("mask.png", self.context)
        self.assertEqual(result, dict(self.context, mask_id="mask-1"))
        self.store.artifact.assert_called_once_with(
            "s", "src", revision=3, owner=("ui", "", "local")
        )
        self.store.bind_mask.assert_called_once_with(
            "s", "owned.png", "src", owner=("ui", "", "local")
        )

    def test_no_file_keeps_context_and_a_fresh_upload_waits_for_run(self):
        self.assertIs(self.adopt(None, self.context), self.context)
        self.assertIsNone(self.adopt("mask.png", None))
        self.store.bind_mask.assert_not_called()


class LocalModelActionTests(unittest.TestCase):
    def setUp(self):
        self.files = {"q4": "C:/models/model-q4.gguf", "q8": "C:/models/model-q8.gguf"}
        self.select_source = Mock(return_value={"selection": "updated"})
        self.removed = Mock()
        scope = {
            "gr": GR,
            "Path": Path,
            "model_catalogue": lambda source: {"model": [("Q4", "q4"), ("Q8", "q8")]},
            "select_source": self.select_source,
            "preferred_quant": lambda choices: choices[-1][1],
            "local_models": lambda: self.files,
            "remove_local_model": self.removed,
        }
        for name in (
            "llama_model_status",
            "pending_remove",
            "remove_dialog",
            "llama_quant",
            "llama_msg",
            "llama_use",
        ):
            scope[name] = name
        self.action = function_from(
            "ruinedfooocus/features/chat/ui.py", "local_model_action", dict(scope)
        )
        self.confirm = function_from(
            "ruinedfooocus/features/chat/ui.py", "confirm_remove", dict(scope)
        )

    @staticmethod
    def event(**data):
        return SimpleNamespace(_data=data)

    def test_selecting_keeps_a_valid_quantization_or_uses_the_preferred_one(self):
        self.action("Local models", "q4", "session", self.event(action="select", model="model"))
        self.select_source.assert_called_with("Local models", "model", "q4", "session")
        self.action("Local models", "gone", "session", self.event(action="select", model="model"))
        self.select_source.assert_called_with("Local models", "model", "q8", "session")

    def test_a_missing_model_asks_for_a_refresh_and_remove_opens_the_file_dialog(self):
        missing = self.action("Local models", None, "s", self.event(action="select", model="x"))
        self.assertIn("no longer available", missing["llama_model_status"])
        dialog = self.action("Local models", "q8", "s", self.event(action="remove", model="model"))
        paths = [str(Path(path).resolve()) for path in self.files.values()]
        self.assertEqual(dialog["pending_remove"], paths)
        self.assertEqual(dialog["remove_dialog"]["selected"], paths[1])
        self.removed.assert_not_called()

    def test_confirm_removes_only_a_valid_choice_and_reports_errors(self):
        paths = ["a.gguf", "b.gguf"]
        for data in (dict(action="delete", file_index=5), dict(action="other", file_index=0)):
            with self.subTest(data=data):
                result = self.confirm(paths, "model", "q4", "s", self.event(**data))
                self.assertEqual(result, {"pending_remove": None, "remove_dialog": ""})
        self.removed.assert_not_called()
        result = self.confirm(paths, "model", "q4", "s", self.event(action="delete", file_index=1))
        self.removed.assert_called_once_with("b.gguf", delete_file=True)
        self.assertEqual(result["llama_model_status"], "File deleted.")
        self.removed.reset_mock()
        self.removed.side_effect = OSError("in use")
        failed = self.confirm(paths, "model", "q4", "s", self.event(action="hide", file_index=0))
        self.removed.assert_called_once_with("a.gguf", delete_file=False)
        self.assertEqual(failed["llama_model_status"], "in use")


if __name__ == "__main__":
    unittest.main()
