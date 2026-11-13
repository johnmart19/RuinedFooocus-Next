from ruinedfooocus.core.paths import asset_path, resource_path
from ruinedfooocus.core.model_formats import is_safetensors

"""Workflow validation without starting model workers or downloading files."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, mock_open, patch
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


class WorkflowControlsTests(unittest.TestCase):
    def test_native_translation_markers_are_not_embedded_in_toolbar_button_labels(self):
        # Gradio's custom toolbar button labels bypass native i18n processing.
        # Keep translated actions as ordinary buttons until that path supports it.
        for source in (ROOT / "ruinedfooocus").rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            translated_buttons = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                call = node.value
                if not isinstance(call.func, ast.Attribute) or call.func.attr != "Button":
                    continue
                values = list(call.args) + [kw.value for kw in call.keywords if kw.arg == "value"]
                if any(
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in ("t", "translate_ui")
                    for value in values
                ):
                    translated_buttons.update(
                        target.id for target in node.targets if isinstance(target, ast.Name)
                    )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "buttons" and isinstance(
                        keyword.value, (ast.List, ast.Tuple)
                    ):
                        embedded = {
                            item.id for item in keyword.value.elts if isinstance(item, ast.Name)
                        }
                        with self.subTest(source=source.relative_to(ROOT), line=node.lineno):
                            self.assertFalse(embedded & translated_buttons)

    def test_checkpoint_tools_panel_is_available(self):
        gr = MagicMock()
        namespace = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            "gr": gr,
            "is_safetensors": is_safetensors,
            "CheckpointToolSelection": Mock(),
            "version_js": lambda channel: "(...values) => values",
            "publish_checkpoint_result": Mock(),
            "shared": SimpleNamespace(models=SimpleNamespace(get_names=lambda category: ["model"])),
        }
        create = function_from(
            "ruinedfooocus/features/models/ui_checkpoint_tools.py", "create_ui", namespace
        )
        create(SimpleNamespace(value="model.sft", change=Mock()))
        gr.Column.assert_called_once_with(visible=True, elem_id="checkpoint-tools")
        self.assertTrue(
            any(call.kwargs.get("interactive") is True for call in gr.Button.call_args_list)
        )

    def test_gallery_grid_dimensions_match_the_gradio_contract(self):
        # Gradio 6 serializes legacy lists without rejecting them, but the browser
        # receives an invalid CSS repeat() count and renders one oversized column.
        found = 0
        for source in (ROOT / "ruinedfooocus").rglob("*.py"):
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "Gallery"
                ):
                    continue
                for keyword in node.keywords:
                    if keyword.arg in {"columns", "rows"}:
                        with self.subTest(path=source.relative_to(ROOT), dimension=keyword.arg):
                            value = ast.literal_eval(keyword.value)
                            self.assertTrue(value is None or (type(value) is int and value > 0))
                            found += 1
        self.assertGreater(found, 0)

    def test_generation_initial_update_works_after_changelog_setup(self):
        from ruinedfooocus.ui import hints

        namespace = dict(
            gr=SimpleNamespace(update=lambda **values: values),
            html=SimpleNamespace(make_progress_html=lambda *_: "Waiting"),
            asset_path=asset_path,
            hints=hints,
            ui_hints=hints,
        )
        for name in (
            "run_button",
            "stop_button",
            "progress_html",
            "gallery",
            "main_view",
            "inpaint_view",
            "hint_text",
        ):
            namespace[name] = name
        source = ROOT / "ruinedfooocus/ui/app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        changelog = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "open"
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and child.args[0].value == "update_log.md"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        with patch("builtins.open", mock_open(read_data="Recent changes")):
            exec(
                compile(ast.Module(body=[changelog], type_ignores=[]), str(source), "exec"),
                namespace,
            )
        update = function_from("ruinedfooocus/ui/app.py", "update_clicked", namespace)()
        self.assertTrue(update["hint_text"]["value"].startswith("**LPT:**"))
        self.assertFalse(update["run_button"]["interactive"])
        self.assertTrue(update["stop_button"]["interactive"])

    def test_rewrite_preview_requires_apply_and_does_not_overwrite_new_edits(self):
        from ruinedfooocus.features.prompting.ui_rewrite import safe_rewrite

        gr = SimpleNamespace(Error=ValueError, update=lambda **values: values)
        model = Mock(return_value="A rewritten prompt")
        suggest = function_from(
            "ruinedfooocus/features/prompting/ui_rewrite.py",
            "suggest",
            {"gr": gr, "run_prompt": model, "safe_rewrite": safe_rewrite},
        )
        preview, original, enabled = suggest("my-preset.txt", "Original prompt")
        self.assertEqual(original, "Original prompt")
        self.assertEqual(preview["value"], "A rewritten prompt")
        apply = function_from(
            "ruinedfooocus/features/prompting/ui_rewrite.py", "apply_rewrite", {"gr": gr}
        )
        with self.assertRaisesRegex(ValueError, "prompt changed"):
            apply("New user edit", original, preview["value"])
        result = apply(original, original, preview["value"])
        self.assertEqual(
            result[:2], ("A rewritten prompt", ("Original prompt", "A rewritten prompt"))
        )

    def test_rewrite_undo_protects_later_user_edits(self):
        gr = SimpleNamespace(Error=ValueError, update=lambda **values: values, skip=lambda: "skip")
        undo = function_from(
            "ruinedfooocus/features/prompting/ui_rewrite.py", "undo_rewrite", {"gr": gr}
        )
        record = ("Original", "Applied")
        with self.assertRaisesRegex(ValueError, "overwrite your edits"):
            undo("Later edit", record)
        self.assertEqual(undo("Applied", record)[0], {"value": "Original"})

    def test_image_submission_preserves_prompt_before_it_reaches_the_worker(self):
        policy = function_from(
            "ruinedfooocus/features/prompting/prompt_processing.py", "use_random_prompt", {}
        )
        add = Mock(return_value="task")
        submit = function_from(
            "ruinedfooocus/ui/app.py",
            "append_work",
            {
                "use_random_prompt": policy,
                "worker": SimpleNamespace(add_task=add),
                "get_promptlist": lambda data: data["prompt"].split("---"),
                "ui_controlnet": SimpleNamespace(tool_kind=lambda *_: None),
            },
        )
        config = dict(
            prompt="Keep the black hair and neon street.",
            image_number=2,
            input_image=object(),
            controlnet={"type": "img2img"},
            obp_assume_direct_control=True,
            obp_image_prompt=False,
        )
        self.assertEqual(submit(config), "task")
        self.assertEqual(add.call_args.args[0]["prompt"], config["prompt"])
        self.assertEqual(add.call_args.args[0]["image_total"], 2)
        submit(config | {"inpaint_toggle": True, "input_image": None})
        self.assertEqual(add.call_args.args[0]["prompt"], config["prompt"])
        submit(config | {"obp_image_prompt": True})
        self.assertEqual(add.call_args.args[0]["prompt"], "")
        submit(config | {"input_image": None})
        self.assertEqual(add.call_args.args[0]["prompt"], "")
        self.assertEqual(config["prompt"], "Keep the black hair and neon street.")

    def submit(self, custom=False):
        from ruinedfooocus.features.chat.model_preparation import validate_message

        return function_from(
            "ruinedfooocus/features/chat/ui.py",
            "llama_get_text",
            {
                "gr": SimpleNamespace(Error=ValueError),
                "custom_file": custom,
                "selected_file": "custom.gguf",
                "validate_message": validate_message,
            },
        )

    def test_empty_or_unloaded_messages_are_rejected_and_active_model_accepts_other_candidate(self):
        submit = self.submit()
        for message, active, selected in [
            ("  ", "a", "a"),
            ("hello", None, "a"),
        ]:
            with (
                self.subTest(message=message, active=active, selected=selected),
                self.assertRaises(ValueError),
            ):
                submit(message, True, active, selected)
        self.assertEqual(submit("hello", True, "a", "a"), ("hello", True))
        self.assertEqual(submit("hello", True, "active-B", "candidate-A"), ("hello", True))

    def test_configured_custom_model_remains_usable_without_catalogue_selection(self):
        self.assertEqual(
            self.submit(custom=True)("hello", False, "custom.gguf", None), ("hello", False)
        )
        self.assertEqual(
            self.submit(custom=True)("hello", False, "active.gguf", None), ("hello", False)
        )

    def test_invalid_submit_leaves_conversation_untouched_and_never_calls_worker(self):
        respond = Mock()
        submit = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "submit_message",
            {
                "gr": SimpleNamespace(Warning=Mock(), skip=lambda: "skip"),
                "llama_msg": "message",
                "image_tool_status": "inline_status",
                "editor_workspace": None,
                "llama_get_text": self.submit(),
                "llama_respond": respond,
            },
        )
        history = [{"role": "assistant", "content": "Keep this conversation."}]
        for message, active, error in (
            ("", "a", "Enter a message first."),
            ("Keep this prompt.", None, "Load a model before sending a message."),
        ):
            with self.subTest(message=message, active=active):
                result = list(
                    submit(message, False, "a", "", "", history, active, False, False, None, False)
                )
                self.assertEqual(result, [{"message": "skip", "inline_status": error}])
                self.assertEqual(
                    history, [{"role": "assistant", "content": "Keep this conversation."}]
                )
        respond.assert_not_called()

    def test_chat_detail_does_not_shift_prompt_source_denoise_or_image_settings(self):
        from PIL import Image

        worker = SimpleNamespace(
            add_task=Mock(return_value=1), task_result=Mock(return_value=("results", "done"))
        )
        respond = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "llama_respond",
            {
                "worker": worker,
                "gr": SimpleNamespace(update=lambda **values: values),
                "image_controls": {"base_model_name": None, "custom_steps": None},
                "llama_chat": "chat",
                "llama_history": "history",
                "llama_model_status": "status",
                "llama_download": "download",
                "display_history": lambda history, show: history,
            },
        )
        source = Image.new("RGBA", (23, 17), (3, 4, 5, 87))
        history = [{"role": "user", "content": "Earlier request"}]
        list(
            respond(
                'Keep "CAFÉ 42".',
                "system",
                "[]",
                history,
                "model.gguf",
                False,
                True,
                "review.gguf",
                False,
                False,
                "Uploaded image",
                source,
                0.37,
                True,
                "high",
                "janku.safetensors",
                30,
                reasoning={"mode": "Off", "effort": "Auto"},
            )
        )
        task = worker.add_task.call_args.args[0]
        self.assertEqual(task["image_detail"], "high")
        self.assertEqual(task["reasoning"], {"mode": "Off", "effort": "Auto"})
        self.assertIs(task["image_reference"], source)
        self.assertEqual(task["image_denoise"], 0.37)
        self.assertEqual(
            task["image_settings"], {"base_model_name": "janku.safetensors", "custom_steps": 30}
        )
        last_user = next(
            message for message in reversed(task["history"]) if message["role"] == "user"
        )
        self.assertEqual(last_user["content"], 'Keep "CAFÉ 42".')
        self.assertEqual(len(history), 1)

    def test_missing_chat_reference_preserves_message_without_submitting_or_loading(self):
        respond = Mock()
        workspace = SimpleNamespace()
        workspace.image_context = MethodType(
            function_from(
                "ruinedfooocus/features/gallery/ui_workspace.py",
                "image_context",
                {"ui_request_owner": lambda request: "workflow-fixture"},
            ),
            workspace,
        )
        submit = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "submit_message",
            {
                "gr": SimpleNamespace(Warning=Mock(), skip=lambda: "skip"),
                "llama_msg": "message",
                "image_tool_status": "inline_status",
                "editor_workspace": workspace,
                "llama_get_text": self.submit(),
                "llama_respond": respond,
            },
        )
        history = [{"role": "assistant", "content": "Keep this conversation."}]
        for mode, error in (
            ("Uploaded image", "Upload an image before selecting this source."),
            (
                "Previous generated image",
                "Open the current owned image before selecting this source.",
            ),
        ):
            with self.subTest(mode=mode):
                result = list(
                    submit(
                        "Keep this prompt.",
                        True,
                        "a",
                        "",
                        "",
                        history,
                        "a",
                        False,
                        False,
                        None,
                        False,
                        mode,
                    )
                )
                self.assertEqual(result, [{"message": "skip", "inline_status": error}])
                self.assertEqual(
                    history, [{"role": "assistant", "content": "Keep this conversation."}]
                )
        respond.assert_not_called()

    def test_search_single_page_has_no_phantom_second_page(self):
        from ruinedfooocus.features.gallery.image_metadata import read_metadata

        browser = SimpleNamespace(
            load_images=Mock(
                return_value=(["image.png"], "", [], {"count": 1, "pages": 1, "page": 1})
            ),
            num_images_pages=Mock(return_value=(1, 1)),
        )
        namespace = {
            "gr": SimpleNamespace(
                update=lambda **values: values, Gallery=lambda **values: values, Request=object
            ),
            "browser": browser,
            "localize": __import__(
                "ruinedfooocus.core.translation_manager", fromlist=["localize"]
            ).localize,
            "resolve_locale": __import__(
                "ruinedfooocus.core.translation_manager", fromlist=["resolve_locale"]
            ).resolve_locale,
            "read_metadata": read_metadata,
            "action_buttons": [
                "prompt_btn",
                "setup_btn",
                "edit_btn",
                "inpaint_btn",
                "upscale_btn",
                "rembg_btn",
            ],
            "navigation_controls": ["update_btn", "ib_page", "search_btn", "search_input"],
        }
        namespace.update(
            {
                name: name
                for name in (
                    "gallery",
                    "ib_page",
                    "status_output",
                    "page_records",
                    "ib_range",
                    "pagination",
                    "page_total",
                    "previous_page",
                    "next_page",
                    "selected",
                    "metadata_json",
                    "metadata_output",
                )
            }
        )
        function_from(
            "ruinedfooocus/features/gallery/ui.py",
            "availability",
            namespace,
        )
        function_from("ruinedfooocus/features/gallery/ui.py", "render_page", namespace)
        search = function_from(
            "ruinedfooocus/features/gallery/ui.py",
            "search",
            namespace,
        )
        result = search("test")
        self.assertEqual(result["ib_page"], {"value": 1, "maximum": 1, "interactive": True})
        browser.load_images.assert_called_once_with(
            1, "test", include_paths=True, include_page=True, locale="en"
        )
        browser.num_images_pages.assert_not_called()
        self.assertFalse(hasattr(browser, "filter"))
        self.assertIsNone(result["selected"])
        self.assertEqual(result["metadata_json"], {})
        self.assertTrue(
            all(result[button] == {"interactive": False} for button in namespace["action_buttons"])
        )
        self.assertTrue(
            all(result[control]["interactive"] for control in namespace["navigation_controls"])
        )


if __name__ == "__main__":
    unittest.main()
