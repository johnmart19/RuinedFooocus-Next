"""Button-interactivity guards and actionable error handling for Chat and Models UIs."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

import test_chat_image_ui_events as image_ui_fixture
from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.chat import ui_model_diagnostics, ui_runtime_info
from ruinedfooocus.features.models import (
    ui_adapters,
    ui_checkpoint_tools,
    ui_download,
    ui_model_components,
    ui_popular,
)


def _button(blocks, label):
    translated = str(translate_ui(label))
    for block in blocks.blocks.values():
        if isinstance(block, gr.Button) and str(block.value) in (label, translated):
            return block
    raise KeyError(label)


class ChatButtonGuardsTests(unittest.TestCase):
    def test_chat_buttons_require_loaded_model_message_and_assistant_name(self):
        fixture = image_ui_fixture.ChatImageUiEventsTests()
        app, _, _, _, _ = fixture.make_ui()

        send_btn = _button(app, "Send")
        readiness_btn = _button(app, "Check image readiness")
        save_btn = _button(app, "Save")
        save_as_btn = _button(app, "Save as new")

        self.assertFalse(send_btn.interactive)
        self.assertFalse(readiness_btn.interactive)
        self.assertFalse(save_btn.interactive)
        self.assertFalse(save_as_btn.interactive)

        # Send requires both loaded model and non-empty message
        send_fns = [
            fn
            for fn in app.fns.values()
            if fn.fn is not None and send_btn in fn.outputs and len(fn.inputs) == 2
        ]
        self.assertTrue(send_fns, "Expected reactive Send guard wired to model and message")
        send_guard = send_fns[0].fn
        self.assertFalse(send_guard(None, "hello")["interactive"])
        self.assertFalse(send_guard("model.gguf", "")["interactive"])
        self.assertFalse(send_guard("model.gguf", "   ")["interactive"])
        self.assertTrue(send_guard("model.gguf", "hello")["interactive"])

        # Check image readiness requires an active chat model
        ref_fn = next(
            fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == "reference_controls"
        )
        no_model = ref_fn("run", False, None, "New image", None, None)
        self.assertTrue(no_model[5]["visible"])
        self.assertFalse(no_model[5]["interactive"])
        with_model = ref_fn("run", False, "model.gguf", "New image", None, None)
        self.assertTrue(with_model[5]["visible"])
        self.assertTrue(with_model[5]["interactive"])

        # Assistant Save / Save as new require non-empty assistant name
        save_guard = next(
            fn.fn
            for fn in app.fns.values()
            if fn.fn is not None and save_btn in fn.outputs and save_as_btn in fn.outputs
        )
        empty_res = save_guard("   ")
        self.assertFalse(empty_res[0]["interactive"])
        self.assertFalse(empty_res[1]["interactive"])
        named_res = save_guard("Helper")
        self.assertTrue(named_res[0]["interactive"])
        self.assertTrue(named_res[1]["interactive"])

    def test_runtime_info_cancel_starts_disabled_and_skips_when_idle(self):
        with gr.Blocks() as app:
            profile = gr.State({"settings": {"cpu_everywhere": True}})
            ui_runtime_info.create_runtime_info_controls(SimpleNamespace(cpu=True), profile)

        self.assertTrue(_button(app, "Refresh").interactive)
        self.assertFalse(_button(app, "Cancel runtime refresh").interactive)

        callbacks = {
            getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values() if fn.fn is not None
        }
        self.assertEqual(
            callbacks["begin_refresh"](),
            (gr.update(interactive=False), gr.update(interactive=True)),
        )
        self.assertEqual(
            callbacks["end_refresh"](),
            (gr.update(interactive=True), gr.update(interactive=False)),
        )
        session = ui_runtime_info.RuntimeInfoSession()
        self.assertEqual(callbacks["stop"](session, gr.Request()), gr.skip())

    def test_model_diagnostics_recommendations_guard_tracks_installed_gguf(self):
        with tempfile.TemporaryDirectory() as folder:
            installed = Path(folder) / "demo.gguf"
            installed.write_bytes(b"gguf")
            with gr.Blocks() as app:
                selected = gr.State(None)
                runtime_status = gr.State("")
                sampling_status = gr.State("")
                ui_model_diagnostics.create_model_diagnostics(
                    selected, runtime_status, sampling_status
                )
            rec_button = _button(app, "Refresh sampling recommendations")
            self.assertFalse(rec_button.interactive)

            guard = next(
                fn.fn
                for fn in app.fns.values()
                if getattr(fn.fn, "__name__", None) == "_recommendations_update"
            )
            self.assertFalse(guard(None)["interactive"])
            self.assertFalse(guard(str(Path(folder) / "missing.gguf"))["interactive"])
            self.assertTrue(guard(str(installed))["interactive"])


class ModelsButtonGuardsTests(unittest.TestCase):
    def test_checkpoint_tools_inspect_guard_and_empty_selection_error_handling(self):
        with gr.Blocks() as app:
            base = gr.Dropdown(choices=[], value=None)
            ui_checkpoint_tools.create_ui(base)

        inspect_btn = _button(app, "Inspect")
        self.assertFalse(inspect_btn.interactive)
        self.assertFalse(_button(app, "Check weights").interactive)
        self.assertFalse(_button(app, "Create FP16 copy").interactive)
        self.assertFalse(_button(app, "Create BF16 copy").interactive)
        self.assertFalse(_button(app, "Create FP8 copy").interactive)
        self.assertFalse(_button(app, "Create INT8 copy").interactive)
        self.assertFalse(_button(app, "Benchmark pair").interactive)

        inspect_guard = next(
            fn.fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.outputs == [inspect_btn]
        )
        self.assertFalse(inspect_guard(None)["interactive"])
        self.assertTrue(inspect_guard("demo.safetensors")["interactive"])

        inspect_current = next(
            fn.fn
            for fn in app.fns.values()
            if getattr(fn.fn, "__name__", None) == "inspect_current"
        )
        session = ui_checkpoint_tools.CheckpointToolSelection()
        session.capture(None, 1, select=True)
        payload = inspect_current(None, 0, session, 1)
        self.assertIn("Select an installed checkpoint", payload["updates"][0])
        self.assertTrue(all(not item["interactive"] for item in payload["updates"][1:7]))

    def test_model_components_refresh_and_cancel_guards(self):
        with gr.Blocks() as app:
            base = gr.Dropdown(choices=[], value=None)
            ui_model_components.create_ui(base)

        self.assertFalse(_button(app, "Refresh components").interactive)
        self.assertFalse(_button(app, "Download missing").interactive)
        self.assertFalse(_button(app, "Cancel component download").visible)
        self.assertFalse(_button(app, "Cancel component download").interactive)

        callbacks = {
            getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values() if fn.fn is not None
        }
        # sync_component_cards must NOT show Cancel when idle
        cards_payload = callbacks["sync_component_cards"](
            [["VAE", "vae.safetensors", "Missing"]],
            {"vae": {"file": "vae.safetensors", "download_available": True}},
        )
        self.assertIsInstance(cards_payload, dict)
        self.assertEqual(len(cards_payload["items"]), 1)

        dl_start, cancel_start = callbacks["begin_component_download"]()
        self.assertFalse(dl_start["interactive"])
        self.assertTrue(cancel_start["visible"])
        self.assertTrue(cancel_start["interactive"])

        cancel_end = callbacks["end_component_download"]()
        self.assertFalse(cancel_end["visible"])
        self.assertFalse(cancel_end["interactive"])

    def test_adapters_remove_button_tracks_active_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "active.safetensors").write_bytes(b"lora1")
            (root / "inactive.safetensors").write_bytes(b"lora2")
            embeddings = root / "embeddings"
            embeddings.mkdir()
            preview = root / "preview.png"
            Image.new("RGB", (8, 8)).save(preview)
            models = SimpleNamespace(
                get_names=lambda category: ["active.safetensors", "inactive.safetensors"],
                get_file=lambda category, name: root / name,
            )
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "ruinedfooocus.core.util": SimpleNamespace(
                            _get_model_thumbnail=lambda *args, **kwargs: None
                        )
                    },
                ),
                patch.object(ui_adapters, "t", side_effect=lambda text: text),
            ):
                with gr.Blocks() as app:
                    prompt = gr.Textbox(value="")
                    negative = gr.Textbox(value="")
                    controls = ui_adapters.create_ui(
                        prompt,
                        negative,
                        models=models,
                        directories=lambda: [embeddings],
                        cache_dir=root / "cache",
                        defaults=[("active.safetensors", 0.8)],
                        thumbnail=lambda name: str(preview),
                        keywords=lambda name: "",
                    )

            sync_remove = next(
                fn.fn
                for fn in app.fns.values()
                if getattr(fn.fn, "__name__", None) == "sync_remove_button"
            )
            gallery = [(None, "0.8 - active.safetensors")]
            self.assertFalse(sync_remove(controls.catalog.value, None, gallery, "", "")["interactive"])
            self.assertFalse(
                sync_remove(
                    controls.catalog.value, ("LoRA", "inactive.safetensors"), gallery, "", ""
                )["interactive"]
            )
            self.assertTrue(
                sync_remove(
                    controls.catalog.value, ("LoRA", "active.safetensors"), gallery, "", ""
                )["interactive"]
            )

    def test_download_default_wraps_failures_in_actionable_gradio_error(self):
        fake_shared = SimpleNamespace(
            path_manager=SimpleNamespace(
                get_folder_file_path=Mock(side_effect=OSError("disk full"))
            ),
            models=SimpleNamespace(update_all_models=Mock()),
        )
        with (
            patch.object(ui_download, "offline", return_value=False),
            patch.dict("sys.modules", {"ruinedfooocus.app.state": fake_shared}),
            patch("ruinedfooocus.app.state", fake_shared, create=True),
        ):
            with self.assertRaisesRegex(gr.Error, "Could not download the default image model.*disk full"):
                ui_download.download_default()

    def test_popular_models_download_guard_respects_offline_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = SimpleNamespace(
                model_paths={"cache_path": folder, "checkpoints_path": [folder]}
            )
            models = SimpleNamespace(update_all_models=Mock())
            with gr.Blocks() as app:
                ui_popular.create_ui(manager, models)

            callbacks = {
                getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values() if fn.fn is not None
            }
            entry = {
                "key": "civitai:1",
                "name": "Demo",
                "version": "v1",
                "source": "Civitai",
                "base": "SDXL",
                "category": "checkpoints",
                "downloads": 100,
                "likes": 10,
                "file": "demo.safetensors",
                "bytes": 1024,
                "page": "https://civitai.com/models/1",
            }
            evt = gr.EventData(None, {"key": "civitai:1"})
            with patch.object(ui_popular, "offline", return_value=True):
                _, _, dl_update = callbacks["select"]([entry], evt)
                self.assertFalse(dl_update["interactive"])
                _, cancel_update, dl_after_cancel = callbacks["cancelled"]([entry], "civitai:1")
                self.assertFalse(cancel_update["visible"])
                self.assertFalse(dl_after_cancel["interactive"])
                with self.assertRaisesRegex(gr.Error, "Offline mode"):
                    list(callbacks["run_download"]([entry], "civitai:1", "All"))


if __name__ == "__main__":
    unittest.main()
