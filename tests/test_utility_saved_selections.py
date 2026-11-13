"""Hidden utility fields preserve saved identities without admitting missing files."""

import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

import ruinedfooocus.app
from ruinedfooocus.features.image_processing import native_seedvr2, ui_native_seedvr2, ui_wevisone


class UtilitySavedSelectionTests(unittest.TestCase):
    def fixture(self, saved):
        return SimpleNamespace(
            settings=SimpleNamespace(default_settings=saved),
            path_manager=SimpleNamespace(get_model_folders=lambda _: []),
            add_ctrl=Mock(),
        )

    def test_seedvr2_hidden_initial_and_refresh_preserve_missing_model_and_codec(self):
        saved = {
            "seedvr2_model": "missing-model.safetensors",
            "seedvr2_vae": "missing-codec.safetensors",
        }
        shared = self.fixture(saved)
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            gr.Blocks(analytics_enabled=False) as app,
        ):
            selection, kind, refresh = (
                gr.Dropdown(["None"], value="None"),
                gr.Textbox(""),
                gr.Number(),
            )
            controls = ui_native_seedvr2.create_ui(selection, kind, refresh, lambda *_: "none")
        for name, value in saved.items():
            self.assertEqual(controls[name].preprocess(value), value)
            self.assertFalse(controls[name].allow_custom_value)
        function = next(
            fn
            for fn in app.fns.values()
            if fn.inputs == [controls["seedvr2_model"], controls["seedvr2_vae"]]
        )
        state = SessionState(app)
        asyncio.run(app.process_api(function, list(saved.values()), state=state))
        for name, value in saved.items():
            self.assertEqual(
                state.blocks_config.blocks[controls[name]._id].preprocess(value), value
            )

    def test_inactive_wevisone_preserves_selected_value_without_discovery(self):
        selected = "saved-image-encoder.safetensors"
        shared = self.fixture({"wevisone_model": selected})
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(
                ui_wevisone.wevisone,
                "model_choices",
                side_effect=AssertionError("Inactive navigation must not scan models"),
            ),
            gr.Blocks(analytics_enabled=False) as app,
        ):
            selection, kind, refresh = (
                gr.Dropdown(["None"], value="None"),
                gr.Textbox(""),
                gr.Number(),
            )
            controls = ui_wevisone.create_ui(selection, kind, refresh, lambda *_: "none")
            self.assertEqual(controls["wevisone_model"].preprocess(selected), selected)
            function = next(fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "context")
            result = function.fn("None", "", selected)
            self.assertIn((selected, selected), result[1]["choices"])
        self.assertFalse(controls["wevisone_model"].allow_custom_value)

    def test_saved_missing_seedvr2_identity_is_still_rejected_by_execution_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = SimpleNamespace(get_model_folders=lambda _: [root])
            with self.assertRaises((ValueError, OSError)):
                native_seedvr2.capture_assets(
                    {
                        "input_image": Image.new("RGB", (8, 8)),
                        "seedvr2_model": str(root / "missing.safetensors"),
                        "seedvr2_vae": str(root / "missing-vae.safetensors"),
                    },
                    manager,
                )
