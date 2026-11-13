"""Actual Gradio/API/replay transport for local reference adapters, without models."""

import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
import ruinedfooocus.app
from ruinedfooocus.features.image_editing import ui_reference_adapters as ui
from ruinedfooocus.features.image_editing import reference_adapters as adapters


class ReferenceAdapterUI(unittest.TestCase):
    def build(self, defaults=None, family="Flux", main_family=None, files=("compatible",)):
        registered = {}
        shared = NS(
            settings=NS(default_settings=defaults or {}),
            path_manager=object(),
            add_ctrl=lambda key, obj: registered.update({key: obj}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda name: name,
            )
        )
        self.enterContext(
            patch.object(
                ui,
                "local_choices",
                side_effect=lambda manager, mode, role: list(files) if mode != "none" else [],
            )
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            base = gr.Textbox(value=family)
            main = gr.Textbox(value=main_family) if main_family else None
            refresh = gr.Number(value=0)
            self.effective, self.main = base, main
            controls = ui.create_ui(base, main_model=main, model_refresh=refresh)
        return demo, controls, registered

    def test_preset_effective_checkpoint_not_main_picker_owns_modes_and_actual_callback(self):
        demo, controls, _ = self.build(
            dict(reference_adapter="redux"), family="Flux", main_family="SDXL"
        )
        index = next(key for key, fn in demo.fns.items() if fn.fn.__name__ == "context")
        self.assertIs(demo.fns[index].inputs[0], self.effective)
        self.assertIsNot(demo.fns[index].inputs[0], self.main)
        self.assertEqual(controls["reference_adapter"].value, "redux")
        result = asyncio.run(demo.process_api(index, ["SDXL", "redux"], state=None))
        self.assertEqual(result["data"][1]["value"], "none")
        self.assertEqual(
            [value for _, value in result["data"][1]["choices"]], ["none", "photomaker"]
        )

    def test_native_torch_photomaker_options_are_explicitly_unqualified_in_ui(self):
        _, controls, _ = self.build(
            dict(reference_adapter="photomaker", reference_adapter_model="local.bin"),
            family="SDXL",
            files=("local.bin",),
        )
        self.assertIn(
            ("local.bin · Compatibility unknown", "local.bin"),
            controls["reference_adapter_model"].choices,
        )

    def test_style_redux_torch_model_and_vision_choices_are_unknown_in_actual_ui(self):
        for mode, family in (("style", "SD 1.5"), ("redux", "Flux")):
            with self.subTest(mode=mode):
                _, controls, _ = self.build(
                    dict(reference_adapter=mode), family=family, files=("local.PT2",)
                )
                for role in ("model", "vision"):
                    self.assertIn(
                        ("local.PT2 · Compatibility unknown", "local.PT2"),
                        controls["reference_adapter_" + role].choices,
                    )

    def test_actual_tool_context_hides_adapter_and_restores_without_resetting_choices(self):
        demo, controls, _ = self.build(dict(reference_adapter="redux"))
        with demo:
            selection = gr.Textbox(value="Custom...")
            kind = gr.Textbox(value="Control")
            inpaint = gr.Checkbox(value=False)
            ui.bind_tool_context(
                controls,
                self.effective,
                selection,
                kind,
                inpaint,
                lambda selected, tool: tool.lower() if selected == "Custom..." else selected,
            )
        index = next(key for key, fn in demo.fns.items() if fn.fn.__name__ == "adapter_context")
        for tool, masked, visible in (
            ("Control", False, False),
            ("Img2img", True, False),
            ("Img2img", False, True),
            ("Rembg", False, False),
        ):
            result = asyncio.run(
                demo.process_api(index, ["Flux", "Custom...", tool, masked], state=None)
            )
            self.assertEqual(result["data"][0]["visible"], visible)
        self.assertEqual(controls["reference_adapter"].value, "redux")
        self.assertEqual(len(demo.fns[index].outputs), 1)

    def test_inactive_ui_capture_omits_adapter_but_explicit_api_keeps_rejection(self):
        source = dict(
            reference_adapter="redux",
            reference_adapter_model="saved",
            reference_adapter_image=object(),
        )
        for controls, masked in (
            ({"type": "control"}, False),
            ({"type": "edit"}, False),
            ({"type": "img2img"}, True),
        ):
            data = dict(source, inpaint_toggle=masked)
            captured = adapters.omit_inactive_ui_reference(data, controls)
            self.assertNotIn("reference_adapter", captured)
            self.assertNotIn("reference_adapter_image", captured)
            self.assertEqual(data["reference_adapter_model"], "saved")
            self.assertEqual(
                adapters.omit_inactive_ui_reference(dict(data, task_type="tool_call"), controls)[
                    "reference_adapter"
                ],
                "redux",
            )
        self.assertEqual(
            adapters.omit_inactive_ui_reference(source, {"type": "img2img"})[
                "reference_adapter_model"
            ],
            "saved",
        )

    def test_saved_local_choices_zero_and_separate_reference_image_are_registered(self):
        _, controls, registered = self.build(
            dict(
                reference_adapter="redux",
                reference_adapter_model="missing",
                reference_adapter_vision="compatible",
                reference_adapter_strength=0,
            )
        )
        self.assertEqual(controls["reference_adapter"].value, "redux")
        self.assertEqual(controls["reference_adapter_strength"].value, 0)
        self.assertEqual(controls["reference_adapter_model"].value, "missing")
        self.assertIn(
            ("Selected file · Unavailable: missing", "missing"),
            controls["reference_adapter_model"].choices,
        )
        self.assertIsNone(controls["reference_adapter_image"].value)
        self.assertNotIn("input_image", registered)
        self.assertTrue(set(adapters.FIELDS).issubset(registered))

    def test_actual_process_api_model_transition_clears_incompatible_reference_without_loading(
        self,
    ):
        demo, controls, _ = self.build(dict(reference_adapter="redux"))
        index = next(key for key, fn in demo.fns.items() if fn.fn.__name__ == "context")
        result = asyncio.run(demo.process_api(index, ["SDXL", "redux"], state=None))
        self.assertEqual(result["data"][1]["value"], "none")
        self.assertEqual(result["data"][2:5], ["", "", None])
        self.assertEqual(
            [value for _, value in result["data"][1]["choices"]], ["none", "photomaker"]
        )
        controls["reference_adapter"].choices = [("None", "none"), ("PhotoMaker", "photomaker")]
        self.assertEqual(controls["reference_adapter"].preprocess("redux"), "none")
        with self.assertRaises(gr.Error):
            controls["reference_adapter"].preprocess("arbitrary-remote-adapter")

    def test_refresh_preserves_missing_explicit_files_and_does_not_choose_another(self):
        demo, _, _ = self.build()
        callback = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "update_mode")
        model, vision, *_ = callback("redux", "missing-model", "missing-vision")
        self.assertIn(
            ("Selected file · Unavailable: missing-model", "missing-model"), model["choices"]
        )
        self.assertIn(
            ("Selected file · Unavailable: missing-vision", "missing-vision"), vision["choices"]
        )
        self.assertNotIn("value", model)
        self.assertNotIn("value", vision)

    def test_real_api_transports_explicit_reference_image_and_never_inherits_ui_image(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from ruinedfooocus.features.generation import image_preparation as assets
        from test_api_image_edits import queued_data

        payload = ImageRequest(
            reference_adapter="photomaker",
            reference_adapter_model="native.bin",
            reference_adapter_image="adapter-upload",
            reference_adapter_strength=1,
        ).model_dump(exclude_none=True)
        seen = []
        with patch.object(
            adapters, "capture_assets", side_effect=lambda data, *a: (seen.append(data) or data, ())
        ):
            queued = queued_data("SDXL", payload_options=payload)
            ordinary = queued_data("SDXL")
        self.assertEqual(queued["reference_adapter"], "photomaker")
        self.assertEqual(queued["reference_adapter_model"], "native.bin")
        self.assertEqual(queued["reference_adapter_image"].size, (64, 64))
        self.assertEqual(ordinary["reference_adapter"], "none")
        self.assertIsNone(ordinary["reference_adapter_image"])
        self.assertEqual(len(seen), 2)

    def test_default_and_actual_gallery_replay_keep_choices_but_never_recreate_upload(self):
        from ruinedfooocus.features.generation.defaults import generation_defaults
        import test_image_metadata_boundary as boundary

        metadata = dict(
            prompt="literal",
            reference_adapter="redux",
            reference_adapter_model="local.safetensors",
            reference_adapter_vision="siglip.safetensors",
            reference_adapter_strength=0,
        )
        saved = generation_defaults(metadata)
        self.assertTrue(set(adapters.FIELDS).issubset(saved))
        self.assertEqual(saved["reference_adapter_strength"], 0)
        updates = boundary.gallery(metadata)
        self.assertEqual(updates["reference_adapter"]["value"], "redux")
        self.assertIsNone(updates["reference_adapter_image"]["value"])
        self.assertEqual(
            boundary.gallery({"prompt": "legacy"})["reference_adapter"]["value"], "none"
        )


if __name__ == "__main__":
    unittest.main()
