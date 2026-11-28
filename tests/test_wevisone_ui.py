"""Installed Gradio controls and actual Library/model activation callers."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
import ruinedfooocus.app

from ruinedfooocus.features.image_processing import wevisone, ui_wevisone
from ruinedfooocus.features.image_editing import tool_activation
from ruinedfooocus.features.models.model_identity import file_signature
import test_wevisone as fixtures
from test_utility_sampling_capture import function


class WeVisOneUI(unittest.TestCase):
    def setUp(self):
        self.harness = fixtures.WeVisOneTests("runTest")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.controls = {}
        shared = NS(
            settings=NS(default_settings={}),
            path_manager=self.harness.manager,
            add_ctrl=lambda name, value: self.controls.update({name: value}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        with gr.Blocks(analytics_enabled=False) as self.demo:
            selection = gr.Textbox(value="None")
            kind = gr.Textbox(value="None")
            refresh = gr.Number(value=0)
            ui_wevisone.create_ui(selection, kind, refresh, lambda selected, custom: selected)
        self.session = SessionState(self.demo)

    def context(self, kind, model=None):
        index = next(index for index, fn in self.demo.fns.items() if fn.fn.__name__ == "context")
        return asyncio.run(self.demo.process_api(index, [kind, "None", model], state=self.session))[
            "data"
        ]

    def test_real_mount_registers_picker_text_integer_and_output_controls(self):
        self.assertEqual(set(self.controls), set(wevisone.FIELDS))
        self.assertIsInstance(self.controls["wevisone_query"], gr.Textbox)
        length = self.controls["wevisone_max_length"]
        self.assertIsInstance(length, gr.Number)
        self.assertEqual((length.precision, length.minimum, length.maximum), (0, 1, 32768))
        self.assertEqual(
            {value for _, value in self.controls["wevisone_output"].choices},
            {"cutout", "mask", "overlay"},
        )
        self.assertIsNone(self.controls["wevisone_model"].value)
        matte = self.context("wevisone-matte")
        self.assertTrue(matte[0]["visible"])
        self.assertTrue(matte[2]["visible"])
        self.assertTrue(matte[1]["choices"])
        detect = self.context("wevisone-detect")
        self.assertFalse(detect[2]["visible"])
        self.assertFalse(self.context("edit")[0]["visible"])

    def test_retained_unavailable_picker_label_is_plain_localized_text(self):
        with patch.object(wevisone, "model_choices", return_value=[]):
            rows = ui_wevisone.choices(self.harness.manager, "old.safetensors", locale="uk")
        self.assertEqual(rows[0][1], "old.safetensors")
        self.assertIsInstance(rows[0][0], str)
        self.assertNotIn("__i18n__", rows[0][0])

    def test_actual_activation_callback_updates_real_picker_and_keeps_other_choices(self):
        with gr.Blocks(analytics_enabled=False) as app:
            picker = gr.Dropdown(choices=[], value=None)
            operation = gr.Dropdown(choices=["WeVisOne matting"], value="WeVisOne matting")
            target = gr.Radio(choices=["Generated image"], value="Generated image")
            button = gr.Button()
            controls = dict(
                wevisone_model=picker, cn_selection=operation, automatic_upscale_target=target
            )
            forbidden = Mock(side_effect=AssertionError("generation checkpoint admission reached"))
            activate = function(
                "ruinedfooocus/features/image_editing/ui.py",
                "activate_tool",
                dict(
                    gr=gr,
                    shared=NS(get_ctrl=controls.get, path_manager=self.harness.manager),
                    controlnet=NS(cn_options={"WeVisOne matting": {"type": "wevisone-matte"}}),
                    NEWCN="Custom...",
                    checkpoint_features=forbidden,
                ),
            )

            def use():
                return activate(
                    "wevisone-matte",
                    dict(
                        kind="wevisone-matte",
                        resource="wevisone_model",
                        value=str(self.harness.model),
                    ),
                    checkpoint="unrelated.safetensors",
                )

            event = button.click(use, outputs=list(controls.values()))
        session = SessionState(app)
        result = asyncio.run(app.process_api(event["id"], [], state=session))
        expected = wevisone.model_choices(self.harness.manager)
        self.assertEqual(result["data"][0]["choices"], [list(value) for value in expected])
        actual = session.blocks_config.blocks[picker._id]
        self.assertEqual((actual.value, actual.choices), (str(self.harness.model), expected))
        forbidden.assert_not_called()

    def test_installed_clip_use_activates_exact_model_and_kind_without_checkpoint(self):
        model = self.harness.model
        row = dict(category="clip", local=str(model), local_signature=file_signature(model))
        resource = tool_activation.installed_tool(row, self.harness.manager)
        presets = {label: {"type": kind} for kind, label in wevisone.LABELS.items()}
        values = tool_activation.activation_values(
            "wevisone-matte", resource, presets=presets, custom="Custom"
        )
        self.assertEqual(values["cn_selection"], "WeVisOne matting")
        self.assertEqual(values["wevisone_model"], str(model))
        self.assertNotIn("base_model_name", values)
        self.assertNotIn("cn_type", values)
        model.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            tool_activation.installed_tool(row, self.harness.manager)

    def test_actual_library_replay_clears_source_model_and_generation_tensor_controls(self):
        data = self.harness.capture(wevisone_query="the cat", wevisone_output="overlay")
        wevisone.process(self.harness.owner, data, self.harness.manager, lambda: None)
        metadata = wevisone.output_metadata(data)
        apply = function(
            "ruinedfooocus/features/gallery/ui.py",
            "apply_metadata",
            dict(gr=gr, get_ctrl=lambda key: key),
        )
        result = apply(metadata, None, None)
        self.assertEqual(result["wevisone_query"]["value"], "the cat")
        self.assertEqual(result["wevisone_output"]["value"], "overlay")
        self.assertIsNone(result["wevisone_model"]["value"])
        self.assertIsNone(result["input_image"]["value"])
        self.assertIsNone(result["native_latent_input"]["value"])
        self.assertNotIn("base_model_name", result)

    def test_only_scalar_intent_is_saved_as_startup_defaults(self):
        from ruinedfooocus.features.generation.defaults import generation_defaults
        from ruinedfooocus.core.settings_schema import validate_known

        values = dict(
            wevisone.DEFAULTS,
            wevisone_query="cat",
            wevisone_model="private.safetensors",
            _wevisone_request=object(),
        )
        saved = generation_defaults(values)
        self.assertEqual(saved["wevisone_query"], "cat")
        self.assertNotIn("wevisone_model", saved)
        self.assertNotIn("_wevisone_request", saved)
        for change in (
            dict(wevisone_max_length=True),
            dict(wevisone_query=""),
            dict(wevisone_output="unknown"),
        ):
            with self.assertRaises(ValueError):
                validate_known(dict(wevisone.DEFAULTS, **change))


if __name__ == "__main__":
    unittest.main()
