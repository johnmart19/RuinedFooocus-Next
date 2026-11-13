"""Real installed Gradio binary controls, numeric mount and contextual callbacks."""

import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.data_classes import FileData
from gradio.state_holder import SessionState

import ruinedfooocus.app
from ruinedfooocus.features.image_processing import latent_composition as composition
from ruinedfooocus.features.image_processing import ui_latent_composition as ui


class LatentCompositionUI(unittest.TestCase):
    def setUp(self):
        self.registered = {}
        shared = NS(
            settings=NS(default_settings=dict(latent_composition_x=0, latent_composition_y=16)),
            add_ctrl=lambda name, obj: self.registered.update({name: obj}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        with gr.Blocks(analytics_enabled=False) as self.demo:
            selection = gr.Textbox(value="Compose saved latents")
            kind = gr.Textbox(value=composition.KIND)
            ui.create_ui(selection, kind, lambda selected, value: value)
        self.state = SessionState(self.demo)

    def call(self, predicate, values, *, state=None):
        index = next(index for index, fn in self.demo.fns.items() if predicate(fn))
        return asyncio.run(self.demo.process_api(index, values, state=state or self.state))["data"]

    def test_real_binary_uploads_and_integer_controls_mount_with_empty_inputs_and_exact_zero(self):
        self.assertEqual(
            set(self.registered), set((*composition.INPUTS, composition.MASK, *composition.FIELDS))
        )
        for key in composition.INPUTS:
            self.assertEqual(self.registered[key].type, "binary")
            self.assertIsNone(self.registered[key].value)
        for key in composition.FIELDS[1:]:
            self.assertIsInstance(self.registered[key], gr.Number)
            self.assertEqual(self.registered[key].precision, 0)
        self.assertEqual(self.registered["latent_composition_x"].value, 0)
        self.assertEqual(self.registered["latent_composition_y"].value, 16)

    def test_actual_mode_and_tool_callbacks_only_change_the_relevant_visibility(self):
        mode = lambda fn: fn.fn.__name__ == "operation_context"
        self.assertEqual([v["visible"] for v in self.call(mode, ["masked"])], [False, True])
        self.assertEqual([v["visible"] for v in self.call(mode, ["replace"])], [True, False])
        context = lambda fn: fn.fn.__name__ == "<lambda>" and len(fn.inputs) == 2
        self.assertTrue(self.call(context, ["unused", composition.KIND])[0]["visible"])
        self.assertFalse(self.call(context, ["unused", "img2img"])[0]["visible"])
        self.assertEqual(self.registered["latent_composition_y"].value, 16)

    def test_binary_preprocessing_retains_original_bytes_and_clear_owns_only_its_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "owned.latent"
            original = b"owned transport fixture, not a model"
            path.write_bytes(original)
            for key in composition.INPUTS:
                self.assertEqual(
                    self.registered[key].preprocess(FileData(path=str(path), orig_name=path.name)),
                    original,
                )
            clear = lambda fn: fn.fn.__name__ == "<lambda>" and len(fn.inputs) == 0
            self.assertEqual(self.call(clear, []), [None, None, None])
            self.assertEqual(path.read_bytes(), original)
        fresh = SessionState(self.demo)
        self.assertEqual(self.call(clear, [], state=fresh), [None, None, None])

    def test_inactive_form_values_stay_in_the_form_and_only_scalars_persist(self):
        from ruinedfooocus.features.generation.defaults import generation_defaults
        from ruinedfooocus.core.settings_schema import validate_known

        form = dict(
            latent_composition_mode="masked",
            latent_composition_feather=16,
            latent_composition_primary=b"keep",
            latent_composition_secondary=b"keep",
        )
        active = composition.ui_request(form)
        self.assertEqual(active["latent_composition_feather"], 0)
        self.assertEqual(form["latent_composition_feather"], 16)
        values = dict(
            composition.DEFAULTS,
            latent_composition_x=0,
            latent_composition_primary=b"private",
            latent_composition_secondary=b"private",
            latent_composition_mask=object(),
        )
        saved = generation_defaults(values)
        self.assertEqual(saved["latent_composition_x"], 0)
        for key in (*composition.INPUTS, composition.MASK):
            self.assertNotIn(key, saved)
        with self.assertRaises(ValueError):
            validate_known({"latent_composition_x": True})


if __name__ == "__main__":
    unittest.main()
