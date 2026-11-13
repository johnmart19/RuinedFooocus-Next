"""Control-model choices bind deliberate local selection without callback cycles."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import ruinedfooocus.app
import ruinedfooocus.features.image_editing as editing

from ruinedfooocus.features.image_editing import ui_control_model as ui
from test_control_model_selection import header_file


class ControlModelUI(unittest.TestCase):
    def build(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary).resolve()
        self.local = header_file(
            root / "xl.safetensors",
            {
                "lora_controlnet": [],
                "input_blocks.1.1.transformer_blocks.0.attn2.to_k.down": [128, 2048],
            },
        )
        header_file(root / "wrong.safetensors", {"controlnet_blocks.0.linear.weight": [3072, 3072]})
        (root / "custom.pth").write_bytes(b"not-loaded")
        shared = SimpleNamespace(
            path_manager=SimpleNamespace(get_model_folders=lambda category: [root]),
            settings=SimpleNamespace(default_settings={}),
            add_ctrl=lambda key, component: None,
        )
        controlnet = SimpleNamespace(NEWCN="Custom...", cn_options={"Canny": {"type": "canny"}})
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "ruinedfooocus.features.image_editing.controlnet": controlnet,
                },
            )
        )
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(patch.object(editing, "controlnet", controlnet, create=True))
        # This suite checks compatibility decisions; the browser resolves native
        # translation markers separately. Keep its literal English assertions.
        self.enterContext(patch.object(ui, "t", side_effect=lambda text: text))
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda name: name,
            )
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            selection = gr.Dropdown(choices=["Custom...", "Canny"], value="Canny")
            kind = gr.Dropdown(choices=["Canny"], value="Canny")
            checkpoint = gr.Textbox(value="SDXL")
            self.model_refresh = gr.Number(0)
            model, proof = ui.create_ui(selection, kind, checkpoint, self.model_refresh)
            self.source_controls = [checkpoint, selection, kind]
        return demo, model, proof

    def test_every_native_context_binding_tracks_no_layout_and_keeps_exact_role_graph(self):
        import asyncio

        demo, model, proof = self.build()
        context = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "context")
        bindings = [fn for fn in demo.fns.values() if fn.outputs == context.outputs]
        self.assertEqual(len(bindings), 2)
        expected_inputs = [*self.source_controls, model, proof]
        self.assertEqual(context.inputs, expected_inputs)
        self.assertEqual(
            set(context.targets),
            {
                (component._id, "change")
                for component in [*self.source_controls, proof, self.model_refresh]
            },
        )
        self.assertIsInstance(context.outputs[0], gr.Group)
        self.assertIs(context.outputs[1], model)
        self.assertIsInstance(context.outputs[2], gr.Markdown)
        config = {row["id"]: row for row in demo.get_config_file()["dependencies"]}
        for function in bindings:
            self.assertEqual(function.inputs, expected_inputs)
            self.assertEqual(config[function._id]["show_progress"], "hidden")
            self.assertEqual(config[function._id]["show_progress_on"], [])
            result = asyncio.run(
                demo.process_api(
                    function._id, ["SDXL", "Canny", "Canny", "", None], state=SessionState(demo)
                )
            )
            self.assertEqual(len(result["data"]), 3)
            self.assertTrue(result["data"][0]["visible"])
            self.assertIn("published", result["data"][2])
        refresh = next(function for function in bindings if function is not context)
        self.assertEqual(len(refresh.targets), 1)
        self.assertEqual(refresh.targets[0][1], "click")

    def test_context_filters_compatibility_and_unknown_is_explicit_not_automatic(self):
        demo, model, proof = self.build()
        context = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "context")
        panel, choice, status = context("SDXL", "Canny", "Canny", "", None)
        self.assertTrue(panel["visible"])
        choices = dict(choice["choices"])
        self.assertIn("xl.safetensors", choices.values())
        self.assertNotIn("wrong.safetensors", choices.values())
        self.assertIn("custom.pth · Compatibility unknown", choices)
        self.assertEqual(choice["value"], "")
        self.assertIn("published", status)
        bind = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "bind_selection")
        signature = bind("SDXL", "custom.pth")
        self.assertIn(
            "Compatibility unknown", context("SDXL", "Canny", "Canny", "custom.pth", signature)[2]
        )
        with self.assertRaises(gr.Error):
            bind("SDXL", "wrong.safetensors")
        self.assertFalse(model.allow_custom_value)
        self.assertIsNone(proof.value)

    def test_refresh_preserves_identity_and_graph_does_not_trigger_itself(self):
        demo, model, proof = self.build()
        context = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "context")
        bind = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "bind_selection")
        signature = bind("SDXL", self.local.name)
        self.local.write_bytes(self.local.read_bytes() + b"changed")
        result = context.fn(
            "SDXL", "Canny", "Canny", self.local.name, signature, force_refresh=True
        )
        self.assertEqual(result[1]["value"], self.local.name)
        self.assertIn("changed", result[2])
        self.assertNotIn(proof, context.outputs)
        self.assertNotIn((model._id, "change"), context.targets)
        self.assertIn((proof._id, "change"), context.targets)

    def test_non_xl_native_processors_keep_their_strength_control_and_guide_label(self):
        from test_powerup_family_controls import functions

        scope, models = functions()
        scope["controlnet"].cn_options["Raw guide"] = {"type": "control"}
        for family in ("SD 1.5", "SD 2.0", "Flux", "SD3", "QwenImage", "HunyuanDiT"):
            models.detect_checkpoint_base.return_value = family
            for preset in ("Canny", "Depth", "Raw guide"):
                with self.subTest(family=family, preset=preset):
                    result = scope["input_context"](False, preset, "Canny", "selected")
                    self.assertTrue(result[1]["visible"])
                    self.assertEqual(result[0]["label"], "Control guide")


if __name__ == "__main__":
    unittest.main()
