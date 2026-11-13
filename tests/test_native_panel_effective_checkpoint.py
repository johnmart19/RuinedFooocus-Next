"""Specialist panels follow actual PNG-aware app composition and retain valid drafts."""

import ast
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import ruinedfooocus.app
from ruinedfooocus.features.generation import layered, native_sampling
from ruinedfooocus.features.generation import ui_layered, ui_native_sampling
from ruinedfooocus.features.models.resource_defaults import load_family_recipe
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


class EffectiveNativePanelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.registered = {}
        self.recipes = {
            "ideogram": load_family_recipe("native-ideogram4-conditional"),
            "chroma": load_family_recipe("native-chromaradiance-radiance-x0"),
        }
        self.recipes["ideogram-alternative"] = self.recipes["ideogram"]
        shared = NS(
            settings=NS(default_settings={}),
            models=NS(get_file=lambda category, name: name),
            path_manager=NS(),
            performance_settings=NS(
                model_recommendations=NS(family_details=lambda name: self.recipes.get(name, {}))
            ),
            add_ctrl=lambda key, value: self.registered.update({key: value}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch.object(
                ui_layered, "is_layered_recipe", side_effect=lambda path, _: path == "layered"
            )
        )
        self.enterContext(
            patch.object(
                ui_native_sampling,
                "negative_model_choices",
                side_effect=lambda _, name: ["unconditional.safetensors"],
            )
        )
        source = ROOT / "ruinedfooocus/ui/app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        project = function_from(
            "ruinedfooocus/ui/app.py",
            "update_effective_image_checkpoint",
            dict(gr=gr, settings={"preset_mode": 4}),
        )
        with gr.Blocks(analytics_enabled=False) as self.app:
            self.base = gr.Textbox(value="ordinary")
            self.preset = gr.Textbox(value="")
            self.effective = gr.Textbox(value="ordinary")
            self.metadata_refresh = gr.Number(0)
            namespace = dict(
                gr=gr,
                base_model=self.base,
                preset_selection=self.preset,
                effective_image_checkpoint=self.effective,
                model_metadata_refresh=self.metadata_refresh,
                model_refresh=gr.Button(),
                update_effective_image_checkpoint=project,
                create_native_sampling_ui=ui_native_sampling.create_ui,
                create_layered_ui=ui_layered.create_ui,
            )
            for name in (
                "update_effective_image_checkpoint",
                "create_native_sampling_ui",
                "create_layered_ui",
            ):
                # Instantiate the real app declarations, not a separately wired fixture.
                call = next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and (
                        isinstance(node.func, ast.Name)
                        and node.func.id == name
                        or any(
                            item.arg == "fn"
                            and isinstance(item.value, ast.Name)
                            and item.value.id == name
                            for item in node.keywords
                        )
                    )
                )
                eval(compile(ast.Expression(call), str(source), "eval"), namespace)
        self.state = SessionState(self.app)
        self.project_id = next(key for key, fn in self.app.fns.items() if fn.fn is project)
        self.sampling_id = self.context_id(ui_native_sampling)
        self.layered_id = self.context_id(ui_layered)

    def context_id(self, module):
        return next(
            key
            for key, fn in self.app.fns.items()
            if fn.fn.__module__ == module.__name__ and fn.fn.__name__ == "context"
        )

    def invoke(self, key, values):
        return asyncio.run(self.app.process_api(key, values, state=self.state))["data"]

    def png(self, model):
        path = self.root / (model + ".png")
        metadata = PngInfo()
        metadata.add_text(
            "parameters", json.dumps(dict(software="RuinedFooocus", base_model_name=model))
        )
        Image.new("RGB", (8, 8)).save(path, pnginfo=metadata)
        return str(path)

    def effective_model(self, model):
        return self.invoke(self.project_id, ["ordinary", self.png(model)])[0]["value"]

    def test_real_app_panels_observe_effective_checkpoint_and_metadata_refresh(self):
        for key in (self.sampling_id, self.layered_id):
            function = self.app.fns[key]
            self.assertIs(function.inputs[0], self.effective)
            self.assertEqual(
                set(function.targets),
                {(self.effective._id, "change"), (self.metadata_refresh._id, "change")},
            )
        self.assertEqual(
            set(self.app.fns[self.project_id].targets),
            {
                (self.base._id, "change"),
                (self.preset._id, "change"),
                (self.metadata_refresh._id, "change"),
            },
        )
        self.assertEqual(self.effective_model("ideogram"), "ideogram")
        self.assertEqual(self.base.value, "ordinary")

    def test_png_model_replay_retains_only_compatible_native_parameters_and_choices(self):
        selected = self.effective_model("ideogram")
        first = self.invoke(self.sampling_id, [selected, "ordinary", None, None, None, None, None])
        self.assertTrue(first[0]["visible"])
        self.assertTrue(first[1]["visible"])
        self.assertIn(
            ["unconditional.safetensors", "unconditional.safetensors"], first[1]["choices"]
        )
        selected = self.effective_model("ideogram-alternative")
        for previous in ("ideogram", selected):
            changed = self.invoke(
                self.sampling_id,
                [selected, previous, "unconditional.safetensors", None, None, 0.75, 1.5],
            )
            self.assertEqual(changed[1]["value"], "unconditional.safetensors")
            self.assertEqual([value["value"] for value in changed[4:6]], [0.75, 1.5])
        plan = native_sampling.plan_from_recipe(self.recipes[selected])
        effective = native_sampling.apply_options(
            plan, dict(native_schedule_mu=0.75, native_schedule_std=1.5)
        )
        self.assertEqual(dict(effective.parameters), dict(mu=0.75, std=1.5))
        selected = self.effective_model("chroma")
        incompatible = self.invoke(
            self.sampling_id,
            [selected, "ideogram-alternative", "unconditional.safetensors", None, None, 0.75, 1.5],
        )
        self.assertFalse(incompatible[1]["visible"])
        self.assertIsNone(incompatible[1]["value"])
        self.assertTrue(all(value["value"] is None for value in incompatible[4:6]))
        ordinary = self.invoke(self.sampling_id, ["ordinary", selected, None, 0.4, 0.4, None, None])
        self.assertFalse(ordinary[0]["visible"])
        self.assertTrue(all(value["value"] is None for value in ordinary[2:6]))

    def test_png_layered_context_keeps_uploaded_source_when_hidden_and_shown(self):
        image = Image.new("RGBA", (13, 9), (11, 22, 33, 44))
        selected = self.effective_model("layered")
        shown = self.invoke(self.layered_id, [selected, "ordinary"])
        self.assertTrue(shown[0]["visible"])
        self.assertNotIn("value", shown[1])
        hidden = self.invoke(self.layered_id, ["ordinary", selected])
        self.assertFalse(hidden[0]["visible"])
        self.assertNotIn("value", hidden[1])
        draft = dict(qwen_layer_source=image, qwen_layer_count=3)
        inactive = layered.capture(draft, NS(), None, family="Flux")
        self.assertNotIn("qwen_layer_source", inactive)
        self.assertIs(draft["qwen_layer_source"], image)
        shown_again = self.invoke(self.layered_id, [selected, "ordinary"])
        self.assertTrue(shown_again[0]["visible"])
        self.assertNotIn("value", shown_again[1])


if __name__ == "__main__":
    unittest.main()
