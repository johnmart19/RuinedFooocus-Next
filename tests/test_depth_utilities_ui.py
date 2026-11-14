"""Real Gradio utility contexts and API events preserve independent controls."""

import asyncio
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import ruinedfooocus.app
from ruinedfooocus.features.image_processing import ui_depth_utilities as ui
from ruinedfooocus.features.image_processing import depth_utilities as utilities
from ruinedfooocus.core.paths import resource_path
from test_utility_sampling_capture import function


class DepthUtilitiesUI(unittest.TestCase):
    def test_provenance_and_missing_file_labels_resolve_real_i18n_before_filename_combination(self):
        from gradio.i18n import I18nData
        from ruinedfooocus.core import translation_manager as translations
        from ruinedfooocus.features.image_processing import utility_marigold

        manager = translations.TranslationManager()
        phrases = {
            "Edit2509 source verified": "Джерело Edit2509 підтверджено",
            "Edit2509 source unverified": "Джерело Edit2509 не підтверджено",
            "Unavailable: {name}": "Недоступно: {name}",
        }
        for text, translated in phrases.items():
            manager.aliases.pop(text, None)
            manager.translations["en"][text] = text
            manager.translations.setdefault("uk", {})[text] = translated
        with patch.object(translations, "catalogue", return_value=manager):
            self.assertIsInstance(translations.translate_ui("Edit2509 source verified"), I18nData)
            for proof, text in (
                ({"verified": True}, "Edit2509 source verified"),
                (None, "Edit2509 source unverified"),
            ):
                with patch.object(utility_marigold, "backbone_proof", return_value=proof):
                    label = utility_marigold.component_label(
                        "image_utility_model", object(), "chosen.safetensors", locale="uk"
                    )
                self.assertEqual(label, "chosen.safetensors (" + phrases[text] + ")")
                self.assertNotIn("__i18n__", label)
            with patch.object(ui, "component_choices", return_value=[]):
                values = ui.choices(
                    object(),
                    "marigold-albedo",
                    "image_utility_model",
                    "missing.safetensors",
                    locale="uk",
                )
            self.assertEqual(values, [("Недоступно: missing.safetensors", "missing.safetensors")])
            self.assertNotIn("__i18n__", values[0][0])

    def build(self, kind, saved=None):
        self.controls = {}
        shared = NS(
            path_manager=object(),
            settings=NS(default_settings=saved or {}),
            add_ctrl=lambda key, control: self.controls.update({key: control}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch.object(
                ui,
                "component_choices",
                side_effect=lambda manager, kind, field, **options: (
                    [(f"{kind}/{field}", f"/{kind}/{field}")] if kind in utilities.KINDS else []
                ),
            )
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            choice = gr.Textbox(value=kind)
            tool = gr.Textbox(value=kind)
            self.image = gr.Image(type="pil", image_mode="RGBA")
            refresh = gr.Number(value=0)
            ui.create_ui(choice, tool, self.image, refresh, lambda choice, kind: kind)
        self.demo, self.state = demo, SessionState(demo)

    def context(self, kind, selected=None):
        index = next(
            index for index, fn in self.demo.fns.items() if fn.fn.__name__ == "utility_context"
        )
        return asyncio.run(
            self.demo.process_api(index, [kind, kind, selected, None, None, None], state=self.state)
        )["data"]

    def test_all_capture_fields_registered_and_explicit_zero_survives(self):
        self.build("face-landmarks", {"face_min_confidence": 0, "image_utility_model": "/missing"})
        self.assertEqual(set(self.controls), set(utilities.FIELDS))
        self.assertEqual(self.controls["face_min_confidence"].value, 0)
        self.assertEqual(self.controls["image_utility_model"].value, "/missing")

    def test_required_model_hint_keeps_a_single_choice_unselected(self):
        self.build("moge-depth")
        model = self.controls["image_utility_model"]
        self.assertEqual(model.info.key, "ruinedfooocus.models.select_a_model_or_component")
        self.assertEqual(len(model.choices), 1)
        self.assertIsNone(model.value)
        update = self.context("moge-depth")[2]
        self.assertEqual(len(update["choices"]), 1)
        self.assertIsNone(update["value"])

    def test_contexts_expose_exact_companions_and_preserve_missing_explicit_choice(self):
        self.build("lotus-depth", {"image_utility_model": "/missing"})
        for kind in utilities.KINDS:
            with self.subTest(kind=kind):
                result = self.context(kind, "/missing")
                self.assertTrue(result[0]["visible"])
                self.assertEqual(result[2]["value"], "/missing")
                self.assertIn("/missing", [value for _, value in result[2]["choices"]])
                self.assertEqual(result[3]["visible"], kind in ("lotus-depth", "marigold-albedo"))
                self.assertEqual(result[4]["visible"], kind == "marigold-albedo")
                self.assertEqual(result[5]["visible"], kind == "marigold-albedo")
                self.assertEqual(
                    [result[index]["visible"] for index in (6, 7, 8)],
                    [kind == "moge-depth", kind == "face-landmarks", kind == "pose-map"],
                )
        self.assertFalse(self.context("depth-map")[0]["visible"])

    def test_context_events_cannot_overwrite_source_and_replay_requires_new_source(self):
        self.build("moge-depth")
        for fn in self.demo.fns.values():
            self.assertNotIn(self.image, fn.outputs)
        for kind in utilities.KINDS:
            replay = utilities.replay_values({"controlnet": {"type": kind}})
            self.assertIsNone(replay["input_image"])
            self.assertIsNone(replay["image_source"])
            self.assertEqual(replay["cn_selection"], ui.LABELS[kind])
            self.assertTrue(all(replay[field] is None for field in utilities.REPLAY_RESETS))

    def test_replay_clears_old_files_before_resolving_supported_tool_defaults(self):
        self.build("moge-depth")
        resets = utilities.replay_values({"controlnet": {"type": "moge-depth"}})
        for kind in utilities.KINDS:
            result = self.context(kind, resets["image_utility_model"])
            self.assertTrue(result[0]["visible"])
            if kind in ("face-landmarks", "pose-map"):
                self.assertEqual(result[2]["value"], f"/{kind}/image_utility_model")
            else:
                self.assertIsNone(result[2]["value"])

    def test_actual_image_tools_host_mounts_with_real_number_model_refresh(self):
        # Gradio otherwise writes generated typing files beside custom classes.
        # Mount verification needs the real class/event system, not that output.
        with patch("gradio.component_meta.create_or_modify_pyi"):
            from ruinedfooocus.features.image_editing.tool_type_dropdown import ToolTypeDropdown
        from ruinedfooocus.features.image_processing import ui_native_vision

        controls = {}
        manager = NS(
            upscaler_filenames=[], DOWNLOADABLE_FILES={}, get_model_folders=lambda kind: []
        )
        shared = NS(
            path_manager=manager,
            settings=NS(default_settings={}),
            add_ctrl=lambda key, control: controls.update({key: control}),
            get_ctrl=controls.get,
        )
        presets = json.loads(
            resource_path("defaults", "powerup.default").read_text(encoding="utf-8")
        )
        cn = NS(
            cn_options=presets,
            controlnet_models=dict.fromkeys(
                [
                    "canny",
                    "depth",
                    "sketch",
                    "recolour",
                    "img2img",
                    "upscale",
                    "rembg",
                    "detect",
                    "segment",
                    "depth-map",
                    *utilities.KINDS,
                ]
            ),
        )
        features = dict(
            sdxl=False,
            control=False,
            inpaint=False,
            qwen_local=False,
            qwen_union=False,
            native_edit=False,
            instruct=False,
            sd4x=False,
        )
        scope = dict(
            gr=gr,
            shared=shared,
            add_ctrl=shared.add_ctrl,
            path_manager=manager,
            controlnet=cn,
            NEWCN="Custom...",
            SDXL_CONTROLS=frozenset(("canny", "depth", "sketch", "recolour")),
            checkpoint_features=lambda *args: features,
            t=lambda value: value,
            translate_ui=lambda value: value,
            ToolTypeDropdown=ToolTypeDropdown,
            OUTPAINT_MAX_MARGIN=2048,
            OUTPAINT_MAX_FEATHER=128,
            OUTPAINT_DEFAULT_FEATHER=32,
        )
        for name in (
            "control_types",
            "input_label",
            "image_tool_help",
            "powerup_choices",
            "tool_kind",
        ):
            function("ruinedfooocus/features/image_editing/ui.py", name, scope)
        mount = function("ruinedfooocus/features/image_editing/ui.py", "add_controlnet_tab", scope)
        noop = lambda *args, **kwargs: None
        # Other feature panels keep real control interfaces; their behavior has
        # independent suites. The host and both native utility panels are real.
        modules = {
            "ruinedfooocus.app.state": shared,
            "ruinedfooocus.features.image_editing.ui_depth_control": NS(create_ui=lambda *args: {}),
            "ruinedfooocus.features.image_editing.ui_control_model": NS(
                create_ui=lambda *args: (gr.Dropdown(choices=[""], value=""), gr.State(None))
            ),
            "ruinedfooocus.features.image_editing.ui_guidance": NS(
                create_ui=lambda *args, **kwargs: {
                    "moss_image_edit_format": gr.Textbox(value="Disabled")
                }
            ),
            "ruinedfooocus.features.image_editing.ui_native_workflows": NS(create_ui=noop),
            "ruinedfooocus.features.image_editing.ui_model_patches": NS(create_ui=noop),
            "ruinedfooocus.features.image_editing.ui_lora_hooks": NS(create_ui=noop),
            "ruinedfooocus.features.image_processing.ui_image_composition": NS(create_ui=noop),
            "ruinedfooocus.features.image_editing.ui_reference_adapters": NS(
                create_ui=lambda *args, **kwargs: {}, bind_tool_context=noop
            ),
            "ruinedfooocus.features.image_editing.ui_ordered_references": NS(create_ui=noop),
            "ruinedfooocus.features.image_editing.ui_qwen_union": NS(create_ui=noop),
            "ruinedfooocus.features.image_processing.ui_native_background": NS(create_ui=noop),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ui, "component_choices", return_value=[]),
            patch.object(ui_native_vision, "coco_classes", return_value=[]),
            patch.object(ui_native_vision, "native_model_choices", return_value=[]),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                for key in ("OBP_preset", "OBP_promptenhance", "OBP_modeltype"):
                    controls[key] = gr.Dropdown(choices=["none"], value="none")
                controls["obp_givensubject"] = gr.Textbox()
                controls["image_source"] = gr.State(None)
                refresh = gr.Number(value=0, precision=0, visible=False)
                mount(
                    gr.Image(),
                    gr.ImageEditor(),
                    gr.Textbox(),
                    gr.Number(value=1),
                    gr.Button(),
                    gr.Dropdown(choices=[""], value=""),
                    model_refresh=refresh,
                )
        self.assertTrue(set(utilities.FIELDS) <= controls.keys())
        utility_event = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "utility_context")
        self.assertIn((refresh._id, "change"), utility_event.targets)
        self.assertNotIn((refresh._id, "click"), utility_event.targets)
        self.assertTrue(
            set(utilities.LABELS.values())
            <= {value for _, value in controls["cn_selection"].choices}
        )


if __name__ == "__main__":
    unittest.main()
