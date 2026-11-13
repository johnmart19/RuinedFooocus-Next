import ast
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ruinedfooocus.app  # The fake state module below still needs its real parent package.
from ruinedfooocus.features.image_editing import ui_guidance as ui


class Widget:
    def __init__(self, *args, **kwargs):
        self.value = kwargs.get("value", args[0] if args else None)
        self.properties = kwargs
        self.handlers = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def change(self, fn, **kwargs):
        self.handlers.append((fn, kwargs))

    def input(self, fn, **kwargs):
        self.handlers.append((fn, kwargs))


class GuidanceUI(unittest.TestCase):
    def test_checkpoint_transition_preserves_restored_qwen_schedule_only_for_qwen(self):
        from ruinedfooocus.features.models.model_recommendations import QWEN_IMAGE21_TURBO_SIGMAS

        restored = ui.restore_guidance_metadata(
            {"effective_sampling_sigmas": list(QWEN_IMAGE21_TURBO_SIGMAS)}, family="QwenImage21"
        )
        transition = ui.preset_control_state(
            restored,
            {},
            {},
            7,
            "QwenImage21",
            "turbo",
            "turbo",
            {"checkpoint": "base", "backup": None},
        )
        self.assertEqual(transition["values"]["qwen_image21_schedule"], "turbo_8")
        changed = ui.preset_control_state(
            transition["values"], {}, {}, 7, "SDXL", "sdxl", "sdxl", transition["state"]
        )
        self.assertEqual(changed["values"]["qwen_image21_schedule"], "auto")
        self.assertEqual(ui.qwen_schedule_for_family("invalid", "QwenImage21"), "auto")

    def test_actual_refresh_callback_keeps_utility_guidance_hidden_and_preserves_values(self):
        import gradio as gr
        from gradio.state_holder import SessionState

        registered = {}
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={"cfg_norm_strength": 2}),
            add_ctrl=lambda key, obj: registered.update({key: obj}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ui, "family_for_checkpoint", return_value="Flux"),
            patch.object(ui, "read_preset_metadata", return_value={}),
            patch.object(ui, "effective_preset_checkpoint", side_effect=lambda name, *args: name),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                model = gr.Textbox(value="same checkpoint")
                preset = gr.Textbox(value="")
                refresh = gr.Number(value=0)
                selection = gr.Textbox(value="Segment by points")
                kind = gr.Textbox(value="segment")
                controls = ui.create_ui(
                    model,
                    preset_selection=preset,
                    main_model=model,
                    model_refresh=refresh,
                    tool_selection=selection,
                    tool_type=kind,
                    kind_for=lambda choice, kind: kind,
                )
            index = next(key for key, fn in demo.fns.items() if fn.fn.__name__ == "context_changed")
            fn = demo.fns[index]
            self.assertIn((refresh._id, "change"), fn.targets)
            self.assertNotIn(selection, fn.outputs)
            self.assertNotIn(model, fn.outputs)
            values = [controls[key].value for key in ui.FIELD_SCHEMA]
            values[list(ui.FIELD_SCHEMA).index("cfg_norm_strength")] = "2"
            state = SessionState(demo)
            for tool in ("segment", "detect", "rembg", "upscale", "img2img"):
                result = asyncio.run(
                    demo.process_api(
                        index, [model.value, "", None, *values, selection.value, tool], state=state
                    )
                )["data"]
                self.assertEqual(result[0]["visible"], tool == "img2img")
                norm_index = 3 + list(ui.FIELD_SCHEMA).index("cfg_norm_strength")
                self.assertEqual(result[norm_index]["value"], "2")

    def test_preset_state_hides_controls_restores_backup_and_drops_private_refs(self):
        current = {key: spec["default"] for key, spec in ui.FIELD_SCHEMA.items()}
        current.update(
            sampling_shift=3,
            cfg_norm_strength=2,
            cfg_norm_pre_cfg=True,
            hidream_o1_noise_scale=7.6,
            reference_images=["private"],
        )
        metadata = {
            "software": "RuinedFooocus",
            "moss_image_edit_format": "ai-toolkit",
            "sampling_shift": None,
            "effective_sampling_shift": 5,
        }
        entered = ui.preset_control_state(
            current, {}, metadata, 7, "MOSSImage", "preset-moss", "main-qwen", None
        )
        self.assertTrue(entered["locked"])
        self.assertTrue(entered["clear_references"])
        self.assertEqual(entered["values"]["sampling_shift"], 5)
        self.assertIsNone(entered["values"]["hidream_o1_noise_scale"])
        self.assertNotIn("reference_images", entered["state"]["backup"]["values"])
        restored = ui.preset_control_state(
            entered["values"], {}, {}, 7, "QwenImage", "main-qwen", "main-qwen", entered["state"]
        )
        self.assertFalse(restored["locked"])
        self.assertEqual(restored["values"]["sampling_shift"], 3)
        self.assertEqual(restored["values"]["cfg_norm_strength"], 2)
        self.assertTrue(restored["values"]["cfg_norm_pre_cfg"])
        self.assertEqual(restored["values"]["hidream_o1_noise_scale"], 7.6)
        changed = ui.preset_control_state(
            entered["values"], {}, {}, 7, "SDXL", "new-main", "new-main", entered["state"]
        )
        self.assertIsNone(changed["values"]["sampling_shift"])
        self.assertIsNone(changed["values"]["cfg_norm_strength"])
        self.assertFalse(changed["values"]["cfg_norm_pre_cfg"])
        self.assertIsNone(changed["values"]["hidream_o1_noise_scale"])
        self.assertIsNone(
            ui.preset_control_state(
                current, {}, None, 7, "QwenImage", "bad", "main-qwen", entered["state"]
            )
        )

    def test_reference_visibility_respects_model_and_edit_mode(self):
        self.assertFalse(ui.references_visible("MOSSImage", "Disabled"))
        self.assertTrue(ui.references_visible("MOSSImage", "ai-toolkit"))
        self.assertTrue(ui.references_visible("MageFlow", "Disabled"))
        self.assertTrue(ui.references_visible("Fibo2", "Disabled"))
        self.assertTrue(ui.references_visible("QwenImage", "Disabled"))
        self.assertTrue(ui.references_visible("Flux2Klein9B", "Disabled"))
        self.assertEqual(ui.control_visibility("SDXL"), (False, False, False, True))

    def test_controls_register_and_clear_model_specific_values(self):
        registered = {}
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={}),
            add_ctrl=lambda key, obj: registered.update({key: obj}),
        )
        fake = SimpleNamespace(
            **{
                name: Widget
                for name in (
                    "Accordion",
                    "Group",
                    "Row",
                    "Markdown",
                    "Dropdown",
                    "Slider",
                    "Checkbox",
                    "Textbox",
                    "Gallery",
                )
            },
            update=lambda **values: values,
            on=lambda triggers, fn, **kwargs: [trigger(fn, **kwargs) for trigger in triggers],
        )
        base = Widget(value="moss")
        metadata_refresh = Widget(value=0)
        with (
            patch.object(ui, "gr", fake),
            patch.object(
                ui,
                "family_for_checkpoint",
                side_effect=lambda name: {
                    "moss": "MOSSImage",
                    "sdxl": "SDXL",
                    "qwen21": "QwenImage21",
                }[name],
            ),
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            controls = ui.create_ui(base, model_refresh=metadata_refresh)
            self.assertIs(base.handlers[0][0], metadata_refresh.handlers[0][0])
            self.assertEqual(set(registered), set(controls))
            self.assertTrue(set(ui.FIELD_SCHEMA).issubset(controls))
            self.assertIn("reference_images", registered)
            self.assertFalse(controls["qwen_image21_schedule"].properties["visible"])
            self.assertEqual(controls["qwen_image21_schedule"].value, "auto")
            from ruinedfooocus.features.image_editing.image_guidance import resolve_guidance

            values = {key: controls[key].value for key in ("sampling_shift", "cfg_norm_strength")}
            resolved = resolve_guidance(values, {}, "MOSSImage")
            self.assertIsNone(resolved["sampling_shift"])
            self.assertIsNone(resolved["cfg_norm_strength"])
            changes = base.handlers[0][0]("sdxl", "ai-toolkit")
            self.assertEqual(changes[2], {"visible": False, "value": ""})
            self.assertEqual(changes[3]["value"], "")
            self.assertEqual(changes[4], {"visible": False, "value": []})
            self.assertEqual(changes[9], {"visible": False})
            self.assertEqual(changes[11], {"value": ""})
            self.assertEqual(changes[-1], {"visible": False, "value": "auto"})
            self.assertEqual(
                base.handlers[0][0]("qwen21", "Disabled")[-1], {"visible": True, "value": "auto"}
            )
            self.assertEqual(
                base.handlers[0][0]("qwen21", "Disabled", False, "turbo_8")[-1],
                {"visible": True, "value": "turbo_8"},
            )
            self.assertEqual(
                base.handlers[0][0]("sdxl", "Disabled", False, "turbo_8")[-1],
                {"visible": False, "value": "auto"},
            )
            format_change = controls["moss_image_edit_format"].handlers[0][0]
            self.assertEqual(format_change("moss", "Disabled"), {"visible": False, "value": []})
            self.assertEqual(format_change("moss", "ai-toolkit"), {"visible": True})
            slg_change = controls["moss_image_slg"].handlers[0][0]
            self.assertEqual(slg_change("moss", True), {"visible": True})
            self.assertEqual(slg_change("sdxl", True), {"visible": False})
            self.assertEqual(slg_change("moss", False), {"visible": False})

    def test_api_snapshot_includes_scalar_guidance_but_never_uploaded_references(self):
        source = Path("ruinedfooocus/ui/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        values = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "image_keys" for t in node.targets)
        )
        keys = ast.literal_eval(values)
        self.assertTrue(set(ui.FIELD_SCHEMA).issubset(keys))
        self.assertNotIn("reference_images", keys)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue(
            names.isdisjoint({"perf_name", "perf_save", "performance_add", "performance_delete"})
        )


if __name__ == "__main__":
    unittest.main()
