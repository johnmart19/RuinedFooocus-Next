"""Actual Gradio guidance events stay contextual and scan files only on demand."""

import asyncio
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

import ruinedfooocus.app
from ruinedfooocus.features.image_editing import ui_native_auxiliary as ui
from ruinedfooocus.features.image_editing import gligen_layout, native_hypernetwork


class AuxiliaryUITests(unittest.TestCase):
    def build(self, family="SDXL", defaults=None):
        controls = {}
        shared = NS(
            settings=NS(default_settings=defaults or {}),
            path_manager=object(),
            get_ctrl=controls.get,
            add_ctrl=lambda key, component: controls.update({key: component}),
            models=NS(
                get_models_by_path=lambda category, name, **kw: {"family": name},
                get_model_base=lambda info: info.get("family"),
            ),
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "ruinedfooocus.features.image_editing.ui": NS(
                        tool_kind=lambda selected, kind: (
                            kind if selected == "Custom..." else selected
                        )
                    ),
                },
            )
        )
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        scans = self.enterContext(
            patch.object(
                ui,
                "local_choices",
                side_effect=lambda manager, role: (
                    ["layout.pt"] if role == "gligen" else ["attention.pt"]
                ),
            )
        )
        selection = self.enterContext(
            patch.object(
                ui,
                "selected_asset",
                return_value=NS(
                    public_signature=("layout.pt", 40, "1000000000000000001", "1000000000000000002")
                ),
            )
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            base = gr.Textbox(value=family)
            controls["cn_selection"], controls["cn_type"] = (
                gr.Textbox(value="None"),
                gr.Textbox(value="None"),
            )
            refresh, painting = gr.Number(value=0), gr.Checkbox(value=False)
            panel = ui.create_ui(base, refresh, painting)
        return demo, controls, panel, scans, selection

    def test_default_off_does_not_scan_or_capture_and_registers_exact_fields(self):
        _, controls, panel, scans, selection = self.build()
        scans.assert_not_called()
        selection.assert_not_called()
        self.assertTrue(panel.visible)
        self.assertFalse(panel.open)
        fields = set(gligen_layout.FIELDS + native_hypernetwork.FIELDS)
        self.assertEqual(set(controls) - {"cn_selection", "cn_type"}, fields)
        self.assertIsNone(controls["gligen_model"].value)
        self.assertIsNone(controls["hypernetwork_model"].value)
        self.assertEqual(controls["hypernetwork_strength"].value, 1.0)

    def test_incompatible_startup_clears_hidden_saved_models_without_scanning(self):
        _, controls, panel, scans, selection = self.build(
            "QwenImage", dict(gligen_model="layout.pt", hypernetwork_model="attention.pt")
        )
        self.assertFalse(panel.visible)
        self.assertIsNone(controls["gligen_model"].value)
        self.assertIsNone(controls["hypernetwork_model"].value)
        self.assertIsNone(controls["gligen_model_signature"].value)
        scans.assert_not_called()
        selection.assert_not_called()

    def test_actual_refresh_event_lists_local_names_without_deserializing_or_losing_choice(self):
        demo, controls, _, scans, selection = self.build(defaults={"gligen_model": "layout.pt"})
        event = next(
            index for index, function in demo.fns.items() if function.fn.__name__ == "refresh"
        )
        result = asyncio.run(demo.process_api(event, ["layout.pt", None]))["data"]
        self.assertEqual(scans.call_count, 2)
        self.assertEqual(result[0]["value"], "layout.pt")
        self.assertEqual(result[1]["choices"][-1], ["attention.pt", "attention.pt"])
        selection.assert_not_called()
        self.assertEqual(controls["gligen_model"].choices[-1][1], "layout.pt")

    def test_actual_context_event_clears_incompatible_tools_and_preserves_ordinary_drafts(self):
        demo, _, _, scans, _ = self.build(
            defaults={"gligen_model": "layout.pt", "hypernetwork_model": "attention.pt"}
        )
        event = next(
            index for index, function in demo.fns.items() if function.fn.__name__ == "context"
        )

        def context(family, kind, painting=False):
            return asyncio.run(
                demo.process_api(
                    event, [family, "Custom...", kind, painting, "layout.pt", "attention.pt"]
                )
            )["data"]

        plain = context("SDXL", "None")
        self.assertTrue(plain[0]["visible"])
        self.assertEqual(plain[2]["value"], "layout.pt")
        edited = context("SDXL", "img2img")
        self.assertIsNone(edited[2]["value"])
        self.assertEqual(edited[3]["value"], "attention.pt")
        self.assertFalse(edited[1]["visible"])
        for family, operation in (
            ("QwenImage", "None"),
            ("SDXL", "detect"),
            ("SDXL", "lotus-depth"),
            ("SDXL", "restore-seedvr2"),
        ):
            with self.subTest(family=family, operation=operation):
                result = context(family, operation)
                self.assertFalse(result[0]["visible"])
                self.assertIsNone(result[2]["value"])
                self.assertIsNone(result[3]["value"])
        scans.assert_not_called()

    def test_actual_file_identity_event_keeps_decimal_timestamp_proof_until_explicit_reselection(
        self,
    ):
        demo, controls, _, scans, selection = self.build()
        event = next(
            index
            for index, function in demo.fns.items()
            if function.fn.__name__ == "identity" and function.inputs[0] is controls["gligen_model"]
        )
        state = SessionState(demo)
        # Dropdown choices normally arrive from the explicit expansion event.
        controls["gligen_model"].choices.append(("layout.pt", "layout.pt"))
        asyncio.run(demo.process_api(event, ["layout.pt", None], state=state))
        proof = state[controls["gligen_model_signature"]._id]
        self.assertEqual(proof[2:], ["1000000000000000001", "1000000000000000002"])
        selection.assert_called_once()
        asyncio.run(demo.process_api(event, ["layout.pt", None], state=state))
        selection.assert_called_once()
        asyncio.run(demo.process_api(event, [None, None], state=state))
        self.assertIsNone(state[controls["gligen_model_signature"]._id])
        scans.assert_not_called()


if __name__ == "__main__":
    unittest.main()
