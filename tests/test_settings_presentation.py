"""Native Settings runtime presentation, without application or device startup."""

from collections import Counter
from types import SimpleNamespace
import unittest

import gradio as gr

from ruinedfooocus.backends.comfy.runtime_info import runtime_details
from ruinedfooocus.ui.settings_presentation import create_image_runtime_summary


class SettingsRuntimePresentationTests(unittest.TestCase):
    def test_initialized_cpu_identity_does_not_promote_requested_policies_to_active_state(self):
        rows = runtime_details(
            SimpleNamespace(
                __version__="test-build", version=SimpleNamespace(hip=None, cuda="13.2")
            ),
            SimpleNamespace(type="cpu"),
            "CPU test device",
            SimpleNamespace(
                directml=None,
                disable_async_offload=False,
                async_offload=0,
                cpu_vae=False,
                fast=set(),
                fp16_unet=True,
            ),
            SimpleNamespace(aimdo_enabled=False),
        )
        with gr.Blocks() as app:
            summary, diagnostics = create_image_runtime_summary(rows)
        self.assertEqual(
            summary.value["data"],
            [
                ["Active image backend", "CPU"],
                ["Selected device", "CPU test device"],
                ["PyTorch build", "test-build"],
            ],
        )
        details = dict(diagnostics.value["data"])
        self.assertEqual(details["Compiled CUDA runtime"], "13.2")
        self.assertEqual(details["Image precision requested"], "fp16_unet")
        self.assertEqual(details["Weight transfer streams requested"], "0")
        self.assertEqual(
            Counter(map(tuple, summary.value["data"] + diagnostics.value["data"])),
            Counter(map(tuple, rows)),
        )
        disclosure = next(block for block in app.blocks.values() if isinstance(block, gr.Accordion))
        self.assertFalse(disclosure.open)
        self.assertNotIn(summary, disclosure.children)
        self.assertIn(diagnostics, disclosure.children)
        self.assertFalse(summary.interactive)
        self.assertFalse(diagnostics.interactive)
        self.assertEqual(app.config["dependencies"], [])

    def test_additional_runtime_facts_remain_in_diagnostics_without_assuming_row_positions(self):
        rows = [
            ["Future native policy requested", "Auto"],
            ["Selected device", "Device with | and <characters>"],
            ["Active image backend", "ROCm / HIP"],
            ["PyTorch build", "test-build"],
            ["Qualification", "Initialized state only"],
        ]
        with gr.Blocks():
            summary, diagnostics = create_image_runtime_summary(rows)
        self.assertEqual(dict(summary.value["data"])["Selected device"], rows[1][1])
        self.assertEqual(diagnostics.value["data"], [rows[0], rows[4]])


if __name__ == "__main__":
    unittest.main()
