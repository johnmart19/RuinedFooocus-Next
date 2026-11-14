"""Ordinary UI status copy preserves internal evidence and concrete constraints."""

import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.features.chat import ui_sampling
from ruinedfooocus.packaging import support_matrix
from ui_test_helpers import function_from


class ReleaseStatusWordingTests(unittest.TestCase):
    def test_hardware_display_omits_proof_badges_without_changing_capability_records(self):
        decisions = [
            dict(
                backend="Vulkan",
                operation="llama",
                adapter_id="gpu-0",
                state="installed",
                active=False,
                eligible=True,
                usable=False,
                qualified=False,
                reasons=(),
            )
        ]
        rows = support_matrix.capability_rows(decisions)
        rows[0]["note"] = "Exact runtime/device eligibility has not been verified"
        rows.append(dict(rows[0], status="Not available", note="Needs an Intel GPU"))
        original = copy.deepcopy(rows)
        render = function_from(
            "ruinedfooocus/ui/ui_hardware.py",
            "support_table",
            dict(
                inventory=NS(detect_gpus=Mock(return_value=[])),
                SUPPORT_HEADERS=("Component", "Option", "Status", "Details"),
                localize=lambda text, locale: text,
            ),
        )
        with patch.object(support_matrix, "support_rows", return_value=rows):
            display = render("en")
        self.assertEqual(display["data"][0][2:], ["Installed", ""])
        self.assertEqual(display["data"][1][2:], ["Not available", "Needs an Intel GPU"])
        self.assertEqual(rows, original)
        self.assertFalse(decisions[0]["usable"])
        self.assertFalse(decisions[0]["qualified"])
        self.assertEqual(
            support_matrix.capability_rows(decisions)[0]["status"], "Installed (unverified)"
        )

    def test_chat_defaults_copy_does_not_change_source_status_or_literal_model_name(self):
        source = {"status": "source_unverified", "repositories": []}
        original = copy.deepcopy(source)
        with (
            patch.object(ui_sampling, "cached_sampling_source", return_value=source),
            patch.object(ui_sampling, "offline_mode", return_value=False),
        ):
            message = ui_sampling.source_evidence(Path("custom.gguf"))[0][1]
        self.assertIn("runtime defaults", message)
        self.assertNotIn("unverified", message.lower())
        self.assertEqual(source, original)
        status = ui_sampling.active_runtime_status(
            "Unverified model.gguf", {"runtime_name": "llama.cpp"}
        )
        self.assertIn("Active: Unverified model.gguf", status)
        self.assertIn("Runtime: llama.cpp", status)
        self.assertNotIn("·", status)

    def test_component_summary_omits_unknown_badge_but_keeps_technical_compatibility(self):
        row = dict(
            key="vae_sdxl",
            label="VAE",
            filename="chosen.safetensors",
            status="Present",
            bytes=8,
            location="/chosen.safetensors",
            source=None,
            download_available=False,
            compatibility="unknown",
            compatibility_reason="Unverified architecture metadata",
        )
        state = dict(rows=[row], architecture="SDXL", missing_payload_bytes=0)
        render = function_from(
            "ruinedfooocus/features/models/ui_model_components.py",
            "refresh",
            dict(
                gr=gr,
                selected=lambda name: (Path(name), "SDXL"),
                session_choices=lambda *args: ({}, "identity"),
                component_inventory=lambda *args: state,
                selection_source=lambda *args: "Profile default",
                component_label=lambda key: key,
                offline=lambda: False,
            ),
        )
        shared = NS(settings=NS(default_settings={}), path_manager=object())
        with patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}):
            result = render("chosen.safetensors")
        self.assertNotIn("Unverified", result[0][0][2])
        self.assertNotIn("Ready", result[0][0][2])
        self.assertEqual(result[1]["vae_sdxl"]["compatibility"], "unknown")
        self.assertEqual(result[1]["vae_sdxl"]["compatibility_reason"], row["compatibility_reason"])


if __name__ == "__main__":
    unittest.main()
