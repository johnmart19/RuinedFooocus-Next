"""Unsupported DirectML auxiliary paths explain CPU at runtime, not capture."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.chat import rag_preparation as rag
from ruinedfooocus.features.image_processing import caption_runtime


class AuxiliaryDeviceFallbackReasonTests(unittest.TestCase):
    def test_directml_caption_has_an_explicit_reason_without_a_gpu_attempt(self):
        native = NS(is_directml_enabled=lambda: True, get_torch_device=Mock())
        with (
            patch.dict("sys.modules", {"comfy": NS(model_management=native)}),
            patch.object(caption_runtime, "warn_cpu_fallback") as warning,
        ):
            self.assertEqual(caption_runtime.caption_device(), torch.device("cpu"))
        native.get_torch_device.assert_not_called()
        self.assertIn("DirectML", warning.call_args.args[1])
        self.assertIn("not qualified", warning.call_args.args[1])

    def test_native_cpu_caption_does_not_claim_a_directml_failure(self):
        native = NS(is_directml_enabled=lambda: False, get_torch_device=lambda: torch.device("cpu"))
        with (
            patch.dict("sys.modules", {"comfy": NS(model_management=native)}),
            patch.object(caption_runtime, "warn_cpu_fallback") as warning,
        ):
            self.assertEqual(caption_runtime.caption_device(), torch.device("cpu"))
        warning.assert_not_called()

    def test_rag_reason_is_captured_passively_and_reported_only_at_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, _, _ in rag.MODEL_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"metadata fixture only")
            with (
                patch.object(rag, "data_path", return_value=root),
                patch.object(rag, "snapshot_download", side_effect=AssertionError("No network")),
                patch("ruinedfooocus.core.device_fallback.warn_cpu_fallback") as warning,
            ):
                prepared = rag.capture_chat_rag(
                    [["text", "local context"]],
                    {},
                    launch_args=NS(cpu=False, directml=0, offline=False),
                )
                self.assertFalse(prepared.settings["gpu"])
                self.assertIn("DirectML", prepared.settings["cpu_reason"])
                warning.assert_not_called()
                options = rag.runtime_options(prepared)
                self.assertIsNone(options["gpu"])
                warning.assert_called_once_with(
                    "Chat text embeddings", prepared.settings["cpu_reason"]
                )
                warning.reset_mock()
                explicit = rag.capture_chat_rag(
                    [["text", "local context"]],
                    {"rag_embedding_device": "CPU"},
                    launch_args=NS(cpu=False, directml=0, offline=False),
                )
                self.assertNotIn("cpu_reason", explicit.settings)
                self.assertIsNone(rag.runtime_options(explicit)["gpu"])
                warning.assert_not_called()
                automatic = rag.capture_chat_rag(
                    [["text", "local context"]],
                    {},
                    launch_args=NS(cpu=False, directml=None, offline=False),
                )
                self.assertTrue(rag.runtime_options(automatic)["gpu"])
                self.assertNotIn("cpu_reason", automatic.settings)
                warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
