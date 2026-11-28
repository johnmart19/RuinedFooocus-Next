"""Auxiliary caption and RAG CPU placement paths stay passive at capture."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from ruinedfooocus.features.chat import rag_preparation as rag
from ruinedfooocus.features.image_processing import caption_runtime


class AuxiliaryDeviceFallbackReasonTests(unittest.TestCase):
    def test_native_cpu_caption_returns_cpu_device(self):
        native = NS(get_torch_device=lambda: torch.device("cpu"))
        with patch.dict("sys.modules", {"comfy": NS(model_management=native)}):
            self.assertEqual(caption_runtime.caption_device(), torch.device("cpu"))

    def test_rag_cpu_policy_is_captured_passively_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, _, _ in rag.MODEL_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"metadata fixture only")
            with (
                patch.object(rag, "data_path", return_value=root),
                patch.object(rag, "snapshot_download", side_effect=AssertionError("No network")),
            ):
                global_cpu = rag.capture_chat_rag(
                    [["text", "local context"]],
                    {},
                    launch_args=NS(cpu=True, offline=False),
                )
                self.assertFalse(global_cpu.settings["gpu"])
                self.assertIsNone(rag.runtime_options(global_cpu)["gpu"])

                explicit = rag.capture_chat_rag(
                    [["text", "local context"]],
                    {"rag_embedding_device": "CPU"},
                    launch_args=NS(cpu=False, offline=False),
                )
                self.assertFalse(explicit.settings["gpu"])
                self.assertIsNone(rag.runtime_options(explicit)["gpu"])

                automatic = rag.capture_chat_rag(
                    [["text", "local context"]],
                    {},
                    launch_args=NS(cpu=False, offline=False),
                )
                self.assertTrue(rag.runtime_options(automatic)["gpu"])


if __name__ == "__main__":
    unittest.main()
