"""Native architecture names must not imply incompatible older companions."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models.model_architecture import detect_tensor_architecture
from ruinedfooocus.features.models.image_components import inventory
from test_image_preparation import Manager


class DistinctImageArchitecturesTests(unittest.TestCase):
    def test_native_qwen_image21_keeps_its_distinct_architecture(self):
        config = {"image_model": "qwen_image21"}
        detection = SimpleNamespace(
            unet_prefix_from_state_dict=lambda _: "",
            model_config_from_unet=lambda *args, **kwargs: type(
                "QwenImage21", (), {"unet_config": config}
            )(),
        )
        with patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(empty=lambda shape, device: (shape, device)),
                "comfy": SimpleNamespace(model_detection=detection),
            },
        ):
            self.assertEqual(detect_tensor_architecture({"weight": {"shape": [1]}}), "QwenImage21")

    def test_qwen_image21_offers_distinct_encoder_and_64_channel_vae(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.safetensors"
            header = b'{"weight":{"shape":[1]}}'
            model.write_bytes(len(header).to_bytes(8, "little") + header)
            result = inventory(model, "QwenImage21", {}, Manager(Path(directory)))
            self.assertEqual(
                [row["key"] for row in result["rows"]], ["clip_qwen_image21", "vae_qwen_image21"]
            )


if __name__ == "__main__":
    unittest.main()
