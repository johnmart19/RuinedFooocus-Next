"""Native SD2 uses CLIP-H despite sharing BaseModel with SD1."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility
from test_image_preparation import Manager
from test_model_cache import method
from test_native_model_coverage import PIPELINE, route_info
from test_native_variant_defaults import write_header


class SD2ComponentsTests(unittest.TestCase):
    def test_runtime_uses_exact_native_config_identity_for_shared_base_model(self):
        get_info = method(PIPELINE, "get_clip_and_vae", dict(settings=NS(default_settings={})))
        owner = NS(known_model_info=route_info())
        for native_class, role in (
            ("SD20", "clip_sd4x_h"),
            ("SD15", "clip_l"),
            ("Other", "clip_l"),
        ):
            config = type(native_class, (), {})()
            config.unet_config = {"context_dim": 1024}
            model = type("BaseModel", (), {})()
            model.model_config = config
            self.assertEqual(get_info(owner, NS(model=model))["clip_keys"], [role])

    def test_inventory_sd2_role_rejects_sd1_encoder_and_accepts_native_h(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = write_header(root / "renamed.safetensors", {"unknown.weight": [1]})
            state = inventory(model, "SD 2.0", {}, Manager(root))
            self.assertEqual([row["key"] for row in state["rows"]], ["clip_sd4x_h", "vae_sd"])
        wrong = classify_shapes(
            {"text_model.encoder.layers.0.mlp.fc1.weight": (3072, 768)},
            detector=lambda _: NS(name="CLIP_L"),
        )
        right = classify_shapes(
            {"text_model.encoder.layers.22.mlp.fc1.weight": (4096, 1024)},
            detector=lambda _: NS(name="CLIP_H"),
        )
        self.assertEqual(compatibility("clip_sd4x_h", wrong), "incompatible")
        self.assertEqual(compatibility("clip_sd4x_h", right), "compatible")


if __name__ == "__main__":
    unittest.main()
