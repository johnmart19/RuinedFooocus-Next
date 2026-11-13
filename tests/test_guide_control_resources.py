"""The exact base-six SDXL guide payload reaches the existing signed broker."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.image_editing import guide_control_compat as guide
from ruinedfooocus.features.image_editing.control_candidates import classify_header
from ruinedfooocus.features.image_editing.native_control_recipes import (
    control_contract,
    union_types,
)
from ruinedfooocus.features.models.acquisition import Catalogue
from ruinedfooocus.features.models.model_registry import load_registry


class GuideControlResourceTests(unittest.TestCase):
    KEY = "cn_union_sdxl"
    REVISION = "801a4a3fa3d4c936f4feea95b98607bc6726f80c"
    URL = (
        "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/"
        + REVISION
        + "/diffusion_pytorch_model.safetensors"
    )

    def setUp(self):
        self.downloads = json.loads(
            resource_path("model_catalogue", "native_controlnet_union_sdxl.json").read_text(
                encoding="utf-8"
            )
        )
        self.registry = load_registry()
        self.entry = self.registry.annotations_for(self.KEY, self.downloads[self.KEY])

    def test_exact_publisher_payload_matches_native_header_contract(self):
        source = self.downloads[self.KEY]
        self.assertEqual(
            (source["url"], source["path"], source["filename"]),
            (self.URL, "path_controlnet", "xinsir_controlnet_union_sdxl_1.0.safetensors"),
        )
        self.assertEqual((source["size"], source["sha256"]), (2512030408, guide.UNION_SHA256))
        artifact = self.entry["artifact"]
        self.assertEqual(
            (artifact["pinned_url"], artifact["bytes"], artifact["sha256"]),
            (self.URL, source["size"], source["sha256"]),
        )
        self.assertEqual(
            (self.entry["role"], self.entry["native_family"]), ("control_conditioning", "SDXL")
        )
        header = {
            "task_embedding": {"shape": [6, 320]},
            "control_add_embedding.linear_1.bias": {"shape": [1280]},
            "controlnet_cond_embedding.conv_in.weight": {"shape": [16, 3, 3, 3]},
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight": {
                "shape": [640, 2048]
            },
        }
        self.assertEqual(control_contract(header)["union_modes"], 6)
        self.assertEqual(classify_header(header)["families"], ("SDXL",))
        runtime = self.entry["runtime"]["native_comfy"]
        self.assertEqual(
            (runtime["loader"], runtime["mode_selector"], runtime["union_modes"]),
            ("ControlNetLoader", "SetUnionControlNetType", 6),
        )
        self.assertEqual(runtime["source_pin"], "989d93d0d968e704c589ce0146bb18c77f6b50cc")

    def test_existing_acquisition_capture_uses_exact_signed_request_without_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            manager = SimpleNamespace(
                DOWNLOADABLE_FILES=self.downloads, get_model_folders=lambda _: [root]
            )
            manager.preparation_request = lambda key: PathManager.preparation_request(
                manager, key, destination=root / self.downloads[key]["filename"]
            )
            with patch(
                "ruinedfooocus.features.models.acquisition.preparation_service.submit",
                side_effect=AssertionError("capture must not download"),
            ):
                plan = Catalogue(manager).capture(self.KEY, include_companions=False, reserve_gib=0)
            (request,) = plan.artifacts
            self.assertEqual(
                (request.url, request.expected_size, request.sha256),
                (self.URL, 2512030408, guide.UNION_SHA256),
            )
            self.assertEqual(request.destination, root / self.downloads[self.KEY]["filename"])
            self.assertFalse(request.destination.exists())
            self.downloads[self.KEY]["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "Conflicting sha256"):
                manager.preparation_request(self.KEY)

    def test_six_mode_provenance_keeps_normal_inference_and_quality_limits(self):
        profile = self.registry.profile(self.entry["profile_id"])
        self.assertEqual(tuple(profile["supported_modes"]), union_types()[1:7])
        self.assertEqual(tuple(profile["unsupported_modes"]), ("tile", "repaint"))
        self.assertEqual(
            (union_types().index("depth") - 1, union_types().index("normal") - 1), (1, 4)
        )
        self.assertEqual(profile["provenance"]["revision"], self.REVISION)
        self.assertEqual(
            profile["provenance"]["exact_variant"], "diffusion_pytorch_model.safetensors"
        )
        self.assertEqual(profile["license"]["value"], "Apache-2.0")
        convention = profile["normal_convention"]
        self.assertEqual(convention["publisher_preprocessor"], "NormalBaeDetector scannet.pt")
        self.assertIn("source_qualified_inference", convention["moge_axis_conversion"])
        self.assertIn("Not a direct Xinsir", convention["evidence_scope"])
        self.assertIn("GPU/visual acceptance pending", convention["evidence_scope"])
        self.assertFalse(
            self.entry["chat_weight"]
            or self.entry["uses_chat_runtime"]
            or self.entry["auto_recipe_supported"]
        )


if __name__ == "__main__":
    unittest.main()
