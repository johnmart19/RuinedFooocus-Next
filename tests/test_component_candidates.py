"""VAE alternatives use native architecture headers, never filenames or tensors."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models import component_candidates as candidates


def qwen_wan21_shapes():
    # Selected shapes from the installed Qwen Image VAE's 21,176-byte header,
    # SHA256 636d195d2853366f11a84e91f139695bb3aced8ea17ad00e9a7c7423a075382e.
    # The separately inspected Wan2.1 VAE has the same architecture shapes.
    return {
        "encoder.conv1.weight": (96, 3, 3, 3, 3),
        "decoder.conv1.weight": (384, 16, 3, 3, 3),
        "encoder.head.0.gamma": (384, 1, 1, 1),
        "encoder.head.2.weight": (32, 384, 3, 3, 3),
        "decoder.head.0.gamma": (96, 1, 1, 1),
        "decoder.head.2.weight": (3, 96, 3, 3, 3),
        "decoder.middle.0.residual.0.gamma": (384, 1, 1, 1),
        "conv1.weight": (32, 32, 1, 1, 1),
        "conv2.weight": (16, 16, 1, 1, 1),
        "encoder.downsamples.2.resample.1.weight": (96, 96, 3, 3),
        "encoder.downsamples.5.resample.1.weight": (192, 192, 3, 3),
        "encoder.downsamples.8.resample.1.weight": (384, 384, 3, 3),
        "encoder.downsamples.5.time_conv.weight": (192, 192, 3, 1, 1),
        "encoder.downsamples.8.time_conv.weight": (384, 384, 3, 1, 1),
        "decoder.upsamples.3.resample.1.weight": (192, 384, 3, 3),
        "decoder.upsamples.7.resample.1.weight": (192, 384, 3, 3),
        "decoder.upsamples.11.resample.1.weight": (96, 192, 3, 3),
        "decoder.upsamples.3.time_conv.weight": (768, 384, 3, 1, 1),
        "decoder.upsamples.7.time_conv.weight": (768, 384, 3, 1, 1),
    }


def classify(shapes):
    return candidates.classify_shapes(shapes, detector=lambda _: None)


class VaeCandidateTests(unittest.TestCase):
    def test_observed_rgb_wan21_contract_matches_anima_qwen_and_wan_roles_only(self):
        classification = classify(qwen_wan21_shapes())
        self.assertEqual(classification["vae_layout"], "wan21")
        self.assertEqual(classification["vae_channels"], 16)
        self.assertTrue(classification["vae_contract_valid"])
        for role in ("vae_qwen_image", "vae_wan"):
            self.assertEqual(candidates.compatibility(role, classification), "compatible")
        for role in ("vae_sd", "vae_sdxl", "vae_flux", "vae_sd3", "clip_l"):
            self.assertEqual(candidates.compatibility(role, classification), "incompatible")

    def test_wrong_channels_kernels_scales_and_normalized_layouts_are_excluded(self):
        mutations = (
            ("decoder.conv1.weight", (384, 48, 3, 3, 3)),
            ("encoder.conv1.weight", (96, 4, 3, 3, 3)),
            ("decoder.head.2.weight", (4, 96, 3, 3, 3)),
            ("encoder.head.2.weight", (64, 384, 3, 3, 3)),
            ("conv2.weight", (48, 48, 1, 1, 1)),
            ("decoder.conv1.weight", (384, 16, 1, 3, 3)),
            ("decoder.upsamples.11.resample.1.weight", None),
            ("encoder.downsamples.2.time_conv.weight", (96, 96, 3, 1, 1)),
            ("bn.running_mean", (16,)),
        )
        for key, value in mutations:
            with self.subTest(key=key, shape=value):
                shapes = qwen_wan21_shapes()
                if value is None:
                    shapes.pop(key)
                else:
                    shapes[key] = value
                self.assertEqual(
                    candidates.compatibility("vae_qwen_image", classify(shapes)), "incompatible"
                )

    def test_wan22_and_qwen21_do_not_inherit_the_rgb16_qwen_role(self):
        for temporal_kernel in (1, 3):
            with self.subTest(temporal_kernel=temporal_kernel):
                shapes = qwen_wan21_shapes()
                shapes["decoder.upsamples.0.upsamples.0.residual.2.weight"] = (
                    320,
                    320,
                    temporal_kernel,
                    3,
                    3,
                )
                shapes["decoder.head.2.weight"] = (
                    4 if temporal_kernel == 1 else 12,
                    160,
                    temporal_kernel,
                    3,
                    3,
                )
                shapes["decoder.conv1.weight"] = (
                    1280,
                    64 if temporal_kernel == 1 else 48,
                    temporal_kernel,
                    3,
                    3,
                )
                classification = classify(shapes)
                self.assertEqual(
                    classification["vae_layout"],
                    "qwen_image21" if temporal_kernel == 1 else "wan22",
                )
                self.assertEqual(
                    candidates.compatibility("vae_qwen_image", classification), "incompatible"
                )
                self.assertEqual(
                    candidates.compatibility("vae_wan", classification), "incompatible"
                )

    def test_planar16_is_not_compatible_merely_because_the_channel_count_matches(self):
        shapes = {
            "decoder.conv_in.weight": (512, 16, 3, 3),
            "encoder.conv_in.weight": (128, 3, 3, 3),
            "decoder.conv_out.weight": (3, 128, 3, 3),
            "encoder.down.2.downsample.conv.weight": (512, 512, 3, 3),
            "decoder.up.3.upsample.conv.weight": (512, 512, 3, 3),
        }
        classification = classify(shapes)
        self.assertEqual(candidates.compatibility("vae_flux", classification), "compatible")
        self.assertEqual(candidates.compatibility("vae_qwen_image", classification), "incompatible")

    def test_installed_renamed_header_is_offered_without_tensor_payloads_and_unknown_stays_advanced(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, shapes in (
                ("renamed.safetensors", qwen_wan21_shapes()),
                ("qwen_image_vae.safetensors", {"unknown.conv.weight": (16, 3, 3, 3, 3)}),
            ):
                raw = json.dumps(
                    {key: {"shape": list(shape)} for key, shape in shapes.items()}
                ).encode()
                (root / name).write_bytes(len(raw).to_bytes(8, "little") + raw)
            paths = SimpleNamespace(
                paths={"path_vae": [str(root)]}, get_abspath=Path, EXTENSIONS=[".safetensors"]
            )
            with patch.dict("sys.modules", {"comfy.sd": None}):
                rows = candidates.installed_candidates("vae_qwen_image", paths)
                self.assertEqual(
                    {row["name"]: row["compatibility"] for row in rows},
                    {"renamed.safetensors": "compatible", "qwen_image_vae.safetensors": "unknown"},
                )
                self.assertEqual(
                    candidates.validate_local_choice(
                        "vae_qwen_image", "renamed.safetensors", paths
                    ),
                    "renamed.safetensors",
                )
                with self.assertRaisesRegex(ValueError, "unverified"):
                    candidates.validate_local_choice(
                        "vae_qwen_image", "qwen_image_vae.safetensors", paths
                    )
                self.assertEqual(
                    candidates.validate_local_choice(
                        "vae_qwen_image", "qwen_image_vae.safetensors", paths, allow_unknown=True
                    ),
                    "qwen_image_vae.safetensors",
                )
