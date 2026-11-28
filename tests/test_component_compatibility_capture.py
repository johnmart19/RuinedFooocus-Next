"""Local shape mismatches fail before preparation/native weight loading."""

import ast
from enum import Enum
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models import component_candidates as candidates
from ruinedfooocus.features.models import component_profiles as profiles
from ruinedfooocus.features.generation.image_preparation import capture_image_assets

ROOT = Path(__file__).resolve().parents[1]


def header(path, shapes):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({name: {"shape": list(shape)} for name, shape in shapes.items()}).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)
    return path


def planar(channels):
    return {
        "decoder.conv_in.weight": (512, channels, 3, 3),
        "encoder.conv_in.weight": (128, 3, 3, 3),
        "decoder.conv_out.weight": (3, 128, 3, 3),
        "encoder.down.2.downsample.conv.weight": (512, 512, 3, 3),
        "decoder.up.3.upsample.conv.weight": (512, 512, 3, 3),
    }


class ComponentCompatibilityCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = header(self.root / "renamed.safetensors", {})
        self.paths = SimpleNamespace(
            paths={"path_clip": [str(self.root / "clip")], "path_vae": [str(self.root / "vae")]},
            get_abspath=Path,
            DOWNLOADABLE_FILES={},
            EXTENSIONS=[".safetensors", ".gguf"],
        )
        # Execute the pinned native classifier itself, without importing its GPU runtime.
        source = ROOT / "repositories/ComfyUI/comfy/sd.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if getattr(node, "name", "") in ("TEModel", "detect_te_model")
        ]
        scope = {"Enum": Enum}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
        native = SimpleNamespace(detect_te_model=scope["detect_te_model"])
        patched = patch.dict("sys.modules", {"comfy.sd": native, "comfy.model_detection": None})
        patched.start()
        self.addCleanup(patched.stop)
        # Inventory must not attempt runtime detection on these deliberately empty checkpoints.
        inspected = patch(
            "ruinedfooocus.features.models.model_architecture.inspect_tensor_model",
            return_value=None,
        )
        inspected.start()
        self.addCleanup(inspected.stop)

    def test_wrong_same_name_encoder_and_vae_reject_frozen_api_choices(self):
        for key, name, shapes in (
            (
                "clip_l",
                "clip_l.safetensors",
                {"text_model.encoder.layers.30.mlp.fc1.weight": (5120, 1280)},
            ),
            ("vae_sdxl", "sdxl_vae.safetensors", planar(16)),
        ):
            with self.subTest(key=key):
                category = "clip" if key.startswith("clip_") else "vae"
                header(self.root / category / name, shapes)
                state = profiles.component_inventory(
                    self.model, "SDXL", {}, self.paths, {key: name}
                )
                row = next(row for row in state["rows"] if row["key"] == key)
                self.assertEqual(row["compatibility"], "incompatible")
                with self.assertRaisesRegex(ValueError, key + ".*incompatible"):
                    profiles.freeze_choices(self.model, "SDXL", {}, self.paths, {key: name})
                (self.root / category / name).unlink()

    def test_replaced_component_is_rechecked_after_snapshot_and_by_preparation(self):
        name = "renamed-vae.safetensors"
        path = header(self.root / "vae" / name, planar(4))
        choices = profiles.freeze_choices(self.model, "SDXL", {}, self.paths, {"vae_sdxl": name})
        # Replace the same path after submission: neither its name nor old header is proof.
        header(path, planar(16))
        models = SimpleNamespace(
            get_file=lambda *args: self.model,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
            get_model_base=lambda value: value["baseModel"],
        )
        with self.assertRaisesRegex(ValueError, "vae_sdxl.*incompatible"):
            capture_image_assets(
                {"base_model_name": self.model.name, profiles.REQUEST_KEY: choices},
                self.paths,
                {},
                models=models,
            )

    def test_bundled_components_take_priority_over_incompatible_legacy_globals(self):
        header(self.model, {"conditioner.embedders.0.x": (1,), "first_stage_model.x": (1,)})
        header(self.root / "vae" / "wrong.safetensors", planar(16))
        choices = profiles.freeze_choices(
            self.model, "SDXL", {"vae_sdxl": "wrong.safetensors"}, self.paths
        )
        self.assertTrue(all(value is None for value in choices.values()))

    def test_unknown_deliberate_choice_is_retained_and_not_automatically_qualified(self):
        header(self.root / "clip" / "custom.safetensors", {"custom.weight": (2, 3)})
        value = candidates.validate_local_choice(
            "clip_l", "custom.safetensors", self.paths, allow_unknown=True
        )
        self.assertEqual(
            profiles.freeze_choices(self.model, "SDXL", {}, self.paths, {"clip_l": value})[
                "clip_l"
            ],
            value,
        )
        self.assertEqual(
            candidates.installed_candidates("clip_l", self.paths)[0]["compatibility"], "unknown"
        )

    def test_native_mixed_gguf_and_scaled_fp8_encoder_set_is_rejected(self):
        header(
            self.root / "clip" / "scaled.safetensors",
            {
                "text_model.encoder.layers.0.mlp.fc1.weight": (3072, 768),
                "scaled_fp8": (2,),
            },
        )
        # A missing GGUF may be acquired later; the set is already known invalid.
        choices = {"clip_l": "scaled.safetensors", "clip_t5": "encoder.gguf"}
        state = profiles.component_inventory(self.model, "Flux", {}, self.paths, choices)
        with self.assertRaisesRegex(ValueError, "mixed GGUF encoder loader"):
            candidates.validate_inventory(state)
        self.assertEqual(
            candidates.installed_candidates("clip_l", self.paths, companion_rows=state["rows"])[0][
                "compatibility"
            ],
            "incompatible",
        )
        with self.assertRaisesRegex(ValueError, "different encoder"):
            candidates.validate_local_choice(
                "clip_l",
                "scaled.safetensors",
                self.paths,
                allow_unknown=True,
                companion_rows=state["rows"],
            )

    def test_gguf_logical_shape_agreement_does_not_certify_its_native_loader(self):
        path = self.root / "clip" / "renamed.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"metadata fixture")
        payload = {"text_model.encoder.layers.0.mlp.fc1.weight": {"shape": [3072, 768]}}
        with patch(
            "ruinedfooocus.features.models.gguf_metadata.read_gguf_header",
            return_value=(payload, "unknown"),
        ):
            self.assertEqual(
                candidates.inspect_local_component("clip_l", path)["compatibility"], "unknown"
            )
        for role in ("clip_hydit_bert", "clip_hydit_mt5"):
            self.assertEqual(
                candidates.inspect_local_component(role, path)["compatibility"], "incompatible"
            )

    def test_gguf_is_ineligible_for_image_vae_even_with_advanced_override(self):
        folder = self.root / "vae"
        folder.mkdir()
        names = ("looks-like-an-image-codec.gguf", "minimax-h3-video.GgUf")
        for name in names:
            (folder / name).write_bytes(b"this file must not be passed to a reader")
        with patch.object(candidates, "header_shapes") as read_shapes:
            rows = candidates.installed_candidates("vae_sdxl", self.paths)
            self.assertEqual({row["name"] for row in rows}, set(names))
            self.assertTrue(all(row["compatibility"] == "incompatible" for row in rows))
            for name in names:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        ValueError, "GGUF files are unsupported for still-image VAEs"
                    ),
                ):
                    candidates.validate_local_choice(
                        "vae_sdxl", name, self.paths, allow_unknown=True
                    )
            read_shapes.assert_not_called()

    def test_saved_gguf_image_vae_is_rejected_at_request_capture_even_if_missing(self):
        models = SimpleNamespace(
            get_file=lambda *args: self.model,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
            get_model_base=lambda value: value["baseModel"],
        )
        choices = {"vae_sdxl": "removed.GGUF"}
        with self.assertRaisesRegex(ValueError, "vae_sdxl.*GGUF files are unsupported"):
            profiles.freeze_choices(self.model, "SDXL", {}, self.paths, choices)
        with self.assertRaisesRegex(ValueError, "vae_sdxl.*GGUF files are unsupported"):
            capture_image_assets(
                {"base_model_name": self.model.name, profiles.REQUEST_KEY: choices},
                self.paths,
                {},
                models=models,
            )

    def test_native_clip_marker_with_wrong_dimensions_is_not_a_compatible_encoder(self):
        path = header(
            self.root / "clip" / "clip_l.safetensors",
            {
                "text_model.encoder.layers.0.mlp.fc1.weight": (5120, 1280),
            },
        )
        self.assertEqual(
            candidates.inspect_local_component("clip_l", path)["compatibility"], "incompatible"
        )

    def test_flux2_packed_bn_contract_cannot_be_replaced_by_plain_128_channels(self):
        shapes = planar(32)
        shapes.update(
            {
                "encoder.conv_out.weight": (64, 512, 3, 3),
                "post_quant_conv.weight": (32, 32, 1, 1),
                "bn.running_mean": (128,),
                "bn.running_var": (128,),
            }
        )
        path = header(self.root / "vae" / "renamed.safetensors", shapes)
        self.assertEqual(
            candidates.inspect_local_component("vae_flux2", path)["compatibility"], "compatible"
        )
        self.assertEqual(
            candidates.inspect_local_component("vae_mage_flow", path)["compatibility"],
            "incompatible",
        )
        for changed in (dict(shapes, **{"bn.running_var": (64,)}), planar(128)):
            header(path, changed)
            self.assertEqual(
                candidates.inspect_local_component("vae_flux2", path)["compatibility"],
                "incompatible",
            )

    def test_mage_dconv_cod_geometry_is_distinct_from_flux2_even_with_same_latents(self):
        shapes = {
            "student.dconv_encoder.proj_out.weight": (256, 384, 1, 1),
            "student.dconv_encoder.patch_cond_embed.weight": (768, 3, 16, 16),
            "student.dconv_encoder.z_proj.weight": (384, 128, 1, 1),
            "student.dconv_encoder.blocks.20.conv2.weight": (384, 1, 3, 3),
            "pipeline.y_embedder.decoder.conv_in.weight": (384, 128, 3, 3),
            "pipeline.y_embedder.decoder.conv_out.weight": (384, 384, 3, 3),
            "pipeline.y_embedder_x.weight": (8192, 384, 1, 1),
            "pipeline.blocks.20.conv2.weight": (384, 1, 3, 3),
            "pipeline.final_layer.linear.weight": (3, 32),
        }
        path = header(self.root / "vae" / "custom.safetensors", shapes)
        self.assertEqual(
            candidates.inspect_local_component("vae_mage_flow", path)["compatibility"], "compatible"
        )
        self.assertEqual(
            candidates.inspect_local_component("vae_flux2", path)["compatibility"], "incompatible"
        )
        shapes["student.dconv_encoder.patch_cond_embed.weight"] = (768, 3, 8, 8)
        header(path, shapes)
        self.assertEqual(
            candidates.inspect_local_component("vae_mage_flow", path)["compatibility"],
            "incompatible",
        )


if __name__ == "__main__":
    unittest.main()
