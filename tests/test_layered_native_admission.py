"""Positive native Layered topology admits manual capture without a scalar label."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from gguf import GGUFWriter
import numpy as np

from ruinedfooocus.features.generation import layered
from ruinedfooocus.features.models.model_architecture import inspect_model_file
from ruinedfooocus.features.models.native_recipes import compatible_native_recipe
import test_native_variant_defaults as fixtures


def shapes(*, layered=True):
    result = {
        "txt_norm.weight": [3584],
        "img_in.weight": [3072, 64],
        **{f"transformer_blocks.{index}.attn.norm_q.weight": [128] for index in range(60)},
    }
    if layered:
        result["time_text_embed.addition_t_embedding.weight"] = [2, 3072]
    return result


class LayeredNativeAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.dict(fixtures.SHAPES, {"QwenImage": shapes()}))
        self.enterContext(fixtures.native_backend())

    def checkpoint(self, name="unrelated.safetensors", *, positive=True, metadata=None):
        path = self.root / name
        if path.suffix == ".gguf":
            writer = GGUFWriter(path, "qwen_image")
            for key, shape in shapes(layered=positive).items():
                writer.add_tensor(key, np.zeros(shape, dtype=np.float16))
            if metadata:
                for key, value in metadata.items():
                    writer.add_string(key, value)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            return path
        return fixtures.write_header(path, shapes(layered=positive), metadata=metadata)

    def test_actual_native_detector_flags_survive_header_projection(self):
        contract = inspect_model_file(self.checkpoint())
        self.assertIs(contract["native_config"].get("use_additional_t_cond"), True)
        self.assertEqual(contract["native_config"].get("default_ref_method"), "negative_index")

    def test_actual_manual_capture_uses_positive_topology_in_every_safe_container(self):
        for name in ("unrelated.safetensors", "renamed.sft", "unknown-quantization.gguf"):
            with self.subTest(name=name):
                path = self.checkpoint(name)
                recipe = compatible_native_recipe(path, {})
                self.assertEqual(recipe.get("id"), "native-qwenimage-layered")
                self.assertIs(recipe["auto_recipe_supported"], False)
                self.assertIs(recipe["artifact_qualified"], False)
                data = layered.capture(
                    dict(
                        base_model_name=path.name,
                        qwen_layer_count=2,
                        width=64,
                        height=64,
                        image_number=1,
                        task_type="tool_call",
                        custom_steps=4,
                        cfg=2.5,
                        sampler_name="euler",
                        scheduler="simple",
                    ),
                    NS(model_paths={}),
                    NS(get_file=lambda *args: path),
                    family="QwenImage",
                )
                self.assertEqual(data[layered.KEY].layers, 2)
                self.assertEqual(
                    (data["custom_steps"], data["sampler_name"], data["scheduler"]),
                    (4, "euler", "simple"),
                )

    def test_plain_qwen_or_scalar_layered_label_cannot_forge_positive_topology(self):
        for metadata in (None, {"modelspec.variant": "layered"}):
            with self.subTest(metadata=metadata):
                path = self.checkpoint("ordinary.safetensors", positive=False, metadata=metadata)
                self.assertNotEqual(
                    compatible_native_recipe(path, {}).get("id"), "native-qwenimage-layered"
                )

    def test_conflicting_scalar_variant_does_not_promote_layered_to_an_auto_recipe(self):
        path = self.checkpoint(metadata={"modelspec.variant": "base"})
        self.assertEqual(compatible_native_recipe(path, {}), {})


if __name__ == "__main__":
    unittest.main()
