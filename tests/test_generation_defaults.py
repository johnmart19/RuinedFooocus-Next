"""Explicit saving stores preferences, never a previous image or request state."""

import unittest

from ruinedfooocus.features.generation.defaults import generation_defaults, startup_loras


class GenerationDefaultsTests(unittest.TestCase):
    def test_saves_lotus_depth_intent_without_uploads(self):
        fields = dict(
            cn_depth_preprocess="auto",
            cn_depth_model="lotus",
            cn_depth_lotus_model="local.safetensors",
            cn_depth_lotus_vae="codec.safetensors",
        )
        saved = generation_defaults(dict(fields, input_image=object()))
        self.assertEqual({key: saved[key] for key in fields}, fields)
        self.assertNotIn("input_image", saved)

    def test_saves_arbitrary_loras_and_copies_lists_without_uploads(self):
        styles = ["My style"]
        loras = [(None, f"0.5 - custom - {i}.safetensors") for i in range(7)]
        values = {
            "base_model_name": "image.safetensors",
            "style_selection": styles,
            "loras": loras,
            "image_number": 0,
            "input_image": object(),
            "prompt": "Keep 2 red apples.",
        }
        result = generation_defaults(values)
        self.assertEqual(len(result["startup_loras"]), 7)
        self.assertEqual(result["startup_loras"][0]["name"], "custom - 0.safetensors")
        self.assertEqual(result["image_number"], 1)
        self.assertNotIn("input_image", result)
        self.assertEqual(result["prompt"], values["prompt"])
        styles.append("Later change")
        self.assertEqual(result["style"], ["My style"])

    def test_explicit_empty_list_disables_old_loras_but_legacy_still_loads(self):
        old = {"lora_1_model": "legacy.safetensors", "lora_1_weight": 0.75}
        self.assertEqual(startup_loras(old), [("legacy.safetensors", 0.75)])
        self.assertEqual(startup_loras(old | {"startup_loras": []}), [])

    def test_active_preset_cannot_save_misleading_hidden_controls(self):
        with self.assertRaisesRegex(ValueError, "Clear the generation preset"):
            generation_defaults({"preset_selection": "preset.png"})

    def test_nonfinite_lora_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            generation_defaults({"loras": [(None, "nan - style.safetensors")]})

    def test_saved_image_controls_still_use_full_settings_validation(self):
        with self.assertRaisesRegex(ValueError, "grid_columns"):
            generation_defaults({"grid_columns": 0})


if __name__ == "__main__":
    unittest.main()
