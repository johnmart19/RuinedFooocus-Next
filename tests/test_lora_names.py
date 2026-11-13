"""Short LoRA names typed in prompts resolve to installed files without loading weights."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ruinedfooocus.features.models.lora_names import merge_loras, resolve_lora_name  # noqa: E402

INSTALLED = [
    r"styles\Moss-RealismEnhancer-V1.safetensors",
    "TrendCraft_Detailer-v2.0A-PonyXL.safetensors",
    "TrendCraft_Detailer-v2.4I-Illustrious.safetensors",
    "detail.pt",
]


class ResolveTests(unittest.TestCase):
    def test_exact_stem_case_and_folder_free_names(self):
        for typed in (
            r"styles\Moss-RealismEnhancer-V1.safetensors",
            "styles/Moss-RealismEnhancer-V1",
            "Moss-RealismEnhancer-V1",
            "moss-realismenhancer-v1",
            "realismenhancer",
        ):
            self.assertEqual(resolve_lora_name(typed, INSTALLED), INSTALLED[0], typed)
        self.assertEqual(resolve_lora_name("DETAIL", INSTALLED), "detail.pt")
        self.assertEqual(resolve_lora_name("illustrious", INSTALLED), INSTALLED[2])

    def test_ambiguous_names_list_candidates(self):
        with self.assertRaisesRegex(ValueError, "matches several.*PonyXL.*Illustrious"):
            resolve_lora_name("TrendCraft", INSTALLED)

    def test_unknown_names_keep_download_and_missing_file_handling(self):
        self.assertEqual(resolve_lora_name("absent", INSTALLED), "absent.safetensors")
        self.assertEqual(resolve_lora_name("absent.pt", INSTALLED), "absent.pt")


class MergeTests(unittest.TestCase):
    def test_prompt_weight_overrides_selected_card_once(self):
        selected = [{"name": INSTALLED[2], "weight": 1.0, "hash": None}]
        typed = [
            {"name": "illustrious", "weight": 0.6, "hash": None},
            {"name": "realismenhancer", "weight": 0.4, "hash": None},
        ]
        merged = merge_loras(selected, typed, INSTALLED)
        self.assertEqual(
            [(row["name"], row["weight"]) for row in merged],
            [(INSTALLED[2], 0.6), (INSTALLED[0], 0.4)],
        )

    def test_selected_cards_need_no_lookup(self):
        selected = [{"name": "x.safetensors", "weight": 0.5, "hash": "h"}]
        self.assertEqual(merge_loras(selected, [], []), selected)


if __name__ == "__main__":
    unittest.main()
