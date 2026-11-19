"""Stable Cascade Stage B discovery used when a request names only Stage C."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.generation import cascade

ROOT = Path(__file__).resolve().parents[1]


def models(families, *, broken=()):
    def detect(path):
        if path in broken:
            raise OSError("unreadable")
        return families[path]

    return SimpleNamespace(
        get_names=lambda kind: [*families, "missing"],
        get_file=lambda kind, name: None if name == "missing" else name,
        detect_checkpoint_base=detect,
    )


class StageBSelectionTests(unittest.TestCase):
    def test_only_bundled_stage_b_checkpoints_are_listed_in_catalogue_order(self):
        families = {
            "sdxl": "SDXL",
            "b-lite": "Stable_Cascade_B",
            "c": "Stable_Cascade_C",
            "b-old": "StableCascade_B",
            "b-unbundled": "Stable_Cascade_B",
            "b-broken": "Stable_Cascade_B",
        }

        def bundle(path, stage):
            self.assertEqual(stage, "b")
            if path == "b-unbundled":
                raise ValueError("not a bundle")

        with patch.object(cascade, "require_bundle", bundle):
            names = cascade.installed_stage_b(models(families, broken={"b-broken"}))
        self.assertEqual(names, ["b-lite", "b-old"])

    def test_preparation_uses_the_single_installed_stage_b_and_says_when_it_cannot(self):
        source = (ROOT / "ruinedfooocus/features/generation/image_preparation.py").read_text(
            encoding="utf-8"
        )
        branch = source[source.index('if not result.get("cascade_stage_b"):') :]
        branch = branch[: branch.index("sampling_settings(result)")]
        self.assertIn("installed_stage_b(models)", branch)
        self.assertIn("len(available) != 1", branch)
        self.assertIn('result["cascade_stage_b"] = available[0]', branch)
        self.assertIn("Select a Stable Cascade Stage B checkpoint in Cascade settings.", branch)
        # The choice is made before Stage B's own defaults are looked up.
        self.assertIn("stage_b_defaults(result.get(", branch)

    def test_the_settings_panel_lists_the_same_checkpoints(self):
        from ruinedfooocus.features.image_editing import ui_native_workflows

        with patch.object(cascade, "installed_stage_b", return_value=["only"]) as listed:
            self.assertEqual(ui_native_workflows.stage_b_choices("models"), ["only"])
        listed.assert_called_once_with("models")


if __name__ == "__main__":
    unittest.main()
