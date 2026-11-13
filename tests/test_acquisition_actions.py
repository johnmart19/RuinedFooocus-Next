"""Selected-resource action policy and native component activation callers."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models.acquisition_actions import (
    action_state,
    component_targets,
    use_component,
)
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.component_profiles import checkpoint_identity


class AcquisitionActionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.checkpoint, self.component = (
            self.root / "checkpoint.safetensors",
            self.root / "encoder.gguf",
        )
        self.checkpoint.write_bytes(b"checkpoint header fixture")
        self.component.write_bytes(b"component header fixture")
        self.row = dict(
            key="clip/encoder.gguf",
            category="clip",
            local=str(self.component),
            local_signature=file_signature(self.component),
            allowed=True,
            size_matches=True,
        )
        self.roles = [
            dict(key="clip_l", category="clip", status="Installed"),
            dict(key="clip_g", category="clip", status="Embedded in checkpoint"),
            dict(key="vae_sdxl", category="vae", status="Installed"),
        ]

    def test_installed_component_targets_only_compatible_external_role_and_sets_session_choice(
        self,
    ):
        original = copy.deepcopy(self.row)
        candidate = dict(path=str(self.component), name="encoder.gguf", compatibility="compatible")
        with (
            patch(
                "ruinedfooocus.features.models.component_profiles.component_inventory",
                return_value=dict(rows=self.roles),
            ),
            patch(
                "ruinedfooocus.features.models.component_candidates.installed_candidates",
                return_value=[candidate],
            ) as scan,
        ):
            targets = component_targets(self.row, self.checkpoint, "SDXL", {}, object())
            self.assertEqual([target["key"] for target in targets], ["clip_l"])
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(scan.call_args.kwargs["companion_rows"], self.roles)
            values = dict(prompt="keep exactly", image_component_profiles={})
            choices, scope = use_component(
                self.row, "clip_l", self.checkpoint, "SDXL", values, object()
            )
            self.assertEqual(choices, dict(clip_l="encoder.gguf"))
            self.assertEqual(scope, checkpoint_identity(self.checkpoint))
            self.assertEqual(values, dict(prompt="keep exactly", image_component_profiles={}))
        self.assertEqual(self.row, original)

    def test_unknown_incompatible_locked_and_same_name_other_root_never_activate(self):
        for compatibility, path, locked in (
            ("unknown", self.component, False),
            ("incompatible", self.component, False),
            ("compatible", self.root / "other/encoder.gguf", False),
            ("compatible", self.component, True),
        ):
            with self.subTest(compatibility=compatibility, path=path, locked=locked):
                roles = [dict(self.roles[0], locked=locked)]
                with (
                    patch(
                        "ruinedfooocus.features.models.component_profiles.component_inventory",
                        return_value=dict(rows=roles),
                    ),
                    patch(
                        "ruinedfooocus.features.models.component_candidates.installed_candidates",
                        return_value=[
                            dict(path=str(path), name="encoder.gguf", compatibility=compatibility)
                        ],
                    ),
                ):
                    self.assertEqual(
                        component_targets(self.row, self.checkpoint, "SDXL", {}, object()), []
                    )
                    with self.assertRaisesRegex(ValueError, "compatible"):
                        use_component(self.row, "clip_l", self.checkpoint, "SDXL", {}, object())

    def test_replaced_component_and_checkpoint_scope_are_revalidated_before_use(self):
        self.component.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            component_targets(self.row, self.checkpoint, "SDXL", {}, object())
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            use_component(self.row, "clip_l", self.checkpoint, "SDXL", {}, object(), scope="stale")

    def test_selected_primary_download_resume_retry_offline_and_companions(self):
        missing = dict(key="model-a", local=None, allowed=True, size_matches=True)
        self.assertEqual(action_state(missing)["label"], "Download")
        self.assertEqual(
            action_state(missing, transfer=dict(resumable=True))["label"], "Resume download"
        )
        for state in ("failed", "cancelled"):
            job = dict(selection_key="model-a", state=state, active=False)
            self.assertEqual(action_state(missing, job=job)["label"], "Retry download")
        offline = action_state(missing, offline=True)
        self.assertTrue(offline["download"])
        self.assertFalse(offline["download_enabled"])
        installed = action_state(self.row, can_use=True, companions_missing=True)
        self.assertTrue(installed["use"])
        self.assertTrue(installed["secondary"])
        self.assertEqual(installed["label"], "Download missing companions")

    def test_download_a_does_not_block_using_installed_b_and_never_redownloads_it(self):
        job = dict(selection_key="model-a", active=True, state="running")
        selected_a = action_state(dict(key="model-a", local=None, allowed=True), job=job)
        self.assertTrue(selected_a["cancel"])
        self.assertFalse(selected_a["download"])
        selected_b = action_state(self.row, job=job, can_use=True)
        self.assertTrue(selected_b["use"])
        self.assertFalse(selected_b["download"])
        self.assertFalse(selected_b["cancel"])
        invalid = action_state(dict(self.row, size_matches=False), can_use=True)
        self.assertFalse(invalid["use"])


if __name__ == "__main__":
    unittest.main()
