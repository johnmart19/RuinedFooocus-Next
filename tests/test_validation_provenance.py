"""Native comparison fingerprints follow the maintained package layout."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from tools.validate_comfy_workflows import provenance, revision


class ValidationProvenanceTests(unittest.TestCase):
    def test_nested_export_uses_declared_commit_and_backend_pin_without_ancestor_git(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "enclosing-checkout/cache/export"
            root.mkdir(parents=True)
            pin = "b" * 40
            identity = {
                "kind": "source-archive",
                "source_commit": "a" * 40,
                "backend_pins": {"ComfyUI": pin},
            }
            (root / "distribution.json").write_text(json.dumps(identity), encoding="utf-8")
            with patch(
                "tools.validate_comfy_workflows.subprocess.check_output",
                side_effect=AssertionError("Do not inherit parent Git"),
            ):
                self.assertEqual(revision(root), "a" * 40)
                self.assertEqual(revision(root / "repositories/ComfyUI"), pin)
                observed = provenance(root)
            self.assertIsNone(observed["app_dirty"])
            self.assertEqual(observed["declared_export_identity"]["backend_pins"], {"ComfyUI": pin})

    def test_unmarked_or_invalid_nested_export_does_not_inherit_parent_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache/export"
            root.mkdir(parents=True)
            for manifest in (
                None,
                {},
                {
                    "kind": "source-archive",
                    "source_commit": "a" * 40,
                    "backend_pins": {"ComfyUI": "invalid"},
                },
            ):
                if manifest is not None:
                    (root / "distribution.json").write_text(json.dumps(manifest), encoding="utf-8")
                with patch(
                    "tools.validate_comfy_workflows.subprocess.check_output", return_value=folder
                ) as git:
                    self.assertIsNone(revision(root))
                    self.assertIsNone(provenance(root)["app_dirty"])
                    self.assertTrue(
                        all(call.args[0][-1] == "--show-toplevel" for call in git.call_args_list)
                    )

    def test_real_checkout_keeps_git_revision_and_dirty_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def git(command, **kwargs):
                return {
                    "--show-toplevel": str(root),
                    "HEAD": "c" * 40,
                    "--porcelain": " M owned.py",
                }[command[-1]]

            with patch("tools.validate_comfy_workflows.subprocess.check_output", side_effect=git):
                self.assertEqual(revision(root), "c" * 40)
                self.assertTrue(provenance(root)["app_dirty"])

    def test_feature_backend_and_resource_changes_invalidate_source_fingerprint(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("tools.validate_comfy_workflows.subprocess.check_output", return_value=b""),
        ):
            root = Path(folder)
            feature = root / "ruinedfooocus/features/generation/image_pipeline.py"
            backend = root / "ruinedfooocus/backends/comfy/image_compute.py"
            recipe = root / "ruinedfooocus/resources/model_catalogue/model.json"
            for path, content in (
                (feature, "prompt = 'original'"),
                (backend, "precision = 'auto'"),
                (recipe, '{"url":"source"}'),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            before = provenance(root)
            for path in (feature, backend, recipe):
                self.assertIn(path.relative_to(root).as_posix(), before["source_files_sha256"])
                path.write_text(path.read_text() + "\n")
                after = provenance(root)
                self.assertNotEqual(before["source_tree_sha256"], after["source_tree_sha256"])
                before = after
