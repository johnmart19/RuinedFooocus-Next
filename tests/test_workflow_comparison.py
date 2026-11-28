import unittest
import tempfile
from pathlib import Path

import numpy as np

from tools.validate_comfy_workflows import capture, compare, cleanup_runtime, provenance
from unittest.mock import Mock, patch


class WorkflowReportTests(unittest.TestCase):
    def snapshot(self, value):
        arrays = {}
        return capture(value, arrays), arrays

    def test_identical_nested_artifacts(self):
        a, aa = self.snapshot({"latent": np.zeros((1, 4, 2, 2)), "settings": {"seed": 7}})
        b, bb = self.snapshot({"latent": np.zeros((1, 4, 2, 2)), "settings": {"seed": 7}})
        self.assertTrue(compare(a, b, aa, bb)["passed"])

    def test_tolerance_does_not_hide_settings_mismatch(self):
        a, aa = self.snapshot({"output": np.array([1.0]), "settings": {"seed": 7}})
        b, bb = self.snapshot({"output": np.array([1.001]), "settings": {"seed": 8}})
        result = compare(a, b, aa, bb, atol=0.01)
        self.assertFalse(result["passed"])
        self.assertTrue(result["numeric"]["root/output"]["within_tolerance"])

    def test_tolerated_numeric_error_is_distinct_from_exact_match(self):
        a, aa = self.snapshot({"output": np.array([1.0])})
        b, bb = self.snapshot({"output": np.array([1.001])})
        result = compare(a, b, aa, bb, atol=0.01)
        self.assertTrue(result["passed"])
        self.assertFalse(result["exact_fingerprints_equal"])

    def test_missing_artifact_and_nonfinite_output_fail(self):
        a, aa = self.snapshot({"output": np.array([np.nan])})
        b, bb = self.snapshot({"output": np.array([np.nan])})
        self.assertFalse(compare(a, b, aa, bb)["passed"])
        self.assertFalse(compare(a, {}, aa, {})["passed"])

    def test_provenance_tracks_dirty_source_bytes_without_weight_hashing(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1] / "tmp"
        ) as directory:
            root = Path(directory)
            (root / "modules" / "nested").mkdir(parents=True)
            source = root / "modules" / "nested" / "example.py"
            source.write_text("value = 1\n", encoding="utf-8")
            weight = root / "weight.safetensors"
            weight.write_bytes(b"excluded")

            def git(command, **kwargs):
                self.assertEqual(command[:3], ["git", "-C", str(root)])
                if command[3:] == ["rev-parse", "--show-toplevel"]:
                    return str(root)
                self.assertEqual(command[3:], ["status", "--porcelain"])
                return " M modules/nested/example.py\n"

            read_bytes = Path.read_bytes

            def read_source(path):
                self.assertNotEqual(path, weight, "Provenance must not read model weights")
                return read_bytes(path)

            with (
                patch("tools.validate_comfy_workflows.subprocess.check_output", side_effect=git),
                patch.object(Path, "read_bytes", read_source),
            ):
                first = provenance(root)
                source.write_text("value = 2\n", encoding="utf-8")
                second = provenance(root)
            self.assertTrue(second["app_dirty"])
            self.assertEqual(set(second["source_files_sha256"]), {"modules/nested/example.py"})
            self.assertNotEqual(first["source_tree_sha256"], second["source_tree_sha256"])
            self.assertNotEqual(first["source_files_sha256"], second["source_files_sha256"])

    def test_cleanup_synchronizes_before_releasing_native_owners(self):
        order = []
        memory, torch, gc = Mock(), Mock(), Mock()
        torch.cuda.is_initialized.return_value = True
        torch.cuda.synchronize.side_effect = lambda: order.append("synchronize")
        memory.unload_all_models.side_effect = lambda: order.append("unload")
        memory.reset_cast_buffers.side_effect = lambda: order.append("cast_buffers")
        memory.cleanup_models.side_effect = lambda: order.append("models")
        result = cleanup_runtime(memory, torch, gc)
        self.assertEqual(order, ["synchronize", "unload", "cast_buffers", "models"])
        self.assertTrue(result["process_exit_requires_external_verification"])
        self.assertEqual(gc.collect.call_count, 2)

    def test_model_objects_cannot_be_silently_ignored(self):
        with self.assertRaisesRegex(TypeError, "Unsupported artifact"):
            self.snapshot({"model": object()})


if __name__ == "__main__":
    unittest.main()
