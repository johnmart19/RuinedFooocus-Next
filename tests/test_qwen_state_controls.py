"""State matching must retain sequential failures and reject incomplete evidence."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.compare_qwen_state_controls import compare, SOURCE_FILES


class QwenStateControlTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        kwargs = dict(
            family="qwen",
            seed=123,
            steps=20,
            sampling_shift=3.0,
            diffusion_model="weights.safetensors",
            source_image="source.png",
        )
        self.report = dict(
            comfy_revision="a" * 40,
            comparison={"passed": False},
            spec=dict(
                runtime_args=["--use-quad-cross-attention"],
                app=dict(entrypoint="tools.comfy_feature_app:image_edit", kwargs=kwargs),
                native=dict(
                    entrypoint="tools.comfy_feature_reference:image_edit",
                    kwargs=copy.deepcopy(kwargs),
                ),
            ),
            provenance={"source_files_sha256": {name: "c" * 64 for name in SOURCE_FILES}},
            runs={},
        )
        for side, digit in (("app", "1"), ("native", "2")):
            tensor = dict(kind="tensor", shape=[1, 2], dtype="torch.float32", sha256=digit * 64)
            self.report["runs"][side] = {
                "artifacts": dict(
                    conditioning={"positive": [copy.deepcopy(tensor)]},
                    latent=copy.deepcopy(tensor),
                    output=copy.deepcopy(tensor),
                    settings={"steps": 20},
                )
            }
        self.native = copy.deepcopy(self.report)
        self.native["spec"]["app"]["entrypoint"] = "tools.comfy_feature_reference:image_edit"

    def write(self, name, report, *, digest=True, exit_code=1, error=None):
        path = self.root / (name + "-comparison.json")
        body = json.dumps(report).encode()
        path.write_bytes(body)
        process = dict(exit_code=exit_code, passed=exit_code == 0)
        if digest:
            process["comparison_sha256"] = hashlib.sha256(body).hexdigest()
        if error:
            process["error"] = error
        process_path = self.root / (name + "-process.json")
        process_path.write_text(json.dumps(process))
        return path, process_path

    def pair(self):
        return self.write("application", self.report)[0], self.write("native", self.native)[0]

    def test_exact_cold_cold_and_warm_warm_retains_sequential_failures(self):
        result = compare(*self.pair())
        self.assertTrue(result["state_matched_core_equivalence_supported"])
        self.assertTrue(result["complete_source_provenance_verified"])
        self.assertEqual(
            result["original_sequential_results"], [{"passed": False}, {"passed": False}]
        )
        self.assertIn("Does not establish sequential cold/warm equality", result["scope"])
        self.assertNotEqual(self.report["runs"]["app"], self.report["runs"]["native"])

    def test_changed_seed_or_source_hash_rejects_even_identical_tensor_hashes(self):
        for change in ("seed", "source"):
            with self.subTest(change=change):
                changed = copy.deepcopy(self.native)
                if change == "seed":
                    changed["spec"]["app"]["kwargs"]["seed"] = 999
                else:
                    changed["provenance"]["source_files_sha256"][SOURCE_FILES[0]] = "f" * 64
                a = self.write("application", self.report)[0]
                b = self.write("native", changed)[0]
                self.assertFalse(compare(a, b)["state_matched_core_equivalence_supported"])

    def test_missing_empty_invalid_or_zero_sized_tensor_fingerprints_reject(self):
        for change in ("missing", "empty", "bad_hex", "empty_tensor"):
            with self.subTest(change=change):
                report = copy.deepcopy(self.report)
                artifacts = report["runs"]["app"]["artifacts"]
                if change == "missing":
                    del artifacts["conditioning"]
                elif change == "empty":
                    artifacts["output"] = {}
                elif change == "bad_hex":
                    artifacts["latent"]["sha256"] = "z" * 64
                else:
                    artifacts["latent"]["shape"] = [0, 2]
                with self.assertRaises(ValueError):
                    compare(
                        self.write("application", report)[0], self.write("native", self.native)[0]
                    )

    def test_missing_stale_crashed_or_wrong_exit_process_records_reject(self):
        for change in ("missing", "stale_hash", "stale_legacy", "crash", "wrong_exit", "null_hash"):
            with self.subTest(change=change):
                a, process = self.write("application", self.report, digest=change != "stale_legacy")
                b = self.write("native", self.native)[0]
                if change == "missing":
                    process.unlink()
                elif change == "stale_legacy":
                    os.utime(process, ns=(1, 1))
                else:
                    record = json.loads(process.read_text())
                    if change == "stale_hash":
                        record["comparison_sha256"] = "f" * 64
                    elif change == "null_hash":
                        record["comparison_sha256"] = None
                    elif change == "crash":
                        record.update(exit_code=2, error="native runtime crashed")
                    else:
                        record["exit_code"] = 0
                    process.write_text(json.dumps(record))
                self.assertFalse(compare(a, b)["state_matched_core_equivalence_supported"])

    def test_current_legacy_record_and_passing_raw_exit_zero_are_accepted(self):
        a, process = self.write("application", self.report, digest=False)
        os.utime(process, ns=(a.stat().st_mtime_ns + 1_000_000, a.stat().st_mtime_ns + 1_000_000))
        b = self.write("native", self.native)[0]
        self.assertTrue(compare(a, b)["state_matched_core_equivalence_supported"])
        passed = copy.deepcopy(self.report)
        passed["comparison"]["passed"] = True
        a = self.write("application", passed, exit_code=0)[0]
        self.assertTrue(compare(a, b)["state_matched_core_equivalence_supported"])

    def test_native_crash_with_missing_artifacts_is_not_equivalence_evidence(self):
        crashed = copy.deepcopy(self.native)
        crashed["runs"]["native"] = {"error": "model failed"}
        with self.assertRaisesRegex(ValueError, "Missing or failed native"):
            compare(self.write("application", self.report)[0], self.write("native", crashed)[0])


if __name__ == "__main__":
    unittest.main()
