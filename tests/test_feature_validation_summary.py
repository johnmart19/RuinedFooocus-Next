"""CPU-only integrity checks for image comparison reports and contact sheets."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.summarize_feature_validation import summarize

ROOT = Path(__file__).resolve().parents[1]


class SummaryIntegrityTests(unittest.TestCase):
    def test_exit_record_must_match_current_comparison(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            report = root / "case-comparison.json"
            process = root / "case-process.json"
            report.write_text(json.dumps({"comparison": {"passed": True}}))
            self.assertEqual(summarize(root)["reports"]["case"]["status"], "exit unrecorded")
            process.write_text(json.dumps({"exit_code": 0, "comparison_sha256": "0" * 64}))
            self.assertEqual(summarize(root)["reports"]["case"]["status"], "process record stale")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            process.write_text(json.dumps({"exit_code": 0, "comparison_sha256": digest}))
            # Matching content is reliable even if timestamps were preserved on copy.
            os.utime(process, ns=(1, 1))
            self.assertEqual(summarize(root)["reports"]["case"]["status"], "passed")
            process.write_text(json.dumps({"exit_code": 0}))
            os.utime(process, ns=(1, 1))
            self.assertEqual(summarize(root)["reports"]["case"]["status"], "process record stale")
            timestamp = report.stat().st_mtime_ns + 1000000
            os.utime(process, ns=(timestamp, timestamp))
            self.assertEqual(summarize(root)["reports"]["case"]["status"], "passed")

    def test_image_contact_sheet_preserves_inputs_and_ignores_removed_video_routes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            image = root / "case-app.png"
            Image.new("RGB", (32, 24), "blue").save(image)
            original = image.read_bytes()
            video = root / "old-app.mp4"
            video.write_bytes(b"historical output is not decoded")
            result = summarize(root)
            self.assertEqual(result["images"], ["case-app.png"])
            self.assertNotIn("media", result)
            self.assertTrue((root / "contact-sheet.jpg").is_file())
            self.assertEqual(image.read_bytes(), original)
            self.assertEqual(video.read_bytes(), b"historical output is not decoded")


if __name__ == "__main__":
    unittest.main()
