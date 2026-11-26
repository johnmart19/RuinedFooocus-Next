"""Atomic step-preview writes, live-preview cleanup, and interrupt recovery."""

import ast
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.app.resource_lifecycle import JobIdentity, WorkItem, cleanup_interrupt, job_scope
from ruinedfooocus.features.gallery.index_store import LIVE_PREVIEW, discard_live_previews

ROOT = Path(__file__).resolve().parents[1]


def _load_worker_helpers():
    source = (ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    scope = {
        "Path": Path,
        "shared": NS(path_manager=NS(model_paths={"temp_preview_path": ""})),
        "current_job": lambda: None,
    }
    for name in ("_preview_path", "_save_step_preview"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "async_worker.py", "exec"), scope)
    return scope["_preview_path"], scope["_save_step_preview"]


_preview_path, _save_step_preview = _load_worker_helpers()


class LivePreviewAndStopRecoveryTests(unittest.TestCase):
    def test_save_step_preview_writes_unique_atomic_jpegs_and_prunes_old_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "preview.jpg"
            item = WorkItem(JobIdentity(1, "abc123", ("ui", "", "local", "sess")), threading.Event())
            gen_data = {
                "_job": item,
                "_image_capture_paths": NS(model_paths={"temp_preview_path": str(base)}),
            }
            frame = Image.new("RGB", (64, 64), color=(180, 40, 40))

            p1 = Path(_save_step_preview(frame, gen_data, final=False))
            p2 = Path(_save_step_preview(frame, gen_data, final=False))
            p3 = Path(_save_step_preview(frame, gen_data, final=False))

            self.assertNotEqual(p1, p2)
            self.assertNotEqual(p2, p3)
            self.assertTrue(LIVE_PREVIEW.fullmatch(p2.name))
            self.assertTrue(LIVE_PREVIEW.fullmatch(p3.name))
            # Only the 2 most recent step previews remain on disk.
            self.assertFalse(p1.exists())
            self.assertTrue(p2.is_file())
            self.assertTrue(p3.is_file())
            # Both remaining files are complete, readable JPEGs with no leftover .tmp files.
            for path in (p2, p3):
                with Image.open(path) as opened:
                    self.assertEqual(opened.size, (64, 64))
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])

            final_path = Path(_save_step_preview(frame, gen_data, final=True))
            self.assertEqual(final_path.name, "preview-abc123.jpg")
            self.assertTrue(final_path.is_file())
            self.assertEqual(gen_data["_last_preview_path"], str(final_path))

    def test_discard_live_previews_removes_all_step_files_for_job_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_base = root / "preview-job1.jpg"
            step1 = root / "preview-job1-1.jpg"
            step2 = root / "preview-job1-2.jpg"
            stray_tmp = root / ".preview-job1-3.tmp"
            other_job = root / "preview-job2-1.jpg"
            for path in (job_base, step1, step2, stray_tmp, other_job):
                path.write_bytes(b"data")

            discard_live_previews(job_base, keep=[])

            self.assertFalse(job_base.exists())
            self.assertFalse(step1.exists())
            self.assertFalse(step2.exists())
            self.assertFalse(stray_tmp.exists())
            self.assertTrue(other_job.is_file())

    def test_discard_live_previews_preserves_kept_preview_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_base = root / "preview-job1.jpg"
            step1 = root / "preview-job1-1.jpg"
            job_base.write_bytes(b"grid")
            step1.write_bytes(b"step")

            discard_live_previews(job_base, keep=[str(job_base)])

            self.assertTrue(job_base.is_file())
            self.assertFalse(step1.exists())

    def test_cleanup_interrupt_resets_native_flag_for_owning_job(self):
        calls = []
        fake_mm = NS(interrupt_current_processing=lambda value: calls.append(value))
        item = WorkItem(JobIdentity(2, "def456", ("ui", "", "local", "sess")), threading.Event())
        item.state["native_interrupt"] = True
        with patch.dict(sys.modules, {"comfy.model_management": fake_mm}):
            with job_scope(item):
                cleanup_interrupt(item)
        self.assertFalse(item.state.get("native_interrupt", False))
        self.assertEqual(calls, [False])


if __name__ == "__main__":
    unittest.main()
