"""Validate artifact integrity and overwrite protection without network access."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tools import download_validation_assets as download


class ValidationAssetTests(unittest.TestCase):
    def test_state_writer_never_opens_predictable_temporary_path(self):
        # Emulate the open behavior of a fixed temporary symlink without needing
        # Windows link privileges: opening it redirects to an unrelated target.
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            asset = self.range_asset()
            with patch.object(download, "ROOT", root):
                stage = download.stage_folder(asset)
                stage.mkdir(parents=True)
                _, record, state, _ = download.range_state(asset)
                unrelated = root / "unrelated.txt"
                unrelated.write_bytes(b"preserve unrelated content")
                fixed = record.with_suffix(".json.tmp")
                fixed.write_bytes(b"preserve fixed placeholder")
                original_open = Path.open

                def redirected_open(path, *args, **kwargs):
                    return original_open(unrelated if path == fixed else path, *args, **kwargs)

                with patch.object(Path, "open", redirected_open):
                    download.write_range_state(record, state)
                self.assertEqual(unrelated.read_bytes(), b"preserve unrelated content")
                self.assertEqual(fixed.read_bytes(), b"preserve fixed placeholder")
                self.assertEqual(json.loads(record.read_text(encoding="utf-8")), state)
                self.assertEqual(list(stage.glob("range-state-*.tmp")), [])
                original_record = record.read_bytes()
                with patch.object(download.json, "dump", side_effect=OSError("write interrupted")):
                    with self.assertRaises(OSError):
                        download.write_range_state(record, state)
                self.assertEqual(record.read_bytes(), original_record)
                self.assertEqual(list(stage.glob("range-state-*.tmp")), [])
                self.assertEqual(unrelated.read_bytes(), b"preserve unrelated content")

    def range_asset(self):
        return dict(
            destination="models/test.bin",
            revision="a" * 40,
            repo="test/repository",
            file="nested/test.bin",
            sha256=hashlib.sha256(b"goodgood").hexdigest(),
            bytes=8,
        )

    def response(self, body, content_range="bytes 0-3/8", status=206):
        class Response:
            status_code = status
            headers = {"Content-Range": content_range}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            def iter_content(self, size):
                yield body

        return Response()

    def test_ranged_timeout_preserves_verified_prefix_and_retries_only_remaining_range(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            asset = self.range_asset()
            with (
                patch.object(download, "ROOT", root),
                patch.object(download, "RANGE_CHUNK", 4),
                patch(
                    "requests.get",
                    side_effect=[self.response(b"good"), TimeoutError("interrupted")],
                ) as request,
            ):
                with self.assertRaises(TimeoutError):
                    download.acquire(asset, ranged=True)
                payload, record, state, prefix = download.range_state(asset)
                self.assertEqual(prefix, 4)
                self.assertEqual(download.remaining_bytes(asset, ranged=True), 4)
                self.assertEqual(download.remaining_bytes(asset), 8)
                self.assertEqual(request.call_args_list[1].kwargs["headers"]["Range"], "bytes=4-7")
                with payload.open("ab") as handle:
                    handle.write(b"uncommitted tail")
                with patch(
                    "requests.get", return_value=self.response(b"good", "bytes 4-7/8")
                ) as retry:
                    result = download.acquire(asset, ranged=True)
                self.assertEqual(retry.call_args.kwargs["headers"]["Range"], "bytes=4-7")
                self.assertEqual(result["status"], "verified")
                self.assertFalse(payload.exists())
                self.assertFalse(record.exists())
            self.assertEqual((root / "models/test.bin").read_bytes(), b"goodgood")

    def test_invalid_or_short_range_never_commits_bytes(self):
        for response in (
            self.response(b"good", status=200),
            self.response(b"good", "bytes 1-4/8"),
            self.response(b"bad"),
            self.response(b"too long"),
        ):
            with (
                self.subTest(headers=response.headers, status=response.status_code),
                tempfile.TemporaryDirectory(dir=download.ROOT) as folder,
            ):
                root = Path(folder)
                (root / "models").mkdir()
                with (
                    patch.object(download, "ROOT", root),
                    patch.object(download, "RANGE_CHUNK", 4),
                    patch("requests.get", return_value=response),
                ):
                    with self.assertRaises(ValueError):
                        download.acquire(self.range_asset(), ranged=True)
                    self.assertEqual(download.range_state(self.range_asset())[3], 0)
                self.assertFalse((root / "models/test.bin").exists())

    def test_prefix_corruption_is_rejected_and_unrelated_partials_are_not_credited(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            asset = self.range_asset()
            with patch.object(download, "ROOT", root), patch.object(download, "RANGE_CHUNK", 4):
                stage = download.stage_folder(asset)
                stage.mkdir(parents=True)
                (stage / "unrelated.incomplete").write_bytes(b"ignored!")
                self.assertEqual(download.remaining_bytes(asset, ranged=True), 8)
                payload, record, state, _ = download.range_state(asset)
                payload.write_bytes(b"evil")
                state["chunks"].append(dict(bytes=4, sha256=hashlib.sha256(b"good").hexdigest()))
                download.write_range_state(record, state)
                with self.assertRaisesRegex(ValueError, "integrity"):
                    download.remaining_bytes(asset, ranged=True)

    def test_cli_preflight_credits_only_verified_ranged_prefix(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            (root / "tools").mkdir()
            asset = self.range_asset()
            (root / "tools/validation_assets.json").write_text(
                json.dumps({"assets": [asset], "provenance": {}})
            )
            with patch.object(download, "ROOT", root), patch.object(download, "RANGE_CHUNK", 4):
                stage = download.stage_folder(asset)
                stage.mkdir(parents=True)
                payload, record, state, _ = download.range_state(asset)
                payload.write_bytes(b"good")
                state["chunks"].append(dict(bytes=4, sha256=hashlib.sha256(b"good").hexdigest()))
                download.write_range_state(record, state)
                with (
                    patch("sys.argv", ["download", "--download", "--ranged"]),
                    patch.object(
                        download.shutil,
                        "disk_usage",
                        return_value=SimpleNamespace(free=10 * 1024**3 + 4),
                    ),
                    patch.object(
                        download,
                        "acquire",
                        return_value=dict(status="verified", destination=asset["destination"]),
                    ) as acquire,
                ):
                    self.assertEqual(download.main(), 0)
                    acquire.assert_called_once_with(asset, ranged=True)

    def test_completed_stage_is_verified_and_promoted_without_transport(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            asset = self.range_asset()
            with patch.object(download, "ROOT", root):
                stage = download.stage_folder(asset)
                stage.mkdir(parents=True)
                (stage / "test.bin").write_bytes(b"goodgood")
                self.assertEqual(download.remaining_bytes(asset), 0)
                with (
                    patch("requests.get", side_effect=AssertionError("No HTTP expected")),
                    patch.dict("sys.modules", {"ruinedfooocus.core.hub_download": None}),
                ):
                    self.assertEqual(download.acquire(asset)["status"], "verified")
                self.assertEqual((root / "models/test.bin").read_bytes(), b"goodgood")

    def test_ranged_promotion_preserves_competing_target_and_own_verified_payload(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            asset = self.range_asset()

            def response(*args, **kwargs):
                (root / "models/test.bin").write_bytes(b"other owner")
                return self.response(b"goodgood", "bytes 0-7/8")

            with patch.object(download, "ROOT", root), patch("requests.get", side_effect=response):
                with self.assertRaises(FileExistsError):
                    download.acquire(asset, ranged=True)
                payload, _, _, prefix = download.range_state(asset)
                self.assertEqual(prefix, 8)
                self.assertEqual(payload.read_bytes(), b"goodgood")
                self.assertEqual((root / "models/test.bin").read_bytes(), b"other owner")

    def test_timeout_keeps_partial_and_retry_reuses_identical_hub_directory(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            asset = dict(
                destination="models/test.bin",
                revision="a" * 40,
                repo="test/repository",
                file="nested/test.bin",
                sha256=hashlib.sha256(b"good").hexdigest(),
                bytes=4,
            )
            calls = []

            def resumable_download(url, path):
                calls.append(path)
                partial = path.parent / ".cache/huggingface/download/test.incomplete"
                if len(calls) == 1:
                    partial.parent.mkdir(parents=True)
                    partial.write_bytes(b"go")
                    raise TimeoutError("interrupted after two bytes")
                self.assertEqual(partial.read_bytes(), b"go")
                path.write_bytes(partial.read_bytes() + b"od")
                partial.unlink()

            fake_hub = type("Hub", (), {"download_hub_file": staticmethod(resumable_download)})
            with (
                patch.object(download, "ROOT", root),
                patch.dict("sys.modules", {"ruinedfooocus.core.hub_download": fake_hub}),
            ):
                with self.assertRaises(TimeoutError):
                    download.acquire(asset)
                stage = root / "models/.validation-downloads" / asset["sha256"]
                self.assertEqual(
                    (stage / ".cache/huggingface/download/test.incomplete").read_bytes(), b"go"
                )
                self.assertFalse((root / "models/test.bin").exists())
                result = download.acquire(asset)
            self.assertEqual(calls, [stage / "test.bin", stage / "test.bin"])
            self.assertEqual(result["status"], "verified")
            self.assertEqual((root / "models/test.bin").read_bytes(), b"good")
            self.assertFalse((stage / "test.bin").exists())
            self.assertTrue((stage / ".cache/huggingface/download").is_dir())

    def test_existing_mismatch_is_preserved_and_matching_file_is_accepted(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            path = root / "models/test.bin"
            path.parent.mkdir()
            path.write_bytes(b"bad")
            asset = dict(
                destination="models/test.bin",
                revision="a" * 40,
                sha256=hashlib.sha256(b"good").hexdigest(),
                bytes=4,
            )
            with patch.object(download, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Size mismatch"):
                    download.acquire(asset)
                self.assertEqual(path.read_bytes(), b"bad")
                path.write_bytes(b"evil")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    download.acquire(asset)
                self.assertEqual(path.read_bytes(), b"evil")
                path.write_bytes(b"good")
                with patch.object(download, "verify", wraps=download.verify) as verify:
                    result = download.acquire(asset)
                verify.assert_called_once()
                self.assertEqual(
                    result,
                    dict(
                        destination="models/test.bin",
                        status="verified",
                        action="existing",
                        bytes=4,
                        sha256=asset["sha256"],
                        actual_sha256=asset["sha256"],
                    ),
                )
                asset["destination"] = "models/../outside.bin"
                with self.assertRaisesRegex(ValueError, "inside ROOT/models"):
                    download.destination(asset)

    def test_download_report_has_summary_compatible_digest_evidence(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            digest = hashlib.sha256(b"good").hexdigest()
            asset = dict(
                destination="models/test.bin",
                revision="a" * 40,
                repo="test/repository",
                file="nested/test.bin",
                sha256=digest,
                bytes=4,
            )
            fake_hub = type(
                "Hub",
                (),
                {"download_hub_file": staticmethod(lambda url, path: path.write_bytes(b"good"))},
            )
            with (
                patch.object(download, "ROOT", root),
                patch.dict("sys.modules", {"ruinedfooocus.core.hub_download": fake_hub}),
                patch.object(download, "verify", wraps=download.verify) as verify,
            ):
                result = download.acquire(asset)
            verify.assert_called_once()
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["action"], "downloaded")
            self.assertEqual(result["sha256"], result["actual_sha256"])
            self.assertEqual((root / "models/test.bin").read_bytes(), b"good")

    def test_a_file_created_during_download_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=download.ROOT) as folder:
            root = Path(folder)
            (root / "models").mkdir()
            asset = dict(
                destination="models/test.bin",
                revision="a" * 40,
                repo="test/repository",
                file="test.bin",
                sha256=hashlib.sha256(b"good").hexdigest(),
                bytes=4,
            )

            def competing_download(url, path):
                path.write_bytes(b"good")
                (root / "models/test.bin").write_bytes(b"other owner")

            fake_hub = type("Hub", (), {"download_hub_file": staticmethod(competing_download)})
            with (
                patch.object(download, "ROOT", root),
                patch.dict("sys.modules", {"ruinedfooocus.core.hub_download": fake_hub}),
            ):
                with self.assertRaises(FileExistsError):
                    download.acquire(asset)
            self.assertEqual((root / "models/test.bin").read_bytes(), b"other owner")


if __name__ == "__main__":
    unittest.main()
