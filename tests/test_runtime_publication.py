"""Bounded directory publication retries use inert files and no native execution."""

from concurrent.futures import CancelledError
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.packaging import install_lock as publication


def windows_error(code):
    error = PermissionError(errno.EACCES, "fixture publication denied")
    error.winerror = code
    return error


class RuntimePublicationTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.source, self.target = root / "stage", root / "runtime"
        self.source.mkdir()
        (self.source / "payload.bin").write_bytes(b"inert runtime")
        self.enterContext(patch.object(publication, "_WINDOWS", True))
        self.sleep = self.enterContext(patch.object(publication.time, "sleep"))
        self.rename = publication.os.rename

    def test_transient_access_and_sharing_denials_then_success_revalidate(self):
        errors = iter([windows_error(5), windows_error(32), None])

        def attempt(source, target):
            error = next(errors)
            if error:
                raise error
            self.rename(source, target)

        validate, cancelled = Mock(), Mock()
        with patch.object(publication.os, "rename", side_effect=attempt) as rename:
            publication.publish_runtime_directory(
                self.source, self.target, validate=validate, check_cancelled=cancelled
            )
        self.assertEqual(rename.call_count, 3)
        self.assertEqual(validate.call_count, 3)
        self.assertGreater(cancelled.call_count, 3)
        self.assertEqual((self.target / "payload.bin").read_bytes(), b"inert runtime")
        self.assertFalse(self.source.exists())

    def test_persistent_access_denied_preserves_original_error_and_source(self):
        original = windows_error(5)
        with patch.object(publication.os, "rename", side_effect=original) as rename:
            with self.assertRaises(PermissionError) as raised:
                publication.publish_runtime_directory(self.source, self.target)
        self.assertIs(raised.exception, original)
        self.assertEqual(rename.call_count, 6)
        self.assertLessEqual(sum(call.args[0] for call in self.sleep.call_args_list), 1.551)
        self.assertTrue((self.source / "payload.bin").is_file())
        self.assertFalse(self.target.exists())

    def test_cancellation_during_wait_does_not_publish(self):
        cancelled = Mock(side_effect=[None, CancelledError()])
        with patch.object(publication.os, "rename", side_effect=windows_error(32)) as rename:
            with self.assertRaises(CancelledError):
                publication.publish_runtime_directory(
                    self.source, self.target, check_cancelled=cancelled
                )
        self.assertEqual(rename.call_count, 1)
        self.assertTrue(self.source.exists())
        self.assertFalse(self.target.exists())

    def test_destination_appearing_after_denial_is_never_replaced(self):
        def racing(source, target):
            self.target.mkdir()
            (self.target / "manual.bin").write_bytes(b"retain")
            raise windows_error(5)

        with patch.object(publication.os, "rename", side_effect=racing) as rename:
            with self.assertRaises(FileExistsError):
                publication.publish_runtime_directory(self.source, self.target)
        self.assertEqual(rename.call_count, 1)
        self.assertEqual((self.target / "manual.bin").read_bytes(), b"retain")
        self.assertTrue(self.source.exists())

    def test_nonwindows_and_unrelated_errors_are_not_retried(self):
        for windows, code in ((False, 5), (True, 3), (True, 17)):
            with self.subTest(windows=windows, code=code):
                with (
                    patch.object(publication, "_WINDOWS", windows),
                    patch.object(
                        publication.os, "rename", side_effect=windows_error(code)
                    ) as rename,
                ):
                    with self.assertRaises(PermissionError):
                        publication.publish_runtime_directory(self.source, self.target)
                self.assertEqual(rename.call_count, 1)
        self.sleep.assert_not_called()

    def test_source_validation_failure_after_denial_stops_without_retrying_rename(self):
        validate = Mock(side_effect=[None, RuntimeError("source changed")])
        with patch.object(publication.os, "rename", side_effect=windows_error(5)) as rename:
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                publication.publish_runtime_directory(self.source, self.target, validate=validate)
        self.assertEqual(rename.call_count, 1)
        self.assertTrue(self.source.exists())
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
