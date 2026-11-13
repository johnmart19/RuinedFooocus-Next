"""Recognize only the owning checkout before offering to close a listener."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.app.startup_port import ruinedfooocus_owner
from ruinedfooocus.core.paths import APP_ROOT


class StartupPortTests(unittest.TestCase):
    def test_supported_launchers_are_recognized_without_terminating_them(self):
        for script in ("launch.py", "entry_with_update.py"):
            with self.subTest(script=script):
                process = Mock()
                process.name.return_value = "python.exe"
                process.cmdline.return_value = ["python", "-u", script]
                process.cwd.return_value = str(APP_ROOT)
                with (
                    patch(
                        "psutil.net_connections",
                        return_value=[
                            SimpleNamespace(
                                pid=987654, status="LISTEN", laddr=SimpleNamespace(port=7860)
                            )
                        ],
                    ),
                    patch("psutil.Process", return_value=process),
                ):
                    self.assertIs(ruinedfooocus_owner(7860), process)
                process.create_time.assert_called_once()
                process.terminate.assert_not_called()

    def test_same_filename_from_another_checkout_is_not_recognized(self):
        process = Mock()
        process.name.return_value = "python.exe"
        process.cmdline.return_value = [
            "python",
            str(APP_ROOT.parent / "another-checkout" / "launch.py"),
        ]
        with (
            patch(
                "psutil.net_connections",
                return_value=[
                    SimpleNamespace(pid=987654, status="LISTEN", laddr=SimpleNamespace(port=7860))
                ],
            ),
            patch("psutil.Process", return_value=process),
        ):
            self.assertIsNone(ruinedfooocus_owner(7860))
        process.terminate.assert_not_called()
