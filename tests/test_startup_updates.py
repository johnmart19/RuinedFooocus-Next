"""Updater failures continue startup while SAFE publication and argv survive."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pygit2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "entry_with_update.py").read_text(encoding="utf-8")


class StartupUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.origin = pygit2.init_repository(self.root / "origin", initial_head="main")
        self.addCleanup(self.origin.free)
        self.first = self.commit("first")
        self.path = self.root / "checkout"
        self.repo = pygit2.clone_repository(str(self.root / "origin"), str(self.path))
        self.addCleanup(self.repo.free)

    def commit(self, contents):
        (Path(self.origin.workdir) / "source.txt").write_text(contents, encoding="utf-8")
        self.origin.index.add("source.txt")
        self.origin.index.write()
        signature = pygit2.Signature("Startup Fixture", "fixture@example.invalid")
        return self.origin.create_commit(
            "HEAD",
            signature,
            signature,
            "Fixture",
            self.origin.index.write_tree(),
            [] if self.origin.head_is_unborn else [self.origin.head.target],
        )

    def execute(self, *arguments, failure=None):
        bootstrap = types.ModuleType("ruinedfooocus.app.bootstrap")
        bootstrap.__all__ = []
        output = io.StringIO()
        argv = [str(self.path / "entry_with_update.py"), "--skip-dependency-install", *arguments]
        from ruinedfooocus.packaging.git_transport import bounded_git_operation

        with (
            redirect_stdout(output),
            patch.object(sys, "argv", argv),
            patch.object(sys, "path", list(sys.path)),
            patch.dict(os.environ, RF_OFFLINE="0"),
            patch("os.chdir"),
            patch("os.execv") as restart,
            patch.dict(sys.modules, {"ruinedfooocus.app.bootstrap": bootstrap}),
            patch(
                "ruinedfooocus.packaging.git_transport.bounded_git_operation",
                side_effect=failure or bounded_git_operation,
            ) as network,
        ):
            exec(compile(SOURCE, "entry_with_update.py", "exec"), {"__file__": argv[0]})
        self.assertIn("Starting RuinedFooocus Python", output.getvalue().splitlines()[0])
        self.assertIn("Preparing application runtime", output.getvalue())
        return output.getvalue(), network, restart, argv

    def test_latest_source_continues_without_restarting(self):
        output, network, restart, _ = self.execute()
        self.assertIn("You have the latest version", output)
        network.assert_called_once_with("fetch", self.path.resolve(), "origin")
        restart.assert_not_called()

    def test_timeout_is_visible_and_does_not_apply_a_cached_remote_revision(self):
        target = self.commit("remote update")
        self.repo.remotes["origin"].fetch()
        output, _, restart, _ = self.execute(failure=TimeoutError("fake slow Git deadline"))
        self.assertIn("fake slow Git deadline", output)
        self.assertIn("Continuing with installed source", output)
        self.assertEqual(self.repo.head.target, self.first)
        self.assertNotEqual(self.repo.head.target, target)
        restart.assert_not_called()

    def test_offline_and_abbreviated_offline_never_fetch(self):
        for option in ("--offline", "--of"):
            with self.subTest(option=option):
                output, network, restart, _ = self.execute(option)
                self.assertIn("Offline mode. No update", output)
                network.assert_not_called()
                restart.assert_not_called()

    def test_dirty_and_detached_source_never_fetch(self):
        path = self.path / "source.txt"
        path.write_text("user modification", encoding="utf-8")
        output, network, _, _ = self.execute()
        self.assertIn("Tracked source changes found", output)
        network.assert_not_called()
        self.assertEqual(path.read_text(), "user modification")
        path.write_text("first", encoding="utf-8")
        self.repo.set_head(self.first)
        output, network, _, _ = self.execute()
        self.assertIn("Detached source revision", output)
        network.assert_not_called()

    def test_fast_forward_restarts_with_the_exact_interpreter_and_arguments(self):
        target = self.commit("remote update")
        output, _, restart, argv = self.execute("--settings", "profile with spaces", "--cpu")
        self.assertEqual(self.repo.head.target, target)
        self.assertEqual((self.path / "source.txt").read_text(), "remote update")
        self.assertIn("Update succeeded", output)
        restart.assert_called_once_with(sys.executable, [sys.executable, *argv])


if __name__ == "__main__":
    unittest.main()
