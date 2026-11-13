"""Pinned backend updates preserve local work, including checkout-time conflicts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pygit2

from ruinedfooocus.packaging.launch_util import git_clone


class BackendUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.origin = pygit2.init_repository(self.root / "origin", initial_head="main")
        self.addCleanup(self.origin.free)
        self.first = self.commit("first")
        self.clone = pygit2.clone_repository(str(self.root / "origin"), str(self.root / "clone"))
        self.addCleanup(self.clone.free)
        self.second = self.commit("second")

    def commit(self, text):
        path = Path(self.origin.workdir) / "backend.py"
        path.write_text(text, encoding="utf-8")
        self.origin.index.add("backend.py")
        self.origin.index.write()
        signature = pygit2.Signature("Backend Fixture", "fixture@example.invalid")
        parents = [] if self.origin.head_is_unborn else [self.origin.head.target]
        return self.origin.create_commit(
            "HEAD", signature, signature, "Backend fixture", self.origin.index.write_tree(), parents
        )

    def test_clean_backend_updates_to_exact_pin(self):
        self.assertTrue(
            git_clone(
                str(self.root / "origin"), self.root / "clone", "Fixture", hash=str(self.second)
            )
        )
        self.assertEqual(self.clone.head.target, self.second)
        self.assertEqual((self.root / "clone/backend.py").read_text(), "second")

    def test_current_pin_does_not_fetch_or_checkout(self):
        repo = Mock(wraps=self.clone)
        repo.head = self.clone.head
        with patch.object(pygit2, "Repository", return_value=repo):
            self.assertTrue(
                git_clone("unused", self.root / "clone", "Fixture", hash=str(self.first))
            )
        self.assertEqual(repo.remotes.mock_calls, [])
        repo.checkout_tree.assert_not_called()

    def test_locally_present_target_does_not_need_network(self):
        self.clone.remotes["origin"].fetch()
        with patch("ruinedfooocus.packaging.git_transport.bounded_git_operation") as network:
            self.assertTrue(
                git_clone("unused", self.root / "clone", "Fixture", hash=str(self.second))
            )
        network.assert_not_called()
        self.assertEqual(self.clone.head.target, self.second)

    def test_fetch_timeout_keeps_the_existing_backend(self):
        with patch(
            "ruinedfooocus.packaging.git_transport.bounded_git_operation",
            side_effect=TimeoutError("fake slow Git deadline"),
        ):
            self.assertFalse(
                git_clone("unused", self.root / "clone", "Fixture", hash=str(self.second))
            )
        self.assertEqual(self.clone.head.target, self.first)
        self.assertEqual((self.root / "clone/backend.py").read_text(), "first")

    def test_interrupted_clone_does_not_publish_a_partial_checkout(self):
        destination = self.root / "new checkout"

        def interrupted(operation, staged, source):
            Path(staged).mkdir()
            (Path(staged) / "partial").write_text("incomplete", encoding="utf-8")
            raise TimeoutError("fake slow Git deadline")

        with patch(
            "ruinedfooocus.packaging.git_transport.bounded_git_operation", side_effect=interrupted
        ):
            self.assertFalse(git_clone("unused", destination, "Fixture", hash=str(self.second)))
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob("rf-git-clone-*")))

    def test_dirty_backend_and_untracked_inputs_are_left_unchanged(self):
        path = self.root / "clone/backend.py"
        path.write_text("other chat's edits", encoding="utf-8")
        note = self.root / "clone/local-note.txt"
        note.write_text("user input", encoding="utf-8")
        self.assertFalse(
            git_clone(
                str(self.root / "origin"), self.root / "clone", "Fixture", hash=str(self.second)
            )
        )
        self.assertEqual(self.clone.head.target, self.first)
        self.assertEqual(path.read_text(), "other chat's edits")
        self.assertEqual(note.read_text(), "user input")

    def test_safe_checkout_rejects_a_conflict_even_if_precheck_misses_it(self):
        path = self.root / "clone/backend.py"
        path.write_text("concurrent edits", encoding="utf-8")
        repo = Mock(wraps=self.clone)
        repo.remotes = self.clone.remotes
        repo.status.return_value = {}
        with patch.object(pygit2, "Repository", return_value=repo):
            self.assertFalse(
                git_clone(
                    str(self.root / "origin"), self.root / "clone", "Fixture", hash=str(self.second)
                )
            )
        repo.checkout_tree.assert_called_once()
        repo.set_head.assert_not_called()
        self.assertEqual(path.read_text(), "concurrent edits")
        self.assertEqual(self.clone.head.target, self.first)


if __name__ == "__main__":
    unittest.main()
