"""Use disposable Git fixtures to check source update preservation."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import pygit2

from ruinedfooocus.packaging.source_updates import fast_forward


class SourceUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "--quiet")
        self.git("config", "user.name", "Update Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.hooksPath", str(self.root / "no-hooks"))
        self.git("config", "core.autocrlf", "false")
        (self.root / "launch.py").write_text("original\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("future.txt\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Original fixture")
        self.base = self.git("rev-parse", "HEAD").decode().strip()
        (self.root / "launch.py").write_text("updated\n", encoding="utf-8")
        (self.root / "future.txt").write_text("new source\n", encoding="utf-8")
        self.git("add", "launch.py")
        self.git("add", "-f", "future.txt")
        self.git("commit", "--quiet", "-m", "Update fixture")
        self.target = self.git("rev-parse", "HEAD").decode().strip()
        self.git("checkout", "--quiet", "-b", "current", self.base)
        self.repo = pygit2.Repository(str(self.root))
        self.expected = (self.repo.head.name, self.repo.head.target)

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.root), *arguments])

    def test_clean_fast_forward_updates_files_and_reference(self):
        fast_forward(self.repo, self.repo[self.target], self.expected)
        self.assertEqual(str(self.repo.head.target), self.target)
        self.assertEqual((self.root / "launch.py").read_text(), "updated\n")
        self.assertEqual((self.root / "future.txt").read_text(), "new source\n")
        self.assertFalse(self.repo.status())

    def test_dirty_source_remains_unchanged(self):
        (self.root / "launch.py").write_text("other chat work\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            fast_forward(self.repo, self.repo[self.target], self.expected)
        self.assertEqual(str(self.repo.head.target), self.base)
        self.assertEqual((self.root / "launch.py").read_text(), "other chat work\n")

    def test_unrelated_untracked_work_does_not_disable_safe_update(self):
        note = self.root / "unpublished.py"
        note.write_text("local work\n", encoding="utf-8")
        fast_forward(self.repo, self.repo[self.target], self.expected)
        self.assertEqual(str(self.repo.head.target), self.target)
        self.assertEqual(note.read_text(), "local work\n")

    def test_untracked_update_collision_preserves_file_and_head(self):
        note = self.root / "unpublished.py"
        note.write_text("local work\n", encoding="utf-8")
        self.git("checkout", "--quiet", "--detach", self.target)
        self.git("add", "-f", "unpublished.py")
        note.write_text("remote source\n", encoding="utf-8")
        self.git("add", "-f", "unpublished.py")
        self.git("commit", "--quiet", "-m", "New source collision")
        target = self.git("rev-parse", "HEAD").decode().strip()
        self.git("checkout", "--quiet", "current")
        note.write_text("local work\n", encoding="utf-8")
        with self.assertRaises(pygit2.GitError):
            fast_forward(self.repo, self.repo[target], self.expected)
        self.assertEqual(note.read_text(), "local work\n")
        self.assertEqual(str(self.repo.head.target), self.base)

    def test_ignored_file_collision_keeps_user_file_and_head(self):
        (self.root / "future.txt").write_text("user resource\n", encoding="utf-8")
        self.assertFalse(self.repo.status())
        with self.assertRaises(pygit2.GitError):
            fast_forward(self.repo, self.repo[self.target], self.expected)
        self.assertEqual(str(self.repo.head.target), self.base)
        self.assertEqual((self.root / "launch.py").read_text(), "original\n")
        self.assertEqual((self.root / "future.txt").read_text(), "user resource\n")

    def test_changed_branch_is_not_advanced(self):
        self.git("checkout", "--quiet", "-b", "other")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            fast_forward(self.repo, self.repo[self.target], self.expected)
        self.assertEqual(self.repo.head.shorthand, "other")
        self.assertEqual(str(self.repo.head.target), self.base)


if __name__ == "__main__":
    unittest.main()
