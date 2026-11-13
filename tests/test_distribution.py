"""Source archives exclude user data and preserve an existing artifact."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from ruinedfooocus.packaging.distribution import build_source_archive, source_checkout


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "--quiet")
        files = {
            "launch.py": 'print("source")\n',
            "ruinedfooocus/app/bootstrap.py": 'git_repos = [{"name":"ComfyUI","hash":"pinned"}]\n',
            "ruinedfooocus/resources/prompt_presets/default.txt": "Default\nKeep the scene.",
            "settings/settings.json": '{"private":true}',
            "outputs/private.png": "private",
            "models/README.md": "# Model folders\n",
            "models/background_removal/put_birefnet_models_here": "Place BiRefNet here.\n",
            "models/diffusers/put_local_diffusers_bundles_here": "Place local bundles here.\n",
            "models/inbox/Place_models_to_be_sorted_here": "",
            "models/clip/put_text_encoder_models_here": "legacy alias",
            "models/diffusers/put_private_data_here": "unlisted marker-shaped file",
            "models/diffusers/example/model_index.json": "private bundle manifest",
            "models/diffusers/example/transformer/model.safetensors": "private bundle weights",
            "models/diffusers/example/put_local_diffusers_bundles_here": "private nested file",
            "models/private.safetensors": "original",
            "repositories/ComfyUI/runtime.py": "local runtime",
            "venv/pyvenv.cfg": "local interpreter",
            "nested/PRIVATE.GGUF": "original",
            "AGENTS.md": "local instructions",
        }
        for name, text in files.items():
            file = self.repo / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
        self.git("add", ".")
        self.commit()

    def commit(self):
        self.git(
            "-c",
            "user.name=Archive Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=" + str(self.root / "no-hooks"),
            "commit",
            "--quiet",
            "-m",
            "Source fixture",
        )

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.repo), *arguments])

    def test_archive_has_exact_source_and_resources_without_user_data_or_weights(self):
        output = self.root / "source.zip"
        revision = self.git("rev-parse", "HEAD").decode().strip()
        (self.repo / "launch.py").write_text("uncommitted", encoding="utf-8")
        marker = self.repo / "models/diffusers/put_local_diffusers_bundles_here"
        marker.write_text("uncommitted placement text", encoding="utf-8")
        self.git("add", str(marker))
        result = build_source_archive(self.repo, output, revision)
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertEqual(archive.read("launch.py"), b'print("source")\n')
            self.assertIn("ruinedfooocus/resources/prompt_presets/default.txt", names)
            self.assertFalse(
                any(
                    name.startswith(("settings/", "outputs/", "repositories/", "venv/"))
                    for name in names
                )
            )
            self.assertEqual(
                {name for name in names if name.startswith("models/") and not name.endswith("/")},
                {
                    "models/README.md",
                    "models/background_removal/put_birefnet_models_here",
                    "models/diffusers/put_local_diffusers_bundles_here",
                    "models/inbox/Place_models_to_be_sorted_here",
                },
            )
            self.assertNotIn("models/clip/", names)
            self.assertNotIn("models/diffusers/example/", names)
            self.assertEqual(
                archive.read("models/diffusers/put_local_diffusers_bundles_here"),
                b"Place local bundles here.\n",
            )
            extracted = self.root / "extracted"
            archive.extractall(extracted)
            for directory in ("background_removal", "diffusers", "inbox"):
                self.assertTrue((extracted / "models" / directory).is_dir())
            self.assertNotIn("nested/PRIVATE.GGUF", names)
            self.assertNotIn("AGENTS.md", names)
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(json.loads(archive.read("distribution.json")), result)
        self.assertEqual(result["backend_pins"]["ComfyUI"], "pinned")
        self.assertEqual(result["source_commit"], revision)
        self.assertFalse(result["runtime_bundled"])
        self.assertFalse(result["models_bundled"])
        self.assertEqual((self.repo / "models/private.safetensors").read_text(), "original")

    def test_allowlisted_names_only_admit_small_regular_files(self):
        target = self.repo / "link-target"
        target.write_text("../private.safetensors", encoding="utf-8")
        blob = self.git("hash-object", "-w", str(target)).decode().strip()
        symlink = "models/checkpoints/put_checkpoints_here"
        self.git("update-index", "--add", "--cacheinfo", f"120000,{blob},{symlink}")
        directory = self.repo / "models/loras/put_loras_here"
        directory.mkdir(parents=True)
        (directory / "private.txt").write_text("arbitrary nested data", encoding="utf-8")
        oversized = self.repo / "models/vae/put_vae_here"
        oversized.parent.mkdir()
        oversized.write_bytes(b"x" * (16 * 1024 + 1))
        maximum = self.repo / "models/vae_approx/put_approximate_vae_decoders_here"
        maximum.parent.mkdir()
        maximum.write_bytes(b"x" * (16 * 1024))
        self.git("add", str(directory), str(oversized), str(maximum))
        self.commit()

        output = self.root / "source.zip"
        build_source_archive(self.repo, output)
        with zipfile.ZipFile(output) as archive:
            self.assertNotIn(symlink, archive.namelist())
            self.assertFalse(any(name.startswith("models/loras/") for name in archive.namelist()))
            self.assertNotIn("models/vae/put_vae_here", archive.namelist())
            self.assertEqual(
                archive.getinfo("models/vae_approx/put_approximate_vae_decoders_here").file_size,
                16 * 1024,
            )

    def test_existing_artifact_is_never_overwritten_or_deleted(self):
        output = self.root / "source.zip"
        output.write_bytes(b"keep this artifact")
        with self.assertRaises(FileExistsError):
            build_source_archive(self.repo, output)
        self.assertEqual(output.read_bytes(), b"keep this artifact")

    def test_same_revision_produces_repeatable_archive_bytes(self):
        first, second = self.root / "first.zip", self.root / "second.zip"
        build_source_archive(self.repo, first)
        build_source_archive(self.repo, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_source_detection_accepts_worktrees_and_extracted_archive_has_no_git(self):
        self.assertTrue(source_checkout(self.repo))
        folder = self.root / "worktree"
        folder.mkdir()
        (folder / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        self.assertTrue(source_checkout(folder))
        (folder / ".git").unlink()
        self.assertFalse(source_checkout(folder))


if __name__ == "__main__":
    unittest.main()
