"""Frozen diffusion artifacts: CPU-only acquisition and publication fixtures."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from ruinedfooocus.backends.sdcpp import artifacts, contracts
from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.packaging import install_lock as publication


class SdcppArtifactTests(unittest.TestCase):
    def test_pinned_primary_artifacts_have_size_hash_and_platform_limits(self):
        self.assertEqual(len(artifacts.ARTIFACTS), 4)
        for item in artifacts.ARTIFACTS:
            request = item.preparation_request(Path(tempfile.gettempdir()) / item.filename)
            self.assertEqual(request.sha256, item.sha256)
            self.assertEqual(request.expected_size, item.size)
            self.assertIn(contracts.RELEASE, request.url)
            if item.system == "Linux":
                self.assertIn("Ubuntu 26.04", item.platform_contract)
                self.assertIn("WSL separate", item.platform_contract)
        self.assertIsNone(artifacts.artifact_for("Darwin", "x86_64", "vulkan"))
        with self.assertRaises(ValueError):
            artifacts.artifact_for("Windows", "arm64", "vulkan")

    def fixture_bundle(self, members):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, data in members.items():
                bundle.writestr(name, data)
        content = archive.getvalue()
        item = artifacts.Artifact(
            "Windows",
            "cpu",
            "fixture.zip",
            len(content),
            hashlib.sha256(content).hexdigest(),
            "fixture",
        )
        return item, content

    def test_preparation_publishes_verified_complete_payload_and_reuses_cache(self):
        item, content = self.fixture_bundle(
            {"sd-cli.exe": b"inert", "stable-diffusion.cpp.txt": b"MIT", "ggml.txt": b"MIT"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []

            def transfer(request, staging, progress, cancelled):
                calls.append(request)
                path = staging / "fixture.part"
                path.write_bytes(content)
                return path

            service = PreparationService(transport=transfer, staging_root=root / "stage")
            original_rename = publication.os.rename
            publications = []

            def rename(source, destination):
                if Path(source).parent.name.startswith("rf-sdcpp-"):
                    publications.append((source, destination))
                    if len(publications) < 3:
                        error = PermissionError("fixture SDCPP publication denied")
                        error.winerror = 5 if len(publications) == 1 else 32
                        raise error
                return original_rename(source, destination)

            try:
                with (
                    patch.object(artifacts, "ARTIFACTS", (item,)),
                    patch.object(publication, "_WINDOWS", True),
                    patch.object(publication.time, "sleep"),
                    patch.object(publication.os, "rename", side_effect=rename),
                ):
                    candidate = artifacts.prepare_runtime(
                        item,
                        root / "runtime",
                        archive=root / "archive.zip",
                        check_cancelled=lambda: None,
                        service=service,
                    )
                    again = artifacts.prepare_runtime(
                        item,
                        root / "runtime",
                        archive=root / "archive.zip",
                        check_cancelled=lambda: None,
                        service=service,
                    )
                    self.assertEqual(candidate, again)
                    self.assertFalse(candidate.device_verified)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(len(publications), 3)
                    (root / "runtime/sd-cli.exe").write_bytes(b"tampered")
                    with self.assertRaisesRegex(RuntimeError, "integrity"):
                        artifacts.open_runtime(root / "runtime", item)
                    self.assertEqual((root / "runtime/sd-cli.exe").read_bytes(), b"tampered")
            finally:
                service.close()

    def test_unsafe_archive_or_missing_notices_never_publishes_partial_runtime(self):
        for members in (
            {"../escape": b"bad"},
            {"C:stream": b"bad"},
            {"nested/file": b"bad"},
            {"NUL.txt": b"bad"},
            {"alias.dll.": b"bad"},
            {"sd-cli.exe": b"inert"},
        ):
            item, content = self.fixture_bundle(members)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "archive.zip"
                archive.write_bytes(content)
                service = PreparationService(staging_root=root / "stage")
                try:
                    with patch.object(artifacts, "ARTIFACTS", (item,)):
                        with self.assertRaises(RuntimeError):
                            artifacts.prepare_runtime(
                                item,
                                root / "runtime",
                                archive=archive,
                                check_cancelled=lambda: None,
                                service=service,
                            )
                    self.assertFalse((root / "runtime").exists())
                    self.assertFalse(list(root.glob("rf-sdcpp-*")))
                finally:
                    service.close()
