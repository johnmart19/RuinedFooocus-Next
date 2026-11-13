"""Pinned userspace bridge for the native ROCm 10.0 child on WSL.

Extract only the verified library, without installing a system package or running
its maintainer scripts. Image Torch's newer SDK is independent.
"""

import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import time

import requests

from ruinedfooocus.core.paths import data_path
from ruinedfooocus.packaging.install_lock import installation_lock
from ruinedfooocus.packaging.startup_options import skip_dependency_install
from ruinedfooocus.backends.llama import llama_installer as installer

VERSION = "1.2.2"
URL = f"https://github.com/ROCm/librocdxg/releases/download/v{VERSION}/rocdxg-roct_{VERSION}_amd64.deb"
ARCHIVE_SHA256 = "28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541"
LIBRARY_SHA256 = "cf5a7459c31c1e12ae20d3f0755377d4d7872c7466fd6ea2b8050ba3f97a4523"
SYSTEM_LIBRARY = Path("/opt/rocm/lib/librocdxg.so")


def library_directory(*, acquire=False, progress_callback=None, check_cancelled=None):
    installer._check(check_cancelled)
    directory = data_path("cache", "llama.cpp", f"rocdxg-{VERSION}")
    library = directory / "librocdxg.so"
    with installation_lock(directory, timeout=0):
        for candidate in (library, SYSTEM_LIBRARY):
            if (
                candidate.is_file()
                and installer._file_sha256(candidate, check_cancelled=check_cancelled)
                == LIBRARY_SHA256
            ):
                return candidate.parent
        if not acquire or skip_dependency_install():
            raise RuntimeError(
                "Native ROCm WSL bridge is missing or unverified; load online once to acquire ROCDXG"
            )
        if not shutil.which("dpkg-deb"):
            raise RuntimeError("Acquiring the native ROCm WSL bridge requires dpkg-deb")
        directory.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory.parent) as temporary:
            archive = Path(temporary) / "bridge.deb"
            digest = hashlib.sha256()
            received = 0
            started = time.monotonic()
            installer._emit(
                progress_callback,
                "downloading_runtime",
                backend="ROCm",
                package=f"rocdxg-{VERSION}",
                received=0,
                message="Downloading native ROCm WSL bridge",
            )
            print(f"Downloading native ROCm WSL userspace bridge {VERSION}")
            with requests.get(URL, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with archive.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        installer._check(check_cancelled)
                        received += len(chunk)
                        if received > 8 * 1024 * 1024:
                            raise RuntimeError("ROCDXG archive exceeds its download limit")
                        digest.update(chunk)
                        output.write(chunk)
                        installer._emit(
                            progress_callback,
                            "downloading_runtime",
                            backend="ROCm",
                            package=f"rocdxg-{VERSION}",
                            received=received,
                            rate=received / max(time.monotonic() - started, 0.001),
                            message="Downloading native ROCm WSL bridge",
                        )
            installer._check(check_cancelled)
            if digest.hexdigest() != ARCHIVE_SHA256:
                raise RuntimeError("ROCDXG download checksum mismatch")
            installer._emit(
                progress_callback,
                "verifying_runtime",
                backend="ROCm",
                package=f"rocdxg-{VERSION}",
                message="Verifying native ROCm WSL bridge",
            )
            runner = (
                subprocess.run
                if check_cancelled is None
                else (
                    lambda command, **options: installer._run_cancellable(
                        command, check_cancelled=check_cancelled, **options
                    )
                )
            )
            result = runner(
                ["dpkg-deb", "--fsys-tarfile", str(archive)],
                capture_output=True,
                check=True,
                timeout=30,
            )
            with tarfile.open(fileobj=io.BytesIO(result.stdout)) as bundle:
                member = next(
                    (
                        item
                        for item in bundle
                        if item.isfile()
                        and item.name.removeprefix("./") == f"opt/rocm/lib/librocdxg.so.{VERSION}"
                    ),
                    None,
                )
                if member is None or member.size > 4 * 1024 * 1024:
                    raise RuntimeError("ROCDXG package has no supported userspace library")
                with bundle.extractfile(member) as stream:
                    content = stream.read()
            installer._check(check_cancelled)
            if hashlib.sha256(content).hexdigest() != LIBRARY_SHA256:
                raise RuntimeError("ROCDXG library checksum mismatch")
            staged = Path(temporary) / "librocdxg.so"
            staged.write_bytes(content)
            installer._emit(
                progress_callback,
                "staging_runtime",
                backend="ROCm",
                package=f"rocdxg-{VERSION}",
                message="Publishing verified native ROCm WSL bridge",
            )
            installer._check(check_cancelled)
            directory.mkdir(parents=True, exist_ok=True)
            staged.replace(library)
        return directory
