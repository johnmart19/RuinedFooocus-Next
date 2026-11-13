"""Explicit opt-in updates for the official CPU and Vulkan llama.cpp builds."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import time
from urllib.parse import urlsplit
import zipfile

import requests

from ruinedfooocus.backends.llama import llama_installer as installer
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.paths import data_path
from ruinedfooocus.packaging.install_lock import installation_lock
from ruinedfooocus.packaging.startup_options import skip_dependency_install


_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
_DOWNLOAD = "https://github.com/ggml-org/llama.cpp/releases/download"
_ASSET_HOSTS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}
# The reviewed Linux CUDA 12.8 cudart companion is 594 MB.
_MAX_ARCHIVE = 768 * 1024**2
_MAX_EXTRACTED = 2 * 1024**3
_MAX_FILES = 10_000
_CHUNK = 1024**2
_TAG = re.compile(r"b[1-9][0-9]{0,8}\Z")
_VERSION_BUILD = re.compile(
    r"version:\s*(?:b?([0-9]+)(?:\s+\([^)]*\))?"
    r"|[0-9]+\.[0-9]+\.[0-9]+[^\s]*\s+\(build\s+([0-9]+),\s+commit\s+[0-9a-f]+\))",
    re.I,
)


@dataclass(frozen=True)
class PrebuiltUpdate:
    backend: str
    tag: str
    name: str
    url: str
    sha256: str
    size: int
    prerelease: bool

    def archive(self):
        return {"name": self.name, "url": self.url, "sha256": self.sha256}


def _family(backend):
    system, machine = platform.system(), platform.machine().lower()
    if (
        backend not in {"CPU", "Vulkan"}
        or system not in {"Windows", "Linux"}
        or machine not in {"amd64", "x86_64", "arm64", "aarch64"}
    ):
        raise RuntimeError(
            "Select an explicit CPU or Vulkan backend on supported Windows/Linux hardware."
        )
    if os.environ.get("LLAMA_SERVER"):
        raise RuntimeError(
            "LLAMA_SERVER selects a custom binary. Managed updates apply to the packaged runtime."
        )
    return system, machine


def _pointer(backend):
    system, machine = _family(backend)
    return data_path("cache", "llama.cpp", "selected", f"{system}-{machine}-{backend}.json")


def _package(backend, tag):
    system, machine = _family(backend)
    return installer.package_name(
        system, machine, backend == "Vulkan", backend=backend, version=tag
    )


def _read_selected(backend):
    try:
        value = json.loads(_pointer(backend).read_text(encoding="utf-8"))
        tag = value.get("tag") if isinstance(value, dict) else None
        digest = value.get("sha256") if isinstance(value, dict) else None
        if (
            isinstance(tag, str)
            and _TAG.fullmatch(tag)
            # Preserve an explicit verified selection across an app pin update.
            and int(tag[1:]) >= 12110
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            return tag, digest
    except (OSError, ValueError):
        pass
    return installer.VERSION, None


def selected_build(backend, *, progress_callback=None, check_cancelled=None):
    """Return the verified, explicitly selected build for a new Chat process."""
    if backend not in {"CPU", "Vulkan"} or os.environ.get("LLAMA_SERVER"):
        return installer.VERSION, None
    installer._check(check_cancelled)
    try:
        tag, digest = _read_selected(backend)
        if digest:
            package = _package(backend, tag)
            directory = installer.package_directory(package, version=tag)
            executable = installer._install_server(
                directory,
                package,
                acquire=False,
                version=tag,
                backend=backend,
                **installer._callbacks(progress_callback, check_cancelled),
            )
            archive = {"name": package, "url": f"{_DOWNLOAD}/{tag}/{package}", "sha256": digest}
            manifest = json.loads((directory / "rf-runtime.json").read_text(encoding="utf-8"))
            if executable.is_relative_to(directory.resolve()) and manifest.get("archives") == [
                archive
            ]:
                return tag, executable
    except (OSError, RuntimeError, ValueError):
        pass
    return installer.VERSION, None


def configured_build(backend):
    """Read the local selection without a release check or binary probe."""
    try:
        return _read_selected(backend)[0]
    except RuntimeError:
        return None


def _candidate(release, backend):
    if not isinstance(release, dict) or release.get("draft") is not False:
        return None
    tag, assets = release.get("tag_name"), release.get("assets")
    if not isinstance(tag, str) or not _TAG.fullmatch(tag) or not isinstance(assets, list):
        return None
    name = _package(backend, tag)
    url = f"{_DOWNLOAD}/{tag}/{name}"
    for asset in assets:
        if not isinstance(asset, dict) or asset.get("name") != name:
            continue
        digest, size = asset.get("digest"), asset.get("size")
        if (
            asset.get("state") == "uploaded"
            and asset.get("browser_download_url") == url
            and isinstance(digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            and type(size) is int
            and 0 < size <= _MAX_ARCHIVE
        ):
            return PrebuiltUpdate(
                backend, tag, name, url, digest[7:], size, release.get("prerelease") is True
            )
    return None


def check_update(backend, *, progress_callback=None, check_cancelled=None):
    """Discover the highest numbered compatible release in GitHub's recent releases."""
    _family(backend)
    installer._check(check_cancelled)
    if skip_dependency_install():
        raise RuntimeError("Update checks are disabled by offline or skip-install mode.")
    selected, _ = selected_build(
        backend, **installer._callbacks(progress_callback, check_cancelled)
    )
    installer._emit(
        progress_callback,
        "resolving_runtime",
        backend=backend,
        message="Checking official llama.cpp releases",
    )
    installer._check(check_cancelled)
    try:
        with requests.get(
            _RELEASES, params={"per_page": 20}, timeout=(5, 10), stream=True, allow_redirects=False
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"GitHub release check returned HTTP {response.status_code}.")
            payload = bytearray()
            for chunk in response.iter_content(64 * 1024):
                installer._check(check_cancelled)
                payload.extend(chunk)
                if len(payload) > 8 * 1024**2:
                    raise RuntimeError("GitHub release metadata exceeded the size limit.")
    except requests.RequestException as error:
        raise RuntimeError("Could not check the official llama.cpp releases.") from error
    releases = json.loads(payload)
    if not isinstance(releases, list):
        raise RuntimeError("GitHub did not return a release list.")
    candidate = max(
        (
            candidate
            for release in releases
            if (candidate := _candidate(release, backend))
            and int(candidate.tag[1:]) > int(selected[1:])
        ),
        key=lambda candidate: int(candidate.tag[1:]),
        default=None,
    )
    return selected, candidate


def _write_archive(
    response, candidate, archive, progress, *, progress_callback=None, check_cancelled=None
):
    if response.status_code != 200:
        raise RuntimeError(f"GitHub asset download returned HTTP {response.status_code}.")
    received = 0
    digest = hashlib.sha256()
    last_update = 0.0
    started = time.monotonic()
    installer._emit(
        progress_callback,
        "downloading_runtime",
        backend=candidate.backend,
        package=candidate.name,
        version=candidate.tag,
        received=0,
        total=candidate.size,
        message="Downloading llama.cpp",
    )
    with archive.open("wb") as output:
        for chunk in response.iter_content(_CHUNK):
            installer._check(check_cancelled)
            if not chunk:
                continue
            received += len(chunk)
            if received > candidate.size:
                raise RuntimeError("The llama.cpp archive exceeds its published size.")
            digest.update(chunk)
            output.write(chunk)
            now = time.monotonic()
            if now - last_update >= 0.25:
                progress(0.9 * received / candidate.size, "Downloading llama.cpp")
                installer._emit(
                    progress_callback,
                    "downloading_runtime",
                    backend=candidate.backend,
                    package=candidate.name,
                    version=candidate.tag,
                    received=received,
                    total=candidate.size,
                    rate=received / max(now - started, 0.001),
                    message="Downloading llama.cpp",
                )
                last_update = now
    installer._check(check_cancelled)
    if received != candidate.size or digest.hexdigest() != candidate.sha256:
        raise RuntimeError(
            "The llama.cpp archive does not match GitHub's published size and SHA256."
        )
    installer._emit(
        progress_callback,
        "verifying_runtime",
        backend=candidate.backend,
        package=candidate.name,
        version=candidate.tag,
        received=received,
        total=candidate.size,
        message="Verifying llama.cpp",
    )
    installer._check(check_cancelled)


def _download(candidate, archive, progress, *, progress_callback=None, check_cancelled=None):
    callbacks = installer._callbacks(progress_callback, check_cancelled)
    installer._check(check_cancelled)
    options = {"stream": True, "timeout": (10, 30), "allow_redirects": False}
    try:
        with requests.get(candidate.url, **options) as response:
            if not response.is_redirect:
                return _write_archive(response, candidate, archive, progress, **callbacks)
            target = response.headers.get("Location", "")
            parsed = urlsplit(target)
            if (
                parsed.scheme != "https"
                or parsed.netloc not in _ASSET_HOSTS
                or parsed.username
                or parsed.password
            ):
                raise RuntimeError("GitHub asset download returned an unexpected redirect.")
        installer._check(check_cancelled)
        with requests.get(target, **options) as response:
            _write_archive(response, candidate, archive, progress, **callbacks)
    except requests.RequestException as error:
        raise RuntimeError("Could not download the official llama.cpp archive.") from error


def _extract(archive, directory, *, check_cancelled=None):
    installer._check(check_cancelled)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            files = bundle.infolist()
            if len(files) > _MAX_FILES or sum(item.file_size for item in files) > _MAX_EXTRACTED:
                raise RuntimeError("The llama.cpp archive exceeds the extraction limits.")
            for item in files:
                installer._check(check_cancelled)
                path = PurePosixPath(item.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or ":" in item.filename
                    or "\\" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)
                ):
                    raise RuntimeError("The llama.cpp ZIP contains an unsafe path.")
            for item in files:
                installer._check(check_cancelled)
                bundle.extract(item, directory)
    else:
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError("This Python build does not support safe tar extraction.")
        with tarfile.open(archive) as bundle:
            files = []
            size = 0
            for item in bundle:
                installer._check(check_cancelled)
                files.append(item)
                size += item.size
                if len(files) > _MAX_FILES or size > _MAX_EXTRACTED:
                    raise RuntimeError("The llama.cpp archive exceeds the extraction limits.")
                if not (item.isfile() or item.isdir() or item.issym() or item.islnk()):
                    raise RuntimeError("The llama.cpp TAR contains a special file.")
            for item in files:
                installer._check(check_cancelled)
                bundle.extract(item, directory, filter="data")
    installer._check(check_cancelled)


def _verify_executable(executable, candidate, *, progress_callback=None, check_cancelled=None):
    from ruinedfooocus.backends.llama.runtime_probe import probe_runtime

    gpu = candidate.backend == "Vulkan"
    environment = installer.runtime_environment(gpu, backend=candidate.backend, acquire=False)
    installer._emit(
        progress_callback,
        "probing_device",
        backend=candidate.backend,
        package=candidate.name,
        version=candidate.tag,
        message="Verifying staged llama.cpp",
    )
    installer._check(check_cancelled)
    try:
        result = installer._run_cancellable(
            [str(executable), "--version"],
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check_cancelled=check_cancelled,
        )
        build_numbers = {
            match.group(1) or match.group(2)
            for line in (result.stdout + "\n" + result.stderr).splitlines()
            if (match := _VERSION_BUILD.fullmatch(line.strip()))
        }
        if result.returncode or build_numbers != {candidate.tag.removeprefix("b")}:
            raise RuntimeError("The staged llama.cpp executable did not report the selected build.")
        callbacks = installer._callbacks(progress_callback, check_cancelled)
        if progress_callback is not None:
            callbacks["progress_callback"] = lambda event: progress_callback(
                event
                | {
                    "runtime": {
                        "tag": candidate.tag,
                        "backend": candidate.backend,
                        "package": candidate.name,
                    }
                }
            )
        if gpu and not probe_runtime(executable, environment, "Vulkan", **callbacks):
            raise RuntimeError("The staged llama.cpp build did not expose a usable Vulkan device.")
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "The staged llama.cpp executable could not be verified on this host."
        ) from error


def install_update(backend, candidate, progress, *, progress_callback=None, check_cancelled=None):
    """Stage and validate the selected candidate before atomically enabling it."""
    pointer = _pointer(backend)
    installer._check(check_cancelled)
    callbacks = installer._callbacks(progress_callback, check_cancelled)
    if skip_dependency_install():
        raise RuntimeError("Updates are disabled by offline or skip-install mode.")
    if (
        not isinstance(candidate, PrebuiltUpdate)
        or candidate.backend != backend
        or not _TAG.fullmatch(candidate.tag)
        or candidate.name != _package(backend, candidate.tag)
        or candidate.url != f"{_DOWNLOAD}/{candidate.tag}/{candidate.name}"
        or not re.fullmatch(r"[0-9a-f]{64}", candidate.sha256)
        or type(candidate.size) is not int
        or not 0 < candidate.size <= _MAX_ARCHIVE
    ):
        raise RuntimeError("Check for a compatible update before installing it.")
    directory = installer.package_directory(candidate.name, version=candidate.tag)
    with installation_lock(pointer, timeout=0):
        current_tag, _ = _read_selected(backend)
        if int(candidate.tag[1:]) <= int(current_tag[1:]):
            raise RuntimeError("The selected update is no longer newer than the configured build.")
        progress(0, "Downloading llama.cpp")
        if directory.exists():
            executable = installer._install_server(
                directory,
                candidate.name,
                acquire=False,
                version=candidate.tag,
                backend=backend,
                **callbacks,
            )
            saved = json.loads((directory / "rf-runtime.json").read_text(encoding="utf-8"))
            if saved.get("archives") != [candidate.archive()]:
                raise RuntimeError("The cached build differs from the selected GitHub asset.")
            _verify_executable(executable, candidate, **callbacks)
        else:
            directory.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=directory.parent) as temporary:
                archive = Path(temporary) / candidate.name
                extracted = Path(temporary) / "runtime"
                _download(candidate, archive, progress, **callbacks)
                progress(0.92, "Verifying llama.cpp")
                try:
                    _extract(archive, extracted, check_cancelled=check_cancelled)
                except (zipfile.BadZipFile, tarfile.TarError) as error:
                    raise RuntimeError(
                        "The verified llama.cpp archive could not be extracted."
                    ) from error
                executable_name = (
                    "llama-server.exe" if platform.system() == "Windows" else "llama-server"
                )
                executables = list(extracted.rglob(executable_name))
                if (
                    len(executables) != 1
                    or not executables[0].is_file()
                    or executables[0].is_symlink()
                ):
                    raise RuntimeError("The staged archive has no unique llama-server executable.")
                _verify_executable(executables[0], candidate, **callbacks)
                manifest = installer._runtime_manifest(
                    extracted,
                    candidate.name,
                    version=candidate.tag,
                    check_cancelled=check_cancelled,
                )
                if (extracted / "rf-runtime.json").exists() or (
                    extracted / "rf-runtime.json"
                ).is_symlink():
                    raise RuntimeError("The llama.cpp archive contains reserved runtime metadata.")
                save_json(
                    extracted / "rf-runtime.json", manifest | {"archives": [candidate.archive()]}
                )
                installer._check(check_cancelled)
                extracted.rename(directory)
        installer._emit(
            progress_callback,
            "staging_runtime",
            backend=backend,
            package=candidate.name,
            version=candidate.tag,
            message="Selecting verified llama.cpp for the next load",
        )
        installer._check(check_cancelled)
        save_json(pointer, {"tag": candidate.tag, "sha256": candidate.sha256})
    progress(1, "Ready for the next Chat load")
    return candidate.tag


def use_pinned_build(backend):
    pointer = _pointer(backend)
    with installation_lock(pointer, timeout=0):
        pointer.unlink(missing_ok=True)
    return installer.VERSION
