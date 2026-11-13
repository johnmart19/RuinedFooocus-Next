"""Frozen optional artifacts, explicit preparation and immutable runtime publication."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from .contracts import RELEASE, SOURCE_REVISION, RuntimeCandidate, _Cancellation


@dataclass(frozen=True)
class Artifact:
    system: str
    backend: str
    filename: str
    size: int
    sha256: str
    platform_contract: str

    @property
    def url(self):
        return f"https://github.com/leejet/stable-diffusion.cpp/releases/download/{RELEASE}/{self.filename}"

    def preparation_request(self, destination):
        from ruinedfooocus.core.preparation import PreparationRequest

        return PreparationRequest(self.url, destination, self.size, self.sha256)


ARTIFACTS = (
    Artifact(
        "Windows",
        "cpu",
        "sd-master-fc350be-bin-win-cpu-x64.zip",
        18225866,
        "b3da78a4e222caf8b447144b1ec4f51102b5da902a45ab2c288968729962b1b9",
        "Windows x86-64; CPU instruction/runtime loader acceptance required",
    ),
    Artifact(
        "Windows",
        "vulkan",
        "sd-master-fc350be-bin-win-vulkan-x64.zip",
        29995988,
        "b229d9a21858ba91975d061ccfe37fa690254a423bcf77b39e72a1d8435a8b4a",
        "Windows x86-64; Vulkan loader/driver/device acceptance required",
    ),
    Artifact(
        "Linux",
        "cpu",
        "sd-master-fc350be-bin-Linux-Ubuntu-26.04-x86_64.zip",
        29022149,
        "11283f1358e0ac7f17a27f473fd51a53de338f9849b708f701516f5c64837ff0",
        "Ubuntu 26.04 x86-64 publisher build; libc/loader/CPU acceptance required; WSL separate",
    ),
    Artifact(
        "Linux",
        "vulkan",
        "sd-master-fc350be-bin-Linux-Ubuntu-26.04-x86_64-vulkan.zip",
        41894957,
        "bd46e4dae1e145b91321fd7452d7e8e0765c34a95a336c392dd180f455ba8f17",
        "Ubuntu 26.04 x86-64 publisher build; libc/loader/Vulkan acceptance required; WSL separate",
    ),
)


def artifact_for(system, machine, backend):
    """Return publisher metadata, never a claim of device or Linux ABI eligibility."""
    if machine.lower() not in ("amd64", "x86_64"):
        raise ValueError("No pinned optional diffusion artifact for this architecture")
    return next(
        (item for item in ARTIFACTS if item.system == system and item.backend == backend), None
    )


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload(directory):
    files = []
    for path in sorted(directory.iterdir()):
        if path.name == "rf-runtime.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Unexpected non-file in optional diffusion runtime")
        files.append((path.name, path.stat().st_size, _hash_file(path)))
    return files


def open_runtime(directory, artifact, *, device=None):
    """Verify cached payload at explicit preparation/activation, never on rendering."""
    from ruinedfooocus.features.models.model_identity import file_signature

    if artifact not in ARTIFACTS:
        raise ValueError("Only the frozen publisher artifact matrix may be prepared")
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "rf-runtime.json").read_text(encoding="utf-8"))
    if (
        manifest.get("revision") != SOURCE_REVISION
        or manifest.get("archive_sha256") != artifact.sha256
        or manifest.get("files") != [list(item) for item in _payload(directory)]
    ):
        raise RuntimeError(
            "Optional diffusion runtime pin/payload integrity differs from its manifest"
        )
    executable = directory / ("sd-cli.exe" if artifact.system == "Windows" else "sd-cli")
    if not executable.is_file() or not all(
        (directory / name).is_file() for name in ("stable-diffusion.cpp.txt", "ggml.txt")
    ):
        raise RuntimeError(
            "Optional diffusion runtime executable or upstream license notices are missing"
        )
    return RuntimeCandidate(
        executable,
        artifact.backend,
        device or ("cpu" if artifact.backend == "cpu" else ""),
        artifact_sha256=artifact.sha256,
        payload_signatures=tuple(
            tuple(file_signature(directory / item[0])) for item in manifest["files"]
        )
        + (tuple(file_signature(directory / "rf-runtime.json")),),
    )


def prepare_runtime(artifact, directory, *, archive, check_cancelled, progress=None, service=None):
    """Explicit acquisition action; single-flight transfer, lease, lock and atomic publish.

    Caller applies offline/download authorization and platform admission before
    invoking this function. A completed preparation does not qualify execution.
    Existing runtimes are verified and reused, never replaced under live users.
    """
    import zipfile
    from ruinedfooocus.core.preparation import preparation_service
    from ruinedfooocus.packaging.install_lock import installation_lock

    if artifact not in ARTIFACTS:
        raise ValueError("Unknown optional diffusion artifact")
    check_cancelled()
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    if archive == directory or directory in archive.parents:
        raise ValueError("Archive cache must be outside the immutable runtime directory")
    service = service or preparation_service
    request = artifact.preparation_request(archive)
    subscription = service.submit(request, progress)
    try:
        while not subscription.done():
            check_cancelled()
            try:
                subscription.result(timeout=0.1)
            except TimeoutError:
                pass
        subscription.result()
        check_cancelled()
    except BaseException:
        subscription.cancel()
        raise
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (
        service.artifact_lease(archive, cancelled=_Cancellation(check_cancelled)),
        installation_lock(directory, timeout=1),
    ):
        check_cancelled()
        if directory.exists():
            return open_runtime(directory, artifact)
        # Transfer verified the frozen size and hash. Recheck under the read
        # lease because this cached archive may have preceded the subscription.
        if archive.stat().st_size != artifact.size or _hash_file(archive) != artifact.sha256:
            raise RuntimeError("Optional diffusion archive integrity mismatch")
        with tempfile.TemporaryDirectory(prefix="rf-sdcpp-", dir=directory.parent) as temporary:
            stage = Path(temporary) / "runtime"
            stage.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                names = [member.filename.casefold() for member in members]
                if (
                    len(members) > 4096
                    or len(set(names)) != len(names)
                    or sum(m.file_size for m in members) > 2 * 1024**3
                ):
                    raise RuntimeError("Duplicate or oversized optional runtime archive")
                for member in members:
                    check_cancelled()
                    name = member.filename
                    # The publisher's zip -j layout is flat. Refuse nested paths,
                    # symlinks and platform path tricks instead of extracting them.
                    if (
                        not name
                        or name in (".", "..")
                        or any(c in name for c in "/\\:")
                        or any(ord(c) < 32 for c in name)
                        or name != name.rstrip(" .")
                        or re.match(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", name, re.I)
                        or stat.S_ISLNK(member.external_attr >> 16)
                        or member.is_dir()
                    ):
                        raise RuntimeError("Unsafe optional diffusion archive member")
                    with bundle.open(member) as source, (stage / name).open("xb") as output:
                        while chunk := source.read(1024 * 1024):
                            check_cancelled()
                            output.write(chunk)
            manifest = {
                "revision": SOURCE_REVISION,
                "release": RELEASE,
                "archive_sha256": artifact.sha256,
                "files": _payload(stage),
            }
            (stage / "rf-runtime.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            open_runtime(stage, artifact)
            if artifact.system == "Linux":
                (stage / "sd-cli").chmod(0o755)
            check_cancelled()
            os.rename(stage, directory)
        return open_runtime(directory, artifact)
