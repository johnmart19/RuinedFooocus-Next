"""PyTorch and llama.cpp package selection."""

import importlib.metadata
import base64
import hashlib
import json
import os
import re
import sys
import subprocess
import sysconfig
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from ruinedfooocus.packaging.runtime_support import installed_rocm_platform
from ruinedfooocus.packaging.runtime_support import installed_cuda_platform
from ruinedfooocus.packaging.runtime_bundles import bundle_for


def preserve_installed_cuda(bundle, torch_platform, reinstall=False):
    from packaging.version import Version

    return bool(
        bundle
        and not reinstall
        and installed_cuda_platform(bundle) == torch_platform
        and Version(bundle["versions"]["torch"]) >= Version("2.6.0")
    )


def installed_torch_constraints():
    versions = {}
    packages = ["torch", "torchvision"]
    if os.environ.get("TORCH_PLATFORM") == "directml":
        packages += ["numpy", "scipy", "torch-directml"]
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    # Keep the wheel-owned vendor runtime closure, not just the visible
    # Torch companions. An overlay must not upgrade HIP/CUDA libraries behind
    # an unchanged Torch version or redirect them to an older system SDK.
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name.startswith(("rocm-sdk-", "nvidia-", "pytorch-triton", "triton")) or name == "rocm":
            versions[name] = distribution.version
    # Transformers 5.17 disables its torch integration on Torch 2.4, which
    # Microsoft DirectML still requires. Preserve the last verified integration.
    if versions.get("torch", "").split(".")[:2] == ["2", "4"]:
        versions["transformers"] = "5.13.0"
    return {"versions": versions} if versions else None


def preserve_installed_rocm(installed_rocm, torch_platform, reinstall=False):
    if not installed_rocm or reinstall:
        return False
    # An explicit different ROCm version is a request to change environments.
    return torch_platform == installed_rocm_platform(installed_rocm)


@contextmanager
def torch_constraints(installed_rocm):
    """Keep vendor builds intact when resolving transitive dependencies."""
    if not installed_rocm:
        yield ""
        return
    with TemporaryDirectory(prefix="ruinedfooocus-torch-") as directory:
        path = Path(directory) / "constraints.txt"
        path.write_text(
            "".join(f"{name}=={version}\n" for name, version in installed_rocm["versions"].items()),
            encoding="utf-8",
        )
        yield f' --constraint "{path}"'


def torch_install_commands(torchruntime, torch_platform, os_platform, nightly=False):
    if torch_platform == "xpu" and sys.version_info[:2] < bundle_for("xpu").python_min:
        raise RuntimeError("Intel XPU PyTorch wheels need Python 3.11 or newer.")
    if torch_platform in ("cpu", "xpu"):
        # Pinned like CUDA/ROCm; torchruntime would install the newest XPU wheels.
        bundle = bundle_for(torch_platform)
        return [
            [
                f"torch=={bundle.torch}",
                f"torchvision=={bundle.vision}",
                "--index-url",
                bundle.index,
            ]
        ]
    if torch_platform == "directml":
        import platform

        if os_platform not in ("Windows", "Linux") or not (3, 10) <= sys.version_info[:2] <= (
            3,
            12,
        ):
            raise RuntimeError("DirectML requires Python 3.10–3.12 on Windows or WSL.")
        if platform.machine().lower() not in ("amd64", "x86_64"):
            raise RuntimeError("The published DirectML wheels require x86_64 Python.")
        if os_platform == "Linux" and "microsoft" not in platform.release().lower():
            raise RuntimeError(
                "DirectML requires Windows or WSL, not native Linux. Use --cpu or your GPU's native backend."
            )
        return [
            [
                "torch==2.4.1+cpu",
                "torchvision==0.19.1+cpu",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            ["--no-deps", "torch-directml==0.2.5.dev240914", "numpy==1.26.4", "scipy==1.15.3"],
        ]
    if torch_platform == "cu124" and not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError(
            "CUDA 12.4 requires Python 3.10–3.13 in RuinedFooocus. Use Python 3.10 or 3.12 for GTX 1080 Ti."
        )
    # Keep the selected CUDA runtime: newer wheels may drop older GPU support.
    from ruinedfooocus.packaging.cuda_selection import (
        CUDA_PACKAGES,
        cuda_index,
        cuda_wheels_available,
        validate_cuda_runtime,
    )

    if os_platform not in ("Windows", "Linux") or torch_platform not in (*CUDA_PACKAGES, "cu134"):
        return torchruntime.installer.get_install_commands(torch_platform, [])
    validate_cuda_runtime(torch_platform, nightly=nightly)
    if not cuda_wheels_available(torch_platform, nightly=nightly):
        raise RuntimeError(
            f"No {torch_platform} wheel bundle matches Python {sys.version.split()[0]} and this OS/CPU architecture."
        )
    if nightly:
        return [
            [
                "--pre",
                "--upgrade",
                "--force-reinstall",
                "--only-binary=:all:",
                "torch",
                "torchvision",
                "--index-url",
                cuda_index(torch_platform, nightly),
            ],
        ]
    torch, vision = CUDA_PACKAGES[torch_platform]
    packages = [f"torch=={torch}+{torch_platform}", f"torchvision=={vision}+{torch_platform}"]
    commands = [packages + ["--index-url", f"https://download.pytorch.org/whl/{torch_platform}"]]
    # Windows Triton is optional; an unbounded version breaks older torch builds.
    return commands


XLLAMACPP_VERSION = "2026.11.12361"
XLLAMACPP_LLAMA_TAG = "b12361"


def xllamacpp_index(torch_platform, os_platform, version=None, choice="Auto", *, machine=None):
    """Platform-aware LLM wheel policy, independent of the image Torch backend.

    Published xllamacpp 2026.11.12361 wheel matrix:
    - PyPI (CPU / macOS Metal): Win x64, Linux x64/ARM64, macOS x64/ARM64.
    - Vulkan: Win x64/ARM64, Linux x64, macOS x64.
    - CUDA 12.8 / 13.2: Win x64, Linux x64/ARM64.
    - ROCm 6.4 / 7.2: Linux x64 only (conflicts in-process with ROCm 10 PyTorch).

    Selecting an index is acquisition policy, not proof of a usable device.
    Runtime activation separately verifies the selected native device.
    """
    import platform as _platform

    arch = (machine or _platform.machine()).lower()
    x64 = arch in ("amd64", "x86_64")
    arm64 = arch in ("arm64", "aarch64")
    explicit = choice not in (None, "Auto")
    torch_platform = str(torch_platform or "")
    if explicit:
        from ruinedfooocus.packaging.hardware_inventory import XLLAMACPP_INDEXES

        if choice not in XLLAMACPP_INDEXES:
            raise RuntimeError(f"Unknown xllamacpp backend {choice!r}.")
        if choice.startswith("ROCm"):
            if os_platform != "Linux" or not x64:
                raise RuntimeError("xllamacpp ROCm wheels exist only for Linux/WSL x86_64.")
            if torch_platform.startswith("rocm1"):
                raise RuntimeError(
                    f"The xllamacpp {choice} wheel crashes beside ROCm 10 PyTorch. "
                    "Use native llama.cpp with ROCm, or xllamacpp on Vulkan/CPU."
                )
        elif choice.startswith("CUDA"):
            if os_platform not in ("Windows", "Linux") or (os_platform == "Windows" and not x64):
                raise RuntimeError(
                    f"xllamacpp {choice} wheels exist only for Windows x86_64 and Linux x86_64/ARM64."
                )
        elif choice == "Vulkan":
            if (os_platform == "Linux" and not x64) or (os_platform == "Darwin" and not x64):
                raise RuntimeError(
                    "xllamacpp Vulkan wheels exist only for Windows (x86_64/ARM64), Linux x86_64, and Intel macOS."
                )
        backend = XLLAMACPP_INDEXES[choice]
        candidates = [backend] if backend else []
    else:
        from ruinedfooocus.backends.llama.runtime_policy import preferred_backend

        if os_platform == "Linux" and arm64:
            # No Linux ARM64 Vulkan/ROCm wheels exist; select CUDA when on a CUDA Torch build.
            if torch_platform.startswith("cu13"):
                backend = "cu132"
                candidates = ["cu132", "cu128"]
            elif torch_platform.startswith("cu12"):
                backend = "cu128"
                candidates = ["cu128"]
            else:
                backend = None
                candidates = []
        elif preferred_backend(os_platform) == "Vulkan":
            backend = "vulkan"
            candidates = ["vulkan"]
            if torch_platform.startswith("cu13") and (
                (os_platform == "Windows" and x64) or (os_platform == "Linux" and (x64 or arm64))
            ):
                candidates.append("cu132")
            elif torch_platform.startswith("cu12") and (
                (os_platform == "Windows" and x64) or (os_platform == "Linux" and (x64 or arm64))
            ):
                candidates.append("cu128")
            elif (
                os_platform == "Linux"
                and x64
                and torch_platform.startswith("rocm")
                and not torch_platform.startswith("rocm1")
            ):
                candidates += ["rocm-7.2.4", "rocm-6.4.1"]
        else:
            backend = None
            candidates = []
    root = "https://xorbitsai.github.io/xllamacpp/whl"
    if version and backend:
        from ruinedfooocus.packaging.cuda_selection import wheel_available

        selected = None
        for candidate in candidates:
            if wheel_available(f"{root}/{candidate}", "xllamacpp", version):
                if candidate != backend and not explicit:
                    print(
                        f"No matching xllamacpp {backend} wheel for this Python/OS; using {candidate}."
                    )
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                f"No xllamacpp {version} {backend} wheel matches this Python/OS. Use native llama.cpp or a supported Python version."
            )
        backend = selected
    return f"{root}/{backend}" if backend else "https://pypi.org/simple"


def has_cuda_xllamacpp():
    return has_xllamacpp_device("CUDA")


def has_xllamacpp_device(prefix):
    # Check the loaded backend, not just the shared distribution version.
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                # Linux CUDA wheels use the libraries bundled with PyTorch. Load
                # them as the application does before probing xllamacpp's devices.
                "import sys\n"
                "if sys.platform == 'linux':\n"
                "    import torch\n"
                "import xllamacpp as x\n"
                f"print('RF_DEVICE=' + str(int(any(d['name'].startswith({prefix!r}) for d in x.get_device_info()))))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and "RF_DEVICE=1" in result.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _xllamacpp_installation():
    distribution = importlib.metadata.distribution("xllamacpp")
    # RECORD metadata alone can remain unchanged after a damaged DLL. Verify
    # recorded payload bytes without importing the extension or probing devices.
    for file in distribution.files or ():
        expected = file.hash
        if expected is None:
            continue  # RECORD itself and generated entry points may be unhashed.
        if expected.mode != "sha256":
            raise OSError("xllamacpp has an unsupported installed-file hash")
        digest = hashlib.sha256()
        with Path(file.locate()).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        if actual != expected.value:
            raise OSError(f"xllamacpp installed-file integrity mismatch: {file.name}")
    record = next(
        file
        for file in distribution.files or []
        if str(file).replace("\\", "/").endswith(".dist-info/RECORD")
    )
    record = Path(record.locate())
    from ruinedfooocus.core.native_runtime_paths import native_runtime_root

    from ruinedfooocus.packaging.package_plan import current_target, target_identity

    return native_runtime_root("llama.cpp", "xllamacpp-backend.json"), {
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "target": target_identity(current_target()),
    }


def xllamacpp_backend_matches(index):
    """Backend wheels share versions; remember the exact installed wheel source."""
    try:
        marker, installation = _xllamacpp_installation()
        return json.loads(marker.read_text(encoding="utf-8")) == dict(installation, index=index)
    except (importlib.metadata.PackageNotFoundError, StopIteration, OSError, ValueError):
        # Untracked/manual installs are replaced once rather than guessed from Torch.
        return False


def record_xllamacpp_backend(index):
    marker, installation = _xllamacpp_installation()
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(installation, index=index)), encoding="utf-8")
    temporary.replace(marker)


def cached_xllamacpp_backend_matches(index, *, cpu=False):
    """Offline CPU may reuse a tracked GPU wheel; offload is disabled by the loader."""
    if not cpu:
        return xllamacpp_backend_matches(index)
    try:
        marker, installation = _xllamacpp_installation()
        saved = json.loads(marker.read_text(encoding="utf-8"))
        known = ("https://pypi.org/simple", "https://xorbitsai.github.io/xllamacpp/whl/")
        source = saved.pop("index", "")
        return saved == installation and (source == known[0] or source.startswith(known[1]))
    except (importlib.metadata.PackageNotFoundError, StopIteration, OSError, ValueError):
        return False


def cached_xllamacpp_index():
    """Exact tracked source after interpreter and payload verification."""
    try:
        marker, installation = _xllamacpp_installation()
        saved = json.loads(marker.read_text(encoding="utf-8"))
        source = saved.pop("index", "")
        if saved == installation and (
            source == "https://pypi.org/simple"
            or source.startswith("https://xorbitsai.github.io/xllamacpp/whl/")
        ):
            return source
    except (importlib.metadata.PackageNotFoundError, StopIteration, OSError, ValueError):
        pass
    return None


def cleanup_xllamacpp_markers():
    """Old app markers can leave empty dist-info directories after a pip upgrade."""
    for directory in Path(sysconfig.get_path("purelib")).glob("xllamacpp-*.dist-info"):
        marker = directory / "ruinedfooocus-backend.json"
        if marker.is_file():
            marker.unlink()
            if not any(directory.iterdir()):
                directory.rmdir()


def has_vulkan_xllamacpp():
    # Backend wheels share a distribution version; version alone is insufficient.
    try:
        files = importlib.metadata.files("xllamacpp") or []
        return any(
            (
                file.name.lower() == "ggml-vulkan.dll"
                or re.fullmatch(r"vulkan-1(?:-[0-9a-f]+)?\.dll", file.name.lower())
                or re.fullmatch(r"lib(?:ggml-)?vulkan(?:-[0-9a-f]+)?\.so(?:\.[0-9]+)*", file.name.lower())
                or re.fullmatch(r"lib(?:ggml-)?vulkan(?:\.[0-9]+)*\.dylib", file.name.lower())
            )
            and Path(file.locate()).is_file()
            for file in files
        )
    except importlib.metadata.PackageNotFoundError:
        return False


def selected_llm_runtime(profile=None):
    from ruinedfooocus.core.paths import settings_path

    path = settings_path(profile)
    if not path.is_file():
        return "llama.cpp"
    values = json.loads(path.read_text(encoding="utf-8"))
    return values.get("llm_runtime", "llama.cpp")


def saved_xllamacpp_backend(profile=None):
    """Saved xllamacpp backend; Use CPU for everything forces the CPU wheel."""
    from ruinedfooocus.core.paths import settings_path
    from ruinedfooocus.packaging.backend_preferences import effective_backends

    path = settings_path(profile)
    try:
        values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        values = {}
    return effective_backends(values)["xllamacpp"]


def ensure_xllamacpp(
    torch_platform, os_platform, reinstall=False, installed_torch=None, choice="Auto"
):
    from ruinedfooocus.packaging.install_lock import installation_lock

    with installation_lock(Path(sys.prefix) / "xllamacpp"):
        return _ensure_xllamacpp(torch_platform, os_platform, reinstall, installed_torch, choice)


def _ensure_xllamacpp(torch_platform, os_platform, reinstall, installed_torch, choice):
    from ruinedfooocus.packaging.launch_util import is_installed
    from ruinedfooocus.packaging.startup_options import skip_dependency_install

    xlc_version = f"xllamacpp=={XLLAMACPP_VERSION}"
    # No wheel-index/network/device access while validating offline identity.
    index = xllamacpp_index(torch_platform, os_platform, choice=choice)
    if skip_dependency_install():
        if not is_installed(xlc_version):
            raise RuntimeError(
                "XLlamacpp is missing. Launch without --offline/--skip-dependency-install to install it, or select native llama.cpp."
            )
        if not cached_xllamacpp_backend_matches(index, cpu=choice == "CPU"):
            cached = cached_xllamacpp_index() if choice == "Auto" else None
            if cached:
                label = (
                    "CUDA"
                    if cached.rsplit("/", 1)[-1] in ("cu128", "cu132")
                    else "ROCm"
                    if "/rocm-" in cached
                    else None
                )
                if label and has_xllamacpp_device(label):
                    reason = "Preferred Vulkan wheel is not cached; using the verified cached GPU backend"
                    print(f"Chat Auto: {reason} ({label}).")
                    return {"backend": label, "fallback_reason": reason}
            raise RuntimeError(
                "Cached xllamacpp backend/interpreter identity does not match the selection. "
                "Restart online to acquire it, or select native llama.cpp."
            )
        return {"backend": "CPU" if choice == "CPU" else choice, "fallback_reason": ""}
    index = xllamacpp_index(
        torch_platform, os_platform, version=XLLAMACPP_VERSION, choice=choice
    )
    needs_vulkan = index.endswith("/vulkan")
    needs_cuda = index.rsplit("/", 1)[-1] in ("cu128", "cu132")
    needs_rocm = "/rocm-" in index
    wrong_backend = (
        not cached_xllamacpp_backend_matches(index, cpu=choice == "CPU")
        or needs_vulkan
        and not has_vulkan_xllamacpp()
        or needs_cuda
        and not has_cuda_xllamacpp()
        or needs_rocm
        and not has_xllamacpp_device("ROCm")
    )
    installed_before = is_installed(xlc_version)
    if reinstall or not installed_before or wrong_backend:
        if "xllamacpp" in sys.modules:
            raise RuntimeError(
                "xllamacpp is already imported. Restart before changing its native backend."
            )
        cleanup_xllamacpp_markers()
        from ruinedfooocus.packaging.package_plan import (
            PackageLayer,
            current_target,
            execute_plan,
            plan_packages,
            resolve_packages,
        )

        layer = PackageLayer(
            "xllamacpp",
            (xlc_version,),
            index,
            xlc_version.split("==")[1],
            "Apache-2.0",
            hashlib.sha256(xlc_version.encode()).hexdigest(),
            index,
            ("https://github.com/xorbitsai/xllamacpp/releases/download",)
            if index != "https://pypi.org/simple"
            else ("https://files.pythonhosted.org/packages",),
        )
        constraints = tuple(
            f"{name}=={version}"
            for name, version in (installed_torch or {}).get("versions", {}).items()
        )
        artifacts = resolve_packages(
            (layer,),
            index=index,
            extra_indexes=("https://pypi.org/simple",),
            constraints=constraints,
        )
        plan = plan_packages(
            current_target(),
            {"eligible": True},
            (layer,),
            (),
            artifacts,
            environment=sys.prefix,
            recipe=index,
            api_floor=None,
        )
        # All wheel backends support CPU acquisition; actual GPU visibility is
        # verified below/at activation and is never claimed by this package plan.
        execute_plan(plan, plan.digest, force=reinstall or wrong_backend and installed_before)
        if needs_cuda and not has_cuda_xllamacpp():
            raise RuntimeError(
                "xllamacpp CUDA installed but no CUDA device is available. Check the NVIDIA driver, or use native llama.cpp with Vulkan."
            )
        if needs_rocm and not has_xllamacpp_device("ROCm"):
            raise RuntimeError(
                "xllamacpp ROCm installed but no ROCm device is available. Check the AMD ROCm driver, or use Vulkan."
            )
        if needs_vulkan and not has_vulkan_xllamacpp():
            raise RuntimeError("xllamacpp Vulkan installed but its Vulkan libraries are missing.")
        record_xllamacpp_backend(index)
    return {"backend": "CPU" if choice == "CPU" else choice, "fallback_reason": ""}
