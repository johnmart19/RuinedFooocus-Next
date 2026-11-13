from ruinedfooocus.core.paths import data_path
from ruinedfooocus.packaging.startup_options import skip_dependency_install

"""Install optional GPU ONNX wheels without replacing CPU ONNX."""
from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile
import os
import uuid
from contextlib import ExitStack
from ruinedfooocus.packaging.install_lock import installation_lock
from ruinedfooocus.backends.onnx.provider_recipes import (
    MIGRAPHX_INDEX,
    MIGRAPHX_LIBS_INDEX,
    migraphx_decision,
)

VERSION = "1.30.1.202610302006"
ROOT = data_path()
WORKER = Path(__file__).with_name("onnx_worker.py")


def worker_command(root, probe=False, device_id=None, worker=None):
    # Windows venv python.exe is a redirector. Launch the real interpreter so
    # terminating an owned worker cannot leave its interpreter child behind.
    python = getattr(sys, "_base_executable", None) or sys.executable
    command = [
        python,
        "-I",
        "-u",
        str(worker or WORKER),
        "--runtime",
        str(root),
        "--site-dir",
        sysconfig.get_path("purelib"),
    ]
    if probe:
        command.append("--probe")
    if device_id is not None:
        command += ["--device-id", str(device_id)]
    return command


def runtime_path():
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return ROOT / "cache" / "onnx" / abi / VERSION


def runtime_available():
    root = runtime_path()
    try:
        marker = json.loads((root / "verified.json").read_text())
        return (
            marker["version"] == VERSION
            and marker["python"] == list(sys.version_info[:2])
            and hashlib.sha256((root / marker["binary"]).read_bytes()).hexdigest()
            == marker["sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def install_plan(target=None):
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("This optional DirectML runtime requires Windows x64.")
    if not (3, 11) <= sys.version_info[:2] <= (3, 14):
        raise RuntimeError(
            "The DirectML wheel supports Python 3.11–3.14; use native/CPU inference otherwise."
        )
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--no-deps",
        "--target",
        str(target or runtime_path()),
        f"onnxruntime-windowsml=={VERSION}",
    ]


def install_runtime(*, check_cancelled=None):
    from ruinedfooocus.backends.onnx.acquisition import run

    if skip_dependency_install():
        raise RuntimeError("Dependency installation is disabled by offline/install-skip settings.")
    if check_cancelled:
        check_cancelled()
    root = runtime_path()
    root.mkdir(parents=True, exist_ok=True)
    if not runtime_available():
        run(install_plan(), check=True, check_cancelled=check_cancelled, timeout=300)
    # The CPU ONNX dependency already supplies these packages. Validate the
    # isolated wheel's actual requirements instead of silently ignoring them.
    import importlib.metadata as metadata
    from packaging.requirements import Requirement

    for distribution in metadata.distributions(path=[str(root)]):
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            if not requirement.specifier.contains(
                metadata.version(requirement.name), prereleases=True
            ):
                raise RuntimeError(f"Optional ONNX runtime requires {requirement}.")
    result = run(
        worker_command(root, probe=True),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        check_cancelled=check_cancelled,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if not report.get("ok") or report["version"] != VERSION.split(".2026")[0]:
        raise RuntimeError("DirectML GPU validation failed.")
    binary = next((root / "onnxruntime/capi").glob("onnxruntime_pybind11_state*.pyd"))
    marker = dict(
        version=VERSION,
        python=list(sys.version_info[:2]),
        binary=str(binary.relative_to(root)),
        sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    if check_cancelled:
        check_cancelled()
    (root / "verified.json").write_text(json.dumps(marker, indent=2))
    return (
        "Windows GPU ONNX runtime installed; DirectML canary passed (model workload unqualified)."
    )


@dataclass(frozen=True)
class NativeRuntime:
    """A vendor ONNX Runtime build. Each provides the same ``onnxruntime`` import as
    the CPU package, so it lives in its own folder and runs only in the mask worker."""

    package: str
    version: str
    provider: str
    systems: tuple
    index: str = ""
    extras: tuple = ()
    windows_packages: tuple = ()
    python: tuple = ((3, 11), (3, 13))


CUDA12_FEED = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
NATIVE_RUNTIMES = {
    # PyPI onnxruntime-gpu targets CUDA 13; the [cuda, cudnn] extras add the libraries
    # when the installed PyTorch does not already provide CUDA 13.
    "CUDA": NativeRuntime(
        "onnxruntime-gpu",
        "1.32.0",
        "CUDAExecutionProvider",
        ("Windows", "Linux"),
        python=((3, 11), (3, 14)),
    ),
    # Classic OpenVINO build (Intel CPU / Arc GPU / NPU). Linux wheels bundle
    # OpenVINO; Windows needs the matching openvino package beside it.
    "OpenVINO": NativeRuntime(
        "onnxruntime-openvino",
        "1.24.1",
        "OpenVINOExecutionProvider",
        ("Windows", "Linux"),
        windows_packages=("openvino==2025.4.1",),
    ),
    # ROCm 10.1 plugin route; separate from Windows ML and legacy integrated ORT.
    "MIGraphX": NativeRuntime(
        "onnxruntime-ep-migraphx",
        "1.0.0+rocm10.1.0",
        "MIGraphXExecutionProvider",
        ("Linux",),
        python=((3, 12), (3, 12)),
    ),
}
CUDA12_RUNTIME = replace(NATIVE_RUNTIMES["CUDA"], version="1.29.0", index=CUDA12_FEED)
LEGACY_MIGRAPHX = NativeRuntime(
    "onnxruntime-migraphx",
    "1.27.1",
    "MIGraphXExecutionProvider",
    ("Linux",),
    python=((3, 10), (3, 14)),
)


def torch_cuda_major(version=None):
    """CUDA major of the installed PyTorch wheel (``+cu128`` -> 12), from metadata only."""
    if version is None:
        try:
            version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return None
    match = re.search(r"\+cu(\d\d)", version)
    return int(match[1]) if match else None


def native_runtime(name, torch_version=None):
    """The build for ``name``; CUDA follows the installed PyTorch CUDA major (12 or 13)."""
    if name not in NATIVE_RUNTIMES:
        raise ValueError(f"Unknown ONNX runtime {name!r}.")
    if name != "CUDA":
        return NATIVE_RUNTIMES[name]
    major = torch_cuda_major(torch_version)
    if major == 12:
        return CUDA12_RUNTIME
    if major == 13:
        return NATIVE_RUNTIMES["CUDA"]
    # No CUDA PyTorch to share libraries with: bring CUDA 13 and cuDNN 9 wheels.
    return replace(NATIVE_RUNTIMES["CUDA"], extras=("cuda", "cudnn"))


def native_runtime_path(name, runtime=None):
    if runtime is None and name == "MIGraphX":
        current = native_runtime_path(name, native_runtime(name))
        legacy = native_runtime_path(name, LEGACY_MIGRAPHX)
        if not _verified(current, native_runtime(name).version) and _verified(
            legacy, LEGACY_MIGRAPHX.version
        ):
            return legacy
    runtime = runtime or native_runtime(name)
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return ROOT / "cache" / "onnx" / abi / f"{runtime.package}-{runtime.version}"


def _verified(root, version):
    try:
        marker = json.loads((root / "verified.json").read_text())
        valid = (
            marker["version"] == version
            and marker["python"] == list(sys.version_info[:2])
            and hashlib.sha256((root / marker["binary"]).read_bytes()).hexdigest()
            == marker["sha256"]
        )
        for record in marker.get("plugin_libraries", []):
            library = (root / record["path"]).resolve()
            valid = (
                valid
                and library.is_relative_to(root.resolve())
                and hashlib.sha256(library.read_bytes()).hexdigest() == record["sha256"]
            )
        if "onnxruntime-ep-migraphx" in root.name and not marker.get("plugin_libraries"):
            return False
        return valid
    except (OSError, ValueError, KeyError, TypeError):
        return False


def native_runtime_auto_eligible(name):
    if not native_runtime_available(name):
        return False
    root = native_runtime_path(name)
    if name != "MIGraphX" or root.name.startswith("onnxruntime-migraphx-"):
        return True  # Preserve existing CUDA/OpenVINO/legacy compatibility.
    # A canary or generic marker flag is not real workload qualification. The
    # Windows ML contract producer cannot certify Linux's driver/device closure.
    # Keep this new route explicit until its own matching producer is shipped.
    return False


def native_runtime_available(name):
    """Installed and verified by a provider probe; reads the saved record only."""
    try:
        runtime = native_runtime(name)
    except ValueError:
        return False
    return _verified(native_runtime_path(name, runtime), runtime.version) or (
        name == "MIGraphX"
        and _verified(native_runtime_path(name, LEGACY_MIGRAPHX), LEGACY_MIGRAPHX.version)
    )


def native_install_plan(
    name,
    target=None,
    runtime=None,
    *,
    target_info=None,
    adapter=None,
    evidence=None,
    verification_pending=False,
):
    runtime = runtime or native_runtime(name)
    system = platform.system()
    if system not in runtime.systems:
        raise RuntimeError(f"The {name} ONNX runtime is not published for {system}.")
    if platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError(f"The {name} ONNX runtime requires x86_64 Python.")
    low, high = runtime.python
    if not low <= sys.version_info[:2] <= high:
        raise RuntimeError(
            f"The {name} ONNX runtime supports Python {low[0]}.{low[1]} to {high[0]}.{high[1]}."
        )
    packages = [f"{runtime.package}=={runtime.version}"]
    if name == "MIGraphX":
        if target_info is None:
            raise RuntimeError(
                "MIGraphX plugin acquisition requires an exact native Linux target, AMD GFX/driver and audited wheel closure; WSL is a separate technical-preview route."
            )
        if (
            target_info.system != system
            or target_info.python_abi != f"cp{sys.version_info.major}{sys.version_info.minor}"
            or target_info.python_version.split(".")[:2] != [str(v) for v in sys.version_info[:2]]
        ):
            raise RuntimeError("The MIGraphX target does not match this worker's interpreter/OS.")
        if "microsoft" in platform.release().lower() or "wsl" in platform.release().lower():
            raise RuntimeError(
                "WSL MIGraphX is a separate technical-preview route; native Linux evidence cannot qualify it."
            )
        policy_evidence = {
            key: value for key, value in (evidence or {}).items() if key != "artifact_paths"
        }
        decision = migraphx_decision(target_info, adapter, **policy_evidence)
        pending = {
            "Driver compatibility has not been verified",
            "Provider-specific MIGraphX device identity is unknown",
        }
        if not decision["eligible"] and not (
            verification_pending and set(decision["reasons"]).issubset(pending)
        ):
            raise RuntimeError(
                "MIGraphX plugin is not installable: " + "; ".join(decision["reasons"])
            )
        packages = []
        for artifact in evidence["artifacts"]:
            if len(artifact.sha256) != 64 or any(
                c not in "0123456789abcdef" for c in artifact.sha256.lower()
            ):
                raise RuntimeError(
                    "Every artifact in the MIGraphX dependency closure needs SHA-256 evidence."
                )
            source = artifact.source
            if not source.endswith(".whl"):
                source = source.rstrip("/") + "/" + artifact.filename
            from packaging.utils import canonicalize_name

            local = evidence.get("artifact_paths", {}).get(canonicalize_name(artifact.name))
            if local:
                local = Path(local).resolve()
                with local.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != artifact.sha256:
                        raise RuntimeError("The acquired wheel hash changed: " + artifact.name)
                source = local.as_uri()
            packages.append(f"{artifact.name} @ {source}#sha256={artifact.sha256}")
    if system == "Windows":
        packages += list(runtime.windows_packages)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--no-deps",
        "--target",
        str(target or native_runtime_path(name, runtime)),
        *packages,
    ]
    if runtime.index:
        command += ["--index-url", runtime.index]
    if name == "MIGraphX":
        command += ["--extra-index-url", MIGRAPHX_INDEX, "--extra-index-url", MIGRAPHX_LIBS_INDEX]
    return command


def missing_requirements(root, extras=()):
    """Requirements of the isolated wheels that neither the app nor the folder provides.

    A requirement the app already has at an incompatible version is an error: the
    worker shares the app's packages and must not silently load a mismatch.
    """
    from packaging.requirements import Requirement

    distributions = list(importlib.metadata.distributions(path=[str(root)]))
    local = {
        str(distribution.metadata["Name"]).lower().replace("_", "-")
        for distribution in distributions
    }
    environments = [{"extra": extra} for extra in extras] or [{"extra": ""}]
    missing = set()
    for distribution in distributions:
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not any(
                requirement.marker.evaluate(environment) for environment in environments
            ):
                continue
            if requirement.name.lower().replace("_", "-") in local:
                continue
            try:
                installed = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                missing.add(f"{requirement.name}{requirement.specifier}")
                continue
            if not requirement.specifier.contains(installed, prereleases=True):
                raise RuntimeError(f"Optional ONNX runtime requires {requirement}.")
    return sorted(missing)


def _pip_target(root, packages, index="", *, check_cancelled=None):
    from ruinedfooocus.backends.onnx.acquisition import run

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--no-deps",
        "--target",
        str(root),
        *packages,
    ]
    run(command + (["--index-url", index] if index else []), check_cancelled=check_cancelled)


def install_native_runtime(
    name, *, target_info=None, adapter=None, evidence=None, check_cancelled=None
):
    runtime = native_runtime(name)
    target = native_runtime_path(name, runtime)
    with installation_lock(target):
        return _install_native_runtime_locked(
            name,
            target_info=target_info,
            adapter=adapter,
            evidence=evidence,
            check_cancelled=check_cancelled,
            runtime=runtime,
        )


def _install_native_runtime_locked(
    name, *, target_info=None, adapter=None, evidence=None, check_cancelled=None, runtime=None
):
    runtime = runtime or native_runtime(name)
    with ExitStack() as cleanup:
        verification_pending = False
        if (
            name == "MIGraphX"
            and (target_info is None or evidence is None)
            and not _verified(native_runtime_path(name, runtime), runtime.version)
        ):
            if skip_dependency_install():
                raise RuntimeError(
                    "Dependency installation is disabled by offline/install-skip settings."
                )
            from ruinedfooocus.backends.onnx.acquisition import prepare_migraphx

            prepared = cleanup.enter_context(
                prepare_migraphx(adapter, check_cancelled=check_cancelled)
            )
            target_info, adapter, evidence = prepared.target, prepared.adapter, prepared.evidence
            verification_pending = True
        return _install_native_runtime(
            name,
            target_info=target_info,
            adapter=adapter,
            evidence=evidence,
            check_cancelled=check_cancelled,
            cleanup=cleanup,
            verification_pending=verification_pending,
            runtime=runtime,
        )


def _install_native_runtime(
    name,
    *,
    target_info=None,
    adapter=None,
    evidence=None,
    check_cancelled=None,
    cleanup,
    verification_pending=False,
    runtime=None,
):
    """Install the selected vendor build in isolation, probe its provider, record it."""
    if skip_dependency_install():
        raise RuntimeError("Dependency installation is disabled by offline/install-skip settings.")
    from ruinedfooocus.backends.onnx.acquisition import run

    if check_cancelled:
        check_cancelled()
    runtime = runtime or native_runtime(name)
    root = native_runtime_path(name, runtime)
    staging = None
    final = root
    if not _verified(root, runtime.version):
        # Validate before creating any cache paths. Publish only after a successful
        # worker canary; preserve the previous folder for explicit rollback.
        native_install_plan(
            name,
            root,
            runtime,
            target_info=target_info,
            adapter=adapter,
            evidence=evidence,
            verification_pending=verification_pending,
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = tempfile.TemporaryDirectory(prefix=".onnx-prepare-", dir=root.parent)
        cleanup.enter_context(staging)
        root = Path(staging.name)
        plan = native_install_plan(
            name,
            root,
            runtime,
            target_info=target_info,
            adapter=adapter,
            evidence=evidence,
            verification_pending=verification_pending,
        )
        run(plan, check_cancelled=check_cancelled)
        # Add missing libraries (for example CUDA/cuDNN wheels) without duplicating
        # packages the app provides; a few rounds cover their own dependencies.
        for _ in range(4):
            missing = missing_requirements(root, runtime.extras)
            if not missing:
                break
            if name == "MIGraphX":
                raise RuntimeError(
                    "The audited MIGraphX artifact closure omitted dependencies: "
                    + ", ".join(missing)
                )
            _pip_target(root, missing, check_cancelled=check_cancelled)
        else:
            raise RuntimeError(f"Could not resolve the {name} ONNX runtime requirements: {missing}")
    command = worker_command(root, probe=True) + ["--provider", name]
    if name == "MIGraphX":
        if adapter is None:
            saved = json.loads((root / "verified.json").read_text())
            adapter = saved["adapter"]
        if verification_pending:
            discovered = run(
                worker_command(root) + ["--provider", name, "--list-devices"],
                check_cancelled=check_cancelled,
                timeout=60,
                capture_output=True,
                text=True,
            )
            devices = json.loads(discovered.stdout.strip().splitlines()[-1])["devices"]
            matches = [
                item for item in devices if adapter.get("onnx_device_id") in (None, item["id"])
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "MIGraphX device binding is ambiguous; select an explicit provider device. Devices: "
                    + json.dumps(devices)
                )
            adapter = dict(adapter, onnx_device_id=matches[0]["id"])
        command += ["--ep-device", adapter["onnx_device_id"]]
    result = run(
        command,
        check_cancelled=check_cancelled,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    lines = result.stdout.strip().splitlines()
    try:
        report = json.loads(lines[-1]) if lines else {}
    except ValueError:
        report = {}
    if result.returncode != 0 or not report.get("ok"):
        reason = report.get("error") or (result.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"{name} ONNX runtime installed, but its provider failed: {reason}")
    expected_ort = "1.29.0" if name == "MIGraphX" else runtime.version
    if report.get("version") != expected_ort:
        raise RuntimeError(
            f"{name} worker loaded ORT {report.get('version')}; expected {expected_ort}."
        )
    if staging is None:
        return (
            f"{name} ONNX runtime {runtime.version} canary verified; model qualification unchanged."
        )
    if name == "MIGraphX":
        if report.get("device_identity") != adapter["onnx_device_id"]:
            raise RuntimeError("The MIGraphX canary ran on a different provider device.")
        adapter = dict(adapter, **{"rocm10.1_driver_supported": True})
        policy_evidence = {key: value for key, value in evidence.items() if key != "artifact_paths"}
        decision = migraphx_decision(target_info, adapter, **policy_evidence)
        if not decision["eligible"]:
            raise RuntimeError(
                "MIGraphX verification did not satisfy its exact recipe: "
                + "; ".join(decision["reasons"])
            )
    binary = next(
        path
        for path in (root / "onnxruntime" / "capi").glob("onnxruntime_pybind11_state*")
        if path.suffix in (".pyd", ".so")
    )
    marker = dict(
        version=runtime.version,
        python=list(sys.version_info[:2]),
        binary=str(binary.relative_to(root)),
        sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        provider=runtime.provider,
    )
    if name == "MIGraphX":
        marker["plugin_libraries"] = report.get("plugin_libraries", [])
        if not marker["plugin_libraries"]:
            raise RuntimeError("MIGraphX plugin canary supplied no native-library evidence.")
        marker["recipe"] = "onnx-migraphx-plugin-rocm10.1-ort1.29"
        marker["qualification"] = "canary_only_not_model_qualified"
        marker["artifacts"] = [
            dict(
                name=a.name,
                version=a.version,
                filename=a.filename,
                source=a.source,
                sha256=a.sha256,
            )
            for a in evidence["artifacts"]
        ]
        marker["target"] = dict(
            system=target_info.system,
            machine=target_info.machine,
            python_abi=target_info.python_abi,
            environment=target_info.environment,
        )
        marker["adapter"] = adapter
        marker["provider_device_id"] = report["device_identity"]
    (root / "verified.json").write_text(json.dumps(marker, indent=2))
    if check_cancelled:
        check_cancelled()
    if staging:
        previous = final.with_name(final.name + ".previous-" + uuid.uuid4().hex)
        if final.exists():
            os.replace(final, previous)
        try:
            os.replace(root, final)
        except BaseException:
            if previous.exists() and not final.exists():
                os.replace(previous, final)
            raise
        finally:
            staging.cleanup()
    return f"{name} ONNX runtime {runtime.version} installed; {runtime.provider} verified."
