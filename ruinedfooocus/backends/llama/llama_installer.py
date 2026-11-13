"""Official llama.cpp binaries, cached separately from models and Python packages."""

import hashlib
from ruinedfooocus.core.paths import data_path
import json
import os
from pathlib import Path
import platform
import shutil
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import venv
import uuid

import requests
from ruinedfooocus.packaging.startup_options import skip_dependency_install
from ruinedfooocus.backends.llama.runtime_policy import preferred_backend

VERSION = "b12366"
CUDA_BACKEND = "CUDA 13.5"
RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/"


def cuda_backend(version=VERSION):
    build = re.fullmatch(r"b([0-9]+)", version)
    return "CUDA 13.5" if build and int(build[1]) >= 12327 else "CUDA 13.4"


def canonical_backend(backend, *, version=VERSION):
    return (
        cuda_backend(version)
        if backend in ("CUDA 13.0", "CUDA 13.2", "CUDA 13.4", "CUDA 13.5")
        else backend
    )


def _check(check_cancelled):
    if check_cancelled is not None:
        check_cancelled()


def _emit(progress_callback, phase, *, backend=None, package=None, version=VERSION, **fields):
    if progress_callback is not None:
        progress_callback(
            {"phase": phase, "runtime": {"tag": version, "backend": backend, "package": package}}
            | fields
        )


def _callbacks(progress_callback, check_cancelled):
    return {
        key: value
        for key, value in (
            ("progress_callback", progress_callback),
            ("check_cancelled", check_cancelled),
        )
        if value is not None
    }


def _run_cancellable(command, *, check_cancelled=None, **options):
    """Adapt existing SDK callers to the shared bounded, owned-child runner."""
    if check_cancelled is None:
        return subprocess.run(command, **options)
    from ruinedfooocus.core.bounded_subprocess import run_bounded

    _check(check_cancelled)
    try:
        stdout, stderr = run_bounded(
            command,
            timeout=options.get("timeout", 1800),
            max_output=16 * 1024**2,
            env=options.get("env"),
            check_cancelled=check_cancelled,
        )
    except RuntimeError as error:
        if options.get("check"):
            raise subprocess.CalledProcessError(1, command) from error
        return subprocess.CompletedProcess(command, 1, "", "")
    if options.get("text"):
        stdout, stderr = (
            data.decode("utf-8", errors=options.get("errors", "replace"))
            for data in (stdout, stderr)
        )
    return subprocess.CompletedProcess(command, 0, stdout, stderr)


def validate_cuda_backend(backend):
    backend = canonical_backend(backend)
    if not backend.startswith("CUDA "):
        return
    from ruinedfooocus.packaging.cuda_selection import nvidia_devices

    capabilities, driver = nvidia_devices()
    required = tuple(int(part) for part in backend.split()[1].split("."))
    if not capabilities or driver is None:
        raise RuntimeError(
            "Cannot validate the NVIDIA GPU/driver. Check nvidia-smi or select Vulkan/CPU."
        )
    if driver < required:
        raise RuntimeError(
            f"{backend} requires a newer NVIDIA driver; select Vulkan or update the driver."
        )
    if any(
        capability < 5
        or (required[0] >= 13 and capability < 7.5)
        or (required == (12, 4) and capability >= 10)
        for capability in capabilities
    ):
        raise RuntimeError(
            f"{backend} does not support this GPU architecture. Select a compatible CUDA backend or Vulkan."
        )


def backend_choices(system=None, machine=None, *, version=VERSION):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    choices = ["Auto", "CPU"]
    x64 = machine in ("amd64", "x86_64")
    if system == "Windows" and (x64 or machine in ("arm64", "aarch64")):
        choices += (
            ["Vulkan", "ROCm", "SYCL", "CUDA 12.4", cuda_backend(version)]
            if x64
            else ["Vulkan", cuda_backend(version)]
        )
    elif system == "Linux":
        choices += ["Vulkan"]
        if x64:
            # SYCL is the official Intel GPU build (oneAPI runtime bundled).
            choices += ["ROCm", "SYCL", "CUDA 12.8", cuda_backend(version)]
            from ruinedfooocus.backends.llama.llama_build import LABEL, built_server

            if built_server():
                choices.append(LABEL)
        elif machine in ("arm64", "aarch64"):
            choices += [cuda_backend(version)]
    return choices


def package_name(system, machine, gpu, rocm=False, backend="Auto", *, version=VERSION):
    backend = canonical_backend(backend, version=version)
    if backend not in backend_choices(system, machine, version=version):
        raise RuntimeError(
            f"No official {backend} build for {system}/{machine}; use LLAMA_SERVER for a custom build."
        )
    if backend == "CPU":
        gpu = False
    elif backend != "Auto":
        gpu = True
    arch = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
        machine.lower()
    )
    if not arch:
        raise RuntimeError("Set LLAMA_SERVER to a llama-server binary for this CPU.")
    if system == "Windows":
        backend = {
            "ROCm": "rocm-10.0",
            "SYCL": "sycl",
            "CUDA 12.4": "cuda-12.4",
            cuda_backend(version): "cuda-" + cuda_backend(version).split()[1],
            "Vulkan": "vulkan",
        }.get(backend) or ("vulkan" if gpu else "cpu")
        return f"llama-{version}-bin-win-{backend}-{arch}.zip"
    if system == "Linux":
        use_rocm = gpu and backend == "ROCm"
        if backend.startswith("CUDA "):
            backend = "cuda-" + backend.split()[1] + "-"
        elif backend == "SYCL":
            backend = "sycl-fp16-"
        else:
            backend = "rocm-10.0-" if use_rocm else "vulkan-" if gpu else ""
        return f"llama-{version}-bin-ubuntu-{backend}{arch}.tar.gz"
    if system == "Darwin":
        return f"llama-{version}-bin-macos-{arch}.tar.gz"
    raise RuntimeError("Set LLAMA_SERVER to a llama-server binary for this OS.")


def rocm_target(gpu, windows=False):
    if not gpu or platform.system() != "Linux" and not windows:
        return None
    # Native ROCm is an independent process and SDK. A CPU/CUDA image Torch
    # build must not hide an AMD adapter or choose its target for this engine.
    from ruinedfooocus.packaging.vendor_admission import linux_amd_adapters
    from ruinedfooocus.packaging.rocm_installer import detect_rocm_targets

    targets = (
        sorted({item["gfx"] for item in linux_amd_adapters()})
        if platform.system() == "Linux"
        else []
    )
    targets = targets or detect_rocm_targets(platform.system())
    if not targets:
        return None
    if len(targets) != 1:
        raise RuntimeError(
            "Native ROCm target is ambiguous across AMD adapters; select Vulkan or configure a custom LLAMA_SERVER environment"
        )
    target = targets[0]
    if not re.fullmatch(r"gfx[0-9a-f]+", target):
        raise RuntimeError(f"Cannot select llama.cpp ROCm libraries for GPU {target}")
    return target


def active_rocm_directory(directory):
    """A published SDK path must remain inside its owned cache parent."""
    try:
        saved = json.loads(
            (directory.parent / (directory.name + ".active.json")).read_text(encoding="utf-8")
        )
        if not isinstance(saved, dict) or not isinstance(saved.get("directory"), str):
            return directory
        active = Path(saved["directory"]).resolve()
        if active.parent == directory.parent.resolve() and active.name.startswith(
            directory.name + "-"
        ):
            return active
    except (OSError, ValueError, KeyError):
        pass
    return directory


def ensure_rocm_libraries(
    directory, target, *, reinstall=False, progress_callback=None, check_cancelled=None
):
    """Resolve all SDK wheels before mutation in a separate, ABI-matched venv."""
    from ruinedfooocus.packaging.install_lock import installation_lock
    from ruinedfooocus.packaging.rocm_installer import rocm_metadata_layer
    from packaging.requirements import Requirement
    from dataclasses import replace
    from ruinedfooocus.packaging.package_plan import (
        current_target,
        execute_plan,
        plan_packages,
        resolve_packages,
        save_plan,
    )

    base = directory
    directory = active_rocm_directory(base)
    if directory.exists():
        # Never replace libraries mapped by a running native server. Retain the
        # old version and publish a separate SDK environment after verification.
        directory = base.with_name(base.name + "-" + uuid.uuid4().hex[:12])
    python = directory / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    if skip_dependency_install():
        raise RuntimeError(
            "Native ROCm libraries need acquisition; restart online or select Vulkan/CPU"
        )
    _check(check_cancelled)
    _emit(
        progress_callback,
        "resolving_runtime",
        backend="ROCm",
        package="rocm-sdk-10.0.0",
        message="Resolving isolated ROCm SDK dependencies",
    )
    runner = (
        {
            "runner": lambda command, **kwargs: _run_cancellable(
                command, check_cancelled=check_cancelled, **kwargs
            )
        }
        if check_cancelled is not None
        else {}
    )
    selector = rocm_metadata_layer("10.0.0")
    index = "https://stable.repo.amd.com/rocm/whl-next/"
    layer = replace(
        selector,
        identity="native-llama-rocm",
        requirements=(
            f"rocm[libraries,device-{target}] @ {Requirement(selector.requirements[0]).url}",
        ),
    )
    with installation_lock(base.parent / (base.name + ".prepare")):
        if skip_dependency_install():
            raise RuntimeError(
                "Native ROCm libraries need acquisition; restart online or select Vulkan/CPU"
            )
        artifacts = resolve_packages(
            (layer,), index=index, extra_indexes=("https://pypi.org/simple",), **runner
        )
        plan = plan_packages(
            current_target(),
            {"eligible": True},
            (layer,),
            (),
            artifacts,
            environment=directory,
            recipe="native-rocm10.0",
            api_floor=None,
        )
        if not plan.ready:
            raise RuntimeError(
                "Native ROCm dependency closure is unavailable: " + "; ".join(plan.conflicts)
            )
        if not python.exists():
            _check(check_cancelled)
            if check_cancelled is None:
                venv.EnvBuilder(with_pip=True).create(directory)
            else:
                _run_cancellable(
                    [sys.executable, "-m", "venv", str(directory)],
                    check=True,
                    check_cancelled=check_cancelled,
                )
        _emit(
            progress_callback,
            "downloading_runtime",
            backend="ROCm",
            package="rocm-sdk-10.0.0",
            message="Acquiring isolated ROCm SDK dependencies",
        )
        execute_plan(plan, plan.digest, force=reinstall, interpreter=python, **runner)
        _check(check_cancelled)
        _run_cancellable(
            [
                str(python),
                "-c",
                "import rocm_sdk as r; assert r.__version__=='10.0.0'; assert all(list(r.find_libraries(name)) for name in ('amdhip64','hipblas','hipblaslt'))",
            ],
            check=True,
            check_cancelled=check_cancelled,
        )
        save_plan(plan, directory / "rf-package-plan.json")
        _check(check_cancelled)
        if directory != base:
            pointer = base.parent / (base.name + ".active.json")
            temporary = pointer.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"directory": str(directory.resolve()), "plan_digest": plan.digest}),
                encoding="utf-8",
            )
            temporary.replace(pointer)
    return python


def runtime_environment(
    gpu,
    reinstall=False,
    backend="Auto",
    *,
    acquire=True,
    progress_callback=None,
    check_cancelled=None,
):
    _check(check_cancelled)
    callbacks = _callbacks(progress_callback, check_cancelled)
    backend = canonical_backend(backend)
    from ruinedfooocus.backends.comfy.comfy_compat import child_device_environment

    env = child_device_environment(service="llama")
    if (
        backend == "ROCm"
        and platform.system() == "Windows"
        and gpu
        and not os.environ.get("LLAMA_SERVER")
    ):
        previous_path = env.get("PATH")
        result = windows_rocm_environment(env, reinstall, install=acquire, **callbacks)
        if not acquire and result.get("PATH") == previous_path:
            raise RuntimeError(
                "Native ROCm SDK is not cached; acquire it explicitly before using this alternative"
            )
        return result
    if (
        backend in ("CUDA 12.8", CUDA_BACKEND)
        and platform.system() == "Linux"
        and gpu
        and not os.environ.get("LLAMA_SERVER")
    ):
        # Official Linux CUDA builds load the cudart/cuBLAS bundle extracted beside them.
        directory = package_directory(
            package_name(platform.system(), platform.machine(), gpu, backend=backend)
        )
        libraries = sorted({str(path.parent) for path in directory.rglob("libcudart.so*")})
        env["LD_LIBRARY_PATH"] = ":".join(libraries + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
        return env
    target = rocm_target(gpu) if backend == "ROCm" else None
    if not target or os.environ.get("LLAMA_SERVER"):
        return env
    # Native subprocess dependencies must not replace the app's working torch/ROCm.
    base = data_path("cache", "llama.cpp", "rocm-10.0")
    directory = active_rocm_directory(base)
    python = directory / "bin/python"
    marker = directory / f"{target}.ready"
    if reinstall or not marker.exists():
        if not acquire or skip_dependency_install():
            raise RuntimeError(
                "Native ROCm libraries are missing. Launch without --offline/--skip-dependency-install to install them, or select CPU/Vulkan."
            )
        python = ensure_rocm_libraries(base, target, reinstall=reinstall, **callbacks)
        _check(check_cancelled)
        directory = python.parent.parent
        marker = directory / f"{target}.ready"
        marker.touch()
    site = next(directory.glob("lib/python*/site-packages"))
    libraries = [site / "_rocm_sdk_core/lib", site / "_rocm_sdk_libraries/lib"]
    if Path("/dev/dxg").exists():
        from ruinedfooocus.backends.llama.wsl_bridge import library_directory

        libraries.append(library_directory(acquire=acquire, **callbacks))
        _check(check_cancelled)
    libraries.append(Path("/usr/lib/wsl/lib"))
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in libraries if path.is_dir())
    if os.environ.get("LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] += ":" + os.environ["LD_LIBRARY_PATH"]
    return env


def windows_rocm_environment(
    env, reinstall=False, install=True, *, query=None, progress_callback=None, check_cancelled=None
):
    _check(check_cancelled)
    callbacks = _callbacks(progress_callback, check_cancelled)
    # Reuse a complete ROCm 10 SDK, otherwise install into a separate environment.
    script = (
        "import json, rocm_sdk as r; assert r.__version__ == '10.0.0'; "
        "print(json.dumps(list(dict.fromkeys(str(p.parent) for p in "
        "r.find_libraries('amdhip64', 'hipblas', 'hipblaslt')))))"
    )

    def paths(python):
        _check(check_cancelled)
        if query is not None:
            try:
                return json.loads(query([str(python), "-c", script]))
            except RuntimeError:
                return None
        result = _run_cancellable(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check_cancelled=check_cancelled,
        )
        if result.returncode:
            return None
        return json.loads(result.stdout)

    libraries = None if reinstall else paths(sys.executable)
    if not libraries and (not install or skip_dependency_install()):
        directory = active_rocm_directory(data_path("cache", "llama.cpp", "rocm-10.0-windows"))
        python = directory / "Scripts/python.exe"
        libraries = paths(python) if python.exists() else None
        if not libraries:
            if not install:
                return env
            raise RuntimeError(
                "Native ROCm libraries are missing. Launch without --offline/--skip-dependency-install to install them, or select CPU/Vulkan."
            )
    if not libraries:
        target = rocm_target(True, windows=True)
        if not target:
            raise RuntimeError(
                "Cannot detect the AMD GPU target for ROCm 10. Select Vulkan or install a compatible ROCm SDK."
            )
        base = data_path("cache", "llama.cpp", "rocm-10.0-windows")
        directory = active_rocm_directory(base)
        python = directory / "Scripts/python.exe"
        libraries = paths(python) if python.exists() and not reinstall else None
        if not libraries:
            python = ensure_rocm_libraries(base, target, reinstall=reinstall, **callbacks)
            libraries = paths(python)
        if not libraries:
            raise RuntimeError(
                "ROCm 10 runtime libraries are unavailable. Select Vulkan or check the SDK installation."
            )
    env["PATH"] = os.pathsep.join(libraries + [env.get("PATH", "")])
    return env


def package_directory(package, *, version=VERSION):
    return data_path(
        "cache", "llama.cpp", version, package.removesuffix(".zip").removesuffix(".tar.gz")
    )


def server_path(
    gpu=True, backend="Auto", *, acquire=True, progress_callback=None, check_cancelled=None
):
    _check(check_cancelled)
    override = os.environ.get("LLAMA_SERVER")
    if override:
        executable = shutil.which(override) or override
        if not Path(executable).is_file():
            raise FileNotFoundError(f"LLAMA_SERVER does not exist: {override}")
        return Path(executable).resolve()
    backend = canonical_backend(backend)
    backend = preferred_backend(platform.system(), backend, gpu=gpu)
    from ruinedfooocus.backends.llama.llama_build import LABEL, built_server

    if gpu and backend == LABEL:
        executable = built_server()
        if not executable:
            raise RuntimeError(
                "The local CUDA runtime is unavailable. Build it in Settings or select another backend."
            )
        return executable
    if gpu:
        validate_cuda_backend(backend)
    package = package_name(
        platform.system(),
        platform.machine(),
        gpu,
        False,
        backend,
    )
    directory = package_directory(package)
    from ruinedfooocus.packaging.install_lock import installation_lock

    _emit(progress_callback, "resolving_runtime", backend=backend, package=package)
    # Fail promptly rather than block cancellation behind another installer.
    with installation_lock(directory, timeout=0):
        return _install_server(
            directory,
            package,
            acquire=acquire,
            backend=backend,
            **_callbacks(progress_callback, check_cancelled),
        )


def _file_sha256(path, *, check_cancelled=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _check(check_cancelled)
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_manifest(directory, package, *, version=VERSION, check_cancelled=None):
    files = {
        str(path.relative_to(directory)).replace("\\", "/"): _file_sha256(
            path, check_cancelled=check_cancelled
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "rf-runtime.json"
    }
    return {"version": version, "package": package, "files": files}


def _pinned_archives(package, version):
    manifest = Path(__file__).with_name("releases") / f"{version}.json"
    if not manifest.is_file():
        return None
    assets = {asset["name"]: asset for asset in json.loads(manifest.read_text())["assets"]}
    if package not in assets:
        return None
    names = [package]
    if "-win-cuda-" in package:
        names.append("cudart-llama-bin-win-" + package.split("-bin-win-", 1)[1])
    elif "-ubuntu-cuda-" in package:
        names.append("cudart-" + package)
    return [
        {
            "name": name,
            "url": assets[name]["browser_download_url"],
            "sha256": assets[name]["digest"][7:],
        }
        for name in names
    ]


def adopt_verified_cache(directory, package, trusted_manifest):
    """Migrate an offline cache only against an independently verified manifest.

    Do not manufacture publisher provenance by hashing an unknown executable.
    Portable/offline callers retain their verified source manifest separately.
    """
    expected = _runtime_manifest(directory, package)
    if any(trusted_manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Cached runtime differs from the verified offline manifest")
    if not trusted_manifest.get("archives"):
        raise RuntimeError("Offline manifest has no verified publisher archive identities")
    marker = directory / "rf-runtime.json"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(trusted_manifest, sort_keys=True), encoding="utf-8")
    temporary.replace(marker)


def _install_server(
    directory,
    package,
    *,
    acquire=True,
    version=VERSION,
    backend=None,
    progress_callback=None,
    check_cancelled=None,
):
    _check(check_cancelled)
    callbacks = _callbacks(progress_callback, check_cancelled)
    executable = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    found = list(directory.rglob(executable))
    if found:
        try:
            _emit(
                progress_callback,
                "verifying_runtime",
                backend=backend,
                package=package,
                version=version,
                message="Verifying cached llama.cpp",
            )
            saved = json.loads((directory / "rf-runtime.json").read_text(encoding="utf-8"))
            current = _runtime_manifest(
                directory, package, version=version, check_cancelled=check_cancelled
            )
            if (
                len(found) == 1
                and not found[0].is_symlink()
                and not directory.is_symlink()
                and all(
                    path.resolve().is_relative_to(directory.resolve())
                    for path in directory.rglob("*")
                )
                and isinstance(saved, dict)
                and (
                    _pinned_archives(package, version) is None
                    or saved.get("archives") == _pinned_archives(package, version)
                )
                and all(saved.get(key) == value for key, value in current.items())
            ):
                _check(check_cancelled)
                return found[0].resolve()
        except (OSError, ValueError):
            pass
        raise RuntimeError(
            "Cached native llama.cpp identity/integrity is unverified. "
            f"Restore its verified offline manifest, or explicitly configure LLAMA_SERVER={found[0].resolve()} "
            "to reuse this local binary as a custom runtime. No cache was removed."
        )
    if not acquire or skip_dependency_install():
        raise RuntimeError(
            "Native llama.cpp is missing. Launch without --offline/--skip-dependency-install once to install the selected runtime, or configure LLAMA_SERVER."
        )
    _emit(
        progress_callback,
        "resolving_runtime",
        backend=backend,
        package=package,
        version=version,
        message="Resolving official llama.cpp assets",
    )
    _check(check_cancelled)
    pinned = Path(__file__).with_name("releases") / f"{version}.json"
    if pinned.is_file():
        release = json.loads(pinned.read_text(encoding="utf-8"))
    else:
        with requests.get(RELEASES + version, timeout=(10, 15)) as response:
            response.raise_for_status()
            release = response.json()
    if release.get("tag_name") != version or release.get("draft") is not False:
        raise RuntimeError(
            "The official llama.cpp release identity did not match the requested build."
        )
    assets = release["assets"]
    _check(check_cancelled)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as temporary:
        temporary = Path(temporary)
        extracted = temporary / "runtime"
        archives = [
            download_package(
                package, assets, temporary, extracted, version=version, backend=backend, **callbacks
            )
        ]
        if "-win-cuda-" in package:
            dependency = "cudart-llama-bin-win-" + package.split("-bin-win-", 1)[1]
            archives.append(
                download_package(
                    dependency,
                    assets,
                    temporary,
                    extracted,
                    version=version,
                    backend=backend,
                    **callbacks,
                )
            )
        elif "-ubuntu-cuda-" in package:
            archives.append(
                download_package(
                    "cudart-" + package,
                    assets,
                    temporary,
                    extracted,
                    version=version,
                    backend=backend,
                    **callbacks,
                )
            )
        found = list(extracted.rglob(executable))
        if len(found) != 1 or not found[0].is_file() or found[0].is_symlink():
            raise RuntimeError("The downloaded package has no unique llama-server executable.")
        if (extracted / "rf-runtime.json").exists() or (extracted / "rf-runtime.json").is_symlink():
            raise RuntimeError("The llama.cpp archive contains reserved runtime metadata.")
        _emit(
            progress_callback,
            "verifying_runtime",
            backend=backend,
            package=package,
            version=version,
            message="Verifying staged llama.cpp files",
        )
        (extracted / "rf-runtime.json").write_text(
            json.dumps(
                _runtime_manifest(
                    extracted, package, version=version, check_cancelled=check_cancelled
                )
                | {"archives": archives},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _emit(
            progress_callback,
            "staging_runtime",
            backend=backend,
            package=package,
            version=version,
            message="Publishing verified llama.cpp",
        )
        _check(check_cancelled)
        extracted.rename(directory)
    return next(directory.rglob(executable)).resolve()


def download_package(
    package,
    assets,
    temporary,
    extracted,
    *,
    version=VERSION,
    backend=None,
    progress_callback=None,
    check_cancelled=None,
):
    from ruinedfooocus.backends.llama import llama_update

    asset = next((item for item in assets if item["name"] == package), None)
    url = f"{llama_update._DOWNLOAD}/{version}/{package}"
    if (
        not asset
        or asset.get("state") != "uploaded"
        or asset.get("browser_download_url") != url
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", asset.get("digest") or "")
        or type(asset.get("size")) is not int
        or not 0 < asset["size"] <= llama_update._MAX_ARCHIVE
    ):
        raise RuntimeError(f"No verified official llama.cpp package available: {package}")
    candidate = llama_update.PrebuiltUpdate(
        backend, version, package, url, asset["digest"][7:], asset["size"], True
    )
    print(f"Downloading llama.cpp {version} ({package})")
    archive = temporary / package
    llama_update._download(
        candidate, archive, lambda *_: None, **_callbacks(progress_callback, check_cancelled)
    )
    _emit(
        progress_callback,
        "verifying_runtime",
        backend=backend,
        package=package,
        version=version,
        message="Extracting verified llama.cpp archive",
    )
    llama_update._extract(archive, extracted, check_cancelled=check_cancelled)
    _check(check_cancelled)
    return candidate.archive()
