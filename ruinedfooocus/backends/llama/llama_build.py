"""Optional, isolated Linux CUDA build of the pinned native chat runtime."""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import venv

from ruinedfooocus.packaging.startup_options import skip_dependency_install

LABEL = "CUDA (local build)"
TOOLKIT = "13.5.0"
_build_lock = threading.Lock()


def build_capabilities():
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        return []
    from ruinedfooocus.packaging.cuda_selection import nvidia_devices

    try:
        capabilities, driver = nvidia_devices()
        if (
            capabilities
            and driver
            and driver >= tuple(int(part) for part in TOOLKIT.split(".")[:2])
            and min(capabilities) >= 7.5
        ):
            return sorted({str(round(value * 10)) for value in capabilities})
    except (OSError, RuntimeError):
        pass
    return []


def build_root():
    from ruinedfooocus.backends.llama.llama_installer import VERSION

    from ruinedfooocus.core.paths import data_path

    return data_path("cache", "llama.cpp", VERSION, "local-cuda")


def built_server():
    from ruinedfooocus.backends.llama.llama_installer import VERSION

    root = build_root()
    try:
        data = json.loads((root / "ready.json").read_text())
        executable = root / "bin/llama-server"
        architectures = build_capabilities()
        toolkit = next(
            (root / "build-work/tools").glob("lib/python*/site-packages/nvidia/cu13"), None
        )
        if (
            data["version"] == VERSION
            and data["toolkit"] == TOOLKIT
            and architectures
            and set(architectures) <= set(data["architectures"])
            and toolkit
            and all(
                (toolkit / f"lib/lib{name}.so.13").is_file()
                for name in ("cudart", "cublas", "cublasLt")
            )
            and executable.is_file()
            and os.access(executable, os.X_OK)
        ):
            return executable
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _run(command, stage, log_path, env):
    """Yield bounded log tails, including when a compiler is quiet for a while."""
    from ruinedfooocus.core.child_process import start_process

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + stage + "\n")
        log.flush()
        process, _ = start_process(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                with log_path.open("rb") as source:
                    source.seek(max(0, log_path.stat().st_size - 4000))
                    tail = source.read().decode("utf-8", errors="replace")
                yield stage + "\n\n" + tail
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"{stage} failed. See {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


def build_cuda_runtime():
    """Build on explicit request; never change application Torch or the driver."""
    if skip_dependency_install():
        raise RuntimeError("Dependency installation is disabled by offline/install-skip settings.")
    from ruinedfooocus.backends.llama.llama_installer import VERSION

    if not _build_lock.acquire(blocking=False):
        raise RuntimeError("A native runtime build is already running.")
    try:
        architectures = build_capabilities()
        if not architectures:
            raise RuntimeError(
                f"Local CUDA builds require Linux x86_64, Turing or newer and a CUDA {TOOLKIT}-capable NVIDIA driver."
            )
        if not shutil.which("c++") or not shutil.which("make") or not shutil.which("git"):
            raise RuntimeError(
                "Install system build tools first. On Ubuntu: sudo apt install build-essential git python3-venv"
            )
        root = build_root()
        work = root / "build-work"
        work.mkdir(parents=True, exist_ok=True)
        log_path = root / "build.log"
        log_path.write_text("", encoding="utf-8")
        environment = os.environ.copy()
        # Searching inherited Windows PATH entries makes CMake very slow in WSL.
        environment["PATH"] = os.pathsep.join(
            path
            for path in environment.get("PATH", "").split(os.pathsep)
            if path and not path.startswith("/mnt/")
        )
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        tools = (work / "tools").resolve()
        python = tools / "bin/python"
        if not python.is_file():
            yield "Creating an isolated build environment…"
            venv.EnvBuilder(with_pip=True).create(tools)
        yield from _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"cuda-toolkit[nvcc,cudart,cublas,cccl]=={TOOLKIT}",
                "cmake>=3.24,<5",
            ],
            "Installing build dependencies",
            log_path,
            environment,
        )
        toolkit = next(tools.glob("lib/python*/site-packages/nvidia/cu13"))
        for name in ("cudart", "cublas", "cublasLt"):
            link = toolkit / f"lib/lib{name}.so"
            if not link.exists():
                link.symlink_to(f"lib{name}.so.13")
        source = (work / "source").resolve()
        if not source.exists():
            yield from _run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    VERSION,
                    "https://github.com/ggml-org/llama.cpp.git",
                    str(source),
                ],
                "Downloading pinned llama.cpp source",
                log_path,
                environment,
            )
        tag = subprocess.check_output(
            ["git", "-C", str(source), "describe", "--exact-match", "--tags"],
            env=environment,
            text=True,
        ).strip()
        if (
            tag != VERSION
            or subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], env=environment
            ).strip()
        ):
            raise RuntimeError(
                "The cached llama.cpp source differs from the pinned release. Use a clean source cache."
            )
        build = (work / "build").resolve()
        cmake = tools / "bin/cmake"
        command = [
            str(cmake),
            "-S",
            str(source),
            "-B",
            str(build),
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLLAMA_BUILD_NUMBER={VERSION.removeprefix('b')}",
            f"-DCMAKE_BUILD_RPATH=$ORIGIN;{toolkit / 'lib'}",
            f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{toolkit / 'lib'}",
            "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(architectures),
            f"-DCMAKE_CUDA_COMPILER={toolkit / 'bin/nvcc'}",
            f"-DCUDAToolkit_ROOT={toolkit}",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF",
        ]
        if Path("/usr/lib/wsl/lib/libcuda.so").is_file():
            command.append("-DCUDA_cuda_driver_LIBRARY=/usr/lib/wsl/lib/libcuda.so")
        yield from _run(command, "Configuring CUDA build", log_path, environment)
        yield from _run(
            [
                str(cmake),
                "--build",
                str(build),
                "--target",
                "llama-server",
                "-j",
                str(min(6, max(1, (os.cpu_count() or 2) // 2))),
            ],
            "Compiling llama.cpp",
            log_path,
            environment,
        )
        executable = build / "bin/llama-server"
        result = subprocess.run(
            [str(executable), "--list-devices"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        if not any(line.strip().startswith("CUDA") for line in result.stdout.splitlines()):
            raise RuntimeError(
                "The compiled runtime did not report a CUDA device; it was not enabled."
            )
        # Publish only a validated runtime. Existing builds remain usable during compilation.
        destination = root / "bin"
        destination.mkdir(exist_ok=True)
        (root / "ready.json").unlink(missing_ok=True)
        for path in (build / "bin").iterdir():
            if path.is_file():
                temporary = destination / (path.name + ".new")
                shutil.copy2(path, temporary)
                temporary.replace(destination / path.name)
        subprocess.run(
            [str(destination / "llama-server"), "--list-devices"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        marker = root / "ready.json.new"
        marker.write_text(
            json.dumps(dict(version=VERSION, toolkit=TOOLKIT, architectures=architectures)),
            encoding="utf-8",
        )
        marker.replace(root / "ready.json")
        yield "Build complete. Select CUDA (local build), save Settings, then load your chat model."
    finally:
        _build_lock.release()
