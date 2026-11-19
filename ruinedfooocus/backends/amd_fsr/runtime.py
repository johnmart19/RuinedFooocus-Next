"""Acquire the AMD FSR 2, 3 and 4 still-image runtime on first use.

The runtime is three files beside each other: AMD's two signed DirectX 12 DLLs and
the small helper compiled from `fsr_still.cpp` in this package. The DLLs, the
loader import library and four headers come from AMD's FidelityFX SDK repository
at one pinned commit, each verified by size and SHA256. The helper is compiled
once with the Visual Studio C++ build tools already on the machine; nothing is
installed, and no binary is taken from anywhere but AMD's repository.

AMD's licence requires its notice to accompany the signed binaries, so the SDK's
licence file is placed with them.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

REPOSITORY = "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK"
SDK_VERSION = "2.4.0"
SDK_COMMIT = "7539ae8b6568ad29ed78a753404351daf3b5cca7"
RAW = "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/" + SDK_COMMIT
# Path in the SDK: (bytes, SHA256). Checked against the published tag v2.4.0.
SDK_FILES = {
    "Kits/FidelityFX/api/include/ffx_api.h": (
        10848,
        "4d9dcf243ef5efbf03340a02f16b51a22053243a22e8a9a0b3ce91c26982b4bf",
    ),
    "Kits/FidelityFX/api/include/ffx_api_types.h": (
        13637,
        "53f07c113a9ba493f297656e4481b6539aed0afc9f3cc8da682aa477d270c46c",
    ),
    "Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h": (
        11323,
        "c9afbc3c4c673723e9e5369a666f0ff54560543b5a93537f8ff453bf1c3bf530",
    ),
    "Kits/FidelityFX/upscalers/include/ffx_upscale.h": (
        19576,
        "13091e7fe5fd9c82b1fb7c044b9f08f664d9653f599d2ea500738709236fca4d",
    ),
    "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.lib": (
        2880,
        "384123f21f43f263047250ac28d1b0e9a7ed4ea5b37391af2e89fdc3f964d50c",
    ),
    "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll": (
        39184,
        "e2191d3c0828be45eb8affd4e3ec0378f2293adc345b772a7170edcf3b021ecb",
    ),
    "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll": (
        57664272,
        "aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace",
    ),
    "docs/license.md": (
        65681,
        "410f9c138e766e846acf2dc93db768198def0293a017d7e3296815b451688dbe",
    ),
}
DLLS = ("amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll")
HELPER = "fsr_still.exe"
LICENCE = "AMD-FidelityFX-SDK-license.md"
SOURCE = Path(__file__).resolve().with_name("fsr_still.cpp")
BUILD_TIMEOUT_SECONDS = 300
_lock = threading.Lock()


def offline():
    from ruinedfooocus.app.arguments import args

    return bool(getattr(args, "offline", False)) or any(
        os.environ.get(name, "").upper() in ("1", "TRUE", "YES", "ON")
        for name in ("RF_OFFLINE", "HF_HUB_OFFLINE")
    )


def compiler():
    """Path of `vcvars64.bat` for the installed C++ x64 build tools, or None."""
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        return None
    try:
        found = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    script = Path(found) / "VC/Auxiliary/Build/vcvars64.bat" if found else None
    return script if script is not None and script.is_file() else None


def acquisition_blocker():
    """Why the runtime cannot be fetched and built here, or None when it can."""
    if os.name != "nt":
        return "AMD FSR 2, 3 and 4 need Windows and DirectX 12."
    if offline():
        return "Offline mode: the AMD FSR runtime cannot be downloaded."
    if compiler() is None:
        return (
            "Building the AMD FSR helper needs the Visual Studio C++ build tools "
            "(Desktop development with C++). Install them, or choose FSR 1."
        )
    return None


def sdk_directory():
    from ruinedfooocus.core.cache_contract import configured_cache_root

    return configured_cache_root() / "amd-fsr-sdk" / SDK_COMMIT


def fetch_sdk(directory=None, *, check_cancelled=None, service=None):
    """Download the pinned SDK files that are not already present and verified."""
    from ruinedfooocus.core.preparation import PreparationRequest, preparation_service

    directory = Path(directory) if directory is not None else sdk_directory()
    service = preparation_service if service is None else service
    # One file at a time: the service's queue is small and shared with model downloads.
    for relative, (size, digest) in SDK_FILES.items():
        if check_cancelled is not None:
            check_cancelled()
        # An existing destination is verified against the pin by the service.
        subscription = service.submit(
            PreparationRequest(
                RAW + "/" + relative, directory / relative, expected_size=size, sha256=digest
            )
        )
        while True:
            if check_cancelled is not None:
                check_cancelled()
            try:
                subscription.result(timeout=0.2)
                break
            except TimeoutError:
                if subscription.done():
                    raise
    return directory


def build(sdk, output, *, source=SOURCE, script=None):
    """Compile the helper against `sdk` and place the runtime in `output`."""
    kit = Path(sdk) / "Kits/FidelityFX"
    library = kit / "signedbin/amd_fidelityfx_loader_dx12.lib"
    needed = [library, *(kit / "signedbin" / name for name in DLLS), kit / "api/include/ffx_api.h"]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise RuntimeError("This is not a complete AMD FSR SDK. Missing: " + ", ".join(missing))
    script = compiler() if script is None else script
    if script is None:
        raise RuntimeError("The Visual Studio C++ x64 build tools were not found.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rf-amd-fsr-") as work:
        batch = Path(work) / "build.bat"
        batch.write_text(
            "@echo off\n"
            f'call "{script}" >nul\n'
            "cl /nologo /std:c++17 /O2 /EHsc /W3"
            f' /I"{kit / "api/include"}" /I"{kit / "upscalers/include"}"'
            f' "{source}" /Fo"{work}\\\\" /Fe:"{Path(work) / HELPER}"'
            f' /link "{library}" d3d12.lib dxgi.lib dxguid.lib\n',
            encoding="utf-8",
        )
        try:
            result = subprocess.run(
                ["cmd", "/c", str(batch)],
                cwd=work,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=BUILD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("The AMD FSR helper did not compile in time.") from None
        built = Path(work) / HELPER
        if result.returncode != 0 or not built.is_file():
            detail = " ".join((result.stdout + " " + result.stderr).split())[-300:]
            raise RuntimeError("The AMD FSR helper did not compile. " + detail)
        # Publish the helper last: its presence with both DLLs marks a usable runtime.
        for name in DLLS:
            shutil.copy2(kit / "signedbin" / name, output / name)
        licence = Path(sdk) / "docs/license.md"
        if licence.is_file():
            shutil.copy2(licence, output / LICENCE)
        shutil.copy2(built, output / HELPER)
    return output


def ensure(output, *, check_cancelled=None, ready=None):
    """Make the runtime present in `output`, downloading and building it once.

    `ready()` returns True when the runtime is already usable. Raises ValueError
    with the reason when it cannot be acquired here.
    """
    with _lock:
        if ready is not None and ready():
            return Path(output)
        blocker = acquisition_blocker()
        if blocker is not None:
            raise ValueError(blocker)
        try:
            sdk = fetch_sdk(check_cancelled=check_cancelled)
            if check_cancelled is not None:
                check_cancelled()
            return build(sdk, output)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                "The AMD FSR runtime could not be prepared: "
                + " ".join(str(error).split())[:300]
                + " FSR 1 needs no runtime."
            ) from error
