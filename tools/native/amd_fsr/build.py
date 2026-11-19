"""Build the AMD FSR still-image helper and place it with AMD's runtime DLLs.

Windows only. Needs the Visual Studio C++ build tools and an unpacked AMD FSR SDK
(2.4.0 was used). It compiles `fsr_still.cpp` against the SDK headers and the
signed loader import library, then copies the helper and the two signed DirectX 12
DLLs into the output folder the application looks in. Nothing is downloaded.

    python tools/native/amd_fsr/build.py --sdk D:/FidelityFX-SDK-2.4.0 [--output DIR]

AMD's licence for the signed binaries requires its copyright and permission
notice to accompany redistributions; see `docs/license.md` in the SDK.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DLLS = ("amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll")
HELPER = "fsr_still.exe"


def vcvars():
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise SystemExit("Visual Studio Installer (vswhere.exe) was not found.")
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
        check=False,
    ).stdout.strip()
    script = Path(found) / "VC/Auxiliary/Build/vcvars64.bat" if found else None
    if script is None or not script.is_file():
        raise SystemExit("The Visual Studio C++ x64 build tools were not found.")
    return script


def default_output():
    sys.path.insert(0, str(ROOT))
    from ruinedfooocus.features.image_processing.amd_fsr4 import runtime_directory

    return runtime_directory()


def build(sdk, output):
    kit = Path(sdk) / "Kits/FidelityFX"
    library = kit / "signedbin/amd_fidelityfx_loader_dx12.lib"
    needed = [library, *(kit / "signedbin" / name for name in DLLS), kit / "api/include/ffx_api.h"]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise SystemExit("This is not a complete AMD FSR SDK. Missing: " + ", ".join(missing))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rf-amd-fsr-") as work:
        script = Path(work) / "build.bat"
        script.write_text(
            "@echo off\n"
            f'call "{vcvars()}" >nul\n'
            "cl /nologo /std:c++17 /O2 /EHsc /W3"
            f' /I"{kit / "api/include"}" /I"{kit / "upscalers/include"}"'
            f' "{HERE / "fsr_still.cpp"}" /Fo"{work}\\\\" /Fe:"{Path(work) / HELPER}"'
            f' /link "{library}" d3d12.lib dxgi.lib dxguid.lib\n',
            encoding="utf-8",
        )
        result = subprocess.run(["cmd", "/c", str(script)], cwd=work, check=False)
        built = Path(work) / HELPER
        if result.returncode != 0 or not built.is_file():
            raise SystemExit("The helper did not compile.")
        shutil.copy2(built, output / HELPER)
    for name in DLLS:
        shutil.copy2(kit / "signedbin" / name, output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdk", required=True, help="Folder of the unpacked AMD FSR SDK.")
    parser.add_argument("--output", help="Runtime folder; default is the application's.")
    options = parser.parse_args()
    if os.name != "nt":
        raise SystemExit("The AMD FSR runtime needs Windows and DirectX 12.")
    output = build(options.sdk, options.output or default_output())
    print(f"AMD FSR runtime ready in {output}")


if __name__ == "__main__":
    main()
