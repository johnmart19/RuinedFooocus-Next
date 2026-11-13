"""Automatic AMD wheel selection, separate from OS driver installation."""

import os
import platform
import re
import shutil
import subprocess
import sys
import json
import hashlib
import venv
from dataclasses import dataclass
from ruinedfooocus.packaging.runtime_bundles import bundle_for

# Targets present in the pinned wheel extras and AMD's release/development matrix.
# Artifact eligibility is not hardware/driver qualification of every platform.
_RADEON = frozenset(
    {
        "gfx1010",
        "gfx1011",
        "gfx1012",
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1033",
        "gfx1034",
        "gfx1035",
        "gfx1036",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
    }
)
SUPPORTED_TARGETS = {
    "Windows": _RADEON | {"gfx1200", "gfx1201"},
    "Linux": _RADEON | {"gfx908", "gfx90a", "gfx942", "gfx950", "gfx1200", "gfx1201"},
}
ROCM_PLATFORM = "rocm10.1"


@dataclass(frozen=True)
class RocmInstallPlan:
    platform: str
    args: list


ROCM_INDEX = bundle_for(ROCM_PLATFORM).index

# AMD deliberately publishes this small Python SDK selector only as an sdist.
# Native Torch and SDK payloads must still resolve to publisher wheels.
_ROCM_METADATA_HASHES = {
    "10.0.0": "65b5982249612a310135f5e061b9f28909e8ad718a1ddb3f1abba92298b7216b",
    "10.1.0": "e6616e62ebd1324031681e6e59349d46ee4f143a8d8f4f901a91c35c993041fc",
}


def rocm_metadata_layer(version):
    """Build AMD's hash-pinned Python selector before wheel-only resolution."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from ruinedfooocus.core.paths import data_path
    from ruinedfooocus.packaging.package_plan import PackageLayer

    try:
        source_hash = _ROCM_METADATA_HASHES[version]
    except KeyError as error:
        raise RuntimeError(f"ROCm SDK metadata source is not pinned for {version}") from error
    source = f"{ROCM_INDEX}rocm/rocm-{version}.tar.gz"
    directory = data_path("cache", "installer-tools", "rocm-metadata")
    directory.mkdir(parents=True, exist_ok=True)
    # Build in isolation from both installed packages and earlier device choices.
    with TemporaryDirectory(prefix="build-", dir=directory) as temporary:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--only-binary=:all:",
                "--no-binary=rocm",
                "--no-cache-dir",
                "--wheel-dir",
                temporary,
                source + "#sha256=" + source_hash,
            ],
            check=True,
        )
        wheels = list(Path(temporary).glob("*.whl"))
        if len(wheels) != 1 or wheels[0].name != f"rocm-{version}-py3-none-any.whl":
            raise RuntimeError("ROCm selector build did not produce the expected Python wheel")
        payload = wheels[0].read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        destination = directory / digest / wheels[0].name
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise RuntimeError("Cached ROCm selector differs from its content identity")
        else:
            with destination.open("xb") as output:
                output.write(payload)
    return PackageLayer(
        "amd-sdk-selector",
        (f"rocm @ {destination.as_uri()}",),
        source,
        source_hash,
        "MIT",
        digest,
    )


def parse_rocminfo_targets(output):
    # Match agent names, not ISA lists (which also contain generic gfx targets).
    return sorted(set(re.findall(r"^\s*Name:\s+(gfx[0-9a-f]+)\s*$", output, re.M)))


def detect_rocm_targets(os_platform):
    # AMD-only dependency: never install it from the shared bootstrap requirements.
    try:
        import rocm_bootstrap
    except ModuleNotFoundError:
        # Inventory is not authorization to mutate the running environment.
        # Linux/WSL can use an existing diagnostic; Windows needs bootstrap
        # acquired by the explicit package plan before this discovery route.
        targets = detect_rocminfo_targets() if os_platform == "Linux" else []
        if targets:
            return targets
        from ruinedfooocus.packaging.vendor_admission import linux_amd_adapters

        targets = (
            sorted({item["gfx"] for item in linux_amd_adapters()}) if os_platform == "Linux" else []
        )
        return targets or isolated_gfx_detection()
    from rocm_bootstrap import detect_gfx_targets

    targets = sorted({target.name for target in detect_gfx_targets()})
    if targets or os_platform != "Linux":
        return targets
    return detect_rocminfo_targets()


def isolated_gfx_detection():
    """Acquire the AMD-only metadata helper without changing the main RF base."""
    from ruinedfooocus.core.paths import data_path
    from ruinedfooocus.packaging.install_lock import installation_lock
    from ruinedfooocus.packaging.package_plan import (
        PackageLayer,
        current_target,
        execute_plan,
        plan_packages,
        resolve_packages,
        save_plan,
    )
    from ruinedfooocus.packaging.startup_options import skip_dependency_install

    directory = data_path(
        "cache",
        "installer-tools",
        "rocm-bootstrap-0.3.0-"
        + platform.system()
        + "-"
        + sys.implementation.cache_tag
        + "-"
        + platform.machine(),
    )
    python = directory / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    with installation_lock(directory.parent / (directory.name + ".prepare")):
        marker = directory / "rf-package-plan.json"
        try:
            from ruinedfooocus.packaging.package_plan import target_identity

            ready = python.exists() and json.loads(marker.read_text(encoding="utf-8"))["plan"][
                "target_json"
            ] == target_identity(current_target())
        except (OSError, ValueError, KeyError):
            ready = False
        if not ready:
            if skip_dependency_install():
                raise RuntimeError(
                    "AMD target detection helper is not cached; restart online or select CPU/Vulkan"
                )
            requirement = "rocm-bootstrap==0.3.0"
            layer = PackageLayer(
                "amd-detection-helper",
                (requirement,),
                "https://pypi.org/simple",
                "0.3.0",
                "MIT",
                hashlib.sha256(requirement.encode()).hexdigest(),
                "https://pypi.org/simple",
                ("https://files.pythonhosted.org/packages",),
            )
            artifacts = resolve_packages((layer,))
            plan = plan_packages(
                current_target(),
                {"eligible": True},
                (layer,),
                (),
                artifacts,
                environment=directory,
                recipe="amd-detection-helper",
                api_floor=None,
            )
            if not plan.ready:
                raise RuntimeError(
                    "AMD detection helper package closure is unavailable: "
                    + "; ".join(plan.conflicts)
                )
            if not python.exists():
                venv.EnvBuilder(with_pip=True).create(directory)
            execute_plan(plan, plan.digest, interpreter=python)
            save_plan(plan, marker)
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import json; from importlib.metadata import version; assert version('rocm-bootstrap')=='0.3.0'; from rocm_bootstrap import detect_gfx_targets; print(json.dumps(sorted({item.name for item in detect_gfx_targets()})))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        targets = json.loads(result.stdout)
        if not isinstance(targets, list) or any(
            not re.fullmatch(r"gfx[0-9a-f]+", target) for target in targets
        ):
            raise RuntimeError("AMD detection helper returned an invalid GFX identity")
        return targets


def detect_rocminfo_targets():
    # WSL exposes /dev/dxg, not the KFD topology used by rocm-bootstrap.
    rocminfo = shutil.which("rocminfo")
    if not rocminfo and "microsoft" in platform.release().lower():
        # Acquire the right guest wheels from host architecture evidence before
        # the guest has ROCm installed. This does not certify guest execution.
        rocminfo = shutil.which("clinfo.exe")
    if not rocminfo or os.environ.get("ROCM_BOOTSTRAP_DISABLE_DETECTION") == "1":
        return []
    try:
        result = subprocess.run(
            [rocminfo],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_rocminfo_targets(result.stdout) if result.returncode == 0 else []


def automatic_rocm_plan(os_platform, selected_platform, force=False):
    """Return a checked installation plan, or None for another backend/override."""
    override = os.environ.get("TORCH_PLATFORM")
    if os_platform not in SUPPORTED_TARGETS:
        return None
    target_platform = ROCM_PLATFORM
    if not force and override and override != target_platform:
        return None
    # Keep the existing choice on mixed AMD/NVIDIA or AMD/Intel machines.
    if (
        not force
        and not override
        and selected_platform not in ("cpu", "directml")
        and not selected_platform.startswith("rocm")
    ):
        return None
    targets = []
    if not force and not override and selected_platform in ("cpu", "directml"):
        from torchruntime.device_db import get_gpus

        if not any(gpu.vendor_id.lower() == "1002" for gpu in get_gpus()):
            # WSL can expose AMD through DXG without a PCI device entry.
            if os_platform == "Linux" and "microsoft" in platform.release().lower():
                targets = detect_rocminfo_targets()
            if not targets:
                return None
    explicit = force or override == target_platform
    targets = targets or detect_rocm_targets(os_platform)
    if not targets:
        if explicit:
            raise RuntimeError(
                "ROCm requested, but no AMD GFX target could be determined. "
                "Check the AMD driver and clinfo (Windows) or rocminfo (Linux/WSL)."
            )
        return None
    unsupported = set(targets) - SUPPORTED_TARGETS[os_platform]
    supported = sorted(set(targets) & SUPPORTED_TARGETS[os_platform])
    if unsupported and supported:
        print(f"Leaving unsupported AMD targets unselected: {', '.join(sorted(unsupported))}")
        targets = supported
    elif unsupported and not explicit:
        # Older Radeon GPUs retain the compatible installed runtime or CPU fallback.
        print(
            f"ROCm {target_platform} does not support {', '.join(sorted(unsupported))}; keeping {selected_platform}."
        )
        return None
    elif unsupported:
        raise RuntimeError(
            f"Automatic ROCm installation is not validated for {os_platform}: "
            f"{', '.join(sorted(unsupported))}. Install a compatible AMD bundle manually "
            "or explicitly select another TORCH_PLATFORM."
        )
    if platform.machine().lower() not in ("x86_64", "amd64"):
        if not explicit:
            return None
        raise RuntimeError("Automatic ROCm wheels require x86_64 Python.")
    extras = ",".join(f"device-{target}" for target in targets)
    bundle = bundle_for(target_platform)
    print(f"AMD GPU targets: {', '.join(targets)}")
    return RocmInstallPlan(
        target_platform,
        [
            f"torch[{extras}]=={bundle.torch}",
            # AMD's stable index labels the matching 2.14 vision build as alpha.
            # An unbounded install selects 0.28, whose native extension cannot load.
            f"torchvision[{extras}]=={bundle.vision}",
            "--index-url",
            ROCM_INDEX,
            "--only-binary=torch,torchvision",
        ],
    )


def install_rocm(args, reinstall=False):
    from ruinedfooocus.packaging.package_plan import PackagePlan, execute_plan

    if not isinstance(args, PackagePlan) or not args.recipe.startswith("rocm"):
        raise RuntimeError(
            "ROCm installation requires the frozen combined PackagePlan, not raw pip arguments"
        )
    execute_plan(args, args.digest, force=reinstall)
    # Do not continue with a CPU fallback if an incompatible driver/runtime loaded.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, torchvision; "
            "assert torch.version.hip and torch.cuda.is_available(), "
            "'ROCm installed but no GPU is available; check the AMD driver/WSL runtime'; "
            "weights=torch.ones(8192,768,dtype=torch.float16).to('cuda'); "
            "assert weights[0,0].item()==1; "
            "x=torch.ones(16,16,device='cuda'); y=x@x; torch.cuda.synchronize(); "
            "assert y[0,0].item()==16; "
            "boxes=torch.tensor([[0.,0.,1.,1.]],device='cuda'); "
            "assert torchvision.ops.nms(boxes,torch.ones(1,device='cuda'),0.5).item()==0; "
            "torch.cuda.synchronize(); print('ROCm verified:',torch.cuda.get_device_name(0))",
        ],
        check=True,
    )
