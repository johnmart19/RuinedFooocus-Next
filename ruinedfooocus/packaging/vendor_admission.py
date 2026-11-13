"""Acquisition evidence for pinned ROCm 10.1 and Torch 2.14 XPU recipes.

Windows driver-store versions are read without importing Torch or initializing
accelerators. Published matrices qualify acquisition, never a model workload.
Linux/WSL callers must supply their distinct driver/guest evidence explicitly.
Sources and access date are maintained in docs/runtime-and-troubleshooting.md.
"""

import json
import re
import subprocess
from pathlib import Path
import platform
import os
import shutil


def linux_amd_adapters(sysfs=Path("/sys")):
    """Passive KFD topology + physical PCI slot identity; no HIP/ROCm initialization."""
    adapters = []
    for properties in (sysfs / "class/kfd/kfd/topology/nodes").glob("*/properties"):
        try:
            fields = dict(line.split(maxsplit=1) for line in properties.read_text().splitlines())
            target = int(fields.get("gfx_target_version", "0"))
            if not target:
                continue
            render = sysfs / "class/drm" / ("renderD" + fields["drm_render_minor"]) / "device"
            if int((render / "vendor").read_text().strip(), 16) != 0x1002:
                continue
            uevent = dict(
                line.split("=", 1)
                for line in (render / "uevent").read_text().splitlines()
                if "=" in line
            )
            pci = uevent["PCI_SLOT_NAME"]
            if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci):
                continue
            major, remainder = divmod(target, 10000)
            minor, stepping = divmod(remainder, 100)
            architecture = f"gfx{major}{minor:x}{stepping:x}"
            try:
                module = (sysfs / "module/amdgpu/srcversion").read_text().strip()
            except OSError:
                module = "unknown-module"
            adapters.append(
                dict(
                    adapter_id="native:pci:" + pci.lower(),
                    hardware_id=pci.lower(),
                    identity_stable=True,
                    vendor="amd",
                    kind="gpu",
                    location="native",
                    gfx=architecture,
                    architecture=architecture,
                    driver_identity=platform.release() + ":" + module,
                    rocm10_1_driver_supported=None,
                )
            )
        except (OSError, ValueError, KeyError):
            continue
    return adapters


def linux_image_adapters(observed, *, sysfs=Path("/sys"), amd_targets=()):
    """Enrich physical native DRM/KFD records; preserve WSL host identities."""
    by_pci = {}
    for item in observed:
        by_pci.setdefault(item.get("pci_id"), []).append(item)
    result = []
    amd = {item["hardware_id"]: item for item in linux_amd_adapters(sysfs)}
    for device in (sysfs / "class/drm").glob("renderD*/device"):
        try:
            vendor = int((device / "vendor").read_text().strip(), 16)
            product = int((device / "device").read_text().strip(), 16)
            if vendor not in (0x1002, 0x8086):
                continue
            fields = dict(
                line.split("=", 1)
                for line in (device / "uevent").read_text().splitlines()
                if "=" in line
            )
            slot = fields["PCI_SLOT_NAME"].lower()
            if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", slot):
                continue
            pci_id = f"{vendor:04x}:{product:04x}"
            names = {item.get("name", "") for item in by_pci.get(pci_id, ())}
            record = dict(
                amd.get(slot, {}),
                adapter_id="native:pci:" + slot,
                hardware_id=slot,
                identity_stable=True,
                pci_id=pci_id,
                vendor="amd" if vendor == 0x1002 else "intel",
                kind="gpu",
                location="native",
                name=next(iter(names)) if len(names) == 1 else "",
                kernel_driver=fields.get("DRIVER"),
            )
            result.append(record)
        except (OSError, ValueError, KeyError):
            continue
    if result:
        return result
    # WSL supplies no KFD/DRM topology. A unique host adapter and unique GFX
    # query can share architecture evidence; guest visibility stays separate.
    for item in observed:
        record = dict(item)
        if (
            record.get("vendor") == "amd"
            and len([row for row in observed if row.get("vendor") == "amd"]) == 1
            and len(amd_targets) == 1
        ):
            record["gfx"] = amd_targets[0]
        result.append(record)
    return result


def windows_driver_inventory():
    script = (
        "$drivers = @(Get-CimInstance Win32_PnPSignedDriver -Filter \"DeviceClass='DISPLAY'\" | "
        "Select-Object DeviceID,DriverVersion); "
        "$os = Get-CimInstance Win32_OperatingSystem; "
        "@{build=$os.BuildNumber; drivers=$drivers} | ConvertTo-Json -Depth 4 -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        return json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def _version(value):
    return (
        tuple(int(part) for part in value.split("."))
        if re.fullmatch(r"\d+(?:\.\d+)+", value or "")
        else None
    )


def combined_evidence(values):
    values = tuple(values)
    return False if False in values else None if not values or None in values else True


def windows_recipe_evidence(recipe, adapters, inventory):
    """Return known/unknown compatibility per physical adapter, with reasons.

    A missing metadata row is unknown. A known old driver/OS or unsupported
    device is rejected. Matching a driver version alone never certifies a GPU.
    """
    results = []
    rows = inventory.get("drivers", ())
    if isinstance(rows, dict):
        rows = [rows]
    try:
        build = int(inventory.get("build", ""))
    except (TypeError, ValueError):
        build = None
    for adapter in adapters:
        name = adapter.get("name", "")
        pci = adapter.get("pci_id", "").lower().split(":")
        matches = [
            row
            for row in rows
            if len(pci) == 2 and f"ven_{pci[0]}&dev_{pci[1]}" in row.get("DeviceID", "").lower()
        ]
        hardware_id = adapter.get("hardware_id", "").lower()
        exact = [row for row in matches if row.get("DeviceID", "").lower() == hardware_id]
        matches = exact or matches
        driver = _version(matches[0].get("DriverVersion")) if len(matches) == 1 else None
        if recipe == "rocm10.1":
            # Released Radeon/Ryzen rows; TheRock build-only GFX extras are not
            # a substitute for this device-family contract.
            hardware = bool(
                re.search(
                    r"(?:RX\s+(?:7[6789]\d{2}|9\d{3})|Radeon\s+(?:PRO\s+W[67]\d{3}|AI\s+PRO\s+R9\d{3}|(?:8[024689]\d|7[468]0)M|80[456]\dS))",
                    name,
                    re.I,
                )
            )
            # Admit acquisition on newer drivers; native validation after install
            # still owns runtime compatibility, independently of this lower bound.
            driver_ok = (
                None if driver is None or driver[0] != 32 else driver >= (32, 0, 31041, 5005)
            )
            os_ok = None if build is None or build >= 28000 else build >= 26200
            source = "AMD ROCm 10.1 matrix / RN-RAD-ROCM-10-01"
        elif recipe == "xpu":
            hardware = bool(
                re.search(
                    r"(?:Arc.*(?:[AB]\d{3}|\d{3}[VTH])|Core.*Ultra|Meteor Lake|Arrow Lake|Lunar Lake|Panther Lake)",
                    name,
                    re.I,
                )
            )
            driver_ok = None if driver is None else driver >= (32, 0, 101, 8801)
            os_ok = None if build is None else build >= 22000
            source = "Intel PyTorch 2.14 prerequisites 2026-08-21"
        else:
            raise ValueError(f"No audited Windows acquisition matrix for {recipe}")
        reasons = []
        if not hardware:
            reasons.append("Device is outside the released recipe matrix")
        if os_ok is not True:
            reasons.append(
                "Windows build is unknown"
                if os_ok is None
                else "Windows build is outside the recipe matrix"
            )
        if driver_ok is not True:
            reasons.append(
                "Exact physical-adapter driver is unknown"
                if driver_ok is None
                else "Driver is outside the recipe matrix"
            )
        results.append(
            dict(
                adapter_id=adapter["adapter_id"],
                hardware_id=matches[0].get("DeviceID") if len(matches) == 1 else None,
                hardware_supported=hardware and os_ok,
                driver_supported=driver_ok,
                source=source,
                reasons=tuple(reasons),
            )
        )
    return tuple(results)


def select_installer_adapter(adapters, evidence):
    """Choose one compatible physical pair; never combine two devices' facts."""
    by_id = {item["adapter_id"]: item for item in adapters}
    ranked = sorted(
        evidence,
        key=lambda row: (
            not (row.get("hardware_supported") is True and row.get("driver_supported") is True),
            row.get("hardware_supported") is not True,
            row.get("hardware_supported") is False,
        ),
    )
    for row in ranked:
        if row["adapter_id"] not in by_id:
            continue
        selected = dict(by_id[row["adapter_id"]])
        if row.get("hardware_id"):
            selected.update(
                inventory_adapter_id=selected["adapter_id"],
                hardware_id=row["hardware_id"],
                identity_stable=True,
                adapter_id=selected.get("location", "native") + ":" + row["hardware_id"],
            )
        return selected, row.get("hardware_supported"), row.get("driver_supported")
    return None, None, None


def linux_platform_inventory(*, sysfs=Path("/sys"), dev=Path("/dev"), runner=None):
    """Passive OS/package/module metadata. Never initializes a GPU or installs."""
    runner = subprocess.run if runner is None else runner
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        os_release = {}
    packages = {}
    names = (
        "amdgpu-core",
        "amdgpu-dkms",
        "libze1",
        "intel-level-zero-gpu",
        "libze-intel-gpu1",
        "intel-opencl-icd",
    )
    try:
        result = runner(
            ["dpkg-query", "-W", "-f=${Package}\t${Version}\n", *names],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "\t" in line:
                name, version = line.split("\t", 1)
                packages[name.split(":")[0]] = version
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not packages and shutil.which("rpm"):
        try:
            result = runner(
                [
                    "rpm",
                    "-q",
                    "--qf",
                    "%{NAME}\t%{VERSION}\n",
                    "amdgpu-core",
                    "amdgpu-dkms",
                    "libze1",
                    "intel-level-zero-gpu",
                    "intel-opencl-icd",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            packages.update(
                line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    modules = {}
    for module in ("amdgpu", "xe", "i915"):
        try:
            loaded = (sysfs / "module" / module / "srcversion").read_text().strip()
        except OSError:
            continue
        try:
            result = runner(
                ["modinfo", "-F", "srcversion", module], capture_output=True, text=True, timeout=10
            )
            installed = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            installed = None
        modules[module] = {"loaded": loaded, "installed": installed}
    return dict(
        os_release=os_release,
        kernel=platform.release(),
        packages=packages,
        modules=modules,
        dxg=(dev / "dxg").exists(),
        kfd=(dev / "kfd").exists(),
        render_nodes=tuple(str(path) for path in (dev / "dri").glob("renderD*")),
    )


def guest_device_inventory(adapters, *, runner=None):
    """Explicit acquisition-time guest device query, separate from passive reads.

    Existing clinfo or rocminfo is used without installing tools. A single
    uniquely named host adapter can be bound to a guest listing; duplicate names
    remain unknown until a native API supplies a physical binding.
    """
    clinfo = shutil.which("clinfo")
    executable = clinfo or shutil.which("rocminfo")
    if not executable:
        return {item["adapter_id"]: None for item in adapters}
    runner = subprocess.run if runner is None else runner
    try:
        command = [executable, "-l"] if clinfo else [executable]
        result = runner(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return {item["adapter_id"]: None for item in adapters}
    if result.returncode:
        return {item["adapter_id"]: None for item in adapters}

    def normalized(value):
        value = re.sub(
            r"\((?:TM|R)\)|\b(?:AMD|Intel|Graphics|Corporation)\b", "", value, flags=re.I
        )
        return re.sub(r"[^a-z0-9]", "", value.lower())

    devices = [
        normalized(line.split(":", 1)[-1])
        for line in result.stdout.splitlines()
        if ("device" in line.lower() if clinfo else "marketing name:" in line.lower())
    ]
    names = [normalized(item.get("name", "")) for item in adapters]
    return {
        item["adapter_id"]: (
            True
            if name
            and len(name) > 6
            and names.count(name) == 1
            and sum(name in device for device in devices) == 1
            else None
        )
        for item, name in zip(adapters, names)
    }


def linux_recipe_evidence(
    recipe, adapters, inventory, *, wsl=False, host_inventory=None, guest_visible=None
):
    """Pinned Linux/WSL acquisition matrix; unknown fields remain unknown.

    Kernel/module/package evidence is distinct from active device visibility and
    real-model qualification. ROCm WSL is a technical preview requiring opt-in.
    """
    os_release = inventory.get("os_release", {})
    distro, release = os_release.get("ID"), os_release.get("VERSION_ID")
    pretty = os_release.get("PRETTY_NAME", "")
    kernel_match = re.match(r"(\d+)\.(\d+)", inventory.get("kernel", ""))
    kernel = tuple(map(int, kernel_match.groups())) if kernel_match else None
    packages, modules = inventory.get("packages", {}), inventory.get("modules", {})
    results = []
    for adapter in adapters:
        name = adapter.get("name", "")
        gfx = adapter.get("gfx") or adapter.get("architecture")
        reasons = []
        if recipe == "rocm10.1":
            released_gfx = {
                "gfx1030",
                "gfx1100",
                "gfx1101",
                "gfx1102",
                "gfx1103",
                "gfx1150",
                "gfx1151",
                "gfx1152",
                "gfx1153",
                "gfx1200",
                "gfx1201",
                "gfx908",
                "gfx90a",
                "gfx942",
                "gfx950",
            }
            hardware = None if gfx is None else gfx in released_gfx
            instinct = gfx in {"gfx908", "gfx90a", "gfx942", "gfx950"}
            apu = gfx in {"gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153"}
            if wsl:
                os_ok = (
                    None
                    if distro is None or release is None
                    else distro == "ubuntu"
                    and release in ("22.04", "24.04", "26.04")
                    or distro == "debian"
                    and release == "13"
                )
                host = windows_recipe_evidence(recipe, [adapter], host_inventory or {})[0]
                hardware = combined_evidence((hardware, host["hardware_supported"], os_ok))
                guest_visibility = (guest_visible or {}).get(adapter["adapter_id"])
                driver = combined_evidence(
                    (
                        host["driver_supported"],
                        inventory.get("dxg"),
                        guest_visibility is not False,
                    )
                )
                reasons.append("ROCm 10.1 WSL is a technical preview requiring explicit opt-in")
                if guest_visibility is None:
                    reasons.append(
                        "Guest execution must be verified after installing the ROCm wheels"
                    )
            else:
                ubuntu = {
                    "26.04": ((7, 0), "26.04.1"),
                    "24.04": ((6, 8) if instinct else (7, 0), "24.04.5"),
                    "22.04": ((5, 15) if instinct else (6, 8), "22.04.5"),
                }
                if distro == "ubuntu" and release in ubuntu:
                    required_kernel, point = ubuntu[release]
                    os_ok = (
                        None
                        if kernel is None or not pretty
                        else point in pretty and kernel == required_kernel
                    )
                    if apu and release == "22.04":
                        os_ok = False
                elif distro == "rhel" and release in ("10.2", "9.8") and not apu:
                    required = {"10.2": "6.12.0-211", "9.8": "5.14.0-687"}[release]
                    os_ok = inventory.get("kernel", "").startswith(required)
                elif distro == "debian" and instinct:
                    releases = {
                        "gfx950": {"13": (6, 12), "12": (6, 1)},
                        "gfx942": {"13": (6, 12)},
                        "gfx90a": {"12": (6, 1)},
                        "gfx908": {"12": (6, 1)},
                    }[gfx]
                    os_ok = (
                        None
                        if release not in releases or kernel is None
                        else kernel == releases[release]
                    )
                elif distro in ("sles", "sled") and release == "15.7" and instinct:
                    os_ok = inventory.get("kernel", "").startswith("6.4.0-150700.51")
                elif distro in ("rhel", "ol", "sles", "debian", "rocky"):
                    # Remaining rows require per-device kernel/firmware evidence;
                    # a neighboring released branch is not a compatible default.
                    os_ok = None
                else:
                    os_ok = None if distro is None or release is None else False
                module = modules.get("amdgpu", {})
                loaded = (
                    None
                    if not module.get("loaded") or not module.get("installed")
                    else module["loaded"] == module["installed"]
                )
                branch_match = re.match(
                    r"(?:\d+:)?(\d+\.\d+\.\d+)", packages.get("amdgpu-core", "")
                )
                branch = _version(branch_match[1]) if branch_match else None
                branch_ok = (
                    None
                    if branch is None or branch > (31, 60, 0)
                    else branch in ((31, 50, 0), (31, 60, 0))
                )
                driver = combined_evidence(
                    (
                        loaded,
                        inventory.get("kfd"),
                        True if apu else branch_ok,
                        True if apu or packages.get("amdgpu-dkms") else None,
                    )
                )
                hardware = combined_evidence((hardware, os_ok))
            source = "AMD ROCm 10.1 released matrix / WSL technical preview"
        elif recipe == "xpu":
            hardware = (
                bool(
                    re.search(
                        r"Arc.*(?:[AB]\d{3}|\d{3}[VTH])|Core.*Ultra|Meteor[ -]?Lake|Arrow[ -]?Lake|Lunar[ -]?Lake|Panther[ -]?Lake",
                        name,
                        re.I,
                    )
                )
                if name
                else None
            )
            os_ok = (
                None
                if distro is None or release is None
                else distro == "ubuntu" and release in ("24.04", "26.04")
            )
            if "panther" in name.lower() and not wsl and release == "24.04":
                os_ok = False
            user_runtime = (
                bool(packages.get("libze1"))
                and bool(packages.get("intel-level-zero-gpu") or packages.get("libze-intel-gpu1"))
                and bool(packages.get("intel-opencl-icd"))
            )
            if wsl:
                host = windows_recipe_evidence(recipe, [adapter], host_inventory or {})[0]
                hardware = combined_evidence((hardware, os_ok, host["hardware_supported"]))
                driver = combined_evidence(
                    (
                        host["driver_supported"],
                        inventory.get("dxg"),
                        user_runtime or None,
                        (guest_visible or {}).get(adapter["adapter_id"]),
                    )
                )
            else:
                row = modules.get(adapter.get("kernel_driver"), {})
                loaded = bool(row.get("loaded")) and row.get("loaded") == row.get("installed")
                driver = True if loaded and user_runtime and inventory.get("render_nodes") else None
                hardware = combined_evidence((hardware, os_ok))
            source = "Intel PyTorch 2.14 prerequisites 2026-08-21"
        else:
            raise ValueError(f"No audited Linux acquisition matrix for {recipe}")
        if hardware is not True:
            reasons.append("Released OS/device combination is unknown or unsupported")
        if driver is not True:
            reasons.append(
                "Loaded driver or guest visibility evidence is incomplete or incompatible"
            )
        results.append(
            dict(
                adapter_id=adapter["adapter_id"],
                hardware_supported=hardware,
                driver_supported=driver,
                source=source,
                reasons=tuple(reasons),
                opt_in=wsl and recipe == "rocm10.1",
            )
        )
    return tuple(results)
