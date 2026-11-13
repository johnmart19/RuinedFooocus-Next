"""What each component can use on this PC, and what is installed for it.

One row per component option: ComfyUI (PyTorch), llama.cpp, xllamacpp, ONNX
Runtime and Windows ML. Rows are built from the GPU scan, the operating system
(Windows, WSL or Linux) and installed package metadata or saved install records.
Nothing here imports PyTorch, loads a model or probes a device.
"""

import platform

from ruinedfooocus.packaging import hardware_inventory as inventory

COMPONENTS = ("ComfyUI (PyTorch)", "llama.cpp", "xllamacpp", "ONNX Runtime", "Windows ML")

# Vendor each option needs; None means any machine (CPU) and "gpu" any GPU vendor.
IMAGE_VENDORS = {"cuda": "nvidia", "rocm": "amd", "xpu": "intel", "cpu": None, "directml": "gpu"}
IMAGE_LABELS = {
    "cuda": "CUDA (NVIDIA)",
    "rocm": "ROCm (AMD)",
    "xpu": "XPU (Intel)",
    "cpu": "CPU",
    "directml": "DirectML (deprecated)",
}
LLAMA_VENDORS = {"CPU": None, "Vulkan": "gpu", "ROCm": "amd", "SYCL": "intel"}
XLLAMA_VENDORS = {
    "CPU": None,
    "Vulkan": "gpu",
    "CUDA 12.8": "nvidia",
    "CUDA 13.2": "nvidia",
    "ROCm 7.2": "amd",
}
ONNX_VENDORS = {
    "CPU": None,
    "CUDA": "nvidia",
    "MIGraphX": "amd",
    "OpenVINO": "intel",
    "Windows GPU": "gpu",
}
ONNX_LABELS = {
    "CPU": "CPU",
    "CUDA": "CUDA (NVIDIA)",
    "MIGraphX": "MIGraphX (AMD)",
    "OpenVINO": "OpenVINO (Intel)",
    "Windows GPU": "DirectML (deprecated)",
}
# Windows ML vendor providers and the GPU vendor each one serves.
WINML_VENDORS = {
    "AMDGPUExecutionProvider": "amd",
    "MIGraphXExecutionProvider": "amd",
    "VitisAIExecutionProvider": "amd",
    "NvTensorRtRtxExecutionProvider": "nvidia",
    "OpenVINOExecutionProvider": "intel",
    "QNNExecutionProvider": "qualcomm",
}
# WSL exposes the GPU to Vulkan only through Mesa's Dozen (dzn) driver.
WSL_VULKAN_NOTE = "Needs Mesa's Dozen Vulkan driver in WSL"
VENDOR_NEEDS = {
    "nvidia": "Needs an NVIDIA GPU",
    "amd": "Needs an AMD GPU",
    "intel": "Needs an Intel GPU",
    "gpu": "Needs a GPU",
    "qualcomm": "Needs a Qualcomm Snapdragon PC",
}


def windows_ml_provider_names(providers, gpus, saved="Auto", machine=None):
    """Windows ML providers serving a detected GPU vendor; a saved one stays listed."""
    arm = (machine or platform.machine()).lower() in ("arm64", "aarch64")
    return [
        name
        for name in providers
        if (WINML_VENDORS.get(name) == "qualcomm" and arm)
        or inventory.has_vendor(gpus, WINML_VENDORS.get(name))
        or name == saved
    ]


def _row(component, option, status, note=""):
    return {"component": component, "option": option, "status": status, "note": note}


def _vendor_ok(gpus, vendor):
    if vendor is None:
        return True
    if vendor == "gpu":
        return bool(gpus)
    return inventory.has_vendor(gpus, vendor)


def _unavailable_note(gpus, vendor, os_name, needs_os=None):
    """Why an option is missing: the OS rule first, then the vendor rule."""
    if needs_os and os_name not in needs_os:
        return {
            ("Windows",): "Windows only",
            ("Linux", "WSL"): "Linux and WSL only",
            ("Linux",): "Native Linux only",
        }.get(tuple(needs_os), "Not for this operating system")
    if not _vendor_ok(gpus, vendor):
        return VENDOR_NEEDS.get(vendor, "Not supported on this PC")
    return "Not supported on this PC"


def image_rows(gpus, system, os_name, installed, machine):
    kind = installed.get("kind")
    choices = set(inventory.image_backend_choices(gpus, system, installed, machine))
    rows = []
    for value in ("cuda", "rocm", "xpu", "cpu", "directml"):
        label = IMAGE_LABELS[value]
        if value in choices:
            active = kind == value or (value == "directml" and installed.get("directml"))
            if value == "cpu" and not active:
                rows.append(
                    _row(COMPONENTS[0], label, "Available", "Any installed build runs on CPU")
                )
            else:
                rows.append(
                    _row(
                        COMPONENTS[0],
                        label,
                        "Installed" if active else "Available",
                        (installed.get("torch") or "") if active else "",
                    )
                )
        else:
            note = (
                "Legacy install only"
                if value == "directml" and os_name in ("Windows", "WSL")
                else _unavailable_note(
                    gpus,
                    IMAGE_VENDORS[value],
                    os_name,
                    ("Windows", "WSL") if value == "directml" else ("Windows", "Linux", "WSL"),
                )
            )
            rows.append(_row(COMPONENTS[0], label, "Not available", note))
    return rows


def llama_rows(gpus, system, os_name, machine):
    from ruinedfooocus.backends.llama import llama_installer

    try:
        official = [
            choice
            for choice in llama_installer.backend_choices(system, machine)
            if choice != "Auto"
        ]
    except Exception:
        official = []
    usable = set(inventory.llm_backend_choices(gpus, system, machine))
    try:
        cached = set(inventory.llama_installed(system, machine))
    except Exception:
        cached = set()
    rows = []
    for choice in official:
        vendor = "nvidia" if "CUDA" in choice else LLAMA_VENDORS.get(choice)
        if choice in usable:
            note = WSL_VULKAN_NOTE if choice == "Vulkan" and os_name == "WSL" else ""
            rows.append(
                _row(COMPONENTS[1], choice, "Installed" if choice in cached else "Available", note)
            )
        else:
            rows.append(
                _row(
                    COMPONENTS[1], choice, "Not available", _unavailable_note(gpus, vendor, os_name)
                )
            )
    if not official:
        rows.append(_row(COMPONENTS[1], "CPU", "Not available", "No official build for this CPU"))
    return rows


def xllama_rows(gpus, system, os_name, machine):
    usable = set(inventory.xllamacpp_backend_choices(gpus, system, machine))
    installed = inventory.xllamacpp_installed()
    rows = []
    for choice in inventory.XLLAMACPP_INDEXES:
        vendor = XLLAMA_VENDORS[choice]
        if choice in usable:
            current = installed and installed.get("backend") == choice
            rows.append(
                _row(
                    COMPONENTS[2],
                    choice,
                    "Installed" if current else "Available",
                    installed["version"]
                    if current
                    else WSL_VULKAN_NOTE
                    if choice == "Vulkan" and os_name == "WSL"
                    else "",
                )
            )
        elif (
            choice.startswith("ROCm")
            and os_name in ("Linux", "WSL")
            and _vendor_ok(gpus, "amd")
            and inventory.rocm10_torch()
        ):
            rows.append(
                _row(COMPONENTS[2], choice, "Not available", "Conflicts with ROCm 10 PyTorch")
            )
        else:
            needs = ("Linux", "WSL") if choice.startswith("ROCm") else None
            rows.append(
                _row(
                    COMPONENTS[2],
                    choice,
                    "Not available",
                    _unavailable_note(gpus, vendor, os_name, needs),
                )
            )
    return rows


def onnx_rows(gpus, system, os_name, machine):
    from ruinedfooocus.backends.onnx import onnx_installer

    usable = set(inventory.onnx_backend_choices(gpus, system, machine))
    try:
        import importlib.metadata

        cpu_version = importlib.metadata.version("onnxruntime")
    except Exception:
        cpu_version = None
    rows = []
    for value, label in ONNX_LABELS.items():
        if value not in usable:
            needs = {
                "Windows GPU": ("Windows",),
                "MIGraphX": ("Linux",),
            }.get(value)
            rows.append(
                _row(
                    COMPONENTS[3],
                    label,
                    "Not available",
                    _unavailable_note(gpus, ONNX_VENDORS[value], os_name, needs),
                )
            )
            continue
        if value == "CPU":
            installed, note = cpu_version is not None, cpu_version or ""
        elif value == "Windows GPU":
            installed, note = onnx_installer.runtime_available(), ""
        else:
            installed = onnx_installer.native_runtime_available(value)
            note = onnx_installer.NATIVE_RUNTIMES[value].version if installed else ""
        rows.append(_row(COMPONENTS[3], label, "Installed" if installed else "Available", note))
    return rows


def winml_rows(gpus, os_name, machine):
    from ruinedfooocus.backends.onnx.windows_ml import CATALOG_PROVIDERS, installed_provider

    arm = machine.lower() in ("arm64", "aarch64")
    rows = []
    for name in CATALOG_PROVIDERS:
        label = name.replace("ExecutionProvider", "")
        vendor = WINML_VENDORS.get(name)
        if os_name != "Windows":
            rows.append(_row(COMPONENTS[4], label, "Not available", "Windows only"))
        elif vendor == "qualcomm":
            rows.append(
                _row(
                    COMPONENTS[4],
                    label,
                    "Available" if arm else "Not available",
                    "" if arm else VENDOR_NEEDS["qualcomm"],
                )
            )
        elif not _vendor_ok(gpus, vendor):
            rows.append(_row(COMPONENTS[4], label, "Not available", VENDOR_NEEDS[vendor]))
        else:
            installed = installed_provider(name) is not None
            rows.append(_row(COMPONENTS[4], label, "Installed" if installed else "Available"))
    return rows


def capability_rows(decisions):
    """Render the same evidence consumed by installers and runtime admission."""
    components = {
        "image": COMPONENTS[0],
        "llama": COMPONENTS[1],
        "xllamacpp": COMPONENTS[2],
        "onnx": COMPONENTS[3],
        "windows_ml": COMPONENTS[4],
    }
    statuses = {
        "unavailable": "Not available",
        "unknown": "Unknown",
        "installable": "Installable",
        "installed": "Installed (unverified)",
        "usable": "Usable (workload unverified)",
        "qualified": "Qualified",
    }
    return [
        dict(
            _row(
                components.get(row["operation"], row["operation"]),
                f"{row['backend']} ({row['platform']})"
                if row.get("platform") and row["platform"] != row["backend"]
                else row["backend"],
                "Active" if row["active"] else statuses[row["state"]],
                "; ".join(row["reasons"]),
            ),
            adapter_id=row["adapter_id"],
            platform=row.get("platform"),
            recipe_id=row.get("recipe_id"),
            eligible=row["eligible"],
            usable=row["usable"],
            qualified=row["qualified"],
        )
        for row in decisions
    ]


def support_rows(
    gpus=None, system=None, machine=None, installed=None, release=None, *, decisions=None
):
    """All rows in display order. Each part fails soft so one broken probe hides nothing else."""
    if decisions is not None:
        return capability_rows(decisions)
    system = system or platform.system()
    machine = machine or platform.machine()
    gpus = inventory.detect_gpus() if gpus is None else gpus
    os_name = inventory.os_kind(system, release)
    installed = inventory.installed_runtimes(system) if installed is None else installed
    rows = []
    for build in (
        lambda: image_rows(gpus, system, os_name, installed, machine),
        lambda: llama_rows(gpus, system, os_name, machine),
        lambda: xllama_rows(gpus, system, os_name, machine),
        lambda: onnx_rows(gpus, system, os_name, machine),
        lambda: winml_rows(gpus, os_name, machine),
    ):
        try:
            rows += build()
        except Exception as error:  # Informational table: report, never break Settings.
            rows.append(_row("?", "?", "Unknown", str(error)))
    # Legacy callers supply a vendor catalogue, not resolved artifacts/device
    # evidence. Do not call those routes available or executable on that basis.
    for row in rows:
        if row["status"] in ("Available", "Installed"):
            row["status"] = (
                "Installed (unverified)" if row["status"] == "Installed" else "Candidate"
            )
            row["note"] = "; ".join(
                filter(
                    None, (row["note"], "Exact runtime/device eligibility has not been verified")
                )
            )
    return rows
