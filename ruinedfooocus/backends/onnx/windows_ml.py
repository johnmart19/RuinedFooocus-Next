"""Explicit Windows ML acquisition; inference stays in the existing ONNX worker."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import math
import tempfile

from ruinedfooocus.core.paths import data_path
from ruinedfooocus.packaging.install_lock import installation_lock
from ruinedfooocus.backends.onnx.qualification import (
    matching_record,
    evidence_valid,
    current_contract,
    fingerprint,
)

PROVIDERS = ("Auto", "Windows ML", "DirectML")
CATALOG_PACKAGES = (
    "wasdk-Microsoft.Windows.AI.MachineLearning==2.3.0",
    "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap==2.3.0",
    "onnxruntime-windowsml==1.25.2.202605110140",
    "winrt-runtime==3.2.1",
    "winrt-Windows.ApplicationModel==3.2.1",
    "winrt-Windows.Foundation==3.2.1",
    "winrt-Windows.Foundation.Collections==3.2.1",
)
# Catalogue eligibility only. Actual device discovery/assignment remains
# authoritative: a provider supporting a class does not certify local hardware.
PROVIDER_DEVICE_TYPES = {
    "AMDGPUExecutionProvider": ("GPU",),
    "MIGraphXExecutionProvider": ("GPU",),
    "NvTensorRtRtxExecutionProvider": ("GPU",),
    "OpenVINOExecutionProvider": ("CPU", "GPU", "NPU"),
    "QNNExecutionProvider": ("GPU", "NPU"),
    "VitisAIExecutionProvider": ("NPU",),
}
CATALOG_PROVIDERS = tuple(PROVIDER_DEVICE_TYPES)
DEVICE_TYPES = ("Auto", "CPU", "GPU", "NPU")


def provider_preference(values):
    # The old explicit Windows GPU choice requested isolated DirectML. Retain
    # that intent without rewriting saved settings or enabling DML for fresh Auto.
    return values.get(
        "rembg_provider", "DirectML" if values.get("rembg_backend") == "Windows GPU" else "Auto"
    )


def manifest_path():
    return data_path("cache", "onnx", "windows-ml", "provider.json")


def _catalog_rows(document):
    if (
        not isinstance(document, dict)
        or document.get("catalog_version") != "2.3.0"
        or not isinstance(document.get("providers"), list)
    ):
        raise ValueError("The local catalogue returned invalid metadata")
    rows = []
    for row in document["providers"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or type(row.get("certified")) is not bool
        ):
            raise ValueError("The local catalogue returned an invalid provider")
        name = row["name"]
        if name in CATALOG_PROVIDERS:
            rows.append(
                {
                    "name": name,
                    "certified": row["certified"],
                    "bridge_compatible": bridge_compatible(name, document["catalog_version"]),
                    "device_types": list(PROVIDER_DEVICE_TYPES[name]),
                }
            )
    return rows


def catalog_status():
    """Read the last explicit local catalogue inspection; never load a provider."""
    try:
        path = manifest_path().with_name("catalogue.json")
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Catalogue snapshot is oversized")
        document = json.loads(raw)
        if (
            document.get("system") != platform.system()
            or document.get("machine") != platform.machine()
            or document.get("python") != list(sys.version_info[:2])
        ):
            raise ValueError("Catalogue snapshot belongs to a different host/interpreter")
        rows = _catalog_rows(document)
        return {
            "providers": rows,
            "reason": "Cached local catalogue; Refresh updates offered providers. Acquisition is not model qualification.",
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {
            "providers": [],
            "reason": "Local Windows ML catalogue is unknown. Install catalogue support explicitly, then Refresh.",
        }


def refresh_catalog(*, check_cancelled=None, cpu=False):
    """Explicit metadata inspection through the existing isolated bridge; no acquisition."""
    if cpu:
        return {
            "providers": [],
            "reason": "CPU authority is active; no Windows ML catalogue or device process was started.",
        }
    if platform.system() != "Windows":
        raise RuntimeError("Windows ML catalogue inspection requires Windows")
    from ruinedfooocus.backends.onnx.acquisition import run
    from ruinedfooocus.backends.onnx.onnx_installer import worker_command
    from ruinedfooocus.core.config_io import save_json

    root = manifest_path().parent
    bridge = root / "catalog-2.3.0"
    if not bridge.is_dir():
        raise RuntimeError(
            "Install Windows ML catalogue/provider support explicitly before Refresh; Refresh never downloads it."
        )
    command = worker_command(bridge, worker=Path(__file__).with_name("winml_catalog.py"))
    command += ["--list-providers"]
    result = run(
        command, check_cancelled=check_cancelled, capture_output=True, text=True, timeout=45
    )
    document = json.loads(result.stdout.strip().splitlines()[-1])
    _catalog_rows(document)
    document.update(
        system=platform.system(), machine=platform.machine(), python=list(sys.version_info[:2])
    )
    if check_cancelled:
        check_cancelled()
    with installation_lock(root):
        save_json(root / "catalogue.json", document)
    return catalog_status()


@lru_cache(maxsize=8)
def _verified_library(filename, size, modified, checksum):
    with Path(filename).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == checksum


def bridge_compatible(name, catalog_version="2.3.0"):
    from packaging.version import Version

    return name != "AMDGPUExecutionProvider" or Version(catalog_version) >= Version("2.5.96")


def _record_rank(record):
    # Only actual workload evidence may outrank stable catalogue order. Acquired
    # records never become Auto-qualified merely because they appear first.
    evidence = record.get("workload_evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    qualified = evidence_valid(evidence, evidence.get("contract"))
    cost = evidence.get("warm_end_to_end_ms")
    if (
        not isinstance(cost, (float, int))
        or isinstance(cost, bool)
        or not math.isfinite(cost)
        or cost < 0
    ):
        cost = float("inf")
    return (
        not qualified,
        cost if qualified else float("inf"),
        CATALOG_PROVIDERS.index(record["name"]),
    )


def installed_providers(name="Auto", *, verify_payload=True):
    """Read acquisition records; runtime use verifies payloads, passive UI does not."""
    if platform.system() != "Windows":
        return []
    try:
        document = json.loads(manifest_path().read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return []
        records = document.get("providers", {document.get("name"): document})
        if not isinstance(records, dict):
            return []
        valid = []
        for key in CATALOG_PROVIDERS:
            if key not in records or name not in ("Auto", key):
                continue
            try:
                record = records[key]
                if not isinstance(record, dict):
                    continue
                library = Path(record["library_path"])
                if (
                    record["name"] != key
                    or not library.is_absolute()
                    or library.suffix.lower() != ".dll"
                    or not bridge_compatible(key, record.get("catalog_version", "2.3.0"))
                ):
                    continue
                stat = library.stat()
                if not verify_payload or _verified_library(
                    str(library), stat.st_size, stat.st_mtime_ns, record["sha256"]
                ):
                    valid.append(record)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(valid, key=_record_rank)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return []


def installed_provider(name="Auto"):
    return next(
        (
            record
            for record in installed_providers(name)
            if name != "Auto" or matching_record(record)
        ),
        None,
    )


def install_plan(target):
    if platform.system() != "Windows" or platform.machine().lower() not in (
        "amd64",
        "x86_64",
        "arm64",
        "aarch64",
    ):
        raise RuntimeError("Windows ML provider acquisition requires Windows x64 or ARM64.")
    if sys.getwindowsversion().build < 26100:
        raise RuntimeError("Windows ML provider acquisition requires Windows 11 24H2 or later.")
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError(
            "The qualified Windows ML Python bridge supports Python 3.11–3.13; use CPU or an independently qualified provider otherwise."
        )
    # The publisher's Python bridge pins an older ORT. Never put it in the app
    # environment; it is used only to acquire the certified provider package.
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--target",
        str(target),
        *CATALOG_PACKAGES,
    ]


def install_provider(name="Auto", device_type="Auto", *, check_cancelled=None):
    # Catalogue installation and the read/merge/publish of every provider record
    # share one target. A per-provider lock would still lose another vendor's row.
    if name == "Auto" and device_type == "CPU":
        return "The built-in CPU ONNX provider is already available; no vendor package or device probe is needed."
    with installation_lock(manifest_path().parent):
        return _install_provider(name, device_type, check_cancelled=check_cancelled)


def _install_provider(name="Auto", device_type="Auto", *, check_cancelled=None):
    from ruinedfooocus.packaging.startup_options import skip_dependency_install
    from ruinedfooocus.backends.onnx.onnx_installer import worker_command
    from ruinedfooocus.backends.onnx.acquisition import run

    if device_type not in DEVICE_TYPES:
        raise ValueError("Choose Auto, CPU, GPU or NPU.")
    if name != "Auto" and name not in CATALOG_PROVIDERS:
        raise ValueError("Choose a listed Windows ML execution provider.")
    if name != "Auto" and device_type != "Auto" and device_type not in PROVIDER_DEVICE_TYPES[name]:
        raise ValueError(f"The selected provider does not target {device_type} devices.")
    if device_type == "CPU" and name == "Auto":
        return "The built-in CPU ONNX provider is already available; no vendor package or device probe is needed."
    if not bridge_compatible(name):
        raise RuntimeError(
            "AMDGPU requires Windows ML 2.5.96+; the audited Python projection is 2.3.0. "
            "No compatible Python/App SDK/ORT closure has been qualified. Use DirectML or CPU."
        )
    if skip_dependency_install():
        raise RuntimeError("Windows ML installation is disabled by offline/skip-dependency policy.")
    if check_cancelled:
        check_cancelled()
    if any(
        device_type == "Auto" or device_type in PROVIDER_DEVICE_TYPES[record["name"]]
        for record in installed_providers(name)
    ):
        return "The certified Windows ML provider installation record is current; model qualification is unchanged."
    root = manifest_path().parent
    catalogue = root / "catalog-2.3.0"
    run(install_plan(catalogue), check_cancelled=check_cancelled)
    command = worker_command(catalogue, worker=Path(__file__).with_name("winml_catalog.py"))
    command += ["--ep-name", name, "--ep-type", device_type]
    try:
        result = run(
            command,
            check_cancelled=check_cancelled,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
    except subprocess.CalledProcessError as error:
        diagnostic = (error.stderr or error.stdout or "Provider acquisition failed.").strip()
        # The bridge's final exception line includes the local catalogue. Keep
        # its useful diagnostic rather than exposing only the command/exit code.
        raise RuntimeError(diagnostic.splitlines()[-1][:2000]) from error
    record = json.loads(result.stdout.strip().splitlines()[-1])
    if record.get("name") not in CATALOG_PROVIDERS:
        raise RuntimeError("No certified compatible Windows ML provider was acquired.")
    if check_cancelled:
        check_cancelled()
    _publish_provider(record)
    return "Certified Windows ML provider acquired. Select a named vendor explicitly; vendor Auto and compiled-model cache reuse require matching real-model qualification."


def _publish_provider(record):
    """Caller holds the shared catalogue/manifest installation lock."""
    root = manifest_path().parent
    root.mkdir(parents=True, exist_ok=True)
    try:
        document = json.loads(manifest_path().read_text(encoding="utf-8"))
        records = document.get(
            "providers", {document["name"]: document} if "name" in document else {}
        )
        if not isinstance(records, dict):
            records = {}
    except (OSError, ValueError, TypeError, AttributeError):
        records = {}
    records[record["name"]] = record
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix="provider-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump({"providers": records}, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, manifest_path())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_qualification(name, contract, evidence):
    """Coordinator publishes measured real-Bria acceptance, never a canary summary."""
    if name not in CATALOG_PROVIDERS:
        raise ValueError("Choose an explicit Windows ML vendor to qualify.")
    with installation_lock(manifest_path().parent):
        record = installed_provider(name)
        if not record or not isinstance(contract, dict):
            raise RuntimeError(
                "The provider installation/qualification contract is missing or stale."
            )
        current = current_contract(record, contract.get("device"), contract.get("device_type"))
        observed = dict(evidence, schema=1, contract=contract, cache_identity=fingerprint(contract))
        if current != contract or not evidence_valid(observed, current):
            raise RuntimeError(
                "Real Bria quality/performance/assignment evidence or its exact current contract is incomplete/stale."
            )
        _publish_provider(
            dict(
                record,
                qualification="real_model_qualified_exact_contract",
                workload_evidence=observed,
            )
        )


def hardware(device):
    return getattr(device, "hardware_device", None) or device.device


def device_identity(device):
    item = hardware(device)
    identity = {
        "ep": device.ep_name,
        "vendor": item.vendor_id,
        "id": item.device_id,
        "type": str(item.type),
        "metadata": dict(item.metadata),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def available_devices(ort, record, device_type="Auto"):
    if device_type not in DEVICE_TYPES:
        raise ValueError("Choose Auto, CPU, GPU or NPU.")
    return [
        device
        for device in ort.get_ep_devices()
        if device.ep_name == record["name"]
        and (
            device_type == "Auto"
            or hardware(device).type == getattr(ort.OrtHardwareDeviceType, device_type)
        )
    ]


def select_device(ort, record, identity="Auto", device_type="Auto"):
    devices = available_devices(ort, record, device_type)
    if identity != "Auto":
        devices = [device for device in devices if device_identity(device) == identity]
        if len(devices) != 1:
            raise RuntimeError(
                "The selected Windows ML device is missing or ambiguous; refresh devices."
            )
    if not devices:
        raise RuntimeError(
            "The installed Windows ML provider exposes no compatible " + device_type + " device."
        )

    def rank(device):
        kind = getattr(hardware(device).type, "name", str(hardware(device).type)).split(".")[-1]
        return ({"GPU": 0, "NPU": 1, "CPU": 2}.get(kind, 3), device_identity(device))

    return sorted(devices, key=rank)[0]


def register_provider(ort, record):
    # Keep process-local DLL handles alive for the entire session. No global PATH
    # edits, system SDK lookup or copying libraries into Torch/ORT folders.
    import ctypes

    if not callable(getattr(ort, "register_execution_provider_library", None)) or not callable(
        getattr(ort, "get_ep_devices", None)
    ):
        raise RuntimeError("The selected ORT does not expose the required plugin/device API.")
    if not bridge_compatible(record["name"], record.get("catalog_version", "2.3.0")):
        raise RuntimeError(
            "The Windows ML projection does not meet this provider's minimum version."
        )
    cpp = ctypes.WinDLL("msvcp140.dll", winmode=0x800)
    directory = os.add_dll_directory(str(Path(record["library_path"]).parent))
    try:
        ort.register_execution_provider_library(record["name"], record["library_path"])
    except BaseException:
        directory.close()
        raise
    return cpp, directory


def select_installed_device(ort, name="Auto", identity="Auto", device_type="Auto"):
    failures = []
    for record in installed_providers(name):
        if name == "Auto" and not matching_record(record):
            failures.append(record["name"] + ": no matching real-model qualification")
            continue
        handles = None
        try:
            handles = register_provider(ort, record)
            selected_identity = identity
            if name == "Auto" and identity == "Auto":
                selected_identity = record["workload_evidence"]["contract"]["device"]
            selected = select_device(ort, record, selected_identity, device_type)
            kind = getattr(hardware(selected).type, "name", str(hardware(selected).type)).split(
                "."
            )[-1]
            if name == "Auto" and not matching_record(
                record, device_id=device_identity(selected), device_type=kind
            ):
                raise RuntimeError(
                    "The observed provider device/runtime/model qualification is stale."
                )
            return record, selected, handles
        except (RuntimeError, ValueError, OSError, AttributeError) as error:
            failures.append(record["name"] + ": " + str(error))
            if handles:
                handles[1].close()
    raise RuntimeError("No installed Windows ML provider/device is usable. " + "; ".join(failures))


def discover_installed_devices(ort, name="Auto", device_type="Auto"):
    devices, handles, rejected = [], [], []
    for record in installed_providers(name):
        try:
            owned = register_provider(ort, record)
            # Keep all DLL directories alive while any discovered EP is loaded.
            handles.append(owned)
            devices.extend(available_devices(ort, record, device_type))
        except (RuntimeError, ValueError, OSError, AttributeError) as error:
            rejected.append(record["name"] + ": " + str(error))
    if not devices:
        raise RuntimeError(
            "No installed Windows ML provider exposes a compatible device. " + "; ".join(rejected)
        )
    return sorted(devices, key=device_identity), handles, rejected


def discover_devices(name="Auto", device_type="Auto", *, check_cancelled=None):
    """An explicit UI action; never called by status rendering or model lists."""
    import sysconfig
    from ruinedfooocus.backends.onnx.onnx_installer import worker_command

    if device_type == "CPU" and name == "Auto":
        return []
    if not installed_providers(name):
        raise RuntimeError("Install/verify a Windows ML provider first.")
    command = worker_command(sysconfig.get_path("purelib"))
    command += ["--provider", "Windows ML", "--manifest", str(manifest_path()), "--list-devices"]
    command += ["--ep-name", name, "--ep-type", device_type]
    from ruinedfooocus.backends.onnx.acquisition import run

    result = run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        check_cancelled=check_cancelled,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])["devices"]
