"""ORT plugin registration is worker-only and never edits global vendor paths."""

import hashlib
import importlib
from pathlib import Path


def register_migraphx(ort, root, expected_libraries=None):
    if not callable(getattr(ort, "register_execution_provider_library", None)) or not callable(
        getattr(ort, "get_ep_devices", None)
    ):
        raise RuntimeError("This ORT lacks the plugin registration/device API.")
    plugin = importlib.import_module("onnxruntime_ep_migraphx")
    if not all(
        callable(getattr(plugin, name, None)) for name in ("get_ep_names", "get_library_paths")
    ):
        raise RuntimeError(
            "The MIGraphX package has no supported native plugin API (possibly a placeholder)."
        )
    names, libraries = list(plugin.get_ep_names()), list(plugin.get_library_paths())
    if not names or len(names) != len(libraries) or len(set(names)) != len(names):
        raise RuntimeError("The MIGraphX plugin returned invalid aliases/library pairs.")
    records = []
    root = Path(root).resolve()
    for name, filename in zip(names, libraries):
        library = Path(filename)
        if (
            name != "MIGraphXExecutionProvider"
            or not library.is_absolute()
            or library.suffix != ".so"
        ):
            raise RuntimeError("The MIGraphX plugin alias/native library is invalid.")
        library = library.resolve()
        if not library.is_relative_to(root) or not library.is_file():
            raise RuntimeError(
                "The MIGraphX native payload is missing/outside its isolated environment."
            )
        checksum = hashlib.sha256(library.read_bytes()).hexdigest()
        record = {"name": name, "path": str(library.relative_to(root)), "sha256": checksum}
        if expected_libraries is not None and record not in expected_libraries:
            raise RuntimeError("The MIGraphX native library differs from its installation record.")
        records.append(record)
    if expected_libraries is not None and records != expected_libraries:
        raise RuntimeError("The MIGraphX native library closure changed.")
    for record in records:
        ort.register_execution_provider_library(record["name"], str(root / record["path"]))
    devices = [
        device
        for device in ort.get_ep_devices()
        if device.ep_name == "MIGraphXExecutionProvider"
        and (getattr(device, "hardware_device", None) or device.device).vendor_id == 0x1002
    ]
    if not devices:
        raise RuntimeError("The registered MIGraphX plugin exposes no AMD device.")
    return devices, records
