"""Isolated GPU mask worker; never imported into the Torch process."""

import argparse
import base64
import json
from pathlib import Path
import sys
import time

# Vendor builds installed by onnx_installer.install_native_runtime.
NATIVE_PROVIDERS = {
    "CUDA": "CUDAExecutionProvider",
    "MIGraphX": "MIGraphXExecutionProvider",
    "OpenVINO": "OpenVINOExecutionProvider",
}
OPENVINO_DEVICES = {"Auto": "GPU", "CPU": "CPU", "GPU": "GPU", "NPU": "NPU"}
CANARY = "CAk6RgoOCgF4CgF4EgF5IgNBZGQSCmdwdS1jYW5hcnlaEwoBeBIOCgwIARIICgIIAgoCCAJiEwoBeRIOCgwIARIICgIIAgoCCAJCBAoAEBE="


def initialization_progress(phase, started):
    print(
        json.dumps(
            {
                "event": "initialization_progress",
                "phase": phase,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        ),
        flush=True,
    )


def run(
    runtime,
    probe=False,
    device_id=None,
    site_dir=None,
    provider="DirectML",
    manifest=None,
    ep_device="Auto",
    list_devices=False,
    ep_name="Auto",
    ep_type="Auto",
    profile_root=None,
):
    started = time.monotonic()

    def progress(phase):
        if not probe and not list_devices:
            initialization_progress(phase, started)

    progress("loading runtime")
    if site_dir:
        import site

        site.addsitedir(site_dir)
        sys.path.remove(site_dir)
        sys.path.insert(0, site_dir)
    sys.path.insert(0, str(Path(runtime).resolve()))
    if provider == "CUDA":
        # Load PyTorch's CUDA libraries first when it has them (same CUDA major).
        try:
            import torch  # noqa: F401
        except Exception:
            pass
    import onnxruntime as ort

    if provider == "CUDA" and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    if provider == "OpenVINO" and sys.platform == "win32":
        try:
            import onnxruntime.tools.add_openvino_win_libs as openvino_libraries

            openvino_libraries.add_openvino_libs_to_path()
        except Exception:
            pass
    import numpy as np

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    plugin_libraries = None
    progress("selecting provider and device")
    if provider == "Windows ML":
        from ruinedfooocus.backends.onnx.windows_ml import (
            installed_provider,
            installed_providers,
            device_identity,
            hardware,
            manifest_path,
            select_installed_device,
            discover_installed_devices,
        )

        record = (
            next(iter(installed_providers(ep_name)), None)
            if list_devices
            else installed_provider(ep_name)
        )
        if (
            record is None
            or manifest is None
            or Path(manifest).resolve() != manifest_path().resolve()
        ):
            raise RuntimeError(
                "The Windows ML installation record is missing or changed; install/verify it explicitly."
            )
        if list_devices:
            devices, handles, rejected = discover_installed_devices(ort, ep_name, ep_type)
            print(
                json.dumps(
                    {
                        "devices": [
                            {
                                "id": device_identity(d),
                                "label": str(hardware(d).type)
                                + " · "
                                + d.ep_name
                                + " · "
                                + str(dict(hardware(d).metadata)),
                            }
                            for d in devices
                        ],
                        "rejected": rejected,
                    }
                ),
                flush=True,
            )
            return
        record, selected, handles = select_installed_device(ort, ep_name, ep_device, ep_type)
        options.add_provider_for_devices([selected], {})
        # Factories in SessionOptions own the exact EP/device; an empty provider
        # list prevents rembg's legacy provider detection from replacing them.
        providers = []
        expected_provider = selected.ep_name
        active_type = getattr(hardware(selected).type, "name", str(hardware(selected).type))
    elif provider == "MIGraphX" and not Path(runtime).name.startswith("onnxruntime-migraphx-"):
        from ruinedfooocus.backends.onnx.plugins import register_migraphx
        from ruinedfooocus.backends.onnx.windows_ml import device_identity

        marker = Path(runtime) / "verified.json"
        installed = json.loads(marker.read_text()) if marker.is_file() else {}
        expected = installed.get("plugin_libraries")
        if ep_device == "Auto":
            ep_device = installed.get("provider_device_id", "Auto")
        devices, plugin_libraries = register_migraphx(ort, runtime, expected)
        if list_devices:
            print(
                json.dumps(
                    {
                        "devices": [
                            {
                                "id": device_identity(device),
                                "ep": device.ep_name,
                                "metadata": dict(
                                    (
                                        getattr(device, "hardware_device", None) or device.device
                                    ).metadata
                                ),
                            }
                            for device in devices
                        ]
                    }
                ),
                flush=True,
            )
            return
        if not callable(getattr(options, "add_provider_for_devices", None)):
            raise RuntimeError("This ORT lacks SessionOptions.add_provider_for_devices.")
        matches = [
            device
            for device in devices
            if ep_device == "Auto" or device_identity(device) == ep_device
        ]
        if not matches or (ep_device != "Auto" and len(matches) != 1):
            raise RuntimeError("The selected MIGraphX device is missing or ambiguous.")
        selected = sorted(matches, key=device_identity)[0]
        options.add_provider_for_devices([selected], {})
        providers = []
        expected_provider = selected.ep_name
        active_type = "GPU"
    elif provider in NATIVE_PROVIDERS:
        expected_provider = NATIVE_PROVIDERS[provider]
        if expected_provider not in ort.get_available_providers():
            raise RuntimeError(f"The isolated ONNX runtime has no {expected_provider}.")
        provider_options = {}
        active_type = "GPU"
        if provider == "OpenVINO":
            active_type = OPENVINO_DEVICES.get(ep_type, "GPU")
            provider_options = {"device_type": active_type}
        elif provider == "CUDA" and device_id is not None:
            provider_options = {"device_id": str(device_id)}
        providers = [(expected_provider, provider_options), "CPUExecutionProvider"]
    else:
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("The isolated ONNX runtime has no DirectML provider.")
        provider_options = (
            {"device_id": str(device_id)}
            if device_id is not None
            else {"performance_preference": "high_performance", "device_filter": "gpu"}
        )
        providers = [("DmlExecutionProvider", provider_options), "CPUExecutionProvider"]
        expected_provider = "DmlExecutionProvider"
        active_type = "GPU"
    if probe:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            base64.b64decode(CANARY),
            sess_options=options,
            providers=providers[:1],
            enable_fallback=False,
        )
        session.disable_fallback()
        if expected_provider not in session.get_providers():
            raise RuntimeError("The requested GPU provider was not assigned.")
        x = np.ones((2, 2), dtype=np.float32)
        np.testing.assert_array_equal(session.run(None, {"x": x})[0], x * 2)
        print(
            json.dumps(
                {
                    "ok": True,
                    "version": ort.__version__,
                    "plugin_libraries": plugin_libraries,
                    "device_identity": device_identity(selected)
                    if plugin_libraries is not None
                    else None,
                }
            ),
            flush=True,
        )
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ruinedfooocus.backends.onnx.model_contracts import (
        create_audited_mask_session,
        model_identity,
    )
    from ruinedfooocus.backends.onnx.execution import finish_assignment, cache_identity
    from PIL import Image

    # The parent uses its own rembg version's normal postprocessing. Only mask
    # inference occurs here, with the same inherited package/model default.
    progress("creating native model session")
    session = create_audited_mask_session(
        expected_provider, sess_opts=options, providers=providers, profile_root=profile_root
    )
    progress("checking model identity and runtime contract")
    model_evidence = model_identity(session)
    qualification_contract = None
    if provider == "Windows ML":
        from ruinedfooocus.backends.onnx.qualification import (
            current_contract,
            runtime_identity,
            evidence_valid,
        )

        qualification_contract = current_contract(
            record,
            device_identity(selected),
            active_type,
            model=model_evidence,
            runtime=runtime_identity(ort),
        )
        if ep_name == "Auto" and not evidence_valid(
            record.get("workload_evidence"), qualification_contract
        ):
            raise RuntimeError(
                "The real-model qualification contract changed during Windows ML initialization."
            )
    if expected_provider not in session.inner_session.get_providers():
        raise RuntimeError(
            "The requested GPU provider did not initialize; refusing silent CPU inference."
        )
    progress("publishing readiness")
    print(
        json.dumps(
            {
                "ready": True,
                "model": session.model_name,
                "provider": expected_provider,
                "device_type": active_type,
                "ort": ort.__version__,
                "model_identity": model_evidence,
                "initialization_seconds": time.monotonic() - started,
                "phase": "initialized_not_executed",
                "graph_assignment": session._rf_graph_assignment,
                "qualification_contract": qualification_contract,
                "cache_identity": cache_identity(
                    model_sha256=model_evidence["sha256"],
                    preprocessing=model_evidence.get("preprocessing"),
                    shape=[1, 3, 1024, 1024] if session.model_name == "bria-rmbg" else None,
                    dtype="float32" if session.model_name == "bria-rmbg" else None,
                    provider=expected_provider,
                    runtime=ort.__version__,
                    device=device_identity(selected)
                    if provider == "Windows ML" or plugin_libraries is not None
                    else None,
                    driver=None,
                    options=session.inner_session.get_provider_options(),
                ),
                "compiled_model_cache_reuse": False,
                "device_identity": device_identity(selected)
                if provider == "Windows ML" or plugin_libraries is not None
                else None,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            return
        try:
            request_started = time.monotonic()
            with Image.open(request["input"]) as image:
                masks = session.predict(image.convert("RGB"))
            evidence = finish_assignment(session, expected_provider)
            paths = []
            for i, mask in enumerate(masks):
                path = Path(request["output"]) / f"mask-{i}.png"
                mask.save(path)
                paths.append(str(path))
            print(
                json.dumps(
                    {
                        "masks": paths,
                        "assignment": evidence,
                        "inference_seconds": time.monotonic() - request_started,
                    }
                ),
                flush=True,
            )
        except Exception as error:
            print(
                json.dumps({"error": f"{type(error).__name__}: {error}", "phase": "inference"}),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--site-dir")
    parser.add_argument(
        "--provider", choices=["DirectML", "Windows ML", *NATIVE_PROVIDERS], default="DirectML"
    )
    parser.add_argument("--manifest")
    parser.add_argument("--ep-device", default="Auto")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--ep-name", default="Auto")
    parser.add_argument("--ep-type", choices=["Auto", "CPU", "GPU", "NPU"], default="Auto")
    parser.add_argument("--profile-root")
    args = parser.parse_args()
    try:
        run(
            args.runtime,
            args.probe,
            args.device_id,
            args.site_dir,
            args.provider,
            args.manifest,
            args.ep_device,
            args.list_devices,
            args.ep_name,
            args.ep_type,
            args.profile_root,
        )
    except Exception as error:
        import traceback

        traceback.print_exc(file=sys.stderr)
        # Initialization errors must reach the parent's visible CPU recovery
        # status too, not just the worker log or a generic child-exited message.
        print(
            json.dumps({"error": f"{type(error).__name__}: {error}", "phase": "initialization"}),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
