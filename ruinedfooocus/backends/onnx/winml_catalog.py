"""Acquire a certified vendor EP through the isolated publisher Python bridge."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def select_provider(providers, certification, name="Auto", device_type="Auto"):
    from ruinedfooocus.backends.onnx.windows_ml import (
        CATALOG_PROVIDERS,
        PROVIDER_DEVICE_TYPES,
        bridge_compatible,
    )

    providers = list(providers)
    candidates = [
        provider
        for provider in providers
        if provider.name in CATALOG_PROVIDERS
        and name in ("Auto", provider.name)
        and provider.certification == certification
        and bridge_compatible(provider.name)
        and (device_type == "Auto" or device_type in PROVIDER_DEVICE_TYPES[provider.name])
    ]
    # Prefer a provider dedicated to the requested class, e.g. VitisAI for an
    # AMD NPU over a multi-device provider also offered on the same machine.
    candidates.sort(
        key=lambda provider: (
            len(PROVIDER_DEVICE_TYPES[provider.name]) if device_type != "Auto" else 0,
            CATALOG_PROVIDERS.index(provider.name),
        )
    )
    if not candidates:
        offered = (
            ", ".join(
                f"{provider.name.removesuffix('ExecutionProvider')} "
                f"({'certified' if provider.certification == certification else 'not certified'})"
                for provider in providers
                if provider.name in CATALOG_PROVIDERS
            )
            or "none"
        )
        selected = name.removesuffix("ExecutionProvider")
        kind = "" if device_type == "Auto" else f"{device_type} "
        raise RuntimeError(
            f"Windows ML offers no certified compatible {kind}provider "
            f"for {selected}. Local catalogue: {offered}. "
            "Check device, architecture, runtime and driver requirements."
        )
    return candidates[0]


def catalogue_rows(providers, certification):
    """Only existing publisher metadata properties; never ensure_ready/device APIs."""
    from ruinedfooocus.backends.onnx.windows_ml import (
        CATALOG_PROVIDERS,
        PROVIDER_DEVICE_TYPES,
        bridge_compatible,
    )

    return [
        {
            "name": provider.name,
            "certified": provider.certification == certification,
            "bridge_compatible": bridge_compatible(provider.name),
            "device_types": list(PROVIDER_DEVICE_TYPES[provider.name]),
        }
        for provider in providers
        if provider.name in CATALOG_PROVIDERS
    ]


def catalogue(runtime, site_dir, *, name="Auto", device_type="Auto", inspect_only=False):
    import site

    site.addsitedir(site_dir)
    sys.path.insert(0, str(Path(runtime).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import onnxruntime  # Load the bridge's declared ORT DLL in this process only.

    if onnxruntime.__version__ != "1.25.2":
        raise RuntimeError(
            "The Windows ML catalogue bridge requires its isolated publisher ORT build."
        )
    from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
        initialize,
        InitializeOptions,
    )

    with initialize(options=InitializeOptions.NONE):
        import winui3.microsoft.windows.ai.machinelearning as winml

        catalog = winml.ExecutionProviderCatalog.get_default()
        providers = list(catalog.find_all_providers())
        if inspect_only:
            rows = catalogue_rows(providers, winml.ExecutionProviderCertification.CERTIFIED)
            providers.clear()
            catalog = None
            return {"catalog_version": "2.3.0", "providers": rows}
        selected = select_provider(
            providers, winml.ExecutionProviderCertification.CERTIFIED, name, device_type
        )
        result = selected.ensure_ready_async().get()
        if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
            raise RuntimeError("Windows ML acquisition failed: " + result.diagnostic_text)
        library = Path(selected.library_path)
        with library.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        record = {
            "name": selected.name,
            "library_path": str(library),
            "sha256": checksum,
            "catalog_version": "2.3.0",
            "qualification": "acquired_only_not_model_qualified",
        }
        selected = None
        providers.clear()
        catalog = None
    return record


def acquire(runtime, site_dir, name="Auto", device_type="Auto"):
    return catalogue(runtime, site_dir, name=name, device_type=device_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--site-dir", required=True)
    parser.add_argument("--ep-name", default="Auto")
    parser.add_argument("--ep-type", choices=["Auto", "CPU", "GPU", "NPU"], default="Auto")
    parser.add_argument("--list-providers", action="store_true")
    args = parser.parse_args()
    result = catalogue(
        args.runtime,
        args.site_dir,
        name=args.ep_name,
        device_type=args.ep_type,
        inspect_only=args.list_providers,
    )
    print(json.dumps(result), flush=True)
