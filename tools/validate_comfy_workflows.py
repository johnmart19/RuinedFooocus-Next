"""Compare app/native adapters sequentially, without launching a server.

Usage (run with the project's Python environment)::
  python tools/validate_comfy_workflows.py case.json --report outputs/comparison.json

The trusted local JSON spec contains ``app`` and ``native`` objects, each with
``entrypoint`` (``package.module:function`` or ``path/to/adapter.py:function``)
and optional JSON ``kwargs``. Both functions must return the same named artifact
mapping, e.g. {"latent": tensor, "conditioning": conditioning, "output": images,
"settings": {...}}. Call official Comfy nodes in the native adapter. Adapters
must load only existing local weights and must not launch servers or subprocesses.
An adapter may return only the artifacts relevant to its feature; the report lists
exactly what was checked and never claims coverage beyond those artifacts.

Optional spec fields: ``runtime_args`` (application CLI flags), ``atol``, ``rtol``.
Use identical seed/settings and independent fresh pipeline/model objects in each
adapter. The harness unloads Comfy models between adapters. No graph executor is
reimplemented here. Importing this module or using --help does not import Torch.
Any adapter or cleanup exception aborts the pair before another model load.
Failure tracebacks remain in the report; failed cleanup is not retried. A failed
pair must not be retried without a separately authorized device/runtime check.
Source exports report the structurally validated distribution commit/pins as
declared identity alongside measured source hashes; they never inherit a parent
checkout's Git revision or dirty status.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def capture(value, arrays, path="root"):
    """Produce JSON fingerprints and CPU numeric snapshots, releasing GPU refs."""
    if hasattr(value, "tensors") and isinstance(value.tensors, (tuple, list)):
        return {"nested_tensors": capture(list(value.tensors), arrays, path + "/nested_tensors")}
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu().contiguous()
        # Byte views preserve bfloat16 and float8 without unsupported NumPy casts.
        raw = tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        arrays[path] = tensor.double().numpy().copy()
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if hasattr(value, "__array__"):
        import numpy as np

        array = np.asarray(value)
        if array.dtype.kind not in "buifc":
            raise TypeError(f"Unsupported array dtype at {path}: {array.dtype}")
        arrays[path] = array.copy()
        return {
            "kind": "array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"Artifact keys must be strings at {path}")
        return {key: capture(item, arrays, f"{path}/{key}") for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [capture(item, arrays, f"{path}/{index}") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Unsupported artifact at {path}: {type(value).__name__}; return tensors/settings, not model objects"
    )


def compare(left, right, left_arrays, right_arrays, atol=0.0, rtol=0.0):
    import numpy as np

    differences = []
    numeric = {}
    for path in sorted(left_arrays.keys() | right_arrays.keys()):
        if path not in left_arrays or path not in right_arrays:
            differences.append(path)
            continue
        a, b = left_arrays[path], right_arrays[path]
        same_shape = a.shape == b.shape
        close = same_shape and bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
        numeric[path] = {"within_tolerance": close, "same_shape": same_shape}
        if same_shape and a.size:
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            numeric[path]["max_abs_error"] = (
                float(delta.max()) if np.isfinite(delta).all() else None
            )
            numeric[path]["mean_abs_error"] = (
                float(delta.mean()) if np.isfinite(delta).all() else None
            )
        if not close:
            differences.append(path)

    def structure(value):
        if isinstance(value, dict):
            return {
                k: structure(v)
                for k, v in value.items()
                if not (k == "sha256" and value.get("kind") in ("tensor", "array"))
            }
        if isinstance(value, list):
            return [structure(v) for v in value]
        return value

    settings_equal = structure(left) == structure(right)
    return {
        "passed": settings_equal and not differences,
        "structure_settings_equal": settings_equal,
        "exact_fingerprints_equal": left == right,
        "atol": atol,
        "rtol": rtol,
        "numeric": numeric,
        "failed_artifacts": differences,
    }


def load_entrypoint(entrypoint):
    module_name, function = entrypoint.rsplit(":", 1)
    if module_name.endswith(".py"):
        source = Path(module_name).resolve()
        spec = importlib.util.spec_from_file_location(
            "_comfy_validation_adapter_" + hashlib.sha256(str(source).encode()).hexdigest()[:12],
            source,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, function)


def initialize(runtime_args):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.append(str(ROOT / "repositories" / "ComfyUI"))
    sys.path.append(str(ROOT / "repositories"))
    os.environ.update(
        RF_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        GRADIO_ANALYTICS_ENABLED="False",
    )
    sys.argv = [str(ROOT / "ruinedfooocus/app/bootstrap.py"), "--offline", *runtime_args]
    from ruinedfooocus.app.arguments import args
    from ruinedfooocus.backends.comfy.comfy_compat import (
        configure_device_environment,
        initialize_backend,
    )

    configure_device_environment(args)
    initialize_backend(args)


def export_identity(root):
    """Read declared archive identity; file hashes still measure the actual source."""
    try:
        value = json.loads((Path(root) / "distribution.json").read_text(encoding="utf-8"))
        pins = value.get("backend_pins")
        if (
            value.get("kind") != "source-archive"
            or re.fullmatch(r"[0-9a-f]{40}", value.get("source_commit", "")) is None
            or not isinstance(pins, dict)
            or not pins
            or any(
                not isinstance(key, str)
                or not isinstance(pin, str)
                or re.fullmatch(r"[0-9a-f]{40}", pin) is None
                for key, pin in pins.items()
            )
        ):
            return None
        return {"source_commit": value["source_commit"], "backend_pins": pins}
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def git_root(path):
    """Never inherit the Git identity of a source export's enclosing checkout."""
    try:
        root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return bool(root) and Path(root).resolve() == Path(path).resolve()
    except (OSError, subprocess.SubprocessError):
        return False


def revision(path):
    path = Path(path).resolve()
    exported = export_identity(path)
    if exported is not None:
        return exported["source_commit"]
    if path.parent.name == "repositories":
        exported = export_identity(path.parent.parent)
        key = {"ComfyUI": "ComfyUI", "molbal_comfyui_gguf": "molbal/ComfyUI-GGUF"}.get(path.name)
        if exported is not None and key is not None:
            return exported["backend_pins"].get(key)
    if not git_root(path):
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(root=ROOT):
    files = set(root.glob("modules/**/*.py")) | set(root.glob("tools/*.py"))
    files.update(root.glob("ruinedfooocus/**/*.py"))
    files.update(root.glob("ruinedfooocus/resources/**/*.json"))
    files.update(root.glob("ruinedfooocus/resources/defaults/*"))
    files.update(
        root / name
        for name in (
            "ruinedfooocus/app/state.py",
            "ruinedfooocus/app/arguments.py",
            "ruinedfooocus/ui/app.py",
            "ruinedfooocus/app/bootstrap.py",
            "entry_with_update.py",
            "ruinedfooocus/resources/defaults/index.json",
            "pip/modules.txt",
            "requirements_versions.txt",
        )
    )
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if path.is_file()
    }
    tree = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        dirty = (
            bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"], text=True
                ).strip()
            )
            if export_identity(root) is None and git_root(root)
            else None
        )
    except (OSError, subprocess.SubprocessError):
        dirty = None
    runtime = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        runtime.update(torch=str(torch.__version__), hip=getattr(torch.version, "hip", None))
        if torch.cuda.is_initialized():
            runtime["gpu_name"] = torch.cuda.get_device_name()
    return {
        "source_tree_sha256": tree,
        "source_files_sha256": hashes,
        "app_dirty": dirty,
        "declared_export_identity": export_identity(root),
        "runtime": runtime,
    }


def run(spec):
    initialize(spec.get("runtime_args", []))
    import gc
    import torch
    import comfy.model_management as memory

    report = {
        "app_revision": revision(ROOT),
        "comfy_revision": revision(ROOT / "repositories/ComfyUI"),
        "spec": spec,
        "runs": {},
        "provenance": provenance(),
    }
    snapshots = {}
    cleanup_errors = []
    for side in ("app", "native"):
        failed = False
        config = spec[side]
        arrays = {}
        result = None
        try:
            function = load_entrypoint(config["entrypoint"])
            started = time.perf_counter()
            with torch.inference_mode():
                result = function(**config.get("kwargs", {}))
            # Materialization synchronizes device work before recording timing.
            if not isinstance(result, dict) or not result:
                raise ValueError("Adapters must return a nonempty named artifact mapping")
            fingerprints = capture(result, arrays)
            report["runs"][side] = {
                "seconds_including_capture": time.perf_counter() - started,
                "artifacts": fingerprints,
            }
            snapshots[side] = arrays
            del result
        except Exception:
            failed = True
            report["runs"][side] = {"error": traceback.format_exc()}
        finally:
            # Do not retain failed adapter results through the next run/exit.
            result = None
            try:
                cleanup_runtime(memory, torch, gc)
            except Exception:
                failed = True
                cleanup_errors.append({"side": side, "error": traceback.format_exc()})
        if failed:
            # A device/runtime failure must never trigger another model load.
            report["aborted_after"] = side
            break
    if len(snapshots) == 2:
        report["comparison"] = compare(
            report["runs"]["app"]["artifacts"],
            report["runs"]["native"]["artifacts"],
            snapshots["app"],
            snapshots["native"],
            spec.get("atol", 0.0),
            spec.get("rtol", 0.0),
        )
    else:
        report["comparison"] = {
            "passed": False,
            "reason": "An adapter or its cleanup failed; equivalence was not established",
        }
    if "aborted_after" not in report:
        try:
            report["runtime_cleanup"] = cleanup_runtime(memory, torch, gc)
        except Exception:
            cleanup_errors.append({"side": "shutdown", "error": traceback.format_exc()})
    if cleanup_errors:
        report["cleanup_errors"] = cleanup_errors
        report["comparison"]["passed"] = False
    return report


def cleanup_runtime(memory, torch, gc):
    """Release native residency/cast owners before interpreter module teardown.

    CPU snapshots already own their bytes. Never force-close an mmap still owned
    by a live tensor, and never deinitialize AIMDO while native owners remain.
    This cannot certify successful Py_Exit; callers must also require exit code 0.
    """
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    memory.unload_all_models()
    gc.collect()
    # Native cast caches can own pinned buffers and their backing tensor storage.
    memory.reset_cast_buffers()
    memory.cleanup_models()
    gc.collect()
    memory.soft_empty_cache()
    mmap_module = sys.modules.get("comfy_aimdo.model_mmap")
    mmap_class = getattr(mmap_module, "ModelMMAP", None)
    live = sum(type(value) is mmap_class for value in gc.get_objects()) if mmap_class else None
    return {
        "completed": True,
        "live_model_mmaps": live,
        "process_exit_requires_external_verification": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    report_path = args.report.resolve()
    try:
        report = run(spec)
    except Exception:
        report = {"spec": spec, "error": traceback.format_exc(), "comparison": {"passed": False}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Comparison report: {report_path}")
    return 0 if report["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
