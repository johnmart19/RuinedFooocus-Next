"""CPU-only cold/cold and warm/warm fingerprint comparison of paired reports.

Example: python tools/compare_qwen_state_controls.py app-report.json native-repeat.json
  --output outputs/qwen-state-control.json

This never changes or overrides the sequential comparison result in either input.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

CORE = ("conditioning", "latent", "output", "settings")
SOURCE_FILES = (
    "ruinedfooocus/features/generation/image_pipeline.py",
    "ruinedfooocus/features/generation/pipeline_utils.py",
    "ruinedfooocus/backends/comfy/model_attention.py",
    "ruinedfooocus/features/models/resource_defaults.py",
    "tools/comfy_feature_app.py",
    "tools/comfy_feature_reference.py",
)


def _arguments(report, side):
    values = dict(report["spec"][side]["kwargs"])
    values.pop("output_png", None)
    for key, default in (
        ("diagnostics", False),
        ("moss_slg", False),
        ("lora_strength", 1.0),
        ("sampling_shift", None),
        ("cfg_norm_strength", None),
    ):
        values.setdefault(key, default)
    return values


def _tensor_hashes(value, path=""):
    found = {}
    if isinstance(value, dict):
        if value.get("kind") in ("tensor", "array"):
            digest = value.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError("Missing tensor fingerprint: " + path)
            shape = value.get("shape")
            if (
                not isinstance(shape, list)
                or any(type(size) is not int or size <= 0 for size in shape)
                or not isinstance(value.get("dtype"), str)
                or not value["dtype"]
            ):
                raise ValueError("Invalid or empty tensor fingerprint: " + path)
            found[path] = {
                "sha256": digest,
                "shape": value.get("shape"),
                "dtype": value.get("dtype"),
            }
        else:
            for key, item in value.items():
                found.update(_tensor_hashes(item, path + "/" + key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_tensor_hashes(item, path + "/" + str(index)))
    return found


def _process_record(path, body, report):
    suffix = "-comparison.json"
    process_path = path.with_name(path.name.removesuffix(suffix) + "-process.json")
    result = {"path": str(process_path), "valid": False}
    try:
        if not path.name.endswith(suffix):
            raise ValueError("Expected a *-comparison.json report for process correlation")
        process_body = process_path.read_bytes()
        process = json.loads(process_body)
        result["process_record_sha256"] = hashlib.sha256(process_body).hexdigest()
        passed = report.get("comparison", {}).get("passed")
        if type(passed) is not bool:
            raise ValueError("Missing raw sequential comparison status")
        expected = 0 if passed else 1
        result.update(expected_exit_code=expected, exit_code=process.get("exit_code"))
        if (
            type(process.get("exit_code")) is not int
            or process["exit_code"] != expected
            or process.get("error")
        ):
            raise ValueError(
                "Process failed/crashed or its exit code disagrees with the recorded comparison"
            )
        if "comparison_sha256" in process:
            digest = process["comparison_sha256"]
            result["correlation_method"] = "report_sha256"
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
                or digest.lower() != hashlib.sha256(body).hexdigest()
            ):
                raise ValueError("Process record report SHA256 is missing, invalid or stale")
        else:
            result["correlation_method"] = "legacy_record_mtime_not_older_than_report"
            if process_path.stat().st_mtime_ns < path.stat().st_mtime_ns:
                raise ValueError("Legacy process record is stale")
        result["valid"] = True
    except (OSError, ValueError, TypeError, AttributeError) as error:
        result["error"] = str(error)
    return result


def compare(app_path, native_path):
    paths = [Path(app_path).resolve(), Path(native_path).resolve()]
    bodies = [path.read_bytes() for path in paths]
    left, right = [json.loads(body) for body in bodies]
    for report in (left, right):
        for side in ("app", "native"):
            run = report.get("runs", {}).get(side, {})
            if not isinstance(run.get("artifacts"), dict) or run.get("error"):
                raise ValueError("Missing or failed " + side + " artifact run")
    process_records = [
        _process_record(path, body, report)
        for path, body, report in zip(paths, bodies, (left, right))
    ]
    checks = {
        "app_entrypoint": left["spec"]["app"]["entrypoint"]
        .replace(".", "/")
        .endswith("comfy_feature_app:image_edit"),
        "app_reference_entrypoint": left["spec"]["native"]["entrypoint"]
        .replace(".", "/")
        .endswith("comfy_feature_reference:image_edit"),
        "native_control_entrypoints": all(
            right["spec"][side]["entrypoint"]
            .replace(".", "/")
            .endswith("comfy_feature_reference:image_edit")
            for side in ("app", "native")
        ),
        "runtime_args_equal": left["spec"].get("runtime_args", [])
        == right["spec"].get("runtime_args", []),
        "comfy_revision_equal": bool(left.get("comfy_revision"))
        and left["comfy_revision"] == right.get("comfy_revision"),
        "all_workload_arguments_equal": all(
            _arguments(report, side) == _arguments(left, "app")
            for report in (left, right)
            for side in ("app", "native")
        ),
        "qwen_workload": _arguments(left, "app").get("family") == "qwen",
        "run_order_app_then_native": all(
            list(report["runs"]) == ["app", "native"] for report in (left, right)
        ),
        "external_process_records_valid": all(record["valid"] for record in process_records),
    }
    sources = [
        report.get("provenance", {}).get("source_files_sha256", {}) for report in (left, right)
    ]
    source_checks = {
        name: (
            bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(sources[0][name])))
            and sources[0][name] == sources[1][name]
            if name in sources[0] and name in sources[1]
            else None
        )
        for name in SOURCE_FILES
        if name in sources[0] or name in sources[1]
    }
    checks["no_recorded_source_conflict"] = all(
        value is not False for value in source_checks.values()
    )
    source_complete = len(source_checks) == len(SOURCE_FILES) and all(
        value is True for value in source_checks.values()
    )
    phases = {}
    for side, phase in (("app", "cold_first_run"), ("native", "warm_second_run")):
        a, b = left["runs"][side]["artifacts"], right["runs"][side]["artifacts"]
        fields = {name: name in a and name in b and a[name] == b[name] for name in CORE}
        hashes = {
            name: {
                "application_report": _tensor_hashes(a.get(name), name),
                "native_control_report": _tensor_hashes(b.get(name), name),
            }
            for name in CORE
        }
        for name in ("conditioning", "latent", "output"):
            if any(not value for value in hashes[name].values()):
                raise ValueError(
                    "Required nonempty tensor fingerprints missing: " + phase + "/" + name
                )
        phases[phase] = {
            "exact_core_artifact_equality": fields,
            "all_core_artifacts_equal": all(fields.values()),
            "tensor_fingerprints": hashes,
        }
    supported = all(checks.values()) and all(
        item["all_core_artifacts_equal"] for item in phases.values()
    )
    return {
        "method": "Exact recorded artifact fingerprints, cold first run versus cold first run; warm second run versus warm second run.",
        "scope": "State-matched evidence only. Does not establish sequential cold/warm equality, quality, or fresh-process repeatability.",
        "inputs": [
            {"path": str(path), "report_sha256": hashlib.sha256(body).hexdigest()}
            for path, body in zip(paths, bodies)
        ],
        "external_process_records": process_records,
        "safeguards": checks,
        "source_checks": source_checks,
        "phase_comparisons": phases,
        "complete_source_provenance_verified": source_complete,
        "source_provenance_limit": None
        if source_complete
        else "At least one input report lacks relevant source hashes; artifact equality does not prove identical source trees.",
        "state_matched_core_equivalence_supported": supported,
        "original_sequential_results": [report.get("comparison", {}) for report in (left, right)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("application_report", type=Path)
    parser.add_argument("native_control_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.application_report, args.native_control_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("State-matched evidence supported:", result["state_matched_core_equivalence_supported"])
    return 0 if result["state_matched_core_equivalence_supported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
