"""Cheap catalogue projections: no downloads, weight hashing or runtime imports."""

from pathlib import Path

from ruinedfooocus.features.models.model_capabilities import catalogue_artifact, task_eligibility
from ruinedfooocus.features.models.model_registry import (
    load_registry,
    RegistryError,
    unique_payload_bytes,
)


def bundle_presence(items, downloads, roots, selected=None):
    """Snapshot paths/sizes only; never load, hash, fetch or reassign aliases."""
    rows, missing = [], []
    unresolved = False
    for index, item in enumerate(items):
        entry = downloads.get(item["catalogue_key"])
        if entry is None:
            rows.append(["File presence", f"{item['catalogue_key']}: no live download tuple"])
            unresolved = True
            continue
        candidates = (
            [Path(selected)]
            if index == 0 and selected
            else [Path(folder) / entry["filename"] for folder in roots.get(entry["path"], ())]
        )
        if not candidates:
            rows.append(["File presence", f"{item['catalogue_key']}: storage location unknown"])
            unresolved = True
            continue
        found = next((path for path in candidates if path.is_file()), None)
        size = item["artifact"]["bytes"]
        if found is not None:
            observed = found.stat().st_size
            if size is not None and size != observed:
                state = "Present, size differs; content verification required"
                unresolved = True
            else:
                state = "Present by path/size; checksum unverified"
            location = found
        else:
            state = "Missing"
            location = candidates[0]
            missing.append(item)
        rows.append(["File presence", f"{item['catalogue_key']}: {state} ({location})"])
    total = None if unresolved else unique_payload_bytes(missing)
    rows.append(
        [
            "Missing payload bytes",
            size_label(total)
            + "; unique missing published payloads, not RAM/VRAM; no access or checksum guarantee",
        ]
    )
    return rows


def model_evidence(value, downloads, runtime, *, roots=None, projector="Auto"):
    """Display attributed metadata without claiming a local basename verifies content."""
    if not value:
        return [["Selection", "Choose a model."]]
    row = catalogue_artifact(Path(value).name, downloads)
    local = Path(value).is_absolute()
    result = [
        ["Selection", str(value)],
        ["Runtime source", f"{runtime.name} / {runtime.build or 'unknown'} ({runtime.origin})"],
    ]
    if row is None:
        return result + [
            [
                "Evidence",
                "Custom or unrecognized artifact. No publisher defaults or resource estimates inferred.",
            ]
        ]
    registry = load_registry()
    profile = registry.profile(row["profile_id"])
    result += [
        ["Identity", row["label"] + " / " + row["quantization"]["label"]],
        [
            "Evidence scope",
            "Catalogue candidate by filename; local content is verified only when loaded."
            if local
            else "Exact registered download tuple; device execution remains unverified.",
        ],
        ["Architecture", row["architecture"] or "Unknown"],
        ["Published file size", size_label(row["artifact"]["bytes"])],
        ["Text support", task_eligibility(row, runtime).reason],
        ["Image support", task_eligibility(row, runtime, "image").reason],
        [
            "Download observation",
            f"{row['artifact'].get('availability', 'unknown')} ({row['artifact'].get('checked_utc', 'unknown')}); not an access guarantee",
        ],
        ["Weight source", row["artifact"].get("pinned_url") or "Unknown"],
    ]
    license_info = profile.get("license", {})
    result.append(["Underlying license", license_info.get("declared_upstream") or "Unknown"])
    if license_info.get("restriction_summary"):
        result.append(["License notes", license_info["restriction_summary"]])
    declaration = row.get("repository_license_declaration", {})
    result.append(
        [
            "Artifact repository license",
            declaration.get("license_name") or declaration.get("license") or "Unknown",
        ]
    )
    release = profile.get("release", {})
    result.append(
        [
            "Publisher announcement",
            release.get("announced_release_date")
            or "Unknown (repository modification is not a release date)",
        ]
    )
    for mode in ("text", "image"):
        companions = [registry.artifact(identity) for identity in row["companions"][mode]]
        result.append(
            [
                f"{mode.capitalize()} companions",
                ", ".join(item["catalogue_key"] for item in companions) or "None declared",
            ]
        )
        if companions:
            total = unique_payload_bytes([row, *companions])
            result.append(
                [
                    f"{mode.capitalize()} bundle bytes",
                    size_label(total)
                    + " (unique published payloads; not RAM/VRAM or remaining download size)",
                ]
            )
        if roots is not None and (mode == "text" or companions):
            if mode == "image" and projector != "Auto":
                result.append(
                    [
                        "Image bundle selection",
                        "Projector disabled; automatic image bundle is not loaded."
                        if projector == "None"
                        else f"Custom projector: {projector}. Automatic bundle accounting does not apply.",
                    ]
                )
                continue
            items = [row, *companions]
            for item in items:
                if item["catalogue_key"] in downloads:
                    registry.annotations_for(
                        item["catalogue_key"], downloads[item["catalogue_key"]]
                    )
            for label, description in bundle_presence(
                items, downloads, roots, str(value) if local else None
            ):
                result.append([f"{mode.capitalize()} {label.lower()}", description])
    for estimate in row.get("hardware", {}).get("planning_profiles", ()):
        result.append(
            [
                "Unmeasured planning envelope",
                f"{estimate.get('mode', 'unknown')}; context {estimate.get('context_tokens', 'unknown')}; sequences {estimate.get('parallel_sequences', 'unknown')}; offload {estimate.get('offload', 'unknown')}; full-offload VRAM GiB {estimate.get('estimated_full_offload_vram_range_gib', 'unknown')}; system RAM GiB {estimate.get('recommended_system_ram_gib', 'unknown')}. Not an execution requirement or measured peak.",
            ]
        )
    for benchmark in registry.benchmarks(row["profile_id"]):
        evidence = benchmark.get("evidence", {})
        result.append(
            [
                "Published evidence (not reproduced)",
                f"{benchmark.get('dataset', 'unknown dataset')}: {benchmark.get('metric', 'unknown metric')} = {benchmark.get('value', 'unknown')}; "
                f"unit {benchmark.get('unit', 'unknown')}; precision {benchmark.get('precision', 'unknown')}; "
                f"protocol {benchmark.get('protocol', {}).get('protocol_id', 'not stated')}; "
                f"evaluator {evidence.get('evaluator', 'unknown')}; quantized artifact tested: {evidence.get('quantized_artifact_tested', 'unknown')}; "
                f"source {evidence.get('source_url', 'not stated')}. No universal or cross-protocol ranking.",
            ]
        )
    return result


def catalogue_local_file(value, downloads, roots):
    """Existing declared destination only; a custom same-named file is separate."""
    if not value or Path(value).is_absolute():
        return None
    for key in (str(value), "llm/" + str(value), "vision/" + str(value)):
        entry = downloads.get(key)
        if not isinstance(entry, dict):
            continue
        for folder in (roots or {}).get(entry.get("path"), ()):
            root = Path(folder).resolve()
            target = (root / entry.get("filename", "")).resolve()
            if target.is_relative_to(root) and target.is_file():
                return target
    return None


def model_action_state(value, downloads, session=None, *, roots=None):
    """Stat/job controls only; no weight reads, hashing, downloads or activation."""
    path = Path(value).expanduser() if value else None
    present = bool(path and path.is_absolute() and path.is_file())
    if path is not None and not path.is_absolute():
        present = catalogue_local_file(value, downloads, roots) is not None
    downloadable = (
        bool(value)
        and not (path and path.is_absolute())
        and any(
            key in downloads for key in (str(value), "llm/" + str(value), "vision/" + str(value))
        )
    )
    transfer = getattr(session, "transfer", None)
    pending = transfer is not None and not transfer.done()
    return {
        "download_visible": bool(downloadable and not present and not pending),
        "cancel_visible": pending,
        "load_available": present,
    }


def size_label(value):
    return "Unknown" if value is None else f"{value / 1024**3:.2f} GiB ({value:,} bytes)"


def safe_model_evidence(value, downloads, runtime, **options):
    try:
        result = model_evidence(value, downloads, runtime, **options)
        path = (
            Path(value)
            if value and Path(value).is_absolute()
            else catalogue_local_file(value, downloads, options.get("roots"))
        )
        if path is not None and path.is_file():
            from ruinedfooocus.features.chat.ui_sampling import source_evidence

            result += source_evidence(path)
        return result
    except (RegistryError, OSError) as error:
        return [["Evidence unavailable", str(error)]]
