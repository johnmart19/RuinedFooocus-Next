"""Split legacy registry data, or verify indexed resources against their source digest."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.models.model_registry import ModelRegistry, load_registry
from ruinedfooocus.features.models.model_registry import metadata_digest as document_digest


def baseline_manifest(document):
    """Retain exact migration identities when later features add new resources."""
    return {
        "profiles": list(document["profiles"]),
        "artifact_ids": [entry["id"] for entry in document["artifacts"]],
        "benchmark_digests": [document_digest(row) for row in document["benchmarks"]],
        "alias_digests": [document_digest(row) for row in document.get("payload_alias_groups", [])],
        "provenance": document.get("provenance", {}),
        "source_document_sha256": document_digest(document),
        "download_identities_sha256": document_digest(
            {
                entry["catalogue_key"]: entry["download_identity"]
                for entry in document["artifacts"]
                if entry["state"] == "existing"
            }
        ),
    }


def baseline_projection(document, baseline):
    artifacts = {entry["id"]: entry for entry in document["artifacts"]}
    retirement = baseline.get("scope_retirement", {})

    def active(key):
        retired = retirement.get(key, [])
        if (
            not isinstance(retired, list)
            or not all(isinstance(value, str) for value in retired)
            or len(retired) != len(set(retired))
            or not set(retired) <= set(baseline[key])
        ):
            raise ValueError("Invalid migration scope retirement.")
        if key == "profiles" and set(retired) & set(document["profiles"]):
            raise ValueError("Retired migration profiles remain active.")
        if key == "artifact_ids" and set(retired) & set(artifacts):
            raise ValueError("Retired migration artifacts remain active.")
        return [value for value in baseline[key] if value not in retired]

    def retained(rows, digests):
        available = defaultdict(list)
        for row in rows:
            available[document_digest(row)].append(row)
        result = []
        for digest in digests:
            if not available[digest]:
                raise ValueError("An original migration evidence record changed or disappeared.")
            result.append(available[digest].pop(0))
        return result

    return {
        "schema_version": 1,
        "profiles": {identity: document["profiles"][identity] for identity in active("profiles")},
        "artifacts": [artifacts[identity] for identity in active("artifact_ids")],
        "benchmarks": retained(document["benchmarks"], active("benchmark_digests")),
        "payload_alias_groups": retained(
            document.get("payload_alias_groups", []), active("alias_digests")
        ),
        "provenance": baseline["provenance"],
    }


def split(
    document, destination, *, migration_baseline=None, artifact_paths=None, profile_paths=None
):
    """Publish shards before the index; never modify download catalogue tuples."""
    ModelRegistry(document)
    destination = Path(destination).resolve()
    artifact_paths = {} if artifact_paths is None else artifact_paths
    identities = {entry["id"] for entry in document["artifacts"]}
    if not isinstance(artifact_paths, dict) or any(
        identity not in identities
        or not isinstance(path, str)
        or re.fullmatch(r"artifacts/[a-zA-Z0-9_-]+\.json", path) is None
        for identity, path in artifact_paths.items()
    ):
        raise ValueError("Invalid existing artifact shard mapping.")
    profile_paths = {} if profile_paths is None else profile_paths
    if not isinstance(profile_paths, dict) or any(
        identity not in document["profiles"]
        or not isinstance(path, str)
        or re.fullmatch(r"families/[a-zA-Z0-9_.-]+\.json", path) is None
        for identity, path in profile_paths.items()
    ):
        raise ValueError("Invalid existing profile shard mapping.")
    if migration_baseline is None and (destination / "index.json").is_file():
        previous = json.loads((destination / "index.json").read_text(encoding="utf-8"))
        migration_baseline = previous.get("migration_baseline")
    for entry in document["artifacts"]:
        if re.fullmatch(r"[a-zA-Z0-9_-]+", entry["role"]) is None:
            raise ValueError(f"Unsafe model role resource name: {entry['role']}")
    index = dict(
        schema_version=2,
        provenance=document.get("provenance", {}),
        profiles={},
        artifacts={},
        qualification={},
        shards={},
        benchmark_order=[],
        payload_alias_groups=document.get("payload_alias_groups", []),
        source_document_sha256=document_digest(document),
    )
    if migration_baseline is not None:
        index["migration_baseline"] = migration_baseline

    def write(name, payload):
        if name in index["shards"]:
            raise ValueError(f"Duplicate normalized shard path: {name}")
        path = (destination / name).resolve()
        if not path.is_relative_to(destination) or path == destination:
            raise ValueError(f"Model resource escapes destination: {name}")
        value = dict(schema_version=1, **payload)
        save_json(path, value)
        index["shards"][name] = dict(sha256=document_digest(value))

    qualification, profiles, artifacts = {}, defaultdict(dict), defaultdict(dict)
    for identity, profile in document["profiles"].items():
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", identity)
        path = profile_paths.get(identity, f"families/{name}.json")
        index["profiles"][identity] = dict(path=path, keys=list(profile))
        values = dict(profile)
        if "qualification" in values:
            qualification[identity] = {"qualification": values.pop("qualification")}
        profiles[path][identity] = values
    for path, entries in profiles.items():
        write(path, {"profiles": entries})
    for entry in document["artifacts"]:
        path = artifact_paths.get(entry["id"], f"artifacts/{entry['role']}.json")
        artifacts[path][entry["id"]] = entry
        index["artifacts"][entry["id"]] = dict(
            path=path,
            **{key: entry[key] for key in ("catalogue_key", "state", "role", "profile_id")},
        )
    for path, entries in artifacts.items():
        write(path, {"artifacts": entries})
    for row in document["benchmarks"]:
        identity = row["profile_id"]
        qualification.setdefault(identity, {}).setdefault("benchmarks", []).append(row)
        index["benchmark_order"].append(identity)
    for identity, payload in qualification.items():
        path = "qualification/" + re.sub(r"[^a-zA-Z0-9_.-]", "_", identity) + ".json"
        index["qualification"][identity] = path
        write(path, {"profile_id": identity, **payload})
    index["download_identities_sha256"] = document_digest(
        {
            entry["catalogue_key"]: entry["download_identity"]
            for entry in document["artifacts"]
            if entry["state"] == "existing"
        }
    )
    index_path = (destination / "index.json").resolve()
    if not index_path.is_relative_to(destination):
        raise ValueError("Model index escapes destination.")
    save_json(index_path, index)
    return index


def verify(destination):
    registry = load_registry(destination)
    document = registry.to_document()
    ModelRegistry(document)
    index = json.loads((Path(destination) / "index.json").read_text(encoding="utf-8"))
    if document_digest(document) != index["source_document_sha256"]:
        raise ValueError("Split resources differ from the recorded source document.")
    actual = {
        entry["catalogue_key"]: entry["download_identity"]
        for entry in document["artifacts"]
        if entry["state"] == "existing"
    }
    if document_digest(actual) != index["download_identities_sha256"]:
        raise ValueError("Split resources changed existing download tuples.")
    result = {
        "profiles": len(document["profiles"]),
        "artifacts": len(document["artifacts"]),
        "benchmarks": len(document["benchmarks"]),
        "source_document_sha256": index["source_document_sha256"],
        "download_identities_sha256": index["download_identities_sha256"],
    }
    baseline = index.get("migration_baseline")
    if baseline is not None:
        original = baseline_projection(document, baseline)
        expected = baseline.get("scope_retirement", baseline)
        if document_digest(original) != expected["source_document_sha256"]:
            raise ValueError("Original migration annotations changed.")
        identities = {
            entry["catalogue_key"]: entry["download_identity"]
            for entry in original["artifacts"]
            if entry["state"] == "existing"
        }
        if document_digest(identities) != expected["download_identities_sha256"]:
            raise ValueError("Original migration download identities changed.")
        result["migration_baseline"] = {
            "profiles": len(original["profiles"]),
            "artifacts": len(original["artifacts"]),
            "benchmarks": len(original["benchmarks"]),
            "source_document_sha256": expected["source_document_sha256"],
            "download_identities_sha256": expected["download_identities_sha256"],
        }
        if "scope_retirement" in baseline:
            result["historical_migration_baseline"] = {
                "profiles": len(baseline["profiles"]),
                "artifacts": len(baseline["artifact_ids"]),
                "benchmarks": len(baseline["benchmark_digests"]),
                "source_document_sha256": baseline["source_document_sha256"],
                "download_identities_sha256": baseline["download_identities_sha256"],
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "ruinedfooocus/resources/model_metadata"
    )
    args = parser.parse_args()
    if args.source:
        split(json.loads(args.source.read_text(encoding="utf-8")), args.destination)
    print(json.dumps(verify(args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
