"""Evidence from executed ORT nodes, separate from provider enumeration."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile


class AssignmentError(RuntimeError):
    pass


def profile_options(options, parent=None):
    # No persistent optimized/compiled-model cache: this is first-run evidence.
    directory = tempfile.TemporaryDirectory(prefix="rf-ort-profile-", dir=parent)
    options.enable_profiling = True
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    options.profile_file_prefix = str(Path(directory.name) / "execution")
    return directory


def graph_assignment(session, expected_provider):
    method = getattr(session, "get_provider_graph_assignment_info", None)
    if not callable(method):
        return None
    counts = Counter()
    for subgraph in method():
        counts[subgraph.ep_name] += len(subgraph.get_nodes())
    if not counts.get(expected_provider):
        raise AssignmentError(
            f"No optimized graph nodes assigned to {expected_provider}; observed {dict(counts)}."
        )
    return {
        "scope": "runtime_graph_assignment",
        "provider": expected_provider,
        "nodes": dict(counts),
        "cpu_partitions": counts.get("CPUExecutionProvider", 0),
        "original_graph_node_count": None,
    }


def assignment(session, expected_provider):
    """Read actual executed partitions after inference; never infer from get_providers."""
    filename = session.end_profiling()
    try:
        events = json.loads(Path(filename).read_text(encoding="utf-8"))
        counts = Counter(
            event["args"]["provider"]
            for event in events
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        )
    except (OSError, ValueError, TypeError) as error:
        raise AssignmentError("ORT returned no readable node-assignment profile.") from error
    if not counts.get(expected_provider):
        raise AssignmentError(
            f"No executed nodes were assigned to {expected_provider}; observed {dict(counts)}."
        )
    return {
        "scope": "first_inference_executed_nodes",
        "provider": expected_provider,
        "nodes": dict(counts),
        "cpu_partitions": counts.get("CPUExecutionProvider", 0),
        "all_graph_nodes": False,
    }


def assignment_status(evidence):
    return (
        f"{evidence['provider']}: {evidence['nodes'][evidence['provider']]} executed node events; "
        f"CPU partitions: {evidence['cpu_partitions']}"
    )


def finish_assignment(session, expected_provider):
    # Only the audited factory creates this field. Custom callers may expose no
    # profile; their initialization cannot be represented as execution evidence.
    if vars(session).get("_rf_assignment") is not None:
        return session._rf_assignment
    directory = vars(session).get("_rf_profile")
    if directory is None:
        return None
    if session._rf_assignment is None:
        try:
            session._rf_assignment = assignment(session.inner_session, expected_provider)
        finally:
            directory.cleanup()
            session._rf_profile = None
    return session._rf_assignment


def discard_profile(session):
    directory = vars(session).get("_rf_profile")
    if directory is not None:
        try:
            if session._rf_assignment is None:
                session.inner_session.end_profiling()
        finally:
            directory.cleanup()
            session._rf_profile = None


def cache_identity(**contract):
    """Caller-supplied exact identities; missing evidence cannot certify cache reuse."""
    required = (
        "model_sha256",
        "preprocessing",
        "shape",
        "dtype",
        "provider",
        "runtime",
        "device",
        "driver",
        "options",
    )
    if any(key not in contract or contract[key] is None for key in required):
        return None
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
