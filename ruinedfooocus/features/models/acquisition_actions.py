"""Selected-resource actions over local presence and native component contracts."""

from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature


def validate_local_record(row):
    path = Path(row.get("local") or "")
    if (
        not row.get("local")
        or not row.get("size_matches", True)
        or not path.is_file()
        or file_signature(path) != row.get("local_signature")
    ):
        raise ValueError("The local file changed or is incomplete. Refresh and select it again.")
    return path


def component_targets(row, checkpoint, family, values, paths, overrides=None):
    """Compatible external roles on this checkpoint, never global file menus."""
    from ruinedfooocus.features.models.component_profiles import component_inventory
    from ruinedfooocus.features.models.component_candidates import installed_candidates

    if row.get("category") not in ("clip", "vae", "clip_vision") or not row.get("local"):
        return []
    target = validate_local_record(row).resolve()
    state = component_inventory(checkpoint, family, values, paths, overrides)
    choices = []
    for role in state["rows"]:
        if (
            role["category"] != row["category"]
            or role.get("locked")
            or role["status"] == "Embedded in checkpoint"
        ):
            continue
        for candidate in installed_candidates(role["key"], paths, companion_rows=state["rows"]):
            if (
                Path(candidate["path"]).resolve() == target
                and candidate["compatibility"] == "compatible"
            ):
                choices.append(
                    dict(key=role["key"], name=candidate["name"], checkpoint=str(checkpoint))
                )
                break
    return choices


def use_component(row, key, checkpoint, family, values, paths, overrides=None, scope=None):
    """Return session overrides only; no profile write, download or tensor load."""
    from ruinedfooocus.features.models.component_profiles import (
        checkpoint_identity,
        session_choices,
    )

    identity = checkpoint_identity(checkpoint)
    if scope is not None and scope != identity:
        raise ValueError("The checkpoint changed. Refresh components before using this file.")
    choices, identity = session_choices(checkpoint, values, overrides, scope)
    targets = component_targets(row, checkpoint, family, values, paths, choices)
    target = (
        targets[0]
        if key is None and len(targets) == 1
        else next((item for item in targets if item["key"] == key), None)
    )
    if target is None:
        raise ValueError(
            "This installed file is not a compatible external component for the selected checkpoint."
        )
    return dict(choices, **{target["key"]: target["name"]}), identity


def action_state(
    row, *, can_use=False, companions_missing=False, transfer=None, job=None, offline=False
):
    """One selected-resource primary; companion acquisition remains secondary."""
    active = bool(job and job.get("active"))
    owns = active and job.get("selection_key") == (row or {}).get("key")
    result = dict(
        download=False,
        download_enabled=False,
        use=False,
        cancel=owns,
        secondary=False,
        label="Download",
    )
    if row is None or owns:
        return result
    if row.get("local"):
        result["use"] = bool(can_use and row.get("size_matches", True))
        result["download"] = bool(
            companions_missing and row.get("allowed") and row.get("size_matches", True)
        )
        result["secondary"] = result["use"]
        result["label"] = "Download missing companions"
    else:
        result["download"] = bool(row.get("allowed"))
        if transfer and transfer.get("resumable"):
            result["label"] = "Resume download"
        elif (
            job
            and job.get("selection_key") == row.get("key")
            and job.get("state") in ("failed", "cancelled")
        ):
            result["label"] = "Retry download"
    result["download_enabled"] = bool(result["download"] and not active and not offline)
    return result
