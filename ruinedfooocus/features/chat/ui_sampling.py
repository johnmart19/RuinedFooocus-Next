"""Read-only selected-source evidence and initialized sampling presentation."""

from ruinedfooocus.features.chat.model_profiles import SAMPLERS, validate_sampling
from ruinedfooocus.features.chat.sampling_sources import cached_sampling_source, offline_mode


def active_runtime_status(model, result):
    rows = [
        f"Active: {model}",
        f"Runtime: {result.get('runtime_name') or 'Unknown'} · {result.get('runtime_label') or 'Unverified'}",
    ]
    if result.get("fallback_reason"):
        rows.append("Backend selection: " + result["fallback_reason"])
    return "\n\n".join(rows)


def source_evidence(path):
    source = cached_sampling_source(path)
    status = source.get("status", "not_prepared")
    messages = {
        "reviewed": "Exact content verified; packaged publisher recommendations, when published, apply on load.",
        "reviewed_variant": "Exact content matched the reviewed variant revision; its packaged sampling advice, when published, applies on load. Runtime and reasoning support remain separate.",
        "source_unverified": "Source found, publisher/variant unverified; these hints do not supply automatic defaults.",
        "offline": "Recommendations unavailable offline; runtime defaults and explicit overrides remain usable.",
        "not_found": "No qualified publisher recommendations found; runtime defaults and explicit overrides remain usable.",
        "not_prepared": "Source not prepared. Import, Load or Refresh sampling recommendations to inspect it.",
    }
    rows = [["Selected sampling source", messages.get(status, messages["not_prepared"])]]
    if offline_mode():
        rows.append(["Source mode", "Offline; cached evidence is retained."])
    if source.get("last_lookup") == "unavailable":
        rows.append(["Source refresh", "Lookup unavailable; current cached evidence retained."])
    for repository in source.get("repositories", ())[:4]:
        if isinstance(repository, str):
            rows.append(["Repository hint (not identity)", repository])
    if source.get("status") == "reviewed_variant":
        rows.append(
            [
                "Reviewed variant revision",
                f"{source.get('variant_repository', 'Unknown')} @ {source.get('variant_revision', 'Unknown')}",
            ]
        )
    for observation in source.get("observations", ())[:4]:
        if not isinstance(observation, dict):
            continue
        config = observation.get("generation_config", {})
        try:
            config = validate_sampling(config)
        except ValueError:
            continue
        if config:
            rows.append(
                [
                    "Reported source settings (not active)",
                    ", ".join(f"{key}: {value}" for key, value in config.items()),
                ]
            )
    return rows


def active_sampling_status(state):
    if not isinstance(state, dict):
        return "No active Chat model sampling state."
    values = validate_sampling(
        {key: value for key, value in state.get("effective", {}).items() if key in SAMPLERS}
    )
    origins = state.get("origins", {})
    labels = {
        "publisher_defaults": "Publisher / verified variant",
        "prompt_enhancement": "Task default",
        "saved_overrides": "Saved override",
        "native_cli": "Native server argument",
        "request_overrides": "Explicit request",
    }
    title = {
        "publisher_defaults": "Recommended settings",
        "user_overrides": "Your overrides",
        "task_defaults": "Task defaults",
    }.get(state.get("status"), "Runtime defaults")
    rows = [f"**{title} · active model**"]
    if values:
        table = ["| Setting | Effective value | Source |", "|---|---|---|"]
        table += [
            f"| {key} | {value} | {labels.get(origins.get(key), 'Runtime')} |"
            for key, value in values.items()
        ]
        rows.append("\n".join(table))
    rows.append("Other values use runtime defaults.")
    return "\n\n".join(rows)
