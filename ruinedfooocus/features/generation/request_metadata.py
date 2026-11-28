"""Resolve imported request intent before consulting any generation resources."""

from ruinedfooocus.features.gallery.image_metadata import read_metadata, render_values

SOURCE_GENERATION = "_source_generation"


def capture_source_generation(data, *, store=None, owner=None):
    """Capture the current owned editor source's most recent generation recipe."""
    from ruinedfooocus.features.gallery.editor_session import default_store
    from ruinedfooocus.features.gallery.image_metadata import generation_provenance

    result = dict(data)
    result.pop(SOURCE_GENERATION, None)
    ticket = result.get("_editor_operation")
    if ticket is None:
        return result
    selected = store or default_store()
    selected.validate(ticket, owner=owner)
    snapshot = selected.snapshot(ticket.session_id, owner=owner)
    if snapshot.revision != ticket.revision or snapshot.head_id != ticket.source_id:
        from concurrent.futures import CancelledError

        raise CancelledError("The editor source changed while capturing its provenance.")
    steps = {step["result_id"]: step for step in snapshot.steps}
    head, visited = snapshot.head_id, set()
    while head in steps and head not in visited:
        visited.add(head)
        step = steps[head]
        execution = step.get("execution", {})
        provenance = generation_provenance(
            execution.get("source_generation")
            if execution.get("kind") == "processing"
            else execution
        )
        if provenance is not None:
            result[SOURCE_GENERATION] = provenance
            return result
        if execution.get("kind") != "processing":
            return result
        head = step["parent_id"]
    if head == snapshot.original.artifact_id:
        provenance = generation_provenance(snapshot.source_generation)
        if provenance is not None:
            result[SOURCE_GENERATION] = provenance
    return result


def attach_source_generation(data, metadata):
    """Utility execution stays top-level; its source recipe remains separate."""
    from ruinedfooocus.features.gallery.image_metadata import generation_provenance

    provenance = generation_provenance(data.get(SOURCE_GENERATION))
    if provenance is None:
        return metadata
    return dict(metadata, kind="processing", source_generation=provenance)


def _prompt_document(data):
    prompt = data.get("prompt")
    if isinstance(prompt, list) and len(prompt) == 1:
        prompt = prompt[0]
    return prompt, read_metadata(prompt)


def metadata_intent(data):
    """Restore model/control intent without sampling, model, prompt or file work."""
    from ruinedfooocus.features.generation.image_preparation import image_control_data

    if "_api_job" in data or data.get("task_type") == "tool_call" or data.get("_metadata_prompt"):
        return dict(data), None
    prompt, imported = _prompt_document(data)
    if imported is None:
        return dict(data), None
    if not isinstance(imported["prompt"], str):
        raise ValueError("Image metadata needs a text prompt.")
    result = image_control_data(dict(data, prompt=prompt))
    for key in ("base_model_name", "base_model_hash"):
        if key in imported:
            result[key] = imported[key]
    return result, imported


def capture_sampling_replay(data, models, *, _literal_boundary=True):
    """Freeze the existing metadata importer before queue/default state can drift.

    API/tool prompts are literal at the submission boundary. The explicit legacy
    importer delegates with that boundary disabled; its caller already decides
    whether a metadata document is being imported.
    """
    result = dict(data)
    if (
        result.get("_metadata_prompt")
        or _literal_boundary
        and ("_api_job" in result or result.get("task_type") == "tool_call")
    ):
        return result
    _, meta = _prompt_document(result)
    if meta is None:
        return result
    if not isinstance(meta["prompt"], str):
        raise ValueError("Image metadata needs a text prompt.")
    from ruinedfooocus.features.image_editing.image_guidance import restore_guidance_metadata

    checkpoint = meta.get("base_model_name", result.get("base_model_name"))
    family = "Unknown"
    if checkpoint:
        info = models.get_models_by_path("checkpoints", checkpoint, fetch=False)
        family = models.get_model_base(info)
        path = models.get_file("checkpoints", checkpoint)
        detect = getattr(models, "detect_checkpoint_base", None)
        if path and callable(detect):
            family = detect(path) or family
    restored = restore_guidance_metadata(meta, family=family)
    result.update(render_values(meta))
    result["_metadata_prompt"] = True
    result.update(restored)
    result["style_selection"] = []
    result["performance_selection"] = "Own settings"
    if meta.get("width") and meta.get("height"):
        from ruinedfooocus.features.generation.size_plan import replay_size

        width, height = replay_size(meta)
        result["width"] = result["custom_width"] = width
        result["height"] = result["custom_height"] = height
        result["aspect_ratios_selection"] = "Custom..."
        result["size_mode"] = "target" if "size_plan" in meta else "legacy_base"
    result["automatic_upscale"] = meta.get("automatic_upscale", "Off")
    result["automatic_upscale_target"] = "Generated image"
    if "steps" in meta:
        result["custom_steps"] = int(meta["steps"])
    return result


def restore_utility_metadata(data, imported):
    """Replay inert metadata fields without checkpoint/guidance/prompt resolution."""
    if imported is None:
        return data
    result = dict(data)
    result.update(render_values(imported))
    result.update(
        _metadata_prompt=True,
        style_selection=[],
        performance_selection="Own settings",
        automatic_upscale=imported.get("automatic_upscale", "Off"),
        automatic_upscale_target="Generated image",
    )
    if "steps" in imported:
        result["custom_steps"] = int(imported["steps"])
    return result
