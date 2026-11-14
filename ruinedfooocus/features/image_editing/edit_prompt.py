"""Literal Editor conditioning from the already captured owned source recipe."""

from copy import deepcopy

METADATA_KEY = "image_edit_prompt"
REQUEST_KEY = "_editor_prompt_metadata"
MAX_SOURCE_CHARS = 16384
MAX_PRIOR_CHARS = 4096
MAX_PRIOR_INSTRUCTIONS = 3


def _previous(source):
    previous = source.get(METADATA_KEY)
    if not isinstance(previous, dict) or previous.get("schema") != 1:
        return None
    if previous.get("mode") not in ("img2img", "inpaint") or not isinstance(
        previous.get("source_prompt"), str
    ):
        return None
    history = previous.get("previous_instructions")
    if (
        not isinstance(history, list)
        or len(history) > MAX_PRIOR_INSTRUCTIONS
        or not all(isinstance(value, str) for value in history)
        or not isinstance(previous.get("instruction"), str)
    ):
        return None
    return previous


def render_prompt(instruction, data):
    """Keep requested text intact; only ordinary Editor img2img adds source context."""
    from ruinedfooocus.features.image_editing.image_workflows import KEY, WorkflowStep

    data.pop(REQUEST_KEY, None)
    step = data.get(KEY)
    if not isinstance(step, WorkflowStep) or step.action not in ("prompt-edit", "inpaint"):
        return instruction
    mode = data.get("controlnet", {}).get("type")
    if step.action == "inpaint" and data.get("inpaint_toggle"):
        from ruinedfooocus.features.models.image_components import ALIASES

        family = (step.ticket.effective_plan or {}).get("family")
        family = ALIASES.get(family, family)
        mode = (
            "inpaint"
            if family in ("SDXL", "SD1", "SD2", "SD15", "SD20", "Flux")
            else "edit"
            if family == "QwenImage21"
            else None
        )
    if mode not in ("edit", "img2img", "inpaint"):
        return instruction
    data["raw_prompt"] = instruction
    source_prompt, history, omission = "", [], None
    if mode in ("img2img", "inpaint"):
        source = data.get("_source_generation")
        source = source if isinstance(source, dict) else {}
        previous = _previous(source)
        source_prompt = previous["source_prompt"] if previous else source.get("prompt", "")
        if not isinstance(source_prompt, str):
            source_prompt = ""
        if len(source_prompt) > MAX_SOURCE_CHARS:
            source_prompt, omission = "", "Source prompt exceeds the context limit."
        if previous:
            remaining = MAX_PRIOR_CHARS
            for value in reversed([*previous["previous_instructions"], previous["instruction"]]):
                if len(history) == MAX_PRIOR_INSTRUCTIONS or len(value) > remaining:
                    break
                history.insert(0, value)
                remaining -= len(value)
    data[REQUEST_KEY] = dict(
        schema=1,
        mode=mode,
        operation=step.action,
        source_prompt=source_prompt,
        previous_instructions=history,
        instruction=instruction,
        context_omission=omission,
    )
    from ruinedfooocus.features.models.lora_names import parse_loras
    from ruinedfooocus.features.prompting.prompt_resources import RESOURCE

    # Prior text is context, never permission to load another adapter or restorer.
    context = [
        RESOURCE.sub("", parse_loras(value, "")[1]).replace("<facerestore>", "")
        for value in [source_prompt, *history]
    ]
    return "\n".join(value for value in [*context, instruction] if value) or instruction


def attach_metadata(data, metadata):
    descriptor = data.get(REQUEST_KEY)
    return dict(metadata, **{METADATA_KEY: deepcopy(descriptor)}) if descriptor else metadata
