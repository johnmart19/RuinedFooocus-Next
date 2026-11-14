"""Finite, explicit still-image handoffs over editor history and native workers.

This module captures one step at a time. It never acquires assets, starts a
worker, completes history, or executes an operation loop.
"""

from dataclasses import dataclass, replace
import json
import math
from uuid import uuid4

from ruinedfooocus.features.image_editing.mask_handoffs import (
    ResultMaskHandoff,
    inpaint_inputs,
)

KEY = "_image_workflow_step"
EDITOR_KEY = "_editor_operation"
ACTIONS = (
    "remove-background",
    "select-background",
    "generate-background",
    "compose",
    "prompt-edit",
    "inpaint",
    "upscale",
)
SAMPLING = ("generate-background", "prompt-edit", "inpaint")
PRESETS = {
    "background-replacement": dict(
        background_mode="select", composition_mode="SRC_OVER", image_resize_mode="preserve"
    ),
    "portrait-enhancement": dict(image_resize_mode="preserve", denoise=0.3),
}
SEQUENCES = {
    "background-replacement": (
        "remove-background",
        "background",
        "compose",
        "prompt-edit",
        "upscale",
    ),
    "portrait-enhancement": ("prompt-edit", "upscale"),
}
SAMPLING_FIELDS = {
    "base_model_name",
    "base_model_hash",
    "image_engine",
    "prompt",
    "negative",
    "loras",
    "style_selection",
    "custom_steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "clip_skip",
    "seed",
    "image_component_overrides",
    "native_edit_recipe",
    "qwen_local_edit_mode",
    "moss_image_edit_format",
}
ACTION_FIELDS = {
    "remove-background": {"rembg_native_model", "rembg_force"},
    "select-background": {"background_id"},
    "generate-background": SAMPLING_FIELDS,
    "compose": {
        "background_id",
        "foreground_id",
        "composition_mode",
        "composition_x",
        "composition_y",
        "composition_width",
        "composition_height",
    },
    "prompt-edit": SAMPLING_FIELDS | {"denoise"},
    "inpaint": SAMPLING_FIELDS | {"inpaint_denoise", "inpaint_mask_grow"},
    "upscale": {"upscaler"},
}
COMMON_FIELDS = {"target_width", "target_height", "image_resize_mode", "alpha_policy"}


def _json(value):
    """Own bounded JSON data, rejecting images, non-finite values and mutable aliases."""
    try:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("Image workflow settings must be bounded finite JSON values.") from error
    if len(raw) > 65536:
        raise ValueError("Image workflow settings exceed the 64 KiB limit.")
    return raw


def _integer(value, label, minimum=1, maximum=32768):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be a whole number from {minimum} through {maximum}.")
    return value


def _fraction(value, label, *, zero=False):
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or not (0 <= value <= 1 if zero else 0 < value <= 1)
    ):
        raise ValueError(
            f"{label} must be a finite value {'from zero' if zero else 'above zero'} through one."
        )
    return float(value)


def preset_values(name, overrides=None):
    """Persist only the finite scalar intent; resolve models and uploads on use."""
    if name not in PRESETS:
        raise ValueError("Choose background replacement or portrait enhancement.")
    values = dict(PRESETS[name])
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(values):
        raise ValueError("A workflow preset stores only its supported scalar intent.")
    values.update(overrides)
    if values.get("background_mode", "select") not in ("select", "generate"):
        raise ValueError("Choose a selected or generated background.")
    if values.get("composition_mode", "SRC_OVER") != "SRC_OVER":
        raise ValueError("The background preset uses native source-over composition.")
    if values["image_resize_mode"] not in ("preserve", "pad", "crop"):
        raise ValueError("Choose an explicit preserve, pad or crop image frame.")
    if "denoise" in values:
        values["denoise"] = _fraction(values["denoise"], "Denoise")
    return dict(name=name, **values)


@dataclass(frozen=True)
class WorkflowCursor:
    workflow_id: str
    session_id: str
    revision: int
    preset_json: str
    position: int = 0
    foreground_id: str | None = None
    background_id: str | None = None

    @property
    def preset(self):
        return json.loads(self.preset_json)

    @property
    def next_action(self):
        values = self.preset
        sequence = SEQUENCES[values["name"]]
        if self.position == len(sequence):
            return None
        action = sequence[self.position]
        return values["background_mode"] + "-background" if action == "background" else action


@dataclass(frozen=True)
class WorkflowStep:
    ticket: object
    action: str
    parameters_json: str
    mask: ResultMaskHandoff | None = None
    cursor: WorkflowCursor | None = None

    @property
    def parameters(self):
        return json.loads(self.parameters_json)


def begin_workflow(store, session_id, preset, *, owner, overrides=None):
    snapshot = store.snapshot(session_id, owner=owner)
    values = preset_values(preset, overrides)
    return WorkflowCursor(uuid4().hex, session_id, snapshot.revision, _json(values))


def _parameters(action, parameters, source):
    if not isinstance(parameters, dict) or set(parameters) - (
        ACTION_FIELDS[action] | COMMON_FIELDS
    ):
        raise ValueError("Use only settings owned by this image workflow step.")
    values = json.loads(_json(parameters))
    width = _integer(values.setdefault("target_width", source.frame.width), "Target width")
    height = _integer(values.setdefault("target_height", source.frame.height), "Target height")
    if width * height > 64 * 1024 * 1024:
        raise ValueError("The output target exceeds the 64-megapixel limit.")
    mode = values.setdefault("image_resize_mode", "preserve")
    if mode not in ("preserve", "pad", "crop"):
        raise ValueError("Choose an explicit preserve, pad or crop image frame.")
    alpha = values.setdefault("alpha_policy", "preserve")
    if alpha not in ("preserve", "native", "flatten-black"):
        raise ValueError("Choose preserve, native or explicit flatten-black alpha handling.")
    if (
        action in ("compose", "select-background", "generate-background")
        and alpha == "flatten-black"
    ):
        raise ValueError("Bind an explicitly flattened owned image before this workflow action.")
    if (
        action in ("inpaint", "prompt-edit")
        and mode == "preserve"
        and (width, height)
        != (
            source.frame.width,
            source.frame.height,
        )
    ):
        raise ValueError("Choose explicit pad or crop when changing an edit's image frame.")
    if action in ("remove-background", "compose") and (width, height) != (
        source.frame.width,
        source.frame.height,
    ):
        raise ValueError("Background removal and composition do not resize their base canvas.")
    if action == "prompt-edit":
        values["denoise"] = _fraction(values.get("denoise", 0.3), "Denoise")
    if action == "inpaint":
        values["inpaint_denoise"] = _fraction(values.get("inpaint_denoise", 1.0), "Inpaint denoise")
        values["inpaint_mask_grow"] = _integer(
            values.get("inpaint_mask_grow", 20), "Mask expansion", 0, 128
        )
    if action == "remove-background" and type(values.setdefault("rembg_force", False)) is not bool:
        raise ValueError("Force background removal must be true or false.")
    if action in SAMPLING:
        for key in ("prompt", "negative"):
            value = values.setdefault(key, "")
            if not isinstance(value, str) or len(value) > 16384:
                raise ValueError("Prompts and negatives must be bounded literal text.")
        values.setdefault("loras", [])
    if action == "upscale" and (
        not isinstance(values.get("upscaler"), str) or not values["upscaler"].strip()
    ):
        raise ValueError("Select a currently compatible native pixel upscaler.")
    if action == "upscale" and not {"target_width", "target_height"} <= set(parameters):
        raise ValueError("Capture both exact output target dimensions for the upscale step.")
    return values


def capture_step(
    store,
    session_id,
    revision,
    action,
    parameters,
    *,
    owner,
    mask=None,
    effective_plan=None,
    cursor=None,
    family=None,
):
    """Validate identities and freeze one step before worker capture/preparation."""
    if action not in ACTIONS:
        raise ValueError("Choose a supported still-image workflow step.")
    snapshot = store.snapshot(session_id, owner=owner)
    if snapshot.revision != revision:
        raise ValueError("The image workflow revision changed. Refresh the current image.")
    if cursor is not None and (
        not isinstance(cursor, WorkflowCursor)
        or cursor.session_id != session_id
        or cursor.revision != revision
        or cursor.next_action != action
    ):
        raise ValueError("This is not the next explicit step of the current image workflow.")
    if cursor is not None and isinstance(parameters, dict):
        defaults = {
            key: value
            for key, value in cursor.preset.items()
            if key in ACTION_FIELDS[action] | COMMON_FIELDS
        }
        parameters = defaults | parameters
    values = _parameters(action, parameters, snapshot.current)
    if action == "prompt-edit":
        captured_family = (effective_plan or {}).get("family")
        if family and captured_family and family != captured_family:
            raise ValueError("The selected edit family differs from the captured image plan.")
        values["edit_family"] = captured_family or family or ""
        if values.get("base_model_name") and not values["edit_family"]:
            raise ValueError(
                "Resolve the selected checkpoint's current image family before editing."
            )
    if cursor is not None and action == "compose":
        if cursor.background_id != snapshot.current.artifact_id or not cursor.foreground_id:
            raise ValueError("The workflow needs its completed foreground and background.")
        if values.get("foreground_id", cursor.foreground_id) != cursor.foreground_id:
            raise ValueError("The selected foreground is not from this workflow.")
        values["foreground_id"] = cursor.foreground_id
    additional = ()
    if "background_id" in values and "foreground_id" in values:
        raise ValueError("Choose one owned background or foreground, not both.")
    for key in ("background_id", "foreground_id"):
        if key in values:
            ref = store.artifact(session_id, values[key], revision=revision, owner=owner)
            if ref.kind != "image":
                raise ValueError("Backgrounds and foregrounds must be owned images.")
            additional = (ref.artifact_id,)
    if action in ("select-background", "compose") and not additional:
        raise ValueError("Select the owned background or foreground for this step.")
    if action == "select-background":
        for key, actual in (("target_width", ref.frame.width), ("target_height", ref.frame.height)):
            if key in parameters and values[key] != actual:
                raise ValueError("The selected background owns its exact image dimensions.")
            values[key] = actual
    if action == "inpaint":
        if not isinstance(mask, ResultMaskHandoff) or (
            mask.session_id,
            mask.revision,
            mask.source_id,
        ) != (session_id, revision, snapshot.current.artifact_id):
            raise ValueError("Use the selection mask captured for this current image revision.")
        inpaint_inputs(store, mask, owner=owner)
    elif mask is not None:
        raise ValueError("A result selection mask belongs only to the inpaint step.")
    metadata = dict(values)
    if mask is not None:
        metadata["result_mask"] = mask.metadata()
    if cursor is not None:
        metadata["workflow"] = dict(
            workflow_id=cursor.workflow_id, preset=cursor.preset, position=cursor.position
        )
    ticket = store.capture(
        session_id,
        revision,
        action,
        metadata,
        source_id=snapshot.current.artifact_id,
        mask_id=mask.mask_id if mask is not None else None,
        additional_image_ids=additional,
        effective_plan=effective_plan,
        owner=owner,
    )
    return WorkflowStep(ticket, action, _json(values), mask, cursor)


def _image(store, ticket, ref, owner, *, details=False):
    from ruinedfooocus.features.gallery.editing_workflow import working_image

    store.validate(ticket, owner=owner)
    image, information = working_image(ref.path)
    if image.size != ref.frame.size:
        raise ValueError("The prepared working image differs from its captured source frame.")
    store.validate(ticket, owner=owner)
    return (image, information) if details else image


def composition_inputs(step, store, *, owner):
    """Resolve both owned roles again at native capture, preserving the current canvas."""
    if not isinstance(step, WorkflowStep) or step.action != "compose":
        raise ValueError("Capture an owned composition step before resolving its image roles.")
    source = _image(store, step.ticket, step.ticket.source, owner)
    additional = _image(store, step.ticket, step.ticket.additional_images[0], owner)
    if "background_id" not in step.parameters:
        return [source, additional]
    from PIL import Image
    import numpy as np
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

    pixels, _ = prepare_reference(additional, source.size, "pad", multiple=1, preserve_alpha=True)
    background = Image.fromarray((pixels[0].numpy() * 255).round().astype(np.uint8))
    store.validate(step.ticket, owner=owner)
    return [background, source]


def worker_request(step, store, *, owner):
    """Translate to existing worker fields; the caller owns capture/preparation/queue."""
    store.validate(step.ticket, owner=owner)
    if step.action == "select-background":
        raise ValueError("Background selection uses selected_background, without compute.")
    values = step.parameters
    source, information = _image(store, step.ticket, step.ticket.source, owner, details=True)
    data = {key: value for key, value in values.items() if key in SAMPLING_FIELDS}
    data.update(
        task_type="process",
        image_number=1,
        generate_forever=False,
        input_image=source,
        main_view=None,
        inpaint_view=None,
        inpaint_toggle=False,
        automatic_upscale="Off",
        automatic_upscale_target="Generated image",
        aspect_ratios_selection="Custom...",
        custom_width=values["target_width"],
        custom_height=values["target_height"],
        image_resize_mode=values["image_resize_mode"],
        controlnet={},
        image_source=information,
    )
    if values["alpha_policy"] == "flatten-black":
        from PIL import Image

        flat = Image.new("RGBA", source.size, (0, 0, 0, 255))
        flat.alpha_composite(source)
        data["input_image"] = source = flat.convert("RGB")
    if step.action == "remove-background":
        data.update(
            controlnet=dict(type="rembg", force=values["rembg_force"]),
            rembg_force=values["rembg_force"],
            rembg_native_model=values.get("rembg_native_model"),
        )
    elif step.action == "generate-background":
        data.update(input_image=None, size_mode="target")
    elif step.action == "compose":
        data.update(
            controlnet=dict(type="image-composite"),
            input_image=None,
            composition_images=composition_inputs(step, store, owner=owner),
            **{key: value for key, value in values.items() if key.startswith("composition_")},
        )
    elif step.action == "prompt-edit":
        from ruinedfooocus.features.models.image_components import edit_control

        data["controlnet"] = edit_control(
            values["edit_family"],
            values.get("base_model_name", ""),
            values.get("moss_image_edit_format", "Disabled"),
            denoise=values["denoise"],
        )
    elif step.action == "inpaint":
        data.update(inpaint_inputs(store, step.mask, owner=owner))
        if values["alpha_policy"] == "flatten-black":
            data["main_view"] = source
        data.update(
            inpaint_denoise=values["inpaint_denoise"],
            inpaint_mask_grow=values["inpaint_mask_grow"],
        )
    elif step.action == "upscale":
        data.update(
            controlnet=dict(type="upscale", upscaler=values["upscaler"]),
            automatic_upscale=values["upscaler"],
            automatic_upscale_target="Input image only",
            size_mode="target",
            resolution_auto=False,
        )
    if step.action not in SAMPLING:
        # Pure processing must not borrow an image's inherited diffusion settings.
        data.update(prompt="", negative="", loras=[], style_selection=[])
    data.update({KEY: step, EDITOR_KEY: step.ticket})
    store.validate(step.ticket, owner=owner)
    return data


def selected_background(step, store, *, owner):
    """Return the owned selected artifact for success publication without compute."""
    if step.action != "select-background":
        raise ValueError("This step does not select a background image.")
    store.validate(step.ticket, owner=owner)
    ref = step.ticket.additional_images[0]
    return ref.path, dict(
        kind="processing",
        operation="select-background",
        computed=False,
        geometry=dict(
            policy="select-existing-background",
            source=list(step.ticket.source.frame.size),
            result=list(ref.frame.size),
        ),
        alpha_policy="selected background alpha",
        precision_conversion=(
            f"selected existing background: {ref.frame.dtype} {ref.frame.color_space}; original retained"
        ),
    )


def bind_effective_plan(data, plan, *, store=None, owner=None):
    """Bind the worker's final capture before preparation, superseding the draft ticket."""
    step = data.get(KEY)
    if step is None:
        return data
    if not isinstance(step, WorkflowStep) or data.get(EDITOR_KEY) != step.ticket:
        raise ValueError("Bind the final plan only to its matching image workflow ticket.")
    if store is None:
        from ruinedfooocus.features.gallery.editor_session import default_store

        store = default_store()
    store.validate(step.ticket, owner=owner)
    metadata = plan.metadata() if hasattr(plan, "metadata") else plan
    metadata = json.loads(_json(metadata))
    if step.action == "prompt-edit" and metadata is not None:
        family = metadata.get("family")
        if family and step.parameters["edit_family"] != family:
            raise ValueError(
                "The final image plan changed the captured edit family. Recapture the edit."
            )
    if step.ticket.effective_plan == metadata:
        return data
    ticket = store.bind_effective_plan(step.ticket, metadata, owner=owner)
    return dict(data, **{KEY: replace(step, ticket=ticket), EDITOR_KEY: ticket})


def execution_metadata(step, result_size, *, actual=None):
    """Record deliberate geometry/alpha policies alongside actual worker metadata."""
    values = step.parameters
    target = (values["target_width"], values["target_height"])
    if tuple(result_size) != target:
        raise ValueError(
            "The completed workflow output frame differs from its exact captured target."
        )
    result = json.loads(_json(actual or {}))
    if step.action not in SAMPLING:
        for key in (
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "seed",
            "prompt",
            "negative",
            "loras",
        ):
            result.pop(key, None)
        result["kind"] = "processing"
    result["workflow_action"] = step.action
    result["geometry"] = dict(
        policy=(
            "select-existing-background"
            if step.action == "select-background"
            else "exact-native-upscale-uniform-fit-pad"
            if step.action == "upscale"
            else values["image_resize_mode"]
        ),
        source=list(step.ticket.source.frame.size),
        result=list(result_size),
    )
    result["alpha_policy"] = (
        "new background alpha"
        if step.action in ("select-background", "generate-background")
        else "native foreground alpha multiplied by mask"
        if step.action == "remove-background"
        else values["alpha_policy"]
    )
    if step.action == "compose" and "background_id" in values:
        result["background_fit"] = dict(
            policy="uniform fit with transparent padding; no crop or stretch",
            source=list(step.ticket.additional_images[0].frame.size),
            result=list(step.ticket.source.frame.size),
        )
    if step.action != "select-background":
        conversions = [
            dict(
                image_id=ref.artifact_id,
                original_dtype=ref.frame.dtype,
                original_color_space=ref.frame.color_space,
                working_dtype="uint8",
                working_color_space="sRGB",
            )
            for ref in (step.ticket.source, *step.ticket.additional_images)
            if ref.frame.dtype != "uint8" or ref.frame.color_space != "sRGB"
        ]
        if conversions:
            result["working_images"] = conversions
            result["precision_conversion"] = (
                "automatic request-local 8-bit sRGB working images; original artifacts retained"
            )
    return result


def advance_workflow(cursor, step, snapshot):
    """Advance only after P3 reports this exact operation as successfully published."""
    if (
        not isinstance(cursor, WorkflowCursor)
        or step.cursor != cursor
        or step.action != cursor.next_action
        or snapshot.session_id != cursor.session_id
        or snapshot.revision != cursor.revision + 1
        or not any(row.get("operation_id") == step.ticket.operation_id for row in snapshot.steps)
    ):
        raise ValueError("Only the completed current workflow step can advance the workflow.")
    changes = dict(revision=snapshot.revision, position=cursor.position + 1)
    if step.action == "remove-background":
        changes["foreground_id"] = snapshot.current.artifact_id
    elif step.action in ("select-background", "generate-background"):
        changes["background_id"] = snapshot.current.artifact_id
    return replace(cursor, **changes)


def skip_background_removal(store, cursor, *, owner):
    """An already transparent source may advance without a fictitious completed step."""
    from ruinedfooocus.features.gallery.editing_workflow import working_image

    snapshot = store.snapshot(cursor.session_id, owner=owner)
    if cursor.next_action != "remove-background" or snapshot.revision != cursor.revision:
        raise ValueError("Only current background removal can be skipped.")
    ref = store.artifact(
        cursor.session_id, snapshot.current.artifact_id, revision=cursor.revision, owner=owner
    )
    if ref.frame.alpha == "none":
        raise ValueError("The current source has no existing transparency to preserve.")
    image, _ = working_image(ref.path)
    if image.getchannel("A").getextrema()[0] == 255:
        raise ValueError("The current source is opaque; background removal is not a no-op.")
    store.artifact(cursor.session_id, ref.artifact_id, revision=cursor.revision, owner=owner)
    return replace(cursor, position=cursor.position + 1, foreground_id=ref.artifact_id)


def finish_pixels(step, store, result, *, owner):
    """Retain source alpha for an unchanged edit frame before ordinary file output."""
    from PIL import Image
    import numpy as np

    store.validate(step.ticket, owner=owner)
    if isinstance(result, Image.Image):
        result_size = result.size
        has_alpha = "A" in result.getbands() or "transparency" in result.info
    else:
        array = np.asarray(result)
        if array.ndim != 3 or array.shape[-1] not in (3, 4) or not np.isfinite(array).all():
            raise ValueError("The image workflow returned invalid still-image pixels.")
        result_size = (array.shape[1], array.shape[0])
        has_alpha = array.shape[-1] == 4
    execution_metadata(step, result_size)
    if (
        step.action not in ("prompt-edit", "inpaint")
        or step.parameters["alpha_policy"] != "preserve"
        or has_alpha
    ):
        return result
    source = _image(store, step.ticket, step.ticket.source, owner)
    if source.getchannel("A").getextrema() == (255, 255):
        return result
    target = (step.parameters["target_width"], step.parameters["target_height"])
    if target != source.size:
        from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

        pixels, plan = prepare_reference(
            source, target, step.parameters["image_resize_mode"], multiple=1, preserve_alpha=True
        )
        alpha = pixels[0, ..., 3].numpy()
        expected = (plan["width"], plan["height"])
    else:
        alpha = np.array(source.getchannel("A"), dtype=np.float32) / 255
        expected = source.size
    if isinstance(result, Image.Image):
        if result.size != expected:
            raise ValueError(
                "The completed edit frame differs from the captured source-alpha frame."
            )
        output = result.convert("RGBA").copy()
        output.putalpha(Image.fromarray((alpha * 255).round().astype(np.uint8)))
    else:
        value = np.asarray(result)
        if (
            value.ndim != 3
            or value.shape[-1] not in (3, 4)
            or tuple(value.shape[:2]) != (expected[1], expected[0])
            or not np.isfinite(value).all()
        ):
            raise ValueError("The completed edit has no valid captured source-alpha frame.")
        channel = alpha.astype(value.dtype)
        if np.issubdtype(value.dtype, np.integer):
            channel = (alpha * np.iinfo(value.dtype).max).round().astype(value.dtype)
        output = np.concatenate((value[..., :3], channel[..., None]), axis=-1)
    store.validate(step.ticket, owner=owner)
    return output


def finish_worker_output(data, result, *, store=None, owner=None):
    """Typed pre-save hook for the existing worker; unrelated jobs pass through."""
    step = data.get(KEY)
    if step is None:
        return result
    if not isinstance(step, WorkflowStep) or data.get(EDITOR_KEY) != step.ticket:
        raise ValueError("The image workflow output has no matching owned editor ticket.")
    if store is None:
        from ruinedfooocus.features.gallery.editor_session import default_store

        store = default_store()
    return finish_pixels(step, store, result, owner=owner)
