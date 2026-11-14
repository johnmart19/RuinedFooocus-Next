from ruinedfooocus.core.paths import resource_path
import threading
import gc
import torch
import math
import time
import pathlib
from pathlib import Path
import traceback
from concurrent.futures import CancelledError
from functools import wraps
from contextlib import contextmanager
from ruinedfooocus.core.task_errors import describe_error

import json
import ruinedfooocus.app.state as shared
import random

from ruinedfooocus.features.prompting.prompt_processing import (
    process_metadata,
    process_prompt,
    parse_loras,
)
from ruinedfooocus.features.image_editing.shift_attention import shift_attention
from ruinedfooocus.features.models.lora_names import merge_loras

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from ruinedfooocus.core.util import (
    generate_temp_filename,
    TimeIt,
    get_checkpoint_hashes,
    get_lora_hashes,
)
import ruinedfooocus.features.generation.pipelines as pipelines
import ruinedfooocus.features.image_editing.controlnet as control_tools
import comfy.model_management as model_management
from ruinedfooocus.app.state import settings
from ruinedfooocus.app.resource_lifecycle import (
    admission,
    registry,
    current_job,
    current_owner,
    job_scope,
    service_scope,
    cleanup_interrupt,
    service_pipeline,
)

task_lock = threading.Lock()
api_jobs = {}
_active_job = None
_startup_preloaded = False


def interrupt_processing(task_id=None):
    """Cancel only this authenticated UI session's selected/latest job."""
    if not registry.cancel(task_id):
        return False
    item = _active_job
    if item is not None and item.cancelled.is_set() and item.state.get("native_active"):
        item.state["native_interrupt"] = True
        model_management.interrupt_current_processing(True)
    return True


def cancel_owner(owner_prefix):
    """Called with server-authenticated login identity during logout."""
    return registry.cancel_owner(owner_prefix)


def check_interrupt(gen_data):
    job = gen_data.get("_api_job")
    item = gen_data.get("_job") or current_job()
    if (item is not None and item.cancelled.is_set()) or (
        job is not None and job.cancelled.is_set()
    ):
        raise model_management.InterruptProcessingException()
    if item is not None and item.state.get("native_active"):
        model_management.throw_exception_if_processing_interrupted()


def _publish_output(event):
    add_result(*event)


def validate_editor_data(data, *, start=False):
    if data.get("_editor_operation") is None:
        return None
    from ruinedfooocus.features.gallery.editor_session import validate_request

    ticket = validate_request(data, start=start)
    item = data.get("_job") or current_job()
    if item is not None:
        item.state["_editor_request"] = {"_editor_operation": ticket}
    return ticket


def _source_has_opaque_alpha(source):
    """An all-opaque channel carries no transparency for an RGB utility to lose."""
    from PIL import Image

    from ruinedfooocus.features.image_processing.native_image_formats import (
        needs_preview,
        read_image,
    )

    if needs_preview(source.path):
        import numpy as np

        pixels = read_image(source.path).pixels
        if pixels.ndim != 3 or pixels.shape[-1] != 4:
            return False
        maximum = 1.0 if pixels.dtype.kind == "f" else np.iinfo(pixels.dtype).max
        return bool(np.all(pixels[..., 3] == maximum))
    with Image.open(source.path) as image:
        return image.convert("RGBA").getchannel("A").getextrema() == (255, 255)


def stage_editor_result(data, result):
    """Retain scalar publication evidence, never another in-memory image history."""
    ticket = data.get("_editor_operation")
    if ticket is None:
        return
    from ruinedfooocus.features.gallery.editor_session import describe_frame
    from ruinedfooocus.features.image_processing.native_image_formats import image_metadata

    check_interrupt(data)
    validate_editor_data(data)
    if ticket.operation == "segment":
        from pathlib import Path

        from ruinedfooocus.features.gallery.editor_session import default_store
        from ruinedfooocus.features.generation.image_output_batches import validate_artifacts

        validate_artifacts(data, result)
        item = data.get("_job") or current_job()
        if item is None:
            raise RuntimeError("Editor segmentation publication requires its owned worker job.")
        if item.state.get("_editor_publication") is not None:
            raise ValueError("Only one editor mutation is allowed per worker job.")
        output = data.get("segment_output", "cutout")
        if output not in ("mask", "cutout"):
            raise ValueError("Choose native segmentation mask or cutout output.")
        paths = [str(getattr(value, "original", value)) for value in result]
        results = default_store().capture_derived_results(ticket, paths, owner=item.owner)
        execution = dict(
            kind="processing",
            operation="segment",
            output_count=len(results),
            native_vision_source=dict(requested_model=Path(data["native_vision_model"]).name),
        )
        if paths:
            metadata = image_metadata(paths[0]).get("parameters", {})
            metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
            execution["native_vision_source"].update(metadata.get("native_vision_source") or {})
        if getattr(result, "plan", None) is not None:
            execution["native_output_group"] = dict(
                kind=result.plan.kind,
                count=len(results),
                completed_generations=result.completed_generations,
            )
        execution["native_vision_source"].update(output=output, output_count=len(results))
        item.state["_editor_request"] = {"_editor_operation": ticket}
        item.state["_editor_publication"] = dict(
            kind="segment-results", results=results, output=output, execution=execution
        )
        return
    if not result or len(result) != 1:
        raise ValueError("An editor operation must finish with exactly one image artifact.")
    item = data.get("_job") or current_job()
    if item is None:
        raise RuntimeError("Editor publication requires its owned worker job.")
    original = getattr(result[0], "original", result[0])
    frame = describe_frame(original)
    parameters = image_metadata(original).get("parameters", {})
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    execution = dict(parameters) if isinstance(parameters, dict) else {}
    step = data.get("_image_workflow_step")
    if step is not None:
        from ruinedfooocus.features.image_editing.image_workflows import execution_metadata

        execution = execution_metadata(step, frame.size, actual=execution)
    else:
        processing = data.get("_sampling_captured") is False
        execution["kind"] = "processing" if processing else "generation"
        execution["operation"] = ticket.operation
        execution["geometry"] = dict(
            policy=data.get("image_resize_mode", "preserve"),
            source=list(ticket.source.frame.size),
            result=list(frame.size),
        )
        from ruinedfooocus.features.image_processing.still_image_utilities import ADJUSTMENT_KINDS

        adjustment = ticket.operation in ADJUSTMENT_KINDS
        if adjustment:
            execution["geometry"]["policy"] = "native " + str(
                data.get("transform_operation", ticket.operation)
            )
            if ticket.source.frame.dtype != frame.dtype:
                execution["precision_conversion"] = (
                    f"native adjustment output export: {ticket.source.frame.dtype} to {frame.dtype}"
                )
        if ticket.source.frame.alpha != "none" and frame.alpha == "none":
            if (
                ticket.operation == "image-mask"
                and execution.get("image_utility_source", {}).get("mask_role")
                == "inpaint-selection"
            ):
                execution["alpha_policy"] = "explicit grayscale selection mask extraction"
            elif not _source_has_opaque_alpha(ticket.source):
                raise ValueError("This image operation did not preserve its source alpha.")
            else:
                validate_editor_data(data)
                execution["alpha_policy"] = "discard redundant fully opaque source alpha"
    source_conversion = data.get("image_source", {}).get("conversion")
    if source_conversion and source_conversion != "none":
        execution["precision_conversion"] = source_conversion
    item.state["_editor_request"] = {"_editor_operation": ticket}
    if item.state.get("_editor_publication") is not None:
        raise ValueError("Only one editor mutation is allowed per worker job.")
    item.state["_editor_publication"] = (str(original), execution)


def publish_editor_result(item):
    """Called under registry.condition at the existing terminal publication boundary."""
    from ruinedfooocus.features.gallery.editor_session import complete_request, default_store

    pending = item.state.get("_editor_publication")
    original = item.state.get("_editor_original")
    if pending is None and original is None:
        return None
    item.check()
    if isinstance(pending, dict) and pending.get("kind") == "segment-results":
        snapshot = default_store().complete_derived_results(
            item.state["_editor_request"]["_editor_operation"],
            pending["results"],
            output=pending["output"],
            execution=pending["execution"],
            owner=item.owner,
            check_cancelled=item.check,
        )
    elif pending is not None:
        path, execution = pending
        snapshot = complete_request(
            item.state["_editor_request"],
            path,
            execution,
            owner=item.owner,
            check_cancelled=item.check,
        )
    else:
        path, execution = original
        snapshot = default_store().start(path, provenance=execution, owner=item.owner)
    value = dict(status="completed", snapshot=snapshot.to_dict())
    item.state["editor_result"] = value
    item.state.pop("_editor_publication", None)
    item.state.pop("_editor_original", None)
    if item.external is not None:
        item.external.editor_result = value
    return value


def stage_editor_original(path):
    from ruinedfooocus.features.gallery.editor_session import describe_frame
    from ruinedfooocus.features.gallery.image_metadata import read_metadata
    from ruinedfooocus.features.image_processing.native_image_formats import image_metadata

    item = current_job()
    if item is None:
        raise RuntimeError("A new editor image requires its owned worker job.")
    item.check()
    describe_frame(path)
    if item.state.get("_editor_publication") or item.state.get("_editor_original"):
        raise ValueError("Only one image mutation is allowed per Chat turn.")
    item.state["_editor_original"] = (
        str(path),
        read_metadata(image_metadata(path).get("parameters")) or {},
    )


def discard_editor_result(item, *, status):
    if status == "error":
        status = "failed"
    original = item.state.pop("_editor_original", None)
    request = item.state.get("_editor_request")
    if request is None or item.state.get("editor_result", {}).get("status") == "completed":
        if original is not None:
            item.state["editor_result"] = dict(status=status, snapshot=None)
        return
    from ruinedfooocus.features.gallery.editor_session import cancel_request, fail_request

    function = cancel_request if status in ("cancelled", "stale") else fail_request
    function(request, owner=item.owner)
    item.state.pop("_editor_publication", None)
    item.state["editor_result"] = dict(status=status, snapshot=None)


def publish_image_tool_error(item, continuation, error):
    from ruinedfooocus.features.chat.chat_tools import image_tool_error

    with registry.condition:
        if item.terminal or item.state.get("image_tool_terminal"):
            return
        terminal = image_tool_error(
            continuation.request.call_id,
            continuation.request.name,
            error,
            cancelled=item.cancelled.is_set(),
        )
        add_result(item.task_id, "image_tool_result", terminal)
        item.state["image_tool_terminal"] = True


def editor_result(task_id):
    """Owner-protected bounded status for the existing UI terminal callback."""
    from copy import deepcopy

    item = registry.get(task_id)
    with registry.condition:
        return deepcopy(item.state.get("editor_result"))


def image_stage(function):
    @wraps(function)
    def run(gen_data):
        from ruinedfooocus.features.image_processing import still_image_utilities as depth_utilities

        item = gen_data.get("_job") or current_job()
        if item is None:
            raise RuntimeError("Image execution requires an owned worker job")
        gen_data = dict(gen_data, _job=item)
        from ruinedfooocus.features.models.effective_image_plan import get_effective_image_plan

        plan = get_effective_image_plan(gen_data)
        if plan is not None:
            plan.verify_request(gen_data)
        if gen_data.get("_editor_operation") is not None:
            validate_editor_data(gen_data, start=True)
        from ruinedfooocus.features.generation.sdcpp_pipeline import ExecutionHooks, HOOKS_KEY

        gen_data[HOOKS_KEY] = ExecutionHooks(
            lambda: check_interrupt(gen_data),
            lambda request, candidate: admission.lease(item, ("compute:shared", "service:sdcpp")),
        )
        from ruinedfooocus.features.generation.image_preparation import image_control_data
        from ruinedfooocus.features.image_processing.image_composition import handles

        control = control_tools.get_settings(gen_data) or {}
        if control.get("type") not in (
            "detect",
            "segment",
            "depth-map",
            "restore-seedvr2",
            "latent-compose",
            *depth_utilities.KINDS,
        ) and not handles(control.get("type")):
            control = control_tools.get_settings(image_control_data(gen_data)) or {}
        group = control.get("type") if control.get("type") in ("rembg", "upscale") else "image"
        if group == "rembg" and gen_data.get("rembg_native_model"):
            group = "image"
        from ruinedfooocus.features.prompting.local_model import prompt_preparation_scope

        with (
            prompt_preparation_scope(item, gen_data),
            service_scope(group, shared.state, item=item),
        ):
            item.state["native_active"] = group != "rembg"
            try:
                result = function(gen_data)
                if gen_data.get("_editor_operation") is not None:
                    stage_editor_result(gen_data, result)
                return result
            finally:
                item.state.pop("native_active", None)
                cleanup_interrupt(item)

    return run


@contextmanager
def chat_stage(gen_data=None):
    item = (gen_data or {}).get("_job") or current_job()
    standalone = item is None
    if standalone:
        item = registry.create()
    try:
        with job_scope(item), service_scope("chat", shared.state, item=item):
            yield
    finally:
        if standalone:
            registry.finish(item)


def chat_operation(function):
    @wraps(function)
    def run(*args, **kwargs):
        with chat_stage():
            return function(*args, **kwargs)

    return run


@contextmanager
def image_handoff(gen_data, chat, original_model, original_vision):
    """Inline compact handoff; retain conversation/settings, never double-queue."""
    with chat_stage(gen_data):
        item = current_job()
        prepared = gen_data.get("_chat_preparation") or getattr(chat, "_prepared_request", None)
        chat.unload()
        error = None
        try:
            yield
        except BaseException as caught:
            error = caught
            raise
        finally:
            shared.state["pipeline"] = chat
            if item.cancelled.is_set() or isinstance(error, RuntimeError):
                gen_data["_chat_restoration_pending"] = str(original_model)
            else:
                options = {"prepared": prepared} if prepared is not None else {}
                chat.load_base_model(original_model, vision=original_vision, **options)


def _preview_path(gen_data):
    paths = gen_data.get("_image_capture_paths") or shared.path_manager
    base = Path(paths.model_paths["temp_preview_path"])
    item = gen_data.get("_job") or current_job()
    token = item.identity.token if item is not None else str(gen_data.get("task_id", "inline"))
    return str(base.with_name(f"{base.stem}-{token}{base.suffix}"))


def is_sha256_hash(input_string):
    # Check if the string is exactly 64 characters long
    if len(input_string) != 64:
        return False
    # Check if the string consists only of hexadecimal characters
    if not all(c in "0123456789abcdefABCDEF" for c in input_string):
        return False
    return True


@image_stage
def _process(gen_data):
    from ruinedfooocus.features.image_processing import still_image_utilities as depth_utilities

    check_interrupt(gen_data)
    from ruinedfooocus.features.models.effective_image_plan import get_effective_image_plan

    effective_plan = get_effective_image_plan(gen_data)

    def metadata_hashes(category, name):
        models = gen_data.get("_image_capture_models")
        if models is None:
            return (get_checkpoint_hashes if category == "checkpoints" else get_lora_hashes)(name)
        from ruinedfooocus.core.util import _get_model_hashes

        cache = getattr(models, "cache_paths", {}).get(category)
        return (
            _get_model_hashes(Path(cache) / Path(name).name)
            if cache is not None
            else {"SHA256": ""}
        )

    from ruinedfooocus.features.image_processing import latent_composition

    if latent_composition.handles(control_tools.get_settings(gen_data).get("type")):
        result = latent_composition.process(gen_data, lambda: check_interrupt(gen_data))
        return latent_composition.validate_result(gen_data, result)
    res = []
    metadatastrings = []
    from ruinedfooocus.features.generation.request_metadata import (
        metadata_intent,
        restore_utility_metadata,
    )

    from ruinedfooocus.features.image_processing import image_composition

    composition_utility = image_composition.handles(
        control_tools.get_settings(gen_data).get("type")
    )
    native_utility = composition_utility or control_tools.get_settings(gen_data).get(
        "type", ""
    ).lower() in (
        "detect",
        "segment",
        "depth-map",
        "restore-seedvr2",
        *depth_utilities.KINDS,
    )
    if composition_utility:
        gen_data = image_composition.worker_data(gen_data)
        imported = None
    elif native_utility:
        if depth_utilities.is_utility(control_tools.get_settings(gen_data).get("type")):
            worker_data = depth_utilities.worker_data
        elif control_tools.get_settings(gen_data).get("type", "").lower() == "restore-seedvr2":
            from ruinedfooocus.features.image_processing.native_seedvr2 import worker_data
        else:
            from ruinedfooocus.features.image_processing.native_vision import worker_data

        gen_data = worker_data(gen_data)
        imported = None
    else:
        gen_data, imported = metadata_intent(gen_data)
    sampling = not native_utility and control_tools.get_settings(gen_data).get(
        "type", ""
    ).lower() not in ("rembg", "upscale", "detect", "segment", "depth-map", "restore-seedvr2")
    if sampling and gen_data.get("_sampling_captured") is False:
        raise ValueError("This utility request needs a new generation submission before sampling.")
    captured_loras = (
        effective_plan.loras_for_prompt(gen_data["prompt"])
        if sampling and effective_plan is not None
        else None
    )
    # Imported model/control intent chooses the defaults; imported sampling wins later.
    initial_family = (
        effective_plan.family
        if sampling and effective_plan is not None
        else shared.models.get_model_base(
            shared.models.get_models_by_path("checkpoints", gen_data.get("base_model_name", ""))
        )
        if sampling
        else None
    )
    if sampling and not gen_data.get("_sampling_captured"):
        gen_data = shared.performance_settings.apply(gen_data, initial_family)
    if (
        not native_utility
        and "_api_job" not in gen_data
        and gen_data.get("task_type") != "tool_call"
    ):
        component_snapshot = gen_data.get("image_component_overrides")
        gen_data = (
            process_metadata(gen_data) if sampling else restore_utility_metadata(gen_data, imported)
        )
        from ruinedfooocus.features.image_editing.reference_adapters import (
            omit_inactive_ui_reference,
        )

        gen_data = omit_inactive_ui_reference(gen_data, control_tools.get_settings(gen_data))
        if component_snapshot is not None:
            gen_data["image_component_overrides"] = component_snapshot

    route_data = (
        dict(gen_data, prompt="") if gen_data.get("_bound_image_tool") is True else gen_data
    )
    if effective_plan is not None and effective_plan.checkpoint_path:
        route_data = dict(route_data, base_model_name=effective_plan.checkpoint_path)
    pipeline = pipelines.update(route_data)
    if "image_engine_route" in route_data:
        gen_data["image_engine_route"] = route_data["image_engine_route"]
    if pipeline == None:
        print(f"ERROR: No pipeline")
        return
    try:
        # See if pipeline wants to pre-parse gen_data
        gen_data = pipeline.parse_gen_data(gen_data)
    except:
        pass

    image_number = gen_data["image_number"]

    loras = []

    for lora_data in gen_data["loras"] if sampling and gen_data["loras"] is not None else []:
        if is_sha256_hash(lora_data[0]):
            hash = lora_data[0]
        else:
            hash = None

        w, l = lora_data[1].split(" - ", 1)
        if not l == "None":
            loras.append({"name": l, "weight": float(w), "hash": hash})

    if sampling:
        if "silent" not in gen_data:
            _publish_output(
                [
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Loading base model: {gen_data['base_model_name']}", None),
                ]
            )

        if "base_model_hash" not in gen_data:
            gen_data["base_model_hash"] = metadata_hashes(
                "checkpoints",
                effective_plan.checkpoint_path
                if effective_plan is not None
                else gen_data["base_model_name"],
            )["SHA256"]

        pipeline.load_base_model(
            effective_plan.checkpoint_path
            if effective_plan is not None and "sdcpp" not in pipeline.pipeline_type
            else gen_data["base_model_name"],
            hash=gen_data.get("base_model_hash", None),
            **(
                {"component_overrides": gen_data["image_component_overrides"]}
                if "image_component_overrides" in gen_data and "sdxl" in pipeline.pipeline_type
                else {}
            ),
            **(
                {"prepared_data": gen_data}
                if "sdxl" in pipeline.pipeline_type and "sdcpp" not in pipeline.pipeline_type
                else {}
            ),
        )

        if "silent" not in gen_data:
            _publish_output([gen_data["task_id"], "preview", (-1, "Loading LoRA models ...", None)])
    elif not native_utility:
        # Retain the utility's initializer without resolving a diffusion checkpoint.
        controls = control_tools.get_settings(gen_data)
        utility_name = controls.get("upscaler") or gen_data.get("rembg_native_model")
        pipeline.load_base_model(utility_name or controls.get("type"), hash=None)

    steps = gen_data["custom_steps"] if sampling else 1
    gen_data["steps"] = steps

    resolutions = gen_data.get("_image_capture_resolutions") or getattr(
        shared, "resolution_settings", None
    )
    if not sampling:
        source = gen_data.get("input_image")
        width, height = (
            image_composition.captured(gen_data).output_size
            if composition_utility
            else source.size
            if source is not None
            else (1, 1)
        )
        if gen_data.get("controlnet", {}).get("type") == "restore-seedvr2":
            width, height = gen_data["custom_width"], gen_data["custom_height"]
        gen_data["width"], gen_data["height"] = width, height
    elif gen_data["aspect_ratios_selection"] == resolutions.CUSTOM_RESOLUTION:
        width, height = (gen_data["custom_width"], gen_data["custom_height"])
    else:
        width, height = resolutions.aspect_ratios[gen_data["aspect_ratios_selection"]]

    if "width" in gen_data:
        width = gen_data["width"]
    else:
        gen_data["width"] = width
    if "height" in gen_data:
        height = gen_data["height"]
    else:
        gen_data["height"] = height

    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_PLAN_KEY, SizePlan

    size_plan = gen_data.get(SIZE_PLAN_KEY)
    if isinstance(size_plan, SizePlan):
        width, height = size_plan.content
        gen_data["width"], gen_data["height"] = width, height

    try:
        seed = int(gen_data["seed"])
    except:
        seed = -1
    max_seed = 2**32
    if not isinstance(seed, int) or seed == -1:
        seed = random.randint(0, max_seed)

    all_steps = steps * max(image_number, 1)
    with open(resource_path("render.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    status = random.choice(lines)
    status = f"{status}"

    def callback(step, x0, x, total_steps, y):
        global status
        check_interrupt(gen_data)
        if "silent" in gen_data:
            return

        # If we only generate 1 image, skip the last preview
        if (
            (not gen_data["generate_forever"])
            and shared.state["preview_total"] == 1
            and steps == step
        ):
            return

        done_steps = i * steps + step
        try:
            status
        except NameError:
            status = None

        # Get status based on time (for very slow generations)
        status = lines[int(time.time() // 10) % len(lines)]

        grid_xsize = math.ceil(math.sqrt(shared.state["preview_total"]))
        grid_ysize = math.ceil(shared.state["preview_total"] / grid_xsize)
        grid_max = max(grid_xsize, grid_ysize)
        pwidth = int(width * grid_xsize / grid_max)
        pheight = int(height * grid_ysize / grid_max)
        if shared.state["preview_grid"] is None:
            shared.state["preview_grid"] = Image.new("RGB", (pwidth, pheight))
        if y is not None:
            if isinstance(y, Image.Image):
                image = y
            elif isinstance(y, str):
                image = Image.open(y)
            else:
                image = Image.fromarray(y)
            grid_xpos = int((shared.state["preview_count"] % grid_xsize) * (pwidth / grid_xsize))
            grid_ypos = int(
                math.floor(shared.state["preview_count"] / grid_xsize) * (pheight / grid_ysize)
            )
            image = image.resize((int(width / grid_max), int(height / grid_max)))
            shared.state["preview_grid"].paste(image, (grid_xpos, grid_ypos))

            shared.state["preview_grid"].save(
                _preview_path(gen_data),
                optimize=True,
                quality=35 if step < total_steps else 70,
            )

            preview = _preview_path(gen_data)
        else:
            preview = None

        if gen_data.get("task_type", None) != "tool_call":
            _publish_output(
                [
                    gen_data["task_id"],
                    "preview",
                    (
                        int(
                            100
                            * (gen_data["index"][0] + done_steps / all_steps)
                            / max(gen_data["index"][1], 1)
                        ),
                        f"{status} - {step}/{total_steps}",
                        preview,
                    ),
                ]
            )

    # TODO: this should be an "inital ok gen_data" at the beginning of the function
    if "input_image" not in gen_data:
        gen_data["input_image"] = None
    if "main_view" not in gen_data:
        gen_data["main_view"] = None

    stop_batch = False
    from ruinedfooocus.features.generation.image_output_batches import (
        PLAN_KEY,
        ImageOutputBatch,
        ImageBatchArtifacts,
    )

    output_plan = gen_data.get(PLAN_KEY)
    completed_groups = 0
    prompt_styles = gen_data["style_selection"] if sampling else []
    if control_tools.get_settings(gen_data).get("preserve_details"):
        prompt_styles = []
    for i in range(max(image_number, 1)):
        gen_data["seed"] = abs(seed)  # Update seed
        if "_native_tensor_plan" in gen_data:
            gen_data["_native_tensor_generation_index"] = i
        start_step = 0
        denoise = None
        with TimeIt("Pipeline process"):
            try:
                check_interrupt(gen_data)
                p_txt, n_txt = (
                    ("", "")
                    if not sampling
                    else process_prompt(
                        prompt_styles, gen_data["prompt"], gen_data["negative"], gen_data
                    )
                )
                if not sampling:
                    used_loras = []
                else:
                    distance = float(i) / max(image_number - 1.0, 1.0)
                    p_txt = shift_attention(p_txt, distance)
                    n_txt = shift_attention(n_txt, distance)
                    check_interrupt(gen_data)
                    # Load LoRAs
                    parsed_loras, p_txt, n_txt = parse_loras(p_txt, n_txt)
                    used_loras = merge_loras(
                        loras,
                        parsed_loras,
                        (gen_data.get("_image_capture_models") or shared.models).get_names("loras")
                        if parsed_loras
                        else [],
                    )
                    if "_native_minimax_image" in gen_data and used_loras:
                        raise ValueError(
                            "Clear inline LoRAs for the native MiniMax image PDD graph."
                        )
                    if (
                        effective_plan is not None
                        and "sdxl" in pipeline.pipeline_type
                        and "sdcpp" not in pipeline.pipeline_type
                    ):
                        pipeline.load_loras(used_loras, captured_loras=captured_loras)
                    else:
                        pipeline.load_loras(used_loras)
                gen_data["positive_prompt"] = p_txt
                gen_data["negative_prompt"] = n_txt

                imgs = pipeline.process(
                    gen_data=gen_data,
                    callback=callback,
                )
            except model_management.InterruptProcessingException:
                shared.state["interrupted"] = True
                stop_batch = True
                imgs = []
            except Exception as iex:
                stop_batch = True
                imgs = []
                traceback.print_exc()
                print(f"ERROR: {iex}")
                if "_api_job" in gen_data:
                    raise
                else:
                    add_result(gen_data["task_id"], "error", describe_error(iex))

        if not stop_batch:
            if isinstance(imgs, ImageOutputBatch):
                if imgs.plan != output_plan:
                    raise ValueError(
                        "The native image output group was not captured for this request."
                    )
                completed_groups += 1
            elif output_plan is not None:
                raise ValueError(
                    "The native pipeline did not return its captured image output group."
                )

        for output_index, x in enumerate(imgs):
            if gen_data.get("_image_workflow_step") is not None:
                from ruinedfooocus.features.image_editing.image_workflows import (
                    finish_worker_output,
                )

                x = finish_worker_output(gen_data, x)
            from ruinedfooocus.features.image_processing.native_image_formats import (
                advanced_output,
                format_settings,
                write_image,
            )

            export_settings = format_settings(gen_data)
            paths = gen_data.get("_image_capture_paths") or shared.path_manager
            folder = paths.model_paths["temp_outputs_path"]
            local_temp_filename = generate_temp_filename(
                folder=folder,
                extension=export_settings["output_format"],
            )
            dir_path = Path(local_temp_filename).parent
            dir_path.mkdir(parents=True, exist_ok=True)
            metadata = None
            prompt = (
                {
                    "Prompt": p_txt,
                    "Negative": n_txt,
                    "raw_prompt": gen_data.get("raw_prompt", gen_data.get("prompt", p_txt)),
                    "raw_negative": gen_data.get("raw_negative", gen_data.get("negative", n_txt)),
                    "style_provenance": gen_data.get("style_provenance", {}),
                    "steps": steps,
                    "cfg": gen_data["cfg"],
                    # Pipelines update the size when img2img/edits keep the source aspect.
                    "width": gen_data.get("width", width),
                    "height": gen_data.get("height", height),
                    "seed": abs(seed),
                    "sampler_name": gen_data["sampler_name"],
                    "scheduler": gen_data["scheduler"],
                    "base_model_name": gen_data["base_model_name"],
                    "base_model_hash": gen_data["base_model_hash"],
                    "loras": [
                        [
                            f"{metadata_hashes('loras', lora['name'])['SHA256']}",
                            f"{lora['weight']} - {lora['name']}",
                        ]
                        for lora in used_loras
                    ],
                    "start_step": start_step,
                    "denoise": gen_data.get("sampling_denoise", denoise),
                    "clip_skip": gen_data["clip_skip"],
                    "comment": gen_data.get(
                        "_image_capture_settings", settings.default_settings
                    ).get("meta_comment", ""),
                    "software": "RuinedFooocus",
                }
                if sampling
                else dict(
                    software="RuinedFooocus",
                    kind="processing",
                    width=gen_data.get("width", width),
                    height=gen_data.get("height", height),
                    controlnet=dict(control_tools.get_settings(gen_data)),
                )
            )
            if "image_component_overrides" in gen_data:
                prompt["image_component_overrides"] = dict(gen_data["image_component_overrides"])
            if gen_data.get("rembg_native_model"):
                prompt["rembg_native_model"] = gen_data["rembg_native_model"]
                if gen_data.get("native_background_source"):
                    prompt["native_background_source"] = dict(gen_data["native_background_source"])
            if gen_data.get("native_vision_source"):
                from ruinedfooocus.features.image_processing.native_vision import FIELDS

                prompt = dict(
                    software="RuinedFooocus",
                    width=width,
                    height=height,
                    controlnet=dict(gen_data["controlnet"]),
                )
                if gen_data["controlnet"].get("type", "").lower() == "depth-map":
                    prompt["controlnet"] = {"type": "depth-map"}
                else:
                    prompt.update((key, gen_data[key]) for key in FIELDS if key in gen_data)
                    if prompt.get("native_vision_model"):
                        prompt["native_vision_model"] = Path(prompt["native_vision_model"]).name
                prompt["native_vision_source"] = dict(gen_data["native_vision_source"])
            if gen_data.get("native_seedvr2_source"):
                from ruinedfooocus.features.image_processing.native_seedvr2 import FIELDS

                prompt = dict(
                    software="RuinedFooocus",
                    width=width,
                    height=height,
                    controlnet={"type": "restore-seedvr2"},
                    seed=gen_data["seed"],
                    native_seedvr2_source=dict(gen_data["native_seedvr2_source"]),
                )
                prompt.update((key, gen_data[key]) for key in FIELDS if key in gen_data)
                for key in ("seedvr2_model", "seedvr2_vae"):
                    prompt[key] = Path(prompt[key]).name
            if gen_data.get("native_tensor_source"):
                prompt["native_tensor_source"] = dict(gen_data["native_tensor_source"])
                for key in (
                    "native_tensor_family",
                    "native_tensor_workflow",
                    "native_latent_denoise",
                    "native_export_latent",
                    "native_export_conditioning",
                    "native_export_handoff",
                ):
                    if key in gen_data:
                        prompt[key] = gen_data[key]
            from copy import deepcopy

            for key in (
                "sampling_defaults_source",
                "model_recipe_path",
                "model_recipe_sha256",
                "model_recipe_overrides",
                "clip_layer_source",
                "cascade_stage_b",
                "bernini_low_model",
                "bernini_split_step",
                "bernini_lightx2v",
                "bernini_high_lora_strength",
                "bernini_low_lora_strength",
                "bernini_ref_max_size",
                "native_bernini_image",
                "bernini_source_frame",
                "cascade_b_steps",
                "cascade_b_cfg",
                "cascade_b_sampler",
                "cascade_b_scheduler",
                "cascade_stage_b_source",
                "cascade_compression",
                "cascade_source_denoise",
                "native_cascade",
                "cascade_super_resolution",
                "cascade_super_resolution_model",
                "cascade_control_strength",
                "cascade_control_start",
                "cascade_control_end",
                "cascade_control_denoise",
                "cascade_vision_enabled",
                "cascade_vision_model",
                "cascade_vision_strengths",
                "cascade_vision_crop",
                "native_cascade_guidance",
                "sdxl_refiner_model",
                "sdxl_refiner_switch_step",
                "sdxl_refiner_cfg",
                "sdxl_refiner_clip",
                "sdxl_refiner_source",
                "refiner_noise_augmentation",
                "pid_input_vae",
                "pid_latent_format",
                "pid_degrade_sigma",
                "pid_conditioning",
                "reference_adapter",
                "reference_adapter_model",
                "reference_adapter_vision",
                "reference_adapter_strength",
                "reference_adapter_info",
                "native_edit_recipe",
                "krea_style_lora",
                "krea_style_strength",
                "native_krea_style",
                "native_reference_conditioning",
                "native_inpaint_conditioning",
                "native_sampling",
                "qwen_layer_count",
                "native_layered_conditioning",
                "native_minimax_image",
                "native_component_layout",
                "gligen_guidance",
                "hypernetwork_guidance",
                "native_negative_model",
                "native_schedule_alpha",
                "native_schedule_beta",
                "native_schedule_mu",
                "native_schedule_std",
                "native_model_patch",
                "native_control_recipe",
            ):
                if key in gen_data:
                    prompt[key] = deepcopy(gen_data[key])
            if gen_data.get("_native_gligen_layout") is not None:
                from ruinedfooocus.features.image_editing.gligen_layout import FIELDS

                prompt.update((key, deepcopy(gen_data[key])) for key in FIELDS if key in gen_data)
            if gen_data.get("_native_hypernetwork") is not None:
                from ruinedfooocus.features.image_editing.native_hypernetwork import FIELDS

                prompt.update((key, deepcopy(gen_data[key])) for key in FIELDS if key in gen_data)
            from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

            for key in (*model_patches.FIELDS, *native_control_recipes.FIELDS):
                if key in gen_data:
                    prompt[key] = deepcopy(gen_data[key])
            from ruinedfooocus.features.image_editing import lora_hooks

            for key in (*lora_hooks.FIELDS, "native_lora_hooks"):
                if key in gen_data:
                    prompt[key] = deepcopy(gen_data[key])
            if isinstance(imgs, ImageOutputBatch):
                prompt["native_output_group"] = dict(
                    kind=imgs.plan.kind,
                    index=output_index,
                    count=len(imgs),
                    generation_index=i,
                )
                if imgs.metadata:
                    prompt["native_image_output"] = dict(imgs.metadata[output_index])
            if (gen_data.get("controlnet") or {}).get("type") == "qwen-union":
                from ruinedfooocus.features.image_editing.qwen_image21_union import (
                    settings as union_settings,
                )

                prompt["controlnet"] = union_settings(gen_data["controlnet"])
            if gen_data.get("image_guidance"):
                # Flat effective fields round-trip through imported image
                # metadata; the grouped form also identifies the applied recipe.
                prompt.update(gen_data["image_guidance"])
                prompt["image_guidance"] = dict(gen_data["image_guidance"])
                prompt["reference_image_count"] = gen_data.get("reference_image_count", 0)
            if gen_data.get("image_geometry"):
                prompt["image_geometry"] = gen_data["image_geometry"]
                from ruinedfooocus.features.image_editing.qwen_image21_local_edit import (
                    replay_values as local_edit_recipe,
                )

                # Geometry records the applied mode; a hidden UI radio may still
                # contain a value on an unrelated text-to-image request.
                prompt.update(local_edit_recipe({"image_geometry": gen_data["image_geometry"]}))
            if gen_data.get("depth_preprocessing"):
                prompt["depth_preprocessing"] = dict(gen_data["depth_preprocessing"])
            if gen_data.get("_sdcpp_metadata"):
                prompt["image_engine"] = gen_data["image_engine"]
                prompt["image_engine_details"] = dict(gen_data["_sdcpp_metadata"])
            if "upscale" in pipeline.pipeline_type:
                prompt.update(
                    width=x.shape[1],
                    height=x.shape[0],
                    upscaler=control_tools.get_settings(gen_data)["upscaler"],
                )
            elif gen_data.get("automatic_upscale_model"):
                prompt.update(
                    generation_width=prompt["width"],
                    generation_height=prompt["height"],
                    width=x.shape[1],
                    height=x.shape[0],
                    automatic_upscale=gen_data["automatic_upscale"],
                    upscaler=gen_data["automatic_upscale_model"],
                )
            if gen_data.get("size_plan"):
                prompt["size_plan"] = dict(gen_data["size_plan"])
                prompt["size_mode"] = "target"
                prompt["generation_width"] = gen_data["size_plan"]["content_width"]
                prompt["generation_height"] = gen_data["size_plan"]["content_height"]
                prompt["width"], prompt["height"] = x.shape[1], x.shape[0]
                prompt["automatic_upscale"] = gen_data.get("automatic_upscale", "Off")
                if gen_data.get("automatic_upscale_target") == "Input image only":
                    prompt["automatic_upscale_target"] = "Input image only"
                    prompt["upscale_source_size"] = gen_data.get("upscale_source_size")
                    prompt["upscale_source"] = gen_data.get("upscale_source")
            if gen_data.get("image_source"):
                prompt["image_source"] = dict(gen_data["image_source"])
            if gen_data.get("control_guide"):
                prompt["control_guide"] = dict(gen_data["control_guide"])
            if gen_data.get("image_utility_source"):
                prompt = depth_utilities.output_metadata(gen_data)
            if gen_data.get("native_composition_source"):
                prompt = dict(
                    software="RuinedFooocus",
                    width=image_composition.captured(gen_data).output_size[0],
                    height=image_composition.captured(gen_data).output_size[1],
                    controlnet=dict(gen_data["controlnet"]),
                    native_composition_source=gen_data["native_composition_source"],
                    **{key: gen_data[key] for key in image_composition.FIELDS},
                    **export_settings,
                )
            from ruinedfooocus.features.image_editing.edit_prompt import attach_metadata

            prompt = attach_metadata(gen_data, prompt)
            if not sampling:
                from ruinedfooocus.features.generation.request_metadata import (
                    attach_source_generation,
                )

                prompt = attach_source_generation(gen_data, prompt)
            metadata = PngInfo()
            metadata.add_text("parameters", json.dumps(prompt))

            if "preview_count" not in shared.state:
                shared.state["preview_count"] = 0
            shared.state["preview_count"] += 1
            if isinstance(x, str) or isinstance(x, (pathlib.WindowsPath, pathlib.PosixPath)):
                if advanced_output(export_settings) and Path(x).suffix.lower() in (
                    ".png",
                    ".exr",
                    ".avif",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                ):
                    from ruinedfooocus.features.image_processing.native_image_formats import (
                        convert_color,
                        read_export_source,
                    )

                    source = read_export_source(x)
                    prompt["output_source_dtype"] = str(source.pixels.dtype)
                    pixels = convert_color(
                        source.pixels, source.color_space, export_settings["output_color_space"]
                    )
                    prompt.update(export_settings)
                    write_image(local_temp_filename, pixels, export_settings, prompt)
                else:
                    if advanced_output(export_settings):
                        raise ValueError(
                            "This operation did not return a supported still image for native export."
                        )
                    local_temp_filename = x
            elif advanced_output(export_settings):
                prompt.update(export_settings)
                write_image(local_temp_filename, x, export_settings, prompt)
            else:
                if not isinstance(x, Image.Image):
                    x = Image.fromarray(x)
                x.save(local_temp_filename, pnginfo=metadata)

            try:
                metadata = {
                    "parameters": json.dumps(prompt),
                    "file_path": str(Path(local_temp_filename).relative_to(folder)),
                }
                if "browser" in shared.shared_cache:
                    shared.shared_cache["browser"].add_image(
                        local_temp_filename,
                        Path(local_temp_filename).relative_to(folder),
                        metadata,
                        commit=True,
                    )
            except:
                pass

            display_filename = local_temp_filename
            if advanced_output(export_settings):
                from ruinedfooocus.core.cache_contract import configured_cache_root
                from ruinedfooocus.features.image_processing.native_image_formats import (
                    display_image,
                )

                display_filename = str(
                    display_image(local_temp_filename, configured_cache_root(paths))
                )
            if advanced_output(export_settings) and "_api_job" not in gen_data:
                from ruinedfooocus.features.image_processing.native_image_outputs import (
                    NativeImageOutput,
                )

                res.append(NativeImageOutput(str(local_temp_filename), display_filename))
            else:
                res.append(local_temp_filename)
            metadatastrings.append(json.dumps(prompt))
            shared.state["last_image"] = local_temp_filename

        if seed > -1:
            seed += 1
        if stop_batch:
            break
    if output_plan is not None and not stop_batch and completed_groups == max(image_number, 1):
        return ImageBatchArtifacts(
            res,
            output_plan,
            completed_groups,
            tensor_files=gen_data.get("_native_tensor_exports", ()),
        )
    if gen_data.get("_native_tensor_exports") and not stop_batch:
        from ruinedfooocus.features.generation.native_tensor_io import TensorGenerationArtifacts

        return TensorGenerationArtifacts(res, gen_data["_native_tensor_exports"])
    if not stop_batch and gen_data.get("image_utility_source", {}).get("operation") in (
        "wevisone-detect",
        "wevisone-matte",
    ):
        from ruinedfooocus.features.image_processing.wevisone import WeVisOneArtifacts

        return WeVisOneArtifacts(res, gen_data["image_utility_source"])
    return res


def preload_startup_model():
    if not settings.default_settings.get("preload_image_model", False):
        return
    try:
        from ruinedfooocus.features.models.component_profiles import snapshot_request

        request = snapshot_request(
            {"base_model_name": settings.default_settings.get("base_model")},
            shared.models,
            settings.default_settings,
            shared.path_manager,
        )
        captured_source = any(
            key in request
            for key in (
                "_local_diffusers_bundle",
                "_sensenova_shard_bundle",
                "_native_approximate_codec",
            )
        )
        if captured_source:
            from ruinedfooocus.features.generation.image_preparation import capture_image_assets

            request = capture_image_assets(
                request,
                shared.path_manager,
                {},
                offline=True,
                models=shared.models,
                values=settings.default_settings,
            )
        pipeline = pipelines.update(request)
        if pipeline is not None:
            pipeline.load_base_model(
                request["base_model_name"],
                **(
                    {"component_overrides": request["image_component_overrides"]}
                    if "image_component_overrides" in request
                    else {}
                ),
                **(
                    {"prepared_data": request}
                    if captured_source
                    and "sdxl" in pipeline.pipeline_type
                    and "sdcpp" not in pipeline.pipeline_type
                    else {}
                ),
            )
    except Exception as error:
        # A missing/broken startup model must not prevent later requests.
        print(f"Startup model: {describe_error(error)}")
        traceback.print_exc()


def worker():
    global _active_job

    def job_start(gen_data):
        shared.state["preview_grid"] = None
        shared.state["preview_total"] = max(gen_data["image_total"], 1)
        shared.state["preview_count"] = 0

    def job_stop():
        shared.state["preview_grid"] = None
        shared.state["preview_total"] = 0
        shared.state["preview_count"] = 0

    def reset_preview():
        shared.state["preview_grid"] = None
        shared.state["preview_count"] = 0

    def process(gen_data):
        from ruinedfooocus.features.image_processing import latent_composition

        if latent_composition.handles(control_tools.get_settings(gen_data).get("type")):
            # Preserve the tensor-only completion object for the existing File UI.
            add_result(gen_data["task_id"], "results", _process(gen_data))
            return
        # Check some needed items
        if not "image_total" in gen_data:
            gen_data["image_total"] = 1
        if not "generate_forever" in gen_data:
            gen_data["generate_forever"] = False

        shared.state["preview_total"] = max(gen_data["image_total"], 1)

        results = []
        metadatastrings = []
        while True:
            reset_preview()
            gen_data["index"] = (0, (gen_data["image_total"]))
            if isinstance(gen_data["prompt"], list):
                tmp_data = gen_data.copy()
                for prompt in gen_data["prompt"]:
                    tmp_data["prompt"] = prompt
                    if gen_data["generate_forever"]:
                        reset_preview()
                    results.extend(_process(tmp_data))
                    if shared.state["interrupted"]:
                        break
                    tmp_data["index"] = (tmp_data["index"][0] + 1, tmp_data["index"][1])
            else:
                gen_data["index"] = (0, 1)
                results.extend(_process(gen_data))

            if not (gen_data["generate_forever"] and shared.state["interrupted"] == False):
                break

        # Prepend preview-grid (maybe)
        if (
            "preview_grid" in shared.state
            and shared.state["preview_grid"] is not None
            and shared.state["preview_total"] > 1
            and ("show_preview" not in gen_data or gen_data["show_preview"] == True)
            and not gen_data["generate_forever"]
        ):
            results = [_preview_path(gen_data)] + results

        add_result(gen_data["task_id"], "results", results)

    def txt2txt_process(gen_data):

        pipeline = pipelines.update(gen_data)
        if pipeline == None:
            raise RuntimeError("No chat pipeline is available.")

        try:
            # See if pipeline wants to pre-parse gen_data
            gen_data = pipeline.parse_gen_data(gen_data)
        except:
            pass

        try:
            results = pipeline.process(gen_data)
        except Exception as error:
            traceback.print_exc()
            results = {"error": describe_error(error)}

        check_interrupt(gen_data)
        from ruinedfooocus.features.chat.chat_tools import ImageToolContinuation

        if isinstance(results, ImageToolContinuation):
            return results
        add_result(gen_data["task_id"], "results", results)

    def handler(gen_data):
        match gen_data["task_type"]:
            case "external_api":
                group = getattr(gen_data["_api_job"].operation, "resource_group", None) or "utility"
                validate = getattr(gen_data["_api_job"].operation, "validate", None)
                if validate is not None:
                    validate()
                if getattr(gen_data["_api_job"].operation, "compute_required", True):
                    with service_scope(group, shared.state):
                        gen_data["_api_job"].run(gen_data["task_id"])
                else:
                    gen_data["_api_job"].run(gen_data["task_id"])
            case "process":
                process(gen_data)
            case "llama":
                with service_scope("chat", shared.state):
                    return txt2txt_process(gen_data)
            case "chat_image_tool":
                return resume_chat_image_tool(gen_data)
            case _:
                raise ValueError(f"Unknown task type: {gen_data['task_type']}")

    while True:
        selected = registry.take()
        if selected is not None:
            item, task = selected
            _active_job = item
            scope = job_scope(item)
            scope.__enter__()
            deferred = False
            try:
                item.check()
                validate_editor_data(task)
                continuation = handler(task)
                if continuation is not None:
                    defer_chat_image_tool(item, task, continuation)
                    deferred = True
            except (model_management.InterruptProcessingException, CancelledError) as error:
                shared.state["interrupted"] = True
                discard_editor_result(
                    item,
                    status="stale"
                    if isinstance(error, CancelledError) and not item.cancelled.is_set()
                    else "cancelled",
                )
                if task.get("_image_tool_continuation") is not None:
                    publish_image_tool_error(item, task["_image_tool_continuation"], error)
                if "_api_job" in task:
                    task["_api_job"].fail("Generation stopped.")
                elif task.get("task_type") in ("llama", "chat_image_tool"):
                    add_result(
                        task["task_id"],
                        "results",
                        {"error": "The operation was stopped. You can retry."},
                    )
                else:
                    add_result(task["task_id"], "results", [])
            except Exception as error:
                # Loading, metadata and prompt preparation can fail before the
                # sampling try/except. Always finish the request and keep serving.
                traceback.print_exc()
                discard_editor_result(item, status="failed")
                if task.get("_image_tool_continuation") is not None:
                    publish_image_tool_error(item, task["_image_tool_continuation"], error)
                if "_api_job" in task:
                    task["_api_job"].fail(describe_error(error))
                elif task.get("task_type") in ("llama", "chat_image_tool"):
                    add_result(task["task_id"], "results", {"error": describe_error(error)})
                else:
                    add_result(task["task_id"], "error", describe_error(error))
                    add_result(task["task_id"], "results", [])
            finally:
                try:
                    cleanup_interrupt(item)
                finally:
                    if deferred:
                        pass
                    elif item.external is not None and not item.external.finished:
                        item.external.fail(
                            "The worker ended before returning a result; retry the request"
                        )
                    elif item.external is None and not item.terminal:
                        registry.publish(
                            item.task_id,
                            "results",
                            {
                                "error": "The worker ended before returning a result; retry the request"
                            },
                        )
                    if not deferred:
                        if item.state.get("_editor_request"):
                            discard_editor_result(
                                item,
                                status="cancelled" if item.cancelled.is_set() else "failed",
                            )
                        registry.finish(item)
                        api_jobs.pop(item.task_id, None)
                    _active_job = None
                    scope.__exit__(None, None, None)
                    task = None


def defer_chat_image_tool(item, data, continuation):
    """Prepare a validated Chat action after its outer compute scope has exited."""
    from ruinedfooocus.app.api_runtime import prepare_chat_image_tool
    from ruinedfooocus.app.preparation_admission import preparation_admission
    from ruinedfooocus.features.chat.chat_tools import ImageToolContinuation

    if not isinstance(continuation, ImageToolContinuation):
        raise TypeError("The Chat worker returned an invalid continuation")
    if item.state.get("image_tool_dispatched"):
        raise ValueError("Only one image tool action is allowed per Chat turn")
    if continuation.request.owner != item.owner:
        raise PermissionError("The image tool belongs to a different session")
    item.check()
    item.state["image_tool_dispatched"] = True

    def prepare():
        item.check()
        result = prepare_chat_image_tool(
            continuation.request,
            gen_data=data,
            pipeline=continuation.pipeline,
            check_cancelled=item.check,
        )
        item.check()
        return result

    def ready(prepared):
        resumed = dict(
            data,
            task_type="chat_image_tool",
            _image_tool_continuation=continuation,
            _prepared_image_tool=prepared,
        )
        if registry.enqueue(item, resumed):
            start_worker()

    def failed(error):
        discard_editor_result(
            item,
            status="cancelled"
            if item.cancelled.is_set()
            else "stale"
            if isinstance(error, CancelledError)
            else "failed",
        )
        publish_image_tool_error(item, continuation, error)
        add_result(item.task_id, "results", {"error": describe_error(error)})
        registry.finish(item)

    item.preparation = preparation_admission.submit(
        prepare, ready, failed, cancelled=item.cancelled
    )


def resume_chat_image_tool(data):
    """Resume the same owned job; a nested image action never creates another job."""
    from ruinedfooocus.app.api_runtime import execute_chat_image_tool
    from ruinedfooocus.features.chat.chat_tools import dispatch_image_tool, public_image_tool_result

    continuation = data["_image_tool_continuation"]
    check_interrupt(data)
    result = dispatch_image_tool(
        continuation.request,
        lambda request: execute_chat_image_tool(
            request,
            gen_data=data,
            pipeline=continuation.pipeline,
            prepared=data["_prepared_image_tool"],
            check_cancelled=lambda: check_interrupt(data),
        ),
        check_cancelled=lambda: check_interrupt(data),
    )
    check_interrupt(data)
    response = continuation.resume(result)
    check_interrupt(data)
    public = public_image_tool_result(result)
    item = data.get("_job") or current_job()
    with registry.condition:
        item.check()
        if result["status"] in ("error", "cancelled"):
            discard_editor_result(item, status=result["status"])
        editor = publish_editor_result(item)
        if editor is not None:
            public["editor"] = editor["snapshot"]
        add_result(data["task_id"], "image_tool_result", public)
        item.state["image_tool_terminal"] = True
        add_result(data["task_id"], "results", response)


def capture_image_data(data):
    """Freeze an effective sampling recipe before any queue/preparation wait."""
    from ruinedfooocus.features.image_processing import latent_composition
    from ruinedfooocus.features.image_processing.image_composition import handles
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings
    from ruinedfooocus.features.generation.image_preparation import (
        capture_image_assets,
        capture_pending_image_selection,
        pending_image_selection,
    )
    from ruinedfooocus.features.models.effective_image_plan import (
        capture_effective_image_plan,
        get_effective_image_plan,
    )
    from copy import copy, deepcopy

    data = dict(data)
    prompt = data.get("prompt")
    if (
        isinstance(prompt, list)
        and len(prompt) > 1
        and "_api_job" not in data
        and data.get("task_type") != "tool_call"
    ):
        from ruinedfooocus.features.gallery.image_metadata import read_metadata

        if any(read_metadata(value) is not None for value in prompt):
            raise ValueError("Apply saved image settings before starting a prompt batch.")
    if data.get("editor_context") is not None and data.get("_editor_operation") is None:
        from ruinedfooocus.app.api_runtime import bind_editor_source

        data = bind_editor_source(data, data["editor_context"])
    if data.get("_editor_operation") is not None:
        validate_editor_data(data)
    from ruinedfooocus.features.generation.request_metadata import capture_source_generation

    data = capture_source_generation(data)
    controls = control_tools.get_settings(data)
    previous_plan = get_effective_image_plan(data)
    if previous_plan is not None:
        previous_plan.verify_request(data, controls)
    if latent_composition.handles(controls.get("type")) or handles(controls.get("type")):
        result = capture_image_assets(
            dict(data, **format_settings(data)),
            shared.path_manager,
            controls,
            offline=getattr(shared.models, "offline", False),
        )
        result = capture_effective_image_plan(
            dict(result, _sampling_captured=False),
            shared.models,
            {},
            shared.path_manager,
            controls=controls,
        )
        if result.get("_image_workflow_step") is not None:
            from ruinedfooocus.features.image_editing.image_workflows import bind_effective_plan

            result = bind_effective_plan(result, result.get("_effective_image_plan"))
        elif result.get("_editor_operation") is not None:
            from ruinedfooocus.features.gallery.editor_session import bind_effective_plan

            result = bind_effective_plan(result)
        elif result.get("editor_context") is not None:
            from ruinedfooocus.features.gallery.editor_session import capture_request

            result = capture_request(result, result["editor_context"], controls["type"])
        return result
    values = deepcopy(data.get("_image_capture_settings", settings.default_settings))
    performance = data.get("_image_capture_performance")
    if performance is None:
        performance = copy(getattr(shared, "performance_settings", None))
        for key in ("performance_options", "legacy_options"):
            if hasattr(performance, key):
                setattr(performance, key, deepcopy(getattr(performance, key)))
    resolutions = data.get("_image_capture_resolutions")
    if resolutions is None:
        resolutions = copy(getattr(shared, "resolution_settings", None))
        if hasattr(resolutions, "aspect_ratios"):
            resolutions.aspect_ratios = deepcopy(resolutions.aspect_ratios)
    paths = data.get("_image_capture_paths")
    if paths is None:
        paths = copy(shared.path_manager)
        for key in ("paths", "model_paths", "DOWNLOADABLE_FILES"):
            if hasattr(paths, key):
                setattr(paths, key, deepcopy(getattr(paths, key)))
    models = data.get("_image_capture_models")
    if models is None:
        models = copy(shared.models)
        for key in ("model_dirs", "cache_paths", "names", "model_hash"):
            if hasattr(models, key):
                setattr(models, key, deepcopy(getattr(models, key)))
    data.update(
        _image_capture_settings=values,
        _image_capture_performance=performance,
        _image_capture_resolutions=resolutions,
        _image_capture_paths=paths,
        _image_capture_models=models,
    )
    from ruinedfooocus.features.image_processing import still_image_utilities as depth_utilities

    native_tools = ("detect", "segment", "depth-map", "restore-seedvr2", *depth_utilities.KINDS)
    if control_tools.get_settings(data).get("type", "").lower() not in native_tools:
        from ruinedfooocus.features.generation.request_metadata import metadata_intent

        data, _ = metadata_intent(data)
    pending = capture_pending_image_selection(
        data, paths, offline=getattr(models, "offline", False)
    )
    if pending is not None:
        from ruinedfooocus.app.api_runtime import _preflight_pending_worker_image
        from ruinedfooocus.features.generation.request_metadata import capture_sampling_replay

        pending = capture_sampling_replay(pending, models)
        return _preflight_pending_worker_image(
            pending, pending_image_selection(pending), resolutions
        )

    def capture_plan(result, *, family=None, bind=True):
        result = capture_effective_image_plan(
            result,
            models,
            values,
            paths,
            family=family,
            controls=control_tools.get_settings(result),
            performance=performance,
            resolutions=resolutions,
        )
        if not bind:
            return result
        if result.get("_image_workflow_step") is not None:
            from ruinedfooocus.features.image_editing.image_workflows import bind_effective_plan

            result = bind_effective_plan(result, result.get("_effective_image_plan"))
        elif result.get("_editor_operation") is not None:
            from ruinedfooocus.features.gallery.editor_session import bind_effective_plan

            result = bind_effective_plan(result)
        elif result.get("editor_context") is not None and result.get("_editor_operation") is None:
            from ruinedfooocus.features.gallery.editor_session import capture_request

            controls = control_tools.get_settings(result)
            operation = (
                "inpaint" if result.get("inpaint_toggle") else controls.get("type") or "generate"
            )
            result = capture_request(result, result["editor_context"], operation)
        return result

    controls = control_tools.get_settings(data)
    data = latent_composition.omit_inactive(data)
    from ruinedfooocus.features.models.component_profiles import snapshot_request
    from ruinedfooocus.features.generation.sdcpp_pipeline import capture_image_request
    from ruinedfooocus.features.prompting.local_model import capture_prompt_preparation

    result = {key: value if key.startswith("_") else deepcopy(value) for key, value in data.items()}
    result.update(format_settings(result))
    controls = control_tools.get_settings(result)
    from ruinedfooocus.features.image_editing.ordered_native_references import omit_inactive_recipe

    from ruinedfooocus.features.generation.size_plan import capture_size_plan

    native_utility = controls.get("type", "").lower() in native_tools
    sampling = not native_utility and controls.get("type", "").lower() not in ("rembg", "upscale")
    family = None
    if sampling and result.get("base_model_name"):
        family = models.get_model_base(
            models.get_models_by_path("checkpoints", result["base_model_name"], fetch=False)
        )
    from ruinedfooocus.features.image_editing import guide_handoffs

    if result.get(guide_handoffs.HANDOFF_KEY) is not None or guide_handoffs.KEY in result:
        from ruinedfooocus.features.gallery.editor_session import default_store
        from ruinedfooocus.app.resource_lifecycle import current_owner

        result = guide_handoffs.capture_request(
            result, store=default_store(), owner=current_owner(), manager=paths, family=family
        )
        controls = control_tools.get_settings(result)
    result = omit_inactive_recipe(result, controls, family)
    if sampling:
        if result.get("base_model_name") and not result.get("_sampling_captured"):
            result = performance.apply(result, family)
            result["_sampling_captured"] = True
        from ruinedfooocus.features.generation.request_metadata import capture_sampling_replay

        result = capture_sampling_replay(result, models)
        controls = control_tools.get_settings(result)
    from ruinedfooocus.features.image_editing.native_control_recipes import capture_source_size

    result = capture_source_size(result, controls)
    result = capture_size_plan(result, resolutions, family=family, controls=controls)
    from ruinedfooocus.features.image_editing.reference_adapters import omit_inactive_ui_reference

    result = omit_inactive_ui_reference(result, controls)
    if not sampling:
        # False records an intentional utility capture; absence is a legacy caller.
        # A later sampling operation must recapture its recipe before enqueue.
        result["_sampling_captured"] = False
        result.pop("image_component_overrides", None)
        result.pop("_local_diffusers_bundle", None)
        from ruinedfooocus.features.generation.image_preparation import omit_generation_conditioning

        result = omit_generation_conditioning(result)
    else:
        result = snapshot_request(result, models, values, paths)
    if sampling:
        result = capture_prompt_preparation(result, values, paths)
        result = capture_plan(result, family=family, bind=False)
    else:
        result.pop("_prompt_preparation", None)
    result = capture_image_assets(
        result,
        paths,
        controls,
        offline=getattr(models, "offline", False),
        models=models,
        values=values,
    )
    if not native_utility:
        result = capture_image_request(
            result,
            models,
            values,
            paths,
            resolution_settings=resolutions,
        )
    # Native operation owners may normalize their own controls during capture.
    # The draft freezes component roles; only this final plan crosses preparation.
    result = dict(result)
    result.pop("_effective_image_plan", None)
    return capture_plan(result, family=family if sampling else None)


def prepare_image_data(data, *, check_cancelled, progress=None):
    from ruinedfooocus.features.generation.sdcpp_pipeline import prepare_image_request
    from ruinedfooocus.features.generation.image_preparation import (
        pending_image_selection,
        prepare_image_assets,
        prepare_pending_image_selection,
    )

    check_cancelled()
    from ruinedfooocus.features.image_editing import guide_handoffs

    if guide_handoffs.KEY in data:
        from ruinedfooocus.features.gallery.editor_session import default_store
        from ruinedfooocus.features.models.effective_image_plan import get_effective_image_plan
        from ruinedfooocus.app.resource_lifecycle import current_owner

        plan = get_effective_image_plan(data)
        guide_handoffs.validate_request(
            data,
            store=default_store(),
            owner=current_owner(),
            manager=data["_image_capture_paths"],
            family=plan.family,
        )
    if data.get("_editor_operation") is not None:
        validate_editor_data(data)
    if pending_image_selection(data) is not None:
        data = prepare_pending_image_selection(
            data, check_cancelled=check_cancelled, progress=progress
        )
        check_cancelled()
        data = capture_image_data(data)
        if data.get("_editor_operation") is not None:
            validate_editor_data(data)
    data = prepare_image_assets(data, check_cancelled=check_cancelled, progress=progress)
    data = prepare_image_request(data, check_cancelled=check_cancelled)
    if data.get("_image_workflow_step") is not None:
        from ruinedfooocus.features.image_editing.image_workflows import bind_effective_plan

        data = bind_effective_plan(data, data.get("_effective_image_plan"))
    elif data.get("_editor_operation") is not None:
        from ruinedfooocus.features.gallery.editor_session import bind_effective_plan

        data = bind_effective_plan(data)
    if data.get("_editor_operation") is not None:
        validate_editor_data(data)
    return data


# Use this to add a task, then use task_result() to get data from the pipeline
def add_task(gen_data):
    gen_data = dict(gen_data)
    if gen_data.get("task_type") == "process":
        gen_data = capture_image_data(gen_data)
    if gen_data.get("task_type") == "llama" and not gen_data.get("unload"):
        if "image_tool_intent" not in gen_data and gen_data.get("force_image") is True:
            gen_data["image_tool_intent"] = "run"
        from ruinedfooocus.features.chat.model_preparation import capture_chat_request
        from ruinedfooocus.features.chat.llama_models import local_models

        gen_data = capture_chat_request(
            gen_data,
            settings.default_settings,
            shared.path_manager,
            local_models(),
        )
        permissions = gen_data.get("_chat_tool_permissions")
        if permissions is not None:
            permissions.check()
            allow_image = permissions.enable_image_generation
            gen_data["enable_image_generation"] = allow_image
            gen_data["rewrite_image_prompt"] = permissions.rewrite_image_prompt
            gen_data["update_create_prompt"] = permissions.update_create_prompt
        else:
            allow_image = gen_data.get(
                "enable_image_generation", gen_data.get("image_tool_intent") in ("preview", "run")
            )
            gen_data["enable_image_generation"] = allow_image
        if gen_data.get("load_only") or gen_data.get("_chat_attachments") or not allow_image:
            gen_data.pop("image_settings", None)
        elif gen_data.get("image_settings"):
            from copy import deepcopy
            from ruinedfooocus.app.api_runtime import _image_runtime_snapshot

            # A text turn must not validate, acquire or initialize optional image assets.
            gen_data["image_settings"] = deepcopy(gen_data["image_settings"])
            gen_data["_image_capture_runtime"] = _image_runtime_snapshot(shared)
    api_job = gen_data.get("_api_job")
    item = registry.create(
        owner=api_job.owner if api_job is not None else None,
        cancelled=api_job.cancelled if api_job is not None else None,
    )
    item.external = api_job
    editor_request = (
        gen_data if api_job is None else getattr(api_job.operation, "editor_request", {})
    )
    if editor_request.get("_editor_operation") is not None:
        item.state["_editor_request"] = {"_editor_operation": editor_request["_editor_operation"]}
    task_id = item.task_id
    gen_data["task_id"], gen_data["_job"] = task_id, item
    attachments = gen_data.get("_chat_attachments")
    if attachments is not None:
        try:
            attachments.check(item.owner, item.cancelled)
        except Exception:
            registry.finish(item)
            raise
    if api_job is not None:
        api_jobs[task_id] = api_job
        api_job.task_id = task_id
        item.on_finish = lambda: api_jobs.pop(task_id, None)

    def ready(prepared):
        if registry.enqueue(item, prepared.copy()):
            start_worker()

    def failed(error):
        discard_editor_result(item, status="cancelled" if item.cancelled.is_set() else "failed")
        if api_job is not None:
            api_job.fail(describe_error(error))
            api_jobs.pop(task_id, None)
        else:
            add_result(task_id, "results", {"error": describe_error(error)})
        registry.finish(item)

    def image_progress(received, total, speed):
        amount = (
            f"{received:,} / {total:,} bytes" if total else f"{received:,} bytes (total unknown)"
        )
        add_result(
            task_id,
            "preview",
            (100 * received / total if total else -1, "Preparing image files: " + amount, None),
        )

    def prepare_owned():
        item.check()
        validate_editor_data(gen_data)
        from ruinedfooocus.features.prompting.local_model import prepare_prompt_data

        if gen_data.get("task_type") == "llama" and not gen_data.get("unload"):
            from ruinedfooocus.features.chat.model_preparation import prepare_chat_request

            prepare_chat_request(
                gen_data,
                lambda *progress: add_result(task_id, "download", progress),
                cancelled=item.cancelled,
            )
        elif api_job is not None and hasattr(api_job.operation, "prepare"):
            api_job.operation.prepare(api_job)
        elif gen_data.get("task_type") == "process":
            prepare_prompt_data(gen_data, cancelled=item.cancelled)
            return prepare_image_data(
                gen_data,
                check_cancelled=item.check,
                progress=image_progress,
            )
        item.check()
        return gen_data

    def prepare():
        with job_scope(item):
            return prepare_owned()

    from ruinedfooocus.features.generation.sdcpp_pipeline import needs_preparation
    from ruinedfooocus.features.chat.model_preparation import chat_artifacts
    from ruinedfooocus.features.prompting.local_model import needs_prompt_preparation
    from ruinedfooocus.features.generation.image_preparation import needs_image_preparation

    needs_io = (
        (
            gen_data.get("task_type") == "llama"
            and not gen_data.get("unload")
            and (
                any(not artifact.destination.is_file() for artifact in chat_artifacts(gen_data))
                or gen_data.get("load_only")
                or gen_data.get("_rag_preparation") is not None
            )
        )
        or needs_preparation(gen_data)
        or needs_prompt_preparation(gen_data)
        or needs_image_preparation(gen_data)
        or (api_job is not None and hasattr(api_job.operation, "prepare"))
    )
    if needs_io:
        from ruinedfooocus.app.preparation_admission import preparation_admission

        try:
            item.preparation = preparation_admission.submit(prepare, ready, failed)
        except Exception as error:
            failed(error)
    else:
        ready(gen_data)
    return task_id


# Pipelines use this to add results
def add_result(task_id, flag, product):
    if task_id is None or task_id in api_jobs:
        return
    with registry.condition:
        item = registry.get(task_id, trusted=True)
        if item.terminal:
            return
        if flag == "results" and (
            item.state.get("_editor_publication") or item.state.get("_editor_original")
        ):
            publish_editor_result(item)
        registry.publish(task_id, flag, product)


# Use the task_id from add_task() to wait for data
def task_result(task_id):
    while True:
        result = registry.result(task_id)
        if result is not None:
            return result


worker_thread = None


def start_worker():
    """Start after pipeline imports complete; a later request can restart a dead worker."""
    global worker_thread
    with task_lock:
        if worker_thread is None or not worker_thread.is_alive():
            worker_thread = threading.Thread(target=_worker_entry, daemon=True)
            worker_thread.start()


def _worker_entry():
    global _startup_preloaded
    try:
        if not _startup_preloaded and settings.default_settings.get("preload_image_model", False):
            from ruinedfooocus.app.resource_lifecycle import JobIdentity, WorkItem

            _startup_preloaded = True
            startup = WorkItem(
                JobIdentity(0, "startup-preload", ("internal", "startup")), threading.Event()
            )
            with job_scope(startup), service_scope("image", shared.state, item=startup):
                preload_startup_model()
        worker()
    except BaseException:
        item = _active_job
        if item is not None:
            job = api_jobs.pop(item.task_id, None)
            if job is not None:
                job.fail("The worker stopped unexpectedly; retry the request")
            else:
                registry.publish(
                    item.task_id,
                    "results",
                    {"error": "The worker stopped unexpectedly; retry the request"},
                )
            registry.finish(item)
        raise
    finally:
        registry.fail_ready("The worker stopped unexpectedly; retry the request")
