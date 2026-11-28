"""Bounded native LoRA hook schedules, captured before the compute queue."""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from functools import wraps
import json
import math
from pathlib import Path, PurePosixPath
import sys

from ruinedfooocus.core.model_formats import NATIVE_STATE_DICT_SUFFIXES
from ruinedfooocus.core.cache_contract import canonical_path_key
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.image_editing import lora_hook_sources as source_recipes

KEY = "_lora_hook_plan"
MODES = ("off", "keyframes", "interpolated", "floats")
INTERPOLATIONS = ("linear", "ease_in", "ease_out", "ease_in_out")
DEFAULTS = {
    "lora_hook_mode": "off",
    "lora_hook_model": "",
    "lora_hook_model_strength": 1.0,
    "lora_hook_clip_strength": 0.0,
    "lora_hook_schedule_clip": False,
    "lora_hook_keyframes": '[{"start":0,"strength":1},{"start":0.5,"strength":0}]',
    "lora_hook_interpolation": "linear",
    "lora_hook_start": 0.0,
    "lora_hook_end": 1.0,
    "lora_hook_strength_start": 1.0,
    "lora_hook_strength_end": 0.0,
    "lora_hook_keyframe_count": 5,
    **source_recipes.DEFAULTS,
}
FIELDS = tuple(DEFAULTS)
# Explicitly reviewed native scheduled text/reference callers. Families without an
# external CLIP keep model-only hooks. New catalogue families need their own review.
CLIP_FAMILIES = frozenset(
    (
        "Anima",
        "AuraFlow",
        "BaseModel",
        "Boogu",
        "Chroma",
        "ChromaRadiance",
        "CosmosPredict2",
        "ErnieImage",
        "Fibo2",
        "Flux",
        "Flux2",
        "Flux2Klein4B",
        "Flux2Klein9B",
        "Flux3",
        "HiDream",
        "HunyuanDiT",
        "HunyuanImage21",
        "HunyuanImage21Refiner",
        "Ideogram4",
        "JoyImage",
        "Kandinsky5Image",
        "Krea2",
        "Lens",
        "LongCatImage",
        "Lumina2",
        "MageFlow",
        "MOSSImage",
        "MingImage",
        "NewBieImage",
        "Omnigen2",
        "PiD",
        "PixArt",
        "PixelDiTT2I",
        "QwenImage",
        "QwenImage21",
        "SD20",
        "SD3",
        "SDXL",
        "SDXLRefiner",
        "SD21UNCLIP",
        "SD15_instructpix2pix",
        "SDXL_instructpix2pix",
        "SD_X4Upscaler",
        "StableCascade_C",
        "ZImage",
        "ZImagePixelSpace",
    )
)


def _number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number between {low} and {high}.")
    return float(value)


def settings(data, *, _sources=True):
    values = {key: data.get(key, default) for key, default in DEFAULTS.items()}
    if values["lora_hook_mode"] not in MODES:
        raise ValueError("Choose Off, manual, interpolated or strength-list LoRA keyframes.")
    name = values["lora_hook_model"]
    relative = PurePosixPath(str(name).replace("\\", "/"))
    if (
        not isinstance(name, str)
        or len(name) > 1024
        or (name and (relative.is_absolute() or ".." in relative.parts or ":" in name))
    ):
        raise ValueError("Choose a local scheduled LoRA from its configured folder.")
    values["lora_hook_model"] = relative.as_posix() if name else ""
    for key in ("lora_hook_model_strength", "lora_hook_clip_strength"):
        values[key] = _number(values[key], -20, 20, key)
    if type(values["lora_hook_schedule_clip"]) is not bool:
        raise ValueError("CLIP scheduling must be enabled or disabled.")
    if values["lora_hook_interpolation"] not in INTERPOLATIONS:
        raise ValueError("Choose a native LoRA hook interpolation.")
    for key in ("lora_hook_start", "lora_hook_end"):
        values[key] = _number(values[key], 0, 1, key)
    if values["lora_hook_start"] >= values["lora_hook_end"]:
        raise ValueError("Interpolated LoRA hook start must precede its end.")
    for key in ("lora_hook_strength_start", "lora_hook_strength_end"):
        values[key] = _number(values[key], 0, 10, key)
    count = values["lora_hook_keyframe_count"]
    count = _number(count, 2, 100, "LoRA keyframe count")
    if count != int(count):
        raise ValueError("Choose between 2 and 100 interpolated LoRA keyframes.")
    values["lora_hook_keyframe_count"] = int(count)
    text = values["lora_hook_keyframes"]
    if not isinstance(text, str) or len(text) > 16384:
        raise ValueError("LoRA keyframes must be a JSON list of at most 100 entries.")
    try:
        rows = json.loads(text)
    except ValueError:
        raise ValueError(
            "LoRA keyframes must be a JSON list of start and strength pairs."
        ) from None
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("LoRA keyframes require between 1 and 100 entries.")
    keyframes = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"start", "strength"}:
            raise ValueError("Each LoRA keyframe requires only start and strength.")
        start = _number(row["start"], 0, 1, "LoRA keyframe start")
        strength = _number(row["strength"], -20, 20, "LoRA keyframe strength")
        if keyframes and start <= keyframes[-1][0]:
            raise ValueError("LoRA keyframe starts must be strictly increasing.")
        keyframes.append((start, strength))
    if keyframes[0][0] != 0:
        raise ValueError("The first manual LoRA keyframe must start at 0.")
    values["lora_hook_keyframes"] = json.dumps(
        [dict(start=start, strength=strength) for start, strength in keyframes],
        separators=(",", ":"),
    )
    values.update(source_recipes.settings(values, settings, _number, sources=_sources))
    return values, tuple(keyframes)


def replay_values(data):
    return settings(data)[0]


def scheduled_clip_requested(data):
    if data.get("lora_hook_mode", "off") == "off" or not data.get("lora_hook_schedule_clip", False):
        return False
    values, _ = settings(data)
    return any(
        source.settings()["lora_hook_clip_strength"] != 0
        for source in source_recipes.specs(values, settings)
    )


def capabilities(family):
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS

    family = ALIASES.get(family, family)
    return dict(
        model=family in IMAGE_COMPONENTS,
        clip=family in CLIP_FAMILIES,
        modes=list(MODES),
        source_types=list(source_recipes.SOURCE_TYPES),
        max_sources=8,
    )


def choices(manager, source_type="lora"):
    """List native state dictionaries only when enabled; never load weights/headers."""
    result = set()
    for folder in manager.get_model_folders(
        "checkpoints" if source_type == "checkpoint" else "loras"
    ):
        root = Path(folder).resolve()
        pending, seen = [root], set()
        while pending:
            directory = pending.pop().resolve()
            if not directory.is_relative_to(root) or directory in seen:
                continue
            seen.add(directory)
            try:
                children = list(directory.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif (
                    path.suffix.lower() in NATIVE_STATE_DICT_SUFFIXES
                    and path.is_file()
                    and path.resolve().is_relative_to(root)
                ):
                    result.add(path.relative_to(root).as_posix())
    return sorted(result, key=str.casefold)


@dataclass(frozen=True)
class HookPlan:
    checkpoint: tuple
    family: str
    artifact: object
    values: tuple
    keyframes: tuple
    sources: tuple = ()

    def settings(self):
        return dict(self.values)


def capture_assets(data, manager, models, controls):
    result = dict(data)
    result.pop(KEY, None)
    result.pop("native_lora_hooks", None)
    # Off is deliberately zero file/header/native-hook work, even with a stale name.
    if result.get("lora_hook_mode", "off") == "off":
        return result, ()
    kind = str((controls or {}).get("type", "")).lower()
    if kind in ("rembg", "upscale", "detect", "segment", "depth-map"):
        result.update(DEFAULTS)
        return result, ()
    values, frames = settings(result)
    specs = source_recipes.specs(values, settings)
    if not specs:
        raise ValueError("Choose at least one scheduled LoRA or checkpoint source.")
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("LoRA hook scheduling requires the native image engine.")
    from ruinedfooocus.features.generation.image_preparation import (
        _artifact,
        _control_checkpoint_family,
    )
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS

    checkpoint = models.get_file("checkpoints", result.get("base_model_name")) if models else None
    if checkpoint is None:
        raise ValueError("Choose an installed image checkpoint for LoRA hook scheduling.")
    family = _control_checkpoint_family(models, result.get("base_model_name"))
    family = ALIASES.get(family, family)
    if family not in IMAGE_COMPONENTS:
        raise ValueError("LoRA hook scheduling requires a native still-image diffusion checkpoint.")
    sources = []
    for index, spec in enumerate(specs):
        options = spec.settings()
        if not any(options[key] for key in ("lora_hook_model_strength", "lora_hook_clip_strength")):
            raise ValueError("Each scheduled source needs a nonzero model or CLIP strength.")
        if options["lora_hook_clip_strength"] and family not in CLIP_FAMILIES:
            raise ValueError(
                "This checkpoint has no external CLIP target. Set CLIP strength to 0 for model-only hooks."
            )
        category = "checkpoints" if spec.kind == "checkpoint" else "loras"
        item = _artifact(
            manager, "lora-hook" if index == 0 else f"lora-hook-{index}", category, spec.name
        )
        path = item.request.destination.resolve()
        if (
            item.signature is None
            or path.suffix.lower() not in NATIVE_STATE_DICT_SUFFIXES
            or not any(
                path.is_relative_to(Path(root).resolve())
                for root in manager.get_model_folders(category)
            )
        ):
            raise ValueError(
                "Install a native LoRA state dictionary or checkpoint in its configured model folder."
            )
        if tuple(file_signature(path)) != item.signature:
            raise ValueError("The scheduled source changed during capture. Select it again.")
        sources.append(source_recipes.CapturedSource(spec, item))
    result.update(values)
    result[KEY] = HookPlan(
        tuple(file_signature(checkpoint)),
        family,
        sources[0].artifact,
        tuple(values.items()),
        frames,
        tuple(sources),
    )
    result["image_engine"] = "native"
    return result, tuple(source.artifact for source in sources)


def _build_source(plan, native, loaders):
    values = plan.settings()
    kind = plan.spec.kind if isinstance(plan, source_recipes.CapturedSource) else "lora"
    identity = kind, canonical_path_key(plan.artifact.signature[0])
    if identity not in loaders:
        loaders[identity] = (
            native.CreateHookModelAsLoraModelOnly()
            if kind == "checkpoint"
            else native.CreateHookLoraModelOnly()
        )
    loader = loaders[identity]
    if kind == "checkpoint":
        if values["lora_hook_clip_strength"] == 0:
            hooks = loader.create_hook_model_only(
                plan.artifact.name, values["lora_hook_model_strength"]
            )[0]
        else:
            hooks = loader.create_hook(
                plan.artifact.name,
                values["lora_hook_model_strength"],
                values["lora_hook_clip_strength"],
            )[0]
    elif values["lora_hook_clip_strength"] == 0:
        hooks = loader.create_hook_model_only(
            plan.artifact.name, values["lora_hook_model_strength"]
        )[0]
    else:
        hooks = loader.create_hook(
            plan.artifact.name,
            values["lora_hook_model_strength"],
            values["lora_hook_clip_strength"],
        )[0]
    frames = None
    if values["lora_hook_mode"] == "keyframes":
        for start, strength in plan.keyframes:
            frames = native.CreateHookKeyframe().create_hook_keyframe(strength, start, frames)[0]
    elif values["lora_hook_mode"] == "interpolated":
        frames = native.CreateHookKeyframesInterpolated().create_hook_keyframes(
            values["lora_hook_strength_start"],
            values["lora_hook_strength_end"],
            values["lora_hook_interpolation"],
            values["lora_hook_start"],
            values["lora_hook_end"],
            values["lora_hook_keyframe_count"],
        )[0]
    else:
        frames = native.CreateHookKeyframesFromFloats().create_hook_keyframes(
            json.loads(values["lora_hook_strengths"]),
            values["lora_hook_start"],
            values["lora_hook_end"],
        )[0]
    return native.SetHookKeyframes().set_hook_keyframes(hooks, frames)[0], frames


def build_hooks(plan, native):
    groups, frames, loaders = [], [], {}
    for source in plan.sources or (plan,):
        group, keyframes = _build_source(source, native, loaders)
        groups.append(group)
        frames.append(keyframes)
    if len(groups) == 1:
        combined = groups[0]
    elif len(groups) == 2:
        combined = native.CombineHooks().combine_hooks(*groups)[0]
    elif len(groups) <= 4:
        combined = native.CombineHooksFour().combine_hooks(*groups)[0]
    else:
        combined = native.CombineHooksEight().combine_hooks(*groups)[0]
    return combined, frames[0]


def _patch_count(patcher, hooks):
    return sum(len(patcher.hook_patches.get(hook.hook_ref, {})) for hook in hooks.hooks)


def _strength(hook, target):
    return hook._strength_clip if target == "clip" else hook._strength_model


def _require_mapping(patcher, hooks, target):
    for index, hook in enumerate(hooks.hooks, 1):
        if _strength(hook, target) != 0 and not patcher.hook_patches.get(hook.hook_ref):
            label = "CLIP" if target == "clip" else "diffusion-model"
            raise ValueError(f"Scheduled source {index} has no matching {label} hook weights.")
    return _patch_count(patcher, hooks)


def _require_source_weights(hooks):
    for hook in hooks.hooks:
        if hook.need_weight_init:
            continue
        for target, weights in (("model", hook.weights), ("clip", hook.weights_clip)):
            if _strength(hook, target) != 0 and not weights:
                raise ValueError(
                    f"The checkpoint-as-LoRA source has no {target.upper()} weights for its requested target."
                )


def _require_hook_weights(patcher, hooks, target="model"):
    """The pinned GGUF static adapter does not implement packed hook updates."""
    from comfy.model_patcher import get_key_weight
    from comfy.utils import get_attr

    for hook in hooks.hooks:
        for key in patcher.hook_patches.get(hook.hook_ref, {}):
            weight, set_weight, convert_weight = get_key_weight(patcher.model, key)
            qtype = getattr(weight, "tensor_type", None)
            packed_ggml = qtype is not None and getattr(qtype, "name", "") not in ("F32", "F16")
            module = get_attr(patcher.model, key.rsplit(".", 1)[0]) if "." in key else None
            packed_custom = bool(getattr(module, "_quantized", False)) and callable(
                getattr(module, "install_patch_entries", None)
            )
            if (packed_ggml or packed_custom) and set_weight is None and convert_weight is None:
                raise ValueError(
                    "Scheduled hooks on these packed GGUF weights still require a quantized-hook adapter. Use a static LoRA or floating-point weights."
                )
            if not hook.need_weight_init:
                source_weights = hook.weights_clip if target == "clip" else hook.weights
                source_weight = source_weights[key][1][0]
                if tuple(source_weight.shape) != tuple(weight.shape):
                    raise ValueError(
                        "The checkpoint-as-LoRA source has a different weight shape from its requested target."
                    )


@contextmanager
def request_scope(owner, data, manager, check_cancelled):
    plan = (data or {}).get(KEY)
    if plan is None:
        if (data or {}).get("lora_hook_mode", "off") != "off":
            raise ValueError("Capture the LoRA hook schedule before generation.")
        yield
        return
    if not isinstance(plan, HookPlan):
        raise ValueError("Capture the LoRA hook schedule before generation.")
    from ruinedfooocus.app.resource_lifecycle import require_compute_owner
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from comfy import hooks as native_hooks
    from comfy.patcher_extension import CallbacksMP
    from comfy_extras import nodes_hooks as native
    import folder_paths

    require_compute_owner()
    check_cancelled()
    if tuple(getattr(owner, "base_file_signature", ())) != plan.checkpoint:
        raise ValueError("The LoRA hook schedule belongs to a different loaded checkpoint.")
    if tuple(file_signature(plan.checkpoint[0])) != plan.checkpoint:
        raise ValueError("The checkpoint changed after LoRA hook submission. Submit it again.")
    values = plan.settings()
    data.update(values)
    original, conditions = owner.xl_base_patched, owner.conditions
    tracked, model_ids, clip_ids = [], set(), set()
    hooks = None
    previous = getattr(owner, "_request_lora_hooks", None)
    callback_key = f"rf-lora-hooks-{id(tracked)}"

    def track(patcher, target):
        if all(patcher is not old for old in tracked):
            tracked.append(patcher)
            patcher.remove_callbacks_with_key(CallbacksMP.ON_CLONE, callback_key)
            patcher.add_callback_with_key(
                CallbacksMP.ON_CLONE, callback_key, lambda old, new: track(new, target)
            )
        (clip_ids if target == "clip" else model_ids).add(id(patcher.model))

    def register(hook, patcher, options, target, registered):
        clip = target.get("target") == native_hooks.EnumWeightTarget.Clip
        return bool(_strength(hook, "clip" if clip else "model")) and id(patcher.model) in (
            clip_ids if clip else model_ids
        )

    with ExitStack() as stack:
        files, leased = [], {}
        sources = plan.sources or (
            source_recipes.CapturedSource(
                source_recipes.SourceSpec("lora", plan.artifact.name, plan.values, plan.keyframes),
                plan.artifact,
            ),
        )
        for source in sources:
            item = source.artifact
            identity = canonical_path_key(item.signature[0])
            if identity not in leased:
                leased[identity] = stack.enter_context(
                    prepared_asset(data, item.role, item.name, manager, source.category)
                )
            path = leased[identity]
            if (
                tuple(file_signature(path)) != item.signature
                or Path(folder_paths.get_full_path_or_raise(source.category, item.name)).resolve()
                != path.resolve()
            ):
                raise ValueError(
                    "The native scheduled source differs from the captured local file."
                )
            files.append((path, item))
        try:
            hooks, frames = build_hooks(plan, native)
            _require_source_weights(hooks)
            if any(tuple(file_signature(path)) != item.signature for path, item in files):
                raise ValueError("A scheduled source changed while loading. Submit it again.")
            check_cancelled()
            model_hooks = native_hooks.HookGroup()
            clip_hooks = native_hooks.HookGroup()
            for hook in hooks.hooks:
                if _strength(hook, "model") != 0:
                    model_hooks.add(hook)
                if _strength(hook, "clip") != 0:
                    clip_hooks.add(hook)
            unet = original.unet
            if model_hooks.hooks:
                unet = unet.clone()
                track(unet, "model")
            clip = original.clip
            if clip_hooks.hooks:
                if clip is None:
                    raise ValueError("This checkpoint has no CLIP target for the scheduled LoRA.")
                clip = clip.clone(disable_dynamic=True)
                track(clip.patcher, "clip")
            for hook in hooks.hooks:
                hook.custom_should_register = register
            model_count = clip_count = 0
            if model_hooks.hooks:
                unet.register_all_hook_patches(
                    model_hooks,
                    native_hooks.create_target_dict(native_hooks.EnumWeightTarget.Model),
                )
                model_count = _require_mapping(unet, model_hooks, "model")
                _require_hook_weights(unet, model_hooks)
            if clip_hooks.hooks:
                clip = native.SetClipHooks().apply_hooks(
                    clip, values["lora_hook_schedule_clip"], False, clip_hooks
                )[0]
                track(clip.patcher, "clip")
                clip_count = _require_mapping(clip.patcher, clip_hooks, "clip")
                _require_hook_weights(clip.patcher, clip_hooks, "clip")
            check_cancelled()
            owner.xl_base_patched = owner.StableDiffusionModel(
                unet, original.vae, clip, original.clip_vision
            )
            owner.conditions = None
            owner._request_lora_hooks = model_hooks if model_hooks.hooks else None
            native_nodes = []
            for source in sources:
                options = source.settings()
                name = (
                    "CreateHookModelAsLora"
                    if source.spec.kind == "checkpoint"
                    else "CreateHookLora"
                )
                if options["lora_hook_clip_strength"] == 0:
                    name += "ModelOnly"
                schedule = {
                    "keyframes": "CreateHookKeyframe",
                    "interpolated": "CreateHookKeyframesInterpolated",
                    "floats": "CreateHookKeyframesFromFloats",
                }[options["lora_hook_mode"]]
                for node in (name, schedule, "SetHookKeyframes"):
                    if node not in native_nodes:
                        native_nodes.append(node)
            if len(sources) > 1:
                native_nodes.append(
                    "CombineHooks2"
                    if len(sources) == 2
                    else "CombineHooks4"
                    if len(sources) <= 4
                    else "CombineHooks8"
                )
            if model_count:
                native_nodes.append("SetModelHooksOnCond")
            if clip_count:
                native_nodes.append("SetClipHooks")
            data["native_lora_hooks"] = {
                "native_nodes": native_nodes,
                "target": "primary-checkpoint",
                "model_patches": model_count,
                "clip_patches": clip_count,
                "file_signature": list(plan.artifact.signature),
                "checkpoint_signature": list(plan.checkpoint),
                "sources": [
                    dict(
                        type=source.spec.kind,
                        name=source.artifact.name,
                        signature=list(source.artifact.signature),
                        model_strength=_strength(hook, "model"),
                        clip_strength=_strength(hook, "clip"),
                        keyframes=[
                            dict(
                                start=float(frame.start_percent),
                                strength=float(frame.strength),
                                guarantee_steps=frame.guarantee_steps,
                            )
                            for frame in hook.hook_keyframe.keyframes
                        ],
                    )
                    for source, hook in zip(sources, hooks.hooks, strict=True)
                ],
                "keyframes": [
                    dict(
                        start=float(frame.start_percent),
                        strength=float(frame.strength),
                        guarantee_steps=frame.guarantee_steps,
                    )
                    for frame in frames.keyframes
                ],
            }
            yield
        finally:
            active_error = sys.exc_info()[0] is not None
            cleanup_error = None
            for patcher in reversed(tracked):
                try:
                    patcher.clean_hooks()
                except Exception as error:
                    cleanup_error = cleanup_error or error
                finally:
                    patcher.remove_callbacks_with_key(CallbacksMP.ON_CLONE, callback_key)
            if hooks is not None:
                hooks.reset()
            owner.xl_base_patched, owner.conditions = original, conditions
            if previous is None:
                owner.__dict__.pop("_request_lora_hooks", None)
            else:
                owner._request_lora_hooks = previous
            if cleanup_error is not None and not active_error:
                raise cleanup_error


def scoped(function):
    @wraps(function)
    def process(owner, gen_data=None, callback=None):
        if not gen_data or gen_data.get("lora_hook_mode", "off") == "off" and KEY not in gen_data:
            return function(owner, gen_data=gen_data, callback=callback)
        from ruinedfooocus.app import async_worker as worker
        from ruinedfooocus.app.state import path_manager

        with request_scope(owner, gen_data, path_manager, lambda: worker.check_interrupt(gen_data)):
            return function(owner, gen_data=gen_data, callback=callback)

    return process


def conditioning(owner, positive, negative):
    hooks = getattr(owner, "_request_lora_hooks", None)
    if hooks is None:
        return positive, negative
    from comfy_extras.nodes_hooks import SetModelHooksOnCond

    node = SetModelHooksOnCond()
    return node.attach_hook(positive, hooks)[0], node.attach_hook(negative, hooks)[0]
