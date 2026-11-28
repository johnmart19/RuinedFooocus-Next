"""Captured native Cascade super-resolution and ordered CLIP vision guidance."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature

KEY = "_cascade_guidance"
_read_paths = ContextVar("cascade_guidance_read_paths", default=None)
FIELDS = (
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
)
IMAGE_FIELD = "cascade_vision_images"
DEFAULTS = dict(
    cascade_super_resolution=False,
    cascade_super_resolution_model=None,
    cascade_control_strength=1,
    cascade_control_start=0,
    cascade_control_end=1,
    cascade_control_denoise=1,
    cascade_vision_enabled=False,
    cascade_vision_model=None,
    cascade_vision_strengths=[],
    cascade_vision_crop="none",
)


def replay_values(metadata):
    return {key: metadata.get(key, DEFAULTS[key]) for key in FIELDS}


@dataclass(frozen=True)
class VisionReference:
    size: tuple[int, int]
    sha256: str
    pixels: bytes = field(repr=False)

    def image(self):
        from PIL import Image

        return Image.frombytes("RGB", self.size, self.pixels)


@dataclass(frozen=True)
class CascadeGuidance:
    primary: tuple
    primary_name: str
    super_resolution: bool
    control_model: str | None
    control_strength: float
    start: float
    end: float
    denoise: float
    vision_model: str | None
    vision: tuple[VisionReference, ...]
    strengths: tuple[float, ...]
    crop: str


@contextmanager
def _asset(data, role, name, manager, category):
    active = _read_paths.get()
    if active is not None and active[0] is data.get(KEY):
        yield active[1][(role, name)]
        return
    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    with prepared_asset(data, role, name, manager, category) as path:
        yield path


def stage_read_leases(function):
    """Keep lazy native primary/control/vision reads under the existing job admission."""

    @wraps(function)
    def run(owner, gen_data=None, callback=None):
        plan = (gen_data or {}).get(KEY)
        if not isinstance(plan, CascadeGuidance):
            return function(owner, gen_data=gen_data, callback=callback)
        from ruinedfooocus.app.state import path_manager
        from ruinedfooocus.features.generation.image_preparation import prepared_asset

        selections = [("cascade-primary", plan.primary_name, "checkpoints")]
        if plan.control_model:
            selections.append(("cascade-control", plan.control_model, "controlnet"))
        if plan.vision_model:
            selections.append(("cascade-vision", plan.vision_model, "clip_vision"))
        with ExitStack() as stack:
            paths = {
                (role, name): stack.enter_context(
                    prepared_asset(gen_data, role, name, path_manager, category)
                )
                for role, name, category in selections
            }
            signatures = {key: tuple(file_signature(path)) for key, path in paths.items()}
            token = _read_paths.set((plan, paths))
            try:
                result = function(owner, gen_data=gen_data, callback=callback)
                if any(
                    tuple(file_signature(path)) != signatures[key] for key, path in paths.items()
                ):
                    raise ValueError(
                        "A Cascade guidance source changed while sampling. Submit again."
                    )
                return result
            finally:
                _read_paths.reset(token)

    return run


def _number(data, key, default, low, high):
    value = data.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}.")
    return float(value)


def _name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Select an installed Cascade guidance component.")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("Select Cascade guidance components from the configured model folders.")
    return path.as_posix()


def compatible_file(role, path):
    from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES, SAFETENSORS_SUFFIXES

    # Native CLIP vision reads these weights-only containers during Run. A
    # picker can offer an explicit unknown choice without deserializing it.
    if role == "vision" and Path(path).suffix.casefold() in NATIVE_TORCH_SUFFIXES:
        return True
    if Path(path).suffix.casefold() not in SAFETENSORS_SUFFIXES:
        return False
    try:
        header = cached_safetensors_header(path)

        def shape(key):
            return header.get(key, {}).get("shape")

        if role == "control":
            return (
                shape("backbone.0.weight") == [16384, 16, 1, 1]
                and shape("backbone.2.weight") == [1024, 16384, 1, 1]
                and shape("backbone.10.blocks.0.weight") == [1024]
                and shape("backbone.11.weight") == [1280, 1024, 1, 1]
                and all(
                    shape(f"projections.{index}.2.weight") == [2048, 1280, 1, 1]
                    for index in range(8)
                )
            )
        if role != "vision":
            return False
        return (
            shape("vision_model.embeddings.patch_embedding.weight") == [1024, 3, 14, 14]
            and shape("vision_model.embeddings.position_embedding.weight")
            in ([257, 1024], [577, 1024])
            and shape("vision_model.encoder.layers.0.layer_norm1.weight") == [1024]
            and shape("vision_model.encoder.layers.23.layer_norm1.weight") == [1024]
            and shape("vision_model.encoder.layers.22.layer_norm1.weight") == [1024]
            and "vision_model.encoder.layers.30.layer_norm1.weight" not in header
            and "vision_model.encoder.layers.47.layer_norm1.weight" not in header
            and shape("visual_projection.weight") == [768, 1024]
        ) or (
            shape("visual.conv1.weight") == [1024, 3, 14, 14]
            and shape("visual.positional_embedding") in ([257, 1024], [577, 1024])
            and shape("visual.transformer.resblocks.23.ln_1.weight") == [1024]
            and shape("visual.transformer.resblocks.22.ln_1.weight") == [1024]
            and shape("visual.transformer.resblocks.0.ln_1.weight") == [1024]
            and shape("visual.transformer.resblocks.0.attn.in_proj_weight") == [3072, 1024]
            and "visual.transformer.resblocks.30.ln_1.weight" not in header
            and "visual.transformer.resblocks.47.ln_1.weight" not in header
            and shape("visual.proj") == [1024, 768]
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def local_choices(manager, role):
    names = set()
    for root in manager.get_model_folders("controlnet" if role == "control" else "clip_vision"):
        root = Path(root).resolve()
        pending, visited = [root], set()
        while pending:
            folder = pending.pop()
            resolved = folder.resolve()
            if not resolved.is_relative_to(root) or resolved in visited:
                continue
            visited.add(resolved)
            try:
                children = sorted(folder.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif (
                    path.is_file()
                    and path.resolve().is_relative_to(root)
                    and compatible_file(role, path)
                ):
                    names.add(path.relative_to(root).as_posix())
    return sorted(names)


def strengths(value, count):
    if isinstance(value, str):
        try:
            value = (
                json.loads(value)
                if value.strip().startswith("[")
                else [float(part.strip()) for part in value.split(",") if part.strip()]
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Enter one vision strength per image, in the displayed order."
            ) from error
    if value in (None, [], ()):
        return (1.0,) * count
    if (
        not isinstance(value, (list, tuple))
        or len(value) != count
        or any(
            type(weight) not in (int, float) or not math.isfinite(weight) or not -10 <= weight <= 10
            for weight in value
        )
    ):
        raise ValueError(
            "Enter one vision strength from -10 through 10 per image, in the displayed order."
        )
    return tuple(map(float, value))


def capture_assets(data, manager, models):
    from PIL import Image, ImageOps
    from ruinedfooocus.features.generation.image_preparation import _artifact

    result = dict(data)
    result.pop(KEY, None)
    result.pop("native_cascade_guidance", None)
    sr, vision = (
        result.get("cascade_super_resolution", False),
        result.get("cascade_vision_enabled", False),
    )
    if type(sr) is not bool or type(vision) is not bool:
        raise ValueError("Choose Cascade guidance with its explicit enable controls.")
    if not sr and not vision:
        return result, ()
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Cascade guidance requires the native image engine.")
    name = result.get("base_model_name")
    primary = models.get_file("checkpoints", name) if models is not None and name else None
    if primary is None or models.detect_checkpoint_base(primary) not in (
        "Stable_Cascade_C",
        "StableCascade_C",
    ):
        raise ValueError("Cascade guidance requires a native Stage C primary checkpoint.")
    from ruinedfooocus.features.generation.cascade import require_bundle, is_source_request

    require_bundle(primary, "c")
    if sr:
        header = cached_safetensors_header(primary)
        mapper = next(
            (
                value.get("shape", ())
                for key, value in header.items()
                if key.endswith("clip_txt_mapper.weight")
            ),
            (),
        )
        if not mapper or mapper[0] != 2048:
            raise ValueError(
                "The native super-resolution adapter requires the full 2048-wide Stage C model."
            )
    if sr and not is_source_request(result):
        raise ValueError(
            "Add a source image in Native image edit or Image to image for Cascade super-resolution."
        )
    artifacts, selected = [_artifact(manager, "cascade-primary", "checkpoints", _name(name))], {}
    for enabled, role, category, key in (
        (sr, "control", "controlnet", "cascade_super_resolution_model"),
        (vision, "vision", "clip_vision", "cascade_vision_model"),
    ):
        if enabled:
            selected[role] = _name(result.get(key))
            item = _artifact(manager, "cascade-" + role, category, selected[role])
            if item.signature is None or not compatible_file(role, item.request.destination):
                raise ValueError(f"Choose an installed compatible native Cascade {role} component.")
            artifacts.append(item)
    references = []
    if vision:
        images = result.get(IMAGE_FIELD) or []
        if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 8:
            raise ValueError("Add one through eight Cascade vision images.")
        for image in images:
            if isinstance(image, (list, tuple)) and len(image) == 2:
                image = image[0]
            if not isinstance(image, Image.Image):
                raise ValueError("Cascade vision references must be uploaded images.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = image.tobytes()
            references.append(
                VisionReference(image.size, hashlib.sha256(pixels).hexdigest(), pixels)
            )
    weights = strengths(result.get("cascade_vision_strengths"), len(references)) if vision else ()
    crop = result.get("cascade_vision_crop", "none") if vision else "none"
    if crop not in ("none", "center"):
        raise ValueError("Choose Native full image or Native center crop for Cascade vision.")
    start, end = (
        (
            _number(result, "cascade_control_start", 0, 0, 1),
            _number(result, "cascade_control_end", 1, 0, 1),
        )
        if sr
        else (0.0, 1.0)
    )
    if not start < end:
        raise ValueError("Cascade control requires start before end within zero through one.")
    result[KEY] = CascadeGuidance(
        tuple(file_signature(primary)),
        _name(name),
        sr,
        selected.get("control"),
        _number(result, "cascade_control_strength", 1, 0, 10) if sr else 1.0,
        start,
        end,
        _number(result, "cascade_control_denoise", 1, 0, 1) if sr else 1.0,
        selected.get("vision"),
        tuple(references),
        weights,
        crop,
    )
    result["cascade_vision_strengths"] = list(weights)
    result["image_engine"] = "native"
    return result, tuple(artifacts)


def require_plan(owner, data):
    plan = data.get(KEY)
    if (
        not isinstance(plan, CascadeGuidance)
        or type(owner.xl_base_patched.unet.model).__name__ != "StableCascade_C"
    ):
        raise ValueError("Capture Cascade guidance with its native Stage C model before sampling.")
    if tuple(getattr(owner, "base_file_signature", ())) != plan.primary:
        raise ValueError("The loaded Stage C source differs from the captured guidance request.")
    current = (
        _name(data.get("base_model_name")),
        data.get("cascade_super_resolution", False),
        _name(data.get("cascade_super_resolution_model")) if plan.super_resolution else None,
        _number(data, "cascade_control_strength", 1, 0, 10) if plan.super_resolution else 1.0,
        _number(data, "cascade_control_start", 0, 0, 1) if plan.super_resolution else 0.0,
        _number(data, "cascade_control_end", 1, 0, 1) if plan.super_resolution else 1.0,
        _number(data, "cascade_control_denoise", 1, 0, 1) if plan.super_resolution else 1.0,
        data.get("cascade_vision_enabled", False),
        _name(data.get("cascade_vision_model")) if plan.vision else None,
        strengths(data.get("cascade_vision_strengths"), len(plan.vision)) if plan.vision else (),
        data.get("cascade_vision_crop", "none") if plan.vision else "none",
    )
    expected = (
        plan.primary_name,
        plan.super_resolution,
        plan.control_model,
        plan.control_strength,
        plan.start,
        plan.end,
        plan.denoise,
        bool(plan.vision),
        plan.vision_model,
        plan.strengths,
        plan.crop,
    )
    if current != expected:
        raise ValueError("Cascade guidance settings changed after capture. Submit again.")
    config = owner.xl_base_patched.unet.model.model_config.unet_config
    if plan.super_resolution and config.get("c_hidden", [2048, 2048]) != [2048, 2048]:
        raise ValueError(
            "The native super-resolution adapter requires full Stage C projection widths."
        )
    return plan


def super_resolution_source(vae, image, data, *, mode):
    from PIL import ImageOps
    from comfy_extras.nodes_stable_cascade import StableCascade_SuperResolutionControlnet
    from ruinedfooocus.features.generation.cascade import CascadeLatents, source_framing
    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

    plan = data.get(KEY)
    if not isinstance(plan, CascadeGuidance) or not plan.super_resolution:
        raise ValueError("Capture Cascade super-resolution before source encoding.")
    source_framing(mode)
    if getattr(vae, "downscale_ratio", None) != 32:
        raise ValueError("Cascade super-resolution requires the native 32-fold EffNet encoder.")
    image = ImageOps.exif_transpose(image)
    size = data.get(SIZE_KEY)
    if isinstance(size, SizePlan):
        from ruinedfooocus.features.generation.automatic_upscale import scale_for

        if size.scale != 2 * scale_for(data.get("automatic_upscale", "Off")):
            raise ValueError("Capture a new output size after changing Cascade super-resolution.")
    target = size.content if isinstance(size, SizePlan) else image.size
    pixels, frame = prepare_reference(image, target, "pad", multiple=32)
    hint, stage_c, stage_b = StableCascade_SuperResolutionControlnet.execute(pixels, vae)
    if (
        stage_c["samples"].shape[1] != 16
        or stage_b["samples"].shape[1] != 4
        or hint.shape[-1] != 16
    ):
        raise ValueError(
            "The native Cascade super-resolution encoder returned incompatible stage channels."
        )
    output = dict(frame)
    for key in (
        "width",
        "height",
        "canvas_width",
        "canvas_height",
        "left",
        "top",
        "content_width",
        "content_height",
        "content_left",
        "content_top",
    ):
        output[key] *= 2
    data["height"], data["width"] = (int(side) * 4 for side in stage_b["samples"].shape[-2:])
    data["native_cascade"] = dict(
        native_encoder="StableCascade_SuperResolutionControlnet",
        source_width=image.width,
        source_height=image.height,
        native_source_scale=2,
        source_frame=frame,
        output_frame=output,
        native_input_width=int(pixels.shape[2]),
        native_input_height=int(pixels.shape[1]),
        output_width=data["width"],
        output_height=data["height"],
        stage_c_shape=list(stage_c["samples"].shape),
        stage_b_shape=list(stage_b["samples"].shape),
    )
    return pixels, output, CascadeLatents(stage_c, stage_b, hint)


def apply_guidance(owner, positive, negative, data, manager, pixels, check_cancelled):
    from nodes import CLIPVisionEncode, unCLIPConditioning, ControlNetApplyAdvanced

    plan = require_plan(owner, data)
    cache = getattr(owner, "cascade_guidance_cache", None)
    if cache is None:
        cache = owner.cascade_guidance_cache = {}
    metadata = dict(native_source_scale=2 if plan.super_resolution else 1, vision=[])
    with ExitStack() as stack:
        sources = []

        def load(role, name, category, reader):
            check_cancelled()
            path = stack.enter_context(_asset(data, "cascade-" + role, name, manager, category))
            signature = tuple(file_signature(path))
            sources.append((path, signature))
            previous = cache.get(role)
            if previous is None or previous[0] != signature:
                try:
                    loaded = reader(str(path))
                except (ValueError, RuntimeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"The selected native Cascade {role} state dictionary "
                        "is invalid or incompatible. " + str(error)
                    ) from error
                if loaded is None:
                    raise ValueError(
                        f"The native Cascade {role} loader did not return a component."
                    )
                check_cancelled()
                if tuple(file_signature(path)) != signature:
                    raise ValueError(
                        "The Cascade guidance component changed while loading. Submit again."
                    )
                cache[role] = (signature, loaded)
            metadata[role + "_source"] = dict(name=name, signature=signature)
            return cache[role][1]

        if plan.super_resolution:
            from comfy.controlnet import load_controlnet

            if pixels is None:
                raise ValueError("Cascade super-resolution needs its native EffNet control hint.")
            control = load("control", plan.control_model, "controlnet", load_controlnet)
            if (
                type(control).__name__ != "T2IAdapter"
                or control.channels_in != 16
                or control.compression_ratio != 1
                or getattr(control.t2i_model, "proj_blocks", None) != [0, 4, 8, 12, 51, 55, 59, 63]
            ):
                raise ValueError(
                    "The control component is not the native 16-channel Cascade super-resolution adapter."
                )
            positive, negative = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, control, pixels, plan.control_strength, plan.start, plan.end
            )
            metadata.update(
                control_strength=plan.control_strength,
                start=plan.start,
                end=plan.end,
                denoise=plan.denoise,
                native_control="StableCascade_SuperResolutionControlnet + ControlNetApplyAdvanced",
            )
        if plan.vision:
            from comfy.clip_vision import load as load_vision

            vision = load("vision", plan.vision_model, "clip_vision", load_vision)
            for index, (reference, weight) in enumerate(zip(plan.vision, plan.strengths), 1):
                check_cancelled()
                if weight != 0:
                    import numpy as np
                    import torch

                    image = (
                        torch.from_numpy(np.asarray(reference.image()).copy()).float().unsqueeze(0)
                        / 255
                    )
                    encoded = CLIPVisionEncode().encode(vision, image, plan.crop)[0]
                    if tuple(encoded.image_embeds.shape) != (1, 768):
                        raise ValueError(
                            "Cascade vision requires one native 768-wide projected CLIP image embedding."
                        )
                    positive = unCLIPConditioning().apply_adm(positive, encoded, weight, 0.0)[0]
                metadata["vision"].append(
                    dict(
                        index=index,
                        strength=weight,
                        size=list(reference.size),
                        sha256=reference.sha256,
                    )
                )
            metadata.update(
                vision_crop=plan.crop, native_vision="CLIPVisionEncode + unCLIPConditioning"
            )
        check_cancelled()
        if any(tuple(file_signature(path)) != signature for path, signature in sources):
            raise ValueError("A Cascade guidance source changed during conditioning. Submit again.")
        data["native_cascade_guidance"] = metadata
    return positive, negative
