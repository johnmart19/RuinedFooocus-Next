"""Typed, request-local adapters for the pinned native model-patch nodes."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePosixPath

from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES, is_safetensors
from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature

KEY = "_native_model_patch"
SOURCE_KEY = "_native_model_patch_source"
CATEGORY = "model_patches"
DEFAULTS = dict(
    model_patch_recipe="none",
    model_patch_model="",
    model_patch_vision="",
    model_patch_strength=1.0,
    model_patch_start=0.0,
    model_patch_stop=1.0,
    model_patch_denoise=1.0,
)
FIELDS = tuple(DEFAULTS)
IMAGE_FIELDS = ("model_patch_image", "model_patch_inpaint_image", "model_patch_mask")
RECIPES = {
    "anima-lllite": ("Anima", "AnimaLLLiteApply.apply_patch"),
    "qwen-diffsynth": ("QwenImage", "QwenImageDiffsynthControlnet.diffsynth_controlnet"),
    "zimage-fun": ("ZImage", "ZImageFunControlnet.diffsynth_controlnet"),
    "uso-style": ("Flux", "USOStyleReference.apply_patch"),
}


def recipes_for_family(family):
    from ruinedfooocus.features.models.image_components import ALIASES

    family = ALIASES.get(family, family)
    if family == "FluxSchnell":
        family = "Flux"
    return ["none", *(name for name, (target, _) in RECIPES.items() if target == family)]


def reset_values():
    return {"model_patch_recipe": "none", **dict.fromkeys(IMAGE_FIELDS)}


def replay_values(metadata, *, clear_images=True):
    """Restore named settings only; metadata never restores an uploaded source."""
    values = {key: metadata.get(key, default) for key, default in DEFAULTS.items()}
    settings(values)
    for key in ("model_patch_model", "model_patch_vision"):
        name = values[key]
        relative = PurePosixPath(name.replace("\\", "/")) if isinstance(name, str) else None
        if (
            relative is None
            or len(name) > 512
            or relative.is_absolute()
            or ".." in relative.parts
            or ":" in name
        ):
            raise ValueError("Saved native patch files must be relative local filenames.")
    if clear_images:
        values.update(dict.fromkeys(IMAGE_FIELDS))
    return values


def api_values(payload, decode_image, defaults=None):
    """API recipes are explicit; only matching-checkpoint file/number defaults carry."""
    values = {key: (defaults or {}).get(key, default) for key, default in DEFAULTS.items()}
    values.update((key, payload[key]) for key in FIELDS if key in payload)
    values["model_patch_recipe"] = payload.get("model_patch_recipe", "none")
    values = replay_values(values, clear_images=False)
    for key in IMAGE_FIELDS:
        values[key] = decode_image(payload[key], preserve_alpha=True) if payload.get(key) else None
    return values


def _shape(header, key):
    entry = header.get(key)
    value = entry.get("shape") if isinstance(entry, dict) else None
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def classify_patch(header):
    """Match native ModelPatchLoader branches, never generic ControlNet names."""
    if "lllite_conditioning1.conv1.weight" in header:
        first = _shape(header, "lllite_conditioning1.conv1.weight")
        metadata = header.get("__metadata__") or {}
        if not isinstance(metadata, dict):
            return None
        if (
            len(first) == 4
            and first[1] in (3, 4)
            and metadata.get("lllite.version", "2") == "2"
            and any(
                key.startswith("lllite_dit_blocks_") and key.endswith(".down.weight")
                for key in header
            )
        ):
            return "anima-lllite"
    elif "controlnet_blocks.0.y_rms.weight" in header:
        image = _shape(header, "img_in.weight")
        if len(image) == 2 and image[0] == 3072 and image[1] >= 64 and (image[1] - 64) % 4 == 0:
            return "qwen-diffsynth"
    elif "feature_embedder.mid_layer_norm.bias" in header:
        if (
            _shape(header, "feature_embedder.mid_layer_norm.bias") == (1152,)
            and _shape(header, "feature_embedder.mid_projection.weight") == (3072, 1152)
            and _shape(header, "feature_embedder.mid_embedding_linear.0.weight") == (64, 729)
        ):
            return "uso-style"
    elif "control_all_x_embedder.2-1.weight" in header:
        image = _shape(header, "control_all_x_embedder.2-1.weight")
        if len(image) == 2 and image[0] == 3840 and image[1] in (64, 132):
            return "zimage-fun"
    return None


def file_contract(path, recipe):
    """Safe headers can prove a role; Torch containers require admitted native validation."""
    path = Path(path)
    if recipe not in RECIPES:
        return "incompatible"
    if path.suffix.lower() in NATIVE_TORCH_SUFFIXES:
        return "unknown"
    if not is_safetensors(path):
        return "incompatible"
    try:
        return (
            "compatible"
            if classify_patch(cached_safetensors_header(path)) == recipe
            else "incompatible"
        )
    except (ValueError, OSError, TypeError):
        return "incompatible"


def local_choices(manager, recipe, *, vision=False):
    category = "clip_vision" if vision else CATEGORY
    choices, seen = [], set()
    for folder in manager.get_model_folders(category):
        root = Path(folder).resolve()
        pending, visited = [root], set()
        while pending:
            directory = pending.pop()
            resolved = directory.resolve()
            if not resolved.is_relative_to(root) or resolved in visited:
                continue
            visited.add(resolved)
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif path.is_file() and path.resolve().is_relative_to(root):
                    name = path.relative_to(root).as_posix()
                    if name in seen:
                        continue
                    seen.add(name)
                    status = _vision_contract(path) if vision else file_contract(path, recipe)
                    if status != "incompatible":
                        choices.append((name, status))
    return sorted(choices, key=lambda row: row[0].casefold())


def _vision_contract(path):
    if not is_safetensors(path):
        return "unknown" if Path(path).suffix.lower() in NATIVE_TORCH_SUFFIXES else "incompatible"
    try:
        header = cached_safetensors_header(path)
        compatible = (
            _shape(header, "vision_model.embeddings.patch_embedding.weight") == (1152, 3, 14, 14)
            and _shape(header, "vision_model.embeddings.position_embedding.weight") == (729, 1152)
            and _shape(header, "vision_model.encoder.layers.26.layer_norm1.weight") == (1152,)
        )
        return "compatible" if compatible else "incompatible"
    except (OSError, ValueError, TypeError):
        return "incompatible"


def settings(data):
    recipe = data.get("model_patch_recipe") or "none"
    if recipe not in ("none", *RECIPES):
        raise ValueError("Choose an available native model-patch recipe.")
    strength, start, stop = (
        data.get(key, default)
        for key, default in (
            ("model_patch_strength", 1.0),
            ("model_patch_start", 0.0),
            ("model_patch_stop", 1.0),
        )
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in (strength, start, stop)
    ) or not (-10 <= strength <= 10 and 0 <= start < stop <= 1):
        raise ValueError("Model patches require -10 ≤ strength ≤ 10 and 0 ≤ start < stop ≤ 1.")
    if recipe == "uso-style" and (strength, start, stop) != (1, 0, 1):
        raise ValueError(
            "USO uses its native style projection without strength or interval controls."
        )
    denoise = data.get("model_patch_denoise", 1.0)
    if type(denoise) not in (int, float) or not math.isfinite(denoise) or not 0 < denoise <= 1:
        raise ValueError("Fun source denoise must be greater than 0 and at most 1.")
    return recipe, float(strength), float(start), float(stop)


def _image(value, label, *, mask=False):
    from PIL import Image, ImageOps

    if value is None:
        return None
    if not isinstance(value, Image.Image) or not 0 < value.width * value.height <= 16_777_216:
        raise ValueError(f"Add a {label} image of at most 16 megapixels.")
    result = ImageOps.exif_transpose(value).convert("L" if mask else "RGB").copy()
    if mask and result.getextrema()[1] == 0:
        raise ValueError("The model-patch mask must mark the edit area in white.")
    return result


def inputs(data, recipe):
    guide = _image(data.get("model_patch_image"), "model-patch reference")
    source = _image(data.get("model_patch_inpaint_image"), "model-patch inpaint source")
    mask = _image(data.get("model_patch_mask"), "model-patch mask", mask=True)
    if recipe in ("anima-lllite", "qwen-diffsynth", "uso-style") and guide is None:
        raise ValueError("Add this model patch's reference or guide image.")
    if recipe == "uso-style" and (source is not None or mask is not None):
        raise ValueError("USO uses only its own style reference image.")
    if recipe == "qwen-diffsynth" and source is not None:
        raise ValueError(
            "Qwen DiffSynth uses a guide and optional mask; use AliMama for inpainting."
        )
    if recipe == "anima-lllite" and source is not None:
        raise ValueError(
            "Anima LLLite uses its guide and optional mask without a separate inpaint source."
        )
    if recipe == "zimage-fun" and guide is None and source is None:
        raise ValueError("Add a Z-Image guide or an inpaint source and mask.")
    if (
        source is not None
        and mask is None
        or source is None
        and recipe == "zimage-fun"
        and mask is not None
    ):
        raise ValueError("Add both the model patch's inpaint source and its mask.")
    target = source or guide
    if mask is not None and mask.size != target.size:
        raise ValueError("The model-patch mask must match its source image dimensions.")
    return guide, source, mask


@dataclass(frozen=True)
class CapturedPatch:
    recipe: str
    checkpoint_signature: tuple
    artifacts: tuple
    strength: float
    start: float
    stop: float


def _artifact(manager, recipe, role, name):
    from ruinedfooocus.features.generation.image_preparation import _artifact as image_artifact

    category = "clip_vision" if role == "model-patch-vision" else CATEGORY
    relative = PurePosixPath(str(name or "").replace("\\", "/"))
    if not name or relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise ValueError("Choose an installed native patch file from its configured folder.")
    item = image_artifact(manager, role, category, relative.as_posix())
    path = item.request.destination.resolve()
    if item.signature is None or not any(
        path.is_relative_to(Path(root).resolve()) for root in manager.get_model_folders(category)
    ):
        raise ValueError("The native patch file is missing or outside its configured folder.")
    status = _vision_contract(path) if role == "model-patch-vision" else file_contract(path, recipe)
    if status == "incompatible":
        raise ValueError("The selected native patch or vision file does not match this recipe.")
    if tuple(file_signature(path)) != item.signature:
        raise ValueError("The native patch file changed during capture. Select it again.")
    return item


def capture_assets(data, manager, models, controls):
    """Freeze this recipe's own inputs and exact local artifacts before enqueue."""
    result = dict(data)
    result.pop(KEY, None)
    result.pop(SOURCE_KEY, None)
    result.pop("native_model_patch", None)
    kind = str((controls or {}).get("type", "")).lower()
    if (
        "_api_job" not in result
        and result.get("task_type") != "tool_call"
        and kind not in ("", "none", "img2img")
    ):
        result["model_patch_recipe"] = "none"
        return result, ()
    recipe, strength, start, stop = settings(result)
    if recipe == "none":
        return result, ()
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Native model patches require the native image engine.")
    if (
        kind not in ("", "none", "img2img")
        or result.get("inpaint_toggle")
        or result.get("reference_adapter") not in (None, "", "none")
        or result.get("reference_images")
        or result.get("native_edit_recipe") not in (None, "", "auto")
        or result.get("control_recipe") not in (None, "", "ordinary")
    ):
        raise ValueError(
            "Native model patches use their own guide/source/mask on ordinary image requests."
        )
    checkpoint = models.get_file("checkpoints", result.get("base_model_name"))
    if checkpoint is None:
        raise ValueError("Select an installed image checkpoint for this model patch.")
    from ruinedfooocus.features.generation.image_preparation import _control_checkpoint_family

    family = _control_checkpoint_family(models, result.get("base_model_name"))
    if recipe not in recipes_for_family(family):
        raise ValueError("This model-patch recipe does not match the selected checkpoint family.")
    guide, source, mask = inputs(result, recipe)
    if source is not None and result.get("input_image") is not None:
        raise ValueError(
            "Use the patch's own inpaint source without a separate Image to Image source."
        )
    if source is not None and kind == "img2img":
        raise ValueError("Choose no Image tool when using the Fun patch's own inpaint source.")
    result.update(model_patch_image=guide, model_patch_inpaint_image=source, model_patch_mask=mask)
    if source is not None and result.get("image_resize_mode", "preserve") == "preserve":
        from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY

        result.pop(SIZE_KEY, None)
    artifacts = [_artifact(manager, recipe, "model-patch", result.get("model_patch_model"))]
    if recipe == "uso-style":
        artifacts.append(
            _artifact(manager, recipe, "model-patch-vision", result.get("model_patch_vision"))
        )
    result[KEY] = CapturedPatch(
        recipe, tuple(file_signature(checkpoint)), tuple(artifacts), strength, start, stop
    )
    result["image_engine"] = "native"
    return result, tuple(artifacts)


@dataclass(frozen=True)
class PreparedPatchSource:
    pixels: object
    mask: object
    frame: dict


def has_source(data):
    captured = data.get(KEY)
    return (
        isinstance(captured, CapturedPatch)
        and captured.recipe == "zimage-fun"
        and data.get("model_patch_inpaint_image") is not None
    )


def source_latent(vae, data, target_size, resize_mode, check_cancelled):
    """Encode an explicit Fun inpaint source and its native sampling noise mask."""
    captured = data.get(KEY)
    if not isinstance(captured, CapturedPatch) or captured.recipe != "zimage-fun":
        return None
    _, source, mask = inputs(data, captured.recipe)
    if source is None:
        return None
    from ruinedfooocus.app.resource_lifecycle import require_compute_owner
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference
    from ruinedfooocus.features.image_editing.inpaint_utils import prepare_mask
    from nodes import VAEEncode, SetLatentNoiseMask

    require_compute_owner()
    check_cancelled()
    if (
        vae.latent_channels != 16
        or vae.spacial_compression_encode() != 8
        or vae.output_channels != 3
    ):
        raise ValueError("Fun inpainting requires its 16-channel, 8× RGB VAE.")
    pixels, frame = prepare_reference(source, target_size, resize_mode, multiple=16)
    prepared_mask = prepare_mask(mask, frame)
    check_cancelled()
    latent = VAEEncode().encode(vae, pixels)[0]
    check_cancelled()
    latent = SetLatentNoiseMask().set_mask(latent, prepared_mask)[0]
    data[SOURCE_KEY] = PreparedPatchSource(pixels, prepared_mask, frame)
    return latent, pixels, prepared_mask, frame


def _target(recipe, model, vae):
    config = model.model.model_config
    native = type(config).__name__
    allowed = {
        "anima-lllite": ("Anima",),
        "qwen-diffsynth": ("QwenImage",),
        "zimage-fun": ("ZImage",),
        "uso-style": ("Flux", "FluxSchnell"),
    }[recipe]
    if native not in allowed:
        raise ValueError("The loaded native checkpoint does not match this model-patch recipe.")
    if recipe in ("qwen-diffsynth", "zimage-fun") and (
        vae.latent_channels != 16
        or vae.spacial_compression_encode() != 8
        or vae.output_channels != 3
    ):
        raise ValueError("This native patch requires its 16-channel, 8× RGB VAE.")
    if recipe == "uso-style" and (
        config.unet_config.get("in_channels") != 64 or config.unet_config.get("hidden_size") != 3072
    ):
        raise ValueError("USO requires the ordinary Flux 64-input, 3072-width native model.")


@contextmanager
def apply_scope(owner, model, vae, data, manager, check_cancelled):
    """Hold exact artifact leases through sampling; native nodes own all algorithms."""
    captured = data.get(KEY)
    if captured is None:
        if data.get("model_patch_recipe") not in (None, "", "none"):
            raise ValueError("Capture the native model patch before compute.")
        yield model
        return
    if not isinstance(captured, CapturedPatch):
        raise ValueError("The native model-patch capture is invalid.")
    from ruinedfooocus.app.resource_lifecycle import require_compute_owner
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    import folder_paths
    import numpy as np
    import torch

    require_compute_owner()
    check_cancelled()
    if tuple(getattr(owner, "base_file_signature", ())) != captured.checkpoint_signature:
        raise ValueError("This model patch was captured for a different checkpoint.")
    _target(captured.recipe, model, vae)
    guide, source, mask = inputs(data, captured.recipe)

    def tensor(image):
        return (
            torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255
            if image is not None
            else None
        )

    with ExitStack() as stack:
        paths = []
        for item in captured.artifacts:
            category = "clip_vision" if item.role == "model-patch-vision" else CATEGORY
            path = stack.enter_context(
                prepared_asset(data, item.role, item.name, manager, category)
            )
            if (
                Path(folder_paths.get_full_path_or_raise(category, item.name)).resolve()
                != path.resolve()
            ):
                raise ValueError("The native patch path differs from the captured local file.")
            paths.append(path)
        check_cancelled()
        from comfy_extras import nodes_model_patch as native

        identity = (captured.recipe, *(tuple(file_signature(path)) for path in paths))
        cached = getattr(owner, "native_model_patch_cache", None)
        if cached is None or cached[0] != identity:
            patch = native.ModelPatchLoader().load_model_patch(captured.artifacts[0].name)[0]
            vision = None
            if captured.recipe == "uso-style":
                import comfy.clip_vision

                vision = comfy.clip_vision.load(str(paths[1]))
                if vision is None:
                    raise ValueError("The native USO vision encoder could not be loaded.")
        else:
            patch, vision = cached[1]
        expected = {
            "anima-lllite": native.comfy.ldm.anima.lllite.AnimaLLLite,
            "qwen-diffsynth": native.QwenImageBlockWiseControlNet,
            "zimage-fun": native.comfy.ldm.lumina.controlnet.ZImage_Control,
            "uso-style": native.SigLIPMultiFeatProjModel,
        }[captured.recipe]
        if not isinstance(patch.model, expected):
            raise ValueError("The native loader returned a different model-patch type.")
        if (
            captured.recipe == "anima-lllite"
            and mask is not None
            and patch.model.cond_in_channels != 4
        ):
            raise ValueError("This Anima LLLite patch has no mask input channel.")
        if (
            captured.recipe == "qwen-diffsynth"
            and mask is not None
            and patch.model.additional_in_dim <= 0
        ):
            raise ValueError("This Qwen DiffSynth patch has no mask input channels.")
        if (
            captured.recipe == "zimage-fun"
            and source is not None
            and patch.model.additional_in_dim <= 0
        ):
            raise ValueError("Select an inpaint-capable Z-Image Fun patch for a source and mask.")
        check_cancelled()
        kwargs = dict(model=model, model_patch=patch)
        if captured.recipe == "anima-lllite":
            output = native.AnimaLLLiteApply().apply_patch(
                **kwargs,
                image=tensor(guide),
                mask=tensor(mask),
                strength=captured.strength,
                start_percent=captured.start,
                end_percent=captured.stop,
            )[0]
        elif captured.recipe == "uso-style":
            from nodes import CLIPVisionEncode

            encoded = CLIPVisionEncode().encode(vision, tensor(guide), "center")[0]
            states, last = encoded.all_hidden_states, encoded.penultimate_hidden_states
            if (
                states.ndim != 4
                or states.shape[1] < 20
                or tuple(states.shape[-2:]) != (729, 1152)
                or tuple(last.shape[-2:]) != (729, 1152)
            ):
                raise ValueError(
                    "USO needs the matching 729-token, 1152-wide SigLIP hidden states."
                )
            output = native.USOStyleReference().apply_patch(**kwargs, clip_vision_output=encoded)[0]
        else:
            node = (
                native.QwenImageDiffsynthControlnet
                if captured.recipe == "qwen-diffsynth"
                else native.ZImageFunControlnet
            )
            prepared = data.get(SOURCE_KEY)
            if source is not None and not isinstance(prepared, PreparedPatchSource):
                raise ValueError(
                    "Prepare the patch source latent and mask before applying its model patch."
                )
            output = node().diffsynth_controlnet(
                **kwargs,
                vae=vae,
                image=tensor(guide),
                inpaint_image=prepared.pixels if source is not None else None,
                mask=prepared.mask if source is not None else tensor(mask),
                strength=captured.strength,
                start_percent=captured.start,
                end_percent=captured.stop,
            )[0]
        check_cancelled()
        if output is model and captured.strength != 0:
            raise ValueError("The native model patch did not create a request-local model clone.")
        # Publish only validated weights after the native application and final
        # cancellation check; failed/cancelled loads cannot poison another job.
        owner.native_model_patch_cache = identity, (patch, vision)
        data["native_model_patch"] = dict(
            recipe=captured.recipe,
            native=RECIPES[captured.recipe][1],
            model=captured.artifacts[0].name,
            vision=captured.artifacts[1].name if len(captured.artifacts) > 1 else None,
            strength=captured.strength,
            start=captured.start,
            stop=captured.stop,
            source_sha256=hashlib.sha256((source or guide).tobytes()).hexdigest(),
            guide_sha256=hashlib.sha256(guide.tobytes()).hexdigest() if guide is not None else None,
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest() if mask is not None else None,
        )
        try:
            yield output
        finally:
            # Native cloned patch objects and their image caches belong to this call.
            del output
