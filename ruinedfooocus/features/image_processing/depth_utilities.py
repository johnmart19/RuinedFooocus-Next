"""Frozen still-image depth, landmark, pose and intrinsic utility contracts."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature

KINDS = ("lotus-depth", "moge-depth", "face-landmarks", "pose-map", "marigold-albedo")
LABELS = {
    "lotus-depth": "Lotus depth",
    "moge-depth": "MoGe depth",
    "face-landmarks": "Face landmarks",
    "pose-map": "Whole-body pose",
    "marigold-albedo": "Marigold albedo",
}
KEY = "_depth_utility_request"
MOGE_OUTPUT_DEFAULTS = dict(
    moge_output="depth",
    moge_projection="perspective",
    moge_normal_convention="opengl",
    moge_split_resolution=512,
    moge_merge_resolution=1920,
    moge_panorama_batch_size=1,
)
FACE_OUTPUT_DEFAULTS = dict(
    face_output="overlay",
    face_mask_regions="all",
    face_mask_face_oval=True,
    face_mask_lips=False,
    face_mask_left_eye=False,
    face_mask_right_eye=False,
    face_mask_irises=False,
)
EXTRA_UTILITY_DEFAULTS = {**MOGE_OUTPUT_DEFAULTS, **FACE_OUTPUT_DEFAULTS}
DEFAULTS = {
    **EXTRA_UTILITY_DEFAULTS,
    "image_utility_model": "",
    "image_utility_vae": "",
    "image_utility_lora": "",
    "image_utility_conditioning": "",
    "moge_resolution_level": 2,
    "moge_refine_steps": 3,
    "face_detector_variant": "short",
    "face_max_count": 1,
    "face_min_confidence": 0.5,
    "pose_score_threshold": 0.5,
}
FIELDS = tuple(DEFAULTS)
REPLAY_RESETS = {key: None for key in FIELDS if key.startswith("image_utility_")}


@dataclass(frozen=True)
class UtilityFile:
    field: str
    role: str
    category: str
    name: str
    path: Path
    signature: tuple | None
    provenance: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class UtilityRequest:
    kind: str
    size: tuple[int, int]
    rgba: bytes
    files: tuple[UtilityFile, ...]
    parameters: tuple[tuple[str, object], ...]


def is_utility(kind):
    return str(kind).lower() in KINDS


def kind_from(data):
    controls = data.get("controlnet")
    return str(controls.get("type", "")).lower() if isinstance(controls, dict) else ""


def replay_values(metadata):
    """Replay scalar intent; a saved result is never a replacement source image."""
    kind = kind_from(metadata)
    if not is_utility(kind):
        return dict(REPLAY_RESETS)
    parameters = _module(kind).parameters(metadata)
    return {
        **REPLAY_RESETS,
        **parameters,
        "input_image": None,
        "image_source": None,
        "inpaint_toggle": False,
        "cn_selection": LABELS[kind],
        "prompt": "",
        "negative": "",
        "automatic_upscale": "Off",
        "automatic_upscale_target": "Generated image",
        "model_recipe_path": "",
        "reference_adapter": "none",
    }


def output_metadata(data):
    """Utility provenance contains no local paths or fictitious sampling recipe."""
    from copy import deepcopy

    source = deepcopy(data["image_utility_source"])
    result = {
        "software": "RuinedFooocus",
        "prompt": "",
        "negative": "",
        "controlnet": {"type": source["operation"]},
        "width": source["width"],
        "height": source["height"],
        "image_utility_source": source,
    }
    result.update(dict(data[KEY].parameters))
    return result


def _module(kind):
    from importlib import import_module

    suffix = {
        "lotus-depth": "lotus",
        "moge-depth": "moge",
        "face-landmarks": "face",
        "pose-map": "pose",
        "marigold-albedo": "marigold",
    }[kind]
    return import_module(f"ruinedfooocus.features.image_processing.utility_{suffix}")


def roots(manager, categories):
    return tuple(
        dict.fromkeys(
            Path(root).resolve()
            for category in categories
            for root in manager.get_model_folders(category)
        )
    )


def selected_file(manager, value, categories):
    """Select an existing safe tensor below an explicit configured model root."""
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError("Choose each required installed component in Image tools.")
    supplied = Path(value)
    if ".." in supplied.parts:
        raise ValueError("Choose a component inside its configured model folders.")
    for root in roots(manager, categories):
        path = (supplied if supplied.is_absolute() else root / supplied).resolve()
        if path.is_file() and path.is_relative_to(root):
            if path.suffix.lower() not in (".safetensors", ".sft"):
                raise ValueError("This utility requires safe tensor components.")
            return path, path.relative_to(root).as_posix()
    raise ValueError("The selected utility component is not installed in its configured folders.")


def component_choices(manager, kind, field, *, locale=None):
    """Passive installed headers and registered first-use choices; no network or model load."""
    if kind not in KINDS:
        return []
    module = _module(kind)
    spec = next((item for item in module.COMPONENTS if item[0] == field), None)
    if spec is None:
        return []
    _, _, categories, validate = spec
    choices, seen = [], set()
    for root in roots(manager, categories):
        pending, visited = [root], set()
        while pending:
            folder = pending.pop()
            try:
                canonical = folder.resolve()
                if not canonical.is_relative_to(root) or canonical in visited:
                    continue
                visited.add(canonical)
                for path in sorted(folder.iterdir()):
                    if path.is_dir():
                        pending.append(path)
                    elif path.suffix.lower() in (".safetensors", ".sft"):
                        resolved, name = selected_file(manager, str(path.resolve()), categories)
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                        try:
                            validate(resolved)
                        except (OSError, ValueError, TypeError, KeyError):
                            continue
                        label = getattr(module, "component_label", None)
                        display = (
                            label(field, resolved, name, locale=locale)
                            if label is not None
                            else name
                        )
                        choices.append((display, str(resolved)))
            except (OSError, ValueError):
                continue
    if kind == "face-landmarks" and field == "image_utility_model":
        key = "detection/mediapipe_face_fp32.safetensors"
        entry = getattr(manager, "DOWNLOADABLE_FILES", {}).get(key)
        if entry is not None:
            try:
                request = manager.preparation_request(key)
                destination = Path(request.destination).resolve()
                if not destination.exists() and destination not in seen:
                    from ruinedfooocus.core.translation_manager import localize

                    choices.append(
                        (
                            localize(
                                "Download on first use: {name}", locale, name=destination.name
                            ),
                            str(destination),
                        )
                    )
            except (OSError, ValueError):
                pass
    return sorted(choices, key=lambda row: row[0].casefold())


def capture_assets(data, manager, models=None):
    from PIL import Image, ImageOps
    from ruinedfooocus.features.generation.image_preparation import selected_image_artifact

    kind = kind_from(data)
    if kind not in KINDS:
        raise ValueError("Choose a supported still-image utility.")
    source = data.get("input_image")
    if (
        not isinstance(source, Image.Image)
        or min(source.size) < 1
        or source.width * source.height > 16777216
    ):
        raise ValueError("Add one source image of at most 16 megapixels in Image tools.")
    if (
        data.get("inpaint_toggle")
        or data.get("model_recipe_path")
        or data.get("reference_adapter") not in (None, "", "none")
    ):
        raise ValueError(
            "Turn off inpainting, generation recipes and reference adapters for this utility."
        )
    image = ImageOps.exif_transpose(source).convert("RGBA").copy()
    result = dict(data, input_image=image, controlnet={"type": kind})
    module = _module(kind)
    parameters = module.parameters(result)
    files, artifacts = [], []
    for field, role, categories, validate in module.COMPONENTS:
        identities = {
            "lotus-depth": ("LotusD",),
            "sdpose-checkpoint": ("SDPose",),
            "moge-model": ("MoGe", "MoGe2"),
            "face-landmarker": ("FaceLandmarker",),
        }.get(role, ())
        item = selected_image_artifact(
            manager,
            result.get(field),
            role,
            categories,
            identities=identities,
            source_roles=("depth_estimation", "image_utility_model")
            if role == "moge-model"
            else ("depth_estimation",)
            if role == "lotus-depth"
            else ("image_utility_model",),
        )
        path, signature = item.request.destination, item.signature
        name = next(
            path.relative_to(root).as_posix()
            for root in roots(manager, categories)
            if path.is_relative_to(root)
        )
        provenance = None
        if signature is not None:
            validate(path)
            verify = getattr(module, "verify_component", None)
            provenance = verify(field, path) if verify is not None else None
        if signature is not None and signature != tuple(file_signature(path)):
            raise ValueError("A utility component changed during capture. Select it again.")
        result[field] = str(path)
        # The native Lotus ControlNet helper retains its established role/name contract.
        artifact_name = name if kind == "lotus-depth" else str(path)
        files.append(
            UtilityFile(
                field, role, categories[0], name, path, signature, tuple((provenance or {}).items())
            )
        )
        artifacts.append(replace(item, name=artifact_name))
    result.update(parameters)
    result[KEY] = UtilityRequest(
        kind, image.size, image.tobytes(), tuple(files), tuple(parameters.items())
    )
    result.pop("image_utility_source", None)
    return result, tuple(artifacts)


def validate_prepared(data):
    """Complete deferred role/source checks and freeze acquired file identities."""
    request = data.get(KEY)
    if not isinstance(request, UtilityRequest):
        raise ValueError("The utility request was not captured before preparation.")
    module, files = _module(request.kind), []
    for item in request.files:
        before = tuple(file_signature(item.path))
        if item.signature is not None and before != item.signature:
            raise ValueError("A utility component changed after submission. Select it again.")
        validate = next(spec[3] for spec in module.COMPONENTS if spec[0] == item.field)
        validate(item.path)
        verify = getattr(module, "verify_component", None)
        proof = verify(item.field, item.path) if verify is not None else None
        if before != tuple(file_signature(item.path)):
            raise ValueError("A utility component changed during preparation. Select it again.")
        files.append(replace(item, signature=before, provenance=tuple((proof or {}).items())))
    return dict(data, **{KEY: replace(request, files=tuple(files))})


def worker_data(data):
    from ruinedfooocus.features.image_processing.native_vision import (
        worker_data as shared_worker_data,
    )

    result = shared_worker_data(dict(data, native_vision_model=data["image_utility_model"]))
    result.pop("native_vision_model", None)
    return result


def native_call(check_cancelled, function, *args, **kwargs):
    check_cancelled()
    value = function(*args, **kwargs)
    check_cancelled()
    return value


def cached_model(owner, request, loader, check_cancelled, *, source_identity=()):
    """Keep native patchers on the shared image lane, keyed by all file identities."""
    identity = (request.kind, tuple(item.signature for item in request.files))
    if source_identity:
        identity += (tuple(source_identity),)
    cached = getattr(owner, "_depth_utility_cache", None)
    if cached is None or cached[0] != identity:
        check_cancelled()
        clear_model(owner)
        loaded, patchers = loader()
        try:
            check_cancelled()
        except BaseException:
            from comfy.model_management import unload_model_and_clones

            for patcher in patchers:
                unload_model_and_clones(patcher)
            raise
        cached = owner._depth_utility_cache = (identity, loaded, tuple(patchers))
    return cached[1]


def clear_model(owner):
    from comfy.model_management import unload_model_and_clones

    lotus = getattr(owner, "_lotus_depth", None)
    if lotus is not None:
        owner._lotus_depth = None
        unload_model_and_clones(lotus[1])
        unload_model_and_clones(lotus[2].patcher)
    cached = getattr(owner, "_depth_utility_cache", None)
    if cached is not None:
        owner._depth_utility_cache = None
        for patcher in cached[2]:
            unload_model_and_clones(patcher)


def process(owner, data, manager, check_cancelled):
    import numpy as np
    import torch
    from PIL import Image
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output

    request = data.get(KEY)
    data.pop("image_utility_source", None)
    kind = kind_from(data)
    if not isinstance(request, UtilityRequest) or request.kind != kind:
        raise ValueError("The utility request was not captured before compute. Submit it again.")
    image = Image.frombytes("RGBA", request.size, request.rgba)
    pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)[None]
    check_cancelled()
    if kind == "lotus-depth" and getattr(owner, "_depth_utility_cache", None) is not None:
        clear_model(owner)
    with ExitStack() as leases:
        for item in request.files:
            if kind == "lotus-depth":
                # The reused Lotus helper owns both read leases for its entire
                # native run. A second read behind a waiting writer can deadlock.
                continue
            name = str(item.path)
            path = leases.enter_context(
                prepared_asset(data, item.role, name, manager, item.category)
            )
            if path.resolve() != item.path or tuple(file_signature(path)) != item.signature:
                raise ValueError("A utility component changed. Submit the image again.")
        verify_request = getattr(_module(kind), "verify_captured_request", None)
        if verify_request is not None:
            verify_request(request)
        with torch.inference_mode():
            output, details = _module(kind).render(
                owner, request, pixels, data, manager, check_cancelled
            )
        check_cancelled()
        if tuple(output.shape) != tuple(pixels.shape) or not torch.isfinite(output).all():
            raise ValueError("The native utility returned invalid image dimensions or values.")
        for item in request.files:
            if tuple(file_signature(item.path)) != item.signature:
                raise ValueError(
                    "A utility component changed during processing. Submit the image again."
                )
        precision = output[0].detach().float().cpu().numpy().copy()
        # Maps and overlays retain source coverage. Native estimators consume RGB only.
        precision = np.concatenate(
            (precision, np.asarray(image.getchannel("A"), dtype=np.float32)[..., None] / 255),
            axis=-1,
        )
        data["image_utility_source"] = {
            "operation": kind,
            "components": [
                {"role": item.role, "name": item.path.name, "size": item.signature[1]}
                for item in request.files
            ],
            "width": image.width,
            "height": image.height,
            "alpha": "source",
            **dict(request.parameters),
            **details,
        }
        if advanced_output(data):
            return [precision]
        result = Image.fromarray((np.clip(precision, 0, 1) * 255).round().astype(np.uint8), "RGBA")
        return [result]
