"""Captured native image assets prepared before the existing compute queue."""

from concurrent.futures import TimeoutError
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, replace
from copy import deepcopy
from functools import wraps
import inspect
import math
import os
from pathlib import Path
from urllib.parse import urlparse

from ruinedfooocus.core.preparation import PreparationRequest, preparation_service, verify_file
from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_native_image_assets"
PENDING_KEY = "_pending_image_selection"
_component_leases = ContextVar("native_image_component_leases", default=None)
CONTROL_MODELS = {
    "canny": "control-lora-canny-rank128.safetensors",
    "depth": "control-lora-depth-rank128.safetensors",
    "recolour": "control-lora-recolor-rank128.safetensors",
    "sketch": "control-lora-sketch-rank128-metadata.safetensors",
}


@dataclass(frozen=True)
class ImageArtifact:
    role: str
    name: str
    request: PreparationRequest
    signature: tuple | None
    receipt_required: bool = False


@dataclass(frozen=True)
class ImageAssets:
    artifacts: tuple[ImageArtifact, ...]
    offline: bool
    automatic_upscale_name: str | None = None
    automatic_upscale_scale: int = 1
    depth_name: str | None = None
    reserve_gib: float = 0
    allow_unknown_size: bool = True
    cache_root: Path | None = None
    source_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageRequirement:
    role: str
    name: str
    destination: Path
    status: str
    source: str
    integrity: str
    expected_bytes: int | None
    reason: str = ""


@dataclass(frozen=True)
class ImageReadiness:
    state: str
    requirements: tuple[ImageRequirement, ...]
    missing_bytes: int | None
    reasons: tuple[str, ...] = ()
    source_warnings: tuple[str, ...] = ()
    fit_status: str = "Not measured; download bytes are not runtime memory requirements."


@dataclass(frozen=True)
class PendingImageSelection:
    catalogue_key: str
    profile_id: str
    artifact_id: str
    native_family: str
    native_class: str
    behavioral_variant: str | None
    catalogue_entry: tuple[tuple[str, object], ...]
    source_identity: tuple
    primary: ImageArtifact


def pending_image_selection(data):
    pending = data.get(PENDING_KEY)
    return pending if isinstance(pending, PendingImageSelection) else None


def _local_prompt_command(data):
    prompt = data.get("prompt", "")
    return (
        isinstance(prompt, str)
        and data.get("task_type") != "tool_call"
        and "_api_job" not in data
        and not data.get("_metadata_prompt")
        and (
            prompt.lower() == "ruinedfooocuslogo"
            or prompt.startswith("#!")
            or prompt.lower().startswith("search:")
        )
    )


def _pending_source_identity(row):
    artifact = row["artifact"]
    return (
        row["id"],
        row["profile_id"],
        row.get("native_family"),
        row.get("native_class", row.get("architecture")),
        row.get("behavioral_variant"),
        row["download_identity"],
        artifact.get("bytes"),
        artifact.get("sha256"),
        artifact.get("pinned_url"),
        row.get("evidence", {}).get("default_recipe_id"),
    )


def capture_pending_image_selection(
    data,
    manager,
    *,
    offline=False,
    reserve_gib=0,
    allow_unknown_size=True,
):
    """Freeze a missing, explicitly selected trusted primary; never acquire it here."""
    if pending_image_selection(data) is not None:
        return data
    from ruinedfooocus.features.image_processing import (
        still_image_utilities,
        image_composition,
        latent_composition,
    )

    kind = str((data.get("controlnet") or {}).get("type", "")).lower()
    if (
        data.get("task_type") in ("llama", "chat")
        or _local_prompt_command(data)
        or kind
        in (
            "detect",
            "segment",
            "depth-map",
            "restore-seedvr2",
            "upscale",
            "rembg",
            *still_image_utilities.KINDS,
            *image_composition.KINDS,
        )
        or latent_composition.handles(kind)
    ):
        return None
    _validate_policy(reserve_gib, allow_unknown_size)
    name = data.get("base_model_name")
    if not isinstance(name, str) or not name or len(name) > 1024:
        return None
    supplied = Path(name)
    if ".." in supplied.parts:
        return None
    categories = ("checkpoints", "diffusion_models")
    # Existing custom/uncurated files keep their normal header-based admission.
    if any(
        (supplied if supplied.is_absolute() else Path(root) / supplied).is_file()
        for category in categories
        for root in manager.get_model_folders(category)
    ):
        return None
    from ruinedfooocus.features.models.model_registry import load_registry
    from ruinedfooocus.features.models.model_identity import is_native_checkpoint_source
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS
    from ruinedfooocus.features.models.image_scope import require_image_family

    registry, matches = load_registry(), []
    for key, info in getattr(manager, "DOWNLOADABLE_FILES", {}).items():
        category = info.get("path", "").removeprefix("path_")
        if category not in categories:
            continue
        filename = info.get("filename", "")
        roots = tuple(Path(root).resolve() for root in manager.get_model_folders(category))
        candidates = tuple((root / filename).resolve() for root in roots)
        if name not in (key, filename) and not (
            supplied.is_absolute() and supplied.resolve() in candidates
        ):
            continue
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("The registered image filename leaves its configured model folder.")
        row = registry.annotations_for(key, info)
        if row is None or row.get("state") != "existing" or not is_native_checkpoint_source(row):
            raise ValueError(
                "This missing primary has no trusted image source contract. Install and inspect it locally."
            )
        family = row["native_family"]
        require_image_family(family)
        native_class = row.get("native_class", row.get("architecture"))
        if (
            ALIASES.get(family, family) not in IMAGE_COMPONENTS
            or not isinstance(native_class, str)
            or not native_class
        ):
            raise ValueError(
                "The selected source has no connected image loader contract. Choose a supported image source."
            )
        if row.get("gated") or row.get("access") in ("gated", "denied", "restricted"):
            raise ValueError(
                "This image source requires access approval. Install the authorized file locally."
            )
        request = manager.preparation_request(key)
        if not any(
            request.destination.is_relative_to(root) for root in roots
        ) or request.destination.suffix.lower() not in (".safetensors", ".sft", ".gguf"):
            raise ValueError(
                "The selected primary needs a supported file in its configured image model folders."
            )
        matches.append((key, dict(info), row, request))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Choose one unambiguous registered primary image source.")
    key, info, row, request = matches[0]
    if request.destination.is_file():
        return None
    # Public intent and source images are request-owned; runtime job objects stay references.
    result = {key: value if key.startswith("_") else deepcopy(value) for key, value in data.items()}
    for stale in (KEY, "_effective_image_plan", "_sdcpp_snapshot", "_sdcpp_prepared"):
        result.pop(stale, None)
    primary = ImageArtifact(
        "primary", str(request.destination), request, None, receipt_required=True
    )
    result[PENDING_KEY] = PendingImageSelection(
        key,
        row["profile_id"],
        row["id"],
        row["native_family"],
        row.get("native_class", row.get("architecture")),
        row.get("behavioral_variant"),
        tuple(info.items()),
        _pending_source_identity(row),
        primary,
    )
    result[KEY] = ImageAssets(
        (primary,),
        bool(offline)
        or any(
            os.environ.get(key, "").upper() in ("1", "TRUE", "YES", "ON")
            for key in ("RF_OFFLINE", "HF_HUB_OFFLINE")
        ),
        reserve_gib=reserve_gib,
        allow_unknown_size=allow_unknown_size,
        cache_root=Path(manager.model_paths["cache_path"]).resolve()
        if getattr(manager, "model_paths", {}).get("cache_path")
        else None,
    )
    return result


def _validate_primary_source(pending, path, cache_root):
    from ruinedfooocus.features.models.image_components import ALIASES
    from ruinedfooocus.features.models.image_scope import require_image_checkpoint
    from ruinedfooocus.features.models.model_identity import (
        cached_safetensors_header,
        read_prepared_native_source,
        verified_native_source,
    )
    from ruinedfooocus.features.models.native_workflow_admission import (
        require_available_image_workflow,
    )

    from ruinedfooocus.features.models.model_architecture import native_detector_ready

    details = {}
    if native_detector_ready():
        from ruinedfooocus.features.models.model_architecture import inspect_model_file

        details = inspect_model_file(path)
        native_class = details.get("native_class")
    else:
        from ruinedfooocus.features.models.utility_models import passive_native_class

        native_class = passive_native_class(path, {pending.native_class}, include_gguf=True)
    if native_class != pending.native_class:
        raise ValueError(
            "The prepared primary's native header differs from its selected source. Inspect it before use."
        )
    actual_family = details.get("architecture")
    if pending.native_family == "BerniniImage":
        from ruinedfooocus.features.generation.bernini_image import require_header

        require_header(path, "high")
        actual_family = "BerniniImage"
    if actual_family and ALIASES.get(actual_family, actual_family) != ALIASES.get(
        pending.native_family, pending.native_family
    ):
        raise ValueError(
            "The prepared primary's family differs from its selected source. Inspect it before use."
        )
    require_image_checkpoint(path, pending.native_family)
    metadata = details.get("metadata", {})
    if not metadata and is_safetensors(path):
        metadata = cached_safetensors_header(path).get("__metadata__", {})
    elif not metadata and path.suffix.lower() == ".gguf":
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        header, _ = read_gguf_header(path)
        metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        metadata = {}
    variants = {
        str(metadata[key]).strip().casefold().replace("_", "-")
        for key in ("modelspec.variant", "variant", "model_variant", "training_variant")
        if metadata.get(key)
    }
    expected = pending.behavioral_variant
    if expected and variants and variants != {expected.casefold().replace("_", "-")}:
        raise ValueError(
            "The prepared primary declares a different behavioral variant. Inspect it before use."
        )
    if expected and not variants:
        proof = read_prepared_native_source(path, cache_root) if cache_root is not None else {}
        source = verified_native_source(path, proof)
        if source is None or source.get("behavioral_variant") != expected:
            raise ValueError(
                "The primary variant is unverified. Refresh its immutable source metadata or install and Inspect the file locally."
            )
    # Final P1 capture owns operation-specific plans such as MiniMax's image recipe.
    if pending.native_family != "MiniMaxH3Image":
        require_available_image_workflow(path, family=pending.native_family)


def prepare_pending_image_selection(
    data, *, check_cancelled, progress=None, service=preparation_service
):
    """Stage one only; the caller recaptures P1/companions from the installed bytes."""
    pending = pending_image_selection(data)
    if pending is None:
        return data
    from ruinedfooocus.features.models.model_registry import load_registry

    row = load_registry().annotations_for(pending.catalogue_key, dict(pending.catalogue_entry))
    if row is None or _pending_source_identity(row) != pending.source_identity:
        raise ValueError(
            "The selected primary source contract changed. Select it again before preparation."
        )
    primary_data = dict(data)
    primary_data.pop(PENDING_KEY, None)
    prepared = prepare_image_assets(
        primary_data, check_cancelled=check_cancelled, progress=progress, service=service
    )
    current = load_registry().annotations_for(pending.catalogue_key, dict(pending.catalogue_entry))
    if current is None or _pending_source_identity(current) != pending.source_identity:
        raise ValueError(
            "The selected primary source contract changed during preparation. Select it again."
        )
    captured = prepared[KEY]
    item = captured.artifacts[0]
    with service.artifact_lease(
        item.request.destination, cancelled=_Cancellation(check_cancelled)
    ) as path:
        if tuple(file_signature(path)) != item.signature:
            raise ValueError("The prepared primary changed. Select it again.")
        check_cancelled()
        _validate_primary_source(pending, path, captured.cache_root)
        check_cancelled()
    result = dict(
        prepared, base_model_name=str(item.request.destination), _prepared_primary_asset=item
    )
    result["_image_source_warnings"] = captured.source_warnings
    result["_image_preparation_policy"] = (captured.reserve_gib, captured.allow_unknown_size)
    result.pop(PENDING_KEY, None)
    result.pop(KEY, None)
    return result


def selected_image_artifact(
    manager,
    value,
    role,
    categories,
    *,
    identities=(),
    source_roles=(),
    allowed_roots=None,
):
    """Bind an explicit local file or a compatible registered acquisition tuple.

    Publisher workflow annotations are provenance, never acquisition authority.
    Missing custom paths and arbitrary URLs cannot become download selections.
    """
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError("Choose each required installed or registered image component.")
    supplied = Path(value)
    if ".." in supplied.parts:
        raise ValueError("Choose an image component inside its configured model folders.")
    roots = (
        tuple(Path(root).resolve() for root in allowed_roots)
        if allowed_roots is not None
        else tuple(
            dict.fromkeys(
                Path(root).resolve()
                for category in categories
                for root in manager.get_model_folders(category)
            )
        )
    )
    for root in roots:
        path = (supplied if supplied.is_absolute() else root / supplied).resolve()
        if path.is_file() and path.is_relative_to(root):
            if path.suffix.lower() not in (".safetensors", ".sft"):
                raise ValueError("This image utility requires safe tensor components.")
            return ImageArtifact(
                role, str(path), PreparationRequest("", path), tuple(file_signature(path))
            )
    from ruinedfooocus.features.models.model_registry import load_registry

    matches = []
    for key, info in getattr(manager, "DOWNLOADABLE_FILES", {}).items():
        category = info.get("path", "").removeprefix("path_")
        if category not in categories:
            continue
        filename = info.get("filename", "")
        candidates = tuple(
            (Path(root).resolve() / filename).resolve()
            for root in manager.get_model_folders(category)
        )
        if value not in (key, filename) and not (
            supplied.is_absolute() and supplied.resolve() in candidates
        ):
            continue
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("The registered image filename leaves its configured model folder.")
        annotation = load_registry().annotations_for(key, info)
        if (
            annotation is None
            or annotation.get("state") != "existing"
            or annotation.get("role") not in source_roles
        ):
            continue
        source_identities = {
            annotation.get(field)
            for field in ("native_family", "native_class", "architecture", "role")
        }
        if not identities or not source_identities.intersection(identities):
            continue
        if annotation.get("gated") or annotation.get("access") in ("gated", "denied", "restricted"):
            raise ValueError(
                "This image source requires access approval. Install the authorized file locally."
            )
        request = manager.preparation_request(key)
        if not any(request.destination.is_relative_to(root) for root in roots):
            raise ValueError("The image download destination leaves its configured model folders.")
        if request.destination.suffix.lower() not in (".safetensors", ".sft"):
            raise ValueError("This image utility requires safe tensor components.")
        matches.append(ImageArtifact(role, str(request.destination), request, None))
    if len(matches) != 1:
        raise ValueError(
            "Choose an installed file or one compatible registered source for this image tool."
        )
    return matches[0]


def image_readiness(data):
    """Passive captured-file status: no network, hashing, tensor imports or compute."""
    captured = data.get(KEY)
    if not isinstance(captured, ImageAssets):
        return ImageReadiness("ready", (), 0)
    rows, reasons, missing, requests = [], [], {}, {}
    for item in captured.artifacts:
        request, reason = item.request, ""
        path = request.destination
        if path in requests and requests[path] != request:
            reason = f"Image roles have conflicting acquisition metadata for {path.name}. Choose them again."
        requests[path] = request
        present = path.is_file()
        if item.signature is not None:
            try:
                current = tuple(file_signature(path)) if present else None
            except OSError:
                current, present = None, False
            if not present:
                reason = f"Captured image {item.role} was removed after submission: {item.name}. Choose it again."
            elif current != item.signature:
                reason = f"Image {item.role} changed after submission: {item.name}."
        if not present:
            missing[path] = request.expected_size
            if captured.offline:
                reason = f"Offline mode: image {item.role} is missing: {item.name}."
            elif not reason and not request.url:
                reason = f"The selected local image {item.role} is missing: {item.name}. Install it locally."
            elif not reason:
                url = urlparse(request.url)
                if url.scheme != "https" and not (
                    url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
                ):
                    reason = f"Image {item.role} has an invalid acquisition source. Refresh its source details."
                elif request.expected_size is None and not captured.allow_unknown_size:
                    reason = f"Image {item.role} has an unknown download size. Allow unknown sizes or install it locally."
        if reason:
            reasons.append(reason)
        rows.append(
            ImageRequirement(
                item.role,
                item.name,
                path,
                "blocked" if reason else "installed" if present else "missing",
                request.url or "Local file",
                "SHA256 supplied"
                if request.sha256
                else "Size only"
                if request.expected_size
                else "Unknown",
                request.expected_size,
                reason,
            )
        )
    if missing and not reasons:
        try:
            _check_disk_capacity(captured)
        except (OSError, ValueError) as error:
            reasons.append(str(error))
    total = None if any(size is None for size in missing.values()) else sum(missing.values())
    return ImageReadiness(
        "blocked"
        if reasons
        else "preparable"
        if missing or any(item.signature is None for item in captured.artifacts)
        else "ready",
        tuple(rows),
        total,
        tuple(dict.fromkeys(reasons)),
        captured.source_warnings,
    )


def _check_disk_capacity(captured):
    from ruinedfooocus.features.models.acquisition import _space

    requests = {
        item.request.destination: item.request
        for item in captured.artifacts
        if not item.request.destination.is_file()
    }
    for request in requests.values():
        needed = sum(
            other.expected_size or 0
            for other in requests.values()
            if other.destination.anchor == request.destination.anchor
        )
        if needed + captured.reserve_gib * 1024**3 > _space(request.destination.parent):
            raise ValueError(
                "The selected image files would exceed the free-space reserve. Free disk space or change the captured reserve."
            )
    if requests and captured.cache_root is not None:
        needed = sum(request.expected_size or 0 for request in requests.values())
        if needed + captured.reserve_gib * 1024**3 > _space(captured.cache_root):
            raise ValueError(
                "The selected image files would exceed the preparation cache free-space reserve."
            )


def _validate_prepared_utilities(data, artifacts, service, check_cancelled):
    controls = data.get("controlnet") or {}
    kind = str(controls.get("type", "")).lower()
    from ruinedfooocus.features.image_processing import depth_utilities

    if kind in ("detect", "segment", "depth-map"):
        from ruinedfooocus.features.image_processing import native_vision as utility
    elif kind == "restore-seedvr2":
        from ruinedfooocus.features.image_processing import native_seedvr2 as utility
    elif depth_utilities.is_utility(kind):
        from ruinedfooocus.features.image_processing import still_image_utilities

        utility = still_image_utilities._handler(kind)
    else:
        return data
    with ExitStack() as leases:
        unique = {item.request.destination: item for item in artifacts}
        for _, item in sorted(unique.items(), key=lambda pair: str(pair[0])):
            path = leases.enter_context(
                service.artifact_lease(
                    item.request.destination, cancelled=_Cancellation(check_cancelled)
                )
            )
            if tuple(file_signature(path)) != item.signature:
                raise ValueError(
                    "An image utility component changed during preparation. Submit it again."
                )
        check_cancelled()
        result = utility.validate_prepared(data)
        check_cancelled()
        return result


class _Cancellation:
    def __init__(self, check):
        self.check = check

    def is_set(self):
        self.check()
        return False


def _artifact(manager, role, category, name, key=None, *, local_only=False):
    """Resolve local roots/catalogue once; this never calls a download resolver."""
    name = str(name)
    catalog = manager.DOWNLOADABLE_FILES
    key = (
        key
        if key in catalog
        else next(
            (
                key
                for key, entry in catalog.items()
                if entry.get("path") == "path_" + category and entry.get("filename") == name
            ),
            None,
        )
    )
    filename = (
        catalog[key]["filename"]
        if key
        else CONTROL_MODELS.get(name, name)
        if role == "controlnet"
        else name
    )
    supplied = Path(name).expanduser()
    local = (
        supplied
        if supplied.is_absolute() and supplied.is_file()
        else next(
            (
                root / filename
                for root in manager.get_model_folders(category)
                if (root / filename).is_file()
            ),
            None,
        )
    )
    if local is not None:
        local = local.resolve()
        request = (
            manager.preparation_request(key, destination=local)
            if key and not local_only
            else PreparationRequest("", local)
        )
        return ImageArtifact(role, name, request, tuple(file_signature(local)))
    if key is None or local_only:
        raise FileNotFoundError(f"Image {role} is not installed: {name}. Choose an installed file.")
    request = manager.preparation_request(key)
    return ImageArtifact(role, name, request, None)


def image_control_data(data):
    """Resolve metadata-owned control fields without applying sampling settings."""
    from ruinedfooocus.features.gallery.image_metadata import read_metadata

    imported = (
        read_metadata(data.get("prompt"))
        if "_api_job" not in data and data.get("task_type") != "tool_call"
        else None
    )
    result = dict(data)
    if imported:
        from ruinedfooocus.features.generation.native_tensor_io import (
            replay_values as replay_tensors,
        )

        result.update(replay_tensors(imported, result))
        result.update(
            (key, value)
            for key, value in imported.items()
            if key == "controlnet" or key.startswith("cn_")
        )
        result["automatic_upscale"] = imported.get("automatic_upscale", "Off")
        # process_metadata replays upscaling after generation, never input-only.
        result["automatic_upscale_target"] = "Generated image"
        from ruinedfooocus.features.models.yaml_recipes import replay_values

        if result.get("_sampling_captured") is not False:
            result.update(replay_values(imported))
        from ruinedfooocus.features.image_editing.reference_adapters import (
            replay_values as replay_reference,
        )

        if result.get("_sampling_captured") is not False:
            result.update(replay_reference(imported))
            if imported.get("native_edit_recipe") == "krea2-style":
                from ruinedfooocus.features.image_editing.krea_style import values as krea_values

                style = krea_values(imported)
                result.update(krea_style_lora=style.name, krea_style_strength=style.strength)
        from ruinedfooocus.features.image_editing.qwen_image21_union import (
            replay_values as replay_union,
        )

        result.update(replay_union(imported))
        from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

        for feature in (model_patches, native_control_recipes):
            result.update(feature.replay_values(imported, clear_images=False))
        from ruinedfooocus.features.image_editing import lora_hooks

        result.update(lora_hooks.replay_values(imported))
        from ruinedfooocus.features.image_editing.qwen_image21_local_edit import (
            replay_values as replay_local_edit,
        )

        result.update(replay_local_edit(imported))
        from ruinedfooocus.features.generation.layered import replay_values as replay_layered

        result.update(replay_layered(imported))
        from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
            replay_values as replay_auxiliary,
        )

        result.update(replay_auxiliary(imported))
        from ruinedfooocus.features.generation.minimax_image import replay_values as replay_minimax

        result.update(replay_minimax(imported))
        from ruinedfooocus.features.generation.bernini_image import replay_values as replay_bernini

        result.update(replay_bernini(imported))
        for key in (
            "cascade_stage_b",
            "cascade_b_steps",
            "cascade_b_cfg",
            "cascade_b_sampler",
            "cascade_b_scheduler",
            "cascade_stage_b_source",
            "cascade_compression",
            "cascade_source_denoise",
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
            "sdxl_refiner_model",
            "sdxl_refiner_switch_step",
            "sdxl_refiner_cfg",
            "sdxl_refiner_clip",
            "refiner_noise_augmentation",
            "pid_input_vae",
            "pid_latent_format",
            "pid_degrade_sigma",
            "pid_conditioning",
        ):
            if key in imported:
                result[key] = deepcopy(imported[key])
    return result


def omit_generation_conditioning(data):
    """Utilities ignore hidden generation choices without changing UI/profile state."""
    result = dict(data)
    from ruinedfooocus.features.generation.cascade_guidance import (
        FIELDS,
        IMAGE_FIELD,
        KEY as CASCADE_KEY,
    )

    for key in (*FIELDS, IMAGE_FIELD, CASCADE_KEY, "native_cascade_guidance"):
        result.pop(key, None)
    for key in (
        "model_recipe_path",
        "model_recipe_sha256",
        "model_recipe_overrides",
        "clip_layer_source",
        "_native_yaml_recipe",
        "_native_approximate_codec",
        "reference_adapter",
        "reference_adapter_model",
        "reference_adapter_vision",
        "reference_adapter_strength",
        "reference_adapter_image",
        "reference_adapter_info",
        "krea_style_lora",
        "krea_style_strength",
        "native_krea_style",
        "_native_krea_style",
        "_reference_adapter_capture",
        "qwen_layer_count",
        "qwen_layer_source",
        "native_layered_conditioning",
        "native_component_layout",
        "_native_layered_plan",
        "_native_component_layout",
        "gligen_model",
        "gligen_model_signature",
        "gligen_layout",
        "gligen_guidance",
        "_native_gligen_layout",
        "hypernetwork_model",
        "hypernetwork_model_signature",
        "hypernetwork_strength",
        "hypernetwork_guidance",
        "_native_hypernetwork",
        "native_minimax_image",
        "_native_minimax_image",
        "native_bernini_image",
        "_native_bernini_image",
    ):
        result.pop(key, None)
    from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

    for feature in (model_patches, native_control_recipes):
        for key in (*feature.FIELDS, *feature.IMAGE_FIELDS, feature.KEY, feature.SOURCE_KEY):
            result.pop(key, None)
    result.pop("native_model_patch", None)
    result.pop("native_control_recipe", None)
    from ruinedfooocus.features.image_editing import lora_hooks

    for key in (*lora_hooks.FIELDS, lora_hooks.KEY, "native_lora_hooks"):
        result.pop(key, None)
    return result


def _control_checkpoint_family(models, name):
    if models is None or not name:
        return None
    primary = models.get_file("checkpoints", name)
    detect = getattr(models, "detect_checkpoint_base", None)
    family = detect(primary) if primary is not None and callable(detect) else None
    return family or models.get_model_base(
        models.get_models_by_path("checkpoints", name, fetch=False)
    )


def _validate_policy(reserve_gib, allow_unknown_size):
    if (
        type(reserve_gib) not in (int, float)
        or not math.isfinite(reserve_gib)
        or not 0 <= reserve_gib <= 10000
        or type(allow_unknown_size) is not bool
    ):
        raise ValueError(
            "Choose a finite disk reserve from 0 through 10000 GiB and an explicit unknown-size policy."
        )


def _effective_plan(data):
    if "_effective_image_plan" not in data:
        return None
    from ruinedfooocus.features.models.effective_image_plan import get_effective_image_plan

    return get_effective_image_plan(data)


def _plan_component_assets(data, manager, existing):
    plan = _effective_plan(data)
    if plan is None:
        return None
    plan.verify_files()
    approximate = {
        (item.name, item.request.destination) for item in existing if item.role == "approximate_vae"
    }
    artifacts = []
    for row in plan.required_components:
        if row.embedded or row.category == "operation":
            continue
        if row.path is None:
            raise FileNotFoundError(
                f"Image component {row.key} has no selected file. Choose the required component."
            )
        path = Path(row.path).resolve()
        if (row.key, path) in approximate:
            continue
        if (
            row.signature is None
            and not row.download_key
            and (row.acquisition is not None or not path.is_file())
        ):
            raise FileNotFoundError(
                f"Image component {row.key} is missing: {row.filename}. Install it locally or reset this role to Automatic."
            )
        if row.catalogue_identity is not None:
            from ruinedfooocus.features.models.model_registry import download_identity

            entry = manager.DOWNLOADABLE_FILES.get(row.download_key)
            if entry is None or download_identity(entry) != row.catalogue_identity:
                raise ValueError(
                    f"Image component {row.key} acquisition metadata changed after capture. Submit it again."
                )
        request = (
            manager.preparation_request(row.download_key, destination=path)
            if row.download_key
            else PreparationRequest("", path)
        )
        if row.category == "loras":
            from ruinedfooocus.features.models import lora_sources

            selected = tuple(
                item for item in lora_sources.captured(data) or () if row.key == "lora:" + item.name
            )
            if not selected or any(item.request != request for item in selected):
                raise ValueError(
                    "The selected LoRA acquisition metadata changed after capture. Submit it again."
                )
        if row.acquisition is not None:
            if request != row.acquisition:
                raise ValueError(
                    f"Image component {row.key} acquisition metadata changed after capture. Submit it again."
                )
            request = row.acquisition
        if row.acquisition is None and (
            row.source
            and request.url != row.source
            or row.expected_bytes is not None
            and request.expected_size != row.expected_bytes
        ):
            raise ValueError(
                f"Image component {row.key} acquisition metadata changed after capture. Submit it again."
            )
        signature = tuple(row.signature) if row.signature is not None else None
        artifacts.append(ImageArtifact(row.category, row.key, request, signature))
    return tuple(artifacts)


def capture_image_assets(
    data,
    manager,
    controls,
    *,
    offline=False,
    models=None,
    values=None,
    reserve_gib=0,
    allow_unknown_size=True,
):
    if "_image_preparation_policy" in data:
        reserve_gib, allow_unknown_size = data["_image_preparation_policy"]
    _validate_policy(reserve_gib, allow_unknown_size)
    offline = bool(offline) or any(
        os.environ.get(key, "").upper() in ("1", "TRUE", "YES", "ON")
        for key in ("RF_OFFLINE", "HF_HUB_OFFLINE")
    )
    cache_root = getattr(manager, "model_paths", {}).get("cache_path")

    def captured(artifacts, *options):
        return ImageAssets(
            tuple(artifacts),
            offline,
            *options,
            reserve_gib=reserve_gib,
            allow_unknown_size=allow_unknown_size,
            cache_root=Path(cache_root).resolve() if cache_root else None,
            source_warnings=tuple(data.get("_image_source_warnings", ())),
        )

    from ruinedfooocus.features.models.image_scope import require_image_operation

    require_image_operation(data)
    from ruinedfooocus.features.image_processing import image_composition

    if image_composition.handles((controls or {}).get("type")):
        result = image_composition.capture(data, str(controls["type"]).lower())
        result[KEY] = captured(())
        return result
    from ruinedfooocus.features.image_processing import latent_composition

    if latent_composition.handles((controls or {}).get("type")):
        result = latent_composition.capture(data, manager)
        result[KEY] = captured(())
        return result
    data = latent_composition.omit_inactive(data)
    from ruinedfooocus.features.image_editing import qwen_image21_union
    from ruinedfooocus.features.image_processing import still_image_utilities as depth_utilities

    native_utility = (
        isinstance(controls, dict)
        and str(controls.get("type", "")).lower()
        in (
            "detect",
            "segment",
            "depth-map",
            "restore-seedvr2",
        )
        or isinstance(controls, dict)
        and depth_utilities.is_utility(controls.get("type"))
    )
    result = (
        dict(data, controlnet=controls, prompt="")
        if native_utility
        else image_control_data(dict(data, controlnet=controls))
    )
    from ruinedfooocus.features.generation.image_output_batches import PLAN_KEY

    # A fresh source operation owns its output contract; saved display metadata
    # and a previous request's private plan are never completion evidence.
    result.pop(PLAN_KEY, None)
    result.pop("native_output_group", None)
    result.pop("native_image_output", None)
    result.pop("native_cascade", None)
    result.pop(image_composition.KEY, None)
    result.pop("native_composition_source", None)
    for key in image_composition.IMAGE_FIELDS:
        result.pop(key, None)
    controls = result["controlnet"]
    if result.get("task_type") == "llama" or _local_prompt_command(result):
        return result
    if not isinstance(controls, dict):
        raise ValueError("Image control settings must be a mapping.")
    controls = deepcopy(controls)
    if controls:
        result["controlnet"] = controls
    kind = str(controls.get("type", "")).lower()
    native_utility = kind in (
        "detect",
        "segment",
        "depth-map",
        "restore-seedvr2",
    ) or depth_utilities.is_utility(kind)
    if kind == "restore-seedvr2":
        from ruinedfooocus.features.image_processing.native_seedvr2 import (
            capture_assets,
            worker_data,
        )

        result, items = capture_assets(omit_generation_conditioning(result), manager)
        result[KEY] = captured(items)
        return worker_data(result)
    if depth_utilities.is_utility(kind):
        # Standalone image operations own their assets and bypass generation
        # checkpoints, Layered layouts, and hidden patch/control recipes.
        result, items = depth_utilities.capture_assets(
            omit_generation_conditioning(result), manager, models
        )
        result[KEY] = captured(items)
        return depth_utilities.worker_data(result)
    if kind in ("detect", "segment", "depth-map", "restore-seedvr2", "rembg", "upscale"):
        result = omit_generation_conditioning(result)
    if (
        models is not None
        and result.get("base_model_name")
        and kind not in ("detect", "segment", "depth-map", "restore-seedvr2", "upscale", "rembg")
    ):
        from ruinedfooocus.features.generation.layered import capture as capture_layered
        from ruinedfooocus.features.models.utility_models import is_image_utility

        metadata = models.get_models_by_path("checkpoints", result["base_model_name"], fetch=False)
        if is_image_utility(metadata):
            raise ValueError(
                "This installed model is an image utility. Choose its matching Image tool."
            )
        family = models.get_model_base(metadata)
        from ruinedfooocus.features.models.image_scope import (
            require_image_family,
            require_image_checkpoint,
        )

        require_image_family(family)
        if family == "SD_X4Upscaler":
            # Reject a missing SD4x source before any checkpoint or component lookup.
            from ruinedfooocus.features.image_editing.sd4x_conditioning import (
                require_sd4x_input,
            )

            require_sd4x_input(
                kind, result.get("input_image"), inpainting=result.get("inpaint_toggle", False)
            )

        from ruinedfooocus.features.models.native_workflow_admission import (
            require_available_image_workflow,
        )

        getter = getattr(models, "get_file", None)
        checkpoint = getter("checkpoints", result["base_model_name"]) if callable(getter) else None
        if checkpoint is not None:
            require_image_checkpoint(checkpoint, family)
        if family == "BerniniImage":
            from ruinedfooocus.features.generation.bernini_image import capture_assets

            result.setdefault("image_engine", (values or {}).get("image_engine", "auto"))
            result, bernini_assets = capture_assets(result, checkpoint, manager, values)
            pending_loras = tuple(
                item
                for item in _plan_component_assets(result, manager, ()) or ()
                if item.role == "loras"
            )
            result[KEY] = captured((*bernini_assets, *pending_loras))
            return result
        result.pop("_native_bernini_image", None)
        result.pop("native_bernini_image", None)
        result.setdefault("image_engine", (values or {}).get("image_engine", "auto"))
        if family == "MiniMaxH3Image":
            from ruinedfooocus.features.generation.minimax_image import capture as capture_minimax

            result = capture_minimax(result, checkpoint)
            controls = result["controlnet"]
            kind = str(controls.get("type", "")).lower()
        else:
            result.pop("_native_minimax_image", None)
            result.pop("native_minimax_image", None)
        require_available_image_workflow(
            checkpoint,
            family=family,
            prepared_data=result,
        )

        result.setdefault("image_engine", (values or {}).get("image_engine", "auto"))
        result = capture_layered(result, manager, models, family=family)
        controls = result["controlnet"]
        kind = str(controls.get("type", "")).lower()
    auxiliary_assets = []
    if (result.get("gligen_model") or result.get("hypernetwork_model")) and str(
        result.get("image_engine", (values or {}).get("image_engine", "auto"))
    ).startswith("sdcpp-"):
        raise ValueError("GLIGEN and hypernetworks require the native image engine.")
    if result.get("gligen_model") or result.get("_native_gligen_layout"):
        from ruinedfooocus.features.image_editing.gligen_layout import capture_assets

        result, captured_auxiliary = capture_assets(
            result,
            manager,
            models,
            _control_checkpoint_family(models, result.get("base_model_name")),
        )
        auxiliary_assets.extend(captured_auxiliary)
    if result.get("hypernetwork_model") or result.get("_native_hypernetwork"):
        from ruinedfooocus.features.image_editing.native_hypernetwork import capture_assets

        result, captured_auxiliary = capture_assets(
            result,
            manager,
            models,
            _control_checkpoint_family(models, result.get("base_model_name")),
        )
        auxiliary_assets.extend(captured_auxiliary)
    from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

    result = native_control_recipes.capture(result, manager, models, controls)
    result, patch_assets = model_patches.capture_assets(result, manager, models, controls)
    from ruinedfooocus.features.generation.native_tensor_io import enabled as tensor_io_enabled

    if (
        not native_utility
        and kind not in ("rembg", "upscale")
        and tensor_io_enabled(result)
        and str(result.get("image_engine", (values or {}).get("image_engine", "auto"))).startswith(
            "sdcpp-"
        )
    ):
        raise ValueError("Native latent and conditioning files require the native image engine.")
    from ruinedfooocus.features.image_editing import lora_hooks

    result, hook_assets = lora_hooks.capture_assets(result, manager, models, controls)
    if str(result.get("image_engine", (values or {}).get("image_engine", "auto"))).startswith(
        "sdcpp-"
    ) and not (kind == "rembg" and result.get("rembg_native_model") or native_utility):
        from ruinedfooocus.features.models.approximate_codecs import is_choice

        if any(
            is_choice(value) for value in (result.get("image_component_overrides") or {}).values()
        ):
            raise ValueError("Full approximate codecs require the native image engine.")
        if kind == qwen_image21_union.KIND:
            raise ValueError("Qwen Image 2.1 Union requires the native image engine.")
        if controls.get("model"):
            raise ValueError("Selected ControlNet/T2I files require the native image engine.")
        if result.get("sdxl_refiner_model"):
            raise ValueError("SDXL base/refiner staging requires the native image engine.")
        if (
            kind == "depth"
            and controls.get("depth_model") == "lotus"
            and controls.get("depth_preprocess", "none") != "none"
        ):
            raise ValueError("Lotus depth preprocessing requires the native image engine.")
        if result.get("model_recipe_path"):
            raise ValueError("YAML recipes require the native image engine.")
        if result.get("reference_adapter") not in (None, "", "none"):
            raise ValueError("Reference adapters require the native image engine.")
        result[KEY] = captured(())
        return result
    artifacts = [*patch_assets, *hook_assets, *auxiliary_assets]
    primary = result.get("_prepared_primary_asset")
    if isinstance(primary, ImageArtifact) and models is not None:
        checkpoint = models.get_file("checkpoints", result.get("base_model_name"))
        if checkpoint is not None and Path(checkpoint).resolve() == primary.request.destination:
            if tuple(file_signature(checkpoint)) != primary.signature:
                raise ValueError(
                    "The prepared primary changed before final capture. Select it again."
                )
            artifacts.append(primary)
    if "_native_minimax_image" in result:
        artifacts.append(
            _artifact(manager, "minimax_model", "checkpoints", result["base_model_name"])
        )
    if native_utility:
        from ruinedfooocus.features.image_processing.native_vision import (
            capture_assets,
            worker_data,
        )

        # Self-contained utility: no checkpoint, component or upscaler capture.
        result, item = capture_assets(result, manager, kind)
        result[KEY] = captured((item,))
        return worker_data(result)
    if kind == "rembg" and result.get("rembg_native_model"):
        from ruinedfooocus.features.image_processing.native_background import validate_model

        item = _artifact(
            manager, "background-removal", "background_removal", result["rembg_native_model"]
        )
        if item.signature is not None:
            validate_model(item.request.destination)
        artifacts.append(item)
    from ruinedfooocus.features.models.local_diffusers import BUNDLE_KEY, LocalDiffusersBundle

    bundle = result.get(BUNDLE_KEY)
    bundle = bundle if isinstance(bundle, LocalDiffusersBundle) else None
    if isinstance(bundle, LocalDiffusersBundle):
        bundle.verify()
        artifacts.extend(
            ImageArtifact("bundle", item.role, PreparationRequest("", item.path), item.signature)
            for item in bundle.files
        )
    from ruinedfooocus.features.models.sensenova_bundle import KEY as SENSE_KEY, SenseNovaBundle

    sense_bundle = result.get(SENSE_KEY)
    if isinstance(sense_bundle, SenseNovaBundle):
        sense_bundle.verify()
        artifacts.extend(
            ImageArtifact(
                "sensenova-bundle",
                str(item.path),
                PreparationRequest("", item.path),
                item.signature,
            )
            for item in sense_bundle.files
        )
    if kind == qwen_image21_union.KIND:
        if models is not None and result.get("base_model_name"):
            from ruinedfooocus.features.models.image_components import ALIASES

            family = _control_checkpoint_family(models, result["base_model_name"])
            if ALIASES.get(family, family) != "QwenImage21":
                raise ValueError("Qwen Image 2.1 Union requires a Qwen Image 2.1 checkpoint.")
        result, patch = qwen_image21_union.capture(result, manager, controls)
        controls = result["controlnet"]
        artifacts.append(patch)
    from ruinedfooocus.features.image_editing.control_candidates import (
        CONTROL_KINDS,
        selected_control,
        model_family,
    )

    if kind in CONTROL_KINDS and (
        result.get("input_image") is not None or native_control_recipes.KEY in result
    ):
        family = _control_checkpoint_family(models, result.get("base_model_name"))
        name = controls.get("model")
        if name:
            if models is None:
                raise ValueError(
                    "Capture the selected control model with the checkpoint inventory."
                )
            path, signature, info = selected_control(
                manager, name, family, controls.get("model_signature")
            )
            item = _artifact(manager, "controlnet", "controlnet", name)
            if item.request.destination.resolve() != path or item.signature != tuple(signature):
                raise ValueError(
                    "The selected control model changed during capture. Select it again."
                )
            artifacts.append(replace(item, name=kind))
        else:
            if kind == "control" or models is not None and model_family(family) != "SDXL":
                raise ValueError("Choose a local control model for this image checkpoint.")
            artifacts.append(_artifact(manager, "controlnet", "controlnet", kind, "cn_" + kind))
    depth_name = None
    if kind == "depth" and result.get("input_image") is not None:
        from ruinedfooocus.features.image_editing.depth_control import (
            depth_settings,
            captured_depth_name,
        )

        options = depth_settings(controls)
        if options["depth_preprocess"] != "none" and options["depth_model"] == "lotus":
            from ruinedfooocus.features.image_editing.lotus_depth import (
                validate_model,
                validate_vae,
                selected_file,
            )

            for role, category, key, validate in (
                ("lotus-depth", "checkpoints", "depth_lotus_model", validate_model),
                ("lotus-vae", "vae", "depth_lotus_vae", validate_vae),
            ):
                local = selected_file(
                    manager, options[key], "model" if role == "lotus-depth" else "vae"
                )
                item = replace(_artifact(manager, role, category, str(local)), name=options[key])
                if item.signature is None:
                    raise ValueError("Lotus depth requires explicitly installed local components.")
                validate(item.request.destination)
                artifacts.append(item)
        elif options["depth_preprocess"] != "none":
            from ruinedfooocus.features.generation.sdcpp_pipeline import cpu_requested

            depth_name = captured_depth_name(
                options["depth_model"], force_cpu=cpu_requested(values or {})
            )
            artifacts.append(
                _artifact(manager, "depth-preprocessor", "geometry_estimation", depth_name)
            )
    if kind == "upscale":
        name = controls.get("upscaler")
        if not name or name == "None":
            raise ValueError("Choose an upscaler in Image tools.")
        artifacts.append(_artifact(manager, "upscaler", "upscalers", name))
    auto_name, auto_scale = None, 1
    if kind not in ("upscale", "rembg"):
        from ruinedfooocus.features.generation.automatic_upscale import resolve

        auto_name, auto_scale = resolve(result.get("automatic_upscale"))
        if auto_name:
            artifacts.append(_artifact(manager, "upscaler", "upscalers", auto_name))
        from ruinedfooocus.features.models.approximate_codecs import (
            KEY as APPROXIMATE_KEY,
            capture_request as capture_approximate,
        )

        result = capture_approximate(result, manager)
        pair = result.get(APPROXIMATE_KEY)
        if pair is not None:
            for item in pair.files:
                artifacts.append(
                    ImageArtifact(
                        "approximate_vae",
                        pair.slot + ":" + item.role,
                        PreparationRequest("", item.path),
                        item.signature,
                    )
                )
        if (
            models is not None
            and result.get("base_model_name")
            and bundle is None
            and not isinstance(sense_bundle, SenseNovaBundle)
        ):
            from ruinedfooocus.features.models.component_profiles import component_inventory
            from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS

            name = result["base_model_name"]
            metadata = models.get_models_by_path("checkpoints", name, fetch=False)
            from ruinedfooocus.features.models.utility_models import is_image_utility

            if is_image_utility(metadata):
                raise ValueError(
                    "This installed model is an image utility. Choose its matching Image tool."
                )
            family = models.get_model_base(metadata)
            from ruinedfooocus.features.models.native_component_layout import (
                KEY as LAYOUT_KEY,
                current_layout,
            )

            checkpoint = models.get_file("checkpoints", name)
            layout = (
                current_layout(checkpoint, manager)
                if checkpoint is not None and family == "QwenImage"
                else None
            )
            if layout is not None:
                result[LAYOUT_KEY] = layout
                result["native_component_layout"] = layout.metadata()
            else:
                result.pop(LAYOUT_KEY, None)
                result.pop("native_component_layout", None)
            if family in ("PiD", "HunyuanImage21Refiner"):
                if (
                    kind != "edit"
                    or result.get("input_image") is None
                    or result.get("inpaint_toggle")
                ):
                    raise ValueError(
                        "This native restoration/refiner requires Image tools Edit with a source image and no mask."
                    )
                if family == "PiD":
                    from ruinedfooocus.features.image_editing.pid_conditioning import (
                        validate_input_codec,
                        degrade_sigma,
                    )

                    degrade_sigma(result)
                    codec = result.get("pid_input_vae")
                    if not codec:
                        raise ValueError("Select PiD's input VAE before submitting restoration.")
                    item = _artifact(manager, "pid_input_vae", "vae", codec)
                    if item.signature is None:
                        raise FileNotFoundError(
                            "Install the selected PiD input VAE locally before restoration."
                        )
                    validate_input_codec(
                        item.request.destination,
                        result.get("pid_latent_format"),
                        models.get_file("checkpoints", name),
                    )
                    artifacts.append(item)
                else:
                    from ruinedfooocus.features.image_editing.hunyuan_refiner import (
                        noise_augmentation,
                    )

                    noise_augmentation(result)
            if family in ("Stable_Cascade_B", "StableCascade_B"):
                raise ValueError(
                    "Select Stage C as the primary checkpoint and Stage B in Cascade settings."
                )
            if family in ("Stable_Cascade_C", "StableCascade_C"):
                from ruinedfooocus.features.generation.cascade import (
                    sampling_settings,
                    require_bundle,
                    stage_b_defaults,
                )

                for key, value in stage_b_defaults(result.get("cascade_stage_b"), models).items():
                    if result.get(key) in (None, ""):
                        result[key] = value
                sampling_settings(result)
                require_bundle(models.get_file("checkpoints", name), "c")
                stage_b = models.get_file("checkpoints", result["cascade_stage_b"])
                if stage_b is None or models.detect_checkpoint_base(stage_b) != "Stable_Cascade_B":
                    raise ValueError(
                        "Choose an installed native Stable Cascade Stage B checkpoint."
                    )
                require_bundle(stage_b, "b")
                artifacts.append(
                    _artifact(manager, "cascade_stage_b", "checkpoints", result["cascade_stage_b"])
                )
            if family == "SD_X4Upscaler":
                from ruinedfooocus.features.image_editing.sd4x_conditioning import (
                    require_sd4x_input,
                )

                require_sd4x_input(
                    kind, result.get("input_image"), inpainting=result.get("inpaint_toggle", False)
                )
            checkpoint = models.get_file("checkpoints", name)
            if (
                checkpoint
                and (is_safetensors(checkpoint) or Path(checkpoint).suffix.lower() == ".gguf")
                and ALIASES.get(family, family) in IMAGE_COMPONENTS
            ):
                planned = _plan_component_assets(result, manager, artifacts)
                inventory = (
                    component_inventory(
                        checkpoint,
                        family,
                        values or {},
                        manager,
                        result.get("image_component_overrides"),
                    )
                    if planned is None
                    else None
                )
                from ruinedfooocus.features.models.component_candidates import validate_inventory

                if inventory is not None:
                    validate_inventory(inventory)
                artifacts.extend(planned or ())
                for row in inventory["rows"] if inventory is not None else ():
                    if row["status"] == "Embedded in checkpoint" or row.get("approximate_codec"):
                        continue
                    location = Path(row["location"]).resolve()
                    key = row["download_key"]
                    if not location.is_file() and not row["download_available"]:
                        raise FileNotFoundError(
                            f"Image component {row['key']} is missing: {row['filename']}. Install it locally or reset this role to Automatic."
                        )
                    request = (
                        manager.preparation_request(key, destination=location)
                        if key
                        else PreparationRequest("", location)
                    )
                    artifacts.append(
                        ImageArtifact(
                            row["category"],
                            row["key"],
                            request,
                            tuple(file_signature(location)) if location.is_file() else None,
                        )
                    )
        if result.get("base_model_name"):
            from ruinedfooocus.features.generation.cascade_guidance import (
                capture_assets as capture_cascade_guidance,
            )

            result, cascade_assets = capture_cascade_guidance(result, manager, models)
            artifacts.extend(cascade_assets)
            from ruinedfooocus.features.generation.sdxl_refiner import (
                capture_assets as capture_refiner,
            )

            result, refiner_assets = capture_refiner(result, manager, models, values)
            artifacts.extend(refiner_assets)
            from ruinedfooocus.features.generation.native_sampling import (
                capture_assets as capture_sampling,
            )

            result, sampling_assets = capture_sampling(result, manager, models)
            artifacts.extend(sampling_assets)

    from ruinedfooocus.features.models.yaml_recipes import capture_request, captured_recipe

    if result.get("model_recipe_path") and models is None:
        raise ValueError(
            "Capture the selected YAML recipe with the installed checkpoint inventory."
        )
    result.setdefault("image_engine", (values or {}).get("image_engine", "auto"))
    result = capture_request(result, models, manager, controls)
    recipe = captured_recipe(result)
    if recipe is not None:
        for role, item in (("yaml-recipe", recipe.recipe), ("yaml-checkpoint", recipe.checkpoint)):
            artifacts.append(
                ImageArtifact(
                    role, str(item.source), PreparationRequest("", item.path), item.signature
                )
            )
    from ruinedfooocus.features.image_editing.reference_adapters import (
        capture_assets as capture_references,
    )

    result, references = capture_references(result, manager, models, controls)
    artifacts.extend(references)
    from ruinedfooocus.features.image_editing.krea_style import capture_assets as capture_krea_style

    result, style_assets = capture_krea_style(result, manager, models, controls)
    artifacts.extend(style_assets)
    result[KEY] = captured(artifacts, auto_name, auto_scale, depth_name)
    from ruinedfooocus.features.models import gguf_projectors

    result.pop(gguf_projectors.KEY, None)
    result = gguf_projectors.capture_assets(result, artifacts)
    if kind not in ("rembg", "upscale") and tensor_io_enabled(result):
        from ruinedfooocus.features.generation.native_tensor_io import (
            capture_request as capture_tensors,
        )

        result = capture_tensors(result, manager, models)
    return result


def needs_image_preparation(data):
    captured = data.get(KEY)
    return isinstance(captured, ImageAssets) and any(
        item.signature is None or not item.request.destination.is_file()
        for item in captured.artifacts
    )


def prepare_image_assets(data, *, check_cancelled, progress=None, service=preparation_service):
    if pending_image_selection(data) is not None:
        raise ValueError(
            "Prepare the selected primary and recapture its image plan before preparing companions."
        )
    from ruinedfooocus.features.models.yaml_recipes import captured_recipe

    recipe = captured_recipe(data)
    from ruinedfooocus.features.models import gguf_projectors

    gguf_projectors.captured(data)
    if recipe is not None:
        recipe.verify(_Cancellation(check_cancelled))
    captured = data.get(KEY)
    if not isinstance(captured, ImageAssets):
        return data
    check_cancelled()
    plan = _effective_plan(data)
    if plan is not None:
        plan.verify_request(data)
    readiness = image_readiness(data)
    if readiness.state == "blocked":
        missing_local = any(
            row.reason and "removed after submission" in row.reason
            for row in readiness.requirements
        )
        error = FileNotFoundError if missing_local else ValueError
        raise error(" ".join(readiness.reasons))
    _check_disk_capacity(captured)
    subscriptions = []
    try:
        # A captured local identity cannot become a new acquisition after removal.
        # Check the complete set before starting any missing-at-capture transfer.
        submitted = set()
        for item in captured.artifacts:
            check_cancelled()
            if item.signature is not None and not item.request.destination.is_file():
                if captured.offline:
                    raise ValueError(f"Offline mode: image {item.role} is missing: {item.name}.")
                raise FileNotFoundError(
                    f"Captured image {item.role} was removed after submission: {item.name}. Choose it again."
                )
        for item in captured.artifacts:
            check_cancelled()
            if (
                item.role in ("gligen", "hypernetwork")
                and not item.request.url
                and not item.request.destination.is_file()
            ):
                raise FileNotFoundError(
                    "The selected local image model was removed. Choose it again."
                )
            if item.request.destination.is_file() and not (
                item.receipt_required and item.signature is None
            ):
                continue
            if captured.offline and not item.request.destination.is_file():
                raise ValueError(f"Offline mode: image {item.role} is missing: {item.name}.")
            if item.signature is not None:
                raise FileNotFoundError(
                    f"Captured image {item.role} was removed after submission: {item.name}. Choose it again."
                )
            if not item.request.url:
                raise FileNotFoundError(
                    "The selected local image model was removed. Choose it again."
                )
            if item.role == "approximate_vae":
                raise FileNotFoundError(
                    "An approximate codec file was removed. Refresh and choose again."
                )
            if item.request.destination not in submitted:
                subscriptions.append(service.submit(item.request, progress))
                submitted.add(item.request.destination)
        for subscription in subscriptions:
            while True:
                check_cancelled()
                try:
                    subscription.result(timeout=0.1)
                    break
                except TimeoutError:
                    if subscription.done():
                        raise
                    continue
        check_cancelled()
        result = dict(data)
        artifacts = []
        receipts = {
            subscription.request.destination: subscription.verified_signature
            for subscription in subscriptions
        }
        from ruinedfooocus.core.cache_contract import file_identity

        for item in captured.artifacts:
            with service.artifact_lease(
                item.request.destination, cancelled=_Cancellation(check_cancelled)
            ) as path:
                signature = tuple(file_signature(path))
                if item.signature is not None and item.signature != signature:
                    raise ValueError(f"Image {item.role} changed after submission: {item.name}.")
                receipt = receipts.get(path)
                if item.signature is None and (
                    receipt is None or dict(receipt) != file_identity(path)
                ):
                    verify_file(path, item.request, _Cancellation(check_cancelled))
                artifacts.append(replace(item, signature=signature))
        warnings = list(captured.source_warnings)
        if captured.cache_root is not None:
            from ruinedfooocus.features.models.model_identity import record_prepared_native_source

            for subscription in subscriptions:
                check_cancelled()
                receipt = receipts.get(subscription.request.destination)
                if receipt is not None:
                    try:
                        record_prepared_native_source(
                            subscription.request.destination,
                            subscription.request,
                            receipt,
                            captured.cache_root,
                        )
                    except (OSError, ValueError):
                        warnings.append(
                            "Downloaded files retained; source details could not be cached. Use Inspect before Model Defaults."
                        )
        result[KEY] = replace(
            captured, artifacts=tuple(artifacts), source_warnings=tuple(dict.fromkeys(warnings))
        )
        result = _validate_prepared_utilities(result, artifacts, service, check_cancelled)
        result = gguf_projectors.capture_assets(result, artifacts)
        if plan is not None:
            from ruinedfooocus.features.models.effective_image_plan import (
                finalize_prepared_image_plan,
            )

            result = finalize_prepared_image_plan(data, result)
            plan = _effective_plan(result)
        if recipe is not None:
            recipe.verify(_Cancellation(check_cancelled))
        if plan is not None:
            plan.verify_request(result)
        check_cancelled()
        return result
    finally:
        for subscription in subscriptions:
            if not subscription.done():
                subscription.cancel()


@contextmanager
def prepared_asset(data, role, name, manager, category, *, service=preparation_service):
    """Protect the exact captured file while a native loader reads it."""
    captured = (data or {}).get(KEY)
    if isinstance(captured, ImageAssets):
        item = next(
            (item for item in captured.artifacts if item.role == role and item.name == str(name)),
            None,
        )
        if item is None:
            raise ValueError(f"Image {role} was not captured before compute: {name}.")
    else:
        key = "cn_" + str(name) if role == "controlnet" else None
        item = _artifact(manager, role, category, name, key)
    if item.signature is None:
        raise FileNotFoundError(f"Prepare image {role} before loading: {name}.")
    from ruinedfooocus.app.resource_lifecycle import current_job

    job = (data or {}).get("_job") or (data or {}).get("_api_job") or current_job()
    with service.artifact_lease(
        item.request.destination, cancelled=getattr(job, "cancelled", None)
    ) as path:
        if tuple(file_signature(path)) != item.signature:
            raise ValueError(f"Image {role} changed after submission. Submit it again: {name}.")
        yield path


def component_read_leases(function):
    """Hold all exact external component read leases until native loading ends."""

    @wraps(function)
    def load(*args, **kwargs):
        if _component_leases.get() is not None:
            # Recursive missing-component fallback shares the outer reader.
            # A waiting writer must not block a second read of our own files.
            return function(*args, **kwargs)
        data = (
            inspect.signature(function).bind_partial(*args, **kwargs).arguments.get("prepared_data")
        )
        from ruinedfooocus.features.models.yaml_recipes import captured_recipe, recipe_read_scope

        recipe = captured_recipe(data)
        with ExitStack() as stack:
            token = _component_leases.set(stack)
            try:
                captured = (data or {}).get(KEY)
                artifacts = captured.artifacts if isinstance(captured, ImageAssets) else ()
                # Cascade's stage scope already owns its primary read lease.
                if not any(item.role == "cascade-primary" for item in artifacts):
                    from ruinedfooocus.app.resource_lifecycle import current_job

                    job = (data or {}).get("_job") or (data or {}).get("_api_job") or current_job()
                    for item in artifacts:
                        if item.role == "primary":
                            path = stack.enter_context(
                                preparation_service.artifact_lease(
                                    item.request.destination,
                                    cancelled=getattr(job, "cancelled", None),
                                )
                            )
                            if tuple(file_signature(path)) != item.signature:
                                raise ValueError(
                                    "The prepared primary changed before native loading. Submit it again."
                                )
                if recipe is not None:
                    from ruinedfooocus.app.resource_lifecycle import current_job

                    job = (data or {}).get("_job") or (data or {}).get("_api_job") or current_job()
                    stack.enter_context(
                        recipe_read_scope(recipe, cancelled=getattr(job, "cancelled", None))
                    )
                from ruinedfooocus.features.models.sensenova_bundle import (
                    KEY as SENSE_KEY,
                    SenseNovaBundle,
                    read_scope as sense_read_scope,
                )

                sense_bundle = (data or {}).get(SENSE_KEY)
                if isinstance(sense_bundle, SenseNovaBundle):
                    from ruinedfooocus.app.resource_lifecycle import current_job

                    job = (data or {}).get("_job") or (data or {}).get("_api_job") or current_job()
                    stack.enter_context(
                        sense_read_scope(sense_bundle, cancelled=getattr(job, "cancelled", None))
                    )
                from ruinedfooocus.features.models.approximate_codecs import (
                    KEY as APPROXIMATE_KEY,
                    ApproximateCodec,
                    read_scope as approximate_read_scope,
                )

                pair = (data or {}).get(APPROXIMATE_KEY)
                if isinstance(pair, ApproximateCodec):
                    from ruinedfooocus.app.resource_lifecycle import current_job

                    job = (data or {}).get("_job") or (data or {}).get("_api_job") or current_job()
                    stack.enter_context(
                        approximate_read_scope(pair, cancelled=getattr(job, "cancelled", None))
                    )
                return function(*args, **kwargs)
            finally:
                _component_leases.reset(token)

    return load


def component_file(data, role, slot, filename, manager):
    stack = _component_leases.get()
    if stack is None:
        raise RuntimeError("Native component loading requires an owned file-read context.")
    captured = (data or {}).get(KEY)
    return stack.enter_context(
        prepared_asset(
            data, role, slot if isinstance(captured, ImageAssets) else filename, manager, role
        )
    )
