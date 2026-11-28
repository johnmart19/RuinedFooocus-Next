"""Request-owned safe native tensor inputs and opt-in generation exports."""

import base64
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import types

from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_native_tensor_plan"
EXPORT_KEY = "_native_tensor_exports"
INPUTS = ("native_latent_input", "native_positive_input", "native_negative_input")
FIELDS = (
    *INPUTS,
    "native_tensor_family",
    "native_tensor_workflow",
    "native_latent_denoise",
    "native_export_latent",
    "native_export_conditioning",
    "native_export_handoff",
)
MAX_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_HEADER = 128 * 1024
# Native model configs, latent channels and CLIPTextEncode widths. These are
# ordinary still consumers, not an admission claim for layered/staged models.
FAMILIES = {
    "SD15": (4, 768),
    "SD20": (4, 1024),
    "SDXL": (4, 2048),
    "SD3": (16, 4096),
    "Flux": (16, 4096),
    "FluxSchnell": (16, 4096),
}
MODEL_INPUT_CHANNELS = {"SD15": 4, "SD20": 4, "SDXL": 4, "SD3": 16, "Flux": 64, "FluxSchnell": 64}
DTYPES = {"F16": 2, "BF16": 2, "F32": 4}
CONTEXT_FIELDS = frozenset(
    {
        "family",
        "model_sha256",
        "vae_type",
        "channels",
        "spatial_scale",
        "component_signature",
        "lora_signature",
    }
)
POOLED = {"SD15": 768, "SD20": 1024, "SDXL": 1280, "SD3": 2048, "Flux": 768, "FluxSchnell": 768}
SCALARS = frozenset(
    {
        "guidance",
        "width",
        "height",
        "crop_w",
        "crop_h",
        "target_width",
        "target_height",
        "aesthetic_score",
        "strength",
        "start_percent",
        "end_percent",
        "set_area_to_bounds",
    }
)


@dataclass(frozen=True)
class TensorInput:
    role: str
    path: Path
    signature: tuple
    sha256: str
    dtype: str
    shape: tuple
    context: tuple
    legacy_scale: float = 1.0


@dataclass(frozen=True)
class TensorPlan:
    family: str
    directory: Path
    inputs: tuple
    export_latent: bool
    export_conditioning: bool
    context: str = ""
    export_handoff: bool = False


@dataclass(frozen=True)
class TensorExport:
    path: str
    kind: str
    signature: tuple
    sha256: str
    family: str
    seed: int = 0
    generation_index: int = 0


class TensorGenerationArtifacts(list):
    """Images remain images; native tensors are separate request-owned downloads."""

    def __init__(self, images, tensors):
        super().__init__(images)
        self.tensor_files = tuple(tensors)


def enabled(data):
    return any(data.get(key) is not None for key in INPUTS) or any(
        data.get(key) is True
        for key in ("native_export_latent", "native_export_conditioning", "native_export_handoff")
    )


def replay_values(metadata, data):
    source = metadata.get("native_tensor_source")
    if source is None:
        return {}
    from .native_tensor_context import SAVED_FAMILIES

    if (
        not isinstance(source, dict)
        or not isinstance(source.get("family"), str)
        or source["family"] not in SAVED_FAMILIES
    ):
        raise ValueError("Invalid saved native tensor family context.")
    entries = source.get("inputs", [])
    if not isinstance(entries, list) or len(entries) > 3:
        raise ValueError("Invalid saved native tensor input context.")
    roles = {"latent": INPUTS[0], "positive": INPUTS[1], "negative": INPUTS[2]}
    seen = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("role"), str)
            or entry["role"] not in roles
            or entry["role"] in seen
        ):
            raise ValueError("Invalid saved native tensor role.")
        seen.add(entry["role"])
        if data.get(roles[entry["role"]]) is None:
            raise ValueError(
                "Re-upload the original native tensor inputs before replaying this image."
            )
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or hashlib.sha256(_raw(data[roles[entry["role"]]])).hexdigest() != digest
        ):
            raise ValueError("The uploaded tensor does not match the saved replay input identity.")
    result = {"native_tensor_family": source.get("family", "auto")}
    if source.get("workflow") is True:
        result["native_tensor_workflow"] = True
    if "native_latent_denoise" in metadata:
        result["native_latent_denoise"] = metadata["native_latent_denoise"]
    for key in ("native_export_latent", "native_export_conditioning", "native_export_handoff"):
        value = metadata.get(key, False)
        if type(value) is not bool:
            raise ValueError("Invalid saved native tensor export choice.")
        result[key] = value
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate native tensor header keys are not allowed.")
        result[key] = value
    return result


def tensor_header(raw, role, family):
    """Bound dtype/rank/count/bytes before a native reader can allocate tensors."""
    from . import native_tensor_context as workflow

    if workflow.has_context(raw):
        return workflow.header(raw, role, family)
    if not isinstance(raw, bytes) or not 10 <= len(raw) <= MAX_BYTES:
        raise ValueError("A native tensor file must contain at most 32 MiB.")
    size = int.from_bytes(raw[:8], "little")
    if not 2 <= size <= min(MAX_HEADER, len(raw) - 8):
        raise ValueError("Invalid or oversized native safetensors header.")
    try:
        header = json.loads(raw[8 : 8 + size], object_pairs_hook=_object)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise ValueError("Invalid native safetensors header.") from error
    if not isinstance(header, dict) or not 1 <= len(header) <= 18:
        raise ValueError("Native tensor files require a bounded tensor dictionary.")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str)
        or len(key) > 128
        or not isinstance(value, str)
        or len(value) > 32768
        for key, value in metadata.items()
    ):
        raise ValueError("Native tensor metadata must contain bounded strings.")
    rows = {key: value for key, value in header.items() if key != "__metadata__"}
    allowed = (
        {"latent_tensor", "latent_format_version_0"}
        if role == "latent"
        else {"conditioning", "pooled_output"}
    )
    if not rows or set(rows) - allowed:
        raise ValueError("Use a native latent or single ordinary text-conditioning file.")
    primary = "latent_tensor" if role == "latent" else "conditioning"
    if primary not in rows:
        raise ValueError("The file does not contain the requested native tensor role.")
    intervals = []
    for name, entry in rows.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Invalid native tensor descriptor.")
        shape, dtype, offsets = entry["shape"], entry["dtype"], entry["data_offsets"]
        if (
            not isinstance(dtype, str)
            or dtype not in DTYPES
            or not isinstance(shape, list)
            or len(shape) > 4
            or any(type(side) is not int or not 0 <= side <= 32768 for side in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError("Unsupported native tensor dtype, rank or dimensions.")
        count = math.prod(shape)
        if (
            offsets[0] < 0
            or offsets[1] - offsets[0] != count * DTYPES[dtype]
            or offsets[1] > len(raw) - 8 - size
        ):
            raise ValueError("Native tensor byte offsets disagree with its bounded shape.")
        if name == "latent_format_version_0":
            if shape != [0] or dtype != "F32":
                raise ValueError("Invalid native latent version marker.")
        elif not shape or min(shape) < 1 or shape[0] != 1:
            raise ValueError("Native tensor inputs require exactly one nonempty still sample.")
        if count:
            intervals.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Native tensor byte ranges must be contiguous and nonoverlapping.")
        cursor = end
    if cursor != len(raw) - 8 - size:
        raise ValueError("Native tensor file contains trailing or unreferenced bytes.")
    if family not in FAMILIES:
        raise ValueError("Choose a supported ordinary still-image tensor family.")
    channels, width = FAMILIES[family]
    shape = rows[primary]["shape"]
    if role == "latent":
        if (
            len(shape) != 4
            or shape[1] != channels
            or max(shape[-2:]) > 1024
            or math.prod(shape[-2:]) * 64 > 16_777_216
        ):
            raise ValueError("The latent shape does not match the selected still model family.")
        if "latent_format_version_0" not in rows and family not in ("SD15", "SD20"):
            raise ValueError("Legacy unmarked latents require the original SD1/SD2 codec context.")
    elif len(shape) != 3 or shape[-1] != width or shape[1] > 16384:
        raise ValueError("Conditioning width/rank does not match the selected text encoder family.")
    if "pooled_output" in rows:
        pooled = rows["pooled_output"]["shape"]
        if len(pooled) != 2 or pooled[-1] != POOLED[family]:
            raise ValueError("Invalid pooled text-conditioning output.")
    elif role != "latent" and family in ("SDXL", "SD3", "Flux", "FluxSchnell"):
        raise ValueError("This model family requires its matching pooled text-conditioning output.")
    try:
        options = json.loads(metadata.get("conditioning_options", "{}"), object_pairs_hook=_object)
        context = json.loads(metadata.get("rf_tensor_context", "{}"), object_pairs_hook=_object)
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid native conditioning or source context metadata.") from error
    if not isinstance(context, dict) or set(context) - CONTEXT_FIELDS:
        raise ValueError("Invalid native tensor source context.")
    if context.get("family", family) != family:
        raise ValueError("The native tensor belongs to a different model family.")
    if context and (
        set(context) != CONTEXT_FIELDS
        or not isinstance(context["model_sha256"], str)
        or context["model_sha256"]
        and not re.fullmatch(r"[0-9a-fA-F]{64}", context["model_sha256"])
        or not isinstance(context["vae_type"], str)
        or not 1 <= len(context["vae_type"]) <= 128
        or type(context["channels"]) is not int
        or context["channels"] != channels
        or type(context["spatial_scale"]) is not int
        or context["spatial_scale"] != 8
        or any(
            not isinstance(context[key], str) or not re.fullmatch(r"[0-9a-f]{64}", context[key])
            for key in ("component_signature", "lora_signature")
        )
    ):
        raise ValueError("Invalid native tensor model/codec context.")
    if (
        not isinstance(options, dict)
        or set(options) - SCALARS
        or any(
            type(value) not in (bool, int, float)
            or not -1e9 <= value <= 1e9
            or not math.isfinite(value)
            for value in options.values()
        )
    ):
        raise ValueError("Only bounded ordinary text-conditioning options can be imported.")
    return header, context


def _raw(value):
    if isinstance(value, bytes):
        if len(value) > MAX_BYTES:
            raise ValueError("A native tensor file must contain at most 32 MiB.")
        return value
    if not isinstance(value, str) or len(value) > (MAX_BYTES * 4 // 3 + 8):
        raise ValueError(
            "Use an uploaded native tensor or bounded base64 data, never a local path."
        )
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Native tensor inputs must be base64 safetensors data.") from error


def _directory(manager):
    from ruinedfooocus.core.cache_contract import configured_cache_root

    root = configured_cache_root(manager).resolve()
    parent = root / "native-tensor-jobs"
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.resolve().is_relative_to(root):
        raise ValueError("Native tensor storage escaped its configured cache.")
    folder = parent / secrets.token_hex(24)
    folder.mkdir(mode=0o700)
    return folder.resolve()


def capture_request(data, manager, models):
    """No call, file, tensor read or clone is needed when every control is Off."""
    if not enabled(data):
        return data
    from ruinedfooocus.features.models.model_architecture import inspect_model_file

    _check_job(data)
    result = dict(data)
    path = models.get_file("checkpoints", result.get("base_model_name"))
    details = inspect_model_file(path) if path else {}
    family = details.get("native_class")
    if result.get("native_export_handoff") and not result.get("sdxl_refiner_model"):
        raise ValueError(
            "Native noisy handoff exports require an explicit SDXL base/refiner workflow."
        )
    if details.get("native_config", {}).get("in_channels") == 9 and not result.get(
        "inpaint_toggle"
    ):
        raise ValueError(
            "Specialized inpainting checkpoints require their original source and mask."
        )
    from . import native_tensor_context as workflow

    raws = [_raw(result[key]) for key in INPUTS if result.get(key) is not None]
    if workflow.needs_context(result, details, raws):
        return workflow.capture(result, manager, models, family)
    if family not in FAMILIES:
        raise ValueError(
            "Native tensor inputs/exports currently need ordinary SD1/SD2/SDXL/SD3/Flux generation."
        )
    if (
        details.get("native_config", {}).get("in_channels", MODEL_INPUT_CHANNELS[family])
        != MODEL_INPUT_CHANNELS[family]
    ):
        raise ValueError(
            "Specialized image and inpainting checkpoints need their own tensor context."
        )
    if (
        str(result.get("image_engine", "auto")).startswith("sdcpp-")
        or (result.get("controlnet") or {}).get("type") not in (None, "")
        or result.get("input_image") is not None
        or result.get("inpaint_toggle")
        or result.get("reference_images")
        or result.get("reference_adapter") not in (None, "", "none")
        or result.get("cascade_stage_b")
        or result.get("native_negative_model")
        or result.get("gligen_model")
        or result.get("hypernetwork_model")
    ):
        raise ValueError(
            "Native tensor files require ordinary text-to-image without staged, image or patch inputs."
        )
    for key in ("native_export_latent", "native_export_conditioning"):
        if type(result.get(key, False)) is not bool:
            raise ValueError("Native tensor export controls must be booleans.")
    denoise = result.get("native_latent_denoise", 0.5)
    if type(denoise) not in (int, float) or not math.isfinite(denoise) or not 0 < denoise <= 1:
        raise ValueError(
            "Native latent denoise must be a finite number greater than zero and at most one."
        )
    result["native_latent_denoise"] = float(denoise)
    if (
        type(result.get("image_number", 1)) is not int
        or not 1 <= result.get("image_number", 1) <= 4
    ):
        raise ValueError("Native tensor jobs support one through four image repetitions.")
    parsed, total = [], 0
    for field, role in zip(INPUTS, ("latent", "positive", "negative")):
        value = result.get(field)
        if value is None:
            continue
        raw = _raw(value)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Native tensor inputs together exceed 64 MiB.")
        header, context = tensor_header(
            raw, "latent" if role == "latent" else "conditioning", family
        )
        declared = result.get("native_tensor_family")
        if declared not in (None, "auto", family) or not context and declared != family:
            raise ValueError(
                "Declare the original model family for native files without saved RF context."
            )
        parsed.append((field, role, raw, header, context))
    directory = _directory(manager)
    inputs = []
    for field, role, raw, header, context in parsed:
        _check_job(result)
        destination = directory / (role + (".latent" if role == "latent" else ".safetensors"))
        with destination.open("xb") as target:
            target.write(raw)
        primary = header["latent_tensor" if role == "latent" else "conditioning"]
        inputs.append(
            TensorInput(
                role,
                destination,
                tuple(file_signature(destination)),
                hashlib.sha256(raw).hexdigest(),
                primary["dtype"],
                tuple(primary["shape"]),
                tuple(sorted(context.items())),
                1 / 0.18215
                if role == "latent" and "latent_format_version_0" not in header
                else 1.0,
            )
        )
        result.pop(field, None)
        if role == "latent":
            width, height = primary["shape"][-1] * 8, primary["shape"][-2] * 8
            requested = result.pop("_native_tensor_requested_size", "auto")
            if requested not in (None, "auto", f"{width}x{height}"):
                raise ValueError(
                    "Requested size must match the imported latent's complete native image."
                )
            from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY

            result.pop(SIZE_KEY, None)
            result.pop("size_plan", None)
            result.update(
                aspect_ratios_selection="Custom...",
                custom_width=width,
                custom_height=height,
                width=width,
                height=height,
                size_mode="legacy_base",
            )
    result[KEY] = TensorPlan(
        family,
        directory,
        tuple(inputs),
        result.get("native_export_latent", False),
        result.get("native_export_conditioning", False),
    )
    result[EXPORT_KEY] = []
    result["native_tensor_source"] = {
        "family": family,
        "inputs": [
            {
                "role": item.role,
                "sha256": item.sha256,
                "dtype": item.dtype,
                "shape": list(item.shape),
                **(
                    {"working_dtype": "float32", "native_legacy_scale": item.legacy_scale}
                    if item.role == "latent"
                    else {}
                ),
            }
            for item in inputs
        ],
        "required_context": "original model, matching VAE and matching text encoders; files do not supply these companions",
    }
    return result


def _scoped(function, receiver, paths, *args, **kwargs):
    """Bind the pinned native body to request-local paths without changing globals."""
    function = getattr(function, "__func__", function)
    scope = dict(function.__globals__, folder_paths=paths)
    bound = types.FunctionType(
        function.__code__, scope, function.__name__, function.__defaults__, function.__closure__
    )
    return bound(receiver, *args, **kwargs)


def _check_job(data):
    from ruinedfooocus.app.resource_lifecycle import current_job

    job = data.get("_job") or data.get("_api_job") or current_job()
    if callable(getattr(job, "check", None)):
        job.check()
    elif getattr(job, "cancelled", None) is not None and job.cancelled.is_set():
        raise CancelledError("Native tensor request was cancelled.")
    return job


def _runtime(owner, data):
    _check_job(data)
    plan = data.get(KEY)
    if not isinstance(plan, TensorPlan):
        raise ValueError("Native tensor inputs/exports were not captured for this request.")
    model = owner.xl_base_patched.unet.model
    family = type(model.model_config).__name__
    vae = owner.xl_base_patched.vae
    if (
        family != plan.family
        or family not in FAMILIES
        or vae.latent_channels != FAMILIES[family][0]
        or vae.latent_dim != 2
        or vae.spacial_compression_decode() != 8
        or getattr(model.model_config, "unet_config", {}).get(
            "in_channels", MODEL_INPUT_CHANNELS.get(family)
        )
        != MODEL_INPUT_CHANNELS.get(family)
    ):
        raise ValueError(
            "The loaded model and codec do not match the captured native tensor family."
        )
    context = {
        "family": family,
        "model_sha256": data.get("base_model_hash") or "",
        "vae_type": type(vae.first_stage_model).__name__,
        "channels": vae.latent_channels,
        "spatial_scale": 8,
        "component_signature": hashlib.sha256(
            json.dumps(
                [
                    [Path(stamp[0]).name, *stamp[1:]]
                    for stamp in getattr(owner, "component_file_signatures", ()) or ()
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "lora_signature": hashlib.sha256(
            json.dumps(
                [
                    [[Path(stamp[0]).name, *stamp[1:]], weight]
                    for stamp, weight in getattr(owner, "xl_base_patched_hash", ()) or ()
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    for item in plan.inputs:
        saved = dict(item.context)
        for key in (
            "vae_type",
            "channels",
            "spatial_scale",
            "component_signature",
            "lora_signature",
        ):
            if key in saved and saved[key] != context[key]:
                raise ValueError("The input tensor requires its original matching codec context.")
        if saved.get("model_sha256") and saved["model_sha256"] != context["model_sha256"]:
            raise ValueError("The input tensor was exported with a different model identity.")
    return plan, context


@contextmanager
def _input_file(item, data):
    from ruinedfooocus.core.preparation import preparation_service

    job = _check_job(data)
    with preparation_service.artifact_lease(
        item.path, cancelled=getattr(job, "cancelled", None)
    ) as path:
        if (
            tuple(file_signature(path)) != item.signature
            or hashlib.sha256(path.read_bytes()).hexdigest() != item.sha256
        ):
            raise ValueError("The captured native tensor changed before compute.")
        yield path
        _check_job(data)
        if tuple(file_signature(path)) != item.signature:
            raise ValueError("The native tensor changed while its reader was active.")


def conditioning_inputs(owner, data):
    if data[KEY].context:
        # Workflow files contain final numeric conditioning, including reference
        # tensors. Replace it after native source/mask attachment exactly once.
        return {}
    plan, _ = _runtime(owner, data)
    from comfy_extras.nodes_cond import ConditioningLoader

    values = {}
    for item in plan.inputs:
        if item.role == "latent":
            continue
        with _input_file(item, data) as path:
            paths = types.SimpleNamespace(get_full_path_or_raise=lambda category, name: str(path))
            values[item.role] = _scoped(
                ConditioningLoader.execute, ConditioningLoader, paths, path.name
            )[0]
            import torch

            if any(
                not torch.isfinite(tensor).all()
                for entry in values[item.role]
                for tensor in (
                    entry[0],
                    *[value for value in entry[1].values() if isinstance(value, torch.Tensor)],
                )
            ):
                raise ValueError("Native conditioning contains non-finite values.")
    return values


def latent_input(owner, data):
    plan = data[KEY]
    if plan.context:
        item = next((item for item in plan.inputs if item.role == "latent"), None)
        return load_latent_item(owner, data, item) if item is not None else None
    plan, _ = _runtime(owner, data)
    item = next((item for item in plan.inputs if item.role == "latent"), None)
    if item is None:
        return None
    from nodes import LoadLatent

    with _input_file(item, data) as path:
        paths = types.SimpleNamespace(get_annotated_filepath=lambda name: str(path))
        latent = _scoped(LoadLatent.load, LoadLatent(), paths, path.name)[0]
        import torch

        if not torch.isfinite(latent["samples"]).all():
            raise ValueError("Native latent input contains non-finite values.")
        if (data.get("width"), data.get("height")) != (
            latent["samples"].shape[-1] * 8,
            latent["samples"].shape[-2] * 8,
        ):
            raise ValueError(
                "The captured native latent geometry changed. Capture the request again."
            )
        return latent


def _context_file(path, context):
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + size])
    header.setdefault("__metadata__", {})["rf_tensor_context"] = json.dumps(context)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as target:
        target.write(len(encoded).to_bytes(8, "little") + encoded + raw[8 + size :])
    temporary.replace(path)


def _export_bounds(value, role, family):
    import torch

    if role == "latent":
        if not isinstance(value, dict) or not isinstance(value.get("samples"), torch.Tensor):
            raise ValueError("Export a native sampled latent dictionary.")
        tensors = {"latent_tensor": value["samples"]}
    else:
        if not isinstance(value, (list, tuple)) or len(value) != 1:
            raise ValueError(
                "Native conditioning export requires one uncombined conditioning entry."
            )
        tensor, options = value[0]
        if not isinstance(options, dict) or set(options) - {*SCALARS, "pooled_output"}:
            raise ValueError(
                "Only ordinary text conditioning can be exported; image, GLIGEN and hook context need their own serializer contract."
            )
        tensors = {"conditioning": tensor}
        if any(
            type(option) not in (bool, int, float)
            or not -1e9 <= option <= 1e9
            or not math.isfinite(option)
            for name, option in options.items()
            if name != "pooled_output"
        ):
            raise ValueError("Native conditioning export requires finite ordinary scalar options.")
        if "pooled_output" in options:
            tensors["pooled_output"] = options["pooled_output"]
    if any(
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or tensor.ndim not in (2, 3, 4)
        or min(tensor.shape) < 1
        or tensor.shape[0] != 1
        for tensor in tensors.values()
    ):
        raise ValueError("Native tensor export needs one bounded floating point still sample.")
    if (
        sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        > MAX_BYTES - MAX_HEADER
    ):
        raise ValueError("The native tensor export exceeds its 32-MiB limit before copying.")
    primary = tensors["latent_tensor" if role == "latent" else "conditioning"]
    channels, width = FAMILIES[family]
    if role == "latent":
        if (
            primary.ndim != 4
            or primary.shape[1] != channels
            or max(primary.shape[-2:]) > 1024
            or math.prod(primary.shape[-2:]) * 64 > 16_777_216
        ):
            raise ValueError("The sampled latent does not match its bounded still family.")
    elif primary.ndim != 3 or primary.shape[-1] != width or primary.shape[1] > 16384:
        raise ValueError("The text conditioning does not match its native encoder family.")
    if "pooled_output" in tensors:
        if tuple(tensors["pooled_output"].shape) != (1, POOLED[family]):
            raise ValueError("The pooled conditioning does not match its native encoder family.")
    elif role != "latent" and family in ("SDXL", "SD3", "Flux", "FluxSchnell"):
        raise ValueError("This family requires its matching pooled conditioning for export.")
    if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise ValueError("Native tensor export contains non-finite values.")


def export_tensors(owner, data, latent, positive, negative, *, stage_b=None, frame=None):
    """Caller guards this function; Off never captures or clones any tensors."""
    if data[KEY].context:
        from .native_tensor_context import export

        return export(owner, data, latent, positive, negative, stage_b=stage_b, frame=frame)
    plan, context = _runtime(owner, data)
    from nodes import SaveLatent
    from comfy_extras.nodes_cond import SaveConditioning

    outputs = []
    requested = ([("latent", latent)] if plan.export_latent else []) + (
        [("positive", positive), ("negative", negative)] if plan.export_conditioning else []
    )
    generation_index = len(data.get(EXPORT_KEY, ())) // max(1, len(requested))
    for role, value in requested:
        _check_job(data)
        _export_bounds(value, role, plan.family)
        prefix = role + "-" + secrets.token_hex(12)
        paths = types.SimpleNamespace(
            get_save_image_path=lambda *args: (str(plan.directory), prefix, 0, "", prefix),
            get_output_directory=lambda: str(plan.directory),
        )
        if role == "latent":
            saver = SaveLatent.__new__(SaveLatent)
            saver.output_dir = str(plan.directory)
            _scoped(SaveLatent.save, saver, paths, value, filename_prefix=prefix)
        else:
            _scoped(SaveConditioning.execute, SaveConditioning, paths, value, prefix)
        path = plan.directory / (
            prefix + "_00000_" + (".latent" if role == "latent" else ".safetensors")
        )
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError("The native tensor export exceeds its 32-MiB limit.")
        _context_file(path, context)
        tensor_header(
            path.read_bytes(), "latent" if role == "latent" else "conditioning", plan.family
        )
        outputs.append(
            TensorExport(
                str(path),
                role,
                tuple(file_signature(path)),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                plan.family,
                data.get("seed", 0),
                generation_index,
            )
        )
    _check_job(data)
    data.setdefault(EXPORT_KEY, []).extend(outputs)
    return outputs


def apply_conditioning_inputs(owner, data, positive, negative, *, frame=None):
    plan = data.get(KEY)
    if not isinstance(plan, TensorPlan) or not plan.context:
        return positive, negative
    from .native_tensor_context import load_conditioning, validate_frame

    data["_native_tensor_frame"] = frame
    validate_frame(data, frame)
    values = {
        item.role: load_conditioning(owner, data, item)
        for item in plan.inputs
        if item.role in ("positive", "negative")
    }
    return values.get("positive", positive), values.get("negative", negative)


def load_latent_item(owner, data, item):
    """Read one captured companion through the native reader and its context gate."""
    if dict(item.context).get("version") == 2:
        from .native_tensor_context import load_latent

        return load_latent(owner, data, item)
    from dataclasses import replace

    plan = data[KEY]
    return latent_input(owner, dict(data, **{KEY: replace(plan, inputs=(item,), context="")}))


def capture_latent_companion(raw, data, manager, models):
    """Composition receives bytes and a captured request, never an arbitrary path."""
    source = dict(
        data, native_latent_input=raw, native_positive_input=None, native_negative_input=None
    )
    captured = capture_request(source, manager, models)
    return next(item for item in captured[KEY].inputs if item.role == "latent")


def assert_composable(primary, secondary):
    """Native composition keeps four axes; pixel offsets use the declared codec."""
    from .native_tensor_context import canonical

    def identity(item):
        value = dict(item.context)
        value["layout"] = {
            key: child for key, child in value.get("layout", {}).items() if key != "composition"
        }
        return canonical(value)

    if (
        primary.dtype != secondary.dtype
        or primary.shape != secondary.shape
        or identity(primary) != identity(secondary)
    ):
        raise ValueError(
            "Latent composition requires matching original dtype, shape and model/codec/text context."
        )
    context = dict(primary.context)
    if (
        context.get("version") != 2
        or len(primary.shape) != 4
        or primary.shape[0] != 1
        or context.get("layout", {}).get("kind") != "native-still"
        or context["layout"].get("scale") not in (8, 16)
    ):
        raise ValueError(
            "Native composition requires locally receipted rank-four still tensors at codec scale eight or sixteen; layer/frame axes and paired stages need their own operation."
        )


def capture_composition_inputs(primary_raw, secondary_raw, manager, *, data=None):
    from .native_latent_composition_io import capture_composition_inputs as capture

    return capture(primary_raw, secondary_raw, manager, data=data)


def load_composition_latent(data, item):
    from .native_latent_composition_io import load_composition_latent as load

    return load(data, item)


def save_composed_latent(data, primary, latent, *, composition):
    from .native_latent_composition_io import save_composed_latent as save

    return save(data, primary, latent, composition=composition)


def sdxl_handoff_state(owner, data, latent, *, payload, export=False):
    """Resume or explicitly export the native noisy split, never a final latent."""
    from dataclasses import replace
    from .native_tensor_context import export as workflow_export

    saved = data.get("_native_tensor_handoff")
    if saved is not None and saved != payload:
        raise ValueError(
            "The saved SDXL split requires its original schedule, sources, prompts, noise mask and seed."
        )
    if export and data.get("native_export_handoff"):
        plan = data.get(KEY)
        if not isinstance(plan, TensorPlan) or not plan.context or not plan.export_handoff:
            raise ValueError("Capture the native noisy handoff export before sampling.")
        selected = dict(data, **{KEY: replace(plan, export_latent=True, export_conditioning=False)})
        workflow_export(
            owner,
            selected,
            latent,
            None,
            None,
            handoff=payload,
            frame=data.get("_native_tensor_frame"),
        )
    return saved is not None
