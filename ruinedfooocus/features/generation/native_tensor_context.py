"""Bounded, locally receipted native workflow tensors; no executable graph state."""

import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import types

from ruinedfooocus.features.models.model_identity import file_signature

MARKER = "rf_tensor_workflow"
VERSION = 2
MAX_ENTRIES = 64
MAX_TENSORS = 256
SAVED_FAMILIES = {
    "SD15",
    "SD20",
    "SDXL",
    "SDXLRefiner",
    "SD3",
    "Flux",
    "FluxSchnell",
    "QwenImage",
    "QwenImage21",
    "Flux2",
    "HunyuanImage21Refiner",
    "Stable_Cascade_C",
}
FLOATS = {"F16": 2, "BF16": 2, "F32": 4}
DTYPES = FLOATS | {"I64": 8, "I32": 4, "BOOL": 1}
OBJECT_OPTIONS = {"hooks", "gligen", "control", "model_conds"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def raw_header(raw):
    from .native_tensor_io import MAX_BYTES, MAX_HEADER, _object

    if not isinstance(raw, bytes) or not 10 <= len(raw) <= MAX_BYTES:
        raise ValueError("Native workflow files must contain at most 32 MiB.")
    size = int.from_bytes(raw[:8], "little")
    if not 2 <= size <= min(MAX_HEADER, len(raw) - 8):
        raise ValueError("Invalid bounded native workflow header.")
    try:
        header = json.loads(raw[8 : 8 + size], object_pairs_hook=_object)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise ValueError("Invalid native workflow JSON header.") from error
    if not isinstance(header, dict):
        raise ValueError("A native workflow file requires a tensor dictionary.")
    return header, size


def has_context(raw):
    header, _ = raw_header(raw)
    metadata = header.get("__metadata__", {})
    return isinstance(metadata, dict) and MARKER in metadata


def header(raw, role, family):
    from .native_tensor_io import _object

    if not isinstance(family, str) or family not in SAVED_FAMILIES:
        raise ValueError("Invalid saved native workflow family.")
    values, size = raw_header(raw)
    metadata = values.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or len(k) > 128 or not isinstance(v, str) or len(v) > 65536
        for k, v in metadata.items()
    ):
        raise ValueError("Invalid native workflow string metadata.")
    try:
        context = json.loads(metadata[MARKER], object_pairs_hook=_object)
    except (KeyError, ValueError, RecursionError) as error:
        raise ValueError("Missing bounded saved workflow context.") from error
    if (
        not isinstance(context, dict)
        or set(context) != {"version", "family", "capture", "runtime", "layout", "entries"}
        or type(context["version"]) is not int
        or context["version"] != VERSION
        or context["family"] != family
        or family not in SAVED_FAMILIES
        or not all(isinstance(context[k], dict) for k in ("capture", "runtime", "layout"))
        or not isinstance(context["entries"], list)
        or len(context["entries"]) > MAX_ENTRIES
    ):
        raise ValueError("The saved workflow family/context does not match this request.")
    rows = {k: v for k, v in values.items() if k != "__metadata__"}
    if not 1 <= len(rows) <= MAX_TENSORS:
        raise ValueError("Too many native workflow tensors.")
    cursor, ranges = 0, []
    for name, row in rows.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.]{1,128}", name):
            raise ValueError("Invalid native workflow tensor name.")
        if not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Invalid native workflow tensor descriptor.")
        shape, dtype, offsets = row["shape"], row["dtype"], row["data_offsets"]
        if (
            not isinstance(dtype, str)
            or dtype not in DTYPES
            or not isinstance(shape, list)
            or len(shape) > 5
            or any(type(n) is not int or not 0 <= n <= 32768 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or offsets[0] < 0
            or offsets[1] > len(raw) - 8 - size
            or offsets[1] - offsets[0] != math.prod(shape) * DTYPES[dtype]
        ):
            raise ValueError("Native workflow dtype/rank/shape/byte bounds disagree.")
        if name == "latent_format_version_0":
            if shape != [0] or dtype != "F32":
                raise ValueError("Invalid native latent version marker.")
        elif not shape or min(shape) < 1:
            raise ValueError("Workflow tensors must be nonempty.")
        if math.prod(shape):
            ranges.append(offsets)
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Workflow tensor byte ranges must be contiguous and nonoverlapping.")
        cursor = end
    if cursor != len(raw) - 8 - size:
        raise ValueError("Workflow tensor file contains unreferenced bytes.")
    layout = context["layout"]
    if role == "latent":
        primary = rows.get("latent_tensor", {})
        if (
            primary.get("dtype") not in FLOATS
            or primary.get("shape") != layout.get("shape")
            or "latent_format_version_0" not in rows
            or set(rows)
            - {"latent_tensor", "latent_format_version_0", "rf_noise_mask", "rf_stage_b"}
        ):
            raise ValueError("A saved native workflow latent requires its exact marked layout.")
        validate_layout(family, layout, rows)
    else:
        entries = context["entries"]
        if not entries or not isinstance(entries[0], dict) or entries[0].get("prefix") != "":
            raise ValueError("Saved workflow conditioning needs its original entries.")
        allowed = set()
        for index, entry in enumerate(entries):
            prefix = "" if index == 0 else f"rf_entry_{index}."
            if (
                not isinstance(entry, dict)
                or set(entry) != {"prefix", "options", "names"}
                or entry["prefix"] != prefix
            ):
                raise ValueError("Invalid saved conditioning entry context.")
            if not isinstance(entry["options"], dict) or not isinstance(entry["names"], list):
                raise ValueError("Invalid saved conditioning options.")
            if not 1 <= len(entry["names"]) <= MAX_TENSORS or len(entry["options"]) > MAX_TENSORS:
                raise ValueError("Saved conditioning names/options exceed their bounded count.")
            for name, value in entry["options"].items():
                safe_option(name, value)
            primary = rows.get(prefix + "conditioning", {})
            shape = primary.get("shape", [])
            if (
                primary.get("dtype") not in FLOATS
                or len(shape) != 3
                or shape[0] != 1
                or shape[1] > 16384
            ):
                raise ValueError("Saved text conditioning requires one bounded native text sample.")
            for name in entry["names"]:
                if (
                    not isinstance(name, str)
                    or not 1 <= len(name) <= 128
                    or name in OBJECT_OPTIONS
                    or not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[0-9]+)?", name)
                ):
                    raise ValueError("Unknown or executable native conditioning state.")
                key = prefix + name
                if key not in rows or key in allowed:
                    raise ValueError("Unknown or duplicate saved conditioning tensor name.")
                allowed.add(key)
        if set(rows) != allowed:
            raise ValueError("Unreferenced saved conditioning tensors.")
    return values, context


def validate_layout(family, layout, rows):
    shape = layout.get("shape", [])
    width, height = layout.get("width"), layout.get("height")
    if (
        len(shape) not in (4, 5)
        or shape[0] != 1
        or any(type(side) is not int or not 1 <= side <= 8192 for side in (width, height))
        or width * height > 16_777_216
    ):
        raise ValueError("Invalid saved still workflow geometry.")
    kind = layout.get("kind")
    if kind == "qwen-layers":
        layers = layout.get("layers")
        if (
            family != "QwenImage"
            or type(layers) is not int
            or not 1 <= layers <= 16
            or shape != [1, 16, layers + 1, height // 8, width // 8]
            or width % 16
            or height % 16
        ):
            raise ValueError(
                "Layered tensors must retain reconstruction plus every native layer slot."
            )
    elif kind == "cascade-prior":
        companion = rows.get("rf_stage_b", {})
        if (
            family != "Stable_Cascade_C"
            or len(shape) != 4
            or shape[1] != 16
            or companion.get("shape") != [1, 4, height // 4, width // 4]
            or companion.get("dtype") not in FLOATS
        ):
            raise ValueError("Cascade tensors require the original paired C16/B4 stage grids.")
    elif kind == "hunyuan-refiner":
        if family != "HunyuanImage21Refiner" or shape != [1, 32, 1, height // 16, width // 16]:
            raise ValueError("Hunyuan still refiner tensors require one 32-channel native frame.")
    elif kind in ("native-still", "sdxl-handoff"):
        scale, channels = layout.get("scale"), layout.get("channels")
        if (
            type(scale) is not int
            or scale not in (8, 16)
            or type(channels) is not int
            or channels not in (4, 16, 64)
            or shape[-2:] != [height // scale, width // scale]
            or shape[1] != channels
            or len(shape) == 5
            and shape[2] != 1
        ):
            raise ValueError("The saved native latent rank/channel/spatial layout changed.")
        if kind == "sdxl-handoff" and (
            family != "SDXL" or not isinstance(layout.get("handoff"), dict)
        ):
            raise ValueError("A noisy SDXL split requires its full original stage/sigma context.")
    else:
        raise ValueError("This native workflow stage has no saved tensor consumer.")
    if "rf_noise_mask" in rows:
        mask = rows["rf_noise_mask"]
        if (
            len(mask["shape"]) not in (3, 4, 5)
            or mask["shape"][0] != 1
            or mask["dtype"] not in FLOATS
        ):
            raise ValueError("The native noise mask requires one bounded floating still mask.")


def safe_option(name, value):
    if not isinstance(name, str) or len(name) > 128 or name in OBJECT_OPTIONS:
        raise ValueError(
            "Native files cannot serialize executable GLIGEN, ControlNet or hook objects."
        )
    if type(value) in (bool, int, float):
        if not math.isfinite(value) or not -1e9 <= value <= 1e9:
            raise ValueError("Conditioning options must be finite and bounded.")
    elif not isinstance(value, str) or len(value) > 1024:
        raise ValueError("Native conditioning scalar options must use bounded native primitives.")


def _stamp(value):
    return _signature_stamp(file_signature(value))


def _signature_stamp(stamp):
    """Preserve the loaded/captured bytes' stamp while hiding its local path."""
    from ruinedfooocus.core.cache_contract import canonical_path_key

    return [
        Path(stamp[0]).name,
        stamp[1],
        *(str(v) for v in stamp[2:]),
        hashlib.sha256(canonical_path_key(stamp[0]).encode()).hexdigest(),
    ]


def _pixels(value):
    from PIL import Image, ImageOps

    if value is None:
        return None
    if isinstance(value, dict):
        return {
            key: _pixels(child)
            for key, child in value.items()
            if key in ("background", "layers", "composite")
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            raise ValueError("Saved workflow context supports at most sixteen source images.")
        return [_pixels(child) for child in value]
    if not isinstance(value, Image.Image):
        # These are existing request-owned Main/editor uploads, not tensor paths.
        with Image.open(value) as image:
            value = ImageOps.exif_transpose(image).convert("RGBA")
    else:
        value = ImageOps.exif_transpose(value).convert("RGBA")
    if value.width * value.height > 16_777_216:
        raise ValueError("Saved workflow source exceeds sixteen megapixels.")
    return dict(size=list(value.size), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def capture_context(data, models):
    fields = (
        "controlnet",
        "inpaint_toggle",
        "image_resize_mode",
        "native_edit_recipe",
        "krea_style_lora",
        "krea_style_strength",
        "qwen_layer_count",
        "cascade_stage_b",
        "cascade_compression",
        "cascade_b_steps",
        "cascade_b_cfg",
        "cascade_b_sampler",
        "cascade_b_scheduler",
        "sdxl_refiner_model",
        "sdxl_refiner_clip",
        "sdxl_refiner_switch_step",
        "sdxl_refiner_cfg",
        "refiner_noise_augmentation",
        "gligen_model",
        "gligen_layout",
        "hypernetwork_model",
        "hypernetwork_strength",
    )
    settings = {
        key: data[key] for key in fields if data.get(key) is not None and data.get(key) != ""
    }
    settings["inpaint_toggle"] = bool(data.get("inpaint_toggle"))
    for key, value in data.items():
        if hooks_active(data) and key.startswith("lora_hook_") and value is not None:
            settings[key] = value
    hook_plan = data.get("_lora_hook_plan")
    if hooks_active(data) and hook_plan is not None:
        settings["resolved_hook_sources"] = [
            dict(
                settings=source.settings(),
                kind=getattr(getattr(source, "spec", None), "kind", "lora"),
                signature=_signature_stamp(source.artifact.signature),
            )
            for source in getattr(hook_plan, "sources", ()) or (hook_plan,)
        ]
    sources = source_context(data)
    checkpoints = {}
    for field in ("base_model_name", "cascade_stage_b", "sdxl_refiner_model"):
        if data.get(field):
            path = models.get_file("checkpoints", data[field])
            if path is None:
                raise ValueError("Install the original saved workflow checkpoint companions.")
            checkpoints[field] = _stamp(path)
    return dict(settings=settings, sources=sources, checkpoints=checkpoints)


def source_context(data):
    sources = {
        key: _pixels(data.get(key))
        for key in ("input_image", "reference_images")
        if data.get(key) is not None
    }
    if data.get("inpaint_toggle"):
        sources.update(main=_pixels(data.get("main_view")), mask=_pixels(data.get("inpaint_view")))
    layered = data.get("_native_layered_plan")
    if layered is not None:
        sources["layered"] = layered.metadata()
    return sources


def needs_context(data, details, raws):
    from .native_tensor_io import FAMILIES

    return (
        data.get("native_tensor_workflow") is True
        or hooks_active(data)
        or details.get("native_class") not in FAMILIES
        or details.get("native_config", {}).get("in_channels") in (8, 9)
        or any(
            data.get(key)
            for key in (
                "input_image",
                "inpaint_toggle",
                "reference_images",
                "cascade_stage_b",
                "sdxl_refiner_model",
                "_native_layered_plan",
                "hypernetwork_model",
                "gligen_model",
            )
        )
        or any(has_context(raw) for raw in raws)
    )


def hooks_active(data):
    return str(data.get("lora_hook_mode", "off")).casefold() not in ("", "off")


def model_hooks_active(data):
    if not hooks_active(data):
        return False
    plan = data.get("_lora_hook_plan")
    if plan is None:
        return True
    sources = getattr(plan, "sources", ()) or (plan,)
    return any(source.settings().get("lora_hook_model_strength", 1) != 0 for source in sources)


def capture(data, manager, models, family):
    from .native_tensor_io import (
        TensorInput,
        TensorPlan,
        KEY,
        EXPORT_KEY,
        INPUTS,
        MAX_TOTAL_BYTES,
        _raw,
        _directory,
        _check_job,
    )

    if family not in SAVED_FAMILIES or str(data.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("This native still workflow has no saved tensor consumer.")
    if data.get("native_tensor_family") not in (None, "auto", family):
        raise ValueError("The declared tensor family differs from the selected native workflow.")
    if data.get("reference_adapter") not in (None, "", "none") or data.get("native_negative_model"):
        raise ValueError(
            "IP-Adapter model patches and dual-model sampler objects need separate reconstruction contracts."
        )
    if ((data.get("controlnet") or {}).get("type") or "").lower() not in (
        "",
        "none",
        "edit",
        "img2img",
    ):
        raise ValueError("Native conditioning files cannot serialize live ControlNet objects.")
    if (data.get("gligen_model") or model_hooks_active(data)) and (
        data.get("native_export_conditioning")
        or any(data.get(key) is not None for key in INPUTS[1:])
    ):
        raise ValueError(
            "Native files cannot serialize executable GLIGEN or hook conditioning; latent-only capture requires their original rebuilt request."
        )
    for key in (
        "native_export_latent",
        "native_export_conditioning",
        "native_export_handoff",
        "native_tensor_workflow",
    ):
        if type(data.get(key, False)) is not bool:
            raise ValueError("Native workflow controls must be booleans.")
    count = data.get("image_number", 1)
    denoise = data.get("native_latent_denoise", 0.5)
    if (
        type(count) is not int
        or not 1 <= count <= 4
        or type(denoise) not in (int, float)
        or not math.isfinite(denoise)
        or not 0 < denoise <= 1
    ):
        raise ValueError(
            "Native workflow jobs need one through four repetitions and denoise greater than zero through one."
        )
    request_context = capture_context(data, models)
    parsed, total, geometry = [], 0, None
    for field, role in zip(INPUTS, ("latent", "positive", "negative")):
        if data.get(field) is None:
            continue
        raw = _raw(data[field])
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Native workflow inputs together exceed 64 MiB.")
        values, context = header(raw, "latent" if role == "latent" else "conditioning", family)
        if canonical(context["capture"]) != canonical(request_context):
            raise ValueError(
                "Reattach the original saved workflow source and model companions with matching settings."
            )
        if geometry is not None and context["layout"] != geometry:
            raise ValueError("Native input slots must share the exact original workflow geometry.")
        geometry = context["layout"]
        parsed.append((field, role, raw, values, context))
    directory = _directory(manager)
    inputs = []
    result = dict(data, native_tensor_workflow=True, native_latent_denoise=float(denoise))
    for field, role, raw, values, context in parsed:
        _check_job(data)
        verify_receipt(raw, context, directory)
        destination = directory / (role + (".latent" if role == "latent" else ".safetensors"))
        destination.write_bytes(raw)
        primary = values["latent_tensor" if role == "latent" else "conditioning"]
        inputs.append(
            TensorInput(
                role,
                destination,
                tuple(file_signature(destination)),
                hashlib.sha256(raw).hexdigest(),
                primary["dtype"],
                tuple(primary["shape"]),
                tuple(sorted(context.items())),
            )
        )
        result.pop(field, None)
    if geometry is not None:
        from .size_plan import KEY as SIZE_KEY, SizePlan

        width, height = geometry["width"], geometry["height"]
        size = geometry.get("size_plan")
        plan = SizePlan.from_metadata(size) if size is not None else None
        expected = plan.target if plan is not None else (width, height)
        requested = result.pop("_native_tensor_requested_size", "auto")
        if requested not in (None, "auto", f"{expected[0]}x{expected[1]}"):
            raise ValueError(
                "Requested size must match the saved workflow's complete output frame."
            )
        if (
            geometry["kind"] == "qwen-layers"
            and result.get("qwen_layer_count") != geometry["layers"]
        ):
            raise ValueError(
                "Choose the original saved layer count, including its native reconstruction slot."
            )
        if geometry["kind"] == "sdxl-handoff":
            if count != 1:
                raise ValueError("Resume one noisy SDXL split with its original seed and schedule.")
            result["native_latent_denoise"] = geometry["handoff"]["denoise"]
        result.update(
            width=width,
            height=height,
            custom_width=width,
            custom_height=height,
            aspect_ratios_selection="Custom...",
            size_mode="legacy_base",
        )
        if plan is not None:
            result[SIZE_KEY], result["size_plan"] = plan, plan.metadata()
        else:
            result.pop(SIZE_KEY, None)
            result.pop("size_plan", None)
    result[KEY] = TensorPlan(
        family,
        directory,
        tuple(inputs),
        result.get("native_export_latent", False),
        result.get("native_export_conditioning", False),
        canonical(request_context),
        result.get("native_export_handoff", False),
    )
    result[EXPORT_KEY] = []
    result["native_tensor_source"] = dict(
        family=family,
        workflow=True,
        inputs=[
            dict(role=item.role, sha256=item.sha256, dtype=item.dtype, shape=list(item.shape))
            for item in inputs
        ],
        required_context="original locally receipted workflow, model, codecs, text encoders and source uploads",
    )
    return result


def receipt_root(directory):
    cache = directory.parent.parent.resolve()
    target = cache / "native-tensor-contexts"
    if not target.resolve().is_relative_to(cache):
        raise ValueError("Saved tensor provenance must remain inside the configured cache.")
    return target


def verify_receipt(raw, context, directory):
    key = hashlib.sha256(raw).hexdigest()
    path = receipt_root(directory) / (key + ".json")
    if (
        not path.resolve().is_relative_to(receipt_root(directory).resolve())
        or not path.is_file()
        or path.stat().st_size > 128 * 1024
        or path.read_text(encoding="utf-8") != canonical(context)
    ):
        raise ValueError(
            "This complex tensor requires its original local saved-context receipt; external files lack adequate workflow provenance."
        )


def runtime_context(owner, data):
    plan = data["_native_tensor_plan"]
    vae = owner.xl_base_patched.vae
    model = owner.xl_base_patched.unet.model
    if type(model.model_config).__name__ != plan.family:
        raise ValueError("The loaded workflow model differs from its captured native family.")
    components = [
        _signature_stamp(stamp) for stamp in getattr(owner, "component_file_signatures", ()) or ()
    ]
    loras = [
        [_signature_stamp(stamp), weight]
        for stamp, weight in getattr(owner, "xl_base_patched_hash", ()) or ()
    ]
    assets = [
        [item.role, Path(item.name).name, _stamp(item.request.destination)]
        for item in getattr(data.get("_native_image_assets"), "artifacts", ())
    ]
    config = getattr(model.model_config, "unet_config", {})
    return dict(
        model_sha256=data.get("base_model_hash") or "",
        codec=type(vae.first_stage_model).__name__,
        codec_channels=vae.latent_channels,
        codec_rank=vae.latent_dim,
        scale=vae.spacial_compression_decode(),
        text_encoder=type(getattr(owner.xl_base_patched.clip, "cond_stage_model", None)).__name__
        if hasattr(owner.xl_base_patched, "clip")
        else "unknown",
        components=hashlib.sha256(canonical(components).encode()).hexdigest(),
        loras=hashlib.sha256(canonical(loras).encode()).hexdigest(),
        unet_config=hashlib.sha256(
            canonical(
                {
                    key: config[key]
                    for key in (
                        "in_channels",
                        "image_model",
                        "context_in_dim",
                        "hidden_size",
                        "patch_size",
                    )
                    if key in config
                }
            ).encode()
        ).hexdigest(),
        assets=hashlib.sha256(canonical(assets).encode()).hexdigest(),
        sampling={
            key: data[key]
            for key in ("steps", "cfg", "clip_skip", "sampler_name", "scheduler")
            if key in data
        },
    )


def validate_runtime(owner, data, item=None):
    from .native_tensor_io import _check_job

    _check_job(data)
    plan = data["_native_tensor_plan"]
    if source_context(data) != json.loads(plan.context)["sources"]:
        raise ValueError("The captured source image or mask changed before native tensor use.")
    current = runtime_context(owner, data)
    inputs = (item,) if item is not None else plan.inputs
    for source in inputs:
        saved = dict(source.context)
        if saved.get("version") == VERSION and (
            saved["runtime"] != current or canonical(saved["capture"]) != plan.context
        ):
            raise ValueError(
                "The saved tensor requires the exact original model, codec, text encoders, source and workflow selections."
            )
    return current


def layout(owner, data, latent, extras, *, frame=None):
    vae = owner.xl_base_patched.vae
    shape = list(latent["samples"].shape)
    result = dict(
        kind="native-still",
        shape=shape,
        width=data["width"],
        height=data["height"],
        scale=vae.spacial_compression_decode(),
        channels=shape[1],
        size_plan=data.get("size_plan"),
        frame=frame,
        stage_context={
            key: data[key]
            for key in ("native_cascade", "native_layered_conditioning", "native_source_scale")
            if key in data
        },
    )
    plan = data.get("_native_layered_plan")
    if plan is not None:
        result.update(
            kind="qwen-layers",
            layers=plan.layers,
            output_kind="qwen-layers",
            output_count=plan.layers,
        )
    elif data["_native_tensor_plan"].family == "Stable_Cascade_C":
        result["kind"] = "cascade-prior"
        if "rf_stage_b" not in extras:
            raise ValueError("Export Cascade's original paired Stage B target grid with its prior.")
    elif data["_native_tensor_plan"].family == "HunyuanImage21Refiner":
        result["kind"] = "hunyuan-refiner"
    return result


def validate_frame(data, frame):
    for item in data["_native_tensor_plan"].inputs:
        saved = dict(item.context)
        geometry = saved.get("layout", {})
        if geometry.get("frame") != frame or (geometry.get("width"), geometry.get("height")) != (
            data.get("width"),
            data.get("height"),
        ):
            raise ValueError(
                "The saved tensor source/output frame differs from the current native sampler frame."
            )


def pack(parts, context):
    """Join native safe serializer payloads without tensor coercion or pickle."""
    from .native_tensor_io import MAX_BYTES, MAX_HEADER

    values, payload, offset = {}, [], 0
    for prefix, raw in parts:
        original, size = raw_header(raw)
        for name, row in original.items():
            if name == "__metadata__":
                continue
            start, end = row["data_offsets"]
            if prefix + name in values:
                raise ValueError("Duplicate saved workflow tensor names are not allowed.")
            chunk = raw[8 + size + start : 8 + size + end]
            values[prefix + name] = dict(row, data_offsets=[offset, offset + len(chunk)])
            payload.append(chunk)
            offset += len(chunk)
    values["__metadata__"] = {MARKER: canonical(context)}
    encoded = canonical(values).encode()
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > MAX_HEADER or offset + len(encoded) + 8 > MAX_BYTES:
        raise ValueError("The complete saved workflow exceeds its header/file limit.")
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(payload)


def split_condition(raw, entry):
    values, size = raw_header(raw)
    rows, payload, offset = {}, [], 0
    for name in entry["names"]:
        row = values[entry["prefix"] + name]
        start, end = row["data_offsets"]
        chunk = raw[8 + size + start : 8 + size + end]
        rows[name] = dict(row, data_offsets=[offset, offset + len(chunk)])
        payload.append(chunk)
        offset += len(chunk)
    rows["__metadata__"] = {"conditioning_options": canonical(entry["options"])}
    encoded = canonical(rows).encode()
    encoded += b" " * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(payload)


def load_conditioning(owner, data, item):
    from comfy_extras.nodes_cond import ConditioningLoader
    from .native_tensor_io import _input_file, _scoped, _check_job

    validate_runtime(owner, data, item)
    with _input_file(item, data) as path:
        raw = path.read_bytes()
        _, context = header(raw, "conditioning", data["_native_tensor_plan"].family)
        result = []
        for index, entry in enumerate(context["entries"]):
            _check_job(data)
            local = path.with_name(path.stem + f"-entry-{index}.safetensors")
            local.write_bytes(split_condition(raw, entry))
            paths = types.SimpleNamespace(get_full_path_or_raise=lambda *_: str(local))
            result.extend(
                _scoped(ConditioningLoader.execute, ConditioningLoader, paths, local.name)[0]
            )
        finite(result)
        return result


def finite(value):
    import torch

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Native workflow tensors contain non-finite values.")
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite(child)
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)


def load_latent(owner, data, item):
    import safetensors.torch
    from nodes import LoadLatent
    from .native_tensor_io import _input_file, _scoped

    validate_runtime(owner, data, item)
    with _input_file(item, data) as path:
        paths = types.SimpleNamespace(get_annotated_filepath=lambda *_: str(path))
        latent = _scoped(LoadLatent.load, LoadLatent(), paths, path.name)[0]
        extras = safetensors.torch.load_file(str(path), device="cpu")
        if "rf_noise_mask" in extras:
            latent["noise_mask"] = extras["rf_noise_mask"]
        if "rf_stage_b" in extras:
            latent["_rf_stage_b"] = {"samples": extras["rf_stage_b"]}
        saved = dict(item.context)
        if saved["layout"]["kind"] == "sdxl-handoff":
            data["_native_tensor_handoff"] = saved["layout"]["handoff"]
        finite(latent)
        return latent


def export(owner, data, latent, positive, negative, *, stage_b=None, handoff=None, frame=None):
    import safetensors.torch
    import torch
    from nodes import SaveLatent
    from comfy_extras.nodes_cond import SaveConditioning
    from .native_tensor_io import (
        TensorExport,
        EXPORT_KEY,
        _check_job,
        _scoped,
        MAX_BYTES,
        MAX_HEADER,
    )

    plan = data["_native_tensor_plan"]
    runtime = validate_runtime(owner, data)
    requested = ([("latent", latent)] if plan.export_latent else []) + (
        [("positive", positive), ("negative", negative)] if plan.export_conditioning else []
    )
    outputs = []
    index = data.get("_native_tensor_generation_index", 0)
    extras = {}
    if "noise_mask" in latent:
        extras["rf_noise_mask"] = latent["noise_mask"]
    if stage_b is not None:
        extras["rf_stage_b"] = stage_b["samples"]
    geometry = layout(owner, data, latent, extras, frame=frame)
    if handoff is not None:
        geometry.update(kind="sdxl-handoff", handoff=handoff)
    if plan.export_latent:
        descriptions = {"latent_tensor": {"shape": geometry["shape"]}}
        for name, tensor in extras.items():
            descriptions[name] = {
                "shape": list(tensor.shape),
                "dtype": "F32" if tensor.is_floating_point() else "invalid",
            }
        validate_layout(plan.family, geometry, descriptions)
    for role, value in requested:
        _check_job(data)
        entries = []
        tensors = [value["samples"], *extras.values()] if role == "latent" else []
        if role != "latent":
            if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_ENTRIES:
                raise ValueError("Export at most sixty-four safe native conditioning entries.")
            for tensor, options in value:
                if not isinstance(options, dict) or set(options) & OBJECT_OPTIONS:
                    raise ValueError(
                        "Native files cannot serialize executable GLIGEN, ControlNet or hook objects."
                    )
                tensors.append(tensor)
                for name, option in options.items():
                    if isinstance(option, torch.Tensor):
                        tensors.append(option)
                    elif (
                        isinstance(option, (list, tuple))
                        and option
                        and all(isinstance(child, torch.Tensor) for child in option)
                    ):
                        if len(option) > 16:
                            raise ValueError(
                                "At most sixteen native reference tensors are allowed."
                            )
                        tensors.extend(option)
                    elif option is not None:
                        safe_option(name, option)
        primaries = [value["samples"]] if role == "latent" else [entry[0] for entry in value]
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tensor.ndim != (len(geometry["shape"]) if role == "latent" else 3)
            or min(tensor.shape) < 1
            or tensor.shape[0] != 1
            or role != "latent"
            and tensor.shape[1] > 16384
            for tensor in primaries
        ):
            raise ValueError(
                "Native workflow primary tensors require their original floating batch-one layout before copying."
            )
        if (
            len(tensors) > MAX_TENSORS
            or any(
                not isinstance(t, torch.Tensor)
                or t.ndim > 5
                or t.dtype
                not in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.int64,
                    torch.int32,
                    torch.bool,
                )
                for t in tensors
            )
            or sum(t.numel() * t.element_size() for t in tensors) > MAX_BYTES - MAX_HEADER
        ):
            raise ValueError(
                "Saved workflow export exceeds its dtype/rank/byte bounds before copying."
            )
        finite(tensors)
        prefix = ("sdxl-handoff" if handoff is not None else role) + "-" + secrets.token_hex(12)
        parts = []
        values = [value] if role == "latent" else [[[tensor, options]] for tensor, options in value]
        for number, single in enumerate(values):
            _check_job(data)
            stem = prefix + f"-entry-{number}"
            paths = types.SimpleNamespace(
                get_save_image_path=lambda *_, stem=stem: (str(plan.directory), stem, 0, "", stem),
                get_output_directory=lambda: str(plan.directory),
            )
            suffix = ".latent" if role == "latent" else ".safetensors"
            local = plan.directory / (stem + "_00000_" + suffix)
            if role == "latent":
                saver = SaveLatent.__new__(SaveLatent)
                saver.output_dir = str(plan.directory)
                _scoped(SaveLatent.save, saver, paths, single, filename_prefix=stem)
            else:
                _scoped(SaveConditioning.execute, SaveConditioning, paths, single, stem)
            raw = local.read_bytes()
            native_header, _ = raw_header(raw)
            entry_prefix = "" if number == 0 else f"rf_entry_{number}."
            if role != "latent":
                entries.append(
                    dict(
                        prefix=entry_prefix,
                        names=[k for k in native_header if k != "__metadata__"],
                        options=json.loads(
                            native_header.get("__metadata__", {}).get("conditioning_options", "{}")
                        ),
                    )
                )
            parts.append((entry_prefix, raw))
        if role == "latent" and extras:
            parts.append(
                (
                    "",
                    safetensors.torch.save(
                        {k: v.detach().cpu().contiguous() for k, v in extras.items()}
                    ),
                )
            )
        context = dict(
            version=VERSION,
            family=plan.family,
            capture=json.loads(plan.context),
            runtime=runtime,
            layout=geometry,
            entries=entries,
        )
        raw = pack(parts, context)
        header(raw, "latent" if role == "latent" else "conditioning", plan.family)
        output = plan.directory / (prefix + (".latent" if role == "latent" else ".safetensors"))
        output.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        receipts = receipt_root(plan.directory)
        receipts.mkdir(parents=True, exist_ok=True)
        receipt = receipts / (digest + ".json")
        temporary = receipt.with_suffix("." + secrets.token_hex(8) + ".tmp")
        temporary.write_text(canonical(context), encoding="utf-8")
        temporary.replace(receipt)
        outputs.append(
            TensorExport(
                str(output),
                "sdxl-handoff" if handoff is not None else role,
                tuple(file_signature(output)),
                digest,
                plan.family,
                data.get("seed", 0),
                index,
            )
        )
    _check_job(data)
    data.setdefault(EXPORT_KEY, []).extend(outputs)
    return outputs
