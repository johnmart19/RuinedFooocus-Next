"""Ordered literal text boxes through the pinned native GLIGEN conditioning node."""

from dataclasses import dataclass
import json
import math

from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature
from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
    AuxiliaryAsset,
    artifact,
    attention_layout,
    selected_asset,
    require_ordinary_codec,
    safe_native_state,
)

KEY = "_native_gligen_layout"
FIELDS = ("gligen_model", "gligen_model_signature", "gligen_layout")
RESETS = dict(gligen_model=None, gligen_model_signature=None, gligen_layout="[]")
FAMILIES = frozenset(("BaseModel", "SD20", "SDXL", "SDXLRefiner"))


def guidance_header(path):
    try:
        return cached_safetensors_header(path)
    except (ValueError, RecursionError, TypeError) as error:
        raise ValueError("The GLIGEN file has an invalid bounded safetensors header.") from error


@dataclass(frozen=True)
class TextBox:
    text: str
    x: int
    y: int
    width: int
    height: int

    def metadata(self):
        return dict(text=self.text, x=self.x, y=self.y, width=self.width, height=self.height)


@dataclass(frozen=True)
class GligenPlan:
    asset: AuxiliaryAsset
    checkpoint_signature: tuple
    boxes: tuple[TextBox, ...]


def parse_layout(value):
    if not isinstance(value, str) or len(value) > 32768:
        raise ValueError("GLIGEN layout must be bounded JSON text.")
    try:
        rows = json.loads(value)
    except (ValueError, RecursionError) as error:
        raise ValueError("Enter a JSON list of text, x, y, width and height boxes.") from error
    if not isinstance(rows, list) or not 1 <= len(rows) <= 30:
        raise ValueError("GLIGEN requires one through 30 ordered text boxes.")
    boxes = []
    total_text = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"text", "x", "y", "width", "height"}:
            raise ValueError("Each GLIGEN box needs exactly text, x, y, width and height.")
        text = row["text"]
        if not isinstance(text, str) or not 1 <= len(text) <= 1024 or not text.strip():
            raise ValueError("Each GLIGEN box requires one through 1024 literal text characters.")
        total_text += len(text)
        if total_text > 8192 or any(ord(char) < 32 and char not in "\n\r\t" for char in text):
            raise ValueError("GLIGEN text exceeds its total limit or contains control codes.")
        for key in ("x", "y", "width", "height"):
            number = row[key]
            if type(number) is not int or not 0 <= number <= 32768 or number % 8:
                raise ValueError(
                    "GLIGEN box coordinates and sizes must use whole eight-pixel units."
                )
        if not row["width"] or not row["height"]:
            raise ValueError("GLIGEN boxes must have positive width and height.")
        boxes.append(TextBox(**row))
    return tuple(boxes)


def validate_canvas(plan, width, height):
    if any(type(side) is not int or side < 8 or side % 8 for side in (width, height)):
        raise ValueError("GLIGEN needs an eight-pixel aligned target canvas.")
    if any(box.x + box.width > width or box.y + box.height > height for box in plan.boxes):
        raise ValueError("GLIGEN boxes must remain inside the selected target canvas.")


def header_profile(header):
    """Require complete native fusers instead of accepting random missing weights."""
    if (
        not isinstance(header, dict)
        or len(header) > 4096
        or any(not isinstance(key, str) for key in header)
    ):
        raise ValueError("The GLIGEN state dictionary is invalid or oversized.")

    def shape(key):
        value = header.get(key, {})
        dimensions = value.get("shape") if isinstance(value, dict) else None
        if not isinstance(dimensions, (list, tuple)) or any(
            type(side) is not int or side < 0 for side in dimensions
        ):
            return None
        if value.get("dtype") not in ("F16", "BF16", "F32", "F64"):
            return None
        return tuple(dimensions)

    pooled = shape("position_net.null_positive_feature")
    output = shape("position_net.linears.4.weight")
    if (
        pooled is None
        or output is None
        or len(pooled) != 1
        or not 1 <= pooled[0] <= 4096
        or len(output) != 2
    ):
        raise ValueError("The selected file has no native GLIGEN position network.")
    key_width = output[0]
    expected = {
        "position_net.null_positive_feature": pooled,
        "position_net.null_position_feature": (64,),
        "position_net.linears.0.weight": (512, pooled[0] + 64),
        "position_net.linears.0.bias": (512,),
        "position_net.linears.2.weight": (512, 512),
        "position_net.linears.2.bias": (512,),
        "position_net.linears.4.weight": (key_width, 512),
        "position_net.linears.4.bias": (key_width,),
    }
    if (
        not 1 <= key_width <= 4096
        or pooled[0] != key_width
        or any(shape(key) != value for key, value in expected.items())
    ):
        raise ValueError(
            "The native GLIGEN position-network shapes are incomplete or incompatible."
        )
    elements = sum(math.prod(value) for value in expected.values())
    queries = []
    for section in ("input_blocks", "middle_block", "output_blocks"):
        for index in range(20):
            keys = [key for key in header if f"{section}.{index}." in key and ".fuser." in key]
            if not keys:
                continue
            state = {key.split(".fuser.")[-1]: key for key in keys}
            if len(state) != len(keys):
                raise ValueError("GLIGEN fusers contain duplicate native module positions.")
            linear = shape(state.get("linear.weight", ""))
            if (
                linear is None
                or len(linear) != 2
                or linear[1] != key_width
                or not 1 <= linear[0] <= 4096
            ):
                raise ValueError("The native GLIGEN fuser has an incompatible context width.")
            query = linear[0]
            if query % (8 if key_width == 768 else 64):
                raise ValueError("The native GLIGEN attention heads do not divide its query width.")
            required = {
                "linear.weight": (query, key_width),
                "linear.bias": (query,),
                "alpha_attn": (),
                "alpha_dense": (),
                "norm1.weight": (query,),
                "norm1.bias": (query,),
                "norm2.weight": (query,),
                "norm2.bias": (query,),
                "attn.to_q.weight": (query, query),
                "attn.to_k.weight": (query, query),
                "attn.to_v.weight": (query, query),
                "attn.to_out.0.weight": (query, query),
                "attn.to_out.0.bias": (query,),
                "ff.net.0.proj.weight": (query * 8, query),
                "ff.net.0.proj.bias": (query * 8,),
                "ff.net.2.weight": (query, query * 4),
                "ff.net.2.bias": (query,),
            }
            if any(
                name not in state or shape(state[name]) != wanted
                for name, wanted in required.items()
            ):
                raise ValueError("The native GLIGEN fuser state is incomplete or incompatible.")
            elements += sum(math.prod(value) for value in required.values())
            if elements > 512 * 1024**2:
                raise ValueError("The GLIGEN model exceeds its bounded dense-parameter budget.")
            queries.append(query)
    if not queries:
        raise ValueError("The file contains no native GLIGEN fusers.")
    return pooled[0], key_width, tuple(queries)


def capture_assets(data, manager, models, family):
    result = dict(data)
    previous = result.pop(KEY, None)
    if not result.get("gligen_model"):
        result.pop("gligen_guidance", None)
        return result, ()
    from ruinedfooocus.features.models.image_components import ALIASES

    if ALIASES.get(family, family) not in FAMILIES:
        raise ValueError("GLIGEN needs an ordinary compatible SD image checkpoint and CLIP.")
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("GLIGEN layout requires the native image engine.")
    if (
        result.get("input_image") is not None
        or result.get("inpaint_toggle")
        or (result.get("controlnet") or {}).get("type", "").lower() not in ("", "none")
        or result.get("reference_adapter") not in (None, "", "none")
        or result.get("automatic_upscale") not in (None, "", "Off")
        or result.get("sdxl_refiner_model")
    ):
        raise ValueError(
            "GLIGEN layout uses a new image canvas; clear image tools, refiner and upscaling."
        )
    if any(
        result.get(key) is not None for key in ("native_positive_input", "native_negative_input")
    ) or result.get("native_export_conditioning"):
        raise ValueError(
            "Native conditioning files cannot serialize GLIGEN's executable position model; use a latent-only workflow with the original rebuilt layout."
        )
    from ruinedfooocus.features.generation.native_tensor_io import enabled as tensor_io_enabled

    if tensor_io_enabled(result):
        result["native_tensor_workflow"] = True
    asset = selected_asset(
        manager, "gligen", result["gligen_model"], result.get("gligen_model_signature")
    )
    if is_safetensors(asset.signature[0]):
        header_profile(guidance_header(asset.signature[0]))
    checkpoint = models.get_file("checkpoints", result.get("base_model_name"))
    plan = GligenPlan(
        asset, tuple(file_signature(checkpoint)), parse_layout(result.get("gligen_layout", "[]"))
    )
    if isinstance(previous, GligenPlan) and (
        previous.asset != asset or previous.checkpoint_signature != plan.checkpoint_signature
    ):
        raise ValueError("The queued GLIGEN sources changed. Submit a new request.")
    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan

    size = result.get(SIZE_KEY)
    width, height = (
        size.target
        if isinstance(size, SizePlan)
        else (
            result.get("width", result.get("custom_width")),
            result.get("height", result.get("custom_height")),
        )
    )
    if width is not None and height is not None:
        validate_canvas(plan, width, height)
    result.update(
        {
            KEY: plan,
            "image_engine": "native",
            "gligen_model": asset.name,
            "gligen_model_signature": list(asset.public_signature),
            "gligen_layout": json.dumps(
                [box.metadata() for box in plan.boxes], separators=(",", ":")
            ),
        }
    )
    return result, (artifact(asset),)


def clear_cache(owner):
    cached = getattr(owner, "_native_gligen", None)
    if cached is not None:
        from comfy.model_management import unload_model_and_clones

        owner._native_gligen = None
        unload_model_and_clones(cached[1])


def apply(owner, positive, data, manager, check_cancelled):
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.features.models.yaml_recipes import apply_clip_recipe, captured_recipe
    from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder
    from nodes import GLIGENTextBoxApply
    from ruinedfooocus.features.image_editing.clip_hook_conditioning import direct_conditioning
    import comfy.sd

    plan = data.get(KEY)
    if (
        not isinstance(plan, GligenPlan)
        or tuple(file_signature(plan.checkpoint_signature[0])) != plan.checkpoint_signature
    ):
        raise ValueError("Capture the current GLIGEN checkpoint and layout before generation.")
    validate_canvas(plan, data["width"], data["height"])
    require_ordinary_codec(owner.xl_base_patched.vae)
    queries, contexts, _ = attention_layout(owner.xl_base_patched.unet)
    check_cancelled()
    with prepared_asset(data, "gligen", plan.asset.name, manager, "gligen") as path:
        cached = getattr(owner, "_native_gligen", None)
        if cached is None or cached[0] != plan.asset.signature:
            clear_cache(owner)
            if is_safetensors(path):
                profile = header_profile(guidance_header(path))
            else:
                state = safe_native_state(path)
                profile = header_profile(
                    {
                        key: {
                            "shape": list(value.shape),
                            "dtype": "F32" if value.is_floating_point() else None,
                        }
                        for key, value in state.items()
                        if hasattr(value, "shape")
                    }
                )
                del state
            if profile[2] != queries or contexts != {profile[1]}:
                raise ValueError("GLIGEN fuser positions or context width do not match this UNet.")
            native = comfy.sd.load_gligen(str(path))
            owner._native_gligen = cached = (plan.asset.signature, native, profile)
        if cached[2][2] != queries or contexts != {cached[2][1]}:
            raise ValueError("The cached GLIGEN model does not match this UNet's attention layout.")
        if tuple(file_signature(path)) != plan.asset.signature:
            clear_cache(owner)
            raise ValueError("The GLIGEN file changed during loading. Submit the request again.")
    check_cancelled()
    clip = apply_clip_recipe(owner.xl_base_patched.clip, data["clip_skip"], captured_recipe(data))
    warm_patched_text_encoder(clip)

    class CheckedClip:
        def __init__(self, current):
            self.current = current

        def tokenize(self, *args, **kwargs):
            return self.current.tokenize(*args, **kwargs)

        def encode_from_tokens(self, *args, **kwargs):
            encoded, pooled = self.current.encode_from_tokens(*args, **kwargs)
            if tuple(pooled.shape) != (1, cached[2][0]):
                raise ValueError("GLIGEN needs the matching unprojected CLIP pooled width.")
            return encoded, pooled

    for box in plan.boxes:
        positive = direct_conditioning(
            clip,
            lambda current, selected: GLIGENTextBoxApply().append(
                selected, CheckedClip(current), cached[1], **box.metadata()
            )[0],
            positive,
        )
        check_cancelled()
    data["gligen_guidance"] = dict(
        recipe="native-gligen-layout-v1",
        source=plan.asset.metadata(),
        width=data["width"],
        height=data["height"],
        boxes=[box.metadata() for box in plan.boxes],
        pooled_width=cached[2][0],
        native_units="pixels-divided-by-eight",
    )
    return positive
