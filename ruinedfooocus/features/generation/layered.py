"""Qwen Layered named native workflow; one repetition returns ordered RGBA layers."""

from dataclasses import dataclass
import hashlib

from PIL import Image, ImageOps

from ruinedfooocus.features.generation.image_output_batches import (
    PLAN_KEY,
    OutputPlan,
    ImageOutputBatch,
    with_output_plan,
)

KEY = "_native_layered_plan"
MAX_OUTPUT_PIXELS = 16 * 1024**2
MAX_LAYERS = 16


def is_layered_recipe(path, manager):
    from ruinedfooocus.features.models.native_recipes import current_native_recipe

    recipe = current_native_recipe(path, manager) if path is not None else {}
    return recipe.get("family") == "QwenImage" and recipe.get("variant") == "layered"


def replay_values(metadata):
    result = {}
    if "qwen_layer_count" in metadata:
        result["qwen_layer_count"] = metadata["qwen_layer_count"]
    native = metadata.get("native_layered_conditioning")
    if isinstance(native, dict) and native.get("source"):
        # Pixel hashes identify an upload; they cannot restore the upload itself.
        result["_native_layered_replay_source"] = native["source"]
    return result


@dataclass(frozen=True)
class LayeredPlan:
    layers: int
    recipe_id: str
    identity_basis: str
    source_size: tuple[int, int] | None = None
    source_pixels: bytes | None = None
    signature: tuple = ()
    generations: int = 1

    def source(self):
        return (
            Image.frombytes("RGBA", self.source_size, self.source_pixels)
            if self.source_pixels is not None
            else None
        )

    def metadata(self):
        result = dict(
            recipe=self.recipe_id,
            layer_count=self.layers,
            native_order="latent-slots-1-through-layers",
            codec="vae_qwen_image_layered",
            identity_basis=self.identity_basis,
        )
        if self.source_pixels is not None:
            result["source"] = dict(
                width=self.source_size[0],
                height=self.source_size[1],
                mode="RGBA8",
                working_pixel_sha256=hashlib.sha256(self.source_pixels).hexdigest(),
            )
        return result


def capture(data, manager, models, *, family=None):
    result = dict(data)
    previous = result.pop(KEY, None)
    if family is not None and family != "QwenImage":
        if result.get("task_type") == "tool_call" and result.get("qwen_layer_count") is not None:
            raise ValueError("Select an identified Qwen Image Layered checkpoint for layer output.")
        result.pop("qwen_layer_count", None)
        result.pop("qwen_layer_source", None)
        return result
    from ruinedfooocus.features.models.native_recipes import current_native_recipe
    from ruinedfooocus.features.models.model_identity import file_signature

    name = result.get("base_model_name")
    path = models.get_file("checkpoints", name) if models is not None and name else None
    recipe = current_native_recipe(path, manager) if path is not None else {}
    if recipe.get("family") != "QwenImage" or recipe.get("variant") != "layered":
        if result.get("task_type") == "tool_call" and result.get("qwen_layer_count") is not None:
            raise ValueError("Select an identified Qwen Image Layered checkpoint for layer output.")
        result.pop("qwen_layer_count", None)
        result.pop("qwen_layer_source", None)
        return result
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Qwen Layered requires the native image engine.")
    if result.get("output_format", "png") == "avif":
        raise ValueError(
            "Transparent layers require PNG or EXR output; native still AVIF cannot retain alpha."
        )
    if result.get("automatic_upscale") not in (None, "", "Off"):
        raise ValueError("Layered output uses its native size; turn automatic upscaling off.")
    controls = result.get("controlnet") or {}
    if (
        result.get("inpaint_toggle")
        or controls.get("type", "").lower() not in ("", "img2img", "edit")
        or result.get("reference_images")
        or result.get("reference_adapter") not in (None, "", "none")
        or result.get("model_patch_recipe") not in (None, "", "none")
    ):
        raise ValueError(
            "Layered generation uses one optional source image; clear other image tools and masks."
        )
    count = result.get("qwen_layer_count")
    count = 2 if count is None else count
    if type(count) is not int or not 1 <= count <= MAX_LAYERS:
        raise ValueError("Choose 1 through 16 output layers.")
    generations = result.get("image_number", 1)
    if type(generations) is not int or not 1 <= generations <= 100:
        raise ValueError("Choose a bounded number of layer generation repetitions.")
    width, height = (
        result.get("width", result.get("custom_width")),
        result.get("height", result.get("custom_height")),
    )
    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan

    size = result.get(SIZE_KEY)
    if isinstance(size, SizePlan):
        target_w, target_h = size.target
        if not width or not height or target_w * target_h > width * height:
            width, height = target_w, target_h
    if all(type(value) is int and value > 0 for value in (width, height)):
        validate_budget(count, generations, width, height)
    source = result.get("qwen_layer_source")
    if source is not None and result.get("input_image") is not None:
        raise ValueError("Choose one layer source image.")
    if source is None:
        source = result.get("input_image")
    source_size, pixels = None, None
    signature = tuple(file_signature(path))
    if isinstance(previous, LayeredPlan):
        if previous.signature != signature or previous.recipe_id != recipe["id"]:
            raise ValueError("The layered checkpoint changed after capture. Submit a new request.")
        source_size, pixels = previous.source_size, previous.source_pixels
    elif source is not None:
        if (
            not isinstance(source, Image.Image)
            or not 0 < source.width * source.height <= MAX_OUTPUT_PIXELS
        ):
            raise ValueError("The layer source must be an image of at most 16 megapixels.")
        source = ImageOps.exif_transpose(source).convert("RGBA")
        source_size, pixels = source.size, source.tobytes()
    expected_source = result.pop("_native_layered_replay_source", None)
    if expected_source is not None and (
        not isinstance(expected_source, dict)
        or pixels is None
        or hashlib.sha256(pixels).hexdigest() != expected_source.get("working_pixel_sha256")
    ):
        raise ValueError("Reattach the original RGBA source image to replay these layers.")
    plan = LayeredPlan(
        count,
        recipe["id"],
        recipe.get("identity_basis", "current_native_recipe"),
        source_size,
        pixels,
        signature,
        generations,
    )
    result.update(
        {
            KEY: plan,
            "qwen_layer_count": count,
            "native_layered_conditioning": plan.metadata(),
            "input_image": None,
            "controlnet": {},
            "reference_images": [],
        }
    )
    result.pop("native_edit_recipe", None)
    result.pop("qwen_layer_source", None)
    return with_output_plan(result, OutputPlan("qwen-layers", count, count))


def validate_budget(layers, generations, width, height):
    if width * height * layers * generations > MAX_OUTPUT_PIXELS:
        raise ValueError(
            "Layer output exceeds 16 megapixels across all layers and repetitions. Reduce size, layers or repetitions."
        )


def require_vae(vae):
    if (
        getattr(vae, "latent_channels", None) != 16
        or getattr(vae, "output_channels", None) != 4
        or vae.spacial_compression_encode() != 8
        or vae.spacial_compression_decode() != 8
    ):
        raise ValueError("Qwen Layered requires its 16-channel, eightfold RGBA VAE.")


def empty_latent(plan, vae, width, height):
    from comfy_extras.nodes_qwen import EmptyQwenImageLayeredLatentImage

    require_vae(vae)
    if any(type(side) is not int or side < 16 or side % 16 for side in (width, height)):
        raise ValueError("The native layer canvas must use positive multiples of 16 pixels.")
    validate_budget(plan.layers, plan.generations, width, height)
    return EmptyQwenImageLayeredLatentImage.execute(width, height, plan.layers, batch_size=1)[0]


def source_frame(plan, width, height, resize_mode):
    from ruinedfooocus.features.image_editing.image_geometry import reference_plan

    if plan.source_size is None:
        return None
    frame = reference_plan(plan.source_size, (width, height), resize_mode, multiple=16)
    validate_budget(plan.layers, plan.generations, frame["canvas_width"], frame["canvas_height"])
    return frame


def reference_conditioning(
    plan, vae, positive, negative, width, height, resize_mode, *, frame=None
):
    from nodes import VAEEncode
    from comfy_extras.nodes_edit_model import ReferenceLatent
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

    require_vae(vae)
    source = plan.source()
    if source is None:
        return positive, negative, None
    target = (frame["width"], frame["height"]) if frame is not None else (width, height)
    pixels, prepared_frame = prepare_reference(source, target, resize_mode, preserve_alpha=True)
    if frame is not None and prepared_frame != frame:
        raise ValueError("The captured layer source framing changed before conditioning.")
    frame = prepared_frame
    latent = VAEEncode().encode(vae, pixels)[0]
    return (
        ReferenceLatent.execute(positive, latent)[0],
        ReferenceLatent.execute(negative, latent)[0],
        frame,
    )


def patch_model(model, guidance):
    if type(model.model).__name__ != "QwenImage":
        raise ValueError("Layered generation requires a native Qwen Image checkpoint.")
    if guidance.get("sampling_shift") is not None:
        return model
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

    return ModelSamplingAuraFlow().patch_aura(model, shift=1.0)[0]


def layer_latent(plan, sampled):
    from comfy_extras.nodes_latent import LatentCut, LatentCutToBatch

    shape = sampled["samples"].shape
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 16 or shape[2] != plan.layers + 1:
        raise ValueError("The native layered latent has an unexpected batch/channel/layer shape.")
    layers = LatentCut.execute(sampled, "t", 1, plan.layers)[0]
    return LatentCutToBatch.execute(layers, "t", 1)[0]


def outputs(data, images):
    plan = data.get(KEY)
    if not isinstance(plan, LayeredPlan) or len(images) != plan.layers:
        raise ValueError("The decoded layer count differs from the captured request.")
    if any(len(image.shape) != 3 or image.shape[-1] != 4 for image in images):
        raise ValueError("The native layer decoder must return ordered RGBA images.")
    return ImageOutputBatch(
        tuple(images),
        data[PLAN_KEY],
        tuple(
            dict(
                layer_index=index,
                layer_count=plan.layers,
                latent_slot=index + 1,
                alpha="native-RGBA",
            )
            for index in range(plan.layers)
        ),
    )
