"""The pinned MiniMax H3 still-image graph, with captured source ownership."""

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from ruinedfooocus.features.generation.image_output_batches import (
    ImageOutputBatch,
    OutputPlan,
    PLAN_KEY,
    with_output_plan,
)
from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan
from ruinedfooocus.features.models.model_identity import (
    cached_safetensors_header,
    file_signature,
)

KEY = "_native_minimax_image"
FAMILY = "MiniMaxH3Image"
CLIP_ROLE = "clip_minimax_h3_image"
VAE_ROLE = "vae_minimax_h3_image"
MAX_REFERENCES = 4
MAX_PIXELS = 16 * 1024 * 1024


def source_identity(signature):
    return (Path(signature[0]).name, signature[1], str(signature[2]), str(signature[3]))


def reference_sizes(size):
    """Record and bound the pinned native short-edge/grid policy before loading."""
    short = min(size)
    return tuple(
        tuple(
            max(32, round(side * (min(max(short, 1024), high) / short) / 32) * 32) for side in size
        )
        for high in (1024, 2048)
    )


def checkpoint_contract(path):
    """Require the image topology and its own finite PDD metadata, never a name."""
    from ruinedfooocus.features.models.utility_models import passive_native_class

    before = tuple(file_signature(path))
    if passive_native_class(path, {FAMILY}) != FAMILY:
        raise ValueError("Choose a native MiniMax H3 Image checkpoint, without an audio stream.")
    header = cached_safetensors_header(path)
    try:
        metadata = header.get("__metadata__", {})
        raw = metadata.get("config", "")
        if not isinstance(raw, str) or len(raw) > 1024 * 1024:
            raise ValueError
        config = json.loads(raw)["transformer"]
        steps, sigmas = config["pdd_steps"], config["pdd_sigmas"]
        if (
            type(steps) is not int
            or not 1 <= steps <= 64
            or not isinstance(sigmas, list)
            or len(sigmas) != steps + 1
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in sigmas
            )
            or sigmas[0] != 1
            or sigmas[-1] != 0
            or any(a <= b for a, b in zip(sigmas, sigmas[1:]))
            or config.get("image_model", "minimax_h3_image") != "minimax_h3_image"
            or config.get("latents_dim", 24) != 24
            or config.get("text_dim", 5120) != 5120
        ):
            raise ValueError
        shape = header["final_layer.video_out.weight"]["shape"]
        if not isinstance(shape, list) or shape[0] != 96 * steps:
            raise ValueError
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(
            "MiniMax H3 Image needs its original 24-channel PDD heads and descending sigma metadata."
        ) from error
    if tuple(file_signature(path)) != before:
        raise ValueError("The MiniMax image checkpoint changed during capture.")
    return before, steps, tuple(float(value) for value in sigmas)


def codec_contract(shapes):
    """Positive native sd.py image branch; exclude PH3 and the temporal decoder."""
    return (
        shapes.get("decoder.proj_out.weight", ())[:1] == (768,)
        and shapes.get("decoder.x_embedder.weight", ())[:1] == (1024,)
        and shapes.get("post_quant_conv.weight") == (24, 24, 1, 1)
        and "encoder.conv_in.weight" in shapes
        and "decoder.transformer_blocks.0.scale1" in shapes
        and not any(key.startswith("decoder.blocks.") for key in shapes)
    )


def validate_components(encoder, vae):
    from ruinedfooocus.features.models.component_candidates import header_shapes
    from ruinedfooocus.features.models.native_text_identity import passive_native_text_type

    if (
        Path(encoder).suffix.lower() not in (".safetensors", ".sft")
        or passive_native_text_type(
            header_shapes(encoder), allowed_types=frozenset({"QWEN3VL_32B"})
        )
        != "QWEN3VL_32B"
    ):
        raise ValueError(
            "MiniMax H3 Image requires its 50-layer Qwen3-VL 32B conditioning encoder. "
            "The optional 9B prompt helper and recovered video encoders are different components."
        )
    if Path(vae).suffix.lower() not in (".safetensors", ".sft") or not codec_contract(
        header_shapes(vae)
    ):
        raise ValueError("MiniMax H3 Image requires its 2D image VAE, not a video or PH3 codec.")


@dataclass(frozen=True)
class Reference:
    size: tuple[int, int]
    rgba: bytes

    def metadata(self):
        return dict(
            width=self.size[0],
            height=self.size[1],
            rgba_sha256=hashlib.sha256(self.rgba).hexdigest(),
        )

    def tensor(self):
        import numpy as np
        import torch

        rgba = (
            torch.from_numpy(
                np.frombuffer(self.rgba, dtype=np.uint8)
                .copy()
                .reshape(self.size[1], self.size[0], 4)
            ).float()
            / 255
        )
        return (rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:])).unsqueeze(0)


@dataclass(frozen=True)
class MiniMaxImagePlan:
    checkpoint: tuple
    steps: int
    sigmas: tuple[float, ...]
    references: tuple[Reference, ...]
    replay_components: tuple = ()

    def verify(self, path):
        if checkpoint_contract(path) != (self.checkpoint, self.steps, self.sigmas):
            raise ValueError("The captured MiniMax image checkpoint or PDD schedule changed.")

    def metadata(self):
        return dict(
            version=1,
            family=FAMILY,
            conditioning="MiniMaxH3ImageConditioning",
            sampler="MiniMaxH3ImageModelSampling",
            codec="MiniMaxH3ImageVAE",
            latent_layout="B,24,1,H/16,W/16",
            pdd_steps=self.steps,
            pdd_sigmas=list(self.sigmas),
            checkpoint=list(source_identity(self.checkpoint)),
            references=[source.metadata() for source in self.references],
            reference_sizes=[
                dict(
                    vision=list(reference_sizes(source.size)[0]),
                    vae=list(reference_sizes(source.size)[1]),
                )
                for source in self.references
            ],
            alpha="reference RGBA composited over neutral RGB; opaque RGB output",
        )


def require_plan(data, path):
    plan = (data or {}).get(KEY)
    if not isinstance(plan, MiniMaxImagePlan):
        raise ValueError(
            "Capture the MiniMax H3 Image request with its PDD schedule and image components."
        )
    plan.verify(path)
    return plan


def validate_prepared(data):
    from ruinedfooocus.features.generation.image_preparation import KEY as ASSETS_KEY, ImageAssets

    assets = (data or {}).get(ASSETS_KEY)
    if not isinstance(assets, ImageAssets):
        raise ValueError("Prepare the captured MiniMax image components before loading.")
    selected = {(item.role, item.name): item for item in assets.artifacts}
    plan = data.get(KEY)
    current = tuple(
        (item.role, *source_identity(item.signature))
        for item in assets.artifacts
        if item.signature is not None
    )
    if (
        isinstance(plan, MiniMaxImagePlan)
        and plan.replay_components
        and current != plan.replay_components
    ):
        raise ValueError(
            "The saved MiniMax model/encoder/codec identities no longer match. Select the original components to replay."
        )
    paths = []
    for role, name in (("clip", CLIP_ROLE), ("vae", VAE_ROLE)):
        item = selected.get((role, name))
        if (
            item is None
            or item.signature is None
            or tuple(file_signature(item.request.destination)) != item.signature
        ):
            raise ValueError("The MiniMax image components changed or were not prepared.")
        paths.append(item.request.destination)
    validate_components(*paths)


def replay_values(metadata):
    value = metadata.get("native_minimax_image")
    return {"native_minimax_image": value} if value is not None else {}


def capture(data, path):
    """Freeze existing typed Create/Edit inputs before a preparation/compute wait."""
    result = dict(data)
    signature, steps, sigmas = checkpoint_contract(path)
    generations = result.get("image_number", 1)
    if type(generations) is not int or not 1 <= generations <= 100:
        raise ValueError("Choose 1 through 100 bounded MiniMax image repetitions.")
    controls = result.get("controlnet") or {}
    kind = str(controls.get("type", "")).lower()
    if kind not in ("", "none", "edit") or result.get("inpaint_toggle"):
        raise ValueError("MiniMax H3 Image uses text-to-image or unmasked native Image tools Edit.")
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("MiniMax H3 Image requires the native image engine.")
    if result.get("automatic_upscale", "Off") not in (None, "Off"):
        raise ValueError("Keep automatic upscaling Off for the MiniMax H3 Image graph.")
    # These features insert alternate conditioning/model/schedule state. The
    # first image graph explicitly owns its released PDD execution contract.
    from ruinedfooocus.features.generation.native_tensor_io import enabled

    if (
        enabled(result)
        or result.get("sdxl_refiner_model")
        or result.get("model_recipe_path")
        or result.get("gligen_model")
        or result.get("hypernetwork_model")
        or result.get("reference_adapter") not in (None, "", "none")
        or str(result.get("lora_hook_mode", "off")).lower() != "off"
        or any(
            (
                entry.get("name")
                if isinstance(entry, dict)
                else entry[0]
                if isinstance(entry, (list, tuple)) and entry
                else entry
            )
            not in (None, "", "None")
            for entry in (result.get("loras") or ())
        )
        or result.get("model_patch_recipe") not in (None, "", "none")
        or result.get("control_recipe") not in (None, "", "none", "ordinary")
        or result.get("native_edit_recipe") not in (None, "", "auto")
    ):
        raise ValueError(
            "Clear tensor overrides, staged models and patch recipes for the MiniMax image PDD graph."
        )
    if (
        result.get("sampling_shift") is not None
        or result.get("cfg_norm_strength") is not None
        or result.get("cfg_norm_auto")
        or result.get("cfg_norm_pre_cfg")
        or any(
            result.get(key) is not None for key in ("lcm_noise", "lcm_noise_end", "lcm_noise_clip")
        )
    ):
        raise ValueError(
            "Use the native MiniMax PDD schedule without flow-shift or alternate CFG/noise patches."
        )
    for key, expected in (
        ("steps", steps),
        ("cfg", 1),
        ("sampler_name", "euler"),
        ("scheduler", "simple"),
        ("clip_skip", 1),
    ):
        if key in result and result[key] != expected:
            raise ValueError(f"MiniMax H3 Image requires {key}={expected} for this PDD checkpoint.")
        result[key] = expected
    if not isinstance(result.get("prompt"), str) or not result["prompt"].strip():
        raise ValueError("Describe the MiniMax image or edit in the text prompt.")
    if not isinstance(result.get("negative", ""), str) or result.get("negative", "").strip():
        raise ValueError(
            "MiniMax H3 Image uses native zeroed negative conditioning; clear the negative prompt."
        )
    primary = result.get("input_image")
    supplied_references = result.get("reference_images") or ()
    if not isinstance(supplied_references, (tuple, list)):
        raise ValueError("MiniMax references must be an ordered image list.")
    references = list(supplied_references)
    previous_plan = result.get(KEY)
    if (
        references
        and primary is None
        or kind == "edit"
        and primary is None
        and not isinstance(previous_plan, MiniMaxImagePlan)
    ):
        raise ValueError("Add the primary MiniMax edit image in Image tools.")
    sources = ([primary] if primary is not None else []) + references
    if not sources and isinstance(previous_plan, MiniMaxImagePlan):
        previous_plan.verify(path)
        captured = list(previous_plan.references)
    else:
        captured = []
    if len(sources) > MAX_REFERENCES:
        raise ValueError(
            "Use at most four ordered MiniMax image references including the primary image."
        )
    for source in sources:
        if (
            not isinstance(source, Image.Image)
            or not 0 < source.width * source.height <= MAX_PIXELS
        ):
            raise ValueError("Each MiniMax reference must be an image of at most 16 megapixels.")
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        # Bound the native short-edge reference scaling as well as upload bytes.
        if max(rgba.size) > 8 * min(rgba.size):
            raise ValueError("MiniMax references require an aspect ratio between 1:8 and 8:1.")
        captured.append(Reference(rgba.size, rgba.tobytes()))
    if sum(source.size[0] * source.size[1] for source in captured) > MAX_PIXELS:
        raise ValueError("MiniMax references exceed the combined 16-megapixel input budget.")
    if sum(math.prod(reference_sizes(source.size)[1]) for source in captured) > MAX_PIXELS:
        raise ValueError(
            "Native MiniMax reference scaling exceeds the combined 16-megapixel conditioning budget."
        )
    plan = MiniMaxImagePlan(signature, steps, sigmas, tuple(captured))
    previous = result.get("native_minimax_image")
    if previous is not None:
        if (
            not isinstance(previous, dict)
            or type(previous.get("version")) is not int
            or previous["version"] != 1
            or previous.get("references") != plan.metadata()["references"]
            or not isinstance(previous.get("checkpoint"), (list, tuple))
            or list(previous["checkpoint"]) != plan.metadata()["checkpoint"]
            or previous.get("pdd_steps") != plan.steps
            or previous.get("pdd_sigmas") != list(plan.sigmas)
        ):
            raise ValueError(
                "Reattach the original ordered RGBA references to replay this MiniMax image."
            )
        components = previous.get("components", ())
        if (
            not isinstance(components, (tuple, list))
            or len(components) not in (0, 3)
            or any(
                not isinstance(row, (tuple, list))
                or len(row) != 5
                or not isinstance(row[0], str)
                or row[0] not in ("minimax_model", "clip", "vae")
                or not isinstance(row[1], str)
                or Path(row[1]).name != row[1]
                or type(row[2]) is not int
                or row[2] < 1
                or any(
                    not isinstance(value, str) or not value.isdecimal() or len(value) > 30
                    for value in row[3:]
                )
                for row in components
            )
        ):
            raise ValueError("The saved MiniMax component identities are malformed.")
        from dataclasses import replace

        plan = replace(
            plan,
            replay_components=tuple(tuple(row) for row in components)
            or (
                previous_plan.replay_components
                if isinstance(previous_plan, MiniMaxImagePlan)
                else ()
            ),
        )
    result[KEY] = plan
    result["native_minimax_image"] = plan.metadata()
    if captured:
        result["controlnet"] = dict(type="edit")
    # No PIL object shared with a subsequent upload remains the native source.
    result["input_image"] = None
    result["reference_images"] = []
    return with_output_plan(result, OutputPlan("minimax-image", 1, 1))


def _require_runtime(owner, plan):
    model = owner.xl_base_patched.unet.model
    vae = owner.xl_base_patched.vae
    config = model.model_config.unet_config
    clip_model = owner.xl_base_patched.clip.cond_stage_model
    if (
        type(model).__name__ != FAMILY
        or type(model.model_sampling).__name__ != "MiniMaxH3ImageModelSampling"
        or tuple(config.get("pdd_sigmas", ())) != plan.sigmas
        or config.get("pdd_steps") != plan.steps
        or config.get("latents_dim") != 24
        or not any(base.__name__ == "MiniMaxH3TEModel" for base in type(clip_model).__mro__)
        or type(vae.first_stage_model).__name__ != "MiniMaxH3ImageVAE"
        or vae.latent_channels != 24
        or vae.spacial_compression_encode() != 16
        or vae.spacial_compression_decode() != 16
    ):
        raise ValueError(
            "The loaded MiniMax image graph differs from its captured model/encoder/codec contract."
        )


def process(owner, data, callback, check_cancelled):
    """Execute the native still graph inside the existing serialized image job."""
    import torch
    import comfy.sample
    import comfy.utils
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning
    from nodes import ConditioningZeroOut
    from ruinedfooocus.features.generation.image_preparation import (
        KEY as ASSETS_KEY,
        prepared_asset,
    )
    from ruinedfooocus.features.generation.automatic_upscale import upscale_decoded
    from ruinedfooocus.features.image_processing.native_image_formats import pipeline_images

    plan = data.get(KEY)
    if not isinstance(plan, MiniMaxImagePlan):
        raise ValueError("Capture the MiniMax H3 image request before native execution.")
    if any(
        data.get(key) != value
        for key, value in (
            ("steps", plan.steps),
            ("cfg", 1),
            ("sampler_name", "euler"),
            ("scheduler", "simple"),
            ("clip_skip", 1),
        )
    ):
        raise ValueError("The MiniMax image sampling settings changed after capture.")
    _require_runtime(owner, plan)
    size = data.get(SIZE_KEY)
    if not isinstance(size, SizePlan):
        size = SizePlan.create((data["width"], data["height"]))
    if size.scale != 1:
        raise ValueError("MiniMax H3 Image uses its exact target without a staged upscale factor.")
    size = size.aligned(32, minimum=64, maximum=4096)
    if size.canvas_width * size.canvas_height > MAX_PIXELS:
        raise ValueError("The MiniMax image canvas exceeds 16 megapixels.")
    data[SIZE_KEY] = size
    data["width"], data["height"] = size.canvas
    check_cancelled()
    assets = data.get(ASSETS_KEY)
    if assets is None:
        raise ValueError(
            "Prepare the captured MiniMax model and image components before execution."
        )
    with ExitStack() as stack:
        paths = {}
        for item in assets.artifacts:
            path = stack.enter_context(prepared_asset(data, item.role, item.name, None, item.role))
            paths[(item.role, item.name)] = path
        checkpoint = paths.get(("minimax_model", data["base_model_name"]))
        if checkpoint is None:
            raise ValueError("The MiniMax image model was not captured before compute.")
        plan.verify(checkpoint)
        encoder, vae = paths.get(("clip", CLIP_ROLE)), paths.get(("vae", VAE_ROLE))
        if encoder is None or vae is None:
            raise ValueError(
                "MiniMax H3 Image requires explicit captured external encoder and image VAE components."
            )
        validate_components(encoder, vae)
        validate_prepared(data)
        check_cancelled()
        prompt = data.get("positive_prompt", data["prompt"])
        negative_prompt = data.get("negative_prompt", data.get("negative", ""))
        if (
            not isinstance(prompt, str)
            or not isinstance(negative_prompt, str)
            or negative_prompt.strip()
        ):
            raise ValueError(
                "The native MiniMax graph needs a text prompt and zeroed negative conditioning."
            )
        positive, latent = MiniMaxH3ImageConditioning.execute(
            owner.xl_base_patched.clip,
            prompt,
            *size.canvas,
            vae=owner.xl_base_patched.vae,
            images={
                f"image_{index}": source.tensor() for index, source in enumerate(plan.references, 1)
            },
        )
        expected = (1, 24, 1, size.canvas_height // 16, size.canvas_width // 16)
        if tuple(latent["samples"].shape) != expected:
            raise ValueError("The native MiniMax image node returned a different latent layout.")
        negative = ConditioningZeroOut().zero_out(positive)[0]
        check_cancelled()
        noise = comfy.sample.prepare_noise(latent["samples"], data["seed"])
        pbar = comfy.utils.ProgressBar(plan.steps)

        def progress(step, x0, x, total_steps):
            check_cancelled()
            if callback is not None:
                callback(step, x0, x, total_steps, None)
            pbar.update_absolute(step + 1, total_steps, None)

        # The model already owns MiniMaxH3ImageModelSampling. No flow-shift
        # patch or hand-written replacement sigma curve is applied.
        samples = comfy.sample.sample(
            owner.xl_base_patched.unet,
            noise,
            plan.steps,
            1.0,
            "euler",
            "simple",
            positive,
            negative,
            latent["samples"],
            denoise=1.0,
            callback=progress,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=data["seed"],
        )
        check_cancelled()
        if tuple(samples.shape) != expected:
            raise ValueError("The MiniMax sampler changed the single-image latent layout.")
        decoded = owner.decode_latent(
            dict(samples=samples),
            tiled=data.get("vae_tiled", False),
            tile_size=data.get("vae_tile_size"),
            overlap=data.get("vae_overlap"),
        )
        check_cancelled()
    if tuple(decoded.shape) != (1, size.canvas_height, size.canvas_width, 3):
        raise ValueError("The MiniMax image codec must decode one opaque RGB image.")
    decoded = size.restore_canvas(decoded)
    decoded = upscale_decoded(owner, decoded, data)
    check_cancelled()
    data["native_minimax_image"] = plan.metadata() | dict(
        canvas_width=size.canvas_width,
        canvas_height=size.canvas_height,
        final_width=size.target_width,
        final_height=size.target_height,
        components=[[item.role, *source_identity(item.signature)] for item in assets.artifacts],
    )
    images = pipeline_images(decoded, data)
    if len(images) != 1:
        raise ValueError("The MiniMax still graph must return exactly one image per repetition.")
    return ImageOutputBatch(tuple(images), data[PLAN_KEY], (dict(alpha="opaque-RGB"),))
