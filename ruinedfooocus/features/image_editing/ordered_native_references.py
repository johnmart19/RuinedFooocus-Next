"""Typed ordered edit-reference adapters for pinned native image nodes."""

from dataclasses import dataclass
import hashlib
import math

FLUX2_FAMILIES = frozenset({"Flux2", "Flux2Klein4B", "Flux2Klein9B"})
RECIPES = {
    "boogu-edit": ("Boogu", 16, "TextEncodeBooguEdit", None),
    "krea2-style": ("Krea2", 3, "TextEncodeQwenImageEditPlus", "index_timestep_zero"),
    "qwen-edit": ("QwenImage", 1, "TextEncodeQwenImageEdit", None),
    "qwen-edit-2509": ("QwenImage", 3, "TextEncodeQwenImageEditPlus", None),
    "qwen-edit-2511": ("QwenImage", 3, "TextEncodeQwenImageEditPlus", "index_timestep_zero"),
    "longcat-edit": ("LongCatImage", 1, "TextEncodeQwenImageEdit", "index"),
    "flux2-reference": ("Flux2", 16, "ReferenceMultiLatent", None),
}
RECIPE_CHOICES = ("auto", *RECIPES)
VARIANTS = {
    "boogu-edit": {"edit"},
    "qwen-edit": {"edit"},
    "qwen-edit-2509": {"edit-2509"},
    "qwen-edit-2511": {"edit-2511"},
    "longcat-edit": {"edit"},
}


def reference_family(model):
    family = type(model).__name__
    if family == "Flux2":
        config = getattr(getattr(model, "model_config", None), "unet_config", {})
        return {20: "Flux2Klein4B", 24: "Flux2Klein9B"}.get(
            config.get("depth_single_blocks"), family
        )
    return family


def recipe_for(family, requested="auto", source=None):
    """Auto uses a current verified source; an explicit compatible recipe stays unqualified."""
    if requested not in RECIPE_CHOICES:
        raise ValueError("Choose an available native edit recipe.")
    variant = source.get("behavioral_variant") if source else None
    if requested == "auto":
        if family == "Boogu":
            if variant != "edit":
                raise ValueError(
                    "Automatic editing needs an identified Boogu Edit checkpoint. Inspect it or select Boogu Edit explicitly."
                )
            requested = "boogu-edit"
        elif family == "Krea2":
            requested = "krea2-style"
        elif family in FLUX2_FAMILIES:
            requested = "flux2-reference"
        elif family == "LongCatImage":
            if variant != "edit":
                raise ValueError(
                    "Choose a LongCat Image Edit checkpoint. Inspect it for Automatic, or select LongCat Image Edit in Edit recipe."
                )
            requested = "longcat-edit"
        elif family == "QwenImage":
            requested = {
                "edit": "qwen-edit",
                "edit-2509": "qwen-edit-2509",
                "edit-2511": "qwen-edit-2511",
            }.get(variant)
            if requested is None:
                raise ValueError(
                    "Automatic editing needs an identified Qwen Edit checkpoint. Inspect the model or choose its matching Edit recipe. Use Image to image with a base model."
                )
        else:
            return None
    expected = RECIPES[requested][0]
    matches = family in FLUX2_FAMILIES if expected == "Flux2" else family == expected
    if not matches:
        raise ValueError("The native edit recipe does not match this checkpoint family.")
    if source and (
        source.get("native_family") not in FLUX2_FAMILIES
        if expected == "Flux2"
        else source.get("native_family") != expected
    ):
        raise ValueError("The verified edit artifact does not match this native family.")
    if source and requested in VARIANTS and variant not in VARIANTS[requested]:
        raise ValueError("Choose the checkpoint artifact for the selected native edit variant.")
    return requested


def current_source(name, models, manager):
    """Reuse current source receipts; no name inference, network or background hashing."""
    from ruinedfooocus.core.cache_contract import configured_cache_root
    from ruinedfooocus.features.models.model_identity import (
        read_prepared_native_source,
        verified_native_source,
    )

    path = models.get_file("checkpoints", name)
    if path is None:
        return None
    metadata = models.get_models_by_path("checkpoints", name, fetch=False)
    live_metadata = metadata
    source = verified_native_source(path, metadata)
    if source is None:
        metadata = read_prepared_native_source(path, configured_cache_root(manager))
        source = verified_native_source(path, metadata)
    if source is not None:
        return dict(source, identity_basis="verified publisher artifact")
    from ruinedfooocus.features.models.native_recipes import compatible_native_recipe

    recipe = compatible_native_recipe(path, live_metadata)
    # A family fallback is not evidence of an edit training variant. Explicit
    # current-header variants are admitted without inventing artifact qualification.
    if recipe and not recipe["applicability"].get("default_for_native_configuration"):
        return dict(
            native_family=recipe["family"],
            behavioral_variant=recipe["variant"],
            identity_basis=recipe["identity_basis"],
        )
    return None


def omit_inactive_recipe(data, controls, family=None):
    """Retain the UI choice while excluding it from unrelated queued operations."""
    from ruinedfooocus.features.models.image_components import ALIASES

    result = dict(data)
    family = ALIASES.get(family, family)
    if family != "Krea2" or str((controls or {}).get("type", "")).lower() != "edit":
        result.pop("krea_style_lora", None)
        result.pop("krea_style_strength", None)
    if str((controls or {}).get("type", "")).lower() != "edit":
        result.pop("native_edit_recipe", None)
        if ALIASES.get(family, family) in ("Boogu", "Krea2", "QwenImage", *FLUX2_FAMILIES):
            result["reference_images"] = []
    return result


def _image(value):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    if isinstance(value, Image.Image):
        value = ImageOps.exif_transpose(value).convert("RGB")
        if not 0 < value.width * value.height <= 16_777_216:
            raise ValueError("Each edit reference may contain at most 16 megapixels.")
        value = torch.from_numpy(np.array(value, dtype=np.float32) / 255)[None]
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or value.shape[0] != 1
        or value.shape[-1] not in (3, 4)
    ):
        raise ValueError("Native references require single RGB images in upload order.")
    if not 0 < value.shape[1] * value.shape[2] <= 16_777_216:
        raise ValueError("Each edit reference may contain at most 16 megapixels.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Native edit references must contain finite pixels.")
    with torch.inference_mode(False):
        return value[..., :3].detach().clone()


@dataclass
class OrderedNativeReferences:
    """Own captured image order and share only request-local native VAE results."""

    recipe: str
    vae: object
    images: tuple
    source: dict | None = None

    def __post_init__(self):
        if self.recipe not in RECIPES:
            raise ValueError("Choose an available native edit recipe.")
        _, limit, _, _ = RECIPES[self.recipe]
        if not 1 <= len(self.images) <= limit:
            raise ValueError(
                f"This native recipe accepts one through {limit} ordered references including the primary image."
            )
        self.images = tuple(_image(image) for image in self.images)
        self._input_contract = self._contract()
        self._latents = []
        self._cursor = 0
        expected_channels = 128 if self.recipe == "flux2-reference" else 16
        if getattr(self.vae, "latent_channels", None) != expected_channels:
            raise ValueError(
                "The selected VAE latent channels do not match this native edit recipe."
            )
        if self.recipe == "flux2-reference" and self.vae.spacial_compression_encode() != 16:
            raise ValueError("Flux.2 references require their 128-channel 16-fold VAE.")
        if (
            self.recipe in ("boogu-edit", "krea2-style")
            and self.vae.spacial_compression_encode() != 8
        ):
            raise ValueError("This native edit recipe requires its 16-channel eightfold VAE.")

    def encode(self, pixels):
        """Native image nodes call this in stable slot order for each text encoding."""
        if self._contract() != self._input_contract:
            raise ValueError("The captured native reference input changed after submission.")
        index = self._cursor
        self._cursor += 1
        if index > len(self._latents):
            raise ValueError("The native reference encoder changed its slot order.")
        preprocessing = (tuple(pixels.shape), pixels.dtype, pixels.device)
        if index == len(self._latents):
            self._latents.append((preprocessing, self.vae.encode(pixels)))
        signature, latent = self._latents[index]
        if signature != preprocessing:
            raise ValueError("The native reference encoder changed its captured preprocessing.")
        return latent

    def _contract(self):
        return (
            self.recipe,
            id(self.vae),
            tuple(
                (image.data_ptr(), tuple(image.shape), image.dtype, image.device, image._version)
                for image in self.images
            ),
        )

    def _condition(self, clip, text):
        from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit, TextEncodeQwenImageEditPlus

        self._cursor = 0
        if self.recipe == "boogu-edit":
            return self._paired_boogu(clip, text, "")[0]
        if self.recipe in ("qwen-edit-2509", "qwen-edit-2511", "krea2-style"):
            inputs = {f"image{index}": image for index, image in enumerate(self.images, 1)}
            conditioning = TextEncodeQwenImageEditPlus.execute(
                clip=clip, prompt=text, vae=self, **inputs
            )[0]
        else:
            conditioning = TextEncodeQwenImageEdit.execute(
                clip=clip, prompt=text, vae=self, image=self.images[0]
            )[0]
        method = RECIPES[self.recipe][3]
        if method is not None:
            from comfy_extras.nodes_flux import FluxKontextMultiReferenceLatentMethod

            conditioning = FluxKontextMultiReferenceLatentMethod.execute(conditioning, method)[0]
        return conditioning

    def _paired_boogu(self, clip, prompt, negative_prompt):
        from comfy_extras.nodes_boogu import TextEncodeBooguEdit

        self._cursor = 0
        return tuple(
            TextEncodeBooguEdit.execute(
                clip=clip,
                prompt=prompt,
                negative_prompt=negative_prompt,
                vae=self,
                images={f"image_{index}": image for index, image in enumerate(self.images, 1)},
            )
        )

    def conditioning(self, clip, prompt, negative_prompt):
        if self.recipe == "flux2-reference":
            raise ValueError("Flux.2 attaches VAE references to its ordinary text encoding.")
        from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder

        warm_patched_text_encoder(clip)
        if self.recipe == "boogu-edit":
            return self._paired_boogu(clip, prompt, negative_prompt)
        positive = self._condition(clip, prompt)
        if self.recipe == "krea2-style" and not negative_prompt:
            from nodes import ConditioningZeroOut

            return positive, ConditioningZeroOut().zero_out(positive)[0]
        return positive, self._condition(clip, negative_prompt)

    def scheduled_conditioning(self, clip, prompt, negative_prompt, steps):
        from node_helpers import conditioning_set_values
        from ruinedfooocus.features.prompting.prompt_scheduling import prompt_switch_per_step

        positive, negative = self.conditioning(clip, prompt, negative_prompt)
        schedule = prompt_switch_per_step(prompt, steps)
        cache, switched = {}, []
        for index, text in enumerate(schedule):
            if text not in cache:
                cache[text] = self._condition(clip, text)
            switched += conditioning_set_values(
                cache[text],
                {
                    "start_percent": index / len(schedule),
                    "end_percent": (index + 1) / len(schedule),
                    "_rf_prompt_step": index,
                },
            )
        return positive, negative, switched

    def attach(self, positive, negative):
        if self.recipe != "flux2-reference":
            raise ValueError("This edit recipe requires its vision-aware native text encoder.")
        from comfy_extras.nodes_edit_model import ReferenceMultiLatent

        self._cursor = 0
        return tuple(
            ReferenceMultiLatent.execute(
                positive=positive,
                negative=negative,
                vae=self,
                resolution={"resolution": "keep resolution"},
                images={f"image{index}": image for index, image in enumerate(self.images, 1)},
            )
        )

    def provenance(self):
        return {
            "recipe": self.recipe,
            "native_node": RECIPES[self.recipe][2],
            "reference_method": RECIPES[self.recipe][3] or "native default",
            "source_identity": self.source.get("identity_basis", "verified publisher artifact")
            if self.source
            else "compatible recipe; artifact unqualified",
            "artifact_qualified": False,
            "references": [
                {
                    "index": index,
                    "width": image.shape[2],
                    "height": image.shape[1],
                    "pixel_dtype": str(image.dtype),
                    "pixels_sha256": hashlib.sha256(
                        image.cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                }
                for index, image in enumerate(self.images, 1)
            ],
        }


def flux2_sigmas(steps, width, height, denoise=None):
    """Use the pinned resolution-dependent scheduler, including native denoise slicing."""
    if type(steps) is not int or not 1 <= steps <= 4096:
        raise ValueError("Flux.2 schedule steps must be a whole number from 1 through 4096.")
    if any(type(side) is not int or side <= 0 or side % 16 for side in (width, height)):
        raise ValueError("Flux.2 schedule dimensions must be positive multiples of 16.")
    if denoise is not None and (
        type(denoise) not in (int, float) or not math.isfinite(denoise) or not 0 < denoise <= 1
    ):
        raise ValueError("Flux.2 denoise must be greater than zero and at most one.")
    total = steps if denoise is None else int(steps / denoise)
    if total > 4096:
        raise ValueError("The requested Flux.2 denoise schedule exceeds 4096 native steps.")
    from comfy_extras.nodes_flux import Flux2Scheduler

    return Flux2Scheduler.execute(total, width, height)[0][-steps - 1 :]
