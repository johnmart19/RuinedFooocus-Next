"""Qwen Image 2.1 paired conditioning through the pinned native node."""

from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder
from ruinedfooocus.features.image_editing.image_geometry import prepare_reference, reference_plan


def text_to_image_conditioning(clip, prompt, negative_prompt, width, height):
    """Keep native T2I flags and metadata while preserving the requested target grid.

    No reference images are supplied here. The native joint node creates a square
    empty latent; native interpolation reshapes that zero tensor to the request's
    non-square grid without changing channels, device or dtype.
    """
    if any(type(side) is not int or side <= 0 or side % 16 for side in (width, height)):
        raise ValueError("Qwen Image 2.1 dimensions must be positive multiples of 16 pixels.")
    from comfy.utils import common_upscale
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21

    warm_patched_text_encoder(clip)
    positive, negative, latent = TextEncodeQwenImage21.execute(
        clip=clip,
        prompt=prompt,
        negative_prompt=negative_prompt,
        vae=None,
        resolution=width,
        images={},
    )
    if latent["samples"].shape[-2:] != (height // 16, width // 16):
        latent = latent.copy()
        latent["samples"] = common_upscale(
            latent["samples"], width // 16, height // 16, "nearest-exact", "disabled"
        )
    return positive, negative, latent


def _prepare_reference(image, size, mode):
    from PIL import Image

    if not isinstance(image, Image.Image) or not 0 < image.width * image.height <= 16777216:
        raise ValueError("Each Qwen Image 2.1 reference must be an image of at most 16 megapixels.")
    plan = reference_plan(image.size, size, mode, multiple=32)
    if max(plan["canvas_width"], plan["canvas_height"]) > 4096:
        raise ValueError("Qwen Image 2.1 reference canvases must be at most 4096 pixels per side.")
    return prepare_reference(image, size, mode, multiple=32, preserve_alpha=True)


class QwenImage21References:
    """Own one edit's aligned image tensors and share their VAE results across prompt switches."""

    def __init__(self, vae, primary, additional, size, mode, *, paint_mask=None, edit_mode=None):
        if len(additional) > 15:
            raise ValueError(
                "Use at most 16 Qwen Image 2.1 references including the primary image."
            )
        if (
            getattr(vae, "latent_channels", None) != 64
            or getattr(vae, "latent_dim", None) != 2
            or getattr(vae, "output_channels", None) != 4
            or vae.spacial_compression_encode() != 16
            or vae.spacial_compression_decode() != 16
        ):
            raise ValueError(
                "Qwen Image 2.1 reference editing requires its 64-channel 16× RGBA VAE."
            )
        from ruinedfooocus.features.image_editing.image_guidance import reference_image_map

        self._composite = None
        primary_size = primary.size if paint_mask is not None and mode == "preserve" else size
        pixels, self.frame = _prepare_reference(primary, primary_size, mode)
        if paint_mask is not None:
            from ruinedfooocus.features.image_editing.qwen_image21_local_edit import (
                local_edit_mode,
                prepare_paint_guide,
            )

            edit_mode = local_edit_mode(edit_mode)
            original = pixels
            pixels, coverage = prepare_paint_guide(original, paint_mask, self.frame)
            self.frame["qwen_local_edit_mode"] = edit_mode
            if edit_mode == "mask":
                self._composite = original, coverage
        self.images = reference_image_map(
            pixels, additional, lambda image: _prepare_reference(image, size, "preserve")[0]
        )
        self.vae = vae
        self._grids = {
            image.data_ptr(): (1, 64, image.shape[1] // 16, image.shape[2] // 16)
            for image in self.images.values()
        }
        self._latents = {}

    def finish(self, decoded):
        if self._composite is None:
            return decoded
        from ruinedfooocus.features.image_editing.qwen_image21_local_edit import (
            composite_local_edit,
        )

        return composite_local_edit(decoded, *self._composite)

    def encode(self, pixels):
        # resolution=0 preserves these 32-aligned inputs. The node's batch slice
        # shares storage with the corresponding prepared reference.
        key = pixels.data_ptr()
        if key not in self._grids:
            raise ValueError("The native Qwen Image 2.1 node changed an aligned reference canvas.")
        if key not in self._latents:
            latent = self.vae.encode(pixels)
            if tuple(latent.shape) != self._grids[key]:
                raise ValueError("The Qwen Image 2.1 VAE returned an incompatible reference grid.")
            self._latents[key] = latent
        return self._latents[key]

    def conditioning(self, clip, prompt, negative_prompt):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        warm_patched_text_encoder(clip)
        positive, negative, latent = TextEncodeQwenImage21.execute(
            clip=clip,
            prompt=prompt,
            negative_prompt=negative_prompt,
            vae=self,
            resolution=0,
            images=self.images,
        )
        if tuple(latent["samples"].shape) != self._grids[self.images["image_1"].data_ptr()]:
            raise ValueError(
                "The native Qwen Image 2.1 edit target must match its first reference grid."
            )
        if any(
            not rows
            or any(len(meta.get("image_slots", ())) != len(self.images) for _, meta in rows)
            for rows in (positive, negative)
        ):
            raise ValueError(
                "The Qwen Image 2.1 text encoder did not produce the required image slots."
            )
        return positive, negative, latent
