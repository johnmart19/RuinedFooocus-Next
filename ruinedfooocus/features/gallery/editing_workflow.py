"""Source-local Library editing over the same request capture used by Create."""

from copy import deepcopy
import hashlib

import numpy as np
from PIL import Image, ImageOps

from ruinedfooocus.features.gallery.actions import source_path


def open_editor(record, metadata=None, *, owner=None, store=None):
    """Adopt a trusted Library selection; saved generation context remains inherited."""
    from ruinedfooocus.features.gallery.editor_session import default_store
    from ruinedfooocus.features.gallery.image_metadata import read_metadata

    return (store or default_store()).start(
        source_path(record), provenance=read_metadata(metadata) or {}, owner=owner
    )


def editor_source_record(context, *, owner=None, store=None):
    """Resolve a current opaque source before reading pixels or preparing resources."""
    from ruinedfooocus.features.gallery.actions import source_record
    from ruinedfooocus.features.gallery.editor_session import default_store

    if not isinstance(context, dict) or not {"session_id", "revision", "source_id"} <= set(context):
        raise ValueError("Choose the current owned editor image.")
    selected = store or default_store()
    snapshot = selected.snapshot(context["session_id"], owner=owner)
    ref = selected.artifact(
        context["session_id"], context["source_id"], revision=context["revision"], owner=owner
    )
    if ref.kind != "image" or ref.artifact_id != snapshot.current.artifact_id:
        raise ValueError("Choose the current owned editor image.")
    return source_record(ref.path)


def _owned_library_mask(context, *, owner, store):
    from ruinedfooocus.features.gallery.editor_session import default_store

    selected = store or default_store()
    if context.get("mask_id") is None:
        raise ValueError("Adopt an owned selection mask before editor inpainting.")
    source = selected.artifact(
        context["session_id"], context["source_id"], revision=context["revision"], owner=owner
    )
    mask = selected.artifact(
        context["session_id"], context["mask_id"], revision=context["revision"], owner=owner
    )
    if (
        mask.kind != "mask"
        or mask.source_image_id != source.artifact_id
        or mask.source_frame_id != source.frame_id
        or mask.frame.size != source.frame.size
    ):
        raise ValueError("The owned selection mask belongs to another source frame.")
    with Image.open(mask.path) as original:
        selection = ImageOps.exif_transpose(original).convert("L").copy()
    if mask.inverted:
        selection = ImageOps.invert(selection)
    selected.artifact(
        context["session_id"], mask.artifact_id, revision=context["revision"], owner=owner
    )
    layer = Image.new("RGBA", selection.size, (255, 255, 255, 0))
    layer.putalpha(selection)
    return {"layers": [layer]}


def working_source(record):
    """Prepare a private working image from the validated original, never its preview."""
    path = source_path(record)
    image, details = working_image(path)
    source_path(record)
    return image, details


def working_image(path):
    """Decode owned/trusted pixels as sRGB8 for model inputs, without changing files."""
    from io import BytesIO
    from pathlib import Path

    from ruinedfooocus.features.image_processing.native_image_formats import (
        MAX_BYTES,
        MAX_PIXELS,
        convert_color,
        needs_preview,
        read_image,
    )

    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose a still image no larger than 512 MiB.")
    if needs_preview(path):
        imported = read_image(path)
        pixels = convert_color(imported.pixels, imported.color_space, "sRGB")
        array = (pixels.clamp(0, 1).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)
        image.info.update(getattr(imported, "metadata", {}))
        details = dict(
            original_dtype=str(imported.pixels.dtype),
            original_color_space=imported.color_space,
            conversion="native color conversion from original; image tools use an 8-bit working image",
        )
    else:
        with Image.open(path) as original:
            if (
                getattr(original, "n_frames", 1) != 1
                or not 0 < original.width * original.height <= MAX_PIXELS
            ):
                raise ValueError("Choose a still image for image tools.")
            image = ImageOps.exif_transpose(original).copy()
            pixels = np.asarray(image)
            dtype = str(pixels.dtype)
            if pixels.dtype != np.uint8:
                pixels = convert_color(pixels, "sRGB", "sRGB")
                array = (pixels.clamp(0, 1).numpy() * 255).astype(np.uint8)
                image = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)
                image.info.update(original.info)
        details = dict(
            original_dtype=dtype,
            original_color_space="sRGB",
            conversion="none" if dtype == "uint8" else "image tools use an 8-bit working image",
        )
    # Native codecs retain high precision above. Pillow's CMS applies embedded
    # profiles to the final model-input copy; alpha is not a color channel.
    profile = image.info.get("icc_profile")
    if profile is None and path.suffix.lower() in (".png", ".avif"):
        with Image.open(path) as original:
            profile = original.info.get("icc_profile")
    if profile and details["original_color_space"] == "sRGB":
        from PIL import ImageCms

        alpha = (
            image.convert("RGBA").getchannel("A")
            if "A" in image.getbands() or "transparency" in image.info
            else Image.new("L", image.size, 255)
        )
        information = dict(image.info)
        rgb = image.convert("RGB") if image.mode in ("RGBA", "P", "LA") else image
        destination = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        try:
            source_profile = ImageCms.ImageCmsProfile(BytesIO(profile))
            if source_profile.profile.xcolor_space.strip() == "RGB":
                rgb = rgb.convert("RGB")
            image = ImageCms.profileToProfile(
                rgb, source_profile, destination, outputMode="RGB"
            ).convert("RGBA")
        except (ImageCms.PyCMSError, OSError, TypeError, ValueError) as error:
            raise ValueError(
                "The image's embedded color profile could not be converted to sRGB."
            ) from error
        image.putalpha(alpha)
        image.info.update(information, icc_profile=destination.tobytes())
        details.update(
            original_color_space="embedded ICC",
            conversion="native precision conversion and embedded ICC to sRGB; image tools use an 8-bit working image",
        )
    image = image.convert("RGBA").copy()
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    details.update(
        filename=path.name,
        sha256=digest,
        width=image.width,
        height=image.height,
        working_dtype="uint8",
        working_color_space="sRGB",
    )
    return image, details


def _automatic_adjustment_output(data, original, action):
    """Preserve native samples when an Editor request still owns untouched defaults."""
    from ruinedfooocus.features.image_processing.native_image_formats import (
        ImportedImage,
        format_settings,
    )

    if data.get("_editor_output_explicit") is not False:
        return
    choice = format_settings(data)
    if choice != format_settings():
        return
    pixels = original.pixels if isinstance(original, ImportedImage) else np.asarray(original)
    space = original.color_space if isinstance(original, ImportedImage) else "sRGB"
    hdr = action == "image-color" and data.get("color_operation") == "sdr-to-hdr"
    if pixels.dtype == np.uint8 and space == "sRGB" and not hdr:
        return
    scalar = action == "image-mask"
    floating = np.issubdtype(pixels.dtype, np.floating) or space != "sRGB" or hdr
    selected = dict(
        output_format="exr" if floating else "png",
        output_bit_depth="32-bit float" if floating else "16-bit",
        output_color_space="sRGB" if scalar else "HDR" if hdr else space,
        output_crf=choice["output_crf"],
    )
    data.update(format_settings(selected))
    data["image_source"]["output_selection"] = dict(
        policy="automatic precision-preserving Editor output",
        selected=selected,
    )


def build_library_request(
    controls,
    record,
    *,
    prompt,
    negative,
    checkpoint,
    action,
    resets,
    features,
    presets,
    custom,
    strength=None,
    mask=None,
    tool_resource=None,
    prompt_processing=False,
    editor_context=None,
    editor_store=None,
    owner=None,
):
    """Build one explicit operation; no cached source, reference or brush is reused."""
    from ruinedfooocus.features.image_editing.tool_activation import activation_values
    from ruinedfooocus.features.image_editing import model_patches, native_control_recipes
    from ruinedfooocus.features.image_processing import still_image_utilities as utilities

    if not isinstance(prompt, str) or not isinstance(negative, str):
        raise ValueError("Prompts must be text.")
    if editor_context is not None:
        owned = editor_source_record(editor_context, owner=owner, store=editor_store)
        if source_path(owned) != source_path(record):
            raise ValueError("The Library selection does not match the owned editor image.")
        if action == "inpaint":
            mask = _owned_library_mask(editor_context, owner=owner, store=editor_store)
    source_path(record)
    data = deepcopy(controls)
    data.update(deepcopy(resets))
    data.update(model_patches.reset_values())
    from ruinedfooocus.features.image_editing import lora_hooks

    data.update(lora_hooks.DEFAULTS)
    data.update(native_control_recipes.reset_values())
    from ruinedfooocus.features.image_processing.image_composition import KEY, reset_values

    data.update(reset_values())
    data.pop(KEY, None)
    data.update(
        prompt=prompt,
        negative=negative,
        base_model_name=checkpoint,
        preset_selection=None,
        _imported_prompt=None,
        _metadata_prompt=True,
        _apply_prompt_processing=bool(prompt_processing),
        _image_guidance_checkpoint=checkpoint,
        style_selection=deepcopy(controls.get("style_selection", [])) if prompt_processing else [],
        obp_assume_direct_control=False,
        obp_image_prompt=False,
        auto_negative=bool(controls.get("auto_negative")) if prompt_processing else False,
        image_number=1,
        generate_forever=False,
        automatic_upscale_target="Generated image",
        native_edit_recipe="auto",
        gligen_model=None,
        gligen_model_signature=None,
        gligen_layout="[]",
        hypernetwork_model=None,
        hypernetwork_model_signature=None,
        hypernetwork_strength=1.0,
    )
    if checkpoint != controls.get("base_model_name"):
        # Choices belong to the checkpoint whose settings were captured, not a new model.
        data["image_component_overrides"] = {}
        data["_image_component_scope"] = None
        data["_image_guidance_checkpoint"] = checkpoint
    if action == "generate":
        data["cn_selection"] = "None"
        return _capture_editor(data, editor_context, action, owner=owner, store=editor_store)
    if action in utilities.ADJUSTMENT_KINDS:
        from ruinedfooocus.features.image_processing.native_image_formats import ImportedImage

        path = source_path(record)
        original = utilities.read_adjustment_source(path)
        pixels = original.pixels if isinstance(original, ImportedImage) else np.asarray(original)
        dtype = str(pixels.dtype)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        source_path(record)
        data.update(activation_values(action, presets=presets, custom=custom, features=features))
        data.update(
            input_image=original,
            controlnet={"type": action},
            automatic_upscale="Off",
            _native_main_source=None,
            image_source=dict(
                filename=path.name,
                sha256=digest,
                width=pixels.shape[1],
                height=pixels.shape[0],
                original_dtype=dtype,
                working_dtype=dtype,
                original_color_space=original.color_space
                if isinstance(original, ImportedImage)
                else "sRGB",
                conversion="none",
            ),
        )
        if editor_context is not None:
            _automatic_adjustment_output(data, original, action)
        for field in getattr(utilities._handler(action), "IMAGE_FIELDS", ()):
            data[field] = controls.get(field)
        if (
            utilities.uses_owned_selection(action, data)
            and editor_context
            and editor_context.get("mask_id")
            and data.get("mask_image") is None
        ):
            mask = editor_store.artifact(
                editor_context["session_id"],
                editor_context["mask_id"],
                revision=editor_context["revision"],
                owner=owner,
            )
            if mask.source_image_id != editor_context["source_id"]:
                raise ValueError("The selection mask belongs to another image frame.")
            data["mask_image"] = mask.path
        data.pop("base_model_name", None)
        data.pop("base_model_hash", None)
        return _capture_editor(data, editor_context, action, owner=owner, store=editor_store)
    image, details = working_source(record)
    data.update(main_view=image.copy(), input_image=image.copy(), image_source=details)
    if action == "inpaint":
        if not features.get("inpaint"):
            raise ValueError("Select a checkpoint with a supported inpainting route.")
        from ruinedfooocus.features.image_editing.inpaint_utils import source_and_mask

        source_and_mask(image, mask, preserve_alpha=bool(features.get("qwen_local")))
        data.update(inpaint_toggle=True, inpaint_view=deepcopy(mask), cn_selection="None")
    elif action == "upscale":
        data.update(upscale_upload=image.copy(), automatic_upscale_target="Input image only")
        if tool_resource is not None:
            data["automatic_upscale"] = tool_resource
        if data.get("automatic_upscale", "Off") == "Off":
            data["automatic_upscale"] = "2× General (RealESRGAN)"
    elif action in ("edit", "rembg", "depth-map", "restore-seedvr2"):
        kind = "img2img" if action == "edit" and not features.get("native_edit") else action
        data.update(activation_values(kind, presets=presets, custom=custom, features=features))
        if action in ("rembg", "depth-map") and tool_resource is not None:
            data["rembg_native_model" if action == "rembg" else "native_vision_model"] = (
                tool_resource
            )
        if kind == "img2img" and strength is not None:
            data["cn_strength"] = strength
        if action == "restore-seedvr2":
            if tool_resource is not None:
                data["seedvr2_model"] = tool_resource
            data.update(prompt="", negative="", automatic_upscale="Off")
            data.pop("base_model_name", None)
            data.pop("base_model_hash", None)
    else:
        raise ValueError("Unknown image action.")
    return _capture_editor(data, editor_context, action, owner=owner, store=editor_store)


def _capture_editor(data, context, operation, *, owner, store):
    if context is None:
        return data
    from ruinedfooocus.features.gallery.editor_session import capture_request

    return capture_request(data, context, operation, owner=owner, store=store)
