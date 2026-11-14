"""Image component inventory and explicit catalog downloads; no tensor runtime."""

from pathlib import Path
import shutil

from ruinedfooocus.features.models.resource_defaults import resource_name
from ruinedfooocus.core.model_folders import model_folders
from ruinedfooocus.core.model_formats import image_vae_format_error, is_safetensors
from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES
from ruinedfooocus.features.models.model_identity import read_safetensors_header
from ruinedfooocus.features.models.model_registry import (
    RegistryError,
    load_registry,
    unique_payload_bytes,
)
from ruinedfooocus.features.models.image_scope import retired_family, unsupported_tensor_family

# Resource keys mirror the image backend's architecture configurations.
_COMPONENT_KEYS = {
    "Anima": (("clip_qwen3_06b",), "vae_qwen_image"),
    "AuraFlow": (("clip_aura",), "vae_auraflow"),
    "BaseModel": (("clip_l",), "vae_sd"),
    "SD20": (("clip_sd4x_h",), "vae_sd"),
    "Boogu": (("clip_qwen3vl_8b_scaled",), "vae_flux"),
    "Chroma": (("clip_t5",), "vae_flux"),
    "ChromaRadiance": (("clip_t5",), "pixel_space"),
    "CosmosPredict2": (("clip_oldt5",), "vae_wan"),
    "ErnieImage": (("clip_ministral3",), "vae_flux2"),
    "Fibo2": (("clip_qwen3vl_4b",), "vae_flux2"),
    "Flux": (("clip_l", "clip_t5"), "vae_flux"),
    "Flux2": (("clip_mistral3",), "vae_flux2"),
    "Flux2Klein4B": (("clip_qwen3_4b",), "vae_flux2"),
    "Flux2Klein9B": (("clip_qwen3_8b",), "vae_flux2"),
    "Flux3": (("clip_qwen3vl_4b",), "vae_flux2"),
    "HiDream": (("clip_g", "clip_l", "clip_llama", "clip_t5"), "vae_flux"),
    "HiDreamO1": ((), "pixel_space"),
    "HunyuanDiT": (("clip_hydit_bert", "clip_hydit_mt5"), "vae_sdxl"),
    "HunyuanImage21": (("clip_qwen25", "clip_hunyuan_byt5"), "vae_hunyuan_image21"),
    "HunyuanImage21Refiner": (("clip_qwen25",), "vae_hunyuan_refiner"),
    "Ideogram4": (("clip_qwen3vl_8b",), "vae_flux2"),
    "JoyImage": (("clip_joyimage_8b",), "vae_wan"),
    "Kandinsky5Image": (("clip_qwen25", "clip_l"), "vae_flux"),
    "Krea2": (("clip_qwen3vl_4b_scaled",), "vae_qwen_image"),
    "Lens": (("clip_lens_gpt_oss",), "vae_flux2"),
    "Lumina2": (("clip_gemma",), "vae_lumina2"),
    "LongCatImage": (("clip_qwen25",), "vae_flux"),
    "MageFlow": (("clip_qwen3vl_4b",), "vae_mage_flow"),
    "MOSSImage": (("clip_qwen35_9b",), "vae_flux2"),
    "MingImage": (("clip_ming_image",), "vae_ming_image"),
    "MiniMaxH3Image": (("clip_minimax_h3_image",), "vae_minimax_h3_image"),
    "NewBieImage": (("clip_gemma3", "clip_jina"), "vae_flux"),
    "Omnigen2": (("clip_omnigen2_3b",), "vae_flux"),
    "PixArt": (("clip_t5",), "vae_pixart"),
    "PixelDiTT2I": (("clip_gemma2_it_elm",), "pixel_space"),
    "PiD": (("clip_gemma2_it_elm",), "pixel_space"),
    "QwenImage": (("clip_qwen25",), "vae_qwen_image"),
    "QwenImage21": (("clip_qwen_image21",), "vae_qwen_image21"),
    "SenseNovaU15": ((), "pixel_space"),
    "SD3": (("clip_l", "clip_g", "clip_t5"), "vae_sd3"),
    "SD21UNCLIP": (("clip_sd4x_h",), "vae_sd"),
    "StableCascade_C": (("clip_g",), "vae_cascade_c"),
    "SDXL": (("clip_l", "clip_g"), "vae_sdxl"),
    "SDXLRefiner": (("clip_g",), "vae_sdxl"),
    "SD15_instructpix2pix": (("clip_l",), "vae_sd"),
    "SDXL_instructpix2pix": (("clip_l", "clip_g"), "vae_sdxl"),
    "SD_X4Upscaler": (("clip_sd4x_h",), "vae_sd4x"),
    "BerniniImage": (("clip_bernini_umt5",), "vae_bernini"),
    "ZImage": (("clip_qwen3_4b",), "vae_lumina2"),
    "ZImagePixelSpace": (("clip_qwen3_4b",), "pixel_space"),
}
IMAGE_COMPONENTS = {
    name: {"clip_names": list(clips), "vae_name": vae}
    for name, (clips, vae) in _COMPONENT_KEYS.items()
}
ALIASES = {
    "joyimage": "JoyImage",
    "Stable_Cascade_C": "StableCascade_C",
    "SD21UnclipL": "SD21UNCLIP",
    "SD21UnclipH": "SD21UNCLIP",
    "HunyuanDiT1": "HunyuanDiT",
    "SD 1.5": "BaseModel",
    "SD 2.0": "SD20",
    "SDXL 1.0": "SDXL",
    "SSD1B": "SDXL",
    "KOALA_700M": "SDXL",
    "KOALA_1B": "SDXL",
    "Segmind_Vega": "SDXL",
    "CosmosT2IPredict2": "CosmosPredict2",
    "Pony": "SDXL",
    "Illustrious": "SDXL",
    "NoobAI": "SDXL",
    "Pony XL": "SDXL",
    "Illustrious XL": "SDXL",
    "NoobAI XL": "SDXL",
    "Animagine XL": "SDXL",
    "Animagine XL 3.1": "SDXL",
    "Flux.1 D": "Flux",
    "Flux.1 S": "Flux",
    "FluxInpaint": "Flux",
    "SD 3": "SD3",
    "SD3.5": "SD3",
    "SD 3.5": "SD3",
    "SD 3.5 Large": "SD3",
    "Qwen": "QwenImage",
    "Qwen-Image": "QwenImage",
    "MOSS Image": "MOSSImage",
    "MOSS-Image": "MOSSImage",
    "Qwen Image": "QwenImage",
    "Krea 2": "Krea2",
    "Mage-Flow": "MageFlow",
    "Mage Flow": "MageFlow",
    "Lumina 2": "Lumina2",
    "Z-Image": "ZImage",
    "Z-Image Turbo": "ZImage",
    "HiDream O1": "HiDreamO1",
    "FLUX.1 dev": "Flux",
    "FLUX.1 schnell": "Flux",
    "Flux.2": "Flux2",
    "Flux.3 Klein": "Flux3",
}
# Only unambiguous GGUF tags: e.g. lumina2 and qwen_image cover several
# architectures and need the initialized backend's tensor-shape classifier.
GGUF_ARCHITECTURES = {
    "sdxl": "SDXL",
    "sd3": "SD3",
    "aura": "AuraFlow",
    "hidream": "HiDream",
    "cosmos": "CosmosPredict2",
    "moss_image": "MOSSImage",
    "fibo2": "Fibo2",
    "ideogram": "Ideogram4",
    "krea2": "Krea2",
    "flux3": "Flux3",
    "qwen_image21": "QwenImage21",
}


# Families with an explicit native source-image editing caller.
NATIVE_EDIT_FAMILIES = (
    "Boogu",
    "Krea2",
    "MiniMaxH3Image",
    "PiD",
    "HunyuanImage21Refiner",
    "SD21UNCLIP",
    "MageFlow",
    "Fibo2",
    "JoyImage",
    "QwenImage",
    "QwenImage21",
    "HiDreamO1",
    "SD15_instructpix2pix",
    "SDXL_instructpix2pix",
    "SD_X4Upscaler",
    "BerniniImage",
    "StableCascade_C",
)


def native_edit_supported(family, name="", moss_format="Disabled"):
    """Whether a checkpoint can edit an image natively; shared by the UI and the API."""
    family = family or ""
    canonical = ALIASES.get(family, family)
    return (
        canonical in NATIVE_EDIT_FAMILIES
        or canonical in ("Flux2", "Flux2Klein4B", "Flux2Klein9B", "LongCatImage")
        or canonical == "MOSSImage"
        and moss_format != "Disabled"
        or "kontext" in f"{name} {family}".lower()
    )


IMG2IMG_DENOISE = 0.25  # Gentle default: keeps the source recognisable.


def edit_control(family, name="", moss_format="Disabled", denoise=IMG2IMG_DENOISE):
    """How to edit a supplied image: natively when possible, else image-to-image."""
    if native_edit_supported(family, name, moss_format):
        return {"type": "edit"}
    return {"type": "img2img", "denoise": denoise}


def inventory(checkpoint, family, settings, paths, component_overrides=None):
    """Resolve local image dependencies. This function never downloads or hashes."""
    from ruinedfooocus.backends.comfy.component_selection import (
        component_settings,
        normalize_component_overrides,
        validate_component_slots,
    )

    choices = normalize_component_overrides(component_overrides)
    settings = component_settings(settings, choices)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("Select an installed image checkpoint")
    tensors = read_safetensors_header(checkpoint) if is_safetensors(checkpoint) else None
    from ruinedfooocus.features.generation.bernini_image import primary_family

    bernini = primary_family(checkpoint)
    if bernini:
        family = bernini
    excluded = unsupported_tensor_family(tensors) if tensors is not None else None
    if retired_family(family) or excluded and not bernini:
        raise ValueError("This panel supports image checkpoints only")
    gguf_architecture = None
    is_gguf = checkpoint.suffix.lower() == ".gguf"
    if is_gguf:
        # Read shape metadata before choosing dependencies; cached family can be stale.
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        tensors, arch = read_gguf_header(checkpoint)
        if arch in ("wan", "ltxv", "minimax_h3") or unsupported_tensor_family(tensors):
            raise ValueError("This panel supports image checkpoints only")
        gguf_architecture = GGUF_ARCHITECTURES.get(arch)
    detected = None
    verified_source = None
    if tensors is None and checkpoint.suffix.lower() in NATIVE_TORCH_SUFFIXES:
        from ruinedfooocus.features.models.model_identity import verified_native_source
        import json

        cache_root = getattr(paths, "model_paths", {}).get("cache_path")
        if cache_root is not None:
            try:
                cached = (Path(cache_root) / "checkpoints" / checkpoint.name).with_suffix(".json")
                metadata = json.loads(cached.read_text(encoding="utf-8"))
                verified_source = verified_native_source(checkpoint, metadata)
                if verified_source is not None:
                    family = verified_source["native_family"]
                    tensors = {}
            except (OSError, ValueError, TypeError):
                verified_source = None
    # The detector creates meta tensors, but imports backend/device modules.
    # Reuse it only in an initialized app; standalone inventory stays Torch-free.
    native_info = None
    from ruinedfooocus.features.models.model_architecture import native_detector_ready

    if tensors is not None and native_detector_ready():
        from ruinedfooocus.features.models.model_architecture import inspect_tensor_model

        native_info = inspect_tensor_model(tensors)
        detected = native_info["architecture"] if native_info is not None else None
        if bernini:
            detected = bernini
    if retired_family(detected) or (
        is_gguf and arch == "hyvid" and detected not in ("HunyuanImage21", "HunyuanImage21Refiner")
    ):
        raise ValueError("This panel supports image checkpoints only")
    canonical = (detected or gguf_architecture) if is_gguf else (detected or family)
    architecture = ALIASES.get(canonical, canonical)
    if architecture not in IMAGE_COMPONENTS:
        if is_gguf:
            raise ValueError(
                "GGUF architecture cannot be identified safely; component downloads are disabled"
            )
        raise ValueError("Image architecture is unknown; refresh local model metadata first")
    if tensors is None:
        raise ValueError("Component inspection supports safetensors and image GGUF checkpoints")
    names = list(tensors or {})
    embedded_clip = any(
        name.startswith(("cond_stage_model.", "conditioner.embedders.", "text_encoders."))
        for name in names
    )
    embedded_vae = any(name.startswith("first_stage_model.") for name in names)
    if native_info is not None:
        embedded_clip = embedded_clip or any(
            name.startswith(native_info["text_encoder_prefixes"]) for name in names
        )
        embedded_vae = embedded_vae or any(
            name.startswith(native_info["vae_prefixes"]) for name in names
        )
    if verified_source is not None:
        bundled = verified_source.get("bundled_components", ())
        embedded_clip = "clip" in bundled
        embedded_vae = "vae" in bundled
    clips, vae = (
        IMAGE_COMPONENTS[architecture]["clip_names"],
        IMAGE_COMPONENTS[architecture]["vae_name"],
    )
    from ruinedfooocus.features.models.native_component_layout import current_layout

    layout = (
        current_layout(checkpoint, paths, family=architecture)
        if architecture == "QwenImage"
        else None
    )
    if layout is not None:
        clips, vae = layout.clip_roles, layout.vae_role
    validate_component_slots(
        choices,
        {"clip_keys": clips, "vae_key": vae},
        bundled_clip=embedded_clip,
        bundled_vae=embedded_vae,
    )
    from ruinedfooocus.features.models.component_candidates import inspect_local_component
    from ruinedfooocus.features.models.approximate_codecs import is_choice, inventory_row

    rows = []
    for key in (*clips, *(() if vae == "pixel_space" else (vae,))):
        if is_choice(choices.get(key)):
            rows.append(
                dict(inventory_row(key, choices[key], paths), embedded_fallback=embedded_vae)
            )
            continue
        category = "clip" if key.startswith("clip_") else "vae"
        embedded = embedded_clip if category == "clip" else embedded_vae
        # Native bundled slots take priority. A profile-wide override can still
        # serve another checkpoint, but is not the effective component here.
        filename = "Built into checkpoint" if embedded else resource_name(key, settings)
        override = bool(settings.get(key)) and not embedded
        format_error = (
            image_vae_format_error(filename) if category == "vae" and not embedded else None
        )
        roots = model_folders(paths.paths, category, paths.get_abspath)
        found = (
            None
            if embedded
            else next((root / filename for root in roots if (root / filename).is_file()), None)
        )
        catalog = next(
            (
                (name, item)
                for name, item in paths.DOWNLOADABLE_FILES.items()
                if item.get("path") == "path_" + category and item.get("filename") == filename
            ),
            None,
        )
        download_key = None
        if catalog and roots and not format_error and not Path(filename).is_absolute():
            target = (roots[0] / filename).resolve()
            if target.is_relative_to(roots[0]) and not embedded and not override:
                download_key = catalog[0]
        inspected = (
            {"compatibility": "incompatible", "reason": format_error}
            if format_error
            else inspect_local_component(key, found)
            if found
            else {}
        )
        rows.append(
            dict(
                key=key,
                filename=filename,
                category=category,
                override=override,
                status="Embedded in checkpoint"
                if embedded
                else "Installed"
                if found
                else "Missing",
                location=str(
                    checkpoint
                    if embedded
                    else found or (roots[0] / filename if roots else filename)
                ),
                source=catalog[1].get("url", "") if catalog and not embedded else "",
                bytes=found.stat().st_size if found else None,
                download_key=download_key,
                download_available=bool(download_key and not found and not embedded),
                compatibility="embedded" if embedded else inspected.get("compatibility", "unknown"),
                compatibility_reason=inspected.get("reason", ""),
                scaled_fp8=inspected.get("scaled_fp8", False),
            )
        )
    # This image route uses DualCLIPLoaderGGUF for a mixed external encoder set.
    # Its pinned load_data rejects a scaled_fp8 companion before constructing CLIP.
    if any(
        row["category"] == "clip"
        and row["status"] != "Embedded in checkpoint"
        and Path(row["filename"]).suffix.lower() == ".gguf"
        for row in rows
    ):
        for row in rows:
            if row["category"] == "clip" and row["scaled_fp8"]:
                row["compatibility"] = "incompatible"
                row["compatibility_reason"] = (
                    "The native mixed GGUF encoder loader cannot combine GGUF and scaled FP8 "
                    "encoders. Choose an encoder set supported by the same loader."
                )
    missing = []
    try:
        registry = load_registry()
    except RegistryError:
        registry = None
    for row in rows:
        annotation = None
        key = row["download_key"]
        if key and registry is not None:
            try:
                annotation = registry.annotations_for(key, paths.DOWNLOADABLE_FILES[key])
            except RegistryError:
                pass  # Optional metadata never changes native component selection.
        row["published_bytes"] = annotation["artifact"]["bytes"] if annotation else None
        if row["status"] == "Missing":
            missing.append(annotation or {"id": row["key"], "artifact": {"bytes": None}})
    return dict(
        family=family,
        architecture=architecture,
        native_component_layout=layout.metadata() if layout is not None else None,
        rows=rows,
        missing_payload_bytes=unique_payload_bytes(missing),
        note=("Pixel-space model: no VAE required. " if vae == "pixel_space" else "")
        + "Compatibility uses local architecture metadata. Unverified files are not qualified "
        "alternatives, and local presence does not verify checksums. "
        "Missing custom overrides require a local file, or reset the component to Automatic in Settings.",
    )


def download_missing(
    checkpoint, family, settings, paths, *, offline=False, progress=None, component_overrides=None
):
    """Explicit user action. Recompute selected image requirements; accept no client file list."""
    if offline:
        raise ValueError("Offline mode: component downloads are disabled")
    state = inventory(checkpoint, family, settings, paths, component_overrides=component_overrides)
    downloaded = []
    for row in state["rows"]:
        if row["download_available"]:
            if Path(row["location"]).exists():
                continue
            if progress:
                progress(0, desc="Downloading " + row["filename"])

            def update(received, total, speed):
                if progress:
                    progress(
                        received / total if total else 0, desc="Downloading " + row["filename"]
                    )

            paths.download_file(row["download_key"], progress=update)
            downloaded.append(row["filename"])
    return downloaded


def disk_free(paths):
    root = paths.paths.get("path_clip", [])
    root = root[0] if isinstance(root, list) and root else root
    path = paths.get_abspath(root)
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free
