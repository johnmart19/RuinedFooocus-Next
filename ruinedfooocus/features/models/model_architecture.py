"""Ask the installed backend to classify tensor shapes without loading weights."""

from functools import lru_cache
from copy import deepcopy
from pathlib import Path


def inspect_model_file(path):
    """Signature-cached native classification of safe tensor headers, never weights."""
    from ruinedfooocus.features.models.model_identity import file_signature

    path = Path(path).resolve()
    before = tuple(file_signature(path))
    result = _inspect_model_file(str(path), before)
    if tuple(file_signature(path)) != before:
        raise OSError("Model changed during native header inspection")
    result = deepcopy(result)
    from ruinedfooocus.features.generation.bernini_image import primary_family

    if family := primary_family(path):
        result["architecture"] = family
    return result


@lru_cache(maxsize=256)
def _inspect_model_file(path, signature):
    from ruinedfooocus.features.models.model_identity import cached_safetensors_header

    if Path(path).suffix.lower() == ".gguf":
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        header, _ = read_gguf_header(path)
    else:
        header = cached_safetensors_header(path)
    metadata = header.get("__metadata__")
    from ruinedfooocus.features.models.native_pixel_sources import source_error

    if error := source_error(header):
        return {
            "unsupported_reason": error,
            "metadata": metadata.copy() if isinstance(metadata, dict) else {},
        }
    details = inspect_tensor_model(header, include_config=True)
    return (details or {}) | {"metadata": metadata.copy() if isinstance(metadata, dict) else {}}


def detect_tensor_architecture(header):
    result = inspect_tensor_model(header)
    return result["architecture"] if result is not None else None


def inspect_tensor_model(header, *, include_config=False):
    """Native architecture and declared embedded prefixes, using only meta tensors."""
    from ruinedfooocus.features.models.native_pixel_sources import source_error

    if source_error(header):
        return None
    import torch
    from comfy import model_detection

    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(type(size) is not int or size < 0 or size > 2**31 for size in shape)
        ):
            continue
        try:
            tensors[name] = torch.empty(shape, device="meta")
        except (RuntimeError, OverflowError):
            continue
    if not tensors:
        return None
    metadata = header.get("__metadata__")
    metadata = metadata if isinstance(metadata, dict) else None
    prefixes = dict.fromkeys(
        (
            model_detection.unet_prefix_from_state_dict(tensors),
            "",
            "model.diffusion_model.",
            "net.",
            "model.",
        )
    )
    errors = (KeyError, IndexError, ValueError, TypeError, RuntimeError, AttributeError)

    def label(supported):
        name = type(supported).__name__
        config = supported.unet_config
        if name == "Lumina2" and "clip_text_dim" in config:
            return "NewBieImage"
        if name == "Flux2":
            return {20: "Flux2Klein4B", 24: "Flux2Klein9B"}.get(
                config.get("depth_single_blocks"), "Flux2"
            )
        return {
            "SD15": "SD 1.5",
            "SD20": "SD 2.0",
            "SDXL": "SDXL 1.0",
            "Flux": "Flux.1 D",
            "FluxSchnell": "Flux.1 S",
            "PixArtAlpha": "PixArt",
            "PixArtSigma": "PixArt",
            "HunyuanVideo": "Hunyuan Video",
        }.get(name, name)

    def result(supported):
        details = {
            "architecture": label(supported),
            "vae_prefixes": tuple(getattr(supported, "vae_key_prefix", ())),
            "text_encoder_prefixes": tuple(getattr(supported, "text_encoder_key_prefix", ())),
        }
        if include_config:
            keys = (
                "image_model",
                "use_additional_t_cond",
                "default_ref_method",
                "hidden_size",
                "num_layers",
                "in_channels",
                "patch_size",
                "model_dim",
                "visual_embed_dim",
                "flow_map",
                "depth",
                "depth_single_blocks",
                "stable_cascade_stage",
                "c_cond",
                "c_hidden",
                "lq_latent_channels",
                "latent_spatial_down_factor",
                "decoder_hidden_size",
                "decoder_num_res_blocks",
                "decoder_max_freqs",
                "decoder_in_channels",
                "use_x0",
                "pdd_steps",
                "latents_dim",
                "text_dim",
            )
            details["native_class"] = type(supported).__name__
            details["native_config"] = {
                key: supported.unet_config[key] for key in keys if key in supported.unet_config
            }
            if (
                type(supported).__name__ == "Stable_Cascade_C"
                and supported.unet_config.get("c_cond") == 2048
            ):
                for prefix in prefixes:
                    mapper = tensors.get(prefix + "clip_txt_mapper.weight")
                    output = tensors.get(prefix + "clf.1.weight")
                    if (
                        mapper is not None
                        and tuple(mapper.shape) == (2048, 1280)
                        and output is not None
                        and tuple(output.shape) == (16, 2048, 1, 1)
                    ):
                        details["native_variant"] = "full-c"
                        break
            if type(supported).__name__ == "Flux3":
                # The native detector distinguishes flow-map weights by presence.
                # Qualifying a recipe additionally checks the conditioning geometry.
                hidden = supported.unet_config.get("hidden_size")
                for prefix in prefixes:
                    image = tensors.get(prefix + "content_mode_blocks.image.0.attn_out.weight")
                    modulation = tensors.get(prefix + "single_stream_modulations.txt.lin.weight")
                    target = tensors.get(prefix + "ref_time_in.in_layer.weight")
                    if (
                        type(hidden) is int
                        and hidden > 0
                        and image is not None
                        and tuple(image.shape) == (hidden, hidden)
                        and modulation is not None
                        and tuple(modulation.shape) == (3 * hidden, hidden)
                        and target is not None
                        and tuple(target.shape) == (hidden, 256)
                    ):
                        details["native_variant"] = "flow_map"
                        break
        return details

    for prefix in prefixes:
        try:
            supported = model_detection.model_config_from_unet(tensors, prefix, metadata=metadata)
            if supported is not None:
                return result(supported)
        except errors:
            # Partial/unsupported headers are a normal metadata miss.
            continue
    # Match native sd.load_diffusion_model_state_dict's conversion/fallback order.
    # Native conversion mutates the mapping; meta tensors and caller headers stay owned.
    for prefix in prefixes:
        candidate = {
            name[len(prefix) :]: value for name, value in tensors.items() if name.startswith(prefix)
        }
        if not candidate:
            continue
        try:
            converted = model_detection.convert_diffusers_mmdit(dict(candidate))
            supported = (
                model_detection.model_config_from_unet(converted, "", metadata=metadata)
                if converted is not None
                else model_detection.model_config_from_diffusers_unet(candidate)
            )
            if supported is not None:
                return result(supported)
        except errors:
            continue
    return None
