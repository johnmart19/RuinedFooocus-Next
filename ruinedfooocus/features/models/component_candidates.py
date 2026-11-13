"""Conservative local component alternatives from bounded, cached shape metadata."""

from pathlib import Path
import sys
from types import SimpleNamespace

from ruinedfooocus.core.model_folders import model_folders
from ruinedfooocus.core.model_formats import image_vae_format_error, is_safetensors
from ruinedfooocus.features.models.model_identity import cached_safetensors_header

TEXT_TYPES = {
    "clip_bernini_umt5": {"T5_XXL"},
    "clip_l": {"CLIP_L"},
    "clip_sd4x_h": {"CLIP_H"},
    "clip_g": {"CLIP_G"},
    "clip_t5": {"T5_XXL"},
    "clip_oldt5": {"T5_XXL_OLD"},
    "clip_omnigen2_3b": {"QWEN25_3B"},
    "clip_aura": {"T5_XXL"},
    "clip_llama": {"LLAMA3_8"},
    "clip_gemma": {"GEMMA_2_2B"},
    "clip_gemma2_it_elm": {"GEMMA_2_2B"},
    "clip_gemma3": {"GEMMA_3_4B"},
    "clip_jina": {"JINA_CLIP_2"},
    "clip_joyimage_8b": {"QWEN3VL_8B"},
    "clip_qwen25": {"QWEN25_7B"},
    "clip_qwen3_06b": {"QWEN3_06B"},
    "clip_qwen3_4b": {"QWEN3_4B"},
    "clip_qwen3_8b": {"QWEN3_8B"},
    "clip_qwen3vl_4b": {"QWEN3VL_4B"},
    "clip_qwen3vl_4b_scaled": {"QWEN3VL_4B"},
    "clip_qwen3vl_8b": {"QWEN3VL_8B"},
    "clip_qwen3vl_8b_scaled": {"QWEN3VL_8B"},
    "clip_qwen35_9b": {"QWEN35_9B"},
    "clip_ministral3": {"MINISTRAL_3_3B"},
    "clip_mistral3": {"MISTRAL3_24B_PRUNED_FLUX2"},
    "clip_lens_gpt_oss": {"GPT_OSS_20B"},
    "clip_ming_image": {"MING_IMAGE"},
    "clip_hunyuan_byt5": {"BYT5_SMALL_GLYPH"},
    "clip_qwen_image21": {"QWEN3VL_8B"},
    "clip_minimax_h3_image": {"QWEN3VL_32B"},
}
# Native planar AutoencoderKL / AutoencodingEngine, eightfold spatial scale.
# Other VAE formats stay unknown rather than extrapolating from their filenames.
VAE_CHANNELS = {
    "vae_bernini": 16,
    "vae_sd": 4,
    "vae_sd4x": 4,
    "vae_sdxl": 4,
    "vae_pixart": 4,
    "vae_auraflow": 4,
    "vae_flux": 16,
    "vae_sd3": 16,
    "vae_lumina2": 16,
    "vae_qwen_image": 16,
    "vae_qwen_image_layered": 16,
    "vae_wan": 16,
    "vae_flux2": 128,
    "vae_mage_flow": 128,
    "vae_ming_image": 16,
    "vae_qwen_image21": 64,
    "vae_hunyuan_image21": 64,
    "vae_hunyuan_refiner": 32,
    "vae_minimax_h3_image": 24,
}
VAE_LAYOUTS = {key: "planar" for key in VAE_CHANNELS}
VAE_LAYOUTS.update(vae_qwen_image="wan21", vae_wan="wan21")
VAE_LAYOUTS["vae_bernini"] = "wan21"
VAE_LAYOUTS["vae_sd4x"] = "planar4x"
VAE_LAYOUTS.update(vae_flux2="planar_bn_packed", vae_mage_flow="mage")
VAE_LAYOUTS["vae_ming_image"] = "wan21_rgba"
VAE_LAYOUTS["vae_qwen_image_layered"] = "wan21_rgba"
VAE_LAYOUTS.update(vae_qwen_image21="qwen_image21", vae_hunyuan_image21="hunyuan_image21")
VAE_LAYOUTS["vae_hunyuan_refiner"] = "hunyuan_refiner"
VAE_LAYOUTS["vae_minimax_h3_image"] = "minimax_image"


def wan21_rgb_contract(shapes, *, image_channels=3):
    """Native sd.py Wan2.1 branch: RGB/RGBA, 16 latents and three spatial resamples.

    Shapes follow comfy/ldm/wan/vae.py's dim_mult=[1, 2, 4, 4], two residual
    blocks and temporal downsampling [False, True, True]. This does not infer
    compatibility from the shared latent count of unrelated planar/3D VAEs.
    """
    gamma = shapes.get("decoder.head.0.gamma", ())
    if len(gamma) != 4 or gamma[1:] != (1, 1, 1) or gamma[0] <= 0:
        return False
    dim = gamma[0]
    expected = {
        "encoder.conv1.weight": (dim, image_channels, 3, 3, 3),
        "decoder.conv1.weight": (4 * dim, 16, 3, 3, 3),
        "encoder.head.0.gamma": (4 * dim, 1, 1, 1),
        "encoder.head.2.weight": (32, 4 * dim, 3, 3, 3),
        "decoder.head.2.weight": (image_channels, dim, 3, 3, 3),
        "decoder.middle.0.residual.0.gamma": (4 * dim, 1, 1, 1),
        "conv1.weight": (32, 32, 1, 1, 1),
        "conv2.weight": (16, 16, 1, 1, 1),
        "encoder.downsamples.2.resample.1.weight": (dim, dim, 3, 3),
        "encoder.downsamples.5.resample.1.weight": (2 * dim, 2 * dim, 3, 3),
        "encoder.downsamples.8.resample.1.weight": (4 * dim, 4 * dim, 3, 3),
        "encoder.downsamples.5.time_conv.weight": (2 * dim, 2 * dim, 3, 1, 1),
        "encoder.downsamples.8.time_conv.weight": (4 * dim, 4 * dim, 3, 1, 1),
        "decoder.upsamples.3.resample.1.weight": (2 * dim, 4 * dim, 3, 3),
        "decoder.upsamples.7.resample.1.weight": (2 * dim, 4 * dim, 3, 3),
        "decoder.upsamples.11.resample.1.weight": (dim, 2 * dim, 3, 3),
        "decoder.upsamples.3.time_conv.weight": (8 * dim, 4 * dim, 3, 1, 1),
        "decoder.upsamples.7.time_conv.weight": (8 * dim, 4 * dim, 3, 1, 1),
    }
    return (
        all(shapes.get(name) == shape for name, shape in expected.items())
        and "encoder.downsamples.2.time_conv.weight" not in shapes
        and "decoder.upsamples.11.time_conv.weight" not in shapes
        and "bn.running_mean" not in shapes
    )


def header_shapes(path):
    if is_safetensors(path):
        header = cached_safetensors_header(path)
    elif Path(path).suffix.lower() == ".gguf":
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        header, _ = read_gguf_header(path)
    else:
        return {}
    return {
        name: tuple(entry["shape"])
        for name, entry in header.items()
        if name != "__metadata__"
        and isinstance(entry, dict)
        and isinstance(entry.get("shape"), list)
        and len(entry["shape"]) <= 8
        and all(type(size) is int and 0 <= size <= 2**31 for size in entry["shape"])
    }


def classify_shapes(shapes, detector=None):
    """Reuse the initialized native text detector without importing its runtime."""
    result = {}
    result["bernini_umt5"] = (
        shapes.get("shared.weight") == (256384, 4096)
        and shapes.get("encoder.block.23.layer.1.DenseReluDense.wi_1.weight") == (10240, 4096)
    )
    if shapes.get("bert.encoder.layer.0.attention.self.query.weight") == (1024, 1024):
        result["hydit_encoder"] = "bert"
    if shapes.get("encoder.block.23.layer.1.DenseReluDense.wi_1.weight") == (5120, 2048):
        proto = shapes.get("mt5xl.spiece_model", shapes.get("spiece_model", ()))
        result["hydit_encoder"] = (
            "mt5"
            if len(proto) == 1 and 0 < proto[0] <= 16 * 1024 * 1024
            else "mt5_without_tokenizer"
        )
    if detector is None:
        detector = getattr(sys.modules.get("comfy.sd"), "detect_te_model", None)
    if detector is None:
        from ruinedfooocus.features.models.native_text_identity import passive_native_text_type

        native_type = passive_native_text_type(
            shapes,
            allowed_types=frozenset({"WEVISONE_2B", "WEVISONE_4B", "QWEN3VL_2B", "QWEN3VL_32B"}),
        )
        if native_type is not None:
            result["text_type"] = native_type
    if detector is not None:
        detected = detector({name: SimpleNamespace(shape=shape) for name, shape in shapes.items()})
        if detected is not None:
            result["text_type"] = detected.name
            if detected.name in ("GPT_OSS_20B", "MING_IMAGE"):
                tokenizer = shapes.get("tokenizer_json", ())
                result["text_contract_valid"] = (
                    len(tokenizer) == 1 and 0 < tokenizer[0] <= 16 * 1024 * 1024
                )
            clip_contracts = {
                "CLIP_L": ("text_model.encoder.layers.0.mlp.fc1.weight", (3072, 768)),
                "CLIP_H": ("text_model.encoder.layers.22.mlp.fc1.weight", (4096, 1024)),
                "CLIP_G": ("text_model.encoder.layers.30.mlp.fc1.weight", (5120, 1280)),
            }
            if detected.name in clip_contracts:
                marker, expected = clip_contracts[detected.name]
                result["text_contract_valid"] = shapes.get(marker) == expected
    if result.get("text_type") in ("WEVISONE_2B", "WEVISONE_4B"):
        result["text_role"] = "image_utility_encoder"
    decoder, encoder = (
        shapes.get("decoder.conv_in.weight", ()),
        shapes.get("encoder.conv_in.weight", ()),
    )
    planar_geometry = (
        "encoder.down.2.downsample.conv.weight" in shapes
        and "decoder.up.3.upsample.conv.weight" in shapes
    ) or (
        # sd.py converts Diffusers keys only with this native conversion marker.
        "decoder.up_blocks.0.resnets.0.norm1.weight" in shapes
        and "encoder.down_blocks.2.downsamplers.0.conv.weight" in shapes
        and "decoder.up_blocks.0.upsamplers.0.conv.weight" in shapes
    )
    if (
        len(decoder) == len(encoder) == 4
        and encoder[1] == 3
        and shapes.get("decoder.conv_out.weight", (None,))[0] == 3
        and planar_geometry
    ):
        result["vae_channels"] = decoder[1]
        result["vae_layout"] = "planar"
        if "bn.running_mean" in shapes:
            # Native sd.py/AutoencoderKL packs 2x2 latent pixels around BN:
            # Flux2 is 32 base channels -> 128 effective channels, scale 8 -> 16.
            channels = 4 * decoder[1]
            result.update(
                vae_channels=channels,
                vae_layout="planar_bn_packed",
                vae_contract_valid=(
                    shapes.get("bn.running_mean") == shapes.get("bn.running_var") == (channels,)
                    and shapes.get("encoder.conv_out.weight") == (2 * decoder[1], decoder[0], 3, 3)
                    and shapes.get("post_quant_conv.weight") == (decoder[1], decoder[1], 1, 1)
                ),
            )
    if (
        len(decoder) == len(encoder) == 4
        and decoder[1] == 4
        and encoder[1] == 3
        and shapes.get("decoder.conv_out.weight", (None,))[0] == 3
        and "encoder.down.1.downsample.conv.weight" in shapes
        and "decoder.up.2.upsample.conv.weight" in shapes
        and "encoder.down.2.downsample.conv.weight" not in shapes
        and "decoder.up.3.upsample.conv.weight" not in shapes
        and "bn.running_mean" not in shapes
    ):
        result.update(vae_channels=4, vae_layout="planar4x")
    if "decoder.middle.0.residual.0.gamma" in shapes:
        if "decoder.upsamples.0.upsamples.0.residual.2.weight" in shapes:
            # Native Wan2.2 and Qwen Image2.1 branches have different layouts,
            # spatial scales and latent counts from the Qwen Image/Anima slot.
            head = shapes.get("decoder.head.2.weight", ())
            result["vae_layout"] = "qwen_image21" if len(head) == 5 and head[2] == 1 else "wan22"
            if result["vae_layout"] == "qwen_image21":
                # Native WanVAE2.2 image branch: patch_size1, temporal_kernel1,
                # RGBA,64 channels and four spatial resamples (16x).
                enc = shapes.get("encoder.conv1.weight", ())
                dec = shapes.get("decoder.head.0.gamma", ())
                valid = len(enc) == 5 and len(dec) == 4
                if valid:
                    dim, out = enc[0], dec[0]
                    expected = {
                        "encoder.conv1.weight": (dim, 4, 1, 3, 3),
                        "encoder.head.2.weight": (128, 8 * dim, 1, 3, 3),
                        "decoder.conv1.weight": (8 * out, 64, 1, 3, 3),
                        "decoder.head.2.weight": (4, out, 1, 3, 3),
                    }
                    for index, factor in enumerate((1, 2, 4, 8)):
                        expected[f"encoder.downsamples.{index}.downsamples.2.resample.1.weight"] = (
                            factor * dim,
                            factor * dim,
                            3,
                            3,
                        )
                    for index, factor in enumerate((8, 8, 4, 2)):
                        expected[f"decoder.upsamples.{index}.upsamples.3.resample.1.weight"] = (
                            factor * out,
                            factor * out,
                            3,
                            3,
                        )
                    valid = all(shapes.get(key) == shape for key, shape in expected.items())
                result.update(vae_channels=64, vae_contract_valid=valid)
        else:
            rgba = shapes.get("encoder.conv1.weight", (None, None))[1] == 4
            result.update(
                vae_layout="wan21_rgba" if rgba else "wan21",
                vae_contract_valid=wan21_rgb_contract(shapes, image_channels=4 if rgba else 3),
            )
            wan_decoder = shapes.get("decoder.conv1.weight", ())
            if len(wan_decoder) == 5:
                result["vae_channels"] = wan_decoder[1]
    if "student.dconv_encoder.proj_out.weight" in shapes:
        # Pinned MageVAE has a 16x RGB DConvEncoder and a separate 128-channel
        # CoD/DConv decoder. Its shared latent count does not make Flux2 a Mage codec.
        expected = {
            "student.dconv_encoder.proj_out.weight": (256, 384, 1, 1),
            "student.dconv_encoder.patch_cond_embed.weight": (768, 3, 16, 16),
            "student.dconv_encoder.z_proj.weight": (384, 128, 1, 1),
            "student.dconv_encoder.blocks.20.conv2.weight": (384, 1, 3, 3),
            "pipeline.y_embedder.decoder.conv_in.weight": (384, 128, 3, 3),
            "pipeline.y_embedder.decoder.conv_out.weight": (384, 384, 3, 3),
            "pipeline.y_embedder_x.weight": (8192, 384, 1, 1),
            "pipeline.blocks.20.conv2.weight": (384, 1, 3, 3),
            "pipeline.final_layer.linear.weight": (3, 32),
        }
        result.update(
            vae_channels=128,
            vae_layout="mage",
            vae_contract_valid=all(shapes.get(name) == shape for name, shape in expected.items()),
        )
    if len(decoder) == 4 and decoder[1] == 64:
        # sd.py selects the planar Hunyuan Image codec before ordinary KL.
        # Its five PixelShuffle/Unshuffle stages distinguish 32x from Qwen16x.
        expected = {
            "encoder.conv_in.weight": (128, 3, 3, 3),
            "decoder.conv_in.weight": (1024, 64, 3, 3),
            "encoder.conv_out.weight": (128, 1024, 3, 3),
            "decoder.conv_out.weight": (3, 128, 3, 3),
        }
        for index, (inp, out) in enumerate(
            ((128, 64), (256, 128), (512, 128), (512, 256), (1024, 256))
        ):
            expected[f"encoder.down.{index}.downsample.conv.weight"] = (out, inp, 3, 3)
        for index, (inp, out) in enumerate(
            ((1024, 4096), (1024, 2048), (512, 2048), (512, 1024), (256, 512))
        ):
            expected[f"decoder.up.{index}.upsample.conv.weight"] = (out, inp, 3, 3)
        result.update(
            vae_channels=64,
            vae_layout="hunyuan_image21",
            vae_contract_valid=all(shapes.get(key) == shape for key, shape in expected.items()),
        )
    if len(decoder) == 5 and decoder[1] == 32:
        # Native sd.py's single-image Hunyuan refiner branch: 3D codec,
        # four spatial and two temporal resamples, refiner_vae=False.
        expected = {
            "encoder.conv_in.weight": (128, 3, 3, 3, 3),
            "decoder.conv_in.weight": (1024, 32, 3, 3, 3),
            "encoder.conv_out.weight": (64, 1024, 3, 3, 3),
            "decoder.conv_out.weight": (3, 128, 3, 3, 3),
        }
        for index, (inp, out) in enumerate(((128, 64), (256, 128), (512, 128), (1024, 128))):
            expected[f"encoder.down.{index}.downsample.conv.weight"] = (out, inp, 3, 3, 3)
        for index, (inp, out) in enumerate(((1024, 8192), (1024, 4096), (512, 1024), (256, 512))):
            expected[f"decoder.up.{index}.upsample.conv.weight"] = (out, inp, 3, 3, 3)
        result.update(
            vae_channels=32,
            vae_layout="hunyuan_refiner",
            vae_contract_valid=all(shapes.get(key) == shape for key, shape in expected.items()),
        )
    from ruinedfooocus.features.generation.minimax_image import codec_contract

    if codec_contract(shapes):
        result.update(vae_channels=24, vae_layout="minimax_image", vae_contract_valid=True)
    return result


def compatibility(key, classification):
    """Return compatible, incompatible or unknown; unknown is never a promise."""
    if key == "clip_bernini_umt5":
        return "compatible" if classification.get("bernini_umt5") else "incompatible"
    if key in ("clip_hydit_bert", "clip_hydit_mt5"):
        kind = classification.get("hydit_encoder")
        if kind is not None:
            return (
                "compatible"
                if kind == {"clip_hydit_bert": "bert", "clip_hydit_mt5": "mt5"}[key]
                else "incompatible"
            )
        return "incompatible" if "text_type" in classification else "unknown"
    if key in VAE_LAYOUTS and "vae_layout" in classification:
        if classification["vae_layout"] != VAE_LAYOUTS[key] or not classification.get(
            "vae_contract_valid", True
        ):
            return "incompatible"
    if key.startswith("clip_") and "text_type" in classification:
        if classification.get("text_role") == "image_utility_encoder":
            return "incompatible"
        if not classification.get("text_contract_valid", True):
            return "incompatible"
        expected = TEXT_TYPES.get(key)
        return (
            "compatible"
            if expected and classification["text_type"] in expected
            else "incompatible"
            if expected
            else "unknown"
        )
    if key.startswith("vae_") and "vae_channels" in classification:
        expected = VAE_CHANNELS.get(key)
        return (
            "compatible"
            if expected == classification["vae_channels"]
            else "incompatible"
            if expected
            else "unknown"
        )
    if (key.startswith("clip_") and "vae_layout" in classification) or (
        key.startswith("vae_") and "text_type" in classification
    ):
        return "incompatible"
    return "unknown"


def inspect_local_component(key, path):
    """Inspect one selected image component without loading weights or assuming a name.

    GGUF encoders can require native key conversion, tokenizer data or a visual
    companion. Logical shape agreement alone cannot qualify that loader route.
    """
    path = Path(path)
    gguf = path.suffix.lower() == ".gguf"
    result = {"compatibility": "unknown", "scaled_fp8": False, "format": path.suffix.lower()}
    if key.startswith("vae_") and (reason := image_vae_format_error(path)):
        return dict(result, compatibility="incompatible", reason=reason)
    if gguf and key in ("clip_hydit_bert", "clip_hydit_mt5"):
        return dict(
            result,
            compatibility="incompatible",
            reason="The native HunyuanDiT split encoder loader does not accept GGUF files.",
        )
    try:
        shapes = header_shapes(path)
        result["scaled_fp8"] = "scaled_fp8" in shapes
        classification = classify_shapes(shapes)
        state = compatibility(key, classification)
        if classification.get("text_role") == "image_utility_encoder":
            result["reason"] = (
                "WeVisOne-VL is an image detection/matting utility encoder, not a generation conditioning component."
            )
        # Native GGUF remapping and tokenizer/vision acquisition are deliberately
        # not reproduced by this metadata-only picker.
        result["compatibility"] = "unknown" if gguf and state == "compatible" else state
    except (OSError, ValueError, TypeError, AttributeError, KeyError, IndexError) as error:
        if "Q4_PT GGUF files are retired" in str(error):
            result.update(compatibility="incompatible", reason=str(error))
    return result


def validate_inventory(state):
    """Reject known mismatches at submission, keeping unknown advanced choices."""
    for row in state["rows"]:
        if row.get("compatibility") == "incompatible":
            raise ValueError(
                f"Image component {row['key']} ({row['filename']}) is incompatible. "
                + (row.get("compatibility_reason") or "Choose a compatible installed component.")
            )


def _with_companions(key, candidate, rows):
    if not key.startswith("clip_") or candidate["compatibility"] == "incompatible":
        return candidate
    companions = [
        row
        for row in rows
        if row["key"] != key
        and row["category"] == "clip"
        and row["status"] != "Embedded in checkpoint"
    ]
    if (
        candidate["format"] == ".gguf"
        and any(row.get("scaled_fp8") for row in companions)
        or candidate["scaled_fp8"]
        and any(Path(row["filename"]).suffix.lower() == ".gguf" for row in companions)
    ):
        return dict(candidate, compatibility="incompatible")
    return candidate


def installed_candidates(key, paths, *, companion_rows=()):
    from ruinedfooocus.features.models.approximate_codecs import installed_pairs

    category = (
        "clip_vision"
        if key in ("clip_vision", "clip_hunyuan_vision")
        else "clip"
        if key.startswith("clip_")
        else "vae"
    )
    rows, names = installed_pairs(key, paths), set()
    for root in model_folders(paths.paths, category, paths.get_abspath):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in paths.EXTENSIONS:
                continue
            name = path.relative_to(root).as_posix()
            if name in names:
                continue
            names.add(name)
            inspected = _with_companions(key, inspect_local_component(key, path), companion_rows)
            rows.append(dict(name=name, path=str(path), **inspected))
    return rows


def validate_local_choice(key, value, paths, *, allow_unknown=False, companion_rows=()):
    if not value:
        return None
    candidate = next(
        (
            row
            for row in installed_candidates(key, paths, companion_rows=companion_rows)
            if row["name"] == value
        ),
        None,
    )
    if candidate is None:
        raise ValueError("The component is no longer installed. Refresh and choose again.")
    if candidate["compatibility"] == "incompatible":
        raise ValueError(
            candidate.get("reason")
            or "This installed component belongs to a different encoder or VAE architecture."
        )
    if candidate["compatibility"] == "unknown" and not allow_unknown:
        raise ValueError(
            "Compatibility is unverified. Use the explicit advanced override to choose it."
        )
    return candidate["name"]
