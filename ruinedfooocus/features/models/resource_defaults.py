"""Automatic component filenames shared by loaders and Settings (no GPU imports)."""

from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import (
    RegistryError,
    _read_document,
    _require,
    metadata_digest,
)


@lru_cache(maxsize=2)
def _recipe_index(path, modified, size):
    document = _read_document(path)
    _require(
        isinstance(document, dict)
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1,
        "Unsupported family recipe schema.",
    )
    for key in ("recipes", "aliases", "variants"):
        _require(isinstance(document.get(key), dict), f"Invalid family recipe {key}.")
    for label, descriptor in document["recipes"].items():
        _require(
            isinstance(label, str)
            and isinstance(descriptor, dict)
            and isinstance(descriptor.get("path"), str)
            and descriptor.get("media") in ("image", "chat"),
            "Invalid family recipe reference.",
        )
        _require(
            all(
                isinstance(descriptor.get(key), str) and bool(descriptor[key])
                for key in ("id", "family", "variant")
            ),
            "Missing family recipe identity.",
        )
    identities = [row["id"] for row in document["recipes"].values()]
    _require(len(identities) == len(set(identities)), "Duplicate family recipe identity.")
    _require(
        all(
            isinstance(alias, str) and isinstance(label, str) and label in document["recipes"]
            for alias, label in document["aliases"].items()
        ),
        "Invalid family recipe alias.",
    )
    _require(
        all(
            isinstance(family, str)
            and isinstance(variants, dict)
            and all(
                isinstance(variant, str) and isinstance(label, str) and label in document["recipes"]
                for variant, label in variants.items()
            )
            for family, variants in document["variants"].items()
        ),
        "Invalid behavioral variant reference.",
    )
    native = document.get("native_recipes", {})
    _require(
        isinstance(native, dict)
        and all(
            isinstance(name, str)
            and isinstance(labels, list)
            and len(labels) <= 64
            and all(isinstance(label, str) and label in document["recipes"] for label in labels)
            and len(labels) == len(set(labels))
            for name, labels in native.items()
        ),
        "Invalid native family recipe references.",
    )
    return document


def _size(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(item) is int and 64 <= item <= 32768 for item in value)
    )


@lru_cache(maxsize=128)
def _recipe_file(path, modified, size, digest):
    _require(size <= 64 * 1024, "Oversized family recipe.")
    document = _read_document(path)
    _require(metadata_digest(document) == digest, "Changed family recipe resource.")
    _require(
        isinstance(document, dict)
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1,
        "Unsupported family recipe schema.",
    )
    for key in ("id", "family", "media", "variant"):
        _require(
            isinstance(document.get(key), str) and bool(document[key]),
            f"Missing family recipe {key}.",
        )
    sizes = document.get("recommended_sizes")
    _require(
        isinstance(sizes, list) and len(sizes) <= 128 and all(_size(value) for value in sizes),
        "Invalid recommended family sizes.",
    )
    default = document.get("default_size")
    _require(
        default is None or (_size(default) and default in sizes), "Invalid default family size."
    )
    _require(
        isinstance(document.get("sampling"), dict)
        and isinstance(document.get("applicability"), dict)
        and isinstance(document.get("provenance"), dict),
        "Invalid family recipe contract.",
    )
    for key, low, high in (("custom_steps", 1, 200), ("cfg", 0, 20), ("clip_skip", 1, 5)):
        value = document["sampling"].get(key)
        _require(
            value is None
            or (
                type(value) in (int, float)
                and math.isfinite(value)
                and low <= value <= high
                and (key == "cfg" or int(value) == value)
            ),
            "Invalid family sampling value.",
        )
    for key in ("sampler_name", "scheduler"):
        value = document["sampling"].get(key)
        _require(
            value is None or (isinstance(value, str) and bool(value)),
            "Invalid family sampling name.",
        )
    applicability = document["applicability"]
    if "native_class" in applicability:
        _require(
            isinstance(applicability["native_class"], str)
            and isinstance(applicability.get("architecture"), str)
            and isinstance(applicability.get("native_config"), dict)
            and isinstance(applicability.get("behavioral_variants"), list)
            and all(isinstance(value, str) for value in applicability["behavioral_variants"])
            and type(applicability.get("auto_recipe_supported")) is bool,
            "Invalid native family recipe applicability.",
        )
    provenance = document["provenance"]
    _require(
        all(
            isinstance(provenance.get(key), str) and bool(provenance[key])
            for key in ("kind", "label", "source_url", "source_revision")
        ),
        "Invalid family recipe provenance.",
    )
    return document


def _recipes_root(root=None):
    path = Path(root) if root is not None else resource_path("defaults")
    path = path.resolve()
    index_path = path / "index.json"
    try:
        stat = index_path.stat()
        return path, _recipe_index(str(index_path), stat.st_mtime_ns, stat.st_size)
    except OSError as error:
        raise RegistryError(f"Family recipes unavailable: {index_path}") from error


def _recipe(root, descriptor):
    path = (root / descriptor["path"]).resolve()
    _require(path.is_relative_to(root), "Family recipe escapes resource root.")
    try:
        stat = path.stat()
    except OSError as error:
        raise RegistryError(f"Family recipe unavailable: {path}") from error
    document = _recipe_file(str(path), stat.st_mtime_ns, stat.st_size, descriptor.get("sha256"))
    _require(
        all(document[key] == descriptor.get(key) for key in ("id", "family", "variant", "media")),
        "Inconsistent family recipe reference.",
    )
    return document


def load_family_recipe(family, *, variant=None, media="image", root=None):
    """Read one compatible family recipe; behavioral variants are explicit."""
    if not isinstance(family, str) or (variant is not None and not isinstance(variant, str)):
        return {}
    root, index = _recipes_root(root)
    label = index["aliases"].get(family, family)
    descriptor = index["recipes"].get(label)
    if descriptor is None:
        descriptor = next((row for row in index["recipes"].values() if row["id"] == family), None)
    if descriptor is None:
        return {}
    if variant is not None:
        variants = index["variants"].get(descriptor["family"], {})
        label = variants.get(variant.casefold())
        if label is None:
            if variant.casefold() != descriptor["variant"]:
                return {}
        else:
            descriptor = index["recipes"][label]
    if descriptor["media"] != media:
        return {}
    return deepcopy(_recipe(root, descriptor))


def native_family_recipes(native_class, *, root=None):
    """Read only one native class's variant recipes; no registry/history lookup."""
    root, index = _recipes_root(root)
    result = []
    for label in index.get("native_recipes", {}).get(native_class, []):
        document = _recipe(root, index["recipes"][label])
        _require(
            document["applicability"].get("native_class") == native_class,
            "Inconsistent native family recipe reference.",
        )
        result.append(deepcopy(document))
    return result


def has_native_family_recipes(family, *, root=None):
    """Whether this family has explicit native variant admission contracts."""
    _, index = _recipes_root(root)
    return any(
        index["recipes"][label]["family"] == family
        for labels in index.get("native_recipes", {}).values()
        for label in labels
    )


def performance_defaults(root=None):
    """Legacy image preset projection from the same family resources."""
    root, index = _recipes_root(root)
    return {
        label: deepcopy(document["legacy"])
        for label, descriptor in index["recipes"].items()
        if (document := _recipe(root, descriptor)).get("legacy") is not None
    }


def native_workflow_example(family, *, native_class, variant=None, samplers=(), schedulers=()):
    """A reviewed connected T2I example, separate from publisher recommendations.

    Callers must first establish an ordinary compatible native configuration.
    Variants, unresolved controls and specialized schedules retain their explicit
    recipe contracts. Loading a snapshot never scans source or acquires files.
    """
    from ruinedfooocus.features.models.native_snapshot import get_image_family_facts
    from ruinedfooocus.features.models.model_recommendations import validated

    facts = get_image_family_facts(family)
    example = facts.metadata().get("example_recipe") if facts is not None else None
    if (
        not isinstance(example, dict)
        or example.get("operation") != "generate"
        or example.get("unresolved")
        or example.get("behavioral_variant") != variant
        or native_class not in example.get("native_classes", ())
    ):
        return {}
    sampling = example.get("sampling", {})
    if not isinstance(sampling, dict):
        return {}
    required = {"custom_steps", "cfg", "sampler_name", "scheduler"}
    if not required.issubset(sampling):
        return {}
    checked = validated(
        sampling,
        samplers or (sampling["sampler_name"],),
        schedulers or (sampling["scheduler"],),
    )
    if "clip_skip" in sampling:
        required.add("clip_skip")
    if any(key not in checked or checked[key] != sampling[key] for key in required):
        return {}
    return deepcopy(example)


RESOURCE_DEFAULTS = {
    "clip_bernini_umt5": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae_bernini": "Wan2_1_VAE_bf16.safetensors",
    "clip_hydit_bert": "hydit_clip.safetensors",
    "clip_hydit_mt5": "hydit_mt5xl.safetensors",
    "clip_qwen35_9b": "qwen3.5_9b_int8_convrot.safetensors",
    "clip_aura": "clip_aura.safetensors",
    "clip_g": "clip_g.safetensors",
    "clip_gemma": "gemma_2_2b_fp16.safetensors",
    "clip_gemma2_it_elm": "gemma_2_2b_it_elm_fp8_scaled.safetensors",
    "clip_gemma3": "gemma_3_4b_it_bf16.safetensors",
    "clip_jina": "jina_clip_v2_bf16.safetensors",
    "clip_joyimage_8b": "qwen3vl_8b_joyimage_edit_int8_convrot.safetensors",
    "clip_l": "clip_l.safetensors",
    "clip_sd4x_h": "sd4x_clip_h.safetensors",
    "clip_llama": "llama_q2.gguf",
    "clip_lens_gpt_oss": "gpt_oss_20b_nvfp4.safetensors",
    "clip_ming_image": "ming_image_0.1_ling_mini_2.0_int8_convrot.safetensors",
    "clip_hunyuan_byt5": "byt5_small_glyphxl_fp16.safetensors",
    "clip_qwen_image21": "qwen3vl_8b_int8_convrot.safetensors",
    "clip_ministral3": "ministral-3-3b.safetensors",
    "clip_mistral3": "mistral_3_small_flux2_fp8.safetensors",
    "clip_oldt5": "t5xxl_old_fp32-q4_0.gguf",
    "clip_omnigen2_3b": "qwen_2.5_vl_fp16.safetensors",
    "clip_qwen25": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "clip_qwen3_06b": "qwen_3_06b_base.safetensors",
    "clip_qwen3_4b": "Qwen3-4B-Q4_K_M.gguf",
    "clip_qwen3_8b": "Qwen3-8B-Q8_0.gguf",
    "clip_minimax_h3_image": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "vae_minimax_h3_image": "minimax_h3_image_vae_fp16.safetensors",
    "clip_qwen3vl_4b": "qwen3vl_4b_bf16.safetensors",
    "clip_qwen3vl_4b_scaled": "qwen3vl_4b_fp8_scaled.safetensors",
    "clip_qwen3vl_8b": "qwen3vl_8b_fp8_scaled.safetensors",
    "clip_qwen3vl_8b_scaled": "qwen3vl_8b_fp8_scaled.safetensors",
    "clip_t5": "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
    "vae_auraflow": "auraflow_vae_fp32.safetensors",
    "vae_flux": "ae.safetensors",
    "vae_flux2": "flux2-vae.safetensors",
    "vae_lumina2": "lumina2_vae_fp32.safetensors",
    "vae_mage_flow": "mage_flow_vae_bf16.safetensors",
    "vae_ming_image": "ming_image_vae_bf16.safetensors",
    "vae_hunyuan_image21": "hunyuan_image_2.1_vae_fp16.safetensors",
    "vae_qwen_image21": "qwen_image_2.1_vae_bf16.safetensors",
    "vae_cascade_c": "stable_cascade_stage_c.safetensors",
    "vae_hunyuan_refiner": "hunyuan_image_refiner_vae_fp16.safetensors",
    "vae_pixart": "pixart_vae_fp16.safetensors",
    "vae_qwen_image": "qwen_image_vae.safetensors",
    "vae_qwen_image_layered": "qwen_image_layered_vae.safetensors",
    "vae_sd": "sd15_vae.safetensors",
    "vae_sd4x": "sd4x_vae.safetensors",
    "vae_sd3": "sd3_vae.safetensors",
    "vae_sdxl": "sdxl_vae.safetensors",
    "vae_wan": "wan_2.1_vae.safetensors",
}


def resource_name(key, values):
    return values.get(key) or RESOURCE_DEFAULTS.get(key)


RESOURCE_LABELS = {
    "clip_bernini_umt5": "Bernini-R UMT5 XXL",
    "vae_bernini": "Bernini-R RGB Wan 2.1 codec",
    "vae_hunyuan_refiner": "Hunyuan Image refiner VAE (32 codec channels, 16×)",
    "vae_cascade_c": "Stable Cascade C bundled EffNet codec",
    "clip_hunyuan_byt5": "Hunyuan Image 2.1 ByT5 glyph encoder",
    "clip_qwen_image21": "Qwen Image 2.1 Qwen3-VL 8B",
    "vae_hunyuan_image21": "Hunyuan Image 2.1 VAE (64 channels, 32×)",
    "vae_qwen_image21": "Qwen Image 2.1 RGBA VAE (64 channels, 16×)",
    "clip_hydit_bert": "HunyuanDiT Chinese RoBERTa",
    "clip_hydit_mt5": "HunyuanDiT MT5-XL with tokenizer",
    "clip_qwen35_9b": "MOSS Image Qwen 3.5 9B",
    "clip_aura": "AuraFlow encoder",
    "clip_g": "CLIP G (SDXL / SD3 / HiDream)",
    "clip_l": "CLIP L (SD / SDXL / Flux / SD3 / HiDream / Hunyuan)",
    "clip_sd4x_h": "SD2 CLIP-H (UnCLIP / SD4x)",
    "clip_t5": "T5 (Flux / HiDream / SD3 / PixArt)",
    "clip_gemma": "Lumina 2 Gemma",
    "clip_gemma2_it_elm": "PixelDiT Gemma 2",
    "clip_gemma3": "NewBieImage Gemma 3",
    "clip_jina": "NewBieImage Jina",
    "clip_joyimage_8b": "JoyImage native Qwen3 VL 8B",
    "clip_ministral3": "ErnieImage Ministral",
    "clip_mistral3": "Flux 2 Mistral",
    "clip_llama": "HiDream Llama",
    "clip_lens_gpt_oss": "Lens GPT-OSS 20B with tokenizer",
    "clip_ming_image": "Ming Image thinker and connector with tokenizer",
    "clip_oldt5": "Cosmos Predict 2 T5",
    "clip_omnigen2_3b": "OmniGen2 Qwen2.5 VL 3B",
    "clip_qwen25": "Qwen Image Qwen 2.5 VL",
    "clip_qwen3_06b": "Anima Qwen 3 0.6B",
    "clip_qwen3_4b": "Flux 2 Klein 4B / Z Image Qwen 3",
    "clip_qwen3_8b": "Flux 2 Klein 9B Qwen 3 8B",
    "clip_qwen3vl_4b": "Mage-Flow Qwen 3 VL",
    "clip_qwen3vl_4b_scaled": "Krea 2 Qwen 3 VL",
    "clip_qwen3vl_8b": "Ideogram 4 Qwen 3 VL",
    "clip_qwen3vl_8b_scaled": "Boogu Image Qwen 3 VL",
    "clip_minimax_h3_image": "MiniMax H3 Image Qwen3-VL 32B",
    "vae_minimax_h3_image": "MiniMax H3 Image VAE",
    "vae_sdxl": "SDXL / Pony / Illustrious VAE",
    "vae_sd": "SD 1.5 VAE",
    "vae_sd4x": "SD4x VAE (4× codec)",
    "vae_qwen_image": "Qwen Image / Anima VAE",
    "vae_qwen_image_layered": "Qwen Image Layered RGBA VAE (16 channels, 8×)",
    "vae_flux": "Flux VAE",
    "vae_ming_image": "Ming Image RGBA VAE",
    "vae_lumina2": "Lumina 2 / Z Image VAE",
    "vae_sd3": "SD3 / HiDream VAE",
}
