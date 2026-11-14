"""Reject retired media at image admission without loading model weights."""

from pathlib import Path


_RETIRED_FAMILIES = frozenset(
    """GenmoMochi LTXV LTXAV LTXV2 MiniMaxH3 StableAudio3 StableAudio ACEStep
    ACEStep15 MiniMaxMusic3 YuE2 Stable_Zero123 SV3D_u SV3D_p Hunyuan3Dv2mini
    Hunyuan3Dv2 Hunyuan3Dv2_1 TripoSplat Trellis2 HunyuanVideo HunyuanVideo15
    HunyuanVideo15_SR_Distilled HunyuanVideoSkyreelsI2V HunyuanVideoI2V CosmosT2V
    CosmosI2V CosmosI2VPredict2 WAN22_T2V WAN21_CausalAR_T2V WAN21_FunControl2V
    WAN21_Vace WAN21_Camera WAN22_Camera WAN22_S2V WAN21_HuMo WAN22_Animate
    WAN_Animate2 WAN21_FlowRVS WAN21_SCAIL WAN21_SCAIL2 WAN22_RelayVSR
    WAN22_WanDancer Kandinsky5 CogVideoX_Inpaint CogVideoX_I2V CogVideoX_T2V
    SVD_img2vid WAN21_T2V WAN21_I2V""".split()
) | {"Wan Video", "Wan 2.2 5B", "Hunyuan Video", "LTXV 2.3", "LTXV 2.5"}
_RETIRED_IDS = ("video/", "audio/", "music/", "speech/", "3d/", "mesh/", "splat/")
_RETIRED_OPERATIONS = frozenset(
    ("video", "audio", "music", "speech", "3d", "mesh", "splat", "text-to-video", "image-to-video")
)
_RETIRED_SUFFIXES = frozenset(
    (
        ".mp4",
        ".webm",
        ".mov",
        ".avi",
        ".mkv",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".glb",
        ".gltf",
        ".obj",
        ".ply",
        ".stl",
    )
)
MESSAGE = "RuinedFooocus supports still images and local Chat; this media operation is unavailable."


def retired_family(family):
    return family in _RETIRED_FAMILIES


def require_image_family(family):
    if retired_family(family):
        raise ValueError(MESSAGE)


def admitted_native_family(family):
    """Native registrations need a real RF image consumer before admission."""
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS
    from ruinedfooocus.features.models.utility_models import IMAGE_UTILITY_CLASSES

    return (
        ALIASES.get(family, family) in IMAGE_COMPONENTS
        or family in IMAGE_UTILITY_CLASSES
        or family == "Stable_Cascade_B"
    )


def require_image_operation(data):
    """Admit a still-image request before an input loader or model download."""
    for key in ("operation", "task_type", "media", "output_type"):
        if str(data.get(key, "")).casefold() in _RETIRED_OPERATIONS:
            raise ValueError(MESSAGE)
    name = str(data.get("model", "")).replace("\\", "/")
    if name.casefold().startswith(_RETIRED_IDS):
        raise ValueError(MESSAGE)


def require_image_path(path):
    if Path(path).suffix.casefold() in _RETIRED_SUFFIXES:
        raise ValueError(MESSAGE)


def unsupported_tensor_family(tensors):
    """Positive retired signatures only; shared image architectures stay distinct."""
    for prefix in ("", "model.diffusion_model.", "diffusion_model."):

        def has(key):
            return prefix + key in tensors

        def shape(key):
            return tensors.get(prefix + key, {}).get("shape", [])

        if has("head.modulation") and has("patch_embedding.weight"):
            return "Wan 2.2 5B" if shape("head.head.weight")[:1] == [192] else "Wan Video"
        if has("txt_in.individual_token_refiner.blocks.0.norm1.weight"):
            projection = shape("img_in.proj.weight")
            if (
                any(name.startswith(prefix + "vector_in.") for name in tensors)
                or len(projection) == 5
                and projection[1] in (32, 33)
                and projection[2:] != [1, 1, 1]
                or shape("vision_in.proj.0.weight")[:1] == [1152]
            ):
                return "Hunyuan Video"
        if has("video_patch_proj.weight") and has("audio_patch_proj.weight"):
            return "MiniMaxH3"
        if has("adaln_single.emb.timestep_embedder.linear_1.bias") and not has(
            "pos_embed.proj.bias"
        ):
            return "LTXV"
    return None


def require_image_checkpoint(path, family=None):
    """Inspect safe headers; never infer an execution backend from a filename."""
    from ruinedfooocus.features.generation.bernini_image import primary_family

    # Only this exact publisher header/size is a Bernini still candidate.
    # Its dedicated loader additionally requires signed preparation receipts.
    if primary_family(path):
        if family not in (None, "BerniniImage"):
            raise ValueError(MESSAGE)
        return
    require_image_family(family)
    path = Path(path)
    require_image_path(path)
    from ruinedfooocus.core.model_formats import is_safetensors

    architecture = None
    if is_safetensors(path):
        from ruinedfooocus.features.models.model_identity import cached_safetensors_header

        tensors = cached_safetensors_header(path)
    elif path.suffix.lower() == ".gguf":
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        tensors, architecture = read_gguf_header(path)
        if architecture in ("wan", "ltxv", "minimax_h3"):
            raise ValueError(MESSAGE)
    else:
        return
    require_image_family(unsupported_tensor_family(tensors))
    # Native classification is metadata-only, and is available after app startup.

    from ruinedfooocus.features.models.model_architecture import native_detector_ready

    if native_detector_ready():
        from ruinedfooocus.features.models.model_architecture import inspect_tensor_model

        details = inspect_tensor_model(tensors)
        if details and not admitted_native_family(details["architecture"]):
            raise ValueError(MESSAGE)
        if architecture == "hyvid" and (
            not details
            or details["architecture"] not in ("HunyuanImage21", "HunyuanImage21Refiner")
        ):
            raise ValueError(MESSAGE)
    elif architecture == "hyvid":
        raise ValueError(MESSAGE)
