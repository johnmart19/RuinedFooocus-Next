"""Opt-in provider endpoints and an OpenAPI image-generation tool."""

import asyncio
import base64
import json
import hashlib
import mimetypes
import os
import queue
import secrets
import time
from concurrent.futures import CancelledError
from pathlib import Path
from urllib.parse import urlparse
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationError
from starlette.datastructures import UploadFile

from ruinedfooocus.features.chat.attachments import MAX_FILES, MAX_FILE_BYTES
from ruinedfooocus.features.image_editing.inpaint_utils import (
    OUTPAINT_MAX_FEATHER,
    OUTPAINT_MAX_MARGIN,
)


class ChatAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(
        min_length=1,
        max_length=120,
        description="Plain .txt, .md, .csv, .json, .pdf or .docx filename.",
    )
    data: str = Field(
        min_length=1,
        max_length=4 * ((MAX_FILE_BYTES + 2) // 3),
        description="Standard base64 of file bytes. Both the file and extracted UTF-8 text must fit 256 KiB.",
    )


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "chat/default"
    messages: list[dict] = Field(min_length=1)
    stream: bool = False
    n: Literal[1] = 1
    rf_attachments: list[ChatAttachment] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_FILES,
        description="Optional stateless local grounding for the final user message.",
    )

    @field_validator("messages")
    @classmethod
    def text_messages(cls, messages):
        for message in messages:
            if message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError("Unsupported message role.")
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict) or item.get("type") not in ("text", "image_url"):
                        raise ValueError("Supported content types are text and image_url.")
                    if item["type"] == "text" and not isinstance(item.get("text"), str):
                        raise ValueError("Text content requires a text string.")
                    if item["type"] == "image_url":
                        url = (
                            item.get("image_url", {}).get("url")
                            if isinstance(item.get("image_url"), dict)
                            else None
                        )
                        if (
                            message["role"] != "user"
                            or not isinstance(url, str)
                            or not url.startswith("data:image/")
                        ):
                            raise ValueError(
                                "Vision inputs must be user messages with base64 image data URLs."
                            )
                        from ruinedfooocus.app.api_runtime import decode_image

                        decode_image(url)
            elif content is not None and not isinstance(content, str):
                raise ValueError("Message content must be text.")
        return messages

    @model_validator(mode="after")
    def attachment_turn(self):
        if self.rf_attachments is not None:
            from ruinedfooocus.features.chat.attachment_retrieval import attachment_question

            attachment_question(self.messages)
        return self


class ControlNetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "canny",
        "depth",
        "recolour",
        "sketch",
        "control",
        "img2img",
        "edit",
        "rembg",
        "detect",
        "segment",
    ]
    model: str | None = Field(
        default=None,
        max_length=1024,
        description="Explicit installed ControlNet/T2I adapter relative filename; native image engine only.",
    )
    strength: float = Field(default=1, ge=0, le=2)
    start: float = Field(default=0, ge=0, le=1)
    stop: float = Field(default=1, ge=0, le=1)
    edge_low: float = Field(default=0.2, ge=0, le=1)
    edge_high: float = Field(default=0.8, ge=0, le=1)
    denoise: float = Field(default=0.25, gt=0, le=1)
    force: bool = Field(
        default=False, description="Reprocess transparent images with background removal."
    )
    depth_preprocess: Literal["auto", "none"] = Field(
        default="auto",
        description="Auto estimates a depth map from the input photo; none uses an uploaded depth map.",
    )
    depth_model: Literal["auto", "small", "base", "lotus"] = "auto"
    depth_lotus_model: str = ""
    depth_lotus_vae: str = ""
    depth_resolution: int = Field(default=504, ge=140, le=2520, multiple_of=14)
    depth_resize_method: Literal["upper_bound_resize", "lower_bound_resize"] = "upper_bound_resize"
    depth_normalization: Literal["v2_style", "min_max"] = "v2_style"
    depth_dtype: Literal["default", "fp16", "bf16", "fp32"] = "default"


class DepthMapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["depth-map"]


class DepthUtilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lotus-depth", "moge-depth", "face-landmarks", "pose-map", "marigold-albedo"]


class CompositionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image-grid", "image-composite"]


class SeedVR2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["restore-seedvr2"]


CurveCoordinate = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
CurveTable = Annotated[
    list[tuple[CurveCoordinate, CurveCoordinate]], Field(min_length=2, max_length=32)
]


class ImageShaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image-curves", "glsl-effect"]


class ImageFilterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image-blur", "image-sharpen", "adjust-brightness", "adjust-contrast"]


class ImageAdjustmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image-transform", "image-mask", "image-color"]


class LatentCompositionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["latent-compose"]


class WeVisOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["wevisone-detect", "wevisone-matte"]


MAX_IMAGE_PROMPT_CHARS = 131072
MAX_IMAGE_UPLOAD_BYTES = 21000000
MAX_INLINE_IMAGE_CHARS = 28000000
MAX_EDIT_FORM_BYTES = 2 * MAX_IMAGE_UPLOAD_BYTES + MAX_INLINE_IMAGE_CHARS + 1024**2


class MiniMaxImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: Annotated[int, Field(strict=True, ge=1, le=16777216)]
    height: Annotated[int, Field(strict=True, ge=1, le=16777216)]
    rgba_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


MiniMaxFileIdentity = tuple[
    Annotated[str, Field(min_length=1, max_length=1024)],
    Annotated[int, Field(strict=True, ge=1)],
    Annotated[str, Field(pattern=r"^[0-9]{1,30}$")],
    Annotated[str, Field(pattern=r"^[0-9]{1,30}$")],
]


class MiniMaxImageReplay(BaseModel):
    """Inert same-app provenance; uploaded images/components are resolved anew."""

    model_config = ConfigDict(extra="ignore")
    version: Literal[1]
    checkpoint: MiniMaxFileIdentity
    pdd_steps: Annotated[int, Field(strict=True, ge=1, le=64)]
    pdd_sigmas: Annotated[
        list[Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]],
        Field(min_length=2, max_length=65),
    ]
    references: Annotated[list[MiniMaxImageReference], Field(max_length=4)]
    components: Annotated[
        list[
            tuple[
                Literal["minimax_model", "clip", "vae"],
                Annotated[str, Field(min_length=1, max_length=1024)],
                Annotated[int, Field(strict=True, ge=1)],
                Annotated[str, Field(pattern=r"^[0-9]{1,30}$")],
                Annotated[str, Field(pattern=r"^[0-9]{1,30}$")],
            ]
        ],
        Field(max_length=3),
    ] = Field(default_factory=list)


class EditorContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    revision: int = Field(ge=1)
    source_id: str = Field(pattern=r"^image_[0-9a-f]{32}$")
    mask_id: str | None = Field(default=None, pattern=r"^mask_[0-9a-f]{32}$")
    additional_image_ids: list[Annotated[str, Field(pattern=r"^image_[0-9a-f]{32}$")]] = Field(
        default_factory=list, max_length=32
    )


class EditorOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_image: str = Field(min_length=1, max_length=28000000)


class EditorMoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=1)


class RFImageToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    call: dict
    intent: Literal["text", "preview", "run"]
    enable_image_generation: bool = True
    rewrite_image_prompt: bool = False
    update_create_prompt: bool = False
    image_prompt_target: Literal["new", "supplied"] = "new"
    image_prompt_original: str | None = Field(default=None, max_length=12000)
    editor_context: EditorContext | None = None
    stream: bool = False
    response_format: Literal["url", "b64_json"] = "url"


class _GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(
        default="",
        max_length=MAX_IMAGE_PROMPT_CHARS,
        description="Complete visual description, up to 131072 characters. Native encoder chunking/context and available memory still apply. Put output dimensions in size, not only in this text.",
    )
    model: str = Field(
        default="image/default",
        description="Use image/default for the selected checkpoint, or an image/ ID from available_models. Do not use chat model IDs.",
    )
    n: int = Field(default=1, ge=1, le=4)
    editor_context: EditorContext | None = None
    size: str | None = Field(
        default=None,
        pattern=r"^(auto|[0-9]+x[0-9]+)$",
        description="Output WIDTHxHEIGHT; omit to inherit Main's size. Images pad the native processing canvas separately, preserving exact target pixels. With automatic_upscale, size is the final target and native sampling uses target/factor. Native model and application pixel limits apply to the working canvas. Source-preserving edits and SD4x retain their own geometry.",
    )
    response_format: Literal["url", "b64_json"] = "url"
    negative_prompt: str | None = Field(default=None, max_length=MAX_IMAGE_PROMPT_CHARS)
    performance: str | None = None
    steps: int | None = Field(default=None, ge=1, le=200)
    cfg: float | None = Field(default=None, ge=0, le=30)
    sampler: str | None = None
    scheduler: str | None = None
    clip_skip: int | None = Field(default=None, ge=1, le=12)
    minimax_image_context: MiniMaxImageReplay | None = Field(
        default=None,
        description="Optional saved native_minimax_image provenance. Reattach the exact original ordered references and select the same local model, 32B conditioning encoder and image codec; this field never supplies tensors or paths.",
    )
    model_recipe_path: str | None = Field(
        default=None,
        max_length=4096,
        description="Explicit local YAML recipe from configured checkpoint/model folders; native ordinary SD only.",
    )
    model_recipe_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    clip_layer_source: Literal["request", "recipe"] | None = Field(
        default=None,
        description="Clip Skip remains request-owned unless recipe is explicitly selected; conflicts with explicit clip_skip.",
    )
    seed: int | None = Field(default=None, ge=-1, le=4294967295)
    user: str | None = None
    input_image: str | None = Field(
        default=None,
        max_length=MAX_INLINE_IMAGE_CHARS,
        description="Base64 image, image data URL, or unexpired image URL returned by this API. No arbitrary URLs or file paths.",
    )
    reference_adapter: Literal["none", "style", "redux", "photomaker"] | None = None
    reference_adapter_model: str | None = Field(default=None, max_length=1024)
    reference_adapter_vision: str | None = Field(default=None, max_length=1024)
    reference_adapter_strength: float | None = Field(default=None, ge=0, le=10, allow_inf_nan=False)
    reference_adapter_image: str | None = Field(
        default=None,
        max_length=28000000,
        description="Explicit adapter reference: base64 image or an owned API image URL, separate from input_image.",
    )
    controlnet: ControlNetRequest | None = None
    styles: list[str] | None = None
    loras: list["LoraRequest"] | None = None
    qwen_image21_schedule: Literal["auto", "standard", "turbo_8"] | None = Field(
        default=None,
        description=(
            "Qwen Image 2.1 sampling schedule. Auto follows Model Defaults, standard uses the "
            "selected scheduler, and turbo_8 requires 8 steps, CFG 1, Euler and Simple."
        ),
    )
    moss_image_edit_format: (
        Literal["Disabled", "ai-toolkit", "ai-toolkit (kv_cache off)", "Moss-Image-Edit"] | None
    ) = None
    moss_image_edit_strength: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    moss_image_slg: bool | None = None
    moss_image_slg_double_layers: str | None = Field(default=None, max_length=1024)
    moss_image_slg_single_layers: str | None = Field(default=None, max_length=1024)
    moss_image_slg_start: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    moss_image_slg_end: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    sampling_shift: float | None = Field(default=None, ge=0.01, le=100, allow_inf_nan=False)
    cfg_norm_strength: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    cfg_norm_auto: bool | None = Field(
        default=None,
        description="Use the selected model recipe's CFG normalization when no strength is set.",
    )
    cascade_stage_b: str | None = Field(default=None, max_length=1024)
    bernini_low_model: str | None = Field(default=None, max_length=1024)
    bernini_split_step: int | None = Field(default=None, ge=1, le=199, strict=True)
    bernini_lightx2v: bool = Field(default=False, strict=True)
    bernini_high_lora_strength: float = Field(default=3, ge=-100, le=100, allow_inf_nan=False)
    bernini_low_lora_strength: float = Field(default=1.5, ge=-100, le=100, allow_inf_nan=False)
    bernini_ref_max_size: int = Field(default=848, ge=16, le=8192, strict=True)
    cascade_b_steps: int | None = Field(default=None, ge=1, le=200)
    cascade_b_cfg: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False)
    cascade_b_sampler: str | None = Field(default=None, max_length=80)
    cascade_b_scheduler: str | None = Field(default=None, max_length=80)
    cascade_compression: int = Field(default=42, ge=4, le=128, strict=True)
    cascade_source_denoise: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    cascade_super_resolution: bool = Field(default=False, strict=True)
    cascade_super_resolution_model: str | None = Field(default=None, max_length=1024)
    cascade_control_strength: float = Field(default=1, ge=0, le=10, allow_inf_nan=False)
    cascade_control_start: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    cascade_control_end: float = Field(default=1, ge=0, le=1, allow_inf_nan=False)
    cascade_control_denoise: float = Field(default=1, ge=0, le=1, allow_inf_nan=False)
    cascade_vision_enabled: bool = Field(default=False, strict=True)
    cascade_vision_model: str | None = Field(default=None, max_length=1024)
    cascade_vision_images: list[Annotated[str, Field(max_length=28000000)]] | None = Field(
        default=None, max_length=8
    )
    cascade_vision_strengths: (
        list[Annotated[float, Field(ge=-10, le=10, allow_inf_nan=False)]] | None
    ) = Field(default=None, max_length=8)
    cascade_vision_crop: Literal["none", "center"] = "none"
    sdxl_refiner_model: str | None = Field(default=None, max_length=1024)
    sdxl_refiner_switch_step: int | None = Field(default=None, ge=1, le=199, strict=True)
    sdxl_refiner_cfg: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    sdxl_refiner_clip: str | None = Field(default=None, max_length=1024)
    refiner_noise_augmentation: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    pid_input_vae: str | None = Field(default=None, max_length=1024)
    rembg_native_model: str | None = Field(default=None, max_length=1024)
    native_vision_model: str | None = Field(
        default=None,
        max_length=1024,
        description="Installed safetensors model for detect (RT-DETR v4), segment (SAM 3 / 3.1), or depth-map (Depth Anything 3).",
    )
    image_utility_model: str | None = Field(default=None, max_length=1024)
    image_utility_vae: str | None = Field(default=None, max_length=1024)
    image_utility_lora: str | None = Field(default=None, max_length=1024)
    image_utility_conditioning: str | None = Field(default=None, max_length=1024)
    moge_resolution_level: int | None = Field(default=None, ge=0, le=9, strict=True)
    moge_refine_steps: int | None = Field(default=None, ge=0, le=8, strict=True)
    moge_output: Literal["depth", "normal"] | None = None
    moge_projection: Literal["perspective", "panorama"] | None = None
    moge_normal_convention: Literal["opengl", "directx"] | None = None
    moge_split_resolution: int | None = Field(default=None, ge=256, le=1024, strict=True)
    moge_merge_resolution: int | None = Field(default=None, ge=256, le=8192, strict=True)
    moge_panorama_batch_size: int | None = Field(default=None, ge=1, le=12, strict=True)
    face_detector_variant: Literal["short", "full", "both"] | None = None
    face_max_count: int | None = Field(default=None, ge=1, le=16, strict=True)
    face_output: Literal["overlay", "region-mask"] | None = None
    face_mask_regions: Literal["all", "custom"] | None = None
    face_mask_face_oval: bool | None = Field(default=None, strict=True)
    face_mask_lips: bool | None = Field(default=None, strict=True)
    face_mask_left_eye: bool | None = Field(default=None, strict=True)
    face_mask_right_eye: bool | None = Field(default=None, strict=True)
    face_mask_irises: bool | None = Field(default=None, strict=True)
    face_min_confidence: float | None = Field(
        default=None, ge=0, le=1, strict=True, allow_inf_nan=False
    )
    pose_score_threshold: float | None = Field(
        default=None, ge=0, le=1, strict=True, allow_inf_nan=False
    )
    vision_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    detect_class: str | None = Field(default=None, max_length=80)
    detect_max_count: int | None = Field(default=None, ge=1, le=1000)
    segment_positive_points: str | None = Field(
        default=None,
        max_length=16384,
        description='JSON pixel points such as [{"x":100,"y":80}]; at least one for segment.',
    )
    segment_negative_points: str | None = Field(default=None, max_length=16384)
    segment_refine_iterations: int | None = Field(default=None, ge=0, le=5)
    segment_output: Literal["cutout", "mask"] | None = None
    seedvr2_model: str | None = Field(default=None, max_length=1024)
    seedvr2_vae: str | None = Field(default=None, max_length=1024)
    seedvr2_scale: float | None = Field(default=None, ge=1, le=4, allow_inf_nan=False)
    seedvr2_color: Literal["none", "lab", "wavelet", "adain"] | None = None
    wevisone_model: str | None = Field(default=None, max_length=1024)
    wevisone_query: str | None = Field(default=None, max_length=4096)
    wevisone_max_length: int | None = Field(default=None, strict=True, ge=1, le=32768)
    wevisone_output: Literal["cutout", "mask", "overlay"] | None = None
    native_latent_input: str | None = Field(default=None, max_length=44739252)
    native_latent_denoise: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    native_positive_input: str | None = Field(default=None, max_length=44739252)
    native_negative_input: str | None = Field(default=None, max_length=44739252)
    native_tensor_family: (
        Literal[
            "auto",
            "SD15",
            "SD20",
            "SDXL",
            "SDXLRefiner",
            "SD3",
            "Flux",
            "FluxSchnell",
            "QwenImage",
            "QwenImage21",
            "Flux2",
            "HunyuanImage21Refiner",
            "Stable_Cascade_C",
        ]
        | None
    ) = None
    native_tensor_workflow: bool = False
    native_export_latent: bool = False
    native_export_conditioning: bool = False
    native_export_handoff: bool = False
    latent_composition_primary: str | None = Field(default=None, max_length=44739252)
    latent_composition_secondary: str | None = Field(default=None, max_length=44739252)
    latent_composition_mask: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    latent_composition_mode: Literal["replace", "masked"] = "replace"
    latent_composition_x: int = Field(default=0, strict=True, ge=0, le=16384)
    latent_composition_y: int = Field(default=0, strict=True, ge=0, le=16384)
    latent_composition_feather: int = Field(default=0, strict=True, ge=0, le=16384)
    pid_latent_format: Literal["flux", "sd3", "sdxl", "qwenimage"] | None = None
    pid_degrade_sigma: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    cfg_norm_pre_cfg: bool | None = Field(
        default=None,
        description="Normalize before CFG when normalization strength is set; may amplify guidance.",
    )
    hidream_o1_noise_scale: float | None = Field(default=None, ge=0, le=64, allow_inf_nan=False)
    hidream_o1_seam_smoothing: bool | None = Field(
        default=None,
        description="Auto enables seams for full O1 and disables them for dev; true/false explicitly overrides Auto.",
    )
    hidream_o1_seam_start: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    hidream_o1_seam_end: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    hidream_o1_seam_pattern: Literal["single_shift", "symmetric"] | None = None
    hidream_o1_seam_passes: Literal["2", "4", "ramp_2_4", "ramp_2_4_8"] | None = None
    hidream_o1_seam_blend: Literal["average", "window", "median"] | None = None
    hidream_o1_seam_strength: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    lcm_noise: float | None = Field(default=None, ge=0, le=64, allow_inf_nan=False)
    lcm_noise_end: float | None = Field(default=None, ge=0, le=64, allow_inf_nan=False)
    lcm_noise_clip: float | None = Field(default=None, ge=0, le=10, allow_inf_nan=False)
    image_resize_mode: Literal["preserve", "pad", "crop"] | None = Field(
        default=None,
        description="Preserve keeps the full input frame and only downscales to the working pixel budget. Pad fits the full frame inside the selected size. Crop explicitly fills the selected size without stretching.",
    )
    vae_tiled: bool | None = Field(
        default=None,
        description="Force native tiled VAE decoding. Auto also tiles when the image exceeds VRAM.",
    )
    vae_tile_size: int | None = Field(
        default=None,
        ge=64,
        le=4096,
        multiple_of=32,
        description="VAE tile edge in pixels; omit for automatic VRAM fitting.",
    )
    vae_overlap: int | None = Field(
        default=None,
        ge=0,
        le=4096,
        multiple_of=32,
        description="VAE tile overlap in pixels; omit for the native default.",
    )
    reference_images: list[Annotated[str, Field(max_length=28000000)]] | None = Field(
        default=None, max_length=15
    )


BorderPixels = Annotated[int, Field(strict=True, ge=0, le=OUTPAINT_MAX_MARGIN)]


class OutpaintBorders(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: BorderPixels = 0
    top: BorderPixels = 0
    right: BorderPixels = 0
    bottom: BorderPixels = 0
    feather: Annotated[int, Field(strict=True, ge=0, le=OUTPAINT_MAX_FEATHER)] | None = None

    @model_validator(mode="after")
    def require_extension(self):
        if not any((self.left, self.top, self.right, self.bottom)):
            raise ValueError("Set at least one positive canvas border.")
        return self


class QwenImage21UnionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["qwen-union"]
    model: str | None = Field(default=None, max_length=1024)
    model_signature: (
        tuple[
            str,
            Annotated[int, Field(strict=True, ge=0)],
            Annotated[int, Field(strict=True, ge=0)],
            Annotated[int, Field(strict=True, ge=0)],
        ]
        | None
    ) = None
    mode: Literal[
        "canny", "depth", "grayscale", "hed", "lineart", "mlsd", "pose", "scribble", "inpaint"
    ] = "canny"
    strength: float = Field(default=1.0, ge=0, le=2)
    start: float = Field(default=0.0, ge=0, le=1)
    stop: float = Field(default=1.0, ge=0, le=1)


GuidanceFileIdentity = tuple[
    Annotated[str, Field(max_length=1024)],
    Annotated[int, Field(strict=True, ge=0, le=2147483648)],
    Annotated[str, Field(pattern=r"^[0-9]{1,24}$")],
    Annotated[str, Field(pattern=r"^[0-9]{1,24}$")],
]


class PrecomputedControlGuide(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(min_length=1, max_length=MAX_INLINE_IMAGE_CHARS)
    kind: Literal["depth", "normal"]
    convention: Literal["white-near", "moge-opengl", "moge-directx", "normal-bae"]
    denoise: float = Field(default=0.35, strict=True, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def compatible_convention(self):
        if (self.kind == "depth") != (self.convention == "white-near"):
            raise ValueError("Declare white-near depth or the exact supported normal convention.")
        return self


class ImageRequest(_GenerationRequest):
    control_guide: PrecomputedControlGuide | None = Field(
        default=None,
        description="Perspective precomputed depth/normal map, separate from original input_image. Uses existing inline or owned API media transport; never an Editor result ticket. Original and guide must be exact-frame 8-bit sRGB images.",
    )
    composition_images: (
        list[Annotated[str, Field(min_length=1, max_length=MAX_INLINE_IMAGE_CHARS)]] | None
    ) = Field(default=None, min_length=1, max_length=32)
    composition_mask: str | None = Field(
        default=None, min_length=1, max_length=MAX_INLINE_IMAGE_CHARS
    )
    grid_columns: int = Field(default=2, strict=True, ge=1, le=20)
    grid_cell_width: int = Field(default=512, strict=True, ge=32, le=2048)
    grid_cell_height: int = Field(default=512, strict=True, ge=32, le=2048)
    grid_padding: int = Field(default=8, strict=True, ge=0, le=50)
    composition_mode: str = Field(default="SRC_OVER", max_length=32)
    composition_x: int = Field(default=0, strict=True, ge=0, le=16384)
    composition_y: int = Field(default=0, strict=True, ge=0, le=16384)
    composition_width: int = Field(default=0, strict=True, ge=0, le=16384)
    composition_height: int = Field(default=0, strict=True, ge=0, le=16384)

    @model_validator(mode="after")
    def validate_editor(self):
        if self.editor_context is not None:
            if self.n != 1:
                raise ValueError("An editor operation returns exactly one image.")
            if self.input_image is not None or self.mask is not None:
                raise ValueError("An editor handle supplies its own image and selection mask.")
        return self

    @model_validator(mode="after")
    def validate_precomputed_guide(self):
        if self.control_guide is None:
            return self
        if self.editor_context is not None or not self.input_image or self.n != 1:
            raise ValueError(
                "An uploaded guide requires one explicit original input_image without Editor handles."
            )
        if self.mask is not None or self.borders is not None:
            raise ValueError(
                "Precomputed guided editing uses an original source without an inpaint mask."
            )
        if (
            self.controlnet is None
            or self.controlnet.type not in ("depth", "control")
            or not self.controlnet.model
        ):
            raise ValueError(
                "Select an explicit installed depth/normal ControlNet for the uploaded guide."
            )
        if (
            "depth_preprocess" in self.controlnet.model_fields_set
            and self.controlnet.depth_preprocess != "none"
        ):
            raise ValueError("A precomputed guide cannot run another depth estimator.")
        self.controlnet.depth_preprocess = "none"
        return self

    @model_validator(mode="after")
    def validate_composition(self):
        from ruinedfooocus.features.image_processing.image_composition import (
            FIELDS,
            handles,
            request_values,
        )

        kind = self.controlnet.type if self.controlnet is not None else None
        if not handles(kind) and self.model_fields_set.intersection(FIELDS):
            raise ValueError("Composition controls require image-grid or image-composite.")
        request_values(self.model_dump(exclude_none=True))
        return self

    curve_master: CurveTable | None = None
    curve_red: CurveTable | None = None
    curve_green: CurveTable | None = None
    curve_blue: CurveTable | None = None
    curve_interpolation: Literal["linear", "monotone_cubic"] | None = None
    shader_effect: Literal["invert", "bt709-grayscale", "posterize"] | None = None
    shader_amount: float | None = Field(default=None, strict=True, ge=0, le=1, allow_inf_nan=False)
    shader_levels: int | None = Field(default=None, strict=True, ge=2, le=32)
    filter_radius: int | None = Field(default=None, strict=True, ge=1, le=31)
    filter_sigma: float | None = Field(
        default=None, strict=True, ge=0.1, le=10, allow_inf_nan=False
    )
    filter_strength: float | None = Field(
        default=None, strict=True, ge=0, le=5, allow_inf_nan=False
    )
    filter_factor: float | None = Field(default=None, strict=True, ge=0, le=2, allow_inf_nan=False)
    transform_operation: (
        Literal["resize", "crop", "pad", "rotate", "flip", "crop-to-mask"] | None
    ) = None
    transform_width: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_height: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_fit: Literal["contain", "stretch", "center-crop"] | None = None
    transform_interpolation: Literal["nearest-exact", "bilinear", "area", "bicubic"] | None = None
    transform_x: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_y: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_left: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_top: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_right: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_bottom: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_padding_style: Literal["transparent", "color", "mirror"] | None = None
    transform_padding_color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    transform_feather: int | None = Field(default=None, strict=True, ge=0, le=16384)
    transform_rotation: Literal["none", "90 degrees", "180 degrees", "270 degrees"] | None = None
    transform_flip: Literal["horizontal", "vertical"] | None = None
    mask_operation: (
        Literal["grow", "feather", "composite", "channel", "color", "threshold", "invert"] | None
    ) = None
    mask_expand: int | None = Field(default=None, strict=True, ge=-16384, le=16384)
    mask_tapered_corners: bool | None = Field(default=None, strict=True)
    mask_left: int | None = Field(default=None, strict=True, ge=0, le=16384)
    mask_top: int | None = Field(default=None, strict=True, ge=0, le=16384)
    mask_right: int | None = Field(default=None, strict=True, ge=0, le=16384)
    mask_bottom: int | None = Field(default=None, strict=True, ge=0, le=16384)
    mask_composite_operation: Literal["multiply", "add", "subtract", "and", "or", "xor"] | None = (
        None
    )
    mask_channel: Literal["red", "green", "blue", "alpha"] | None = None
    mask_color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    mask_threshold: float | None = Field(default=None, strict=True, ge=0, le=1, allow_inf_nan=False)
    mask_invert: bool | None = Field(default=None, strict=True)
    mask_image: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    mask_reference: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    color_operation: (
        Literal["blend", "transfer", "quantize", "text-overlay", "morphology", "sdr-to-hdr"] | None
    ) = None
    color_blend_mode: (
        Literal["normal", "multiply", "screen", "overlay", "soft_light", "difference"] | None
    ) = None
    color_blend_factor: float | None = Field(
        default=None, strict=True, ge=0, le=1, allow_inf_nan=False
    )
    color_transfer_method: Literal["reinhard_lab", "mkl_lab", "histogram"] | None = None
    color_transfer_strength: float | None = Field(
        default=None, strict=True, ge=0, le=10, allow_inf_nan=False
    )
    color_quantize_colors: int | None = Field(default=None, strict=True, ge=1, le=256)
    color_quantize_dither: (
        Literal["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"] | None
    ) = None
    color_reference: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    color_text: str | None = Field(default=None, max_length=2048)
    color_text_font_size: float | None = Field(
        default=None, ge=0.5, le=50, strict=True, allow_inf_nan=False
    )
    color_text_color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    color_text_position: Literal["top", "bottom"] | None = None
    color_text_align: Literal["left", "center", "right"] | None = None
    color_text_outline: bool | None = Field(default=None, strict=True)
    color_morphology_operation: (
        Literal["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"] | None
    ) = None
    color_morphology_kernel: int | None = Field(default=None, ge=3, le=999, strict=True)
    color_hdr_luminance: float | None = Field(
        default=None, ge=0, le=10000, strict=True, allow_inf_nan=False
    )
    segment_text: str | None = Field(default=None, max_length=1024)
    segment_boxes: str | None = Field(default=None, max_length=32768)
    segment_individual_masks: bool | None = None
    segment_max_instances: int | None = Field(default=None, ge=1, le=64)
    qwen_layer_count: Annotated[int, Field(strict=True, ge=1, le=16)] | None = Field(
        default=None,
        description="Ordered RGBA outputs per Qwen Layered generation; n remains the repetition count.",
    )

    @model_validator(mode="after")
    def validate_cascade_guidance(self):
        if self.cascade_super_resolution:
            if not self.input_image or not self.cascade_super_resolution_model:
                raise ValueError(
                    "Cascade super-resolution requires input_image and its installed adapter."
                )
            if self.cascade_control_start >= self.cascade_control_end:
                raise ValueError("Cascade control requires start before end.")
        if self.cascade_vision_enabled:
            if not self.cascade_vision_model or not self.cascade_vision_images:
                raise ValueError(
                    "Cascade vision requires an installed vision model and reference images."
                )
            if self.cascade_vision_strengths and len(self.cascade_vision_strengths) != len(
                self.cascade_vision_images
            ):
                raise ValueError("Cascade vision requires one strength per reference image.")
        return self

    native_negative_model: str | None = Field(default=None, max_length=1024)
    native_schedule_alpha: float | None = Field(default=None, ge=0.01, le=50)
    native_schedule_beta: float | None = Field(default=None, ge=0.01, le=50)
    native_schedule_mu: float | None = Field(default=None, ge=-10, le=10)
    native_schedule_std: float | None = Field(default=None, ge=0.1, le=5)
    gligen_model: str | None = Field(
        default=None,
        max_length=1024,
        description="Local relative GLIGEN filename under configured gligen roots. Null turns Off; omission inherits saved controls for the same checkpoint. Matching native SD UNet/CLIP required.",
    )
    gligen_model_signature: GuidanceFileIdentity | None = None
    gligen_layout: str | None = Field(
        default=None,
        max_length=32768,
        description="JSON list of 1..30 ordered objects with literal text,x,y,width,height. Whole multiples of eight in target pixels; boxes stay inside a fresh eight-aligned canvas.",
    )
    hypernetwork_model: str | None = Field(
        default=None,
        max_length=1024,
        description="Local relative nested native Torch hypernetwork filename under configured hypernetwork roots; attention dimensions validated before sampling.",
    )
    hypernetwork_model_signature: GuidanceFileIdentity | None = None
    hypernetwork_strength: float | None = Field(
        default=None, strict=True, ge=-10, le=10, allow_inf_nan=False
    )
    native_edit_recipe: Literal[
        "auto",
        "qwen-edit",
        "qwen-edit-2509",
        "qwen-edit-2511",
        "longcat-edit",
        "flux2-reference",
        "boogu-edit",
        "krea2-style",
    ] = "auto"

    krea_style_lora: str | None = Field(default=None, max_length=512)
    krea_style_strength: float | None = Field(default=None, ge=-10, le=10, allow_inf_nan=False)

    model_patch_recipe: Literal[
        "none", "anima-lllite", "qwen-diffsynth", "zimage-fun", "uso-style"
    ] = "none"
    model_patch_model: str | None = Field(default=None, max_length=512)
    model_patch_vision: str | None = Field(default=None, max_length=512)
    model_patch_strength: float = Field(default=1.0, ge=-10, le=10, allow_inf_nan=False)
    model_patch_start: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    model_patch_stop: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    model_patch_denoise: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    model_patch_image: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    model_patch_inpaint_image: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    model_patch_mask: str | None = Field(
        default=None,
        max_length=MAX_INLINE_IMAGE_CHARS,
        description="Patch mask in the source image frame: white changes, black preserves.",
    )
    control_recipe: Literal["ordinary", "union", "alimama-inpaint"] = "ordinary"
    control_union_type: Literal[
        "auto",
        "openpose",
        "depth",
        "hed/pidi/scribble/ted",
        "canny/lineart/anime_lineart/mlsd",
        "normal",
        "segment",
        "tile",
        "repaint",
    ] = "auto"
    control_inpaint_mode: Literal["mask", "outpaint"] = "mask"
    control_inpaint_denoise: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    control_inpaint_image: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    control_inpaint_mask: str | None = Field(
        default=None,
        max_length=MAX_INLINE_IMAGE_CHARS,
        description="AliMama mask in the dedicated source frame: white changes, black preserves.",
    )
    control_outpaint_left: BorderPixels = 0
    control_outpaint_top: BorderPixels = 0
    control_outpaint_right: BorderPixels = 0
    control_outpaint_bottom: BorderPixels = 0
    control_outpaint_feather: Annotated[int, Field(strict=True, ge=0, le=OUTPAINT_MAX_FEATHER)] = 32

    lora_hook_mode: Literal["off", "keyframes", "interpolated", "floats"] | None = None
    lora_hook_source_type: Literal["lora", "checkpoint"] | None = None
    lora_hook_sources: str | None = Field(default=None, max_length=131072)
    lora_hook_strengths: str | None = Field(default=None, max_length=8192)
    lora_hook_model: str | None = Field(default=None, max_length=1024)
    lora_hook_model_strength: float | None = Field(default=None, ge=-20, le=20, allow_inf_nan=False)
    lora_hook_clip_strength: float | None = Field(default=None, ge=-20, le=20, allow_inf_nan=False)
    lora_hook_schedule_clip: bool | None = None
    lora_hook_keyframes: str | None = Field(default=None, max_length=16384)
    lora_hook_interpolation: Literal["linear", "ease_in", "ease_out", "ease_in_out"] | None = None
    lora_hook_start: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    lora_hook_end: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    lora_hook_strength_start: float | None = Field(default=None, ge=0, le=10, allow_inf_nan=False)
    lora_hook_strength_end: float | None = Field(default=None, ge=0, le=10, allow_inf_nan=False)
    lora_hook_keyframe_count: Annotated[int, Field(strict=True, ge=2, le=100)] | None = None

    @model_validator(mode="after")
    def validate_lora_hooks(self):
        from ruinedfooocus.features.image_editing import lora_hooks

        values = self.model_dump(include=set(lora_hooks.FIELDS), exclude_none=True)
        lora_hooks.settings(values)
        return self

    @model_validator(mode="after")
    def validate_native_patches(self):
        from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

        model_patches.settings(self.model_dump(include=set(model_patches.FIELDS)))
        patch = self.model_patch_recipe != "none"
        control = self.control_recipe != "ordinary"
        if patch or control:
            if (
                patch
                and control
                or self.mask is not None
                or self.borders is not None
                or self.reference_images
                or self.reference_adapter not in (None, "none")
                or self.native_edit_recipe != "auto"
            ):
                raise ValueError(
                    "Named patch/control recipes use their own inputs without another source recipe."
                )
        if patch:
            if self.controlnet is not None and self.controlnet.type != "img2img":
                raise ValueError("Native model patches require no Image tool or ordinary img2img.")
            if self.model_patch_recipe != "zimage-fun" and not self.model_patch_image:
                raise ValueError("Add this model patch's own guide or style reference.")
            if self.model_patch_recipe == "uso-style" and (
                self.model_patch_inpaint_image or self.model_patch_mask
            ):
                raise ValueError("USO uses only its own style reference image.")
            if self.model_patch_recipe != "zimage-fun" and self.model_patch_inpaint_image:
                raise ValueError("This model patch has no separate inpaint source.")
            if self.model_patch_recipe == "zimage-fun":
                if bool(self.model_patch_inpaint_image) != bool(self.model_patch_mask):
                    raise ValueError("Add both the Fun inpaint source and its white-edit mask.")
                if not (self.model_patch_image or self.model_patch_inpaint_image):
                    raise ValueError("Add a Fun guide or inpaint source and mask.")
                if self.model_patch_inpaint_image and (
                    self.input_image or self.controlnet is not None
                ):
                    raise ValueError(
                        "Use the Fun patch's own source without a separate input_image or Image tool."
                    )
        elif any(getattr(self, key) for key in model_patches.IMAGE_FIELDS):
            raise ValueError("Patch images require an explicit model_patch_recipe.")
        if control:
            from ruinedfooocus.features.image_editing.control_candidates import CONTROL_KINDS

            if (
                not isinstance(self.controlnet, ControlNetRequest)
                or self.controlnet.type not in CONTROL_KINDS
            ):
                raise ValueError("Select a native ControlNet tool for this named control recipe.")
            if self.control_recipe == "alimama-inpaint":
                if self.controlnet.type != "control" or self.input_image:
                    raise ValueError(
                        "AliMama uses controlnet.type=control and its own source image."
                    )
                if (
                    not self.control_inpaint_image
                    or self.control_inpaint_mode == "mask"
                    and not self.control_inpaint_mask
                ):
                    raise ValueError(
                        "Add the AliMama source and white-edit mask, or choose canvas extension."
                    )
                if self.control_inpaint_mode == "outpaint" and not any(
                    getattr(self, key) for key in native_control_recipes.OUTPAINT_FIELDS[:-1]
                ):
                    raise ValueError("Extend at least one AliMama canvas edge.")
            elif self.control_inpaint_image or self.control_inpaint_mask:
                raise ValueError("Ordinary Union uses input_image as its guide.")
        elif any(getattr(self, key) for key in native_control_recipes.IMAGE_FIELDS):
            raise ValueError("AliMama inputs require control_recipe=alimama-inpaint.")
        return self

    @model_validator(mode="after")
    def validate_native_edit_recipe(self):
        if self.krea_style_lora is not None or self.krea_style_strength is not None:
            if (
                self.native_edit_recipe not in ("auto", "krea2-style")
                or not self.input_image
                or self.mask is not None
                or self.borders is not None
                or self.controlnet is not None
                and self.controlnet.type != "edit"
            ):
                raise ValueError(
                    "Krea style settings require the unmasked native reference operation."
                )
        if self.native_edit_recipe == "auto":
            return self
        if not self.input_image or self.mask is not None or self.borders is not None:
            raise ValueError("A native edit recipe requires an unmasked primary input image.")
        if self.controlnet is not None and self.controlnet.type != "edit":
            raise ValueError("The native edit recipe requires the native edit operation.")
        limit = (
            2
            if self.native_edit_recipe in ("qwen-edit-2509", "qwen-edit-2511", "krea2-style")
            else 15
            if self.native_edit_recipe in ("flux2-reference", "boogu-edit")
            else 0
        )
        if len(self.reference_images or []) > limit:
            raise ValueError(
                "Too many ordered reference images for the selected native edit recipe."
            )
        return self

    output_format: Literal["png", "exr", "avif"] = "png"
    output_bit_depth: str | None = Field(default=None, max_length=32)
    output_color_space: str | None = Field(default=None, max_length=32)
    output_crf: int = Field(default=18, strict=True, ge=1, le=63)

    @model_validator(mode="after")
    def validate_output_format(self):
        from ruinedfooocus.features.image_processing.native_image_formats import format_settings

        format_settings(
            self.model_dump(
                include={"output_format", "output_bit_depth", "output_color_space", "output_crf"}
            )
        )
        return self

    automatic_upscale: str = Field(
        default="Off",
        max_length=1024,
        description="Registered pixel-upscaler selection or filename. size remains the exact final target; native latent padding is removed before enlargement.",
    )
    automatic_upscale_factor: Literal[2, 4] | None = Field(
        default=None,
        description="Optional required native factor; a mismatched selected model is rejected.",
    )
    controlnet: (
        ControlNetRequest
        | DepthMapRequest
        | DepthUtilityRequest
        | SeedVR2Request
        | ImageFilterRequest
        | ImageAdjustmentRequest
        | WeVisOneRequest
        | ImageShaderRequest
        | CompositionRequest
        | LatentCompositionRequest
        | QwenImage21UnionRequest
        | None
    ) = None
    qwen_union_guide: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    qwen_union_source: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    qwen_union_mask: str | None = Field(default=None, max_length=MAX_INLINE_IMAGE_CHARS)
    mask: str | None = Field(
        default=None,
        max_length=MAX_INLINE_IMAGE_CHARS,
        description="Inline PNG with alpha matching the oriented input image. Transparent pixels are editable.",
    )
    borders: OutpaintBorders | None = Field(
        default=None,
        description="Extend the source by whole pixels per side. Mutually exclusive with mask.",
    )
    inpaint_denoise: float | None = Field(default=None, strict=True, gt=0, le=1)
    inpaint_mask_grow: int | None = Field(default=None, strict=True, ge=0, le=128)
    qwen_local_edit_mode: Literal["paint", "mask"] | None = Field(
        default=None,
        description=(
            "Qwen Image 2.1 with a PNG mask only. Paint guides the edit with a red mark. "
            "Mask also preserves source pixels outside the marked area and is the default."
        ),
    )

    @model_validator(mode="after")
    def validate_depth_map(self):
        if isinstance(self.controlnet, ImageShaderRequest):
            if not self.input_image:
                raise ValueError("Curves and GLSL effects require input_image.")
            if self.n != 1:
                raise ValueError("Curves and GLSL effects return one source-sized image.")
        if isinstance(self.controlnet, ImageFilterRequest):
            if not self.input_image:
                raise ValueError("A still-image filter requires input_image.")
            if self.n != 1:
                raise ValueError("A still-image filter returns one source-sized image.")
        if isinstance(self.controlnet, DepthUtilityRequest):
            if not self.input_image or not (self.image_utility_model or "").strip():
                raise ValueError(
                    "This image utility requires input_image and an installed image_utility_model."
                )
            if self.n != 1:
                raise ValueError("A still-image utility returns one image per request.")
            if (
                self.controlnet.type in ("lotus-depth", "marigold-albedo")
                and not (self.image_utility_vae or "").strip()
            ):
                raise ValueError("Select the utility's required VAE.")
            if self.controlnet.type == "marigold-albedo" and not all(
                (value or "").strip()
                for value in (self.image_utility_lora, self.image_utility_conditioning)
            ):
                raise ValueError("Marigold albedo requires its albedo LoRA and saved conditioning.")
        if isinstance(self.controlnet, DepthMapRequest):
            if not self.input_image or not (self.native_vision_model or "").strip():
                raise ValueError(
                    "Depth map requires input_image and an installed native_vision_model."
                )
            if self.n != 1:
                raise ValueError("Depth map returns one image per request.")
        return self

    @model_validator(mode="after")
    def validate_adjustment(self):
        from ruinedfooocus.features.image_processing import still_image_utilities as utilities

        fields = set(utilities.ADJUSTMENT_DEFAULTS) | set(utilities.ADJUSTMENT_IMAGES)
        if not isinstance(self.controlnet, ImageAdjustmentRequest):
            if self.model_fields_set & fields:
                raise ValueError(
                    "Adjustment controls require image-transform, image-mask or image-color."
                )
            return self
        if not (self.input_image or self.editor_context) or self.n != 1:
            raise ValueError("An adjustment requires one source image or owned Editor image.")
        if self.mask is not None or self.borders is not None or self.reference_images:
            raise ValueError(
                "Use adjustment-specific inputs without generation references or borders."
            )
        if self.automatic_upscale not in (None, "Off"):
            raise ValueError(
                "Adjustments use their captured output size without automatic upscaling."
            )
        handler = utilities._handler(self.controlnet.type)
        allowed = set(handler.FIELDS) | set(getattr(handler, "IMAGE_FIELDS", ()))
        if self.model_fields_set & fields - allowed:
            raise ValueError("Use only controls belonging to the selected adjustment group.")
        handler.parameters(self.model_dump(exclude_none=True))
        if self.color_operation == "sdr-to-hdr" and (
            self.output_format != "exr" or self.output_color_space != "HDR"
        ):
            raise ValueError("SDR-to-HDR requires explicit EXR output in HDR color space.")
        return self

    @model_validator(mode="after")
    def validate_utility_outputs(self):
        from ruinedfooocus.features.image_processing.depth_utilities import (
            MOGE_OUTPUT_DEFAULTS,
            FACE_OUTPUT_DEFAULTS,
        )
        from ruinedfooocus.features.image_processing import utility_moge, utility_face

        kind = self.controlnet.type if self.controlnet is not None else None
        for selected, fields, owner in (
            ("moge-depth", MOGE_OUTPUT_DEFAULTS, utility_moge),
            ("face-landmarks", FACE_OUTPUT_DEFAULTS, utility_face),
        ):
            if self.model_fields_set & set(fields) and kind != selected:
                raise ValueError(
                    "Utility output controls require their matching MoGe or face tool."
                )
            if kind == selected:
                owner.parameters(self.model_dump(exclude_none=True))
        return self

    @model_validator(mode="after")
    def validate_seedvr2(self):
        if isinstance(self.controlnet, SeedVR2Request):
            if not self.input_image or not all(
                (value or "").strip() for value in (self.seedvr2_model, self.seedvr2_vae)
            ):
                raise ValueError("SeedVR2 requires input_image, seedvr2_model and seedvr2_vae.")
            if (
                self.n != 1
                or self.reference_images
                or self.mask is not None
                or self.borders is not None
            ):
                raise ValueError(
                    "SeedVR2 restores one still image without references, a mask or borders."
                )
        return self

    @model_validator(mode="after")
    def validate_latent_composition(self):
        from ruinedfooocus.features.image_processing.latent_composition import request_values

        request_values(self.model_dump(exclude_none=True))
        return self

    @model_validator(mode="after")
    def validate_wevisone(self):
        from ruinedfooocus.features.image_processing import wevisone

        if not isinstance(self.controlnet, WeVisOneRequest):
            if self.wevisone_model is not None:
                raise ValueError("A WeVisOne encoder requires WeVisOne detection or matting.")
            return self
        if not self.input_image or not (self.wevisone_model or "").strip():
            raise ValueError("WeVisOne requires a source image and an installed WeVisOne encoder.")
        if (
            self.n != 1
            or self.reference_images
            or self.mask is not None
            or self.borders is not None
        ):
            raise ValueError(
                "WeVisOne processes one still image without references, masks or borders."
            )
        if self.automatic_upscale not in (None, "Off") or any(
            (
                self.native_latent_input,
                self.native_positive_input,
                self.native_negative_input,
                self.native_export_latent,
                self.native_export_conditioning,
                self.native_export_handoff,
            )
        ):
            raise ValueError(
                "WeVisOne preserves its source canvas without generation tensors or upscaling."
            )
        wevisone.parameters(self.model_dump(exclude_none=True))
        return self

    @model_validator(mode="after")
    def validate_native_tensors(self):
        requested = (
            any(
                value is not None
                for value in (
                    self.native_latent_input,
                    self.native_positive_input,
                    self.native_negative_input,
                )
            )
            or self.native_export_latent
            or self.native_export_conditioning
            or self.native_export_handoff
        )
        if requested and (
            self.controlnet is not None
            and self.controlnet.type not in ("edit", "img2img")
            or self.n > 4
            or self.reference_adapter not in (None, "none")
        ):
            raise ValueError(
                "Native tensor files require one through four still generation repetitions with serializable conditioning."
            )
        return self

    @model_validator(mode="after")
    def validate_qwen_union(self):
        if isinstance(self.controlnet, QwenImage21UnionRequest):
            if self.reference_images:
                raise ValueError("Union uses its own guide, source image and mask.")
            if self.controlnet.mode != "inpaint" and not (
                self.qwen_union_guide or self.input_image
            ):
                raise ValueError("Add a Union control guide for the selected guide type.")
            if bool(self.qwen_union_source) != bool(self.qwen_union_mask) or (
                self.controlnet.mode == "inpaint" and not self.qwen_union_source
            ):
                raise ValueError("Add both a Union inpaint source and its black-and-white mask.")
        elif any((self.qwen_union_guide, self.qwen_union_source, self.qwen_union_mask)):
            raise ValueError("Union inputs require controlnet.type=qwen-union.")
        return self

    @model_validator(mode="after")
    def validate_masked_edit(self):
        owned_mask = self.editor_context is not None and self.editor_context.mask_id is not None
        if self.qwen_local_edit_mode is not None and self.mask is None and not owned_mask:
            raise ValueError("qwen_local_edit_mode requires a PNG mask.")
        if self.mask is None and self.borders is None:
            if not owned_mask and (
                self.inpaint_denoise is not None or self.inpaint_mask_grow is not None
            ):
                raise ValueError("Inpainting options require a mask or canvas borders.")
            return self
        if self.mask is not None and self.borders is not None:
            raise ValueError("Choose either a mask or canvas borders.")
        if not self.input_image and self.editor_context is None:
            raise ValueError("A source image is required for masked editing or canvas extension.")
        if (
            self.controlnet is not None
            or self.reference_adapter not in (None, "none")
            or self.reference_adapter_image is not None
            or self.reference_adapter_model is not None
            or self.reference_adapter_vision is not None
            or self.reference_adapter_strength is not None
        ):
            raise ValueError(
                "Masked editing and canvas extension require their own image operation."
            )
        return self


class LoraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    strength: float = Field(default=1, ge=-2, le=2)


ImageRequest.model_rebuild()


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input_image: str | None = Field(default=None, max_length=28000000)
    editor_context: EditorContext | None = None
    response_format: Literal["url", "b64_json"] = "url"
    size: str | None = Field(
        default=None,
        pattern=r"^[0-9]+x[0-9]+$",
        description="Explicit exact target, uniformly fit with padding. Omit for source size multiplied by the model's native factor.",
    )

    @model_validator(mode="after")
    def validate_source(self):
        if bool(self.input_image) == (self.editor_context is not None):
            raise ValueError("Choose one inline image or one owned editor source.")
        return self


class HistogramRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_image: str = Field(min_length=1, max_length=28000000)


HistogramBin = Annotated[int, Field(ge=0, le=16777216)]
HistogramChannel = Annotated[list[HistogramBin], Field(min_length=256, max_length=256)]


class HistogramSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(ge=1, le=16777216)
    height: int = Field(ge=1, le=16777216)
    input_mode: str
    working_mode: Literal["RGB8"]
    has_alpha: bool
    has_icc_profile: bool
    color_policy: str
    alpha_policy: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sha256_layout: str


class HistogramBins(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rgb: HistogramChannel
    luminance: HistogramChannel
    red: HistogramChannel
    green: HistogramChannel
    blue: HistogramChannel


class HistogramResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object: Literal["image.histogram"]
    native_node: Literal["comfy_extras.nodes_curve.ImageHistogram"]
    native_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source: HistogramSource
    quantization: str
    rgb_composite: str
    luminance_weights: tuple[float, float, float]
    bins: HistogramBins


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input_image: str = Field(max_length=28000000)
    prompt: str = Field(default="Describe the image and any visible issues.", max_length=16000)
    system_prompt: str = Field(default="", max_length=16000)
    include_system_prompt: bool = False
    image_detail: Literal["standard", "high"] = "standard"


def _validation_message(error):
    return " ".join(
        f"{'.'.join(map(str, item['loc']))} — {item['msg']}" if item["loc"] else item["msg"]
        for item in error.errors()
    )


def _bounded_edit_request(request):
    received = 0

    async def receive():
        nonlocal received
        message = await request.receive()
        received += len(message.get("body", b""))
        if received > MAX_EDIT_FORM_BYTES:
            raise HTTPException(413, "Image edit form exceeds the size limit.")
        return message

    return Request(request.scope, receive=receive)


async def _read_edit_upload(upload, label):
    if not isinstance(upload, UploadFile):
        raise HTTPException(400, f"Upload {label} as a file.")
    raw = await upload.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if len(raw) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(400, f"The {label} exceeds the size limit.")
    return base64.b64encode(raw).decode()


def create_apps(backend, key=None):
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if key and (
            credentials is None or not secrets.compare_digest(credentials.credentials, key)
        ):
            raise HTTPException(401, "Invalid API key.", headers={"WWW-Authenticate": "Bearer"})

    api = FastAPI(title="RuinedFooocus API", version="1", dependencies=[Depends(authorize)])
    tools = FastAPI(title="RuinedFooocus tools", version="1", dependencies=[Depends(authorize)])
    assets = {}
    editor_owner = (
        "api",
        "image-tools",
        hashlib.sha256(("rf-image-tools:" + key).encode("utf-8")).hexdigest()
        if key
        else secrets.token_hex(16),
    )

    def inline_image(value):
        if value and value.startswith(("http://", "https://")):
            path = urlparse(value).path
            token = path.removeprefix("/v1/images/")
            entry = assets.get(token) if path.startswith("/v1/images/") else None
            if entry is None or time.time() - entry[1] > 86400 or not entry[0].is_file():
                raise HTTPException(
                    400, "Use an image data URL or an unexpired image URL returned by this API."
                )
            if entry[0].stat().st_size > MAX_IMAGE_UPLOAD_BYTES:
                raise HTTPException(400, "Input image exceeds the size limit.")
            return base64.b64encode(entry[0].read_bytes()).decode()
        return value

    async def http_error(request, error):
        return JSONResponse(
            {
                "error": {
                    "message": str(error.detail),
                    "type": "invalid_request_error",
                    "code": error.status_code,
                }
            },
            status_code=error.status_code,
            headers=error.headers,
        )

    async def validation_error(request, error):
        endpoint = request.scope.get("endpoint")
        private_chat = (
            endpoint is chat_completions
            and isinstance(error.body, dict)
            and "rf_attachments" in error.body
        )
        message = (
            _validation_message(error)
            if private_chat
            or endpoint in (image_generations, image_generation, editor_open, rf_image_tool)
            else str(error)
        )
        return JSONResponse(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
            status_code=400,
        )

    for app in (api, tools):
        app.add_exception_handler(HTTPException, http_error)
        app.add_exception_handler(RequestValidationError, validation_error)

    def select(model, kind):
        try:
            return backend.resolve_model(model, kind)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    def submit(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except PermissionError as error:
            raise HTTPException(403, "This image handle belongs to another session.") from error
        except CancelledError as error:
            raise HTTPException(
                409, "The editor image changed. Refresh its current revision."
            ) from error
        except RuntimeError as error:
            raise HTTPException(429, str(error)) from error

    async def events(job, request):
        deadline = time.monotonic() + 1800
        try:
            while True:
                if await request.is_disconnected():
                    return
                if time.monotonic() >= deadline:
                    raise HTTPException(504, "Generation timed out.")
                try:
                    kind, value = await asyncio.to_thread(job.events.get, True, 0.25)
                except queue.Empty:
                    continue
                if kind == "error":
                    raise HTTPException(500, value)
                if kind == "invalid_request":
                    raise HTTPException(400, value)
                yield kind, value
                if kind == "result":
                    return
        finally:
            job.cancelled.set()

    @api.get("/models")
    @tools.get("/models", operation_id="available_models")
    def list_models():
        return {
            "object": "list",
            "data": [
                dict(
                    id=name,
                    object="model",
                    created=0,
                    owned_by="ruinedfooocus",
                    type=info["kind"],
                    **(
                        {
                            "family": "Unknown",
                            "generation_available": False,
                            "unavailable_reason": info["unavailable_reason"],
                        }
                        if info.get("unavailable_reason")
                        else {}
                    ),
                )
                for name, info in backend.models().items()
            ],
        }

    @api.get("/capabilities")
    @tools.get("/capabilities", operation_id="generation_settings")
    def capabilities():
        return backend.capabilities()

    @api.post("/rf/editor/sessions")
    def editor_open(body: EditorOpenRequest):
        payload = body.model_dump()
        payload["input_image"] = inline_image(payload["input_image"])
        return submit(backend.open_editor_session, payload, owner=editor_owner)

    @api.get("/rf/editor/sessions/{session_id}")
    def editor_snapshot(session_id: str):
        return submit(backend.editor_session_state, session_id, owner=editor_owner)

    @api.post("/rf/editor/sessions/{session_id}/{action}")
    def editor_move(session_id: str, action: Literal["undo", "redo"], body: EditorMoveRequest):
        return submit(
            backend.editor_session_state,
            session_id,
            owner=editor_owner,
            action=action,
            revision=body.revision,
        )

    def tool_reply(value, body, request):
        from ruinedfooocus.features.chat.chat_tools import public_image_tool_result

        result = public_image_tool_result(value)
        data = []
        for filename in value.get("images", ()):
            path = Path(filename)
            if body.response_format == "b64_json":
                data.append({"b64_json": base64.b64encode(path.read_bytes()).decode("ascii")})
            else:
                now = time.time()
                for token in list(assets):
                    if now - assets[token][1] > 86400:
                        assets.pop(token)
                while len(assets) >= 256:
                    assets.pop(next(iter(assets)))
                token = secrets.token_urlsafe(24)
                assets[token] = (path, now)
                data.append({"url": str(request.url.replace(path="/v1/images/" + token, query=""))})
        result["data"] = data
        return result

    @api.post("/rf/image-tools")
    async def rf_image_tool(body: RFImageToolRequest, request: Request):
        job = submit(backend.image_tool, body.model_dump(exclude_none=True), owner=editor_owner)
        if body.stream:

            async def stream_tool():
                try:
                    async for kind, value in events(job, request):
                        if kind == "result":
                            yield "data: " + json.dumps(tool_reply(value, body, request)) + "\n\n"
                except HTTPException as error:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "call_id": body.call.get("id"),
                                "status": "error",
                                "error": {"code": error.status_code, "message": str(error.detail)},
                            }
                        )
                        + "\n\n"
                    )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_tool(),
                media_type="text/event-stream",
                headers={"Cache-Control": "private, no-store"},
            )
        async for kind, value in events(job, request):
            if kind == "result":
                return JSONResponse(
                    tool_reply(value, body, request), headers={"Cache-Control": "private, no-store"}
                )
        raise HTTPException(499, "Client disconnected.")

    @api.post("/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request):
        model, selected = select(body.model, "chat")
        payload = body.model_dump(exclude_none=True)
        payload["model"] = model
        job = submit(backend.chat, payload, selected)
        cache_control = "private, no-store" if body.rf_attachments is not None else "no-cache"
        if body.stream:

            async def stream():
                try:
                    async for kind, value in events(job, request):
                        if kind == "chunk":
                            yield "data: " + json.dumps(value) + "\n\n"
                except HTTPException as error:
                    yield (
                        "data: "
                        + json.dumps(
                            {"error": {"message": error.detail, "code": error.status_code}}
                        )
                        + "\n\n"
                    )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream(), media_type="text/event-stream", headers={"Cache-Control": cache_control}
            )
        async for kind, value in events(job, request):
            if kind == "result":
                if body.rf_attachments is not None:
                    return JSONResponse(value, headers={"Cache-Control": cache_control})
                return value
        raise HTTPException(499, "Client disconnected.")

    async def generate(body, request, kind="image"):
        from ruinedfooocus.features.image_processing.image_composition import handles
        from ruinedfooocus.features.image_processing.still_image_utilities import (
            KINDS as DEPTH_UTILITIES,
        )

        if kind != "image" and any(
            getattr(body, field, None)
            for field in (
                "native_latent_input",
                "native_positive_input",
                "native_negative_input",
                "native_export_latent",
                "native_export_conditioning",
                "native_export_handoff",
                "latent_composition_primary",
                "latent_composition_secondary",
                "latent_composition_mask",
            )
        ):
            raise HTTPException(
                400, "Native tensor inputs and exports require still-image generation."
            )
        native_utility = (
            kind == "image"
            and body.controlnet is not None
            and (
                body.controlnet.type
                in (
                    "detect",
                    "segment",
                    "depth-map",
                    "restore-seedvr2",
                    "latent-compose",
                    *DEPTH_UTILITIES,
                )
                or handles(body.controlnet.type)
            )
        )
        if native_utility:
            selected = {}  # The typed utility model is selected independently.
            resolved_model = body.model
        else:
            resolved_model, selected = select(body.model, kind)
        operation = backend.upscale if kind == "upscaler" else backend.images
        payload = body.model_dump(exclude_none=True)
        payload["model"] = resolved_model
        if kind == "image":
            from ruinedfooocus.features.models.image_scope import require_image_operation

            try:
                require_image_operation(payload)
            except ValueError as error:
                raise HTTPException(400, str(error)) from None
        # Explicit null selects Auto/Off; omission inherits the UI/default snapshot.
        from ruinedfooocus.features.image_editing.image_guidance import FIELD_SCHEMA

        for key, spec in FIELD_SCHEMA.items():
            if spec["default"] is None and key in body.model_fields_set:
                payload[key] = getattr(body, key)
        for key in ("gligen_model", "hypernetwork_model"):
            if key in body.model_fields_set:
                payload[key] = getattr(body, key)
        if payload.get("input_image"):
            payload["input_image"] = inline_image(payload["input_image"])
        if payload.get("control_guide"):
            payload["control_guide"]["image"] = inline_image(payload["control_guide"]["image"])
        if payload.get("latent_composition_mask"):
            payload["latent_composition_mask"] = inline_image(payload["latent_composition_mask"])
        if payload.get("composition_images"):
            payload["composition_images"] = [
                inline_image(value) for value in payload["composition_images"]
            ]
        if payload.get("composition_mask"):
            payload["composition_mask"] = inline_image(payload["composition_mask"])
        for key in (
            "qwen_union_guide",
            "qwen_union_source",
            "qwen_union_mask",
            "model_patch_image",
            "model_patch_inpaint_image",
            "model_patch_mask",
            "control_inpaint_image",
            "control_inpaint_mask",
        ):
            if payload.get(key):
                payload[key] = inline_image(payload[key])
        if payload.get("reference_images"):
            payload["reference_images"] = [
                inline_image(value) for value in payload["reference_images"]
            ]
        if payload.get("cascade_vision_images"):
            payload["cascade_vision_images"] = [
                inline_image(value) for value in payload["cascade_vision_images"]
            ]
        job = submit(
            operation,
            payload,
            selected,
            **({"owner": editor_owner} if payload.get("editor_context") is not None else {}),
        )
        async for kind, value in events(job, request):
            if kind != "result":
                continue
            data = []
            for filename in value:
                path = Path(filename)
                if body.response_format == "b64_json":
                    data.append({"b64_json": base64.b64encode(path.read_bytes()).decode()})
                else:
                    now = time.time()
                    for token in list(assets):
                        if now - assets[token][1] > 86400:
                            assets.pop(token)
                    while len(assets) >= 256:
                        assets.pop(next(iter(assets)))
                    token = secrets.token_urlsafe(24)
                    assets[token] = (path, now)
                    # Both apps live on the same host; media is served by /v1.
                    url = str(request.url.replace(path="/v1/images/" + token, query=""))
                    data.append({"url": url})
            reply = {"created": int(time.time()), "data": data}
            if getattr(job, "editor_result", None) is not None:
                reply["editor"] = job.editor_result
            from ruinedfooocus.features.image_processing.wevisone import WeVisOneArtifacts

            if isinstance(value, WeVisOneArtifacts):
                reply["wevisone"] = value.wevisone
            tensor_files = getattr(value, "tensor_files", ())
            if tensor_files:
                from ruinedfooocus.features.models.model_identity import file_signature

                tensors = []
                for artifact in tensor_files:
                    if tuple(file_signature(artifact.path)) != artifact.signature:
                        raise HTTPException(
                            400, "A generated native tensor changed before download."
                        )
                    path = Path(artifact.path)
                    entry = {
                        "kind": artifact.kind,
                        "family": artifact.family,
                        "sha256": artifact.sha256,
                        "seed": artifact.seed,
                        "generation_index": artifact.generation_index,
                    }
                    if body.response_format == "b64_json":
                        entry["b64_json"] = base64.b64encode(path.read_bytes()).decode()
                    else:
                        token = secrets.token_urlsafe(24)
                        assets[token] = (path, time.time())
                        entry["url"] = str(
                            request.url.replace(path="/v1/images/" + token, query="")
                        )
                    tensors.append(entry)
                reply["native_tensors"] = tensors
            return reply
        raise HTTPException(499, "Client disconnected.")

    @api.post("/images/generations")
    async def image_generations(body: ImageRequest, request: Request):
        return await generate(body, request)

    @api.post(
        "/images/edits",
        openapi_extra={
            "requestBody": {
                "required": True,
                "description": (
                    "Upload exactly one source image using image or image[]. "
                    f"Each file may contain up to {MAX_IMAGE_UPLOAD_BYTES:,} bytes."
                ),
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "oneOf": [{"required": ["image"]}, {"required": ["image[]"]}],
                            "not": {"required": ["mask", "borders"]},
                            "properties": {
                                "image": {"type": "string", "format": "binary"},
                                "image[]": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "Alternate field name for the single source image.",
                                },
                                "mask": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": (
                                        "PNG with alpha matching the oriented source dimensions. "
                                        "Transparent pixels are editable."
                                    ),
                                },
                                "borders": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "contentSchema": {
                                        "$ref": "#/components/schemas/OutpaintBorders"
                                    },
                                    "description": "JSON text. At least one border must be positive.",
                                    "examples": ['{"left":128,"bottom":96,"feather":16}'],
                                },
                                "model": {"type": "string", "default": "image/default"},
                                "prompt": {"type": "string", "maxLength": MAX_IMAGE_PROMPT_CHARS},
                                "size": {"type": "string", "pattern": r"^(auto|[0-9]+x[0-9]+)$"},
                                "response_format": {"type": "string", "enum": ["url", "b64_json"]},
                                "controlnet": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON image tool settings. Use depth-map with native_vision_model for a standalone depth map.",
                                    "examples": ['{"type":"depth-map"}'],
                                },
                                "native_vision_model": {
                                    "type": "string",
                                    "description": "Installed model in the configured roots for the selected image tool.",
                                },
                                "filter_radius": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON integer 1–31 for image-blur/image-sharpen, smaller than both source dimensions.",
                                },
                                "curve_master": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "2–32 normalized [input, output] pairs for the master RGB curve.",
                                },
                                "curve_red": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "2–32 normalized [input, output] pairs for the red curve.",
                                },
                                "curve_green": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "2–32 normalized [input, output] pairs for the green curve.",
                                },
                                "curve_blue": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "2–32 normalized [input, output] pairs for the blue curve.",
                                },
                                "curve_interpolation": {
                                    "type": "string",
                                    "enum": ["linear", "monotone_cubic"],
                                },
                                "shader_effect": {
                                    "type": "string",
                                    "enum": ["invert", "bt709-grayscale", "posterize"],
                                },
                                "shader_amount": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON number 0–1 for the selected named effect.",
                                },
                                "shader_levels": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON integer 2–32 for posterize.",
                                },
                                "filter_sigma": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON number 0.1–10 for image-blur/image-sharpen.",
                                },
                                "filter_strength": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON number 0–5 for image-sharpen.",
                                },
                                "filter_factor": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON number 0–2 for adjust-brightness/adjust-contrast; 1 preserves RGB.",
                                },
                                "inpaint_denoise": {
                                    "type": "string",
                                    "description": "JSON number greater than 0 and at most 1. Requires mask or borders.",
                                },
                                "inpaint_mask_grow": {
                                    "type": "string",
                                    "description": "JSON integer from 0 to 128. Requires mask or borders.",
                                },
                                "qwen_local_edit_mode": {
                                    "type": "string",
                                    "enum": ["paint", "mask"],
                                    "description": "Qwen Image 2.1 with a PNG mask only. Defaults to mask.",
                                },
                            },
                        },
                        "encoding": {"mask": {"contentType": "image/png"}},
                    },
                },
            },
        },
    )
    async def image_edits(request: Request):
        """Edit one uploaded image with an optional PNG mask or JSON canvas borders."""
        async with _bounded_edit_request(request).form(
            max_files=2, max_fields=20, max_part_size=MAX_INLINE_IMAGE_CHARS
        ) as form:
            uploads = form.getlist("image") + form.getlist("image[]")
            if len(uploads) != 1 or not isinstance(uploads[0], UploadFile):
                raise HTTPException(400, "Upload exactly one image using image or image[].")
            masks = form.getlist("mask")
            if len(masks) > 1:
                raise HTTPException(400, "Upload at most one mask file.")
            values = {
                name: value
                for name, value in form.items()
                if name not in ("image", "image[]", "mask")
            }
            for name in (
                "styles",
                "loras",
                "controlnet",
                "reference_images",
                "borders",
                "inpaint_denoise",
                "inpaint_mask_grow",
                "filter_radius",
                "seedvr2_scale",
                "wevisone_max_length",
                "native_tensor_workflow",
                "native_export_handoff",
                "native_export_latent",
                "native_export_conditioning",
                "filter_sigma",
                "filter_strength",
                "filter_factor",
                "curve_master",
                "curve_red",
                "curve_green",
                "curve_blue",
                "shader_amount",
                "shader_levels",
            ):
                if name in values:
                    try:
                        values[name] = json.loads(values[name])
                    except (TypeError, ValueError) as error:
                        raise HTTPException(400, f"{name} must be JSON.") from error
            values["input_image"] = await _read_edit_upload(uploads[0], "source image")
            if masks:
                values["mask"] = await _read_edit_upload(masks[0], "PNG mask")
        try:
            body = ImageRequest.model_validate(values)
        except ValidationError as error:
            raise HTTPException(400, _validation_message(error)) from error
        return await generate(body, request)

    @tools.post("/image_generation", operation_id="image_generation")
    async def image_generation(body: ImageRequest, request: Request):
        """Create an image now when the user asks to draw, generate or regenerate one.

        Supply a visual prompt and pass any requested resolution in size as WIDTHxHEIGHT.
        Otherwise keep defaults. No extra confirmation is needed for a generation request.
        Do not generate for prompt-only requests. Display the returned URL as a Markdown image.
        """
        return await generate(body, request)

    @api.post("/images/upscale")
    @tools.post("/image_upscale", operation_id="image_upscale")
    async def image_upscale(body: UpscaleRequest, request: Request):
        """Upscale an inline image using a known upscaler model ID."""
        return await generate(body, request, "upscaler")

    @api.post("/images/histogram", response_model=HistogramResponse)
    async def image_histogram(body: HistogramRequest, request: Request):
        payload = {"input_image": inline_image(body.input_image)}
        job = submit(backend.histogram, payload)
        async for kind, value in events(job, request):
            if kind == "result":
                return value
        raise HTTPException(499, "Client disconnected.")

    @api.post("/images/review")
    @tools.post("/image_review", operation_id="image_review")
    async def image_review(body: ReviewRequest, request: Request):
        """Inspect an inline image with a vision model and return feedback, without generating a revision."""
        _, selected = select(body.model, "vision")
        payload = body.model_dump()
        payload["input_image"] = inline_image(payload["input_image"])
        job = submit(backend.review, payload, selected)
        async for kind, value in events(job, request):
            if kind == "result":
                return {"feedback": value}
        raise HTTPException(499, "Client disconnected.")

    # Opaque expiring URLs can be rendered by clients that cannot attach API headers.
    # The mapping contains only files generated by this API, never caller-supplied paths.
    async def image_file(token: str):
        entry = assets.get(token)
        if entry is None or time.time() - entry[1] > 86400 or not entry[0].is_file():
            raise HTTPException(404, "Image URL expired or was not found.")
        return FileResponse(
            entry[0],
            filename=entry[0].name
            if entry[0].suffix.lower() in (".latent", ".safetensors")
            else None,
            media_type={".exr": "image/x-exr", ".avif": "image/avif"}.get(
                entry[0].suffix.lower(),
                mimetypes.guess_type(entry[0].name)[0] or "application/octet-stream",
            ),
        )

    # A separate router avoids inheriting the bearer dependency for capability URLs.
    from fastapi import APIRouter

    media = APIRouter()
    media.add_api_route("/images/{token}", image_file, methods=["GET"], include_in_schema=False)
    api.router.routes.extend(media.routes)
    return api, tools


def validate_options(args):
    key = args.api_key or os.environ.get("RF_API_KEY")
    listen = args.listen or os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    if args.api and (args.share or listen not in ("localhost", "127.0.0.1", "::1")) and not key:
        raise ValueError("--api with --listen or --share requires --api-key or RF_API_KEY.")
    return key


def install(app, key=None):
    import ruinedfooocus.app.api_runtime as api_runtime

    api, tools = create_apps(api_runtime, key)
    app.mount("/v1", api)
    app.mount("/tools", tools)
    app.state.rf_openai_api_requires_key = bool(key)
