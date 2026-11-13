"""Model-free native blend, reference grading and explicit palette quantization."""

from dataclasses import dataclass
import re

from ruinedfooocus.features.image_processing import color_native as native
from ruinedfooocus.features.image_processing.image_geometry_tools import (
    FrozenPixels,
    NATIVE_REVISION,
    capture_source,
)
from ruinedfooocus.features.image_processing.native_vision import _number

KINDS = ("image-color",)
LABELS = {"image-color": "Finish color"}
KEY = "_image_color_request"
OPERATIONS = ("blend", "transfer", "quantize", "text-overlay", "morphology", "sdr-to-hdr")
BLEND_MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "difference")
TRANSFER_METHODS = ("reinhard_lab", "mkl_lab", "histogram")
DITHERS = ("none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16")
MORPHOLOGY_OPERATIONS = ("erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat")
TEXT_POSITIONS = ("top", "bottom")
TEXT_ALIGNMENTS = ("left", "center", "right")
MAX_MORPHOLOGY_WORK = 64 * 1024 * 1024
DEFAULTS = dict(
    color_operation="blend",
    color_blend_mode="normal",
    color_blend_factor=0.5,
    color_transfer_method="reinhard_lab",
    color_transfer_strength=1.0,
    color_quantize_colors=256,
    color_quantize_dither="none",
    color_text="",
    color_text_font_size=5.0,
    color_text_color="#FFFFFF",
    color_text_position="top",
    color_text_align="left",
    color_text_outline=True,
    color_morphology_operation="erode",
    color_morphology_kernel=3,
    color_hdr_luminance=650.0,
)
FIELDS = tuple(DEFAULTS)
IMAGE_FIELDS = ("color_reference",)
NODES = {
    "blend": "ImageBlend",
    "transfer": "ColorTransfer",
    "quantize": "ImageQuantize",
    "text-overlay": "TextOverlay",
    "morphology": "Morphology",
    "sdr-to-hdr": "ImageSDRToHDR",
}


@dataclass(frozen=True)
class ColorRequest:
    source: FrozenPixels
    reference: FrozenPixels | None
    parameters: tuple[tuple[str, object], ...]
    native_signatures: tuple
    output_settings: tuple[tuple[str, object], ...]


def parameters(data):
    values = {name: data.get(name, default) for name, default in DEFAULTS.items()}
    for name, choices in (
        ("color_operation", OPERATIONS),
        ("color_blend_mode", BLEND_MODES),
        ("color_transfer_method", TRANSFER_METHODS),
        ("color_quantize_dither", DITHERS),
        ("color_text_position", TEXT_POSITIONS),
        ("color_text_align", TEXT_ALIGNMENTS),
        ("color_morphology_operation", MORPHOLOGY_OPERATIONS),
    ):
        if not isinstance(values[name], str) or values[name] not in choices:
            raise ValueError(f"Choose a supported {name.replace('_', ' ')}.")
    values["color_blend_factor"] = _number(values["color_blend_factor"], "Blend factor", 0, 1)
    values["color_transfer_strength"] = _number(
        values["color_transfer_strength"], "Color transfer strength", 0, 10
    )
    values["color_quantize_colors"] = _number(
        values["color_quantize_colors"], "Palette colors", 1, 256, integer=True
    )
    values["color_text_font_size"] = _number(
        values["color_text_font_size"], "Text font size", 0.5, 50
    )
    values["color_morphology_kernel"] = _number(
        values["color_morphology_kernel"], "Morphology kernel", 3, 999, integer=True
    )
    values["color_hdr_luminance"] = _number(
        values["color_hdr_luminance"], "HDR target luminance", 0, 10000
    )
    if type(values["color_text_outline"]) is not bool:
        raise ValueError("Text outline must be true or false.")
    if not isinstance(values["color_text"], str) or len(values["color_text"]) > 2048:
        raise ValueError("Text overlays accept at most 2048 text characters.")
    if values["color_text"].replace("\\n", "\n").count("\n") >= 64:
        raise ValueError("Text overlays accept at most 64 lines.")
    if not isinstance(values["color_text_color"], str) or not re.fullmatch(
        r"#[0-9a-fA-F]{6}", values["color_text_color"]
    ):
        raise ValueError("Choose a six-digit RGB text color.")
    values["color_text_color"] = values["color_text_color"].upper()
    return values


def _samples(source):
    import numpy as np

    array = np.frombuffer(source.pixels, dtype=source.dtype).reshape(
        source.size[1], source.size[0], source.channels
    )
    array = array.astype(np.float32) / (
        255 if source.dtype == "uint8" else 65535 if source.dtype == "uint16" else 1
    )
    return np.repeat(array, 3, axis=-1) if source.channels == 1 else array


def capture_assets(data, manager, models=None):
    from PIL import Image
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings

    values = parameters(data)
    if (
        data.get("inpaint_toggle")
        or data.get("model_recipe_path")
        or data.get("reference_adapter") not in (None, "", "none")
    ):
        raise ValueError(
            "Turn off inpainting, generation recipes and reference adapters before finishing color."
        )
    source = capture_source(data.get("input_image"), data.get("image_source"))
    operation = values["color_operation"]
    reference = None
    if operation in ("blend", "transfer"):
        if data.get("color_reference") is None:
            raise ValueError("Add a reference image for blending or color transfer.")
        reference = capture_source(data["color_reference"])
        if operation == "blend" and reference.size != source.size:
            raise ValueError("Blend images must share one oriented frame; resize explicitly first.")
        if source.color_space != reference.color_space:
            raise ValueError(
                "The source and reference color spaces must match; convert color explicitly first."
            )
    elif data.get("color_reference") is not None:
        raise ValueError("This operation uses only the main image; clear the reference image.")
    sdr_only = operation in ("transfer", "quantize", "text-overlay", "sdr-to-hdr")
    if sdr_only and source.color_space != "sRGB":
        raise ValueError(
            "This native operation requires encoded sRGB; convert color explicitly first."
        )
    if operation == "text-overlay":
        if source.dtype != "uint8":
            raise ValueError(
                "Native text overlays require an 8-bit sRGB source; make an explicit working copy first."
            )
        pixel_size = max(1, round(values["color_text_font_size"] / 100 * source.size[1]))
        if pixel_size > 512 or pixel_size * len(values["color_text"]) > 262144:
            raise ValueError(
                "Text overlay font work is too large; reduce text length or font size."
            )
    if operation == "morphology":
        passes = 1 if values["color_morphology_operation"] in ("erode", "dilate") else 2
        work = source.size[0] * source.size[1] * values["color_morphology_kernel"] ** 2 * passes
        if work > MAX_MORPHOLOGY_WORK:
            raise ValueError(
                "Native morphology exceeds the 64-million-sample work budget; reduce image or kernel size."
            )
    for frame in (source, reference):
        if frame is None:
            continue
        samples = _samples(frame)
        if frame.channels == 4 and (samples[..., 3].min() < 0 or samples[..., 3].max() > 1):
            raise ValueError("Image alpha must stay between zero and one.")
        if sdr_only and (samples[..., :3].min() < 0 or samples[..., :3].max() > 1):
            raise ValueError(
                "This native operation needs RGB values from zero to one; tone-map HDR explicitly first."
            )
    choice = format_settings(data)
    if operation == "morphology" and choice["output_format"] != "exr":
        rgb = _samples(source)[..., :3]
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Out-of-range floating-point morphology requires EXR output.")
    if source.channels == 4 and choice["output_format"] == "avif":
        raise ValueError("Native AVIF cannot preserve alpha. Choose PNG or EXR.")
    output_color_space = "HDR" if operation == "sdr-to-hdr" else source.color_space
    if choice["output_color_space"] != output_color_space:
        raise ValueError(
            "SDR-to-HDR requires HDR output color space."
            if operation == "sdr-to-hdr"
            else "Output color space must match the source; convert color explicitly first."
        )
    if operation == "sdr-to-hdr" and choice["output_format"] != "exr":
        raise ValueError("SDR-to-HDR requires floating-point EXR output with HDR color space.")
    if source.color_space != "sRGB" and choice["output_format"] != "exr":
        raise ValueError("Linear and HDR image blending requires EXR output.")
    retains_precision = operation != "quantize" or source.channels == 4
    if (
        retains_precision
        and any(frame is not None and frame.dtype != "uint8" for frame in (source, reference))
        and not (
            choice["output_format"] == "exr"
            or (choice["output_format"] == "png" and choice["output_bit_depth"] == "16-bit")
        )
    ):
        raise ValueError("Precision color tools require PNG16 or EXR output.")
    if choice["output_format"] == "avif" and min(source.size) < 64:
        raise ValueError("Native still AVIF needs dimensions of at least 64 pixels per side.")
    # Float samples are frozen unchanged; native blend's documented clamp is an
    # effect policy, never an implicit uint8 working-copy conversion.
    request = ColorRequest(
        source, reference, tuple(values.items()), native.source_signatures(), tuple(choice.items())
    )
    result = dict(
        data, **values, controlnet={"type": KINDS[0]}, input_image=Image.new("L", source.size)
    )
    result[KEY] = request
    result.update(choice)
    result.pop("color_reference", None)
    result.pop("image_utility_source", None)
    return result, ()


def worker_data(data):
    from PIL import Image
    from ruinedfooocus.features.generation.image_preparation import KEY as ASSET_KEY
    from ruinedfooocus.features.generation.request_metadata import SOURCE_GENERATION
    from ruinedfooocus.features.gallery.editor_session import KEY as EDITOR_KEY
    from ruinedfooocus.features.gallery.image_metadata import generation_provenance
    from ruinedfooocus.features.image_processing.image_filters import worker_data as common

    request = data.get(KEY)
    if not isinstance(request, ColorRequest):
        raise ValueError("Capture the color-tool images before processing them.")
    result = {
        key: value
        for key, value in data.items()
        if key
        in (
            KEY,
            ASSET_KEY,
            EDITOR_KEY,
            SOURCE_GENERATION,
            "_image_workflow_step",
            "_api_job",
            "_job",
            "task_id",
            "task_type",
            "index",
            "silent",
            "show_preview",
            "output_format",
            "output_bit_depth",
            "output_color_space",
            "output_crf",
        )
    }
    provenance = generation_provenance(result.pop(SOURCE_GENERATION, None))
    if provenance is not None:
        result[SOURCE_GENERATION] = provenance
    result.update(dict(request.parameters))
    result.update(dict(request.output_settings))
    result.update(controlnet={"type": KINDS[0]}, input_image=Image.new("L", request.source.size))
    return common(result)


def process(owner, data, manager, check_cancelled):
    import numpy as np
    import torch
    from PIL import Image
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output
    from ruinedfooocus.features.image_processing.utility_device import run_utility

    request = data.get(KEY)
    data.pop("image_utility_source", None)
    if not isinstance(request, ColorRequest) or data.get("controlnet", {}).get("type") != KINDS[0]:
        raise ValueError("Capture this color tool's images before processing. Submit them again.")
    values = dict(request.parameters)
    data.update(dict(request.output_settings))
    operation = values["color_operation"]
    check_cancelled()
    width, height = request.source.size
    channels = 4 if request.source.channels == 4 else 3

    def execute(policy):
        # Every attempt owns new tensors. A failed accelerator attempt cannot
        # leave partially modified inputs in the CPU recovery attempt.
        ops = native.operations(
            request.native_signatures,
            check_cancelled,
            policy=policy,
            transfer=operation == "transfer",
            finishing=operation
            if operation in ("text-overlay", "morphology", "sdr-to-hdr")
            else None,
        )
        with torch.device(policy.device), torch.inference_mode():
            source = (
                torch.from_numpy(_samples(request.source).copy()).unsqueeze(0).to(policy.device)
            )
            reference = (
                torch.from_numpy(_samples(request.reference).copy()).unsqueeze(0).to(policy.device)
                if request.reference
                else None
            )
            if operation == "blend":
                result = ops.Blend.execute(
                    source[..., :3],
                    reference[..., :3],
                    values["color_blend_factor"],
                    values["color_blend_mode"],
                )[0]
            elif operation == "transfer":
                result = ops.ColorTransfer.execute(
                    source[..., :3],
                    reference[..., :3],
                    values["color_transfer_method"],
                    {"source_stats": "per_frame"},
                    values["color_transfer_strength"],
                )[0]
            elif operation == "quantize":
                result = ops.Quantize.execute(
                    source, values["color_quantize_colors"], values["color_quantize_dither"]
                )[0]
            elif operation == "text-overlay":
                result = ops.TextOverlay.execute(
                    source,
                    values["color_text"],
                    values["color_text_font_size"],
                    values["color_text_color"],
                    values["color_text_position"],
                    values["color_text_align"],
                    values["color_text_outline"],
                )[0]
            elif operation == "morphology":
                result = ops.Morphology.execute(
                    source[..., :3],
                    values["color_morphology_operation"],
                    values["color_morphology_kernel"],
                )[0]
            else:
                result = ops.ImageSDRToHDR.execute(
                    source[..., :3],
                    {"algorithm": "lottes", "target_luminance": values["color_hdr_luminance"]},
                )[0]
            if request.source.channels == 4 and operation not in ("quantize", "text-overlay"):
                # Native intermediates may live on CPU; keep original alpha at
                # float32 on the native output device.
                result = torch.cat((result, source[..., 3:].to(result.device)), dim=-1)
        check_cancelled()
        if native.source_signatures() != request.native_signatures:
            raise ValueError(
                "The native color source changed during processing. Submit the images again."
            )
        if (
            not isinstance(result, torch.Tensor)
            or tuple(result.shape) != (1, height, width, channels)
            or not torch.isfinite(result).all()
        ):
            raise ValueError(
                "The native color operation returned invalid dimensions or non-finite pixels."
            )
        # Include synchronization/copy in the guarded attempt: an asynchronous
        # device failure here must receive the same CPU recovery as execute().
        return result[0].detach().cpu().float().numpy().copy()

    host_reason = {
        "text-overlay": "The native text overlay uses Pillow's bundled-font renderer on host images.",
        "quantize": "The native palette quantizer requires NumPy/Pillow host images.",
    }.get(operation)
    execution = run_utility(
        NODES[operation], execute, check_cancelled=check_cancelled, cpu_only_reason=host_reason
    )
    pixels = execution.value
    information = dict(
        operation=KINDS[0],
        native_node=NODES[operation],
        native_revision=NATIVE_REVISION,
        width=width,
        height=height,
        source_sha256=request.source.source_sha256,
        source_dtype=request.source.dtype,
        color_space="HDR" if operation == "sdr-to-hdr" else request.source.color_space,
        source_color_space=request.source.color_space,
        has_icc_profile=request.source.has_icc_profile,
        reference_sha256=request.reference.source_sha256 if request.reference else None,
        reference_size=list(request.reference.size) if request.reference else None,
        frame="oriented source frame; no implicit reference resize or crop",
        alpha="native source-over glyph alpha union; alpha outside glyphs unchanged"
        if operation == "text-overlay"
        else "source alpha samples preserved; reference alpha does not select pixels",
        working_dtype="float32",
        compute=execution.metadata(),
        color_policy="native sRGB to Rec.2020 HLG expansion; EXR writer receives HDR input"
        if operation == "sdr-to-hdr"
        else "native encoded samples; no ICC or color-space conversion",
        output_precision=(
            "RGB palette uses native uint8 quantization; alpha float samples preserved"
            if operation == "quantize"
            else "native bundled-default-font RGBA8 glyph layer on explicit sRGB8 source; native glyph alpha union"
            if operation == "text-overlay"
            else "native Kornia float32 RGB morphology; source alpha retained"
            if operation == "morphology"
            else "native float32 Rec.2020 HLG; EXR codec stores scene-linear samples"
            if operation == "sdr-to-hdr"
            else "native 256-bin histogram matching; alpha float samples preserved"
            if operation == "transfer" and values["color_transfer_method"] == "histogram"
            else "native float32; blend and nonzero color transfer clamp RGB to zero through one"
        ),
        **values,
    )
    check_cancelled()
    data["image_utility_source"] = information
    if advanced_output(data):
        return [pixels]
    return [Image.fromarray((np.clip(pixels, 0, 1) * 255).round().astype(np.uint8))]


def output_metadata(data):
    from copy import deepcopy

    request = data[KEY]
    return dict(
        software="RuinedFooocus",
        prompt="",
        negative="",
        controlnet={"type": KINDS[0]},
        width=request.source.size[0],
        height=request.source.size[1],
        image_utility_source=deepcopy(data["image_utility_source"]),
        **dict(request.parameters),
        **dict(request.output_settings),
    )


def replay_values(metadata):
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings

    return dict(
        **parameters(metadata),
        **format_settings(metadata),
        color_reference=None,
        input_image=None,
        image_source=None,
        inpaint_toggle=False,
        cn_selection=LABELS[KINDS[0]],
        prompt="",
        negative="",
        automatic_upscale="Off",
        automatic_upscale_target="Generated image",
        model_recipe_path="",
        reference_adapter="none",
    )
