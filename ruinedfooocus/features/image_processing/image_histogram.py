"""Bounded still-image analysis through the pinned native histogram node."""

from concurrent.futures import CancelledError
from dataclasses import dataclass
import hashlib

MAX_PIXELS = 16777216
CHANNELS = ("rgb", "luminance", "red", "green", "blue")
NATIVE_NODE = "comfy_extras.nodes_curve.ImageHistogram"
NATIVE_REVISION = "fca60fe70e5b84c5c6b058263653d15cb3401e44"


@dataclass(frozen=True)
class HistogramRequest:
    size: tuple[int, int]
    rgba: bytes
    input_mode: str
    has_alpha: bool
    has_icc_profile: bool
    source_sha256: str


@dataclass(frozen=True)
class HistogramResult:
    request: HistogramRequest
    bins: tuple[tuple[int, ...], ...]

    def as_dict(self):
        request = self.request
        return {
            "object": "image.histogram",
            "native_node": NATIVE_NODE,
            "native_revision": NATIVE_REVISION,
            "source": {
                "width": request.size[0],
                "height": request.size[1],
                "input_mode": request.input_mode,
                "working_mode": "RGB8",
                "has_alpha": request.has_alpha,
                "has_icc_profile": request.has_icc_profile,
                "color_policy": "encoded RGB; no ICC conversion or linearization",
                "alpha_policy": "all pixels included; alpha ignored",
                "sha256": request.source_sha256,
                "sha256_layout": "oriented RGBA8 bytes with width and height",
            },
            "quantization": "clip(value * 255, 0, 255), then truncate to uint8",
            "rgb_composite": "integer floor of the per-bin R/G/B count average",
            "luminance_weights": [0.2126, 0.7152, 0.0722],
            "bins": dict(zip(CHANNELS, (list(values) for values in self.bins))),
        }


def capture(image):
    from PIL import Image, ImageOps

    if (
        not isinstance(image, Image.Image)
        or min(image.size) < 1
        or image.width * image.height > MAX_PIXELS
    ):
        raise ValueError("Add one source image of at most 16 megapixels to analyze.")
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        raise ValueError("Histogram analysis requires an 8-bit RGB, grayscale or palette image.")
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    oriented = ImageOps.exif_transpose(image).convert("RGBA")
    rgba = oriented.tobytes()
    identity = oriented.width.to_bytes(4, "little") + oriented.height.to_bytes(4, "little")
    return HistogramRequest(
        oriented.size,
        rgba,
        image.mode,
        has_alpha,
        bool(image.info.get("icc_profile")),
        hashlib.sha256(identity + rgba).hexdigest(),
    )


def analyze(request: HistogramRequest, cancel=None):
    import numpy as np
    import torch
    from comfy_extras.nodes_curve import ImageHistogram

    def check_cancelled():
        if cancel is not None and cancel.is_set():
            raise CancelledError("Histogram analysis cancelled.")

    check_cancelled()
    if not isinstance(request, HistogramRequest):
        raise ValueError("Capture a source image before histogram analysis.")
    width, height = request.size
    if (
        min(request.size) < 1
        or width * height > MAX_PIXELS
        or len(request.rgba) != width * height * 4
    ):
        raise ValueError("The captured histogram source is invalid.")
    pixels = np.frombuffer(request.rgba, dtype=np.uint8).reshape(height, width, 4)
    tensor = (
        torch.from_numpy(pixels[:, :, :3].copy()).to(dtype=torch.float32).div_(255).unsqueeze(0)
    )
    check_cancelled()
    output = ImageHistogram.execute(tensor)
    check_cancelled()
    values = getattr(output, "result", output)
    if len(values) != len(CHANNELS):
        raise RuntimeError("The native histogram returned an invalid channel count.")
    bins = []
    for channel in values:
        if len(channel) != 256 or any(
            type(value) is not int or not 0 <= value <= width * height for value in channel
        ):
            raise RuntimeError("The native histogram returned invalid bins.")
        bins.append(tuple(channel))
    if any(sum(channel) != width * height for channel in bins[1:]):
        raise RuntimeError("The native histogram returned an invalid pixel count.")
    return HistogramResult(request, tuple(bins))


def operation(request):
    """Freeze before enqueueing; cancellation never publishes a partial result."""

    def run(job, task_id):
        return analyze(request, job.cancelled).as_dict()

    return run


def chart(result):
    """Render only bounded numeric node results into an accessible static SVG."""
    from html import escape
    from ruinedfooocus.core.translation_manager import translate_ui

    series = result["bins"]
    peak = max(1, *(max(series[name]) for name in CHANNELS))
    colors = {
        "rgb": "#aaaaaa",
        "luminance": "#e8ba4a",
        "red": "#ef6666",
        "green": "#57bd86",
        "blue": "#6c9fe8",
    }
    labels = {
        "rgb": translate_ui("RGB"),
        "luminance": translate_ui("Luminance"),
        "red": translate_ui("Red"),
        "green": translate_ui("Green"),
        "blue": translate_ui("Blue"),
    }
    paths = []
    for name in CHANNELS:
        points = " ".join(
            f"{24 + index * 2},{184 - int(value * 160 / peak)}"
            for index, value in enumerate(series[name])
        )
        paths.append(
            f'<polyline fill="none" stroke="{colors[name]}" stroke-width="1.5" points="{points}"/>'
        )
    legend = " · ".join(
        f'<span style="color:{colors[name]}">{escape(labels[name])}</span>' for name in CHANNELS
    )
    description = escape(
        translate_ui("Image histogram; five channels, bins zero to 255"), quote=True
    )
    return (
        f'<div role="img" aria-label="{description}">'
        '<svg viewBox="0 0 558 212" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M24 16V184H534" stroke="currentColor" fill="none"/>'
        + "".join(paths)
        + '<text x="24" y="204" fill="currentColor">0</text><text x="508" y="204" fill="currentColor">255</text>'
        + f'<text x="28" y="14" fill="currentColor">{peak}</text></svg>{legend}</div>'
    )
