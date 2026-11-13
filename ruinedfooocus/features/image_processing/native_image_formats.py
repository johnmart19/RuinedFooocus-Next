"""Bounded still-image transport around the pinned native PyAV image codecs."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from PIL import Image

FORMAT_FIELDS = ("output_format", "output_bit_depth", "output_color_space", "output_crf")
STILL_SUFFIXES = frozenset({".png", ".exr", ".avif"})
# A full 16-Mpixel RGBA float32 EXR is 256 MiB before codec/header overhead.
# One shared 512-MiB file limit covers native outputs and bounded external imports.
MAX_BYTES = 512 * 1024 * 1024
MAX_PIXELS = 16_777_216
MAX_METADATA = 2 * 1024 * 1024
OPTIONS = {
    "png": (("8-bit", "16-bit"), ("sRGB",)),
    "exr": (("16-bit float", "32-bit float"), ("sRGB", "HDR", "linear", "linear ACEScg")),
    "avif": (("auto", "8-bit YUV420", "10-bit YUV420"), ("sRGB", "HDR", "HDR PQ")),
}


def format_settings(values=None):
    """Validate a complete immutable export choice without importing native compute."""
    values = values or {}
    kind = values.get("output_format", "png")
    if kind not in OPTIONS:
        raise ValueError("Choose PNG, EXR or still AVIF output.")
    depths, spaces = OPTIONS[kind]
    depth = values.get("output_bit_depth") or depths[0]
    space = values.get("output_color_space") or spaces[0]
    crf = values.get("output_crf", 18)
    if depth not in depths or space not in spaces:
        raise ValueError(f"Choose a supported {kind.upper()} bit depth and input color space.")
    if type(crf) is not int or not 1 <= crf <= 63:
        raise ValueError("AVIF quality must be a whole CRF value from 1 through 63.")
    return dict(zip(FORMAT_FIELDS, (kind, depth, space, crf)))


def advanced_output(values):
    choice = format_settings(values)
    return choice["output_format"] != "png" or choice["output_bit_depth"] != "8-bit"


def pipeline_images(decoded, values):
    """Keep decoder precision for explicit exports; retain ordinary PNG byte behavior."""
    if advanced_output(values):
        return [image.detach().cpu().float().numpy().copy() for image in decoded]
    return [np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8) for image in decoded]


def _native():
    # Bootstrap already owns the native source path, dependencies and CPU/device policy.
    from comfy_extras import nodes_images

    return nodes_images


def _pixels(value):
    import torch

    if isinstance(value, Image.Image):
        value = np.asarray(value)
    if isinstance(value, torch.Tensor):
        pixels = value.detach().cpu().float().clone()
    else:
        array = np.asarray(value)
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255
        elif array.dtype == np.uint16:
            array = array.astype(np.float32) / 65535
        elif not np.issubdtype(array.dtype, np.floating):
            raise ValueError("Image exports require uint8, uint16 or floating point pixels.")
        pixels = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    if pixels.ndim == 2:
        pixels = pixels.unsqueeze(-1)
    if pixels.ndim != 3 or pixels.shape[-1] not in (1, 3, 4):
        raise ValueError("Choose a single grayscale, RGB or RGBA still image.")
    if not 0 < pixels.shape[0] * pixels.shape[1] <= MAX_PIXELS:
        raise ValueError("Still images may contain at most 16,777,216 pixels.")
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("Still image pixels must be finite.")
    return pixels


def write_image(path, pixels, values=None, metadata=None):
    """Write a new image atomically using native algorithms; never replace an original."""
    choice = format_settings(values)
    kind, depth, space, crf = (choice[key] for key in FORMAT_FIELDS)
    destination = Path(path)
    if destination.exists():
        raise ValueError("Export destination already exists. Choose a new filename.")
    if destination.suffix.lower() != f".{kind}":
        raise ValueError("Export filename must match the selected image format.")
    tensor = _pixels(pixels)
    if kind == "avif" and tensor.shape[-1] == 4:
        raise ValueError("Native still AVIF does not support alpha. Choose PNG or EXR for RGBA.")
    if kind == "avif" and (tensor.shape[0] < 64 or tensor.shape[1] < 64):
        raise ValueError("Native still AVIF needs dimensions of at least 64 pixels per side.")
    native = _native()
    information = dict(metadata or {})
    information.update(choice)
    information["native_export"] = {
        "node": "SaveImageAdvanced",
        "source_dtype": str(getattr(pixels, "dtype", getattr(pixels, "mode", "unknown"))),
        "channels": tensor.shape[-1],
        "stored_color_space": "scene-linear" if kind == "exr" else space,
        "alpha": tensor.shape[-1] == 4,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".rf-export-", suffix=f".{kind}", dir=destination.parent
    )
    os.close(descriptor)
    try:
        if kind == "avif":
            native._save_avif(
                tensor.unsqueeze(0),
                temporary,
                depth,
                space,
                crf,
                metadata={"parameters": information},
            )
        else:
            encoded = native._encode_image(tensor, kind, depth, space)
            inject = native.inject_png_metadata if kind == "png" else native.inject_exr_metadata
            if kind == "png":
                encoded = inject(encoded, None, {"parameters": information})
            else:
                encoded = inject(encoded, None, {"parameters": information}, space)
            Path(temporary).write_bytes(encoded)
        # Reuse the platform's atomic, no-clobber publication contract.
        if Path(temporary).stat().st_size > MAX_BYTES:
            raise ValueError("Encoded still image exceeds the 512 MiB file limit.")
        from ruinedfooocus.core.preparation import _publish_no_clobber

        _publish_no_clobber(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def _exr_attributes(path):
    """Read the bounded header, never an EXR payload or an executable container."""
    attributes = {}
    with Path(path).open("rb") as source:
        header = source.read(MAX_METADATA + 1)
    if header[:4] != b"\x76\x2f\x31\x01":
        raise ValueError("Not an OpenEXR still image.")
    offset = 8
    while offset < min(len(header), MAX_METADATA) and header[offset]:
        try:
            end = header.index(b"\0", offset)
            name = header[offset:end].decode("utf-8")
            offset = end + 1
            end = header.index(b"\0", offset)
            kind = header[offset:end].decode("ascii")
            length = struct.unpack_from("<i", header, end + 1)[0]
            offset = end + 5
            if length < 0 or offset + length > min(len(header), MAX_METADATA):
                raise ValueError("OpenEXR metadata exceeds the supported header limit.")
            attributes[name] = (kind, header[offset : offset + length])
            offset += length
        except (IndexError, UnicodeDecodeError, struct.error, ValueError) as error:
            raise ValueError("Invalid or oversized OpenEXR metadata header.") from error
    if offset >= min(len(header), MAX_METADATA):
        raise ValueError("Invalid or oversized OpenEXR metadata header.")
    return attributes


def image_metadata(path, *, suffix=None):
    """Read RF metadata embedded by the actual native codec, without model imports."""
    path = Path(path)
    suffix = suffix or path.suffix.lower()
    result = {}
    if suffix == ".exr":
        attrs = _exr_attributes(path)
        value = attrs.get("parameters", (None, b""))[1]
        if value:
            result["parameters"] = value.decode("utf-8")
    else:
        with Image.open(path) as image:
            result.update(
                {
                    key: value
                    for key, value in image.info.items()
                    if key in ("parameters", "comment")
                }
            )
            if suffix == ".avif":
                for value in image.getexif().values():
                    if isinstance(value, str) and value.startswith("parameters:"):
                        result["parameters"] = value[len("parameters:") :]
    if any(
        isinstance(value, (str, bytes)) and len(value) > MAX_METADATA for value in result.values()
    ):
        raise ValueError("Image metadata exceeds the supported 2 MiB limit.")
    return result


@dataclass(frozen=True)
class ImportedImage:
    pixels: np.ndarray
    color_space: str
    bit_depth: str
    metadata: dict


@dataclass(frozen=True)
class ImagePrecision:
    width: int
    height: int
    bits: int
    has_alpha: bool


def _pixel_precision(width, height, pixel_format, max_pixels):
    if not 0 < width * height <= max_pixels:
        raise ValueError(f"Still images may contain at most {max_pixels:,} pixels.")
    components = pixel_format.components if pixel_format is not None else ()
    if not components or any(component.bits <= 0 for component in components):
        raise ValueError("The image precision could not be determined.")
    return ImagePrecision(
        width,
        height,
        max(component.bits for component in components),
        any(component.is_alpha for component in components),
    )


def _first_frame(container, max_pixels=MAX_PIXELS):
    """Use native decoded-frame metadata before any RGB8 conversion."""
    if len(container.streams.video) != 1:
        raise ValueError("Choose one still image.")
    stream = container.streams.video[0]
    if not 0 < stream.width * stream.height <= max_pixels:
        raise ValueError(f"Still images may contain at most {max_pixels:,} pixels.")
    frames = container.decode(stream)
    frame = next(frames, None)
    if frame is None:
        raise ValueError("The still image contains no pixels.")
    precision = _pixel_precision(frame.width, frame.height, frame.format, max_pixels)
    return frame, frames, precision


def encoded_precision(encoded, max_pixels=MAX_PIXELS):
    """Inspect bounded inline pixels with the same CPU reader as native imports.

    No temporary file, RGB conversion, model or hardware decoder is created.
    Native metadata covers the primary image and optional auxiliary alpha stream.
    Missing precision falls back to each component's first frame only. This keeps
    inline first-frame policy separate from strict original-file import admission.
    """
    import av

    if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_BYTES:
        raise ValueError("Choose an encoded image no larger than 512 MiB.")
    with av.open(BytesIO(encoded), mode="r") as container:
        streams = container.streams.video
        if not 1 <= len(streams) <= 2:
            raise ValueError("Choose one inline image with at most one alpha component.")
        precisions = []
        for index, stream in enumerate(streams):
            if not 0 < stream.width * stream.height <= max_pixels:
                raise ValueError(f"Still images may contain at most {max_pixels:,} pixels.")
            pixel_format = stream.codec_context.format
            components = pixel_format.components if pixel_format is not None else ()
            if components and all(component.bits > 0 for component in components):
                precision = _pixel_precision(stream.width, stream.height, pixel_format, max_pixels)
            else:
                # Reopen so decoding one stream cannot consume another's packets.
                with av.open(BytesIO(encoded), mode="r") as decoder:
                    frame = next(decoder.decode(decoder.streams.video[index]), None)
                    if frame is None:
                        raise ValueError("The inline image contains no pixels.")
                    precision = _pixel_precision(
                        frame.width, frame.height, frame.format, max_pixels
                    )
            precisions.append(precision)
        primary = precisions[0]
        return ImagePrecision(
            primary.width,
            primary.height,
            max(value.bits for value in precisions),
            any(value.has_alpha for value in precisions)
            or any(stream.metadata.get("title") == "Alpha" for stream in streams),
        )


def _read_avif_auxiliary_alpha(path, container):
    """Decode one native 8-bit Color/Alpha still through Pillow's AVIF codec."""
    streams = container.streams.video
    by_title = {stream.metadata.get("title"): stream for stream in streams}
    if len(streams) != 2 or set(by_title) != {"Color", "Alpha"}:
        raise ValueError("Choose one still image.")
    size = (by_title["Color"].width, by_title["Color"].height)
    if not 0 < size[0] * size[1] <= MAX_PIXELS or any(
        (stream.width, stream.height) != size for stream in streams
    ):
        raise ValueError("AVIF color and alpha must share one bounded image canvas.")
    counts = {stream.index: 0 for stream in streams}
    transfer = 2
    # Demux both components together: decoding one stream must not consume the
    # other's packets. A second frame fails before any original is published.
    for packet in container.demux(streams):
        for frame in packet.decode():
            counts[packet.stream.index] += 1
            if counts[packet.stream.index] != 1:
                raise ValueError("Animated images are outside the still-image importer.")
            precision = _pixel_precision(frame.width, frame.height, frame.format, MAX_PIXELS)
            if precision.bits != 8:
                raise ValueError(
                    "Auxiliary-alpha AVIF import requires 8-bit color and alpha; "
                    "choose PNG or EXR to preserve higher precision."
                )
            if (precision.width, precision.height) != size:
                raise ValueError("AVIF color and alpha must share one image canvas.")
            if packet.stream.index == by_title["Color"].index:
                transfer = int(frame.color_trc)
            elif len(frame.format.components) != 1:
                raise ValueError("The AVIF auxiliary component is not grayscale alpha.")
    if any(count != 1 for count in counts.values()):
        raise ValueError("Choose one complete color-and-alpha still image.")
    with Image.open(path) as image:
        if (
            image.format != "AVIF"
            or getattr(image, "n_frames", 1) != 1
            or image.mode != "RGBA"
            or image.size != size
        ):
            raise ValueError("Choose one RGBA AVIF still image.")
        pixels = np.asarray(image).copy()
    if pixels.dtype != np.uint8 or pixels.shape != (size[1], size[0], 4):
        raise ValueError("The AVIF decoder did not preserve 8-bit RGBA pixels.")
    return ImportedImage(
        pixels,
        {18: "HDR", 16: "HDR PQ"}.get(transfer, "sRGB"),
        "8-bit",
        image_metadata(path, suffix=".avif"),
    )


def read_image(path, *, suffix=None):
    """Decode one bounded native still while preserving integer or float precision."""
    import av

    path = Path(path)
    suffix = suffix or path.suffix.lower()
    if suffix not in STILL_SUFFIXES or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose a PNG, EXR or still AVIF no larger than 512 MiB.")
    with path.open("rb") as source:
        signature = source.read(128)
    valid = {
        ".png": signature.startswith(b"\x89PNG\r\n\x1a\n"),
        ".exr": signature.startswith(b"\x76\x2f\x31\x01"),
        ".avif": signature[4:8] == b"ftyp" and b"avif" in signature[8:],
    }
    if not valid[suffix]:
        raise ValueError("Image contents do not match the selected still-image format.")
    if suffix == ".exr":
        _exr_attributes(path)
        if struct.unpack_from("<I", signature, 4)[0] & 0x1800:
            raise ValueError("Deep and multipart EXR files require a separate image workflow.")
    with av.open(str(path), mode="r") as container:
        if suffix == ".avif" and len(container.streams.video) == 2:
            return _read_avif_auxiliary_alpha(path, container)
        frame, frames, precision = _first_frame(container)
        alpha = precision.has_alpha
        grayscale = len(frame.format.components) == 1
        bits = precision.bits
        transfer = int(frame.color_trc)
        if suffix == ".exr":
            target = "grayf32le" if grayscale else ("gbrapf32le" if alpha else "gbrpf32le")
        elif bits > 8:
            target = "gray16le" if grayscale else ("rgba64le" if alpha else "rgb48le")
        else:
            target = "gray" if grayscale else ("rgba" if alpha else "rgb24")
        if suffix == ".exr" and frame.format.name.endswith("f16le"):
            # PyAV does not expose half-float ndarray conversion. swscale's f16→f32
            # conversion can quantize through integers; read the native half planes.
            order = [0] if grayscale else ([2, 0, 1, 3] if alpha else [2, 0, 1])
            planes = [
                np.frombuffer(frame.planes[index], dtype="<f2")
                .reshape(frame.height, frame.planes[index].line_size // 2)[:, : frame.width]
                .astype(np.float32)
                for index in order
            ]
            pixels = planes[0] if grayscale else np.stack(planes, axis=-1)
        else:
            pixels = frame.to_ndarray(format=target).copy()
        if next(frames, None) is not None:
            raise ValueError("Animated images are outside the still-image importer.")
    color = "sRGB"
    if suffix == ".avif":
        color = {18: "HDR", 16: "HDR PQ"}.get(transfer, "sRGB")
    metadata = image_metadata(path, suffix=suffix)
    if suffix == ".exr":
        attrs = _exr_attributes(path)
        chromaticities = attrs.get("chromaticities", (None, b""))[1]
        color = "linear"
        if len(chromaticities) == 32:
            red_x, red_y = struct.unpack("<8f", chromaticities)[:2]
            if abs(red_x - 0.713) < 0.001 and abs(red_y - 0.293) < 0.001:
                color = "linear ACEScg"
            elif abs(red_x - 0.708) < 0.001:
                color = "linear Rec.2020"
    return ImportedImage(
        pixels, color, f"{bits}-bit" + (" float" if suffix == ".exr" else ""), metadata
    )


def needs_preview(path):
    path = Path(path)
    if path.suffix.lower() in (".exr", ".avif"):
        return True
    if path.suffix.lower() == ".png":
        with path.open("rb") as source:
            header = source.read(25)
        return header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) == 25 and header[24] == 16
    return False


def display_image(path, cache):
    """A derived 8-bit sRGB browser copy; actions continue to use the original path."""
    path = Path(path)
    if not needs_preview(path):
        return path
    stat = path.stat()
    identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
    target = (
        Path(cache)
        / "native-image-previews"
        / f"{hashlib.sha256(identity.encode()).hexdigest()}.png"
    )
    if target.is_file():
        return target
    imported = read_image(path)
    pixels = _pixels(imported.pixels)
    if imported.color_space == "linear":
        from comfy_execution.preview_tonemap import linear_to_preview

        # The native preview performs display tone mapping; stored pixels stay untouched.
        preview = linear_to_preview(pixels)
    else:
        pixels = convert_color(pixels, imported.color_space, "sRGB")
        array = (pixels.clamp(0, 1).numpy() * 255).astype(np.uint8)
        preview = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".preview-", suffix=".png", dir=target.parent)
    os.close(descriptor)
    try:
        from PIL.PngImagePlugin import PngInfo

        metadata = PngInfo()
        if imported.metadata.get("parameters"):
            metadata.add_text("parameters", imported.metadata["parameters"])
        preview.save(temporary, format="PNG", pnginfo=metadata)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def convert_color(pixels, source, destination):
    """Reuse the native color-space node for an explicit display/export conversion."""
    import torch

    tensor = _pixels(pixels)
    if source == destination:
        return tensor
    native = _native()
    if tensor.shape[-1] == 1:
        tensor = tensor.expand(-1, -1, 3)
    if source == "linear Rec.2020":
        rgb = native._convert_rgb_primaries(tensor[..., :3], native._REC2020_TO_REC709)
        tensor = torch.cat((rgb, tensor[..., 3:]), dim=-1) if tensor.shape[-1] == 4 else rgb
        source = "linear"
    return native.ImageColorSpace.execute(tensor, source, destination)[0].cpu()


def read_export_source(path):
    """Ordinary Library files may also be converted; their actual 8-bit depth is retained."""
    path = Path(path)
    if path.suffix.lower() in STILL_SUFFIXES:
        return read_image(path)
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose a still image no larger than 512 MiB.")
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1 or not 0 < image.width * image.height <= MAX_PIXELS:
            raise ValueError("Choose one still image of at most 16,777,216 pixels.")
        return ImportedImage(
            np.asarray(image.convert("RGBA" if "A" in image.getbands() else "RGB")).copy(),
            "sRGB",
            "8-bit",
            dict(image.info),
        )


def import_image(path, folder):
    """Copy an admitted upload byte for byte; original precision and source stay intact."""
    path = Path(path)
    if path.suffix.lower() not in STILL_SUFFIXES or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose a PNG, EXR or still AVIF no larger than 512 MiB.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".rf-import-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(descriptor, "wb") as destination, path.open("rb") as source:
            copied = 0
            while chunk := source.read(1024 * 1024):
                copied += len(chunk)
                if copied > MAX_BYTES:
                    raise ValueError("Still image exceeds the 512 MiB import limit.")
                destination.write(chunk)
        imported = read_image(temporary, suffix=path.suffix.lower())
        target = (
            folder
            / f"imported-{Path(temporary).stem.removeprefix('.rf-import-')}{path.suffix.lower()}"
        )
        from ruinedfooocus.core.preparation import _publish_no_clobber

        _publish_no_clobber(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target, imported
