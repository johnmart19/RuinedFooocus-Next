"""CPU equivalents for RF's finite recipes; no general GLSL interpreter."""


def sample_lut(values, lut):
    import numpy as np
    import torch
    from torch.nn.functional import grid_sample

    # GLSL recipe coordinates are (value * 255 + .5) / 256. With
    # align_corners=True this samples the same 256 texel centers and interpolates
    # between them, including exact endpoints. Native CurveInput owns the LUT.
    texture = torch.from_numpy(np.asarray(lut, dtype=np.float32).copy()).reshape(1, 1, 1, 256)
    horizontal = values.clamp(0, 1) * 2 - 1
    grid = torch.stack((horizontal, torch.zeros_like(horizontal)), dim=-1)
    return grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=True)[
        :, 0
    ]


def _luminance(pixels):
    return (pixels * pixels.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1, keepdim=True)


def tonal_range(pixels):
    """Black and white points for Auto contrast: the 0.5% and 99.5% luminance levels.

    A few stray dark or bright pixels do not hold the stretch back, and a flat
    image keeps its tone because the two points coincide.
    """
    return _range(_luminance(pixels))


def _range(values):
    count = values.numel()
    ordered = values.flatten().float().sort().values
    low = float(ordered[min(count - 1, int(count * 0.005))])
    high = float(ordered[max(0, min(count - 1, int(count * 0.995)))])
    return (low, high) if high - low > 1e-5 else (0.0, 1.0)


def channel_ranges(pixels):
    """Per-channel black and white points for Auto color, which also removes a colour cast."""
    ranges = [_range(pixels[..., channel]) for channel in range(3)]
    return tuple(low for low, _ in ranges), tuple(high for _, high in ranges)


def median3(torch, pixels):
    """3x3 per-channel median with replicated edges, as the Reduce noise recipe computes it."""
    from torch.nn.functional import pad

    padded = pad(pixels.movedim(-1, 1), (1, 1, 1, 1), mode="replicate")
    windows = (
        padded.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*padded.shape[:2], *pixels.shape[1:3], 9)
    )
    return windows.median(dim=-1).values.movedim(1, -1)


def _saturation(torch, pixels, amount):
    luminance = _luminance(pixels)
    return luminance + (pixels - luminance) * (1 + amount)


def _vibrance(torch, pixels, amount):
    luminance = _luminance(pixels)
    spread = pixels.amax(dim=-1, keepdim=True) - pixels.amin(dim=-1, keepdim=True)
    return luminance + (pixels - luminance) * (1 + amount * (1 - spread))


def _exposure(torch, pixels, amount):
    return pixels * 2 ** (2 * amount)


def _midtones(torch, pixels, amount):
    return pixels.clamp(0, 1) ** (2**-amount)


def _temperature(torch, pixels, amount):
    return pixels * pixels.new_tensor([1 + 0.25 * amount, 1, 1 - 0.25 * amount])


def _tint(torch, pixels, amount):
    return pixels * pixels.new_tensor([1 + 0.125 * amount, 1 - 0.25 * amount, 1 + 0.125 * amount])


def _hue(torch, pixels, amount):
    import math

    angle = math.pi * amount
    axis = pixels.new_full((3,), 0.57735026918963)
    turned = torch.linalg.cross(axis.expand_as(pixels), pixels, dim=-1)
    along = axis * (pixels * axis).sum(dim=-1, keepdim=True)
    return pixels * math.cos(angle) + turned * math.sin(angle) + along * (1 - math.cos(angle))


def _sepia(torch, pixels, amount):
    matrix = pixels.new_tensor(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
    )
    return torch.lerp(pixels, (pixels @ matrix.T).clamp(0, 1), amount)


def _vignette(torch, pixels, amount):
    height, width = pixels.shape[1:3]
    rows = (torch.arange(height, dtype=pixels.dtype) + 0.5) / height - 0.5
    columns = (torch.arange(width, dtype=pixels.dtype) + 0.5) / width - 0.5
    distance = torch.sqrt(rows[:, None] ** 2 + columns[None, :] ** 2) * 1.41421356237310
    edge = ((distance - 0.35) / 0.65).clamp(0, 1)
    edge = edge * edge * (3 - 2 * edge)  # GLSL smoothstep
    return pixels * (1 - amount * edge)[None, :, :, None]


ADJUSTMENTS = {
    "saturation": _saturation,
    "vibrance": _vibrance,
    "exposure": _exposure,
    "midtones": _midtones,
    "temperature": _temperature,
    "tint": _tint,
    "hue": _hue,
    "sepia": _sepia,
    "vignette": _vignette,
}


def render(request, pixels, curves, check_cancelled):
    import torch
    from ruinedfooocus.features.image_processing.depth_utilities import native_call

    options = dict(request.parameters)
    if request.kind == "image-curves":
        luts = [native_call(check_cancelled, curve.to_lut) for curve in curves.values()]
        master = [
            native_call(check_cancelled, sample_lut, pixels[..., channel], luts[0])
            for channel in range(3)
        ]
        channels = [
            native_call(check_cancelled, sample_lut, master[channel], luts[channel + 1])
            for channel in range(3)
        ]
        return torch.stack(channels, dim=-1).clamp(0, 1), [
            "CurveEditor",
            "CurveInput.to_lut",
            "torch.grid_sample",
        ]

    from comfy_extras.nodes_post_processing import Blend

    if request.recipe == "invert":
        from nodes import ImageInvert

        effect = native_call(check_cancelled, ImageInvert().invert, pixels)[0]
        operators = ["ImageInvert", "ImageBlend"]
    elif request.recipe == "bt709-grayscale":
        from kornia.color import rgb_to_grayscale

        weights = pixels.new_tensor([0.2126, 0.7152, 0.0722])
        gray = native_call(check_cancelled, rgb_to_grayscale, pixels.movedim(-1, 1), weights)
        effect = gray.movedim(1, -1).expand_as(pixels)
        operators = ["kornia.rgb_to_grayscale(BT.709)", "ImageBlend"]
    elif request.recipe == "posterize":
        # Native palette quantization and bit-dropping posterizers implement
        # different operations. This is the finite RF recipe's exact arithmetic.
        check_cancelled()
        steps = options["shader_levels"] - 1
        effect = torch.floor(pixels.clamp(0, 1) * steps + 0.5) / steps
        check_cancelled()
        operators = ["torch.floor(RF posterize recipe)", "ImageBlend"]
    elif request.recipe == "auto-contrast":
        check_cancelled()
        low, high = tonal_range(pixels)
        effect = ((pixels - low) / max(high - low, 0.00001)).clamp(0, 1)
        check_cancelled()
        operators = ["torch(RF auto-contrast recipe)", "ImageBlend"]
    elif request.recipe == "auto-color":
        check_cancelled()
        low, high = (pixels.new_tensor(values) for values in channel_ranges(pixels))
        effect = ((pixels - low) / (high - low).clamp_min(0.00001)).clamp(0, 1)
        check_cancelled()
        operators = ["torch(RF auto-color recipe)", "ImageBlend"]
    elif request.recipe == "levels":
        check_cancelled()
        black, white = options["shader_black"], options["shader_white"]
        ranged = ((pixels - black) / max(white - black, 0.00001)).clamp(0, 1)
        effect = ranged ** (1 / options["shader_gamma"])
        check_cancelled()
        operators = ["torch(RF levels recipe)", "ImageBlend"]
    elif request.recipe == "denoise":
        effect = native_call(check_cancelled, median3, torch, pixels)
        operators = ["torch.median(RF 3x3 recipe)", "ImageBlend"]
    elif request.recipe == "rcas-sharpen":
        from ruinedfooocus.features.image_processing import amd_fsr1

        effect = native_call(check_cancelled, amd_fsr1.rcas, pixels.clamp(0, 1), 0.0)
        operators = ["AMD FSR 1 RCAS (tensor port)", "ImageBlend"]
    elif request.recipe in ADJUSTMENTS:
        # These recipes are their own arithmetic; amount is the adjustment, not an opacity.
        check_cancelled()
        result = ADJUSTMENTS[request.recipe](torch, pixels, float(options["shader_amount"]))
        check_cancelled()
        return result.clamp(0, 1), [f"torch(RF {request.recipe} recipe)"]
    else:
        raise ValueError("CPU execution accepts only the reviewed named image effects.")
    result = native_call(
        check_cancelled, Blend.execute, pixels, effect, options["shader_amount"], "normal"
    )[0]
    return result, operators
