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
