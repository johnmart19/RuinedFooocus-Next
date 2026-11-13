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
    else:
        raise ValueError("CPU execution accepts only the reviewed named image effects.")
    result = native_call(
        check_cancelled, Blend.execute, pixels, effect, options["shader_amount"], "normal"
    )[0]
    return result, operators
