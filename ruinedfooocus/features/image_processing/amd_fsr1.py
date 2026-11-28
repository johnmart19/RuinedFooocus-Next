"""AMD FidelityFX Super Resolution 1 for still images: EASU upscaling and RCAS sharpening.

A tensor port of the float paths of `ffx_fsr1.h` from the AMD FSR SDK 2.4.0
(`Kits/FidelityFX/upscalers/fsr3/include/gpu/fsr1/ffx_fsr1.h`), which AMD
publishes under the MIT licence reproduced below. The shader's approximate
reciprocals are exact divisions here, and divisions the shader leaves to GPU
behaviour at zero are defined explicitly. FSR 1 is the SDK's only spatial,
single-image technique; its temporal upscalers, frame generation and denoisers
need motion vectors, depth and a swap chain and do not apply to a still image.

Copyright (C) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

SOURCE = "AMD FSR 1 (EASU + RCAS), tensor port of ffx_fsr1.h from AMD FSR SDK 2.4.0"
# EASU is designed for up to 2x per axis; larger targets take several passes.
MAX_PASS_SCALE = 2.0
# RCAS sharpness is in stops of reduction: 0 is maximum sharpening.
DEFAULT_SHARPNESS = 0.2
RCAS_LIMIT = 0.25 - 1.0 / 16.0
ROWS_PER_CHUNK = 256

# Tap offsets from the 'f' texel, in the header's order: b c i j f e k l h g o n.
_TAPS = (
    (0, -1),
    (1, -1),
    (-1, 1),
    (0, 1),
    (0, 0),
    (-1, 0),
    (1, 1),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 2),
    (0, 2),
)
_B, _C, _I, _J, _F, _E, _K, _L, _H, _G, _O, _N = range(12)


def _luma(color):
    # "Luma times 2" from the header: 0.5 R + G + 0.5 B.
    return color[..., 2] * 0.5 + (color[..., 0] * 0.5 + color[..., 1])


def _set(torch, direction, length, weight, a, b, c, d, e):
    """Accumulate edge direction and length for one bilinear corner (fsrEasuSetFloat)."""
    for axis, (low, high) in enumerate(((b, d), (a, e))):
        span = torch.maximum((high - c).abs(), (c - low).abs())
        change = high - low
        ratio = torch.where(span > 0, change.abs() / span.clamp_min(1e-30), torch.zeros_like(span))
        direction[axis] += change * weight
        length += ratio.clamp(0, 1) ** 2 * weight
    return length


def _easu_rows(torch, source, out_width, out_height, row_start, row_stop):
    height, width = source.shape[1:3]
    dtype, device = source.dtype, source.device
    rows = torch.arange(row_start, row_stop, device=device, dtype=dtype)
    columns = torch.arange(out_width, device=device, dtype=dtype)
    # con0: input/output scale, and the half-texel shift between the two grids.
    position_y = (rows + 0.5) * (height / out_height) - 0.5
    position_x = (columns + 0.5) * (width / out_width) - 0.5
    base_y, base_x = position_y.floor(), position_x.floor()
    fraction_y = (position_y - base_y)[:, None]
    fraction_x = (position_x - base_x)[None, :]
    index_y, index_x = base_y.long(), base_x.long()

    taps = []
    for offset_x, offset_y in _TAPS:
        tap_y = (index_y + offset_y).clamp(0, height - 1)
        tap_x = (index_x + offset_x).clamp(0, width - 1)
        taps.append(source[:, tap_y][:, :, tap_x])
    luma = [_luma(tap) for tap in taps]

    direction = [torch.zeros_like(luma[0]), torch.zeros_like(luma[0])]
    length = torch.zeros_like(luma[0])
    corners = (
        ((1 - fraction_x) * (1 - fraction_y), (_B, _E, _F, _G, _J)),
        (fraction_x * (1 - fraction_y), (_C, _F, _G, _H, _K)),
        ((1 - fraction_x) * fraction_y, (_F, _I, _J, _K, _N)),
        (fraction_x * fraction_y, (_G, _J, _K, _L, _O)),
    )
    for weight, names in corners:
        length = _set(torch, direction, length, weight, *(luma[name] for name in names))

    # Normalize the direction, with the header's clean-up close to zero.
    magnitude = direction[0] ** 2 + direction[1] ** 2
    flat = magnitude < 1.0 / 32768.0
    scale = torch.where(flat, torch.ones_like(magnitude), magnitude.clamp_min(1e-30).rsqrt())
    direction_x = torch.where(flat, torch.ones_like(magnitude), direction[0]) * scale
    direction_y = direction[1] * scale

    # {0 to 2} to {0 to 1}, shaped with a square.
    length = (length * 0.5) ** 2
    stretch = (direction_x**2 + direction_y**2) / torch.maximum(
        direction_x.abs(), direction_y.abs()
    ).clamp_min(1e-30)
    length_x = 1 + (stretch - 1) * length
    length_y = 1 - 0.5 * length
    lobe = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * length
    clip = 1 / lobe

    color = torch.zeros_like(taps[0])
    total = torch.zeros_like(luma[0])
    for tap, (offset_x, offset_y) in zip(taps, _TAPS):
        delta_x, delta_y = offset_x - fraction_x, offset_y - fraction_y
        rotated_x = (delta_x * direction_x + delta_y * direction_y) * length_x
        rotated_y = (delta_x * -direction_y + delta_y * direction_x) * length_y
        distance = torch.minimum(rotated_x**2 + rotated_y**2, clip)
        # Lanczos-2 approximation: a base with a movable negative lobe, times a window.
        base = (25.0 / 16.0) * (2.0 / 5.0 * distance - 1) ** 2 - (25.0 / 16.0 - 1.0)
        weight = base * (lobe * distance - 1) ** 2
        color += tap * weight[..., None]
        total += weight
    # Normalize, then de-ring against the four nearest texels.
    nearest = torch.stack([taps[name] for name in (_F, _G, _J, _K)])
    return torch.minimum(
        nearest.amax(dim=0), torch.maximum(nearest.amin(dim=0), color / total[..., None])
    )


def easu(pixels, width, height, check_cancelled=None):
    """Edge-adaptive spatial upsampling of BHWC RGB in [0, 1] to (width, height)."""
    import torch

    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError("AMD FSR 1 upscaling needs RGB pixels.")
    chunks = []
    for start in range(0, height, ROWS_PER_CHUNK):
        if check_cancelled is not None:
            check_cancelled()
        chunks.append(
            _easu_rows(torch, pixels, width, height, start, min(height, start + ROWS_PER_CHUNK))
        )
    return torch.cat(chunks, dim=1)


def rcas(pixels, sharpness=DEFAULT_SHARPNESS):
    """Robust contrast-adaptive sharpening of BHWC RGB in [0, 1] (FsrRcasF)."""
    import torch
    import torch.nn.functional as functional

    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError("AMD FSR 1 sharpening needs RGB pixels.")
    if not 0 <= sharpness <= 2:
        raise ValueError("RCAS sharpness is 0 (maximum) to 2 stops of reduction.")
    padded = functional.pad(pixels.movedim(-1, 1), (1, 1, 1, 1), mode="replicate").movedim(1, -1)
    b, d = padded[:, :-2, 1:-1], padded[:, 1:-1, :-2]
    e = padded[:, 1:-1, 1:-1]
    f, h = padded[:, 1:-1, 2:], padded[:, 2:, 1:-1]
    luma_b, luma_d, luma_e, luma_f, luma_h = (_luma(value) for value in (b, d, e, f, h))
    ring = torch.stack((b, d, f, h))
    minimum, maximum = ring.amin(dim=0), ring.amax(dim=0)
    ring_luma = torch.stack((luma_b, luma_d, luma_f, luma_h)).amin(dim=0)
    zero, one = torch.zeros_like(luma_e), torch.ones_like(luma_e)
    lower = torch.where(ring_luma > 0, luma_e / ring_luma.clamp_min(1e-30), one).clamp(0, 1)
    hit_minimum = (
        torch.where(
            maximum > 0, minimum / (4 * maximum).clamp_min(1e-30), torch.zeros_like(minimum)
        )
        * lower[..., None]
    )
    denominator = 4 * minimum - 4
    hit_maximum = torch.where(
        denominator < 0, (1 - maximum) / denominator.clamp_max(-1e-30), torch.zeros_like(minimum)
    )
    lobe = torch.maximum(-hit_minimum, hit_maximum).amax(dim=-1)
    lobe = torch.maximum(torch.full_like(lobe, -RCAS_LIMIT), torch.minimum(lobe, zero))
    lobe = (lobe * 2.0**-sharpness)[..., None]
    return ((lobe * (b + d + h + f) + e) / (4 * lobe + 1)).clamp(0, 1)


def pass_sizes(source, target):
    """Intermediate (width, height) steps of at most 2x per axis, ending at target."""
    import math

    (width, height), (target_width, target_height) = source, target
    if target_width < width or target_height < height:
        raise ValueError(
            "AMD FSR 1 only enlarges. Choose a target at least as large as the source in "
            "both dimensions, or another interpolation."
        )
    passes = max(
        1,
        math.ceil(math.log(max(target_width / width, target_height / height), MAX_PASS_SCALE)),
    )
    return [
        (
            round(width * (target_width / width) ** (step / passes)),
            round(height * (target_height / height) ** (step / passes)),
        )
        for step in range(1, passes)
    ] + [(target_width, target_height)]


def upscale(pixels, width, height, *, sharpness=DEFAULT_SHARPNESS, check_cancelled=None):
    """FSR 1 for BHWC in [0, 1]: EASU passes then RCAS on colour; alpha is bicubic."""
    import torch
    import torch.nn.functional as functional

    channels = pixels.shape[-1]
    if channels not in (1, 3, 4):
        raise ValueError("AMD FSR 1 accepts grayscale, RGB or RGBA pixels.")
    color = (pixels[..., :1].expand(-1, -1, -1, 3) if channels == 1 else pixels[..., :3]).clamp(
        0, 1
    )
    for step_width, step_height in pass_sizes((pixels.shape[2], pixels.shape[1]), (width, height)):
        color = easu(color, step_width, step_height, check_cancelled)
    if check_cancelled is not None:
        check_cancelled()
    color = rcas(color, sharpness)
    if channels == 1:
        return color[..., :1]
    if channels == 3:
        return color
    alpha = functional.interpolate(
        pixels[..., 3:].movedim(-1, 1), size=(height, width), mode="bicubic", align_corners=False
    ).movedim(1, -1)
    return torch.cat((color, alpha.clamp(0, 1)), dim=-1)
