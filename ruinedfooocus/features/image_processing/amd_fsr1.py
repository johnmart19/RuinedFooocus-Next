"""RCAS, the contrast-adaptive sharpener from AMD FidelityFX Super Resolution 1.

A tensor port of `FsrRcasF` from `ffx_fsr1.h` in the AMD FSR SDK 2.4.0
(`Kits/FidelityFX/upscalers/fsr3/include/gpu/fsr1/ffx_fsr1.h`), which AMD publishes
under the MIT licence reproduced below. The shader's approximate reciprocals are
exact divisions here, and divisions it leaves to GPU behaviour at zero are defined
explicitly. Upscaling uses AMD FSR 4 through AMD's runtime; see `amd_fsr4.py`.

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

DEFAULT_SHARPNESS = 0.2
RCAS_LIMIT = 0.25 - 1.0 / 16.0


def _luma(color):
    # "Luma times 2" from the header: 0.5 R + G + 0.5 B.
    return color[..., 2] * 0.5 + (color[..., 0] * 0.5 + color[..., 1])


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
