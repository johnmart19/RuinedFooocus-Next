"""AMD FSR 2, 3 and 4 upscaling for still images, through AMD's own runtime.

These are temporal upscalers; FSR 4 is the ML one. Each expects a moving camera,
motion vectors and a depth buffer. A still has none of these, so the small native helper in
`tools/native/amd_fsr` presents the image as a static camera sequence: each frame
is the source resampled at that frame's sub-pixel jitter, with zero motion and a
flat depth buffer, and FSR's own sharpening pass enabled. The result is a clean,
well anti-aliased enlargement; it does not invent detail the way a generative
upscaler does.

The helper drives AMD's signed `amd_fidelityfx_*_dx12.dll` on the GPU with
DirectX 12, outside Torch. It needs Windows, a GPU and driver for which the AMD
FSR API offers the chosen provider, and the runtime files in `runtime_directory()`.
Nothing here downloads them.
"""

import os
from pathlib import Path
import struct
import subprocess
from uuid import uuid4

HELPER = "fsr_still.exe"
RUNTIME_FILES = (HELPER, "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll")
# Version prefixes the AMD FSR API reports for its upscaler providers.
PROVIDERS = {"fsr4": "4.", "fsr3": "3.", "fsr2": "2."}
LABELS = {"fsr4": "AMD FSR 4", "fsr3": "AMD FSR 3", "fsr2": "AMD FSR 2"}


def source(version):
    return LABELS[version] + " upscaler (AMD FSR API, DirectX 12) on a static jittered sequence"


# FSR's presets go up to 3x; two keeps quality, and larger targets take several passes.
MAX_PASS_SCALE = 2.0
DEFAULT_SHARPNESS = 1.0
# Sharpening limit for every pass after the first. Sharpening an already sharpened
# FSR 2 or 3 result at full strength turns their jitter residue into a checkerboard
# along edges (seen at 4x on 2026-11-19); FSR 4 stays clean at full strength.
LATER_PASS_SHARPNESS = {"fsr4": 1.0, "fsr3": 0.8, "fsr2": 0.65}
TIMEOUT_SECONDS = 120
MAGIC = b"RFI1"


def runtime_directory():
    """Where the helper and AMD's DLLs live: RF_AMD_FSR_DIR, else the data root."""
    configured = os.environ.get("RF_AMD_FSR_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    from ruinedfooocus.core.paths import data_path

    return data_path("runtime", "amd-fsr")


def unavailable_reason(directory=None):
    """None when the AMD runtime can be attempted here; otherwise the reason, in plain words."""
    if os.name != "nt":
        return "AMD FSR 2, 3 and 4 need Windows and DirectX 12."
    directory = Path(directory) if directory is not None else runtime_directory()
    missing = [name for name in RUNTIME_FILES if not (directory / name).is_file()]
    if missing:
        return (
            "AMD FSR 2, 3 and 4 need their runtime in "
            + str(directory)
            + ". Build it with tools/native/amd_fsr/build.py and the AMD FSR SDK. Missing: "
            + ", ".join(missing)
            + "."
        )
    return None


def pass_sizes(source, target):
    """Intermediate (width, height) steps of at most 2x per axis, ending at target."""
    import math

    (width, height), (target_width, target_height) = source, target
    if target_width < width or target_height < height:
        raise ValueError(
            "AMD FSR only enlarges. Choose a target at least as large as the source in "
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


def write_raw(path, rgba, width, height):
    if len(rgba) != width * height * 4:
        raise ValueError("The FSR input does not match its dimensions.")
    with open(path, "wb") as stream:
        stream.write(MAGIC + struct.pack("<II", width, height))
        stream.write(rgba)


def read_raw(path, width, height):
    with open(path, "rb") as stream:
        header = stream.read(12)
        if header[:4] != MAGIC or struct.unpack("<II", header[4:]) != (width, height):
            raise ValueError("The AMD FSR helper returned an unexpected image.")
        pixels = stream.read(width * height * 4 + 1)
    if len(pixels) != width * height * 4:
        raise ValueError("The AMD FSR helper returned an incomplete image.")
    return pixels


def _failure(error, label):
    detail = " ".join((getattr(error, "stderr", b"") or b"").decode("utf-8", "replace").split())
    if "requested upscaler version is not available" in detail:
        return ValueError(
            label + " is not offered by this GPU or driver. Update the AMD driver, or choose "
            "another FSR version."
        )
    return ValueError(label + " could not run on this GPU. " + detail[-240:])


def run_pass(rgba, size, target, *, version, sharpness, directory, work, check_cancelled=None):
    """One helper run: RGBA8 bytes at size -> RGBA8 bytes at target."""
    from ruinedfooocus.core.bounded_subprocess import BoundedProcessError, run_bounded

    name = uuid4().hex
    source_path, result_path = work / (name + ".in.raw"), work / (name + ".out.raw")
    try:
        write_raw(source_path, rgba, *size)
        try:
            run_bounded(
                [
                    str(directory / HELPER),
                    str(source_path),
                    str(result_path),
                    str(target[0]),
                    str(target[1]),
                    "0",
                    "1",
                    PROVIDERS[version],
                    repr(float(sharpness)),
                ],
                timeout=TIMEOUT_SECONDS,
                check_cancelled=check_cancelled,
            )
        except BoundedProcessError as error:
            raise _failure(error, LABELS[version]) from None
        except subprocess.TimeoutExpired:
            raise ValueError(LABELS[version] + " did not finish in time on this GPU.") from None
        return read_raw(result_path, *target)
    finally:
        source_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def upscale(
    pixels,
    width,
    height,
    *,
    version="fsr4",
    sharpness=DEFAULT_SHARPNESS,
    check_cancelled=None,
    work=None,
):
    """FSR 2, 3 or 4 for BHWC in [0, 1]: colour through the helper in 8-bit; alpha is bicubic."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    if version not in PROVIDERS:
        raise ValueError("Choose AMD FSR 2, 3 or 4 for the AMD runtime.")
    reason = unavailable_reason()
    if reason is not None:
        raise ValueError(reason)
    if not 0 <= sharpness <= 1:
        raise ValueError("FSR sharpness is 0 to 1.")
    channels = pixels.shape[-1]
    if pixels.ndim != 4 or pixels.shape[0] != 1 or channels not in (1, 3, 4):
        raise ValueError("AMD FSR accepts one grayscale, RGB or RGBA image.")
    directory = runtime_directory()
    if work is None:
        from ruinedfooocus.core.cache_contract import configured_cache_root

        work = configured_cache_root() / "amd-fsr"
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    color = (pixels[..., :1].expand(-1, -1, -1, 3) if channels == 1 else pixels[..., :3]).clamp(
        0, 1
    )
    size = (pixels.shape[2], pixels.shape[1])
    array = (color[0].detach().float().cpu().numpy() * 255).round().astype(np.uint8)
    rgba = np.concatenate((array, np.full((*array.shape[:2], 1), 255, np.uint8)), axis=-1)
    data = rgba.tobytes()
    first = True
    for step in pass_sizes(size, (width, height)):
        if check_cancelled is not None:
            check_cancelled()
        if step != size:
            data = run_pass(
                data,
                size,
                step,
                version=version,
                sharpness=sharpness if first else min(sharpness, LATER_PASS_SHARPNESS[version]),
                directory=directory,
                work=work,
                check_cancelled=check_cancelled,
            )
            size = step
            first = False
    result = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)[..., :3]
    result = torch.from_numpy(result.astype(np.float32) / 255).unsqueeze(0)
    result = result.to(device=pixels.device, dtype=pixels.dtype)
    if channels == 1:
        return result[..., :1]
    if channels == 3:
        return result
    alpha = functional.interpolate(
        pixels[..., 3:].movedim(-1, 1), size=(height, width), mode="bicubic", align_corners=False
    ).movedim(1, -1)
    return torch.cat((result, alpha.clamp(0, 1)), dim=-1)
