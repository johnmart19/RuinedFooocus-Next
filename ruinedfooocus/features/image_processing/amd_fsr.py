"""AMD FidelityFX Super Resolution for still images: one option, four versions.

FSR 4 (ML), FSR 3 and FSR 2 are temporal upscalers run through AMD's own runtime
on the GPU (`amd_fsr4.py`). FSR 1 is the spatial EASU + RCAS pair, ported to
tensors, and runs on any device including CPU (`amd_fsr1.py`).
"""

INTERPOLATION = "amd-fsr"
VERSIONS = ("fsr4", "fsr3", "fsr2", "fsr1")
DEFAULT_VERSION = "fsr4"


def check(version, *, dtype, color_space, source_size, target_size, cpu):
    """Refuse an unsuitable request before queueing, with the reason."""
    from ruinedfooocus.features.image_processing import amd_fsr1, amd_fsr4

    if version not in VERSIONS:
        raise ValueError("Choose FSR 4, 3, 2 or 1.")
    if color_space != "sRGB":
        raise ValueError("AMD FSR works on sRGB sources.")
    if version == "fsr1":
        if dtype not in ("uint8", "uint16"):
            raise ValueError("AMD FSR 1 works on 8-bit or 16-bit sources.")
        amd_fsr1.pass_sizes(source_size, target_size)
        return
    label = amd_fsr4.LABELS[version]
    if cpu:
        raise ValueError(
            label + " runs on the GPU and is unavailable in CPU mode. FSR 1 works on any device."
        )
    if dtype != "uint8":
        raise ValueError(label + " works on 8-bit sources. FSR 1 also accepts 16-bit.")
    reason = amd_fsr4.unavailable_reason()
    if reason is not None:
        raise ValueError(reason + " FSR 1 needs no runtime.")
    amd_fsr4.pass_sizes(source_size, target_size)


def upscale(version, pixels, width, height, *, check_cancelled=None):
    from ruinedfooocus.features.image_processing import amd_fsr1, amd_fsr4

    if version == "fsr1":
        return amd_fsr1.upscale(pixels, width, height, check_cancelled=check_cancelled)
    return amd_fsr4.upscale(pixels, width, height, version=version, check_cancelled=check_cancelled)


def source(version):
    from ruinedfooocus.features.image_processing import amd_fsr1, amd_fsr4

    return amd_fsr1.SOURCE if version == "fsr1" else amd_fsr4.source(version)
