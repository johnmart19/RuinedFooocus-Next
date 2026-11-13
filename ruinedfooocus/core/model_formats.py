"""Native model file suffixes, without runtime imports."""

from pathlib import Path

SAFETENSORS_SUFFIXES = (".safetensors", ".sft")
# These names use the pinned native torch.load(weights_only=True) reader.
# The suffix identifies a candidate, not arbitrary pickle/export compatibility.
NATIVE_TORCH_SUFFIXES = (".pth", ".ckpt", ".bin", ".pt", ".pt2", ".pkl")
NATIVE_STATE_DICT_SUFFIXES = (*NATIVE_TORCH_SUFFIXES, *SAFETENSORS_SUFFIXES)


def is_safetensors(path):
    return Path(path).suffix.lower() in SAFETENSORS_SUFFIXES


def image_vae_format_error(path):
    if Path(path).suffix.lower() == ".gguf":
        return (
            "GGUF files are unsupported for still-image VAEs. "
            "Choose a compatible native state-dictionary VAE, such as a safetensors file."
        )
    return None


def require_image_vae_format(path):
    if reason := image_vae_format_error(path):
        raise ValueError(reason)
