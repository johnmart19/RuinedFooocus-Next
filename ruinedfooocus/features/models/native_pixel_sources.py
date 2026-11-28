"""Keep the official L2P U-Net decoder distinct from native Zeta pixel models."""

from functools import lru_cache
from pathlib import Path

from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.features.models.model_identity import file_signature

L2P_ERROR = (
    "This L2P checkpoint uses a convolutional pixel decoder that the native image engine "
    "does not support. Choose a compatible Zeta pixel-space checkpoint."
)


def source_error(header):
    """Recognize the reviewed decoder topology, never a filename or model label."""

    def shape(key):
        value = header.get(key)
        return value.get("shape", ()) if isinstance(value, dict) else ()

    for prefix in ("", "model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        first = shape(prefix + "local_decoder.enc1.0.weight")
        middle = shape(prefix + "local_decoder.bottleneck.0.weight")
        final = shape(prefix + "local_decoder.out_conv.weight")
        if (
            isinstance(first, (list, tuple))
            and len(first) == 4
            and list(first[1:]) == [3, 3, 3]
            and isinstance(middle, (list, tuple))
            and len(middle) == 4
            and list(middle[2:]) == [1, 1]
            and isinstance(final, (list, tuple))
            and len(final) == 4
            and final[0] == 3
            and list(final[2:]) == [1, 1]
        ):
            return L2P_ERROR
    return None


@lru_cache(maxsize=128)
def _file_error(path, signature):
    if Path(path).suffix.lower() == ".gguf":
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        header, _ = read_gguf_header(path)
    else:
        from ruinedfooocus.features.models.model_identity import cached_safetensors_header

        header = cached_safetensors_header(path)
    return source_error(header)


def require_native_pixel_source(path):
    """Refuse known incompatible L2P sources before either native file loader.

    Other containers and malformed files retain their existing native reader's
    validation; this is a named topology guard, not a replacement file reader.
    """
    path = Path(path)
    if not (is_safetensors(path) or path.suffix.lower() == ".gguf"):
        return
    try:
        before = tuple(file_signature(path))
        error = _file_error(str(path), before)
    except (OSError, ValueError):
        return
    if tuple(file_signature(path)) != before:
        raise ValueError("The image checkpoint changed during source inspection. Submit it again.")
    if error is not None:
        raise ValueError(error)
