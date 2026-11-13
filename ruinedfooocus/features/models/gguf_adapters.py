"""Admit native GGUF LoRAs using bounded, detached factor descriptors."""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

from ruinedfooocus.core.model_formats import NATIVE_STATE_DICT_SUFFIXES
from ruinedfooocus.features.models.gguf_metadata import read_gguf_header
from ruinedfooocus.features.models.model_identity import file_signature


@dataclass(frozen=True)
class GgufLora:
    signature: tuple
    architecture: str | None
    target_count: int
    factor_types: tuple[int, ...]


def inspect_gguf_lora(path):
    """Match the pinned native adapter contract without mapping or reading weights.

    Admission establishes a LoRA container, not compatibility with a selected image
    model. The native application path still validates every resolved target.
    """
    path = Path(path)
    before = tuple(file_signature(path))
    result = _inspect(str(path), before)
    if tuple(file_signature(path)) != before:
        raise OSError("GGUF LoRA changed during inspection. Refresh and select again.")
    return result


@lru_cache(maxsize=256)
def _inspect(path, signature):
    from gguf import GGMLQuantizationType as Q

    header, architecture = read_gguf_header(path, include_storage=True)
    metadata = header.get("__metadata__", {})
    if metadata.get("general.type") != "adapter":
        raise ValueError("GGUF file is not an adapter (general.type must be 'adapter').")
    if metadata.get("adapter.type") != "lora":
        raise ValueError("GGUF adapter is not a LoRA (adapter.type must be 'lora').")
    alpha = metadata.get("adapter.lora.alpha")
    if alpha is not None and not math.isfinite(alpha):
        raise ValueError("GGUF LoRA alpha must be finite.")
    supported = {int(kind) for kind in (Q.F32, Q.F16, Q.BF16, Q.Q8_0)}
    pairs, targets, types = {}, {}, set()
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        suffix = next((part for part in (".lora_a", ".lora_b") if name.endswith(part)), None)
        if suffix is None:
            raise ValueError("GGUF LoRA accepts only paired .lora_a and .lora_b factor tensors.")
        base = name[: -len(suffix)]
        target = base.removesuffix(".weight")
        if target in targets and targets[target] != base:
            raise ValueError("GGUF LoRA factors name duplicate native targets.")
        targets[target] = base
        pair = pairs.setdefault(base, {})
        factor = suffix[-1]
        shape, kind = entry["storage_shape"], entry["gguf_type"]
        if not target or factor in pair:
            raise ValueError("GGUF LoRA factors name duplicate native targets.")
        if kind not in supported:
            raise ValueError("GGUF LoRA factors support only F32, F16, BF16, and Q8_0.")
        if len(shape) != 2 or shape != entry["shape"]:
            raise ValueError("GGUF LoRA factors must use their stored 2-D shape.")
        pair[factor] = tuple(shape)
        types.add(kind)
    if not pairs:
        raise ValueError("GGUF LoRA contains no factor tensors.")
    for pair in pairs.values():
        if set(pair) != {"a", "b"} or pair["a"][0] != pair["b"][1]:
            raise ValueError("GGUF LoRA needs paired factors with matching rank dimensions.")
    return GgufLora(signature, architecture, len(pairs), tuple(sorted(types)))


def discover_image_loras(roots):
    """Keep native file candidates and admit typed GGUF adapters in root priority."""
    names, seen = [], set()
    for folder in roots:
        folder = Path(folder)
        resolved = folder.resolve()
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            name = str(path.relative_to(folder))
            if name in seen:
                continue
            seen.add(name)
            if path.suffix.lower() in NATIVE_STATE_DICT_SUFFIXES:
                names.append(name)
            elif path.suffix.lower() == ".gguf":
                try:
                    adapter = inspect_gguf_lora(path)
                    if Path(adapter.signature[0]).is_relative_to(resolved):
                        names.append(name)
                except (OSError, ValueError):
                    # Wrong-role or partial files are never exposed as image LoRAs.
                    continue
    return names
