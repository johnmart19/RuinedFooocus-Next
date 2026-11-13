"""Resolve short LoRA names typed in prompts to installed LoRA files."""

from pathlib import Path, PurePath, PurePosixPath
import re
from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES, SAFETENSORS_SUFFIXES
from ruinedfooocus.features.models.model_identity import file_signature

EXTENSIONS = (*SAFETENSORS_SUFFIXES, *NATIVE_TORCH_SUFFIXES, ".gguf")


def _key(text):
    return str(text).replace("\\", "/").casefold()


def _stem(name):
    path = PurePath(str(name).replace("\\", "/"))
    return path.stem if path.suffix.lower() in EXTENSIONS else path.name


def is_gguf_lora_name(name):
    return PurePosixPath(_key(name)).suffix == ".gguf"


def parse_loras(prompt, negative):
    """Extract literal inline LoRA tags with the worker's existing weight rules."""
    pattern = re.compile(r"<lora:([^<>:]+?)(?::([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?>")
    loras = []
    for text in (prompt, negative):
        for match in re.finditer(pattern, text):
            weight = match.group(2)
            loras.append(
                {
                    "name": match.group(1).strip(),
                    "weight": 1.0 if weight is None else float(weight),
                    "hash": None,
                }
            )
    return loras, re.sub(pattern, "", prompt), re.sub(pattern, "", negative)


def capture_gguf_lora(name, roots):
    """Capture an explicit adapter file from the configured local LoRA roots."""
    from ruinedfooocus.features.models.gguf_adapters import inspect_gguf_lora

    normalized = str(name).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not is_gguf_lora_name(normalized)
        or relative.is_absolute()
        or ":" in normalized
        or ".." in relative.parts
    ):
        raise ValueError("Choose a GGUF LoRA by its relative filename in a configured LoRA folder.")
    for root in roots:
        root = Path(root).resolve()
        source = root.joinpath(*relative.parts)
        if source.is_file():
            try:
                signature = file_signature(source)
            except OSError as error:
                raise ValueError(
                    "The GGUF LoRA changed before loading. Submit the request again."
                ) from error
            if not Path(signature[0]).is_relative_to(root):
                raise ValueError("The GGUF LoRA resolves outside its configured LoRA folder.")
            if inspect_gguf_lora(source).signature != tuple(signature):
                raise ValueError("The GGUF LoRA changed before loading. Submit the request again.")
            return source, signature
    raise FileNotFoundError(f"Could not find installed GGUF LoRA: {name}")


def resolve_lora_name(name, installed):
    """Return the installed LoRA a prompt name refers to.

    Accepts the exact relative filename, the filename without extension or
    folder, case-insensitively, or a part of the name that matches exactly one
    installed LoRA. Ambiguous names raise ValueError listing the candidates.
    Unknown names are returned unchanged (with ``.safetensors`` when no extension
    was given) so missing-file handling and catalogue downloads still apply.
    """
    name = str(name).strip()
    installed = list(installed or [])
    if name in installed:
        return name
    wanted = _key(name)
    wanted_stem = _key(_stem(name))
    rules = (
        lambda item: _key(item) == wanted,
        lambda item: _key(PurePath(_key(item)).with_suffix("")) == wanted,
        lambda item: _key(_stem(item)) == wanted_stem,
        lambda item: wanted_stem and wanted_stem in _key(_stem(item)),
    )
    for rule in rules[:1] if is_gguf_lora_name(name) else rules:
        matches = list(dict.fromkeys(item for item in installed if rule(item)))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            shown = ", ".join(_stem(item) for item in matches[:5])
            more = f" and {len(matches) - 5} more" if len(matches) > 5 else ""
            raise ValueError(
                f"LoRA name '{name}' matches several installed LoRAs ({shown}{more}). "
                "Type more of its name."
            )
    has_extension = PurePath(name).suffix.lower() in EXTENSIONS
    return name if has_extension else f"{name}.safetensors"


def merge_loras(selected, typed, installed):
    """Combine card-selected and prompt LoRAs, applying each file once.

    A prompt entry overrides the weight of the same LoRA selected as a card.
    """
    merged = {}
    for lora in selected:
        merged[_key(lora["name"])] = dict(lora)
    for lora in typed:
        resolved = resolve_lora_name(lora["name"], installed)
        merged[_key(resolved)] = {**lora, "name": resolved}
    return list(merged.values())
