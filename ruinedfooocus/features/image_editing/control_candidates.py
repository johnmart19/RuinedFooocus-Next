"""Local control-model choices from the pinned native loader's header branches."""

from functools import lru_cache
from pathlib import Path

from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES, SAFETENSORS_SUFFIXES
from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature

CONTROL_KINDS = frozenset(("canny", "depth", "sketch", "recolour", "control"))


def model_family(family):
    # Keep SD1/SD2 distinct; their component registry intentionally shares a VAE.
    if family in ("SD 1.5", "SD15"):
        return "SD15"
    if family in ("SD 2.0", "SD20"):
        return "SD20"
    from ruinedfooocus.features.models.image_components import ALIASES

    family = ALIASES.get(family, family)
    return {
        "BaseModel": "SD",
        "SDXL 1.0": "SDXL",
        "SDXL": "SDXL",
        "Flux": "Flux",
        "SD3": "SD3",
        "QwenImage": "QwenImage",
        "HunyuanDiT": "HunyuanDiT",
    }.get(family)


def classify_header(header):
    """Recognize only branches in comfy.controlnet.load_controlnet_state_dict.

    Unknown is a visible explicit choice, never an automatic compatibility claim.
    No filename or dtype contributes to a family decision.
    """
    keys = set(header)

    def shape(key):
        row = header.get(key, {})
        value = row.get("shape", ()) if isinstance(row, dict) else ()
        return value if isinstance(value, (tuple, list)) else ()

    kind, families = "ControlNet", ()
    if "after_proj_list.18.bias" in keys:
        families = ("HunyuanDiT",)
    elif "controlnet_blocks.0.linear.weight" in keys:
        families = ("Flux",)
    elif "controlnet_blocks.0.weight" in keys:
        if "double_blocks.0.img_attn.norm.key_norm.scale" in keys:
            families = ("Flux",)
        elif "pos_embed_input.proj.weight" in keys:
            families = ("SD3",)
        elif "transformer_blocks.0.img_mlp.net.0.proj.weight" in keys:
            families = ("QwenImage",)
        elif "controlnet_x_embedder.weight" in keys:
            families = ("Flux",)
    elif {
        "control_blocks.0.after_proj.weight",
        "control_blocks.0.img_mod.1.weight",
        "control_img_in.weight",
    }.issubset(keys):
        families = ("QwenImage",)
    elif (
        "lora_controlnet" in keys
        or any(key.endswith("zero_convs.0.0.weight") for key in keys)
        or "controlnet_cond_embedding.conv_in.weight" in keys
    ):
        contexts = {
            shape(key)[1]
            for key in keys
            if key.endswith(("attn2.to_k.weight", "attn2.to_k.down")) and len(shape(key)) == 2
        }
        if len(contexts) == 1:
            families = {768: ("SD15",), 1024: ("SD20",), 2048: ("SDXL",)}.get(contexts.pop(), ())
    elif "body.0.in_conv.weight" in keys:
        kind = "T2I adapter"
        channels = shape("body.0.in_conv.weight")
        if len(channels) == 4 and channels[1] in (64, 192):
            families = ("SD15", "SD20")
    elif "conv_in.weight" in keys and "body.0.block2.weight" in keys:
        kind = "T2I adapter"
        channels = shape("conv_in.weight")
        if len(channels) == 4:
            families = {
                64: ("SD15", "SD20"),
                192: ("SD15", "SD20"),
                256: ("SDXL",),
                768: ("SDXL",),
            }.get(channels[1], ())
    elif "adapter.body.0.resnets.0.block1.weight" in keys:
        kind = "T2I adapter"
        channels = shape("adapter.conv_in.weight")
        if len(channels) == 4:
            families = {
                64: ("SD15", "SD20"),
                192: ("SD15", "SD20"),
                256: ("SDXL",),
                768: ("SDXL",),
            }.get(channels[1], ())
    return {"kind": kind, "families": families}


def inspect_control(path):
    try:
        return classify_header(cached_safetensors_header(path))
    except (OSError, ValueError):
        return {"kind": "Control model", "families": ()}


def compatibility(info, family):
    target = model_family(family)
    if target is None:
        return "incompatible"
    families = info.get("families", ())
    if not families:
        return "unknown"
    if target == "SD":
        return "unknown" if set(families).intersection(("SD15", "SD20")) else "incompatible"
    return "compatible" if target in families else "incompatible"


@lru_cache(maxsize=16)
def _local_files(roots):
    result, names = [], set()
    for root_name in roots:
        root = Path(root_name)
        pending, visited = [root], set()
        while pending:
            directory = pending.pop()
            try:
                resolved = directory.resolve()
                if not resolved.is_relative_to(root) or resolved in visited:
                    continue
                visited.add(resolved)
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif (
                    path.suffix.lower() in (*SAFETENSORS_SUFFIXES, *NATIVE_TORCH_SUFFIXES)
                    and path.is_file()
                    and path.resolve().is_relative_to(root)
                ):
                    name = path.relative_to(root).as_posix()
                    if name not in names:
                        names.add(name)
                        result.append((name, str(path)))
    return tuple(sorted(result))


def local_candidates(manager, family, *, refresh=False):
    roots = tuple(str(Path(root).resolve()) for root in manager.get_model_folders("controlnet"))
    if refresh:
        _local_files.cache_clear()
    rows = []
    for name, path in _local_files(roots):
        try:
            if not any(Path(path).resolve().is_relative_to(Path(root)) for root in roots):
                continue
            signature = file_signature(path)
            info = inspect_control(path)
            if file_signature(path) != signature:
                continue
            status = compatibility(info, family)
            if status != "incompatible":
                rows.append(
                    {
                        "name": name,
                        "path": path,
                        "signature": signature,
                        "compatibility": status,
                        "kind": info["kind"],
                    }
                )
        except OSError:
            continue
    return rows


def selected_control(manager, name, family, expected_signature=None):
    """Re-resolve ordered roots at submission; selection never downloads a file."""
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("Choose an installed ControlNet or T2I adapter from the local list.")
    for folder in manager.get_model_folders("controlnet"):
        root = Path(folder).resolve()
        path = root / name
        if path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("The selected control model moved outside its configured folder.")
            signature = file_signature(path)
            if expected_signature is not None and (
                not isinstance(expected_signature, (list, tuple))
                or list(expected_signature) != signature
            ):
                raise ValueError("The selected control model changed. Refresh and select it again.")
            info = inspect_control(path)
            if file_signature(path) != signature:
                raise ValueError(
                    "The selected control model changed while inspecting it. Select it again."
                )
            if compatibility(info, family) == "incompatible":
                raise ValueError(
                    "This control model is incompatible with the selected image checkpoint."
                )
            if path.suffix.lower() not in (*SAFETENSORS_SUFFIXES, *NATIVE_TORCH_SUFFIXES):
                raise ValueError("Choose a supported native control-model container.")
            return resolved, signature, info
    raise ValueError("The selected control model is no longer installed. Refresh the local list.")


def replay_values(metadata):
    controls = metadata.get("controlnet")
    controls = controls if isinstance(controls, dict) else {}
    return {
        "cn_model": controls.get("model") or "",
        "cn_model_signature": controls.get("model_signature"),
    }
