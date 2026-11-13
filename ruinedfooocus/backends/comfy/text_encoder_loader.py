"""Reader dispatch for native mixed GGUF/state-dictionary text encoders."""

from pathlib import Path
from contextlib import nullcontext


class _GGUFReaderPath(str):
    """Preserve filesystem spelling for the pin's case-sensitive reader branch."""

    def endswith(self, suffix, *bounds):
        if suffix == ".gguf" and not bounds:
            return self.casefold().endswith(suffix)
        return super().endswith(suffix, *bounds)


def has_gguf_encoder(paths):
    return any(Path(path).suffix.casefold() == ".gguf" for path in paths)


def load_mixed_clip(paths, clip_type, *, projectors=None):
    """Keep native parsing/patching while admitting the discovered suffix casing.

    Pinned CLIPLoaderGGUF.load_data uses a case-sensitive suffix test. This
    adapter changes reader selection and retains native validation/progress.
    Explicit single-encoder Qwen3-VL text-only metadata selects the pin's
    Dynamic VRAM loader, including its device eligibility and fallback.
    Paths and file bytes are never renamed or normalized to lowercase.
    """
    from ruinedfooocus.features.models import gguf_projectors
    from ruinedfooocus.features.models.gguf_metadata import qwen3vl_text_only_variant

    paths = [str(path) for path in paths]
    text_only = [
        path
        for path in paths
        if Path(path).suffix.casefold() == ".gguf" and qwen3vl_text_only_variant(path)
    ]
    if text_only and len(paths) != 1:
        raise ValueError("The text-only Qwen3-VL Q8_CR encoder must be loaded by itself.")
    with gguf_projectors.read_scope(projectors) if projectors is not None else nullcontext():
        if text_only:
            from molbal_comfyui_gguf.nodes import _load_dynamic_gguf_clip

            return _load_dynamic_gguf_clip(
                [_GGUFReaderPath(paths[0])], clip_type, enable_q8_native=True
            )
        return _load_mixed_clip(paths, clip_type, projectors)


def _load_mixed_clip(paths, clip_type, projectors):
    from comfy.utils import load_torch_file
    from molbal_comfyui_gguf.loader import gguf_clip_loader
    from molbal_comfyui_gguf.nodes import DualCLIPLoaderGGUF, GGUFLoadProgress
    from ruinedfooocus.features.models.gguf_projectors import bind

    paths = [str(path) for path in paths]
    progress = GGUFLoadProgress(paths)
    data = []
    for path in paths:
        if Path(path).suffix.casefold() == ".gguf":
            reader = (
                bind(gguf_clip_loader, path, projectors)
                if projectors is not None
                else gguf_clip_loader
            )
            state, extra = reader(
                path, progress_callback=progress.callback_for(path), return_extra=True
            )
            if (
                extra.get("arch_str") == "qwen3vl"
                and extra.get("qwen3vl_text_only")
                and "int8_tensorwise" in extra.get("gguf_quant_formats", ())
            ):
                raise ValueError(
                    "Qwen3-VL text-only Q8_CR requires the native single-CLIP Dynamic VRAM "
                    "loader; this mixed image encoder route cannot load it."
                )
        else:
            state = load_torch_file(path, safe_load=True)
            if "scaled_fp8" in state:
                raise ValueError(
                    "Mixing scaled FP8 with GGUF is not supported by the native text encoder "
                    "loader. Select regular compatible encoders."
                )
        data.append(state)
        progress.complete_file(path)
    return DualCLIPLoaderGGUF().load_patcher(paths, clip_type, data)
