"""Run auxiliary caption models on the selected device without retaining VRAM."""

import logging

import torch
from ruinedfooocus.core.device_fallback import warn_cpu_fallback


def caption_device():
    from comfy import model_management

    # Transformers generation has no qualified DirectML operator coverage.
    if model_management.is_directml_enabled():
        warn_cpu_fallback(
            "image captioning",
            "Transformers caption inference is not qualified for the selected DirectML backend",
        )
        return torch.device("cpu")
    return model_management.get_torch_device()


def run_caption(model, infer):
    from comfy import model_management

    device = caption_device()
    model.eval()
    devices = [device] if device.type == "cpu" else [device, torch.device("cpu")]
    for target in devices:
        try:
            if target.type != "cpu":
                size = sum(t.numel() * t.element_size() for t in model.parameters())
                model_management.free_memory(size * 1.3 + 512 * 1024**2, target)
            model.to(target)
            with torch.inference_mode():
                return infer(target)
        except torch.OutOfMemoryError:
            if target.type == "cpu":
                raise
            logging.warning("Caption inference exceeded GPU memory; retrying on CPU.")
        finally:
            model.to("cpu")
