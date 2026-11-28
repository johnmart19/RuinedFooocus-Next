"""Run auxiliary caption models on the selected device without retaining VRAM."""

import logging

import torch


def caption_device():
    from comfy import model_management

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
