"""Image captioning through native Transformers and the shared serial worker."""

from functools import lru_cache
from pathlib import Path
import json
import logging
import os

import torch

from ruinedfooocus.features.image_processing.caption_runtime import caption_device, run_caption


@lru_cache(maxsize=1)
def caption_model(kind):
    from transformers import (
        AutoProcessor,
        BlipForConditionalGeneration,
        Florence2ForConditionalGeneration,
    )
    from ruinedfooocus.app.arguments import args

    options = {"local_files_only": args.offline or os.environ.get("RF_OFFLINE") == "1"}
    if kind == "brainblip":
        processor = AutoProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", **options
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "braintacles/brainblip", dtype=torch.float32, **options
        )
    else:
        checkpoint = "florence-community/Florence-2-base"
        processor = AutoProcessor.from_pretrained(checkpoint, **options)
        model = Florence2ForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.float32, **options
        )
    return processor, model


def brainblip_look(image, prompt, gr):
    processor, model = caption_model("brainblip")

    def infer(device):
        inputs = processor(image.convert("RGB"), return_tensors="pt").to(device)
        out = model.generate(
            **inputs, min_length=40, max_new_tokens=150, num_beams=5, repetition_penalty=1.40
        )
        return processor.decode(out[0], skip_special_tokens=True)

    return run_caption(model, infer)


def clip_look(image, prompt, gr):
    from clip_interrogator import Config, Interrogator
    from ruinedfooocus.app.state import path_manager
    from comfy import model_management

    def infer(device):
        config = Config(
            device=device,
            clip_model_name="ViT-L-14/openai",
            cache_path=str(Path(path_manager.model_paths["cache_path"]) / "clip-interrogator"),
        )
        config.apply_low_vram_defaults()
        interrogator = None
        try:
            if device.type != "cpu":
                model_management.free_memory(4 * 1024**3, device)
            with torch.inference_mode():
                interrogator = Interrogator(config)
                return interrogator.interrogate(image.convert("RGB"))
        finally:
            if interrogator is not None:
                for model in (interrogator.caption_model, interrogator.clip_model):
                    if model is not None:
                        model.to("cpu")

    device = caption_device()
    try:
        return infer(device)
    except torch.OutOfMemoryError:
        if device.type == "cpu":
            raise
        logging.warning("CLIP interrogation exceeded GPU memory; retrying on CPU.")
    return infer(torch.device("cpu"))


def florence_look(image, prompt, gr):
    image = image.convert("RGB")
    task = "<MORE_DETAILED_CAPTION>"
    processor, model = caption_model("florence")

    def infer(device):
        inputs = processor(text=task, images=image, return_tensors="pt").to(device)
        generated = model.generate(**inputs, max_new_tokens=2048, num_beams=6, do_sample=False)
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        result = processor.post_process_generation(text, task=task, image_size=image.size)
        return result[task]

    return run_caption(model, infer)


looks = {"brainblip": brainblip_look, "clip": clip_look, "florence": florence_look}


def queued_caption(image, kind, gr):
    from ruinedfooocus.app.api_runtime import submit, wait_for_result

    gr.Info(f"{kind.title()} is creating your prompt. Waiting for the image worker.")
    # Upload callbacks otherwise race generation and its GPU memory controller.
    job = submit(lambda job, task_id: looks[kind](image, kind, gr))
    return wait_for_result(job)


def metadata_prompt(image):
    """Read only saved prompt text; a PNG recipe is not a textbox value."""
    try:
        parameters = image.info.get("parameters")
    except AttributeError:
        return None
    if not isinstance(parameters, str) or not parameters:
        return None
    try:
        metadata = json.loads(parameters)
    except ValueError:
        # Other tools also save ordinary, unstructured prompt text here.
        return parameters
    if isinstance(metadata, dict):
        metadata = {key.lower(): value for key, value in metadata.items()}
        for key in ("prompt", "raw_prompt"):
            if isinstance(metadata.get(key), str):
                return metadata[key]
    return None


def look(image, prompt, gr):
    selected = prompt.strip(" :")
    if selected in looks:
        return queued_caption(image, selected, gr)
    if prompt:
        return prompt
    saved_prompt = metadata_prompt(image)
    if saved_prompt is not None:
        return saved_prompt
    from ruinedfooocus.app.state import settings

    return queued_caption(image, settings.default_settings.get("interrogator", "florence"), gr)
