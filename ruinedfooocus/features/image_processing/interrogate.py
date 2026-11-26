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
    elif kind == "florence-large":
        checkpoint = "florence-community/Florence-2-large"
        processor = AutoProcessor.from_pretrained(checkpoint, **options)
        model = Florence2ForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.float32, **options
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


def _florence_infer(image, *, kind="florence", style=None):
    from ruinedfooocus.features.prompting.image_to_prompt import clean_prompt_output

    image = image.convert("RGB")
    task = "<CAPTION>" if style == "concise" else "<MORE_DETAILED_CAPTION>"
    processor, model = caption_model(kind)

    def infer(device):
        inputs = processor(text=task, images=image, return_tensors="pt").to(device)
        generated = model.generate(**inputs, max_new_tokens=2048, num_beams=6, do_sample=False)
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        result = processor.post_process_generation(text, task=task, image_size=image.size)
        raw = result[task]
        return clean_prompt_output(raw, style=style) if style is not None else raw

    return run_caption(model, infer)


def florence_look(image, prompt, gr, *, style=None, **_kwargs):
    return _florence_infer(image, kind="florence", style=style)


def florence_large_look(image, prompt, gr, *, style=None, **_kwargs):
    return _florence_infer(image, kind="florence-large", style=style)


def vlm_look(
    image,
    prompt,
    gr,
    *,
    style="tags",
    extra_instructions="",
    vision_model=None,
    detail="standard",
    check_cancelled=None,
    prepared=None,
):
    """Describe an image into a styled prompt using a local Vision LLM (GGUF + mmproj)."""
    from concurrent.futures import CancelledError
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.app.resource_lifecycle import service_pipeline
    from ruinedfooocus.features.chat.llama_models import local_models
    from ruinedfooocus.features.chat.llama_pipeline import pipeline as create_llama_pipeline
    from ruinedfooocus.features.chat.llama_vision import image_content
    from ruinedfooocus.features.chat.model_preparation import capture_model, prepare_chat_request
    from ruinedfooocus.features.prompting.image_to_prompt import (
        clean_prompt_output,
        resolve_vision_model,
        vlm_messages,
    )

    if prepared is None:
        chosen = resolve_vision_model(
            vision_model,
            settings_values=shared.settings.default_settings,
            local=local_models(),
            manager=shared.path_manager,
        )
        if not chosen:
            raise ValueError(
                "No local Vision LLM is installed. Install a vision GGUF + mmproj in Models, "
                "or switch Engine to Auto / Florence-2."
            )
        prepared = capture_model(
            chosen,
            shared.settings.default_settings,
            shared.path_manager,
            vision=True,
            local=local_models(),
        )
        if not prepared.present:
            prepare_chat_request({"_chat_preparation": prepared}, cancelled=None)

    llama = create_llama_pipeline()
    try:
        if check_cancelled:
            check_cancelled()
        resident = service_pipeline("chat")
        if resident is not None and resident is not llama:
            resident.unload()
        llama.load_base_model(
            str(prepared.model),
            vision=True,
            prepared=prepared,
            check_cancelled=check_cancelled,
        )
        if check_cancelled:
            check_cancelled()
        payload = image_content(image, detail=detail)
        messages = vlm_messages(
            payload, style=style, extra_instructions=extra_instructions
        )
        parts = []

        def collect(chunk):
            if check_cancelled:
                check_cancelled()
            for item in chunk.get("choices", []):
                text = item.get("delta", {}).get("content")
                if text:
                    parts.append(text)

        parameters = {
            "stream": True,
            "_rf_task": "image_to_prompt",
            "reasoning_effort": "none",
            "max_tokens": max(256, int(prepared.settings.get("llm_hp_max_tokens", 384) or 384)),
            "temperature": 0.3,
            "messages": messages,
        }
        llama.llm.handle_chat_completions(parameters, collect)
        if check_cancelled:
            check_cancelled()
        return clean_prompt_output("".join(parts), style=style)
    except CancelledError:
        raise
    finally:
        llama.unload()


def auto_look(image, prompt, gr, *, style="tags", extra_instructions="", vision_model=None, detail="standard", **kwargs):
    """Use the best locally installed Vision LLM if available, otherwise fall back to Florence-2."""
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.features.chat.llama_models import local_models
    from ruinedfooocus.features.prompting.image_to_prompt import resolve_vision_model

    chosen = resolve_vision_model(
        vision_model,
        settings_values=shared.settings.default_settings,
        local=local_models(),
        manager=shared.path_manager,
        auto_only=True,
    )
    if chosen:
        return vlm_look(
            image,
            prompt,
            gr,
            style=style,
            extra_instructions=extra_instructions,
            vision_model=chosen,
            detail=detail,
            **kwargs,
        )
    return florence_look(image, prompt, gr, style=style)


looks = {
    "auto": auto_look,
    "vlm": vlm_look,
    "florence": florence_look,
    "florence-large": florence_large_look,
    "brainblip": brainblip_look,
    "clip": clip_look,
}


def queued_caption(
    image,
    kind,
    gr,
    *,
    style=None,
    extra_instructions="",
    vision_model=None,
    detail="standard",
):
    from ruinedfooocus.app.api_runtime import submit, wait_for_result

    label = {
        "auto": "Auto (Vision LLM / Florence)",
        "vlm": "Vision LLM",
        "florence": "Florence-2",
        "florence-large": "Florence-2 Large",
        "brainblip": "BrainBLIP",
        "clip": "CLIP",
    }.get(kind, kind.title())
    gr.Info(f"{label} is creating your prompt. Waiting for the image worker.")
    uses_vlm = kind == "vlm"
    if kind == "auto":
        try:
            from ruinedfooocus.app import state as shared
            from ruinedfooocus.features.chat.llama_models import local_models
            from ruinedfooocus.features.prompting.image_to_prompt import resolve_vision_model

            uses_vlm = bool(
                resolve_vision_model(
                    vision_model,
                    settings_values=shared.settings.default_settings,
                    local=local_models(),
                    manager=shared.path_manager,
                    auto_only=True,
                )
            )
        except Exception:
            uses_vlm = False

    def operation(job, task_id):
        fn = looks[kind]
        if kind in ("vlm", "auto", "florence", "florence-large"):
            kwargs = {"style": style}
            if kind in ("vlm", "auto"):
                kwargs.update(
                    extra_instructions=extra_instructions,
                    vision_model=vision_model,
                    detail=detail,
                    check_cancelled=job.check if hasattr(job, "check") else None,
                )
            return fn(image, kind, gr, **kwargs)
        result = fn(image, kind, gr)
        if style is not None:
            from ruinedfooocus.features.prompting.image_to_prompt import clean_prompt_output

            return clean_prompt_output(result, style=style)
        return result

    operation.resource_group = "chat" if uses_vlm else "caption"
    # Upload callbacks otherwise race generation and its GPU memory controller.
    job = submit(operation)
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
