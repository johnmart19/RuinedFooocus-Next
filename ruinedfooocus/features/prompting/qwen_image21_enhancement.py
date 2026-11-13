"""Official Qwen Image 2.1 rewrites through the pinned native text generator."""

from dataclasses import dataclass
import json
import os
import re
import secrets

from PIL import Image, ImageOps

from ruinedfooocus.core.paths import data_path
from ruinedfooocus.core.preparation import PreparationRequest


DEFAULT_TOKENS = 4096
MAX_TOKENS = 24000
MAX_REFERENCES = 10
MAX_REFERENCE_PIXELS = 16777216
_REFERENCE = re.compile(r"<image([1-9]\d*)>")
_RATIO = re.compile(r"[1-9][0-9]{0,4}:[1-9][0-9]{0,4}")
_CHAT_CONTROL = re.compile(r"<\|[^<>]*\|>")
_SYSTEM_PROMPTS = {
    "t2i": (
        "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b",
        10045,
        "a77c9a06c59b120741141d9514b95682bb8761d02bec49ca61def7b2b3d9fb99",
    ),
    "i2i": (
        "72927bc08afc99b7888ceb7d7d51a12db3700bbd",
        18344,
        "e378fea686a1431581ba4c654d332ae96adad633f144ae738ec8ce9c4fd66439",
    ),
}


@dataclass(frozen=True)
class Rewrite:
    prompt: str
    ratio: str
    follow_reference: int | None = None


def system_prompt_request(mode):
    revision, size, sha256 = _SYSTEM_PROMPTS[mode]
    model = "Qwen-Image-2.1-PE-" + mode.upper()
    return PreparationRequest(
        f"https://huggingface.co/Qwen/{model}/resolve/{revision}/system_prompt.txt",
        data_path("cache", "qwen-image-2.1-pe", revision, "system_prompt.txt"),
        expected_size=size,
        sha256=sha256,
    )


def weight_filename(mode):
    return f"qwen3.5_9b_qwen_image_2.1_pe_{mode}.int8_convrot.safetensors"


def reference_images(primary, additional):
    """Capture ordered RGB previews of the active edit references."""
    from ruinedfooocus.features.image_editing.image_guidance import normalize_reference_images

    if not isinstance(primary, Image.Image) or not 0 < primary.width * primary.height <= (
        MAX_REFERENCE_PIXELS
    ):
        raise ValueError("Add a primary image of at most 16 megapixels in Image tools.")
    if isinstance(additional, (list, tuple)) and len(additional) >= MAX_REFERENCES:
        raise ValueError(
            "Qwen prompt enhancement supports up to 10 references including the primary."
        )
    primary = ImageOps.exif_transpose(primary).convert("RGBA")
    ordered = [primary, *normalize_reference_images(additional, primary, "QwenImage21")]
    references = []
    for image in ordered:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        references.append(background)
    return tuple(references)


def parse_answer(text, reference_count):
    """Extract the published JSON fields from the final answer."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.I)
    try:
        payload = json.loads(fenced[1] if fenced else text)
    except (ValueError, RecursionError):
        raise ValueError("Qwen did not finish a valid rewrite. Try a higher token limit.") from None
    if not isinstance(payload, dict):
        raise ValueError("Qwen returned an invalid rewrite.")
    prompt = payload.get("rewritten_prompt")
    if not isinstance(prompt, str) or not prompt.strip() or _CHAT_CONTROL.search(prompt):
        raise ValueError("Qwen returned an invalid rewritten prompt.")
    ratio = payload.get("wh_ratio", "")
    follow = payload.get("ratio_follow", "")
    if not isinstance(ratio, str) or not isinstance(follow, str):
        raise ValueError("Qwen returned an invalid aspect-ratio suggestion.")
    valid_indices = {str(index) for index in range(1, reference_count + 1)}
    follow_match = _REFERENCE.fullmatch(follow)
    if ratio and not follow and _RATIO.fullmatch(ratio):
        index = None
    elif not ratio and follow_match and follow_match[1] in valid_indices:
        index = int(follow_match[1])
    else:
        raise ValueError("Qwen returned an invalid aspect-ratio suggestion.")
    if any(value not in valid_indices for value in _REFERENCE.findall(prompt)):
        raise ValueError("Qwen referred to an image that was not supplied.")
    return Rewrite(prompt.strip(), ratio, index)


def generate_answer(
    path,
    system_prompt,
    prompt,
    references=(),
    *,
    max_tokens=DEFAULT_TOKENS,
    seed=42,
    check_cancelled=None,
):
    """Call the native node inside the caller's compute lease."""
    import numpy as np
    import torch
    from comfy import sd, utils
    from comfy_extras.nodes_textgen import TextGenerateV2

    if check_cancelled is not None:
        check_cancelled()
    clip = sd.load_clip(
        [str(path)],
        clip_type=sd.CLIPType.QWEN_IMAGE if references else sd.CLIPType.STABLE_DIFFUSION,
    )
    images = {
        f"image{index}": torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None, ...]
        for index, image in enumerate(references, 1)
    }
    previous_hook = utils.PROGRESS_BAR_HOOK

    def progress(current, total, preview, *, node_id=None):
        check_cancelled()
        if previous_hook is not None:
            previous_hook(current, total, preview, node_id=node_id)

    if check_cancelled is not None:
        check_cancelled()
        utils.set_progress_bar_global_hook(progress)
    try:
        with torch.inference_mode():
            return TextGenerateV2.execute(
                clip=clip,
                prompt=prompt,
                max_length=max_tokens,
                sampling_mode={
                    "sampling_mode": "on",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                    "seed": seed,
                },
                images=images,
                thinking=True,
                system_prompt=system_prompt,
            ).result[0]
    finally:
        if check_cancelled is not None:
            utils.set_progress_bar_global_hook(previous_hook)


def rewrite_prompt(prompt, references=(), *, max_tokens=DEFAULT_TOKENS, seed=None):
    """Prepare official artifacts outside compute, then use the owned image worker."""
    from ruinedfooocus.app import async_worker as worker, state as shared
    from ruinedfooocus.app.api_runtime import submit, wait_for_result
    from ruinedfooocus.app.resource_lifecycle import require_compute_owner
    from ruinedfooocus.features.generation.image_preparation import (
        KEY,
        ImageArtifact,
        ImageAssets,
        prepare_image_assets,
        prepared_asset,
    )
    from ruinedfooocus.features.models.model_identity import file_signature

    prompt = prompt.strip()
    if not prompt or _CHAT_CONTROL.search(prompt):
        raise ValueError("Enter a plain-language image prompt before rewriting it.")
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError(f"Maximum rewrite tokens must be between 1 and {MAX_TOKENS}.")
    if len(references) > MAX_REFERENCES:
        raise ValueError(
            "Qwen prompt enhancement supports up to 10 references including the primary."
        )

    mode = "i2i" if references else "t2i"
    manager = shared.path_manager
    filename = weight_filename(mode)
    weight = next(
        (
            root / filename
            for root in manager.get_model_folders("clip")
            if (root / filename).is_file()
        ),
        None,
    )
    request = manager.preparation_request("clip/" + filename, destination=weight)
    signature = tuple(file_signature(weight)) if weight is not None else None
    if signature is not None and request.expected_size not in (None, signature[1]):
        raise ValueError(f"The Qwen prompt enhancer has an unexpected file size: {filename}.")
    system = system_prompt_request(mode)
    captured = {
        KEY: ImageAssets(
            (
                ImageArtifact("prompt enhancer", filename, request, signature),
                ImageArtifact("prompt enhancer system prompt", "system_prompt.txt", system, None),
            ),
            bool(shared.args.offline or os.environ.get("RF_OFFLINE") == "1"),
        )
    }
    prepared = None

    def prepare(job):
        nonlocal prepared
        prepared = prepare_image_assets(
            captured, check_cancelled=lambda: worker.check_interrupt({"_api_job": job})
        )

    def operation(job, _task_id):
        from comfy import model_management

        require_compute_owner()

        def check():
            worker.check_interrupt({"_api_job": job})

        with (
            prepared_asset(prepared, "prompt enhancer", filename, manager, "clip") as path,
            prepared_asset(
                prepared, "prompt enhancer system prompt", "system_prompt.txt", manager, "clip"
            ) as system_path,
        ):
            try:
                text = generate_answer(
                    path,
                    system_path.read_text(encoding="utf-8").strip(),
                    prompt,
                    references,
                    max_tokens=max_tokens,
                    seed=secrets.randbits(32) if seed is None else seed,
                    check_cancelled=check,
                )
                check()
                return parse_answer(text, len(references))
            finally:
                model_management.unload_all_models()
                model_management.soft_empty_cache()

    operation.prepare = prepare
    operation.resource_group = "image"
    return wait_for_result(submit(operation))
