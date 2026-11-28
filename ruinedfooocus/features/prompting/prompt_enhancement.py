"""Shared image-prompt enhancement through the configured local chat model."""

import json
import re
from ruinedfooocus.features.prompting.local_model import run_prompt

PROMPT_ADDITION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "prompt_addition",
        "schema": {
            "type": "object",
            "properties": {"addition": {"type": "string"}},
            "required": ["addition"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def image_prompt_instruction(model_type, preset_focus=""):
    format_hint = (
        "Use concise comma-separated visual tags."
        # One Button tag formats (SDXL, anime) get tags; natural-language models get prose.
        if model_type in ("Anime Model", "SDXL", "SD1.5")
        else "Use one short sentence."
    )
    instruction = (
        "Return only up to 20 words of neutral lighting or surface-texture details compatible with the scene. "
        "The application keeps the scene verbatim: never repeat it. "
        "Do not add subjects, objects, counts, colors, actions, geometry changes, overlays, frames, "
        "quoted text, tags, weights or explanations. Respect exclusions, language and medium. "
        'Return JSON with one "addition" string, empty if nothing is needed. '
        'Example addition: "soft diffuse light, subtle surface texture". ' + format_hint
    )
    if preset_focus:
        instruction += (
            " Preset direction: "
            + preset_focus
            + " Apply this direction only where compatible with the supplied scene; "
            "explicit subject details and requested medium take precedence."
        )
    return instruction


def enhance_prompt(prompt, model_type="", preset_focus="", *, check_cancelled=None):
    if not prompt.strip():
        return prompt
    return run_prompt(
        prompt,
        instruction=image_prompt_instruction(model_type, preset_focus),
        check_cancelled=check_cancelled,
    )


def image_prompt_result(choice, original):
    if choice.get("finish_reason") != "stop":
        return original
    text = choice.get("message", {}).get("content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Some templates prefill the opening marker; only its closing marker reaches content.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError:
            return original
        if not isinstance(payload, dict) or not isinstance(payload.get("addition"), str):
            return original
        text = payload["addition"].strip()
    if "<think>" in text or not text or original in text:
        return original
    # Enhancement appends to the exact original. A model cannot safely introduce
    # counts or executable prompt syntax through an allegedly visual addition.
    if len(text.split()) > 20 or re.search(
        r"\d|<|>|__|\[|\]|\{|\}|\(|\)|---|:|\||\b(?:AND|BREAK)\b", text
    ):
        return original
    # Keep proper names and negative instructions in the original payload: the
    # enhancer is restricted to neutral light/texture language, not scene edits.
    if re.search(r"\b(?:replace|remove|instead|ignore|change|without|no|not|must)\b", text, re.I):
        return original
    quoted = r'"([^"\n]*)"|“([^”\n]*)”|«([^»\n]*)»'
    original_quotes = {
        next(part for part in match if part) for match in re.findall(quoted, original) if any(match)
    }
    additions = {
        next(part for part in match if part) for match in re.findall(quoted, text) if any(match)
    }
    if not additions <= original_quotes:
        return original
    return original + "\n" + text
