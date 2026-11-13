"""Read-only style discovery shared by prompt composition and the UI."""

import csv
from dataclasses import dataclass, replace
from functools import lru_cache
import locale
from pathlib import Path
import re

from ruinedfooocus.core.paths import data_path, resource_path

STYLE_ALIASES = {
    value: "Local chat model"
    for name in ("Flufferizer", "Hyperprompt", "Erniehancer", "SuperPrompt", "Local chat model")
    for value in (name, "Style: " + name)
}
PROCESSORS = {
    "None": ("{prompt}", ""),
    "Style: Pick Random": ("{prompt}", ""),
    "LoRA keywords": ("{prompt} {lora_keywords}", ""),
    "Local chat model": ("{prompt}", ""),
}
GROUP_NAMES = {
    "SDXL COMFYUI PORT": "General styles",
    "ADS": "Advertising",
    "ARTSTYLE": "Art styles",
    "FUTURISTIC": "Futuristic",
    "GAME": "Games",
    "MISC": "Miscellaneous",
    "PAPERCRAFT": "Papercraft",
    "PHOTO": "Photography",
    "ARTIFY": "Artify",
}


@dataclass(frozen=True)
class Style:
    name: str
    prompt: str
    negative_prompt: str
    group: str


def is_heading(name):
    return isinstance(name, str) and name.lstrip().startswith(">>>")


def normalize_selection(selected):
    """Legacy headers are inert, including selections restored from old PNGs."""
    if isinstance(selected, str):
        selected = [selected]
    return list(
        dict.fromkeys(
            STYLE_ALIASES.get(name, name)
            for name in selected or ()
            if isinstance(name, str) and name != "None" and not is_heading(name)
        )
    )


@lru_cache(maxsize=16)
def _read_styles(content, encoding):
    # Only the two style CSVs enter this cache, never the large prompt datasets.
    # Content keys detect same-size edits even when file timestamps are unchanged.
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode(encoding)
    # Preserve read_text's universal-newline behavior, including quoted CSV cells.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    group = "Custom"
    records = []
    for row in csv.DictReader(text.splitlines(keepends=True)):
        name = row.get("name")
        if not name:
            continue
        if is_heading(name):
            heading = name.lstrip("> ").strip()
            group = GROUP_NAMES.get(heading.upper(), heading.title())
        elif name not in STYLE_ALIASES and name not in PROCESSORS:
            records.append(
                Style(name, row.get("prompt") or "", row.get("negative_prompt") or "", group)
            )
    return tuple(records)


def load_catalog(default_file=None, user_file=None):
    """User rows override bundled rows by name; neither file is ever rewritten."""
    default_file = (
        Path(default_file) if default_file else resource_path("defaults", "styles.default")
    )
    user_file = Path(user_file) if user_file else data_path("settings", "styles.csv")
    catalog = {}
    for path in (default_file, user_file):
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            continue
        for style in _read_styles(content, locale.getpreferredencoding(False)):
            if style.name in catalog:
                style = replace(style, group=catalog[style.name].group)
            catalog[style.name] = style
    return catalog


def style_templates(default_file=None, user_file=None):
    return PROCESSORS | {
        name: (style.prompt, style.negative_prompt)
        for name, style in load_catalog(default_file, user_file).items()
    }


def style_label(name):
    label = name.removeprefix("Style: ").removeprefix("Artify: ")
    for prefix in (
        "sai-",
        "ads-",
        "artstyle-",
        "futuristic-",
        "game-",
        "misc-",
        "papercraft-",
        "photo-",
    ):
        label = label.removeprefix(prefix)
    return label[:1].upper() + label[1:]


def merge_selection(current, scope, selected):
    """Changing a filtered group must not drop selections hidden by the search."""
    scope = set(scope)
    selected = [name for name in normalize_selection(selected) if name in scope]
    result = [
        name for name in normalize_selection(current) if name not in scope or name in selected
    ]
    return result + [name for name in selected if name not in result]


def resolve_inline_styles(prompt, selected, templates, *, deferred=()):
    """Consume known directives once; unknown text and deferred tools stay literal."""
    selected = normalize_selection(selected)

    def inline(match):
        name = STYLE_ALIASES.get(match.group(1), match.group(1))
        if is_heading(name):
            return ""
        key = f"Style: {name}"
        if key not in templates:
            key = name
        if key in templates and key not in deferred:
            selected.append(key)
            return ""
        return match.group(0)

    prompt = re.sub(r"<style:([^>]+)>", inline, prompt)
    return normalize_selection(selected), prompt
