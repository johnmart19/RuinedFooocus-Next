"""Offline theme resolution; network acquisition is an explicit Settings action."""

import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr

from ruinedfooocus.core.paths import data_path

BUILTINS = {
    "": ("Default", gr.themes.Default),
    "builtin:soft": ("Soft", gr.themes.Soft),
    "builtin:monochrome": ("Monochrome", gr.themes.Monochrome),
}
MAX_BYTES = 1024 * 1024
SYSTEM_FONT = ("Arial", "sans-serif")
MONO_FONT = ("Consolas", "monospace")


def theme_id(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Choose a theme or enter an author/theme ID.")
    value = value.strip()
    if len(value) > 256:
        raise ValueError("Theme IDs must be at most 256 characters.")
    if value in ("", "None"):
        return ""
    if value in BUILTINS:
        return value
    if not re.fullmatch(r"[\w-]+/[\w.-]+(?:@[\w.,<>=~^*+-]+)?", value, re.ASCII):
        raise ValueError("Custom theme IDs use author/theme, optionally followed by @version.")
    from huggingface_hub.utils import validate_repo_id
    from semantic_version import SimpleSpec

    repo, _, version = value.partition("@")
    validate_repo_id(repo)
    if version:
        SimpleSpec(version)
    return value


def cache_file(value):
    key = hashlib.sha256(theme_id(value).encode()).hexdigest()
    return data_path("cache", "themes", key + ".json")


def normalized_schema(schema):
    """Accept only native theme tokens, without remote fonts or stylesheet code."""
    if not isinstance(schema, dict) or not isinstance(schema.get("theme"), dict):
        raise ValueError("The download is not a Gradio theme schema.")
    # to_dict omits None-valued dark defaults that other native themes set.
    # The instance owns the complete native token catalogue.
    defaults = vars(gr.themes.Default(font=SYSTEM_FONT, font_mono=MONO_FONT))
    tokens = {}
    for name, value in schema["theme"].items():
        if not isinstance(name, str):
            raise ValueError("Theme token names must be strings.")
        if name.startswith("_") or name in ("name", "font", "font_mono", "custom_css"):
            continue
        if name not in defaults or not isinstance(value, str) or len(value) > 8192:
            raise ValueError(f"Unsupported theme token: {name}.")
        if re.search(r"[\\\x00-\x1f]|/\*|\*/", value):
            raise ValueError(f"Theme token {name} contains unsupported CSS escaping.")
        if re.search(
            r"[{}<>;\"']|@|(?:url|image|image-set|src|expression)\s*\(|javascript:", value, re.I
        ):
            # Retain only Gradio's own checkbox/radio data images. Arbitrary
            # SVG/CSS cannot introduce remote assets or style-tag content.
            if name not in ("checkbox_check", "radio_circle") or value != defaults[name]:
                raise ValueError(f"Theme token {name} contains external or unsupported CSS.")
        tokens[name] = value
    # Let the native compiler enforce its token-reference rules before caching
    # or launch. Applying tokens to Default retains missing native defaults.
    theme = gr.themes.Default(font=SYSTEM_FONT, font_mono=MONO_FONT)
    for name, value in tokens.items():
        for reference in re.findall(r"\*([\w_]+)\b", value):
            if reference not in defaults:
                raise ValueError(f"Unknown theme token reference: {reference}.")
        setattr(theme, name, value)
    theme._get_theme_css()
    return {"theme": tokens}


def _read_cached(value):
    path = cache_file(value)
    saved = _read_cache_json(path)
    if not isinstance(saved, dict) or saved.get("id") != value:
        raise ValueError("Cached theme identity does not match the selection.")
    return normalized_schema(saved["schema"])


def _parse_json(content):
    try:
        return json.loads(content)
    except (ValueError, RecursionError) as error:
        raise ValueError("Theme JSON is invalid.") from error


def _read_cache_json(path):
    with path.open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Cached theme exceeds the supported size.")
    return _parse_json(content)


def validate_selection(value, previous=None):
    """Keep unavailable legacy intent; require new custom choices to be cached."""
    if (
        value == previous
        and isinstance(value, str)
        and value.strip() not in BUILTINS
        and value.strip() != "None"
    ):
        return value
    value = theme_id(value)
    if value not in BUILTINS:
        try:
            _read_cached(value)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("Download this custom theme before saving it.") from error
    return value


def theme_choices(current=None):
    choices = [(name, key) for key, (name, _) in BUILTINS.items()]
    folder = data_path("cache", "themes")
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        try:
            value = theme_id(_read_cache_json(path)["id"])
            _read_cached(value)
            if value not in [key for _, key in choices]:
                choices.append((value, value))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if (
        isinstance(current, str)
        and current not in ("", "None")
        and current not in [value for _, value in choices]
    ):
        choices.append((str(current) + " (not cached)", current))
    return choices


def resolve_theme(value):
    """Never acquire during startup, including for saved legacy Hub IDs."""
    warning = ""
    try:
        value = theme_id(value)
        if value in BUILTINS:
            theme = BUILTINS[value][1](font=SYSTEM_FONT, font_mono=MONO_FONT)
        else:
            theme = gr.themes.Default(font=SYSTEM_FONT, font_mono=MONO_FONT)
            for key, token in _read_cached(value)["theme"].items():
                setattr(theme, key, token)
    except (OSError, ValueError, KeyError, TypeError) as error:
        warning = f"Theme unavailable: {error} Using Default; the saved choice is unchanged."
        value = ""
        theme = gr.themes.Default(font=SYSTEM_FONT, font_mono=MONO_FONT)
    if value in BUILTINS:
        # Native Default's orange buttons and Soft's hover/selected states use
        # pale fills with white text. Use darker shades from their own palette.
        # Their light-mode help and secondary text also need darker neutrals.
        if value != "builtin:monochrome":
            theme.set(
                body_text_color_subdued="*neutral_600",
                block_info_text_color="*neutral_600",
                button_primary_background_fill="*primary_700",
                button_primary_background_fill_dark="*primary_700",
                button_primary_background_fill_hover="*primary_800",
                button_primary_background_fill_hover_dark="*primary_800",
                button_primary_text_color="white",
                button_primary_text_color_dark="white",
                button_primary_text_color_hover="white",
                button_primary_text_color_hover_dark="white",
                checkbox_label_background_fill_selected="*primary_700",
                checkbox_label_background_fill_selected_dark="*primary_700",
                checkbox_label_text_color_selected="white",
                checkbox_label_text_color_selected_dark="white",
            )
        theme.set(
            color_accent="*primary_700",
            button_secondary_text_color_hover="*body_text_color",
            button_secondary_text_color_hover_dark="*body_text_color",
            link_text_color="*primary_700",
            link_text_color_hover="*primary_800",
            link_text_color_active="*primary_800",
            link_text_color_visited="*primary_700",
            link_text_color_dark="*primary_300",
            link_text_color_hover_dark="*primary_200",
            link_text_color_active_dark="*primary_200",
            link_text_color_visited_dark="*primary_300",
        )
        if value == "builtin:soft":
            theme.set(
                block_label_text_color="*primary_800",
                block_title_text_color="*primary_800",
                button_secondary_background_fill_hover_dark="*neutral_700",
            )
    for name, spacing in zip(
        ("xxs", "xs", "sm", "md", "lg", "xl", "xxl"),
        (".125rem", ".25rem", ".375rem", ".5rem", ".75rem", "1rem", "1.5rem"),
    ):
        setattr(theme, "spacing_" + name, spacing)
    # Root text size is a browser preference. Native px text tokens otherwise
    # ignore that preference, and downloaded geometry must not enlarge cards.
    for name, pixels in zip(
        ("xxs", "xs", "sm", "md", "lg", "xl", "xxl"), (9, 10, 12, 14, 16, 22, 26)
    ):
        setattr(theme, "text_" + name, f"{pixels / 16:g}rem")
    return theme.set(
        layout_gap=".75rem",
        block_padding=".75rem",
        input_padding=".625rem",
        block_radius=".375rem",
        input_radius=".375rem",
        button_large_radius=".375rem",
        button_small_radius=".375rem",
    ), warning


def acquire_theme(value, *, offline=False):
    """Fetch a bounded, revision-bound public schema using Gradio's version rules."""
    if offline:
        raise ValueError("Offline mode: custom theme downloads are disabled.")
    value = theme_id(value)
    if value in BUILTINS:
        return value
    from gradio.themes.base import get_matching_version, get_theme_assets
    from huggingface_hub import HfApi, hf_hub_url
    import requests

    repo, _, version = value.partition("@")
    info = HfApi(token=False).space_info(repo, files_metadata=True, timeout=15)
    if info.private or not info.sha:
        raise ValueError("Choose a public theme with a published revision.")
    # Native discovery also treats non-schema files in themes/ as versions.
    # Filter those first; retain Gradio's semantic-version matching behavior.
    from types import SimpleNamespace

    siblings = [
        item
        for item in info.siblings
        if re.fullmatch(r"themes/theme_schema@[^/]+\.json", item.rfilename)
    ]
    asset = get_matching_version(
        get_theme_assets(SimpleNamespace(siblings=siblings)), version or None
    )
    if asset is None:
        raise ValueError("No theme version matches that ID.")
    filename = asset.filename
    metadata = next(item for item in info.siblings if item.rfilename == filename)
    if metadata.size is not None and metadata.size > MAX_BYTES:
        raise ValueError("Theme exceeds the supported size.")
    url = hf_hub_url(repo, filename, repo_type="space", revision=info.sha)
    payload = bytearray()
    with requests.get(url, stream=True, timeout=(15, 30)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65536):
            payload.extend(chunk)
            if len(payload) > MAX_BYTES:
                raise ValueError("Theme exceeds the supported size.")
    schema = normalized_schema(_parse_json(payload))
    saved = {"id": value, "revision": info.sha, "schema": schema}
    serialized = json.dumps(saved, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Cached theme exceeds the supported size.")
    destination = cache_file(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return value
