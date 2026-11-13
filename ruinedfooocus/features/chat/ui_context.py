"""Readable conversation context without loading or probing a model."""

from pathlib import Path
import re

from ruinedfooocus.core.translation_manager import localize


def _label(value):
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", str(value).replace("\n", " "))


def context_text(selected, loaded, assistant, *, offline=False, locale=None):
    from gradio.i18n import I18nData

    if isinstance(assistant, I18nData):
        assistant = localize(assistant.key.removeprefix("ruinedfooocus."), locale)
    text = localize(
        "Assistant: {assistant} · Selected: {selected} · Loaded: {loaded}",
        locale,
        assistant=_label(assistant),
        selected=_label(Path(selected).name) if selected else localize("No model selected", locale),
        loaded=_label(Path(loaded).name) if loaded else localize("No model loaded", locale),
    )
    if not loaded:
        text += "\n\n" + localize("Press Load to start chatting.", locale)
        if offline:
            text += " " + localize("Offline: only installed or imported models work.", locale)
    return text
