"""Stable image dimensions across model-specific recommendation labels."""

import re

import gradio as gr


def dimension_value(value):
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"([0-9]+)x([0-9]+)(?: \((?:.+ recommended|Generic fallback)\))?", value)
    if match is None:
        return value  # Named saved sizes and Custom keep their selection identity.
    return f"{int(match[1])}x{int(match[2])}"


def choice_values(choices):
    result, seen = [], set()
    for choice in choices or ():
        label, value = choice if isinstance(choice, (tuple, list)) else (choice, choice)
        value = dimension_value(value)
        if value not in seen:
            result.append((label, value))
            seen.add(value)
    return result


class ResolutionDropdown(gr.Dropdown):
    is_template = True

    def __init__(self, choices=None, *, value=None, **kwargs):
        super().__init__(choices=choice_values(choices), value=dimension_value(value), **kwargs)

    def get_config(self, cls=None):
        # Native serialization inspects the constructor signature; retain the
        # complete Dropdown schema behind this field's normalization wrapper.
        return super().get_config(cls or gr.Dropdown)

    def preprocess(self, payload):
        value = dimension_value(payload)
        if value in [entry for _, entry in self.choices]:
            return super().preprocess(value)
        # A queued callback may carry the preceding model's dimension value.
        # Admit only bounded pixel dimensions; the receiving callback still
        # resolves automatic/manual ownership and the native request size plan.
        if isinstance(value, str) and re.fullmatch(r"[0-9]+x[0-9]+", value):
            width, height = map(int, value.split("x"))
            if (
                1 <= min(width, height)
                and max(width, height) <= 32768
                and width * height <= 67108864
            ):
                return value
        return super().preprocess(payload)

    def postprocess(self, value):
        return super().postprocess(dimension_value(value))
