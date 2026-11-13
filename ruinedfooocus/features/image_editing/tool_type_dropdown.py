"""Keep queued image-tool type values valid across checkpoint transitions."""

import gradio as gr

from ruinedfooocus.features.generation.image_preparation import CONTROL_MODELS

_TOOL_TYPES = frozenset(name.capitalize() for name in CONTROL_MODELS) | {
    "Img2img",
    "Edit",
    "Control",
    "Upscale",
    "Rembg",
    "Image-transform",
    "Image-mask",
    "Image-color",
}


class ToolTypeDropdown(gr.Dropdown):
    # Reuse Gradio's built-in dropdown renderer and dynamic update reconstruction.
    is_template = True

    def preprocess(self, payload):
        if payload in ("Image-transform", "Image-mask", "Image-color"):
            return payload
        choices = [value for _, value in self.choices]
        # A model-change event narrows the choices before dependent queued events
        # necessarily receive its new value. Normalize only known old tool types;
        # arbitrary values still go through the native dropdown validation.
        if isinstance(payload, str) and payload in _TOOL_TYPES and payload not in choices:
            payload = choices[0] if choices else None
        return super().preprocess(payload)
