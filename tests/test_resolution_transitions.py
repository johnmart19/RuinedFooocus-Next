"""Actual Gradio size preprocessing survives queued model recommendation changes."""

import asyncio
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.models.image_scope import (
    require_image_checkpoint,
    require_image_operation,
)
from ruinedfooocus.features.models.resolution_dropdown import ResolutionDropdown
from ruinedfooocus.features.models.resolutions import ResolutionSettings


class ResolutionTransitionTests(unittest.TestCase):
    def test_native_config_and_localized_label_survive_dimension_updates(self):
        old = "1024x1024 (SDXL recommended)"
        new = "832x1216 (Flux recommended)"
        properties = dict(
            label=translate_ui("Output target"),
            info=translate_ui("Choosing a size keeps your override until you enable this again."),
            interactive=True,
            show_label=True,
            min_width=240,
            scale=2,
            elem_classes=["size-target"],
        )
        with gr.Blocks(analytics_enabled=False) as app:
            size = ResolutionDropdown([old, "Custom..."], value=old, **properties)
            native = gr.Dropdown(
                [(old, "1024x1024"), ("Custom...", "Custom...")],
                value="1024x1024",
                **properties,
            )
            refresh = gr.Button()
            refresh.click(lambda: gr.update(choices=[new, "Custom..."], value=new), outputs=size)
        self.assertEqual(size.get_config(), native.get_config())
        config = app.get_config_file()
        props = next(item["props"] for item in config["components"] if item["id"] == size._id)
        self.assertIn("ruinedfooocus.shell.output_target", str(props["label"]))
        state = SessionState(app)
        result = asyncio.run(app.process_api(0, [], state=state))
        self.assertEqual(result["data"][0]["value"], "832x1216")
        self.assertEqual(tuple(result["data"][0]["choices"][0]), (new, "832x1216"))
        updated = state.blocks_config.blocks[size._id].get_config()
        for field in properties:
            self.assertEqual(updated[field], native.get_config()[field])
        self.assertFalse(updated["allow_custom_value"])

    def test_queued_sdxl_to_flux_and_rapid_updates_preprocess_stable_dimensions(self):
        old = "1024x1024 (SDXL 1.0 recommended)"
        with gr.Blocks(analytics_enabled=False) as app:
            size = ResolutionDropdown([old, "Custom..."], value=old)
            family = gr.Textbox()
            refresh = gr.Button()
            output = gr.Textbox()
            refresh.click(
                lambda label: gr.update(choices=[label, "Custom..."], value=label),
                inputs=family,
                outputs=size,
            )
            size.change(lambda value: value, inputs=size, outputs=output)
        state = SessionState(app)

        async def transitions():
            previous = old
            for label in (
                "1024x1024 (Flux.3 Klein recommended)",
                "2048x2048 (QwenImage recommended)",
                old,
            ):
                await app.process_api(0, [label], state=state)
                result = await app.process_api(1, [previous], state=state)
                self.assertEqual(result["data"], [previous.split(" ", 1)[0]])
                previous = label
            self.assertEqual(state.blocks_config.blocks[size._id].choices[0], (old, "1024x1024"))

        asyncio.run(transitions())

    def test_named_saved_sizes_custom_and_legacy_recommendations_keep_their_intent(self):
        saved = "1920x1080 (Cinema · SDXL)"
        with gr.Blocks(analytics_enabled=False):
            size = ResolutionDropdown(
                [saved, "1024x1024 (SDXL recommended)", "Custom..."], value=saved
            )
        self.assertEqual(size.preprocess(saved), saved)
        self.assertEqual(size.preprocess("Custom..."), "Custom...")
        self.assertEqual(size.preprocess("1024x1024 (Flux.3 Klein recommended)"), "1024x1024")
        self.assertFalse(size.allow_custom_value)

    def test_only_this_typed_field_accepts_bounded_stale_dimensions(self):
        with gr.Blocks(analytics_enabled=False):
            size = ResolutionDropdown(["2048x2048 (QwenImage recommended)", "Custom..."])
            ordinary = gr.Dropdown(["2048x2048"])
        self.assertEqual(size.preprocess("1024x1024"), "1024x1024")
        with self.assertRaises(gr.Error):
            ordinary.preprocess("1024x1024")
        for invalid in ("wrong", "0x1024", "32769x1", "9999x9999"):
            with self.subTest(value=invalid), self.assertRaises(gr.Error):
                size.preprocess(invalid)

    def test_known_recommendation_and_uncached_compatible_family_have_honest_defaults(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(ResolutionSettings, "RESOLUTIONS_FILE", Path(temporary) / "sizes.json"),
        ):
            settings = ResolutionSettings()
            _, selected, automatic = settings.model_selection("SDXL", None, True)
            self.assertIn("recommended", selected)
            self.assertEqual(settings.aspect_ratios[selected], (1024, 1024))
            self.assertTrue(automatic)
            performance = SimpleNamespace(
                model_recommendations=SimpleNamespace(
                    cached_resolution=lambda _: (None, None), metadata=lambda _: {}
                ),
                family_preset=lambda *_: None,
            )
            choices, selected, automatic = settings.model_selection(
                "Flux3", selected, True, "uncatalogued-alternate-precision.safetensors", performance
            )
            self.assertEqual(selected, "1024x1024 (Generic fallback)")
            self.assertIn(selected, choices)
            self.assertTrue(automatic)
            self.assertEqual(settings.get_aspect_ratios("1024x1024"), (1024, 1024))

    def test_renamed_alternate_precision_native_image_is_not_subject_to_a_hash_allowlist(self):
        supported = type("SDXL", (), {"unet_config": {}})()
        detection = SimpleNamespace(
            unet_prefix_from_state_dict=lambda _: "",
            model_config_from_unet=lambda *args, **kwargs: supported,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                "sys.modules",
                {
                    "comfy": SimpleNamespace(model_detection=detection),
                    "comfy.model_detection": detection,
                },
            ),
        ):
            for dtype in ("F16", "F8_E4M3"):
                path = Path(temporary) / f"unregistered-renamed-{dtype}.safetensors"
                header = json.dumps(
                    {
                        "conv_in.weight": {
                            "shape": [320, 4, 3, 3],
                            "dtype": dtype,
                            "data_offsets": [0, 0],
                        }
                    }
                ).encode()
                path.write_bytes(struct.pack("<Q", len(header)) + header)
                require_image_checkpoint(path, "Unknown")
        with self.assertRaisesRegex(ValueError, "still images and local Chat"):
            require_image_operation({"model": "video/default"})
