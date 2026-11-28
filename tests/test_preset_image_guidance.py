"""PNG preset guidance restoration without UI startup or generation."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    restore_guidance_metadata,
)
from ruinedfooocus.features.generation.ui_request import build_ui_request


class PresetImageGuidanceTests(unittest.TestCase):
    def test_grouped_fields_restore_and_flat_fields_take_precedence(self):
        group = dict(
            moss_image_edit_format="ai-toolkit",
            moss_image_edit_strength=2,
            moss_image_slg=True,
            sampling_shift=None,
            cfg_norm_strength=1,
            effective_sampling_shift=5,
            reference_images=["private/path.png"],
        )
        result = restore_guidance_metadata(
            {"image_guidance": group, "moss_image_edit_strength": 0, "cfg_norm_strength": None}
        )
        self.assertEqual(
            result,
            dict(
                moss_image_edit_format="ai-toolkit",
                moss_image_edit_strength=0,
                moss_image_slg=True,
                sampling_shift=5,
                cfg_norm_strength=None,
            ),
        )
        self.assertTrue(set(result).issubset(FIELD_SCHEMA))

    def test_valid_explicit_shift_wins_and_auto_promotes_only_valid_actual(self):
        for auto in (None, ""):
            self.assertEqual(
                restore_guidance_metadata(
                    {"sampling_shift": auto, "effective_sampling_shift": 3.1}
                ),
                {"sampling_shift": 3.1},
            )
        self.assertEqual(
            restore_guidance_metadata({"effective_sampling_shift": 5}), {"sampling_shift": 5}
        )
        self.assertEqual(
            restore_guidance_metadata({"sampling_shift": 4, "effective_sampling_shift": 5}),
            {"sampling_shift": 4},
        )
        for invalid in (True, 0, 101, float("inf"), float("nan"), "not a number"):
            self.assertEqual(restore_guidance_metadata({"effective_sampling_shift": invalid}), {})
        self.assertEqual(
            restore_guidance_metadata({"sampling_shift": 0, "effective_sampling_shift": 5}), {}
        )
        self.assertEqual(
            restore_guidance_metadata({"effective_sampling_shift": 1.15}, family="Flux"), {}
        )
        self.assertEqual(
            restore_guidance_metadata(
                {"effective_sampling_shift": 1.15, "sampling_shift_supported": False}
            ),
            {},
        )

    def test_old_or_malformed_metadata_does_not_add_untrusted_values(self):
        self.assertEqual(restore_guidance_metadata({"steps": 20}), {})
        self.assertEqual(restore_guidance_metadata({"image_guidance": "not a dictionary"}), {})
        self.assertEqual(
            restore_guidance_metadata(
                {
                    "reference_images": ["image.png"],
                    "moss_image_edit_format": "unknown",
                    "moss_image_slg": "false",
                }
            ),
            {},
        )

    def test_ui_request_builder_restores_guidance_only_when_sampling_preset_bit_enabled(self):
        recipe = dict(
            software="RuinedFooocus",
            steps=30,
            cfg=4,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=1,
            image_guidance={
                "moss_image_edit_format": "Moss-Image-Edit",
                "sampling_shift": None,
                "effective_sampling_shift": 5,
            },
        )
        import ruinedfooocus.features.image_editing as editing

        with tempfile.TemporaryDirectory() as folder:
            preset = Path(folder) / "preset.png"
            info = PngInfo()
            info.add_text("parameters", json.dumps(recipe))
            Image.new("RGB", (1, 1)).save(preset, pnginfo=info)
            values = dict(
                preset_selection=str(preset),
                base_model_name="moss",
                moss_image_edit_format="Disabled",
            )
            controlnet = SimpleNamespace(get_settings=lambda _: {})
            for mode in (0, 2):
                with self.subTest(preset_mode=mode):
                    family = Mock(return_value="MOSSImage")
                    with (
                        patch.dict(
                            "sys.modules",
                            {"ruinedfooocus.features.image_editing.controlnet": controlnet},
                        ),
                        patch.object(editing, "controlnet", controlnet, create=True),
                    ):
                        restored = build_ui_request(
                            values, models=None, settings={}, preset_mode=mode, family_for=family
                        )
                    self.assertEqual(values["moss_image_edit_format"], "Disabled")
                    self.assertNotIn("sampling_shift", values)
                    if mode:
                        family.assert_called_once_with("moss")
                        self.assertEqual(restored["moss_image_edit_format"], "Moss-Image-Edit")
                        self.assertEqual(restored["sampling_shift"], 5)
                        for key in ("steps", "cfg", "sampler_name", "scheduler", "clip_skip"):
                            self.assertEqual(restored[key], recipe[key])
                    else:
                        family.assert_not_called()
                        self.assertEqual(restored, values | {"generate_forever": False})


if __name__ == "__main__":
    unittest.main()
