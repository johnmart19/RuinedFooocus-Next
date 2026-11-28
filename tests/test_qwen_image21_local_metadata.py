"""Save and replay applied Qwen paint modes without restoring image inputs."""

import json
import unittest

from ruinedfooocus.features.gallery.image_metadata import render_values
from ruinedfooocus.features.generation.image_preparation import image_control_data
from ruinedfooocus.features.image_editing.qwen_image21_local_edit import replay_values
from test_image_metadata_boundary import gallery, replay
from test_image_upscale_metadata import saved_parameters


class QwenImage21LocalMetadataTests(unittest.TestCase):
    def test_png_saves_only_the_applied_local_mode_and_keeps_source_fields_private(self):
        for mode in ("paint", "mask"):
            with self.subTest(mode=mode):
                data, size = saved_parameters(
                    (385, 321),
                    scale=1,
                    metadata={
                        "qwen_local_edit_mode": "mask" if mode == "paint" else "paint",
                        "image_geometry": {
                            "width": 385,
                            "height": 321,
                            "qwen_local_edit_mode": mode,
                        },
                        "main_view": "private-source.png",
                        "input_image": "private-reference.png",
                        "inpaint_view": {"layers": ["private-mask.png"]},
                        "reference_images": ["private-additional.png"],
                    },
                )
                self.assertEqual(size, (385, 321))
                self.assertEqual(data["qwen_local_edit_mode"], mode)
                self.assertNotIn("private-", json.dumps(data))
                self.assertNotIn("inpaint_toggle", data)
        ordinary, _ = saved_parameters(
            (128, 128), scale=1, metadata={"qwen_local_edit_mode": "mask"}
        )
        self.assertNotIn("qwen_local_edit_mode", ordinary)

    def test_flat_and_legacy_geometry_mode_replay_before_capture_without_source_injection(self):
        for recipe in (
            {"qwen_local_edit_mode": "mask"},
            {"image_geometry": {"qwen_local_edit_mode": "mask"}},
        ):
            with self.subTest(recipe=recipe):
                metadata = {
                    "prompt": "Make the red area blue.",
                    **recipe,
                    "input_image": "private-source.png",
                    "inpaint_view": {"layers": ["private-mask.png"]},
                    "reference_images": ["private-reference.png"],
                }
                values = render_values(metadata)
                self.assertEqual(values["qwen_local_edit_mode"], "mask")
                self.assertFalse(
                    {"input_image", "inpaint_view", "reference_images"} & values.keys()
                )
                source, editor = object(), object()
                captured = image_control_data(
                    dict(prompt=json.dumps(metadata), main_view=source, inpaint_view=editor)
                )
                self.assertEqual(captured["qwen_local_edit_mode"], "mask")
                self.assertIs(captured["main_view"], source)
                self.assertIs(captured["inpaint_view"], editor)
                self.assertNotIn("input_image", captured)
                self.assertNotIn("reference_images", captured)
                imported = replay(json.dumps(metadata))
                self.assertEqual(imported["qwen_local_edit_mode"], "mask")
                self.assertNotIn("inpaint_toggle", imported)

    def test_library_restores_scalar_choice_without_enabling_an_editor_or_references(self):
        for recipe in (
            {"qwen_local_edit_mode": "paint"},
            {"image_geometry": {"qwen_local_edit_mode": "paint"}},
        ):
            with self.subTest(recipe=recipe):
                updates = gallery(
                    dict(
                        prompt="Edit the red area.",
                        **recipe,
                        input_image="private-source.png",
                        inpaint_view={"layers": ["private-mask.png"]},
                        reference_images=["private-additional.png"],
                    )
                )
                self.assertEqual(updates["qwen_local_edit_mode"], {"value": "paint"})
                self.assertFalse(
                    {
                        "input_image",
                        "main_view",
                        "inpaint_view",
                        "reference_images",
                        "inpaint_toggle",
                    }
                    & updates.keys()
                )

    def test_invalid_mode_is_rejected_and_flat_mode_takes_precedence_over_legacy_geometry(self):
        self.assertEqual(
            replay_values(
                {
                    "qwen_local_edit_mode": "mask",
                    "image_geometry": {"qwen_local_edit_mode": "paint"},
                }
            ),
            {"qwen_local_edit_mode": "mask"},
        )
        for mode in (True, "invalid", {"input_image": "private.png"}):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Paint guide or Mask"):
                replay_values({"qwen_local_edit_mode": mode})
