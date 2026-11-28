"""Explicit Library submissions freeze the selected source and preserve Create intent."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
from PIL import Image

from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.gallery.editing_workflow import build_library_request, working_source
from ruinedfooocus.features.generation.ui_request import build_ui_request


class LibraryRequestTests(unittest.TestCase):
    def test_plain_library_requests_do_not_reuse_create_patch_or_control_recipe_uploads(self):
        self.controls.update(
            model_patch_recipe="uso-style",
            model_patch_image=Image.new("RGB", (8, 8)),
            model_patch_inpaint_image=Image.new("RGB", (8, 8)),
            model_patch_mask=Image.new("L", (8, 8)),
            control_recipe="alimama-inpaint",
            control_inpaint_image=Image.new("RGB", (8, 8)),
            control_inpaint_mask=Image.new("L", (8, 8)),
        )
        for action in ("generate", "edit"):
            request = self.build(action)
            self.assertEqual(request["model_patch_recipe"], "none")
            self.assertEqual(request["control_recipe"], "ordinary")
            for key in (
                "model_patch_image",
                "model_patch_inpaint_image",
                "model_patch_mask",
                "control_inpaint_image",
                "control_inpaint_mask",
            ):
                self.assertIsNone(request[key])
            self.assertEqual(request["loras"], self.controls["loras"])
        self.assertEqual(self.controls["model_patch_recipe"], "uso-style")
        self.assertEqual(self.controls["control_recipe"], "alimama-inpaint")

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.path = self.root / "selected.png"
        Image.new("RGBA", (73, 121), (20, 40, 60, 127)).save(self.path)
        self.record = source_record(self.path)
        self.controls = dict(
            prompt="Create draft",
            negative="keep negative",
            base_model_name="selected-model",
            custom_width=1216,
            custom_height=832,
            size_mode="target",
            image_resize_mode="preserve",
            cn_strength=0.37,
            automatic_upscale="Off",
            image_component_overrides={"vae": "explicit.safetensors"},
            loras=[("thumbnail", "0.75 - detail.safetensors")],
            reference_images=[Image.new("RGB", (3, 4), "red")],
            _native_main_source={"path": "another-session.exr", "display": "same.png"},
        )
        self.resets = dict(
            input_image=None,
            reference_images=[],
            reference_adapter_image=None,
            inpaint_view=None,
            inpaint_toggle=False,
            _native_main_source=None,
            _generation_artifacts=[],
            _generation_originals=[],
            native_edit_recipe="auto",
        )
        self.features = dict(native_edit=False, inpaint=True, qwen_local=False)

    def build(self, action="edit", **kwargs):
        return build_library_request(
            self.controls,
            self.record,
            prompt="Changed instruction",
            negative="literal negative",
            checkpoint="selected-model",
            action=action,
            resets=self.resets,
            features=self.features,
            presets={"Img2Img": {"type": "img2img"}, "Remove": {"type": "rembg"}},
            custom="Custom...",
            **kwargs,
        )

    def test_selected_source_alpha_denoise_geometry_components_and_loras_are_captured(self):
        original = self.path.read_bytes()
        request = self.build(strength=0.37)
        self.assertEqual(request["prompt"], "Changed instruction")
        self.assertEqual(request["negative"], "literal negative")
        self.assertEqual(request["input_image"].getpixel((0, 0)), (20, 40, 60, 127))
        self.assertEqual(request["input_image"].size, (73, 121))
        for key in (
            "custom_width",
            "custom_height",
            "size_mode",
            "image_resize_mode",
            "cn_strength",
            "image_component_overrides",
            "loras",
        ):
            self.assertEqual(request[key], self.controls[key])
        self.assertIsNone(request["_native_main_source"])
        self.assertEqual(request["_generation_artifacts"], [])
        self.assertEqual(request["reference_images"], [])
        self.assertEqual(request["image_source"]["filename"], self.path.name)
        self.assertNotIn(str(self.root), str(request["image_source"]))
        self.controls["loras"].clear()
        self.controls["image_component_overrides"]["vae"] = "new"
        self.path.write_bytes(b"changed after capture")
        self.assertEqual(request["input_image"].getpixel((0, 0)), (20, 40, 60, 127))
        self.assertEqual(request["image_component_overrides"]["vae"], "explicit.safetensors")
        self.assertEqual(len(request["loras"]), 1)
        self.assertNotEqual(original, self.path.read_bytes())

    def test_generate_clears_old_source_and_another_session_cannot_supply_it(self):
        generated = self.build("generate")
        self.assertIsNone(generated["input_image"])
        self.assertIsNone(generated["_native_main_source"])
        self.assertEqual(generated["reference_images"], [])
        second = self.root / "other.png"
        Image.new("RGBA", (29, 31), (90, 80, 70, 60)).save(second)
        other_controls = copy.deepcopy(self.controls)
        self.controls["main_view"] = "old-original.exr"
        second_request = build_library_request(
            other_controls,
            source_record(second),
            prompt="session B",
            negative="",
            checkpoint="selected-model",
            action="edit",
            resets=self.resets,
            features=self.features,
            presets={"Img2Img": {"type": "img2img"}},
            custom="Custom...",
        )
        self.assertEqual(second_request["input_image"].size, (29, 31))
        self.assertEqual(second_request["input_image"].getpixel((0, 0)), (90, 80, 70, 60))

    def test_missing_changed_source_and_absent_or_empty_mask_fail_before_admission(self):
        for mask in (None, {"layers": [Image.new("RGBA", (73, 121))]}):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "Paint a mask"):
                self.build("inpaint", mask=mask)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "changed or was removed"):
            self.build()

    def test_inpaint_copies_only_explicit_current_brush_and_capability_is_generic(self):
        layer = Image.new("RGBA", (73, 121))
        layer.putpixel((2, 3), (255, 255, 255, 255))
        mask = {"layers": [layer]}
        request = self.build("inpaint", mask=mask)
        self.assertTrue(request["inpaint_toggle"])
        layer.paste((0, 0, 0, 0), (0, 0, 73, 121))
        self.assertEqual(
            request["inpaint_view"]["layers"][0].getchannel("A").getbbox(), (2, 3, 3, 4)
        )
        self.features["inpaint"] = False
        with self.assertRaisesRegex(ValueError, "supported inpainting"):
            self.build("inpaint", mask=mask)

    def test_precision_source_uses_original_and_reports_working_conversion(self):
        values = SimpleNamespace(
            pixels=np.zeros((121, 73, 4), dtype=np.uint16), color_space="linear"
        )
        converted = SimpleNamespace(
            clamp=lambda *_: SimpleNamespace(
                numpy=lambda: np.full((121, 73, 4), 0.5, dtype=np.float32)
            )
        )
        module = "ruinedfooocus.features.image_processing.native_image_formats"
        with (
            patch(module + ".needs_preview", return_value=True),
            patch(module + ".read_image", return_value=values) as read,
            patch(module + ".convert_color", return_value=converted) as convert,
        ):
            image, info = working_source(self.record)
        read.assert_called_once_with(self.path)
        self.assertEqual(convert.call_args.args[1:], ("linear", "sRGB"))
        self.assertEqual(image.getpixel((0, 0)), (127, 127, 127, 127))
        self.assertEqual(info["original_dtype"], "uint16")
        self.assertIn("8-bit working image", info["conversion"])

    def test_shared_ui_builder_selects_input_upscale_without_mutating_controls(self):
        image = Image.new("RGBA", (13, 17))
        values = dict(
            automatic_upscale="4× UltraSharp",
            automatic_upscale_target="Input image only",
            upscale_upload=image,
            image_number=1,
        )
        module = "ruinedfooocus.features.image_editing.controlnet"
        with patch.dict("sys.modules", {module: SimpleNamespace(get_settings=lambda _: {})}):
            result = build_ui_request(values, models=None, settings={})
        self.assertIs(result["input_image"], image)
        self.assertNotIn("input_image", values)
        self.assertFalse(result["generate_forever"])

    def test_saved_rendered_prompt_is_literal_unless_current_prompt_processing_is_explicit(self):
        self.controls.update(
            style_selection=["selected-style"],
            auto_negative=True,
            obp_assume_direct_control=True,
            obp_image_prompt=True,
        )
        literal = self.build()
        self.assertEqual(literal["style_selection"], [])
        self.assertFalse(literal["auto_negative"])
        processed = self.build(prompt_processing=True)
        self.assertEqual(processed["style_selection"], ["selected-style"])
        self.assertTrue(processed["auto_negative"])
        self.assertFalse(processed["obp_assume_direct_control"])
        self.assertFalse(processed["obp_image_prompt"])
        self.assertTrue(processed["_metadata_prompt"])
        self.assertTrue(processed["_apply_prompt_processing"])
        from test_image_metadata_boundary import function

        styled = Mock(return_value=("styled instruction", "styled negative"))
        process_prompt = function(
            "ruinedfooocus/features/prompting/prompt_processing.py",
            "process_prompt",
            dict(
                use_random_prompt=lambda *_: False,
                load_styles=lambda: {"selected-style": {}},
                apply_style=styled,
            ),
        )
        processed["auto_negative"] = False
        self.assertEqual(
            process_prompt(processed["style_selection"], "instruction", "negative", processed),
            ("styled instruction", "styled negative"),
        )
        self.assertEqual(styled.call_args.args[0], ["selected-style"])

    def test_command_like_and_metadata_like_library_instructions_remain_literal(self):
        from test_image_metadata_boundary import function, replay, route
        from ruinedfooocus.features.generation.request_metadata import metadata_intent

        for text in (
            "#!clean cache",
            "search: original",
            '{"prompt":"nested","base_model_name":"wrong"}',
        ):
            with self.subTest(text=text):
                data = self.build()
                data["prompt"] = text
                intended, imported = metadata_intent(data)
                self.assertIsNone(imported)
                self.assertEqual(intended["base_model_name"], "selected-model")
                self.assertEqual(replay(text, _metadata_prompt=True)["prompt"], text)
                _, factories = route(data)
                factories["image_pipeline"].pipeline.assert_called_once()
                for name in ("hashbang_pipeline", "search_pipeline", "template_pipeline"):
                    factories[name].pipeline.assert_not_called()
                process_prompt = function(
                    "ruinedfooocus/features/prompting/prompt_processing.py", "process_prompt", {}
                )
                self.assertEqual(process_prompt([], text, "negative", data), (text, "negative"))


if __name__ == "__main__":
    unittest.main()
