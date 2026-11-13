"""New image labels are extracted without services, and native controls keep wire values."""

import ast
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr

import ruinedfooocus.app
from ruinedfooocus.core.translation_manager import TranslationManager, native_key
from ruinedfooocus.features.image_processing import ui_image_adjustments
from ruinedfooocus.features.image_processing import ui_utility_outputs
from ruinedfooocus.features.image_processing.ui_color_operations import HDR_MIRROR
from tools import check_translations


class ImageToolsLocalizationTests(unittest.TestCase):
    def test_scoped_helper_extracts_displayed_schema_without_importing_service_code(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        folder = root / "ruinedfooocus/features/image_processing"
        folder.mkdir(parents=True)
        (folder / "image_mask_tools.py").write_text(
            'OPERATIONS = ("threshold", "invert")\n'
            'LABELS = {"image-mask": "Edit mask"}\n'
            'raise RuntimeError("Service must never execute during extraction")\n',
            encoding="utf-8",
        )
        (folder / "image_color_tools.py").write_text(
            'OPERATIONS = ("text-overlay", "sdr-to-hdr")\n'
            'MORPHOLOGY_OPERATIONS = ("erode", "dilate")\n'
            'TEXT_POSITIONS = ("top", "bottom")\n'
            'TEXT_ALIGNMENTS = ("left", "center", "right")\n'
            'raise RuntimeError("Color service must not execute")\n',
            encoding="utf-8",
        )
        source = (
            'field("mask_operation", "Mask operation", choices=mask.OPERATIONS, info="Help text")\n'
            'field("color_method", "Color method", choices=("histogram", "soft_light"))\n'
            'field("operation", "Color operation", choices=color.OPERATIONS)\n'
            'field("crop", "Crop mode", choices=("crop-to-mask",))\n'
            'field("morph", "Morphology operation", choices=color.MORPHOLOGY_OPERATIONS)\n'
            'for name in ("left", "right"):\n'
            '    field("wire_" + name, "Padding " + name)\n'
            'prompt = "Do not translate uploaded prose"\n'
        )
        tree = ast.parse(source)
        with patch.object(check_translations, "ROOT", root):
            constants, labels = check_translations.adjustment_schema()
            extracted = {
                node.value
                for node in check_translations.static_nodes(tree, adjustment_constants=constants)
            }
        self.assertEqual(
            extracted,
            {
                "Mask operation",
                "Help text",
                "Threshold",
                "Invert",
                "Color method",
                "Histogram (256-bin)",
                "Soft light",
                "Padding left",
                "Padding right",
                "Color operation",
                "Text overlay",
                "SDR to HDR",
                "Crop mode",
                "Crop to selection",
                "Morphology operation",
                "Erode",
                "Dilate",
            },
        )
        self.assertEqual([node.value for _, node in labels], ["Edit mask"])
        self.assertEqual(list(check_translations.static_nodes(tree)), [])
        self.assertNotIn("mask_operation", extracted)
        self.assertNotIn("soft_light", extracted)

    def test_actual_adjustment_controls_have_catalogued_labels_and_literal_wire_choices(self):
        manager = TranslationManager(legacy_root=Path("no-custom-image-tools-catalogues"))
        controls = {}
        shared = NS(
            settings=NS(default_settings={"color_text": "猫 {literal}\nSecond line"}),
            add_ctrl=lambda name, control: controls.update({name: control}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False):
                selection = gr.Dropdown(choices=["Custom..."], value="Custom...")
                kind = gr.Dropdown(choices=["Image-transform"], value="Image-transform")
                ui_image_adjustments.create_ui(
                    selection, kind, lambda choice, custom: custom.lower()
                )
                controls.update(ui_utility_outputs.moge_controls({}))
                controls.update(ui_utility_outputs.face_controls({}))
        for field, control in controls.items():
            with self.subTest(field=field):
                self.assertTrue(hasattr(control.label, "key"))
                key = control.label.key.removeprefix("ruinedfooocus.")
                for locale in manager.manifest["locales"]:
                    self.assertTrue(key in manager.translations[locale], f"{locale}: {key}")
                for label, wire in getattr(control, "choices", ()):
                    if field in (HDR_MIRROR, "moge_normal_convention"):
                        self.assertEqual(
                            label,
                            {"opengl": "OpenGL", "directx": "DirectX"}.get(wire, wire),
                        )
                        continue
                    self.assertTrue(hasattr(label, "key"))
                    self.assertIsInstance(wire, str)
                    choice_key = label.key.removeprefix("ruinedfooocus.")
                    for locale in manager.manifest["locales"]:
                        self.assertTrue(
                            choice_key in manager.translations[locale], f"{locale}: {choice_key}"
                        )
        self.assertEqual(
            [wire for _, wire in controls["transform_operation"].choices],
            ["resize", "crop", "pad", "rotate", "flip", "crop-to-mask"],
        )
        self.assertEqual(
            [wire for _, wire in controls["color_transfer_method"].choices],
            ["reinhard_lab", "mkl_lab", "histogram"],
        )
        self.assertEqual(controls["transform_padding_color"].value, "#000000")
        self.assertEqual(controls["color_operation"].value, "blend")
        self.assertEqual(
            [wire for _, wire in controls["color_operation"].choices],
            ["blend", "transfer", "quantize", "text-overlay", "morphology", "sdr-to-hdr"],
        )
        self.assertEqual(controls["color_text"].value, "猫 {literal}\nSecond line")
        self.assertEqual(
            controls[HDR_MIRROR].choices,
            [("16-bit float", "16-bit float"), ("32-bit float", "32-bit float")],
        )
        self.assertEqual(
            controls["moge_normal_convention"].choices,
            [("OpenGL", "opengl"), ("DirectX", "directx")],
        )
        self.assertEqual(manager.language, "en")

    def test_new_image_and_model_labels_have_existing_locale_drafts(self):
        manager = TranslationManager(legacy_root=Path("no-custom-image-tools-catalogues"))
        self.assertFalse(manager.errors)
        for text in (
            "Adjust image",
            "Use result as selection mask",
            "Transform image",
            "Edit mask",
            "Finish color",
            "Histogram (256-bin)",
            "Boogu Edit: up to sixteen references",
            "Krea2 style reference: up to three images",
            "Krea2 style strength",
            "Bernini-R image edit · two stages",
            "High-to-low switch step",
            "Bernini reference long edge",
        ):
            key = manager.aliases[text]
            self.assertEqual(manager.translations["en"][key], text)
            self.assertTrue(native_key(key).startswith("ruinedfooocus."))
            for locale, metadata in manager.manifest["locales"].items():
                with self.subTest(locale=locale, text=text):
                    self.assertTrue(key in manager.translations[locale], f"{locale}: {key}")
                    if locale != "en":
                        self.assertEqual(metadata["review"], "draft")
        self.assertEqual(manager.language, "en")


if __name__ == "__main__":
    unittest.main()
