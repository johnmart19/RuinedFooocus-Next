from ruinedfooocus.core.paths import asset_path, resource_path

"""Image UI regression checks; no server, model loading or generation."""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import gradio as gradio
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.features.image_editing.inpaint_utils import editor_payload, source_and_mask
from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    capabilities,
    resolve_guidance,
    restore_guidance_metadata,
)
from ruinedfooocus.features.image_editing.ui_guidance import preset_control_state

ROOT = Path(__file__).resolve().parents[1]


def function_body(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class ImageUIAuditTests(unittest.TestCase):
    def setUp(self):
        stub = patch.dict(
            "sys.modules",
            {"ruinedfooocus.app.arguments": SimpleNamespace(args=SimpleNamespace(offline=True))},
        )
        stub.start()
        self.addCleanup(stub.stop)

    def test_inpaint_toggle_keeps_full_feedback_on_checkbox_and_native_editor_data(self):
        path = ROOT / "ruinedfooocus/features/image_editing/ui.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definition = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "inpaint_checked"
        )
        with gradio.Blocks(analytics_enabled=False) as app:
            toggle, image, editor, model = (
                gradio.Checkbox(),
                gradio.Image(),
                gradio.ImageEditor(),
                gradio.Textbox(),
            )
            namespace = dict(
                gr=gradio,
                inpaint_toggle=toggle,
                main_view=image,
                inpaint_view=editor,
                base_model=model,
                editor_payload=editor_payload,
                checkpoint_features=lambda _: {"qwen_local": False},
            )
            registration = ast.Expression(definition.decorator_list[0])
            event = eval(compile(registration, str(path), "eval"), namespace)
            callback = function_body(str(path.relative_to(ROOT)), "inpaint_checked", namespace)
            event(callback)
        config = app.config["dependencies"][-1]
        self.assertEqual(config["show_progress"], "full")
        self.assertEqual(config["show_progress_on"], [toggle._id])
        self.assertEqual(config["outputs"], [editor._id, toggle._id])
        with patch.object(gradio, "Warning") as warning:
            for enabled in (False, True):
                result = asyncio.run(
                    app.process_api(config["id"], [enabled, None, "model"], state=SessionState(app))
                )
                self.assertTrue(result["data"][0]["interactive"])
                if enabled:
                    self.assertFalse(result["data"][1]["value"])
            warning.assert_called_once()

    def test_legacy_preset_does_not_inherit_another_checkpoints_shift(self):
        current = {key: spec["default"] for key, spec in FIELD_SCHEMA.items()}
        current.update(sampling_shift=3, cfg_norm_strength=2)
        transition = preset_control_state(
            current, {}, {"software": "RuinedFooocus"}, 7, "SDXL", "sdxl", "qwen", None
        )
        self.assertIsNone(transition["values"]["sampling_shift"])
        self.assertIsNone(transition["values"]["cfg_norm_strength"])
        self.assertEqual(resolve_guidance(transition["values"], {}, "SDXL")["sampling_shift"], None)
        cleared = preset_control_state(
            transition["values"], {}, {}, 7, "QwenImage", "qwen", "qwen", transition["state"]
        )
        self.assertEqual(cleared["values"]["sampling_shift"], 3)
        recorded = preset_control_state(
            current, {}, {"sampling_shift": 5}, 7, "MOSSImage", "moss", "qwen", None
        )
        self.assertEqual(recorded["values"]["sampling_shift"], 5)

    def test_inpaint_source_normalizes_orientation_and_color_without_upscaling(self):
        image = Image.new("RGBA", (30, 20))
        image.getexif()[274] = 6
        payload = editor_payload(image)
        self.assertEqual(payload["background"].size, (20, 30))
        self.assertEqual(payload["background"].mode, "RGB")
        self.assertEqual(image.size, (30, 20))
        self.assertEqual(payload["layers"][0].getchannel("A").getbbox(), None)
        tall = editor_payload(Image.new("RGB", (1, 1200)))
        self.assertEqual(tall["background"].size, (1, 600))
        mask = np.zeros((6, 4, 4), dtype=np.uint8)
        mask[:3, :, 3] = 255
        source, alpha = source_and_mask(image, {"layers": [mask]})
        self.assertEqual(source.mode, "RGB")
        self.assertEqual(alpha.size, source.size)
        self.assertEqual(alpha.mode, "L")
        self.assertGreater(alpha.getextrema()[1], 0)

    def test_missing_or_invalid_inpaint_mask_is_rejected_before_model_work(self):
        image = Image.new("RGB", (8, 8))
        for editor in (
            None,
            {},
            {"layers": []},
            {"layers": [np.zeros((8, 8, 3), dtype=np.uint8)]},
            {"layers": [np.zeros((8, 8, 4), dtype=np.uint8)]},
        ):
            with self.assertRaises(ValueError):
                source_and_mask(image, editor)
        gr = SimpleNamespace(update=lambda **values: values, Warning=Mock())
        callback = function_body(
            "ruinedfooocus/features/image_editing/ui.py",
            "inpaint_checked",
            dict(
                gr=gr,
                editor_payload=editor_payload,
                main_view="main",
                inpaint_view="editor",
                inpaint_toggle="toggle",
            ),
        )
        result = callback(True, None)
        self.assertEqual(result["toggle"], {"value": False})
        self.assertNotIn("main", result)
        self.assertNotIn(
            "visible", result["editor"]
        )  # Visibility follows the reset toggle in one shared callback.
        self.assertTrue(result["editor"]["interactive"])
        self.assertTrue(callback(False, None)["editor"]["interactive"])
        gr.Warning.assert_called_once()

    def metadata_callback(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        mapping = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "METADATA_CONTROLS"
                for target in node.targets
            )
        )
        namespace = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            json=json,
            gr=SimpleNamespace(update=lambda **values: values, Error=ValueError, Info=Mock()),
            get_ctrl=lambda key: key,
            METADATA_CONTROLS=ast.literal_eval(mapping),
            FIELD_SCHEMA=FIELD_SCHEMA,
            capabilities=capabilities,
            resolve_guidance=resolve_guidance,
            restore_guidance_metadata=restore_guidance_metadata,
        )
        return function_body("ruinedfooocus/features/gallery/ui.py", "apply_metadata", namespace)

    def test_browser_metadata_restores_guidance_without_private_references(self):
        callback = self.metadata_callback()
        metadata = dict(
            prompt="fox",
            image_guidance=dict(
                moss_image_edit_format="ai-toolkit",
                moss_image_edit_strength=2,
                moss_image_slg=True,
                sampling_shift=None,
                effective_sampling_shift=5,
                cfg_norm_strength=None,
                reference_images=["private.png"],
            ),
        )
        with (
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                return_value="MOSSImage",
            ),
            patch(
                "ruinedfooocus.features.generation.automatic_upscale.choices", return_value=["Off"]
            ),
        ):
            updates = callback(metadata, "moss", "")
        self.assertEqual(updates["sampling_shift"], {"value": 5})
        self.assertEqual(updates["moss_image_edit_format"]["value"], "ai-toolkit")
        self.assertEqual(updates["cfg_norm_strength"], {"value": None})
        self.assertNotIn("reference_images", updates)
        with (
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                return_value="Krea2",
            ),
            patch(
                "ruinedfooocus.features.generation.automatic_upscale.choices", return_value=["Off"]
            ),
        ):
            updates = callback(metadata, "krea", "")
        self.assertIsNone(updates["sampling_shift"]["value"])
        self.assertEqual(updates["moss_image_edit_format"]["value"], "Disabled")

    def test_browser_metadata_reports_invalid_input_and_active_preset(self):
        callback = self.metadata_callback()
        with self.assertRaisesRegex(ValueError, "Clear the active preset"):
            callback({"prompt": "fox"}, "model", "preset.png")
        with self.assertRaisesRegex(ValueError, "valid generation metadata"):
            callback("invalid JSON", "model", "")


if __name__ == "__main__":
    unittest.main()
