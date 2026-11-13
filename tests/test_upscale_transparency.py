from ruinedfooocus.core.paths import asset_path, resource_path

"""Exercise real Gradio preprocessing configured by both app upload controls."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.data_classes import ImageData
import numpy as np
from PIL import Image, PngImagePlugin


ROOT = Path(__file__).resolve().parents[1]


class TransparentUploadTests(unittest.TestCase):
    def test_main_preview_preserves_uploaded_file_pixels_and_metadata(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "main_view" for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        )
        options = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in assignment.value.keywords
            if keyword.arg in ("type", "image_mode")
        }
        uploader = gr.Image(**options)
        rgba = Image.fromarray(
            np.array([[[255, 0, 0, 0], [0, 255, 0, 128], [0, 0, 255, 255]]], dtype=np.uint8)
        )
        images = [
            rgba,
            rgba.quantize(method=Image.Quantize.FASTOCTREE),
            rgba.convert("LA"),
            Image.fromarray(np.array([[0, 1024, 65535]], dtype=np.uint16)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for image in images:
                with self.subTest(mode=image.mode):
                    path = Path(directory) / "upload.png"
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("parameters", "original prompt and settings")
                    image.save(path, pnginfo=metadata)
                    original_bytes = path.read_bytes()
                    uploaded = uploader.preprocess(ImageData(path=str(path), orig_name=path.name))
                    self.assertEqual(Path(uploaded).read_bytes(), original_bytes)
                    with Image.open(uploaded) as result, Image.open(path) as original:
                        self.assertEqual(result.mode, original.mode)
                        np.testing.assert_array_equal(np.array(result), np.array(original))
                        self.assertEqual(result.info, original.info)

    def test_standalone_conversion_preserves_alpha_stored_only_in_a_palette(self):
        import torch

        rgba = Image.fromarray(
            np.array(
                [
                    [[255, 0, 0, 0], [0, 255, 0, 64]],
                    [[0, 0, 255, 128], [255, 255, 255, 255]],
                ],
                dtype=np.uint8,
            )
        )
        palette = rgba.quantize(method=Image.Quantize.FASTOCTREE)
        self.assertEqual(palette.palette.mode, "RGBA")
        self.assertNotIn("transparency", palette.info)
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/image_processing/upscale_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            torch=torch,
            np=np,
            worker=Mock(),
            InterruptibleUpscaler=lambda model, data: model,
            controlnet=SimpleNamespace(get_settings=lambda data: {"upscaler": "test"}),
        )
        exec(
            compile(ast.Module(body=[process], type_ignores=[]), "upscale_conversion", "exec"),
            scope,
        )
        owner = SimpleNamespace(load_upscaler_model=Mock(return_value="native"))
        for image, mode in (
            (rgba, "RGBA"),
            (palette, "RGBA"),
            (rgba.convert("LA"), "RGBA"),
            (rgba.convert("RGBa"), "RGBA"),
            (rgba.convert("RGB"), "RGB"),
        ):
            with self.subTest(mode=image.mode):
                expected = np.array(image.convert(mode))

                def upscale(model, pixels):
                    self.assertEqual(pixels.shape[-1], expected.shape[-1])
                    np.testing.assert_allclose(pixels[0].numpy(), expected / 255, rtol=0, atol=1e-7)
                    return (pixels.repeat_interleave(2, 1).repeat_interleave(2, 2),)

                native = SimpleNamespace(ImageUpscaleWithModel=SimpleNamespace(execute=upscale))
                with patch.dict("sys.modules", {"comfy_extras.nodes_upscale_model": native}):
                    output = scope["process"](owner, {"input_image": image, "task_id": 1})[0]
                np.testing.assert_array_equal(output, expected.repeat(2, 0).repeat(2, 1))

    def test_both_uploaders_preserve_rgba_palette_and_grayscale_transparency(self):
        rgba = Image.fromarray(
            np.array(
                [
                    [[255, 0, 0, 0], [0, 255, 0, 64]],
                    [[0, 0, 255, 128], [255, 255, 255, 255]],
                ],
                dtype=np.uint8,
            )
        )
        images = [rgba, rgba.quantize(method=Image.Quantize.FASTOCTREE), rgba.convert("LA")]
        for source, name in [
            ("ruinedfooocus/ui/app.py", "upscale_upload"),
            ("ruinedfooocus/features/image_editing/ui.py", "input_image"),
        ]:
            tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
            assignment = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name for target in node.targets
                )
                and isinstance(node.value, ast.Call)
            )
            options = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in assignment.value.keywords
                if keyword.arg in ("type", "image_mode")
            }
            # Use the actual app options, including Gradio's default if removed.
            uploader = gr.Image(**options)
            with tempfile.TemporaryDirectory() as directory:
                for image in images:
                    with self.subTest(uploader=name, mode=image.mode):
                        path = Path(directory) / "transparent.png"
                        image.save(path)
                        uploaded = uploader.preprocess(
                            ImageData(path=str(path), orig_name=path.name)
                        )
                        self.assertEqual(uploaded.mode, "RGBA")
                        np.testing.assert_array_equal(
                            np.array(uploaded), np.array(image.convert("RGBA"))
                        )


if __name__ == "__main__":
    unittest.main()
