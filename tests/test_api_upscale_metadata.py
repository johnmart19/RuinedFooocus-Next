"""Standalone upscale keeps uploaded PNG setup text and transparent pixels."""

import base64
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from ruinedfooocus.app import api_runtime
import ruinedfooocus.app


class ApiUpscaleMetadataTests(unittest.TestCase):
    def test_native_result_retains_source_setup_text_profile_and_alpha(self):
        image = Image.new("RGBA", (2, 3), (30, 70, 160, 64))
        info = PngInfo()
        parameters = json.dumps(
            {"Prompt": "Preserve exact words", "seed": 17, "width": 2, "height": 3}
        )
        info.add_text("parameters", parameters)
        info.add_text("Comment", "source comment")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", pnginfo=info, icc_profile=b"source profile")
        payload = {"input_image": base64.b64encode(buffer.getvalue()).decode()}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.png"
            worker = SimpleNamespace(check_interrupt=Mock())
            shared = SimpleNamespace(
                state={},
                models=SimpleNamespace(offline=True),
                path_manager=SimpleNamespace(
                    model_paths={"temp_outputs_path": folder},
                    DOWNLOADABLE_FILES={},
                    get_model_folders=lambda category: [Path(folder)],
                ),
            )
            (Path(folder) / "native2x").write_bytes(b"inert native mock weights")
            native = SimpleNamespace(
                pipeline=lambda: SimpleNamespace(
                    process=lambda data: [np.asarray(data["input_image"]).repeat(2, 0).repeat(2, 1)]
                )
            )
            with (
                # Patch the package attribute too: 'import ruinedfooocus.app.state as x' reads it.
                patch.object(ruinedfooocus.app, "state", shared, create=True),
                patch.dict(
                    "sys.modules",
                    {
                        "ruinedfooocus.app.state": shared,
                        "ruinedfooocus.app.async_worker": worker,
                        "ruinedfooocus.features.image_processing.upscale_pipeline": native,
                        "ruinedfooocus.core.util": SimpleNamespace(
                            generate_temp_filename=lambda **kwargs: str(output)
                        ),
                    },
                ),
                patch.object(
                    api_runtime,
                    "submit",
                    side_effect=lambda operation: (
                        operation.prepare(SimpleNamespace()),
                        operation(SimpleNamespace(), 1),
                    )[1],
                ),
            ):
                self.assertEqual(api_runtime.upscale(payload, {"name": "native2x"}), [str(output)])
            with Image.open(output) as result:
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (4, 6))
                self.assertEqual(result.getpixel((0, 0)), (30, 70, 160, 64))
                self.assertEqual(result.info["parameters"], parameters)
                self.assertEqual(result.info["Comment"], "source comment")
                self.assertEqual(result.info["icc_profile"], b"source profile")
