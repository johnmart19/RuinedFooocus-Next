"""Native masked editing and canvas extension through the public ASGI routes."""

import base64
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.model_folders import model_folders
from ruinedfooocus.features.image_editing.inpaint_utils import prepare_inpaint, source_and_mask
from ruinedfooocus.features.prompting import local_model
from ruinedfooocus.integrations import openai_api


def image_bytes(image, format="PNG", **options):
    data = io.BytesIO()
    image.save(data, format=format, **options)
    return data.getvalue()


def inline(raw):
    return base64.b64encode(raw).decode()


class InpaintPaths:
    get_abspath = staticmethod(Path)

    def __init__(self, root):
        self.paths = {"path_checkpoints": [str(root)]}
        self.DOWNLOADABLE_FILES = {}

    def get_model_folders(self, category):
        return model_folders(self.paths, category, self.get_abspath)


class NativeInpaintApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        checkpoint = root / "fixture.safetensors"
        header = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
        checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        self.selected = dict(name=checkpoint.name, path=str(checkpoint), family="SDXL")
        self.settings = {"performance": "Own settings", "resolution": "default"}
        state = SimpleNamespace(
            state={},
            settings=SimpleNamespace(default_settings=self.settings),
            performance_settings=SimpleNamespace(
                CUSTOM_PERFORMANCE="Own settings",
                choices_for_model=lambda *_: ["Own settings"],
                selection_for_model=lambda *_: "Own settings",
                get_perf_options=lambda *_: dict(
                    custom_steps=8, cfg=4, sampler_name="euler", scheduler="simple", clip_skip=1
                ),
            ),
            models=SimpleNamespace(
                get_models_by_path=lambda *_, **__: {"baseModel": self.selected["family"]},
                get_model_base=lambda metadata: metadata["baseModel"],
                detect_checkpoint_base=Mock(return_value=None),
                get_file=lambda *_: checkpoint,
            ),
            path_manager=InpaintPaths(root),
            resolution_settings=SimpleNamespace(CUSTOM_RESOLUTION="Custom..."),
        )
        self.prepared = []

        def prepare(data, **_):
            self.prepared.append(data)
            return data

        worker = SimpleNamespace(
            prepare_image_data=prepare,
            check_interrupt=lambda *_: None,
            _process=lambda _: [root / "owned-output.png"],
        )
        (root / "owned-output.png").write_bytes(b"owned output fixture")

        def submit(operation):
            job = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event())
            operation.prepare(job)
            job.events.put(("result", operation(job, 1)))
            return job

        self.submit = Mock(side_effect=submit)
        app_module = sys.modules["ruinedfooocus.app"]
        samplers = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
        )
        for patcher in (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": state,
                    "ruinedfooocus.app.async_worker": worker,
                    "comfy.samplers": samplers,
                },
            ),
            patch.object(app_module, "state", state, create=True),
            patch.object(app_module, "async_worker", worker, create=True),
            patch.object(api_runtime, "submit", self.submit),
            patch.object(
                api_runtime, "image_defaults", {"inpaint_denoise": 0.11, "inpaint_mask_grow": 1}
            ),
            patch.object(local_model, "prepare_prompt_data"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        backend = SimpleNamespace(
            images=api_runtime.images,
            resolve_model=lambda *_: ("image/fixture.safetensors", self.selected),
        )
        api, self.tools = openai_api.create_apps(backend, "secret")
        self.client = TestClient(api, headers={"Authorization": "Bearer secret"})
        self.addCleanup(self.client.close)
        self.size = (91, 127)
        self.source = image_bytes(Image.new("RGB", self.size))
        layer = Image.new("RGBA", self.size, (0, 0, 0, 255))
        layer.putpixel((7, 12), (1, 2, 3, 0))
        layer.putpixel((8, 12), (1, 2, 3, 128))
        self.mask = image_bytes(layer)

    def json_edit(self, **fields):
        return self.client.post(
            "/images/generations",
            json={
                "prompt": "edit this image",
                "input_image": inline(self.source),
                **fields,
            },
        )

    def test_openapi_describes_bounded_multipart_image_aliases_and_edit_fields(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        document = response.json()
        body = document["paths"]["/images/edits"]["post"]["requestBody"]
        self.assertTrue(body["required"])
        self.assertIn(f"{openai_api.MAX_IMAGE_UPLOAD_BYTES:,}", body["description"])
        multipart = body["content"]["multipart/form-data"]
        schema = multipart["schema"]
        self.assertEqual(schema["oneOf"], [{"required": ["image"]}, {"required": ["image[]"]}])
        self.assertEqual(schema["not"], {"required": ["mask", "borders"]})
        for key in ("image", "image[]", "mask"):
            self.assertEqual(schema["properties"][key]["format"], "binary")
        self.assertEqual(multipart["encoding"]["mask"]["contentType"], "image/png")
        borders = schema["properties"]["borders"]
        self.assertEqual(
            (borders["type"], borders["contentMediaType"]), ("string", "application/json")
        )
        self.assertEqual(borders["contentSchema"], {"$ref": "#/components/schemas/OutpaintBorders"})
        border_schema = document["components"]["schemas"]["OutpaintBorders"]
        self.assertEqual(
            border_schema["properties"]["left"]["maximum"], openai_api.OUTPAINT_MAX_MARGIN
        )
        self.assertGreater(json.loads(borders["examples"][0])["left"], 0)
        self.submit.assert_not_called()

    def test_multipart_png_alpha_and_exif_reach_the_native_inpaint_request(self):
        exif = Image.Exif()
        exif[274] = 6
        source = image_bytes(Image.new("RGB", self.size[::-1]), format="JPEG", exif=exif)
        response = self.client.post(
            "/images/edits",
            files={"image[]": ("source.jpg", source), "mask": ("mask.png", self.mask)},
            data={"prompt": "change it", "inpaint_denoise": "0.67", "inpaint_mask_grow": "0"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = self.prepared[0]
        self.assertEqual((data["custom_width"], data["custom_height"]), self.size)
        self.assertEqual(data["inpaint_denoise"], 0.67)
        self.assertEqual(data["inpaint_mask_grow"], 0)
        self.assertTrue(data["inpaint_toggle"])
        self.assertEqual(data["image_engine"], "native")
        self.assertIsNone(data["input_image"])
        self.assertFalse(data["controlnet"])
        self.assertNotIn("mask", data)
        actual_source, actual_mask = source_and_mask(data["main_view"], data["inpaint_view"])
        self.assertEqual(actual_source.size, self.size)
        self.assertEqual([actual_mask.getpixel((x, 12)) for x in (7, 8, 9)], [255, 127, 0])

    def test_json_mask_keeps_explicit_canvas_size_and_request_defaults(self):
        mask = "data:image/png;base64," + inline(self.mask)
        response = self.json_edit(mask=mask, size="113x75")
        self.assertEqual(response.status_code, 200, response.text)
        with TestClient(self.tools, headers={"Authorization": "Bearer secret"}) as client:
            response = client.post(
                "/image_generation",
                json={
                    "prompt": "edit this image",
                    "input_image": inline(self.source),
                    "mask": mask,
                    "size": "113x75",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        for data in self.prepared:
            size = data["custom_width"], data["custom_height"]
            self.assertEqual(size, (113, 75))
            self.assertEqual(data["main_view"].size, self.size)
            self.assertEqual(data["image_resize_mode"], "pad")
            self.assertEqual(data["inpaint_denoise"], 1)
            self.assertEqual(data["inpaint_mask_grow"], 20)
            source, editable = source_and_mask(data["main_view"], data["inpaint_view"])
            pixels, _, frame = prepare_inpaint(source, editable, size, data["image_resize_mode"])
            self.assertEqual(pixels.shape[1:3], (80, 128))
            self.assertEqual((frame["width"], frame["height"]), size)

    def test_json_and_multipart_outpaint_take_the_exact_source_plus_borders(self):
        borders = {"left": 13, "right": 1, "bottom": 8, "feather": 0}
        self.selected["family"] = "FluxInpaint"
        for response in (
            self.json_edit(borders=borders, size="105x135", inpaint_denoise=0.42),
            self.client.post(
                "/images/edits",
                files={"image": ("source.png", self.source)},
                data={
                    "prompt": "extend",
                    "borders": json.dumps(borders),
                    "inpaint_denoise": "0.42",
                },
            ),
        ):
            self.assertEqual(response.status_code, 200, response.text)
        for data in self.prepared:
            self.assertEqual((data["custom_width"], data["custom_height"]), (105, 135))
            self.assertEqual(data["main_view"].size, self.size)
            self.assertEqual(data["inpaint_denoise"], 0.42)
            self.assertEqual(data["outpaint_left"], 13)
            self.assertEqual(data["outpaint_top"], 0)
            self.assertEqual(data["outpaint_feather"], 0)
            self.assertNotIn("inpaint_view", data)

    def test_invalid_inputs_never_queue_an_edit(self):
        mask = inline(self.mask)
        borders = {"right": 1}
        for fields, message in (
            ({"mask": mask, "input_image": None}, "source image is required"),
            ({"mask": mask, "borders": borders}, "either a mask or canvas borders"),
            ({"mask": mask, "controlnet": {"type": "img2img"}}, "own image operation"),
            ({"mask": mask, "reference_adapter": "redux"}, "own image operation"),
            ({"mask": mask, "reference_images": [mask]}, "Additional references require"),
            ({"mask": mask, "inpaint_denoise": 0}, "greater than 0"),
            ({"inpaint_mask_grow": 1}, "require a mask or canvas borders"),
            ({"mask": "file:///private.png"}, "valid inline PNG mask"),
            ({"mask": "https://example.invalid/mask.png"}, "valid inline PNG mask"),
            (
                {"mask": inline(image_bytes(Image.new("RGBA", self.size), format="TIFF"))},
                "valid inline PNG mask",
            ),
            ({"mask": inline(self.source)}, "alpha channel"),
            (
                {"mask": inline(image_bytes(Image.new("RGBA", self.size, (0, 0, 0, 255))))},
                "transparent pixel",
            ),
            (
                {"mask": inline(image_bytes(Image.new("RGBA", self.size[::-1])))},
                "match the oriented source",
            ),
            ({"mask": mask, "size": "128x128", "image_resize_mode": "preserve"}, "pad or crop"),
            ({"mask": mask, "size": "2049x128"}, "2048 pixels"),
            ({"borders": {}}, "positive canvas border"),
            ({"borders": {"left": 0.5}}, "valid integer"),
            ({"borders": {"right": 1025}}, "less than or equal to 1024"),
            ({"borders": borders, "size": "90x128"}, "produces 92x127"),
            ({"borders": borders, "image_resize_mode": "crop"}, "preserves the source"),
            (
                {
                    "borders": borders,
                    "input_image": inline(image_bytes(Image.new("RGB", (4096, 64)))),
                },
                "4096 pixels on each side",
            ),
        ):
            with self.subTest(fields=tuple(fields), message=message):
                response = self.json_edit(**fields)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(message, response.json()["error"]["message"])
                self.assertNotIn(mask, response.text)
        self.submit.assert_not_called()

    def test_ineligible_checkpoint_and_optional_engine_fail_before_enqueue(self):
        for family in ("Flux", "SDXLRefiner", "SDXL_instructpix2pix"):
            self.selected["family"] = family
            response = self.json_edit(mask=inline(self.mask))
            self.assertEqual(response.status_code, 400)
        self.assertIn(
            "require an SDXL, nine-channel SD1/SD2 or Flux Inpaint checkpoint", response.text
        )
        self.selected["family"] = "SDXL"
        self.settings["image_engine"] = "sdcpp-vulkan"
        response = self.json_edit(borders={"top": 1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("require the native image engine", response.text)
        self.submit.assert_not_called()

    def test_qwen_paint_and_mask_use_native_reference_route_preserving_source_alpha(self):
        self.selected["family"] = "QwenImage21"
        raw = image_bytes(Image.new("RGBA", self.size, (11, 87, 149, 40)))
        for mode in (None, "paint", "mask"):
            with self.subTest(mode=mode):
                files = {"image": ("source.png", raw), "mask": ("mask.png", self.mask)}
                form = {"prompt": "Recolor the red marked area."}
                if mode is not None:
                    form["qwen_local_edit_mode"] = mode
                    form["reference_images"] = json.dumps([inline(raw)])
                response = self.client.post("/images/edits", files=files, data=form)
                self.assertEqual(response.status_code, 200, response.text)
                data = self.prepared[-1]
                self.assertEqual(data["qwen_local_edit_mode"], mode or "mask")
                self.assertEqual((data["custom_width"], data["custom_height"]), self.size)
                self.assertEqual(data["main_view"].getpixel((0, 0)), (11, 87, 149, 40))
                self.assertNotIn("inpaint_denoise", data)
                self.assertNotIn("inpaint_mask_grow", data)
                self.assertEqual(len(data["reference_images"]), int(mode is not None))
                if mode:
                    self.assertEqual(data["reference_images"][0].mode, "RGBA")

    def test_qwen_local_options_reject_wrong_operation_or_legacy_sampler_controls(self):
        mask = inline(self.mask)
        self.selected["family"] = "QwenImage21"
        for fields, message in (
            ({"borders": {"left": 1}}, "Canvas extension requires"),
            ({"qwen_local_edit_mode": "paint"}, "requires a PNG mask"),
            ({"mask": mask, "inpaint_denoise": 0.7}, "does not use inpaint denoise"),
            ({"mask": mask, "inpaint_mask_grow": 2}, "does not use inpaint denoise"),
        ):
            with self.subTest(fields=tuple(fields)):
                response = self.json_edit(**fields)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(message, response.text)
        self.selected["family"] = "SDXL"
        response = self.json_edit(mask=mask, qwen_local_edit_mode="paint")
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("requires a Qwen Image 2.1 checkpoint", response.text)
        self.submit.assert_not_called()

    def test_qwen_mask_with_references_defaults_to_mask_and_other_families_still_reject(self):
        self.selected["family"] = "QwenImage21"
        reference = inline(image_bytes(Image.new("RGBA", self.size, (11, 87, 149, 40))))
        fields = {"mask": inline(self.mask), "reference_images": [reference]}
        response = self.json_edit(**fields)
        self.assertEqual(response.status_code, 200, response.text)
        data = self.prepared[-1]
        self.assertEqual(data["qwen_local_edit_mode"], "mask")
        self.assertEqual(len(data["reference_images"]), 1)
        self.assertEqual(data["reference_images"][0].getpixel((0, 0)), (11, 87, 149, 40))

        self.selected["family"] = "SDXL"
        self.submit.reset_mock()
        response = self.json_edit(**fields)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Additional references require", response.text)
        self.submit.assert_not_called()

    def test_upload_byte_and_pixel_bounds(self):
        files = {"image": ("source.png", self.source), "mask": ("mask.png", self.mask)}
        with patch.object(openai_api, "MAX_EDIT_FORM_BYTES", 40):
            response = self.client.post("/images/edits", files=files)
            self.assertEqual(response.status_code, 413, response.text)
        with patch.object(openai_api, "MAX_IMAGE_UPLOAD_BYTES", len(self.source)):
            response = self.client.post("/images/edits", files=files)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("PNG mask exceeds the size limit", response.text)
        oversized = inline(image_bytes(Image.new("1", (4097, 4096))))
        response = self.json_edit(input_image=oversized, borders={"left": 1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("16 megapixels", response.text)
        self.submit.assert_not_called()
