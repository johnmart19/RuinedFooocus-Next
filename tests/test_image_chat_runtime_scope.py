"""Actual API admission and safe model headers preserve image/Chat-only scope."""

import ast
import base64
from io import BytesIO
import json
from pathlib import Path
import queue
import sqlite3
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.models import image_scope
from ruinedfooocus.integrations.openai_api import create_apps

ROOT = Path(__file__).resolve().parents[1]


def completed(value):
    events = queue.Queue()
    events.put(("result", value))
    return SimpleNamespace(events=events, cancelled=threading.Event())


def inline_image(*frames):
    stream = BytesIO()
    frames[0].save(
        stream,
        format="GIF" if len(frames) > 1 else "PNG",
        save_all=len(frames) > 1,
        append_images=frames[1:],
    )
    return base64.b64encode(stream.getvalue()).decode()


class ImageChatScopeTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimpleNamespace(
            models=Mock(return_value={}),
            capabilities=Mock(return_value={}),
            resolve_model=Mock(
                side_effect=lambda model, kind: (
                    model,
                    {"kind": kind, "family": "SDXL", "name": "image.safetensors"},
                )
            ),
            images=Mock(return_value=completed([])),
            chat=Mock(return_value=completed("Local reply")),
        )
        api, tools = create_apps(self.backend, "scope-fixture")
        app = FastAPI()
        app.mount("/v1", api)
        app.mount("/tools", tools)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer scope-fixture"}

    def test_retired_http_and_tool_routes_are_absent_from_actual_apps(self):
        for endpoint in (
            "/v1/videos/generations",
            "/v1/audio/generations",
            "/v1/meshes/generations",
            "/tools/video_generation",
        ):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(
                    self.client.post(
                        endpoint, headers=self.headers, json={"prompt": "test"}
                    ).status_code,
                    404,
                )
        for endpoint in ("/v1/openapi.json", "/tools/openapi.json"):
            schema = self.client.get(endpoint, headers=self.headers).json()
            self.assertNotIn("VideoRequest", schema.get("components", {}).get("schemas", {}))

    def test_retired_model_ids_and_timing_are_rejected_before_backend_admission(self):
        for changes in (
            {"model": "video/default"},
            {"model": "audio/music"},
            {"model": "3d/mesh"},
            {"duration": 3},
            {"fps": 24},
            {"video_fps": 24},
        ):
            with self.subTest(changes=changes):
                response = self.client.post(
                    "/v1/images/generations",
                    headers=self.headers,
                    json={"prompt": "test", **changes},
                )
                self.assertEqual(response.status_code, 400)
        self.backend.images.assert_not_called()

    def test_retired_cached_families_fail_before_input_decode_or_queue(self):
        for family in ("LTXV 2.5", "WAN21_T2V", "WAN21_I2V", "StableAudio", "Trellis2"):
            with (
                self.subTest(family=family),
                patch.object(api_runtime, "decode_image") as decode,
                patch.object(api_runtime, "submit") as submit,
            ):
                with self.assertRaisesRegex(ValueError, "still images and local Chat"):
                    api_runtime.images(
                        {"prompt": "test", "input_image": "not decoded"}, {"family": family}
                    )
                decode.assert_not_called()
                submit.assert_not_called()

    def test_actual_image_and_chat_apps_keep_image_options_and_system_messages(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=self.headers,
            json={
                "prompt": "test",
                "n": 2,
                "size": "768x1024",
                "steps": 12,
                "cfg": 4.5,
                "loras": [{"name": "style.safetensors", "strength": 0.7}],
                "styles": ["3d-render"],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = self.backend.images.call_args.args[0]
        self.assertEqual(
            (payload["n"], payload["size"], payload["steps"], payload["cfg"]),
            (2, "768x1024", 12, 4.5),
        )
        self.assertEqual(payload["loras"][0]["strength"], 0.7)
        messages = [
            {"role": "system", "content": "Keep this assistant preset."},
            {"role": "user", "content": "Explain this image."},
        ]
        response = self.client.post(
            "/v1/chat/completions", headers=self.headers, json={"messages": messages}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.backend.chat.call_args.args[0]["messages"], messages)

    def test_single_frame_images_keep_alpha_and_animated_inputs_are_refused(self):
        source = Image.new("RGBA", (4, 3), (10, 20, 30, 64))
        decoded = api_runtime.decode_image(inline_image(source), preserve_alpha=True)
        self.assertEqual(
            (decoded.mode, decoded.size, decoded.getpixel((0, 0))),
            ("RGBA", (4, 3), (10, 20, 30, 64)),
        )
        with self.assertRaises(ValueError):
            api_runtime.decode_image(
                inline_image(source.convert("RGB"), Image.new("RGB", source.size, "white"))
            )

    def test_local_folder_names_do_not_override_image_headers(self):
        header = {
            "img_in.proj.weight": {
                "dtype": "F32",
                "shape": [3072, 64, 1, 1, 1],
                "data_offsets": [0, 0],
            },
            "txt_in.individual_token_refiner.blocks.0.norm1.weight": {
                "dtype": "F32",
                "shape": [3072],
                "data_offsets": [0, 0],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            for folder in ("3d", "video", "Wan Video"):
                with self.subTest(folder=folder):
                    name = folder + "/image-style.safetensors"
                    image_scope.require_image_operation({"base_model_name": name})
                    path = Path(temporary) / name
                    path.parent.mkdir(parents=True)
                    content = json.dumps(header).encode()
                    path.write_bytes(struct.pack("<Q", len(content)) + content)
                    image_scope.require_image_checkpoint(path, "HunyuanImage21Refiner")

    def test_shared_image_families_and_text_styles_are_not_media_operations(self):
        for family in (
            "HunyuanImage21",
            "HunyuanImage21Refiner",
            "MiniMaxH3Image",
            "CosmosPredict2",
            "JoyImage",
            "QwenImage21",
        ):
            with self.subTest(family=family):
                image_scope.require_image_family(family)
                self.assertTrue(image_scope.admitted_native_family(family))
        image_scope.require_image_operation(
            {"task_type": "process", "prompt": "3D render of a videogame musician"}
        )

    def test_retired_sources_are_refused_before_image_decoders(self):
        from ruinedfooocus.features.gallery.actions import source_path

        for suffix in (".mp4", ".wav", ".glb", ".ply"):
            with (
                self.subTest(suffix=suffix),
                self.assertRaisesRegex(ValueError, "still images and local Chat"),
            ):
                source_path({"path": "original" + suffix, "signature": []})

    def test_existing_library_index_hides_retired_outputs_without_deleting_rows(self):
        from ruinedfooocus.features.gallery.index_store import ensure_schema, page_snapshot

        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        ensure_schema(connection)
        connection.executemany(
            "INSERT INTO images(fullpath,path,json) VALUES(?,?,?)",
            [("original.mp4", "original.mp4", "{}"), ("image.PNG", "image.PNG", "{}")],
        )
        connection.commit()
        count, _, _, rows = page_snapshot(connection, 1, "", 100)
        self.assertEqual(count, 1)
        self.assertEqual(rows[0][0], "image.PNG")
        self.assertEqual(connection.execute("SELECT count(*) FROM images").fetchone()[0], 2)

    def test_animated_sources_are_static_artwork_and_never_library_edit_inputs(self):
        from ruinedfooocus.features.gallery.actions import source_record, source_path
        from ruinedfooocus.features.models.model_identity import file_signature
        from ruinedfooocus.features.models.model_previews import valid_preview, write_preview

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix, format in ((".gif", "GIF"), (".webp", "WEBP"), (".png", "PNG")):
                with self.subTest(format=format):
                    path = root / ("original" + suffix)
                    first = Image.new("RGBA", (5, 4), "red")
                    first.save(
                        path,
                        format=format,
                        save_all=True,
                        append_images=[Image.new("RGBA", first.size, "blue")],
                        duration=100,
                    )
                    original = path.read_bytes()
                    record = {"path": str(path), "signature": file_signature(path)}
                    for admission, value in ((source_record, path), (source_path, record)):
                        with self.assertRaisesRegex(ValueError, "single still image"):
                            admission(value)
                    thumbnail = write_preview(original, root / f"artwork-{format}{suffix}")
                    with Image.open(thumbnail) as image:
                        self.assertEqual(getattr(image, "n_frames", 1), 1)
                    self.assertFalse(valid_preview(path))
                    self.assertEqual(path.read_bytes(), original)
                    still = root / ("still" + suffix)
                    first.save(still, format=format)
                    self.assertEqual(source_path(source_record(still)), still.resolve())

    def test_mixed_saved_image_preset_is_retained_with_legacy_keys(self):
        from ruinedfooocus.features.models.performance import PerformanceSettings

        mixed = {"image_models": ["SDXL"], "video_models": ["LTXV"], "custom_steps": 17, "cfg": 4.5}
        retired = {"video_models": ["LTXV"], "custom_steps": 8}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            defaults, saved = root / "defaults.json", root / "performance.json"
            defaults.write_text("{}", encoding="utf-8")
            saved.write_text(
                json.dumps({"My image preset": mixed, "Retired preset": retired}), encoding="utf-8"
            )
            with (
                patch.object(PerformanceSettings, "DEFAULT_PERFORMANCE_FILE", defaults),
                patch.object(PerformanceSettings, "PERFORMANCE_FILE", saved),
            ):
                settings = PerformanceSettings.__new__(PerformanceSettings)
                options = settings.load_performance()
            self.assertEqual(options["My image preset"], mixed)
            self.assertNotIn("Retired preset", options)
            self.assertEqual(
                json.loads(saved.read_text(encoding="utf-8")),
                {"My image preset": mixed, "Retired preset": retired},
            )

    def test_rf_dispatch_has_no_import_of_a_retired_runtime_package(self):
        for folder in ("app", "features", "ui", "integrations", "backends/comfy"):
            for path in (ROOT / "ruinedfooocus" / folder).rglob("*.py"):
                if "resources" in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    imports = (
                        [name.name for name in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                        if isinstance(node, ast.ImportFrom)
                        else []
                    )
                    self.assertFalse(
                        any(name.startswith("ruinedfooocus.features.video") for name in imports),
                        str(path),
                    )
