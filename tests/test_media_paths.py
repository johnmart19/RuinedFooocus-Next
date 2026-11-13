"""Real Gradio media transport with an external data root, auth and upload fences."""

import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr
from gradio.exceptions import InvalidPathError
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.integrations.media_paths import PUBLIC_CACHE_MEDIA, owner_media_paths
from ruinedfooocus.integrations.web_access import PrivateResponses


class MediaPathsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.source, self.data, self.system_temp = (
            self.root / name for name in ("source", "external-data", "system-temp")
        )
        for path in (self.source, self.data, self.system_temp):
            path.mkdir()
        self.cache = self.data / "cache"
        self.outputs = self.data / "outputs"
        self.outputs.mkdir()
        self.manager = SimpleNamespace(
            model_paths=dict(cache_path=self.cache, temp_outputs_path=self.outputs)
        )
        self.models = SimpleNamespace(
            cache_paths=dict(
                checkpoints=self.cache / "checkpoints",
                loras=self.cache / "loras",
                embeddings=self.cache / "embeddings",
            )
        )
        self.allowed = owner_media_paths(self.manager, self.models, {})
        self.media = []
        for name in PUBLIC_CACHE_MEDIA:
            path = self.cache / name / "owned" / "image.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 4), "blue").save(path)
            self.media.append(path)
        self.secret = self.data / "settings" / "secrets.json"
        self.secret.parent.mkdir()
        self.secret.write_text('{"private":"fixture"}', encoding="utf-8")

    def test_only_dedicated_media_leaves_are_added_and_existing_roots_remain(self):
        allowed = set(self.allowed)
        for name in PUBLIC_CACHE_MEDIA:
            self.assertIn(str(self.cache / name), allowed)
        for path in (
            self.data,
            self.cache,
            self.secret.parent,
            self.data / "models",
            self.cache / "hashes",
        ):
            self.assertNotIn(str(path), allowed)
        self.assertIn(str(self.outputs), allowed)
        self.assertTrue(set(str(path) for path in self.models.cache_paths.values()) <= allowed)
        archive = self.root / "explicit-archive"
        self.assertIn(
            str(archive),
            owner_media_paths(self.manager, self.models, {"archive_folders": [str(archive)]}),
        )

    def test_real_output_callback_reproduces_missing_media_root_then_accepts_each_owned_leaf(self):
        with gr.Blocks(analytics_enabled=False) as app:
            output = gr.Gallery()
            button = gr.Button()
            button.click(lambda: [str(path) for path in self.media], outputs=output)
        self.addCleanup(app.close)
        app.has_launched = True
        # Model the portable source/data/system-temp separation. The production
        # _check_allowed function and file-cache movement run unchanged.
        with (
            patch("gradio.processing_utils.os.getcwd", return_value=str(self.source)),
            patch(
                "gradio.processing_utils.tempfile.gettempdir", return_value=str(self.system_temp)
            ),
        ):
            app.allowed_paths = [str(self.outputs)]
            with self.assertRaises(InvalidPathError):
                asyncio.run(app.process_api(0, [], state=SessionState(app)))
            app.allowed_paths = self.allowed
            result = asyncio.run(app.process_api(0, [], state=SessionState(app)))["data"][0]
            self.assertEqual(len(result), len(PUBLIC_CACHE_MEDIA))
            for item in result:
                self.assertEqual(Image.open(item["image"]["path"]).size, (3, 4))
            with self.assertRaises(InvalidPathError):
                from gradio.context import LocalContext

                token = LocalContext.blocks.set(app)
                try:
                    gr.processing_utils._check_allowed(self.secret, False)
                finally:
                    LocalContext.blocks.reset(token)

    def test_public_output_root_does_not_make_an_unuploaded_input_valid(self):
        with gr.Blocks(analytics_enabled=False) as app:
            uploaded = gr.File()
            button = gr.Button()
            button.click(lambda value: value, inputs=uploaded, outputs=gr.File())
        self.addCleanup(app.close)
        app.has_launched, app.allowed_paths = True, self.allowed
        data = gr.data_classes.FileData(path=str(self.media[0])).model_dump()
        with self.assertRaisesRegex(InvalidPathError, "not uploaded"):
            asyncio.run(app.process_api(0, [data], state=SessionState(app)))

    def test_generated_default_preview_passes_launched_html_callback_with_external_data_root(self):
        from ruinedfooocus.core.paths import data_path
        from ruinedfooocus.features.models.ui_acquisition import card_data

        self.manager.DOWNLOADABLE_FILES = {}
        row = dict(
            key="checkpoints/missing.safetensors",
            label="No publisher artwork",
            category="checkpoints",
            category_label="Image models",
            local=None,
            destination=str(self.data / "models/missing.safetensors"),
            annotation={},
            status="Missing",
            quantization="fp16",
            task="Text-to-image",
        )

        def render():
            return card_data([row], row["key"], self.manager)[0]

        with gr.Blocks(analytics_enabled=False) as app:
            output = gr.HTML(value={}, html_template="<div></div>")
            gr.Button().click(render, outputs=output)
        self.addCleanup(app.close)
        app.has_launched = True
        leaf = self.cache / "default-previews"
        with (
            patch.dict("os.environ", {"RF_DATA_DIR": str(self.data)}),
            patch("gradio.processing_utils.os.getcwd", return_value=str(self.source)),
            patch(
                "gradio.processing_utils.tempfile.gettempdir", return_value=str(self.system_temp)
            ),
        ):
            self.assertEqual(data_path("cache", "default-previews"), leaf)
            # Reproduce the actual failure using the pre-P0 media contract,
            # then exercise the installed serializer with the one added leaf.
            app.allowed_paths = [path for path in self.allowed if Path(path) != leaf]
            with self.assertRaises(InvalidPathError):
                asyncio.run(app.process_api(0, [], state=SessionState(app)))
            app.allowed_paths = self.allowed
            card = asyncio.run(app.process_api(0, [], state=SessionState(app)))["data"][0]["cards"][
                0
            ]
            for name in ("preview", "fallback"):
                self.assertTrue(card[name]["url"].startswith("/gradio_api/file="))
                with Image.open(card[name]["path"]) as image:
                    self.assertEqual(image.size, (512, 512))
            self.assertEqual(card["preview"]["path"], card["fallback"]["path"])
            self.assertRegex(
                Path(card["fallback"]["path"]).name, r"^image-model-[0-9a-f]{16}\.jpeg$"
            )
            self.assertNotIn(str(self.data), app.allowed_paths)
            self.assertNotIn(str(self.cache), app.allowed_paths)

    def test_authenticated_file_routes_and_guest_blocking_remain_intact(self):
        from ruinedfooocus.features.models.resource_cards import (
            DEFAULT_PREVIEW_ROLES,
            default_preview,
        )

        defaults = [
            Path(default_preview(labels[0], self.cache))
            for labels in DEFAULT_PREVIEW_ROLES.values()
        ]
        with gr.Blocks(analytics_enabled=False) as blocks:
            gr.Textbox()
        self.addCleanup(blocks.close)
        app = gr.mount_gradio_app(
            FastAPI(),
            blocks,
            path="/owner",
            auth=("owner", "password"),
            allowed_paths=self.allowed,
            ssr_mode=False,
            mcp_server=False,
        )
        client = TestClient(app)
        self.addCleanup(client.close)
        paths = ["/owner/gradio_api/file=" + path.as_posix() for path in [*self.media, *defaults]]
        for path in paths:
            self.assertEqual(client.get(path).status_code, 401)
        self.assertEqual(
            client.post(
                "/owner/login", data={"username": "owner", "password": "password"}
            ).status_code,
            200,
        )
        for path in paths:
            self.assertEqual(client.get(path).status_code, 200)
        for path in defaults:
            response = client.get("/owner/gradio_api/file=" + path.as_posix())
            self.assertEqual(response.content, path.read_bytes())
            self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(
            client.get("/owner/gradio_api/file=" + self.secret.as_posix()).status_code, 403
        )
        guest = TestClient(PrivateResponses(FastAPI(), guest=True))
        self.addCleanup(guest.close)
        self.assertEqual(guest.get("/gradio_api/file=" + self.media[0].as_posix()).status_code, 404)


if __name__ == "__main__":
    unittest.main()
