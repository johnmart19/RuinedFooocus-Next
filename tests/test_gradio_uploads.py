"""Identical uploads at once and cancelled uploads, through Gradio's real upload route.

The Windows lock (WinError 32) behind the reported failure is simulated by a
move that raises `PermissionError`; these checks do not hold a real file lock.
"""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr
from starlette.middleware import Middleware

from ruinedfooocus.integrations import gradio_uploads
from ruinedfooocus.integrations.gradio_uploads import UploadGuard, move_to_cache

PNG = bytes.fromhex("89504e470d0a1a0a") + b"layer" * 200


class MoveToCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.folder = self.root / "hash"
        self.folder.mkdir()

    def temporary(self, data=PNG):
        path = self.root / f"tmp{len(list(self.root.glob('tmp*')))}"
        path.write_bytes(data)
        return path

    def test_new_upload_is_moved_and_a_stored_copy_is_reused(self):
        first, dest = self.temporary(), self.folder / "layer_0.png"
        move_to_cache([str(first)], [str(dest)])
        self.assertEqual((first.exists(), dest.read_bytes()), (False, PNG))
        second = self.temporary()

        def locked(*args):
            raise PermissionError(32, "The file is used by another process")

        # The stored file is never copied over, so a lock on it cannot matter.
        with patch.object(gradio_uploads.shutil, "move", locked):
            move_to_cache([str(second)], [str(dest)])
        self.assertFalse(second.exists())
        self.assertEqual(dest.read_bytes(), PNG)

    def test_a_copy_stored_by_another_request_meanwhile_is_accepted(self):
        source, dest = self.temporary(), self.folder / "image.png"

        def raced(src, dst):
            Path(dst).write_bytes(PNG)
            raise PermissionError(32, "The file is used by another process")

        with patch.object(gradio_uploads.shutil, "move", raced):
            move_to_cache([str(source)], [str(dest)])
        self.assertFalse(source.exists())
        # A different file under that name is a real failure and is not hidden.
        other = self.temporary(PNG + b"x")
        with (
            patch.object(gradio_uploads.shutil, "move", raced),
            self.assertRaises(PermissionError),
        ):
            move_to_cache([str(other)], [str(self.folder / "other.png")])
        self.assertTrue(other.exists())


class UploadRouteTests(unittest.TestCase):
    def setUp(self):
        self.uploads = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict("os.environ", {"GRADIO_TEMP_DIR": str(self.uploads)}))
        with gr.Blocks(analytics_enabled=False) as blocks:
            gr.ImageEditor()
        self.addCleanup(blocks.close)
        self.app = gr.mount_gradio_app(
            FastAPI(),
            blocks,
            path="/",
            ssr_mode=False,
            mcp_server=False,
            app_kwargs={"middleware": [Middleware(UploadGuard)]},
        )

    def leftovers(self):
        return sorted(path.name for path in self.uploads.rglob("tmp*"))

    def test_the_same_picture_sent_twice_at_once_and_again_later_is_stored_once(self):
        def locked(*args):
            raise PermissionError(32, "The file is used by another process")

        with TestClient(self.app) as client:
            files = [
                ("files", ("layer_0.png", PNG, "image/png")),
                ("files", ("image.png", PNG, "image/png")),
            ]
            first = client.post("/gradio_api/upload", files=files)
            self.assertEqual(first.status_code, 200, first.text)
            with patch.object(gradio_uploads.shutil, "move", locked):
                second = client.post("/gradio_api/upload", files=files[:1])
            self.assertEqual(second.status_code, 200, second.text)
        stored = [Path(path) for path in first.json()]
        self.assertEqual(Path(second.json()[0]), stored[0])
        self.assertEqual({path.parent for path in stored}, {stored[0].parent})
        self.assertEqual([path.read_bytes() for path in stored], [PNG, PNG])
        self.assertEqual(self.leftovers(), [])

    def test_a_cancelled_upload_ends_quietly_and_leaves_no_partial_file(self):
        boundary = "rf-boundary"
        head = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="files"; filename="layer_0.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode() + PNG
        events = [
            {"type": "http.request", "body": head, "more_body": True},
            {"type": "http.disconnect"},
        ]
        sent = []

        async def receive():
            return events.pop(0) if events else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = dict(
            type="http",
            asgi={"version": "3.0"},
            http_version="1.1",
            method="POST",
            scheme="http",
            path="/gradio_api/upload",
            raw_path=b"/gradio_api/upload",
            root_path="",
            query_string=b"",
            headers=[
                (b"host", b"testserver"),
                (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            ],
            client=("127.0.0.1", 50000),
            server=("testserver", 80),
        )

        async def run():
            # Start the application the way a server does, then send the request.
            lifespan = iter([{"type": "lifespan.startup"}])

            async def lifespan_receive():
                return next(lifespan, {"type": "lifespan.shutdown"})

            task = asyncio.ensure_future(
                self.app({"type": "lifespan"}, lifespan_receive, lambda message: _noop())
            )
            await asyncio.sleep(0.05)
            await self.app(scope, receive, send)
            task.cancel()

        async def _noop():
            return None

        asyncio.run(run())
        self.assertEqual(sent, [])
        self.assertEqual(self.leftovers(), [])


if __name__ == "__main__":
    unittest.main()
