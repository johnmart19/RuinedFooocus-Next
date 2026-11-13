"""Native Gradio completion and progress contracts; no network listener or model work."""

import ast
import asyncio
import copy
from dataclasses import replace
import gzip
import hashlib
import io
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import brotli
from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image
from starlette.middleware import Middleware

from ruinedfooocus.integrations.web_access import PrivateResponses
from ruinedfooocus.ui import gradio_frontend as frontend


ROOT = Path(__file__).resolve().parents[1]
ASSETS = files("gradio").joinpath("templates/frontend/assets")
CONTRACT = frontend.CONTRACTS[gr.__version__]
TRACKER = {"6.31.0": "statustracker-DQF-KcRT.js", "6.32.0": "statustracker-s2_3vshN.js"}[
    gr.__version__
]
SYNTAX = {
    "6.31.0": {"event": "s", "stream": "n", "status": "c"},
    "6.32.0": {"event": "o", "stream": "e", "status": "s"},
}[gr.__version__]


class GradioFrontendTests(unittest.TestCase):
    def test_actual_native_completion_and_tracker_contract(self):
        original = ASSETS.joinpath(CONTRACT.asset).read_bytes()
        patched = frontend._patch_blocks(original, CONTRACT)
        self.assertEqual(hashlib.sha256(original).hexdigest(), CONTRACT.original_sha256)
        self.assertEqual(hashlib.sha256(patched).hexdigest(), CONTRACT.patched_sha256)
        result = subprocess.run(
            ["node", "tests/test_gradio_frontend.js"],
            cwd=ROOT,
            input=json.dumps(
                {
                    "original": original.decode(),
                    "patched": patched.decode(),
                    "syntax": SYNTAX,
                    "tracker": ASSETS.joinpath(TRACKER).read_text(encoding="utf-8"),
                }
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unknown_bundle_is_rejected_without_modifying_it(self):
        source = ASSETS.joinpath(CONTRACT.asset).read_bytes() + b"\n"
        with self.assertRaisesRegex(ValueError, "hash differs"):
            frontend._patch_blocks(source, CONTRACT)
        self.assertTrue(source.endswith(b"\n"))

    def test_exact_branch_and_corrected_hash_are_required_for_each_release(self):
        original = ASSETS.joinpath(CONTRACT.asset).read_bytes()
        with self.assertRaisesRegex(ValueError, "does not match exactly once"):
            frontend._patch_blocks(original, replace(CONTRACT, before=b"missing branch"))
        with self.assertRaisesRegex(ValueError, "corrected Blocks asset hash differs"):
            frontend._patch_blocks(original, replace(CONTRACT, after=b"changed branch"))
        other = next(value for value in frontend.CONTRACTS.values() if value != CONTRACT)
        with self.assertRaisesRegex(ValueError, "native Blocks asset hash differs"):
            frontend._patch_blocks(original, other)

    def test_native_png_encoding_keeps_integer_clipping_and_original_pixels(self):
        original = Image.new("I", (6, 1))
        original.putdata([-1, 0, 1, 65535, 65536, 2147483647])
        original.info["parameters"] = "literal image metadata"
        pixels = original.tobytes()
        encoded = gr.processing_utils.encode_pil_to_bytes(original, "png")
        with Image.open(io.BytesIO(encoded)) as result:
            self.assertEqual(list(result.getdata()), [0, 0, 1, 65535, 65535, 65535])
            self.assertEqual(result.info["parameters"], "literal image metadata")
        self.assertEqual(original.mode, "I")
        self.assertEqual(original.tobytes(), pixels)

    def test_native_png_encoding_preserves_rgba_alpha_and_text_metadata(self):
        original = Image.new("RGBA", (7, 3), (20, 40, 60, 127))
        original.putpixel((6, 2), (100, 120, 140, 0))
        original.info["parameters"] = "literal prompt <tag>"
        encoded = gr.processing_utils.encode_pil_to_bytes(original, "png")
        with Image.open(io.BytesIO(encoded)) as result:
            self.assertEqual(result.mode, "RGBA")
            self.assertEqual(result.tobytes(), original.tobytes())
            self.assertEqual(result.info["parameters"], "literal prompt <tag>")

    def test_unsupported_version_or_asset_falls_through_and_warns_once(self):
        async def native(scope, receive, send):
            await send({"type": "http.response.start", "status": 202, "headers": []})
            await send({"type": "http.response.body", "body": b"native"})

        with (
            patch.object(frontend, "_warned", False),
            patch.object(frontend._log, "warning") as log,
        ):
            with patch.object(frontend, "version", return_value="unknown"):
                guard = frontend.GradioCompletionGuard(native)
                frontend.GradioCompletionGuard(native)
            with patch.object(frontend, "_patch_blocks", side_effect=ValueError("changed asset")):
                frontend.GradioCompletionGuard(native)
            self.assertIsNone(guard.asset)
            self.assertEqual(log.call_count, 1)
            messages = []

            async def send(message):
                messages.append(message)

            asyncio.run(
                guard(
                    {"type": "http", "path": "/assets/" + CONTRACT.asset, "method": "GET"},
                    None,
                    send,
                )
            )
            self.assertEqual(messages[0]["status"], 202)
            self.assertEqual(messages[1]["body"], b"native")

    def test_native_regular_and_prefixed_guest_routes_remain_private_on_warm_requests(self):
        original = ASSETS.joinpath(CONTRACT.asset).read_bytes()
        corrected = frontend._patch_blocks(original, CONTRACT)
        for prefix, guest in (("/", False), ("/proxy/rf", True)):
            with self.subTest(prefix=prefix):
                with gr.Blocks(analytics_enabled=False) as blocks:
                    gr.Textbox()
                self.addCleanup(blocks.close)
                app = gr.mount_gradio_app(
                    FastAPI(),
                    blocks,
                    path=prefix,
                    auth=("fixture-user", "fixture-password") if guest else None,
                    ssr_mode=False,
                    mcp_server=False,
                    app_kwargs={
                        "middleware": [
                            Middleware(PrivateResponses, guest=guest),
                            Middleware(frontend.GradioCompletionGuard),
                        ]
                    },
                )
                base = prefix.rstrip("/")
                with TestClient(app) as client:
                    endpoint = base + "/assets/" + CONTRACT.asset
                    tag = 'W/"' + CONTRACT.patched_sha256 + '"'
                    for encoding, expected in (
                        ("identity", None),
                        ("gzip", "gzip"),
                        ("br", "br"),
                        (None, "br"),
                    ):
                        for conditional in (
                            None,
                            '"' + CONTRACT.original_sha256 + '"',
                            '"' + CONTRACT.patched_sha256 + '"',
                            tag,
                        ):
                            with self.subTest(encoding=encoding, conditional=conditional):
                                headers = {} if encoding is None else {"Accept-Encoding": encoding}
                                if conditional is not None:
                                    headers["If-None-Match"] = conditional
                                with client.stream("GET", endpoint, headers=headers) as response:
                                    wire = b"".join(response.iter_raw())
                                    self.assertEqual(response.status_code, 200)
                                    self.assertEqual(
                                        response.headers.get("content-encoding"), expected
                                    )
                                    self.assertEqual(
                                        response.headers["content-length"], str(len(wire))
                                    )
                                    decoded = (
                                        gzip.decompress(wire)
                                        if expected == "gzip"
                                        else (brotli.decompress(wire) if expected == "br" else wire)
                                    )
                                    self.assertEqual(decoded, corrected)
                                head = client.head(endpoint, headers=headers)
                                self.assertEqual(head.status_code, 200)
                                self.assertEqual(head.content, b"")
                                self.assertNotIn("content-length", head.headers)
                                self.assertNotIn("content-encoding", head.headers)
                                for reply in (response, head):
                                    self.assertEqual(reply.headers["etag"], tag)
                                    self.assertEqual(
                                        reply.headers["cache-control"], "private, no-store"
                                    )
                                    self.assertEqual(
                                        reply.headers["referrer-policy"], "no-referrer"
                                    )
                                    self.assertIn("accept-encoding", reply.headers["vary"].lower())
                    for _ in range(2):
                        page = client.get(base + "/")
                        self.assertEqual(page.status_code, 200)
                        self.assertEqual(page.headers["cache-control"], "private, no-store")
                    unchanged = client.get(base + "/assets/" + TRACKER)
                    self.assertEqual(unchanged.status_code, 200)
                    self.assertEqual(unchanged.content, ASSETS.joinpath(TRACKER).read_bytes())
                    config = client.get(base + "/config")
                    self.assertEqual(config.status_code, 401 if guest else 200)
        self.assertEqual(ASSETS.joinpath(CONTRACT.asset).read_bytes(), original)

    def test_hidden_scope_normalization_preserves_native_functions_and_data_errors(self):
        def echo(value):
            if value == "bad":
                raise gr.Error("native error preserved")
            return value

        with gr.Blocks(analytics_enabled=False) as blocks:
            source = gr.Textbox()
            output = gr.Textbox()
            notice = gr.Markdown()
            events = []
            for mode, scope in (
                ("hidden", None),
                ("hidden", []),
                ("hidden", [notice]),
                ("full", None),
                ("minimal", None),
                ("full", [notice]),
            ):
                events.append(
                    source.change(
                        echo,
                        source,
                        output,
                        show_progress=mode,
                        show_progress_on=scope,
                        queue=False,
                        concurrency_id="native-owner",
                        concurrency_limit=1,
                    )
                )
            gr.Button().click(None, cancels=events[0], queue=False)
        self.addCleanup(blocks.close)
        config = copy.deepcopy(blocks.config)
        before = {key: copy.deepcopy(fn.get_config()) for key, fn in blocks.fns.items()}
        identities = {key: dict(vars(fn)) for key, fn in blocks.fns.items()}
        changed = {
            key
            for key, fn in blocks.fns.items()
            if fn.show_progress == "hidden" and fn.show_progress_on is None
        }
        self.assertIn(events[0]["id"], changed)
        self.assertEqual(frontend.normalize_hidden_progress(blocks), len(changed))
        for key, fn in blocks.fns.items():
            expected = before[key] | {"show_progress_on": []} if key in changed else before[key]
            self.assertEqual(fn.get_config(), expected)
            for name, value in identities[key].items():
                if name != "show_progress_on" or key not in changed:
                    self.assertIs(getattr(fn, name), value, (key, name))
        for row in config["dependencies"]:
            if row["id"] in changed:
                row["show_progress_on"] = []
        self.assertEqual(blocks.config, config)
        current = blocks.config
        self.assertEqual(frontend.normalize_hidden_progress(blocks), 0)
        self.assertIs(blocks.config, current)

        async def check_data_errors():
            result = await blocks.process_api(events[0]["id"], ["ok"], state=SessionState(blocks))
            self.assertEqual(result["data"], ["ok"])
            with self.assertRaisesRegex(gr.Error, "native error preserved"):
                await blocks.process_api(events[0]["id"], ["bad"], state=SessionState(blocks))

        asyncio.run(check_data_errors())

    def test_both_launchers_normalize_before_launch_and_keep_private_middleware_outermost(self):
        for filename, name in (("app.py", "main_tabs"), ("ui_guest.py", "app")):
            with self.subTest(filename=filename):
                tree = ast.parse((ROOT / "ruinedfooocus/ui" / filename).read_text(encoding="utf-8"))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                normalize = next(
                    node
                    for node in calls
                    if isinstance(node.func, ast.Name)
                    and node.func.id == "normalize_hidden_progress"
                )
                launch = next(
                    node
                    for node in calls
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "launch"
                )
                self.assertEqual(normalize.args[0].id, name)
                self.assertEqual(launch.func.value.id, name)
                self.assertLess(normalize.lineno, launch.lineno)
                value = next(item.value for item in launch.keywords if item.arg == "app_kwargs")
                options = eval(
                    compile(ast.Expression(value), filename, "eval"),
                    {
                        "Middleware": Middleware,
                        "PrivateResponses": PrivateResponses,
                        "GradioCompletionGuard": frontend.GradioCompletionGuard,
                    },
                )
                self.assertEqual(
                    [item.cls for item in options["middleware"]],
                    [PrivateResponses, frontend.GradioCompletionGuard],
                )


if __name__ == "__main__":
    unittest.main()
