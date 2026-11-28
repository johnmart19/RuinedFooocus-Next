"""Example clients keep framework transport and the production endpoint contracts."""

import ast
from concurrent.futures import TimeoutError
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote

from tools.api_examples import ExampleError, login, main, output_lines, parser, run

ROOT = Path(__file__).resolve().parents[1]


class ApiExampleTests(unittest.TestCase):
    def test_search_keeps_absolute_urls_including_encoded_whitespace(self):
        urls = ["http://localhost:7860/gradio_api/file=D:/outputs/a%20b.png"]
        self.assertEqual(output_lines("search", urls, "http://localhost:7860"), urls)
        self.assertEqual(output_lines("search", [], "http://localhost:7860"), [])
        with self.assertRaises(ExampleError):
            output_lines("search", ["outputs/a.png"], "http://localhost:7860")

    def test_framework_serializes_exact_unicode_multiline_prompt_without_downloading(self):
        prompt = 'Say "привіт"\nC:\\folder\\a.json and $(echo unchanged)'
        options = parser().parse_args(["llama", prompt])
        client = Mock()
        client.submit.return_value.result.return_value = prompt
        factory = Mock(return_value=client)
        self.assertEqual(run(options, client_factory=factory, environ={}), [prompt])
        self.assertFalse(factory.call_args.kwargs["download_files"])
        self.assertFalse(factory.call_args.kwargs["analytics_enabled"])
        self.assertIsNone(factory.call_args.kwargs["auth"])
        self.assertEqual(client.submit.call_args.kwargs["user"], prompt)
        self.assertEqual(client.submit.call_args.kwargs["api_name"], "/llama")
        client.close.assert_called_once()

    def test_timeout_requests_framework_cancel_and_closes_client(self):
        client = Mock()
        client.submit.return_value.result.side_effect = TimeoutError
        with self.assertRaisesRegex(ExampleError, "work may continue"):
            run(parser().parse_args(["search"]), client_factory=lambda *a, **k: client, environ={})
        client.submit.return_value.cancel.assert_called_once()
        client.close.assert_called_once()

    def test_ui_credentials_are_paired_and_not_part_of_url_or_arguments(self):
        self.assertEqual(login({"RF_GRADIO_USERNAME": "a", "RF_GRADIO_PASSWORD": "b"}), ("a", "b"))
        with self.assertRaisesRegex(ExampleError, "Set both"):
            login({"RF_GRADIO_USERNAME": "private"})
        for url in (
            "file:///etc/passwd",
            "http://user:secret@localhost",
            "http://localhost/?token=x",
        ):
            with redirect_stdout(StringIO()), patch("sys.stderr", StringIO()):
                with self.assertRaises(SystemExit):
                    parser().parse_args(["search", "--base-url", url])

    def test_image_filedata_uses_remote_path_and_actual_file_equals_route(self):
        self.assertEqual(
            output_lines("image", {"path": "C:\\outputs\\a #?.png"}, "http://localhost:7860"),
            ["http://localhost:7860/gradio_api/file=C:/outputs/a%20%23%3F.png"],
        )
        url = "https://example.org/gradio_api/file=/server/a.png"
        self.assertEqual(output_lines("image", {"url": url}, "http://localhost:7860"), [url])


class ProductionGradioExampleTests(unittest.TestCase):
    """Real Gradio/auth/queue with add_api from source and inert model services."""

    @classmethod
    def setUpClass(cls):
        import gradio as gr
        import ruinedfooocus.app

        cls.folder = tempfile.TemporaryDirectory(prefix="rf-api-examples-")
        cls.media = Path(cls.folder.name) / "image #1.png"
        cls.media.write_bytes(b"fixture media; no model or image execution")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.runtime = ModuleType("ruinedfooocus.app.api_runtime")
        cls.runtime.configured_chat_model = Mock(return_value=("fixture", {}))
        cls.runtime.chat = Mock(
            side_effect=lambda request, selected: {
                "choices": [{"message": {"content": request["messages"][-1]["content"]}}]
            }
        )
        cls.runtime.legacy_images = Mock(return_value=[str(cls.media)])
        cls.runtime.wait_for_result = lambda result: result
        cls.search = Mock(return_value=[str(cls.media)])
        tree = ast.parse((ROOT / "ruinedfooocus/app/api.py").read_text(encoding="utf-8"))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"add_api", "image_file_url"}
        ]
        scope = {
            "gr": gr,
            "Path": Path,
            "quote": quote,
            "shared": SimpleNamespace(local_url=cls.url),
            "settings": SimpleNamespace(default_settings={}),
            "search_pipeline": SimpleNamespace(search=cls.search),
            "get_last_image": lambda: str(cls.media),
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "app/api.py", "exec"), scope)
        # Callbacks look the runtime up when called, so the inert stand-in stays
        # installed (module and package attribute) until tearDownClass.
        cls.runtime_patches = [
            patch.dict(sys.modules, {cls.runtime.__name__: cls.runtime}),
            patch.object(ruinedfooocus.app, "api_runtime", cls.runtime, create=True),
        ]
        for runtime_patch in cls.runtime_patches:
            runtime_patch.start()
        with gr.Blocks(analytics_enabled=False) as cls.demo:
            scope["add_api"]()
        try:
            cls.demo.launch(
                server_name="127.0.0.1",
                server_port=port,
                prevent_thread_lock=True,
                inbrowser=False,
                quiet=True,
                auth=("fixture", "private"),
                allowed_paths=[cls.folder.name],
            )
        except BaseException:
            cls.demo.close()
            cls.folder.cleanup()
            for runtime_patch in reversed(cls.runtime_patches):
                runtime_patch.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()
        cls.folder.cleanup()
        for runtime_patch in reversed(cls.runtime_patches):
            runtime_patch.stop()

    def options(self, command, *args):
        return parser().parse_args([command, *args, "--base-url", self.url, "--timeout", "20"])

    def execute(self, command, *args):
        return run(
            self.options(command, *args),
            environ={
                "RF_GRADIO_USERNAME": "fixture",
                "RF_GRADIO_PASSWORD": "private",
            },
        )

    def test_search_actual_callback_preserves_server_url(self):
        result = self.execute("search", "a b", "--max-results", "7")
        self.assertEqual(
            result,
            [self.url + "/gradio_api/file=" + quote(self.media.resolve().as_posix(), safe="/:")],
        )
        self.search.assert_called_with("search: max:10 max:7 a b")

    def test_llama_actual_callback_receives_exact_prompt(self):
        prompt = 'Привіт "quoted"\nline two \\ $(no command)'
        self.assertEqual(self.execute("llama", prompt), [prompt])
        self.assertEqual(self.runtime.chat.call_args.args[0]["messages"][-1]["content"], prompt)

    def test_generate_image_actual_filedata_contract_without_model_execution(self):
        result = self.execute("image", "inert fixture")
        self.assertEqual(len(result), 1)
        self.assertIn("/gradio_api/file=", result[0])
        self.assertNotIn("{", result[0])
        self.runtime.legacy_images.assert_called_with("inert fixture")

    def test_private_gradio_requires_login_and_main_reports_failure_without_secret(self):
        output = StringIO()
        with patch.dict(os.environ, {"RF_GRADIO_USERNAME": "", "RF_GRADIO_PASSWORD": ""}):
            with patch("sys.stderr", output):
                status = main(["search", "--base-url", self.url, "--timeout", "20"])
        self.assertEqual(status, 1)
        self.assertNotIn("private", output.getvalue())


if __name__ == "__main__":
    unittest.main()
