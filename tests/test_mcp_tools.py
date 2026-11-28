"""The Gradio API endpoints that --mcp publishes as MCP tools are described for callers."""

from pathlib import Path
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import gradio as gr
from gradio.utils import get_function_description, get_function_params
from ui_test_helpers import function_from


class McpToolTests(unittest.TestCase):
    def setUp(self):
        # Registration needs no runtime (it is only used when a tool is called), so
        # add_api is loaded from source without importing ComfyUI.
        runtime = ModuleType("ruinedfooocus.app.api_runtime")
        patcher = patch.dict(sys.modules, {"ruinedfooocus.app.api_runtime": runtime})
        patcher.start()
        self.addCleanup(patcher.stop)
        import ruinedfooocus.app as package  # an empty package

        attribute = patch.object(package, "api_runtime", runtime, create=True)
        attribute.start()
        self.addCleanup(attribute.stop)
        add_api = function_from(
            "ruinedfooocus/app/api.py",
            "add_api",
            {"gr": gr, "settings": None, "search_pipeline": None, "image_file_url": str},
        )
        add_api.__globals__["get_last_image"] = function_from(
            "ruinedfooocus/app/api.py", "get_last_image", {"Path": Path}
        )
        with gr.Blocks() as self.demo:
            add_api()
        self.endpoints = {
            fn.api_name: fn for fn in self.demo.fns.values() if fn.api_name is not None
        }

    def test_every_public_tool_describes_itself_and_each_parameter(self):
        public = {name: fn for name, fn in self.endpoints.items() if fn.api_visibility == "public"}
        self.assertEqual(set(public), {"llama", "generate_image", "search", "api_version"})
        for name, fn in public.items():
            with self.subTest(tool=name):
                description, parameters, _ = get_function_description(fn.fn)
                self.assertTrue(description.strip())
                for parameter, *_ in get_function_params(fn.fn):
                    self.assertTrue(parameters.get(parameter, "").strip(), parameter)

    def test_search_returns_a_list_and_the_slideshow_path_is_not_a_tool(self):
        search = self.endpoints["search"].fn
        self.assertEqual(search.__annotations__["return"], list[str])
        self.assertEqual(self.endpoints["last_image"].api_visibility, "undocumented")


class ToolServerTests(unittest.TestCase):
    def test_every_tool_server_operation_has_a_description(self):
        from unittest.mock import Mock

        from ruinedfooocus.integrations.openai_api import create_apps

        _, tools = create_apps(Mock(), None)
        for path, operations in tools.openapi()["paths"].items():
            for method, operation in operations.items():
                with self.subTest(operation=f"{method} {path}"):
                    self.assertTrue(operation.get("description", "").strip())


if __name__ == "__main__":
    unittest.main()
