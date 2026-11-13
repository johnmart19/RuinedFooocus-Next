"""Exercise the guest builder with real Gradio configuration and no server/model."""

import ast
import html
import json
import os
from pathlib import Path
import queue
import secrets
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from starlette.middleware import Middleware

from ruinedfooocus.core.translation_manager import catalogue, locale_change_js, translate_ui
from ruinedfooocus.integrations.web_access import PrivateResponses, login_token, validate_web_access
from ruinedfooocus.ui import themes
from ruinedfooocus.ui.gradio_frontend import GradioCompletionGuard, normalize_hidden_progress


class GuestLocalizationTests(unittest.TestCase):
    def build_guest(self, settings_values=None, custom_schema=None):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/ui_guest.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        definitions = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        captured = {}
        html_tree = ast.parse((source.parent / "html.py").read_text(encoding="utf-8"))
        runtime = {"json": json}
        runtime_nodes = [
            node
            for node in html_tree.body
            if isinstance(node, ast.Assign)
            and node.targets[0].id in ("head", "sidebar_head", "scripts")
            or isinstance(node, ast.FunctionDef)
            and node.name == "startup_head"
        ]
        exec(
            compile(ast.Module(body=runtime_nodes, type_ignores=[]), "guest_runtime", "exec"),
            runtime,
        )
        captured["runtime"] = SimpleNamespace(**runtime)

        class StopIdleLoop(Exception):
            pass

        backend = ModuleType("ruinedfooocus.app.api_runtime")
        backend.models = Mock(return_value={})
        requests = ModuleType("ruinedfooocus.integrations.openai_api")
        requests.ImageRequest = Mock()
        server = SimpleNamespace(get=lambda *a, **k: lambda callback: callback)

        def launch(demo, **kwargs):
            captured.update(demo=demo, launch=kwargs)
            demo.i18n_instance = kwargs.get("i18n")
            return server, "private-fixture", None

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, RF_DATA_DIR=folder),
        ):
            values = {"ui_language": "en", **(settings_values or {})}
            if custom_schema is not None:
                cache = themes.cache_file(values["theme"])
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(
                    json.dumps({"id": values["theme"], "schema": custom_schema}), encoding="utf-8"
                )
            shared = SimpleNamespace(
                path_manager=SimpleNamespace(model_paths={"temp_outputs_path": folder}),
                settings=SimpleNamespace(default_settings=values),
            )
            namespace = {
                "shared": shared,
                "gr": gr,
                "Path": Path,
                "secrets": secrets,
                "time": SimpleNamespace(sleep=Mock(side_effect=StopIdleLoop), monotonic=lambda: 0),
                "version": SimpleNamespace(version="fixture"),
                "html": html,
                "queue": queue,
                "HTTPException": HTTPException,
                "Request": Request,
                "FileResponse": FileResponse,
                "Middleware": Middleware,
                "PrivateResponses": PrivateResponses,
                "GradioCompletionGuard": GradioCompletionGuard,
                "normalize_hidden_progress": normalize_hidden_progress,
                "login_token": login_token,
                "validate_web_access": validate_web_access,
                "translate_ui": translate_ui,
                "catalogue": catalogue,
                "locale_change_js": locale_change_js,
                "app_html": captured["runtime"],
            }
            exec(
                compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
                namespace,
            )
            args = SimpleNamespace(
                auth="fixture-user/fixture-password",
                guest=True,
                api=False,
                mcp=False,
                share=False,
                listen="127.0.0.1",
                port=7912,
                nobrowser=True,
                language="en",
            )
            with (
                patch.dict(sys.modules, {backend.__name__: backend, requests.__name__: requests}),
                patch("ruinedfooocus.app.api_runtime", backend, create=True),
                patch.object(gr.Blocks, "launch", launch),
                patch.object(gr.Blocks, "close"),
                self.assertRaises(StopIdleLoop),
            ):
                namespace["launch_guest"](args)
        return captured

    def test_guest_builder_attaches_native_catalogue_and_profile_locale_with_private_access(self):
        captured = self.build_guest()
        launch_args = captured["launch"]
        self.assertIn("i18n", launch_args)
        config = captured["demo"].get_config_file()
        self.assertEqual(config["i18n_translations"]["en"]["ruinedfooocus.common.prompt"], "Prompt")
        self.assertEqual(
            config["i18n_translations"]["en"]["ruinedfooocus.common.generate"], "Generate"
        )
        load = next(
            event
            for event in config["dependencies"]
            if event["js"] and "changeLocale" in event["js"]
        )
        self.assertIn('const defaultLocale = "en"', load["js"])
        self.assertEqual(load["outputs"], [])
        self.assertFalse(load["queue"])
        self.assertEqual(launch_args["auth"], ["fixture-user", "fixture-password"])
        self.assertEqual(launch_args["allowed_paths"], [])
        self.assertFalse(launch_args["share"])
        self.assertTrue(launch_args["app_kwargs"]["middleware"][0].kwargs["guest"])
        self.assertTrue(
            all(event["api_visibility"] == "private" for event in config["dependencies"])
        )

    def test_guest_applies_saved_native_custom_and_fallback_themes_with_exact_restart_id(self):
        for value, schema, constructor, themed in (
            ("", None, gr.themes.Default, False),
            ("builtin:soft", None, gr.themes.Soft, True),
            ("author/theme", {"theme": {"body_text_color": "#123456"}}, gr.themes.Default, True),
            ("author/missing", None, gr.themes.Default, False),
        ):
            with self.subTest(theme=value), patch("builtins.print"):
                captured = self.build_guest({"theme": value}, schema)
                launch = captured["launch"]
                demo = captured["demo"]
                self.assertIsInstance(launch["theme"], constructor)
                self.assertEqual(launch["js"], captured["runtime"].scripts)
                self.assertEqual(
                    launch["head"],
                    captured["runtime"].startup_head(demo.app_id, themed=themed, sidebar=False),
                )
                self.assertNotIn("14.5rem", launch["head"])
                self.assertNotIn("css", launch)
                self.assertEqual(launch["allowed_paths"], [])
                self.assertTrue(launch["app_kwargs"]["middleware"][0].kwargs["guest"])
                if value == "author/theme":
                    self.assertEqual(launch["theme"].body_text_color, "#123456")


if __name__ == "__main__":
    unittest.main()
