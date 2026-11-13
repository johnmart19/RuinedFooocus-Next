"""Owner API workspace status and examples without real models or traffic."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruinedfooocus.app import api_runtime, resource_lifecycle
from ruinedfooocus.integrations.openai_api import ChatRequest, ImageRequest, install
from ruinedfooocus.ui import ui_api_workspace as workspace


class ApiWorkspaceTests(unittest.TestCase):
    def test_active_mount_marker_matches_actual_auth_without_recording_key(self):
        app = FastAPI()
        fixture_key = "example-secret-never-sent-to-workspace"
        with patch.object(api_runtime, "models", return_value={}):
            self.assertIsNone(getattr(app.state, "rf_openai_api_requires_key", None))
            install(app, fixture_key)
            self.assertIs(app.state.rf_openai_api_requires_key, True)
            with TestClient(app) as client:
                self.assertEqual(client.get("/v1/models").status_code, 401)
                self.assertEqual(
                    client.get(
                        "/v1/models", headers={"Authorization": "Bearer " + fixture_key}
                    ).status_code,
                    200,
                )
        no_key = FastAPI()
        install(no_key)
        self.assertIs(no_key.state.rf_openai_api_requires_key, False)
        failed = FastAPI()
        with (
            patch.object(failed, "mount", side_effect=[None, RuntimeError("mount failed")]),
            self.assertRaisesRegex(RuntimeError, "mount failed"),
        ):
            install(failed, fixture_key)
        self.assertIsNone(getattr(failed.state, "rf_openai_api_requires_key", None))

    def test_running_status_uses_public_ids_and_separates_default_from_loaded_model(self):
        root = Path.cwd() / "private-workspace-fixture"
        chat_path = root / "loaded.gguf"
        image_path = root / "same.sft"
        rows = {
            "chat/selected.gguf": {"kind": "chat", "path": root / "selected.gguf"},
            "chat/loaded.gguf": {"kind": "chat", "path": chat_path},
            "image/nested/same.sft": {"kind": "image", "path": image_path},
            "video/private-model": {"kind": "video", "path": root / "video"},
        }
        active = {
            "chat": SimpleNamespace(llm=object(), loaded_paths=(chat_path,)),
            "image": SimpleNamespace(
                xl_base=object(),
                xl_base_hash="same.sft",
                base_file_signature=(image_path,),
            ),
        }
        model = Mock(
            side_effect=lambda _, kind: (
                {
                    "chat": "chat/selected.gguf",
                    "image": "image/nested/same.sft",
                }[kind],
                {},
            )
        )
        shared = SimpleNamespace(
            server_app=SimpleNamespace(state=SimpleNamespace(rf_openai_api_requires_key=True)),
            local_url="http://localhost:9250/",
            settings=SimpleNamespace(default_settings={"startup_api": False}),
        )
        with (
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(api_runtime, "models", return_value=rows),
            patch.object(api_runtime, "resolve_model", model),
            patch.object(resource_lifecycle, "service_pipeline", side_effect=active.get),
        ):
            status, visible, base, links, table, ids, chat, images, masked = workspace._refresh(
                "en"
            )
        self.assertIn("enabled", status)
        self.assertTrue(visible["visible"])
        self.assertEqual(base, "http://localhost:9250/v1")
        self.assertIn("/v1/openapi.json", links)
        self.assertIn("/tools/openapi.json", links)
        self.assertEqual(
            table,
            [
                ["Chat", "2", "chat/selected.gguf", "chat/loaded.gguf"],
                ["Images", "1", "image/nested/same.sft", "image/nested/same.sft"],
            ],
        )
        self.assertEqual(
            ids.splitlines(), ["chat/loaded.gguf", "chat/selected.gguf", "image/nested/same.sft"]
        )
        self.assertNotIn(root.name, str(table) + ids + chat + images + masked)
        for code in (chat, images, masked):
            self.assertIn("Bearer YOUR_API_KEY", code)
            self.assertIn(base, code)
            ast.parse(code)
        self.assertIn("for line in response", chat)
        self.assertIn('Path("mask.png")', masked)

    def test_disabled_workspace_does_not_read_catalogue_even_with_restart_preference(self):
        shared = SimpleNamespace(
            server_app=SimpleNamespace(state=SimpleNamespace()),
            local_url="http://localhost:9250/",
            settings=SimpleNamespace(default_settings={"startup_api": True}),
        )
        with (
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(
                api_runtime, "models", side_effect=AssertionError("disabled API inventory")
            ),
        ):
            status, group, *rest = workspace._refresh("en")
        self.assertIn("disabled", status)
        self.assertFalse(group["visible"])
        self.assertFalse(any(rest))

    def test_client_examples_serialize_payloads_without_credentials_or_shell_quoting(self):
        base = workspace.api_base_url("http://username:secret@[::]:9250/?key=ignored")
        self.assertEqual(base, "http://[::1]:9250/v1")
        self.assertEqual(workspace.api_base_url("http://0.0.0.0:9230/"), "http://127.0.0.1:9230/v1")
        for path, schema, payload, mask in (
            (
                "/chat/completions",
                ChatRequest,
                {"messages": [{"role": "user", "content": "hello"}], "stream": True},
                False,
            ),
            (
                "/images/generations",
                ImageRequest,
                {"model": "image/default", "prompt": "A tree"},
                True,
            ),
        ):
            code = workspace.request_example(base, path, payload, False, mask=mask)
            response = Mock()
            response.__enter__ = Mock(return_value=iter((b"data: [DONE]\n",)))
            response.__exit__ = Mock()
            if mask:
                response.__enter__.return_value = SimpleNamespace(read=lambda: '{"data": []}')
            with (
                patch("urllib.request.urlopen", return_value=response) as open_url,
                patch.object(Path, "read_bytes", return_value=b"fixture"),
                patch("builtins.print"),
            ):
                exec(compile(code, "workspace-example", "exec"), {})
            request = open_url.call_args.args[0]
            self.assertEqual(request.full_url, base + path)
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("secret", code)
            schema.model_validate(json.loads(request.data))

    def test_connections_workspace_uses_private_events_and_native_copy_controls(self):
        with gr.Blocks(analytics_enabled=False) as demo:
            with gr.Tab("Connections") as tab:
                workspace.create_api_workspace(tab)
        configuration = demo.get_config_file()
        events = configuration["dependencies"]
        self.assertEqual(len(events), 4)
        self.assertTrue(all(event["api_visibility"] == "private" for event in events))
        copy = [row for row in configuration["components"] if row["type"] in ("code", "textbox")]
        self.assertEqual(len(copy), 5)
        self.assertTrue(all(row["props"]["buttons"] == ["copy"] for row in copy))
