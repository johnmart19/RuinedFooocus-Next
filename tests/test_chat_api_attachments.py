"""Test the authenticated Chat attachment wire contract and serial worker boundary."""

from base64 import b64encode
from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
import queue
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from weakref import ref

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from ruinedfooocus.app import api_runtime
import ruinedfooocus.app as app_modules
from ruinedfooocus.features.chat.attachment_retrieval import InlineAttachmentTurn
from ruinedfooocus.features.chat.attachments import MAX_FILE_BYTES
from ruinedfooocus.features.chat.model_preparation import ModelPreparation
import ruinedfooocus.features.generation as generation_modules
from ruinedfooocus.integrations.openai_api import create_apps


def source(name, text):
    raw = text.encode() if isinstance(text, str) else text
    return {"name": name, "data": b64encode(raw).decode("ascii")}


class ChatApiAttachmentsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = ModelPreparation(Path("fixture.gguf"), (), '{"llm_n_ctx":8192}')
        self.settings = SimpleNamespace(default_settings={})
        self.shared = SimpleNamespace(settings=self.settings, path_manager=object())
        self.pipeline = SimpleNamespace(
            llm=SimpleNamespace(handle_chat_completions=Mock(side_effect=self.complete)),
            model_settings="same",
            current_model_settings=Mock(return_value="same"),
            load_base_model=Mock(),
            vision_projector="fixture-projector.gguf",
        )
        self.pipelines = SimpleNamespace(update=Mock(return_value=self.pipeline))
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": self.shared,
                    "ruinedfooocus.features.generation.pipelines": self.pipelines,
                },
            )
        )
        self.stack.enter_context(patch.object(app_modules, "state", self.shared, create=True))
        self.stack.enter_context(
            patch.object(generation_modules, "pipelines", self.pipelines, create=True)
        )
        self.stack.enter_context(
            patch("ruinedfooocus.features.chat.llama_models.local_models", return_value={})
        )
        self.capture_model = self.stack.enter_context(
            patch(
                "ruinedfooocus.features.chat.model_preparation.capture_model",
                return_value=self.model,
            )
        )
        self.rag = object()
        self.capture_rag = self.stack.enter_context(
            patch(
                "ruinedfooocus.features.chat.rag_preparation.capture_chat_rag",
                return_value=self.rag,
            )
        )
        self.prepare = self.stack.enter_context(
            patch("ruinedfooocus.features.chat.model_preparation.prepare_chat_request")
        )
        self.rank = self.stack.enter_context(
            patch(
                "ruinedfooocus.features.chat.attachment_retrieval.rank_passages",
                side_effect=self.rerank,
            )
        )
        self.submit = self.stack.enter_context(
            patch.object(api_runtime, "submit", side_effect=self.execute_operation)
        )
        backend = SimpleNamespace(
            chat=api_runtime.chat,
            resolve_model=Mock(return_value=("chat/fixture", {"path": "fixture.gguf"})),
        )
        api, _ = create_apps(backend, "fixture-key")
        app = FastAPI()
        app.mount("/v1", api)
        self.client = self.stack.enter_context(TestClient(app))
        self.headers = {"Authorization": "Bearer fixture-key"}
        self.route = "/v1/chat/completions"
        self.question = "What is the harbor code and folder color?"
        self.messages = [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": self.question},
        ]
        self.attachments = [
            source("note.md", "# Harbor\nThe code is HARBOR-7."),
            source("note.md", "The folder color is purple."),
        ]

    def rerank(self, candidates, question, rag, item, check):
        self.assertIs(rag, self.rag)
        self.assertIn(self.question, question)
        self.pipelines.update.assert_called()
        check()
        return candidates

    @staticmethod
    def execute_operation(operation):
        # Keep unittest.TestCase.run intact so the runner executes these cases.
        job = api_runtime.Job(operation)
        if hasattr(operation, "prepare"):
            try:
                operation.prepare(job)
            except RuntimeError as error:
                job.fail(error)
                return job
        job.run(1)
        return job

    @staticmethod
    def complete(request, callback):
        metadata = {"id": "chatcmpl-fixture", "created": 42, "object": "chat.completion.chunk"}
        callback(
            dict(
                metadata,
                choices=[
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "HARBOR-7; purple."},
                        "finish_reason": None,
                    }
                ],
            )
        )
        callback(
            dict(
                metadata,
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "fixture",
                                    "function": {"name": "generate_image", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            )
        )

    def post(self, **overrides):
        payload = {"messages": self.messages, "rf_attachments": self.attachments, **overrides}
        return self.client.post(self.route, headers=self.headers, json=payload)

    def test_grounded_json_and_legacy_requests_are_independent(self):
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        expected = [
            {"id": "D1", "name": "note.md", "start": 1, "end": 2},
            {"id": "D2", "name": "note.md", "start": 1, "end": 1},
        ]
        self.assertEqual(response.json()["rf_attachment_sources"], expected)
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        native = self.pipeline.llm.handle_chat_completions.call_args.args[0]
        self.assertEqual(native["messages"][0], self.messages[0])
        self.assertIn('"id": "D1"', native["messages"][-1]["content"])
        self.assertIn("The folder color is purple.", native["messages"][-1]["content"])
        self.assertNotIn("rf_attachments", native)
        self.assertNotIn("tools", native)
        self.assertFalse(self.capture_model.call_args.kwargs["vision"])
        self.prepare.assert_called_once()
        self.assertIs(self.prepare.call_args.args[0]["_rag_preparation"], self.rag)

        self.prepare.reset_mock()
        response = self.client.post(
            self.route, headers=self.headers, json={"messages": self.messages}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("rf_attachment_sources", response.json())
        self.assertIn("tool_calls", response.json()["choices"][0]["message"])
        self.assertEqual(
            self.pipeline.llm.handle_chat_completions.call_args.args[0]["messages"], self.messages
        )
        self.prepare.assert_not_called()

    def test_api_model_load_captures_request_identity_and_native_cancellation(self):
        self.pipeline.model_settings = "old-load"
        job = SimpleNamespace(cancelled=Event())

        def load(model, **options):
            self.assertEqual(model, str(self.model.model))
            self.assertIs(options["prepared"], self.model)
            self.assertEqual(options["load_id"], "api-chat-41")
            options["check_cancelled"]()
            job.cancelled.set()
            options["check_cancelled"]()

        self.pipeline.load_base_model.side_effect = load
        with patch.object(api_runtime, "submit", side_effect=lambda operation: operation):
            operation = api_runtime.chat(
                {"model": "chat/fixture", "messages": self.messages}, {"path": "fixture.gguf"}
            )
        from concurrent.futures import CancelledError

        with self.assertRaisesRegex(CancelledError, "cancelled"):
            operation(job, 41)
        self.pipeline.llm.handle_chat_completions.assert_not_called()

    def test_first_sse_chunk_carries_metadata_and_preserves_vision_input(self):
        buffer = BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, format="PNG")
        image = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + b64encode(buffer.getvalue()).decode(),
                "detail": "high",
            },
        }
        messages = [{"role": "user", "content": [{"type": "text", "text": self.question}, image]}]
        response = self.post(messages=messages, stream=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        lines = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in lines[:-1]]
        self.assertEqual([item["id"] for item in chunks[0]["rf_attachment_sources"]], ["D1", "D2"])
        self.assertNotIn("rf_attachment_sources", chunks[1])
        self.assertEqual(chunks[1]["choices"][0]["delta"], {})
        native = self.pipeline.llm.handle_chat_completions.call_args.args[0]
        self.assertIn("HARBOR-7", native["messages"][-1]["content"][0]["text"])
        self.assertEqual(native["messages"][-1]["content"][1], image)
        self.assertTrue(self.capture_model.call_args.kwargs["vision"])

    def test_no_matching_passages_returns_empty_metadata(self):
        self.rank.side_effect = None
        self.rank.return_value = ()
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["rf_attachment_sources"], [])
        self.assertEqual(
            self.pipeline.llm.handle_chat_completions.call_args.args[0]["messages"], self.messages
        )

    def test_pdf_and_docx_bytes_share_actual_json_and_sse_grounding_contract(self):
        from test_document_attachments import docx, pdf

        attachments = [
            source("harbor.pdf", pdf("The code is HARBOR-7.")),
            source("folder.docx", docx("The folder color is purple.")),
        ]
        expected = [
            {"id": "D1", "name": "harbor.pdf", "start": 1, "end": 2, "page": 1},
            {"id": "D2", "name": "folder.docx", "start": 1, "end": 1},
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.post(rf_attachments=attachments, stream=stream)
                self.assertEqual(response.status_code, 200, response.text)
                if stream:
                    chunks = [
                        line.removeprefix("data: ")
                        for line in response.text.splitlines()
                        if line.startswith("data: ")
                    ]
                    self.assertEqual(chunks[-1], "[DONE]")
                    metadata = json.loads(chunks[0])
                else:
                    metadata = response.json()
                self.assertEqual(metadata["rf_attachment_sources"], expected)
                content = self.pipeline.llm.handle_chat_completions.call_args.args[0]["messages"][
                    -1
                ]["content"]
                self.assertIn("HARBOR-7", content)
                self.assertIn("purple", content)
                self.assertNotIn(attachments[0]["data"], content)

    def test_csv_and_json_are_literal_lines_with_provenance_in_json_and_sse(self):
        csv = 'place,code,formula\nHarbor,HARBOR-7,=HYPERLINK("https://example.invalid/")'
        json_text = '{\n  "$ref":"file:///private/notes.json",\n  "folder":"purple",'
        attachments = [source("harbor.csv", csv), source("folder.JSON", json_text)]
        expected = [
            {"id": "D1", "name": "harbor.csv", "start": 1, "end": 2},
            {"id": "D2", "name": "folder.JSON", "start": 1, "end": 3},
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.post(rf_attachments=attachments, stream=stream)
                self.assertEqual(response.status_code, 200, response.text)
                if stream:
                    chunks = [
                        line.removeprefix("data: ")
                        for line in response.text.splitlines()
                        if line.startswith("data: ")
                    ]
                    self.assertEqual(chunks[-1], "[DONE]")
                    metadata = json.loads(chunks[0])
                else:
                    metadata = response.json()
                self.assertEqual(metadata["rf_attachment_sources"], expected)
                candidates = self.rank.call_args.args[0]
                self.assertEqual([candidate.text for candidate in candidates], [csv, json_text])

    def test_auth_tool_conflicts_and_wire_validation(self):
        response = self.client.post(
            self.route, json={"messages": self.messages, "rf_attachments": self.attachments}
        )
        self.assertEqual(response.status_code, 401)
        invalid = (
            {"rf_attachments": []},
            {"rf_attachments": self.attachments * 3},
            {"rf_attachments": [source("../private.txt", "private")]},
            {"rf_attachments": [source("../private.json", "{}")]},
            {"rf_attachments": [source("bad.csv", b"col,value\n\x00\x01")]},
            {"rf_attachments": [source("bad.json", b'{"value":"\xff\xfe"}')]},
            {"rf_attachments": [source("nested.rag", '{"version":1,"sources":[]}')]},
            {"rf_attachments": [{**self.attachments[0], "path": "do-not-echo-this-path"}]},
            {"rf_attachments": [{"name": "wrong.txt", "data": "not base64"}]},
            {"rf_attachments": [source("huge.csv", b"a" * (MAX_FILE_BYTES + 1))]},
            {"messages": [{"role": "assistant", "content": "Existing answer"}]},
            {"tools": [{"type": "function", "function": {"name": "generate_image"}}]},
            {"function_call": "auto"},
        )
        for request in invalid:
            with self.subTest(fields=tuple(request)):
                response = self.post(**request)
                self.assertEqual(response.status_code, 400, response.text[:500])
                self.assertNotIn("do-not-echo-this-path", response.text)
        self.submit.assert_not_called()
        self.capture_model.assert_not_called()

    def test_cancelled_unstarted_job_discards_captured_inline_text(self):
        self.submit.side_effect = api_runtime.Job
        recorded = []
        capture = InlineAttachmentTurn.capture

        def track(*args):
            turn = capture(*args)
            recorded.append(ref(turn))
            return turn

        with patch.object(InlineAttachmentTurn, "capture", side_effect=track):
            job = api_runtime.chat(
                {
                    "model": "chat/fixture",
                    "messages": self.messages,
                    "rf_attachments": self.attachments,
                },
                {"path": "fixture.gguf"},
            )
        self.assertIsNotNone(recorded[0]())
        job.cancelled.set()
        self.assertIsNone(recorded[0]())
        self.assertEqual(job.events.get_nowait(), ("error", "Request cancelled."))
        self.prepare.assert_not_called()

    def test_cancellation_during_ranking_never_dispatches_to_the_native_client(self):
        retained = []

        def cancel(candidates, question, rag, job, check):
            retained.extend(ref(passage) for passage in candidates)
            job.cancelled.set()
            check()

        self.rank.side_effect = cancel
        response = self.post()
        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(response.json()["error"]["message"], "Request cancelled.")
        self.pipeline.llm.handle_chat_completions.assert_not_called()
        # The rank mock records the call itself; remove that test-owned reference.
        self.rank.reset_mock()
        self.assertTrue(retained)
        self.assertTrue(all(passage() is None for passage in retained))


if __name__ == "__main__":
    unittest.main()
