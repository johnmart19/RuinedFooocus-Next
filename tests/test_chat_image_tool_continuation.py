"""Actual Chat continuation producer/caller checks for the atomic worker feature."""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_chat_image_tools import CAPABILITY, call, context
from ruinedfooocus.features.chat.chat_tools import (
    ChatPermissionSession,
    IMAGE_TOOL_GUIDANCE,
    ImageToolContinuation,
)

ROOT = Path(__file__).resolve().parents[1]


def process_method():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
    )
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
    )
    return next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "process"
    )


def run_turn(chunks, *, tool_context=None, capability=None, data=None, changed=False):
    requests = []
    worker = SimpleNamespace(add_result=Mock(), check_interrupt=Mock())
    cap = CAPABILITY if capability is None else capability

    def answer(request, callback):
        requests.append(request)
        for chunk in chunks:
            callback(chunk)

    adapter = SimpleNamespace(handle_chat_completions=answer, reasoning_state=lambda data: {})
    pipeline = SimpleNamespace(
        llm=adapter,
        model_settings="same",
        current_model_settings=lambda *args, **kwargs: "same",
        vision_projector=None,
        embeddings=None,
        embeddings_hash="[]",
        image_tool_capability=Mock(
            side_effect=[cap, {**cap, "binding": "changed"}] if changed else None, return_value=cap
        ),
    )
    scope = {
        "settings": SimpleNamespace(default_settings={"enable_llm_tools": True}),
        "DEFAULT_MODEL": "fixture.gguf",
        "worker": worker,
        "selected_image_reference": lambda *args: (None, ""),
        "json": json,
        "IMAGE_TOOL_GUIDANCE": IMAGE_TOOL_GUIDANCE,
        "re": re,
        "Path": Path,
        "transient_vision_messages": lambda messages: messages,
        "TimeIt": lambda *args: nullcontext(),
        "image_tool_history": lambda *args: [],
    }
    exec(
        compile(ast.Module(body=[process_method()], type_ignores=[]), "llama_pipeline", "exec"),
        scope,
    )
    values = {
        "task_id": 17,
        "model": "fixture.gguf",
        "system": "You are Cat. Use only meows.",
        "history": [{"role": "user", "content": "Create a new image"}],
        "enable_image_tools": True,
        "_chat_preparation": SimpleNamespace(vision=False, settings={"enable_llm_tools": True}),
        **(data or {}),
    }
    if tool_context is not None:
        values["_image_tool_context"] = tool_context
        values["_chat_tool_permissions"] = tool_context.permissions
    return scope["process"](pipeline, values), requests, worker, scope


def tool_chunks(tool_call=None, *, finish="tool_calls"):
    native = call() if tool_call is None else tool_call
    return [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, **native}]}, "finish_reason": finish}]}
    ]


class ChatToolCallerTests(unittest.TestCase):
    def test_prompt_only_permission_does_not_yield_or_enter_image_continuation(self):
        policy = ChatPermissionSession().capture(
            ("test", "owner", "tab"), update_create_prompt=True
        )
        captured = context(intent="text", permissions=policy)
        result, requests, worker, _ = run_turn(
            tool_chunks(call("propose_create_prompt", {"prompt": "A blue cup"})),
            tool_context=captured,
        )
        self.assertIsInstance(result, str)
        self.assertNotIsInstance(result, ImageToolContinuation)
        self.assertEqual(
            [tool["function"]["name"] for tool in requests[0]["tools"]], ["propose_create_prompt"]
        )
        proposals = [
            item.args[2]
            for item in worker.add_result.call_args_list
            if item.args[1] == "chat_prompt_proposal"
        ]
        self.assertEqual(proposals[0]["prompt_proposal"]["prompt"], "A blue cup")

    def test_revoked_completed_image_cannot_render_through_saved_continuation(self):
        session = ChatPermissionSession()
        policy = session.capture(("test", "owner", "tab"), enable_image_generation=True)
        result, _, _, _ = run_turn(tool_chunks(), tool_context=context(permissions=policy))
        self.assertIsInstance(result, ImageToolContinuation)
        session.revoke(policy.owner)
        with self.assertRaisesRegex(ValueError, "revoked"):
            result.resume({"status": "completed", "images": ["inert.png"]})

    def test_text_prompt_only_and_critique_return_text_without_continuation(self):
        chunks = [{"choices": [{"delta": {"content": "Meow"}, "finish_reason": "stop"}]}]
        for intent in ("text", "preview", "run"):
            result, requests, worker, _ = run_turn(chunks, tool_context=context(intent=intent))
            self.assertEqual(result, "Meow")
            self.assertTrue(requests[0]["messages"][0]["content"].startswith("You are Cat."))
            self.assertFalse(
                any(
                    item.args[1] == "image_tool_result" for item in worker.add_result.call_args_list
                )
            )

    def test_valid_tool_yields_without_dispatch_and_resumes_only_once(self):
        result, requests, worker, _ = run_turn(tool_chunks(), tool_context=context())
        self.assertIsInstance(result, ImageToolContinuation)
        self.assertEqual(result.original_model, "fixture.gguf")
        self.assertFalse(result.original_vision)
        self.assertFalse(requests[0]["parallel_tool_calls"])
        self.assertFalse(
            any(item.args[1] == "image_tool_result" for item in worker.add_result.call_args_list)
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "fixture image.png"
            image.write_bytes(b"inert")
            answer = result.resume(
                {
                    "status": "completed",
                    "call_id": "call_fixture",
                    "images": [str(image)],
                    "editor": {"current_id": "image_id"},
                }
            )
            self.assertIn("![Image](gradio_api/file=", answer)
            with self.assertRaisesRegex(RuntimeError, "already"):
                result.resume({"status": "completed"})
        terminals = [
            item for item in worker.add_result.call_args_list if item.args[1] == "image_tool_result"
        ]
        self.assertEqual(terminals, [])

    def test_force_does_not_upgrade_app_text_intent_or_missing_runtime_binding(self):
        for captured, capability in (
            (context(intent="text"), CAPABILITY),
            (context(), {}),
            (None, CAPABILITY),
        ):
            result, requests, _, _ = run_turn(
                tool_chunks(),
                tool_context=captured,
                capability=capability,
                data={"force_image": True},
            )
            self.assertIsInstance(result, str)
            self.assertIn("rejected", result)
            self.assertNotIn("tools", requests[0])

    def test_parallel_truncated_unknown_and_changed_runtime_never_yield(self):
        for chunks, changed in (
            (tool_chunks(finish="length"), False),
            (tool_chunks(call("shell")), False),
            (
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, **call()},
                                        {"index": 1, **call(call_id="other")},
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                ],
                False,
            ),
            (tool_chunks(), True),
        ):
            result, _, worker, _ = run_turn(chunks, tool_context=context(), changed=changed)
            self.assertIsInstance(result, str)
            self.assertIn("rejected", result)
            self.assertEqual(
                len(
                    [
                        item
                        for item in worker.add_result.call_args_list
                        if item.args[1] == "image_tool_result"
                    ]
                ),
                1,
            )

    def test_preview_and_failed_restoration_render_without_publishing(self):
        for terminal in (
            {"status": "preview"},
            {"status": "error", "error": {"code": "restore", "message": "reload required"}},
        ):
            result, _, worker, _ = run_turn(tool_chunks(), tool_context=context(intent="preview"))
            answer = result.resume(terminal)
            self.assertNotIn("![Image]", answer)
            self.assertEqual(
                len(
                    [
                        item
                        for item in worker.add_result.call_args_list
                        if item.args[1] == "image_tool_result"
                    ]
                ),
                0,
            )

    def test_missing_completed_image_raises_before_worker_publication(self):
        result, _, worker, _ = run_turn(tool_chunks(), tool_context=context())
        with self.assertRaisesRegex(RuntimeError, "no completed image"):
            result.resume({"status": "completed"})
        self.assertFalse(
            any(item.args[1] == "image_tool_result" for item in worker.add_result.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
