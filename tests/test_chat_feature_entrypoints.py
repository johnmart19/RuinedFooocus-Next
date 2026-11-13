"""Real public Chat contracts must work with their declared local dependencies."""

from base64 import b64encode
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def tiny_gguf(path):
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + string("general.architecture")
        + struct.pack("<I", 8)
        + string("llama")
    )
    return path


class ChatFeatureEntrypointTests(unittest.TestCase):
    def test_profiled_client_constructs_with_source_sampling(self):
        from ruinedfooocus.features.chat.model_profiles import ProfiledChatClient
        from ruinedfooocus.features.models.model_capabilities import RuntimeIdentity
        from ruinedfooocus.features.models.model_identity import file_signature
        from ruinedfooocus.features.models.model_registry import load_registry

        artifact = load_registry().artifact(
            "existing:llm/Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf"
        )
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_gguf(Path(directory) / "identity-only.gguf")
            for selected in (None, artifact):
                for defaults in (False, True):
                    with self.subTest(known=selected is not None, defaults=defaults):
                        native = Mock()
                        client = ProfiledChatClient(
                            native,
                            selected,
                            {"llm_use_model_defaults": defaults},
                            initialized=True,
                            template="{{ tools }} tool_calls {% if enable_thinking %}x{% endif %}",
                            resources=(file_signature(model),),
                            initialized_runtime=RuntimeIdentity("llama.cpp", "b12366"),
                        )
                        messages = [{"role": "user", "content": "Literal café 0007"}]
                        client.handle_chat_completions(
                            {"messages": messages, "temperature": 0, "top_k": 0}
                        )
                        sent = native.handle_chat_completions.call_args.args[0]
                        self.assertEqual(sent["messages"], messages)
                        self.assertEqual((sent["temperature"], sent["top_k"]), (0, 0))
                        self.assertEqual(
                            client.image_tool_capability["available"], selected is not None
                        )
                        self.assertEqual(
                            client.reasoning_capability["control"],
                            "enable_thinking" if selected is not None else None,
                        )
                        if not defaults:
                            self.assertEqual(client.sampling_state()["effective"], {})
                        client.close()

    def test_chat_capture_imports_url_helper_without_embeds(self):
        from ruinedfooocus.app import resource_lifecycle as life, state as shared
        from ruinedfooocus.core.paths import data_path
        from ruinedfooocus.core.util import url_to_filename
        from ruinedfooocus.features.chat.attachments import AttachmentSession
        from ruinedfooocus.features.chat.model_preparation import capture_chat_request
        from ruinedfooocus.features.chat import rag_preparation

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, RF_OFFLINE="1", RF_DATA_DIR=directory),
        ):
            model = tiny_gguf(Path(directory) / "request.gguf")
            values = {"llama_localfile": str(model), "llama_mmproj": "None", "cpu_everywhere": True}
            plain = capture_chat_request({}, values, shared.path_manager, {})
            self.assertEqual(plain["_chat_preparation"].model, model.resolve())
            self.assertEqual(plain["_context_preparation"], ())

            # Capture needs the finite local snapshot identities, never embedding execution.
            cache = data_path("cache", "rag", rag_preparation.MODEL_REVISION)
            for name, _, _ in rag_preparation.MODEL_FILES:
                target = cache / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"identity fixture, never loaded")
            owner = ("ui", "entrypoints", "chat-capture")
            attachments = AttachmentSession(owner)
            attachments.attach(
                0, [{"name": "note.txt", "data": b64encode(b"literal 0007").decode()}]
            )
            item = life.registry.create(owner=owner)
            self.addCleanup(life.registry.finish, item)
            url = "https://example.invalid/reference.txt"
            with life.job_scope(item):
                captured = capture_chat_request(
                    {
                        "history": [{"role": "user", "content": "What is literal?"}],
                        "embed": json.dumps([["url", url]]),
                        "_chat_attachments": attachments.turn(),
                    },
                    values,
                    shared.path_manager,
                    {},
                    launch_args=SimpleNamespace(cpu=True, directml=None, offline=True),
                )
            request = captured["_context_preparation"][0]
            self.assertEqual(request.url, url)
            self.assertEqual(
                request.destination, data_path("cache", "embeds", url_to_filename(url))
            )
            self.assertFalse(request.destination.exists())
            self.assertEqual(captured["_chat_attachments"].files[0].text, "literal 0007")
            self.assertEqual(captured["_rag_preparation"].settings, {"gpu": False, "offline": True})

    def test_runtime_choices_and_metadata_reach_declared_state(self):
        from ruinedfooocus.app import state as shared
        from ruinedfooocus.backends.llama import llama_runtime_info as info
        from ruinedfooocus.core.paths import data_path

        previous = shared.state.get("pipeline")
        self.addCleanup(shared.state.__setitem__, "pipeline", previous)
        shared.state["pipeline"] = SimpleNamespace(
            llm=object(), runtime_name="xllamacpp", runtime_label="CPU"
        )
        self.assertIn(("xllamacpp [CPU]", "xllamacpp"), info.runtime_choices())
        name = "llama-server.exe" if os.name == "nt" else "llama-server"
        binary = data_path("cache", "llama.cpp", "b-entrypoint", "CPU", name)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"not an executable; passive identity only")
        self.addCleanup(binary.unlink)
        with patch.object(info, "run_bounded", side_effect=AssertionError("No native query")):
            self.assertIn("b-entrypoint / CPU", info.runtime_metadata())
            self.assertIn("CPU authority is active", info.runtime_info(cpu=True))


if __name__ == "__main__":
    unittest.main()
