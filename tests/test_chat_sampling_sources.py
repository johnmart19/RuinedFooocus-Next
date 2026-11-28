"""Trusted source discovery is explicit preparation, never inference/render I/O."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gguf

from ruinedfooocus.features.chat.model_preparation import runtime_selection_key
from ruinedfooocus.features.chat import sampling_sources as sources
from ruinedfooocus.features.chat.model_profiles import ProfiledChatClient
from ruinedfooocus.features.chat.ui_sampling import (
    active_sampling_status,
    source_evidence,
    active_runtime_status,
)
from ruinedfooocus.features.models.model_identity import file_signature
from ui_test_helpers import function_from

REVISION = "a" * 40
REPO = "Publisher/Variant-GGUF"
PROFILE = {
    "identity": {"gguf_architecture": "qwen3"},
    "publisher_settings": {"generation_config": {"temperature": 0.6, "top_p": 0.9}},
}


class Registry:
    def __init__(self, row):
        self.row = row

    def entries(self, **kwargs):
        return (self.row,)

    def profile(self, name):
        return PROFILE if name == "language:variant" else {}


class ChatSamplingSourceTests(unittest.TestCase):
    def test_active_status_reports_cpu_fallback_instead_of_only_model_name(self):
        text = active_runtime_status(
            "model.gguf",
            {
                "runtime_name": "xllamacpp",
                "runtime_label": "CPU",
                "fallback_reason": "Vulkan has no visible GPU",
            },
        )
        self.assertIn("model.gguf", text)
        self.assertIn("xllamacpp · CPU", text)
        self.assertIn("Vulkan has no visible GPU", text)

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cache = self.root / "sources"
        self.row = dict(
            id="registered:variant",
            profile_id="language:variant",
            role="language_model",
            chat_weight=True,
            artifact=dict(
                bytes=999999,
                sha256="0" * 64,
                pinned_url=f"https://huggingface.co/{REPO}/resolve/{REVISION}/known.gguf",
            ),
        )
        self.registry = Registry(self.row)
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_registry.load_registry",
                return_value=self.registry,
            )
        )
        self.enterContext(patch.dict("os.environ", {"RF_OFFLINE": "0", "HF_HUB_OFFLINE": "0"}))

    def gguf(self, name="renamed.gguf", architecture="qwen3", repository=REPO, label="quant-A"):
        path = self.root / name
        writer = gguf.GGUFWriter(str(path), architecture)
        writer.add_string("general.source.huggingface.repository", repository)
        writer.add_string("general.name", label)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path

    def refresh(self, path, **kwargs):
        return sources.refresh_sampling_source(path, cache_dir=self.cache, offline=False, **kwargs)

    def remote(self, revision=REVISION, repository=REPO):
        return dict(
            source_url=f"https://huggingface.co/{repository}/blob/{revision}/README.md",
            generationConfig=dict(
                temperature=0.2, repetition_penalty=1.1, top_p=False, prompt="ignore controls"
            ),
        )

    def test_exact_registered_content_survives_rename_and_offline_without_network(self):
        path = self.gguf(repository="https://127.0.0.1/private")
        self.row["artifact"].update(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        with patch(
            "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
            side_effect=AssertionError("offline network"),
        ):
            source = sources.refresh_sampling_source(path, cache_dir=self.cache, offline=True)
        self.assertEqual(source["status"], "reviewed")
        self.assertIs(sources.reviewed_source_artifact(path, source), self.row)
        self.assertEqual(source["repositories"], [])
        path.unlink()  # GGUF reader releases its Windows mmap.

    def test_different_quantizations_in_reviewed_variant_revision_share_sampling_only(self):
        checksums = []
        with patch(
            "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
            return_value=self.remote(),
        ) as remote:
            for name, label in (("unexpected-A.gguf", "Q4"), ("unexpected-B.gguf", "Q8")):
                path = self.gguf(name, label=label)
                source = self.refresh(path)
                checksums.append(source["sha256"])
                self.assertEqual(source["status"], "reviewed_variant")
                self.assertIsNone(sources.reviewed_source_artifact(path, source))
                client = ProfiledChatClient(Mock(), None, {}, sampling_source=source)
                self.assertEqual(client.parameters({})["temperature"], 0.6)
                self.assertIsNone(client.reasoning_capability["control"])
                self.assertIn("variant sampling baseline", client.sampling_state()["message"])
                self.assertEqual(
                    source["observations"][0]["generation_config"],
                    {"temperature": 0.2, "repeat_penalty": 1.1},
                )
            self.assertEqual(remote.call_count, 2)
        self.assertNotEqual(*checksums)

    def test_family_repository_wrong_revision_and_wrong_architecture_do_not_qualify(self):
        for architecture, response in (
            ("llama", self.remote()),
            ("qwen3", self.remote(revision="b" * 40)),
            ("qwen3", self.remote(repository="Unreviewed/FineTune")),
        ):
            path = self.gguf(architecture=architecture)
            with patch(
                "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
                return_value=response,
            ):
                source = self.refresh(path, force=True)
            self.assertEqual(source["status"], "source_unverified")
            client = ProfiledChatClient(Mock(), None, {}, sampling_source=source)
            self.assertEqual(client.parameters({}), {})
            self.assertIn("unverified", client.sampling_state()["message"])

    def test_cache_offline_failure_retention_and_signature_replacement(self):
        path = self.gguf()
        with patch(
            "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
            return_value=self.remote(),
        ):
            source = self.refresh(path)
        with (
            patch(
                "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
                side_effect=AssertionError("cache must not fetch"),
            ),
            patch(
                "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                side_effect=AssertionError("cache must not hash"),
            ),
        ):
            self.assertEqual(self.refresh(path), source)
            self.assertEqual(
                sources.refresh_sampling_source(
                    path, cache_dir=self.cache, offline=True, force=False
                ),
                source,
            )
            self.assertEqual(
                sources.refresh_sampling_source(
                    path, cache_dir=self.cache, offline=True, force=True
                ),
                source,
            )
        with patch(
            "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
            return_value=None,
        ):
            retained = self.refresh(path, force=True)
        self.assertEqual(retained["status"], "reviewed_variant")
        self.assertEqual(retained["last_lookup"], "unavailable")
        path.write_bytes(path.read_bytes() + b"replacement")
        self.assertEqual(sources.cached_sampling_source(path, cache_dir=self.cache), {})
        self.assertEqual(sources.verified_sampling_profile(source), {})

    def test_effective_status_matches_request_precedence_and_never_mutates_messages(self):
        profile = {"publisher_settings": {"generation_config": {"temperature": 0.6, "top_p": 0.9}}}
        with patch("ruinedfooocus.features.chat.model_profiles.profile_for", return_value=profile):
            client = ProfiledChatClient(
                Mock(),
                self.row,
                {"llm_sampling": {"temperature": 0}, "llama_server_args": "--top-p 0.3"},
            )
        request = dict(
            _rf_task="prompt_enhancement",
            top_p=0,
            messages=[{"role": "user", "content": "Keep exactly"}],
        )
        original = deepcopy(request)
        for task in ("chat", "prompt_enhancement"):
            for explicit in ({}, {"top_p": 0}):
                data = dict(_rf_task=task, stream=False, max_tokens=0, **explicit)
                state = client.sampling_state(data)
                actual = client.parameters(data)
                self.assertEqual(
                    state["effective"], {key: actual[key] for key in state["effective"]}
                )
                self.assertEqual(
                    state["origins"]["top_p"], "request_overrides" if explicit else "native_cli"
                )
                self.assertIs(actual["stream"], False)
                self.assertEqual(actual["max_tokens"], 0)
        with patch.object(
            sources, "refresh_sampling_source", side_effect=AssertionError("inference network/hash")
        ):
            effective = client.sampling_state(request)
            actual = client.parameters(request)
            rendered = active_sampling_status(effective)
        self.assertEqual(
            {key: actual[key] for key in effective["effective"]}, effective["effective"]
        )
        self.assertEqual(
            effective["origins"], {"temperature": "saved_overrides", "top_p": "request_overrides"}
        )
        self.assertEqual(actual["messages"], original["messages"])
        self.assertEqual(request, original)
        self.assertIn("Saved override", rendered)
        self.assertIn("Explicit request", rendered)
        client = ProfiledChatClient(
            Mock(), None, {"llm_use_model_defaults": False, "llm_sampling": {"temperature": 0}}
        )
        self.assertEqual(client.parameters({})["temperature"], 0)

    def test_repository_hints_never_authorize_arbitrary_urls_and_render_is_read_only(self):
        for value in (
            "http://huggingface.co/a/b",
            "https://huggingface.co@127.0.0.1/a/b",
            "https://huggingface.co/a/b?token=x",
            "file:///a/b",
            "https://evil/a/b",
        ):
            self.assertIsNone(sources.huggingface_repository(value))
        self.assertEqual(sources.huggingface_repository(f"https://huggingface.co/{REPO}"), REPO)
        path = self.gguf()
        with patch.object(
            sources,
            "refresh_sampling_source",
            side_effect=AssertionError("render must not prepare"),
        ):
            self.assertIn("not prepared", source_evidence(path)[0][1])
        self.assertFalse(self.cache.exists())

    def test_explicit_load_prepares_sources_but_ordinary_chat_does_not(self):
        from ruinedfooocus.features.chat.model_preparation import prepare_chat_request

        path = self.gguf()
        prepared = SimpleNamespace(model=path, artifacts=())
        with (
            patch("ruinedfooocus.features.chat.rag_preparation.prepare_rag"),
            patch.object(sources, "prepare_sampling_source") as prepare,
        ):
            prepare_chat_request({"load_only": True, "_chat_preparation": prepared})
            prepare.assert_called_once_with(path)
            prepare.reset_mock()
            prepare_chat_request({"_chat_preparation": prepared})
            prepare.assert_not_called()

    def test_selected_refresh_updates_only_evidence_without_replacing_active_state(self):
        path = self.gguf()
        evidence = Mock(return_value=[["Selected", "candidate B"]])
        callback = function_from(
            "ruinedfooocus/features/chat/ui_model_diagnostics.py",
            "refresh_sampling_recommendations",
            dict(
                Path=Path,
                selected_evidence=evidence,
                path_manager=SimpleNamespace(DOWNLOADABLE_FILES={}),
                catalogue_local_file=lambda *args: None,
                _roots=lambda: {},
            ),
        )
        with patch.object(sources, "prepare_sampling_source", return_value={}) as prepare:
            self.assertEqual(callback(str(path)), [["Selected", "candidate B"]])
            prepare.assert_called_once_with(path, force=True)
        with patch.object(
            sources,
            "prepare_sampling_source",
            side_effect=AssertionError("missing candidate must not prepare"),
        ):
            self.assertIn("installed", callback(None)[-1][-1])

    def test_successful_load_displays_worker_sampling_snapshot_not_selected_metadata(self):
        from ruinedfooocus.features.chat.model_preparation import ChatModelSession

        path = self.gguf()
        state = dict(
            status="user_overrides",
            effective={"temperature": 0},
            origins={"temperature": "saved_overrides"},
        )
        prepared = SimpleNamespace(model=path, present=True, settings={})
        controls = {
            name: name
            for name in (
                "llama_active_model",
                "active_status",
                "sampling_status",
                "chat_reasoning_capability",
                "llama_model_status",
                "llama_msg",
                "llama_use",
                "cancel_load",
                "llama_download",
                "preparation_status",
            )
        }
        scope = dict(
            Path=Path,
            settings=SimpleNamespace(default_settings={}),
            runtime_selection_key=runtime_selection_key,
            path_manager=object(),
            capture_model=lambda *a, **kw: prepared,
            local_models=lambda: {},
            model_catalogue=lambda source: {"Model": [("quant", str(path))]},
            ModelTransfer=lambda value: SimpleNamespace(result=lambda: value),
            worker=SimpleNamespace(
                add_task=lambda data: 1,
                task_result=lambda task: ("results", {"ready": True, "sampling_state": state}),
            ),
            gr=SimpleNamespace(update=lambda **kw: kw, Request=object),
            **controls,
        )
        import ast
        import importlib

        # Real helpers imported by the Chat UI; the fixture replaces only the
        # worker, files and widgets. Modules needing ComfyUI stay out of CPU checks.
        source = "ruinedfooocus/features/chat/ui.py"
        helpers = {}
        for node in ast.parse(Path(source).read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                try:
                    module = importlib.import_module(node.module)
                except ImportError:
                    continue
                for alias in node.names:
                    if hasattr(module, alias.name):
                        helpers[alias.asname or alias.name] = getattr(module, alias.name)
        # Load/Download button projection has its own tests.
        scope = {**helpers, "action_updates": lambda *args, **kwargs: {}, **scope}
        for helper in ("error_controls", "settle_superseded_load"):
            function_from(source, helper, scope)
        callback = function_from(source, "use_model", scope)
        session = ChatModelSession()
        session.active = "previous-model"
        with patch.object(
            sources,
            "cached_sampling_source",
            side_effect=AssertionError("active status must use actual loaded snapshot"),
        ):
            updates = list(callback(str(path), "Model", "Local models", "Load", session))
        self.assertEqual(session.active, str(path))
        # The last update only refreshes Load/Download buttons.
        status = next(u["sampling_status"] for u in reversed(updates) if "sampling_status" in u)
        self.assertIn("Your overrides", status)
        self.assertIn("| temperature | 0 | Saved override |", status)

    def test_malformed_cached_repository_fields_and_unknown_status_render_safely(self):
        path = self.gguf()
        cache = sources._cache_file(path, self.cache)
        cache.parent.mkdir(parents=True)
        for repositories in (1, "Publisher/Variant", ["http://127.0.0.1"]):
            cache.write_text(
                json.dumps(
                    dict(
                        schema=sources.SCHEMA,
                        signature=file_signature(path),
                        status="offline",
                        repositories=repositories,
                    )
                )
            )
            self.assertEqual(sources.cached_sampling_source(path, cache_dir=self.cache), {})
        with patch(
            "ruinedfooocus.features.chat.ui_sampling.cached_sampling_source",
            return_value={"status": "future_status"},
        ):
            self.assertIn("not prepared", source_evidence(path)[0][1])

    def test_same_model_refreshes_serialize_and_other_models_remain_independent(self):
        a = self.gguf("A.gguf", label="A")
        b = self.gguf("B.gguf", label="B")
        a_hash = hashlib.sha256(a.read_bytes()).hexdigest()
        entered, release, same_entered, second_started = (threading.Event() for _ in range(4))
        counter = 0
        guard = threading.Lock()

        def remote(repository, checksum):
            nonlocal counter
            if checksum == a_hash:
                with guard:
                    counter += 1
                    first = counter == 1
                if first:
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("Failed-refresh barrier was not released")
                    return None
                same_entered.set()
            return self.remote()

        def same_model():
            second_started.set()
            return self.refresh(a, force=True)

        with (
            patch(
                "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
                side_effect=remote,
            ),
            ThreadPoolExecutor(max_workers=3) as executor,
        ):
            failed = executor.submit(self.refresh, a, force=True)
            try:
                self.assertTrue(entered.wait(5))
                other = executor.submit(self.refresh, b, force=True)
                self.assertEqual(other.result(timeout=5)["status"], "reviewed_variant")
                same = executor.submit(same_model)
                self.assertTrue(second_started.wait(5))
                self.assertFalse(same_entered.wait(0.1))
            finally:
                release.set()
            failed.result(timeout=5)
            self.assertEqual(same.result(timeout=5)["status"], "reviewed_variant")
        self.assertEqual(
            sources.cached_sampling_source(a, cache_dir=self.cache)["status"], "reviewed_variant"
        )


if __name__ == "__main__":
    unittest.main()
