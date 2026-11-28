"""Verified Qwen3 candidate settings and actual loader template handoff, CPU only."""

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.chat.model_profiles import (
    QWEN3_CANDIDATE,
    ProfiledChatClient,
    chat_load_profile,
    configure_embedded_profile,
)
from ruinedfooocus.features.models.model_registry import download_identity, load_registry


class Qwen3CandidateTests(unittest.TestCase):
    def artifact(self):
        return load_registry().artifact("existing:llm/" + QWEN3_CANDIDATE)

    def test_exact_download_joins_pinned_metadata_and_quality_stays_pending(self):
        downloads = json.loads(resource_path("model_catalogue", "llm.json").read_text())
        item = downloads["llm/" + QWEN3_CANDIDATE]
        row = self.artifact()
        self.assertEqual(download_identity(item), row["download_identity"])
        self.assertEqual((item["size"], item["sha256"]), (1834426016, row["artifact"]["sha256"]))
        self.assertIn("/90862c4b9d2787eaed51d12237eafdfe7c5f6077/", item["url"])
        profile = load_registry().profile(row["profile_id"])
        self.assertEqual(profile["qualification"]["status"], "candidate_not_default")
        self.assertFalse(profile["capabilities"]["tools"]["quantized_app_quality_verified"])

    def test_load_defaults_keep_embedded_template_and_explicit_values(self):
        values = {"llm_sampling": {"temperature": 0}, "llm_chat_reasoning_mode": "On"}
        original = deepcopy(values)
        plan = chat_load_profile(self.artifact(), values)
        self.assertEqual(values, original)
        self.assertEqual((plan.settings["llm_n_ctx"], plan.settings["llm_n_predict"]), (2048, 256))
        self.assertEqual((plan.batch, plan.microbatch), (256, 128))
        self.assertIsNone(plan.template)
        self.assertIsNone(plan.template_sha256)
        self.assertNotIn("--chat-template-file", plan.native_defaults)
        params = SimpleNamespace(chat_template="native embedded template")
        configure_embedded_profile(params, plan)
        self.assertEqual(params.chat_template, "native embedded template")
        self.assertEqual((params.n_batch, params.n_ubatch), (256, 128))
        for explicit in (
            {"llm_n_ctx": 0, "llm_n_predict": -1},
            {"llm_n_ctx": None, "llm_n_predict": None},
            {"llm_n_ctx": 4096, "llm_n_predict": 512, "llm_use_model_defaults": False},
        ):
            self.assertEqual(chat_load_profile(self.artifact(), explicit).settings, explicit)

    def test_wrong_artifact_never_gets_candidate_defaults(self):
        original = json.loads(json.dumps(self.artifact(), default=dict))
        for key, value in (("id", "other"), ("profile_id", "other")):
            row = deepcopy(original)
            row[key] = value
            self.assertFalse(chat_load_profile(row, {}).candidate)
        for key, value in (("bytes", 1), ("sha256", "0" * 64)):
            row = deepcopy(original)
            row["artifact"][key] = value
            self.assertFalse(chat_load_profile(row, {}).candidate)

    def test_real_profile_boolean_off_and_publisher_sampling_preserve_messages(self):
        # The actual GGUF template is separately header-verified in acceptance.
        # This tests real registry/runtime eligibility plus the template API.
        template = "{% if enable_thinking %}<think>{% endif %}"
        for runtime in ("llama.cpp", "xllamacpp"):
            with patch.dict(os.environ, {"LLAMA_SERVER": ""}):
                adapter = ProfiledChatClient(
                    Mock(),
                    self.artifact(),
                    {"llm_runtime": runtime, "llm_chat_reasoning_mode": "Off"},
                    template=template,
                    initialized=True,
                )
            self.assertEqual(adapter.reasoning_capability["modes"], ("Auto", "Off", "On"))
            data = {"messages": [{"role": "user", "content": "Answer exactly 7"}], "tools": []}
            result = adapter.parameters(data)
            self.assertIs(result["chat_template_kwargs"]["enable_thinking"], False)
            self.assertEqual(
                (result["temperature"], result["top_p"], result["top_k"]), (0.7, 0.8, 20)
            )
            self.assertEqual(result["messages"], data["messages"])
            self.assertNotIn("chat_template_kwargs", data)
            self.assertEqual(adapter.parameters({"temperature": 0})["temperature"], 0)

    def test_actual_loader_binds_embedded_template_when_candidate_has_no_override(self):
        source = resource_path().parent / "features/chat/llama_pipeline.py"
        # Evaluate the actual production call's template argument without
        # importing full app/Torch or starting either native runtime.
        tree = ast.parse(source.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ProfiledChatClient"
        ]
        self.assertEqual(len(calls), 1)
        expression = next(item.value for item in calls[0].keywords if item.arg == "template")
        embedded = Mock(return_value="actual embedded template")
        plan = chat_load_profile(self.artifact(), {})
        scope = dict(
            load_profile=plan,
            annotation=self.artifact(),
            llm_path=Path("owned.gguf"),
            embedded_reasoning_template=embedded,
        )
        value = eval(compile(ast.Expression(expression), str(source), "eval"), scope)
        self.assertEqual(value, "actual embedded template")
        embedded.assert_called_once_with(Path("owned.gguf"))


if __name__ == "__main__":
    unittest.main()
