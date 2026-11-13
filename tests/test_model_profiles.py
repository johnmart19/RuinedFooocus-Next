"""Every Chat route shares sampler precedence without modifying prompt content."""

import copy
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
import unittest

from ruinedfooocus.features.chat.model_profiles import (
    ProfiledChatClient,
    cli_sampling,
    native_load_defaults,
    validate_sampling,
    initialized_reasoning,
    controlled_reasoning_kwargs,
    embedded_reasoning_template,
)
from ruinedfooocus.features.models.model_registry import load_registry


class ChatProfileTests(unittest.TestCase):
    def artifact(self, name):
        return load_registry().artifact("existing:llm/" + name)

    def controlled_adapter(self, name, template, settings=None):
        return ProfiledChatClient(
            Mock(), self.artifact(name), settings or {}, template=template, initialized=True
        )

    def test_initialized_boolean_template_off_is_false_and_has_no_graded_effort(self):
        adapter = self.controlled_adapter(
            "Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf",
            "{% if enable_thinking %}think{% endif %}",
        )
        self.assertEqual(adapter.reasoning_capability["modes"], ("Auto", "Off", "On"))
        self.assertEqual(adapter.reasoning_capability["efforts"], ())
        self.assertEqual(len(adapter.reasoning_capability["template_sha256"]), 64)
        original = {"_rf_reasoning": {"mode": "Off", "effort": "Auto"}, "max_tokens": 0}
        request = adapter.parameters(original)
        self.assertIs(request["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(request["max_tokens"], 0)
        self.assertNotIn("_rf_reasoning", request)
        self.assertEqual(original["_rf_reasoning"]["mode"], "Off")
        with self.assertRaisesRegex(ValueError, "effort"):
            adapter.parameters({"_rf_reasoning": {"mode": "On", "effort": "Low"}})

    def test_gpt_oss_exposes_exact_effort_without_fictional_off(self):
        adapter = self.controlled_adapter("gpt-oss-20b-MXFP4.gguf", "{{ reasoning_effort }}")
        self.assertEqual(adapter.reasoning_capability["modes"], ("Auto", "On"))
        self.assertEqual(adapter.reasoning_capability["efforts"], ("Low", "Medium", "High"))
        for effort in ("Low", "Medium", "High"):
            request = adapter.parameters({"_rf_reasoning": {"mode": "On", "effort": effort}})
            self.assertEqual(request["chat_template_kwargs"], {"reasoning_effort": effort.lower()})
        with self.assertRaisesRegex(ValueError, "off switch"):
            adapter.parameters({"_rf_reasoning": {"mode": "Off"}})

    def test_absent_legacy_rewrite_mode_preserves_policy_but_explicit_auto_uses_template(self):
        name = "Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf"
        legacy = self.controlled_adapter(
            name, "{{ enable_thinking }}", {"llm_rewrite_reasoning": "Low"}
        )
        self.assertIs(
            legacy.parameters({"_rf_task": "rewrite"})["chat_template_kwargs"]["enable_thinking"],
            False,
        )
        explicit = self.controlled_adapter(
            name,
            "{{ enable_thinking }}",
            {"llm_rewrite_reasoning": "Low", "llm_rewrite_reasoning_mode": "Auto"},
        )
        self.assertNotIn("chat_template_kwargs", explicit.parameters({"_rf_task": "rewrite"}))
        effort = self.controlled_adapter(
            "gpt-oss-20b-MXFP4.gguf",
            "{{ reasoning_effort }}",
            {"llm_rewrite_reasoning_mode": "On", "llm_rewrite_reasoning_effort": "High"},
        )
        self.assertEqual(
            effort.parameters({"_rf_task": "prompt_enhancement"})["chat_template_kwargs"],
            {"reasoning_effort": "high"},
        )

    def test_capability_requires_initialized_exact_profile_template_and_runtime(self):
        artifact = self.artifact("gpt-oss-20b-MXFP4.gguf")
        for entry, template, initialized, settings in (
            (artifact, "{{ reasoning_effort }}", False, {}),
            (None, "{{ reasoning_effort }}", True, {}),
            (artifact, "{{ content }}", True, {}),
            (artifact, "{% broken syntax %}", True, {}),
            (
                artifact,
                "{{ reasoning_effort }}",
                True,
                {"llama_server_args": "--chat-template-file custom.jinja"},
            ),
            (
                artifact,
                "{{ reasoning_effort }}",
                True,
                {"llama_server_args": "--reasoning-budget=0"},
            ),
        ):
            with self.subTest(template=template, initialized=initialized, settings=settings):
                capability = initialized_reasoning(
                    entry, settings, template, initialized=initialized
                )
                self.assertIsNone(capability["control"])
                self.assertEqual(controlled_reasoning_kwargs(capability, {"mode": "Auto"}), {})
        from ruinedfooocus.features.models.model_capabilities import RuntimeIdentity

        with patch(
            "ruinedfooocus.features.models.model_capabilities.configured_runtime",
            return_value=RuntimeIdentity("llama.cpp", None),
        ):
            self.assertIsNone(
                initialized_reasoning(artifact, {}, "{{ reasoning_effort }}", initialized=True)[
                    "control"
                ]
            )

    def test_lfm3_controls_remain_gated_until_budget_contract_is_resolved(self):
        capability = initialized_reasoning(
            self.artifact("LFM3-1.7B-Q4_K_M.gguf"), {}, "{{ enable_thinking }}", initialized=True
        )
        self.assertIsNone(capability["control"])
        self.assertIn("unresolved", capability["reason"])
        with self.assertRaisesRegex(ValueError, "unresolved"):
            controlled_reasoning_kwargs(capability, {"mode": "Off"})

    def test_template_metadata_cache_uses_file_identity_and_bounds_template_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.gguf"
            path.write_bytes(b"metadata fixture")
            field = Mock()
            field.contents.return_value = "{{ enable_thinking }}"
            reader = Mock()
            reader.get_field.return_value = field
            with patch("gguf.GGUFReader", return_value=reader) as create:
                self.assertEqual(embedded_reasoning_template(path), "{{ enable_thinking }}")
                self.assertEqual(embedded_reasoning_template(path), "{{ enable_thinking }}")
                create.assert_called_once_with(str(path.resolve()))
                path.write_bytes(b"changed metadata fixture")
                field.contents.return_value = "x" * (1024 * 1024 + 1)
                self.assertEqual(embedded_reasoning_template(path), "")
                self.assertEqual(create.call_count, 2)

    def test_saved_unknown_controls_are_inert_and_request_snapshots_are_independent(self):
        saved = {"llm_chat_reasoning_mode": "Off"}
        first = self.controlled_adapter(
            "Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf", "{{ enable_thinking }}", saved
        )
        saved["llm_chat_reasoning_mode"] = "On"
        second = self.controlled_adapter(
            "Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf", "{{ enable_thinking }}", saved
        )
        self.assertIs(first.parameters({})["chat_template_kwargs"]["enable_thinking"], False)
        self.assertIs(second.parameters({})["chat_template_kwargs"]["enable_thinking"], True)
        unknown = ProfiledChatClient(Mock(), None, {"llm_chat_reasoning_mode": "future-value"})
        self.assertEqual(unknown.parameters({"messages": []}), {"messages": []})
        self.assertIn("inactive", unknown.reasoning_state({})["status"])
        with self.assertRaises(ValueError):
            unknown.parameters({"_rf_reasoning": {"mode": "future-value"}})
        explicit = first.parameters(
            {
                "_rf_reasoning": {"mode": "Off"},
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 0},
            }
        )
        self.assertIs(explicit["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(explicit["chat_template_kwargs"]["reasoning_budget"], 0)

    def test_trusted_publisher_sampling_covers_streaming_completion_and_rewrite(self):
        client = Mock()
        artifact = self.artifact("LFM3-1.7B-Q4_K_M.gguf")
        adapter = ProfiledChatClient(client, artifact, {})
        for route in (
            {"stream": True},
            {"stream": False},
            {"_rf_task": "rewrite"},
            {"_rf_task": "prompt_enhancement"},
        ):
            with self.subTest(route=route):
                adapter.handle_chat_completions(
                    {"messages": [{"role": "user", "content": "A sign reading SUN 42"}], **route}
                )
                request = client.handle_chat_completions.call_args.args[0]
                self.assertEqual(request["temperature"], 0.1)
                self.assertEqual(request["top_k"], 50)
                self.assertEqual(request["repeat_penalty"], 1.1)
                self.assertNotIn("_rf_task", request)
                if "_rf_task" in route:
                    self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(native_load_defaults(artifact), ["--reasoning-budget", "0"])
        self.assertEqual(native_load_defaults(None), [])

    def test_request_cli_user_and_publisher_precedence_preserves_zero_and_false(self):
        adapter = ProfiledChatClient(
            Mock(),
            self.artifact("LFM3-1.7B-Q4_K_M.gguf"),
            {
                "llm_sampling": {"temperature": 0.6, "top_k": 0},
                "llama_server_args": "--temp=0.7 --top-p 0.8",
            },
        )
        self.assertEqual(adapter.parameters({})["temperature"], 0.7)
        request = adapter.parameters(
            {
                "temperature": 0,
                "top_p": 0,
                "max_tokens": 0,
                "_rf_task": "rewrite",
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        self.assertEqual(
            (request["temperature"], request["top_p"], request["top_k"], request["max_tokens"]),
            (0, 0, 0, 0),
        )
        self.assertIs(request["chat_template_kwargs"]["enable_thinking"], False)

    def test_unknown_and_opted_out_models_keep_embedded_defaults_and_explicit_controls(self):
        for artifact, settings in [
            (None, {}),
            (self.artifact("LFM3-1.7B-Q4_K_M.gguf"), {"llm_use_model_defaults": False}),
        ]:
            adapter = ProfiledChatClient(Mock(), artifact, settings)
            self.assertEqual(
                adapter.parameters({"messages": [], "_rf_task": "rewrite"}), {"messages": []}
            )
            self.assertEqual(
                adapter.parameters({"chat_template_kwargs": {"enable_thinking": False}}),
                {"chat_template_kwargs": {"enable_thinking": False}},
            )

    def test_wrapper_requests_keep_json_booleans_without_native_cli_leakage(self):
        adapter = ProfiledChatClient(
            Mock(), None, {"llm_runtime": "xllamacpp", "llama_server_args": "--temp 0.9"}
        )
        request = adapter.parameters({"chat_template_kwargs": {"enable_thinking": False}})
        self.assertNotIn("temperature", request)
        self.assertIs(request["chat_template_kwargs"]["enable_thinking"], False)

    def test_native_template_tools_schemas_and_literal_messages_are_unchanged(self):
        adapter = ProfiledChatClient(Mock(), self.artifact("gpt-oss-20b-MXFP4.gguf"), {})
        original = {
            "messages": [
                {"role": "system", "content": "Keep policy."},
                {"role": "user", "content": 'Quote: "sunflower_42" <lora:detail:0.5>'},
            ],
            "tools": [{"type": "function", "function": {"name": "generate_image"}}],
            "response_format": {"type": "json_object"},
            "_rf_task": "rewrite",
        }
        saved = copy.deepcopy(original)
        request = adapter.parameters(original)
        self.assertEqual(original, saved)
        for key in ("messages", "tools", "response_format"):
            self.assertIs(request[key], original[key])
        self.assertEqual(request["chat_template_kwargs"], {"reasoning_effort": "low"})
        with self.assertRaisesRegex(ValueError, "supports reasoning effort"):
            adapter.parameters({"chat_template_kwargs": {"reasoning_effort": "off"}})

    def test_hybrid_thinking_selects_matching_recommendation(self):
        artifact = self.artifact("Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf")
        adapter = ProfiledChatClient(Mock(), artifact, {})
        self.assertEqual(adapter.parameters({})["temperature"], 1.0)
        request = adapter.parameters({"_rf_task": "rewrite"})
        self.assertEqual(request["temperature"], 0.7)
        self.assertIs(request["chat_template_kwargs"]["enable_thinking"], False)

    def test_invalid_overrides_fail_and_close_is_idempotent(self):
        for override in (
            {"temperature": float("nan")},
            {"top_p": 1.1},
            {"top_k": True},
            {"unknown": 1},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_sampling(override)
        self.assertEqual(
            cli_sampling("--temp 0.2 --temp=0 --top-k 0"), {"temperature": 0, "top_k": 0}
        )
        close = Mock()
        adapter = ProfiledChatClient(Mock(), None, {}, close=close)
        adapter.close()
        adapter.close()
        close.assert_called_once()

    def test_sampling_form_blanks_clear_overrides_without_losing_zero_or_false(self):
        from ruinedfooocus.core.settings_validation import validate_settings

        saved = validate_settings(
            {
                "llm_use_model_defaults": False,
                "llm_sampling__temperature": "0",
                "llm_sampling__top_k": "0",
                "llm_sampling__top_p": None,
                "llm_sampling__repeat_penalty": "",
            }
        )
        self.assertEqual(saved["llm_sampling"], {"temperature": 0, "top_k": 0})
        self.assertIs(saved["llm_use_model_defaults"], False)
        self.assertFalse(any(key.startswith("llm_sampling__") for key in saved))
        self.assertEqual(validate_settings({"llm_sampling__temperature": None})["llm_sampling"], {})
        for values in ({"llm_sampling__unknown": 1}, {"llm_sampling__top_p": 1.1}):
            with self.assertRaises(ValueError):
                validate_settings(values)
        for value in ("nan", "inf", "bad", "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_settings({"llm_sampling__top_k": value})


if __name__ == "__main__":
    unittest.main()
