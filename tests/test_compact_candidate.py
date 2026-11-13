"""Exact compact-candidate format/configuration, without weights or devices."""

from contextlib import nullcontext
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ui_test_helpers import function_from
from ruinedfooocus.features.chat.model_profiles import (
    COMPACT_CANDIDATE,
    COMPACT_TEMPLATE_SHA256,
    ProfiledChatClient,
    chat_load_profile,
    configure_embedded_profile,
)
from ruinedfooocus.features.models.model_registry import load_registry, download_identity
from ruinedfooocus.features.models.model_capabilities import Eligibility, RuntimeIdentity
from ruinedfooocus.core.paths import resource_path

ROOT = Path(__file__).resolve().parents[1]


class CompactCandidateTests(unittest.TestCase):
    def artifact(self):
        return load_registry().artifact("existing:llm/" + COMPACT_CANDIDATE)

    def test_pinned_candidate_joins_without_changing_existing_default(self):
        downloads = json.loads(resource_path("model_catalogue", "llm.json").read_text())
        item = downloads["llm/" + COMPACT_CANDIDATE]
        artifact = self.artifact()
        self.assertEqual(download_identity(item), artifact["download_identity"])
        self.assertIn("/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/", item["url"])
        self.assertEqual(item["size"], artifact["artifact"]["bytes"])
        with patch.dict(
            "sys.modules",
            {"ruinedfooocus.app.state": SimpleNamespace(settings=None, path_manager=None)},
        ):
            models = runpy.run_path(str(ROOT / "ruinedfooocus/features/chat/llama_models.py"))
        self.assertEqual(models["DEFAULT_MODEL"], "LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf")
        self.assertIn("candidate", models["model_label"]("LFM2.5-1.2B-Instruct"))

    def test_absent_defaults_and_explicit_zero_none_false_survive_without_mutation(self):
        values = {"llm_use_model_defaults": False, "llm_sampling": {"temperature": 0}}
        original = deepcopy(values)
        plan = chat_load_profile(self.artifact(), values)
        self.assertEqual(
            (
                plan.settings["llm_n_ctx"],
                plan.settings["llm_n_predict"],
                plan.batch,
                plan.microbatch,
            ),
            (2048, 256, 256, 128),
        )
        self.assertEqual(plan.template_sha256, COMPACT_TEMPLATE_SHA256)
        self.assertEqual(values, original)
        for explicit in (
            {"llm_n_ctx": 0, "llm_n_predict": -1},
            {"llm_n_ctx": None, "llm_n_predict": None},
        ):
            self.assertEqual(chat_load_profile(self.artifact(), explicit).settings, explicit)

    def test_missing_wrong_hash_or_wrong_profile_never_reads_template(self):
        artifact = json.loads(json.dumps(dict(self.artifact()), default=lambda x: dict(x)))
        bad_hash = deepcopy(artifact)
        bad_hash["artifact"]["sha256"] = "0" * 64
        bad_size = deepcopy(artifact)
        bad_size["artifact"]["bytes"] = 1
        bad_profile = deepcopy(artifact)
        bad_profile["profile_id"] = "other"
        with patch(
            "ruinedfooocus.features.chat.model_profiles.resource_path",
            side_effect=AssertionError("Unverified candidate must not read resource"),
        ):
            for row in (None, bad_hash, bad_size, bad_profile):
                plan = chat_load_profile(row, {})
                self.assertFalse(plan.candidate)
                self.assertEqual(plan.settings, {})

    def test_same_named_small_custom_file_is_not_a_verified_candidate(self):
        from ruinedfooocus.features.models.model_capabilities import verified_local_artifact

        downloads = json.loads(resource_path("model_catalogue", "llm.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / COMPACT_CANDIDATE
            path.write_bytes(b"not the published model")
            artifact = verified_local_artifact(path, downloads)
            self.assertIsNone(artifact)
            self.assertFalse(chat_load_profile(artifact, {}).candidate)

    def test_corrupt_resource_fails_and_embedded_receives_exact_publisher_bytes(self):
        plan = chat_load_profile(self.artifact(), {"llm_runtime": "xllamacpp"})
        params = SimpleNamespace()
        configure_embedded_profile(params, plan)
        self.assertEqual((params.n_batch, params.n_ubatch), (256, 128))
        self.assertEqual(
            hashlib.sha256(params.chat_template.encode()).hexdigest(), COMPACT_TEMPLATE_SHA256
        )
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.jinja"
            bad.write_text("changed")
            with patch(
                "ruinedfooocus.features.chat.model_profiles.resource_path", return_value=bad
            ):
                with self.assertRaisesRegex(ValueError, "has changed"):
                    chat_load_profile(self.artifact(), {})

    def test_custom_native_template_remains_authoritative_without_fake_effective_hash(self):
        values = {
            "llama_server_args": '--chat-template-file "custom template.jinja" --batch-size 64'
        }
        plan = chat_load_profile(self.artifact(), values)
        self.assertIsNone(plan.template)
        self.assertIsNone(plan.template_sha256)
        self.assertNotIn("--chat-template-file", plan.native_defaults)
        self.assertEqual(plan.settings["llama_server_args"], values["llama_server_args"])
        kwargs_only = chat_load_profile(
            self.artifact(), {"llama_server_args": "--chat-template-kwargs {}"}
        )
        self.assertEqual(kwargs_only.template_sha256, COMPACT_TEMPLATE_SHA256)
        # NativeServer places model_defaults before explicit CLI, so explicit
        # batch/context/template switches retain their existing precedence.
        source = (ROOT / "ruinedfooocus/backends/llama/llama_server.py").read_text()
        self.assertLess(source.index("*model_defaults"), source.index("*extra,"))

    def test_profile_sampling_is_shared_and_request_values_win(self):
        client = ProfiledChatClient(Mock(), self.artifact(), {})
        data = {
            "messages": [{"role": "user", "content": "7"}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 0,
        }
        value = client.parameters(data)
        self.assertEqual(value["messages"], data["messages"])
        self.assertIs(value["stream"], False)
        self.assertEqual((value["temperature"], value["max_tokens"]), (0, 0))
        self.assertEqual(
            (
                value["top_k"],
                value["top_p"],
                value["min_p"],
                value["repeat_penalty"],
                value["presence_penalty"],
            ),
            (50, 1, 0, 1.05, 0),
        )

    def test_defaulted_model_cache_key_matches_request_snapshot_and_template_identity(self):
        function = function_from(
            "ruinedfooocus/features/chat/llama_pipeline.py",
            "current_model_settings",
            {
                "settings": SimpleNamespace(default_settings={}),
                "chat_load_profile": chat_load_profile,
                "DEFAULT_MODEL": "unchanged",
                "json": json,
            },
        )
        owner = SimpleNamespace(loaded_paths=(), model_annotation=self.artifact())
        before = function(owner, COMPACT_CANDIDATE, values={})
        plan = chat_load_profile(self.artifact(), {})
        after = function(owner, COMPACT_CANDIDATE, values=plan.settings)
        self.assertEqual(before, after)
        self.assertIn(plan.identity, before)

    def test_actual_loader_branches_share_plan_without_runtime_or_acquisition(self):
        from ruinedfooocus.features.models import model_capabilities as caps
        from ruinedfooocus.packaging import gpu_installer
        from ruinedfooocus.backends.llama import runtime_policy

        captured = {}

        class Native:
            runtime_label = "fixture"
            fallback_reason = ""

            def __init__(self, model, settings, **kwargs):
                captured["native"] = (deepcopy(settings), kwargs)

            def close(self):
                pass

        def embedded(params):
            captured["embedded"] = params
            return Mock()

        manager = SimpleNamespace(DOWNLOADABLE_FILES={})
        state = SimpleNamespace(path_manager=manager)
        function = function_from(
            "ruinedfooocus/features/chat/llama_pipeline.py",
            "load_base_model",
            {
                "settings": SimpleNamespace(default_settings={}),
                "DEFAULT_MODEL": "unused",
                "Path": Path,
                "resolve_model": Mock(side_effect=AssertionError("No acquisition")),
                "projector_path": lambda *a, **k: None,
                "projector_setting": lambda values: "None",
                "TimeIt": lambda text: nullcontext(),
                "force_cpu_for_auxiliary": lambda args: False,
                "NativeServer": Native,
                "ProfiledChatClient": ProfiledChatClient,
                "chat_load_profile": chat_load_profile,
                "configure_embedded_profile": configure_embedded_profile,
                "embedded_reasoning_template": Mock(
                    side_effect=AssertionError("Use effective bundled template")
                ),
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            weight = Path(folder) / "tiny.gguf"
            weight.write_bytes(b"CPU fixture")
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "ruinedfooocus.app.state": state,
                        "ruinedfooocus.app.arguments": SimpleNamespace(
                            args=SimpleNamespace(directml=None)
                        ),
                        "xllamacpp": SimpleNamespace(
                            CommonParams=lambda: SimpleNamespace(model=SimpleNamespace()),
                            Server=embedded,
                        ),
                    },
                ),
                patch.object(
                    caps, "configured_runtime", return_value=RuntimeIdentity("fixture", None)
                ),
                patch.object(caps, "local_architecture", return_value="lfm2"),
                patch.object(
                    caps, "architecture_eligibility", return_value=Eligibility("unknown", "fixture")
                ),
                patch.object(caps, "verified_local_artifact", return_value=self.artifact()),
                patch.object(gpu_installer, "ensure_xllamacpp", return_value={}),
                patch.object(gpu_installer, "installed_torch_constraints", return_value={}),
                patch.object(
                    runtime_policy, "admit_xllamacpp_devices", return_value=("fixture", True, "")
                ),
            ):
                for runtime in ("llama.cpp", "xllamacpp"):
                    values = {"llm_runtime": runtime, "llm_use_model_defaults": False}
                    prepared = SimpleNamespace(
                        settings=values, model=weight, projector=None, present=True
                    )
                    owner = SimpleNamespace(
                        unload=Mock(), current_model_settings=lambda *a, **k: ()
                    )
                    function(owner, str(weight), prepared=prepared, _leased=True)
                    self.assertEqual(prepared.settings, values)
                    self.assertEqual(
                        owner.llm.reasoning_capability["template_sha256"], COMPACT_TEMPLATE_SHA256
                    )
        native_settings, native_args = captured["native"]
        self.assertEqual(native_settings["llm_n_ctx"], 2048)
        self.assertEqual(native_settings["llm_n_predict"], 256)
        self.assertIn(
            str(resource_path("chat_templates", "lfm25-instruct-publisher.jinja")),
            native_args["model_defaults"],
        )
        params = captured["embedded"]
        self.assertEqual(
            (params.n_ctx, params.n_predict, params.n_batch, params.n_ubatch), (2048, 256, 256, 128)
        )
        self.assertEqual(
            hashlib.sha256(params.chat_template.encode()).hexdigest(), COMPACT_TEMPLATE_SHA256
        )

    def test_source_archive_generic_inclusion_covers_template_and_license_after_publication(self):
        from ruinedfooocus.packaging.distribution import USER_DIRECTORIES, WEIGHT_SUFFIXES

        for name in ("lfm25-instruct-publisher.jinja", "LFM-LICENSE.txt", "NOTICE.md"):
            path = resource_path("chat_templates", name).relative_to(ROOT)
            self.assertNotIn(path.parts[0], USER_DIRECTORIES)
            self.assertNotIn(path.suffix, WEIGHT_SUFFIXES)
            self.assertTrue((ROOT / path).is_file())

    def test_settings_absent_fields_stay_blank_and_profile_blanks_remove_overrides(self):
        from ruinedfooocus.core.settings import prospective_profile
        from ruinedfooocus.core.profile_store import save_profile, read_profile
        from ruinedfooocus.core.settings_validation import validate_settings

        source = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text())
        for values in ({}, {"llm_n_ctx": 0, "llm_n_predict": -1}):
            for key in ("llm_n_ctx", "llm_n_predict"):
                node = next(
                    n
                    for n in ast.walk(source)
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == key for t in n.targets)
                )
                scope = {
                    "gr": SimpleNamespace(Number=lambda **kwargs: kwargs),
                    "t": lambda text: text,
                    "settings": SimpleNamespace(default_settings=values),
                }
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), "candidate UI", "exec"), scope
                )
                self.assertEqual(scope[key]["value"], values.get(key))
        old = {"llm_n_ctx": 8192, "llm_n_predict": -1, "custom": "retain"}
        blank, paths = prospective_profile(
            old, {}, validate_settings({"llm_n_ctx": None, "llm_n_predict": None}), set()
        )
        explicit, _ = prospective_profile(
            old, {}, validate_settings({"llm_n_ctx": 0, "llm_n_predict": -1}), set()
        )
        self.assertEqual(blank, {"custom": "retain"})
        self.assertEqual(explicit, {"llm_n_ctx": 0, "llm_n_predict": -1, "custom": "retain"})
        with tempfile.TemporaryDirectory() as folder:
            for values in (blank, explicit):
                save_profile(Path(folder), values, paths)
                self.assertEqual(read_profile(Path(folder) / "settings.json"), values)

    def test_actual_save_callback_accepts_cleared_chat_numbers_and_publishes_no_null(self):
        from ruinedfooocus.core.settings_validation import validate_settings
        from ruinedfooocus.core.profile_store import save_profile, read_profile
        from ruinedfooocus.core.settings_schema import validate_known

        # Do not broaden the raw startup/schema meaning of null or invalid text.
        with self.assertRaises(ValueError):
            validate_known({"llm_n_ctx": None})
        for invalid in ("", "bad", False):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_settings({"llm_n_predict": invalid})

        with tempfile.TemporaryDirectory() as folder:
            owner = SimpleNamespace(
                default_settings={"llm_n_ctx": 8192, "llm_n_predict": -1, "custom": "retain"},
                name=None,
            )
            paths = SimpleNamespace(
                paths={"custom_path": "retain"}, DEFAULT_PATHS={}, set_settings_path=Mock()
            )

            def publish(values, path_values, *, profile, on_publish):
                save_profile(Path(folder), values, path_values)
                owner.default_settings = values
                on_publish()

            owner.save_profile = publish
            callback = function_from(
                "ruinedfooocus/ui/ui_settings.py",
                "save_clicked",
                {
                    "state": {"setting_name": ["llm_n_ctx", "llm_n_predict"]},
                    "validate_settings": validate_settings,
                    "settings": owner,
                    "path_manager": paths,
                    "models": SimpleNamespace(),
                    "performance_settings": SimpleNamespace(
                        model_recommendations=SimpleNamespace()
                    ),
                    "gr": SimpleNamespace(Error=ValueError, Info=Mock()),
                },
            )
            with patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.app.arguments": SimpleNamespace(
                        args=SimpleNamespace(offline=True, settings=None)
                    )
                },
            ):
                callback(None, None)
                self.assertEqual(read_profile(Path(folder) / "settings.json"), {"custom": "retain"})
                callback(0, -1)
                self.assertEqual(
                    read_profile(Path(folder) / "settings.json"),
                    {"custom": "retain", "llm_n_ctx": 0, "llm_n_predict": -1},
                )
            self.assertEqual(paths.paths, {"custom_path": "retain"})


if __name__ == "__main__":
    unittest.main()
