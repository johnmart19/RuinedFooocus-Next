"""Passive parser fixtures and inert native launch; never execute a binary."""

import json
import ast
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from ruinedfooocus.backends.llama import llama_server, moe_cache
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.features.generation.defaults import STARTUP_DEFAULTS

HOST_CACHE = "--cpu-moe --moe-cache-mib 256"


def parser_fixture(path, supported=True):
    path.write_bytes(
        b"inert executable fixture\0"
        + (
            b"--moe-cache-cpu-share\0LLAMA_ARG_MOE_CACHE_CPU_SHARE\0"
            if supported
            else b"--moe-cache-mib\0"
        )
    )
    return path


def modular_fixture(root):
    executable = root / "llama-server.exe"
    executable.write_bytes(b"inert modular launcher")
    library = parser_fixture(root / "llama-common.dll")
    identity, original = next(iter(moe_cache._MODULAR_PARSERS.items()))
    version, package = identity
    fingerprints = (
        original[0],
        *(hashlib.sha256(path.read_bytes()).hexdigest() for path in (executable, library)),
    )
    manifest = {
        "version": version,
        "package": package,
        "archives": [
            {
                "name": package,
                "sha256": fingerprints[0],
                "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{version}/{package}",
            }
        ],
        "files": {"llama-server.exe": fingerprints[1], "llama-common.dll": fingerprints[2]},
    }
    marker = root / "rf-runtime.json"
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    return executable, library, marker, manifest, {identity: fingerprints}


class MoeCacheShareTests(unittest.TestCase):
    def test_modular_parser_requires_registered_archive_and_exact_launcher_and_dll(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, library, marker, manifest, trusted = modular_fixture(Path(directory))
            # A manifest claiming the real archive cannot bless arbitrary bytes.
            self.assertFalse(moe_cache.parser_capability(executable))
            with patch.dict(moe_cache._MODULAR_PARSERS, trusted, clear=True):
                self.assertTrue(moe_cache.parser_capability(executable))
                state = moe_cache.moe_cache_cpu_share_capability(
                    {"llama_server_args": HOST_CACHE}, executable, gpu=True, environment={}
                )
                self.assertTrue(state["supported"])
                self.assertEqual(state["arguments"], ("--moe-cache-cpu-share", "0.5"))
                executable.write_bytes(b"different modular launcher")
                self.assertFalse(moe_cache.parser_capability(executable))

    def test_modular_dll_and_manifest_changes_invalidate_positive_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, library, marker, manifest, trusted = modular_fixture(Path(directory))
            with patch.dict(moe_cache._MODULAR_PARSERS, trusted, clear=True):
                self.assertTrue(moe_cache.parser_capability(executable))
                original = library.read_bytes()
                library.write_bytes(original + b"untrusted replacement")
                self.assertFalse(moe_cache.parser_capability(executable))
                # Changing the local receipt cannot override the registered DLL.
                manifest["files"]["llama-common.dll"] = hashlib.sha256(
                    library.read_bytes()
                ).hexdigest()
                marker.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertFalse(moe_cache.parser_capability(executable))
                library.write_bytes(original)
                manifest["files"]["llama-common.dll"] = next(iter(trusted.values()))[2]
                marker.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertTrue(moe_cache.parser_capability(executable))
                manifest["archives"][0]["sha256"] = "0" * 64
                marker.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertFalse(moe_cache.parser_capability(executable))
                marker.unlink()
                self.assertFalse(moe_cache.parser_capability(executable))

    def test_modular_unknown_neighbor_and_bounded_reads_stay_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, library, marker, manifest, trusted = modular_fixture(Path(directory))
            with patch.dict(moe_cache._MODULAR_PARSERS, trusted, clear=True):
                with patch.object(moe_cache, "_MAX_MANIFEST", 8):
                    self.assertFalse(moe_cache.parser_capability(executable))
                moe_cache._modular_parser_option.cache_clear()
                with patch.object(moe_cache, "_MAX_BINARY", 1):
                    moe_cache._binary_evidence.cache_clear()
                    self.assertFalse(moe_cache.parser_capability(executable))
                moe_cache._binary_evidence.cache_clear()
                moe_cache._modular_parser_option.cache_clear()
                self.assertTrue(moe_cache.parser_capability(executable))
                manifest["package"] = "unregistered.zip"
                marker.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertFalse(moe_cache.parser_capability(executable))
                marker.write_text("not a runtime receipt", encoding="utf-8")
                self.assertFalse(moe_cache.parser_capability(executable))

    def test_changed_requested_share_invalidates_existing_chat_load_identity(self):
        source = (
            Path(__file__).resolve().parents[1] / "ruinedfooocus/features/chat/llama_pipeline.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "current_model_settings"
        )
        scope = {"json": json}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "llama_pipeline", "exec"), scope)
        owner = SimpleNamespace(loaded_paths=())
        original = scope["current_model_settings"](
            owner, "fixture.gguf", values=dict(moe_cache.DEFAULTS)
        )
        for changes in (
            {"llama_moe_cache_cpu_share_enabled": False},
            {"llama_moe_cache_cpu_share": 0.8},
        ):
            modified = {**moe_cache.DEFAULTS, **changes}
            self.assertNotEqual(
                original, scope["current_model_settings"](owner, "fixture.gguf", values=modified)
            )

    def test_enabled_midpoint_is_rf_default_and_existing_opt_out_survives(self):
        self.assertIs(STARTUP_DEFAULTS["llama_moe_cache_cpu_share_enabled"], True)
        self.assertEqual(STARTUP_DEFAULTS["llama_moe_cache_cpu_share"], 0.5)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, RF_DATA_DIR=directory),
        ):
            path = Path(directory) / "settings/settings.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"saved_unrelated": "preserved"}), encoding="utf-8")
            owner = SettingsManager(defaults=STARTUP_DEFAULTS)
            self.assertTrue(owner.default_settings["llama_moe_cache_cpu_share_enabled"])
            self.assertEqual(owner.default_settings["llama_moe_cache_cpu_share"], 0.5)
            values = {
                "llama_moe_cache_cpu_share_enabled": False,
                "llama_moe_cache_cpu_share": 0,
                "saved_unrelated": "preserved",
            }
            path.write_text(json.dumps(values), encoding="utf-8")
            owner = SettingsManager(defaults=STARTUP_DEFAULTS)
            self.assertFalse(owner.default_settings["llama_moe_cache_cpu_share_enabled"])
            self.assertEqual(owner.default_settings["llama_moe_cache_cpu_share"], 0)
            self.assertEqual(owner.default_settings["saved_unrelated"], "preserved")

    def test_schema_rejects_nonboolean_and_nonfinite_out_of_range_shares(self):
        self.assertEqual(
            validate_settings({"llama_moe_cache_cpu_share": "0.25"})["llama_moe_cache_cpu_share"],
            0.25,
        )
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "oops"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_settings({"llama_moe_cache_cpu_share": value})
        with self.assertRaises(ValueError):
            validate_settings({"llama_moe_cache_cpu_share_enabled": "maybe"})

    def test_navigation_capability_never_probes_or_opens_an_executable(self):
        with patch.object(
            moe_cache, "parser_capability", side_effect=AssertionError("No binary supplied")
        ):
            state = moe_cache.moe_cache_cpu_share_capability({})
        self.assertTrue(state["requested_enabled"])
        self.assertFalse(state["supported"])
        self.assertFalse(state["active"])
        self.assertIn("b12110", state["reason"])

    def test_positive_parser_detection_is_signature_cached_and_invalidates_on_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            self.assertTrue(moe_cache.parser_capability(executable))
            executable.write_bytes(b"replacement lacking parser option")
            self.assertFalse(moe_cache.parser_capability(executable))
            executable.unlink()
            self.assertFalse(moe_cache.parser_capability(executable))

    def test_supported_host_cache_emits_default_and_toggle_off_omits_app_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            for enabled in (True, False):
                state = moe_cache.moe_cache_cpu_share_capability(
                    {"llama_server_args": HOST_CACHE, "llama_moe_cache_cpu_share_enabled": enabled},
                    executable,
                    gpu=True,
                    environment={},
                )
                self.assertEqual(state["active"], enabled)
                self.assertEqual(
                    state["arguments"], ("--moe-cache-cpu-share", "0.5") if enabled else ()
                )
                self.assertEqual(state["effective_share"], 0.5 if enabled else None)

    def test_conditions_are_required_and_no_memory_expert_placement_is_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            for args, gpu in (
                ("", True),
                ("--cpu-moe", True),
                ("--moe-cache-mib 256", True),
                (HOST_CACHE, False),
                (HOST_CACHE + " -ngl 0", True),
                (HOST_CACHE + " --device none", True),
            ):
                state = moe_cache.moe_cache_cpu_share_capability(
                    {"llama_server_args": args}, executable, gpu=gpu, environment={}
                )
                self.assertFalse(state["active"])
                self.assertEqual(state["arguments"], ())
                self.assertEqual(state["extra_arguments"], tuple(args.split()))

    def test_verified_moe_cache_can_apply_preference_without_forcing_host_experts(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            for moe in (True, False):
                state = moe_cache.moe_cache_cpu_share_capability(
                    {"llama_server_args": "--moe-cache-mib 256"},
                    executable,
                    gpu=True,
                    environment={},
                    model_is_moe=moe,
                )
                self.assertEqual(bool(state["arguments"]), moe)
                self.assertNotIn("--cpu-moe", state["extra_arguments"])
                self.assertEqual(state["extra_arguments"], ("--moe-cache-mib", "256"))

    def test_explicit_cli_wins_over_app_toggle_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            for tail, expected in (
                (" --moe-cache-cpu-share 0.8", 0.8),
                (" --moe-cache-cpu-share=0", 0),
            ):
                args = HOST_CACHE + tail
                state = moe_cache.moe_cache_cpu_share_capability(
                    {"llama_server_args": args, "llama_moe_cache_cpu_share_enabled": False},
                    executable,
                    gpu=True,
                    environment={"LLAMA_ARG_MOE_CACHE_CPU_SHARE": "0.9"},
                )
                self.assertEqual(state["source"], "native_cli")
                self.assertEqual(state["selected_share"], expected)
                self.assertEqual(state["active"], expected != 0)
                self.assertEqual(state["arguments"], ())
                self.assertEqual(state["extra_arguments"], tuple(args.split()))

    def test_effective_settings_environment_and_cli_offload_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            for settings, tail, environment, active in (
                ({"llm_n_gpu_layers": 0}, "", {}, False),
                ({}, "", {"LLAMA_ARG_DEVICE": "none"}, False),
                ({}, "", {"LLAMA_ARG_N_GPU_LAYERS": "0"}, False),
                ({"llm_n_gpu_layers": 0}, " --gpu-layers all", {}, True),
                ({}, " --device Vulkan0", {"LLAMA_ARG_DEVICE": "none"}, True),
                ({"cpu_everywhere": True}, "", {}, False),
                ({"llama_backend": "CPU"}, "", {}, False),
            ):
                with self.subTest(settings=settings, tail=tail, environment=environment):
                    state = moe_cache.moe_cache_cpu_share_capability(
                        {**settings, "llama_server_args": HOST_CACHE + tail},
                        executable,
                        gpu=True,
                        environment=environment,
                    )
                    self.assertEqual(state["active"], active)
                    self.assertEqual(bool(state["arguments"]), active)

    def test_each_tensor_assignment_must_place_experts_on_cpu_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            for tensor, active in (
                ("ffn_exps=CUDA0,output=CPU", False),
                ("output=CPU", False),
                ("ffn_exps=CPU,output=CUDA0", True),
                ("ffn_exps=CPU", True),
            ):
                with self.subTest(tensor=tensor):
                    state = moe_cache.moe_cache_cpu_share_capability(
                        {"llama_server_args": "--moe-cache-mib 256 --override-tensor " + tensor},
                        executable,
                        gpu=True,
                        environment={},
                    )
                    self.assertEqual(state["active"], active)
                    self.assertEqual(bool(state["arguments"]), active)

    def test_unsupported_cli_is_inactive_without_unknown_flag_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe", False)
            state = moe_cache.moe_cache_cpu_share_capability(
                {"llama_server_args": HOST_CACHE + " --moe-cache-cpu-share=0.8 --temp 0.7"},
                executable,
                gpu=True,
                environment={},
            )
            self.assertFalse(state["supported"])
            self.assertFalse(state["active"])
            self.assertEqual(state["selected_share"], 0.8)
            self.assertEqual(state["arguments"], ())
            self.assertNotIn("--moe-cache-cpu-share=0.8", state["extra_arguments"])
            self.assertIn("--temp", state["extra_arguments"])
            self.assertNotIn("extra_arguments", moe_cache.public_moe_cache_state(state))
            with self.assertLogs(moe_cache.__name__, level="WARNING") as logs:
                moe_cache.warn_unsupported(state, executable)
                moe_cache.warn_unsupported(state, executable)
            self.assertEqual(len(logs.output), 1)

    def test_xllamacpp_keeps_saved_preference_inactive(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = parser_fixture(Path(directory) / "llama-server.exe")
            state = moe_cache.moe_cache_cpu_share_capability(
                {"llm_runtime": "xllamacpp", "llama_server_args": HOST_CACHE},
                executable,
                gpu=True,
                environment={},
            )
            self.assertTrue(state["requested_enabled"])
            self.assertFalse(state["supported"])
            self.assertIn("xllamacpp", state["reason"])

    def test_invalid_app_and_cli_values_fail_before_runtime_resolution(self):
        for settings in (
            {"llama_moe_cache_cpu_share": 2},
            {"llama_server_args": "--moe-cache-cpu-share NaN"},
            {"llama_server_args": "--moe-cache-cpu-share"},
            {"llama_server_args": "--moe-cache-cpu-share -1 --moe-cache-cpu-share 0.5"},
        ):
            with (
                self.subTest(settings=settings),
                patch.object(llama_server, "resolve_runtime") as resolve,
            ):
                with self.assertRaises(ValueError):
                    llama_server.Server("not-a-model", settings)
                resolve.assert_not_called()

    def test_actual_server_launch_uses_passive_capability_and_safe_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(b"synthetic model never read")
            executable = parser_fixture(root / "llama-server.exe")
            process = Mock()
            process.poll.return_value = None
            session = Mock(headers={})
            session.get.return_value.ok = True
            with (
                patch.dict(os.environ, RF_DATA_DIR=directory),
                patch.object(llama_server.requests, "Session", return_value=session),
                patch.object(
                    llama_server,
                    "resolve_runtime",
                    return_value=(executable, {}, True, "Vulkan", ""),
                ),
                patch.object(llama_server, "projector_path", return_value=None),
                patch.object(llama_server, "start_process", return_value=(process, None)) as start,
                patch.object(
                    llama_server.subprocess, "run", side_effect=AssertionError("No process probes")
                ),
            ):
                server = llama_server.Server(model, {"llama_server_args": HOST_CACHE})
                command = start.call_args.args[0]
                self.assertEqual(command[command.index("--moe-cache-cpu-share") + 1], "0.5")
                self.assertEqual(server.moe_cache_cpu_share["effective_share"], 0.5)
                server.close()
            self.assertEqual(model.read_bytes(), b"synthetic model never read")


if __name__ == "__main__":
    unittest.main()
