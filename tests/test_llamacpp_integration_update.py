"""Focused unit tests for the llama.cpp b12803 integration and control wiring."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.llama import llama_installer, llama_server, llama_update, moe_cache
from ruinedfooocus.features.chat import model_profiles, ui_reasoning_settings, ui_sampling


class LlamaCppReleasePinTests(unittest.TestCase):
    def test_version_and_compatible_cached_versions(self):
        self.assertEqual(llama_installer.VERSION, "b12803")
        self.assertEqual(
            llama_installer.COMPATIBLE_CACHED_VERSIONS,
            ("b12366", "b12327", "b12110"),
        )
        self.assertEqual(llama_installer.cuda_backend("b12803"), "CUDA 13.5")

    def test_pinned_release_manifest_covers_all_official_packages(self):
        manifest_path = Path(llama_installer.__file__).with_name("releases") / "b12803.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["tag_name"], "b12803")
        self.assertIs(manifest["draft"], False)
        self.assertGreaterEqual(len(manifest["assets"]), 30)

        targets = [
            ("Windows", "AMD64", "CPU"),
            ("Windows", "AMD64", "Vulkan"),
            ("Windows", "AMD64", "ROCm"),
            ("Windows", "AMD64", "SYCL"),
            ("Windows", "AMD64", "CUDA 12.4"),
            ("Windows", "AMD64", "CUDA 13.5"),
            ("Windows", "ARM64", "CPU"),
            ("Windows", "ARM64", "Vulkan"),
            ("Windows", "ARM64", "CUDA 13.5"),
            ("Linux", "x86_64", "CPU"),
            ("Linux", "x86_64", "Vulkan"),
            ("Linux", "x86_64", "ROCm"),
            ("Linux", "x86_64", "SYCL"),
            ("Linux", "x86_64", "CUDA 12.8"),
            ("Linux", "x86_64", "CUDA 13.5"),
            ("Linux", "aarch64", "CPU"),
            ("Linux", "aarch64", "Vulkan"),
            ("Linux", "aarch64", "CUDA 13.5"),
            ("Darwin", "arm64", "CPU"),
            ("Darwin", "x86_64", "CPU"),
        ]
        for system, machine, backend in targets:
            pkg = llama_installer.package_name(system, machine, backend != "CPU", backend=backend)
            archives = llama_installer._pinned_archives(pkg, "b12803")
            self.assertIsNotNone(archives, f"Missing pinned archive for {system}/{machine}/{backend}: {pkg}")
            expected_count = 2 if "-cuda-" in pkg else 1
            self.assertEqual(len(archives), expected_count)
            for item in archives:
                self.assertEqual(len(item["sha256"]), 64)
                self.assertTrue(item["url"].endswith("/b12803/" + item["name"]))

    def test_moe_cache_modular_parser_includes_b12803_and_n_cpu_moe_setting(self):
        key = ("b12803", "llama-b12803-bin-win-vulkan-x64.zip")
        self.assertIn(key, moe_cache._MODULAR_PARSERS)
        archive_hash, launcher_hash, parser_hash = moe_cache._MODULAR_PARSERS[key]
        self.assertEqual(len(archive_hash), 64)
        self.assertEqual(len(launcher_hash), 64)
        self.assertEqual(len(parser_hash), 64)

        with patch.object(moe_cache, "parser_capability", return_value=True):
            state = moe_cache.moe_cache_cpu_share_capability(
                {
                    "llm_runtime": "llama.cpp",
                    "llama_moe_cache_cpu_share_enabled": True,
                    "llama_moe_cache_cpu_share": 0.4,
                    "llm_n_cpu_moe": 8,
                    "llama_server_args": "--moe-cache-mib 2048",
                },
                executable=Path("llama-server.exe"),
                gpu=True,
                environment={},
            )
        self.assertTrue(state["active"])
        self.assertEqual(state["arguments"], ("--moe-cache-cpu-share", "0.4"))


class LlamaServerFlagsTests(unittest.TestCase):
    def _launch_command(self, settings, *, tag="b12803", gpu=True, env=None):
        captured = {}

        def fake_start(command, **kwargs):
            captured["cmd"] = list(command)
            proc = Mock()
            proc.poll.return_value = None
            return proc, None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"GGUF")
            exe = root / "llama-server.exe"
            exe.write_bytes(b"EXE")
            with (
                patch.dict(os.environ, RF_DATA_DIR=str(root / "data")),
                patch.object(
                    llama_server,
                    "resolve_runtime",
                    return_value=(exe, dict(env or {}), gpu, "Vulkan" if gpu else "CPU", ""),
                ),
                patch.object(
                    llama_server,
                    "runtime_identity",
                    return_value={"tag": tag, "manifest_verified": True, "signature": None},
                ),
                patch.object(llama_server, "model_header_evidence", return_value={"moe": False}),
                patch.object(
                    llama_server,
                    "moe_cache_cpu_share_capability",
                    return_value={
                        "arguments": (),
                        "extra_arguments": tuple(
                            __import__("shlex").split(settings.get("llama_server_args", ""))
                        ),
                        "supported": False,
                        "source": "app_setting",
                    },
                ),
                patch.object(llama_server, "public_moe_cache_state", return_value={}),
                patch.object(llama_server, "warn_unsupported"),
                patch.object(llama_server, "projector_path", return_value=None),
                patch.object(llama_server, "start_process", side_effect=fake_start),
                patch.object(
                    llama_server.requests.Session,
                    "get",
                    return_value=SimpleNamespace(ok=True),
                ),
                patch.object(
                    llama_server,
                    "startup_evidence",
                    return_value={
                        "quantization": "Q4_K_M",
                        "architecture": "llama",
                        "moe_cache_buffers": [],
                        "host_experts": 0,
                        "state": "full",
                        "context": 8192,
                    },
                ),
            ):
                srv = llama_server.Server(model, settings, gpu=gpu)
                srv.close()
        return captured["cmd"]

    def test_b12803_uses_load_mode_instead_of_removed_mmap_flags(self):
        cmd_off = self._launch_command({"llm_mmap": "Disabled"}, tag="b12803")
        self.assertNotIn("--no-mmap", cmd_off)
        self.assertNotIn("--mmap", cmd_off)
        self.assertEqual(cmd_off[cmd_off.index("--load-mode") + 1], "none")

        cmd_on = self._launch_command({"llm_mmap": "Enabled"}, tag="b12803")
        self.assertNotIn("--mmap", cmd_on)
        self.assertEqual(cmd_on[cmd_on.index("--load-mode") + 1], "mmap")

        cmd_legacy = self._launch_command({"llm_mmap": "Disabled"}, tag="b12366")
        self.assertIn("--no-mmap", cmd_legacy)
        self.assertNotIn("--load-mode", cmd_legacy)

    def test_new_llama_server_controls_and_override_precedence(self):
        cmd = self._launch_command(
            {
                "llm_kv_cache_type": "f8_e4m3",
                "llm_flash_attn": "Auto",
                "llm_batch_size": 1024,
                "llm_ubatch_size": 512,
                "llm_cache_reuse": 256,
                "llm_n_cpu_moe": 6,
                "llm_fit": "Off",
                "llm_reasoning_budget": 1024,
                "llm_reasoning_effort": "high",
            },
            tag="b12803",
        )
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "f8_e4m3")
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "f8_e4m3")
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")
        self.assertEqual(cmd[cmd.index("--batch-size") + 1], "1024")
        self.assertEqual(cmd[cmd.index("--ubatch-size") + 1], "512")
        self.assertEqual(cmd[cmd.index("--cache-reuse") + 1], "256")
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "6")
        self.assertEqual(cmd[cmd.index("--fit") + 1], "off")
        self.assertEqual(cmd[cmd.index("--reasoning-budget") + 1], "1024")
        self.assertEqual(cmd[cmd.index("--reasoning-effort") + 1], "high")
        self.assertIn("--jinja", cmd)

        # Explicit --no-jinja and --load-mode / --fit in extra args must not be duplicated.
        cmd_override = self._launch_command(
            {
                "llm_mmap": "Disabled",
                "llm_fit": "On",
                "llm_cache_reuse": 256,
                "llama_server_args": "--no-jinja --load-mode dio --fit off --cache-reuse 512",
            },
            tag="b12803",
        )
        self.assertNotIn("--jinja", cmd_override)
        self.assertIn("--no-jinja", cmd_override)
        self.assertEqual(cmd_override.count("--load-mode"), 1)
        self.assertEqual(cmd_override[cmd_override.index("--load-mode") + 1], "dio")
        self.assertEqual(cmd_override.count("--fit"), 1)
        self.assertEqual(cmd_override.count("--cache-reuse"), 1)

    def test_ensure_latest_build_handles_missing_data_dir_gracefully(self):
        with patch.object(
            llama_update,
            "skip_dependency_install",
            side_effect=FileNotFoundError("missing data dir"),
        ):
            self.assertIsNone(llama_update.ensure_latest_build("Vulkan"))


class SamplingAndReasoningControlsTests(unittest.TestCase):
    def test_extended_samplers_validation_cli_parsing_and_request_forwarding(self):
        cli = model_profiles.cli_sampling(
            "--temp 0.6 --top-nsigma 1.5 --typical 0.9 --min-p 0.05 "
            "--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 512 "
            "--xtc-probability 0.2 --xtc-threshold 0.15 --repeat-last-n 64 "
            "--presence-penalty 0.5 --frequency-penalty 0.25 --dynatemp-range 0.3 --dynatemp-exp 1.2"
        )
        self.assertEqual(cli["temperature"], 0.6)
        self.assertEqual(cli["top_n_sigma"], 1.5)
        self.assertEqual(cli["typical_p"], 0.9)
        self.assertEqual(cli["min_p"], 0.05)
        self.assertEqual(cli["dry_multiplier"], 0.8)
        self.assertEqual(cli["dry_base"], 1.75)
        self.assertEqual(cli["dry_allowed_length"], 2)
        self.assertEqual(cli["dry_penalty_last_n"], 512)
        self.assertEqual(cli["xtc_probability"], 0.2)
        self.assertEqual(cli["xtc_threshold"], 0.15)
        self.assertEqual(cli["repeat_last_n"], 64)
        self.assertEqual(cli["presence_penalty"], 0.5)
        self.assertEqual(cli["frequency_penalty"], 0.25)
        self.assertEqual(cli["dynatemp_range"], 0.3)
        self.assertEqual(cli["dynatemp_exp"], 1.2)

        # Integer-valued floats from settings_validation textboxes normalize cleanly to int.
        normalized = model_profiles.validate_sampling(
            {"dry_allowed_length": 3.0, "dry_penalty_last_n": -1.0, "repeat_last_n": 128.0}
        )
        self.assertIs(type(normalized["dry_allowed_length"]), int)
        self.assertEqual(normalized["dry_allowed_length"], 3)
        self.assertIs(type(normalized["dry_penalty_last_n"]), int)
        self.assertEqual(normalized["dry_penalty_last_n"], -1)
        self.assertIs(type(normalized["repeat_last_n"]), int)
        self.assertEqual(normalized["repeat_last_n"], 128)

        client = model_profiles.ProfiledChatClient(
            Mock(),
            None,
            {
                "llm_runtime": "llama.cpp",
                "llm_use_model_defaults": False,
                "llm_sampling": {
                    "top_n_sigma": 2.0,
                    "dry_multiplier": 0.7,
                    "xtc_probability": 0.1,
                    "min_p": 0.05,
                },
            },
        )
        params = client.parameters({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(params["top_n_sigma"], 2.0)
        self.assertEqual(params["dry_multiplier"], 0.7)
        self.assertEqual(params["xtc_probability"], 0.1)
        self.assertEqual(params["min_p"], 0.05)

        status = ui_sampling.active_sampling_status(client.sampling_state())
        self.assertIn("top_n_sigma", status)
        self.assertIn("dry_multiplier", status)
        self.assertIn("xtc_probability", status)

    def test_extended_reasoning_effort_levels_and_short_flag(self):
        validated = model_profiles.validate_reasoning_request({"mode": "On", "effort": "Xhigh"})
        self.assertEqual(validated, {"mode": "On", "effort": "Xhigh"})
        kwargs = model_profiles.controlled_reasoning_kwargs(
            {
                "control": "reasoning_effort",
                "modes": ("Auto", "Off", "On"),
                "efforts": ("Minimal", "Low", "Medium", "High", "Xhigh", "Max"),
                "reason": "",
            },
            {"mode": "On", "effort": "Max"},
        )
        self.assertEqual(kwargs, {"reasoning_effort": "max"})

        # Short flag -rea in llama_server_args yields native server control over UI reasoning.
        cap = model_profiles.initialized_reasoning(
            {"artifact": {"sha256": "a" * 64}, "profile_id": "test"},
            {"llm_runtime": "llama.cpp", "llama_server_args": "-rea off"},
            "{{ reasoning_effort }}",
            initialized=True,
        )
        self.assertIsNone(cap["control"])
        self.assertIn("Custom template or server reasoning arguments", cap["reason"])

        # ui_reasoning_settings forwards extended effort choices without dropping them.
        conv = ui_reasoning_settings.conversation_reasoning(
            "On",
            "Xhigh",
            {},
            capability={"modes": ("Auto", "Off", "On"), "efforts": ("Low", "Medium", "High", "Xhigh")},
        )
        self.assertEqual(conv, {"mode": "On", "effort": "Xhigh"})


if __name__ == "__main__":
    unittest.main()
