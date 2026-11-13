"""Bounded native startup fixtures and cancellation; no binaries or GPU calls."""

from concurrent.futures import CancelledError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.llama import llama_server, runtime_probe
from ruinedfooocus.backends.llama.runtime_events import (
    MAX_LOG_BYTES,
    MAX_ROWS,
    RuntimeLoad,
    native_request,
    parse_startup,
    runtime_identity,
)
from ruinedfooocus.features.chat.model_preparation import runtime_selection_key


class RuntimeEventsTests(unittest.TestCase):
    def test_native_fit_preserves_context_zero_manual_layers_and_cpu_authority(self):
        state = native_request({"llm_n_ctx": 0}, [])
        self.assertEqual(state["gpu_layers"], "auto")
        self.assertEqual(state["context_policy"], "full_model")
        self.assertEqual(native_request({"llm_n_gpu_layers": -2}, [])["gpu_layers"], "all")
        state = native_request({}, ["-c", "4096", "-ngl=7", "-ctk", "q8_0", "--fit", "off"])
        self.assertEqual(
            (state["context"], state["gpu_layers"], state["cache_type_k"], state["fit"]),
            ("4096", "7", "q8_0", "off"),
        )
        cpu = native_request({}, ["--device", "Vulkan0", "-ngl", "all"], gpu=False)
        self.assertEqual(
            (cpu["gpu_layers"], cpu["device"], cpu["kv_offload"]), ("0", "none", False)
        )

    def test_final_native_initialization_replaces_earlier_fit_tensor_evidence(self):
        text = (
            "tensor blk.0.ffn_up_exps.weight (2048 MiB MXFP4) buffer type overridden to CPU\n"
            "load_tensors: offloaded 25/25 layers to GPU\n"
            "load_tensors: CPU_Mapped model buffer size = 0.00 MiB\n"
            "load_tensors: offloaded 25/25 layers to GPU\n"
            "load_tensors: Vulkan0 model buffer size = 1000.00 MiB\n"
            "srv main: model loaded\n"
        )
        parsed = parse_startup(text)
        self.assertEqual(parsed["state"], "full")
        self.assertFalse(parsed["host_experts"])
        self.assertFalse(parsed["tensors"])

    def test_host_expert_environment_and_cli_controls_remain_explicit(self):
        state = native_request(
            {},
            ["--n-cpu-moe", "0", "-ot", "output=Vulkan0"],
            environment={
                "LLAMA_ARG_CPU_MOE": "true",
                "LLAMA_ARG_N_CPU_MOE": "12",
                "LLAMA_ARG_OVERRIDE_TENSOR": "ffn_exps=CPU",
            },
        )
        self.assertTrue(state["cpu_experts"])
        self.assertEqual(state["cpu_expert_layers"], "0")
        self.assertEqual(state["tensor_overrides"], "ffn_exps=CPU,output=Vulkan0")
        self.assertFalse(
            native_request({}, [], environment={"LLAMA_ARG_CPU_MOE": "false"})["cpu_experts"]
        )
        self.assertTrue(
            native_request({}, ["--cpu-moe"], environment={"LLAMA_ARG_CPU_MOE": "false"})[
                "cpu_experts"
            ]
        )

    def test_native_cache_configuration_proves_host_experts_but_not_miss_execution(self):
        parsed = parse_startup(
            "load_tensors: offloaded 25/25 layers to GPU\n"
            "load_tensors: Vulkan0 model buffer size = 1000.00 MiB\n"
            "impl: Vulkan0 MoE cache size = 128.00 MiB for 9000.00 MiB of host experts, 50% of the misses on the CPU\n"
            "srv main: model loaded\n"
        )
        self.assertEqual(parsed["state"], "mixed")
        self.assertTrue(parsed["host_experts"])
        self.assertEqual(parsed["moe_cache_buffers"][0]["configured_cpu_share"], 0.5)
        self.assertNotIn("observed", parsed["moe_cache_buffers"][0])

    def test_all_layers_alone_never_prove_full_gpu_or_hide_host_experts(self):
        text = "load_tensors: offloaded 25/25 layers to GPU\nload_tensors: Vulkan0 model buffer size = 1000.00 MiB\nload_tensors: CPU_Mapped model buffer size = 5000.00 MiB\n"
        self.assertEqual(parse_startup(text)["state"], "unknown")
        mixed = parse_startup(
            "tensor blk.0.ffn_up_exps.weight (2048 MiB MXFP4) buffer type overridden to CPU\n"
            + text
            + "srv main: model loaded\n"
        )
        self.assertEqual(mixed["state"], "mixed")
        self.assertIs(mixed["host_experts"], True)
        measured = parse_startup(
            text + "srv main: model loaded\n", host_input={"input_bytes": 200 * 1024**2}
        )
        self.assertEqual(measured["state"], "mixed")
        self.assertEqual(
            parse_startup(text, host_input={"input_bytes": 200 * 1024**2})["state"], "unknown"
        )

    def test_initialized_full_cpu_and_input_buffer_attribution_are_separate(self):
        full = "load_tensors: offloaded 25/25 layers to GPU\nload_tensors: Vulkan0 model buffer size = 1000.00 MiB\nllama_context: n_ctx = 4096\nllama_kv_cache: Vulkan0 KV buffer size = 100.00 MiB\nllama_context: CPU compute buffer size = 5.00 MiB\n"
        self.assertEqual(parse_startup(full)["state"], "unknown")
        self.assertEqual(parse_startup(full + "srv main: model loaded\n")["state"], "full")
        host = full + "load_tensors: CPU_Mapped model buffer size = 200.00 MiB\n"
        self.assertEqual(parse_startup(host)["state"], "unknown")
        self.assertEqual(
            parse_startup(
                host + "srv main: model loaded\n", host_input={"input_bytes": 200 * 1024**2}
            )["state"],
            "full",
        )
        cpu = parse_startup(
            "load_tensors: offloaded 0/25 layers to GPU\nload_tensors: CPU_Mapped model buffer size = 1000.00 MiB\nsrv main: model loaded\n"
        )
        self.assertEqual(cpu["state"], "cpu")
        self.assertEqual(
            parse_startup(
                "load_tensors: CPU_Mapped model buffer size = 1000.00 MiB\nsrv main: model loaded\n"
            )["state"],
            "cpu",
        )
        parsed = parse_startup(full)
        self.assertEqual(parsed["context"], 4096)
        self.assertEqual(parsed["kv_buffers"][0]["mib"], 100)
        self.assertTrue(parsed["compute_buffers"][0]["host"])

    def test_json_native_fit_estimates_malformed_and_truncated_stay_unknown(self):
        rows = [
            {
                "kind": "device",
                "name": "Vulkan0",
                "model": 12000,
                "free": 2000,
                "secret": "never public",
            }
        ]
        parsed = parse_startup(
            json.dumps({"type": "fit_memory_breakdown", "data": {"unit": "MiB", "rows": rows}})
        )
        self.assertEqual(parsed["state"], "unknown")
        self.assertNotIn("secret", parsed["fit_estimates"][0])
        full = "load_tensors: offloaded 1/1 layers to GPU\nload_tensors: Vulkan0 model buffer size = 1.00 MiB\n"
        self.assertEqual(parse_startup(full + "x" * MAX_LOG_BYTES)["state"], "unknown")
        self.assertEqual(parse_startup('{"type":\nsecret API-key log')["state"], "unknown")

    def test_malformed_sizes_and_row_caps_cannot_certify_full_placement(self):
        malformed = parse_startup("load_tensors: Vulkan0 model buffer size = 1.2.3 MiB\n")
        self.assertTrue(malformed["truncated"])
        text = (
            "load_tensors: offloaded 1/1 layers to GPU\n"
            + ("load_tensors: Vulkan0 model buffer size = 1.00 MiB\n" * (MAX_ROWS + 1))
            + "srv main: model loaded\n"
        )
        parsed = parse_startup(text)
        self.assertEqual(len(parsed["model_buffers"]), MAX_ROWS)
        self.assertEqual(parsed["state"], "unknown")

    def test_invalidated_load_cannot_publish_ready_or_old_error(self):
        callback = Mock()
        load = RuntimeLoad("old", callback)
        load.emit(phase="loading_model", received=12, total=0, rate=4)
        self.assertEqual(load.snapshot()["transfer"], {"received": 12, "total": 0, "rate": 4})
        callback.call_args.args[0]["phase"] = "mutated"
        self.assertEqual(load.snapshot()["phase"], "loading_model")
        load.cancel()
        with self.assertRaises(CancelledError):
            load.emit(phase="ready", result={})
        load.emit(phase="failed")
        self.assertEqual(callback.call_count, 1)

    def test_runtime_identity_binds_manifest_to_actual_child_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / "llama-server.exe"
            executable.write_bytes(b"inert")
            first = runtime_identity(executable, "Vulkan")
            manifest = {
                "version": "b12327",
                "package": "fixture",
                "files": {executable.name: first["sha256"]},
            }
            (directory / "rf-runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(runtime_identity(executable, "Vulkan")["manifest_verified"])
            self.assertFalse(
                runtime_identity(executable, "Vulkan", custom=True)["manifest_verified"]
            )
            executable.write_bytes(b"replaced")
            self.assertFalse(runtime_identity(executable, "Vulkan")["manifest_verified"])

    def test_selection_key_ignores_unrelated_ui_drafts_but_tracks_source_and_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            model.write_bytes(b"fixture")
            first = runtime_selection_key(model, {"llm_n_ctx": 0})
            self.assertEqual(
                first, runtime_selection_key(model, {"llm_n_ctx": 0, "gallery_page": 2})
            )
            self.assertNotEqual(first, runtime_selection_key(model, {"llm_n_ctx": 4096}))
            self.assertNotEqual(first, runtime_selection_key(model, {"llm_n_ctx": 0}, vision=True))
            model.write_bytes(b"replacement")
            self.assertNotEqual(first, runtime_selection_key(model, {"llm_n_ctx": 0}))

    def test_probe_passes_owned_cancellation_and_never_swallows_it_as_fallback(self):
        check = Mock(side_effect=CancelledError)
        with patch.object(runtime_probe, "run_bounded", side_effect=CancelledError) as run:
            with self.assertRaises(CancelledError):
                llama_server._probe_runtime("fixture", {}, check_cancelled=check)
        self.assertIs(run.call_args.kwargs["check_cancelled"], check)

    def test_cancel_in_health_wait_terminates_only_owned_server_and_no_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "fixture.gguf"
            model.write_bytes(b"inert")
            process = Mock()
            process.poll.return_value = None
            session = Mock(headers={})
            session.get.return_value.ok = False
            events = []

            def check():
                if session.get.called:
                    raise CancelledError()

            with (
                patch.dict("os.environ", RF_DATA_DIR=temporary),
                patch.object(llama_server.requests, "Session", return_value=session),
                patch.object(
                    llama_server,
                    "resolve_runtime",
                    return_value=(Path("fixture.exe"), {}, True, "Vulkan", ""),
                ),
                patch.object(llama_server, "projector_path", return_value=None),
                patch.object(llama_server, "start_process", return_value=(process, None)),
                patch.object(llama_server.time, "sleep"),
            ):
                with self.assertRaises(CancelledError):
                    llama_server.Server(
                        model,
                        {},
                        load_id="owned",
                        progress_callback=events.append,
                        check_cancelled=check,
                    )
            process.terminate.assert_called_once()
            session.close.assert_called_once()
            self.assertEqual(events[-1]["phase"], "cancelled")
            self.assertFalse(any(event["phase"] == "ready" for event in events))


if __name__ == "__main__":
    unittest.main()
