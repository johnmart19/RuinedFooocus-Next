"""Fake runtime/device queries verify full-refresh ownership and passive CPU policy."""

from concurrent.futures import CancelledError
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from fastapi.testclient import TestClient
from gradio.routes import App

from ruinedfooocus.backends.llama import llama_runtime_info as info
from ruinedfooocus.backends.llama import llama_installer
from ruinedfooocus.features.chat.ui_runtime_info import (
    RuntimeInfoSession,
    create_runtime_info_controls,
)


class RuntimeInfoRefreshTests(unittest.TestCase):
    def test_native_cancel_route_removes_only_this_sessions_waiting_refresh(self):
        with gr.Blocks() as app:
            profile = gr.State({"settings": {"cpu_everywhere": True}})
            create_runtime_info_controls(SimpleNamespace(cpu=True), profile)
        app.queue(max_size=8)
        refresh = next(
            function
            for function in app.fns.values()
            if getattr(function.fn, "__name__", None) == "inspect"
        )
        config = app.get_config_file()
        self.assertTrue(any(row["cancels"] == [refresh._id] for row in config["dependencies"]))
        server = App.create_app(app)
        client = TestClient(server)
        self.addCleanup(client.close)
        events = []
        with patch(
            "ruinedfooocus.features.chat.ui_runtime_info.runtime_info",
            side_effect=AssertionError("queued refresh must not query a runtime"),
        ):
            for session in ("runtime-owner-a", "runtime-owner-b"):
                response = client.post(
                    "/gradio_api/queue/join",
                    json={"fn_index": refresh._id, "session_hash": session, "data": [None, None]},
                )
                self.assertEqual(response.status_code, 200, response.text)
                events.append(response.json()["event_id"])
            queue = app._queue.event_queue_per_concurrency_id["chat-runtime-inspection"]
            self.assertEqual([event._id for event in queue.queue], events)
            response = client.post(
                "/gradio_api/cancel",
                json={
                    "fn_index": refresh._id,
                    "session_hash": "runtime-owner-a",
                    "event_id": events[0],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), {"success": True})
            self.assertEqual([event._id for event in queue.queue], [events[1]])
            self.assertNotIn(events[0], app._queue.event_ids_to_events)
            self.assertIn(events[1], app._queue.event_ids_to_events)
            client.post(
                "/gradio_api/cancel",
                json={
                    "fn_index": refresh._id,
                    "session_hash": "runtime-owner-b",
                    "event_id": events[1],
                },
            )

    def test_windows_sdk_metadata_reuses_the_same_bounded_query_without_install(self):
        with (
            patch.object(info, "run_bounded", return_value=(b'["owned-sdk/bin"]', b"")) as run,
            patch.object(
                llama_installer.subprocess,
                "run",
                side_effect=AssertionError("unbounded metadata child"),
            ),
            patch.object(
                llama_installer, "ensure_rocm_libraries", side_effect=AssertionError("install")
            ),
        ):
            result = llama_installer.windows_rocm_environment(
                {"PATH": "old"}, install=False, query=info._run
            )
        self.assertEqual(result["PATH"], os.pathsep.join(("owned-sdk/bin", "old")))
        self.assertEqual(run.call_args.kwargs["max_output"], 16 * 1024)
        self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_entire_refresh_has_one_owned_deadline_output_cap_and_offline_child(self):
        check = Mock()
        with patch.object(info, "run_bounded", return_value=(b"fixture runtime", b"")) as run:
            self.assertEqual(info.runtime_info(check_cancelled=check), "fixture runtime")
        check.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[1:3], ["-I", "-c"])
        self.assertIn("_inspect_runtime_info", command[-1])
        options = run.call_args.kwargs
        self.assertEqual(options["timeout"], 45)
        self.assertEqual(options["max_output"], 64 * 1024)
        self.assertIs(options["check_cancelled"], check)
        self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(options["env"]["TRANSFORMERS_OFFLINE"], "1")

    def test_global_cpu_displays_package_backend_identity_without_any_child_or_device_query(self):
        with (
            patch.object(info, "_installed_binaries", return_value=([], False)),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.xllamacpp_installed",
                return_value={"version": "fixture", "backend": "Vulkan"},
            ),
            patch.object(info, "run_bounded", side_effect=AssertionError("device child")),
            patch.object(info, "_run", side_effect=AssertionError("device query")),
        ):
            result = info.runtime_info(cpu=True)
        self.assertIn("recorded wheel backend: Vulkan", result)
        self.assertIn("device queries were not run", result)

    def test_discovery_caps_build_count_and_keeps_build_backend_identity(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, RF_DATA_DIR=folder, LLAMA_SERVER=""),
        ):
            root = Path(folder) / "cache/llama.cpp"
            executable = "llama-server.exe" if os.name == "nt" else "llama-server"
            for number in range(12):
                binary = root / f"b{number + 12000}" / f"backend-{number}" / executable
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"inert; never executed")
            binary = root / "b99999/build-work/hidden" / executable
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"must not query")
            binaries, truncated = info._installed_binaries()
            self.assertEqual(len(binaries), 8)
            self.assertTrue(truncated)
            self.assertTrue(
                all(
                    "b12" in info._build_label(binary) and "backend-" in info._build_label(binary)
                    for binary in binaries
                )
            )
            self.assertTrue(all("build-work" not in binary.parts for binary in binaries))

    def test_session_cancel_isolated_duplicate_fenced_and_child_completion_joined(self):
        first = RuntimeInfoSession()
        other = deepcopy(first)
        ready = Event()
        errors = []

        def fake_runtime(**options):
            ready.set()
            while not first.cancelled.wait(0.01):
                options["check_cancelled"]()
            options["check_cancelled"]()

        def refresh():
            try:
                first.refresh({}, SimpleNamespace(cpu=False))
            except BaseException as error:
                errors.append(error)

        with patch(
            "ruinedfooocus.features.chat.ui_runtime_info.runtime_info", side_effect=fake_runtime
        ):
            thread = Thread(target=refresh)
            thread.start()
            self.assertTrue(ready.wait(2))
            with self.assertRaisesRegex(ValueError, "already running"):
                with first.operation():
                    pass
            with other.operation() as untouched:
                first.cancel()
                self.assertFalse(untouched.is_set())
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CancelledError)
        self.assertIsNone(first.cancelled)

    def test_saved_cpu_profile_is_captured_and_controls_are_private_and_cancellable(self):
        launch = SimpleNamespace(cpu=False)
        with gr.Blocks() as app:
            profile = gr.State({"settings": {"cpu_everywhere": True}})
            create_runtime_info_controls(launch, profile)
        callbacks = {
            getattr(function.fn, "__name__", ""): function for function in app.fns.values()
        }
        self.assertEqual(callbacks["inspect"].concurrency_id, "chat-runtime-inspection")
        self.assertEqual(callbacks["inspect"].concurrency_limit, 2)
        self.assertFalse(callbacks["stop"].queue)
        self.assertTrue(
            all(callbacks[name].api_visibility == "private" for name in ("inspect", "stop"))
        )
        with patch(
            "ruinedfooocus.features.chat.ui_runtime_info.runtime_info", return_value="CPU metadata"
        ) as runtime:
            session = RuntimeInfoSession()
            events = list(
                callbacks["inspect"].fn(
                    session, {"settings": {"cpu_everywhere": True}}, gr.Request()
                )
            )
        self.assertEqual(events[-1], ("CPU metadata", ""))
        self.assertTrue(runtime.call_args.kwargs["cpu"])


if __name__ == "__main__":
    unittest.main()
