"""Real bounded localhost HTTP/Gradio/SQLite overlap; no app or native runtime import."""

import ast
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

import gradio as gr
from gradio_client import Client
import requests

from tools.concurrency_fixture import Fixture, ROOT, build_app


class ConcurrencyServiceTests(unittest.TestCase):
    def wait_until(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            threading.Event().wait(0.02)
        self.fail("Controlled fixture did not reach its completion boundary")

    def test_real_http_transfer_and_gradio_requests_overlap_cancel_and_keep_previous_sqlite_snapshot(
        self,
    ):
        temporary = tempfile.TemporaryDirectory(prefix="rf-service-concurrency-")
        self.addCleanup(temporary.cleanup)
        fixture = Fixture(Path(temporary.name) / "owned", deadline=60)
        self.addCleanup(fixture.close)
        app = build_app(fixture)
        self.addCleanup(app.close)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            inbrowser=False,
            share=False,
            quiet=True,
        )
        first = Client(f"http://127.0.0.1:{port}", verbose=False, max_workers=4)
        second = Client(f"http://127.0.0.1:{port}", verbose=False, max_workers=4)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        durations = []
        direct_durations = []
        http = requests.Session()
        http.trust_env = False
        self.addCleanup(http.close)

        def call(client, endpoint, *args):
            started = time.perf_counter()
            result = client.predict(*args, api_name="/" + endpoint)
            durations.append(
                {"operation": endpoint, "http_rpc_ms": (time.perf_counter() - started) * 1000}
            )
            return result

        def direct(client, endpoint, *args):
            started = time.perf_counter()
            response = http.post(
                f"http://127.0.0.1:{port}/gradio_api/run/{endpoint}",
                json={"data": list(args), "session_hash": client.session_hash},
                timeout=5,
            )
            response.raise_for_status()
            direct_durations.append(
                {"operation": endpoint, "http_run_ms": (time.perf_counter() - started) * 1000}
            )
            return response.json()["data"][0]

        self.assertEqual(call(first, "download")["download"], "running")
        self.assertTrue(fixture.http_started.wait(5))
        self.assertEqual(call(second, "download")["http_transfers"], 1)
        self.assertEqual(call(first, "cancel_download")["download"], "cancelled")
        self.assertEqual(call(second, "poll_download")["download"], "running")
        self.assertEqual(call(first, "use_installed")["selected"], "installed B")
        self.assertFalse(fixture._download_gate.is_set())

        descriptor, acknowledgement = call(first, "ack_refresh", "", 1, True, False)
        self.assertEqual(acknowledgement["acknowledged"], 1)
        pending = first.submit(descriptor, api_name="/prepare_refresh")
        self.assertTrue(fixture.scan_started[(first.session_hash, 1)].wait(5))
        self.assertEqual(call(second, "browse", "")["count"], 128)
        self.assertEqual(call(second, "navigate_chat")["destination"], "Chat")
        self.assertFalse(pending.done())
        for _ in range(20):
            self.assertEqual(direct(second, "filter", "fixture 1")["count"], 16)
        self.assertFalse(pending.done())
        self.assertEqual(direct(first, "cancel_refresh")["count"], 128)
        stale = pending.result(timeout=5)
        self.assertTrue(all(value.get("__type__") == "update" for value in stale))
        self.assertEqual(fixture.browse("verify")["count"], 128)

        # Two sessions hold independent tickets; cancelling A cannot cancel B.
        other, _ = call(second, "ack_refresh", "", 1, True, False)
        other_pending = second.submit(other, api_name="/prepare_refresh")
        self.assertTrue(fixture.scan_started[(second.session_hash, 1)].wait(5))
        call(first, "cancel_refresh")
        self.assertFalse(other_pending.done())
        self.assertEqual(call(first, "browse", "")["count"], 128)
        call(second, "release_refresh")
        self.assertEqual(other_pending.result(timeout=5)[0]["version"], 1)

        # New acknowledgement cancels an earlier scan before its publication.
        old_descriptor, _ = call(first, "ack_refresh", "", 2, True, False)
        old = first.submit(old_descriptor, api_name="/prepare_refresh")
        self.assertTrue(fixture.scan_started[(first.session_hash, 2)].wait(5))
        latest, _ = call(first, "ack_refresh", "", 3, False, False)
        latest_result = first.submit(latest, api_name="/prepare_refresh").result(timeout=5)[0]
        self.assertEqual(latest_result["version"], 3)
        self.assertTrue(all(value.get("__type__") == "update" for value in old.result(timeout=5)))

        before = call(first, "browse", "")
        broken, _ = call(first, "ack_refresh", "", 4, False, True)
        failed = first.submit(broken, api_name="/prepare_refresh").result(timeout=5)[0]
        self.assertIn("Controlled metadata failure", failed["error"])
        self.assertEqual(
            (failed["count"], failed["revision"]), (before["count"], before["revision"])
        )
        retry, _ = call(first, "ack_refresh", "", 5, False, False)
        self.assertNotIn(
            "error", first.submit(retry, api_name="/prepare_refresh").result(timeout=5)[0]
        )
        warm, _ = call(first, "ack_refresh", "", 6, False, False)
        self.assertEqual(
            first.submit(warm, api_name="/prepare_refresh").result(timeout=5)[0]["parsed"], 0
        )

        self.assertEqual(call(second, "cancel_download")["download"], "cancelled")
        call(first, "release_download")
        self.wait_until(lambda: not fixture.transfers._jobs)
        self.assertFalse(fixture.request.destination.exists())
        call(first, "reset_download")
        call(first, "download")
        self.assertTrue(fixture.http_started.wait(5))
        call(first, "release_download")
        self.wait_until(lambda: fixture.poll(first.session_hash)["download"] == "complete")
        self.assertEqual(
            hashlib.sha256(fixture.request.destination.read_bytes()).hexdigest(),
            fixture.request.sha256,
        )

        for _ in range(20):
            call(first, "filter", "fixture 1")
        warm_times = sorted(row["http_rpc_ms"] for row in durations if row["operation"] == "filter")
        direct_times = sorted(
            row["http_run_ms"] for row in direct_durations if row["operation"] == "filter"
        )
        receipt = {
            **fixture.receipt(),
            "http_rpc": durations,
            "warm_filter_rpc_p50_ms": warm_times[len(warm_times) // 2],
            "warm_filter_rpc_p95_ms": warm_times[int(len(warm_times) * 0.95) - 1],
            "queue_free_http": direct_durations,
            "warm_filter_direct_http_p50_ms": direct_times[len(direct_times) // 2],
            "warm_filter_direct_http_p95_ms": direct_times[int(len(direct_times) * 0.95) - 1],
            "browser_delivery": "not measured by HTTP/Gradio client",
        }
        evidence = ROOT / "tmp" / "concurrency-service-receipt.json"
        evidence.parent.mkdir(exist_ok=True)
        evidence.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"\nCPU/IO real-service receipt: {evidence}")

    def test_installed_gradio_limits_generation_loading_to_its_progress_control(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        event = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(
                keyword.arg == "fn"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "generate_clicked"
                for keyword in node.keywords
            )
        )
        names = {node.id for node in ast.walk(event) if isinstance(node, ast.Name)} - {
            "generate_clicked",
            "refresh_seed",
            "state",
        }
        with gr.Blocks(analytics_enabled=False) as app:
            scope = {name: gr.Textbox() for name in names if name not in {"shared"}}
            scope["shared"] = type(
                "Shared",
                (),
                {"get_ctrl": staticmethod(lambda name: scope.setdefault(name, gr.Textbox()))},
            )()
            scope.update(
                state={"ctrls_obj": []},
                generate_clicked=lambda: None,
                refresh_seed=lambda *args: None,
            )
            exec(compile(ast.Expression(event), "Create generation event", "eval"), scope)
        generation = next(
            function for function in app.fns.values() if function.fn is scope["generate_clicked"]
        )
        self.assertGreater(len(generation.outputs), 1)
        self.assertIn(scope["progress_html"], generation.outputs)
        self.assertEqual(generation.show_progress_on, [scope["progress_html"]])
        self.assertEqual(generation.show_progress, "minimal")


if __name__ == "__main__":
    unittest.main()
