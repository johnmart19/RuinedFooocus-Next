"""Bounded localhost CPU/IO acceptance; synthetic bytes/images, never model execution.

Uses production artifact preparation, request fencing, admission and SQLite index
contracts in an inert Gradio UI. It does not import application state or native
runtimes. Service timings are not browser acknowledgement/paint measurements.
"""

import argparse
import asyncio
from concurrent.futures import CancelledError
from contextlib import closing, contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.app.resource_lifecycle import FairAdmission, JobRegistry, LatestUIRequests
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.gallery.index_store import (
    connect_index,
    incremental_scan,
    page_snapshot,
)


class Fixture:
    def __init__(self, root, *, count=128, deadline=30):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self._source_identity = self._capture_source_identity()
        self.deadline = deadline
        self._lock = threading.RLock()
        self._download_gate, self.http_started = threading.Event(), threading.Event()
        self._requests, self._refreshes, self.measurements = {}, {}, []
        self.scan_started = {}
        self.fence, self.pool = LatestUIRequests(), PreparationAdmission()
        self.jobs, self.admission = JobRegistry(), FairAdmission()
        self.outputs = self.root / "synthetic-library"
        self.outputs.mkdir()
        self.database = self.root / "cache/images.db"
        for index in range(count):
            self._image(self.outputs / f"fixture-{index:04d}.png", f"fixture {index % 8}")
        with closing(connect_index(self.database)) as connection:
            self.cold_scan = incremental_scan(connection, [self.outputs], self._metadata)
        self.installed = self.root / "installed-B.inert"
        self.installed.write_bytes(b"installed B: synthetic fixture, no model format or runtime")
        self.payload = b"inert transfer fixture\n" * 100000
        self.http_requests = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/inert-payload":
                    self.send_error(404)
                    return
                with owner._lock:
                    owner.http_requests += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(owner.payload)))
                self.end_headers()
                try:
                    self.wfile.write(owner.payload[: 1024 * 1024])
                    self.wfile.flush()
                    owner.http_started.set()
                    owner._download_gate.wait(owner.deadline)
                    self.wfile.write(owner.payload[1024 * 1024 :])
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()
        self.transfers = PreparationService(staging_root=self.root / "transfer-cache")
        self.request = PreparationRequest(
            f"http://127.0.0.1:{self.http.server_port}/inert-payload",
            self.root / "download-A.inert",
            expected_size=len(self.payload),
            sha256=hashlib.sha256(self.payload).hexdigest(),
        )

    @staticmethod
    def _image(path, prompt):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        metadata = PngInfo()
        metadata.add_text("parameters", json.dumps({"Prompt": prompt, "width": 7, "height": 11}))
        Image.new("RGBA", (7, 11), (20, 40, 60, 127)).save(path, pnginfo=metadata)

    @staticmethod
    def _metadata(path):
        from PIL import Image

        with Image.open(path) as image:
            return dict(image.info)

    @contextmanager
    def measured(self, operation, owner):
        started = time.perf_counter()
        try:
            yield
        finally:
            with self._lock:
                self.measurements.append(
                    {
                        "operation": operation,
                        "owner": owner,
                        "service_ms": (time.perf_counter() - started) * 1000,
                    }
                )

    def download(self, owner):
        with self.measured("download_ack", owner), self._lock:
            previous = self._requests.get(owner)
            if previous is not None and not previous.done():
                raise ValueError("This session already subscribed to A")
            self._requests[owner] = self.transfers.submit(self.request)
            return self.poll(owner)

    def poll(self, owner):
        with self._lock:
            subscriber = self._requests.get(owner)
        state = "idle"
        if subscriber is not None:
            state = "running"
            if subscriber.done():
                try:
                    subscriber.result()
                    state = "complete"
                except CancelledError:
                    state = "cancelled"
                except Exception as error:
                    state = str(error)
        return {
            "download": state,
            "published": self.request.destination.is_file(),
            "http_transfers": self.http_requests,
        }

    def cancel_download(self, owner):
        with self.measured("download_cancel_ack", owner), self._lock:
            subscriber = self._requests.get(owner)
            if subscriber is not None:
                subscriber.cancel()
            return self.poll(owner)

    def release_download(self):
        self._download_gate.set()
        return {"released": "inert HTTP body"}

    def reset_download(self):
        with self._lock:
            with self.transfers.artifact_lease(
                self.request.destination, write=True, blocking=False
            ):
                self.request.destination.unlink(missing_ok=True)
                self._download_gate.clear()
                self.http_started.clear()
            return {"reset": "inert A only"}

    def use_installed(self, owner):
        with self.measured("use_installed_B", owner):
            job = self.jobs.create(owner=("fixture", owner))
            try:
                with self.admission.lease(job):
                    digest = hashlib.sha256(self.installed.read_bytes()).hexdigest()
                return {"selected": "installed B", "sha256": digest, "native_execution": False}
            finally:
                self.jobs.finish(job)

    def browse(self, owner, term=""):
        with self.measured("browse", owner), closing(sqlite3.connect(self.database)) as connection:
            count, pages, page, rows = page_snapshot(connection, 1, term, 24)
            revision = connection.execute(
                "SELECT revision FROM library_snapshot WHERE id=1"
            ).fetchone()[0]
            return {
                "count": count,
                "pages": pages,
                "page": page,
                "revision": revision,
                "files": [row[1] for row in rows],
            }

    def acknowledge(self, owner, version, term="", *, hold=True, fail=False):
        with self.measured("refresh_ack", owner):
            ticket = self.fence.begin(("fixture", owner), "fixture-library", int(version))
            with self._lock:
                self._refreshes[(owner, int(version))] = (ticket, threading.Event())
                self.scan_started[(owner, int(version))] = threading.Event()
            return {"version": int(version), "term": term, "hold": hold, "fail": fail}

    async def prepare(self, owner, descriptor):
        version = int(descriptor["version"])
        try:
            ticket = self.fence.get(("fixture", owner), "fixture-library", version, start=True)
        except (CancelledError, asyncio.CancelledError):
            return {"superseded": version}
        with self._lock:
            _, gate = self._refreshes[(owner, version)]

        def work():
            with (
                self.measured("refresh_work", owner),
                closing(sqlite3.connect(self.database)) as connection,
            ):
                if descriptor["hold"] or descriptor["fail"]:
                    identity = hashlib.sha256(owner.encode()).hexdigest()[:12]
                    self._image(
                        self.outputs / f"candidate-{identity}-{version}.png", "pending fixture"
                    )

                def metadata(path):
                    self.scan_started[(owner, version)].set()
                    if descriptor["fail"]:
                        raise OSError("Controlled metadata failure; previous index retained")
                    if descriptor["hold"]:
                        expires = time.monotonic() + self.deadline
                        while not gate.wait(0.02):
                            if ticket.cancelled.is_set():
                                raise CancelledError()
                            if time.monotonic() >= expires:
                                raise TimeoutError("Controlled refresh deadline")
                    return self._metadata(path)

                result = incremental_scan(
                    connection,
                    [self.outputs],
                    metadata,
                    cancelled=ticket.cancelled,
                    publish=lambda commit: self.fence.publish(ticket, commit),
                )
                return self.fence.publish(
                    ticket,
                    lambda: {
                        "version": version,
                        "parsed": result.parsed,
                        "reused": result.reused,
                        **self.browse(owner, descriptor["term"]),
                    },
                )

        try:
            result = await asyncio.wrap_future(
                self.pool.submit_work(work, cancelled=ticket.cancelled)
            )
            return self.fence.publish(ticket, lambda: result)
        except (CancelledError, asyncio.CancelledError):
            return {"superseded": version}
        except (OSError, TimeoutError) as error:
            return {"error": str(error), **self.browse(owner)}
        finally:
            self.fence.finish(ticket)
            with self._lock:
                self._refreshes.pop((owner, version), None)

    def cancel_refresh(self, owner):
        with self.measured("refresh_cancel_ack", owner):
            self.fence.cancel(("fixture", owner), "fixture-library")
            return {"cancelled": "refresh", **self.browse(owner)}

    def release_refresh(self, owner):
        with self._lock:
            for (session, _), (_, gate) in self._refreshes.items():
                if session == owner:
                    gate.set()
        return {"released": "this session's scan"}

    @staticmethod
    def _capture_source_identity():
        paths = [
            "tools/concurrency_fixture.py",
            "ruinedfooocus/core/preparation.py",
            "ruinedfooocus/app/preparation_admission.py",
            "ruinedfooocus/app/resource_lifecycle.py",
            "ruinedfooocus/features/gallery/index_store.py",
            "ruinedfooocus/ui/app.py",
        ]
        return {
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": {
                path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
            },
        }

    def receipt(self):
        import gradio

        return {
            "scope": "CPU/IO synthetic localhost fixture; no model execution/acquisition, installs or GPU probes; service timings do not establish browser delivery",
            "source_revision": self._source_identity["source_revision"],
            "source_sha256": dict(self._source_identity["source_sha256"]),
            "gradio": gradio.__version__,
            "captured_at_unix": time.time(),
            "python": sys.version,
            "pid": os.getpid(),
            "owned_root": str(self.root),
            "http_port": self.http.server_port,
            "http_transfers": self.http_requests,
            "cold_scan": vars(self.cold_scan),
            "measurements": list(self.measurements),
        }

    def export_receipt(self):
        path = self.root / "exported-receipt.json"
        save_json(path, self.receipt())
        return {"receipt": str(path), "source_revision": self._source_identity["source_revision"]}

    def close(self):
        self._download_gate.set()
        with self._lock:
            for ticket, gate in self._refreshes.values():
                ticket.cancelled.set()
                gate.set()
        self.pool.close()
        self.transfers.close()
        self.http.shutdown()
        self.http.server_close()
        self.http_thread.join(5)


def build_app(fixture):
    import gradio as gr

    def owner(request):
        return request.session_hash

    with gr.Blocks(analytics_enabled=False) as app:
        gr.Markdown(
            "# Concurrency acceptance fixture\nSynthetic CPU/IO only. A and B are inert bytes; Use never loads a model."
        )
        navigation = gr.JSON(label="Navigation acknowledgement")
        with gr.Tabs():
            with gr.Tab("Create") as create:
                gr.Textbox(value="Keep this Create draft", label="Create draft", interactive=True)
                use = gr.Button("Use installed B (inert)")
                installed_status = gr.JSON(label="Installed B status")
            with gr.Tab("Chat") as chat:
                gr.Textbox(value="Keep this Chat draft", label="Chat draft", interactive=True)
            with gr.Tab("Library") as library:
                term = gr.Textbox(label="Filter synthetic Library")
                version = gr.Number(value=0, visible=False)
                descriptor = gr.JSON(visible=False)
                hold = gr.Checkbox(value=True, label="Hold refresh at a metadata boundary")
                fail = gr.Checkbox(value=False, label="Fail metadata read (rollback fixture)")
                refresh = gr.Button("Refresh Library")
                cancel_refresh = gr.Button("Cancel refresh")
                release_refresh = gr.Button("Release refresh")
                browse = gr.Button("Browse previous committed snapshot")
                page = gr.JSON(value=fixture.browse("initial"), label="Committed Library snapshot")
                scan_status = gr.JSON(label="Refresh acknowledgement/progress")
            with gr.Tab("Settings") as settings:
                download = gr.Button("Download A (local inert fixture)")
                cancel_download = gr.Button("Cancel download A")
                release_download = gr.Button("Release download A")
                reset_download = gr.Button("Reset inert A after completion")
                export_receipt = gr.Button("Export source and service receipt")
                receipt_status = gr.JSON(label="Exported receipt")
                transfer_status = gr.JSON(label="Transfer status")
                transfer_timer = gr.Timer(0.25)

        def download_ack(request: gr.Request):
            return fixture.download(owner(request))

        def download_cancel(request: gr.Request):
            return fixture.cancel_download(owner(request))

        def poll(request: gr.Request):
            return fixture.poll(owner(request))

        def installed(request: gr.Request):
            return fixture.use_installed(owner(request))

        def acknowledge(term, version, hold, fail, request: gr.Request):
            descriptor = fixture.acknowledge(owner(request), version, term, hold=hold, fail=fail)
            return descriptor, {"acknowledged": version}

        async def prepare(descriptor, request: gr.Request):
            result = await fixture.prepare(owner(request), descriptor)
            if "superseded" in result:
                return gr.skip(), gr.skip()
            return result, result

        def cancel(request: gr.Request):
            return fixture.cancel_refresh(owner(request))

        def release(request: gr.Request):
            return fixture.release_refresh(owner(request))

        def read(term, request: gr.Request):
            return fixture.browse(owner(request), term)

        download.click(
            download_ack,
            outputs=transfer_status,
            queue=False,
            show_progress="hidden",
            api_name="download",
        )
        export_receipt.click(
            fixture.export_receipt,
            outputs=receipt_status,
            queue=False,
            show_progress="hidden",
            api_name="export_receipt",
        )
        cancel_download.click(
            download_cancel,
            outputs=transfer_status,
            queue=False,
            show_progress="hidden",
            api_name="cancel_download",
        )
        release_download.click(
            fixture.release_download,
            outputs=transfer_status,
            queue=False,
            show_progress="hidden",
            api_name="release_download",
        )
        reset_download.click(
            fixture.reset_download,
            outputs=transfer_status,
            queue=False,
            show_progress="hidden",
            api_name="reset_download",
        )
        transfer_timer.tick(
            poll,
            outputs=transfer_status,
            queue=False,
            show_progress="hidden",
            api_name="poll_download",
        )
        use.click(
            installed,
            outputs=installed_status,
            queue=False,
            show_progress="hidden",
            api_name="use_installed",
        )
        refresh.click(
            acknowledge,
            inputs=[term, version, hold, fail],
            outputs=[descriptor, scan_status],
            js="(term, version, hold, fail) => [term, (window.__rfFixtureVersion = (window.__rfFixtureVersion || 0) + 1), hold, fail]",
            queue=False,
            show_progress="hidden",
            api_name="ack_refresh",
        )
        descriptor.change(
            prepare,
            inputs=descriptor,
            outputs=[page, scan_status],
            concurrency_id="fixture_library",
            concurrency_limit=2,
            trigger_mode="always_last",
            show_progress="minimal",
            show_progress_on=scan_status,
            api_name="prepare_refresh",
        )
        cancel_refresh.click(
            cancel,
            outputs=scan_status,
            queue=False,
            show_progress="hidden",
            api_name="cancel_refresh",
        )
        release_refresh.click(
            release,
            outputs=scan_status,
            queue=False,
            show_progress="hidden",
            api_name="release_refresh",
        )
        browse.click(
            read, inputs=term, outputs=page, queue=False, show_progress="hidden", api_name="browse"
        )
        term.input(
            read, inputs=term, outputs=page, queue=False, show_progress="hidden", api_name="filter"
        )
        for tab, name in (
            (create, "Create"),
            (chat, "Chat"),
            (library, "Library"),
            (settings, "Settings"),
        ):

            def navigate(name=name):
                return {"destination": name, "server_time": time.time()}

            tab.select(
                navigate,
                outputs=navigation,
                queue=False,
                show_progress="hidden",
                api_name=f"navigate_{name.lower()}",
            )
    return app.queue(max_size=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "tmp" / ("concurrency-fixture-" + uuid.uuid4().hex)
    )
    parser.add_argument("--port", type=int, default=7906)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900 or not 1024 <= args.port <= 65535:
        parser.error("Use a 1..900 second lifetime and an unprivileged port")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    fixture = Fixture(args.root)
    app = None
    try:
        app = build_app(fixture)
        app.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            prevent_thread_lock=True,
            inbrowser=False,
            share=False,
            quiet=True,
        )
        save_json(
            fixture.root / "source-receipt.json", {**fixture.receipt(), "gradio_port": args.port}
        )
        print(
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{args.port}",
                    "owned_root": str(fixture.root),
                    "receipt": str(fixture.root / "source-receipt.json"),
                    "pid": os.getpid(),
                }
            ),
            flush=True,
        )
        expires = time.monotonic() + args.seconds
        stop = fixture.root / "stop"
        while not stop.is_file() and time.monotonic() < expires:
            threading.Event().wait(min(0.25, max(0, expires - time.monotonic())))
    finally:
        if app is not None:
            app.close()
        fixture.close()
        save_json(fixture.root / "final-receipt.json", fixture.receipt())


if __name__ == "__main__":
    main()
