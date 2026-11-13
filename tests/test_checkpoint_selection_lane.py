"""Native queue selection scheduling with fake file work; no server or model runtime."""

import ast
import asyncio
from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import subprocess
from types import SimpleNamespace
import unittest

import gradio as gr
from gradio.queueing import Event
from gradio.state_holder import SessionState

from ruinedfooocus.features.models.model_identity import file_signature
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_ui_scope(values):
    source = ROOT / "ruinedfooocus/features/models/ui_checkpoint_tools.py"
    names = {
        "checkpoint_source",
        "CheckpointToolSelection",
        "selected_result",
        "checkpoint_result",
        "publish_checkpoint_result",
        "run_action",
        "create_ui",
    }
    nodes = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    scope = dict(
        gr=gr,
        Path=Path,
        threading=threading,
        CancelledError=CancelledError,
        file_signature=file_signature,
        version_js=function_from(
            "ruinedfooocus/features/models/ui_requests.py", "version_js", {"json": json}
        ),
        stamp=function_from("ruinedfooocus/features/models/ui_requests.py", "stamp", {}),
        translate_ui=lambda value: value,
        NATIVE_TORCH_SUFFIXES=(".ckpt",),
        is_safetensors=lambda name: str(name).endswith(".safetensors"),
        SafetensorError=ValueError,
    )
    scope.update(values)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


class CheckpointSelectionLaneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-checkpoint-selection-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = {name: self.root / name for name in ("a.safetensors", "b.safetensors")}
        for path in self.paths.values():
            path.write_bytes(b"owned fake header")
        self.names = list(self.paths)
        self.started, self.release = threading.Event(), threading.Event()
        self.addCleanup(self.release.set)
        self.scope = checkpoint_ui_scope(
            dict(
                shared=SimpleNamespace(
                    models=SimpleNamespace(get_names=lambda _: list(self.names))
                ),
                _selected=lambda name: (self.paths[name], {}),
                tools=SimpleNamespace(
                    inspect_checkpoint=lambda _: {"dtypes": {"F32": 8}, "saving_bytes": 4}
                ),
            )
        )

        def fake_action(name, action, *_args, **_kwargs):
            if action in ("check", "convert"):
                self.started.set()
                if not self.release.wait(5):
                    raise AssertionError("Fake file work was not released")
                if action == "convert":
                    self.names.append("a.fp16.safetensors")
            return f"{action}: {name}"

        self.scope["run_action"] = fake_action
        with gr.Blocks() as self.demo:
            self.picker = gr.Dropdown(choices=self.names, value=self.names[0])
            self.scope["create_ui"](self.picker)
        self.demo.queue()
        self.functions = {fn.fn.__name__: fn for fn in self.demo.fns.values() if fn.fn is not None}
        self.session = self.scope["CheckpointToolSelection"]()

    def select(self, name, version):
        self.functions["begin_selection"].fn(name, self.session, version)
        return self.functions["selection_current"].fn(name, self.session, version)["updates"]

    def schedule(self, name):
        fn = self.functions[name]
        queue = self.demo._queue
        queue.create_event_queue_for_fn(fn)
        event = Event("owned-session", fn, None, None)
        queue.event_queue_per_concurrency_id[fn.concurrency_id].queue.append(event)
        return event

    def take(self):
        selected = self.demo._queue.get_events()
        self.assertIsNotNone(selected, "The native scheduler blocked an independent selection")
        events, _, group = selected
        self.demo._queue.event_queue_per_concurrency_id[group].current_concurrency += 1
        return events[0]

    def test_selection_runs_during_serial_file_work_and_late_report_is_excluded(self):
        for action in ("check_current", "convert_current"):
            with self.subTest(action=action):
                self.session = self.scope["CheckpointToolSelection"]()
                self.started.clear()
                self.release.clear()
                self.select("a.safetensors", 1)
                self.schedule(action)
                active = self.take()
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(active.fn.fn, "a.safetensors", self.session, 1)
                try:
                    self.assertTrue(self.started.wait(2))
                    # All explicit disk tools still share the occupied native group.
                    self.schedule("inspect_current")
                    self.assertIsNone(self.demo._queue.get_events())
                    self.functions["begin_selection"].fn("b.safetensors", self.session, 2)
                    selection = self.schedule("selection_current")
                    self.assertIs(self.take(), selection)
                    result = selection.fn.fn("b.safetensors", self.session, 2)["updates"]
                    self.assertEqual(result[0], "inspect: b.safetensors")
                    self.assertFalse(future.done())
                    self.release.set()
                    old_result = future.result(timeout=2)["updates"]
                    if action == "check_current":
                        self.assertEqual(old_result, [gr.skip()])
                    else:
                        self.assertEqual(old_result[0], gr.skip())
                        self.assertIn(
                            ("a.fp16.safetensors", "a.fp16.safetensors"), old_result[1]["choices"]
                        )
                        self.assertNotIn("value", old_result[1])
                finally:
                    self.release.set()
                    executor.shutdown(wait=True, cancel_futures=True)
                    for group in self.demo._queue.event_queue_per_concurrency_id.values():
                        group.current_concurrency = 0
                        group.queue.clear()

    def test_file_replacement_and_superseded_passive_result_are_fenced(self):
        session = self.session
        a = session.capture("a.safetensors", 1, select=True)
        session.capture("b.safetensors", 2, select=True)
        with self.assertRaises(CancelledError):
            session.publish(a, lambda: "stale")
        with self.assertRaises(CancelledError):
            session.capture("a.safetensors", 1, select=True)
        b = session.capture("b.safetensors", 2)
        self.paths["b.safetensors"].write_bytes(b"replacement with a distinct signature")
        with self.assertRaises(CancelledError):
            session.publish(b, lambda: "stale source")
        self.assertEqual(
            self.functions["check_current"].fn("b.safetensors", session, 2)["updates"], [gr.skip()]
        )

    def test_selection_ticket_invalidates_a_running_passive_report_before_next_report_runs(self):
        self.functions["begin_selection"].fn("a.safetensors", self.session, 1)

        def blocked_report(*_args, **_kwargs):
            self.started.set()
            if not self.release.wait(5):
                raise AssertionError("Fake header read was not released")
            return "old passive report"

        self.scope["run_action"] = blocked_report
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self.functions["selection_current"].fn, "a.safetensors", self.session, 1
        )
        try:
            self.assertTrue(self.started.wait(2))
            self.functions["begin_selection"].fn("b.safetensors", self.session, 2)
            self.release.set()
            self.assertEqual(future.result(timeout=2)["updates"], [gr.skip()] * 3)
        finally:
            self.release.set()
            executor.shutdown(wait=True, cancel_futures=True)

    def test_native_json_result_delivered_after_new_selection_cannot_overwrite_it(self):
        state = SessionState(self.demo)
        self.release.set()

        def call(name, inputs):
            response = asyncio.run(
                self.demo.process_api(self.functions[name]._id, inputs, state=state)
            )["data"][0]
            return response.model_dump() if hasattr(response, "model_dump") else response

        call("begin_selection", ["a.safetensors", None, 1])
        old_results = [
            call("inspect_current", ["a.safetensors", 0, None, 1]),
            call("check_current", ["a.safetensors", None, 1]),
            call("convert_current", ["a.safetensors", None, 1]),
        ]
        native_choices = gr.Dropdown(choices=self.names).choices
        self.assertEqual(
            old_results[2]["updates"][1]["choices"], [list(pair) for pair in native_choices]
        )
        self.assertNotIn("value", old_results[2]["updates"][1])
        call("begin_selection", ["b.safetensors", None, 2])
        current = call("selection_current", ["b.safetensors", None, 2])
        dependencies = {item["id"]: item for item in self.demo.config["dependencies"]}

        def publisher(name):
            producer = self.functions[name]
            self.assertIsInstance(producer.outputs[0], gr.JSON)
            client = next(
                item for item in dependencies.values() if item["trigger_after"] == producer._id
            )
            self.assertFalse(client["queue"])
            return client["js"]

        payload = dict(
            current=current,
            current_js=publisher("selection_current"),
            old_results=old_results,
            old_js=[
                publisher(name) for name in ("inspect_current", "check_current", "convert_current")
            ],
        )
        harness = r"""
const assert = require('node:assert/strict'), vm = require('node:vm');
const data = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const scope = {window: {__rfModelUIRequests: {'checkpoint-tools-selection': 1}}};
const copy = vm.runInNewContext('(' + data.old_js[2] + ')', scope)(data.old_results[2]);
assert.equal(copy[0], 'convert: a.safetensors');
const nativeChoices = copy[1].choices.map(([label, value]) => [label, value]);
assert.deepEqual(JSON.parse(JSON.stringify(nativeChoices)), [
    ['a.safetensors', 'a.safetensors'], ['b.safetensors', 'b.safetensors'],
    ['a.fp16.safetensors', 'a.fp16.safetensors']
]);
let selected = 'a.safetensors';
if ('value' in copy[1]) selected = copy[1].value;
assert.equal(selected, 'a.safetensors');
scope.window.__rfModelUIRequests['checkpoint-tools-selection'] = 2;
const current = vm.runInNewContext('(' + data.current_js + ')', scope)(data.current);
assert.equal(current[0], 'inspect: b.safetensors');
const visible = current.slice();
for (let index = 0; index < data.old_results.length; index++) {
    const updates = vm.runInNewContext('(' + data.old_js[index] + ')', scope)(data.old_results[index]);
    assert.ok(updates.every(update => update.__type__ === 'update' && Object.keys(update).length === 1));
}
assert.deepEqual(visible, current);
assert.ok(data.old_results[2].updates[1].choices.some(([label, value]) =>
    label === 'a.fp16.safetensors' && value === 'a.fp16.safetensors'));
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("a.fp16.safetensors", self.names)

    def test_sessions_are_independent_and_selection_event_is_finitely_bounded(self):
        first = self.session
        second = self.scope["CheckpointToolSelection"]()
        ticket = first.capture("a.safetensors", 1, select=True)
        second.capture("b.safetensors", 9, select=True)
        self.assertEqual(first.publish(ticket, lambda: "current"), "current")
        passive = self.functions["selection_current"]
        self.assertEqual(passive.concurrency_limit, 2)
        self.assertEqual(passive.trigger_mode, "always_last")
        self.assertEqual(passive.show_progress_on, [])
        self.assertFalse(self.functions["begin_selection"].queue)
        self.assertEqual(self.functions["begin_selection"].trigger_mode, "multiple")
        for name in ("inspect_current", "check_current", "convert_current"):
            self.assertEqual(self.functions[name].concurrency_id, "checkpoint_tools")
            self.assertEqual(self.functions[name].concurrency_limit, 1)


if __name__ == "__main__":
    unittest.main()
