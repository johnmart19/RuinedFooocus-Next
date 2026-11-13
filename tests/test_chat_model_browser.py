"""Model search and evidence remain cheap, attributable and selection-neutral."""

import json
import ast
import asyncio
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch, Mock

from ruinedfooocus.core.paths import asset_path, resource_path
from ruinedfooocus.features.chat.model_browser import (
    bundle_presence,
    safe_model_evidence,
)
from ruinedfooocus.features.models.model_capabilities import RuntimeIdentity
from ruinedfooocus.features.models.model_registry import RegistryError, load_registry
from ui_test_helpers import function_from
from types import SimpleNamespace
import gradio as gr


class ChatModelBrowserTests(unittest.TestCase):
    def test_hub_selection_reuses_exact_local_family_and_quant_without_loading_or_replacing_draft(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = root / "first/model-Q4.gguf", root / "second/model-Q4.gguf"
            for path in (first, second):
                path.parent.mkdir()
                path.write_bytes(b"inert local fixture")
            catalogue = {"Family": [("Q4 Local", str(first)), ("Q4 Local", str(second))]}
            source, query, candidate = object(), object(), object()
            session = SimpleNamespace(active="keep loaded model", selected=None)
            select_source = Mock(return_value={candidate: str(second)})
            adapter = function_from(
                "ruinedfooocus/features/chat/ui.py",
                "select_catalogue_file",
                dict(
                    Path=Path,
                    local_models=lambda: {str(first): str(first), str(second): str(second)},
                    model_catalogue=lambda _: catalogue,
                    select_source=select_source,
                    llama_source=source,
                    model_query=query,
                ),
            )
            updates = adapter(second, session)
            self.assertEqual(updates, {source: "Local models", query: "", candidate: str(second)})
            select_source.assert_called_once_with("Local models", "Family", str(second), session)
            self.assertEqual(session.active, "keep loaded model")
            with self.assertRaises((OSError, ValueError)):
                adapter(root / "missing/model-Q4.gguf", session)
            source_tree = ast.parse(
                (
                    Path(__file__).resolve().parents[1] / "ruinedfooocus/features/chat/ui.py"
                ).read_text(encoding="utf-8")
            )
            outputs = next(
                node.value
                for node in ast.walk(source_tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "catalogue_selection_outputs"
                    for target in node.targets
                )
            )
            self.assertNotIn("llama_active_model", ast.unparse(outputs))
            self.assertNotIn("history", ast.unparse(outputs))

    def test_source_search_event_keeps_selection_through_filter_and_clear_in_real_gradio_state(
        self,
    ):
        from gradio.state_holder import SessionState
        from ruinedfooocus.features.chat.model_preparation import ChatModelSession
        from ruinedfooocus.features.chat.model_browser import model_action_state

        path = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/chat/ui.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"model_outputs", "search_inputs", "search_outputs"}
        assignments = {
            node.targets[0].id: node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        }
        binding = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "on"
            and any(
                k.arg == "fn" and isinstance(k.value, ast.Name) and k.value.id == "search_models"
                for k in node.value.keywords
            )
        )
        with gr.Blocks() as app:
            scope = dict(
                gr=gr,
                model_catalogue=lambda _: self.catalogue,
                model_label=lambda name: name,
                model_action_state=model_action_state,
                model_roots=lambda: {},
                path_manager=SimpleNamespace(DOWNLOADABLE_FILES=self.downloads),
                settings=SimpleNamespace(default_settings={}),
                configured_runtime=lambda _: self.native,
                initialized_runtime_status=lambda: {"runtime_snapshot": None},
            )
            scope.update(
                llama_source=gr.Radio(["Local models", "Downloadable"], value="Local models"),
                model_query=gr.Textbox(),
                model_session=gr.State(ChatModelSession()),
                llama_selection=gr.State("GPT OSS"),
                llama_selected_quant=gr.State("gpt-oss-20b-MXFP4.gguf"),
                llama_model=gr.Dropdown(["GPT OSS"], value="GPT OSS"),
                local_picker=gr.HTML({"choices": [["GPT OSS", "GPT OSS"]], "selected": "GPT OSS"}),
                llama_quant=gr.Dropdown(["gpt-oss-20b-MXFP4.gguf"], value="gpt-oss-20b-MXFP4.gguf"),
                llama_model_status=gr.Markdown(),
                llama_msg=gr.Textbox(),
                llama_use=gr.Button("Load", interactive=True),
                download_candidate=gr.Button("Download", visible=False),
                cancel_download=gr.Button("Cancel"),
                cancel_load=gr.Button("Cancel loading"),
            )
            function_from("ruinedfooocus/features/chat/ui.py", "action_updates", scope)
            function_from("ruinedfooocus/features/chat/ui.py", "clear_model_selection", scope)
            function_from("ruinedfooocus/features/chat/ui.py", "search_models", scope)
            exec(
                compile(
                    ast.Module(
                        body=[
                            assignments[name]
                            for name in ("model_outputs", "search_inputs", "search_outputs")
                        ]
                        + [binding],
                        type_ignores=[],
                    ),
                    str(path),
                    "exec",
                ),
                scope,
            )
        dependency = app.config["dependencies"][0]
        self.assertEqual(dependency["trigger_mode"], "always_last")
        self.assertFalse(dependency["queue"])
        targets = {tuple(target) for target in dependency["targets"]}
        self.assertTrue(
            {(scope["model_query"]._id, event) for event in ("input", "change", "submit")}.issubset(
                targets
            )
        )
        state = SessionState(app)
        session = ChatModelSession()
        session.active = "loaded-active.gguf"
        session.select("gpt-oss-20b-MXFP4.gguf")
        revision = session.revision
        state[scope["model_session"]._id] = session
        state[scope["llama_selection"]._id] = "GPT OSS"
        state[scope["llama_selected_quant"]._id] = "gpt-oss-20b-MXFP4.gguf"
        result = asyncio.run(
            app.process_api(0, ["Local models", None, None, "lfm", None], state=state)
        )
        self.assertEqual(state[scope["llama_selection"]._id], "GPT OSS")
        self.assertEqual(state[scope["llama_selected_quant"]._id], "gpt-oss-20b-MXFP4.gguf")
        self.assertEqual(session.selected, "gpt-oss-20b-MXFP4.gguf")
        self.assertEqual(session.revision, revision)
        self.assertEqual(session.active, "loaded-active.gguf")
        outputs = {
            component: data for component, data in zip(scope["model_outputs"], result["data"])
        }
        picker = outputs[scope["local_picker"]]["value"]
        self.assertEqual(picker["choices"], [["LFM3", "LFM3"]])
        self.assertEqual((picker["selected"], picker["selected_label"]), ("GPT OSS", "GPT OSS"))
        self.assertEqual(outputs[scope["llama_quant"]], gr.skip())
        self.assertEqual(outputs[scope["llama_use"]], gr.skip())
        self.assertTrue(state[scope["llama_use"]._id].interactive)
        self.assertFalse(state[scope["download_candidate"]._id].visible)
        self.assertFalse(outputs[scope["llama_model"]]["visible"])
        self.assertTrue(outputs[scope["local_picker"]]["visible"])
        settled = asyncio.run(
            app.process_api(0, ["Local models", None, None, "", None], state=state)
        )
        self.assertEqual(settled["changed_state_ids"], [])
        restored = dict(zip(scope["model_outputs"], settled["data"]))
        self.assertEqual(restored[scope["local_picker"]]["value"]["selected"], "GPT OSS")
        self.assertIn(["GPT OSS", "GPT OSS"], restored[scope["local_picker"]]["value"]["choices"])
        self.assertEqual(state[scope["llama_selected_quant"]._id], "gpt-oss-20b-MXFP4.gguf")
        self.assertEqual(session.selected, "gpt-oss-20b-MXFP4.gguf")
        self.assertEqual(session.revision, revision)

    def test_local_picker_keeps_selected_label_while_its_row_is_filtered_out(self):
        script = asset_path("chat_local_models.js").read_text(encoding="utf-8")
        harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const script = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const node = () => ({
  textContent: '', children: [], setAttribute() {}, addEventListener() {},
  focus() {}, append(...children) { this.children.push(...children); },
  replaceChildren() { this.children = []; }
});
const summary = node(), options = node(), details = node(), events = [];
const props = { value: {
  choices: [['LFM3', 'lfm']], selected: 'gpt', selected_label: 'GPT-OSS 20B'
} };
let render;
vm.runInNewContext(script, {
  props, document: { createElement: node }, trigger: (...event) => events.push(event),
  element: { querySelector: name => ({ summary, details, '.model-options': options })[name] },
  watch: (name, callback) => { render = callback; }
});
assert.equal(summary.textContent, 'GPT-OSS 20B');
assert.equal(options.children.length, 1);
assert.equal(options.children[0].children[0].textContent, 'LFM3');
props.value.choices.push(['GPT-OSS 20B', 'gpt']);
render();
assert.equal(summary.textContent, 'GPT-OSS 20B');
assert.equal(options.children.length, 2);
props.value = { choices: [], selected: null, selected_label: null };
render();
assert.equal(summary.textContent, 'Choose a local model');
assert.deepEqual(events, []);
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(script),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_transfer_snapshot_follows_configured_roots_and_unique_missing_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "second").mkdir()
            (root / "second/model.gguf").write_bytes(b"data")
            items = [
                {"id": key, "catalogue_key": key, "artifact": {"bytes": 4, "sha256": sha}}
                for key, sha in (("model", "a"), ("projector", "b"), ("alias", "b"))
            ]
            downloads = {
                key: {"filename": key + ".gguf", "path": "path_llm"}
                for key in ("model", "projector", "alias")
            }
            with patch.object(Path, "open", side_effect=AssertionError("No weight reads")):
                rows = bundle_presence(
                    items, downloads, {"path_llm": [root / "first", root / "second"]}
                )
            self.assertIn("Present by path/size; checksum unverified", rows[0][1])
            self.assertIn("Missing", rows[1][1])
            self.assertIn("(4 bytes)", rows[-1][1])  # One missing payload, two target aliases.

    def test_unknown_missing_sizes_and_wrong_local_sizes_remain_unresolved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.gguf"
            path.write_bytes(b"custom")
            item = {"id": "model", "catalogue_key": "model", "artifact": {"bytes": 4}}
            downloads = {"model": {"filename": "model.gguf", "path": "path_llm"}}
            rows = bundle_presence([item], downloads, {"path_llm": [folder]}, str(path))
            self.assertIn("size differs", rows[0][1])
            self.assertIn("Unknown", rows[-1][1])
            item["artifact"]["bytes"] = None
            rows = bundle_presence([item], downloads, {"path_llm": [folder]})
            self.assertIn("Unknown", rows[-1][1])
            rows = bundle_presence([item], downloads, {})
            self.assertIn("storage location unknown", rows[0][1])

    def test_disabled_and_custom_projectors_do_not_claim_automatic_missing_bytes(self):
        for projector in ("None", "/custom/projector.gguf"):
            rows = dict(
                safe_model_evidence(
                    "SmolVLM-256M-Instruct-Q8_0.gguf",
                    self.downloads,
                    self.native,
                    roots={"path_vision": ["missing-folder"]},
                    projector=projector,
                )
            )
            self.assertIn("Image bundle selection", rows)
            self.assertNotIn("Image missing payload bytes", rows)

    @classmethod
    def setUpClass(cls):
        cls.downloads = {}
        for path in resource_path("model_catalogue").glob("*.json"):
            cls.downloads.update(json.loads(path.read_text(encoding="utf-8")))
        cls.native = RuntimeIdentity("llama.cpp", "b12110")
        cls.catalogue = {
            "LFM3": [("Q4_K_M", "LFM3-1.7B-Q4_K_M.gguf")],
            "SmolVLM": [("Q8_0", "SmolVLM-256M-Instruct-Q8_0.gguf")],
            "GPT OSS": [("MXFP4", "gpt-oss-20b-MXFP4.gguf")],
            "Unknown": [("Original", "custom.gguf")],
        }

    def test_installed_size_does_not_establish_publisher_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "LFM3-1.7B-Q4_K_M.gguf"
            path.write_bytes(b"custom fixture")
            evidence = dict(safe_model_evidence(str(path), self.downloads, self.native))
            self.assertIn("local content is verified only when loaded", evidence["Evidence scope"])
            self.assertIn("Unmeasured planning envelope", evidence)
            paired = dict(
                safe_model_evidence("SmolVLM-256M-Instruct-Q8_0.gguf", self.downloads, self.native)
            )
            self.assertIn("not RAM/VRAM", paired["Image bundle bytes"])

    def test_missing_evidence_keeps_custom_selection_and_unknown_values_honest(self):
        rows = safe_model_evidence("custom.gguf", self.downloads, self.native)
        self.assertIn("No publisher defaults", dict(rows)["Evidence"])
        with patch(
            "ruinedfooocus.features.chat.model_browser.load_registry",
            side_effect=RegistryError("fixture unavailable"),
        ):
            rows = safe_model_evidence("LFM3-1.7B-Q4_K_M.gguf", self.downloads, self.native)
        self.assertEqual(rows, [["Evidence unavailable", "fixture unavailable"]])

    def test_all_live_chat_metadata_renders_without_converting_unknowns_to_scores(self):
        for artifact in load_registry().entries(role="language_model"):
            if not artifact["chat_weight"]:
                continue
            with self.subTest(artifact=artifact["id"]):
                rows = safe_model_evidence(
                    Path(artifact["catalogue_key"]).name, self.downloads, self.native
                )
                self.assertNotEqual(rows[0][0], "Evidence unavailable")
                self.assertTrue(all(isinstance(cell, str) for row in rows for cell in row))

    def test_search_filters_rows_without_clearing_valid_selection_or_changing_active_runtime(self):
        component_names = (
            "llama_model",
            "local_picker",
            "llama_selection",
            "llama_selected_quant",
            "llama_quant",
            "llama_model_status",
        )
        namespace = dict(
            gr=SimpleNamespace(update=lambda **values: values),
            model_catalogue=lambda _: self.catalogue,
            model_label=lambda name: name,
            path_manager=SimpleNamespace(DOWNLOADABLE_FILES=self.downloads),
            settings=SimpleNamespace(default_settings={}),
            configured_runtime=lambda _: self.native,
            **{name: name for name in component_names},
        )
        clear = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "clear_model_selection",
            dict(namespace, action_updates=lambda *args: {}),
        )
        namespace["clear_model_selection"] = clear
        view = function_from("ruinedfooocus/features/chat/ui.py", "search_models", namespace)
        for source in ("Downloadable", "Local models"):
            from ruinedfooocus.features.chat.model_preparation import ChatModelSession

            session = ChatModelSession()
            session.active = "loaded-model.gguf"
            session.select("gpt-oss-20b-MXFP4.gguf")
            session.select = Mock(wraps=session.select)
            result = view(
                source,
                "GPT OSS",
                "gpt-oss-20b-MXFP4.gguf",
                "no results",
                session,
            )
            self.assertEqual(set(result), {"llama_model", "local_picker"})
            self.assertIsNone(result["llama_model"]["value"])
            self.assertEqual(result["local_picker"]["value"]["selected"], "GPT OSS")
            self.assertEqual(result["local_picker"]["value"]["selected_label"], "GPT OSS")
            self.assertEqual(result["llama_model"]["choices"], [])
            self.assertEqual(session.selected, "gpt-oss-20b-MXFP4.gguf")
            self.assertEqual(session.active, "loaded-model.gguf")
            session.select.assert_not_called()
            removed = view(source, "GPT OSS", "missing-quant.gguf", "", session)
            self.assertIsNone(removed["llama_selection"])
            self.assertIsNone(removed["llama_selected_quant"])
            self.assertIsNone(removed["local_picker"]["value"]["selected"])
            session.select.assert_called_once_with(None)

    def test_empty_dropdown_event_cannot_replace_canonical_selection(self):
        from unittest.mock import Mock

        choices = Mock(side_effect=AssertionError("Dismissal must not pick another quant"))
        choose = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "select_model",
            {"selection_choices": choices, "gr": gr},
        )
        self.assertEqual(choose(None, "Downloadable"), gr.skip())
        self.assertEqual(choose("", "Local models"), gr.skip())
        choices.assert_not_called()

    def test_dismissal_of_same_family_preserves_quant_but_new_family_selects_deliberately(self):
        from unittest.mock import Mock

        select = Mock(return_value={"selection": "fixture"})
        choose = function_from(
            "ruinedfooocus/features/chat/ui.py", "choose_model", {"select_model": select, "gr": gr}
        )
        session = object()
        self.assertEqual(
            choose(None, "Downloadable", "saved", "missing-Q5.gguf", session), gr.skip()
        )
        select.assert_not_called()
        choose("saved", "Downloadable", "saved", "missing-Q5.gguf", session)
        select.assert_called_once_with(
            "saved", "Downloadable", "missing-Q5.gguf", session, preserve=True
        )
        select.reset_mock()
        choose("other", "Downloadable", "saved", "missing-Q5.gguf", session)
        select.assert_called_once_with("other", "Downloadable", None, session, preserve=False)

    def test_empty_model_and_quant_events_serialize_the_actual_ten_output_contract(self):
        from gradio.state_holder import SessionState

        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/chat/ui.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "model_outputs"
        )
        with gr.Blocks(analytics_enabled=False) as app:
            scope = {
                "gr": gr,
                "selection_choices": Mock(side_effect=AssertionError("No model selected")),
            }
            for name in (item.id for item in assignment.value.elts):
                scope[name] = (
                    gr.State("preserved")
                    if name in ("llama_selection", "llama_selected_quant")
                    else gr.Textbox()
                )
            exec(
                compile(ast.Module(body=[assignment], type_ignores=[]), str(source), "exec"), scope
            )
            select = function_from("ruinedfooocus/features/chat/ui.py", "select_model", scope)
            choose = function_from("ruinedfooocus/features/chat/ui.py", "choose_model", scope)
            button = gr.Button()
            button.click(
                lambda: choose(None, "Downloadable", "saved", "saved.gguf", None),
                outputs=scope["model_outputs"],
            )
            button.click(
                lambda: select(None, "Local models", "saved.gguf", None, preserve=True),
                outputs=scope["model_outputs"],
            )
        state = SessionState(app)
        for index in (0, 1):
            result = asyncio.run(app.process_api(index, [], state=state))
            self.assertEqual(len(result["data"]), len(scope["model_outputs"]))
            self.assertTrue(all(value is None or value == gr.skip() for value in result["data"]))
            self.assertEqual(state[scope["llama_selection"]._id], "preserved")
            self.assertEqual(state[scope["llama_selected_quant"]._id], "preserved")


if __name__ == "__main__":
    unittest.main()
