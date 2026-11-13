"""Exact Settings navigation and session-local draft state, without app startup."""

import ast
import asyncio
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.ui.settings_navigation import (
    bind_navigation,
    create_profile_state,
    create_search_results,
    navigation_records,
    source_text,
)


DOM = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
class Node {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.dataset = {}; this.attrs = {}; this.listeners = {}; this.textContent = ''; this.classes=new Set();this.classList={contains:name=>this.classes.has(name)}; }
  append(...nodes) { nodes.forEach(node => node.parent = this); this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, value) { this.attrs[name] = value; }
  closest() { return this.tagName === 'button' && this.dataset.result !== undefined ? this : this.parent?.closest(); }
  contains(node) { return node === this || this.children.some(child => child.contains(node)); }
  click() { for (let node = this; node; node = node.parent) node.listeners.click?.({target:this}); }
  focus() { this.focused = true; }
  scrollIntoView() { this.scrolled = true; }
  getClientRects() { return this.visible ? [1] : []; }
  matches() { return false; }
}
function context(props, children) {
  const host = new Node();
  host.querySelector = selector => children[selector];
  host.append(...new Set(Object.values(children)));
  const callbacks = [];
  const events=[];
  const document = { createElement: tag => new Node(tag), getElementById: id => children[id] };
  const scope = { element: host, props, document, trigger:(name,data)=>events.push({name,data}), watch: (names, fn) => callbacks.push({names:[names].flat(),fn}), requestAnimationFrame: fn => setImmediate(fn) };
  vm.runInNewContext(source.renderer, scope);
  function update(changes) {
    const changed=Object.keys(changes);
    if (changes.props) {
      // Native HTML replaces custom props as one dictionary, not a patch.
      for (const name of Object.keys(props)) if (name !== 'value') delete props[name];
      Object.assign(props,changes.props);
      changed.push(...Object.keys(changes.props));
    }
    if (Object.hasOwn(changes,'value')) props.value=changes.value;
    for (const {names,fn} of callbacks) if (names.some(name => changed.includes(name))) fn();
  }
  function input(...values) {
    const outputs = source.event
      ? vm.runInNewContext(`(${source.event})`, {})(...values)
      : source.events[JSON.stringify(values)];
    assert.ok(Array.isArray(outputs),'JS-only events return the declared output list');
    assert.equal(outputs.length,1);
    const result=outputs[0];
    if (result?.__type__ === 'update') { const {__type__,...changes} = result; update(changes); }
    else update({value:result});
    return result;
  }
  return { host, update, input, scope, events };
}
"""


def run_script(payload, test):
    def check_json(value, path="payload"):
        if isinstance(value, dict):
            for key, item in value.items():
                check_json(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check_json(item, f"{path}[{index}]")
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise AssertionError(f"Non-JSON update at {path}: {type(value).__name__}")

    check_json(payload)

    result = subprocess.run(
        ["node", "-e", DOM + test],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=15,
    )
    if result.returncode:
        raise AssertionError(result.stderr)


def registered_transport(kind):
    """Use the actual Gradio dependency and HTML config, without app/model startup."""
    with patch("ruinedfooocus.core.profile_store.selected_profile", return_value="next"):
        with gr.Blocks() as root:
            query = gr.Textbox()
            results = create_search_results()
            with gr.Tabs():
                with gr.Tab("Profiles"):
                    with gr.Accordion("Startup"):
                        text = gr.Textbox(label="Default listen port", value="original")
                        number = gr.Number(value=None)
            profile, _ = create_profile_state("target", [text, number])
            bind_navigation(root, query, results, {})
    component = results if kind == "search" else profile
    dependency = next(
        event for event in root.config["dependencies"] if event["outputs"] == [component._id]
    )
    expected_inputs = [query._id] if kind == "search" else [text._id, number._id]
    if dependency["inputs"] != expected_inputs or dependency["backend_fn"] != (kind == "search"):
        raise AssertionError(dependency)
    config = component.get_config()
    result = {
        "renderer": config["js_on_load"],
        "event": dependency["js"],
        "props": {"value": config["value"], **config["props"]},
    }
    if kind == "search":
        state = SessionState(root)
        result["events"] = {
            json.dumps([query], separators=(",", ":")): asyncio.run(
                root.process_api(dependency["id"], [query], state=state)
            )["data"]
            for query in ("listen port", "unknown", "", None)
        }
        navigator = next(
            index
            for index, function in root.fns.items()
            if function.fn is not None and function.fn.__name__ == "navigate_setting"
        )
        record = next(row for row in config["props"]["records"] if row["id"] == text.elem_id)
        result["navigation"] = {}
        for token in (2, 3):
            packets = asyncio.run(
                root.process_api(
                    navigator,
                    [],
                    state=state,
                    event_data=gr.EventData(
                        None, {"id": record["id"], "token": token, "locale": "en"}
                    ),
                )
            )["data"]
            result["navigation"][str(token)] = packets[-1]
    return result


class SettingsNavigationTests(unittest.TestCase):
    def test_relocated_models_keep_their_events_outside_settings_search_routes(self):
        with gr.Blocks() as app:
            query = gr.Textbox()
            results = create_search_results()
            with gr.Tabs() as tabs:
                with gr.Tab("Models / downloads") as relocated:
                    with gr.Accordion("Model diagnostics", open=False) as model_details:
                        status = gr.Textbox(value="ready")
                        action = gr.Button("Install test runtime")
                        original_event = action.click(lambda: "unchanged", outputs=status)
                with gr.Tab("Appearance & language") as preferences:
                    with gr.Accordion("Theme and updates", open=False) as appearance:
                        theme = gr.Textbox(label="Theme", value="saved-theme")
            bind_navigation(app, query, results, {}, excluded=(relocated,))
        records = results.get_config()["props"]["records"]
        self.assertEqual([record["label"] for record in records], ["Theme"])
        navigator = next(
            index
            for index, function in app.fns.items()
            if function.fn is not None and function.fn.__name__ == "navigate_setting"
        )
        self.assertNotIn(model_details, app.fns[navigator].outputs)
        self.assertIn(appearance, app.fns[navigator].outputs)
        session = SessionState(app)
        asyncio.run(
            app.process_api(
                navigator,
                [],
                state=session,
                event_data=gr.EventData(None, {"id": theme.elem_id, "token": 1}),
            )
        )
        self.assertTrue(session[appearance._id].open)
        self.assertEqual(session[tabs._id].selected, preferences.id)
        self.assertEqual(
            asyncio.run(app.process_api(original_event["id"], [], state=session))["data"],
            ["unchanged"],
        )
        self.assertEqual(theme.value, "saved-theme")

    def test_native_navigation_properties_survive_first_mount_repeat_and_isolate_sessions(self):
        with gr.Blocks() as app:
            query = gr.Textbox()
            results = create_search_results()
            with gr.Tabs() as tabs:
                with gr.Tab("Profiles"):
                    gr.Textbox(label="Profile name", value="kept")
                with gr.Tab("Image runtime") as target_tab:
                    with gr.Accordion("Advanced native compute", open=False) as outer:
                        with gr.Accordion("Precision", open=False) as inner:
                            field = gr.Dropdown(
                                label="VAE precision", choices=["auto", "fp16"], value="auto"
                            )
                            action = gr.Button("Install sample")
            bind_navigation(app, query, results, {})
        navigator = next(
            index
            for index, function in app.fns.items()
            if function.fn is not None and function.fn.__name__ == "navigate_setting"
        )
        outputs = app.fns[navigator].outputs
        self.assertEqual(set(outputs), {tabs, outer, inner, results})
        self.assertNotIn(field, outputs)
        self.assertNotIn(action, outputs)
        self.assertEqual(target_tab.get_config()["id"], target_tab.elem_id)
        self.assertNotIn(".click()", results.get_config()["js_on_load"])
        a, b = SessionState(app), SessionState(app)
        for token in (1, 2):
            result = asyncio.run(
                app.process_api(
                    navigator,
                    [],
                    state=a,
                    event_data=gr.EventData(
                        None, {"id": field.elem_id, "token": token, "locale": "uk"}
                    ),
                )
            )
            self.assertEqual(a[tabs._id].selected, target_tab.id)
            self.assertTrue(a[outer._id].open)
            self.assertTrue(a[inner._id].open)
            packet = result["data"][-1]
            self.assertNotIn("value", packet)
            self.assertEqual(packet["props"]["focus_target"], {"id": field.elem_id, "token": token})
            self.assertEqual(packet["props"]["locale"], "uk")
            json.dumps(result["data"])
        self.assertFalse(b[outer._id].open)
        self.assertFalse(b[inner._id].open)
        self.assertEqual(field.value, "auto")

    def test_native_navigation_refuses_unregistered_keys_and_invalid_tokens(self):
        with gr.Blocks() as app:
            query = gr.Textbox()
            results = create_search_results()
            with gr.Tabs():
                with gr.Tab("Image runtime"):
                    field = gr.Textbox(label="VAE precision", value="unchanged")
            bind_navigation(app, query, results, {})
        navigator = next(
            index
            for index, function in app.fns.items()
            if function.fn is not None and function.fn.__name__ == "navigate_setting"
        )
        for identity, token in (
            ("unknown", 1),
            ({"path": "private"}, 1),
            (field.elem_id, True),
            (field.elem_id, 0),
        ):
            with self.subTest(identity=identity, token=token), self.assertRaises(gr.Error):
                asyncio.run(
                    app.process_api(
                        navigator,
                        [],
                        state=SessionState(app),
                        event_data=gr.EventData(None, {"id": identity, "token": token}),
                    )
                )
        self.assertEqual(field.value, "unchanged")

    def test_nullable_custom_gguf_textbox_keeps_actual_native_initial_value_and_user_edits(self):
        with gr.Blocks() as app:
            custom = gr.Textbox(label="Custom GGUF path", value=None)
            profile, _ = create_profile_state("target", [custom])
        self.assertIsNone(custom.get_config()["value"])
        config = profile.get_config()
        self.assertEqual(config["value"], [None])
        self.assertEqual(config["props"]["saved"], [None])
        dependency = next(
            event for event in app.config["dependencies"] if event["outputs"] == [profile._id]
        )
        run_script(
            {
                "renderer": config["js_on_load"],
                "event": dependency["js"],
                "props": {"value": config["value"], **config["props"]},
            },
            r"""
const status=new Node(),terms=new Node(),timing=new Node();
const props=structuredClone(source.props);
const browser=context(props,{'[role="status"]':status,dl:terms,'.apply-timing':timing});
assert.equal(status.textContent,props.clean_label);
browser.input(null);assert.equal(status.textContent,props.clean_label);
browser.input('typed-local-model.gguf');assert.equal(status.textContent,props.dirty_label);
browser.update({props:{...props,saved:[null]}});
assert.equal(props.value[0],'typed-local-model.gguf');assert.equal(status.textContent,props.dirty_label);
browser.input('');assert.equal(status.textContent,props.dirty_label);
browser.update({props:{...props,saved:['']}});assert.equal(status.textContent,props.clean_label);
""",
        )

    def test_real_search_locale_packet_keeps_cached_records_and_query_outside_props(self):
        with patch("ruinedfooocus.ui.settings_navigation.localized_output") as localized:
            with gr.Blocks() as app:
                query, locale = gr.Textbox(), gr.Textbox(value="en")
                results = create_search_results()
                with gr.Tabs():
                    with gr.Tab("Image runtime"):
                        gr.Dropdown(label="VAE precision", choices=["auto"], value="auto")
                bind_navigation(app, query, results, {})
                render, render_inputs, render_outputs = localized.call_args.args
                event = gr.Button().click(render, [locale, *render_inputs], render_outputs)
        packet = asyncio.run(app.process_api(event["id"], ["en"], state=SessionState(app)))["data"][
            0
        ]
        self.assertNotIn("value", packet)
        self.assertEqual(set(packet["props"]) - {"__type__"}, set(results.props))
        self.assertIn("VAE precision", [record["label"] for record in packet["props"]["records"]])
        # The production JSON response must contain data, not HTML event callables.
        self.assertIn("VAE precision", json.dumps(packet))

    def test_real_gradio_save_receipts_update_properties_without_returning_draft_value(self):
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/ui_settings.py").read_text(
                encoding="utf-8"
            )
        )
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in {"save_form", "save_as_form", "select_next_launch"}
        ]
        from ruinedfooocus.ui.settings_navigation import profile_receipt

        scope = dict(
            gr=gr,
            profile_receipt=profile_receipt,
            save_clicked=lambda *values, **kwargs: ("saved", {"own_base": values[0]}),
            use_next_launch_clicked=lambda name: "next",
            state={"setting_name": ["ui_settings_name", "startup_port"]},
        )
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), "Settings callbacks", "exec"),
            scope,
        )
        with (
            patch("ruinedfooocus.core.profile_store.selected_profile", return_value="next"),
            patch("ruinedfooocus.ui.settings_navigation.localized_output") as localized,
        ):
            with gr.Blocks() as app:
                target, base = gr.Textbox(value="target"), gr.State({"own_base": "original"})
                text, number = gr.Textbox(value="original"), gr.Number(value=None)
                receipt, receipt_state = create_profile_state("target", [text, number])
                locale = gr.Textbox(value="en")
                render, render_inputs, render_outputs = localized.call_args.args
                localized_event = gr.Button().click(
                    render, [locale, *render_inputs], render_outputs
                )
                status = gr.Textbox()
                save = gr.Button().click(
                    scope["save_form"],
                    [target, base, text, number],
                    [status, target, receipt, base, receipt_state],
                )
                copy = gr.Button().click(
                    scope["save_as_form"],
                    [base, text, number],
                    [status, target, receipt, base, receipt_state],
                )
                next_launch = gr.Button().click(
                    scope["select_next_launch"],
                    [text, receipt_state],
                    [status, receipt, receipt_state],
                )
            a, b = SessionState(app), SessionState(app)
            initial = asyncio.run(app.process_api(localized_event["id"], ["en", None], state=a))[
                "data"
            ][0]
            self.assertEqual(initial["props"]["saved"], ["original", None])
            self.assertEqual(initial["props"]["profile"], "target")
            self.assertNotIn("value", initial)
            result = asyncio.run(
                app.process_api(save["id"], ["target", None, "earlier draft", 1234], state=a)
            )["data"]
            self.assertEqual(result[2]["props"]["saved"], ["earlier draft", 1234])
            self.assertIn("clean_label", result[2]["props"])
            self.assertNotIn("value", result[2])
            self.assertEqual(a[base._id], {"own_base": "earlier draft"})
            self.assertEqual(b[base._id], {"own_base": "original"})
            copied = asyncio.run(app.process_api(copy["id"], [None, "copy", None], state=b))[
                "data"
            ][2]
            self.assertEqual(copied["props"]["profile"], "copy")
            self.assertNotIn("value", copied)
            self.assertIn("running_label", copied["props"])
            self.assertEqual(a[receipt._id].props["profile"], "target")
            next_result = asyncio.run(app.process_api(next_launch["id"], ["copy", None], state=a))[
                "data"
            ][1]
            self.assertEqual(next_result["props"]["next"], "copy")
            self.assertNotIn("value", next_result)
            self.assertEqual(next_result["props"]["saved"], ["earlier draft", 1234])
            self.assertEqual(next_result["props"]["profile"], "target")
            localized_result = asyncio.run(
                app.process_api(localized_event["id"], ["uk", None], state=a)
            )["data"][0]
            self.assertEqual(localized_result["props"]["saved"], ["earlier draft", 1234])
            self.assertEqual(localized_result["props"]["next"], "copy")
            self.assertEqual(set(localized_result["props"]), set(initial["props"]))
            self.assertEqual(a[receipt._id].props["saved"], ["earlier draft", 1234])
            self.assertEqual(b[receipt._id].props["profile"], "copy")
            run_script(
                {
                    "renderer": receipt.get_config()["js_on_load"],
                    "event": app.config["dependencies"][0]["js"],
                    "props": {
                        "value": receipt.get_config()["value"],
                        **receipt.get_config()["props"],
                    },
                    "packets": [initial, result[2], localized_result, next_result],
                },
                r"""
const status=new Node(),terms=new Node(),timing=new Node();
const props=structuredClone(source.props);
const browser=context(props,{'[role="status"]':status,dl:terms,'.apply-timing':timing});
browser.update(source.packets[0]);
assert.equal(status.textContent,props.clean_label);
browser.input('typed while saving',1234);
for (const packet of source.packets.slice(1)) {
  browser.update(packet);
  assert.equal(props.value[0],'typed while saving');
  assert.equal(props.profile,'target');
  assert.equal(status.textContent,props.dirty_label);
  assert.ok(terms.children.every(node=>node.textContent!==undefined));
}
""",
            )

    def test_registry_follows_exact_field_ancestors_without_changing_drafts(self):
        with gr.Blocks() as root:
            with gr.Tabs():
                with gr.Tab(translate_ui("Profiles & storage")) as tab:
                    with gr.Accordion(translate_ui("Startup preferences")) as accordion:
                        port = gr.Number(
                            label=translate_ui("Default listen port"),
                            value=None,
                            placeholder="Automatic",
                            minimum=1,
                        )
                with gr.Tab("Maintenance"):
                    cache = gr.Dropdown(
                        label="Cache to clear",
                        choices=[("Model hash records", "hashes")],
                        value="hashes",
                    )
        before = (port.value, cache.value)
        records = navigation_records(root, {"Maintenance": "exact cache hash library sizes"})
        record = next(item for item in records if item["id"] == port.elem_id)
        self.assertEqual(record["route"], (("tab", tab.elem_id), ("accordion", accordion.elem_id)))
        self.assertEqual(record["label"], "Default listen port")
        self.assertEqual((port.value, cache.value), before)
        self.assertIsNone(port.get_config()["value"])
        self.assertEqual(port.get_config()["placeholder"], "Automatic")
        self.assertIn(
            "hash", next(item for item in records if item["id"] == cache.elem_id)["terms"]
        )
        self.assertNotIn("__i18n__", json.dumps(records))
        self.assertEqual(source_text(translate_ui("Save")), "Save")

    def test_search_delegates_layout_updates_and_only_focuses_after_native_acknowledgement(self):
        run_script(
            registered_transport("search"),
            r"""
(async()=>{
const status=new Node(),list=new Node(),field=new Node(),input=new Node('input');
input.value='unsaved draft';field.querySelector=()=>input;
const props=structuredClone(source.props);
const record=props.records.find(row=>row.label==='Default listen port');
const browser=context(props,{'[role="status"]':status,ul:list,[record.id]:field});
assert.equal(browser.host.rfSearch,undefined);
browser.input('listen port');assert.equal(list.children.length,1);
list.children[0].children[0].click();
assert.equal(browser.events[0].name,'navigate');
assert.equal(browser.events[0].data.id,record.id);assert.equal(browser.events[0].data.token,2);
assert.equal(input.focused,undefined); // A click cannot focus a still-hidden native field.
field.visible=true;
browser.update(source.navigation['2']);
let timeout;
const wait=()=>new Promise(resolve=>setImmediate(resolve));
for(let attempt=0;attempt<8&&!input.focused;attempt++)await wait();
assert.equal(input.focused,true);assert.equal(input.value,'unsaved draft');
assert.equal(props.value,'listen port');assert.equal(field.scrolled,true);
// Locale refresh preserves query; a repeated result uses a fresh native event.
browser.update({props:{...props,found:'localized results'}});
assert.equal(status.textContent,'localized results');
input.focused=false;list.children[0].children[0].click();
assert.equal(browser.events[1].data.token,3);
browser.update(source.navigation['3']);
for(let attempt=0;attempt<8&&!input.focused;attempt++)await wait();
assert.equal(input.focused,true);assert.equal(input.value,'unsaved draft');
browser.input('unknown');assert.equal(status.textContent,props.no_matches);
browser.input('');assert.equal(list.children.length,0);
browser.input(null);assert.equal(props.value,'');
})().catch(error=>{console.error(error);process.exitCode=1;});
""",
        )

    def test_saved_receipt_keeps_newer_draft_and_other_browser_isolated(self):
        run_script(
            registered_transport("profile"),
            r"""
function browser() {
  const status = new Node(), terms = new Node(), timing = new Node();
  const props = structuredClone(source.props);
  return {...context(props, {'[role="status"]':status,dl:terms,'.apply-timing':timing}),props,status};
}
const a = browser(), b = browser();
a.input('newer draft',null);
assert.equal(a.host.dataset.dirty, 'true');
assert.equal(b.host.dataset.dirty, 'false');
// Save began before the latest keystroke. Its successful receipt may update the
// baseline but must not mark the current form clean or write its newer value.
a.update({props:{...a.props,saved:['earlier draft',null]}});
assert.equal(a.status.textContent,a.props.dirty_label);
assert.equal(a.props.value[0],'newer draft');
a.update({props:{...a.props,profile:'copy',saved:['newer draft',null]}});
assert.equal(a.status.textContent,a.props.clean_label);
assert.equal(b.props.profile,'target');
// A locale refresh and next-launch selection do not discard a typed edit.
a.input('typed after save',1234);
a.update({props:{...a.props,next:'copy',dirty_label:'localized dirty',clean_label:'localized saved'}});
assert.equal(a.status.textContent,'localized dirty');
assert.equal(a.props.value[0],'typed after save');
assert.equal(b.status.textContent,b.props.clean_label);
assert.equal(b.props.next,'next');
assert.equal(b.host.rfDraftUpdate,undefined);
""",
        )


if __name__ == "__main__":
    unittest.main()
