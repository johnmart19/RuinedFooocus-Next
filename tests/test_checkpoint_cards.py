"""Checkpoint cards follow canonical picker state without resetting adapter selection."""

import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.features.models.selection import (
    checkpoint_records,
    model_label,
    selected_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


class CheckpointCardTests(unittest.TestCase):
    def test_initial_picker_changes_filters_and_refreshes_use_the_canonical_identity(self):
        source = ROOT / "ruinedfooocus/features/models/ui_browser.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_ui"
        )
        with tempfile.TemporaryDirectory() as directory:
            preview = Path(directory) / "preview.png"
            Image.new("RGB", (1, 1)).save(preview)
            names = ["first.safetensors", "second.safetensors"]
            models = SimpleNamespace(
                get_names=lambda category: names,
                get_models_by_path=lambda *args, **kwargs: {},
                get_file=lambda *_: None,
                cache_paths={"checkpoints": Path(directory) / "cache/checkpoints"},
            )
            localized = []
            scope = dict(
                __file__=str(source),
                Path=Path,
                gr=gr,
                translate_ui=lambda text: text,
                localize=lambda text, locale: text,
                resolve_locale=lambda **kwargs: "en",
                localized_output=lambda fn, inputs, outputs: localized.append(
                    (fn, inputs, outputs)
                ),
                checkpoint_records=checkpoint_records,
                selected_checkpoint=selected_checkpoint,
                model_label=model_label,
                get_checkpoint_thumbnail=lambda name: str(preview),
                create_components_ui=lambda model, model_refresh=None: None,
                create_checkpoint_ui=lambda model, model_refresh=None: None,
            )
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
            with gr.Blocks() as demo:
                picker = gr.Dropdown(choices=names, value=names[0])
                metadata_revision = gr.Number(0, visible=False)
                query, gallery, records, refresh = scope["create_ui"](
                    picker, models, model_refresh=metadata_revision
                )
            cards = next(
                block
                for block in demo.blocks.values()
                if getattr(block, "elem_id", None) == "checkpoint-cards"
            )
            self.assertEqual(cards.value["selected"], records.value[0]["caption"])
            self.assertEqual(cards.value["defaultPreviewLabel"], "Default preview")
            self.assertTrue(all(not card["preview_is_default"] for card in cards.value["cards"]))
            callback = next(
                fn
                for fn in demo.fns.values()
                if getattr(fn.fn, "__name__", "") == "visible_cards_for"
            )
            event = next(
                event for event in demo.config["dependencies"] if event["id"] == callback._id
            )
            self.assertEqual(event["inputs"][:2], [records._id, picker._id])
            self.assertEqual(len(event["inputs"]), 3)
            self.assertIsInstance(demo.blocks[event["inputs"][-1]], gr.Number)
            self.assertIn((picker._id, "change"), [tuple(target) for target in event["targets"]])
            self.assertEqual(
                callback.fn(records.value, names[1])["selected"], records.value[1]["caption"]
            )
            # Filtering hides the highlight without selecting a different checkpoint.
            filtered = refresh("first")[1]
            self.assertEqual(callback.fn(filtered, names[1])["selected"], "")
            self.assertEqual(callback.fn(records.value, "removed.safetensors")["selected"], "")
            self.assertEqual(localized[0][1], [records, picker])
            self.assertEqual(
                localized[0][0]("en", records.value, names[1])["selected"],
                records.value[1]["caption"],
            )

            query_callback = next(
                fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "refresh"
            )
            query_event = next(
                event for event in demo.config["dependencies"] if event["id"] == query_callback._id
            )
            self.assertFalse(query_event["queue"])
            self.assertEqual(query_event["outputs"], [gallery._id, records._id])
            self.assertEqual(query_event["inputs"][0], query._id)
            query_targets = [tuple(target) for target in query_event["targets"]]
            # Gradio.change also covers paste, replacement and clearing;
            # Gradio.input in the installed Textbox depends on keypresses.
            self.assertIn((query._id, "change"), query_targets)
            self.assertIn((query._id, "submit"), query_targets)
            self.assertNotIn((query._id, "input"), query_targets)
            self.assertIn(
                (metadata_revision._id, "change"),
                query_targets,
            )
            projection_event = next(
                event
                for event in demo.config["dependencies"]
                if event["trigger_after"] == query_callback._id
            )
            self.assertEqual(projection_event["outputs"], [cards._id])
            self.assertEqual(projection_event["inputs"][:2], [records._id, picker._id])
            self.assertIn('"checkpoint-cards"', projection_event["js"])
            self.assertFalse(projection_event["queue"])

            state = SessionState(demo)
            request = gr.Request(username="owner", session_hash=f"checkpoint-cards-{id(demo)}")

            def query_response(search, version):
                # The installed Gradio unqueued client drops changed_state_ids.
                # Exercise the explicit chain, without dispatching State.change.
                return asyncio.run(
                    demo.process_api(
                        query_callback._id,
                        [search, version],
                        state=state,
                        request=request,
                    )
                )

            def card_response(current_model, version):
                return asyncio.run(
                    demo.process_api(
                        projection_event["id"],
                        [None, current_model, version],
                        state=state,
                        request=request,
                    )
                )["data"][0]

            response = query_response("second", 2)
            self.assertEqual(response["changed_state_ids"], [records._id])
            payload = card_response(names[0], 1)
            self.assertEqual(
                [card["key"] for card in payload["cards"]], [records.value[1]["caption"]]
            )
            self.assertEqual(payload["selected"], "")
            self.assertEqual(
                payload["requestIdentity"], {"channel": "checkpoint-query", "version": 2}
            )
            self.assertEqual(state[records._id][0]["_view_version"], 2)
            stale = query_response("first", 1)
            self.assertEqual(stale["data"][0], gr.skip())
            self.assertEqual(state[records._id][0]["_view_version"], 2)
            query_response("no matching file", 3)
            self.assertEqual(card_response(names[0], 2)["cards"], [])
            query_response("", 4)
            cleared = card_response(names[0], 3)
            self.assertEqual(len(cleared["cards"]), 2)
            self.assertEqual(cleared["selected"], records.value[0]["caption"])

            # A delayed old card projection cannot return old query rows after
            # newer cards complete. Artwork is already part of the query rows,
            # so pause text assembly rather than repeat preview work here.
            query_response("", 5)
            entered, release = threading.Event(), threading.Event()
            original_localize = scope["localize"]

            def blocked_localize(text, locale):
                if text == "Image models" and not entered.is_set():
                    entered.set()
                    self.assertTrue(release.wait(5))
                return original_localize(text, locale)

            scope["localize"] = blocked_localize
            try:
                with ThreadPoolExecutor(1) as threads:
                    old = threads.submit(card_response, names[0], 4)
                    self.assertTrue(entered.wait(5))
                    try:
                        query_response("second", 6)
                        newest = card_response(names[1], 5)
                    finally:
                        release.set()
                    self.assertEqual(old.result(5)["__type__"], "update")
                    self.assertEqual(newest["selected"], records.value[1]["caption"])
                    self.assertEqual(state[records._id][0]["_view_version"], 6)
            finally:
                release.set()
                scope["localize"] = original_localize

            # A model change can arrive before a query's State is published.
            # Its rejected old-row projection must recover through the chain,
            # which reads the latest model and advances the card counter.
            entered, release = threading.Event(), threading.Event()
            original_metadata = models.get_models_by_path

            def blocked_metadata(*args, **kwargs):
                if not entered.is_set():
                    entered.set()
                    self.assertTrue(release.wait(5))
                return original_metadata(*args, **kwargs)

            models.get_models_by_path = blocked_metadata
            try:
                with ThreadPoolExecutor(1) as threads:
                    pending = threads.submit(query_response, "", 7)
                    self.assertTrue(entered.wait(5))
                    try:
                        self.assertEqual(card_response(names[0], 6)["__type__"], "update")
                    finally:
                        release.set()
                    pending.result(5)
                recovered = card_response(names[0], 7)
                self.assertEqual(recovered["selected"], records.value[0]["caption"])
                self.assertEqual(recovered["pageIdentity"]["version"], 7)
                self.assertEqual(recovered["requestIdentity"]["version"], 7)
            finally:
                release.set()
                models.get_models_by_path = original_metadata

    def test_client_applies_server_selection_and_keeps_local_only_adapter_behavior(self):
        script = (ROOT / "ruinedfooocus/ui/assets/adapter_cards.js").read_text(encoding="utf-8")
        harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
function node(tag='div') {
  return {tagName:tag,dataset:{},attrs:{},children:[],classList:{toggle(){},add(){},remove(){}},
    setAttribute(k,v){this.attrs[k]=v},getAttribute(k){return this.attrs[k]??(k==='src'?this.src??null:null)},
    append(...items){for(const item of items){item.remove();item.parent=this;this.children.push(item)}},
    replaceChildren(...items){for(const child of this.children)child.parent=null;this.children=[];this.append(...items)},
    contains(item){return this.children.includes(item)},
    querySelectorAll(selector){return selector==='button[data-adapter]'?this.children.filter(c=>c.dataset.adapter):[]},
    querySelector(selector){return this.children.find(c=>selector==='img'?c.tagName==='img':c.className?.split(' ').includes(selector.slice(1)))||null},
    insertBefore(item,next){item.remove();item.parent=this;const index=next?this.children.indexOf(next):this.children.length;this.children.splice(index,0,item)},
    remove(){if(this.parent){this.parent.children=this.parent.children.filter(c=>c!==this);this.parent=null}},
    focus(){this.focused=true}};
}
const heading=node(), grid=node(), handlers={}, events=[];
const element={querySelector:s=>s==='.adapter-card-heading'?heading:s==='.adapter-card-grid'?grid:null,
  querySelectorAll:()=>grid.children.filter(c=>c.dataset.adapter),
  addEventListener:(name,fn)=>handlers[name]=fn, contains:()=>false};
const props={value:{heading:'Models',acceptsDrop:false,empty:'Empty',selected:'a',cards:[
  {key:'a',label:'A',kind:'Image model'}, {key:'b',label:'B',kind:'Image model'}]}};
let update;
vm.runInNewContext(source,{props,element,window:{},document:{activeElement:null,createElement:node},
  trigger:(...args)=>events.push(args),watch:(name,fn)=>update=fn});
function pressed(){return grid.children.filter(c=>c.attrs['aria-pressed']==='true').map(c=>c.dataset.adapter)}
assert.deepEqual(pressed(),['a']);
props.value={...props.value,selected:'b'};update();assert.deepEqual(pressed(),['b']);
props.value={...props.value,selected:''};update();assert.deepEqual(pressed(),[]);
props.value={...props.value,selected:'removed'};update();assert.deepEqual(pressed(),[]);
// Other adapter surfaces omit server selection; local clicks survive rerenders.
delete props.value.selected;
const b=grid.children[1]; handlers.click({target:{closest:selector=>selector==='button[data-adapter]'?b:null},detail:1});
assert.deepEqual(pressed(),['b']);update();assert.deepEqual(pressed(),['b']);
assert.equal(events[0][1].key,'b');
// Incremental artwork updates retain the same cards, image node and scroll.
const ordinary=props.value;grid.scrollTop=42;grid.scrollLeft=4;
props.value={previewPatch:[{key:'b',artworkKey:'',preview:{url:'preview.png'}}]};
update();const image=b.querySelector('.adapter-card-media').querySelector('img');
update();assert.equal(b.querySelector('.adapter-card-media').querySelector('img'),image);
assert.equal(grid.children[1],b);assert.deepEqual(pressed(),['b']);assert.equal(grid.scrollTop,42);
props.value={...ordinary,cards:[ordinary.cards[1],ordinary.cards[0]]};update();
assert.equal(grid.children[0],b);assert.deepEqual(pressed(),['b']);assert.equal(grid.scrollLeft,4);
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(script),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
