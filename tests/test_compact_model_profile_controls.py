"""Compact native pager wiring and fenced DOM behavior without app/model startup."""

import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.app.resource_lifecycle import LatestUIRequests
from ruinedfooocus.features.models import ui_requests, ui_acquisition as native
import test_model_acquisition as fixtures

ROOT = Path(__file__).resolve().parents[1]


class CompactControlsTests(unittest.TestCase):
    setUp = fixtures.AcquisitionTests.setUp

    def ui(self):
        fence, pool = LatestUIRequests(), PreparationAdmission(max_workers=2, max_jobs=4)
        self.addCleanup(pool.close)
        self.enterContext(patch.object(ui_requests, "latest_ui_requests", fence))
        self.enterContext(patch.object(ui_requests, "preparation_admission", pool))
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = native.create_ui(
                    self.manager, NS(update_all_models=Mock()), offline=lambda: True
                )
        ui.catalogue.registry = self.registry
        self.addCleanup(ui.preview_cache.close)
        functions = {fn.fn.__name__: fn for fn in app.fns.values() if fn.fn is not None}
        template = ui.catalogue.record("checkpoints/a.safetensors")
        rows = [dict(template, key=f"owned-{i}", label=f"Owned model {i}") for i in range(75)]
        self.enterContext(
            patch.object(
                ui.catalogue,
                "records",
                side_effect=lambda **options: rows[:1] if options.get("query") else rows.copy(),
            )
        )
        return app, ui, functions, fence, rows

    def test_real_gradio_pagination_uses_one_versioned_event_and_server_only_state(self):
        app, ui, functions, _, rows = self.ui()
        self.assertEqual(len(ui.refresh_outputs), 6)
        cards, page = ui.refresh_outputs[1], ui.refresh_outputs[3]
        self.assertIsInstance(page, gr.State)
        callback = functions["render_requested_page"]
        self.assertEqual(callback.outputs, [cards, page])
        self.assertNotIn(ui.selection, callback.outputs)
        self.assertFalse(callback.queue)
        self.assertIn('"acquisition-page"', callback.js)
        self.assertFalse(
            any(target[0] == page._id for fn in app.fns.values() for target in fn.targets)
        )
        self.assertIn("paginate", cards.get_config()["js_on_load"])
        state = SessionState(app)
        state[ui.records._id] = rows
        index = next(index for index, fn in app.fns.items() if fn is callback)
        result = asyncio.run(
            app.process_api(
                index, [None, None, 1], state=state, event_data=gr.EventData(None, {"page": 2})
            )
        )
        self.assertEqual(result["data"][0]["pagination"]["page"], 2)
        self.assertEqual(state[page._id], 2)
        self.assertIsNone(state[ui.selection._id])
        for target in (0, -1, True, "2", None):
            with self.subTest(target=target), self.assertRaises(gr.Error):
                callback.fn(rows, None, gr.EventData(None, {"page": target}))

    def test_counts_exclude_retained_selection_and_bound_absolute_page_requests(self):
        _, ui, functions, _, rows = self.ui()
        for size, expected_pages in ((0, 1), (1, 1), (36, 1), (37, 2), (75, 3)):
            data, current = native.card_data(rows[:size], None, self.manager, 100)
            self.assertEqual(
                (current, data["pagination"]["pages"]), (expected_pages, expected_pages)
            )
        data, _ = native.card_data(
            [rows[0], dict(rows[-1], _details_only=True)], rows[-1]["key"], self.manager
        )
        self.assertEqual(data["pagination"]["pages"], 1)
        self.assertEqual(len(data["cards"]), 1)
        self.assertEqual(data["selected"], rows[-1]["key"])
        self.assertEqual(
            functions["render_requested_page"].fn(rows, None, gr.EventData(None, {"page": 100}))[1],
            3,
        )
        self.assertEqual(ui.refresh.get_config()["scale"], 0)

    def test_rapid_pages_new_query_and_locale_observer_cannot_publish_older_results(self):
        _, ui, functions, fence, rows = self.ui()
        request = NS(username="owner", session_hash="pager", request=None)
        refresh = functions["filter_rows"].fn
        current_rows = refresh("", "All", "discover", "All", "All", None, 1, request, 1)[0]
        chosen = current_rows[-1]["key"]
        functions["select_record"].fn(
            current_rows,
            "text",
            True,
            gr.EventData(None, {"key": chosen}),
            request=request,
            version=1,
        )
        move = functions["render_requested_page"].fn
        render = dict(
            zip(move.__code__.co_freevars, (cell.cell_contents for cell in move.__closure__))
        )["render_page"]
        original = native.card_data
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            if args[3] == 2:
                entered.set()
                self.assertTrue(release.wait(5))
            return original(*args, **kwargs)

        with (
            patch.object(native, "card_data", side_effect=delayed),
            ThreadPoolExecutor(1) as threads,
        ):
            old = threads.submit(
                move,
                current_rows,
                chosen,
                gr.EventData(None, {"page": 2, "viewVersion": 1}),
                request,
                1,
            )
            self.assertTrue(entered.wait(5))
            try:
                newest = move(
                    current_rows,
                    chosen,
                    gr.EventData(None, {"page": 3, "viewVersion": 1}),
                    request,
                    2,
                )
                self.assertEqual(newest[0]["pageIdentity"]["version"], 2)
                changed = refresh(
                    "one match", "All", "discover", "All", "All", chosen, 1, request, 2
                )
                self.assertEqual(changed[1]["pagination"]["pages"], 1)
                self.assertTrue(changed[0][-1]["_details_only"])
            finally:
                release.set()
            self.assertEqual(old.result(5), (gr.skip(), gr.skip()))
        current_rows = changed[0]
        self.assertEqual(
            move(
                current_rows, chosen, gr.EventData(None, {"page": 2, "viewVersion": 1}), request, 3
            ),
            (gr.skip(), gr.skip()),
        )
        move(current_rows, chosen, gr.EventData(None, {"page": 1, "viewVersion": 2}), request, 3)
        entered.clear()
        release.clear()

        def observer(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                self.assertTrue(release.wait(5))
            return original(*args, **kwargs)

        with (
            patch.object(native, "card_data", side_effect=observer),
            ThreadPoolExecutor(1) as threads,
        ):
            old = threads.submit(render, current_rows, chosen, 1, request, None, "en")
            self.assertTrue(entered.wait(5))
            try:
                newest = move(
                    current_rows,
                    chosen,
                    gr.EventData(None, {"page": 1, "viewVersion": 2}),
                    request,
                    4,
                )
                self.assertEqual(newest[0]["pageIdentity"]["version"], 4)
            finally:
                release.set()
            self.assertEqual(old.result(5), gr.skip())
        self.assertEqual(
            fence.peek(("ui", "owner", "local", "pager"), "acquisition-selection").source, chosen
        )

    def test_actual_card_dom_keeps_native_buttons_hidden_bounds_and_latest_page_identity(self):
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8');
class Node {
 constructor(){this.children=[];this.dataset={};this.attrs={};this.listeners={};this.classList={toggle(){}};this.scrollTop=0;this.scrollLeft=0;}
 setAttribute(k,v){this.attrs[k]=v;}
 querySelector(selector){return this.children.find(n=>selector==='.'+n.className);}
 querySelectorAll(){return [];}
 append(...nodes){for(const n of nodes){n.parent=this;this.children.push(n);}}
 remove(){this.parent.children=this.parent.children.filter(n=>n!==this);}
}
const host=new Node(),heading=new Node(),grid=new Node(),pager=new Node(),previous=new Node(),next=new Node(),status=new Node();
previous.dataset.pageOffset='-1';next.dataset.pageOffset='1';previous.type=next.type='button';
pager.querySelector=s=>s.includes('"-1"')?previous:s.includes('"1"')?next:status;
host.querySelector=s=>({'.adapter-card-heading':heading,'.adapter-card-grid':grid,'.adapter-card-pager':pager})[s];
host.querySelectorAll=()=>[];host.addEventListener=(name,fn)=>host.listeners[name]=fn;
const versions={'acquisition-view':1,'acquisition-page':0},events=[],callbacks=[];
const packet=(page,pages,view=1,version=0)=>({heading:'Models',cards:[],selected:'retained',requestIdentity:{channel:'acquisition-view',version:view},pageIdentity:{channel:'acquisition-page',version},pagination:{page,pages,previous:'Previous',next:'Next',label:`Page ${page} of ${pages}`}});
const props={value:packet(1,1)};
const context={window:{__rfModelUIRequests:versions},props,element:host,document:{activeElement:null,createElement:()=>new Node()},watch:(name,fn)=>callbacks.push(fn),trigger:(name,data)=>{events.push({name,data});versions['acquisition-page']++;}};
vm.runInNewContext(source,context);
const update=data=>{props.value=data;callbacks.forEach(fn=>fn());};
const click=button=>host.listeners.click({detail:0,target:{closest:selector=>selector==='button[data-page-offset]'?button:null}});
assert.equal(pager.hidden,true);assert.equal(previous.disabled,true);assert.equal(next.disabled,true);
update(packet(1,3));assert.equal(pager.hidden,false);assert.equal(previous.disabled,true);assert.equal(next.disabled,false);
click(next);click(next);assert.equal(events.length,1);assert.equal(events[0].data.page,2);assert.equal(events[0].data.viewVersion,1);
update(packet(2,3,1,1));click(next);assert.equal(events.length,2);assert.equal(events[1].data.page,3);
update(packet(1,3,1,0));assert.equal(status.textContent,'Page 2 of 3');
versions['acquisition-view']=2;update(packet(3,3,1,2));assert.equal(status.textContent,'Page 2 of 3');click(next);assert.equal(events.length,2);
update(packet(1,1,2,2));assert.equal(pager.hidden,true);assert.equal(previous.disabled,true);assert.equal(next.disabled,true);
assert.equal(previous.type,'button');assert.equal(next.type,'button');
assert.equal(events.every(event=>event.name==='paginate'),true);
"""
        result = subprocess.run(
            ["node", "-e", script, str(ROOT / "ruinedfooocus/ui/assets/adapter_cards.js")],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_settings_container_rules_size_content_without_reducing_text_or_hiding_help(self):
        css = (ROOT / "ruinedfooocus/ui/assets/ui.css").read_text(encoding="utf-8")

        def rule(selector):
            return re.search(re.escape(selector) + r"\s*\{([^{}]*)\}", css)[1]

        button = rule(":is(#app-tabs, #image-stage) #settings-actions > button")
        self.assertRegex(button, r"flex:\s*0\s+1\s+auto\s*!important")
        self.assertIn("align-self: flex-end", button)
        self.assertNotRegex(button, r"font-size|max-height|overflow:\s*hidden")
        self.assertIn(
            "display: none",
            rule('#settings-search-results:has(ul:empty):has([role="status"]:empty)'),
        )
        self.assertNotIn("display: none", rule("#settings-search,\n#settings-search-results"))
        template = (ROOT / "ruinedfooocus/ui/assets/adapter_cards.html").read_text()
        self.assertEqual(template.count('type="button" data-page-offset='), 2)
        self.assertIn('role="status" aria-live="polite"', template)

    def test_actual_profile_action_row_retains_name_help_buttons_and_saved_value(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        row = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and any(
                    keyword.arg == "elem_id"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "settings-actions"
                    for keyword in item.context_expr.keywords
                )
                for item in node.items
            )
        )
        scope = dict(
            gr=gr, t=lambda text: text, settings=NS(name="owned-profile"), add_setting=Mock()
        )
        with gr.Blocks(analytics_enabled=False):
            exec(
                compile(ast.Module(body=[row], type_ignores=[]), "actual profile actions", "exec"),
                scope,
            )
        self.assertEqual(scope["ui_settings_name"].get_config()["value"], "owned-profile")
        self.assertEqual(scope["ui_settings_name"].get_config()["placeholder"], "Optional")
        self.assertEqual(
            [
                scope[key].get_config()["value"]
                for key in ("save_btn", "save_as_btn", "next_launch_btn")
            ],
            ["Save", "Save as profile", "Use next launch"],
        )


if __name__ == "__main__":
    unittest.main()
