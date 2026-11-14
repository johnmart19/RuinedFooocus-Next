"""Native six-destination layout preserves the existing image and model owners."""

import asyncio
import ast
import json
import subprocess
from collections import Counter
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.gallery.ui_workspace import EditorWorkspace
from ruinedfooocus.core.translation_manager import translate_ui
import test_gallery_setup_outputs as gallery_fixtures
import test_gallery_actions as gallery_actions
import test_editor_workspace_events as editor_fixtures
import test_tool_navigation as navigation_fixtures
from ui_test_helpers import function_from


def layout_ids(node):
    return [child["id"] for child in node.get("children", ())] + [
        identity for child in node.get("children", ()) for identity in layout_ids(child)
    ]


class NavigationShellTests(unittest.TestCase):
    @staticmethod
    def navigation_presentation(shell, event, result, state):
        publish = next(
            row for row in shell.app.config["dependencies"] if row["trigger_after"] == event["id"]
        )
        envelope = result["data"][0]
        envelope = envelope.model_dump() if hasattr(envelope, "model_dump") else envelope
        values = asyncio.run(
            shell.app.postprocess_data(shell.app.fns[publish["id"]], envelope["updates"], state)
        )
        return dict(
            zip(
                publish["outputs"],
                [value.model_dump() if hasattr(value, "model_dump") else value for value in values],
            )
        )

    @staticmethod
    def navigation_script(script, arguments, memory):
        harness = """const vm=require('node:vm'), d=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
        const window=d.memory.window||{}, document={getElementById(id){return {getAttribute(name){return name==='aria-selected'&&id===`rf-nav-${d.memory.selected}-button`?'true':'false'}}}};
        const result=vm.runInNewContext('('+d.script+')',{window,document})(...d.arguments);
        process.stdout.write(JSON.stringify({result,window}));"""
        run = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(dict(script=script, arguments=arguments, memory=memory)),
            text=True,
            capture_output=True,
            timeout=10,
        )
        if run.returncode:
            raise AssertionError(run.stderr)
        output = json.loads(run.stdout)
        memory["window"] = output["window"]
        return output["result"]

    def shell(self, gallery=None, translate=lambda value: value):
        create, scope, base, _, create_draft = navigation_fixtures.create_fixture()
        with create:
            workspace = EditorWorkspace()
            run_button = gr.Button("Generate")
            with workspace.panel:
                editor_draft = gr.Textbox("Editor draft")
            for label in ("Generate", "Import image", "Save image"):
                gr.Button(label).click(
                    Mock(side_effect=AssertionError(f"Navigation invoked {label}")),
                    inputs=[],
                    outputs=[],
                )
        with gr.Blocks(analytics_enabled=False) as chat:
            chat_text = gr.Textbox("Chat draft")
            gr.Button("Send").click(lambda value: value, chat_text, chat_text)
        if gallery is None:
            with gr.Blocks(analytics_enabled=False) as gallery:
                gallery_draft = gr.Textbox("Gallery query")
        else:
            gallery_draft = None
        with gr.Blocks(analytics_enabled=False) as settings:
            with gr.Tabs() as settings.models_tabs:
                with gr.Tab("Models / downloads", id="models") as settings.models_tab:
                    model_query = gr.Textbox("Model query")
                    gr.Button("Search").click(lambda value: value, model_query, model_query)
                with gr.Tab("Preferences"):
                    gr.Checkbox(label="Saved preference")
                    settings_draft = gr.Textbox("Settings draft")
        scope["navigation_drafts"] = dict(
            create=create_draft,
            editor=editor_draft,
            library=gallery_draft,
            models=model_query,
            chat=chat_text,
            settings=settings_draft,
        )
        scope["run_button"] = run_button
        source_apps = (create, chat, gallery, settings)
        original_layout = [
            identity for app in source_apps for identity in layout_ids(app.config["layout"])
        ]
        original_registry = {identity for app in source_apps for identity in app.blocks}
        original_events = [event for app in source_apps for event in app.fns.values()]
        create_shell = function_from(
            "ruinedfooocus/ui/app.py",
            "create_navigation_shell",
            dict(gr=gr, t=translate, SimpleNamespace=SimpleNamespace),
        )
        shell = create_shell(
            create,
            chat,
            gallery,
            settings,
            workspace,
            model_tab=scope["model_tab"],
            create_tabs=scope["create_tabs"],
            image_tools_tabs=scope["image_tools_tabs"],
            run_button=run_button,
            title="Navigation fixture",
        )
        return (
            shell,
            workspace,
            base,
            scope,
            settings,
            original_layout,
            original_registry,
            original_events,
        )

    def test_every_heading_line_has_one_native_translation_marker(self):
        shell, *_ = self.shell(translate=translate_ui)
        headings = [
            component
            for component in shell.app.blocks.values()
            if "rf-page-head" in (component.elem_classes or [])
        ]
        # Chat owns its heading inside the conversation; the shell supplies
        # the five other destinations without duplicating that heading.
        self.assertEqual(len(headings), 10)
        for component in headings:
            parent = component.parent
            while parent is not None:
                self.assertIsNot(parent, shell.chat)
                parent = getattr(parent, "parent", None)
        for component in headings:
            with self.subTest(value=component.value):
                self.assertEqual(component.value.count("__i18n__"), 1)
                self.assertTrue(
                    component.value.startswith("<h1>") or component.value.startswith("<p>")
                )

    def test_chat_gets_the_live_create_prompt_separately_from_image_defaults(self):
        prompt, workspace, result = object(), object(), object()
        image_controls = {"negative": object()}
        create = Mock(return_value=result)
        get_ctrl = Mock(return_value=prompt)
        compose = function_from(
            "ruinedfooocus/ui/app.py",
            "create_chat_for_workspace",
            {
                "ui_llama_chat": SimpleNamespace(create_chat=create),
                "shared": SimpleNamespace(get_ctrl=get_ctrl),
            },
        )
        self.assertIs(compose(image_controls, workspace), result)
        get_ctrl.assert_called_once_with("prompt")
        create.assert_called_once_with(
            image_controls, editor_workspace=workspace, prompt_control=prompt
        )
        self.assertEqual(set(image_controls), {"negative"})

    def test_native_move_keeps_every_original_id_and_event_once(self):
        shell, workspace, base, scope, settings, layout, registry, events = self.shell()
        self.assertEqual(
            [(tab.label, tab.id) for tab in shell.tabs.children],
            [
                ("Create", "create"),
                ("Editor", "editor"),
                ("Gallery", "library"),
                ("Models", "models"),
                ("Chat", "chat"),
                ("Settings", "settings"),
            ],
        )
        counts = Counter(layout_ids(shell.app.config["layout"]))
        self.assertTrue(all(counts[identity] == 1 for identity in layout))
        self.assertTrue(registry.issubset(shell.app.blocks))
        self.assertIs(shell.app.blocks[workspace.context._id], workspace.context)
        self.assertIs(shell.app.blocks[base._id], base)
        self.assertIs(scope["model_tab"].parent, shell.model_destinations)
        self.assertIs(settings.models_tab.parent, shell.model_destinations)
        self.assertNotIn(settings.models_tab, settings.models_tabs.children)
        for event in events:
            self.assertEqual(sum(event is current for current in shell.app.fns.values()), 1)
        known = set(shell.app.blocks) | {shell.app._id}
        for event in shell.app.config["dependencies"]:
            with self.subTest(event=event["id"]):
                self.assertTrue(set(event["inputs"]).issubset(known))
                self.assertTrue(set(event["outputs"]).issubset(known))
                for target, name in event["targets"]:
                    if target is None:
                        self.assertEqual(name, "success")
                        self.assertIn(event["trigger_after"], shell.app.fns)
                    else:
                        self.assertIn(target, known)

    def test_two_native_round_trips_keep_all_six_drafts_without_workflow_or_file_callbacks(self):
        shell, workspace, base, scope, *_, workflow_events = self.shell()
        state = SessionState(shell.app)
        context = {"session_id": "owned-image"}
        state[workspace.context._id] = context
        drafts = scope["navigation_drafts"]
        for route, component in drafts.items():
            state[component._id].value = f"Edited {route} draft: keep this text"
        expected_drafts = {route: state[component._id].value for route, component in drafts.items()}
        routes = (
            shell.create,
            shell.editor,
            shell.gallery,
            shell.models,
            shell.chat,
            shell.settings,
        )
        image_tools = next(
            tab for tab in scope["image_tools_tabs"].children if tab.id == "image-tools"
        )
        blocked = []
        for event in workflow_events:
            callback = Mock(side_effect=AssertionError("Navigation invoked an owning workflow"))
            guard = patch.object(event, "fn", callback)
            guard.start()
            self.addCleanup(guard.stop)
            blocked.append(callback)
        workspace.selected_store = Mock(
            side_effect=AssertionError("Navigation accessed image files")
        )

        with patch.object(shell.app, "call_function", wraps=shell.app.call_function) as calls:
            for round_number in range(2):
                for destination in routes:
                    with self.subTest(round=round_number, route=destination.id):
                        event = next(
                            event
                            for event in shell.app.config["dependencies"]
                            if event["targets"] == [(destination._id, "select")]
                        )
                        self.assertFalse(event["queue"])
                        self.assertEqual(event["inputs"][0], workspace.action._id)
                        # The native select event already changed the browser's
                        # tab. Check its transport, not Gradio's rebuilt Tabs cache.
                        result = asyncio.run(
                            shell.app.process_api(
                                event["id"],
                                ["generate", round_number * 6 + routes.index(destination) + 1],
                                state=state,
                            )
                        )
                        updates = self.navigation_presentation(shell, event, result, state)
                        self.assertEqual(updates[shell.tabs._id], gr.skip())
                        self.assertEqual(updates[workspace.destination._id], destination.id)
                        image = destination.id in ("create", "editor")
                        editing = destination.id == "editor"
                        classes = state[shell.image_stage._id].elem_classes or []
                        self.assertEqual("rf-off" not in classes, image)
                        self.assertEqual("rf-editor" in classes, editing)
                        self.assertEqual(state[workspace.panel._id].visible, editing)
                        self.assertEqual(state[image_tools._id].visible, editing)
                        if image:
                            self.assertEqual(state[scope["create_tabs"]._id].selected, "generation")
                        else:
                            self.assertEqual(updates[scope["create_tabs"]._id], gr.skip())
                        self.assertTrue(
                            {component._id for component in drafts.values()}.isdisjoint(updates)
                        )
                        self.assertEqual(
                            {
                                route: state[component._id].value
                                for route, component in drafts.items()
                            },
                            expected_drafts,
                        )
                        self.assertEqual(state[workspace.context._id], context)
                        self.assertEqual(state[base._id].value, base.value)
            self.assertEqual(calls.await_count, 12)
        self.assertEqual(len(blocked), 5)
        for callback in blocked:
            callback.assert_not_called()
        workspace.selected_store.assert_not_called()

    def test_native_navigation_tracks_no_output_progress_and_keeps_toolbar_destinations(self):
        shell, workspace, base, *_ = self.shell()
        routes = {tab._id for tab in shell.tabs.children}
        navigation = [
            event
            for event in shell.app.config["dependencies"]
            if any(target[0] in routes and target[1] == "select" for target in event["targets"])
        ]
        self.assertEqual(len(navigation), 6)
        self.assertTrue(all(event["show_progress"] == "hidden" for event in navigation))
        self.assertTrue(all(event["show_progress_on"] == [] for event in navigation))
        self.assertTrue(
            all(
                next(
                    row
                    for row in shell.app.config["dependencies"]
                    if row["trigger_after"] == event["id"]
                )["outputs"]
                == [control._id for control in shell.outputs]
                for event in navigation
            )
        )

        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        state = SessionState(shell.app)
        context = {"session_id": "retained-source"}
        state[workspace.context._id] = context
        for name, destination in (
            ("model_catalogue_button", "downloads"),
            ("create_models_button", "create-models"),
        ):
            call = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "click"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == name
            )
            with shell.app:
                button = gr.Button(name)
                dependency = eval(
                    compile(ast.Expression(call), str(source), "eval"),
                    {name: button, "shell": shell, "gr": gr},
                )
            self.assertEqual(dependency["show_progress"], "hidden")
            self.assertEqual(dependency["show_progress_on"], [])
            result = asyncio.run(shell.app.process_api(dependency["id"], [], state=state))
            self.assertEqual(len(result["data"]), len(shell.outputs) + 1)
            self.assertEqual(state[shell.tabs._id].selected, "models")
            self.assertEqual(state[shell.model_destinations._id].selected, destination)
            self.assertEqual(state[workspace.context._id], context)
            self.assertEqual(state[base._id].value, base.value)

    def test_late_gallery_response_cannot_hide_editor_or_replace_run_destination(self):
        shell, workspace, *_ = self.shell(translate=translate_ui)
        state = SessionState(shell.app)
        memory = {"selected": "create", "window": {}}
        pending = {}
        routes = {tab.id: tab for tab in shell.tabs.children}
        for destination in ("models", "chat", "settings", "library", "editor"):
            memory["selected"] = destination
            event = next(
                row
                for row in shell.app.config["dependencies"]
                if row["targets"] == [(routes[destination]._id, "select")]
            )
            args = self.navigation_script(event["js"], ["prompt-edit", 0], memory)
            response = asyncio.run(shell.app.process_api(event["id"], args, state=state))
            pending[destination] = (event, response)

        def deliver(destination):
            event, response = pending[destination]
            child = next(
                row
                for row in shell.app.config["dependencies"]
                if row["trigger_after"] == event["id"]
            )
            value = response["data"][0]
            value = value.model_dump() if hasattr(value, "model_dump") else value
            guarded = self.navigation_script(child["js"], [value], memory)
            outputs = asyncio.run(
                shell.app.postprocess_data(shell.app.fns[child["id"]], guarded, state)
            )
            return dict(
                zip(
                    child["outputs"],
                    [row.model_dump() if hasattr(row, "model_dump") else row for row in outputs],
                )
            )

        current = deliver("editor")
        self.assertEqual(current[workspace.destination._id], "editor")
        self.assertEqual(current[shell.outputs[-1]._id]["value"], "Edit image")
        self.assertEqual(state[shell.image_stage._id].elem_classes, ["rf-editor"])
        delayed = deliver("library")
        self.assertEqual(delayed[workspace.destination._id], gr.skip())
        self.assertEqual(state[shell.image_stage._id].elem_classes, ["rf-editor"])
        # A current native selection is authoritative even before its server
        # presentation response has replaced the previous hidden JSON value.
        for destination in ("create", "editor"):
            memory["selected"] = destination
            captured = self.navigation_script(workspace.run_version_js(0), ["library", 0], memory)
            self.assertEqual(captured[0], destination)

    def test_create_projects_new_image_intent_without_clearing_editor_widgets(self):
        with gr.Blocks(analytics_enabled=False):
            workspace = EditorWorkspace()
        context, pixels = {"session_id": "owned"}, object()
        values = dict(
            _editor_destination="create",
            _editor_action="remove-background",
            _editor_context=context,
            input_image=pixels,
            inpaint_toggle=True,
            cn_selection="Background removal",
            cn_type="rembg",
            prompt="Keep my draft",
            base_model_name="chosen.safetensors",
            automatic_upscale_target="Input image only",
            native_latent_input="deliberate-latent.safetensors",
        )
        workspace.selected_store = Mock(
            side_effect=AssertionError("Navigation resolved image resources")
        )
        projected = workspace.build_request(values)
        self.assertIsNone(projected["input_image"])
        self.assertIsNone(projected["_editor_context"])
        self.assertEqual(projected["cn_selection"], "None")
        self.assertEqual(projected["automatic_upscale_target"], "Generated image")
        self.assertEqual(projected["prompt"], values["prompt"])
        self.assertEqual(projected["base_model_name"], values["base_model_name"])
        self.assertEqual(projected["native_latent_input"], values["native_latent_input"])
        self.assertIs(values["input_image"], pixels)
        self.assertIs(values["_editor_context"], context)
        editor = workspace.build_request(
            dict(values, _editor_destination="editor", _editor_action="generate")
        )
        self.assertIs(editor["editor_context"], context)
        workspace.selected_store.assert_not_called()

    def test_gallery_prompt_and_continue_native_events_return_to_create(self):
        gallery, _, controls, callbacks, *_ = gallery_fixtures.setup_fixture()
        shell, workspace, *_ = self.shell(gallery=gallery)
        callbacks["t"] = lambda value: value
        with (
            shell.app,
            patch.object(workspace, "bind_library"),
            patch.object(workspace, "bind_picker"),
        ):
            callbacks["browser"] = object()
            callbacks["open_editor_btn"] = gr.Button("Open in Editor")
            callbacks["bind_actions"](
                shell.tabs,
                next(
                    component
                    for component in shell.app.blocks.values()
                    if isinstance(component, gr.Tabs)
                    and any(child.id == "generation" for child in component.children)
                ),
                lambda *_: {},
                [],
                image_tools_tabs=next(
                    component
                    for component in shell.app.blocks.values()
                    if isinstance(component, gr.Tabs)
                    and any(child.id == "image-tools" for child in component.children)
                ),
                navigation=shell.navigate,
                navigation_outputs=shell.outputs,
                gallery_tab=shell.gallery,
                editor_workspace=workspace,
            )
        state = SessionState(shell.app)
        state[workspace.destination._id].value = "editor"
        state[workspace.context._id] = {"session_id": "prior-source"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "saved.png"
            Image.new("RGB", (8, 12), "blue").save(source)
            for name, values in (
                (
                    "prompt_btn",
                    [source_record(source), {"prompt": "Saved draft"}, "Rendered prompt"],
                ),
                ("continue_btn", []),
            ):
                event = next(
                    fn
                    for fn in shell.app.fns.values()
                    if fn.targets == [(callbacks[name]._id, "click")] and shell.tabs in fn.outputs
                )
                result = asyncio.run(shell.app.process_api(event, values, state=state))
                updates = dict(zip((control._id for control in event.outputs), result["data"]))
                self.assertEqual(state[shell.tabs._id].selected, "create")
                self.assertEqual(updates[workspace.destination._id], "create")
                self.assertFalse(state[workspace.panel._id].visible)
                self.assertIsNone(state[workspace.context._id])
                if name == "prompt_btn":
                    self.assertEqual(updates[controls["prompt"]._id]["value"], "Saved draft")

            model = Path(directory) / "fixture.safetensors"
            model.write_bytes(b"Inert local identity, never loaded")
            models = SimpleNamespace(
                get_names=lambda kind: [model.name] if kind == "checkpoints" else [],
                get_file=lambda kind, name: model if name == model.name else None,
                get_models_by_path=lambda *args, **kwargs: {"_rf_local_sha256": "a" * 64},
            )
            modules = {
                "ruinedfooocus.app": SimpleNamespace(
                    state=SimpleNamespace(
                        models=models, settings=SimpleNamespace(default_settings={})
                    )
                ),
                "ruinedfooocus.core.util": SimpleNamespace(
                    get_lora_thumbnail=lambda *_: None, load_keywords=lambda *_: ""
                ),
            }
            event = next(
                fn
                for fn in shell.app.fns.values()
                if fn.targets == [(callbacks["setup_btn"]._id, "click")]
                and shell.tabs in fn.outputs
            )
            for controlnet, destination in (
                (None, "create"),
                ({"type": "none"}, "create"),
                ({"type": "img2img"}, "editor"),
            ):
                metadata = dict(
                    prompt="Saved draft",
                    base_model_name=model.name,
                    base_model_hash="a" * 64,
                    width=512,
                    height=768,
                    controlnet=controlnet,
                )
                state[workspace.context._id] = {"session_id": "prior-source"}
                with (
                    patch.dict("sys.modules", modules),
                    patch(
                        "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                        return_value="SDXL",
                    ),
                    patch(
                        "ruinedfooocus.features.models.component_profiles.replay_choices",
                        return_value=({}, None),
                    ),
                ):
                    result = asyncio.run(
                        shell.app.process_api(
                            event,
                            [source_record(source), metadata, "1024×1024", None, None, None],
                            state=state,
                        )
                    )
                self.assertEqual(state[shell.tabs._id].selected, destination)
                self.assertEqual(
                    result["data"][
                        [control._id for control in event.outputs].index(workspace.destination._id)
                    ],
                    destination,
                )
                self.assertIsNone(state[workspace.context._id])
                self.assertEqual(state[workspace.action._id].value, "prompt-edit")

    def test_programmatic_inpaint_handoff_keeps_intent_through_native_action_change(self):
        editor = editor_fixtures.EditorWorkspaceEventTests()
        editor.setUp()
        self.addCleanup(editor.doCleanups)
        workspace = editor.workspace
        scope = gallery_actions.functions()
        controls = dict(editor.controls)
        controls.update(
            main_view=editor.remote["preview"],
            inpaint_view=editor.remote["inpaint"],
            inpaint_toggle=editor.remote["inpaint_toggle"],
        )
        with editor.app:
            for name in scope["REFERENCE_RESETS"]:
                if name not in controls:
                    controls[name] = gr.State(None)
            if "automatic_upscale_target" not in controls:
                controls["automatic_upscale_target"] = gr.State("Generated image")
            scope["get_ctrl"] = controls.__getitem__
            scope.update(
                navigation=lambda destination, inner: {
                    editor.destinations: gr.update(selected=destination),
                    workspace.destination: destination,
                },
                editor_workspace=workspace,
                validate_view_record=editor.legacy.validate,
                t=lambda value: value,
                kind="inpaint",
            )
            function_from("ruinedfooocus/features/gallery/ui.py", "navigate", scope)
            prepare = function_from("ruinedfooocus/features/gallery/ui.py", "prepare", scope)
            button = gr.Button("Inpaint in Editor")
            inputs = [gr.State(None) for _ in range(8)]
            button.click(
                prepare,
                inputs=inputs,
                outputs=list(dict.fromkeys([*controls.values(), *workspace.outputs])),
            )
        session = editor.session("inpaint-navigation")
        editor.open_library(session)
        self.assertEqual(editor.value(session, workspace.action), "prompt-edit")
        policy = SimpleNamespace(
            checkpoint_features=lambda *_: {"inpaint": True},
            edit_tool=lambda *_: "Img2Img",
            tool_kind=lambda *_: "img2img",
            controlnet=SimpleNamespace(cn_options={}),
        )
        with patch.dict("sys.modules", {"ruinedfooocus.features.image_editing.ui": policy}):
            editor.call(
                session,
                editor.event(button),
                dict(
                    zip(
                        inputs,
                        [editor.legacy.record, "model", "Disabled", "Off", "None", "", 0.25, None],
                    )
                ),
            )
        self.assertTrue(editor.value(session, editor.remote["inpaint_toggle"]))
        self.assertEqual(editor.value(session, workspace.action), "generate")
        self.assertEqual(session.state[editor.destinations._id].selected, "editor")
        self.assertIsNone(editor.value(session, workspace.context))
        editor.call(session, editor.event(workspace.action, "change"))
        self.assertTrue(editor.value(session, editor.remote["inpaint_toggle"]))
        self.assertIsNotNone(editor.value(session, editor.remote["inpaint"]))
        user_action = next(
            fn
            for fn in editor.app.fns.values()
            if (workspace.action._id, "input") in fn.targets
            and editor.remote["inpaint_toggle"] in fn.outputs
        )
        for action, intended in (
            ("generate", False),
            ("inpaint", True),
            ("remove-background", False),
        ):
            editor.call(session, user_action, {workspace.action: action})
            self.assertEqual(editor.value(session, editor.remote["inpaint_toggle"]), intended)


if __name__ == "__main__":
    unittest.main()
