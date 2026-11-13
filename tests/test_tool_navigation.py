"""The real Create hierarchy and its Hub/Library navigation share one tab group."""

import ast
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.features.gallery.ui_workspace import EditorWorkspace
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]
APP = "ruinedfooocus/ui/app.py"


def source_tree(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def production_tab(path, **identity):
    call = next(
        node
        for node in ast.walk(source_tree(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Tab"
        and all(
            any(
                keyword.arg == key
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == value
                for keyword in node.keywords
            )
            for key, value in identity.items()
        )
    )
    return eval(
        compile(ast.Expression(call), path, "eval"),
        dict(gr=gr, t=lambda text: text, translate_ui=lambda text: text),
    )


def create_fixture():
    source = (ROOT / APP).read_text(encoding="utf-8")
    column = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.With)
        and any("generation-controls" in ast.unparse(item.context_expr) for item in node.items)
    )
    panel_names = {
        "create_output_format_ui",
        "create_native_sampling_ui",
        "create_auxiliary_ui",
        "create_layered_ui",
        "create_tensor_io_ui",
    }

    def layout_nodes(body):
        """Keep production layout contexts; replace unrelated panel internals only."""
        result = []
        for node in body:
            if isinstance(node, ast.With) and any(
                ast.unparse(item.context_expr).startswith(("gr.Tabs(", "gr.Tab(", "gr.Accordion("))
                or ast.unparse(item.context_expr)
                in ("create_controls_tabs", "generation_tab", "model_tab", "details_tab")
                for item in node.items
            ):
                kept = copy.deepcopy(node)
                if any(
                    isinstance(item.optional_vars, ast.Name)
                    and item.optional_vars.id == "generation_tab"
                    for item in node.items
                ):
                    kept.body = ast.parse(
                        """
base_model = gr.Dropdown(choices=["Janku"], value="Janku")
with gr.Accordion("Sampling", elem_id="generation-sampling"):
    gr.Number(value=20)
with gr.Accordion("Size", elem_id="generation-size"):
    gr.Number(value=1024)
with gr.Accordion("Styles", elem_id="generation-style"):
    gr.Checkbox()
image_number = gr.Slider(minimum=1, maximum=50, value=1)
with gr.Accordion("Negative prompt and seed", elem_id="generation-seed"):
    gr.Number(value=37)
with gr.Accordion("Saved settings (PNG)", elem_id="generation-presets"):
    gr.Textbox()
save_defaults_btn = gr.Button("Save current defaults", elem_id="save-generation-defaults")
"""
                    ).body
                else:
                    kept.body = (
                        layout_nodes(node.body)
                        or ast.parse('gr.Markdown("Existing panel content")').body
                    )
                result.append(kept)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id in ("model_tab", "bind_prompt_generation", "inpaint_toggle")
                for target in node.targets
            ):
                result.append(copy.deepcopy(node))
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) in panel_names | {"regroup_generation_options"}
            ):
                result.append(copy.deepcopy(node))
        return result

    layout = ast.fix_missing_locations(ast.Module(body=layout_nodes(column.body), type_ignores=[]))
    scope = {node.id: None for node in ast.walk(layout) if isinstance(node, ast.Name)}
    scope.update(gr=gr, t=lambda text: text, translate_ui=lambda text: text)
    function_from(APP, "regroup_generation_options", scope)

    def prompt_panel(*args, **kwargs):
        with production_tab(
            "ruinedfooocus/features/prompting/ui_random.py", elem_id="prompt-tools"
        ):
            scope["prompt"] = gr.Textbox(value="Keep this draft", interactive=True)
        return lambda *args: None

    def image_panel(*args, **kwargs):
        with production_tab("ruinedfooocus/features/image_editing/ui.py", id="image-tools"):
            scope["operation"] = gr.Dropdown(choices=["None", "Segment"], value="None")
        return gr.Checkbox(value=False, render=False)

    for name in panel_names:
        scope[name] = lambda *args, name=name: gr.Accordion(name)
    scope["ui_onebutton"] = SimpleNamespace(ui_onebutton=prompt_panel)
    scope["ui_controlnet"] = SimpleNamespace(add_controlnet_tab=image_panel)
    with gr.Blocks(analytics_enabled=False) as create:
        with gr.Column():
            exec(compile(layout, APP, "exec"), scope)
    namespace = dict(
        gr=gr,
        shared=SimpleNamespace(gradio_root=create, models=object()),
        model_tab=scope["model_tab"],
    )
    assignments = [
        node
        for node in ast.walk(source_tree(APP))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in ("create_tabs", "image_tools_tabs")
            for target in node.targets
        )
    ]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), APP, "exec"), namespace)
    return create, namespace, scope["base_model"], scope["operation"], scope["prompt"]


class ToolNavigationTests(unittest.TestCase):
    def test_actual_create_layout_groups_native_panels_under_more_options(self):
        create, scope, _, _, _ = create_fixture()
        tabs = scope["create_tabs"]
        self.assertIs(tabs, scope["image_tools_tabs"])
        self.assertEqual(
            [child.label for child in tabs.children],
            ["Generation", "Models", "Prompt tools", "Image tools", "Details"],
        )
        self.assertTrue(all(isinstance(child, gr.Tab) for child in tabs.children))
        layout = create.config["layout"]
        parents = {}

        def visit(node):
            for child in node.get("children", []):
                parents[child["id"]] = node["id"]
                visit(child)

        visit(layout)
        generation = tabs.children[0]
        panels = [
            component
            for component in create.blocks.values()
            if isinstance(component, gr.Accordion) and component.label.startswith("create_")
        ]
        self.assertEqual(len(panels), 5)
        more = next(
            component
            for component in create.blocks.values()
            if component.elem_id == "generation-more"
        )
        self.assertFalse(more.open)
        self.assertEqual(parents[more._id], generation._id)
        self.assertTrue(all(parents[panel._id] == more._id for panel in panels))
        sections = {
            component.elem_id: component
            for component in create.blocks.values()
            if isinstance(component, gr.Accordion) and component.elem_id
        }
        for identity in (
            "generation-sampling",
            "generation-size",
            "generation-style",
            "generation-seed",
        ):
            self.assertEqual(parents[sections[identity]._id], generation._id)
        self.assertEqual(parents[sections["generation-presets"]._id], more._id)
        save = next(
            component
            for component in create.blocks.values()
            if component.elem_id == "save-generation-defaults"
        )
        self.assertIs(generation.children[-1], save)
        self.assertTrue(all(parents[tab._id] == tabs._id for tab in tabs.children))
        changelog = next(
            component
            for component in create.blocks.values()
            if isinstance(component, gr.Accordion) and component.label == "Changelog"
        )
        self.assertEqual(parents[changelog._id], tabs.children[-1]._id)

    def test_create_and_settings_models_have_distinct_frontend_translation_identities(self):
        create = production_tab(APP, id="create-models")
        settings = production_tab("ruinedfooocus/ui/ui_settings.py", id="models")
        self.assertEqual((create.label, settings.label), ("Models", "Models / downloads"))
        self.assertNotEqual(create.id, settings.id)

    def test_production_hub_event_selects_the_actual_image_tools_parent(self):
        create, scope, base, operation, prompt = create_fixture()
        self.assertIs(scope["create_tabs"], scope["image_tools_tabs"])
        image_tab = next(
            child for child in scope["image_tools_tabs"].children if child.id == "image-tools"
        )
        with create:
            workspace = EditorWorkspace()
            run_button = gr.Button("Generate")
        with gr.Blocks(analytics_enabled=False) as settings:
            with gr.Tabs() as settings.models_tabs:
                with gr.Tab("Models / downloads", id="models") as settings.models_tab:
                    selection = gr.Textbox()
                    records = gr.JSON()
                    role = gr.Textbox()
                    overrides = gr.JSON()
                    component_scope = gr.Textbox()
                    activate = gr.Button()
        settings.acquisition = SimpleNamespace(
            selection=selection,
            records=records,
            component_role=role,
            activate=activate,
            inspected_local=lambda row, models: row,
        )
        with gr.Blocks(analytics_enabled=False) as chat:
            gr.Textbox()
        with gr.Blocks(analytics_enabled=False) as gallery:
            gr.Textbox()
        shell = function_from(
            APP,
            "create_navigation_shell",
            dict(gr=gr, t=lambda value: value, SimpleNamespace=SimpleNamespace),
        )(
            create,
            chat,
            gallery,
            settings,
            workspace,
            model_tab=scope["model_tab"],
            create_tabs=scope["create_tabs"],
            image_tools_tabs=scope["image_tools_tabs"],
            run_button=run_button,
            title="Tool activation fixture",
        )
        app, top = shell.app, shell.tabs
        with app:
            scope.update(
                app_settings=settings,
                app_llama_chat=SimpleNamespace(catalogue_selection_inputs=[]),
                shell=shell,
                editor_workspace=workspace,
                base_model=base,
                top_tabs=top,
                component_overrides=overrides,
                component_scope=component_scope,
                chat_catalogue_outputs=[],
                tool_catalogue_outputs=[operation],
                path_manager=object(),
                ui_controlnet=SimpleNamespace(
                    activate_tool=Mock(return_value={operation: gr.update(value="Segment")})
                ),
            )
            function_from(APP, "activate_catalogue_checkpoint", scope)
            assignment = next(
                node
                for node in ast.walk(source_tree(APP))
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "activation_outputs"
                    for target in node.targets
                )
            )
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), APP, "exec"), scope)
            function_from(APP, "activate_destination", scope)
            binding = next(
                node
                for node in ast.walk(source_tree(APP))
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "app_settings.acquisition.activate.click"
            )
            exec(compile(ast.Module(body=[binding], type_ignores=[]), APP, "exec"), scope)
        event = next(
            event
            for event in app.config["dependencies"]
            if event["targets"] == [(activate._id, "click")]
        )
        self.assertIn(scope["image_tools_tabs"]._id, event["outputs"])
        self.assertEqual(len(event["outputs"]), len(set(event["outputs"])))
        self.assertNotIn(prompt._id, event["outputs"])
        row = dict(key="local:sam", task="Image", category="checkpoints")
        state = SessionState(app)
        with patch(
            "ruinedfooocus.features.image_editing.tool_activation.installed_tool",
            return_value=dict(kind="segment"),
        ):
            result = asyncio.run(
                app.process_api(
                    event["id"], [row["key"], [row], "Janku", None, {}, None], state=state
                )
            )
        projected = dict(zip(event["outputs"], result["data"]))
        self.assertEqual(projected[scope["image_tools_tabs"]._id]["selected"], image_tab.id)
        self.assertEqual(projected[scope["create_tabs"]._id]["selected"], "image-tools")
        self.assertEqual(state[scope["image_tools_tabs"]._id].selected, image_tab.id)
        self.assertEqual(state[top._id].selected, "editor")
        self.assertEqual(state[workspace.destination._id], "editor")
        self.assertTrue(state[workspace.panel._id].visible)
        self.assertEqual(state[workspace.action._id].value, "generate")
        self.assertEqual(state.blocks_config.blocks[base._id].value, "Janku")
        self.assertEqual(projected[operation._id]["value"], "Segment")

    def test_library_navigation_uses_the_tools_parent_and_keeps_generation_routes(self):
        create, scope, _, _, prompt = create_fixture()
        with gr.Blocks(analytics_enabled=False) as app:
            with gr.Tabs() as top:
                with gr.Tab("Create", id="create"):
                    create.render()
                with gr.Tab("Library", id="library"):
                    button = gr.Button()
            scope["top_tabs"] = top
            scope["navigation"] = None
            scope["editor_workspace"] = None
            navigate = function_from("ruinedfooocus/features/gallery/ui.py", "navigate", scope)
            button.click(
                lambda: navigate({}, "image-tools"),
                outputs=list(dict.fromkeys([top, scope["create_tabs"], scope["image_tools_tabs"]])),
            )
        state = SessionState(app)
        event = app.config["dependencies"][0]
        result = asyncio.run(app.process_api(event["id"], [], state=state))
        self.assertEqual(result["data"][-1]["selected"], "image-tools")
        self.assertEqual(state[scope["image_tools_tabs"]._id].selected, "image-tools")
        self.assertEqual(
            navigate({}),
            {
                top: gr.update(selected="create"),
                scope["create_tabs"]: gr.update(selected="generation"),
            },
        )
        self.assertNotIn(prompt, navigate({}))
        tree = source_tree("ruinedfooocus/features/gallery/ui.py")
        image_binding = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(
                keyword.arg == "outputs" and "image_names" in ast.unparse(keyword.value)
                for keyword in node.keywords
            )
        )
        outputs = next(
            keyword.value for keyword in image_binding.keywords if keyword.arg == "outputs"
        )
        self.assertIn("image_tools_tabs", ast.unparse(outputs))


if __name__ == "__main__":
    unittest.main()
