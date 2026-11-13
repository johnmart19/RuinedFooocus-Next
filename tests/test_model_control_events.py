"""Execute the production model event graph without importing or launching the app."""

import ast
import asyncio
from functools import wraps
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.features.models.performance import PerformanceSettings
from ruinedfooocus.features.models.resolution_dropdown import ResolutionDropdown
from ruinedfooocus.features.models.resolutions import ResolutionSettings

SOURCE = Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/app.py"


def event_owner(call):
    while isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
        if isinstance(call.func.value, ast.Name):
            return call.func.value.id
        call = call.func.value
    return None


class ModelControlGraph:
    def __init__(self, test, *, selection="Model Defaults", current="1024x1024", automatic=True):
        self.trace = []
        self.cached_size = None
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        with patch.object(
            ResolutionSettings, "RESOLUTIONS_FILE", Path(temporary.name) / "sizes.json"
        ):
            self.resolutions = ResolutionSettings()
        self.resolutions.model_selection = Mock(wraps=self.resolutions.model_selection)

        def refresh(*_):
            self.cached_size = (832, 1216)

        self.publisher = SimpleNamespace(
            refresh=Mock(side_effect=refresh),
            cached_resolution=Mock(
                side_effect=lambda _: (
                    (self.cached_size, "fresh cached recommendation")
                    if self.cached_size
                    else (None, None)
                )
            ),
            metadata=Mock(return_value={}),
            describe=Mock(return_value="Fresh model recommendation"),
        )
        performance = SimpleNamespace(
            MODEL_PERFORMANCE="Model Defaults",
            CUSTOM_PERFORMANCE="Own settings",
            model_recommendations=self.publisher,
            performance_options={"SDXL": {}},
            family_preset=Mock(return_value="SDXL"),
            control_options=Mock(
                return_value=dict(
                    cfg=7, sampler_name="euler", scheduler="simple", clip_skip=2, custom_steps=12
                )
            ),
        )
        for name in ("choices_for_model", "selection_for_model"):
            setattr(performance, name, MethodType(getattr(PerformanceSettings, name), performance))
        self.scope = dict(
            gr=gr,
            shared=SimpleNamespace(
                models=SimpleNamespace(
                    get_models_by_path=Mock(return_value={"baseModel": "SDXL"}),
                    get_model_base=lambda metadata: metadata["baseModel"],
                )
            ),
            performance_settings=performance,
            resolution_settings=self.resolutions,
            resolve_locale=lambda **_: "en",
            KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"]),
        )
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        callback_names = (
            "model_image_controls",
            "model_resolution",
            "performance_control_updates",
            "refresh_recommendation",
        )
        for name in callback_names:
            node = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), self.scope)
            function = self.scope[name]

            @wraps(function)
            def measured(*args, _function=function, _name=name, **kwargs):
                self.trace.append(_name)
                return _function(*args, **kwargs)

            self.scope[name] = measured

        def describe_resolution(*_):
            self.trace.append("resolution_description")
            return "Resolution description"

        def describe_workflow(*_):
            self.trace.append("workflow_description")
            return "Workflow description"

        with gr.Blocks(analytics_enabled=False) as self.app:
            components = dict(
                base_model=gr.Dropdown(["model.safetensors"], value="model.safetensors"),
                model_metadata_refresh=gr.Number(0),
                performance_selection=gr.Dropdown(
                    ["Model Defaults", "Own settings"], value=selection
                ),
                model_family=gr.State("SDXL"),
                aspect_ratios_selection=ResolutionDropdown(
                    ["1024x1024 (SDXL recommended)", current, "Custom..."], value=current
                ),
                resolution_auto=gr.Checkbox(automatic),
                size_mode=gr.State("target"),
                restore_recommended_size=gr.Button(),
                auto_recommendations=gr.Checkbox(True),
                cfg=gr.Number(9),
                sampler_name=gr.Dropdown(["euler"], value="euler"),
                scheduler=gr.Dropdown(["simple"], value="simple"),
                clip_skip=gr.Number(3),
                custom_steps=gr.Number(22),
                recommendation_status=gr.Markdown(),
                resolution_recommendation=gr.Markdown(),
                workflow_help=gr.Markdown(),
            )
            self.scope.update(components)
            self.scope.update(
                recommendation_inputs=[
                    components[name]
                    for name in (
                        "performance_selection",
                        "model_family",
                        "base_model",
                        "auto_recommendations",
                    )
                ],
                performance_outputs=[
                    components[name]
                    for name in ("cfg", "sampler_name", "scheduler", "clip_skip", "custom_steps")
                ],
                resolution_description_inputs=[
                    components[name]
                    for name in (
                        "model_family",
                        "base_model",
                        "aspect_ratios_selection",
                        "resolution_auto",
                    )
                ],
                workflow_inputs=[
                    components[name]
                    for name in (
                        "model_family",
                        "base_model",
                        "performance_selection",
                        "custom_steps",
                        "cfg",
                        "sampler_name",
                        "scheduler",
                    )
                ],
                resolution_recommendation_text=describe_resolution,
                describe_workflow=describe_workflow,
            )
            registration = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "model_controls_event"
                    for target in node.targets
                )
            )
            self.execute(registration)
            for node in sorted(ast.walk(tree), key=lambda node: getattr(node, "lineno", 0)):
                if isinstance(node, ast.Expr) and event_owner(node.value) in (
                    "model_controls_event",
                    "restore_recommended_size",
                    "resolution_auto",
                ):
                    self.execute(node)
        self.values = {component._id: component.value for component in components.values()}
        self.state = SessionState(self.app)
        self.request = gr.Request(session_hash=f"model-controls-{id(self)}")
        self.events = self.app.config["dependencies"]

    def execute(self, node):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), self.scope)

    def run(self, event_id=None):
        async def dispatch(event_id):
            function = self.app.fns[event_id]
            inputs = [
                None if component.stateful else self.values[component._id]
                for component in function.inputs
            ]
            response = await self.app.process_api(
                event_id, inputs, state=self.state, request=self.request
            )
            for component, value in zip(function.outputs, response["data"], strict=True):
                if not component.stateful:
                    if isinstance(value, dict) and value.get("__type__") == "update":
                        if "value" in value:
                            self.values[component._id] = value["value"]
                    else:
                        self.values[component._id] = value
            for event in self.events:
                if event["trigger_after"] == event_id:
                    await dispatch(event["id"])

        asyncio.run(
            dispatch(self.scope["model_controls_event"]["id"] if event_id is None else event_id)
        )

    def value(self, name):
        component = self.scope[name]
        return self.state[component._id] if component.stateful else self.values[component._id]


class ModelControlEventTests(unittest.TestCase):
    def test_one_resolution_update_follows_fresh_recommendation_and_sampling(self):
        graph = ModelControlGraph(self)
        graph.run()
        self.assertEqual(graph.resolutions.model_selection.call_count, 1, graph.trace)
        self.assertEqual(graph.publisher.cached_resolution.call_count, 2)
        self.assertEqual(
            graph.trace,
            [
                "model_image_controls",
                "refresh_recommendation",
                "performance_control_updates",
                "model_resolution",
                "resolution_description",
                "workflow_description",
            ],
        )
        self.assertEqual(graph.value("aspect_ratios_selection"), "832x1216")
        self.assertEqual(graph.value("cfg"), 7)
        self.assertEqual(graph.value("custom_steps"), 12)
        self.assertEqual(graph.value("model_family"), "SDXL")
        triggers = graph.scope["model_controls_event"]["targets"]
        self.assertEqual(
            triggers,
            [
                (graph.scope["base_model"]._id, "change"),
                (graph.scope["model_metadata_refresh"]._id, "change"),
            ],
        )

    def test_explicit_sampling_and_size_keep_ownership_through_metadata_refresh(self):
        graph = ModelControlGraph(
            self, selection="Own settings", current="1216x832", automatic=False
        )
        graph.run()
        self.assertEqual(graph.value("aspect_ratios_selection"), "1216x832")
        self.assertFalse(graph.value("resolution_auto"))
        self.assertEqual(graph.value("performance_selection"), "Own settings")
        self.assertEqual(graph.value("cfg"), 9)
        self.assertEqual(graph.value("custom_steps"), 22)
        graph.publisher.refresh.assert_not_called()
        self.assertEqual(graph.resolutions.model_selection.call_count, 1)

    def test_custom_replay_is_retained_until_restore_recommendation_is_clicked(self):
        graph = ModelControlGraph(self, current="Custom...")
        graph.run()
        self.assertEqual(graph.value("aspect_ratios_selection"), "Custom...")
        self.assertFalse(graph.value("resolution_auto"))
        restore = next(
            event
            for event in graph.events
            if event["targets"] == [(graph.scope["restore_recommended_size"]._id, "click")]
        )
        graph.run(restore["id"])
        self.assertEqual(graph.value("aspect_ratios_selection"), "832x1216")
        self.assertTrue(graph.value("resolution_auto"))
        self.assertEqual(graph.value("size_mode"), "target")

    def test_queued_old_dimensions_keep_manual_ownership_across_family_changes(self):
        for automatic in (True, False):
            with self.subTest(automatic=automatic):
                graph = ModelControlGraph(
                    self,
                    selection="Model Defaults" if automatic else "Own settings",
                    current="1216x832",
                    automatic=automatic,
                )
                graph.run()
                graph.scope["shared"].models.get_models_by_path.return_value = {
                    "baseModel": "Flux3"
                }
                graph.scope["performance_settings"].family_preset.return_value = None
                graph.cached_size = None
                graph.values[graph.scope["auto_recommendations"]._id] = False
                # A queued size value can carry the preceding family's label.
                graph.values[graph.scope["aspect_ratios_selection"]._id] = (
                    "832x1216 (SDXL recommended)"
                )
                graph.resolutions.model_selection.reset_mock()
                graph.publisher.refresh.reset_mock()
                graph.run()
                self.assertEqual(graph.value("model_family"), "Flux3")
                self.assertEqual(
                    graph.value("aspect_ratios_selection"),
                    "1024x1024" if automatic else "832x1216",
                )
                self.assertEqual(graph.value("resolution_auto"), automatic)
                self.assertEqual(graph.resolutions.model_selection.call_count, 1)
                graph.publisher.refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
