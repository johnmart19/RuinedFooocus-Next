"""Native size bindings preserve ownership without cycling or tracking view progress."""

import ast
import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.features.models.resolution_dropdown import ResolutionDropdown
from ruinedfooocus.features.models.resolutions import ResolutionSettings
import test_layout_progress_targets as progress_fixtures


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ruinedfooocus/ui/app.py"


class ResolutionUiEventTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        replacement = patch.object(
            ResolutionSettings, "RESOLUTIONS_FILE", Path(temporary.name) / "sizes.json"
        )
        replacement.start()
        self.addCleanup(replacement.stop)
        self.settings = ResolutionSettings()
        self.performance = SimpleNamespace(
            model_recommendations=SimpleNamespace(
                cached_resolution=lambda _: (None, None), metadata=lambda _: {}
            ),
            family_preset=lambda *_: "SDXL",
        )
        self.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    def fixture(self):
        choices, selected, _ = self.settings.model_selection("SDXL", None, True)
        with gr.Blocks(analytics_enabled=False) as app:
            controls = dict(
                model_family=gr.Textbox(value="SDXL"),
                base_model=gr.Textbox(value="inert.safetensors"),
                aspect_ratios_selection=ResolutionDropdown(choices, value=selected),
                resolution_auto=gr.Checkbox(value=True),
                size_mode=gr.State("target"),
                custom_width=gr.Number(value=1024),
                custom_height=gr.Number(value=1024),
                ratio_name=gr.Textbox(value="Draft name"),
                ratio_save=gr.Button(),
                ratio_delete=gr.Button(),
                target_shortcuts=gr.Row(),
                restore_recommended_size=gr.Button(),
            )
            scope = dict(
                controls,
                gr=gr,
                resolution_settings=self.settings,
                performance_settings=self.performance,
            )
            nodes = []
            for name in ("model_resolution", "aspect_ratios_changed"):
                nodes.append(
                    deepcopy(
                        next(
                            node
                            for node in ast.walk(self.tree)
                            if isinstance(node, ast.FunctionDef) and node.name == name
                        )
                    )
                )
            nodes.extend(
                deepcopy(node)
                for node in ast.walk(self.tree)
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id in ("resolution_auto", "restore_recommended_size")
                and node.value.func.attr in ("input", "click")
            )
            nodes.append(
                deepcopy(
                    next(
                        node
                        for node in ast.walk(self.tree)
                        if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name)
                        and node.target.id == "size_control"
                    )
                )
            )
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
        return app, controls

    def test_programmatic_updates_do_not_reenter_manual_or_auto_ownership(self):
        app, controls = self.fixture()
        dependencies = app.get_config_file()["dependencies"]
        programmatic = [
            row for row in dependencies if any(event == "change" for _, event in row["targets"])
        ]
        self.assertEqual(len(programmatic), 1)
        projection = programmatic[0]
        self.assertEqual(
            projection["targets"], [(controls["aspect_ratios_selection"]._id, "change")]
        )
        self.assertNotIn(controls["aspect_ratios_selection"]._id, projection["outputs"])
        # Auto and dimension updates can only enter ownership callbacks through
        # genuine user input. Their programmatic writes cannot retrigger those.
        for name in ("resolution_auto", "custom_width", "custom_height"):
            events = [
                event
                for row in dependencies
                for component, event in row["targets"]
                if component == controls[name]._id
            ]
            self.assertEqual(events, ["input"])

    def test_native_preset_custom_manual_and_auto_transitions_keep_pixels_and_draft(self):
        app, controls = self.fixture()
        dependencies = app.get_config_file()["dependencies"]
        state = SessionState(app)

        def event(name, kind):
            target = (controls[name]._id, kind)
            return next(row["id"] for row in dependencies if target in row["targets"])

        async def run():
            preset = await app.process_api(
                event("aspect_ratios_selection", "change"), ["1152x896"], state=state
            )
            self.assertEqual(preset["data"][1]["value"], 1152)
            self.assertEqual(preset["data"][2]["value"], 896)
            self.assertNotIn("value", preset["data"][0])  # Saved-name draft survives.
            self.assertNotIn("value", preset["data"][5])  # Programmatic preset preserves Auto.
            repeated = await app.process_api(
                event("aspect_ratios_selection", "change"), ["1152x896"], state=state
            )
            self.assertEqual(repeated["data"], preset["data"])
            self.assertEqual(repeated["changed_state_ids"], [])
            custom = await app.process_api(
                event("aspect_ratios_selection", "change"), ["Custom..."], state=state
            )
            self.assertFalse(custom["data"][5])
            self.assertNotIn("value", custom["data"][1])
            self.assertNotIn("value", custom["data"][2])
            manual = await app.process_api(event("custom_width", "input"), [], state=state)
            self.assertFalse(manual["data"][0])
            automatic = await app.process_api(
                event("resolution_auto", "input"),
                ["SDXL", "inert.safetensors", "Custom...", True],
                state=state,
            )
            self.assertEqual(automatic["data"][0]["value"], "1024x1024")
            self.assertTrue(automatic["data"][1])
            shown = await app.process_api(
                event("aspect_ratios_selection", "change"), ["1024x1024"], state=state
            )
            self.assertEqual((shown["data"][1]["value"], shown["data"][2]["value"]), (1024, 1024))

        asyncio.run(run())

    def test_all_size_view_registrations_keep_outputs_without_native_progress_targets(self):
        registrations = (
            ("aspect_ratios_changed", 7, (6,)),
            ("model_resolution", 2, ()),
            ("resolution_recommendation_text", 1, ()),
            ("update_upscale_size", 1, ()),
        )
        helper = progress_fixtures.LayoutProgressTargetsTests()
        for callback, count, layouts in registrations:
            _, calls = progress_fixtures.declarations(SOURCE, callback)
            for binding, call in enumerate(calls):
                # User Auto/restore callbacks add size_mode to the actual model
                # transition's two outputs; preserve all three in the fixture.
                outputs = next(k.value for k in call.keywords if k.arg == "outputs")
                actual_count = len(outputs.elts) if isinstance(outputs, ast.List) else count
                with self.subTest(callback=callback, line=call.lineno):
                    _, config, components = helper.registered(
                        ("../ui/app.py", callback, actual_count, layouts), binding
                    )
                    self.assertEqual(config["outputs"], [item._id for item in components])
                    self.assertEqual(config["show_progress"], "hidden")
                    self.assertEqual(config["show_progress_on"], [])


if __name__ == "__main__":
    unittest.main()
