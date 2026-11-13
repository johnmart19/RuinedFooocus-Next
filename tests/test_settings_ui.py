"""Settings event contracts without importing the live application or backend."""

import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]


class SettingsProgressTests(unittest.TestCase):
    def test_chat_diagnostics_use_this_sessions_selected_file_and_runtime_projections(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        category = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.optional_vars, ast.Name)
                and item.optional_vars.id == "chat_settings"
                for item in node.items
            )
        )
        call = next(
            node
            for node in category.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "create_model_diagnostics"
        )
        helper_tree = ast.parse(
            (ROOT / "ruinedfooocus/features/chat/ui_model_diagnostics.py").read_text(
                encoding="utf-8"
            )
        )
        helper = next(
            node
            for node in helper_tree.body
            if getattr(node, "name", None) == "create_model_diagnostics"
        )
        evidence, sampling = Mock(return_value=[]), Mock(return_value=[])
        scope = dict(
            gr=gr,
            t=lambda value: value,
            selected_evidence=evidence,
            refresh_sampling_recommendations=sampling,
        )
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "Chat diagnostics", "exec"), scope)
        with gr.Blocks(analytics_enabled=False) as app:
            controls = {
                name: gr.State(value)
                for name, value in (
                    ("chat_selected_quant", "browser-specific.gguf"),
                    ("chat_runtime_status", "initialized runtime"),
                    ("chat_sampling_status", "initialized sampling"),
                )
            }
            scope["get_ctrl"] = controls.__getitem__
            exec(
                compile(
                    ast.Module(body=[call], type_ignores=[]), "Settings diagnostics binding", "exec"
                ),
                scope,
            )
        expand = next(
            event
            for event in app.config["dependencies"]
            if any(target[1] == "expand" for target in event["targets"])
        )
        self.assertEqual(expand["inputs"], [control._id for control in controls.values()])
        group = next(block for block in app.blocks.values() if isinstance(block, gr.Accordion))
        self.assertFalse(group.open)
        evidence.assert_not_called()
        sampling.assert_not_called()

    def test_profile_actions_follow_categories_and_keep_the_registered_form_inputs(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body if getattr(node, "name", None) == "create_settings"
        )
        layout = function.body[0]
        # Exercise production composition and event registration. Replace only
        # expensive feature categories, which would otherwise import backends.
        for node in layout.body:
            if isinstance(node, ast.With) and any(
                isinstance(item.optional_vars, ast.Name)
                and item.optional_vars.id == "settings_tabs"
                for item in node.items
            ):
                node.body = ast.parse(
                    'with gr.Tab("Fixture") as models_tab:\n'
                    '    field = gr.Slider(0, 10, value=5, label="Fixture setting")\n'
                    '    add_setting("fixture", field)\n'
                ).body
        endpoint = next(
            index
            for index, node in enumerate(layout.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "bind_navigation"
        )
        layout.body = layout.body[: endpoint + 1]
        registry = {"setting_name": [], "setting_obj": []}

        def register(name, control):
            registry["setting_name"].append(name)
            registry["setting_obj"].append(control)

        scope = dict(
            gr=gr,
            t=lambda text: text,
            state=registry,
            add_setting=register,
            settings=SimpleNamespace(name="fixture", default_settings={}),
            path_manager=SimpleNamespace(paths={}),
            SETTINGS_INDEX={},
            models_in_settings=True,
        )
        with patch("ruinedfooocus.core.profile_store.selected_profile", return_value="fixture"):
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[layout], type_ignores=[])),
                    "Settings composition",
                    "exec",
                ),
                scope,
            )
        app = scope["app_settings"]
        content = next(
            block for block in app.blocks.values() if block.elem_id == "settings-content"
        )
        self.assertLess(
            content.children.index(scope["settings_tabs"]),
            content.children.index(scope["profile_status"]),
        )
        self.assertLess(
            content.children.index(scope["profile_status"]),
            content.children.index(scope["profile_area"]),
        )
        self.assertEqual(content.children[-1].elem_id, "settings-actions")
        self.assertEqual(registry["setting_name"], ["fixture", "ui_settings_name"])
        self.assertEqual(scope["setting_controls"], [scope["field"], scope["ui_settings_name"]])
        save_event = next(
            event
            for event in app.config["dependencies"]
            if event["targets"] == [(scope["save_btn"]._id, "click")]
        )
        self.assertEqual(
            save_event["inputs"],
            [
                scope["save_target"]._id,
                scope["profile_base"]._id,
                scope["field"]._id,
                scope["ui_settings_name"]._id,
            ],
        )
        self.assertTrue(all(target[1] != "select" for target in save_event["targets"]))

    def test_reloaded_chat_limits_follow_saved_profile_without_restoring_cleared_values(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "current_sampling"
        )
        event = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "current_sampling"
        )
        manager = SimpleNamespace(default_settings={"llm_n_ctx": 8192, "llm_n_predict": -1})
        scope = dict(
            settings=manager,
            SAMPLERS=("temperature",),
            llm_n_ctx="context",
            llm_n_predict="output",
            use_model_defaults="publisher",
            sampling_controls=["temperature"],
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "reload defaults", "exec"), scope
        )
        outputs = next(keyword.value for keyword in event.keywords if keyword.arg == "outputs")
        controls = eval(compile(ast.Expression(outputs), "reload controls", "eval"), scope)
        for saved, expected in (
            ({}, (None, None)),
            ({"llm_n_ctx": 0, "llm_n_predict": -1}, (0, -1)),
            ({"llm_n_ctx": 2048, "llm_n_predict": 256}, (2048, 256)),
        ):
            manager.default_settings = saved
            result = dict(zip(controls, scope["current_sampling"](), strict=True))
            self.assertEqual((result["context"], result["output"]), expected)

    def test_configuration_refresh_declares_every_returned_component(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "update_cfg"
        )
        event = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cfg_timestamp"
            and node.func.attr == "change"
        )
        scope = dict(
            gr=SimpleNamespace(update=lambda **values: values),
            settings={"image_number_max": 50},
            shared=SimpleNamespace(
                state={"last_config": "now"}, settings=SimpleNamespace(name="fixture")
            ),
            performance_settings=SimpleNamespace(choices_for_model=lambda *_: ["Model Defaults"]),
            resolution_settings=SimpleNamespace(
                choices_for_model=lambda _: ["square"],
                selection_for_model=lambda _, value: value,
                set_profile=lambda _: None,
            ),
            model_resolution=lambda family, checkpoint, current, automatic: (
                {"value": current},
                automatic,
            ),
            **{
                key: key
                for key in (
                    "image_number",
                    "performance_selection",
                    "aspect_ratios_selection",
                    "resolution_auto",
                    "cfg_timestamp",
                )
            },
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), "refresh", "exec"), scope)
        result = scope["update_cfg"]("SDXL", "checkpoint", "square", False)
        outputs = next(keyword.value for keyword in event.keywords if keyword.arg == "outputs")
        declared = eval(compile(ast.Expression(outputs), "refresh outputs", "eval"), scope)
        self.assertEqual(set(result), set(declared))
        self.assertEqual(result["aspect_ratios_selection"]["value"], "square")
        self.assertFalse(result["resolution_auto"])

    def test_fast_settings_events_do_not_overlay_other_controls(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        receivers = {
            "settings_search",
            "language",
            "text_size",
            "app_settings",
            "reset_shifts",
            "llm_runtime",
        }
        events = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Name)
                and receiver.id in receivers
                and node.func.attr in {"change", "input", "load", "click"}
            ) or (
                node.func.attr == "tick"
                and isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "Timer"
            ):
                events.append(node)
        # Search is now browser-only field navigation, registered by its helper.
        self.assertEqual(len(events), 6)
        for event in events:
            keywords = {keyword.arg: keyword.value for keyword in event.keywords}
            with self.subTest(line=event.lineno):
                self.assertEqual(ast.literal_eval(keywords["show_progress"]), "hidden")

    def test_installs_refresh_the_support_table_after_completion(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_hardware.py").read_text(encoding="utf-8"))
        refreshed = set()
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "then"
            ):
                continue
            event = node.func.value
            if (
                isinstance(event, ast.Call)
                and isinstance(event.func, ast.Attribute)
                and isinstance(event.func.value, ast.Name)
            ):
                refreshed.add(event.func.value.id)
                keywords = {key.arg: key.value for key in node.keywords}
                self.assertEqual(keywords["outputs"].id, "support")
        self.assertTrue(
            {
                "switch_image",
                "install_llama",
                "build_runtime",
                "install_xllama",
                "install_background",
                "install_winml",
                "onnx_install",
            }
            <= refreshed
        )

    def test_hardware_tab_registers_unique_keys_and_hides_fast_change_progress(self):
        literal_keys = []
        for name in ("ruinedfooocus/ui/ui_settings.py", "ruinedfooocus/ui/ui_hardware.py"):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "add_setting"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    literal_keys.append(node.args[0].value)
        self.assertEqual(len(literal_keys), len(set(literal_keys)))
        for expected in (
            "cpu_everywhere",
            "image_backend",
            "llama_backend",
            "xllamacpp_backend",
            "rembg_backend",
            "rembg_provider",
            "rembg_winml_provider",
            "rembg_winml_type",
            "rembg_winml_device",
        ):
            self.assertIn(expected, literal_keys)
        hardware_tree = ast.parse(
            (ROOT / "ruinedfooocus/ui/ui_hardware.py").read_text(encoding="utf-8")
        )
        change_events = [
            node
            for node in ast.walk(hardware_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "change"
        ]
        self.assertTrue(
            {"image_backend", "llama_backend", "xllama_backend", "cpu_everywhere"}
            <= {
                event.func.value.id
                for event in change_events
                if isinstance(event.func.value, ast.Name)
            }
        )
        for event in change_events:
            keywords = {keyword.arg: keyword.value for keyword in event.keywords}
            self.assertEqual(ast.literal_eval(keywords["show_progress"]), "hidden")


if __name__ == "__main__":
    unittest.main()
