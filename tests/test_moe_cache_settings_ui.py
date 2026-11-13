"""Native Settings events and isolated profiles; no runtime or model is executed."""

import ast
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.backends.llama import moe_cache
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.core.translation_manager import catalogue, localized_outputs
from ruinedfooocus.features.chat import ui_moe_cache as ui
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


class MoeCacheSettingsUiTests(unittest.TestCase):
    def make_ui(self, saved=None):
        saved = dict(moe_cache.DEFAULTS if saved is None else saved)
        registered = {}

        def add_setting(key, control):
            self.assertNotIn(key, registered)
            registered[key] = control

        with gr.Blocks(analytics_enabled=False) as app:
            with gr.Tabs(), gr.Tab("Local Chat") as tab:
                runtime = gr.Dropdown(["llama.cpp", "xllamacpp"], value="llama.cpp")
                arguments = gr.Textbox(value="")
                profile = gr.State({"settings": saved})
                controls = ui.create_moe_cache_settings(
                    tab, saved, add_setting, runtime, arguments, profile, on_load=app.load
                )
        callback = next(fn for fn in app.fns.values() if fn.fn.__name__ == "update_moe_status")
        return app, controls, registered, profile, callback

    def predict(self, app, callback, state, values):
        return asyncio.run(
            app.process_api(
                callback, values, state=state, request=gr.Request(session_hash="fixture")
            )
        )["data"][0]

    def test_native_config_keeps_requested_default_and_saved_zero_editable_without_probe(self):
        with (
            patch.object(moe_cache, "parser_capability", side_effect=AssertionError("No probe")),
            patch.object(ui, "initialized_state", return_value=None),
        ):
            for saved, enabled, share in (
                (None, True, 0.5),
                ({ui.ENABLED: False, ui.SHARE: 0}, False, 0),
            ):
                app, controls, registered, profile, callback = self.make_ui(saved)
                self.assertEqual(set(registered), {ui.ENABLED, ui.SHARE})
                self.assertEqual(controls["enabled"].get_config()["value"], enabled)
                config = controls["share"].get_config()
                self.assertEqual(
                    (config["value"], config["minimum"], config["maximum"]), (share, 0, 1)
                )
                self.assertIn("b12110", controls["status"].value)
                self.assertIsNot(controls["enabled"].interactive, False)
                self.assertEqual(callback.outputs, [controls["status"]])
                dependency = next(
                    item for item in app.config["dependencies"] if item["id"] == callback._id
                )
                self.assertFalse(dependency["queue"])
                self.assertEqual(dependency["show_progress"], "hidden")

    def test_actual_events_keep_two_profile_drafts_separate_and_do_not_write_controls(self):
        with patch.object(ui, "initialized_state", return_value=None):
            app, controls, _, profile, callback = self.make_ui()
            first, second = SessionState(app), SessionState(app)
            first[profile._id] = {"settings": {ui.ENABLED: False, ui.SHARE: 0.25}}
            second[profile._id] = {"settings": {ui.ENABLED: True, ui.SHARE: 0.8}}
            text = self.predict(app, callback, first, ["llama.cpp", False, 0.333, "", None])
            other = self.predict(app, callback, second, ["xllamacpp", True, 0.6, "", None])
        self.assertIn("Form: Off · share 0.333. Saved: Off · share 0.25", text)
        self.assertIn("Form: On · share 0.6. Saved: On · share 0.8", other)
        self.assertIn("xllamacpp", other)
        self.assertEqual(first[profile._id]["settings"][ui.SHARE], 0.25)
        self.assertEqual(second[profile._id]["settings"][ui.SHARE], 0.8)
        self.assertEqual(callback.outputs, [controls["status"]])

    def test_initialized_cli_override_is_distinct_from_form_and_private_tokens_are_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "inert-parser-fixture"
            executable.write_bytes(b"--moe-cache-cpu-share\0LLAMA_ARG_MOE_CACHE_CPU_SHARE\0")
            loaded = moe_cache.moe_cache_cpu_share_capability(
                {"llama_server_args": "--cpu-moe --moe-cache-mib 256 --moe-cache-cpu-share 0.8"},
                executable,
                gpu=True,
                environment={},
            )
            with patch(
                "ruinedfooocus.app.resource_lifecycle.service_pipeline",
                return_value=SimpleNamespace(moe_cache_cpu_share=loaded),
            ) as lookup:
                public = ui.initialized_state()
                text = ui.status_text(
                    "llama.cpp",
                    False,
                    0.25,
                    "--moe-cache-cpu-share=0.8",
                    {"settings": {ui.ENABLED: True, ui.SHARE: 0.5}},
                )
            lookup.assert_called_with("chat")
        self.assertNotIn("arguments", public)
        self.assertNotIn("extra_arguments", public)
        self.assertIn("Form: Off · share 0.25. Saved: On · share 0.5", text)
        self.assertIn("Form selected share: 0.8 · source: Extra llama.cpp arguments", text)
        self.assertIn("Initialized Chat: Active · effective share: 0.8", text)
        self.assertNotIn("--cpu-moe", text)
        self.assertNotIn(str(executable), text)

    def test_invalid_arguments_return_status_without_altering_saved_profile(self):
        with patch.object(ui, "initialized_state", return_value=None):
            app, _, _, profile, callback = self.make_ui()
            state = SessionState(app)
            state[profile._id] = {"settings": dict(moe_cache.DEFAULTS)}
            text = self.predict(
                app, callback, state, ["llama.cpp", True, 0.5, "--moe-cache-cpu-share NaN", None]
            )
        self.assertIn("MoE cache preference is invalid", text)
        self.assertEqual(state[profile._id]["settings"], moe_cache.DEFAULTS)

    def test_actual_profile_save_persists_opt_out_and_fraction_without_activating_runtime(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, RF_DATA_DIR=directory),
        ):
            manager = SettingsManager(
                "moe-ui", defaults={**moe_cache.DEFAULTS, "unrelated": "retained"}
            )
            paths = SimpleNamespace(paths={}, DEFAULT_PATHS={}, set_settings_path=Mock())
            names = ["llm_runtime", "llama_server_args", ui.ENABLED, ui.SHARE]
            save = function_from(
                "ruinedfooocus/ui/ui_settings.py",
                "save_clicked",
                dict(
                    state={"setting_name": names},
                    settings=manager,
                    path_manager=paths,
                    validate_settings=validate_settings,
                    gr=SimpleNamespace(Error=ValueError, Info=Mock()),
                    models=SimpleNamespace(offline=True),
                    performance_settings=SimpleNamespace(
                        model_recommendations=SimpleNamespace(offline=True)
                    ),
                ),
            )
            with patch.object(ui, "initialized_state", return_value=None):
                app, controls, _, profile, callback = self.make_ui(manager.default_settings)
            with app:
                save_button = gr.Button("Save fixture profile")

                def save_profile(runtime, arguments, enabled, share):
                    return save(runtime, arguments, enabled, share, return_snapshot=True)[1]

                save_button.click(
                    save_profile,
                    inputs=[
                        callback.inputs[0],
                        callback.inputs[3],
                        controls["enabled"],
                        controls["share"],
                    ],
                    outputs=profile,
                )
            save_callback = next(fn for fn in app.fns.values() if fn.fn.__name__ == "save_profile")
            state = SessionState(app)
            state[profile._id] = {"settings": manager.default_settings}
            launch = SimpleNamespace(args=SimpleNamespace(settings="moe-ui", offline=True))
            with (
                patch.dict(sys.modules, {"ruinedfooocus.app.arguments": launch}),
                patch.object(ui, "initialized_state", return_value=None),
            ):
                asyncio.run(
                    app.process_api(
                        save_callback,
                        ["llama.cpp", "--moe-cache-cpu-share=0.8", False, 0.333],
                        state=state,
                    )
                )
                text = self.predict(
                    app,
                    callback,
                    state,
                    ["llama.cpp", False, 0.333, "--moe-cache-cpu-share=0.8", None],
                )
            reloaded = SettingsManager("moe-ui")
        self.assertFalse(reloaded.default_settings[ui.ENABLED])
        self.assertEqual(reloaded.default_settings[ui.SHARE], 0.333)
        self.assertEqual(reloaded.default_settings["unrelated"], "retained")
        self.assertIn("Saved: Off · share 0.333", text)
        self.assertIn("Form selected share: 0.8", text)
        self.assertIn("b12110", text)

    def test_locale_refresh_updates_only_status_and_keeps_preferences_literal(self):
        with patch.object(ui, "initialized_state", return_value=None):
            _, controls, _, _, _ = self.make_ui()
            render, inputs, outputs = localized_outputs()[-1]
            manager = catalogue()
            message = "Form selected share: {share} · source: {source}."
            key = manager.aliases[message]
            with patch.dict(
                manager.translations["uk"], {key: "Обрано: {share} · джерело: {source}."}
            ):
                text = render("uk", *(control.value for control in inputs))
        self.assertIn("Обрано: 0.5", text)
        self.assertEqual(outputs, [controls["status"]])
        self.assertTrue(controls["enabled"].value)
        self.assertEqual(controls["share"].value, 0.5)

    def test_settings_places_one_feature_control_after_gpu_layers(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        additions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_moe_cache_settings"
        ]
        self.assertEqual(len(additions), 1)
        setting_calls = {
            node.args[0].value: node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "add_setting"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        self.assertLess(setting_calls["llm_n_gpu_layers"], additions[0].lineno)
        self.assertLess(additions[0].lineno, setting_calls["llm_chat_history"])

    def test_same_app_fresh_session_restores_saved_preferences_only_on_page_load(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, RF_DATA_DIR=directory),
        ):
            with patch.object(ui, "initialized_state", return_value=None):
                app, controls, _, profile, status_callback = self.make_ui()
                self.assertTrue(controls["enabled"].value)
                self.assertEqual(controls["share"].value, 0.5)
                saved_path = Path(directory) / "settings/moe-reload/settings.json"
                saved_path.parent.mkdir(parents=True)
                saved_path.write_text(
                    json.dumps({ui.ENABLED: False, ui.SHARE: 0.333}), encoding="utf-8"
                )
                fresh = SessionState(app)
                fresh[profile._id] = {
                    "profile": "moe-reload",
                    "settings": {**moe_cache.DEFAULTS, "unrelated": "retained"},
                }
                load = next(
                    fn for fn in app.fns.values() if fn.fn.__name__ == "restore_moe_preferences"
                )
                result = asyncio.run(app.process_api(load, ["llama.cpp", "", None], state=fresh))[
                    "data"
                ]
                self.assertEqual(result[:2], [False, 0.333])
                self.assertIn("Saved: Off · share 0.333", result[3])
                self.assertEqual(fresh[profile._id]["settings"]["unrelated"], "retained")
                dependency = next(
                    item for item in app.config["dependencies"] if item["id"] == load._id
                )
                self.assertTrue(all(event == "load" for _, event in dependency["targets"]))
                text = self.predict(app, status_callback, fresh, ["llama.cpp", True, 0.7, "", None])
        self.assertIn("Form: On · share 0.7. Saved: Off · share 0.333", text)
        self.assertEqual(status_callback.outputs, [controls["status"]])

    def test_moe_search_extends_existing_keywords_without_requiring_other_chat_features(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        statements = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "SETTINGS_INDEX"
                    for target in node.targets
                )
            )
            or (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Subscript)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "SETTINGS_INDEX"
            )
        ]
        scope = {}
        exec(compile(ast.Module(body=statements, type_ignores=[]), "Settings index", "exec"), scope)
        search = function_from("ruinedfooocus/ui/ui_settings.py", "search_settings", scope)
        self.assertIn("**Local Chat**", search("MoE cache CPU share"))
        self.assertIn("**Local Chat**", search("llama.cpp"))


if __name__ == "__main__":
    unittest.main()
