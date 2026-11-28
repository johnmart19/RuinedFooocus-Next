"""Maintenance messages use each request's locale without changing user paths."""

import ast
import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.core import translation_manager
from ruinedfooocus.core.maintenance_reset import ResetPlan
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.core.translation_manager import catalogue, localize, resolve_locale
from ruinedfooocus.ui import storage_reset as reset


class StorageResetLocalizationTests(unittest.TestCase):
    def test_registered_load_and_picker_renderers_accept_empty_and_previewed_plan(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, RF_DATA_DIR=str(root)))
        settings = SettingsManager("Portrait-日本語", defaults={"prompt": "factory"})
        manager = PathManager.__new__(PathManager)
        manager.name = settings.name
        manager.set_settings_path(settings.name)
        manager.paths = manager.load_paths()
        manager.model_paths = manager.get_model_paths()

        # Exercise the application's actual locale-first event adapter without
        # importing app.py, whose module body starts the complete application.
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/app.py"
        node = next(
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "localized_renderer"
        )
        scope = {"gr": gr, "resolve_locale": resolve_locale}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        localized_renderer = scope["localized_renderer"]
        bindings = []
        with (
            patch.object(translation_manager, "_LOCALIZED_OUTPUTS", []),
            gr.Blocks(analytics_enabled=False) as app,
        ):
            language = gr.Dropdown(choices=list(catalogue().manifest["locales"]), value="en")
            reset.create_storage_reset(manager, settings)
            for render, inputs, outputs in translation_manager.localized_outputs():
                load = localized_renderer(render, False)
                change = localized_renderer(render, True)
                app.load(load, inputs=inputs, outputs=outputs, queue=False)
                language.change(change, inputs=[language, *inputs], outputs=outputs, queue=False)
                bindings.append((load, change, inputs, outputs))
        self.addCleanup(app.close)
        self.assertEqual(len(bindings), 3)
        functions = {value.fn.__name__: key for key, value in app.fns.items() if value.fn}
        function_id = {value.fn: key for key, value in app.fns.items() if value.fn}
        plan_input = bindings[0][2][0]

        async def check(locale):
            state = SessionState(app)
            request = gr.Request(
                session_hash="maintenance-render-" + locale,
                cookies={"rf_ui_locale": locale},
            )

            async def render_events():
                values = []
                for load, change, inputs, _ in bindings:
                    for callback, data in (
                        (load, [None] * len(inputs)),
                        (change, [locale, *([None] * len(inputs))]),
                    ):
                        values.append(
                            (
                                await app.process_api(
                                    function_id[callback], data, state=state, request=request
                                )
                            )["data"][0]
                        )
                return values

            empty = await render_events()
            self.assertEqual(empty[0]["data"], [])
            self.assertEqual(empty[1]["data"], [])
            self.assertEqual(empty[2:4], ["", ""])
            checked = await app.process_api(
                functions["check_folders"], [], state=state, request=request
            )
            self.assertTrue(checked["data"][0].startswith(localize("Completed", locale)))
            await app.process_api(
                functions["preview_factory"], [None], state=state, request=request
            )
            plan = state[plan_input._id]
            self.assertIsInstance(plan, ResetPlan)
            self.assertEqual(plan.profile, settings.name)
            rendered = await render_events()
            for table in rendered[:2]:
                self.assertEqual(table["data"], reset.target_rows(plan, locale))
            self.assertEqual(rendered[2:4], [reset.retained_text(plan, locale)] * 2)
            for headers in rendered[4:]:
                self.assertEqual(headers["headers"][0], localize("Path", locale))

        for locale in catalogue().manifest["locales"]:
            with self.subTest(locale=locale):
                asyncio.run(check(locale))

    def test_native_locale_callback_preserves_profile_paths_and_byte_units(self):
        profile = "Portrait-日本語"
        path = r"D:\User Inputs\kept-日本語.png"
        plan = SimpleNamespace(action="factory-profile", profile=profile)
        target = SimpleNamespace(path=path, kind="output-file", bytes=123, rows=0)

        def render(request: gr.Request):
            locale = resolve_locale(request=request)
            return (
                reset.retained_text(plan, locale),
                reset.error_text("Linked maintenance target is not eligible: " + path, locale),
                reset.result_text({"complete": True, "profile_reset": True}, locale),
                reset.target_rows(SimpleNamespace(targets=[target]), locale),
            )

        with gr.Blocks(analytics_enabled=False) as app:
            outputs = [gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.JSON()]
            gr.Button().click(render, outputs=outputs)
        self.addCleanup(app.close)
        for locale in catalogue().manifest["locales"]:
            with self.subTest(locale=locale):
                request = gr.Request(
                    session_hash="maintenance-" + locale,
                    cookies={"rf_ui_locale": locale},
                )
                result = asyncio.run(
                    app.process_api(0, [], state=SessionState(app), request=request)
                )["data"]
                self.assertEqual(
                    result[0],
                    localize(
                        "Current profile: {profile}. Other profiles are kept.",
                        locale,
                        profile=profile,
                    ),
                )
                self.assertIn(profile, result[0])
                self.assertIn(path, result[1])
                rows = result[3].root
                self.assertEqual(rows[0][0], path)
                self.assertEqual(rows[0][3], "123 B")
                if locale != "en":
                    self.assertNotIn("Other profiles are kept.", result[0])
                    self.assertNotIn("Linked maintenance target is not eligible", result[1])
                    self.assertNotIn("Current profile defaults restored.", result[2])

    def test_known_core_diagnostics_have_request_local_translations(self):
        warnings = (
            "Cache preview limit exceeded; no reset was performed",
            "Confirm the previewed destructive action first",
            "Maintenance target is outside its owned folder",
            "Preparation is busy; retry after it finishes",
            "Preparation or an artifact is in use; retry after it finishes",
            "Targets changed after preview; recheck them before confirming",
            "The application is busy; finish or stop current work first",
            "The current profile is unavailable",
        )
        for locale in catalogue().manifest["locales"]:
            for warning in warnings:
                with self.subTest(locale=locale, warning=warning):
                    translated = reset.error_text(warning, locale)
                    self.assertEqual(translated, localize(warning, locale))
                    if locale != "en":
                        self.assertNotEqual(translated, warning)
