"""Theme contracts: native tokens, explicit acquisition and private profiles."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.ui import themes

ROOT = Path(__file__).resolve().parents[1]


class ThemeTests(unittest.TestCase):
    def setUp(self):
        private = tempfile.TemporaryDirectory(prefix="rf-theme-")
        self.addCleanup(private.cleanup)
        env = patch.dict(os.environ, RF_DATA_DIR=private.name)
        env.start()
        self.addCleanup(env.stop)

    def cache(self, value="author/theme", schema=None, **extra):
        path = themes.cache_file(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "id": value,
                    "schema": schema or {"theme": {"body_text_color": "#123456"}},
                    **extra,
                }
            ),
            encoding="utf-8",
        )
        return path

    def download(self, value="author/theme", payload=None, *, siblings=None, private=False):
        versions = siblings or [
            SimpleNamespace(rfilename="themes/theme_schema@1.0.0.json", size=100),
            SimpleNamespace(rfilename="themes/theme_schema@2.0.0.json", size=100),
            SimpleNamespace(rfilename="themes/README.md", size=100),
        ]
        info = SimpleNamespace(sha="a" * 40, private=private, siblings=versions)
        api = Mock()
        api.space_info.return_value = info
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [
            payload
            or json.dumps(
                {
                    "theme": {"body_text_color": "#123456"},
                }
            ).encode()
        ]
        with (
            patch("huggingface_hub.HfApi", return_value=api) as factory,
            patch("requests.get", return_value=response) as request,
        ):
            result = themes.acquire_theme(value)
        factory.assert_called_once_with(token=False)
        api.space_info.assert_called_once_with("author/theme", files_metadata=True, timeout=15)
        return result, request

    def test_native_builtins_compile_without_font_or_stylesheet_requests(self):
        for value, (_, constructor) in themes.BUILTINS.items():
            with self.subTest(theme=value):
                theme, warning = themes.resolve_theme(value)
                self.assertIsInstance(theme, constructor)
                self.assertFalse(warning)
                self.assertFalse(theme._stylesheets)
                self.assertFalse(theme._font_css)
                css = theme._get_theme_css()
                self.assertNotIn("@font-face", css)
                self.assertNotIn("https://", css)
                self.assertEqual(theme.font, "Arial, sans-serif")
                self.assertEqual(theme.text_md, "0.875rem")
                self.assertEqual(theme.spacing_lg, ".75rem")

    def test_curated_text_and_selected_colors_meet_normal_text_contrast(self):
        def luminance(color):
            color = {"white": "#ffffff", "black": "#000000"}.get(color, color)
            self.assertRegex(color, r"^#[0-9a-fA-F]{6}$")
            channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
            linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

        pairs = [
            ("body_text_color", "body_background_fill"),
            ("body_text_color_subdued", "body_background_fill"),
            ("block_info_text_color", "block_background_fill"),
            ("block_label_text_color", "block_label_background_fill"),
            ("checkbox_label_text_color_selected", "checkbox_label_background_fill_selected"),
            ("button_secondary_text_color_hover", "button_secondary_background_fill_hover"),
        ]
        pairs += [
            ("button_primary_text_color" + state, "button_primary_background_fill" + state)
            for state in ("", "_hover")
        ]
        pairs += [
            ("link_text_color" + state, "body_background_fill")
            for state in ("", "_hover", "_visited", "_active")
        ]
        for value in themes.BUILTINS:
            theme, _ = themes.resolve_theme(value)
            for text, background in pairs:
                for mode in ("", "_dark"):
                    with self.subTest(theme=value, text=text, mode=mode):
                        levels = sorted(
                            (
                                luminance(theme._get_computed_value(text + mode)),
                                luminance(theme._get_computed_value(background + mode)),
                            )
                        )
                        self.assertGreaterEqual((levels[1] + 0.05) / (levels[0] + 0.05), 4.5)
            # RF's selected Chat source CSS uses white over color_accent.
            self.assertGreaterEqual(
                1.05 / (luminance(theme._get_computed_value("color_accent")) + 0.05), 4.5
            )

    def test_legacy_defaults_and_ids_are_preserved_without_network(self):
        with (
            patch("requests.get", side_effect=AssertionError("network")),
            patch("huggingface_hub.HfApi", side_effect=AssertionError("network")),
        ):
            for value in (None, "", "None", " None "):
                self.assertEqual(themes.theme_id(value), "")
                self.assertFalse(themes.resolve_theme(value)[1])
            for value in ("gradio/soft", "old invalid ID"):
                self.assertEqual(themes.validate_selection(value, previous=value), value)
                self.assertIn((value + " (not cached)", value), themes.theme_choices(value))
                self.assertIn("Using Default", themes.resolve_theme(value)[1])

    def test_invalid_ids_and_version_specs_are_rejected(self):
        for value in (
            [],
            10,
            "https://host/theme",
            "a/../b",
            "a/a..b",
            "a/b@nope",
            "a/b@",
            "a/" + "b" * 256,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                themes.theme_id(value)
        self.assertEqual(
            themes.theme_id(" author/theme@>=1.0.0,<2.0.0 "), "author/theme@>=1.0.0,<2.0.0"
        )

    def test_new_uncached_selection_is_rejected_until_explicit_acquisition(self):
        with self.assertRaisesRegex(ValueError, "Download"):
            themes.validate_selection("author/theme")
        self.cache()
        self.assertEqual(themes.validate_selection("author/theme"), "author/theme")

    def test_native_schema_is_accepted_and_remote_fonts_and_custom_css_are_stripped(self):
        schema = gr.themes.Soft().to_dict()
        schema["theme"].update(
            {
                "_stylesheets": ["https://example.invalid/font.css"],
                "_font_css": ["@import 'https://example.invalid/font.css';"],
                "custom_css": "body { display: none; }",
                "font": "Remote Font",
            }
        )
        normalized = themes.normalized_schema(schema)
        self.cache(schema=normalized)
        theme, warning = themes.resolve_theme("author/theme")
        self.assertFalse(warning)
        self.assertEqual(theme.font, "Arial, sans-serif")
        self.assertNotIn("Remote Font", theme._get_theme_css())
        self.assertFalse(theme.custom_css)
        self.assertFalse(theme._font_css)

    def test_malformed_schema_and_unsafe_css_cannot_be_cached(self):
        invalid = [
            None,
            [],
            {},
            {"theme": []},
            {"theme": {1: "red"}},
            {"theme": {"unknown_token": "red"}},
            {"theme": {"body_text_color": 1}},
        ]
        invalid += [
            {"theme": {"body_text_color": value}}
            for value in (
                "red; display:none",
                "url(https://example.invalid/a)",
                "image-set('a.png' 1x)",
                "u\\72l(a.png)",
                "u/**/rl(a.png)",
                'url("data:image/svg+xml,</style><script>alert(1)</script>")',
                'url("data:image/svg+xml,%3csvg%3e%3cimage href=remote.png /%3e%3c/svg%3e")',
                "*missing_token",
                "*body_text_color_dark",
            )
        ]
        for schema in invalid:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                themes.normalized_schema(schema)

    def test_bad_cache_falls_back_and_choices_skip_it(self):
        path = self.cache()
        for value in ([], {}, {"id": "wrong"}, {"id": "author/theme", "schema": []}):
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(cache=value):
                theme, warning = themes.resolve_theme("author/theme")
                self.assertIsInstance(theme, gr.themes.Default)
                self.assertIn("Using Default", warning)
                self.assertNotIn(("author/theme", "author/theme"), themes.theme_choices())
        path.write_bytes(b"x" * (themes.MAX_BYTES + 1))
        self.assertIn("size", themes.resolve_theme("author/theme")[1])
        for malformed in ("not json", "[" * 2000 + "]" * 2000):
            path.write_text(malformed, encoding="utf-8")
            self.assertIn("Using Default", themes.resolve_theme("author/theme")[1])
            self.assertNotIn(("author/theme", "author/theme"), themes.theme_choices())
        with patch.object(themes.json, "loads", side_effect=RecursionError("deep JSON")):
            self.assertIn("Theme JSON is invalid", themes.resolve_theme("author/theme")[1])
            self.assertNotIn(("author/theme", "author/theme"), themes.theme_choices())

    def test_explicit_download_pins_revision_and_uses_native_version_selection(self):
        value, request = self.download("author/theme@<2.0.0")
        url = request.call_args.args[0]
        self.assertIn("/resolve/" + "a" * 40 + "/", url)
        self.assertTrue(url.endswith("themes/theme_schema%401.0.0.json"))
        self.assertEqual(request.call_args.kwargs, {"stream": True, "timeout": (15, 30)})
        saved = json.loads(themes.cache_file(value).read_text(encoding="utf-8"))
        self.assertEqual(saved["revision"], "a" * 40)
        self.assertIn((value, value), themes.theme_choices())
        self.assertEqual(themes.resolve_theme(value)[0].body_text_color, "#123456")

    def test_latest_native_version_is_used_when_expression_is_omitted(self):
        _, request = self.download()
        self.assertTrue(request.call_args.args[0].endswith("themes/theme_schema%402.0.0.json"))

    def test_offline_and_private_acquisition_do_not_download(self):
        with patch("huggingface_hub.HfApi") as api, patch("requests.get") as request:
            with self.assertRaisesRegex(ValueError, "Offline"):
                themes.acquire_theme("author/theme", offline=True)
            api.assert_not_called()
            request.assert_not_called()
        with self.assertRaisesRegex(ValueError, "public"):
            self.download(private=True)

    def test_invalid_and_oversized_download_preserve_previous_cache(self):
        path = self.cache()
        original = path.read_bytes()
        for payload in (
            b"not json",
            b"[]",
            b"[" * 2000 + b"]" * 2000,
            b"x" * (themes.MAX_BYTES + 1),
        ):
            with self.subTest(payload_size=len(payload)), self.assertRaises(ValueError):
                self.download(payload=payload)
            self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "size"):
            self.download(
                siblings=[
                    SimpleNamespace(
                        rfilename="themes/theme_schema@1.0.0.json",
                        size=themes.MAX_BYTES + 1,
                    )
                ]
            )
        self.assertEqual(path.read_bytes(), original)

    def test_cache_publication_failure_preserves_previous_theme(self):
        path = self.cache()
        original = path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                self.download()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(path.parent.glob("*.tmp")))

    def save_callback(self, manager):
        source = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "save_clicked"
        )
        scope = {
            "state": {"setting_name": ["theme"]},
            "settings": manager,
            "path_manager": SimpleNamespace(paths={}, DEFAULT_PATHS={}, set_settings_path=Mock()),
            "validate_settings": validate_settings,
            "gr": SimpleNamespace(Error=ValueError, Info=Mock()),
            "models": SimpleNamespace(offline=False),
            "performance_settings": SimpleNamespace(
                model_recommendations=SimpleNamespace(offline=False)
            ),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "save_theme", "exec"), scope)
        return scope["save_clicked"]

    def launch_theme(self, manager):
        source = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "launch_app"
        )
        resolver = next(
            index
            for index, node in enumerate(function.body)
            if isinstance(node, ast.ImportFrom) and node.module == "ruinedfooocus.ui.themes"
        )
        scope = {"settings": manager.default_settings}
        exec(
            compile(
                ast.Module(body=function.body[resolver : resolver + 3], type_ignores=[]),
                "launch_theme",
                "exec",
            ),
            scope,
        )
        launch = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "launch"
        )
        binding = next(keyword.value for keyword in launch.keywords if keyword.arg == "theme")
        self.assertEqual(binding.id, "theme")
        return scope["theme"], scope["theme_warning"]

    def test_real_save_reload_and_startup_consumer_for_builtin_custom_and_reset(self):
        self.cache()
        launch = SimpleNamespace(args=SimpleNamespace(settings="themes", offline=True))
        manager = SettingsManager("themes", defaults={"theme": ""})
        callback = self.save_callback(manager)
        for value in ("builtin:soft", "builtin:monochrome", "author/theme", ""):
            with (
                self.subTest(theme=value),
                patch.dict(sys.modules, {"ruinedfooocus.app.arguments": launch}),
            ):
                message = callback(value)
                reloaded = SettingsManager("themes")
                self.assertEqual(reloaded.default_settings["theme"], value)
                theme, warning = self.launch_theme(reloaded)
                self.assertFalse(warning)
                self.assertIn("restart", message)
                self.assertIn("saved theme is used on future launches", message)
                self.assertNotIn("folders and theme are pending restart", message)
                if value == "author/theme":
                    self.assertEqual(theme.body_text_color, "#123456")
                else:
                    self.assertIsInstance(theme, themes.BUILTINS[value][1])

    def test_save_rejects_new_invalid_and_uncached_ids_without_profile_changes(self):
        manager = SettingsManager("themes", defaults={"theme": ""})
        callback = self.save_callback(manager)
        original = manager.settings_path.read_bytes()
        for value in ("bad id", "author/missing"):
            with self.subTest(theme=value), self.assertRaises(ValueError):
                callback(value)
            self.assertEqual(manager.settings_path.read_bytes(), original)
            self.assertEqual(manager.default_settings["theme"], "")

    def test_existing_unavailable_legacy_choice_can_be_saved_and_resets_explicitly(self):
        manager = SettingsManager("themes", defaults={"theme": "old invalid ID"})
        callback = self.save_callback(manager)
        launch = SimpleNamespace(args=SimpleNamespace(settings="themes", offline=True))
        with patch.dict(sys.modules, {"ruinedfooocus.app.arguments": launch}):
            callback("old invalid ID")
            reloaded = SettingsManager("themes")
            self.assertEqual(reloaded.default_settings["theme"], "old invalid ID")
            with patch("builtins.print"):
                self.assertIn("Using Default", self.launch_theme(reloaded)[1])
            callback("None")
            self.assertEqual(SettingsManager("themes").default_settings["theme"], "")

    def test_production_download_callback_binds_offline_policy_and_selection(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "download_theme_clicked"
        )
        scope = {
            "gr": SimpleNamespace(update=lambda **values: values, Error=ValueError),
            "t": lambda text: text,
            "theme_choices": themes.theme_choices,
            "theme_id": themes.theme_id,
        }
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "theme_download", "exec"), scope)
        launch = SimpleNamespace(args=SimpleNamespace(offline=True))
        self.cache()
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.arguments": launch}),
            patch("ruinedfooocus.ui.themes.acquire_theme", return_value="author/theme") as acquire,
        ):
            selection, notice = scope["download_theme_clicked"]("author/theme")
        acquire.assert_called_once_with("author/theme", offline=True)
        self.assertEqual(selection["value"], "author/theme")
        self.assertIn(("author/theme", "author/theme"), selection["choices"])
        self.assertIn("selected", notice)
        self.assertNotIn("restart", notice)

    def live_callback(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "theme_changed"
        )
        scope = {
            "resolve_theme": themes.resolve_theme,
            "theme_id": themes.theme_id,
            "t": lambda text: text,
        }
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "live_theme", "exec"), scope)
        return scope["theme_changed"]

    def test_live_callback_returns_native_css_and_actual_palette_for_custom_builtin_and_reset(self):
        self.cache()
        callback = self.live_callback()
        for value in ("builtin:soft", "builtin:monochrome", "author/theme", "", " None "):
            with self.subTest(theme=value):
                notice, css, themed = callback(value)
                self.assertIn("applied to this browser", notice)
                self.assertNotIn("restart", notice)
                self.assertEqual(css, themes.resolve_theme(value)[0]._get_theme_css())
                self.assertEqual(themed, bool(themes.theme_id(value)))
                if value == "author/theme":
                    self.assertIn("--body-text-color: #123456", css)

    def test_live_callback_reports_unavailable_custom_theme_and_applies_default_palette(self):
        callback = self.live_callback()
        for value in ("author/missing", "invalid old ID"):
            with self.subTest(theme=value):
                notice, css, themed = callback(value)
                self.assertIn("Using Default", notice)
                self.assertEqual(css, themes.resolve_theme("")[0]._get_theme_css())
                self.assertFalse(themed)

    def test_live_event_passes_resolved_palette_state_to_browser(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        event = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "then"
            and isinstance(node.func.value, ast.Call)
            and ast.unparse(node.func.value.func) == "theme.change"
        )
        with gr.Blocks(analytics_enabled=False) as app:
            scope = {
                "theme": gr.Dropdown(choices=["", "builtin:soft"]),
                "theme_status": gr.Markdown(),
                "theme_css": gr.Textbox(visible=False),
                "theme_active": gr.Checkbox(value=False, visible=False),
                "theme_changed": self.live_callback(),
            }
            binding = ast.fix_missing_locations(ast.Module(body=[ast.Expr(event)], type_ignores=[]))
            exec(compile(binding, "live_event", "exec"), scope)
        self.addCleanup(app.close)
        change, applied = app.get_config_file()["dependencies"]
        self.assertEqual(
            change["outputs"],
            [scope[key]._id for key in ("theme_status", "theme_css", "theme_active")],
        )
        self.assertEqual(
            applied["inputs"], [scope[key]._id for key in ("theme_css", "theme_active")]
        )
        self.assertFalse(applied["queue"])

    def test_custom_geometry_keeps_readable_app_spacing_and_text_size(self):
        self.cache(
            schema={
                "theme": {
                    "text_md": "400px",
                    "block_radius": "100px",
                    "block_padding": "100px",
                    "input_padding": "100px",
                }
            }
        )
        theme, warning = themes.resolve_theme("author/theme")
        self.assertFalse(warning)
        self.assertEqual(theme.text_md, "0.875rem")
        self.assertEqual(theme.block_radius, ".375rem")
        self.assertEqual(theme.block_padding, ".75rem")
        self.assertEqual(theme.input_padding, ".625rem")


if __name__ == "__main__":
    unittest.main()
