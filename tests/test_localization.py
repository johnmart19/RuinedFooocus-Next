"""CPU-only catalogue and installed native Gradio integration contracts."""

import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess
from unittest.mock import patch
from tools import check_translations

from ruinedfooocus.core.translation_manager import (
    TranslationManager,
    catalogue,
    locale_change_js,
    locale_choices,
    native_key,
    normalize_locale,
    placeholders,
    read_json,
    translate_ui,
    validate_catalogue,
)


class LocalizationTests(unittest.TestCase):
    def test_current_ui_messages_are_registered_and_new_actions_have_draft_translations(self):
        manager = TranslationManager(legacy_root=Path("no-custom-test-catalogues"))
        english = manager.translations["en"]
        for _, text, path, line in check_translations.messages():
            with self.subTest(path=path, line=line):
                self.assertIn(text, manager.aliases)
                self.assertEqual(english[manager.aliases[text]], text)
        for text in (
            "New",
            "Edit",
            "Assistant name",
            "System prompt",
            "Cancel",
            "Cancel download",
            "Unload",
            "Reasoning preferences",
            "Component role",
            "Installed alternative",
            "Select folder",
            "Add folder",
            "Remove folder",
            "Move up",
            "Move down",
            "Advanced path",
            "Host folder",
            "Open path",
            "Parent folder",
            "Subfolders",
            "Open subfolder",
            "Use folder",
            "Add custom theme",
            "Download theme",
            "Image engine",
            "Optional image engine",
        ):
            for locale in manager.manifest["locales"]:
                with self.subTest(locale=locale, text=text):
                    self.assertIn(manager.aliases[text], manager.translations[locale])

    def test_non_ui_translation_helpers_extract_without_changing_existing_ids_or_locales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = root / "ruinedfooocus/features/prompting/prompt_preview.py"
            helper.parent.mkdir(parents=True)
            helper.write_text(
                'translate_ui (\n    "Pending prompt"\n)\nlocalize("Progress {count}")\nprompt = "Unmarked prompt content"\n',
                encoding="utf-8",
            )
            ui = helper.with_name("ui_tools.py")
            ui.write_text('translate_ui("Visible caption")\n', encoding="utf-8")
            vendor = root / "ruinedfooocus/backends/vendor/example.py"
            vendor.parent.mkdir(parents=True)
            vendor.write_text('translate_ui("Vendor message")\n', encoding="utf-8")
            for relative in (
                "ruinedfooocus/core/settings_schema.py",
                "ruinedfooocus/backends/comfy/image_compute.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            cache_schema = root / "ruinedfooocus/core/cache_maintenance.py"
            cache_schema.write_text(
                'Category("internal-id", "Cached kind", (), True, "Keep source files.")\n',
                encoding="utf-8",
            )
            translations = root / "ruinedfooocus/resources/translations"
            translations.mkdir(parents=True)
            (translations / "en.json").write_text(
                '{"custom.stable_id":"Existing message"}', encoding="utf-8"
            )
            (translations / "legacy-keys.json").write_text(
                '{"Existing message":"custom.stable_id"}', encoding="utf-8"
            )
            (translations / "manifest.json").write_text(
                json.dumps(
                    {
                        "locales": {
                            "en": {"name": "English", "review": "source"},
                            "fr": {"name": "French", "review": "draft"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            french = translations / "fr.json"
            french.write_text('{"custom.stable_id":"Message conservé"}', encoding="utf-8")
            original = french.read_bytes()

            def manager(**kwargs):
                return TranslationManager(root=translations, **kwargs)

            with patch.object(check_translations, "ROOT", root):
                self.assertEqual(check_translations.localized_files(), [helper])
                extracted = [text for _, text, _, _ in check_translations.messages()]
                self.assertCountEqual(
                    extracted,
                    [
                        "Pending prompt",
                        "Progress {count}",
                        "Visible caption",
                        "Cached kind",
                        "Keep source files.",
                    ],
                )
                with (
                    patch.object(check_translations, "TranslationManager", side_effect=manager),
                    patch("sys.argv", ["check_translations.py", "--extract"]),
                ):
                    check_translations.main()
            aliases = read_json(translations / "legacy-keys.json")
            self.assertEqual(aliases["Existing message"], "custom.stable_id")
            self.assertEqual(aliases["Pending prompt"], "prompting.pending_prompt")
            self.assertEqual(aliases["Cached kind"], "settings.cached_kind")
            self.assertEqual(aliases["Keep source files."], "settings.keep_source_files")
            self.assertEqual(french.read_bytes(), original)
            self.assertEqual(
                manager(legacy_root=root / "absent").translate("Pending prompt", locale="fr"),
                "Pending prompt",
            )

    def test_shipped_catalogues_have_valid_english_fallback_and_declared_review(self):
        manager = TranslationManager(legacy_root=Path("no-custom-test-catalogues"))
        self.assertEqual(manager.errors, [])
        english = manager.translations["en"]
        self.assertGreater(len(english), 500)
        for locale in (
            "sv",
            "de",
            "fr",
            "es",
            "pt-BR",
            "uk",
            "ru",
            "ja",
            "zh-CN",
            "zh-TW",
            "ko",
            "it",
            "pl",
            "nl",
            "tr",
        ):
            self.assertEqual(manager.manifest["locales"][locale]["review"], "draft")
            self.assertNotEqual(manager.translate("Generate", locale=locale), "Generate")
            self.assertEqual(
                set(manager.native().translations_dict[locale]),
                {native_key(key) for key in english} | {"common.complete"},
            )
            for text in (
                "Create",
                "Chat",
                "Library",
                "Settings",
                "Appearance & language",
                "Profiles & storage",
                "Hardware & backends",
                "Image runtime",
                "Generation defaults",
                "Local Chat",
                "Connections",
                "Maintenance",
                "Language for this browser",
                "Default language for this profile",
                "Random Prompt — quick",
                "One Button Prompt — configurable",
                "Preview Random Prompt",
                "Generate reviewed Random Prompt preview",
                "Preview One Button prompt",
                "Generate reviewed One Button preview",
                "Prompt variations",
                "Preview variations",
                "Generate reviewed candidates",
                "Use prompt",
                "Reuse full setup",
            ):
                with self.subTest(locale=locale, message=text):
                    self.assertIn(manager.aliases[text], manager.translations[locale])
        self.assertEqual(
            manager.translate("Prompt content: 猫 {literal}", locale="ja"),
            "Prompt content: 猫 {literal}",
        )

    def test_duplicate_ids_types_obsolete_and_placeholders_fail_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.json"
            path.write_text('{"same":"one", "same":"two"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_json(path)
        for invalid in (["text"], {"label": 1}, {1: "text"}):
            with self.assertRaises(ValueError):
                validate_catalogue(invalid)
        with self.assertRaisesRegex(ValueError, "Obsolete"):
            validate_catalogue({"other": "text"}, {"key": "text"})
        with self.assertRaisesRegex(ValueError, "placeholders"):
            validate_catalogue({"key": "Hi {wrong}"}, {"key": "Hi {name}"})
        for changed in ("Hi {name!r}", "Hi {name:4}"):
            with self.assertRaisesRegex(ValueError, "placeholders"):
                validate_catalogue({"key": changed}, {"key": "Hi {name}"})
        self.assertEqual(placeholders("{{literal}} {name}"), {"name"})

    def test_malformed_locale_and_missing_catalogue_fall_back_without_content_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "locales": {
                            "en": {"name": "English", "review": "source"},
                            "sv": {"name": "Svenska", "review": "draft"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "legacy-keys.json").write_text('{"Hello": "core.hello"}', encoding="utf-8")
            (root / "en.json").write_text('{"core.hello":"Hello"}', encoding="utf-8")
            (root / "sv.json").write_text('{"core.hello":7}', encoding="utf-8")
            manager = TranslationManager(root, root / "absent")
            self.assertTrue(manager.errors)
            self.assertEqual(manager.translate("Hello", locale="sv"), "Hello")
            (root / "manifest.json").write_text("[]", encoding="utf-8")
            fallback = TranslationManager(root, root / "absent")
            self.assertTrue(fallback.errors)
            self.assertEqual(fallback.translate("core.hello", locale="sv"), "Hello")
            (root / "manifest.json").write_text('{"locales":{"en":{}}}', encoding="utf-8")
            invalid_metadata = TranslationManager(root, root / "absent")
            self.assertTrue(invalid_metadata.errors)
            self.assertEqual(invalid_metadata.manifest["locales"]["en"]["name"], "English")
            self.assertEqual(
                TranslationManager(root / "missing").translate("untouched"), "untouched"
            )

    def test_legacy_custom_catalogue_is_read_only_and_instance_state_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sv.json"
            original = '{"Generate": "Egen text"}'
            path.write_text(original, encoding="utf-8")
            first = TranslationManager(legacy_root=tmp)
            second = TranslationManager(legacy_root=Path(tmp) / "absent")
            first.set_language("sv")
            self.assertEqual(first.translate("Generate"), "Egen text")
            self.assertEqual(second.translate("Generate"), "Generate")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(second.language, "en")
            path.write_text('{"Generate":"one", "common.generate":"two"}', encoding="utf-8")
            self.assertTrue(
                any(
                    "Duplicate legacy" in error
                    for error in TranslationManager(legacy_root=tmp).errors
                )
            )

    def test_dynamic_values_can_be_escaped_and_missing_values_remain_readable(self):
        manager = TranslationManager()
        self.assertEqual(
            manager.translate("Hi {name}", {"name": '<script>"&'}, escape=True),
            "Hi &lt;script&gt;&quot;&amp;",
        )
        self.assertEqual(manager.translate("Hi {name}", {}), "Hi {name}")
        self.assertEqual(manager.translate("Count {count:d}", {"count": 2}, escape=True), "Count 2")

    def test_former_packaged_custom_directory_is_read_only_and_writable_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "language"
            writable = Path(tmp) / "writable"
            root = Path(tmp) / "translations"
            old.mkdir()
            writable.mkdir()
            root.mkdir()
            for name in ("en.json", "manifest.json", "legacy-keys.json"):
                shutil.copyfile(catalogue().root / name, root / name)
            (old / "sv.json").write_text('{"Generate":"Old custom"}', encoding="utf-8")
            (writable / "sv.json").write_text('{"Generate":"Writable custom"}', encoding="utf-8")
            originals = {
                path: path.read_bytes() for path in (old / "sv.json", writable / "sv.json")
            }
            with (
                patch("ruinedfooocus.core.translation_manager.data_path", return_value=writable),
            ):
                manager = TranslationManager(root=root)
            self.assertEqual(manager.translate("Generate", locale="sv"), "Writable custom")
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_native_component_properties_retain_markers_and_protocol_values(self):
        import gradio as gr
        from gradio.i18n import I18nData

        marker = translate_ui("Generate")
        self.assertIsInstance(marker, I18nData)
        self.assertEqual(marker.key, "ruinedfooocus.common.generate")
        with gr.Blocks() as demo:
            draft = gr.Textbox(
                label=translate_ui("Prompt"),
                placeholder=translate_ui("Type prompt here."),
                value="猫 {literal} --seed 0",
            )
            model = gr.Dropdown(
                label=translate_ui("Model"),
                choices=[("display", "model-id.safetensors")],
                value="model-id.safetensors",
            )
            gr.Button(marker)
            gr.Markdown(translate_ui("Workflow settings"))
            with gr.Tab(translate_ui("Settings"), id="settings"):
                gr.Textbox(info=translate_ui("Optional"))
        config = demo.get_config_file()
        by_id = {item["id"]: item for item in config["components"]}
        self.assertEqual(by_id[draft._id]["props"]["value"], "猫 {literal} --seed 0")
        self.assertEqual(by_id[model._id]["props"]["value"], "model-id.safetensors")
        self.assertIn("__i18n__", str(by_id[draft._id]["props"]["label"]))

    def test_adapter_uses_installed_native_export_and_preserves_browser_state(self):
        import gradio

        script = locale_change_js("sv", initialize=True)
        self.assertIn("native.changeLocale(locale)", script)
        self.assertNotIn("location.reload", script)
        self.assertNotIn("__CORE_ASSET__", script)
        assets = Path(gradio.__file__).parent / "templates/frontend/assets"
        core = next(
            p
            for p in assets.glob("core-*.js")
            if " as changeLocale," in p.read_text(encoding="utf-8")
        )
        self.assertIn(core.name, script)
        routes = (Path(gradio.__file__).parent / "routes.py").read_text(encoding="utf-8")
        self.assertIn('@app.get("/assets/{path:path}")', routes)
        self.assertNotIn("set_language", script)
        self.assertEqual(normalize_locale("pt_br"), "pt-BR")
        self.assertEqual(normalize_locale("zh_cn"), "zh-CN")
        self.assertEqual(normalize_locale("zh_tw"), "zh-TW")
        self.assertEqual(normalize_locale("unknown"), "en")
        self.assertEqual(
            [v for _, v in locale_choices()],
            [
                "auto",
                "en",
                "sv",
                "de",
                "fr",
                "es",
                "pt-BR",
                "uk",
                "ru",
                "ja",
                "zh-CN",
                "zh-TW",
                "ko",
                "it",
                "pl",
                "nl",
                "tr",
            ],
        )

    @unittest.skipUnless(
        shutil.which("node"), "Node is needed to execute the browser adapter contract"
    )
    def test_adapter_executes_with_independent_client_storage_and_state(self):
        subprocess.run(
            [shutil.which("node"), str(Path(__file__).with_name("test_locale_adapter.js"))],
            check=True,
            timeout=15,
        )

    def test_installed_native_adapter_asset_is_served_by_gradio(self):
        import gradio as gr
        from gradio.routes import App
        from fastapi.testclient import TestClient

        script = locale_change_js()
        assets = Path(gr.__file__).parent / "templates/frontend/assets"
        core = next(path for path in assets.glob("core-*.js") if path.name in script)
        with gr.Blocks(analytics_enabled=False) as demo:
            gr.Button(translate_ui("Generate"))
        demo.i18n_instance = catalogue().native()
        client = TestClient(App.create_app(demo))
        response = client.get(f"/assets/{core.name}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, core.read_bytes())
        self.assertIn(" as changeLocale,", response.text)
        config = client.get("/config").json()
        self.assertEqual(
            config["i18n_translations"]["sv"]["ruinedfooocus.common.generate"], "Generera"
        )
        self.assertNotIn("common.settings", config["i18n_translations"]["en"])
        for locale, messages in config["i18n_translations"].items():
            with self.subTest(locale=locale):
                self.assertEqual(messages["common.complete"], "Complete")
                self.assertEqual(
                    {key for key in messages if not key.startswith("ruinedfooocus.")},
                    {"common.complete"},
                )
        self.assertEqual(
            config["i18n_translations"]["en"]["ruinedfooocus.common.settings"], "Settings"
        )


if __name__ == "__main__":
    unittest.main()
