"""Regressions from the November 2026 interface audit, without starting the app."""

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.gallery import index_store
from ruinedfooocus.ui import html
from ruinedfooocus.ui.settings_navigation import source_text


class SettingsSearchLabelTests(unittest.TestCase):
    def test_button_captions_resolve_from_serialized_translation_metadata(self):
        label = translate_ui("Download theme")
        self.assertEqual(source_text(label), "Download theme")
        with gr.Blocks():
            button = gr.Button(label)
        # Buttons keep the caption as the serialized string, not the metadata object.
        self.assertEqual(source_text(button.value), "Download theme")
        self.assertEqual(source_text("__i18n__not json"), "")
        self.assertEqual(source_text("Plain"), "Plain")
        self.assertEqual(source_text(None), "")


class LivePreviewTests(unittest.TestCase):
    def test_only_the_worker_preview_names_are_recognised(self):
        for name in (
            "preview.jpg",
            "preview-431eeb7889124d09beb694a76746288a.jpg",
            "PREVIEW-x.JPEG",
        ):
            self.assertTrue(index_store.LIVE_PREVIEW.fullmatch(name), name)
        for name in ("preview.png", "my preview.jpg", "preview-a.jpg.png", "previews.jpg"):
            self.assertFalse(index_store.LIVE_PREVIEW.fullmatch(name), name)

    def test_a_finished_job_removes_its_preview_and_day_old_ones_but_keeps_results(self):
        _discard_previews = index_store.discard_live_previews
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current, recent, stale, grid, other = (
                root / "preview-current.jpg",
                root / "preview-recent.jpg",
                root / "preview-stale.jpg",
                root / "preview-grid.jpg",
                root / "result.jpg",
            )
            for path in (current, recent, stale, grid, other):
                path.write_bytes(b"x")
            old = time.time() - 2 * 86400
            os.utime(stale, (old, old))
            os.utime(other, (old, old))
            _discard_previews(str(current), keep=[str(other)])
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["preview-grid.jpg", "preview-recent.jpg", "result.jpg"],
            )
            # A preview grid returned as a result outlives its own job.
            _discard_previews(str(grid), keep=[str(grid), str(other)])
            self.assertTrue(grid.exists())
            _discard_previews(str(root / "missing" / "preview-x.jpg"))


class RewritePreviewTests(unittest.TestCase):
    def handler(self):
        from ruinedfooocus.features.prompting import ui_rewrite

        with gr.Blocks() as demo:
            prompt = gr.Textbox()
            ui_rewrite.add_rewrite_controls(prompt)
        return next(
            fn.fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "suggest"
        )

    def test_an_unchanged_rewrite_is_reported_instead_of_offered(self):
        suggest = self.handler()
        target = "ruinedfooocus.features.prompting.ui_rewrite.run_prompt"
        with (
            patch(target, return_value="a red fox "),
            self.assertRaisesRegex(gr.Error, "unchanged"),
        ):
            suggest("preset.txt", "a red fox")
        with patch(target, return_value="a red fox, detailed fur"):
            preview, original, apply = suggest("preset.txt", "a red fox")
        self.assertEqual(preview["value"], "a red fox, detailed fur")
        self.assertEqual(original, "a red fox")
        self.assertTrue(apply["interactive"])


class StartupLanguageTests(unittest.TestCase):
    def test_the_head_applies_the_saved_language_without_waiting_for_load(self):
        head = html.startup_head("1", themed=False, language="ru")
        script = head[head.rindex("<script>") :]
        self.assertIn("window.gradio_config?.i18n_translations", script)
        self.assertIn("ruinedfooocus-ui-language", script)
        self.assertIn('"ru"', script)
        self.assertIn("changeLocale", script)
        # The identity script stays first: other code reads it from the first tag.
        self.assertIn("window.__rfAppId", head[head.index("<script>") : head.index("</script>")])


class ModelDefaultsTableTests(unittest.TestCase):
    def test_narrow_panel_tables_keep_short_cells_on_one_line(self):
        from ruinedfooocus.core.paths import asset_path

        css = asset_path("ui.css").read_text(encoding="utf-8")
        rule = css[css.index("#generation-controls .prose table td:not(:last-child)") :]
        self.assertIn("white-space: nowrap", rule[: rule.index("}")])


class SettingsSearchRankingTests(unittest.TestCase):
    def test_section_keywords_stay_apart_from_a_field_s_own_words(self):
        from ruinedfooocus.ui.settings_navigation import navigation_records

        with gr.Blocks() as demo:
            with gr.Tabs():
                with gr.Tab("Appearance"):
                    gr.Textbox(label="Theme", info="Colours of the interface")
                    gr.Number(label="Maximum batch size")
        records = {
            record["label"]: record
            for record in navigation_records(demo, {"Appearance": "theme language"})
        }
        self.assertEqual(records["Maximum batch size"]["section_terms"], "theme language")
        self.assertNotIn("theme", records["Maximum batch size"]["terms"])
        self.assertIn("Colours", records["Theme"]["terms"])

    def test_the_result_list_prefers_field_matches_over_section_matches(self):
        from ruinedfooocus.core.paths import asset_path

        script = asset_path("settings_search.js").read_text(encoding="utf-8")
        own = script.index("const own = ")
        self.assertIn("${record.label} ${record.terms}", script[own : script.index(";", own)])
        self.assertIn("own.length", script)
        self.assertIn("record.section_terms", script)


class DownloadSelectionPlacementTests(unittest.TestCase):
    def test_the_selection_and_its_actions_are_built_before_the_card_grid(self):
        source = (
            Path(__file__).resolve().parents[1] / "ruinedfooocus/features/models/ui_acquisition.py"
        ).read_text(encoding="utf-8")
        cards = source.index('elem_id="acquisition-cards"')
        self.assertLess(
            source.index('status = gr.Markdown(t("Select a model or component."))'), cards
        )
        self.assertLess(source.index('t("Download selected files")'), cards)
        self.assertLess(source.index('t("Use installed checkpoint in Create")'), cards)


class LeaveImageToolTests(unittest.TestCase):
    def test_a_press_on_the_highlighted_generate_action_registers_as_a_choice(self):
        workspace = (
            Path(__file__).resolve().parents[1] / "ruinedfooocus/features/gallery/ui_workspace.py"
        ).read_text(encoding="utf-8")
        self.assertIn('elem_id="editor-image-action"', workspace)
        hook = html.scripts[html.scripts.index("#editor-image-action label") :]
        hook = hook[: hook.index("{capture: true}")]
        self.assertIn("input.value === 'generate'", hook)
        self.assertIn("input.checked = false", hook)
        self.assertIn(
            "pointerdown", html.scripts[: html.scripts.index("#editor-image-action label")][-120:]
        )


if __name__ == "__main__":
    unittest.main()
