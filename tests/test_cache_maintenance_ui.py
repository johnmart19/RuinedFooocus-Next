"""Maintenance custom text/headers localize without scanning or clearing caches."""

from pathlib import Path
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.i18n import I18nData

from ruinedfooocus.ui import cache_maintenance as ui


class CacheMaintenanceUiTests(unittest.TestCase):
    def test_locations_and_headers_are_plain_localized_text_and_locale_updates_keep_rows(self):
        callbacks = []

        def translated(text, locale=None):
            return ("sv: " if locale == "sv" else "") + text

        with (
            patch.object(
                ui, "cache_roots", return_value=[Path("cache/models"), Path("cache/uploads")]
            ),
            patch.object(
                ui, "maintenance_categories", side_effect=AssertionError("No scan on navigation")
            ),
            patch.object(
                ui, "clear_category", side_effect=AssertionError("No cleanup on navigation")
            ),
            patch.object(ui, "t", side_effect=lambda text: I18nData("test." + text)),
            patch.object(ui, "localize", side_effect=translated),
            patch.object(
                ui,
                "localized_output",
                side_effect=lambda fn, inputs, outputs: callbacks.append((fn, outputs)),
            ),
            gr.Blocks() as app,
        ):
            ui.create_cache_maintenance(object())
            table = next(
                component
                for component in app.blocks.values()
                if isinstance(component, gr.Dataframe)
            )
            locations = next(
                component
                for component in app.blocks.values()
                if isinstance(component, gr.Markdown)
                and str(Path("cache/models")) in str(component.value)
            )
            self.assertNotIn("__i18n__", locations.value)
            self.assertTrue(all(isinstance(label, str) for label in table.headers))
            self.assertNotIn("__i18n__", str(table.headers))
            translated_locations = next(
                fn("sv") for fn, outputs in callbacks if outputs == [locations]
            )
            self.assertIn("sv: Model and Library cache", translated_locations)
            translated_headers = next(fn("sv") for fn, outputs in callbacks if outputs == [table])
            self.assertEqual(translated_headers["headers"][0], "sv: Category")
            self.assertNotIn("value", translated_headers)


if __name__ == "__main__":
    unittest.main()
