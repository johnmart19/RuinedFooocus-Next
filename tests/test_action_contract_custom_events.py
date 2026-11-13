"""Custom HTML event receipts retain source ownership without importing the app."""

from pathlib import Path
import tempfile
import unittest

from tools.audit_action_contracts import inventory


class CustomActionContractTests(unittest.TestCase):
    def fixture(self, files):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for relative, source in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return inventory(root)

    def test_settings_navigation_and_model_pagination_keep_lexical_source_receipts(self):
        document = self.fixture(
            {
                "ruinedfooocus/ui/settings_navigation.py": (
                    "import gradio as gr\n"
                    "def bind_navigation():\n"
                    "    def navigate_setting(event):\n        return event\n"
                    "    results.navigate(navigate_setting, inputs=[], outputs=panels, queue=False)\n"
                ),
                "ruinedfooocus/features/models/ui_acquisition.py": (
                    "import gradio as gr\n"
                    "def create_ui():\n"
                    "    def render_requested_page(rows):\n        return rows\n"
                    "    options = dict(inputs=[records, selection, version], outputs=[cards, page], queue=False, show_progress='hidden')\n"
                    "    cards.paginate(render_requested_page, **options)\n"
                ),
            }
        )
        rows = {row["binding"]: row for row in document["actions"]}
        self.assertEqual(set(rows), {"results.navigate", "cards.paginate"})
        self.assertEqual(
            rows["results.navigate"]["callback_qualified"], "bind_navigation.navigate_setting"
        )
        page = rows["cards.paginate"]
        self.assertEqual(page["callback_qualified"], "create_ui.render_requested_page")
        self.assertEqual(page["inputs"], "[records, selection, version]")
        self.assertEqual(page["outputs"], "[cards, page]")
        self.assertEqual(page["progress"], "'hidden'")
        self.assertFalse(document["unassigned"])
        for row in rows.values():
            self.assertEqual(row["kind"], "event-site")
            self.assertEqual(row["callback_resolution"], "lexical")
            self.assertEqual(row["queue"], "False")
            self.assertEqual(len(row["callback_sha256"]), 64)
            self.assertEqual(len(document["source_sha256"][row["owner"]]), 64)

    def test_client_only_navigation_is_recorded_but_ordinary_same_named_methods_are_not(self):
        document = self.fixture(
            {
                "ruinedfooocus/ui/settings_navigation.py": (
                    "import gradio as gr\n"
                    "navigator.navigate('settings')\n"
                    "service.paginate(items)\n"
                    "results.navigate(fn=None, js='() => []', outputs=panels, queue=False)\n"
                )
            }
        )
        self.assertEqual([row["binding"] for row in document["actions"]], ["results.navigate"])
        self.assertEqual(
            {row["call"] for row in document["excluded_method_calls"]},
            {"navigator.navigate", "service.paginate"},
        )
        self.assertEqual(document["actions"][0]["callback"], "None")


if __name__ == "__main__":
    unittest.main()
