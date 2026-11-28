"""Independent status/runtime/file lanes retain their actual source owners."""

from pathlib import Path
import tempfile
import unittest

from tools.audit_action_contracts import inventory


class ActionOwnerBoundaryTests(unittest.TestCase):
    def fixture(self, files):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for relative, source in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return inventory(root)

    def test_gradio_apis_and_imperative_media_routes_keep_handler_identity(self):
        document = self.fixture(
            {
                "ruinedfooocus/app/api.py": (
                    "import gradio as gr\nfrom fastapi import FastAPI\n"
                    "def get_last_image():\n    return 'owned.png'\n"
                    "def last_image(request):\n    return 'display bytes'\n"
                    "def add_api():\n"
                    "    def api_llama(system, user):\n        return user\n"
                    "    gr.api(api_llama, api_name='llama')\n"
                    "    @gr.api(api_name='api_version')\n"
                    "    def api_version():\n        return 'fixture'\n"
                    "    gr.api(fn=get_last_image, api_name='last_image')\n"
                    "shared.server_app.add_api_route('/last_image', last_image, methods=['GET'])\n"
                    "ordinary.api(get_last_image)\n"
                    "ordinary.add_api_route('not a registration')\n"
                ),
                "ruinedfooocus/integrations/openai_api.py": (
                    "from fastapi import APIRouter\n"
                    "async def image_file(token):\n    return token\n"
                    "media.add_api_route(path='/images/{token}', endpoint=image_file, methods=['GET'])\n"
                ),
                "ruinedfooocus/ordinary.py": (
                    "gr.api(handler, api_name='not_gradio')\n"
                    "service.add_api_route('/not_fastapi', handler)\n"
                ),
            }
        )
        self.assertEqual(document["schema_version"], 3)
        self.assertEqual(len(document["actions"]), 5)
        self.assertFalse(document["unassigned"])
        apis = [row for row in document["actions"] if row["kind"] == "gradio-api"]
        routes = [row for row in document["actions"] if row["kind"] == "route"]
        self.assertEqual(len(apis), 3)
        self.assertEqual(len(routes), 2)
        self.assertEqual(
            {row["api_name"] for row in apis}, {"'llama'", "'api_version'", "'last_image'"}
        )
        for row in apis:
            self.assertEqual(row["callback_resolution"], "lexical")
            self.assertEqual(row["queue"], "Gradio API default")
            self.assertIsNone(row["route"])
        self.assertEqual({row["route"] for row in routes}, {"'/last_image'", "'/images/{token}'"})
        self.assertEqual({row["callback"] for row in routes}, {"last_image", "image_file"})
        for row in routes:
            self.assertEqual(row["group"], "image-media-delivery")
            self.assertEqual(row["http_methods"], "['GET']")
            self.assertEqual(row["queue"], "HTTP request dispatcher")
            self.assertEqual(row["callback_resolution"], "lexical")

    def test_private_activity_actions_are_assigned_without_compute_or_disk_mutation(self):
        document = self.fixture(
            {
                "ruinedfooocus/ui/api_activity.py": (
                    "import gradio as gr\n"
                    "def create_ui():\n"
                    "    def show_activity(request):\n        return snapshot()\n"
                    "    def select_activity(request, event):\n        return snapshot()\n"
                    "    options = dict(outputs=[summary, table], queue=False, show_progress='hidden')\n"
                    "    refresh.click(show_activity, **options)\n"
                    "    tab.select(select_activity, **options)\n"
                )
            }
        )
        self.assertEqual(len(document["actions"]), 2)
        self.assertFalse(document["unassigned"])
        for row in document["actions"]:
            self.assertEqual(row["group"], "private-api-activity")
            self.assertEqual(row["queue"], "False")
            self.assertEqual(row["progress"], "'hidden'")
            self.assertEqual(row["callback_resolution"], "lexical")
            self.assertIn("ApiActivity", row["resource_owner"])
            self.assertIn("No disk", row["persistence"])

    def test_specific_runtime_and_file_owners_precede_generic_ui_patterns(self):
        document = self.fixture(
            {
                "ruinedfooocus/features/chat/ui_runtime_info.py": (
                    "import gradio as gr\n"
                    "def inspect(item):\n    return item.refresh()\n"
                    "def stop(item):\n    item.cancel()\n"
                    "refresh_event = refresh.click(inspect, inputs=session, outputs=details, "
                    "concurrency_id='chat-runtime-inspection', concurrency_limit=2)\n"
                    "cancel.click(stop, inputs=session, outputs=status, queue=False, cancels=[refresh_event])\n"
                ),
                "ruinedfooocus/features/image_processing/ui_native_formats.py": (
                    "import gradio as gr\n"
                    "export.click(lambda record: export_selection(record), inputs=selected, "
                    "outputs=download, concurrency_id='native-image-exports', concurrency_limit=1)\n"
                    "import_button.click(lambda path: import_selection(path), inputs=upload, "
                    "outputs=status, concurrency_id='native-image-exports', concurrency_limit=1)\n"
                ),
            }
        )
        self.assertEqual(len(document["actions"]), 4)
        self.assertFalse(document["unassigned"])
        runtime = [row for row in document["actions"] if row["group"] == "chat-runtime-inspection"]
        files = [
            row for row in document["actions"] if row["group"] == "native-image-file-transport"
        ]
        self.assertEqual(len(runtime), 2)
        self.assertEqual(len(files), 2)
        cancel = next(row for row in runtime if row["callback"] == "stop")
        self.assertEqual(cancel["queue"], "False")
        self.assertEqual(cancel["cancels"], "[refresh_event]")
        inspect = next(row for row in runtime if row["callback"] == "inspect")
        self.assertEqual(inspect["concurrency_limit"], "2")
        for row in files:
            self.assertEqual(row["concurrency_id"], "'native-image-exports'")
            self.assertEqual(row["concurrency_limit"], "1")
            self.assertIn("without clobbering", row["persistence"])


if __name__ == "__main__":
    unittest.main()
