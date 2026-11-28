"""Exercise catalogue services using their real bundled templates and images."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ruinedfooocus.features.chat.assistants import assistant_info
from ui_test_helpers import function_from


class ChatResourcesTests(unittest.TestCase):
    def test_code_blocks_remain_plain_chat_content_with_html_sanitization(self):
        display = function_from(
            "ruinedfooocus/features/chat/ui.py", "display_history", {"re": __import__("re")}
        )
        content = '```python\nprint("<script>alert(1)</script>")\n```'
        self.assertEqual(
            display([{"role": "assistant", "content": content}], False),
            [{"role": "assistant", "content": content}],
        )
        source = (
            Path(__file__).resolve().parents[1] / "ruinedfooocus/features/chat/ui.py"
        ).read_text(encoding="utf-8")
        self.assertIn("sanitize_html=True", source)

    def test_runner_controls_routes_and_assets_are_retired(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "ruinedfooocus/features/chat/chat_code.py",
            "ruinedfooocus/features/chat/browser_python.py",
            "ruinedfooocus/ui/assets/chat_python.html",
            "ruinedfooocus/ui/assets/chat_python_theme.js",
        ):
            self.assertFalse((root / relative).exists(), relative)
        for relative in ("ruinedfooocus/features/chat/ui.py", "ruinedfooocus/app/api.py"):
            source = (root / relative).read_text(encoding="utf-8")
            for obsolete in ("python_runner", "browser_python", "chat_python", "enable_python"):
                self.assertNotIn(obsolete, source)

    def test_general_and_unavailable_assistants_have_decodable_avatars(self):
        with tempfile.TemporaryDirectory() as directory:
            for selected in ("__normal__", str(Path(directory) / "missing.png")):
                with self.subTest(selected=selected):
                    info = assistant_info(selected)
                    with Image.open(info["avatar"]) as image:
                        image.verify()
