"""Where the application may write: prompts and other session content stay disposable.

Static checks of the source. They keep new storage locations and telemetry
switches under review; they do not observe a running application.
"""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
# Top-level folders of the data root, and what each may hold.
LOCATIONS = {
    "cache": "disposable: uploads, editor sessions, gallery index, thumbnails, logs",
    "outputs": "generated images with their own metadata",
    "settings": "configuration only, never prompt or chat text",
    "models": "model files",
    "runtime": "downloaded or built runtimes",
    "assistants": "assistant definitions the user creates deliberately",
    "random_prompt": "wildcard and list files the user provides",
    "language": "interface language choice",
    "reinstall": "installer marker",
    "reinstalltorch": "installer marker",
}


class SessionContentLocationTests(unittest.TestCase):
    def sources(self):
        for path in sorted((ROOT / "ruinedfooocus").rglob("*.py")):
            yield path, path.read_text(encoding="utf-8-sig")

    def test_every_data_folder_the_code_names_is_a_reviewed_location(self):
        found = {}
        for path, text in self.sources():
            for name in re.findall(r"data_path\(\s*[\"']([A-Za-z_.-]+)[\"']", text):
                found.setdefault(name, path.relative_to(ROOT).as_posix())
        unknown = {name: where for name, where in found.items() if name not in LOCATIONS}
        self.assertEqual(
            unknown,
            {},
            "A new top-level data folder: decide whether it can hold prompts, chat or "
            "images, and keep such content under cache or outputs.",
        )
        self.assertTrue({"cache", "outputs", "settings"} <= set(found))

    def test_interface_files_go_to_the_data_cache_and_telemetry_is_off_before_imports(self):
        bootstrap = (ROOT / "ruinedfooocus/app/bootstrap.py").read_text(encoding="utf-8")
        self.assertIn('gradio_cache = str(data_path("cache", "uploads"))', bootstrap)
        self.assertIn('os.environ["GRADIO_TEMP_DIR"] = gradio_cache', bootstrap)
        first_gradio = min(
            (
                bootstrap.index(marker)
                for marker in ("import gradio", "from gradio")
                if marker in bootstrap
            ),
            default=len(bootstrap),
        )
        for switch in (
            'os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"',
            'os.environ["DO_NOT_TRACK"] = "1"',
            'os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"',
        ):
            self.assertIn(switch, bootstrap)
            self.assertLess(bootstrap.index(switch), first_gradio, switch)

    def test_the_browser_keeps_only_interface_preferences(self):
        allowed = {"ruinedfooocus-ui-language", "ruinedfooocus-text-scale"}
        keys = set()
        for path in sorted((ROOT / "ruinedfooocus").rglob("*")):
            if path.suffix not in (".py", ".js", ".html") or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            keys.update(
                re.findall(r"localStorage\.(?:setItem|getItem)\(\s*[\"']([^\"']+)[\"']", text)
            )
            self.assertNotIn("sessionStorage.setItem", text, path.name)
            self.assertNotIn("indexedDB", text, path.name)
        self.assertEqual(keys - allowed, set())

    def test_saved_defaults_never_contain_prompt_text(self):
        from ruinedfooocus.features.generation.defaults import PROMPT_KEYS, generation_defaults

        saved = generation_defaults(
            {"prompt": "a private idea", "negative": "something unwanted", "seed": 7}
        )
        self.assertEqual(PROMPT_KEYS, ("prompt", "negative_prompt"))
        self.assertEqual([saved[key] for key in PROMPT_KEYS], ["", ""])
        self.assertNotIn("private idea", repr(saved))
        self.assertNotIn("unwanted", repr(saved))


if __name__ == "__main__":
    unittest.main()
