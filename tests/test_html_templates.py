from ruinedfooocus.core.paths import asset_path, resource_path

"""App-owned HTML follows the HTML Living Standard basics; no server or browser needed."""
import ast
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGES = (str(asset_path("slideshow.html")),)
FRAGMENTS = (str(asset_path("chat_local_models.html")), str(asset_path("chat_model_remove.html")))
OBSOLETE = {"font", "center", "big", "strike", "marquee", "tt", "frame", "frameset"}
PRESENTATIONAL = {"bgcolor", "align", "border", "cellpadding", "cellspacing", "valign"}


class Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.doctype = None

    def handle_decl(self, decl):
        self.doctype = decl

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag


def parse(path):
    collector = Collector()
    collector.feed((ROOT / path).read_text(encoding="utf-8"))
    return collector


class HtmlTemplateTests(unittest.TestCase):
    def test_control_help_observer_limits_work_to_native_hint_and_label_changes(self):
        result = subprocess.run(
            ["node", "tests/test_control_help.js"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_shell_initializes_measured_image_fit_without_changing_shortcuts(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/html.py").read_text(encoding="utf-8"))
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "scripts"
            or isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "scripts"
        ]
        scope = {"asset_path": lambda name: ROOT / "ruinedfooocus/ui/assets" / name}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "actual shell", "exec"), scope)
        result = subprocess.run(
            ["node", "tests/test_image_layout.js"],
            cwd=ROOT,
            input=json.dumps(scope["scripts"]),
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shortcuts_leave_escape_to_widgets_and_only_run_in_create(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/html.py").read_text(encoding="utf-8"))
        script = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign) and node.targets[0].id == "scripts"
        )
        harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const script = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const handlers = [], clicks = [];
let createVisible = true;
const document = {
  addEventListener(type, fn, options) { if (type === 'keydown' && !options) handlers.push(fn); },
  getElementById(id) { return { click: () => clicks.push(id), getClientRects: () => createVisible ? [1] : [] }; },
  querySelector() { return null; },
};
vm.runInNewContext(script, {
  document, localStorage: { getItem: () => null }, window: { __rfRestartWatch: true }
});
assert.equal(handlers.length, 1);
function press(key, ctrlKey = false, defaultPrevented = false) {
  let prevented = false;
  handlers[0]({ key, ctrlKey, defaultPrevented, preventDefault() { prevented = true; } });
  return prevented;
}
assert.equal(press('Escape'), false);
assert.deepEqual(clicks, []);
assert.equal(press('q', true), true);
assert.equal(press('Enter', true), true);
assert.deepEqual(clicks, ['edit_mode', 'generate']);
createVisible = false;
assert.equal(press('Enter', true), false);
assert.equal(press('q', true), false);
createVisible = true;
assert.equal(press('Enter', true, true), false);
assert.deepEqual(clicks, ['edit_mode', 'generate']);
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(script),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pages_declare_doctype_language_charset_and_title(self):
        for path in PAGES:
            with self.subTest(page=path):
                page = parse(path)
                self.assertEqual(page.doctype, "DOCTYPE html")
                self.assertEqual(page.tags[0], ("html", {"lang": "en"}))
                self.assertEqual(page.tags[1][0], "head")
                self.assertEqual(page.tags[2], ("meta", {"charset": "utf-8"}))
                self.assertIn("title", [tag for tag, _ in page.tags])

    def test_markup_has_no_obsolete_elements_or_presentational_attributes(self):
        for path in PAGES + FRAGMENTS:
            text = (ROOT / path).read_text(encoding="utf-8")
            with self.subTest(file=path):
                self.assertNotRegex(text, r"<[^>]*\s/>")  # void elements without trailing slashes
                for tag, attrs in parse(path).tags:
                    self.assertNotIn(tag, OBSOLETE)
                    self.assertFalse(PRESENTATIONAL & set(attrs), (tag, attrs))
                    if tag == "img":
                        self.assertIn("alt", attrs)
                    if tag == "button":
                        self.assertEqual(attrs.get("type"), "button")

    def test_python_built_markup_avoids_obsolete_tags(self):
        for path in (
            "ruinedfooocus/ui/app.py",
            "ruinedfooocus/features/prompting/ui_random.py",
            "ruinedfooocus/ui/html.py",
        ):
            with self.subTest(file=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                self.assertNotRegex(text, r"<(?:font|center|big|strike|marquee)\b")
                for image in re.findall(r"<img\b[^>]*>", text):
                    self.assertIn("alt=", image)

    def test_progress_bar_is_labelled_and_escapes_status_text(self):
        source = (ROOT / "ruinedfooocus/ui/html.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and node.targets[0].id == "progress_html"
            or isinstance(node, ast.FunctionDef)
            and node.name == "make_progress_html"
        ]
        import html as html_text

        scope = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            "state": {"last_progress": 0},
            "html_text": html_text,
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "ruinedfooocus/ui/html.py", "exec"),
            scope,
        )
        markup = scope["make_progress_html"](40, 'Loading <model> "a&b"')
        self.assertIn("&lt;model&gt; &quot;a&amp;b&quot;", markup)
        tags = [tag for tag, _ in parse_text(markup).tags]
        self.assertEqual(tags[0], "label")
        self.assertIn("progress", tags)
        self.assertNotIn("div", tags)  # phrasing content only inside <label>


def parse_text(text):
    collector = Collector()
    collector.feed(text)
    return collector


if __name__ == "__main__":
    unittest.main()
