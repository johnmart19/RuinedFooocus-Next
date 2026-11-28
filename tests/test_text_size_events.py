"""Exercise the production text-size binding without starting the application."""

import ast
import json
from pathlib import Path
import subprocess
import unittest

import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const data = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const storage = new Map([['ruinedfooocus-text-scale', '200'], ['other-preference', 'keep']]);
const styles = new Map([['--rf-user-text-scale', '200%']]);
const context = vm.createContext({
  document: {documentElement: {style: {setProperty: (key, value) => styles.set(key, value)}}},
  localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
});
const execute = (event, value) => vm.runInContext(`(${event.js.trim().replace(/;$/, '')})`, context)(value);
const dispatch = (name, value) => {
  for (const event of data.dependencies) {
    if (event.targets.some(([id, target]) => id === data.slider && target === name)) {
      execute(event, value);
    }
  }
};
const load = data.dependencies.find(event => event.targets.some(([, target]) => target === 'load'));
assert.equal(execute(load), 200, 'A saved browser size must populate the control');
// Native range/number edits emit input and change; Reset emits change/release only.
dispatch('input', 175);
dispatch('change', 175);
assert.equal(styles.get('--rf-user-text-scale'), '175%');
dispatch('change', 100);
dispatch('release', 100);
assert.equal(styles.get('--rf-user-text-scale'), '100%', 'Reset must apply the rendered size immediately');
assert.equal(storage.get('ruinedfooocus-text-scale'), '100', 'Reset must persist the same browser size');
assert.equal(execute(load), 100, 'Reload must restore the reset size');
assert.equal(storage.get('other-preference'), 'keep');
console.log('Text-size edits, native change-only Reset and browser restore passed');
"""


class TextSizeEventTests(unittest.TestCase):
    def test_native_slider_reset_applies_and_persists_the_displayed_size(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        slider = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "text_size" for target in node.targets
            )
        )
        events = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Attribute) or not isinstance(
                call.func.value, ast.Name
            ):
                continue
            receiver = call.func.value.id
            if receiver == "text_size" or (
                receiver == "app_settings"
                and call.func.attr == "load"
                and any(
                    keyword.arg == "outputs"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "text_size"
                    for keyword in call.keywords
                )
            ):
                events.append(node)
        with gr.Blocks(analytics_enabled=False) as app:
            scope = dict(
                gr=gr,
                t=lambda text: text,
                asset_path=lambda name: ROOT / "ruinedfooocus/ui/assets" / name,
                app_settings=app,
            )
            exec(
                compile(
                    ast.Module(body=[slider, *events], type_ignores=[]), "text-size events", "exec"
                ),
                scope,
            )
        self.addCleanup(app.close)
        dependencies = app.config["dependencies"]
        for event in dependencies:
            self.assertFalse(event["backend_fn"], "Text-size changes must stay browser-only")
        result = subprocess.run(
            ["node", "-e", HARNESS],
            input=json.dumps(dict(slider=scope["text_size"]._id, dependencies=dependencies)),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
