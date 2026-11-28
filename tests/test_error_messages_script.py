"""Messages from events that skip Gradio's queue are shown by the page script."""

import json
import subprocess
import unittest

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.ui import html


HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
function element(tag) {
  return {tag, children: [], attributes: {}, listeners: {}, textContent: '',
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    append(...items) { this.children.push(...items); items.forEach((item) => { item.parent = this; }); },
    remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); }};
}
const body = element('body');
const document = {body, createElement: element, querySelector: () => null,
  getElementById(id) { return body.children.find((child) => child.id === id) || null; }};
const responses = [];
function reply(ok, payload) {
  return {ok, clone() { return {json: async () => payload}; }};
}
const window = {fetch: async (url) => responses.shift()};
const timers = [];
vm.runInNewContext(source, {window, document, setTimeout: (fn) => timers.push(fn), String});
(async () => {
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const notices = () => (document.getElementById('rf-error-messages')?.children || [])
    .map((notice) => notice.children[0].textContent);
  responses.push(reply(false, {error: 'There is no prompt edit to undo.', visible: true}));
  const response = await window.fetch('/gradio_api/run/predict');
  assert.equal(response.ok, false);
  await settle();
  assert.deepEqual(notices(), ['There is no prompt edit to undo.']);
  assert.equal(document.getElementById('rf-error-messages').attributes.role, 'alert');
  // Queued events, successes and errors without app text show nothing new.
  responses.push(reply(false, {error: 'queued'}), reply(true, {error: 'ok'}),
    reply(false, {error: null}), reply(false, {error: 'hidden', visible: false}));
  await window.fetch('/gradio_api/queue/join');
  await window.fetch('/gradio_api/run/predict');
  await window.fetch('/gradio_api/run/predict');
  await window.fetch({url: 'http://x/gradio_api/run/predict'});
  await settle();
  assert.deepEqual(notices(), ['There is no prompt edit to undo.']);
  // The close button and the timer both remove the notice.
  const region = document.getElementById('rf-error-messages');
  region.children[0].children[1].listeners.click();
  assert.equal(region.children.length, 0);
  responses.push(reply(false, {error: 'Second message.'}));
  await window.fetch('/gradio_api/run/predict');
  await settle();
  timers.at(-1)();
  assert.equal(region.children.length, 0);
})().catch((error) => { console.error(error); process.exit(1); });
"""


class ErrorMessageScriptTests(unittest.TestCase):
    def test_unqueued_failures_with_app_text_are_shown_and_dismissable(self):
        source = asset_path("error_messages.js").read_text(encoding="utf-8")
        result = subprocess.run(
            ["node", "-e", HARNESS],
            input=json.dumps(source),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_script_runs_in_the_page_head_before_gradio_starts(self):
        head = html.startup_head("1", themed=False)
        self.assertIn("window.__rfErrorMessages", head)


if __name__ == "__main__":
    unittest.main()
