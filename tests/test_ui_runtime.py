"""Execute the shipped browser runtime and Gradio restart endpoint without a server."""

import ast
import json
from pathlib import Path
import re
import subprocess
import unittest

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware import Middleware

from ruinedfooocus.integrations.web_access import PrivateResponses
from ruinedfooocus.ui import themes

ROOT = Path(__file__).resolve().parents[1]
APP_ID = 18446744073709551614
HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const scenario = process.argv[1];
const settle = () => new Promise(resolve => setImmediate(resolve));
const response = (text, options = {}) => ({
  ok: options.ok ?? true,
  redirected: options.redirected ?? false,
  headers: { get: () => options.type ?? 'application/json; charset=utf-8' },
  text: async () => text,
});
function environment({root = '/rf', prefix = '/gradio_api', startup = true, replies = []} = {}) {
  const ids = new Map(), classes = new Set(), calls = [], intervals = new Map(), timeouts = new Map();
  let timerId = 0, reloads = 0;
  const links = [{href: '/rf/theme.css', disabled: false}, {href: '/rf/app.css', disabled: false}];
  const container = {classList: {toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); }}};
  const document = {
    hidden: false,
    addEventListener() {},
    querySelector: selector => selector === '.gradio-container' ? container : null,
    querySelectorAll: () => links.filter(link => link.href.includes('theme.css')),
    getElementById: id => ids.get(id) || null,
    createElement: tag => ({tagName: tag, textContent: ''}),
    head: {appendChild(element) { ids.set(element.id, element); }},
  };
  const window = {
    location: {href: 'https://example.invalid/rf/settings?__theme=dark#section', reload() { reloads++; }},
    gradio_config: {root, api_prefix: prefix, app_id: Number(input.appId)},
  };
  const context = {
    window, document, URL, AbortController, localStorage: {getItem: () => null},
    setInterval(fn, delay) { const id = ++timerId; intervals.set(id, {fn, delay}); return id; },
    clearInterval: id => intervals.delete(id),
    setTimeout(fn, delay) { const id = ++timerId; timeouts.set(id, {fn, delay}); return id; },
    clearTimeout: id => timeouts.delete(id),
    async fetch(url, options) {
      calls.push({url, options});
      const reply = replies.shift();
      if (typeof reply === 'function') return reply(options);
      if (reply instanceof Error) throw reply;
      return reply || response(`{"app_id":${input.appId}}`);
    },
  };
  vm.createContext(context);
  if (startup) vm.runInContext(input.startup, context);
  vm.runInContext(input.script, context);
  return {
    context, document, window, calls, replies, ids, classes, links, intervals, timeouts,
    reloads: () => reloads,
    tick: () => [...intervals.values()][0]?.fn(),
  };
}
(async () => {
  if (scenario === 'themes') {
    const env = environment();
    await settle();
    assert.equal(env.classes.has('rf-themed'), true);
    assert.equal(env.ids.size, 0); // Startup retains Gradio's native theme stylesheet.
    const apply = vm.runInContext(`(${input.apply})`, env.context);
    apply(input.css.soft, true);
    const live = env.ids.get('rf-live-theme');
    assert.equal(live.textContent, input.css.soft);
    assert.equal(env.links[0].disabled, true);
    assert.equal(env.links[1].disabled, false);
    apply(input.css.custom, true);
    assert.equal(live.textContent, input.css.custom);
    assert.equal(env.classes.has('rf-themed'), true);
    apply(input.css.default, false);
    assert.equal(live.textContent, input.css.default);
    assert.equal(env.classes.has('rf-themed'), false);
    assert.equal(env.ids.size, 1); // Reuse one style node when switching back to Default.
    apply(input.css.default, false); // Missing custom themes use the same Default path.
    assert.equal(env.classes.has('rf-themed'), false);
  } else if (scenario === 'proxy-and-precision') {
    for (const [root, expected] of [
      ['/rf', 'https://example.invalid/rf/gradio_api/app_id'],
      ['/rf/', 'https://example.invalid/rf/gradio_api/app_id'],
      ['https://example.invalid/proxy/rf', 'https://example.invalid/proxy/rf/gradio_api/app_id'],
    ]) {
      const env = environment({root});
      await settle();
      assert.equal(env.calls[0].url, expected);
      assert.equal(env.calls[0].options.cache, 'no-store');
      assert.equal(env.calls[0].options.credentials, 'same-origin');
      assert.equal(env.calls[0].options.redirect, 'error');
      assert.ok(env.calls[0].options.signal instanceof AbortSignal);
      assert.equal([...env.intervals.values()][0].delay, 5000);
      env.replies.push(response(` { "app_id" : "${input.appId}" }\n`));
      await env.tick();
      assert.equal(env.reloads(), 0);
      env.replies.push(response('{"app_id":18446744073709551615}'));
      await env.tick();
      assert.equal(env.reloads(), 1); // Adjacent ids round to the same JavaScript Number.
      assert.equal(env.intervals.size, 0);
    }
  } else if (scenario === 'invalid-and-auth') {
    const env = environment();
    await settle();
    const invalid = [
      response('login', {ok: false}),
      response('<html>login</html>', {type: 'text/html'}),
      response('{"app_id":123}', {redirected: true}),
      response('not json'), response('{}'), response('[]'),
      response('{"app_id":1.5}'), response('{"app_id":-1}'),
      response('{"app_id":123,"error":"login"}'), new Error('server unavailable'),
    ];
    for (const reply of invalid) {
      env.replies.push(reply);
      await env.tick();
      assert.equal(env.reloads(), 0);
    }
    const count = env.calls.length;
    env.document.hidden = true;
    await env.tick();
    assert.equal(env.calls.length, count);
    env.document.hidden = false;
    await env.tick();
    assert.equal(env.reloads(), 0);
    env.replies.push(response('{"app_id":18446744073709551615}'));
    await env.tick();
    assert.equal(env.reloads(), 1);
  } else if (scenario === 'bounded-and-single') {
    const pending = options => new Promise((resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new Error('timeout')), {once: true});
    });
    const env = environment({replies: [pending]});
    assert.equal(env.calls.length, 1);
    await env.tick();
    assert.equal(env.calls.length, 1);
    const timeout = [...env.timeouts.values()][0];
    assert.equal(timeout.delay, 4000);
    timeout.fn();
    await settle();
    assert.equal(env.timeouts.size, 0);
    await env.tick();
    assert.equal(env.calls.length, 2);
    assert.equal(env.reloads(), 0);
    vm.runInContext(input.script, env.context);
    await settle();
    assert.equal(env.calls.length, 2);
    assert.equal(env.intervals.size, 1);
  } else if (scenario === 'startup-baseline') {
    const changed = response('{"app_id":18446744073709551615}');
    const env = environment({replies: [changed]});
    await settle();
    assert.equal(env.reloads(), 1); // A restart before the first poll is still detected.
    const fallback = environment({startup: false, replies: [changed]});
    await settle();
    assert.equal(fallback.reloads(), 0);
    fallback.replies.push(changed);
    await fallback.tick();
    assert.equal(fallback.reloads(), 0);
    fallback.replies.push(response('{"app_id":1}'));
    await fallback.tick();
    assert.equal(fallback.reloads(), 1);
  } else throw new Error('Unknown scenario');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def runtime_payload():
    tree = ast.parse((ROOT / "ruinedfooocus/ui/html.py").read_text(encoding="utf-8"))
    scope = {"json": json}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and node.targets[0].id in ("head", "sidebar_head", "scripts")
        or isinstance(node, ast.FunctionDef)
        and node.name == "startup_head"
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "browser_runtime", "exec"), scope)
    settings = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
    event = next(
        node
        for node in ast.walk(settings)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "then"
        and isinstance(node.func.value, ast.Call)
        and ast.unparse(node.func.value.func) == "theme.change"
    )
    apply = next(ast.literal_eval(kw.value) for kw in event.keywords if kw.arg == "js")
    custom = themes.normalized_schema({"theme": {"body_text_color": "#123456"}})
    custom_theme = gr.themes.Default(font=themes.SYSTEM_FONT, font_mono=themes.MONO_FONT)
    for key, value in custom["theme"].items():
        setattr(custom_theme, key, value)
    return {
        "appId": str(APP_ID),
        "script": scope["scripts"],
        "startup": re.search(r"<script>(.*?)</script>", scope["startup_head"](APP_ID, themed=True))[
            1
        ],
        "apply": apply,
        "css": {
            "default": themes.resolve_theme("")[0]._get_theme_css(),
            "soft": themes.resolve_theme("builtin:soft")[0]._get_theme_css(),
            "custom": custom_theme._get_theme_css(),
        },
    }


class BrowserRuntimeTests(unittest.TestCase):
    def run_scenario(self, name):
        result = subprocess.run(
            ["node", "-e", HARNESS, name],
            input=json.dumps(runtime_payload()),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_live_theme_switch_and_default_reset_use_one_style_and_keep_other_css(self):
        self.run_scenario("themes")

    def test_proxy_roots_credentials_and_exact_app_id_changes(self):
        self.run_scenario("proxy-and-precision")

    def test_auth_redirects_invalid_replies_outages_and_hidden_tabs_do_not_reload(self):
        self.run_scenario("invalid-and-auth")

    def test_polls_timeout_serialize_and_install_only_one_watcher(self):
        self.run_scenario("bounded-and-single")

    def test_startup_id_detects_restarts_before_first_response(self):
        self.run_scenario("startup-baseline")


class RestartEndpointTests(unittest.TestCase):
    def test_native_app_id_remains_exact_and_reachable_with_guest_auth_and_mount_prefix(self):
        with gr.Blocks(analytics_enabled=False) as blocks:
            gr.Textbox()
        self.addCleanup(blocks.close)
        blocks.app_id = APP_ID
        app = gr.mount_gradio_app(
            FastAPI(),
            blocks,
            path="/proxy/rf",
            auth=("fixture-user", "fixture-password"),
            allowed_paths=[],
            ssr_mode=False,
            mcp_server=False,
            app_kwargs={"middleware": [Middleware(PrivateResponses, guest=True)]},
        )
        with TestClient(app) as client:
            endpoint = "/proxy/rf/gradio_api/app_id"
            reply = client.get(endpoint)
            self.assertEqual(reply.status_code, 200)
            self.assertEqual(reply.text, '{"app_id":' + str(APP_ID) + "}")
            self.assertEqual(reply.headers["cache-control"], "private, no-store")
            self.assertEqual(client.get("/proxy/rf/config").status_code, 401)
            self.assertEqual(
                client.post(
                    "/proxy/rf/login",
                    data={"username": "fixture-user", "password": "fixture-password"},
                ).status_code,
                200,
            )
            config = client.get("/proxy/rf/config")
            self.assertEqual(config.status_code, 200)
            self.assertTrue(config.json()["root"].endswith("/proxy/rf"))
            self.assertEqual(client.get(endpoint).text, reply.text)
            client.cookies.clear()
            self.assertEqual(client.get("/proxy/rf/config").status_code, 401)
            blocks.app_id += 1
            restarted = client.get(endpoint)
            self.assertEqual(restarted.status_code, 200)
            self.assertEqual(restarted.json()["app_id"], APP_ID + 1)


if __name__ == "__main__":
    unittest.main()
