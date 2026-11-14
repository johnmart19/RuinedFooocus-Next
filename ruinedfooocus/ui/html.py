from ruinedfooocus.core.paths import asset_path
from pathlib import Path
import json

css = "\n".join(
    asset_path(name).read_text(encoding="utf-8") for name in ("ui.css", "shell.css", "palette.css")
)

# Gradio scopes component CSS below main; outer page spacing belongs in head.
head = """<style>
/* Text and rem spacing follow the browser default up to 1920px wide, then grow
   smoothly with the window: ~117% at 2560px (1440p), 150% at 3840px (4K). */
html { font-size: var(--rf-user-text-scale, clamp(100%, 50% + .4167vw, 150%)); }
@media (max-width: 600px) {
.gradio-container .main.fillable.app { padding: 8px !important; }
}
footer { display: none !important; }</style>"""

sidebar_head = """<style>
/* The fixed sidebar owns the left edge on wide screens. */
@media (min-width: 1001px) {
.gradio-container .main.fillable.app { padding: 1.5rem 1.75rem 2rem calc(14.5rem + 1.75rem) !important; }
}</style>"""


def startup_head(app_id, *, themed=False, sidebar=True):
    from ruinedfooocus.core.paths import asset_path

    # Gradio's random 64-bit id loses precision when read as a JavaScript Number.
    return (
        head
        + (sidebar_head if sidebar else "")
        + (
            f"<script>window.__rfAppId = {json.dumps(str(app_id))}; "
            f"window.__rfThemed = {json.dumps(bool(themed))};</script>"
        )
        # Before Gradio's client starts, so its unqueued requests are observed.
        + f"<script>{asset_path('error_messages.js').read_text(encoding='utf-8')}</script>"
    )


# The label names the progress bar for assistive technology with its status text.
progress_html = """
<label class="loader-container">
  <span class="progress-container">
    <progress value="*number*" max="100"></progress>
  </span>
  <span>*text*</span>
</label>
"""
scripts = r"""
function generate_shortcut(){
  const savedTextScale = Number(localStorage.getItem('ruinedfooocus-text-scale'));
  if (savedTextScale >= 75 && savedTextScale <= 200) {
    document.documentElement.style.setProperty('--rf-user-text-scale', `${savedTextScale}%`);
  }
  // Gradio treats non-navigation keys as search edits, even modifiers and
  // PrintScreen, filtering the list to its selected label. Keep these keys
  // out of dropdown handlers without cancelling the OS screenshot chord.
  const preserveDropdown = (e) => {
    if (['Alt', 'Control', 'Shift', 'Meta', 'PrintScreen'].includes(e.key) &&
        e.target?.closest?.('[role="combobox"][aria-expanded="true"]')) {
      e.stopImmediatePropagation();
      if (e.key === 'Alt') e.preventDefault();
    }
  };
  document.addEventListener('keydown', preserveDropdown, {capture: true});
  document.addEventListener('keyup', preserveDropdown, {capture: true});
  document.addEventListener('keydown', (e) => {
    // Escape belongs to dropdowns/dialogs; it must not hide the image preview.
    if (e.defaultPrevented || !e.ctrlKey) return;
    const handle = e.key === 'Enter' ? 'generate' :
      e.key?.toLowerCase() === 'q' ? 'edit_mode' : null;
    const generate = document.getElementById('generate');
    // Do not dispatch hidden image actions while Chat, Library or Settings is open.
    if (!handle || !generate?.getClientRects().length) return;
    document.getElementById(handle)?.click();
    e.preventDefault();
  });
}
generate_shortcut();
// A chosen theme keeps its own colours; the Default palette steps aside.
window.rfApplyTheme = (css, themed) => {
  const container = document.querySelector('.gradio-container');
  container?.classList.toggle('rf-themed', Boolean(themed));
  if (css === null || css === undefined) return;
  let live = document.getElementById('rf-live-theme');
  if (!live) {
    live = document.createElement('style');
    live.id = 'rf-live-theme';
    document.head.appendChild(live);
  }
  live.textContent = css;
  document.querySelectorAll('link[rel="stylesheet"][href*="theme.css"]')
    .forEach((link) => { link.disabled = true; });
};
window.rfApplyTheme(null, window.__rfThemed);
// A page left open across a server restart holds dead sessions and
// empty panels. Reload once the new server answers.
(() => {
  if (window.__rfRestartWatch) return;
  const config = window.gradio_config || {};
  // Gradio supplies the app root, including reverse-proxy/mount prefixes.
  const root = new URL(config.root || '.', window.location.href);
  root.pathname = root.pathname.replace(/\/?$/, '/');
  root.search = '';
  root.hash = '';
  const endpoint = new URL((config.api_prefix || 'gradio_api').replace(/^\/+/, '') + '/app_id', root);
  let started = window.__rfAppId || null;
  let busy = false;
  let reloading = false;
  const poll = async () => {
    if (document.hidden || busy || reloading) return;
    busy = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const reply = await fetch(endpoint.href, {
        cache: 'no-store', credentials: 'same-origin', redirect: 'error', signal: controller.signal
      });
      if (!reply.ok || reply.redirected || !reply.headers.get('content-type')?.includes('application/json')) return;
      // Compare only the exact id, ignoring JSON whitespace and login/error pages.
      const match = /^\s*\{\s*"app_id"\s*:\s*(?:"(\d+)"|(\d+))\s*\}\s*$/.exec(await reply.text());
      if (!match) return;
      const current = match[1] || match[2];
      if (started === null) started = current;
      else if (current !== started) {
        reloading = true;
        clearInterval(window.__rfRestartWatch);
        window.location.reload();
      }
    } catch (error) { /* server is down; keep waiting */ }
    finally {
      clearTimeout(timeout);
      busy = false;
    }
  };
  window.__rfRestartWatch = setInterval(poll, 5000);
  poll();
})();
"""

scripts += "\n" + asset_path("image_layout.js").read_text(encoding="utf-8")
scripts += "\n" + asset_path("control_help.js").read_text(encoding="utf-8")

import html as html_text

from ruinedfooocus.app.state import state


def make_progress_html(number, text):
    if number == -1:
        number = state["last_progress"]
    else:
        state["last_progress"] = number
    # Status text can name files or prompts; it is text, never markup.
    return progress_html.replace("*number*", str(number)).replace(
        "*text*", html_text.escape(str(text))
    )
