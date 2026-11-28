// Exercise the shipped observer and installed native CSS scoper, without a browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.join(__dirname, "..");
const source = fs.readFileSync(path.join(root, "ruinedfooocus/ui/assets/chat_splitter.js"), "utf8");
const css = fs.readFileSync(path.join(root, "ruinedfooocus/ui/assets/ui.css"), "utf8");
const shell = fs.readFileSync(path.join(root, "ruinedfooocus/ui/assets/shell.css"), "utf8");

function client() {
  const values = new Map();
  const attributes = new Map([["aria-valuenow", "60"]]);
  const classes = new Set();
  const events = {};
  let notify;
  let captured = false;
  const state = { width: 980, font: 16, classWrites: 0 };
  const handle = {
    addEventListener: (name, action) => (events[name] = action),
    getAttribute: (name) => attributes.get(name),
    setAttribute: (name, value) => attributes.set(name, String(value)),
    getClientRects: () => (classes.has("rf-chat-stacked") ? [] : [{}]),
    focus() {},
    setPointerCapture: () => (captured = true),
    hasPointerCapture: () => captured,
    releasePointerCapture: () => (captured = false),
  };
  const row = {
    get clientWidth() {
      return state.width;
    },
    style: {
      getPropertyValue: (name) => values.get(name) || "",
      setProperty: (name, value) => values.set(name, value),
      removeProperty: (name) => values.delete(name),
    },
    classList: {
      contains: (name) => classes.has(name),
      toggle(name, enabled) {
        state.classWrites++;
        if (enabled) classes.add(name);
        else classes.delete(name);
      },
    },
    querySelector: () => ({
      clientWidth: (state.width * Number.parseFloat(values.get("--chat-width") || "60")) / 100,
    }),
    getBoundingClientRect: () => ({ left: 0, width: state.width }),
  };
  vm.runInNewContext(source, {
    element: { querySelector: () => handle, closest: () => row },
    getComputedStyle: () => ({ fontSize: `${state.font}px` }),
    ResizeObserver: class {
      constructor(callback) {
        notify = callback;
      }
      observe(target) {
        assert.equal(target, row);
      }
    },
  });
  return { state, values, attributes, classes, events, notify: () => notify() };
}

const app = client();
app.notify();
assert.equal(app.classes.has("rf-chat-stacked"), false);
app.events.keydown({ key: "ArrowRight", preventDefault() {} });
assert.equal(app.values.get("--chat-width"), "62%");
app.events.pointerdown({ pointerId: 1, preventDefault() {} });
app.events.pointermove({ pointerId: 1, clientX: 490 });
app.events.pointerup({ pointerId: 1 });
assert.equal(app.values.get("--chat-width"), "50%");
const preference = app.values.get("--chat-width");
app.state.width = 689;
app.state.font = 32;
app.notify();
assert.equal(app.classes.has("rf-chat-stacked"), true);
assert.equal(app.values.get("--chat-width"), preference);
const writes = app.state.classWrites;
app.notify();
assert.equal(app.state.classWrites, writes, "unchanged observer results must not rewrite classes");
app.state.width = 0;
app.state.font = 16;
app.notify();
assert.equal(app.state.classWrites, writes, "hidden tabs must not change fit state");
app.state.width = 980;
app.notify();
assert.equal(app.classes.has("rf-chat-stacked"), false);
assert.equal(app.values.get("--chat-width"), preference);
app.state.font = 32;
app.notify();
assert.equal(app.classes.has("rf-chat-stacked"), true, "font-only changes must change fit");
app.state.width = 1920;
app.notify();
assert.equal(app.classes.has("rf-chat-stacked"), false);
app.events.keydown({ key: "Home", preventDefault() {} });
assert.equal(app.values.has("--chat-width"), false);

// Use the installed Gradio implementation, not a copied CSS prefix algorithm.
// These CSSOM records describe the three browser-parsed rule kinds under test.
class CSSStyleRule {
  constructor(selectorText, declarations) {
    this.selectorText = selectorText;
    this.cssText = `${selectorText} { ${declarations} }`;
  }
}
class CSSMediaRule {
  constructor(rule) {
    this.media = { mediaText: "(min-width: 1001px)" };
    this.cssRules = [rule];
    this.cssText = `@media ${this.media.mediaText} { ${rule.cssText} }`;
  }
}
class CSSContainerRule {
  constructor() {
    this.cssText = "@container rf-chat (max-width: 52rem) { #chat-layout { display: flex; } }";
  }
}
function style(selector, stylesheet = css) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = stylesheet.match(new RegExp(`${escaped}\\s*\\{([^}]+)\\}`));
  assert.ok(match, `production selector missing: ${selector}`);
  return new CSSStyleRule(selector, match[1]);
}
const grid = style("#chat-layout:has(#chat-divider)");
const stacked = style("#chat-layout.rf-chat-stacked:has(#chat-divider)");
const imageRow = style("#generation-layout");
const imageWorkspace = style("#generation-workspace");
const imageControls = style("#generation-controls");
const imageStacked = style("#generation-layout.rf-image-stacked > #generation-controls", shell);
const inspectorSticky = shell.match(
  /#generation-controls,\s*#browser-controls,\s*#chat-controls\s*\{([^}]+)\}/,
);
assert.ok(inspectorSticky);
const records = [
  new CSSMediaRule(grid),
  new CSSContainerRule(),
  stacked,
  imageRow,
  imageWorkspace,
  imageControls,
  new CSSMediaRule(new CSSStyleRule("#generation-controls", inspectorSticky[1])),
  imageStacked,
];
const nativeRoot = process.argv[2];
assert.ok(nativeRoot, "pass the installed Gradio frontend asset directory");
const bundle = fs
  .readdirSync(nativeRoot)
  .filter((name) => /^core-.*\.js$/.test(name))
  .map((name) => fs.readFileSync(path.join(nativeRoot, name), "utf8"))
  .find((text) => text.includes(" as prefix_css"));
assert.ok(bundle, "installed native CSS prefixer is required");
const prefixName = bundle.match(/(\w+) as prefix_css/)[1];
const start = bundle.indexOf(`function ${prefixName}(`);
const end = bundle.indexOf("}var ", start) + 1;
assert.ok(start >= 0 && end > start, "native CSS prefixer function boundary changed");
const prefix = vm.runInNewContext(`(${bundle.slice(start, end)})`, {
  R: true,
  document: { createElement: () => ({ remove() {} }) },
  CSSStyleRule,
  CSSMediaRule,
  CSSKeyframesRule: class {},
  CSSKeyframeRule: class {},
  CSSFontFaceRule: class {},
  CSSStyleSheet: class {
    replaceSync() {
      this.cssRules = records;
    }
  },
});
const rendered = prefix(records.map((rule) => rule.cssText).join("\n"), "6-31-0");
assert.equal(rendered.includes("@container"), false, "native scoper drops container rules");
assert.ok(rendered.includes(".contain  #chat-layout.rf-chat-stacked:has(#chat-divider)"));
assert.equal(css.includes("@container rf-chat"), false, "fit must use supported ordinary rules");
const displays = [...rendered.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
  .filter(
    ([, selector, declarations]) =>
      selector.includes("#chat-layout") && /display:/.test(declarations),
  )
  .map(([, selector, declarations], order) => ({
    selector,
    value: declarations.match(/display:\s*(\w+)/)[1],
    weight:
      (selector.match(/#[\w-]+/g) || []).length * 100 + (selector.match(/\.[\w-]+/g) || []).length,
    order,
  }));
function winningDisplay(active) {
  return displays
    .filter((rule) => active || !rule.selector.includes("rf-chat-stacked"))
    .sort((a, b) => a.weight - b.weight || a.order - b.order)
    .at(-1).value;
}
assert.equal(winningDisplay(false), "grid");
assert.equal(winningDisplay(true), "flex", "stack rule must beat the native scoped desktop grid");
assert.match(rendered, /\.contain\s+#generation-layout\s*\{/);
assert.match(imageRow.cssText, /display:\s*flex\s*!important/);
assert.match(imageRow.cssText, /flex-wrap:\s*wrap/);
assert.match(imageWorkspace.cssText, /flex:\s*5\s+1\s+0\s*!important/);
assert.match(imageControls.cssText, /flex:\s*2\s+1\s+0\s*!important/);
const minimum = (rule) =>
  Number(rule.cssText.match(/min-inline-size:\s*min\(100%,\s*([\d.]+)rem\)/)[1]);
for (const [available, font, wraps] of [
  [980, 16, false],
  [689, 32, true],
  [375, 32, true],
  [1920, 32, false],
]) {
  const workspace = Math.min(available, minimum(imageWorkspace) * font);
  const controls = Math.min(available, minimum(imageControls) * font);
  assert.equal(workspace + controls + font > available, wraps);
  assert.ok(workspace > 0 && workspace <= available);
  assert.ok(controls > 0 && controls <= available);
}
function inspectorProperty(property, active) {
  return [...rendered.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
    .filter(
      ([, selector, declarations]) =>
        selector.includes("#generation-controls") &&
        (active || !selector.includes("rf-image-stacked")) &&
        new RegExp(`\\b${property}:`).test(declarations),
    )
    .map(([, selector, declarations], order) => ({
      value: declarations.match(new RegExp(`\\b${property}:\\s*([^;}]+)`))[1].trim(),
      weight:
        (selector.match(/#[\w-]+/g) || []).length * 100 +
        (selector.match(/\.[\w-]+/g) || []).length,
      order,
    }))
    .sort((a, b) => a.weight - b.weight || a.order - b.order)
    .at(-1).value;
}
assert.equal(inspectorProperty("position", false), "sticky");
assert.equal(inspectorProperty("position", true), "static");
assert.equal(inspectorProperty("max-block-size", true), "none");
assert.match(imageStacked.cssText, /overflow:\s*visible/);
console.log("PASS: Chat observer transitions, split preservation, native CSS scoping and cascade");
