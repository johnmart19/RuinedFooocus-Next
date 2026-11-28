// Deterministic DOM/MutationObserver contract fixture; live browser QA is separate.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync("ruinedfooocus/ui/assets/control_help.js", "utf8");
const measure = process.argv.includes("--measure");

function fixture() {
  const counts = { scans: 0, visited: 0, prepared: 0, scheduled: 0, observers: 0 };
  const records = [],
    microtasks = [],
    listeners = new Map();
  let notify;
  class Element {
    constructor(tag = "div", className = "") {
      this.nodeType = 1;
      this.tagName = tag.toUpperCase();
      this.className = className;
      this.childNodes = [];
      this.parentElement = null;
      this.attributes = {};
      this.open = false;
      this.id = "";
      this.classList = {
        contains: (name) => this.className.split(" ").includes(name),
        add: (name) => {
          if (!this.classList.contains(name)) this.className += ` ${name}`;
        },
        remove: (name) => {
          this.className = this.className
            .split(" ")
            .filter((value) => value !== name)
            .join(" ");
        },
      };
    }
    get isConnected() {
      return this === body || !!this.parentElement?.isConnected;
    }
    get textContent() {
      return this.childNodes.map((node) => node.textContent).join("");
    }
    set textContent(value) {
      const previous = this.childNodes;
      previous.forEach((node) => {
        node.parentElement = null;
      });
      this.childNodes = [];
      const text = { nodeType: 3, textContent: value, parentElement: this };
      this.childNodes.push(text);
      this.record(previous, [text]);
    }
    get firstElementChild() {
      return this.childNodes.find((node) => node.nodeType === 1);
    }
    record(removedNodes, addedNodes) {
      if (notify && this.isConnected)
        records.push({ type: "childList", target: this, removedNodes, addedNodes });
    }
    append(...nodes) {
      for (const node of nodes) {
        node.remove?.();
        node.parentElement = this;
        this.childNodes.push(node);
      }
      this.record([], nodes);
    }
    remove() {
      const parent = this.parentElement;
      if (!parent) return;
      parent.childNodes.splice(parent.childNodes.indexOf(this), 1);
      this.parentElement = null;
      parent.record([this], []);
    }
    setAttribute(name, value) {
      this.attributes[name] = value;
    }
    getAttribute(name) {
      return this.attributes[name];
    }
    matches(selector) {
      return selector.split(",").some((part) => {
        part = part.trim();
        if (part.startsWith("#")) return this.id === part.slice(1);
        if (part.startsWith(".")) return this.classList.contains(part.slice(1));
        if (part === '[data-testid="block-info"]')
          return this.attributes["data-testid"] === "block-info";
        return this.tagName === part.toUpperCase();
      });
    }
    closest(selector) {
      return this.matches(selector) ? this : this.parentElement?.closest(selector) || null;
    }
    querySelectorAll(selector) {
      counts.scans++;
      if (selector === ":scope > .rf-control-help") counts.prepared++;
      const found = [];
      const walk = (parent) => {
        for (const child of parent.childNodes) {
          counts.visited++;
          if (child.nodeType !== 1) continue;
          const matches = selector.startsWith(":is(")
            ? child.matches(".info-text") && child.closest("#app-tabs, #image-stage")
            : selector.startsWith(":scope > ")
              ? parent === this && child.matches(selector.slice(9))
              : child.matches(selector);
          if (matches) found.push(child);
          if (!selector.startsWith(":scope > ")) walk(child);
        }
      };
      walk(this);
      return found;
    }
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    }
  }
  const body = new Element("body");
  const scope = new Element();
  scope.id = "image-stage";
  body.append(scope);
  const host = new Element("div", "block");
  const label = new Element("label", "label-text");
  label.textContent = "Denoise";
  const input = new Element("input");
  input.value = "0.37";
  const info = new Element("p", "info-text");
  const link = new Element("a");
  link.setAttribute("href", "https://example.test/help");
  link.textContent = "Original help";
  info.append(link);
  host.append(label, input, info);
  scope.append(host);
  const json = new Element("div", "block");
  const line = new Element("span");
  line.textContent = "status";
  json.append(line);
  for (let i = 0; i < 200; i++) json.append(new Element("span"));
  scope.append(json);
  const outside = new Element("div", "block");
  outside.textContent = "outside";
  body.append(outside);
  const context = vm.createContext({
    window: {},
    document: {
      body,
      createElement: (tag) => new Element(tag),
      querySelectorAll: (selector) => body.querySelectorAll(selector),
      addEventListener: (type, callback) => listeners.set(type, callback),
    },
    MutationObserver: class {
      constructor(callback) {
        counts.observers++;
        notify = callback;
      }
      observe(target, options) {
        assert.equal(target, body);
        assert.equal(
          options.attributes,
          undefined,
          "disclosure attributes must not observe themselves",
        );
      }
    },
    queueMicrotask(callback) {
      counts.scheduled++;
      microtasks.push(callback);
    },
  });
  const flush = () => {
    for (let turn = 0; records.length || microtasks.length; turn++) {
      assert.ok(turn < 30, "observer must settle after its own DOM changes");
      if (records.length) notify(records.splice(0));
      for (const callback of microtasks.splice(0)) callback();
    }
  };
  vm.runInContext(source, context);
  flush();
  return {
    counts,
    records,
    context,
    flush,
    Element,
    host,
    label,
    input,
    info,
    link,
    json,
    line,
    scope,
    outside,
    listeners,
  };
}

const dom = fixture();
const disclosure = dom.host.querySelector(":scope > .rf-control-help");
assert.equal(disclosure.tagName, "DETAILS");
assert.equal(disclosure.firstElementChild.tagName, "SUMMARY");
assert.equal(disclosure.firstElementChild.getAttribute("role"), "button");
assert.equal(disclosure.firstElementChild.getAttribute("aria-label"), "Denoise: Original help");
assert.equal(disclosure.firstElementChild.getAttribute("aria-controls"), dom.info.id);
assert.equal(dom.info.parentElement, dom.host, "Gradio hint ownership is preserved");
assert.equal(dom.link.parentElement, dom.info, "native hint links stay in place");
assert.equal(dom.input.parentElement, dom.host);
assert.equal(dom.input.value, "0.37");

const before = { ...dom.counts };
for (let index = 0; index < 200; index++) {
  const text = dom.line.childNodes[0];
  text.textContent = `stream ${index}`;
  dom.records.push({ type: "characterData", target: text, addedNodes: [], removedNodes: [] });
  dom.flush();
}
const unrelated = Object.fromEntries(
  Object.keys(before).map((key) => [key, dom.counts[key] - before[key]]),
);
console.log(JSON.stringify({ unrelatedTextMutations: 200, ...unrelated }));
if (!measure)
  assert.equal(unrelated.scans, 0, "unrelated JSON/status text must not scan existing subtrees");
if (!measure) assert.equal(unrelated.prepared, 0, "unrelated text must not prepare controls");
if (!measure) assert.equal(unrelated.scheduled, 0, "unrelated text must not schedule help work");

const status = new dom.Element("span");
status.textContent = "idle";
dom.host.append(status);
dom.flush();
const quiet = { ...dom.counts };
status.textContent = "busy";
dom.line.textContent = "JSON text replacement";
dom.outside.textContent = "unrelated outside text";
dom.flush();
if (!measure)
  assert.equal(
    dom.counts.scans,
    quiet.scans,
    "unrelated text replacements must not scan even hint-bearing controls",
  );
if (!measure) assert.equal(dom.counts.scheduled, quiet.scheduled);
const added = new dom.Element("span");
added.textContent = "new JSON token";
const insertion = { ...dom.counts };
dom.json.append(added);
dom.flush();
if (!measure)
  assert.equal(dom.counts.scans - insertion.scans, 1, "inspect only the inserted subtree");
if (!measure)
  assert.equal(dom.counts.visited - insertion.visited, 1, "do not rescan its existing JSON parent");
if (!measure) assert.equal(dom.counts.prepared, insertion.prepared);

dom.label.textContent = "Rauschstärke";
dom.info.textContent = "Neue Hilfe";
dom.flush();
assert.equal(disclosure.firstElementChild.getAttribute("aria-label"), "Rauschstärke: Neue Hilfe");
dom.info.remove();
dom.flush();
assert.equal(dom.host.querySelector(":scope > .rf-control-help"), null);
const wrapper = new dom.Element();
const replacement = new dom.Element("p", "info-text");
replacement.textContent = "Replacement hint";
wrapper.append(replacement);
dom.host.append(wrapper);
dom.flush();
const recreated = dom.host.querySelector(":scope > .rf-control-help");
assert.equal(
  recreated.firstElementChild.getAttribute("aria-label"),
  "Rauschstärke: Replacement hint",
);
wrapper.remove();
dom.flush();
assert.equal(
  dom.host.querySelector(":scope > .rf-control-help"),
  null,
  "removed nested hints remove disclosures",
);
dom.host.append(dom.info);
dom.flush();
dom.label.remove();
const newLabel = new dom.Element("label");
newLabel.setAttribute("data-testid", "block-info");
newLabel.textContent = "Intensité";
dom.host.append(newLabel);
dom.flush();
assert.equal(
  dom.host.querySelector(":scope > .rf-control-help").firstElementChild.getAttribute("aria-label"),
  "Intensité: Neue Hilfe",
);
dom.info.textContent = "";
dom.flush();
assert.equal(
  dom.host.querySelector(":scope > .rf-control-help"),
  null,
  "empty translated hints remove the disclosure",
);
dom.link.textContent = "Aide traduite";
dom.info.append(dom.link);
dom.flush();
assert.equal(
  dom.host.querySelector(":scope > .rf-control-help").firstElementChild.getAttribute("aria-label"),
  "Intensité: Aide traduite",
);
assert.equal(dom.link.getAttribute("href"), "https://example.test/help");
const nested = new dom.Element("div", "block");
const nestedHint = new dom.Element("p", "info-text");
nestedHint.textContent = "Nested hint";
nested.append(nestedHint);
dom.host.append(nested);
dom.flush();
assert.equal(
  nested.querySelectorAll(":scope > .rf-control-help").length,
  1,
  "inserted control owns its disclosure",
);
assert.equal(nestedHint.parentElement, nested);
assert.equal(
  dom.host.querySelectorAll(":scope > .rf-control-help").length,
  1,
  "nested hints do not create a second parent disclosure",
);
const current = dom.host.querySelector(":scope > .rf-control-help");
current.open = true;
dom.listeners.get("toggle")({ target: current });
assert.equal(current.firstElementChild.getAttribute("aria-expanded"), "true");
vm.runInContext(source, dom.context);
dom.flush();
assert.equal(dom.counts.observers, 1, "initialization is idempotent");
assert.equal(dom.host.querySelectorAll(":scope > .rf-control-help").length, 1);
assert.equal(dom.input.value, "0.37", "enhancement never writes control values");
console.log("control help DOM contracts passed");
