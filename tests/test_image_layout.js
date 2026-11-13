// Run the actual assembled shell startup script with measured-layout fixtures.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const script = JSON.parse(fs.readFileSync(0, "utf8"));
const state = { width: 980, workspace: 676, controls: 288, gap: 16, writes: 0, observers: 0 };
const classes = new Set();
let notify;
const row = {
  get clientWidth() {
    return state.width;
  },
  querySelector: (selector) => ({
    getBoundingClientRect: () => ({
      width: selector === "#generation-workspace" ? state.workspace : state.controls,
    }),
  }),
  classList: {
    contains: (name) => classes.has(name),
    toggle(name, enabled) {
      state.writes++;
      if (enabled) classes.add(name);
      else classes.delete(name);
    },
  },
};
const context = vm.createContext({
  document: {
    getElementById: (id) => (id === "generation-layout" ? row : null),
    querySelector: () => null,
    addEventListener() {},
  },
  window: { __rfRestartWatch: true },
  localStorage: { getItem: () => null },
  getComputedStyle: () => ({ columnGap: `${state.gap}px` }),
  ResizeObserver: class {
    constructor(callback) {
      state.observers++;
      notify = callback;
    }
    observe(target) {
      assert.equal(target, row);
    }
  },
});
vm.runInContext(script, context);
assert.equal(state.observers, 1, "actual shell startup must bind the image observer");
assert.equal(classes.has("rf-image-stacked"), false);
Object.assign(state, { width: 689, workspace: 689, controls: 689, gap: 32 });
notify();
assert.equal(classes.has("rf-image-stacked"), true);
const writes = state.writes;
notify();
assert.equal(state.writes, writes, "stable fit must not rewrite the class");
state.width = 0;
notify();
assert.equal(state.writes, writes, "hidden image stages must not change layout state");
Object.assign(state, { width: 980, workspace: 676, controls: 288, gap: 16 });
notify();
assert.equal(classes.has("rf-image-stacked"), false, "wide restore returns to sticky inspector");
Object.assign(state, { workspace: 980, controls: 980, gap: 32 });
notify();
assert.equal(
  classes.has("rf-image-stacked"),
  true,
  "text-scaled wrapping uses actual child widths",
);
Object.assign(state, { width: 375, workspace: 375, controls: 375 });
notify();
assert.equal(classes.has("rf-image-stacked"), true);
vm.runInContext(script, context);
assert.equal(state.observers, 1, "repeated initialization must reuse the same observer");
Object.assign(state, { width: 980, workspace: 676.4, controls: 288.4, gap: 16 });
notify();
assert.equal(classes.has("rf-image-stacked"), false, "allow subpixel width rounding");
console.log("PASS: actual shell mount, measured image fit, hidden guard and wide restore");
