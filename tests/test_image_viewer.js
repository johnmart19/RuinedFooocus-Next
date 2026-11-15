const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const script = fs.readFileSync(process.argv[2], "utf8");
const properties = new Map();
const classes = new Set();
let viewer = {
  style: { setProperty: (name, value) => properties.set(name, value) },
  classList: {
    toggle(name, enabled) {
      if (enabled) classes.add(name);
      else classes.delete(name);
    },
  },
};
const context = vm.createContext({
  window: {},
  document: {
    getElementById: (id) => {
      assert.equal(id, "generation-workspace");
      return viewer;
    },
  },
});
vm.runInContext(script, context);
const api = context.window.rfImageViewer;
assert.equal(properties.get("--rf-viewer-height"), "65dvh");
api.mode("actual");
assert.equal(classes.has("rf-viewer-actual"), true);
api.mode("fit");
assert.equal(classes.size, 0);
for (const [input, result] of [
  [90, "90dvh"],
  [9999, "150dvh"],
  [-1, "25dvh"],
  ["invalid", "65dvh"],
]) {
  api.height(input);
  assert.equal(properties.get("--rf-viewer-height"), result);
}
api.height(85);
vm.runInContext(script, context);
assert.equal(context.window.rfImageViewer, api);
assert.equal(properties.get("--rf-viewer-height"), "85dvh", "remount must keep the user's size");
viewer = null;
assert.doesNotThrow(() => {
  api.mode("actual");
  api.height(100);
});
console.log(
  "PASS: browser-only viewer sizing, actual pixels, bounds, remount and missing-view guards",
);
