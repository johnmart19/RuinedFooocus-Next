const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const listeners = {};
function input(value) {
  return {
    value,
    min: "1",
    max: "32768",
    message: "",
    events: [],
    reports: 0,
    matches: () => true,
    setCustomValidity(message) {
      this.message = message;
    },
    checkValidity() {
      return !this.message && Number(this.value) >= 1 && Number(this.value) <= 32768;
    },
    get validationMessage() {
      return this.message || (this.checkValidity() ? "" : "Out of bounds");
    },
    reportValidity() {
      this.reports++;
    },
    dispatchEvent(event) {
      this.events.push(event.type);
      listeners[event.type]?.({ target: this });
    },
  };
}
const width = input("1440"),
  height = input("1024");
const context = {
  window: {},
  Event: class {
    constructor(type) {
      this.type = type;
    }
  },
  document: {
    addEventListener(name, handler) {
      assert.equal(listeners[name], undefined);
      listeners[name] = handler;
    },
    querySelector(selector) {
      return selector.includes("width") ? width : height;
    },
  },
};
const source = fs.readFileSync("ruinedfooocus/ui/assets/dimension_inputs.js", "utf8");
vm.runInNewContext(source, context);
vm.runInNewContext(source, context); // Startup hooks must not install duplicate handlers.
function paste(target, text) {
  let prevented = false;
  listeners.paste({
    target,
    clipboardData: { getData: () => text },
    preventDefault() {
      prevented = true;
    },
  });
  return prevented;
}
assert.equal(paste(width, "2560"), true);
assert.equal(width.value, "2560"); // Never 14402560.
assert.deepEqual(width.events, ["input"]);
assert.equal(paste(width, "1440 × 2560"), true);
assert.equal(width.value, "1440");
assert.equal(height.value, "2560");
const eventsBefore = width.events.length;
paste(width, "14402560");
assert.equal(width.value, "1440");
assert.equal(width.events.length, eventsBefore);
assert.equal(width.reports, 1);
paste(width, "1920x999999");
assert.equal(width.value, "1440"); // Reject the whole pair without partial updates.
assert.equal(height.value, "2560");
assert.equal(width.events.length, eventsBefore);
assert.equal(paste({ matches: () => false }, "2560"), false);
console.log("Dimension paste replacement, pairs, bounds and listener ownership pass.");
