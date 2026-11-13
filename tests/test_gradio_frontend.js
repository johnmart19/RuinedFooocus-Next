const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { original, patched, tracker, syntax } = JSON.parse(readFileSync(0, "utf8"));
const { event: eventName, stream: streamName, status: statusName } = syntax;

// Execute the installed native LoadingStatus class and actual completion branch.
const trackerStart = tracker.indexOf("var $=class{current={};");
const trackerEnd = tracker.indexOf(",Ge=", trackerStart);
assert.ok(trackerStart >= 0 && trackerEnd > trackerStart);
const Tracker = Function("return " + tracker.slice(trackerStart + 6, trackerEnd))();
function completion(source) {
  const begin = source.indexOf(`if(${eventName}.stage===\`complete\`){`);
  const end = source.indexOf(`else if(${eventName}.stage===\`generating\``, begin);
  assert.ok(begin >= 0 && end > begin);
  return Function(
    "r",
    eventName,
    `let ${streamName}=null,a=[],${statusName}={stage:${eventName}.stage},l=null;submit_loop:{` +
      source.slice(begin, end) +
      `}return ${streamName}`,
  );
}
function run(branch, connection, outputs = [], mode = "hidden") {
  const tracker = new Tracker();
  tracker.register(7, outputs, [666], mode);
  const effects = [];
  const owner = {
    dispatch() {
      throw Error("unexpected trigger");
    },
    dispatch_state_change_events: (event) => effects.push(event),
    update_loading_status: (event) => tracker.update(event),
    update_loading_stati_state: () => effects.push("publish"),
  };
  const event = { stage: "complete" };
  const stream = branch.call(owner, { id: 7, connection_type: connection }, event);
  assert.deepEqual(effects, [event, "publish"]);
  return { tracker, stream };
}
assert.equal(run(completion(original), "sse").tracker.current[666].type, "input");
for (const mode of ["hidden", "full", "minimal"]) {
  const ordinary = run(completion(patched), "sse", [], mode);
  assert.equal(ordinary.stream, null);
  assert.deepEqual(ordinary.tracker.current, {});
  const streamed = run(completion(patched), "stream", [], mode);
  assert.equal(streamed.stream, "closed");
  assert.equal(streamed.tracker.current[666].type, "input");
  const notice = run(completion(patched), "sse", [35], mode).tracker.current;
  assert.equal(notice[35].type, "output");
  assert.equal(notice[35].status, "complete");
  assert.equal(notice[35].show_progress, mode);
  assert.equal(notice[666], undefined);
}
// Existing output entries still complete through native shared tracking.
const shared = new Tracker();
shared.register(7, [35], [666], "hidden");
shared.update({ fn_index: 7, status: "pending" });
shared.update({ fn_index: 7, status: "complete", stream_state: null });
assert.equal(shared.current[35].status, "complete");
assert.equal(shared.current[666], undefined);

const errorBegin = `else if(${eventName}.stage===\`error\`){`;
const errorEnd = `if(${eventName}.type===\`render\`){`;
const errorBranch = (source) =>
  source.slice(source.indexOf(errorBegin), source.indexOf(errorEnd, source.indexOf(errorBegin)));
assert.ok(original.includes(errorBegin) && original.includes(errorEnd));
assert.equal(errorBranch(patched), errorBranch(original));
const validation = `${eventName}.message.forEach((e,t)=>{this.update_state_cb(r.inputs[t],{loading_status:{validation_error:e.is_valid?null:e.message,show_validation_error:!0}},!1)})`;
assert.equal(patched.split(validation).length, 2);
const applyValidation = Function(
  "r",
  eventName,
  validation.replace(`${eventName}.message.forEach`, `return ${eventName}.message.forEach`),
);
const updates = [];
applyValidation.call(
  { update_state_cb: (...args) => updates.push(args) },
  { inputs: [666, 667] },
  {
    message: [
      { is_valid: false, message: "required" },
      { is_valid: true, message: "unused" },
    ],
  },
);
assert.deepEqual(updates, [
  [666, { loading_status: { validation_error: "required", show_validation_error: true } }, false],
  [667, { loading_status: { validation_error: null, show_validation_error: true } }, false],
]);
console.log(
  "Native SSE, streaming, progress output, shared tracker and validation contracts passed.",
);
