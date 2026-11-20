const assert = require("node:assert/strict");
const {
  preset,
  constrain,
  zoomAt,
  resized,
  reframed,
  toggled,
  pinched,
  nudged,
} = require("../ruinedfooocus/ui/assets/image_viewer.js");
const view = { w: 600, h: 400 },
  image = { w: 800, h: 1200 };
const fit = preset("fit", view, image);
assert.equal(fit.scale, 1 / 3);
assert.equal(fit.y, 0);
assert.ok(fit.x > 0, "Fit shows the full portrait with centered side bars");
const fill = preset("fill", view, image);
assert.equal(fill.scale, 0.75);
assert.equal(fill.x, 0);
assert.equal(fill.y, -250, "Fill crops equally and covers the frame");
const actual = preset("actual", view, image);
assert.equal(actual.scale, 1);
assert.deepEqual(constrain({ ...fill, x: 9999, y: 9999 }, view, image), { ...fill, x: 0, y: 0 });
assert.equal(
  constrain({ ...fill, y: -9999 }, view, image).y,
  -500,
  "drag can reach the far edge without blank space",
);
assert.deepEqual(
  reframed({ ...fill, y: -75 }, view, view, image, "fill"),
  { ...fill, y: -75 },
  "rebinding after native UI updates must not recenter a panned image",
);
const point = { x: 300, y: 200 };
const enlarged = zoomAt(actual, 1.2, point, view, image);
for (const axis of ["x", "y"]) {
  assert.ok(
    Math.abs(
      (point[axis] - actual[axis]) / actual.scale - (point[axis] - enlarged[axis]) / enlarged.scale,
    ) < 1e-9,
    "zoom keeps the point beneath the cursor",
  );
}
assert.equal(zoomAt(actual, 1e9, point, view, image).scale, 16);
assert.equal(zoomAt(actual, 1e-9, point, view, image).scale, 0.05);
assert.deepEqual(resized(9999, 9999, 375, 812), { width: 375, height: 1624 });
assert.deepEqual(resized(10, 10, 180, 812), { width: 180, height: 200 });
// Double-click: the whole image goes to actual pixels under the pointer, and back.
const near = (a, b) => Math.abs(a - b) < 1e-9;
const toActual = toggled(fit, point, view, image);
assert.equal(toActual.mode, "actual");
assert.equal(toActual.pose.scale, 1);
assert.ok(
  near((point.x - fit.x) / fit.scale, (point.x - toActual.pose.x) / toActual.pose.scale),
  "double-click keeps the clicked pixel under the pointer",
);
assert.deepEqual(toggled(toActual.pose, point, view, image), { pose: fit, mode: "fit" });
assert.deepEqual(toggled(fill, point, view, image), { pose: fit, mode: "fit" });
// An image that already fits at 1:1 zooms to 2x instead of doing nothing.
const small = { w: 600, h: 400 };
const exact = toggled(preset("fit", view, small), point, view, small);
assert.equal(exact.pose.scale, 2);
assert.equal(exact.mode, "custom");
// Pinch: scale by the finger distance around the midpoint, then follow it.
const from = { mid: { x: 300, y: 200 }, distance: 100 };
const spread = pinched(actual, from, { mid: { x: 300, y: 200 }, distance: 200 }, view, image);
assert.equal(spread.scale, 2);
assert.ok(
  near((300 - actual.x) / actual.scale, (300 - spread.x) / spread.scale),
  "pinch keeps the pixel between the fingers in place",
);
const moved = pinched(actual, from, { mid: { x: 340, y: 230 }, distance: 100 }, view, image);
assert.equal(moved.scale, 1);
assert.deepEqual(
  moved,
  constrain({ ...actual, x: actual.x + 40, y: actual.y + 30 }, view, image),
  "two fingers moving together pan",
);
// Arrow keys move a tenth of the view and never past the image edge.
const centered = constrain({ ...actual, x: -100, y: -400 }, view, image);
assert.deepEqual(nudged(centered, 1, 0, view, image), { ...centered, x: -160 });
assert.deepEqual(nudged(centered, 0, -1, view, image), { ...centered, y: -360 });
assert.deepEqual(nudged(fill, 0, -99, view, image), { ...fill, y: 0 });
console.log(
  "PASS: fit/fill/actual pixels, bounded panning, cursor-anchored zoom, responsive resizing, " +
    "double-click, pinch and arrow keys",
);
