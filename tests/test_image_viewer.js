const assert = require("node:assert/strict");
const {
  preset,
  constrain,
  zoomAt,
  resized,
  reframed,
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
console.log(
  "PASS: fit/fill/actual pixels, bounded panning, cursor-anchored zoom and responsive resizing",
);
