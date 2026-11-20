# Image viewing controls

Create and Editor share one image viewer. The image fills its frame; the
viewing commands float over its lower edge in one glass bar, as in Windows
Photos:

| Control                | Mouse and touch                           | Keyboard (image focused)   |
| ---------------------- | ----------------------------------------- | -------------------------- |
| Zoom out / in          | − / + in the bar, wheel, two-finger pinch | `-` / `+` (also with Ctrl) |
| Whole image (Fit)      | Fit in the bar, double-click when zoomed  | `0` or `Home`              |
| Cover the frame (Fill) | Fill in the bar                           |                            |
| Actual pixels (1:1)    | 1:1 in the bar, double-click when fitted  | `1`                        |
| Fit ↔ actual pixels    | Click the zoom percentage                 | Enter on the percentage    |
| Move a zoomed image    | Drag, or move two fingers together        | Arrow keys                 |

The active mode (Fit, Fill or 1:1) is highlighted in the bar, and each button's
tooltip names its key. Wheel, pinch and double-click zoom stay anchored beneath
the pointer; Ctrl+wheel keeps the browser's own zoom. Double-click on an image
whose whole view already is 1:1 zooms to 2x. Buttons and keys animate the
change briefly, unless the system asks for reduced motion; wheel, drag and
pinch follow the hand directly. Transparent pixels show a checkerboard.

The keyboard behaviour follows WinUI's ScrollView (`docs/api-specs/ScrollingControls/ScrollView.md`
in microsoft/microsoft-ui-xaml): the focused viewer handles zoom and line-step
panning itself, and Ctrl+Plus/Minus zoom the image rather than the page while
it has focus.

Upload, camera and paste appear as a small pill in the top-left corner when the
pointer is over the image or it has keyboard focus, and always on touch screens.
An empty viewer shows Gradio's own drop area and buttons instead. Fullscreen
and remove sit in a matching pill in the top-right corner. The "Image" caption
is kept for screen readers only.

A quiet grip in the lower-right corner resizes the viewer horizontally and
vertically; its width stays inside the workspace. Arrow keys on the grip also
resize it; double-click restores the default frame. The grip is not shown on
touch screens or below 640px, where the viewer already spans the column. On
touch screens a fitted image allows page scrolling; an enlarged image allows
single-finger panning.

These controls affect display only. They do not change generated dimensions,
resample, upload or decode another copy, reload the image, or submit a backend
job. They use native Gradio controls with client-only callbacks and the
small `ui/assets/image_viewer.js` controller. Inpainting's editable canvas and
the existing fullscreen action retain their own behavior.

Transforms keep images inside a clipped frame, with bounded panning that can
reach every edge. Fit and Fill follow frame resizing. Explicit zoom is retained
while resizing; a new source resets custom zoom to Fit. Resize state is attached
to the workspace and survives native image-component/fullscreen rebuilds.
Observers are limited to the image, its immediate workspace and the command
bar. Rebinding aborts old pointer, wheel and key handlers; pointer movement and
wheel zoom paint at most once per animation frame. No page-wide polling or
extra image copies are used. The geometry (fit, fill, anchored zoom, double-click
toggle, pinch and arrow steps) is tested without a browser in
`tests/test_image_viewer.js`.
