# Image viewing controls

Create and Editor share **Fit**, **Fill**, **Actual size**, **− / +** and a
diagonal resize handle below their main image. Fit shows the whole image; Fill
covers the frame. Actual size uses one image pixel per CSS pixel. Drag a cropped
or enlarged image to pan, and use the mouse wheel or −/+ buttons to zoom. Wheel
zoom stays anchored beneath the cursor; Ctrl+wheel retains browser zoom.

Drag the diagonal handle to resize the viewer horizontally and vertically. Its
width stays inside the workspace. Arrow keys on the handle also resize it;
double-click restores the default frame. The initial height is 65% of the browser.
On touch screens a fitted image allows page scrolling; an enlarged image allows
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
Observers are limited to the image and its immediate workspace. Rebinding aborts
old pointer/wheel handlers; pointer movement and wheel zoom paint at most once
per animation frame. No page-wide polling or extra image copies are used.
