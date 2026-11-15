# Image viewing controls

Create and Editor share **Fit / Actual size** controls immediately below their
main image. Fit preserves the complete image inside the viewer. Actual size
shows one image pixel per CSS pixel, with scrolling for images larger than the
available space. **Viewer height (%)** adjusts the frame from 25% to 150% of
the browser height; the initial value is 65%.

These controls affect display only. They do not change generated dimensions,
resample, upload or decode another copy, reload the image, or submit a backend
job. They use native Gradio controls with client-only input callbacks and the
small `ui/assets/image_viewer.js` controller. Inpainting's editable canvas and
the existing fullscreen action retain their own behavior.

The CSS keeps oversized images inside the scrollable upload area. Height uses
viewport units so the selected size follows window resizing on desktop and
mobile. Initialization is idempotent and does not reset an adjusted height when
the shell starts again during navigation. Viewing state is attached to the
workspace, so Gradio rebuilding the image component during fullscreen or source
changes does not erase it.
