// Display-only: no uploads, decoding, generation settings or backend events.
(() => {
  if (window.rfImageViewer) return;
  window.rfImageViewer = {
    mode(value) {
      document
        .getElementById("generation-workspace")
        ?.classList.toggle("rf-viewer-actual", value === "actual");
    },
    height(value) {
      const height = Math.max(25, Math.min(150, Number(value) || 65));
      document
        .getElementById("generation-workspace")
        ?.style.setProperty("--rf-viewer-height", `${height}dvh`);
    },
  };
  window.rfImageViewer.height(65);
})();
