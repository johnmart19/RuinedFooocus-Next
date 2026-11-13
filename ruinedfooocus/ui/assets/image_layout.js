// The shared Create/Editor row is mounted by the existing shell startup hook.
(() => {
  const row = document.getElementById("generation-layout");
  if (!row || row.__rfImageLayoutObserver) return;
  const workspace = row.querySelector("#generation-workspace");
  const controls = row.querySelector("#generation-controls");
  if (!workspace || !controls) return;
  const update = () => {
    if (row.clientWidth <= 0) return;
    // Font-relative CSS minima decide wrapping. Measure their actual result,
    // rather than duplicate those minima or use viewport width as a proxy.
    const gap = Number.parseFloat(getComputedStyle(row).columnGap) || 0;
    const occupied =
      workspace.getBoundingClientRect().width + controls.getBoundingClientRect().width + gap;
    const stacked = occupied > row.clientWidth + 1;
    if (row.classList.contains("rf-image-stacked") !== stacked) {
      row.classList.toggle("rf-image-stacked", stacked);
    }
  };
  row.__rfImageLayoutObserver = new ResizeObserver(update);
  row.__rfImageLayoutObserver.observe(row);
  update();
})();
