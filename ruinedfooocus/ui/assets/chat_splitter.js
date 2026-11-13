const handle = element.querySelector('[role="separator"]');
const row = element.closest("#chat-layout");
const maximumWidth = () => Math.floor(((row.clientWidth - 306) / row.clientWidth) * 100);
const setWidth = (percent) => {
  // Keep the model controls usable even near the desktop/mobile breakpoint.
  const maximum = maximumWidth();
  percent = Math.max(35, Math.min(maximum, percent));
  row.style.setProperty("--chat-width", percent + "%");
  handle.setAttribute("aria-valuemax", Math.floor(maximum));
  handle.setAttribute("aria-valuenow", Math.round(percent));
};
const resetWidth = () => {
  row.style.removeProperty("--chat-width");
  const conversation = row.querySelector("#chat-conversation");
  handle.setAttribute("aria-valuemax", maximumWidth());
  handle.setAttribute(
    "aria-valuenow",
    Math.round((conversation.clientWidth / row.clientWidth) * 100),
  );
};
handle.addEventListener("pointerdown", (event) => {
  handle.focus();
  handle.setPointerCapture(event.pointerId);
  event.preventDefault();
});
handle.addEventListener("pointermove", (event) => {
  if (!handle.hasPointerCapture(event.pointerId)) return;
  const bounds = row.getBoundingClientRect();
  setWidth(((event.clientX - bounds.left) / bounds.width) * 100);
});
handle.addEventListener("pointerup", (event) => handle.releasePointerCapture(event.pointerId));
handle.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home"].includes(event.key)) return;
  event.preventDefault();
  const current = Number(handle.getAttribute("aria-valuenow"));
  if (event.key === "Home") resetWidth();
  else setWidth(current + (event.key === "ArrowLeft" ? -2 : 2));
});
handle.addEventListener("dblclick", resetWidth);
const observer = new ResizeObserver(() => {
  if (row.clientWidth <= 0) return;
  // Native CSS scoping drops container queries. Measure the actual working
  // space here, including font changes, and retain the user's split preference.
  const stacked = row.clientWidth <= 52 * Number.parseFloat(getComputedStyle(row).fontSize);
  if (row.classList.contains("rf-chat-stacked") !== stacked) {
    row.classList.toggle("rf-chat-stacked", stacked);
  }
  if (stacked || !handle.getClientRects().length) return;
  if (row.style.getPropertyValue("--chat-width")) {
    setWidth(Number(handle.getAttribute("aria-valuenow")));
  } else resetWidth();
});
observer.observe(row);
