/* Show app messages from events that skip Gradio's queue.
   Gradio shows gr.Error text only for queued events. A failed unqueued event
   (/run/predict) still returns that text in its response, so show it here. */
(() => {
  if (window.__rfErrorMessages || typeof window.fetch !== "function") return;
  window.__rfErrorMessages = true;
  const original = window.fetch.bind(window);

  const show = (text) => {
    let region = document.getElementById("rf-error-messages");
    if (!region) {
      region = document.createElement("div");
      region.id = "rf-error-messages";
      region.setAttribute("role", "alert");
      region.setAttribute("aria-live", "assertive");
      // Inside the app container so it uses the app's theme variables.
      (document.querySelector(".gradio-container") || document.body).append(region);
    }
    const notice = document.createElement("div");
    notice.className = "rf-error-message";
    const message = document.createElement("span");
    message.textContent = text;
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "×";
    close.setAttribute("aria-label", "Dismiss");
    close.addEventListener("click", () => notice.remove());
    notice.append(message, close);
    region.append(notice);
    setTimeout(() => notice.remove(), 10000);
  };

  window.fetch = async (...args) => {
    const response = await original(...args);
    try {
      const url = String(args[0]?.url ?? args[0] ?? "");
      if (!response.ok && url.includes("/run/predict")) {
        response
          .clone()
          .json()
          .then((body) => {
            if (typeof body?.error === "string" && body.error && body.visible !== false) {
              show(body.error);
            }
          })
          .catch(() => {});
      }
    } catch (error) {
      /* Never interfere with the request itself. */
    }
    return response;
  };
})();
