// Pasting a dimension replaces the old number instead of appending to it.
(() => {
  if (window.__rfDimensionPaste) return;
  window.__rfDimensionPaste = true;
  const selector =
    '#output-target-width input[type="number"], #output-target-height input[type="number"]';
  document.addEventListener(
    "input",
    (event) => {
      if (event.target?.matches?.(selector)) event.target.setCustomValidity("");
    },
    true,
  );
  document.addEventListener(
    "paste",
    (event) => {
      const input = event.target;
      if (!input?.matches?.(selector)) return;
      const text = event.clipboardData?.getData("text/plain")?.trim();
      if (!text) return;
      event.preventDefault();
      const pair = /^(\d+)\s*[x×]\s*(\d+)$/i.exec(text);
      const targets = pair
        ? [
            [document.querySelector('#output-target-width input[type="number"]'), pair[1]],
            [document.querySelector('#output-target-height input[type="number"]'), pair[2]],
          ]
        : [[input, text]];
      for (const [target, value] of targets) {
        if (!target || !/^\d+$/.test(value)) return;
        target.setCustomValidity("");
        const previous = target.value;
        target.value = Number.isSafeInteger(Number(value)) ? value : String(Number(target.max) + 1);
        const valid = target.checkValidity();
        const message = target.validationMessage;
        target.value = previous;
        if (!valid) {
          target.setCustomValidity(message);
          target.reportValidity();
          return;
        }
      }
      for (const [target, value] of targets) target.value = value;
      for (const [target] of targets) target.dispatchEvent(new Event("input", { bubbles: true }));
    },
    true,
  );
})();
