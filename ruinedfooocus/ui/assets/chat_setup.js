async () => {
  const library = document.getElementById("chat-model-library");
  const collapsed = library?.querySelector('button[aria-expanded="false"]');
  if (collapsed) {
    collapsed.click();
    await new Promise(requestAnimationFrame);
  }
  const source = document.getElementById("chat-model-source");
  const load = document.getElementById("chat-model-actions");
  const target = source || load;
  target?.scrollIntoView({ block: "start", behavior: "instant" });
  target
    ?.querySelector('input:not(:disabled),button:not(:disabled),[tabindex="0"]')
    ?.focus({ preventScroll: true });
  return [];
};
