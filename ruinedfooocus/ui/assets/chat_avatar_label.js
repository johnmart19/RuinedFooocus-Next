// Gradio's portrait button has no accessible name and its image has an empty alt.
// Name both after the assistant chosen in the selector below it; display only.
(() => {
  const layout = element.closest("#chat-layout");
  if (!layout || layout.__rfAvatarLabel) return;
  const name = (head) => {
    const button = head.querySelector("#chat-avatar .image-container > button");
    const text = head.querySelector('input[role="listbox"], input')?.value.trim();
    if (!button || !text) return;
    const image = button.querySelector("img");
    if (button.getAttribute("aria-label") !== text) button.setAttribute("aria-label", text);
    if (image && image.getAttribute("alt") !== text) image.setAttribute("alt", text);
  };
  // The side column mounts after this divider. Watch the layout only until it
  // exists, then only the small assistant header, never the streaming chat.
  const bind = () => {
    const head = layout.querySelector("#chat-assistant-head");
    if (!head) return false;
    layout.__rfAvatarLabel.disconnect();
    layout.__rfAvatarLabel = new MutationObserver(() => name(head));
    // A new assistant swaps the portrait's src; the selector's value is a property,
    // so focus refreshes the name too if two assistants share one portrait.
    layout.__rfAvatarLabel.observe(head, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src"],
    });
    head.addEventListener("focusin", () => name(head));
    name(head);
    return true;
  };
  layout.__rfAvatarLabel = new MutationObserver(bind);
  if (!bind()) layout.__rfAvatarLabel.observe(layout, { childList: true, subtree: true });
})();
