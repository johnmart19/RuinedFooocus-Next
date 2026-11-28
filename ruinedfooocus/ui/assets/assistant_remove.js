const dialog = element.querySelector("dialog");
function updateDialog() {
  if (props.value) {
    dialog.querySelector(".assistant-name").textContent = props.value.name;
    if (!dialog.open) dialog.showModal();
  } else if (dialog.open) dialog.close();
}
dialog.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => {
    dialog.close();
    trigger("click", { action: button.dataset.action });
  });
});
dialog.addEventListener("cancel", () => trigger("click", { action: "cancel" }));
watch("value", updateDialog);
updateDialog();
