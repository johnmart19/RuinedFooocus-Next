const dialog = element.querySelector("dialog");
function updateDialog() {
  if (props.value) {
    const select = dialog.querySelector("select");
    select.replaceChildren();
    for (const [label, path] of props.value.files) select.add(new Option(label, path));
    select.value = props.value.selected;
    const updatePath = () => {
      dialog.querySelector(".model-path").textContent = select.value;
    };
    select.onchange = updatePath;
    updatePath();
    if (!dialog.open) dialog.showModal();
  } else if (dialog.open) dialog.close();
}
dialog.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => {
    dialog.close();
    trigger("click", {
      action: button.dataset.action,
      file_index: dialog.querySelector("select").selectedIndex,
    });
  });
});
dialog.addEventListener("cancel", () => trigger("click", { action: "cancel" }));
watch("value", updateDialog);
updateDialog();
