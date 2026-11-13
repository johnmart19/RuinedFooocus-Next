function renderModels() {
  const details = element.querySelector("details");
  const summary = element.querySelector("summary");
  const options = element.querySelector(".model-options");
  const choices = props.value?.choices || [];
  summary.textContent =
    choices.find((item) => item[1] === props.value?.selected)?.[0] ||
    props.value?.selected_label ||
    "Choose a local model";
  options.replaceChildren();
  for (const [label, model] of choices) {
    const row = document.createElement("div");
    const choose = document.createElement("button");
    choose.textContent = label;
    choose.className = "choose-model";
    choose.setAttribute("aria-pressed", String(model === props.value?.selected));
    choose.addEventListener("click", () => {
      details.open = false;
      summary.focus();
      trigger("click", { action: "select", model });
    });
    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.className = "remove-model";
    remove.setAttribute("aria-label", "Remove " + label);
    remove.title = "Remove " + label;
    remove.addEventListener("click", () => {
      details.open = false;
      // Establish the visible return target before the confirmation opens.
      // A deferred focus here would steal focus from the native dialog.
      summary.focus();
      trigger("click", { action: "remove", model });
    });
    row.append(choose, remove);
    options.append(row);
  }
  if (!choices.length) options.textContent = "No local models. Import a GGUF to add one.";
}
watch("value", renderModels);
renderModels();
