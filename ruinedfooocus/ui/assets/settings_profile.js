const status = () => element.querySelector('[role="status"]');
const terms = () => element.querySelector("dl");
function draftStatus() {
  const dirty = JSON.stringify(props.value) !== JSON.stringify(props.saved);
  status().textContent = dirty ? props.dirty_label : props.clean_label;
  element.dataset.dirty = String(dirty);
}
function render() {
  terms().replaceChildren();
  for (const [label, value] of [
    [props.running_label, props.running],
    [props.saved_label, props.profile],
    [props.next_label, props.next],
  ]) {
    const term = document.createElement("dt"),
      description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    terms().append(term, description);
  }
  element.querySelector(".apply-timing").textContent = props.timing;
  draftStatus();
}
// Keep watched property names distinct from Gradio's quoted custom-event names.
watch(Object.keys(props), render);
render();
