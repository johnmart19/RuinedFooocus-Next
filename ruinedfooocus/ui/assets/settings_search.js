const status = () => element.querySelector('[role="status"]');
const list = () => element.querySelector("ul");
let displayed = [];
let navigation = 0;
let focused = 0;
let query = String(props.value || "");

function navigate(record) {
  trigger("navigate", { id: record.id, token: ++navigation, locale: props.locale });
}

async function focusTarget(target) {
  const version = target.token;
  // Native layout properties own selection/open state. Only wait for their
  // resulting field to mount; never synthesize tab or accordion clicks.
  for (let attempt = 0; attempt < 60; attempt++) {
    if (version !== navigation) return;
    const field = document.getElementById(target.id);
    if (field?.getClientRects().length) {
      await new Promise(requestAnimationFrame);
      if (version !== navigation) return;
      if (document.getElementById(target.id) !== field || !field.getClientRects().length) continue;
      const control = field.matches("button,input,textarea,select")
        ? field
        : field.querySelector(
            'input:not([type="hidden"]):not(:disabled),textarea:not(:disabled),select:not(:disabled),button:not(:disabled),[tabindex="0"]',
          );
      field.scrollIntoView({ block: "center", behavior: "instant" });
      if (control) control.focus({ preventScroll: true });
      else {
        field.tabIndex = -1;
        field.focus({ preventScroll: true });
      }
      return;
    }
    await new Promise(requestAnimationFrame);
  }
  status().textContent = props.unavailable;
}

function render() {
  list().replaceChildren();
  const words = String(props.value || "")
    .trim()
    .toLocaleLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  displayed = [];
  if (!words.length) {
    status().textContent = "";
    return;
  }
  const matches = (props.records || []).filter((record) =>
    words.every((word) =>
      `${record.label} ${record.section} ${record.terms}`.toLocaleLowerCase().includes(word),
    ),
  );
  matches.sort(
    (a, b) =>
      Number(words.every((word) => b.label.toLocaleLowerCase().includes(word))) -
      Number(words.every((word) => a.label.toLocaleLowerCase().includes(word))),
  );
  status().textContent = matches.length ? props.found : props.no_matches;
  displayed = matches.slice(0, 30);
  for (const [index, record] of displayed.entries()) {
    const row = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${record.section} › ${record.label}`;
    button.dataset.result = String(index);
    row.append(button);
    list().append(row);
  }
}
// The template can rerender after an output update. Delegate from the stable
// renderer element instead of attaching listeners to replaced result buttons.
element.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-result]");
  if (!button || !element.contains(button)) return;
  const record = displayed[Number(button.dataset.result)];
  if (record) navigate(record);
});
// Gradio 6.31 also scans quoted names in this script for custom events.
// Observe the declared properties without making them look like event names.
watch(Object.keys(props), () => {
  const nextQuery = String(props.value || "");
  if (nextQuery !== query) {
    query = nextQuery;
    navigation++;
  }
  render();
  const target = props.focus_target;
  if (target?.token === navigation && target.token !== focused) {
    focused = target.token;
    focusTarget(target);
  }
});
render();
