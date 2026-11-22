let lastCompJson = null;

function renderComponentCards() {
  const data = props.value || {};
  const nextJson = JSON.stringify(data);
  if (nextJson === lastCompJson) return;
  lastCompJson = nextJson;

  const list = element.querySelector(".rf-comp-list");
  if (!list) return;

  const items = Array.isArray(data.items) ? data.items : [];
  list.replaceChildren();

  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "rf-comp-empty";
    empty.textContent = data.empty || "No external components listed.";
    list.appendChild(empty);
    return;
  }

  for (const item of items) {
    const card = document.createElement("div");
    card.className = "rf-comp-card";

    const head = document.createElement("div");
    head.className = "rf-comp-head";

    const role = document.createElement("span");
    role.className = "rf-comp-role";
    role.textContent = item.label || item.key || "";
    role.title = role.textContent;

    const pill = document.createElement("span");
    pill.className = `rf-comp-pill rf-comp-pill-${item.statusKind || "neutral"}`;
    pill.textContent = item.statusText || "";
    if (item.reason) pill.title = item.reason;

    head.append(role, pill);

    const meta = document.createElement("div");
    meta.className = "rf-comp-meta";

    const file = document.createElement("span");
    file.className = "rf-comp-file";
    file.textContent = item.filename || "";
    file.title = item.location || item.filename || "";

    const source = item.sourceUrl ? document.createElement("a") : document.createElement("span");
    source.className = "rf-comp-source";
    source.textContent = item.selection || "";
    if (item.sourceUrl) {
      source.href = item.sourceUrl;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      source.title = item.sourceUrl;
    }

    meta.append(file, source);
    card.append(head, meta);
    list.appendChild(card);
  }
}

watch("value", renderComponentCards);
renderComponentCards();
