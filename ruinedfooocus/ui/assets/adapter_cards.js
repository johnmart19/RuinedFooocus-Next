const dragType = "application/x-ruinedfooocus-adapter";
let selected = "";
let renderedValue = props.value;
function setPreview(media, url, fallback, badge, isDefault = false) {
  const markDefault = (value) => {
    if (badge) badge.hidden = !value || !badge.textContent;
  };
  const preview = media.querySelector("img") || document.createElement("img");
  if (preview.getAttribute("src") !== url) preview.src = url;
  preview.alt = "";
  preview.draggable = false;
  preview.loading = "lazy";
  markDefault(isDefault || url === fallback);
  preview.onerror = () => {
    if (fallback) {
      markDefault(true);
      if (preview.getAttribute("src") !== fallback) preview.src = fallback;
    }
  };
  if (!media.contains(preview)) media.replaceChildren(preview);
}
function currentView(data) {
  const versions = window.__rfModelUIRequests || {};
  return [data.requestIdentity, data.pageIdentity].every(
    (identity) => !identity || identity.version >= (versions[identity.channel] || 0),
  );
}
function renderPager(data) {
  const pager = element.querySelector(".adapter-card-pager");
  if (!pager) return;
  const value = data.pagination;
  pager.hidden = !value || value.pages <= 1;
  if (!value) return;
  const previous = pager.querySelector('[data-page-offset="-1"]');
  const next = pager.querySelector('[data-page-offset="1"]');
  previous.textContent = value.previous;
  next.textContent = value.next;
  previous.disabled = value.page <= 1;
  next.disabled = value.page >= value.pages;
  pager.querySelector('[role="status"]').textContent = value.label;
  pager.setAttribute("aria-label", value.label);
}
function renderCards() {
  const data = props.value;
  const versions = window.__rfModelUIRequests || {};
  if (!currentView(data)) return;
  if (Array.isArray(data.previewPatch)) {
    const visible = new Map(
      Array.from(element.querySelectorAll("button[data-adapter]")).map((card) => [
        card.dataset.adapter,
        card,
      ]),
    );
    for (const patch of data.previewPatch) {
      const button = visible.get(patch.key);
      if (!button || button.dataset.artwork !== patch.artworkKey || !patch.preview?.url) continue;
      const media = button.querySelector(".adapter-card-media");
      setPreview(
        media,
        patch.preview.url,
        button.dataset.fallback,
        button.querySelector(".adapter-card-default"),
        patch.preview_is_default,
      );
    }
    return;
  }
  renderPager(data);
  const selection = data.selectionIdentity;
  if (
    Object.prototype.hasOwnProperty.call(data, "selected") &&
    (!selection || selection.version >= (versions[selection.channel] || 0))
  ) {
    selected = data.selected;
  }
  const heading = element.querySelector(".adapter-card-heading");
  const grid = element.querySelector(".adapter-card-grid");
  heading.textContent = data.heading;
  grid.setAttribute("aria-label", data.heading);
  grid.classList.toggle("adapter-drop-zone", data.acceptsDrop);
  const previouslyFocused = document.activeElement?.dataset.adapter;
  const scrollTop = grid.scrollTop;
  const scrollLeft = grid.scrollLeft;
  const existing = new Map(
    Array.from(grid.querySelectorAll("button[data-adapter]")).map((card) => [
      card.dataset.adapter,
      card,
    ]),
  );
  grid.querySelector(".adapter-card-empty")?.remove();
  const retained = new Set();
  const rendered = new Map();
  for (const card of data.cards) {
    const button = existing.get(card.key) || document.createElement("button");
    rendered.set(card.key, button);
    retained.add(card.key);
    button.type = "button";
    button.className = "adapter-card";
    button.dataset.adapter = card.key;
    button.dataset.artwork = card.artworkKey || "";
    button.dataset.fallback = card.fallback?.url || "";
    button.draggable = Boolean(data.acceptsDrag);
    button.setAttribute("aria-pressed", String(card.key === selected));
    const media = button.querySelector(".adapter-card-media") || document.createElement("span");
    media.className = "adapter-card-media";
    const defaultBadge =
      button.querySelector(".adapter-card-default") || document.createElement("span");
    defaultBadge.className = "adapter-card-default";
    defaultBadge.textContent = data.defaultPreviewLabel || "";
    defaultBadge.hidden = true;
    if (card.preview?.url || button.dataset.fallback) {
      setPreview(
        media,
        card.preview?.url || button.dataset.fallback,
        button.dataset.fallback,
        defaultBadge,
        card.preview_is_default,
      );
    } else {
      media.textContent = card.kind || data.heading;
    }
    const name = button.querySelector(".adapter-card-name") || document.createElement("span");
    name.className = "adapter-card-name";
    name.textContent = card.label;
    const badge = button.querySelector(".adapter-card-badge") || document.createElement("span");
    badge.className = "adapter-card-badge";
    badge.textContent = [card.kind, card.state].filter(Boolean).join(" | ");
    badge.hidden = !badge.textContent;
    if (!button.contains(media)) button.append(media, name, badge);
    if (!button.contains(defaultBadge)) button.append(defaultBadge);
    if (!grid.contains(button)) grid.append(button);
    if (card.key === previouslyFocused) button.focus({ preventScroll: true });
  }
  for (const [key, button] of existing) {
    if (!retained.has(key)) button.remove();
  }
  // Reorder only when a filter actually changes the order; preview refreshes keep
  // the existing nodes, focus, image decoding and nested scroll position intact.
  data.cards.forEach((card, index) => {
    const button = rendered.get(card.key);
    if (grid.children[index] !== button) grid.insertBefore(button, grid.children[index] || null);
  });
  if (!data.cards.length) {
    const empty = document.createElement("span");
    empty.className = "adapter-card-empty";
    empty.textContent = data.empty;
    grid.append(empty);
  }
  grid.scrollTop = scrollTop;
  grid.scrollLeft = scrollLeft;
  renderedValue = data;
}

function choose(key, action) {
  if (!currentView(renderedValue)) return;
  selected = key;
  for (const card of element.querySelectorAll("[data-adapter]")) {
    card.setAttribute("aria-pressed", String(card.dataset.adapter === key));
  }
  trigger("click", { key, action });
}
function adapterAt(event) {
  return event.target.closest?.("button[data-adapter]");
}

element.addEventListener("click", (event) => {
  const page = event.target.closest?.("button[data-page-offset]");
  if (page) {
    if (page.disabled || !currentView(renderedValue)) return;
    const value = renderedValue.pagination;
    if (!value) return;
    const target = value.page + Number(page.dataset.pageOffset);
    if (target >= 1 && target <= value.pages) {
      trigger("paginate", {
        page: target,
        viewVersion: renderedValue.requestIdentity?.version,
      });
    }
    return;
  }
  const card = adapterAt(event);
  if (card && event.detail <= 1) choose(card.dataset.adapter, "select");
});
element.addEventListener("dblclick", (event) => {
  const card = adapterAt(event);
  if (!card || props.value.acceptsDrop) return;
  event.preventDefault();
  choose(card.dataset.adapter, "activate");
});
element.addEventListener("dragstart", (event) => {
  const card = adapterAt(event);
  if (!card || !props.value.acceptsDrag || !event.dataTransfer) return;
  event.dataTransfer.effectAllowed = "copy";
  event.dataTransfer.setData(dragType, card.dataset.adapter);
});
element.addEventListener("dragover", (event) => {
  if (!props.value.acceptsDrop || !event.dataTransfer?.types.includes(dragType)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
  element.querySelector(".adapter-card-grid").classList.add("adapter-drag-over");
});
element.addEventListener("dragleave", (event) => {
  if (!element.contains(event.relatedTarget)) {
    element.querySelector(".adapter-card-grid").classList.remove("adapter-drag-over");
  }
});
element.addEventListener("drop", (event) => {
  if (!props.value.acceptsDrop || !event.dataTransfer?.types.includes(dragType)) return;
  event.preventDefault();
  element.querySelector(".adapter-card-grid").classList.remove("adapter-drag-over");
  choose(event.dataTransfer.getData(dragType), "activate");
});
watch("value", renderCards);
renderCards();
