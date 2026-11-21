let activeGroup = "__all__";

function styleHash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const GROUP_PALETTES = {
  "General styles": [212, 260],
  "Photography": [28, 198],
  "Art styles": [332, 42],
  "Futuristic": [188, 286],
  "Games": [264, 324],
  "Papercraft": [34, 14],
  "Advertising": [14, 344],
  "Artify": [292, 174],
  "Miscellaneous": [152, 218],
  "Custom": [42, 258],
};

const GROUP_MOTIFS = {
  "Photography":
    '<circle cx="24" cy="24" r="11" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="24" cy="24" r="5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M16 11h16l3 4H13z" fill="none" stroke="currentColor" stroke-width="1.6"/>',
  "Art styles":
    '<path d="M12 31c2-9 9-16 18-19 2 8-4 17-12 20-3 1-6 1-6-1z" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="32" cy="31" r="3" fill="currentColor" opacity="0.65"/><circle cx="25" cy="34" r="2.2" fill="currentColor" opacity="0.45"/>',
  "Futuristic":
    '<polygon points="24,8 38,16 38,32 24,40 10,32 10,16" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="24" cy="24" r="4" fill="currentColor" opacity="0.75"/><path d="M24 8v12M38 16l-10 6M10 16l10 6" stroke="currentColor" stroke-width="1.4" opacity="0.6"/>',
  "Games":
    '<rect x="10" y="16" width="28" height="17" rx="8" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M17 21v7M13.5 24.5h7" stroke="currentColor" stroke-width="1.8"/><circle cx="31" cy="23" r="1.8" fill="currentColor"/><circle cx="34.5" cy="26.5" r="1.8" fill="currentColor"/>',
  "Papercraft":
    '<polygon points="24,9 38,24 24,39 10,24" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M24 9v30M10 24h28" stroke="currentColor" stroke-width="1.3" opacity="0.65"/>',
  "Advertising":
    '<rect x="10" y="11" width="28" height="26" rx="4" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M15 29l6-8 5 6 4-5 4 7" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="17" cy="17" r="2.2" fill="currentColor" opacity="0.8"/>',
  "Artify":
    '<path d="M24 8l3.5 11.5L39 23l-11.5 3.5L24 38l-3.5-11.5L9 23l11.5-3.5z" fill="none" stroke="currentColor" stroke-width="1.8"/>',
  "General styles":
    '<rect x="11" y="11" width="26" height="26" rx="5" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="19" cy="19" r="3" fill="currentColor" opacity="0.75"/><path d="M12 32l8-8 6 5 5-6 6 9" fill="none" stroke="currentColor" stroke-width="1.7"/>',
};

function swatchBackground(item) {
  const h = styleHash(item.name || item.label || "");
  const pair = GROUP_PALETTES[item.group] || [h % 360, (h + 68) % 360];
  const h1 = (pair[0] + (h % 36) - 18 + 360) % 360;
  const h2 = (pair[1] + ((h >> 6) % 42) - 21 + 360) % 360;
  const h3 = (h1 + 35) % 360;
  const angle = 120 + (h % 65);
  const rx = 20 + (h % 60);
  const ry = 18 + ((h >> 4) % 45);
  return `radial-gradient(circle at ${rx}% ${ry}%, hsla(${h3}, 85%, 64%, 0.42) 0%, transparent 62%), linear-gradient(${angle}deg, hsl(${h1} 62% 38%) 0%, hsl(${h2} 56% 20%) 100%)`;
}

function appendFallbackMotif(thumb, item) {
  thumb.style.background = swatchBackground(item);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 48 48");
  svg.setAttribute("class", "rf-style-motif");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = GROUP_MOTIFS[item.group] || GROUP_MOTIFS["General styles"];
  thumb.appendChild(svg);
}

function renderStyles() {
  const data = props.value || {};
  const tabsEl = element.querySelector(".rf-style-tabs");
  const gridEl = element.querySelector(".rf-style-grid");
  if (!tabsEl || !gridEl) return;

  const items = Array.isArray(data.items) ? data.items : [];
  const groups = Array.isArray(data.groups) ? data.groups : [];
  const selectedList = Array.isArray(data.selected) ? data.selected : [];
  const selectedRank = new Map(selectedList.map((name, idx) => [name, idx + 1]));
  const hasQuery = Boolean(data.hasQuery);
  if (hasQuery || (activeGroup !== "__all__" && !groups.includes(activeGroup))) {
    activeGroup = "__all__";
  }

  tabsEl.replaceChildren();
  const tabDefs = [{ id: "__all__", label: data.allLabel || "All" }].concat(
    groups.map((g) => ({ id: g, label: g })),
  );
  for (const tab of tabDefs) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rf-style-tab";
    btn.dataset.group = tab.id;
    btn.setAttribute("aria-pressed", String(activeGroup === tab.id));
    btn.textContent = tab.label;
    tabsEl.appendChild(btn);
  }

  const visible =
    activeGroup === "__all__" ? items : items.filter((item) => item.group === activeGroup);

  gridEl.setAttribute("aria-label", data.heading || "All styles");
  gridEl.replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "rf-style-empty";
    empty.textContent = data.empty || "No matching styles.";
    gridEl.appendChild(empty);
    return;
  }

  const frag = document.createDocumentFragment();
  for (const item of visible) {
    const rank = selectedRank.get(item.name);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rf-style-card";
    btn.dataset.styleName = item.name;
    btn.setAttribute("aria-pressed", String(Boolean(rank)));
    btn.title = `${item.label} (${item.group})`;

    const thumb = document.createElement("span");
    thumb.className = "rf-style-thumb";

    if (item.preview) {
      thumb.style.background = "var(--background-fill-secondary, #18181b)";
      const img = document.createElement("img");
      img.className = "rf-style-img";
      img.src = item.preview;
      img.alt = "";
      img.loading = "lazy";
      img.decoding = "async";
      img.onerror = () => {
        img.remove();
        appendFallbackMotif(thumb, item);
      };
      thumb.appendChild(img);
    } else {
      appendFallbackMotif(thumb, item);
    }

    if (rank) {
      const badge = document.createElement("span");
      badge.className = "rf-style-rank";
      badge.textContent = String(rank);
      thumb.appendChild(badge);
    }

    const label = document.createElement("span");
    label.className = "rf-style-label";
    label.textContent = item.label;

    btn.append(thumb, label);
    frag.appendChild(btn);
  }
  gridEl.appendChild(frag);
}

element.addEventListener("click", (event) => {
  const tab = event.target.closest?.("button.rf-style-tab");
  if (tab) {
    activeGroup = tab.dataset.group || "__all__";
    renderStyles();
    return;
  }
  const card = event.target.closest?.("button.rf-style-card");
  if (card) {
    trigger("click", { name: card.dataset.styleName });
  }
});

watch("value", renderStyles);
renderStyles();
