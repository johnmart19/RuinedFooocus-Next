function renderPicker() {
  const data = props.value || {};
  const aspectLabel = element.querySelector(".rf-aspect-label");
  const aspectGrid = element.querySelector(".rf-aspect-grid");
  const countSection = element.querySelector(".rf-count-section");
  const countLabel = element.querySelector(".rf-count-label");
  const countRow = element.querySelector(".rf-count-row");
  if (!aspectGrid || !countRow) return;

  aspectLabel.textContent = data.aspectLabel || "Aspect ratio";
  aspectGrid.setAttribute("aria-label", aspectLabel.textContent);
  const ratios = Array.isArray(data.ratios) ? data.ratios : [];
  const selectedDim = data.selectedDimension || "";

  aspectGrid.replaceChildren();
  for (const item of ratios) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rf-aspect-tile";
    btn.dataset.kind = "ratio";
    btn.dataset.dimension = item.dimension;
    btn.dataset.choice = item.choice || item.dimension;
    const active = item.dimension === selectedDim;
    btn.setAttribute("aria-pressed", String(active));
    btn.title = `${item.ratio} (${item.w} × ${item.h})`;

    const boxWrap = document.createElement("span");
    boxWrap.className = "rf-aspect-box-wrap";
    const box = document.createElement("span");
    box.className = "rf-aspect-box";
    const rw = Number(item.w) || 1;
    const rh = Number(item.h) || 1;
    const maxDim = 22;
    if (rw >= rh) {
      box.style.width = `${maxDim}px`;
      box.style.height = `${Math.max(9, Math.round((rh / rw) * maxDim))}px`;
    } else {
      box.style.height = `${maxDim}px`;
      box.style.width = `${Math.max(9, Math.round((rw / rh) * maxDim))}px`;
    }
    boxWrap.appendChild(box);

    const text = document.createElement("span");
    text.className = "rf-aspect-text";
    text.textContent = item.ratio;

    const dims = document.createElement("span");
    dims.className = "rf-aspect-dims";
    dims.textContent = `${item.w}×${item.h}`;

    btn.append(boxWrap, text, dims);
    aspectGrid.appendChild(btn);
  }

  const counts = Array.isArray(data.counts) ? data.counts : [1, 2, 4, 8];
  countSection.hidden = counts.length === 0;
  countLabel.textContent = data.countLabel || "Count";
  countRow.setAttribute("aria-label", countLabel.textContent);
  countRow.style.gridTemplateColumns = `repeat(${counts.length}, 1fr)`;
  const activeCount = Number(data.count || 1);

  countRow.replaceChildren();
  for (const n of counts) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rf-count-pill";
    btn.dataset.kind = "count";
    btn.dataset.count = String(n);
    btn.setAttribute("aria-pressed", String(Number(n) === activeCount));
    btn.textContent = String(n);
    countRow.appendChild(btn);
  }
}

element.addEventListener("click", (event) => {
  const btn = event.target.closest?.("button[data-kind]");
  if (!btn) return;
  if (btn.dataset.kind === "ratio") {
    for (const tile of element.querySelectorAll("button.rf-aspect-tile")) {
      tile.setAttribute("aria-pressed", String(tile === btn));
    }
    trigger("click", {
      kind: "ratio",
      dimension: btn.dataset.dimension,
      choice: btn.dataset.choice,
    });
  } else if (btn.dataset.kind === "count") {
    for (const pill of element.querySelectorAll("button.rf-count-pill")) {
      pill.setAttribute("aria-pressed", String(pill === btn));
    }
    trigger("click", {
      kind: "count",
      count: Number(btn.dataset.count),
    });
  }
});

watch("value", renderPicker);
renderPicker();
