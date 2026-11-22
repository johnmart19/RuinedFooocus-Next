let lastPickerJson = null;

function renderPicker() {
  const data = props.value || {};
  const nextJson = JSON.stringify(data);
  if (nextJson === lastPickerJson) return;
  lastPickerJson = nextJson;

  const aspectLabel = element.querySelector(".rf-aspect-label");
  const aspectGrid = element.querySelector(".rf-aspect-grid");
  const countSection = element.querySelector(".rf-count-section");
  const countLabel = element.querySelector(".rf-count-label");
  const countRow = element.querySelector(".rf-count-row");
  if (!aspectGrid || !countRow) return;

  const wasCountInputFocused =
    document.activeElement &&
    document.activeElement.classList?.contains("rf-count-input") &&
    element.contains(document.activeElement);

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
  countLabel.textContent = data.countLabel || "Images to generate";
  countRow.setAttribute("aria-label", countLabel.textContent);
  countRow.style.gridTemplateColumns = `repeat(${counts.length}, minmax(0, 1fr)) minmax(66px, 1.35fr)`;
  const rawCount = data.count === 0 ? 0 : Number(data.count || 1);
  const activeCount = Number.isFinite(rawCount) ? rawCount : 1;
  const maxCount = Number(data.maxCount || 50);
  const isCustomCount = !counts.includes(activeCount);

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

  const inp = document.createElement("input");
  inp.type = "number";
  inp.className = "rf-count-input";
  inp.min = "0";
  inp.max = String(maxCount);
  inp.step = "1";
  inp.placeholder = data.customCountLabel || "Custom";
  inp.value = isCustomCount ? String(activeCount) : "";
  inp.setAttribute("aria-label", data.imageNumberLabel || data.countLabel || "Images to generate");
  inp.title = `${data.imageNumberLabel || data.countLabel || "Images to generate"} (0 = ∞)`;
  inp.dataset.custom = String(isCustomCount);
  countRow.appendChild(inp);

  if (wasCountInputFocused) {
    inp.focus({ preventScroll: true });
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
    const val = Number(btn.dataset.count);
    for (const pill of element.querySelectorAll("button.rf-count-pill")) {
      pill.setAttribute("aria-pressed", String(pill === btn));
    }
    const inp = element.querySelector("input.rf-count-input");
    if (inp) {
      inp.value = "";
      inp.dataset.custom = "false";
    }
    trigger("click", {
      kind: "count",
      count: val,
    });
  }
});

element.addEventListener("change", (event) => {
  const inp = event.target.closest?.("input.rf-count-input");
  if (!inp) return;
  if (inp.value === "") return;
  const parsed = Number(inp.value);
  if (!Number.isFinite(parsed)) return;
  const max = Number(inp.max || 50);
  const clamped = Math.max(0, Math.min(max, Math.round(parsed)));
  const pills = Array.from(element.querySelectorAll("button.rf-count-pill"));
  let matched = false;
  for (const pill of pills) {
    const isMatch = Number(pill.dataset.count) === clamped;
    pill.setAttribute("aria-pressed", String(isMatch));
    if (isMatch) matched = true;
  }
  inp.value = matched ? "" : String(clamped);
  inp.dataset.custom = String(!matched);
  trigger("click", {
    kind: "count",
    count: clamped,
  });
});

watch("value", renderPicker);
renderPicker();
