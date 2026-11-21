let lastPayloadJson = null;

function renderSamplingTable() {
  const data = props.value || {};
  const nextJson = JSON.stringify(data);
  if (nextJson === lastPayloadJson) return;
  lastPayloadJson = nextJson;

  const noticeEl = element.querySelector(".rf-sampling-notice");
  const thSetting = element.querySelector(".rf-st-col-setting");
  const thValue = element.querySelector(".rf-st-col-value");
  const thSource = element.querySelector(".rf-st-col-source");
  const tbody = element.querySelector(".rf-sampling-body");
  if (!tbody) return;

  const activeEl = element.contains(document.activeElement) ? document.activeElement : null;
  const focusField = activeEl?.dataset?.field || null;

  const headers = data.headers || {};
  if (thSetting) thSetting.textContent = headers.setting || "Setting";
  if (thValue) thValue.textContent = headers.value || "Value";
  if (thSource) thSource.textContent = headers.source || "Source";

  if (noticeEl) {
    if (data.notice) {
      noticeEl.textContent = data.notice;
      noticeEl.hidden = false;
    } else {
      noticeEl.textContent = "";
      noticeEl.hidden = true;
    }
  }

  const rows = Array.isArray(data.rows) ? data.rows : [];
  const resetTitle = data.resetLabel || "Reset to default";
  tbody.replaceChildren();

  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.dataset.field = row.key || "";

    // Col 1: Setting label
    const tdLabel = document.createElement("td");
    const labelSpan = document.createElement("span");
    labelSpan.className = "rf-st-label";
    labelSpan.textContent = row.label || row.key || "";
    if (row.info) labelSpan.title = row.info;
    tdLabel.appendChild(labelSpan);

    // Col 2: Direct editable control + optional single-row reset button
    const tdControl = document.createElement("td");
    const controlWrap = document.createElement("div");
    controlWrap.className = "rf-st-control-cell";

    if (row.kind === "select") {
      const sel = document.createElement("select");
      sel.className = "rf-st-select";
      sel.dataset.field = row.key;
      sel.setAttribute("aria-label", row.label || row.key);
      const choices = Array.isArray(row.choices) ? row.choices : [];
      const currentStr = String(row.value ?? "");
      let seen = false;
      for (const ch of choices) {
        const opt = document.createElement("option");
        const val = Array.isArray(ch) ? ch[1] : ch;
        const lbl = Array.isArray(ch) ? ch[0] : ch;
        opt.value = String(val);
        opt.textContent = String(lbl);
        if (opt.value === currentStr) {
          opt.selected = true;
          seen = true;
        }
        sel.appendChild(opt);
      }
      if (!seen && currentStr !== "") {
        const opt = document.createElement("option");
        opt.value = currentStr;
        opt.textContent = currentStr;
        opt.selected = true;
        sel.appendChild(opt);
      }
      controlWrap.appendChild(sel);
    } else if (row.kind === "text") {
      const span = document.createElement("span");
      span.className = "rf-st-static";
      span.textContent = row.value !== undefined && row.value !== null ? String(row.value) : "";
      controlWrap.appendChild(span);
    } else {
      const inp = document.createElement("input");
      inp.type = "number";
      inp.className = "rf-st-input";
      inp.dataset.field = row.key;
      inp.setAttribute("aria-label", row.label || row.key);
      if (row.min !== undefined) inp.min = String(row.min);
      if (row.max !== undefined) inp.max = String(row.max);
      if (row.step !== undefined) inp.step = String(row.step);
      inp.value = row.value !== undefined && row.value !== null ? String(row.value) : "";
      controlWrap.appendChild(inp);
    }

    if (row.canReset && row.defaultValue !== undefined && row.defaultValue !== null) {
      const resetBtn = document.createElement("button");
      resetBtn.type = "button";
      resetBtn.className = "rf-st-reset";
      resetBtn.dataset.resetField = row.key;
      resetBtn.dataset.resetValue = String(row.defaultValue);
      resetBtn.title = `${resetTitle}: ${row.defaultValue}`;
      resetBtn.textContent = "↺";
      controlWrap.appendChild(resetBtn);
    }

    tdControl.appendChild(controlWrap);

    // Col 3: Compact source badge (clickable link when URL exists)
    const tdSource = document.createElement("td");
    const srcWrap = document.createElement("div");
    srcWrap.className = "rf-st-source-wrap";

    const badgeKind = row.sourceKind || "preset";
    const badgeEl = row.sourceUrl ? document.createElement("a") : document.createElement("span");
    badgeEl.className = `rf-st-badge rf-st-badge-${badgeKind}`;
    badgeEl.textContent = row.sourceLabel || "";
    if (row.sourceTitle) badgeEl.title = row.sourceTitle;
    if (row.sourceUrl) {
      badgeEl.href = row.sourceUrl;
      badgeEl.target = "_blank";
      badgeEl.rel = "noopener noreferrer";
    }
    srcWrap.appendChild(badgeEl);
    tdSource.appendChild(srcWrap);

    tr.append(tdLabel, tdControl, tdSource);
    tbody.appendChild(tr);
  }

  if (focusField) {
    const nextFocus = tbody.querySelector(`[data-field="${CSS.escape(focusField)}"]`);
    if (nextFocus && document.activeElement !== nextFocus) {
      nextFocus.focus({ preventScroll: true });
    }
  }
}

element.addEventListener("change", (event) => {
  const target = event.target;
  if (!target || !target.dataset || !target.dataset.field) return;
  const field = target.dataset.field;
  let value = target.value;
  if (target.type === "number") {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    const min = target.min !== "" ? Number(target.min) : -Infinity;
    const max = target.max !== "" ? Number(target.max) : Infinity;
    const clamped = Math.min(max, Math.max(min, parsed));
    value = target.step === "1" ? Math.round(clamped) : Math.round(clamped * 10) / 10;
    target.value = String(value);
  }
  trigger("click", { action: "edit", field, value });
});

element.addEventListener("click", (event) => {
  const btn = event.target.closest?.("button.rf-st-reset");
  if (!btn) return;
  const field = btn.dataset.resetField;
  const raw = btn.dataset.resetValue;
  if (!field) return;
  trigger("click", { action: "reset_field", field, value: raw });
});

watch("value", renderSamplingTable);
renderSamplingTable();
