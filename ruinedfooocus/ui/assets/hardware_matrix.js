function renderMatrix() {
  const data = props.value || {};
  const banner = element.querySelector(".rf-hw-banner");
  const groupsEl = element.querySelector(".rf-hw-groups");
  if (!groupsEl) return;

  const labels = data.labels || {};
  const cpuForced = Boolean(data.cpuForced);

  if (banner) {
    if (cpuForced && labels.cpuForced) {
      banner.textContent = labels.cpuForced;
      banner.hidden = false;
    } else {
      banner.textContent = "";
      banner.hidden = true;
    }
  }

  const groups = Array.isArray(data.groups) ? data.groups : [];
  groupsEl.replaceChildren();

  if (groups.length === 0) {
    const empty = document.createElement("div");
    empty.className = "rf-hw-empty";
    empty.textContent = labels.empty || "No supported backends detected.";
    groupsEl.appendChild(empty);
    return;
  }

  for (const group of groups) {
    const card = document.createElement("section");
    card.className = "rf-hw-group";
    card.dataset.group = group.id || "";

    const head = document.createElement("div");
    head.className = "rf-hw-group-head";

    const title = document.createElement("h3");
    title.className = "rf-hw-group-title";
    title.textContent = group.title || "";
    head.appendChild(title);

    if (group.summary) {
      const summary = document.createElement("span");
      summary.className = "rf-hw-group-summary";
      summary.textContent = group.summary;
      head.appendChild(summary);
    }
    card.appendChild(head);

    const rowsEl = document.createElement("div");
    rowsEl.className = "rf-hw-rows";
    const rows = Array.isArray(group.rows) ? group.rows : [];

    if (rows.length === 0) {
      const empty = document.createElement("div");
      empty.className = "rf-hw-empty";
      empty.textContent = labels.empty || "No supported backends detected.";
      rowsEl.appendChild(empty);
    } else {
      for (const row of rows) {
        const rowEl = document.createElement("div");
        rowEl.className = "rf-hw-row";
        rowEl.dataset.rowId = row.id || "";
        rowEl.dataset.status = row.status || "available";
        rowEl.dataset.main = row.isMain ? "true" : "false";

        const info = document.createElement("div");
        info.className = "rf-hw-info";

        const titleLine = document.createElement("div");
        titleLine.className = "rf-hw-title-line";

        const name = document.createElement("span");
        name.className = "rf-hw-name";
        name.textContent = row.label || "";
        titleLine.appendChild(name);

        if (row.runtimeLabel) {
          const tag = document.createElement("span");
          tag.className = "rf-hw-runtime-tag";
          tag.textContent = row.runtimeLabel;
          titleLine.appendChild(tag);
        }

        info.appendChild(titleLine);

        if (row.detail) {
          const detail = document.createElement("span");
          detail.className = "rf-hw-detail";
          detail.textContent = row.detail;
          info.appendChild(detail);
        }

        const right = document.createElement("div");
        right.className = "rf-hw-right";

        const badge = document.createElement("span");
        badge.className = `rf-hw-badge rf-hw-badge-${row.status || "available"}`;
        badge.textContent = row.statusLabel || "";
        right.appendChild(badge);

        if (row.isMain) {
          const mainBadge = document.createElement("span");
          mainBadge.className = "rf-hw-badge rf-hw-badge-main";
          mainBadge.textContent = labels.main || "Main";
          right.appendChild(mainBadge);
        }

        if (row.canInstall) {
          const installBtn = document.createElement("button");
          installBtn.type = "button";
          installBtn.className = "rf-hw-btn rf-hw-btn-primary";
          installBtn.dataset.action = "install";
          installBtn.dataset.id = row.id || "";
          installBtn.textContent = row.installLabel || labels.install || "Install";
          installBtn.disabled = cpuForced && row.requiresGpu !== false;
          right.appendChild(installBtn);
        }

        if (row.canSetMain && !row.isMain) {
          const mainBtn = document.createElement("button");
          mainBtn.type = "button";
          mainBtn.className = "rf-hw-btn rf-hw-btn-secondary";
          mainBtn.dataset.action = "set_main";
          mainBtn.dataset.id = row.id || "";
          mainBtn.textContent = labels.setMain || "Set as main";
          mainBtn.disabled = cpuForced;
          right.appendChild(mainBtn);
        }

        rowEl.append(info, right);
        rowsEl.appendChild(rowEl);
      }
    }

    card.appendChild(rowsEl);
    groupsEl.appendChild(card);
  }
}

element.addEventListener("click", (event) => {
  const btn = event.target.closest?.("button[data-action]");
  if (!btn || btn.disabled) return;
  trigger("click", {
    action: btn.dataset.action,
    id: btn.dataset.id,
  });
});

watch("value", renderMatrix);
renderMatrix();
