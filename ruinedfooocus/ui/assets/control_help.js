/* Native disclosures keep optional hints available without covering controls. */
(() => {
  if (window.__rfControlHelp || !document.body || typeof MutationObserver !== "function") return;
  const scope = "#app-tabs, #image-stage";
  const relevant = '.info-text, [data-testid="block-info"], .label-text';
  const pending = new Set();
  let scheduled = false;
  let sequence = 0;

  const prepare = (host) => {
    if (!host?.isConnected || !host.closest(scope)) return;
    const info = [...host.querySelectorAll(".info-text")].find(
      (candidate) => candidate.closest(".block") === host,
    );
    let disclosure = host.querySelector(":scope > .rf-control-help");
    if (!info?.textContent.trim()) {
      disclosure?.remove();
      host.classList.remove("rf-help-host");
      return;
    }
    if (!disclosure) {
      disclosure = document.createElement("details");
      disclosure.className = "rf-control-help";
      const summary = document.createElement("summary");
      summary.textContent = "?";
      summary.setAttribute("role", "button");
      disclosure.append(summary);
      // Do not move native Gradio nodes or put another button inside its label.
      host.append(disclosure);
      host.classList.add("rf-help-host");
    }
    const label = host.querySelector('[data-testid="block-info"], .label-text');
    const name = [label?.textContent.trim(), info.textContent.trim()].filter(Boolean).join(": ");
    if (!info.id) info.id = `rf-control-hint-${++sequence}`;
    disclosure.firstElementChild.setAttribute("aria-label", name);
    disclosure.firstElementChild.setAttribute("aria-controls", info.id);
    disclosure.firstElementChild.setAttribute("aria-expanded", String(disclosure.open));
  };

  const queueHost = (element, fallback) => {
    const host = element.closest(".block") || fallback?.closest(".block");
    if (host?.isConnected && host.closest(scope)) pending.add(host);
  };

  const collectSubtree = (node, fallback) => {
    if (node.nodeType !== 1 || node.closest(".rf-control-help")) return;
    if (node.matches(relevant)) queueHost(node, fallback);
    else node.querySelectorAll(relevant).forEach((element) => queueHost(element, fallback));
  };

  const observer = new MutationObserver((changes) => {
    for (const change of changes) {
      const target = change.target.nodeType === 1 ? change.target : change.target.parentElement;
      const owner = target?.closest(relevant);
      if (owner) queueHost(owner);
      // Streaming status/JSON text is unrelated. Inspect only changed subtrees,
      // never their existing parent, to discover newly added or removed hints.
      if (change.type === "childList" && !owner) {
        change.addedNodes.forEach((node) => collectSubtree(node, target));
        change.removedNodes.forEach((node) => collectSubtree(node, target));
      }
    }
    if (scheduled || !pending.size) return;
    scheduled = true;
    queueMicrotask(() => {
      const hosts = [...pending];
      pending.clear();
      scheduled = false;
      hosts.forEach(prepare);
    });
  });
  window.__rfControlHelp = observer;
  document.addEventListener(
    "toggle",
    (event) => {
      const disclosure = event.target;
      if (!disclosure.matches?.(".rf-control-help")) return;
      disclosure.firstElementChild?.setAttribute("aria-expanded", String(disclosure.open));
    },
    true,
  );
  observer.observe(document.body, { childList: true, characterData: true, subtree: true });
  document.querySelectorAll(`:is(${scope}) .info-text`).forEach((info) => {
    const host = info.closest(".block");
    if (host) pending.add(host);
  });
  pending.forEach(prepare);
  pending.clear();
})();
