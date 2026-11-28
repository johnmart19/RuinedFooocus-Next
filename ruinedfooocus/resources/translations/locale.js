async (value) => {
  // Import the same module/store as Gradio's own settings UI.
  const root = window.gradio_config?.root || "";
  const url = new URL(`${root.replace(/\/$/, "")}/assets/${__CORE_ASSET__}`, window.location.href);
  const native = await import(url.href);
  const allowed = new Set(["auto", ...Object.keys(window.gradio_config?.i18n_translations || {})]);
  const defaultLocale = __DEFAULT_LOCALE__;
  let chosen = value;
  if (__INITIALIZE__) {
    try {
      chosen = localStorage.getItem("ruinedfooocus-ui-language") || defaultLocale;
    } catch {
      chosen = defaultLocale;
    }
  }
  if (!allowed.has(chosen)) chosen = "en";
  if (!__INITIALIZE__) {
    try {
      localStorage.setItem("ruinedfooocus-ui-language", chosen);
    } catch {}
  }
  let locale = chosen;
  if (chosen === "auto") {
    const browser = navigator.languages || [navigator.language || "en"];
    locale =
      browser.flatMap((code) => [code, code.split("-")[0]]).find((code) => allowed.has(code)) ||
      "en";
  }
  native.changeLocale(locale);
  // Server-built text (model defaults, counts, statuses) reads the same choice.
  if (typeof document !== "undefined") {
    document.cookie = `rf_ui_locale=${encodeURIComponent(locale)}; path=/; max-age=31536000; SameSite=Lax`;
    // Gradio 6.31 draws tab labels in the page's first locale and re-renders only a
    // tab that gets selected afterwards. Relabel tab buttons from the same catalogue
    // now and whenever Gradio adds or redraws tab bars (lazy panels).
    window.__rfLocale = locale;
    if (!window.__rfTabRelabel) {
      // Built on each run: the config can still be incomplete when this first runs.
      const keysForTabs = () => {
        const tabKeys = new Map();
        for (const component of window.gradio_config?.components || []) {
          if (component.type !== "tabitem") continue;
          const label = component.props?.label;
          const key =
            typeof label === "string" ? label.match(/"key":\s*"([^"]+)"/)?.[1] : label?.key;
          if (!key) continue;
          tabKeys.set(String(component.id), key);
          if (component.props?.id != null) tabKeys.set(String(component.props.id), key);
        }
        return tabKeys;
      };
      const tabSelector = "[role=tab][data-tab-id]";
      const relabel = () => {
        const current = window.__rfLocale || "en";
        const catalogues = window.gradio_config?.i18n_translations || {};
        const tabKeys = keysForTabs();
        for (const button of document.querySelectorAll(tabSelector)) {
          const key = tabKeys.get(button.dataset.tabId);
          const text = key && (catalogues[current]?.[key] ?? catalogues.en?.[key]);
          if (!text) continue;
          const target = [...button.childNodes].find(
            (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim(),
          );
          if (target && target.textContent !== text) target.textContent = text;
          else if (!target && !button.children.length && button.textContent !== text) {
            button.textContent = text;
          }
        }
      };
      let pending = false;
      const schedule = () => {
        if (pending) return;
        pending = true;
        setTimeout(() => {
          pending = false;
          relabel();
        }, 30);
      };
      const affectsTabs = (record) => {
        const target =
          record.target.nodeType === Node.ELEMENT_NODE
            ? record.target
            : record.target.parentElement;
        if (target?.closest(tabSelector)) return true;
        // Streaming messages, cards and progress text do not redraw tab labels.
        // A lazy panel may mount a whole tab bar, so inspect added subtrees too.
        return (
          record.type === "childList" &&
          Array.from(record.addedNodes).some(
            (node) =>
              node.nodeType === Node.ELEMENT_NODE &&
              (node.matches(tabSelector) || node.querySelector(tabSelector)),
          )
        );
      };
      new MutationObserver((records) => {
        if (records.some(affectsTabs)) schedule();
      }).observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
      });
      window.__rfTabRelabel = schedule;
    }
    window.__rfTabRelabel();
  }
  return chosen;
};
