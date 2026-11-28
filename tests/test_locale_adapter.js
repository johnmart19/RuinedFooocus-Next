// Exercise the shipped adapter in independent browser contexts, without a server.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs
  .readFileSync(path.join(__dirname, "../ruinedfooocus/resources/translations/locale.js"), "utf8")
  .trim()
  .replace(/;$/, "")
  .replace("__CORE_ASSET__", JSON.stringify("core-test.js"))
  .replace("__DEFAULT_LOCALE__", JSON.stringify("sv"))
  .replace("await import(url.href)", "await nativeImport(url.href)");

function client(root, languages, saved, dom = {}) {
  const storage = new Map(saved ? [["ruinedfooocus-ui-language", saved]] : []);
  const state = {
    prompt: "猫 {literal} --seed 0",
    history: ["keep"],
    mask: { pixels: [1, 2] },
    settings: { unsaved: true },
  };
  const context = vm.createContext({
    URL,
    navigator: { languages },
    window: {
      gradio_config: { root, i18n_translations: { en: {}, sv: {}, de: {}, "pt-BR": {} } },
      location: { href: "http://localhost:7860/" },
    },
    localStorage: {
      getItem: (key) => storage.get(key),
      setItem: (key, value) => storage.set(key, value),
    },
    nativeImport: async (url) => {
      context.asset = url;
      return {
        changeLocale: (locale) => {
          context.locale = locale;
        },
      };
    },
    state,
    ...dom,
  });
  return {
    context,
    storage,
    state,
    run: (initialize, value) =>
      vm.runInContext(
        `(${source.replaceAll("__INITIALIZE__", String(initialize))})`,
        context,
      )(value),
  };
}

function tabDOM() {
  const counts = { scans: 0, scheduled: 0, observers: 0 };
  const pending = [];
  const tabs = [];
  const body = { nodeType: 1, closest: () => null };
  let notify;
  function tab(id, text) {
    const button = {
      nodeType: 1,
      dataset: { tabId: id },
      children: [],
      matches: () => true,
      closest() {
        return this;
      },
      querySelector: () => null,
    };
    button.childNodes = [{ nodeType: 3, textContent: text, parentElement: button }];
    return button;
  }
  return {
    counts,
    tabs,
    tab,
    body,
    notify: (records) => notify(records),
    flush() {
      for (const callback of pending.splice(0)) callback();
    },
    globals: {
      Node: { TEXT_NODE: 3, ELEMENT_NODE: 1 },
      document: {
        body,
        querySelectorAll() {
          counts.scans++;
          return tabs;
        },
      },
      MutationObserver: class {
        constructor(callback) {
          counts.observers++;
          notify = callback;
        }
        observe() {}
      },
      setTimeout(callback) {
        counts.scheduled++;
        pending.push(callback);
      },
    },
  };
}

(async () => {
  const first = client("", ["de-DE"]);
  const second = client("http://localhost:7860/proxy/", ["pt-BR"], "auto");
  const snapshot = JSON.stringify(first.state);
  assert.equal(await first.run(true), "sv");
  assert.equal(first.context.locale, "sv");
  assert.equal(first.context.asset, "http://localhost:7860/assets/core-test.js");
  assert.equal(await second.run(true), "auto");
  assert.equal(second.context.locale, "pt-BR");
  assert.equal(second.context.asset, "http://localhost:7860/proxy/assets/core-test.js");
  assert.equal(await first.run(false, "de"), "de");
  assert.equal(first.storage.get("ruinedfooocus-ui-language"), "de");
  assert.equal(second.context.locale, "pt-BR");
  assert.equal(JSON.stringify(first.state), snapshot);
  assert.equal(await first.run(false, "unknown"), "en");
  assert.equal(first.context.locale, "en");
  assert.equal(await first.run(false, "auto"), "auto");
  assert.equal(first.context.locale, "de");
  const dom = tabDOM();
  const localized = client("", ["de-DE"], "de", dom.globals);
  const config = localized.context.window.gradio_config;
  const routes = ["create", "editor", "gallery", "models", "chat", "settings"];
  config.components = routes.map((key, index) => ({
    id: index + 1,
    type: "tabitem",
    props: { label: { key } },
  }));
  config.i18n_translations.de = {
    create: "Erstellen",
    editor: "Editor",
    gallery: "Galerie",
    models: "Modelle",
    chat: "Unterhaltung",
    settings: "Einstellungen",
    installed: "Installiert",
  };
  config.i18n_translations.sv = {
    create: "Skapa",
    editor: "Redigerare",
    gallery: "Galleri",
    models: "Modeller",
    chat: "Chatt",
    settings: "Inställningar",
    installed: "Installerat",
  };
  dom.tabs.push(...routes.map((key, index) => dom.tab(String(index + 1), key)));
  await localized.run(true);
  dom.flush();
  routes.forEach((key, index) => {
    assert.equal(dom.tabs[index].childNodes[0].textContent, config.i18n_translations.de[key]);
  });
  const before = { ...dom.counts };
  const message = { nodeType: 3, parentElement: { closest: () => null } };
  const content = { nodeType: 1, matches: () => false, querySelector: () => null };
  for (let delta = 0; delta < 100; delta++) {
    dom.notify([
      { type: "characterData", target: message },
      { type: "childList", target: dom.body, addedNodes: [content] },
    ]);
    dom.flush();
  }
  console.log("Locale observer unrelated text work:", {
    changes: 100,
    tabScans: dom.counts.scans - before.scans,
    timers: dom.counts.scheduled - before.scheduled,
  });
  assert.equal(dom.counts.scans, before.scans, "Unrelated text must not rescan every tab");
  assert.equal(dom.counts.scheduled, before.scheduled);
  // Native lazy panels can add a whole tab bar after the initial page loads.
  const added = dom.tab("7", "Installed");
  dom.tabs.push(added);
  config.components.push({ id: 7, type: "tabitem", props: { label: { key: "installed" } } });
  const wrapper = { nodeType: 1, matches: () => false, querySelector: () => added };
  const marker = { nodeType: 8, parentElement: dom.body };
  for (let mutation = 0; mutation < 20; mutation++) {
    dom.notify([
      { type: "characterData", target: marker },
      { type: "childList", target: dom.body, addedNodes: [wrapper] },
    ]);
  }
  dom.flush();
  assert.equal(dom.counts.scans, before.scans + 1, "One batch of tab mounts needs one scan");
  assert.equal(added.childNodes[0].textContent, "Installiert");
  // Native tab selection can restore its old label; keep that correction reactive.
  added.childNodes[0].textContent = "Installed";
  dom.notify([{ type: "characterData", target: added.childNodes[0] }]);
  dom.flush();
  assert.equal(added.childNodes[0].textContent, "Installiert");
  added.childNodes[0].textContent = "Installed";
  dom.notify([{ type: "childList", target: added, addedNodes: added.childNodes }]);
  dom.flush();
  assert.equal(added.childNodes[0].textContent, "Installiert");
  await localized.run(false, "sv");
  dom.flush();
  routes.forEach((key, index) => {
    assert.equal(dom.tabs[index].childNodes[0].textContent, config.i18n_translations.sv[key]);
  });
  assert.equal(added.childNodes[0].textContent, "Installerat");
  assert.equal(dom.counts.observers, 1, "Locale switches reuse the existing observer");
  console.log("Locale adapter: isolated clients, persistence, Auto, assets and draft state passed");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
