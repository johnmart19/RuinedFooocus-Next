const fence = {
  context(reference) {
    const image = typeof reference === "object" && reference.image === true;
    const id = image ? reference.id : reference;
    const node = document.getElementById(id);
    if (!node) return null;
    const values = Array.from(node.querySelectorAll("input,textarea,select"), (input) => [
      input.value,
      input.checked,
    ]);
    if (image)
      values.push(
        ...Array.from(node.querySelectorAll("img"), (item) => [
          "image",
          item.currentSrc || item.src,
        ]),
      );
    const text = JSON.stringify(values);
    if (!node.rfPromptContext) {
      node.rfPromptContext = { revision: 0, text };
      const changed = () => {
        node.rfPromptContext.revision++;
      };
      node.addEventListener("input", changed);
      node.addEventListener("change", changed);
    }
    if (text !== node.rfPromptContext.text) {
      node.rfPromptContext.revision++;
      node.rfPromptContext.text = text;
    }
    return { id: reference, ...node.rfPromptContext };
  },
  draft(id) {
    const node = document.getElementById(id)?.querySelector("textarea");
    if (!node) return null;
    if (!node.rfDraft) {
      node.rfDraft = { revision: 0, serial: 0, text: node.value };
      node.addEventListener("input", () => {
        node.rfDraft.revision++;
        node.rfDraft.text = node.value;
      });
    }
    const state = node.rfDraft;
    if (state.text !== node.value) {
      state.revision++;
      state.text = node.value;
    }
    return state;
  },
  capture(id, operation, values, config = {}) {
    const state = this.draft(id);
    if (state && config.revoke) state.revocation = (state.revocation || 0) + 1;
    const guard = state
      ? {
          operation,
          revision: state.revision,
          serial: ++state.serial,
          text: state.text,
          revocation: state.revocation || 0,
        }
      : null;
    if (state) {
      const original =
        config.revocation_input == null
          ? null
          : values[config.revocation_input]?.[config.revocation_guard_index ?? 4];
      guard.revoked = !!original && (original.revocation || 0) !== guard.revocation;
      guard.contexts = (config.contexts || []).map((id) => this.context(id)).filter(Boolean);
      values[config.prompt || 0] = state.text;
      const history = config.history_input === null ? null : values[config.history_input];
      guard.invalidHistory =
        Array.isArray(history) &&
        history.length > config.history_revision &&
        history[config.history_revision] !== state.revision;
    }
    return [...values.slice(0, -1), guard];
  },
  publish(id, result, count, config = {}) {
    const skip = () => ({ __type__: "update" });
    const unchanged = Array.from({ length: count + 1 }, skip);
    const state = this.draft(id),
      guard = result?.guard;
    if (!state || !guard || guard.serial !== state.serial) return unchanged;
    if (guard.revoked || (guard.revocation || 0) !== (state.revocation || 0)) return unchanged;
    const contextChanged = (guard.contexts || []).some(
      (context) => JSON.stringify(this.context(context.id)) !== JSON.stringify(context),
    );
    if (config.discard_context_stale && contextChanged) return unchanged;
    if (
      guard.invalidHistory ||
      contextChanged ||
      guard.revision !== state.revision ||
      guard.text !== state.text
    ) {
      return result.stale
        ? [...result.stale, { __type__: "update", value: result.message, visible: true }]
        : [...unchanged.slice(0, -1), { __type__: "update", value: result.message, visible: true }];
    }
    // This synchronous client callback validates the live textarea immediately
    // before Gradio applies its outputs. There is no second server round trip.
    const updates = result.updates;
    const value = typeof updates[0] === "string" ? updates[0] : updates[0]?.value;
    if (typeof value === "string") {
      state.revision++;
      state.text = value;
      if (config.history_output !== null && Array.isArray(updates[config.history_output])) {
        updates[config.history_output] = [...updates[config.history_output]];
        updates[config.history_output][config.history_revision] = state.revision;
      }
    }
    return [...updates, { __type__: "update", value: "", visible: false }];
  },
};
