# Shared model-card preview labels

The `ui/assets/adapter_cards.*` renderer uses the payload's localized
`defaultPreviewLabel` and each card's `preview_is_default` flag to identify category
art. `preview` and `fallback` retain native Gradio FileData URLs. Equality with the
fallback URL also identifies default art. A failed publisher-image decode selects
the fallback once and shows the same label; a failed fallback does not retry.

The label is a visible text span inside the card button, alongside its name and
kind/state text. It contributes to the button's accessible name without announcing
every background artwork update. Images remain decorative, with empty alt text.
Localized labels use text nodes, including after locale changes.

A matching `previewPatch` reuses the existing card and image nodes, preserves
focus and scroll, and clears the default label for publisher artwork. A patch must
match both card key and artwork identity. An explicit default flag or fallback URL
keeps the label visible. Older payloads that omit `defaultPreviewLabel` keep their
existing presentation. Feature callers own the localized text and preview flags;
the renderer does not infer provenance from a filename or invent a translation.

```sh
python tools/check_cpu.py test_card_preview_badge.py test_checkpoint_cards.py test_compact_model_profile_controls.py test_model_ui_requests.py
```

These checks execute the production renderer in a bounded DOM fixture and verify
request fencing and native caller contracts. They do not establish rendered layout,
image quality, mobile behavior or actual browser zoom. Browser acceptance remains
blocked while the in-app provider is unavailable.
