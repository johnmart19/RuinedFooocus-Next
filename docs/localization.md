# UI languages

**Settings → Appearance & language → Language for this browser** changes display
text in place. It uses Gradio's native reactive locale store, without rebuilding
the page. Prompt drafts, Chat history, uploaded references, masks, model IDs and
unsaved controls are not translated or reset. The browser remembers this choice.
**Default language for this profile** is a saved startup default; `--language`
overrides it. **Auto / browser** uses the browser's preferred supported locale.

The authenticated Guest interface attaches the same native catalogue and startup
locale initializer. Its labels resolve after login using the remembered browser
choice or the selected profile/CLI default, with English fallback. Guest mode
does not expose the owner's Settings or language picker, and this translation
setup does not expand its API, folder, installation or media permissions.

English is the complete fallback. Swedish, German, French, Spanish, Portuguese
(Brazil), Ukrainian, Russian, Japanese, Simplified Chinese, Traditional Chinese,
Korean, Italian, Polish, Dutch and Turkish cover every current English message ID
in the shipped UI catalogues. All are marked **draft**: they have not received
native-speaker linguistic review. English fallback remains available for missing
or malformed custom locale data. Backend exceptions, native diagnostics and
dynamic runtime messages may remain English rather than leaking frontend
translation markers into error messages. Complete catalogue coverage does not
mean that every visible runtime string is translated.

The current Library thumbnail, live-theme feedback and automatic backend
installation messages are included in these draft catalogues. Their new
translations still require human linguistic review.

Coverage includes the one-step Editor, explicit prompt tools, primary page
headings and Chat image controls. A shipped-catalogue regression follows the
current English source IDs and placeholder counts, so coverage is not tied to a
fixed release message count. Custom overrides can still be partial.

UI locale has no bearing on a model's language capabilities. No locale selection
rewrites prompts, model filenames, sampler protocol values, dataset rows or saved
metadata. Adding additional locales uses the same catalogue pipeline; no
model-specific language branch is required.

## Catalogue ownership

The new image adjustment, mask, color, Boogu/Krea and Bernini controls also have
AI-drafted translations in the existing locales. They have not received
native-speaker review; model names, file identifiers and operation values stay
literal.

Crop to selection, text, morphology, HDR, MoGe output and face region controls
use the same draft catalogue pipeline. Their operation values, precision labels
and OpenGL/DirectX identifiers remain literal; new drafts still require human
linguistic review.

The authoritative catalogues are in `ruinedfooocus/resources/translations/`:

- `en.json` owns feature-prefixed message IDs and source wording.
- `<locale>.json` supplies reviewed or draft translations of those IDs.
- `manifest.json` declares display names and review status.
- `legacy-keys.json` maps older English-key calls to stable IDs. Existing custom
  `language/*.json` overrides in the writable data root remain read-only inputs.

There is no duplicate shipped `resources/language` catalogue. Catalogues are
validated for string values, duplicate IDs and matching placeholders, conversions
and format specifications. Missing or malformed locale data falls back to English.
The application never calls an LLM or a translation service to display the UI.

Gradio 6.31's tab-label adapter uses one browser-local observer for label redraws
and lazily mounted tab bars. Locale changes still relabel every tab from the
current catalogue. Unrelated body text and child updates, including Chat tokens,
card artwork and progress messages, do not schedule a full tab scan. A burst of
relevant tab changes is coalesced before relabeling; keys are read from the current
configuration so late-mounted panels retain their translations. The CPU DOM
fixture checks these work counts and label recovery; rendered browser latency
requires a separate profile.

## Contribute

The adjustment panel's local `field` helper and its three declared utility enum
sets have a bounded source-only extraction path. It registers the labels users
see, including generated direction labels and choice captions, without importing
the services or adding wire values and uploaded content to the catalogue.

```sh
python tools/check_translations.py --extract
python tools/check_translations.py
python tools/check_cpu.py test_localization.py
```

Extraction registers new presentation strings without rewriting existing IDs or
translations. Review generated IDs before committing; keep protocol/user values
out of the extraction set. Register static labels, help, placeholders and buttons
with `translate_ui`; native Gradio performs per-client resolution. Dynamic messages
can use `TranslationManager.translate(..., locale=..., escape=True)` when rendering
user-provided values as HTML. Never change a shared server locale from a client
callback.

For a new locale, add its catalogue and manifest entry, preserve all placeholders,
then verify complete English fallback, core workflows, Unicode and expanded labels
at mobile and desktop sizes. Declare linguistic review honestly. Tests cover the
installed native asset contract and independent client state; browser checks are
still required after a Gradio update. The adapter deliberately fails clearly if
the installed package no longer has the expected native locale export.

The image/Chat scope cleanup removes only retired media presentation messages
that have no remaining reference in the combined live UI/source. Shared image
messages, frame/3D style wording, Chat personas and prompt datasets are retained.
New source wording falls back to English until a locale translation is reviewed.
