# Prompt data and customization

Random Prompt and configurable One Button share the maintained
`ruinedfooocus/features/prompting/one_button/` engine. They have separate UI state
and purposes; ordinary random selection needs neither an LLM nor image weights.
Optional local-chat enhancement is a separate opt-in processor.

## Prompt tools

**Random Prompt → Generate prompt** builds a draft using independent quick defaults
and the selected checkpoint's prompt format. Its collapsed Prompt options include
a seed (`-1` advances randomness; `0` is an exact seed) and append mode. **One Button
Prompt → Generate prompt** uses the visible preset and optional subject. An entered
subject overrides the preset's subject for that build without saving preset changes.
Choose Custom to reveal the collapsed Advanced Custom options; preset saves remain
explicit. Format and local-chat enhancement are optional collapsed controls.

Both actions build prompt text only. The normal image **Generate** is separate.
Applying a generated draft switches off automatic typed-prompt and image-tool prompt
replacement so a later image request uses the draft. A result produced after a draft
edit waits in Pending prompt with Apply result and Dismiss result. Apply result is
explicit and Undo restores the draft it replaced, provided no later edit would be
lost. Append mode adds a standalone `---` separator line. Neither quick randomness nor default
One Button generation calls the local Chat model; explicitly enabled enhancement does.

Only a line containing `---` (with optional spaces or tabs) separates image prompts
or Evolve parents. Inline lettering such as `"LEFT---RIGHT"`, complete quoted
multiline text, and an escaped literal line `\---` stay unchanged in one prompt.
The escape is retained. Older inline batches must put the separator on its own line;
Append and Generate reviewed variations already produce that format.
If unmatched quotes would combine across an appended or generated batch separator,
the action reports an error; close or escape the quote before trying again.

In Evolve, loading current prompts or promoting a preview to parents clears the
old candidate selection and preview. These resets do not select or apply a
candidate. Preview variations again before choosing a new candidate; invalid
selections and applying after the main draft changed still report an error.

Rewrite uses the selected Chat model only after Preview rewrite. Review the suggestion
and Apply rewrite; Undo protects subsequent edits. Suggestions cannot alter original
quoted lettering. Missing original LoRA/style/embedding references and wildcards are
retained; model-invented LoRA/style/embedding references are removed. These checks
protect literal syntax, not the semantic quality of generated wording.

Rewrite presets are text files in `prompt_presets` under the writable data root:
the first line is the display name and subsequent lines are instructions. Bundled
presets live in `ruinedfooocus/resources/prompt_presets`; legacy `llamas` files still
work. A prompt beginning with `system: ...` followed by a blank line provides a
one-off instruction. Refresh presets is under Rewrite options.

## Source precedence

Custom One Button files are read from `random_prompt/userfiles/` under
`RF_DATA_DIR`, then the legacy clone directory and former package userfiles folder.
Bundled CSVs are read-only defaults. Listing presets, building the UI or loading
configuration does not create user directories or rewrite presets. Explicit preset
saves create a writable destination.

For a logical dataset such as `vehicles`, `vehicles_replace.csv` replaces the
bundled source. An addon applies to the selected logical name: if the source is
`vehicles_light`, its addon is `vehicles_light_addon.csv`; a replaced source uses
`vehicles_replace_addon.csv`. This retains the existing selected-source behavior.
Custom subjects, artists, outfits, prefixes, middle/suffix lists and antilists use
the same user-root resolver. Existing files are never moved by discovery.

Standalone wildcard files use the separate wildcard resolver: custom
`wildcards/name.txt` entries override official entries packaged in
`ruinedfooocus/resources/wildcards/`. Use `__name__` in a prompt. Random Style and
Random Seed are separate controls; neither selects a new random prompt.

## CSV contract

CSV is UTF-8 with an optional BOM. Most lists use semicolons; the artist table
uses commas. Header behavior belongs to each logical caller, not to a filename
guess. Raw rows retain blanks, order and intentional duplicates. List projections
ignore physical empty rows. Scalar list projections retain their existing
case-insensitive deduplication. Gender filtering requires its second column;
malformed short rows produce diagnostics rather than indexing errors. Artist and
negative-word tables validate their required named columns.

Custom CSV syntax is strict. Existing packaged sources retain their legacy quoting
contract where necessary. Light and medium lists now test their actual filenames;
explicit user replacement still takes priority. Parser bug corrections are distinct
from performance changes and may deliberately change formerly broken results.

## Cache and canonical token data

The process lazily caches immutable parsed rows and ordered artist projections by
file identity and parser settings. Changed, added or removed overrides are resolved
again; callers receive fresh mutable lists. No generated prompt or stochastic
selection is cached. Raw row order, artist weights and RNG call order remain
unchanged by caching. The cache has entry and retained-object memory limits; those
limits are not a claim about total process RSS.

The two small style CSVs use a separate content-keyed parsed cache, so same-size
edits refresh even when a filesystem or editor preserves timestamps. Unchanged
style contents reuse parsed rows. This does not scan or hash the large One Button
datasets or model files during style selection or generation.

`csvfiles/tokens.csv` is the canonical token dictionary. The old
`special_lists/tokenwriter.csv` logical name resolves to it instead of shipping a
second identical 590,381-row copy. Its first row is a real token, not a header.
The export helper writes a separate destination outside the packaged sources:

```python
from ruinedfooocus.features.prompting.one_button.csv_reader import export_token_dictionary

export_token_dictionary("tokens-copy.txt")
```

Sorting/deduplication is explicit and affects only the export. No pickle database,
startup token preloading or new storage dependency is required. Plain CSV remains
the editable authoritative format; ZIP distribution already compresses it. Retain
the bundled dataset attribution and One Button licensing when redistributing.

## Prompt integrity

Random and One Button results apply directly when the draft remains current; a
changed draft requires explicit application of the pending result. Preview before
applying variation or rewrite results. An outdated preview cannot overwrite later
prompt edits, and Undo checks that its result is still the current text.
Enhancement keeps the exact original and accepts only a bounded
addition; it rejects new numeric/directive syntax and obvious replacement commands.
These transport checks do not prove that an LLM addition is semantically neutral
or that an image model will follow every detail.
