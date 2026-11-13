# Gallery editing and installed image tools

`gallery/ui_workspace.py` binds the [owned editor history](editor-history.md) to
the existing image widgets. **Open in Editor** validates Gallery's
selection fence, adopts the original through `editing_workflow.open_editor`, and
updates the native Editor destination and source. Chat's deliberate Open action
uses this same controller and owned history. Prompt, negative and available saved
model are published through the existing browser draft fence. Ordinary edits
adopt source dimensions only when the existing size-auto flag permits it;
manual dimensions remain owned by their controls and full-setup target replay
stays explicit. The duplicate Library form is no longer constructed. Its
source-local request builder remains available to legacy callers and utilities.
There is one opaque `editor_context`; it stores no prompt, model or raster copy.
P3 retains original/current/revision/history and the widgets retain current drafts.

`ui/app.py:create_navigation_shell` renders each existing Blocks app once, then
uses Gradio 6.31's public `unrender()`/`render()` to move the original model
containers into Models. The six top IDs are `create`, `editor`, `library`,
`models`, `chat`, `settings`; legacy Create/Chat/Library/Settings IDs stay valid.
One shared image surface is visible for Create and Editor. A destination State
belongs to the existing workspace; it holds no copied image/model/draft. Create
projects source-free intent at submission without clearing Editor's widgets.
Saved-tensor generation inputs remain available. Editor's primary action selector
and the existing `image-tools` tab (labelled **More tools**) share the same worker;
using a legacy tool resets a preceding explicit action. Gallery source handoffs
clear the previous opaque context before preparing the new source.

Settings is constructed with `models_in_settings=False`, excluding the moved
Models subtree from Settings search before navigation is bound. Its acquisition
object is constructed once. Create's model picker, card browser, components,
LoRAs and acquisition activation retain the original objects and listeners.
The native move follows the documented
[Gradio layout API](https://github.com/gradio-app/gradio/blob/main/guides/03_building-with-blocks/02_controlling-layout.md)
(Context7 `/gradio-app/gradio`, consulted 2026-11-12). Native tests check unique
layout IDs, preserved dependency references and actual route events. These are
CPU/configuration proofs, separate from rendered-browser and zoom checks.
The shared image surface is a sibling of the top tabs, so `ui.css` applies its
tokens, action/prompt layout and accessibility rules to both `#app-tabs` and
`#image-stage` through one native `:is()` scope. Neither root receives a CSS
visibility override. The primary action's `.change` listener updates presentation;
only user `.input` changes its inpaint toggle. Programmatic Gallery inpaint and
setup replay retain the toggle and mask restored by their own callbacks.

Gallery construction leaves its image values and records empty, without opening
originals, preparing thumbnails or copying a page to Gradio's cache. Opening the
Gallery tab acknowledges `request_refresh`; its existing bounded `prepare_view`
callback refreshes the incremental index and publishes the first page through
the same owner/version fence as later paging and search. A new index is populated
on that first visit. Action buttons remain disabled until a current image is
selected. `test_gallery_scheduling.py` exercises both empty and existing indexes
through native image preparation and Gradio output processing.

| Entry point                                        | Preparation and execution                                                                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Use prompt in Create                               | Original or rendered text into Create; clear old source/reference/mask state; wait for Generate                                      |
| Reuse full setup                                   | Validate saved resources and geometry; generation setup opens Create, source-bound setup opens Editor; uploads/masks remain required |
| Edit image / Inpaint / Upscale / Remove background | Open a copied selected source in Editor's existing tools; wait for Generate                                                          |
| Start image grid / Start image composition         | Seed the first ordered input from the validated original; add further images in Editor; wait for Generate                            |
| Open in Editor                                     | Adopt the exact original and saved prompt/model/target into the shared editor; Run is explicit                                       |
| Models → Use installed file in Image tools         | Revalidate the installed resource, set its operation and file atomically, open Editor/More tools; wait for Generate                  |

`gallery.editing_workflow.build_library_request` copies the current Create
controls, replaces only explicit Library intent and clears old source-dependent
inputs. It retains sampling options, active LoRAs, framing and current component
choices. A different checkpoint clears component overrides belonging to the
previous checkpoint; its own persisted component defaults remain available.
Manual output dimensions use the existing target-size plan. Upscaling and
background/depth utilities use their existing preparation and native loaders.
Rendered prompts stay literal by default. **Apply current styles and negative
processing** explicitly reuses selected styles and automatic-negative settings;
random prompt replacement remains off for the Library instruction.
The internal rendered-prompt marker prevents command dispatch, metadata-looking
JSON from changing the selected model/components, and implicit batch splitting.
Metadata payloads cannot set this private marker. Selected styles can still use
their own explicit prompt helper; unrelated One Button enhancer settings do not
prepare a helper for this Library request.

`generation.ui_request.build_ui_request` resolves input-only upscaling and PNG
preset intent for both entry points. `ui.app.generate_request` uses the existing
`append_work` → `worker.add_task` → `capture_image_data` → preparation admission
and resource-lifecycle path. Library has no second task queue, model loader or
GPU lane. Stop acts on the same request-local task ID as Create. The shared
result surfaces show new files, which enter the incremental Library index on refresh.

The workspace captures finite P4 steps through `image_workflows.capture_step`
and `worker_request`; native operations and preparation remain their existing
owners. Background selection commits an explicit no-compute step and retains
its foreground ID. Other actions use the existing worker. The final P1 plan
binding and success/history commit stay at the worker terminal boundary.
`editor_result(task_id)` is read once after the existing result event. Only a
completed current snapshot updates the same source widgets. Empty, failed,
cancelled or stale results keep the current image and draft. Native input and
context events invalidate the owner-scoped publication token and cancel the
session's active task. Two browser owners have independent tokens and histories.
Undo/redo/reopen use P3 methods and never update text/model controls.

Readiness calls the accepted request capture and passive P2 status only, then
cancels its draft ticket; it never starts preparation or a worker. Exact-frame
grayscale mask uploads bind through P3. Painted masks use the existing inpaint
working-frame adapter before being adopted as L masks. P4 owns selection versus
cutout validation, union/instance/inversion and output frame/alpha policy.

Selected records keep W05's authenticated owner, view and selection versions.
Open validates its record before and after work. Run rejects an old source or a changed or
missing file before compute admission. The copied pixels and brush belong to
the submission, so later UI/file mutations cannot change the captured request.
References absent from PNG metadata stay required; no previous upload is borrowed.
Full setup declares every restored native field and reference reset in its Gradio
event outputs, including Cascade guidance for ordinary checkpoint images. The
settings and destination update are processed in the same response. The
saved seed also uses Create's canonical visibility update in that response, so
it is editable when Random Seed is cleared. Metadata without a seed preserves
the current seed, random choice and visibility; toggling Random Seed changes
visibility without choosing a different seed.
The focused `test_gallery_setup_outputs.py` checks ordinary SDXL and Cascade setup
through the production callback and real Gradio postprocessing, and checks the
current native replay mappings against the registered output set.
Opening metadata is a deliberate native event; `capture_draft`/`publish_draft`
recheck its serial, text and widget contexts before replacing those controls.
Ordinary completion, history navigation and status events do not replace text,
models, LoRAs, components or user-selected targets. Stop uses the task ID held in
that Gradio session and the authenticated job-registry owner check.

Library inpainting uses `checkpoint_features()['inpaint']`, including current
SDXL aliases, Flux Fill, Qwen Image 2.1 and inspected native nine-channel SD1/2
contracts. Native source/mask conditioning stays with the existing pipeline.
Ordinary edits reset `native_edit_recipe` to Automatic; full setup validates and
restores the saved compatible recipe while asking for new reference uploads.

`image_editing.tool_activation` resolves callable installed upscalers, native
BiRefNet and DA3 by actual local identity and native/header contracts. It rejects
incomplete, replaced, unsupported and priority-shadowed files. Both Library and
the Hub use its operation/resource transaction. Image Tools `.input` handlers
load deliberate user defaults; `.change` handlers update presentation without
overwriting explicit resource or denoise values. Use does not start computation,
download files or persist settings.

Precision originals remain unchanged. Editing uses an explicitly described
8-bit sRGB working copy; `image_source` records original filename/SHA-256,
dimensions, source color/dtype and conversion. Browser display conversion is
separate from native export. Every Library handoff clears or replaces
`_native_main_source`, generation artifact records, original downloads and the
old result gallery in one response. An identical preview never authorizes an
older EXR/PNG16 original.

Focused CPU checks:

```sh
python tools/check_cpu.py test_library_editing_workflow.py test_library_editing_ui.py test_tool_activation.py test_gallery_actions.py test_gallery_scheduling.py test_metadata_guidance_replay.py test_image_metadata_boundary.py test_native_image_output_transport.py test_model_acquisition_ui.py test_acquisition_actions.py
```

These checks cover helper behavior, installed Gradio construction/signature
injection, exact source/alpha/mask preservation, two independent source records,
changed/deleted sources, repeat selection, native-original state invalidation and
atomic tool resources. Actual Gradio session-state checks and a delayed-source
barrier run the registered publication JavaScript in Node, preserving prompt and
negative-prompt typing; session-local Stop rejects another session's task. They
do not certify rendered browser layout, real model
execution, visual quality, GPU performance or sustained stability.
