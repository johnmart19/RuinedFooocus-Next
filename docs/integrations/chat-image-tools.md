# Local Chat image-tool contracts

The bounded image tools are `generate_image`, `edit_image`, `process_image` and
optional `inspect_image`. The separate `propose_create_prompt` tool produces a
prompt proposal without preparing or executing an image. Schema availability follows the initialized local
model/runtime/template and the app's current selected operations. A model family
name, artifact digest or template parser does not establish model behavior,
persona accuracy, visual quality or device qualification.

The app owns **text**, **Preview** or **Run** intent. A model argument cannot
change that intent, choose another owner, change image settings, download a model
or install a runtime. Prompt-only requests use text or an independently authorized
Create prompt proposal; critique remains text. The schemas
permit one tool call per turn and no automatic critique/regenerate cycle.

`generate_image` receives only the current visual prompt. Its validated request
clears selected source, mask, controls and editor lineage. Captured negative
prompt, size, denoise, LoRAs, alpha policy, model/components and effective-plan
data remain app-owned. `edit_image` receives a prompt, the current selected opaque
image ID and an admitted edit operation. Inpaint additionally requires the
selected mask ID; other edit operations require null for that field when it is
present. `process_image` receives the current image ID and one admitted utility.
Utilities do not claim diffusion execution. The shared editor service checks
owner, source revision, geometry, alpha and immutable artifact identity again.

`inspect_image` is available only when the app supplies a selected compatible
initialized vision reviewer. It reports actual pixel observations. Generation
metadata, filenames and inherited prompts are reference data, not vision.
Grounded attachment turns exclude image tools and retain their existing bounded
retrieval and source-provenance controls. Assistant/system/avatar stores, styles,
Random Prompt, One Button and optional rewriting keep their existing owners.

`llama_vision.reference_capabilities` separates source applicability from image
generation permission: compatible vision review can keep the reference selector
visible while generation is Off. Upload is shown only for Uploaded image;
denoise applies only to an available non-native edit source. Selected source,
available pixels, model vision support and initialized projector are distinct.

`attachments.attachment_capabilities` is the picker/help projection of the
actual bounded parser set: `.txt`, `.md`, `.csv`, `.json`, `.docx`, `.rag`, plus
`.pdf` only when the pinned parser metadata matches. Limits remain four files,
256 KiB each and 1 MiB total; choose a `.rag` manifest alone. PDF supports
extractable text, at most 32 pages, without OCR, scanned or encrypted documents.
DOCX extracts document text through the bounded standard-library XML/ZIP child.
Neither this projection nor document parsing prepares a model or probes a GPU.

## Using image actions in Chat

Chat captures three independent permissions for each submitted turn:

- **Image generation** allows one image action using captured Create settings.
  It starts Off and resets for the next turn. Backend Text/Preview/Run intent
  remains supported for existing explicit callers.
- **Rewrite image prompts** allows rewording an explicitly supplied visual
  prompt. It starts Off. Off preserves that target exactly, including spacing,
  and bypasses later style/wildcard/OBP rewrites through the existing literal
  prompt boundary. A new image request instead uses a bounded visual prompt
  constructed from the current request, never the whole raw conversation or an
  unrelated Create draft.
- **Update Create prompt** allows a prompt proposal to update Create. It starts
  Off, does not authorize image generation, and works with generation Off.
  When generation is also On, the completed image action can carry its same
  validated visual prompt as a proposal without making a second tool call.

Existing explicit choices are retained; the legacy `enable_llm_tools` preference
grants neither new prompt permission. A supplied prompt target is explicit;
selecting a reference does not import its old prompt. A model argument cannot
enable a permission or select the target. Older explicit RF image API calls keep
their existing Preview/Run authorization and do not depend on Chat's toggles.

`ChatPermissionSession.capture` creates the frozen `ChatToolPermissions` under
the authenticated browser owner. Queues share its revocation fence, not mutable
settings. A newer turn, Stop, permission revocation or conversation/assistant
change invalidates prior publication. Validation, preparation, dispatch,
execution and continuation publication recheck the same authority.

Create proposals reuse `capture_draft`, `fence_event` and `prompt_preview`.
The final browser callback checks the original text/revision, operation serial
and control context immediately before painting. A changed draft stays intact
with a pending **Apply result** / **Dismiss result** proposal. Apply captures
fresh explicit intent; Undo restores only the accepted before-value while its
after-value and accepted browser revision still match. Pending and Undo records
belong to the browser transaction, and other Create inputs remain unchanged.
Stop, conversation resets and source-context invalidation synchronously advance
the browser's revocation epoch. Apply retains the proposal's original epoch,
so clicking Apply before the server processes cancellation cannot renew a
revoked ticket. A later newly captured turn receives the current epoch.

**New image** ignores retained sources. **Previous generated image** uses the last
owned typed image result, or the current image workspace selection when no such
result is retained. **Uploaded image** adopts that upload through the same image
workspace. Source revision, masks and ownership still have to match. Selecting a
reference never gives the model filesystem authority or changes Create's prompt.

Readiness, preparation, cancellation and errors appear beside the image action.
**Stop reply** cancels this browser's pending image action, including before it
queues. Changing its source, image denoise, Chat model or Create settings
invalidates an older response. A cancelled, replaced or foreign
result cannot enable **Open / Edit image**. A completed owned result enables that
button, which deliberately opens the result for editing. It does not replace
the current image merely because a Chat response finished. Normal assistants,
avatars, model selection, reasoning, vision and bounded attachment retrieval stay
in their existing Chat controls.

Chat retains **Load**, **Import GGUF**, **Download**, **Cancel download** and
**Unload**. Their availability follows selected versus active model state;
detailed backend evidence remains available to its owning diagnostics paths.

The assistant editor keeps **Save**, **Save as new** and **Cancel** together.
**Use at startup** saves the selected assistant as the startup preference.
**Model vision** enables image review; its separate **Vision model** selector
appears when the active Chat model needs a reviewer. High detail sends a larger
PNG and uses more memory; the model may resize it. Originals stay unchanged.

## Boundary and worker integration

[`chat_tools.py`](../../ruinedfooocus/features/chat/chat_tools.py) exports the
strict schemas and guidance.
[`image_tool_requests.py`](../../ruinedfooocus/features/chat/image_tool_requests.py)
owns frozen `ImageToolContext`, validated `ImageToolRequest`, the single-call
stream buffer and `ImageToolContinuation`. Unknown fields, incomplete JSON,
duplicate keys, truncated or parallel calls, paths/URLs used as IDs and
incompatible source/mask operations fail before preparation or compute.
`model_profiles.initialized_image_tools` binds source-qualified transport to the
actual initialized template digest and captured runtime/resource identities.
Unloading, template overrides or unsupported source/runtime bindings disable
that advertisement; metadata access never loads a model or probes a device.

These pure contracts do not execute images on their own. The worker continuation
service is the required consumer: Chat first returns a validated continuation,
then exits its compute scope. Existing preparation admission handles the captured
image operation outside compute and resumes the same owned job. The existing
image handoff performs the operation and restores the captured Chat runtime.
Cancellation, current revision and restoration are checked before publishing a
completed editor step and one terminal tool result. The continuation's renderer
is pure; the worker owns terminal publication and history commit together.

The shared backend service calls existing image operation factories directly.
It must not queue a nested API job or make HTTP requests back into RF. Ordinary
OpenAI-compatible chat completion tools remain client-managed transport. Any
explicit RF server-executed operation route uses this separate validated boundary.
Public tool results carry call IDs, status, opaque editor references and bounded
execution metadata. Local file paths, private settings and owner authority remain
internal; the owning editor DTO and Chat result adapter redact nested paths.

## Focused checks and evidence limits

```sh
python tools/check_cpu.py test_chat_turn_permissions.py test_chat_image_tools.py test_chat_image_tool_continuation.py test_chat_image_service.py test_prompt_draft_fence.py
```

The continuation consumer additionally runs `test_chat_image_tool_continuation.py`
and its actual worker/API cancellation, preparation and transport callers. CPU
fixtures establish input, ownership and transport contracts. They do not establish
real-model tool behavior, image quality, GPU stability or rendered UI behavior.
