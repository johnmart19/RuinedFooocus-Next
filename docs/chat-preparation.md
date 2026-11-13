# Chat model preparation

Chat's model picker selects a candidate. **Download** prepares that file without
loading it or changing the current conversation. **Cancel download** cancels this
session's subscriptions; another session downloading the same artifact continues.
Progress shows a percentage when the server supplies a total size, and received
bytes with an indeterminate bar otherwise.

Click **Load** after the candidate is present. Loading verifies the candidate,
then enters the same serial compute worker used by Chat and image generation.
The active model is shown separately. Selecting or downloading another candidate
does not disable messages to the active model. A completed download never loads
itself, changes the picker, publishes a shared enhancement default, or modifies
conversation history. **Unload** explicitly releases the active Chat runtime.
The saved startup/default path is separate from a model actually loaded in this
browser session.

A preparation or verification failure preserves the active runtime. A failure
after a runtime replacement has begun may leave no active model; the Load result
reports the runtime that actually remains. Loading another model is still a
serial compute operation. There is no simultaneous Chat/image compute guarantee.

## Local text attachments

In the owner WebUI Chat, choose **Attach text files** below the message field.
Attach up to four UTF-8 `.txt`, `.md`, `.csv` or `.json` files, text PDFs or DOCX documents per conversation,
at most 256 KiB each and 1 MiB combined. The picker shows when it reads the selection and
lists ready files with their source IDs. Names may repeat. Use **Remove** beside
the corresponding source ID to remove one file. **New conversation**, clearing
Chat, or choosing a different assistant drops the attachments. Changing the
language model preserves the current conversation's files. Guest Chat is
unavailable. Other standalone file types and persistent libraries are deferred.

PDF and DOCX extraction also has a 256 KiB text limit. PDFs can contain at most
32 pages and must already contain selectable text; encrypted files and OCR are
unsupported. DOCX reads visible body/table text, ignoring field instructions,
links, embedded objects and macros. Layout, headers, footnotes and pagination are
not reconstructed. Extraction never follows URLs or file references. Corrupt,
empty or resource-heavy files return an error without adding part of a selection.
The original binary bytes are discarded after extraction. Parsing is isolated
and bounded; [the document contract](integrations/document-ingestion.md) records
the exact parser, limits and dependency preparation.

Ask a question after the files are ready. Chat looks for relevant excerpts in
each attached file and lists the source ID, filename and line range of passages
supplied to the model. Markdown section headings can help the model interpret
those passages. Chat reads CSV and JSON as literal text. It cites physical line
numbers and leaves formulas, schemas, file paths and URLs as text. Large files
and long conversations may leave some excerpts out.
Check the listed lines when you need to confirm a claim. Chat keeps the original
question and generated reply in history. Uploaded text, excerpts and the source
panel are conversation data held separately from the history and profiles.
For PDFs, source entries also identify the actual page; their line numbers refer
to extracted text including generated page markers. DOCX line numbers refer to
extracted paragraphs and table text, rather than a Word page layout.

**Stop reply** cancels the current attachment-backed reply and restores the
message for retry. Removing or adding a file during that reply also stops it so
the next attempt uses the updated selection. A conversation reset drops the
in-flight reply. Image attachments still use the separate vision controls and
require a compatible model and projector. Image tool calls are disabled while
document excerpts can influence the reply. **Force image generation** uses the
normal image workflow for that turn and leaves the text files attached for later.

The first grounded message may need to prepare the existing local MiniLM model.
Offline use requires its complete pinned snapshot. Preparation downloads model
artifacts when online. Uploaded document bytes stay in the local application.

### Portable `.rag` files

**Export attachments (.rag)** downloads the current sources as
`chat-attachments.rag` in your browser. The download includes the original text
with normalized line endings. PDF/DOCX entries contain extracted text, not the
original binary document. It contains no chat history, model settings,
source IDs or embedding index. Keep the file private if its text is private.

To restore those files, use **Attach text files** and select the `.rag` file by
itself. Chat adds all sources in one operation and assigns new source IDs.
Existing files count toward the same limits. A rejected manifest leaves the
conversation's files intact.

A version 1 manifest is UTF-8 JSON with this structure:

```json
{
  "version": 1,
  "sources": [
    { "name": "notes.txt", "text": "A local note." },
    { "name": "reference.md", "text": "# Reference\nA second source." }
  ]
}
```

Use one to four sources with plain supported attachment filenames and
inline text. Each source can use up to 256 KiB of UTF-8 text, with a combined limit of 1 MiB.
The serialized manifest can use up to 2 MiB plus 4 KiB to accommodate JSON
escaping. Unknown or duplicate fields, unsupported versions, nested manifests
and path or URL references are rejected. Text that mentions a path or URL is
handled as ordinary document text. A restored PDF-named source is inline text;
it is never parsed as a binary document. Original PDF page metadata is omitted
from version 1 manifests, so restored sources cite extracted lines only.
Export creates a download in the browser
without creating a server file or publishing a Gradio file link.

### Chat API attachments

Clients of `POST /v1/chat/completions` can send an optional `rf_attachments`
array. Each entry contains a plain supported `name` and standard base64 `data`
for the file's bytes. Text files must be UTF-8; PDF and DOCX use the same bounded
extraction as the owner WebUI. For example:

```json
{
  "model": "chat/default",
  "messages": [{ "role": "user", "content": "What does the note say?" }],
  "rf_attachments": [{ "name": "note.txt", "data": "SGVsbG8u" }]
}
```

Send one to four files, up to 256 KiB decoded per file and 1 MiB combined. The
last message must be a user message containing text. It can also carry the
usual inline `image_url` parts for a vision-capable model. Send the files with
each request that needs them. `.rag` files, remote URLs and filesystem references
are unsupported in this field. The endpoint uses the configured API bearer key.

A successful JSON completion includes a top-level `rf_attachment_sources`
array. With `"stream": true`, the first JSON SSE chunk carries that field and
the stream still ends with `data: [DONE]`. Each entry gives the source ID,
filename, and inclusive one-based start/end lines, for example
`{"id":"D1","name":"note.txt","start":1,"end":1}`. An empty array means that
no excerpts fit or met retrieval criteria. Source IDs follow upload order and
belong to that request. PDF references additionally include an integer `page`.
Requests that omit `rf_attachments` keep the existing
response shape.

Grounded requests reject active tool definitions and forced tool choices. The
server suppresses tool-shaped native output for those requests. The same local
MiniLM preparation, excerpt limits and offline behavior apply as in the WebUI.
The application releases each request's document bytes and shortlisted passages
when the job completes, fails or is cancelled.

## Implementation and ownership

Local inventory reads only the scalar GGUF role fields needed by the picker.
The bounded header reader skips vocabulary arrays and never maps tensor data.
Source discovery uses the same reader for its selected source fields. Results
remain cached by file identity, so replacement files are inspected again.

After loading, the active-model status reports the initialized runtime, backend
and any fallback reason. Installing a Vulkan build alone does not prove GPU
availability. On WSL, the host's AMD GPU can be available to ROCm while Vulkan
exposes no GPU. Native llama.cpp Auto attempts the matching vendor build before
falling back to CPU when Vulkan cannot initialize a device.
Cached server binaries with missing GPU dependencies receive one acquisition
attempt before CPU fallback. The active status also retains the native backend's
selection explanation. Device execution errors still stop the load explicitly.

`features/chat/model_preparation.py` captures immutable catalogue tuples,
canonical destinations, projector choices and settings before admission. Its
session state keeps selection revision, download subscriptions and active model
separate. `features/chat/ui.py` uses independent Gradio event groups for Download,
Load and messages; cancellation and selection callbacks bypass their queues.
Only the application worker admits compute, so these UI queues grant no extra
GPU concurrency.

Explicit Chat loads enter the existing I/O preparation lane even when weights
are already present. `chat/sampling_sources.py` inspects bounded scalar GGUF
source fields and reuses the shared signature-bound weight-hash cache and trusted
Hugging Face exact-content resolver. At most four repository hints are examined;
only HTTPS Hugging Face repository identifiers reach that resolver. No card,
template, workflow or remote code is executed. Publisher advice comes from the
packaged profile, not arbitrary fetched numeric configuration. A new quantization
may reuse a profile only at its reviewed immutable artifact-repository revision
with matching architecture; runtime and reasoning admission remain independent.

Positive source observations expire after a day, negative observations after
five minutes. Explicit refresh bypasses online TTLs; offline mode keeps current
cached evidence, including on forced refresh. Failed lookup retains current
evidence, and replacing the file invalidates it. Missing, malformed or conflicting
source evidence leaves native runtime defaults and explicit overrides usable.
Preparation failure never unloads an active model. Source changes take effect
for sampling after explicit Load; ordinary inference does not refresh metadata,
hash weights or adopt another selected candidate's advice.

`model_profiles.compose_sampling` supplies both actual client parameters and
effective-field provenance. Active UI status uses the initialized client snapshot;
selected-source details use the separate read-only cache. The optional source
refresh queue owns no compute and holds the normal artifact read lease.
Refreshes take an exclusive lease on that model's metadata cache, so a late
failed lookup cannot erase a successful concurrent result; other models' caches
remain independent. New CPU coverage is `test_chat_sampling_sources.py`, with tiny metadata-only GGUF fixtures,
caller load/refresh checks, cache invalidation, offline retention and full
sampling precedence. These fixtures are never loaded as language models.

`app/async_worker.py::add_task` captures Chat dependencies before appending to
the worker buffer. Missing model, selected reviewer/projector and URL context
files are prepared through `core/preparation.py`. The bounded continuation pool
in `app/preparation_admission.py` defaults to two I/O waiters and eight bounded
preparation jobs. Ready local requests bypass this pool, so blocked transfers do not occupy
compute. No per-request unbounded threads are started.

For assistant context, `features/chat/rag_preparation.py` captures the existing
txtai model and CPU policy with the request. Its finite file closure is prepared
and verified in the same I/O lane before the worker admits Chat compute. Complete
local Hub snapshots at the exact revision are reused; no parallel downloader or
new vector database is introduced. RAG requests use this lane even when their
files are present, so validation stays outside compute. Existing signature-bound
hash caching avoids rereading unchanged model weights on each request. The index
identity includes context-file and prepared model-file signatures; replacing a
file at the same path invalidates the prior index without hashing weights during
compute.

The model remains `sentence-transformers/all-MiniLM-L6-v2`, pinned to official
Hub commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (metadata checked
2026-11-09, Apache-2.0). The eight-file native Transformer/tokenizer/pooling
closure totals 91,567,511 bytes, including the 90,868,376-byte safetensors weight;
ONNX, OpenVINO, TensorFlow and alternate weights are excluded. Sources use the
[immutable publisher revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41).
Small Git files retain pinned revision/size identity; the LFS weight also retains
the publisher's SHA256. Metadata availability is not real-model acceptance.

Offline requests use a complete local closure or fail before acquisition and
compute; a partial directory is insufficient. Model files are not Python package
installs, so `--skip-dependency-install` alone does not disable their normal
online preparation. **Settings → Local Chat → Documents and RAG** shows the
pinned 384-dimension model, exact-size file presence, and the placement policy
for new requests. **Prepare or verify embedding model** checks existing files
and downloads any missing files when online. Cancellation detaches the current
tab's preparation. Chat can continue to use its own preparation subscription.

**Auto** preserves txtai's automatic device choice. **CPU** applies to new
grounded Chat requests after saving the profile. Global CPU and DirectML image
mode also select CPU. The txtai Transformers runtime receives `gpu=None` on
the CPU path, which selects CPU before accelerator probing. An initialized
assistant index exposes its observed device in Settings. Temporary attachment
indexes remain request-owned and close before the reply.

Actual embedding-model loading, indexing and queries remain with the owning
Chat compute stage. The constructor receives an absolute local directory and
passes local-only model arguments through txtai's `vectors` configuration.
Failed or cancelled index construction closes the candidate and preserves the
previous completed index; replacement/context removal/Chat unload release its
owned index and file leases.

`features/chat/{attachments,attachment_ui,attachment_retrieval}.py` owns WebUI
document capture. The native file picker sends bounded encoded bytes through a
private Gradio event. It bypasses the Gradio file upload cache. The server accepts
plain filenames and text bytes, with no path or URL resolution. A random tab
capability augments the authenticated login and Gradio session identity. The
memory-only store retains at most 64 tab entries and expires entries after an
hour of inactivity when the store is accessed. Removing a file or resetting the
conversation invalidates the captured revision and signals its queued or running
jobs. Active replies recheck ownership and revision before publishing results.

The existing I/O preparation lane scans 360-character passages and shortlists
at most 12 per file by question term matches. Chat compute reranks those with
the same pinned local MiniLM model. Its temporary txtai index uses no document
database and closes before native Chat completion. Up to four relevant excerpts
are added to a copy of the current user turn. The byte budget caps added context
at 2,048 bytes and one quarter of the configured model window, with additional
headroom for existing messages, output and active vision inputs. The source
panel uses server-derived metadata and displays only excerpts admitted to the
model context.

`app/api_runtime.py::chat` captures the model/projector request and the API's
inline sources before submission. Attachment requests use the I/O preparation
lane before pipeline selection. Retrieval runs under the Chat compute owner and
appends excerpts to a private copy of the final user turn. API cancellation
detaches the job's preparation subscriptions and clears its corpus and passages
when the job releases its slot. Assistant URL context indexing in the WebUI reads
the captured prepared destination.

`llama_pipeline.py` loads captured paths/settings and retains process-local read
leases while those files belong to its loaded runtime. Import and deletion in
`llama_models.py` take exclusive publication leases. Replacement releases leases
for artifacts no longer loaded; a validation failure before replacement retains
the prior runtime's leases. External programs do not participate in these locks.
The existing loader still verifies GGUF architecture, registered content identity,
projector pairing and file signatures. Native device and CPU policy remains with
the llama.cpp integration.

## Validation and remaining lifecycle work

Run the focused CPU checks with the project interpreter:

```powershell
venv\Scripts\python.exe tools/check_cpu.py test_chat_attachments.py test_rag_manifest.py test_chat_api_attachments.py test_rag_preparation.py test_chat_preparation.py test_chat_boundaries.py test_chat_model_browser.py test_ui_workflows.py test_worker_routing.py test_worker_cleanup.py test_preparation_service.py test_api_live_contracts.py
```

The deterministic fixtures hold a small fake transfer while the real message
validator and reply callback use active B, fence stale completion, cancel or fail
preparation, check immutable paths/settings and exercise known/unknown progress.
They establish source/CPU behavior, not real-model, browser or GPU qualification.

This integration covers Chat weights, explicit/automatic projectors, selected
reviewers, URL context files and the default txtai embedding model. The scoped
worker lifecycle is implemented separately in `app/resource_lifecycle.py`; this
does not qualify every implicit first-use acquisition in caption, ControlNet,
geometry or upscale features. Their topic guides and acceptance evidence retain
those boundaries. Browser, real-model semantics, allocation, performance and
physical concurrency acceptance remain separate from these CPU contracts.
