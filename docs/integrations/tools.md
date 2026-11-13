# Developer tool contracts

The [tool index](../../tools/README.md) owns the complete entrypoint/helper,
consumer, CLI, dependency, platform, mutation and verification inventory. Tools
use an existing project interpreter; help and audit commands never install a
runtime. Windows and WSL have separate environments. Calling a Windows source
file from WSL does not qualify the other checkout or its model runtime.

## Gradio examples

Run the app with its owner API enabled (`--api`). The examples use the app root
URL, including a reverse-proxy prefix if applicable, and the installed
`gradio_client.Client`. Installed contracts checked on 2026-11-11: Gradio
**6.31.0**, gradio_client **2.7.2**. Primary sources are
[`app/api.py`](../../ruinedfooocus/app/api.py) and the installed client's
`Client.submit`, `Job.result`, `Job.cancel` and `Client.close` implementations.
The framework owns JSON escaping, login, endpoint discovery and queue transport.
There is no separate shell SSE parser.

```powershell
venv/Scripts/python.exe tools/api_search.py "all: mountains" --max-results 5 --base-url http://127.0.0.1:7860
venv/Scripts/python.exe tools/api_examples.py llama 'Reply with "hello"' --base-url http://127.0.0.1:7860
```

The first command searches existing Library metadata. The second executes the
configured Chat model. `tools/api_generate_image.sh` or the Python `image`
subcommand executes image generation with the server's current settings and
creates a server output. They are explicit workload examples, not read-only
connectivity checks. Inspect `--help` without contacting an app:

```sh
RF_PYTHON=/absolute/path/to/venv/bin/python sh tools/api_llama.sh --help
RF_PYTHON=/absolute/path/to/venv/bin/python sh tools/api_search.sh 'all: blue sky' --max-results 5
```

For an authenticated UI, set both `RF_GRADIO_USERNAME` and
`RF_GRADIO_PASSWORD` in the caller's environment. The client uses Gradio's
normal login; do not put credentials in a URL or command arguments. Owner
OpenAI-compatible `/v1` bearer authentication is a separate API contract and
does not replace this UI login. Guest mode does not expose the owner API.
Login failures return nonzero without printing credentials or internal errors.

`/search` returns a list of already absolute, percent-encoded
`/gradio_api/file=` URLs. Print each unchanged, including filenames containing
spaces or Unicode; do not prepend an app/file prefix. `/llama` returns text.
`/generate_image` returns Gradio FileData; the helper preserves its server URL
or encodes its remote path using the actual `file=` route. `download_files=False`
prevents creation of the client's default media cache and automatic downloads.
An authenticated file URL still requires the server's login cookie when opened.

`--timeout` bounds the wait (default 300 seconds). Timeout or Ctrl+C requests
Gradio cancellation and closes the client. Gradio cancellation is best effort:
queued work can be removed, while an already running ordinary function may
continue. These examples do not claim that client cancellation stops an active
model job; use the app's owned Stop controls for that workflow.

`test_api_examples.py` runs the production `add_api`/`image_file_url` callback
bodies through a real private authenticated Gradio server and client. The model
service boundary is inert. It verifies login rejection, exact multiline/Unicode/
quoted Chat input, returned search URLs and FileData, without model execution.
Fixture transport proof does not qualify model quality or a full app runtime.

## Registry, links and native pins

The [download audit](../download-audit.md) joins catalogue URLs through the public
`load_registry`, `entries`, `annotations_for` and `profile` interfaces. It records
the exact tuple match, variant, required companion IDs, artifact size/hash,
runtime pins and family source/dependency evidence. Schema-2 family and artifact
shards are validated by the [registry reader](model-registry-schema.md).
Qualification/benchmark history is neither requested nor hashed in this link
inventory. Missing or inconsistent annotations add an unresolved
`registry-validation` row; the existing catalogue URL is preserved. Proposed
artifacts remain unresolved, and research/source URLs remain `reference_only`.
Live probing never silently acquires weights, credentials or gated access.

`split_model_registry.py` with no `--source` verifies current resources without
writing. Its original migration baseline is separate from later additions;
current counts are obtained from data. `--source` explicitly writes split data
to the chosen destination and belongs to the model-data owner.

`audit_native_model_coverage.py` checks declared ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` and GGUF
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41` against the actual native repositories
and audited Git blobs. It uses AST/data, not device imports. Its comprehensive
registry projection deliberately includes historical evidence; the narrower
download audit does not. Counts of nodes/configurations/roles are independent
denominators and establish source coverage only. Keep native revisions and
source hashes with each receipt rather than changing a pin to satisfy a check.

## Validation and distribution

`test_tool_contracts.py` inventories every current root Python/shell tool against
the index and executes Python help or helper imports in child processes that
reject network, process launch, writes and tensor/runtime imports. It also
checks local documentation links and the formatting dependency lock. Shell
syntax/help and literal argument forwarding are separately checked with WSL
`sh`; their installed Gradio closure is not inferred from Windows.

Read-only source commands include the action inventory, offline download audit,
registry verification, translation check and asset listing. Report-producing
commands write only their explicit owned output paths. Formatting and source
archive policy have focused fixtures. Trusted JSON validation cases execute
arbitrary selected adapters and real models: review the spec, local artifact
paths and runtime scope before invoking them. Summarizers write summary/contact
sheets, and batch runners replace their chosen reports/logs; use owned evidence
directories.

Source archives use committed Git blobs and include tracked tool files,
schema-2 resources, docs, translations and approved small model placement
markers. Existing policy excludes weights, environments, settings and private
evidence. This pass reviews that policy and its fixtures; the release owner
builds and byte-verifies the exact final committed archive. See
[development](../development.md) for source/CPU, browser, workload and publication
gates.
