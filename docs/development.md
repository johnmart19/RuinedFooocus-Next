# Development and validation

Start from the actual checkout, branch and existing edits. Preserve other work,
original models, outputs and user data. Use the checkout's virtual environment
and the backend pins in `ruinedfooocus/app/bootstrap.py`.

The [architecture map](architecture.md) locates each feature's UI, service and
native integration. Keep feature code beside its UI and data contracts. Shared
code belongs in `core` only when it has real cross-feature callers. ComfyUI owns
native loading, sampling, LoRA application, upscaling and memory management;
application adapters add request identity, cancellation, metadata and recovery.
Similar names are not evidence that an adapter can be deleted.

The [tool index](../tools/README.md) records each maintained CLI/helper and its
consumer, dependencies, side effects and checks. [Tool contracts](integrations/tools.md)
cover API examples, safe read-only audits and explicit inference tools.

## Focused CPU checks

Use the active checkout interpreter (Windows: `venv\Scripts\python.exe`; Linux:
`venv/bin/python`) for these commands:

```sh
python tools/check_cpu.py test_feature_cherry_pick_manifest.py test_cpu_check_runner.py
python tools/check_cpu.py test_prompt_formats.py test_wildcard_catalog.py
```

The runner shares the feature validator's GPU-initialization/probe guard, expands
filenames before discovery to exclude video modules and gives each invocation
private writable data with Hugging Face offline mode. It does not install
dependencies, start the app or run image generation. A test can still have its
own side effects: inspect newly selected suites and mock external work. Broader
CPU discovery is explicit (`"test_*.py"`), not the default after each edit.

Use changed-feature tests plus direct callers. Record source revision or changed
file hashes, dependency versions, command, test count and failures. Reuse earlier
proof only when its relevant sources and dependencies still match. CPU mocks do
not certify model semantics, GPU stability, or rendered browser behavior.

Many fixtures restore `sys.modules` with `patch.dict`, which also removes any
module first imported while the patch was active. Import a module a test patches
(for example `utility_device`) at the top of the test file, not inside a test:
otherwise a later test patches the dropped copy while the code under test loads
a fresh one. For the same reason the runner imports `torch.distributed.tensor`
before discovery; `torch.testing.assert_close` loads it lazily, and loading it a
second time fails because torch registers its operator library only once.

## Documentation is part of a feature

Each worker updates the guide owned by its feature, or records why the existing
guide remains accurate. Do this alongside code and focused checks, not as an
unowned final cleanup. Link technical backend contracts from
[integrations](integrations/README.md); user actions belong in the user/settings
guides and source ownership in architecture. Avoid duplicating the same status
in a guide, plan, agent instructions and checkpoint.

A useful feature handoff documents:

- UI entry point, source owner, request/data flow and exact native API used.
- Persisted versus requested versus initialized state, apply/restart timing,
  cancellation, concurrency ownership, recovery and compatibility migration.
- Supported OS/device/Python/package closure, optional acquisition, offline and
  explicit CPU behavior. Separate source support from hardware qualification.
- Primary upstream source/release or API version, access date, relevant
  Context7 library ID when used, and the decision to reuse, adapt or defer it.
- Reproducible focused commands and evidence locations, including source/model/
  artifact identities and what is still unverified. No secrets or huge logs.

Summarize the APIs used rather than copying upstream documentation. Validate
Context7 examples against the installed/pinned version. A missing library topic
justifies a narrow lookup; it does not require each worker to repeat the whole
research pass. Keep developer implementation details out of ordinary UI help.
If behavior is only planned, label it as planned. Do not document it as shipped.

Record owner-confirmed non-blocking follow-ups in the [issues index](issues/README.md)
using its bounded template. Keep one source-bound topic per card, with evidence,
impact, owner, next finite action and acceptance. Active acceptance failures remain
current work; an issue card does not close their gate or authorize continued research.

## Feature commits

Each cohesive commit adds `tools/features/<id>.json`, registered through
`tools/feature_commits.json`. Give it the exact unique commit subject, actual
prerequisite IDs and its focused test filenames. Installer features remain
separate. Test additions travel with the feature that can run them; composition
tests belong with the complete UI, not an artificially widened low-level closure.

```sh
python tools/validate_feature_cherry_picks.py --feature <id> --tests --output <report.json>
```

This replays only the declared closure on the recorded upstream base and checks
imports, bundled resources and selected CPU tests. A passing full checkout alone
cannot establish prerequisite independence. The scratch checkout is disposable;
check its resolved path and junctions before removing it.

Closure checks fail unresolved eager imports, including class bodies, and report
unresolved function-body integration imports in `deferred_imports` with their
source locations and scopes. Selected caller tests still own the runtime
dependencies they exercise. Audit the complete candidate with the strict default
`python_contracts(candidate_root)` before publication; every intended local
import, including deferred references, must resolve there.

Native CPU fixtures use exact Git objects, never a vendor working tree. Features
that precede bootstrap declare `native_comfy_revision` or `native_gguf_revision`
alongside their existing `native_comfy` or `native_gguf` flag. A declared revision
must be a full lowercase commit ID and match the consumer's bootstrap pin whenever
bootstrap is present. Missing bootstrap and an undeclared pin cannot fall back to
the vendor HEAD.

Include `Co-authored-by: Codex <noreply@openai.com>` when Codex contributed, preserve
other authors' credit, and verify both the pushed tip and GitHub attribution.
Prefer normal pushes. History replacement needs a verified rollback reference
and an exact force-with-lease against the freshly observed remote.

## Formatting and browser checks

Formatting tools are isolated in `tools/format`; do not reinstall the application
to format a source file. Use a supported Node LTS, version 22 or newer, with npm 9
or newer for the v3 lockfile. This Node floor also covers the native JavaScript
fixtures using `structuredClone`, `Array.at` and `node:buffer`'s `File`.
Create the dedicated formatter environment once (Windows):

```powershell
venv/Scripts/python.exe -m venv tools/format/.venv
tools/format/.venv/Scripts/python.exe -m pip install -r tools/format/requirements.txt
npm ci --prefix tools/format --ignore-scripts
tools/format/.venv/Scripts/python.exe tools/format_code.py --check
```

On Linux/WSL, use `venv/bin/python` and `tools/format/.venv/bin/python` for the
corresponding interpreters. Ruff's install pin and `pyproject.toml` required version
must move together; regenerate the npm lockfile after changing Prettier's exact
pin. Vendored sources, model data and user files are excluded from formatting.

Hidden navigation/view callbacks use `show_progress_on=[]`; native Gradio then
tracks no output progress IDs, including metadata projections reached by startup
value changes. Full-progress view callbacks target their existing leaf controls
rather than layout containers. Scope every binding, including refresh wrappers.
Full feedback uses an existing status or control leaf. Values, errors,
input validation and processing of existing shared tracker entries are unchanged.

Check Markdown links and the documented UI routes after label changes.

Run browser acceptance on a separate CPU/offline profile with automatic model
preload disabled. Do not reuse a user's active session or overwrite their profile.
Check Create, Chat, Library and Settings at desktop and mobile sizes, long labels,
keyboard navigation and draft preservation. Never click Generate for a
navigation-only check.

Control help observes native hint/label changes and inserted or removed subtrees;
unrelated streaming text must not rescan existing controls. The Node DOM contract
in `tests/test_control_help.js` runs through `test_html_templates.py`; real browser
interaction and responsiveness still need their own acceptance evidence.

## Source distribution

```sh
python tools/build_source_archive.py --output dist/RuinedFooocus-source.zip
```

Build only after committing. Verify the archive's revision metadata and Git blob
contents, including docs and translations. The ZIP retains the tracked model
folder README up to 16 KiB; placement markers are excluded. Weights, local
model bundles, settings, environments and private evidence are excluded.
Portable GPU runtime qualification is separate. See
[runtime support](runtime-and-troubleshooting.md).

## Startup time

Measured on Windows with an RX 7900 XTX and a warm disk on 2026-11-20: 46 seconds
from process start to a listening port before, about 20 after. Three things made
the difference, and each is easy to undo by accident:

- **One interface build.** Closing any `with gr.Blocks()` context makes Gradio
  compute the configuration of every component and construct a new web
  application. The interface re-enters such contexts about 25 times while it is
  assembled, which cost 11.5 seconds. `ui/gradio_frontend.py` defers that work
  (`defer_interface_build`) until `launch()`, which builds both once;
  `resume_interface_build` must run before `launch()`. Code that needs
  `Blocks.config` or `Blocks.app` while the interface is being assembled will
  see an empty configuration and no application.
- **Remembered PyTorch validation.** `packaging/runtime_support.py` validates
  the installed bundle by importing torch in a child process, about five seconds
  each time, and startup asked twice. A ROCm or CPU bundle that validated is
  remembered in `cache/torch-probe.json` under the identity of the installed
  torch and torchvision records, so reinstalling or repairing them validates
  again. CUDA is not remembered because its probe also exercises the driver.
- **Imports on first use.** `rembg` and its compiler stack took 1.7 seconds at
  every launch; `rembg_pipeline.py` now imports it when a background is removed.

To see where a launch spends its time, sample the main thread rather than trace
it: a tracing profiler multiplies the cost of import-heavy code and points at
the wrong places.

## Benchmarks

`tools/benchmark_app.py` measures an installation without changing it. Use the
app's interpreter:

```sh
python tools/benchmark_app.py startup --runs 3 --data-dir <profile> -- --offline
python tools/benchmark_app.py generate --base-url http://127.0.0.1:7860 --model anima --runs 4
```

`startup` launches `launch.py` on a free port and reports, per phase, the
seconds until the runtime loads, the interface is built, the port listens and
the page and its configuration answer. `generate` needs an app started with
`--api`; it times `/v1/images/generations` per model, reports the first image
(which includes loading the model) apart from later ones, and on Windows samples
the server process's dedicated GPU memory and busiest GPU engine from the
system counters. Record the CPU load next to any timing: on a shared machine,
other programs changed CPU results by an order of magnitude on 2026-11-20.

Measured 2026-11-20 on an RX 7900 XTX (ROCm, 1024x1024, the model's default
steps, seed 1234, models used one after another in one session); the run was
stopped before Krea 2 finished:

| Model                    | First image | Later images | GPU memory |
| ------------------------ | ----------- | ------------ | ---------- |
| Anima Mayhem 2.9B fp16   | 41 s        | 36-43 s      | 11.5 GB    |
| JANIMA 2.9B bf16         | 37 s        | 32-33 s      | 17.0 GB    |
| Flux 3 Klein INT8        | 32 s        | 9.5-11 s     | 21.9 GB    |
| MOSS Image INT8          | 72 s        | 62-63 s      | 20.1 GB    |
| Krea 2 BSS INT8 (2 runs) | 37 s        | 103-108 s    | 22.0 GB    |

GPU memory grows as models change because earlier ones stay loaded; Krea 2 ran
about three times slower once 22 of 24 GB were in use. This is not yet
explained.

Interface callbacks are measured on the server: wrap
`gradio.blocks.Blocks.process_api` in a scratch launcher to record each
callback's name, trigger and duration, then load the page. On 2026-11-20 a
fresh page load cost 80 callbacks and 19.1 s of server time; most of it came
from the model preview timer, which treated every new page as stale and
re-rendered the model and LoRA lists. It now starts from the revision and
selected-model metadata the page was built with: 65 callbacks and 7.7 s.
