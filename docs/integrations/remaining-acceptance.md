# Remaining acceptance inventory

The full-native parity programme below is stopped. These source-specific receipts
remain historical evidence, not an active task queue or permission to run models.
Current work follows the [image and optional Chat scope](image-chat-scope.md).

This is the historical carry-forward for W09, reconciled on 2026-11-11 against
application `f3bcdcda88181587fb0a331c0659c7b263b452f2`. It records **27 cases**;
none is promoted to real-model, semantic or hardware verification by CPU mocks.
The CPU preparation packet was assembled at integration
`292d93ebc38b34a43f9c16c5f70ab47da4470fb4` and rebased onto
`4d9383a0bea1702e1e7f00f4494b018e7b8bbaab`; the feature IDs below own its changes.
Original test receipts retain their source identities across the rebase.
The archived workorders were retired, not passed. Their recoverable baseline is
`C:/Users/johnmart19/.codex/task-records/archives/ruinedfooocus-plan-20261111-ad6d015/`
(`manifest.json`, `plan.md`, `agents/previous-plan-audit.md`). Missing raw receipts
remain missing evidence, rather than invalidating unrelated documented results.

## Identity and evidence contract

Every new run records full application HEAD and changed-file hashes, native pin,
Python/package/build/backend identities, platform/device, model and component
SHA-256/size, original input SHA-256, complete captured request, command/UI route,
output hashes, elapsed time and evidence path. Record **source/native contract**,
**app transport/geometry**, **semantic/visual**, **performance**, and **stability**
as separate results. A successful HTTP response, provider list, installation
canary or output dimension cannot pass another evidence level.

Current native image pins are ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44` and ComfyUI-GGUF
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`. Native Chat's app pin is b12110;
the already tested updater candidate is b12262. The recorded installed embedded
runtime is xllamacpp `2026.10.11870`, llama.cpp b11870 lineage
`74bca3950054eed1795f525cb9cf06bfe796492d`. Recheck installed identity before a
new run; do not reinstall or assume the current wheel changed with the app.

The configured default profile selects native llama.cpp. The embedded loader
imports xllamacpp in the application interpreter, and its installer targets that
interpreter's `sys.prefix`. The initial read-only audit found no xllamacpp in
`D:/RuinedFooocus/venv` or its installed package plans. The existing isolated ROCm
bootstrap helper is an installer, not an alternate embedded runtime, and also
had no xllamacpp distribution. `W09-xllamacpp-owner.json` retains that audit.

The coordinator then explicitly authorized bounded runtime preparation. The
[official Vulkan index](https://xorbitsai.github.io/xllamacpp/whl/vulkan/xllamacpp/)
and [release asset metadata](https://api.github.com/repos/xorbitsai/xllamacpp/releases/tags/v2026.10.11870-vulkan-windows-2022)
agree on `xllamacpp-2026.10.11870-cp310-abi3-win_amd64.whl`, 22,217,611 bytes,
SHA-256 `ccca35c90805c9b540d87256a2aa490885e1f8c470cbd6bdfe5aa064c3a4c152`.
Its ABI3 tag matches the current CPython 3.12 interpreter. [Package metadata](https://pypi.org/pypi/xllamacpp/2026.10.11870/json)
requires Python >=3.10 and declares no mandatory runtime dependencies. The wheel's
actual METADATA/WHEEL and all 12 hashed payloads were verified before installation.
Only that wheel was installed into the acceptance interpreter with
`--no-index --no-deps`; before/after distribution metadata and RECORD snapshots
show only xllamacpp changed. Version `2026.10.11870` and all 15 hashed installed
RECORD payloads were verified. The retained wheel, reversible action and receipt
are in `runtime-wheels/xllamacpp-vulkan-2026.10.11870/` under the handoff root.

No native runtime import, device probe or model load was performed. The existing
installer still owns recording verified backend provenance before offline
activation. Embedded C01/C03 execution remains in the coordinator's final phase;
package eligibility and installed-file integrity are separate from device,
transport, semantic, performance and stability evidence. The selected default
language model remained unchanged during that runtime-preparation packet. The
later explicit [AtomicChat default selection](atomic-lfm3-default.md) supersedes
the old deferral; it does not qualify the model's behavior or close real cases.

## Case ledger

Owners name the current workorder or source area; final hardware execution is
coordinator-owned. "Pending" means a finite acceptance case remains, not a
confirmed implementation defect. The linked guides retain unaffected evidence.

| ID  | Case and current source owner                                                            | Existing source/artifact/evidence                                                                                                                                                                                                   | Remaining acceptance and expected result                                                                                                                                                                                      |
| --- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C01 | Two-document grounding; W09, `chat/{attachments,attachment_retrieval,llama_pipeline}.py` | Text/CSV/JSON, `.rag` and CPU native API JSON/SSE passed at archived `ad6d015`; pinned MiniLM below                                                                                                                                 | Pending: one compatible xllamacpp query needs both files; accurate admitted source IDs/lines; removal and cancellation revoke the turn.                                                                                       |
| C02 | Text PDF/DOCX; W09, attachment ingestion                                                 | Absent at baseline; `bounded-chat-documents` now passes actual isolated PDF/OOXML extraction and JSON/SSE CPU callers; [document contract](document-ingestion.md)                                                                   | Source/CPU parsing verified. Pending real model grounding only; extraction has explicit empty/encrypted/scanned/corrupt/oversized failures and retains owner/revision fences.                                                 |
| C03 | Direct visual Chat; W09, `chat/llama_vision.py`, native/embedded clients                 | Native SmolVLM 256M Q8/projector WSL result in [WSL acceptance](wsl-model-acceptance.md)                                                                                                                                            | Pending embedded equivalent with exact projector and same original; distinguish native visual input from external reviewer/caption context.                                                                                   |
| C04 | GPU embeddings; W09, `chat/rag_preparation.py`                                           | MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, eight files/91,567,511 bytes; recorded WSL GPU indexing                                                                                                                          | Pending Windows Auto device/index/query evidence; explicit CPU must retain observed CPU and no accelerator probe.                                                                                                             |
| C05 | Context and session/profile isolation; W09, preparation/retrieval/settings               | CPU excerpt budget, ownership, saved/request/initialized snapshots in [Chat preparation](../chat-preparation.md)                                                                                                                    | Pending actual small-window answer, two browser owners, profile switch and model switch; no document leakage or silent setting adoption.                                                                                      |
| C06 | Visible text / reasoning Off; W09, native/embedded reasoning adapter                     | Historical eight-token transport returned empty visible text; b12262 CPU LFM3 replies retained bare reasoning tags                                                                                                                  | Pending matching supported template with Off and useful output budget; visible answer and truthful reasoning state, no hidden output rewriting.                                                                               |
| C07 | Default selection; W09 policy                                                            | LFM2.5: 12/28 pass at `1546996`; Qwen3: 14/28 pass at `46c9995`, both engines; [fixed rubric](local-llm-qualification.md). User selected the pinned [AtomicChat derivative/native processor](atomic-lfm3-default.md) on 2026-11-11. | **User-selected default, quality unqualified.** Prior failed fixtures remain failed; xllamacpp still lacks LFM3. The selection supersedes deferral and does not close semantic, device or lifecycle acceptance. No new sweep. |
| C08 | API JSON/SSE and copied payloads; W09, Chat API; W08 tool examples                       | Real native CPU grounding documented at archived `ad6d015`; source IDs carried in JSON/first SSE chunk                                                                                                                              | Pending copied examples against final private app; exact Unicode/quotes/newlines, auth/cancel, `[DONE]`, unchanged nonattachment response shape.                                                                              |
| C09 | Bounded private activity; W09, API Job/owner UI                                          | Missing implementation at baseline; `private-api-activity` now projects actual Job states into a bounded memory-only [activity view](api-activity.md)                                                                               | Source/CPU privacy and lifecycle verified. Pending rendered final UI/API journey; summaries retain no request, document, filename, key, owner identity, native error or serialized body.                                      |
| R01 | Installed runtime refresh; W09, `llama_runtime_info.py`, Local Chat UI                   | Baseline lacked whole-operation bounds/cancel. `bounded-runtime-inspection` now has 45 s/64 KiB whole-refresh, 15 s/16 KiB query, eight-build bounds and CPU-only metadata; [contract](amd-runtime.md#existing-settings-diagnostic) | Source/CPU contracts verified with fake queries and real inert child ownership. Pending actual device-query responsiveness/cancel in final coordinated hardware phase; no runtime/model acquired or replaced.                 |
| R02 | Native Vulkan updater; W09, `llama_update.py`                                            | b12262 Windows CPU install/version/reply/rollback passed in private cache; [SDK guide](native-llama-sdk.md)                                                                                                                         | Pending installed Vulkan candidate/device/reply/rollback on approved hardware, preserving active runtime and app pin; no package swap in CPU preparation.                                                                     |
| R03 | Live Hub/Xet interrupted resume; W09, `core/hub_download.py`                             | Matched loopback ordinary HTTP transferred 2 instead of 12 MiB; Xet retains Hub SDK transport                                                                                                                                       | Pending one bounded real interrupted/restarted transfer of an already required pinned artifact; verify network byte delta, staging, final size/digest and subscriber isolation. Local range proof does not qualify Xet.       |
| R04 | Chat → image → Chat lifecycle; W09/W05, `app/resource_lifecycle.py`                      | One active Vulkan Chat remained responsive while another model downloaded/cancelled at `21c5762`                                                                                                                                    | Pending sequential fixed assets, captured RAM/VRAM, unload/reuse and exact fallback reason; no simultaneous shared GPU compute claim.                                                                                         |
| R05 | Constrained memory/platforms; W09                                                        | [WSL](wsl-model-acceptance.md) and [Pascal](../runtime-and-troubleshooting.md) cover exact recorded deployments                                                                                                                     | Physical 3/6 GB acceptance blocked when matching hardware unavailable. Do not replace with an allocation estimate or transfer AMD evidence to CUDA/Intel/NPU.                                                                 |
| B01 | Bria Windows ML MIGraphX; W09, ONNX provider owner                                       | Original Bria SHA-256 `5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958`; prior attempts timed out before assignment                                                                                                | Pending real original-model provider assignment, output/alpha/RGB, quality, time/memory, recovery; [qualification gate](onnx-providers.md) remains before Auto.                                                               |
| B02 | WSL MIGraphX discovery; W09, ONNX provider owner                                         | HIP GPU present but ORT platform inventory excludes Microsoft-vendor WSL descriptor; [specific upstream limitation](gpu-background-removal.md#wsl-migraphx-discovery)                                                               | **Blocked upstream for recorded build.** Recheck only changed supported provider contract. No spoofed IDs or lifted acquisition gate.                                                                                         |
| B03 | Native BiRefNet retained proof; W09                                                      | Pinned 444,473,596-byte model; real Windows/Ubuntu/Debian RX 7900 XTX, transparent passthrough and separately injected OOM CPU recovery in [background guide](gpu-background-removal.md)                                            | Retain within exact sources/assets/platforms; rerun only changed dependent route. Does not qualify Bria, CUDA, general quality or sustained stability.                                                                        |
| I01 | Qwen 2.1 ten-reference count; W07                                                        | `ad6d015` base/Turbo INT8 and Q4 paths ran; ten-reference outputs counted eight and fourteen                                                                                                                                        | **Failed semantic case.** Preserve references/prompt/seed; compare one matched native/app case if a wrapper defect is suspected; no reroll to green.                                                                          |
| I02 | Qwen alpha-edge edit; W07                                                                | `ad6d015` 385×321 edit preserved outside RGBA exactly; one retained alpha edit had background/edge artifacts                                                                                                                        | **Failed visual case.** Separate outside-mask equality from edge quality; use same input/mask and native chain.                                                                                                               |
| I03 | PE-T2I transparency prompt; W07/W13                                                      | `ad6d015` preview/Apply/Undo passed; enhancer added checkerboard instructions                                                                                                                                                       | **Failed semantic case.** Preserve optional preview/apply and original prompt; do not silently rewrite the user's request.                                                                                                    |
| I04 | Native outpaint seam/replay; W07                                                         | `2f78324` source/CPU/replay and browser controls passed; real full output absent                                                                                                                                                    | Pending fixed original, mask, padding, prompt/seed and identical native/app chain; exact geometry/replay plus seam review.                                                                                                    |
| N01 | LoRA warm removal case 1; W07                                                            | Old source-bound conditioning matched with differing warm pixels; cold reload matched                                                                                                                                               | Pending same artifact/native-app comparison with exact strengths and warm/cold state; keep prior pixel mismatch.                                                                                                              |
| N02 | LoRA warm removal case 2; W07                                                            | Same bounded old finding as N01; new native GGUF LoRA routing is a different contract                                                                                                                                               | Pending affected original adapter route only; actual trained GGUF LoRA loading is separately owned by W07/W02.                                                                                                                |
| N03 | Six-source edit geometry; W07                                                            | Six original geometries passed; contact-sheet review had visual limits                                                                                                                                                              | Recover six original hashes before replay. Preserve orientation/alpha/size; review individual outputs instead of treating contact-sheet geometry as semantic proof.                                                           |
| N04 | Qwen dark bottom band; W07                                                               | Transport passed; bottom-band visual gate failed; CPU whole-canvas restoration checked                                                                                                                                              | Reconcile later Qwen 2.1 implementation first. If same route remains, compare raw native VAE/latent and restored app output with the original input.                                                                          |
| N05 | Lettering/edit quality; W07                                                              | Old geometry passed with lettering defects                                                                                                                                                                                          | Preserve failed semantic/visual output; rerun only changed applicable native contract using original six identities.                                                                                                          |
| N06 | Upscale prompt semantics; W07                                                            | Old dimensions passed; prompt semantics failed                                                                                                                                                                                      | Distinguish registered native scale/geometry from semantic edit; same source/prompt/native route for affected replay.                                                                                                         |
| I05 | Joy 40-step timeout; W07/W09                                                             | INT8 Joy edit with dedicated encoder/Wan VAE: six-step 384×512 result changed tiles/placement; 40-step hit 180 s client timeout                                                                                                     | Pending one bounded actual request with server lifecycle evidence, source recipe and unchanged input. Timeout is a performance failure; six-step transport does not pass quality.                                             |
| M01 | Inbox live publisher / cross-filesystem; W03/W02                                         | Private startup tiny fixtures passed; 61 CPU checks at `1308b6b`; publisher response and cross-volume publication mocked                                                                                                            | Pending live exact-digest publisher classification and real cross-filesystem staging/sidecar atomic move; unknown/split/bundled remain explicit manual placement.                                                             |

The table contains C01–C09 (9), R01–R05 (5), B01–B03 (3), I01–I05 (5),
N01–N06 (6), and M01 (1): **29 rows**. C07 and B03 are retained policy/proof
rows; the **27-case pending/failed/blocked denominator** excludes these two.
This inventory does not replace W07's complete native capability matrix or W11's
menu/zoom review. It preserves the particular unfinished rows from prior packets.

### CPU evidence correction

The inherited `ChatApiAttachmentsTests` helper was named `run`, which overrode
`unittest.TestCase.run`. Its seven discovered cases executed zero tests, while
the old CPU runner could still return success. Such source/CPU claims are invalid;
the separately recorded real native API calls are unaffected. The helper now has
a distinct name, and both guarded CPU entrypoints compare executed leaf identities
and explicit skip dispositions with discovery. Canonical unittest method/class
decorators, per-test `SkipTest`, and setup fixture skips account only for their
matched discovered leaves; skipped subtests remain distinct. A skipped fixture
cannot excuse an unrelated suppressed case. Executed/skipped counts are reported
separately. All seven JSON/SSE/auth/tool/cancel/release tests actually ran and passed
after the repair. The regression also rejects partial suppression with inert real
child suites. This repairs test execution, not model or hardware qualification.

The saved-reasoning load fixture also now supplies its initialized runtime name,
backend label and fallback state, and checks that the captured load is ready.
It previously failed with a missing `runtime_name` on its test double. The repair
does not change the production reasoning/template control or qualify model output.

## Six original inputs

The coordinator's read-only manifest is
`C:/Users/johnmart19/.codex/task-records/rf-six-originals-20261111.json`.
It freshly verifies the unchanged `D:/Testing` originals. These hashes establish
input identity, not model execution. All six are RGB without alpha; transparency
cases require a separate owned derivative whose identity is also recorded.

| Input | Filename | Bytes     | EXIF-oriented size | SHA-256                                                            |
| ----- | -------- | --------- | ------------------ | ------------------------------------------------------------------ |
| N01   | `1.jpg`  | 7,138,382 | 3364×7480          | `7f320635a5864305c0cd08997982143478fc0db8757dfa99a828bce88bb8b04e` |
| N02   | `2.png`  | 1,681,752 | 928×1664           | `140b25f444a9d8d8ef0317fb53c686a1a6997532e8bf7c7d6ca14bf5dec67c1d` |
| N03   | `3.jpg`  | 3,453,513 | 2758×3270          | `9f55d4df18e7f0d7553e96e13f40481a34aa46f367131e7ddd53604280bdd5c3` |
| N04   | `4.png`  | 1,241,994 | 768×1344           | `751ea025c48e789b10fcf213b13c11688f2caea1442c75070249282ca572456a` |
| N05   | `5.png`  | 1,921,247 | 1344×896           | `7a3c171b46dfd1ba1d5d0986af154db23a0597f9231aa9ed3e74e381a76bf353` |
| N06   | `6.png`  | 1,580,441 | 928×1232           | `176e55181006c580b2874236a3d463b281ec4743f5e5a8ea0741b22460f96fa2` |

N01 is about 25 megapixels. Uniformly scale and pad it to the selected model's
compatible canvas before GPU work; retain raw identity and derived geometry proof.
These input labels do not establish which original a historical N-case used;
that association still comes from its original request/receipt.

## Finite final execution packet

No GPU, probe, model load, download, installation or generation is authorized by
this document. The coordinator schedules the final bounded sequential phase.
Use only existing small/quantized assets; absent closure means blocked, not a
download request. File presence/size was inspected without loading tensors on
2026-11-11: LFM2.5 Q8 (1,246,253,888 bytes), Qwen3 1.7B Q8 (1,834,426,016),
LFM3 Q4_K_M (1,099,627,808), Qwen 2.1 Q4_K_M (4,189,343,904), native BiRefNet,
Janku v7.77 and the existing Joy INT8/dedicated encoder/Wan VAE are available.
Presence does not replace their recorded SHA-256. Prefer Qwen3 or LFM2.5 for
embedded transport fixtures; their failed quality qualification remains unchanged.

1. Capture exact six-original manifest and model/component/runtime identities.
   If an original or its historical hash cannot be recovered, mark that old case
   blocked and use any new synthetic input only under a new case ID.
2. Run C01/C02/C05/C08 together with two tiny local sources containing independent
   fixed facts, one Unicode value and one numerical string. On each compatible
   engine, ask once for both exact facts, inspect cited admitted lines, test a
   small context and removal/cancel/session isolation. Record time to first token,
   indexing/query device and release. One retry is allowed only after a diagnosed
   app defect, with both outputs retained.
3. C03 uses already cached SmolVLM 256M/projector if its full closure is present;
   one original and one fixed visual question, native then embedded. Record direct
   image payload and projector identity separately from answer quality.
4. C06 uses the existing b12262 binary and compatible registered template, Off,
   fixed prompt and sufficient explicit output cap. Record serialized request,
   returned visible/reasoning fields and initialized capability. No default change.
5. R02/R03 each get one predeclared bounded real case. Capture rollback identity
   before an updater action and exact transferred bytes before interrupt/resume.
   Skip acquisition when no already required artifact fits the authorization.
6. R04 uses the smallest present compatible Chat model, one existing Janku image
   recipe at its documented small valid size, then the same Chat prompt. Unload
   between owners, capture peak RAM/VRAM and fallback reason, repeat this short
   sequence three times only if the first passes. A driver/device error ends work.
7. B01 remains an independent original-Bria qualification, never an install
   canary. W07 chooses one matched failing Qwen/outpaint/N-case pair only after
   source reconciliation. I05 gets one unchanged source-recipe request with a
   predeclared server/client deadline; no broad sweep or step reduction to pass.

For each output, retain original and derived hashes, individual visual review,
server completion/cancellation, provider/device evidence and owned-process cleanup.
Report unavailable hardware and upstream/runtime/model limitations explicitly.

## CPU preparation

Use `D:/RuinedFooocus/venv/Scripts/python.exe tools/check_cpu.py` from the isolated
worktree. Existing suites cover attachment ownership/citations, request capture,
reasoning, native update staging and local HTTP resume. New defects and commands
are recorded with their cohesive feature. CPU fixtures cannot close final cases.

| Feature ID                    | Verified CPU scope                                                                                                                                                  | Remaining boundary                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `document-parser-dependency`  | Audited pure-Python wheel/eligibility contract and five package-plan checks                                                                                         | Installer/environment qualification outside the exact coordinator-installed wheel is separate.              |
| `bounded-owned-subprocess`    | Five real inert child cases on Windows, Ubuntu 3.14.4 and Debian 3.13.5; cancellation, overflow, timeout and descendant-held pipes                                  | These are process ownership checks, not Linux runtime, driver or model qualification.                       |
| `executed-cpu-test-contracts` | Actual leaf execution/explicit skip accounting; zero/partial/forged suppression rejected; inherited API tests really run; saved-reasoning fixture identity repaired | A skip is reported as a skip, not an executed assertion. Model reasoning quality is unchanged.              |
| `bounded-chat-documents`      | 30 actual parser/attachment/manifest/API CPU checks, including JSON/SSE page citations and privacy                                                                  | Completion and embedding stages are fakes; no answer semantics, direct vision or GPU indexing is certified. |
| `bounded-runtime-inspection`  | 30 mocked-query/session/SDK/installer checks, including the actual native Gradio queued-cancel route; real canonical child ownership checked separately             | Real device queries and Vulkan updater acceptance remain R01/R02.                                           |
| `private-api-activity`        | 24 actual inert Job/state/UI-wiring/API caller checks, including running-cancel duration through callback release, bounded retention and no content capture         | Rendered UI and actual job performance remain separate.                                                     |

The local handoff/evidence root is
`cache/plan-execution-20261111/` in the isolated `rf-chat-acceptance` worktree.
`W09-posix-Ubuntu.json` and `W09-posix-Debian.json` record actual source hashes
and five successful standard-library-only cases. WSL Git cannot resolve this
Windows-managed worktree's Git pointer, so those receipts explicitly carry the
Windows-verified source reference rather than inventing a Linux checkout HEAD.
Pre-rebase receipts remain bound to their source identities. The original helper
review finding and initial missing preparation prerequisite are retained in the
handoff; final publication uses one complete canonical helper feature.

The final CPU evidence records the initially failing saved-reasoning fixture and
its focused repaired caller result separately. Browser, translation, GPU,
real Hub/Xet, semantic, performance and stability outcomes remain coordinator-owned.
