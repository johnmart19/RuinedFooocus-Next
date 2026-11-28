# Local default LLM qualification

The user superseded the earlier default-change deferral on 2026-11-11 by
selecting [AtomicChat LFM3 with native llama.cpp](atomic-lfm3-default.md).
That explicit choice does not promote any historical candidate or establish
universal behavior. The dated results and decisions below remain historical
evidence for their original artifacts and runtimes.

This is a **qualification record and implementation input**, not a changed app
default. Audited 2026-11-09 at RuinedFooocus
`3ffdc0a0c3448544dcde90620d7ee7b285464885`, with final fixed-rubric runs at
`1546996` and `46c9995`, Windows/RX 7900 XTX.
The alternative-model handoff is research input, not an instruction to install
its entire catalogue. Existing models, profiles and default selection were kept.

## Historical decision — 2026-11-09

The latest requirement is a compact default for **3 GB and 6 GB VRAM**, with
image generation between Chat sessions. Larger-model comparisons have stopped.
Weight-file size is not peak VRAM; bounded context, one slot and release before
image generation are part of acceptance. Physical 3 GB/6 GB hardware is unavailable.

**No tested candidate is qualified as the universal default yet.** Granite 4.2
3B retained system instructions and reviewer uncertainty, but arithmetic/persona
failures remain. The smaller LFM2.5 and SmolLM3 follow-up below found template
defects and residual semantic failures. Gemma E4B's 5.34 GB weights exclude it
from this default, despite useful optional-model results. LFM3's embedded-runtime
incompatibility remains; leaving the current default unchanged does not qualify it.
The bounded LFM2.5 and Qwen3 runtime executions passed, while their declared
quality gates failed. The user explicitly deferred changing the default on
2026-11-09 so verified integration fixes can be published separately. LFM3 stays
selected by the existing default policy; its xllamacpp incompatibility and the
failed candidate qualifications remain visible limitations. This deferral is
not a quality pass or a promotion. No already-researched
compact candidate has established the required complete pass; this finding does
not establish that every small model is incapable.

### xllamacpp release recheck

On 2026-11-09, the latest official Windows Vulkan release is still
[2026.10.11870](https://github.com/xorbitsai/xllamacpp/releases/tag/v2026.10.11870-vulkan-windows-2022).
All 12 hashed members of its wheel match this checkout's installed package.
Reinstalling the same release would not add LFM3 support: its embedded
llama.cpp `b11870` at `74bca3950054eed1795f525cb9cf06bfe796492d`
[registers LFM2 and LFM2MoE, but not LFM3](https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/src/llama-arch.cpp#L135).
The downloaded LFM3 GGUF explicitly requires the `lfm3` architecture; renaming
it cannot supply the missing loader. Standalone `b12110` includes a
[dedicated LFM3 implementation](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/src/models/lfm3.cpp).
That source support is separate from model-quality and device acceptance.

The recheck used official release/submodule metadata, the wheel's RECORD and
local GGUF headers, without runtime imports or GPU execution. The wheel's full
archive digest was not independently recomputed. Local evidence is
`cache/plan-execution-20261109/xllamacpp-11870-lfm3-audit.json`.

| Tested model                                        | Download bytes | Both runtimes load/respond | Observed limitations                                                                                                                                                                                                                                 |
| --------------------------------------------------- | -------------: | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3.5-2B Q4_K_M                                   |  1,270,808,032 | Verified                   | Native called the image tool for a prompt-only request, changed blue mugs to beige, and claimed a cancelled generation succeeded. Embedded answered 0 instead of 7. Extended persona/system/vision-feedback cases were not run after these failures. |
| Qwen3-1.7B Q8_0                                     |  1,834,426,016 | Verified                   | Both obeyed a conflicting user instruction over the system-required name; Cat used human speech; reviewer text/uncertainty changed. Native raw rewrite lost the LoRA-tag delimiters.                                                                 |
| Qwen2.5-7B abliterated-v2 Q4_K_M, already installed |  4,683,074,560 | Verified                   | Both failed the same system-priority case. Embedded called the image tool despite prompt-only instructions. Reviewer uncertainty was lost. Stronger arithmetic in this sample does not qualify every other behavior.                                 |
| Granite 4.2 3B Q4_K_M                               |  2,244,011,552 | Verified                   | System priority, Cat voice, schema, cancellation and reviewer context worked. Arithmetic returned 24/27 instead of 7; a fresh CPU control returned 19. Some persona length/action limits were missed.                                                |

## Compact follow-up and template defects

Both further candidates loaded on both engines with context **2,048**, batch256,
microbatch128, one slot, seed8142 and response cap256. These runs made no image
tool executions. Weights were verified against immutable publisher metadata.

| Candidate                 |  Weight bytes | Corrected-template behavior                                                                                                                                                                                            | Native Vulkan allocation log                           |
| ------------------------- | ------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| LFM2.5 1.2B Instruct Q8_0 | 1,246,253,888 | Valid requested tool/cancel continuation and Mira system identity; refuses prompt-only and watercolor requests, calls image during explicitly text-only Megumin roleplay, loses reviewer uncertainty, arithmetic wrong | 17/17 layers; model1,186.25 + KV24 + compute16.50 MiB  |
| SmolLM3 3B Q4_K_M         | 1,915,305,312 | Corrected template restores image calls and cancellation; still calls image for prompt-only, fails Mira identity, loses uncertainty. Native boat prompt omits watercolor/exclusions; embedded preserves those          | 37/37 layers; model1,819.10 + KV144 + compute20.63 MiB |

Those allocation sums are about **1.20 GiB** and **1.94 GiB**, respectively.
They exclude driver/display allocations and do not establish measured peak memory,
3 GB hardware compatibility or coexistence with a diffusion model. Owned native
process exit and embedded-worker cleanup were observed; the app's full Chat →
image → Chat lifecycle remains task06 acceptance, not proven by these harnesses.

The exact ggml-org SmolLM3 GGUF embeds an older template that ignores standard
`tools` in favor of `xml_tools`, omits assistant tool-call history and leaves the
system turn unclosed when no tools are present. Current
[publisher template](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/chat_template.jinja)
adds the alias but retains the latter two defects. A temporary Apache-2.0-derived
override repairs them and initializes empty system defaults. Five CPU render
checks cover tool declarations, boundaries, call history, results and missing
system message. Nine affected cases per engine were then rerun. The initial
template-defective failures are not independent evidence against the weights.

LFM2.5's GGUF template also omits assistant tool-call history. Its current
[publisher template](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/b9d816ab510f9a28dca37cb182e9da8b0226e13c/chat_template.jinja)
includes it; eight affected cases per engine were rerun with that exact override.
The failures in the table persisted. Native `--chat-template-file` and embedded
`CommonParams.chat_template` supported these controls without modifying weights.
The SmolLM3 override remains a qualification fixture. The exact LFM2.5 publisher
template is now a bundled, identity-matched **candidate format correction**;
this does not qualify LFM2.5 or change the universal default. Both production
engines use the same load-profile helper as the corrected final fixture.

For the verified 1,246,253,888-byte Q8 artifact with SHA256
`f6b981dcb86917fa463f78a362320bd5e2dc45445df147287eedb85e5a30d26a`,
the helper selects `resources/chat_templates/lfm25-instruct-publisher.jinja`
(SHA256 `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6`).
The unmodified template, pinned source attribution and LFM Open License v1.0
are shipped together. Its native `--chat-template-file` and embedded
`CommonParams.chat_template` bindings are independent of the publisher-sampling
checkbox. Explicit custom native templates remain authoritative and do not
inherit a false effective-template identity.

Only absent candidate settings receive context2048/prediction cap256; batch256,
microbatch128 and one slot bound the candidate load. Existing saved context0,
prediction-1/None, explicit sampling and native CLI values keep their established
meaning. Other/custom or content-mismatched files receive no candidate override.
Clearing either numeric field in Settings passes a form-only removal sentinel
through validation before the paired profile save; it removes the override
rather than persisting JSON null. Raw startup/schema null semantics are unchanged.
Saved values outside this bounded recipe are not qualified by its allocation log.
The universal default and explicit model selections remain unchanged.

The final 28-request harness under `cache/default-llm-20261109/` now verifies the
local artifact and exercises this production profile, rather than independently
injecting a cached template. Both cached publisher evidence and shipped template
are checked before/after execution. Its CPU fixture/loader checks establish
configuration and serialization contracts only; final GPU semantics and task06
release/lifecycle acceptance remain separate and pending.

Compact evidence adds **100 requests**: 66 with embedded model templates, 34
with corrected overrides. All completed without request exceptions; semantic
failures above remain. No further compact-model download or GPU sweep is needed
before addressing these caller/template and tool-eligibility contracts.

## Earlier LM Studio results retained as optional evidence

These tests preceded the 3 GB/6 GB steering and are not a request for further
large-model testing. Eight standalone LM Studio weights were identified; one
DFlash draft and seven projectors were not treated as standalone chat models.
RF's RVN duplicate matched the LM Studio file by full SHA256 and was tested once.

- Stock Gemma4 E4B worked in both engines for the sampled tool, system, arithmetic
  and reviewer-grounding cases. Persona length/action limits and some enhancement
  drift remained. Its matching installed projector passed a separate synthetic
  shapes/text vision check and subsequent external-feedback mismatch check in
  both engines: four requests, no image generation, no whole-UI claim.
- E4B Heretic and Gemma12B Heretic had no demonstrated advantage over stock E4B
  for this role. Their sampled persona constraints also failed.
- Stock Qwen3.5 9B and its local abliterated variant failed the system-priority
  and arithmetic samples; the latter also called the image tool for a greeting.
  That exact local abliterated artifact is text-only.
- Saluki27B IQ2 and RVN27B Q4 had mixed system/persona/grounding results. Their
  sizes and, for Saluki, noncommercial license exclude them from this default.
- Installed GPT-OSS20B MXFP4 was a stronger optional tool/enhancement sample than
  its Q4_K_S copy; Q4 produced malformed persona prose and lost uncertainty.
  Both still missed persona constraints. Low reasoning is not disabled reasoning.

Except stock E4B, this expansion was embedded-runtime-only. It adds170 GPU text
requests plus the single fresh Granite CPU control to the initial124. Do not
generalize those results to native-runtime or UI acceptance. Combined text
evidence now totals **395 requests in27 reports, zero request exceptions**;
the four actual vision/feedback requests are recorded separately.

## What was actually exercised

Native llama.cpp **b12110 Vulkan** and installed xllamacpp **2026.10.11870
Vulkan**, with its embedded b11870 lineage, ran sequentially. No runtime packages
were replaced. Tests used the real RF `Server`/`ProfiledChatClient`, image-tool
schema/guidance, prompt-addition instruction/schema/result processing and
`attach_image_feedback`. Model and relevant source hashes are in each report.

The initial eight runs completed **124 text requests without request exceptions**:
11 per engine for Qwen3.5-2B; 17 per engine for each other model. This is a
transport result, **not 124 semantic passes**. Cases covered ordinary chat,
prompt-only versus image-tool selection, cancellation continuation, fresh request
isolation, arithmetic/format following, streaming, three constrained enhancement
scenes and rewrite. Extended checks added a clean system-versus-user conflict,
the bundled Cat/Zug/Megumin personas, supplied external-vision observations and
absence of visual input. Image tools were inspected but **never executed**.

Context was 4,096 tokens, seed 8142, normal response cap 256. Persona cap 150
truncated Granite's Megumin reply; that is an artificial cap, not proof that
ordinary app replies cannot finish. Some persona constraint deviations preceded
truncation. These one-seed samples are regression evidence, not an intelligence
ranking or a general reliability guarantee. Timing fields are incidental, not a
cross-model benchmark. Publisher-recommended ordinary samplers were applied
explicitly; enhancement/rewrite used the app's low-temperature path (0.1).
Granite's publisher instead recommends temperature 1 across tasks: that policy
difference needs a deliberate task-profile decision, not silent reinterpretation.

No app UI/API routing, new default persistence, image generation, CUDA/ROCm LLM
backend or cross-platform acceptance is claimed. The initial vision-context
check used synthetic supplied reviewer text, including an
uncertain character, through the app's real text attachment helper. A text model
must preserve that uncertainty and must not claim direct image inspection.
The bounded Vulkan runs recorded no device errors. Sandbox loopback access
blocked the initial native attempt (WinError 10013). The authorized localhost
retry passed.

Local evidence: `cache/default-llm-20261109/` contains the source-hashed harnesses,
runtime/model JSON reports, template repairs/CPU checks, allocation logs,
acquisition metadata and `summary.json`. Five downloaded compact candidates
remain in `models/llm/`; existing LM Studio models/projector were read in place.
Native processes were closed and embedded workers exited after each run.
No user settings or original weights were changed.

## Exact candidate provenance

| File                                           | Publisher/revision                                                            | SHA-256                                                          |
| ---------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Qwen3.5-2B-Q4_K_M.gguf                         | lmstudio-community/Qwen3.5-2B-GGUF @ bb84e11355a036e28f080c7793fa6d22b7c4e344 | 0bfe35afc9f05b7fac3fa04925e051ac7939a42a8a17ea11afc99701bea826cc |
| Qwen3-1.7B-Q8_0.gguf                           | Qwen/Qwen3-1.7B-GGUF @ 90862c4b9d2787eaed51d12237eafdfe7c5f6077               | 061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a |
| granite-4.2-3b-Q4_K_M.gguf                     | ibm-granite/granite-4.2-3b-GGUF @ c40945d71cd90f249a56985e8155551a9188dc30    | e0406663965846ae22a403456eb826ccce5f450840491f71952f18a7cb78e7d5 |
| Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf | Existing file; independently matches registry/publisher digest                | f57a8677cadbf6b23d200eb233f6b099c513024da9d2527f2be7dd1c6fe48e18 |
| LFM2.5-1.2B-Instruct-Q8_0.gguf                 | LiquidAI/LFM2.5-1.2B-Instruct-GGUF @ 8ed288026e23958ad9dfa92d53ed773a8eee7125 | f6b981dcb86917fa463f78a362320bd5e2dc45445df147287eedb85e5a30d26a |
| SmolLM3-Q4_K_M.gguf                            | ggml-org/SmolLM3-3B-GGUF @ 4965cb60b150737b68a0408c36aeefb65078f894           | 8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e |

All five downloads' full SHA-256 and sizes were verified against pinned Hub
metadata before publication into the local model folder. LFM2.5 uses the custom
LFM Open License1.0; the other four cards identify Apache-2.0. Official Qwen3-1.7B
currently offers Q8_0; do not invent an official Q4 URL. Text-only models cannot
gain vision merely by attaching a projector.

## Abliterated/uncensored alternatives already listed

The existing catalogue includes Qwen2.5-7B abliterated-v2, Qwen3.5-9B Heretic,
chuanli11 Llama3.2-3B uncensored, Rogue Creative 7B and Humanlike Chat 2.0 27B.
Hauhau Qwen3VL-8B Aggressive is in the **conditioning encoder** catalogue, not
the Chat weight list; reclassifying it requires a complete task/companion audit.

- [Qwen2.5 7B derivative](https://huggingface.co/huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2)
  publishes derivative evaluations: IFEval 77.82 versus base 76.44, MMLU-Pro
  42.03 versus 43.12, TruthfulQA 57.81 versus 62.46. These creator results do not
  prove improved intelligence or tool reliability of this quantization.
- [Qwen3.5 9B Heretic](https://huggingface.co/llmfan46/Qwen3.5-9B-Ultra-Uncensored-Heretic-GGUF)
  is the practical optional vision-capable uncensored candidate. Creator
  derivative checks are separate from the copied stock-Qwen benchmark table;
  mapping those scores to RF's exact historical v2 GGUF is unverified. Its
  larger language weights plus matching projector need their own acceptance.
- [Humanlike Chat 2.0](https://huggingface.co/LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF)
  publishes derivative instruction/tool evaluations but is approximately 16.56 GB
  at the selected Q4. Its image tower is retained, while reported fine-tuning and
  evaluation were text-only. It is not a compact-default candidate.
- [Chuanli 3B](https://huggingface.co/chuanli11/Llama-3.2-3B-Instruct-uncensored)
  is compact but lacks a derivative tool benchmark on the examined card.
  [Rogue 7B](https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF)
  documents verbosity/repetition risks, making it a creative option rather than
  the first constrained-tool candidate.

## Final fixed-rubric qualification and pending default decision

The final fixed LFM2.5 fixture ran once on 2026-11-09 at `1546996`: 28 calls
completed across native b12110 and xllamacpp b11870 with full 17/17 Vulkan
offload, one slot and context 2048. Both engines failed the unchanged semantic
rubric: unsolicited image calls during explicit no-image conversation, arithmetic
`13` instead of `7`, persona constraints and external-review uncertainty. The
12 passing cases do not cancel the 16 failures. The candidate remains unqualified;
these measured buffer logs do not establish peak VRAM or smaller-device support.

Qwen3-1.7B Q8 was the next single existing candidate, registered under its exact
official pinned artifact above. Its content-verified profile supplies context
2048, output cap 256, batch 256 and microbatch 128 without changing explicit user
values or the application default. Both engines retain the artifact's own
4100-byte publisher template, SHA-256
`57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361`.
Current base-repository tokenizer metadata contains a different, later template;
it does not silently replace this GGUF's template.

After a verified load, the initialized template exposes real Auto/Off/On thinking
controls. Qualification selects Off through the same saved/request control used
by Chat; it does not strip unfinished reasoning or append a hidden prompt. The
[publisher's nonthinking recipe](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)
uses temperature 0.7, top-p 0.8, top-k 20 and min-p 0. The run declared repeat
penalty 1.05 and presence penalty 0 separately as application sampling values.
Existing saved sampling and explicit request values remain authoritative.

The single Qwen3 run at `46c9995` completed all 28 calls on native b12110 then
xllamacpp b11870, sequentially, with 29/29 Vulkan layers, one slot and context 2048. Its result was **14 passes and 14 failures** under the unchanged rubric.
Both engines calculated 7 correctly but added explanation despite the required
exact-number output. Megumin exceeded sentence/action limits; Zug omitted the
required self-reference. Reviewer replies changed `OPEN 24/1` to `OPEN 241`,
dropped uncertainty, or changed the reported sign to `OPEN 24/7`. Cat's fresh
persona-switch case, prompt-only answers, requested image calls, watercolor
isolation and actual-call cancellation passed. Image calls were recorded without
executing them; no image was generated by either qualification run.

The serialized-request audit found one unchanged system message per request,
correct persona/greeting/current-turn boundaries, no inherited generation history,
exact tools and output caps, and the real initialized `enable_thinking=false`
control. No response ended at the length limit; the longest observed prompt had
647 tokens and its reply fit the 512-token persona cap and 2048-token context.
No exercised profile/template transport defect was demonstrated. This does not
isolate weights or quantization from numerical runtime behavior, or certify the
UI/API paths that the harness did not exercise. Root independently reviewed
representative raw failures and agreed with the failed-gate disposition.

| Candidate      | Final source | Runtime integrity                                 | Quality result            | Vulkan model / KV / compute allocation |
| -------------- | ------------ | ------------------------------------------------- | ------------------------- | -------------------------------------- |
| LFM2.5 1.2B Q8 | `1546996`    | Passed both engines, 17/17 layers, owned shutdown | Failed: 12 pass / 16 fail | 1186.25 / 24 / 16.50 MiB               |
| Qwen3 1.7B Q8  | `46c9995`    | Passed both engines, 29/29 layers, owned shutdown | Failed: 14 pass / 14 fail | 1743.77 / 224 / 13.50 MiB              |

Source, runtime and full model identities were unchanged during each run. Qwen3
used a private managed native cache verified against the official archive; the
earlier LFM2.5 run used the explicit existing native binary override. Native
servers closed and exited; embedded workers exited, with owned job cleanup.
The recorded buffers exclude driver/display costs and are not peak physical
VRAM. **Physical 3 GB/6 GB acceptance remains blocked by unavailable hardware**;
Chat-to-image-to-Chat lifecycle and actual image quality remain separate gates.

Local raw outputs, serialized requests, logs and manifests are preserved under
`cache/default-llm-20261109/final-lfm25-20261109T181004Z-e61be049/` and
`cache/default-llm-20261109/final-qwen3-20261109T182856Z-7feadc86/`.
Each directory contains `manifest.json` and the complete manual
`semantic-verdict.json`; the exact profile qualification metadata records their
SHA-256 identities. The Qwen directory also contains
`prompt-audit-and-candidate-decision.json`. Prior failed runs remain evidence.
Neither candidate is promoted and no saved model/default/profile is changed.
The universal both-runtime quality-default requirement remains unverified and
is explicitly deferred by the user for this release.

Retain failed fixtures. Separate a concise assistant system from tool guidance,
and verify the exact serialized messages/template/sampler before attributing a
failure to weights. Do not weaken assertions or reroll a failed seed until green.
Use a small fixed multi-seed instruction/grounding set for the leading candidate,
including a fresh-context arithmetic control and realistic persona output budget.
Do not extend this into another model sweep. Fix demonstrated message/template
and tool-eligibility gaps first, then rerun only affected compact cases. Keep
large/vision models optional and release their GPU state between stages.

The application must enforce tool eligibility/cancellation and render completion
truthfully; a model's statement is not evidence an image was generated. Preserve
original prompt tags through the existing rewrite guard. Explicit saved models
stay selected. Only promote a default after both engines, Chat/API/rewrite,
enhancement, persona switching and external-review grounding pass their relevant
checks. Genuine multimodal qualification is separate and optional.

Primary runtime/model references:
[Qwen3 official card](https://huggingface.co/Qwen/Qwen3-1.7B),
[Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B),
[Granite 4.2 3B card](https://huggingface.co/ibm-granite/granite-4.2-3b),
[embedded architecture registry](https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/src/llama-arch.cpp).
