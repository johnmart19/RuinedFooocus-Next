# AtomicChat compact Chat default

On 2026-11-11 the user explicitly selected **AtomicChat LFM3 1.7B Uncensored**
as the compact default, superseding the earlier default-change deferral. This is
a selection decision, not a universal behavior or refusal-free guarantee. The
[historical qualification failures](local-llm-qualification.md) remain valid for
their original models, templates, runtimes and source revisions.

Fresh profiles and unset model selections resolve
`LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf` through the ordinary Chat catalogue. Native
`llama.cpp` remains the default processor and **Auto** remains its backend choice.
Explicit saved model/runtime/backend preferences retain precedence. The one
authorized local profile update changes only `llama_localfile` and `llm_runtime`,
with an exact backup and concurrent-write check; there is no automatic migration
of other users' selections or installer change.

The MiniMax source prerequisite supplies metadata for the current combined
registry fingerprint. Chat loads only the selected Atomic GGUF; this packaging
dependency acquires or loads no image-model weights.

## Exact artifact and source

| Field      | Identity                                                                                            |
| ---------- | --------------------------------------------------------------------------------------------------- |
| Repository | [AtomicChat/LFM3-1.7B-Uncensored-GGUF](https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF) |
| Revision   | `3ad712542eb9de349dbd28f2739b69c925cdec7d`                                                          |
| File       | `LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf`                                                               |
| Bytes      | 1,132,208,000                                                                                       |
| SHA-256    | `b22df65673b54ed326f18f74826d76f3b65dadc237ffd6306cd52be04544827d`                                  |

The pinned [creator card](https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF/blob/3ad712542eb9de349dbd28f2739b69c925cdec7d/README.md)
reports a projected rank-1 LoRA baked at W*=2 followed by activation-aware
quantization. The artifact is GGUF v3, `lfm3`, file type 15, with 287 tensor
descriptors. Its [LFM Open License v1.0](https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF/blob/3ad712542eb9de349dbd28f2739b69c925cdec7d/LICENSE)
is a custom license. No public matching full-precision Atomic checkpoint or
adapter was established in the bounded source audit. LiquidAI's public BF16
models establish lineage, not equality with these modified weights.

The separate Atomic family/artifact resources bind the exact download tuple,
size and checksum. They do not inherit LiquidAI benchmarks or older compact
candidate quality results. The original LiquidAI catalogue model stays selectable;
other weights and the supplied `D:/LFM3-1.7B-Q4_K_M.gguf` remain intact.

## Template, sampling and runtime

Use the artifact's own 12,129-byte `tokenizer.chat_template`, SHA-256
`f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818`.
It exactly matches the pinned [LiquidAI template](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/1da06bec6a819da10b54c67e5c90972b46d81871/chat_template.jinja).
The template defaults thinking on and requires mapping objects for historical
tool arguments; it rejects JSON-encoded argument strings. Its presence is not a
tool-quality pass. No template, prompt, geometry, LoRA tag or user request is
silently rewritten by this default change.

The Atomic card gives no sampling directives and the GGUF has no `sampling.*`
fields. Runtime defaults apply unless existing user, native CLI or request
overrides supply values. LiquidAI's base temperature 0.1/top-k 50/repetition 1.1
is not presented as an Atomic recommendation. The existing LFM3 native
`--reasoning-budget 0` load contract applies; explicit CLI values still win.
The unresolved LFM3 reasoning controls remain unavailable.

Native llama.cpp b12110 contains LFM3 support. The app's CPU-only official
`llama-b12110-bin-win-cpu-x64.zip` can be acquired through the existing hash-verified
installer for isolated CPU acceptance. Production Auto selection stays separate.
Installed xllamacpp 2026.10.11870/b11870 remains optional and cannot load LFM3;
its unsupported-architecture message is retained. No wheel or GPU runtime was
replaced to make the model selectable.

## Verification boundary

Focused CPU contracts exercise pinned download preparation, ordinary model/API
selection, fresh native/Auto defaults, explicit saved choices, sampler precedence,
reasoning gating and lazy registry integrity. Run:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_atomic_lfm3_default.py test_chat_model_identity.py test_compact_candidate.py test_model_profiles.py test_model_registry.py test_model_registry_resources.py
```

Real CPU response checks use a private CPU/offline/install-skip profile and the
app's `Server` plus `ProfiledChatClient`, with the exact verified CPU executable.
Two such responses completed on 2026-11-11. The short greeting returned visible
text. The system-name sample asked for Mira but returned LFM's identity, so that
bounded persona observation failed. The stored request and a separate render of
the exact template retain Mira in the system role; no Python/template fixed-name
override was identified. HTTP wire and native rendered prompt were not separately
captured. The owned process ended through Windows termination with exit code 1,
not a claimed graceful exit or stability pass. No retry or quality sweep followed.

These are limited transport observations. GPU execution, 3 GB/6 GB fit, tools,
grounded answers, image follow-up, roleplay quality and sustained lifecycle
acceptance require their own source/artifact-bound evidence; no new sweep is
implied by selecting this default.
