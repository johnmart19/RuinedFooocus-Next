# Numeric tensor files with rebuilt native patches

The [saved workflow tensor consumer](native-tensor-workflow-contexts.md) supports
the combinations below. It continues to serialize numeric data only. Native model
patches are rebuilt from their original installed sources for the admitted request;
the file does not contain executable callbacks, patchers or hook groups.

| Native feature                  | File consumer                                                                                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Hypernetwork                    | Locally receipted latent and numeric conditioning inputs/exports, with the exact source, strength, model, codec and encoder context.                                     |
| GLIGEN                          | Latent-only inputs/exports with the original source and literal layout rebuilt; conditioning files are refused because native `gligen` contains the live position model. |
| CLIP-only LoRA/checkpoint hooks | Numeric conditioning inputs/exports retain every native CLIP interval and pooled tensor. All captured source model strengths must be zero.                               |
| Nonzero model hooks             | Latent-only inputs/exports retain source/settings provenance and require a fresh native model-hook group. Live hook conditioning is refused by the serializer.           |

Selecting tensor I/O with a hypernetwork or GLIGEN automatically selects saved
workflow context. Bare external files cannot supply this provenance; they do not
prove which callback strength or layout produced their values. The existing
source and output-frame checks still apply. Changing a source file, hook source
setting, hypernetwork strength or GLIGEN box layout requires a compatible new
capture rather than silently dropping the selected transform.

Resolved hook sources are recorded from the immutable captured plan, including
their type, effective settings and file signatures. Off ignores stale hook fields.
Checkpoint and component identities include an opaque digest of the local
canonical path as well as file stamps, so same-basename files in different roots
cannot accidentally share a receipt. The paths themselves are omitted. Receipt
reads/writes remain beneath the configured cache and use the existing request's
private storage. No generic graph/file-manager endpoint is added.

Hypernetwork callbacks apply to the primary model's native `attn1` and `attn2`
interfaces. During an SDXL handoff the primary sampling stage uses that request
clone; the refiner retains its own attention. Metadata identifies the target as
`primary-model`. The native dimension/activation/codec checks and source leases
remain in force. Packed modern attention implementations without that legacy
UNet patch interface remain outside this particular native hypernetwork consumer.

GLIGEN still requires its compatible four-channel, scale-eight SD/CLIP canvas.
Its position-network context, UNet attention widths and unprojected pooled width
must agree. The box node consumes one pooled encoding, so scheduled nonzero CLIP
hooks cannot supply its box text. These are concrete native contracts; a file
serializer cannot reconstruct an unsupported attention or pooled layout from a
family label. Executable GLIGEN conditioning and live model-hook groups are
explicitly not captured as file state.

The final pipeline order preserves numeric saved-condition overrides followed by
fresh native model-hook attachment. Latent-only model-hook requests therefore
keep their newly rebuilt group. CLIP-only requests have no executable model group;
their native numeric interval data remains unchanged. Default tensor I/O Off adds
no file capture or tensor copies.

`python tools/check_cpu.py test_native_tensor_patch_contexts.py
test_native_tensor_context.py test_native_auxiliary_callers.py test_native_auxiliary_guidance.py
test_native_lora_hook_callers.py test_layered_generation.py test_sdxl_handoff.py`
checks actual native attention math, GLIGEN rebuilding, numeric serializer round
trips, scheduled CLIP encode bodies, model-hook rebuild identity, ASGI admission,
primary/refiner sampling and the composed pipeline callers. Tiny CPU fixtures
qualify these source contracts, not real-model quality, GPU execution or memory.
