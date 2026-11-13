# Model registry resources

`features/models/model_registry.py` is the single reader for bundled model
annotations. `load_registry()` preserves the `entries`, `artifact`, `profile`,
`annotations_for` and `benchmarks` interface. It imports no application or tensor
runtime, fetches no metadata, reads no model weights and writes no cache.

`resources/model_metadata/index.json` uses schema 2. It maps stable profile IDs
to `families/` files, exact artifact IDs to `artifacts/<role>.json`, and evidence
to `qualification/` files. Each reference has a SHA-256 of canonical JSON values;
formatting changes do not alter metadata identity.
Shard schema 1 identifies the object type through `profiles`, `artifacts`, or
`profile_id` plus qualification/benchmarks. Duplicate JSON keys, unknown schema,
escaping paths, missing references and changed shard bytes fail with
`RegistryError`; optional metadata callers retain their existing recovery.

Index construction and ordinary artifact/profile reads do not read benchmark
history. Accessing a profile's `qualification` key or `benchmarks(profile_id)`
loads that profile's evidence only. `registry_document(path)` explicitly
materializes and validates the legacy schema-1 projection for audits and
migration tools. An explicit path selects that resource root, including audit
`--root` use; it does not substitute another checkout's catalogue.

The original migration identities remain recorded under `migration_baseline`:
167 profiles, 324 artifacts and 135 benchmarks at their historical source. The
image/Chat cleanup records exact retired identity keys and separately pinned
retained projection hashes. The splitter verifies all 142 retained original
profiles, 282 original artifacts and 132 benchmarks without shipping retired
metadata or download registries. Both historical hashes remain available for
Git-history comparison; a removed unrelated original record or changed retained
annotation fails verification.

URLs, paths, filenames and projectors remain owned by `model_catalogue/*.json`.
`annotations_for` checks the complete tuple before returning metadata. Qualification
describes its recorded artifact/runtime/device and never certifies another
quantization. The current registry has 219 profiles, 365 artifacts and 132
benchmark records. The separate [family-default reader](model-defaults.md) serves
geometry and sampling without opening qualification records.

[Native catalogue dispositions](native-model-catalogue.md) contain only reviewed
RF image/utility consumers. Native source recognition is not product admission;
new upstream classes, types and folder roles require an explicit useful RF image
or Chat consumer before becoming active resources.

Modern source tuples retain `native_class`, `native_family`,
`behavioral_variant`, `evidence.default_recipe_id` and their source workflow
component/schedule contract. `image_diffusion_companion` identifies a second
model consumed by a primary artifact; it must not be advertised as an ordinary
image checkpoint merely because its destination is `diffusion_models`. For
example Ideogram 4 references its unconditional model through `companions.image`.
Qwen Layered's RGBA codec is a separate `image_codec` artifact with its own
channel contract. See [modern catalogue extension](native-modern-image-catalogue.md).

Bernini-R's one-frame image source manifest binds the exact high/low pair,
UMT5 encoder, RGB codec and optional Lightx2v adapter. The low model has the
`image_stage_companion` role and appears as a component, with no ordinary image
activation. Both acquisition modes include the required low model, encoder and
codec; the optional adapter remains an explicit workflow choice. Image-specific
resource IDs reuse the upstream physical filenames and payload hashes, preserving
existing local shared components. The pair's 31,149,666,432 bytes and the four
required payloads' 38,139,379,607 bytes describe disk payloads, not runtime memory.
These publisher/header records do not qualify model execution or image quality.

The [extended image and utility catalogue](native-extended-image-catalogue.md)
adds `image_utility_model` for native utility payloads stored beside diffusion
weights. This role carries no ordinary checkpoint activation promise. Public
`auto_recipe_supported` and its reviewed evidence field must agree. Native
availability rows use specific publisher-access or ingestion-gap dispositions
when an exact matching tuple cannot be admitted.

To extend a resource, materialize the current document, change the relevant
profile/artifact/evidence and run `tools.split_model_registry.split` into an owned
temporary directory. Review the projection, then publish shards before the index.
Pass the current index's artifact ID-to-path mapping as `artifact_paths` when
preserving existing custom shards; omitted new artifacts use their role shard.
`profile_paths` similarly preserves custom family filenames. Only direct
`artifacts/<name>.json` and `families/<name>.json` paths for known IDs are accepted.
Never edit an index checksum to hide a tuple change. The splitter writes each file
atomically. Packaged resources are immutable within a release; this operation is
developer tooling, not a user metadata refresh or a model download.

```sh
python tools/split_model_registry.py
python tools/check_cpu.py test_model_registry.py test_model_registry_resources.py test_model_acquisition.py test_native_model_coverage_audit.py
```

The source archive packages these tracked JSON resources normally. User model
metadata, settings, outputs and runtime caches remain outside this schema and
are never migrated or deleted by its reader.

Historical migration acceptance on Windows/Python 3.12 used the six registry/acquisition/
native-audit/model-profile/Chat-browser suites: 67 CPU tests passed. The canonical
merged document digest is
`3c9953484f4571596323629082b21be525f5500f4b9493f8ba9106edc92347f5`; all
324 download identities hash to
`39b1460f216f0e29a361d34c31f3168756e50f3fae82824f1b471c2a5796cdd4`.

The 25-sample metadata-only comparison with baseline `ad6d015` loads one exact
Qwen artifact and its profile display name. Clearing Python reader caches gives
p50/p95 of 19.25/21.29 ms before and 4.89/5.18 ms after. Warm lookup is
0.66/0.71 ms before and 0.80/0.81 ms after. The OS page cache was uncontrolled
and warm; these are not process startup, UI, full-audit or hardware timings.
The source-bound local receipt is
`cache/plan-execution-20261111/W02-registry-lookup.json` (registry source SHA-256
`60dfbaa89ee43bcf082ff60943547441b3242eae6e76bc3449a577e2bed6bc37`).
