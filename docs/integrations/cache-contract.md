# Cache identity and Maintenance

This inventory describes the versioned hash cache, incremental Library index,
derived Gallery thumbnails, verified source proofs, publisher artwork, native
image files and targeted Maintenance operations. The fixture comparison at the
end retains its original `ad6d015` baseline.
`C` means the configured `path_cache`,
resolved by `PathManager.model_paths["cache_path"]`. `D` means
`data_path("cache")`, under `RF_DATA_DIR` when set. They can differ. Maintenance
shows both absolute roots and deduplicates physical aliases. It does not move
existing files or copy external framework caches. Paths and metadata can contain
private filenames, prompts and source URLs.

| Entry/root                                                                                                           | Writer and readers                                                                                                                                          | Identity, invalidation and regeneration                                                                                                                                                                | Retention and cleanup                                                                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C/checkpoints`, `C/loras`, `C/embeddings`                                                                           | `models/model_handler`, `checkpoint_tools`, `model_previews`; `resource_cards`, `image_components`, `model_recommendations`, `core/util`, embedding browser | Legacy basename JSON, Civitai source data, embedded/manual artwork, `.preview.json` v2; explicit refresh or weight signature/source verification. Defaults use packaged registry plus local overrides. | Preserve: offline metadata and user artwork can be irreplaceable. Existing basename readers stay supported; no bulk migration/deletion.                                                                                                               |
| Packaged `model_catalogue`, `model_metadata`, `models/registry`                                                      | Source/registry loaders; capability/default/UI projections                                                                                                  | Download tuple digest, registry schema/profile/artifact/source revision; immutable bounded in-memory loaders invalidate by file signature. Qualification evidence remains separate from annotations.   | Read-only resources, never Maintenance targets.                                                                                                                                                                                                       |
| `C/hashes`                                                                                                           | `model_identity.cached_file_sha256`; Inbox, checkpoint tools, capability/source verification                                                                | Canonical path, size, mtime/ctime ns plus `model-sha256` parser v1; bounded versioned record. Old signature/SHA256 records remain readable. A changed file is rehashed, never substituted.             | Derived; explicit clear of recognized records under an inactive exclusive artifact lease. Hash miss is local I/O, including offline.                                                                                                                  |
| Safetensors/GGUF headers in memory                                                                                   | `model_identity`, GGUF metadata readers; family/native consumers                                                                                            | Bounded LRU keyed by exact file signature/parser; negative results are signature-specific.                                                                                                             | Process lifetime, no disk deletion or model rewrite.                                                                                                                                                                                                  |
| `C/card-previews`                                                                                                    | `ui_adapters` through Gradio PIL cache; card gallery                                                                                                        | Gradio content digest, cached warning/placeholder artwork.                                                                                                                                             | Preserve while UI references are active; currently protected because those readers have no cooperative artifact lease.                                                                                                                                |
| `C/images.db`, SQLite sidecars                                                                                       | `gallery/imagebrowser`, `gallery/index_store`; Library pages/refresh                                                                                        | Index schema v2, canonical source path plus size/mtime/ctime; configured root set and ordered first-root relative identity. Incremental transaction reuses unchanged metadata.                         | Derived. Refresh preserves prior committed snapshot until commit; cancellation/failure rolls back. Maintenance reset is a transaction under inactive exclusive artifact lease, never file unlink. Legacy `D/images.db` is retained when roots differ. |
| `C/library-thumbnails`                                                                                               | `gallery/thumbnails`, `gallery/imagebrowser`; Gradio grid copies                                                                                            | Source canonical path/stat identity, 1024-pixel bound and transform version; EXIF orientation is normalized in the derived WebP.                                                                       | Derived. Recognized copies are pruned toward 256 MiB / 2,048 files after cooperating readers finish. **Library thumbnails** supports size/clear; originals are preserved.                                                                             |
| `D/uploads` / `GRADIO_TEMP_DIR`                                                                                      | Bootstrap and Gradio upload/file cache; image editors, Chat, Gallery                                                                                        | Framework payload digest and session file references; original file identities retained separately for replay.                                                                                         | Active inputs, not expendable previews. Protected in Maintenance. Existing explicit `--clean-cache` is a startup-only option.                                                                                                                         |
| `C/preparation` (new), legacy `D/preparation`                                                                        | `core/preparation`, `hub_download`; subscribers and model publication                                                                                       | Destination canonical identity + URL, expected bytes/SHA256; complete file verified before no-clobber publish. Partial Hub transfers retain resume metadata.                                           | Protected resumable work, including queued subscribers. No automatic deletion or migration.                                                                                                                                                           |
| `C/inbox`, model Inbox sources                                                                                       | `model_inbox`, model handler; discovery/placement                                                                                                           | Per-source signature, destination collision handling, expected publisher size/digest and artifact leases.                                                                                              | Preserve pending/unknown payloads and journals; use Inbox actions, never cache clear.                                                                                                                                                                 |
| `D/embeds`, `D/rag/<revision>`                                                                                       | `chat/model_preparation`, `rag_preparation`; native Chat/retrieval                                                                                          | URL filename for legacy contexts; pinned MiniLM revision/files/digests, model/device/runtime options and source signatures in index identity.                                                          | Downloaded model/context inputs; protected. Attachment corpora/vectors are request scoped in memory and released with request/pipeline ownership, not a shared persistent corpus.                                                                     |
| `D/chat_sources`                                                                                                     | `chat/sampling_sources`; Chat defaults/model browser                                                                                                        | File signature, pinned review/source contract, validated remote repository identity; atomic JSON under artifact write lease.                                                                           | Preserve offline provenance; explicit source refresh, no generic clear.                                                                                                                                                                               |
| `C/clip-interrogator`                                                                                                | `image_processing/interrogate` and native clip-interrogator                                                                                                 | Native CLIP model/text-table keys. Previous models/text_encoders table location has existing compatibility handling.                                                                                   | Protected until native readers can participate in maintenance leases. Never treat actual encoder weights as cache.                                                                                                                                    |
| `D/themes`                                                                                                           | `ui/themes`; theme discovery/activation                                                                                                                     | Theme repository ID, schema and bounded validated token content; atomic cache publication.                                                                                                             | Preserve offline selected themes. Explicit theme refresh only.                                                                                                                                                                                        |
| `D/dependency-checks.json`, `D/package-plans`                                                                        | Bootstrap/dependency_cache/package_plan; startup/install diagnostics                                                                                        | Interpreter/prefix/ABI, source/revision, requirement bytes, installed dist-info metadata; package plan digest.                                                                                         | Keep package-plan receipts. Startup verification marker can be explicitly reset for next startup; root file is not a runtime artifact. Never install during clear.                                                                                    |
| `D/installer-tools`, ROCm detection helpers                                                                          | Packaging ROCm selector/planner                                                                                                                             | Package version/digest, OS/architecture/Python ABI and detection environment; native helper verification.                                                                                              | Protected offline installation/recovery payloads.                                                                                                                                                                                                     |
| `D/llama.cpp`, `D/sdcpp`, optional ONNX provider data                                                                | Native runtime installers, selectors, updater, WindowsML acquisition                                                                                        | Pinned release/package digest, OS/architecture/backend, selected runtime manifest and provider/device qualification. Build products include native compiler/runtime identity.                          | Installed runtimes, active server logs and rollback selections; protected. Dedicated updater/install actions own them.                                                                                                                                |
| `D/qwen-image-2.1-pe`                                                                                                | Explicit prompt-enhancement preparation                                                                                                                     | Publisher revision and system-prompt file identity.                                                                                                                                                    | Protected model workflow input; no implicit fetch on cache miss in offline mode.                                                                                                                                                                      |
| External HF hub/Transformers, Torch hub, rembg, Triton/HIP/CUDA/ONNX caches, pip/OS caches, `models/diffusers_cache` | Their upstream libraries/native runtime and configured environment                                                                                          | Upstream revision/content/ABI/provider/device keys, not a RuinedFooocus directory-name contract. ONNX compilation is provider-owned; no universal app compilation directory exists.                    | Not traversed, migrated or cleared by Maintenance. Rem bg model stores and Diffusers assets can contain weights.                                                                                                                                      |

The newer workflow writers also appear in Maintenance's read-only size inventory:

| Entry/root                | Writer and readers                                                                      | Identity and retention                                                                                                                                              |
| ------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C/model-sources`         | `model_identity.record_prepared_native_source`; installed acquisition/default selection | Exact file signature and verified preparation receipt plus native tuple identity. Protected: retaining proof avoids rehashing model files during browsing.          |
| `C/catalogue-previews`    | `CataloguePreviews`; Model Hub artwork cards                                            | Publisher artwork URL/content identity, bounded validation and atomic publication. Protected for active cards and offline browsing.                                 |
| `C/native-image-previews` | `native_image_formats.display_image`; Library and image display                         | Source canonical path/stat signature; derived 8-bit sRGB PNG. Protected while views/workflows may reference the display copy; original files retain their identity. |
| `C/native-image-exports`  | Library native-format export action; download/result component                          | Unique export result path and selected format/precision/color conversion. Protected user results that may be displayed, downloaded or saved.                        |

Maintenance counts both configured and data-root copies for these four families,
deduplicating root aliases. It reads sizes/stat metadata only, without opening
payloads or validating/rehashing a model. None appears in **Cache to clear**;
direct clear requests are rejected. The separate **Library thumbnails** category
is derived and supports explicit cleanup.

## Gallery thumbnails

`gallery/thumbnails.py` creates disposable WebP grid copies for ordinary still
images under the configured cache root. Small images can use their original path;
unsupported media and failed thumbnail work fall back to the existing display
path. Thumbnail identity includes source path/stat fields, target size and the
transform version. The version invalidates older copies when orientation handling
changes. A unique same-directory temporary file is atomically published only if
the source identity is still current. The source is never rewritten.

The Library keeps original paths in its records for metadata and image actions.
The Gradio grid receives a copied thumbnail path, so its own enlarged preview and
download controls can use that derived image. **Open in Editor** uses the original.

The writer holds an exclusive target lease; a page holds one shared thumbnail
directory lease until Gradio finishes copying all of its paths. Retention runs
once when that page finishes or is cancelled, rather than scanning the retained
cache after every image. The single-image helper uses the same batch contract.
Retention pruning and
Maintenance clear require a nonblocking exclusive directory lease. Active
readers defer pruning and make clear report a skipped entry. Pruning removes the
oldest recognized copies until both the 256 MiB and 2,048-file targets are met.
Unknown/manual files, nested directories, symlinks and reparse entries are
retained, so the bounds apply to recognized completed thumbnails rather than all
contents of the directory. Leases coordinate this process only.

The browser reuses native Gradio copies for its most recent page (at most 100
files). The reuse key includes the upload-cache root, display path and original
source signature; source replacement, a different destination or a missing copy
uses Gradio's native hash-and-copy path again. Reused files are touched as in
native Gradio. Native cache expiry may still delete a copy; each reuse checks
that it exists and recopies it through Gradio if missing. Concurrent page workers
replace their small reuse snapshots rather than mutating one another's entries;
an older worker can lose a future cache hit but cannot substitute a different
source. Original validation, automatic index refresh and latest-request
publication fences still run.

Focused fixtures cover cache reuse, bounded retention, concurrent writers,
page-level retention work, native Gradio delivery reuse/invalidation, cancellation,
copy/clear leases, failure fallback and EXIF orientation while preserving source bytes:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_gallery_page_work.py test_gallery_thumbnails.py test_cache_contract.py test_cache_maintenance.py test_cache_maintenance_ui.py
```

These source/CPU checks do not certify browser rendering, production storage
latency, image-generation quality or multi-process coordination.

## New derived record contract

`core/cache_contract.py` is used by the SHA256 writer. A record contains
`schema_version: 1`, `namespace`, `identity` and `payload`. The filename key is
SHA256 of compact sorted UTF-8 JSON containing the first three fields.
`identity` for local files includes `canonical_path` (platform-normalized),
`size`, `mtime_ns`, `ctime_ns` and the relevant `parser_version`. New consumers
must add source/model revision, runtime/provider/device and profile scope only
when those affect the payload. No unneeded profile/device field belongs in a
content hash. Registry schemas remain the responsibility of their own reader.

Reads are bounded to 64 KiB by default; unknown versions, malformed/oversized
records and mismatched identities are misses. Writers use same-directory
temporary files, flush/fsync and atomic replacement. The prior valid entry
survives disk-full/interrupted replacement. A cache miss never edits a weight,
profile, prompt or source input. Legacy hash records are read compatibly; no
legacy record is removed merely because the new writer exists.

## Ownership and accounting

Artifact leases come from the existing process-local `PreparationService`.
Writers/removers use exclusive leases. Hash readers retain the cache writer
lease while validating or producing a record; loaders own source-file read
leases, and hashing checks source stability before and after I/O. Index
scans retain shared index leases while active. Cleanup attempts a nonblocking exclusive
lease and skips active or queued preparations. Compute still uses the shared
compute admission domain; cache I/O does not grant a compute lease.

Maintenance counts regular files only, refuses symlink/junction/reparse escapes,
does not follow directory links, and reports skipped/error counts. It never
recursively deletes a directory. Roots, models, outputs and profile aliases are
checked before any target is eligible. Unrecognized files remain. Sizes are a
bounded best-effort snapshot, not a promise that disk usage is stable during
concurrent operations. Artifact leases coordinate this application process;
separate processes need their native SQLite/runtime coordination too.

CPU fixtures cover cold/warm reuse, schema errors, interrupted replacement,
custom/Unicode roots, alias escapes, active/queued leases, scan failure and
cancel/retry. Installed Gradio construction and callbacks verify the lazy sizes
panel, derived-only category choices and clear/regenerate against owned fixtures.
The host cannot create the directory-symlink fixture; that case is skipped while
the reparse guard is separately tested. Live Maintenance/browser, real-model
regeneration and hardware latency remain separate acceptance evidence.

A single 1,000-file, 640×480 RGBA PNG fixture comparison against `ad6d015`
reparsed 1,000 files in both baseline cold/warm refreshes (0.945/0.943 s).
Incremental cold refresh parsed 1,000 files (1.266 s); warm refresh reused all
1,000 and parsed zero (0.843 s). Both indexes were disk-backed. Cold indexing has
additional signature/transaction overhead; these local single-run times do not
certify production storage or browser latency. The source-hashed receipt lives
with external task evidence, outside the distributed source tree.
