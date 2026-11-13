# Verified acquisition source metadata

[First-use image preparation](image-preparation.md) consumes the same immutable
completion receipts for explicitly selected primary files and required image
companions. Source cache warnings retain verified files and keep the existing
Inspect recovery; selection/readiness callbacks remain passive.

Successful acquisition can seed native model defaults without hashing the same
multi-gigabyte file again. `PreparationSubscription.verified_signature` is an
immutable `file_identity` mapping captured under the artifact write lease after
the request SHA-256 was actually verified. Pending, failed, cancelled and
size-only preparations expose no certificate.

The acquisition session drains completed receipts once. Its UI completion path
calls `model_identity.record_prepared_native_source(path, request, signature,
cache_root)`; selection and Use callbacks remain read-only. The helper requires
the current canonical destination/size/mtime/ctime to match that receipt, then
joins the exact published URL, SHA-256 and bytes to a registered native artifact.
It refuses stale/path-conflicting receipts and makes no source claim for an
unreviewed artifact. It never reads weights, hashes them again, calls a publisher
or initializes a native runtime.

The versioned `model-native-source` envelope lives under the configured
`cache_root/model-sources/`, keyed by current file identity and parser version.
Publication uses the shared exclusive artifact lease and atomic record writer,
with source identity checks before and after publication. A changed file, missing
artifact, corrupt/unknown envelope or changed registry source is a cache miss.
Metadata write failure leaves the verified download intact and uses the existing
explicit Inspect/retry recovery; a successful transfer is not rolled back.

`ModelRecommendations.metadata` and `Models.get_models_by_path` read this optional
proof before background metadata refresh considers hashing. Only matching current
legacy metadata is merged. An unrelated basename cache cannot relabel the new
source; reading the proof does not rewrite or migrate that legacy file. The
existing explicit refresh path retains its normal cache writes. Saved user model
overrides remain in their existing settings store and retain precedence over the
source recipe. Source checksum verification still does not certify generated
quality, memory fit or GPU execution.

The proof record uses the same `_rf_file_signature`, `_rf_local_sha256` and
`_rf_verified_native_source` schema as explicit Inspect. Its identity is checked
against the current exact registry artifact on every read. Clearing this derived
record never removes the model, original metadata, artwork, prompt, profile or
output; regeneration requires a verified preparation receipt or explicit Inspect.
Canonical paths and source URLs are local metadata, not telemetry.

```sh
python tools/check_cpu.py test_prepared_native_source.py test_model_recommendations.py test_local_model_refresh.py test_model_files.py test_native_ordinary_remaining.py
```

Fixtures cover offline defaults and override preservation without duplicate hash,
source replacement, wrong destination, size-only receipts, unknown/corrupt cache,
registry drift, atomic write failure, mutation during publication, legacy basename
collision and both model-handler read/background-refresh callers. These CPU
fixtures do not download or load a real model.
