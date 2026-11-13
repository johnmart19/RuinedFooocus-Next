# Local SenseNova U1.5 shard bundles

Place the official `model.safetensors.index.json` and all eight
`model-00001-of-00008.safetensors` through `model-00008-of-00008.safetensors`
in one subfolder of a configured checkpoints or diffusion-models folder.
Use **Create → Models → Refresh**. The picker offers the folder's named index;
individual shards are hidden while that index is present. The image API uses
the same relative selection, for example
`image/SenseNova/model.safetensors.index.json` from `GET /v1/models`.
Incomplete, malformed and other-architecture indexes stay visible as one Unknown
bundle with an inspection error. Their eight partial files are not offered as
standalone generation models. The API reports `generation_available=false` and
`unavailable_reason` for that row and refuses generation before queuing it.
The selected model's component status shows the same error.

The native image engine loads this local source. SenseNova uses its native
tokenizer sentinel and pixel codec, with language weights in the diffusion
model; it needs no external text encoder or VAE. Component replacements are
refused. Use Own settings or complete saved settings until a matching sampling
recipe is qualified. This reader does not enable catalogue downloads or change
defaults.

## Source and reader contract

The reviewed primary source is
[SenseNova-U1.5-8B-MoT at a2462cc1b549bace2c0896143bbbcbb3fa217c76](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT/tree/a2462cc1b549bace2c0896143bbbcbb3fa217c76),
Apache-2.0, inspected on 2026-11-11. Its exact index is 113,453 bytes, SHA-256
`14bd9eda56dc3d520fe6127bf158b2eed292d3a7ac73667586390eb8e03491b9`.
Eight bounded HTTP header ranges contain 1,116 BF16 tensors with
35,065,708,928 tensor bytes in total. Published full-file sizes and SHA-256
values are retained in the source fixture; full weights were not downloaded
or hashed by this source check.

`features/models/sensenova_bundle.py` owns this named format. Index JSON is
bounded to 1 MiB, each safetensors header to 16 MiB, tensor counts to 100,000 and
declared tensor bytes to 64 GiB. It rejects duplicate JSON/tensor names, wrong
shard names, path traversal, resolved files outside the bundle folder, conflicting
metadata, unindexed keys, invalid dimensions/dtypes, gaps, overlaps and byte ranges
outside the actual files. BF16, F16 and F32 storage are accepted; a filename does
not establish architecture. The combined detached header must satisfy the pinned
SenseNova native discriminator. Header inspection never maps tensor data. Local
inventory reuses one detached inspection per index, keyed by all nine file
signatures, including missing-file state; it does not reparse all shard headers
for each card or API lookup. Submission still performs a fresh strict capture.

`component_profiles.snapshot_request` freezes the index, all eight source/resolved
paths, file signatures and header digests before queuing. `image_preparation`
carries nine local-only preparation records and holds their read leases through
native loading. The bundle identity participates in model caching; changed shards
invalidate the request even if the index is unchanged. Nested bundle readers share
the owned read scope rather than acquiring a second lease behind a waiting writer.

`backends/comfy/sensenova_bundle.py` calls the pinned
[`comfy.utils.load_torch_file`](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/utils.py)
once per captured shard. A unified dictionary holds those native tensor references;
native mmap/aimdo policy and per-file storage annotations are retained. There is
no on-disk merger, tensor concatenation, format conversion or overwrite of originals.
The dictionary feeds native `load_diffusion_model_state_dict`, whose actual config
must be `SenseNovaU15`. Diffusion and tokenizer reload factories retain the complete
bundle identity and reacquire all source leases. Cancellation and source checks run
before native reads, between shards, after construction and before cache publication.

## Verification and limits

`tests/fixtures/sensenova-u15-headers.json.gz` contains only the exact index and
the eight publisher JSON headers, plus source receipts. Its SHA-256 is
`baf4af263fd55be263c7647a2f61283f5747419f6520b2b728ff2c988256c0f6`.
Tests validate all publisher ranges and index assignments, then call the actual
pinned classifier on meta tensors. Tiny private fixtures exercise native per-file
readers, the production dispatcher, `load_base_model`, tokenizer reload, picker/API
capture, preparation, cache reuse and stale-source refusal. They do not construct
or run the full SenseNova network.

```sh
python tools/check_cpu.py test_sensenova_bundle.py test_api_model_discovery.py test_image_preparation.py test_component_profiles.py test_model_cache.py test_image_settings.py test_native_sensenova_clip.py test_native_extended_catalogue.py test_native_catalogue_dispositions.py
```

No model inference, rendered browser, output quality, performance, memory fit or
GPU stability is qualified here. Local file signatures bind queued requests; they
are not proof that locally installed tensor payloads match publisher hashes.
The catalogue retains no exact acquisition tuple for this bundle.
