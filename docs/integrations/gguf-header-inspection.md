# Bounded GGUF header inspection

`features/models/gguf_metadata.py` reads selected metadata and tensor descriptors
with ordinary file I/O. `read_scalar_strings` and `read_gguf_header` share the
bounded stream cursor; neither constructs `GGUFReader`, creates NumPy mmap/tensor
views, allocates a tokenizer vocabulary, imports Torch or initializes a device.
`model_capabilities.local_architecture` uses the scalar reader too. Native model
loading and dequantization remain with the existing runtime.

The public shape reader still returns `(header, architecture)` with detached
Python lists, dictionaries, integers, booleans and strings. It preserves native
behavioral declarations, `config`, `is_512`, typed original shapes and supported
packed INT4/INT1 shape metadata. Retired Q4_PT forms still require conversion.
The native `gguf.constants.GGML_QUANT_SIZES` table supplies physical block sizes;
the app validates descriptors and payload ranges without reading tensor values.

Limits are 128 MiB of inspected header position, 16 MiB of captured detection
metadata, 100,000 fields/tensors, one million entries in a skipped array, eight
dimensions and 4 KiB keys/tensor names. Unused numeric arrays are sought past;
unused string-array entries are skipped by their lengths. Invalid types, strings,
counts, duplicate names, shapes, alignments, offsets and truncated ranges fail
explicitly. Opened-file and path identities are checked before returning. Windows
`fstat` and `stat` can report different ctime definitions, so each API is compared
against itself and file ID/size/mtime bind the opened source to its path.

The former full `GGUFReader` created vocabulary arrays and tensor mmap views,
then explicitly closed the mapping in `finally`. A malformed-shape exception kept
those views alive through its traceback. Formatting that traceback with standard
Python local capture reproducibly caused a Windows native access violation. The
stream reader retains no mapped data in success or error frames; it also closes
its ordinary file handle on errors. It uses neither a GC workaround nor retained
mapped tensors. This reproduces an unsafe reader path; attribution of the separate
full-app CPU failure requires repeating that exact UI sequence after the fix.

On this Windows/Python 3.12 checkout, one six-worker subprocess scan of six real
local encoder headers took 95.05 seconds with the old reader and 0.77 seconds with
the stream reader. Peak working set was 1,843,937,280 versus 40,611,840 bytes;
peak private commitment was 2,369,658,880 versus 532,013,056 bytes. All six complete
logical-header/architecture digests and file identities matched. These are single
CPU header-scan observations, not app startup, generation, GPU or sustained-memory
qualification. Local source-bound receipts are
`cache/plan-execution-20261111/W02-gguf-real-{before,after}.json`; the exact old
reader snapshot and disposable crash harness remain beside them.

```sh
python tools/check_cpu.py test_gguf_metadata.py test_gguf_streaming_metadata.py test_model_capabilities.py test_component_candidates.py test_image_components.py test_model_profiles.py test_chat_sampling_sources.py test_native_recipe_contracts.py test_compatible_family_defaults.py
```

Checks include native F16/Q8/Q4 and endian parity, physical versus logical packed
shapes, payload-read guards, vocabulary skipping, source replacement, concurrent
plain-Python results and safe retained-error formatting in an isolated subprocess.
