# Semantic startup dependency receipts

`core/dependency_cache.py` owns the `bootstrap` and `maintenance` strings in
`data_path("cache", "dependency-checks.json")`. The updater uses `bootstrap`
before importing package-dependent helpers; `prepare_environment` uses the same
receipt and the maintenance receipt before its pip upgrade and legacy
`llama-cpp-python` removal. A receipt records a successful dependency stage, not
device or model qualification.

Fingerprint version 2 hashes interpreter/executable/prefix/build identity,
Python ABI and cache tag, OS/architecture, installed `dist-info/METADATA` paths,
sizes and modification times, requirement assets and dependency-contract source
bytes. The contract inventory includes root `requirements*.txt`, `pip/**/*.txt`,
the BAT/shell/updater/bootstrap/argument entrypoints, and Python sources under
`packaging`, `core` and `backends`. This intentionally conservative inventory
revalidates changes, additions and removals in those runtime-contract owners,
including imported installer and backend-policy helpers. It never imports them
or probes a device to compute the receipt.

Git HEAD is excluded. Documentation and UI-only commits outside those contract
owners reuse a successful receipt. An archive's JSON metadata is hashed
canonically with only `source_commit` removed; backend pins and other distribution
policy still invalidate it. Archive revision identity and JSON whitespace alone
do not trigger dependency maintenance.

The module captures inherited backend/device visibility and package-index
environment values plus install/device CLI intent at import. CLI context covers
CPU/DirectML, GPU/device selectors, ROCm/CUDA/legacy profiles, selected settings
profile and the optional API closure, including argparse abbreviations. UI
connection options such as port and browser opening are excluded. Capturing
before startup applies those selections keeps generated `TORCH_PLATFORM` and
device masks from producing a different key at the end of every launch. A new
process with changed inherited selection or CLI intent still invalidates it.
These environment values are hashed; they are not written into the receipt.

Runtime package-plan compatibility and device/backend admission remain outside
these cached stages and run through their existing callers. The receipt does
not bypass later CLI application, saved profile backend selection, actual device
admission or native package compatibility checks.

An existing version-1 receipt legitimately misses once after this upgrade. The
next normal startup that permits dependency installation revalidates its stages
and writes version-2 hashes after success. Future docs/UI-only updates then reuse
them. A missing receipt, explicit reinstall request, changed dependency contract,
ABI, startup context or installed package metadata still requires validation.
Offline/skip-install startup does not seed a success receipt, and upgrading this
source does not alter user cache files by itself. An old opaque hash alone cannot
identify which historical input changed.

## Focused verification

```sh
python tools/check_cpu.py test_dependency_cache.py test_resource_paths.py
```

The isolated fixtures cover a real docs/UI-only Git commit, archive revision and
pin/policy changes, requirement/helper changes and removal, ABI and package
metadata drift, next-launch device/CLI changes, and stable startup normalization.
They write only owned temporary fixtures. They do not install packages, rewrite
the user's receipt, start the application or qualify GPU/model execution.
