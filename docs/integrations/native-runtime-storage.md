# Persistent native runtime storage

Managed native installations belong under the writable data root's
`repositories/llama.cpp`, `repositories/onnx` and `repositories/sdcpp`. The default
is the application directory; `RF_DATA_DIR` retains its portable/read-only-install
meaning. Discovery creates nothing. A custom `LLAMA_SERVER` and external Windows
ML provider locations remain external and are never relocated automatically.

These directories retain executable/library payloads, local build sources/tools
needed at runtime, selected llama.cpp versions/backends, xllamacpp wheel-source
provenance and provider/optional-image verification receipts. xllamacpp's actual
wheel remains installed in the application Python environment. Cache removal must
not remove these installations. Server logs and disposable compilation/download
data are not evidence that a runtime is installed or qualified.

Existing installations used `cache/{llama.cpp,onnx,sdcpp}`. The explicit APIs in
`core/native_runtime_paths.py` take a required data root for migration:

- `plan_legacy_runtime_migration(data_root)` reads bounded file counts and bytes
  without hashing, native execution, package installation or directory creation.
- `migrate_legacy_runtime_storage(data_root)` copies through a sibling staging
  directory, checks recorded payload integrity and copied bytes, detects source
  changes, then publishes each whole runtime family. All legacy source data is
  retained. Existing destinations, including incomplete directories, are never
  merged or replaced; unrelated source clones such as `repositories/llama_cpp`
  are unaffected. Symlinks/reparse points and special files are refused.

Stop all native workers and installers before migration. The migration lock
coordinates cooperating installers, not a running older application. Inspect the
plan and free space for the exact installation root first. On failure, preserve
the source and reconcile the reported collision/incomplete payload explicitly.
After migration, existing loaders still check their original runtime/version and
provider receipts; moving bytes grants no new device or model qualification.
When acquisition is disabled and the current llama.cpp package directory is
absent, CPU/Vulkan discovery can reuse the already supported `b12327` or `b12110`
package for the same platform, architecture and backend. It requires the saved
acquisition receipt, canonical official archive identity and unchanged payload
hashes; source-pinned archive digests are also checked where available. This does
not authorize arbitrary older builds. Existing current-pin directories, explicit
updated selections and custom `LLAMA_SERVER` paths retain precedence. A corrupt
current installation fails rather than being hidden by an older installation.
Hardware inventory shares this bounded discovery but checks receipt metadata
only; activation verifies payload hashes and retains the existing device checks.
Inventory presence therefore grants no integrity or GPU qualification.
Owned SDK active-directory pointers are rebased to the copied sibling; arbitrary
external paths and original receipts remain unchanged.

Native directory publication shares `packaging/install_lock.py`'s bounded helper.
On Windows only, access/sharing errors 5/32 receive at most five retries with
1.55 seconds of total waits. Cancellation and source/destination guards run
between attempts; existing destinations are never replaced, and persistent
errors propagate. The same helper publishes initial and updated llama.cpp,
isolated ONNX and SDCPP bundles. ONNX preserves its prior-runtime quarantine and
restores it after failed/cancelled publication; rollback is not cancelled halfway
through recovery. A destination appearing during publication is retained, with
the prior runtime available in quarantine. This addresses observed failures without asserting their
external cause or changing filesystem permissions.

The helper does not migrate models or user outputs, reset selections, download
packages, launch a binary or delete a cache. Separate persistent-data owners cover
legacy model inputs, editor sessions, offline themes and preparation journals.
The runtime migration alone does not make every old cache directory disposable.

Focused fixtures use private inert bundles and the CPU guard:

```powershell
D:/RuinedFooocus/venv/Scripts/python.exe tools/check_cpu.py test_runtime_publication.py test_native_runtime_paths.py test_installer_llama.py test_cached_llama_runtime.py test_llama_update.py test_wsl_llama_bridge.py test_onnx_installer.py
```
