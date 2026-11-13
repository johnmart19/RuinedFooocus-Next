# Startup progress and bounded source updates

The updater prints a flushed status before reading the selected profile or
checking dependencies, then announces environment preparation before importing
the application bootstrap. This provides output independently of Git, package
checks and optional native initialization.

## Source ownership and deadlines

Application fetches and missing pinned-backend fetches run in a short-lived child
using the selected interpreter and the installed pygit2/libgit2. A fetch has a
10-second deadline. Timeout kills and waits for that worker; no detached network
task continues after startup proceeds. Startup reports the failure and continues
with the installed source. A failed application fetch never applies an older
cached remote reference.

First-use backend clones have a separate 300-second deadline and announce that
work before it starts. They clone into an owned temporary sibling and publish the
checkout only after successful acquisition. Interrupted acquisition does not
publish a partially installed backend. A backend pin already present locally
needs no fetch. Each backend check is attempted once during a launch, including
failures, even though both environment preparation and runtime import-path setup
call it. Missing or incompatible installed dependencies can still prevent startup;
a network deadline does not establish native compatibility.

The worker fetches objects and remote references only. The application updater
retains tracked-change detection, detached/diverged-history rejection, locked
branch/HEAD revalidation, SAFE checkout and ignored-file collision protection in
the parent. Successful fast-forward publication restarts with the exact Python
executable and original arguments so new dependency files are checked before
native libraries are imported. User modifications are never reset.

Offline policy is read before any fetch or dependency work, including the saved
profile, `RF_OFFLINE=1`, `--offline` and accepted CLI abbreviations.
`--skip-dependency-install` prevents package changes while still allowing a source
update. Explicit repair markers remain pending while installation is disabled.
Archive installations do not attempt Git application updates. These changes do
not alter package compatibility checks or optional runtime/model acquisition.

## Verification

Use the checkout interpreter:

```sh
python tools/check_cpu.py test_git_transport.py test_startup_updates.py test_source_updates.py test_backend_updates.py test_startup_options.py
```

Fixtures cover a real HTTP Git connection that never responds, worker deadlines,
failed-worker recovery, cached remote references, interrupted clone publication,
local/ignored/untracked source preservation, exact restart arguments and the
attempted-once bootstrap caller. No application packages or models are installed.
The transport uses existing pygit2 `Repository`, `clone_repository` and
`Remote.fetch` APIs; its process deadline is implemented with standard-library
`subprocess.run(timeout=...)` rather than a network callback or a background thread.

Measured launcher conditions and platform limits belong in the
[startup timing report](startup-timings.md). First stdout, HTTP readiness and
desktop-visible terminal appearance are different measurements. A CPU startup
timing does not certify GPU/model performance or browser usability.
