# Startup timing evidence

The 2026-11-11 baseline used Windows checkout `D:/RuinedFooocus`, development
`ad6d015f3fd9843e9d427b010d11b336545dbff0`, Python 3.12.10, Gradio 6.31.0,
ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44` and GGUF integration
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`.

The harness ran the actual BAT through a new `cmd.exe /c` process from an unrelated
working directory. Each app used `--cpu --offline --skip-dependency-install`, a
private initially empty data directory/profile, disabled model preload and an
unused localhost port. It did not install packages, download models or perform
inference. Only its own process tree was stopped after HTTP 200.

| Baseline stage             |       p50 |       p95 |
| -------------------------- | --------: | --------: |
| `cmd.exe` process creation |   1.96 ms |   2.08 ms |
| Hidden console allocated   |  23.87 ms |  24.27 ms |
| First Python stdout        | 192.80 ms | 193.12 ms |
| Localhost `/` HTTP 200     |  16.485 s |  16.598 s |

Three runs reused the same private profile and warmed OS/dependency caches. p95
is the nearest-rank maximum for this small set; it is not a population estimate.
A preceding first-observed launch on another fresh private profile reached HTTP
200 in 22.944 seconds, with first stdout in 221.63 ms. OS caches were not purged,
so this is a fresh-profile observation rather than a reproducible cold disk test.

No pre-terminal regression was reproduced by this programmatic offline path.
The console was deliberately hidden and its allocation was detected separately
from stdout. Physical double-click, shell association and desktop-visible Windows
Terminal appearance remain unmeasured. PowerShell profile startup is outside
this `cmd.exe /c` path. HTTP 200 establishes server readiness; it does not establish
rendered browser usability or optional model/metadata readiness.

The first registered launcher revision, `9522be5`, has the same BAT source as
`ad6d015` (`git diff 9522be5 ad6d015 -- RuinedFooocus.bat` is empty). The shell
launcher subsequently gained first-use system-prerequisite handling. There is
no measured BAT-source regression in that range. A report of slow
desktop terminal appearance needs a separate physical-launch/terminal-host trace.

## After the changes

The same Windows interpreter, native pins, launch command and private-profile
conditions produced this three-run set. The recorded base was `d0c9df1` with W04
source changes; concurrent cache work was present, so readiness is observational,
not a source-isolated performance comparison. The receipt records exact W04 file
hashes. The runtime packet was first committed as `ec6b339`, then mapped to
`10b1be80be6a9b49928966299fa00ad3948eedcc` when the coordinator corrected an
unpublished prerequisite descriptor. The measured W04 files are identical;
the raw evidence retains its original observed revisions and file hashes.

| Windows stage                     |      p50 |       p95 |
| --------------------------------- | -------: | --------: |
| `cmd.exe` process creation        |  2.14 ms |   2.17 ms |
| Hidden console allocated          | 29.16 ms |  29.48 ms |
| First launcher stdout             | 32.13 ms |  40.56 ms |
| First Python stdout               | 87.59 ms |  91.22 ms |
| Environment preparation announced | 98.18 ms | 101.90 ms |
| Web UI import announced           |  2.934 s |   2.997 s |
| Server URL printed                | 16.677 s |  16.869 s |
| Localhost `/` HTTP 200            | 17.446 s |  17.698 s |

First feedback meets the one-second target on this path. Overall readiness did
not improve relative to the baseline; most elapsed time remains after the Web UI
import announcement. The decision is to keep source updates bounded and progress
visible, leaving UI/import optimization to a separately profiled change. There
is no inference-speed or artwork-readiness claim.

Ubuntu WSL used an owned source snapshot at
`/home/johnmart19/.cache/rf-startup-20261111/source w04 проба`: all other app source
was `ad6d015` plus the exact W04 overlay. It reused the existing
`/home/johnmart19/RuinedFooocus/venv` and native repository paths, without changing
that older application checkout. Native pins matched those above. Python was
3.14.4; both platforms had Gradio 6.31.0, pygit2 1.21.0, packaging 26.3 and
Torch 2.14.0+rocm10.1.0, with explicit CPU mode. No GPU workload ran.

The actual shell launcher ran from another directory with a space/non-ASCII source
path and private data. First stdout was 1.50–2.22 ms, first Python status
29.52–36.13 ms. HTTP 200 took 13.931 seconds on the fresh profile, then 8.867 and
8.886 seconds warm. Across three runs, HTTP p50/p95 was 8.886/13.931 seconds.
These are available Ubuntu WSL results, not a qualification of native Linux,
Debian or other hardware. Forced termination of only the owned process group
produced a Python semaphore-cleanup warning; no inference was active.

Windows focused runtime checks passed 27 tests with one native-shell skip.
The isolated Ubuntu snapshot passed the runtime plus launcher group: 29 tests,
one Windows-only skip. Actual launcher fixtures verify explicit/active interpreter
precedence, space/non-ASCII source paths, argument preservation, unbuffered output
and failure exit-code propagation. No application package was replaced.

## Update wait reproduction

The online bottleneck is separate from offline import time. A disposable Git
fixture pointed the actual updater at a loopback HTTP server that accepts a Git
connection and never responds. Only the final application bootstrap was stubbed;
the updater and libgit2 fetch ran in real child processes. The legacy `ad6d015`
updater connected and emitted no stdout during a bounded two-second observation,
then was stopped. The new updater emits status immediately, kills its fetch at
the 10-second deadline and continues to that bootstrap stub: first status was
90.47 ms and continuation 10.181 seconds. Offline reached the stub in 93.06 ms
without connecting; a healthy local Git remote reached it in 318.53 ms. These
single observations cover configured modes rather than estimating percentiles.
These updater-only timings
do not stand in for full application/browser readiness.

Local evidence is retained outside the source tree at
`C:/Users/johnmart19/.codex/task-records/ruinedfooocus-startup-20261111/`:
`baseline-windows-fixed.json`, `final-windows.json`, `final-wsl.json`,
`updater-reproduction.json`, their `.log.json` files and the measurement scripts.
The JSON binds source file SHA-256 values, raw per-run stage
times and the test conditions. An earlier harness cleanup hung after readiness;
that harness fault was corrected using native owned-process-tree termination and
is not an application startup failure.
