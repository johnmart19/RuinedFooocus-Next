# Download link audit

Run the audit from the checkout using its interpreter. It uses the standard
library and pure bundled registry reader, without importing tensor/application
services. It does not initialize a device, install packages, change pins,
use Hub tokens or download model bodies. Offline inventory is the default:

```powershell
venv/Scripts/python.exe tools/audit_download_links.py --output temp/download-inventory.json
venv/Scripts/python.exe tools/audit_download_links.py --live --limit 20 --output temp/download-sample.json
venv/Scripts/python.exe tools/audit_download_links.py --live --compare-metadata --output temp/download-audit.json
venv/Scripts/python.exe tools/check_cpu.py test_download_link_audit.py
```

The third command deliberately audits every inventoried concrete URL. Start with
the bounded sample when checking connectivity. Defaults are four workers, a
12-second per-request timeout, one retry and at least 0.2 seconds between request
starts globally. `--workers`, `--timeout`, `--retries`, `--interval` and `--limit`
have enforced bounds. `--limit` counts unique concrete source URLs, not catalogue
rows. Optional metadata adds at most one lookup group per selected provider/repo;
each group has the same timeout/retry/rate limits. Redirects use urllib's bounded
manual redirect loop and normal TLS verification. Automatic urllib redirects
are disabled so their handlers cannot drain response bodies. Each hop is rate
limited, preserves HEAD or GET/Range, and closes its redirect body without
reading it. At most eight redirects (nine requests) are allowed per HEAD/GET
chain; a HEAD fallback starts a separate bounded GET chain. Cycles, embedded
credentials and HTTPS-to-HTTP downgrades are rejected. No
custom trust bypass is used.
Live runs emit compact JSON progress to stderr every 25 completed unique URLs
and at each stage's completion. A failed worker is saved as `probe_failure` on
its row; other URLs and metadata continue to be reported.

The JSON records source SHA256 values, checkout HEAD, timestamp, settings, owner
path and key/line, exact public URL, revision/version where identifiable, evidence
and status. The file hashes identify dirty source independently of HEAD. Multiple
owners of the same URL share one probe while retaining their own report rows.
Query values, user information and signed redirect parameters are redacted.
The same sanitization applies recursively to requirement strings, resolver
expressions, runtime declarations and other report fields. Signed redirect URLs
remain private request state; only sanitized redirect evidence is reported.
Credential-bearing/query source URLs remain unresolved rather than being probed
with credentials. Network exceptions contain only the exception class.

Coverage includes every category in `resources/model_catalogue/*.json`, including
checkpoints, text/vision/projector encoders, VAEs, ControlNets, LoRAs, face assets,
upscalers, geometry and LLMs. `tools/validation_assets.json` contributes exact
pinned validation artifacts. The model catalogue is authoritative for selectable
downloads; model metadata annotations and research citations do not replace it.

When schema-2 resources are present, the audit validates their exact catalogue
tuple annotations through the public registry interface. Download rows retain
artifact size/hash, variant, runtime pin, required companion IDs and evidence.
Separate `reference_only` profile rows preserve source revisions, source links
and dependency descriptions. Proposed downloads remain unresolved; inconsistent
annotations add an unresolved `registry-validation` row while preserving the
catalogue URL. Qualification/benchmark shards remain lazy and are not read or
hashed. See the [tool contracts](integrations/tools.md) and
[registry schema](integrations/model-registry-schema.md).

The tool also inventories URL literals and resolvable constant f-strings in all
first-party Python modules and root Python entrypoints, explicit package specs,
root requirements, `pip/modules.txt`, runtime bundle/native runtime declarations,
bootstrap Git archive pins and optional native runtime artifacts. It never executes
source to resolve expressions. Dynamic expressions are reported as templates/resolvers.
Static package indexes and API endpoints are labeled separately: a reachable
index cannot establish that a specific wheel exists for an ABI or device.
Probed bases/indexes use separate `endpoint_*` statuses, including
`endpoint_not_found`, `endpoint_authentication_required` and
`endpoint_access_denied`. Raw `probe_status`, reason and HTTP evidence remain in
the row. A composed API base returning 404 is therefore not a broken download
claim. Each endpoint row directs the reviewer to resolve/use the actual API
endpoint or package artifact. Top-level `artifact_counts` and `endpoint_counts`
keep these populations separate. `normalize_endpoint_result(row)` can apply this
interpretation to saved evidence without repeating network requests.
Ordinary source/reference URLs remain `reference_only` and are not live-probed.

The report explicitly retains unresolved acquisition paths: compiled pip plans
and transitive dependencies; llama.cpp platform/backend release assets and
companion libraries; Windows ML vendor catalogue providers; Transformers,
OpenCLIP/CLIP interrogator and rembg session files; and
user-selected Hub/Civitai results. These require exact target/dependency/selection
context. Resolve those manifests separately before claiming every downloadable
artifact has been checked. `complete_artifact_resolution` is always false for
this source inventory. No weights, vendor checkouts, environments, caches, user
profiles, outputs, inputs or prompt datasets are read. Qualification/benchmark
history and developer tools other than the validation manifest are
explicitly outside this acquisition inventory.

Availability uses HEAD. Servers returning 405/501 get a GET with `Range: bytes=0-0`;
the response is closed without reading its body even when the server ignores the
range. Metadata GETs are capped at 1 MiB. Evidence includes response code,
sanitized final URL and selected headers. HTML artifact responses are
`unexpected_content`. Status separates `reachable`, `broken` (404/410),
`authentication_required` (401), `access_denied` (403), `rate_limited`, `transient_failure`, `network_failure`
and other HTTP failures. A 404 can also hide a private repository; confirm access
before deleting a link. A network failure does not establish a broken artifact.
A 403 does not prove a gate: CDN policy, anti-bot rules, authentication and access
approval can all deny public requests. Concrete filename paths are percent-encoded
while existing escapes are preserved.

With `--compare-metadata`, reachable Hub/GitHub artifact owners share a cached
in-run lookup of Hub `main` SHA or GitHub's latest release. An older immutable
Hub pin remains reachable and is labeled `upstream_revision_differs` when `main`
differs. `artifact_change` remains `unknown`: a repository SHA can change because
of a README edit without any model payload change. This audit does not classify
that model as superseded or outdated.
A different GitHub latest tag is labeled `different_latest_release`; it is not
necessarily newer for that artifact's release series. Candidates are explicitly
unvalidated and never replace source URLs. A mutable `main` URL is not evidence
of the latest compatible model or an approved runtime. Header checks do not
validate payload hashes, compatibility, model quality, execution or GPU stability.

## Explicit UI acquisition and cached resources

The selectable catalogue and source inventory are separate from the current UI's
explicit acquisition actions. [Checkpoint components](model-folders.md), Chat
models and matching projectors retain their own preparation/cancellation paths.
Changing an installed component choice or saving an assistant does not download
weights. Saved assistant instructions and user model roots are user data, not
download links to repair or delete.

[Custom themes](themes.md) download a validated Gradio theme schema only when
**Download theme** is clicked; cached and built-in themes remain usable offline.
The action does not acquire theme code, remote fonts or arbitrary stylesheets.
An HTTP-success result alone does not establish that a theme passes that schema
contract.

**Hardware & backends → Optional image engine** separates **Install optional
runtime** from **Verify optional runtime device**. The pinned runtime archives,
payload manifests and notices are documented in the [SDCPP integration
guide](integrations/sdcpp.md). Verification checks the current process's exact
backend namespace; it does not qualify a checkpoint or operation. Normal optional
execution still requires matching acceptance evidence. Windows ML catalogue
availability and provider acquisition likewise do not certify model execution.
Offline/skip-install policy, explicit CPU authority and owned cancellation remain
independent of a link's availability. This audit does not grant hardware or
workload acceptance.
