# Native image facts snapshot

`features/models/native_snapshot.py` owns immutable, source-derived still-image
facts. `native_snapshot.json` beside it is the reviewed local manifest/facts
envelope. The [effective-plan contract](model-defaults.md#effective-image-capture)
owns component selection, recipe precedence, operation admission and size capture.
The existing registry continues to own publisher advice and artifact provenance.
This snapshot adds no detector, sampler, model database or graph executor.

## Runtime contract

`load_native_snapshot()` reads and validates the bundled envelope once per process.
Fresh source clones use this tracked data with no user cache, writable data files
or vendor checkout. Clearing a user cache triggers no source build or network work.
`get_image_family_facts(family, snapshot=...)` resolves only recorded native and RF
family aliases. An unknown family returns `None`; admission still uses existing
native/header/reader rules, including compatible renamed and uncurated files.
`native_sampler_choices(snapshot=None)` returns immutable sampler/scheduler
tuples from those same bundled facts. Cold recommendation callers can validate
native examples without importing `comfy.samplers` or its device-management
dependencies. Missing or legacy enum facts return empty tuples for the existing
labelled fallback.
There is no source scan, Git command, network request, polling or weight hash on
generation. Callers capture the small `snapshot.metadata()` identity and relevant
`family.metadata()` facts; both return owned JSON values from immutable data.

Family facts include native classes/config matches, latent shape/scale contracts,
component prefixes and the existing positive RF component-role overlay. Official
graph model names/URLs/optional hashes are recorded separately as template models.
Graph connectivity determines the active example branch: disconnected helpers,
unselected literal native lazy-switch branches and same-loader alternatives are
optional. A model annotation becomes required
only when that connected loader's captured widget selects its exact filename.
These example roles describe a graph, never authorize downloads or replace a
request's actual captured active components.

Workflow sampling/canvas examples have `source.kind=native-workflow-example`;
they are distinct from publisher recommendations. Their sampling fields use RF's
`custom_steps`, `cfg`, `sampler_name`, `scheduler`, and optional `denoise`/`guidance`
keys. `default_size` contains width and height. `operation=generate` requires a
connected still-image output, empty latent and no source image. Save/Preview sinks
and pinned blueprint declared IMAGE-return interfaces count as source closures.
They do not grant an RF operation. Native class and
explicit artifact behavioral variant remain attached. Conflicting complete
examples produce no selected `example_recipe`. Consumers retain their existing
ordinary-family, variant, operation, sampler/scheduler and strict edit/inpaint
gates before using an example.

V1 `INPUT_TYPES`/`RETURN_TYPES`, native `IO` constants and V3 `io.Schema` ports are
read statically, including native V3 type/enum declarations, keyword port IDs and
literal multi-type ports. Missing optional widgets use explicit native schema
defaults; required missing values remain unresolved. Declared subgraph input
names, native widget defaults and complete proxy-widget maps bind each call.
Native literal primitives are followed only when their complete source body
returns that one input; no function is evaluated. Multiple sampling stages remain
staged even when their values are identical.

Dynamic schemas, unknown nodes, missing/mismatched links, muted
or bypassed active nodes, multiple stages and unvalidated subgraph widget
overrides remain explicit `unresolved` facts. Such an example cannot become an
automatic recipe. Subgraph input/output links are traversed as data; no native
source or node body runs. Configuration rows cannot supply recommended image
dimensions or sampling values.

Published sampling choices come from the pinned `comfy/samplers.py` public
`KSampler.SAMPLERS`/`SCHEDULERS` aliases, literal list concatenations and scheduler
mapping keys. Handler functions never run. Dynamic or mutated declarations block
candidate generation. Their declaration/choice identity joins the required
KSampler schema: additions require semantic review and removals block promotion
while preserving the last good snapshot. This is a source projection, not a
second handwritten sampler list or runtime registry.

## Explicit rebuild and promotion

Accepted sources are ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` and ComfyUI-GGUF
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`, matching bootstrap. The maintainer
tool reuses `audit_native_model_coverage.py` pinned-source, AST and registration
readers and `native_capability_inventory.py` recipe/graph helpers. Vendor sources
are read-only. Official annotation repository/path identities are joined to existing positive
catalogue artifact references to associate families; original URLs and optional
hashes remain attached, and model bytes/revision compatibility stays unqualified.
Filename similarity does not
detect architectures or add files to an allowlist.

Using the existing project interpreter and owned evidence paths:

```sh
python tools/update_native_snapshot.py --native-root repositories/ComfyUI --gguf-root repositories/molbal_comfyui_gguf --candidate tmp/native-image-candidate.json --compare ruinedfooocus/features/models/native_snapshot.json
```

The deterministic identity binds accepted pins, normalized consumed source/overlay
digests, blueprint/official JSON digests, official manifest/package metadata and
the parser version. No timestamps or checkout-specific absolute paths enter the
identity. Recipe JSON must match its installed official manifest. Unrelated
nonrecipe/artwork mismatches are retained as observed/unqualified evidence; they
do not claim the installed package is fully verified or trigger package repair.

Review the delta's added/changed/removed/unresolved schemas and graph roles,
affected family closures, overlay implications and actual caller checks. Then
pass the candidate to `refresh_native_snapshot(candidate_path, cache_path=...)`
or explicitly add `--promote PATH` to the tool. Changed consumed schema semantics
require `reviewed_semantics=True`/`--reviewed-semantics`. Required family/schema
or native-class removal, or breaking required port semantics, blocks promotion. Pin/parser changes
require an explicit code migration; there is no automatic unpinning or update.

The manifest and facts share one envelope, written in a same-directory temporary
file and promoted with `os.replace`. Validation, interrupted writing/replacement
or removal failure preserves the accepted file and last good in-process object.
Successful promotion invalidates only that path's reader cache. A first invalid
or missing load returns `None`, allowing the existing labelled fallback contract.

## Verification and limits

```sh
python tools/check_cpu.py test_native_image_snapshot.py test_native_capability_inventory.py test_native_model_coverage_audit.py
```

Fixtures cover native source nonexecution, deterministic extraction, V1/V3 ports,
selected subgraphs, optional alternatives, example conflicts, changed link and
schema semantics, source/parser/pin invalidation and atomic last-good recovery.
The bundled snapshot records source compatibility only. It does not qualify model
bytes, package artwork, runtime transport, visual quality, memory fit, performance
or GPU stability. Actual request/size/replay caller evidence belongs to the
effective-plan integration and keeps its tested source identities.
