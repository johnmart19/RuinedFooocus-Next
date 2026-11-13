# Shared still-image utility admission and dispatch

- **ID:** RF-ISSUE-001.
- **Status / decision:** Deferred, non-blocking; confirmed by the integration
  steward on 2026-11-11. No failing runtime behavior is recorded for this refactor.
- **Owner:** Generation/integration maintainers.
- **Source:** Grouped still-image utility source
  `0d2fb014d66c6a992eed786e678a690425194050`
  (`native-still-image-utility-workflows`), retaining the original composed
  `62099d027cd16328ad7ab68d35e9705752465d8b` identity for the caller evidence.
  The shared boundaries are
  [API admission/dispatch](../../ruinedfooocus/app/api_runtime.py)
  (`_image_operation`, `_native_vision_operation`),
  [asset capture](../../ruinedfooocus/features/generation/image_preparation.py)
  and the native utility consumer.
- **Evidence / reproduction:** Source review found utility-kind/admission decisions
  repeated across these boundaries while composing depth and SeedVR2. The steward's
  external `rf-layered-generation` evidence contains
  `cache/plan-execution-20261111/integration-depth4-callers.txt` (116 CPU checks) and
  `integration-seed-depth-callers.txt` (75 CPU checks). These are caller-contract
  results, not model/device or image-quality qualification.
  Their SHA-256 values are respectively
  `60246e4f4f978dbdc17fd565a7f8e9cb71a95351e5c9d43ccb90b4bf6be9ecb4` and
  `b2290704ac6159029ebea5f23b641ee00ef571fdfcf0b98581b926c9fee78fa6`.
  The same directory's `grouped-utility-source-equivalence.json`, SHA-256
  `92e5d629cee6ad75168c71c5bec6213c9e54929c8ae26ea07a3b454b3da68905`,
  binds grouped runtime/test/resource blobs to snapshot
  `4a07f953b17eac6ca42206134d290ebf7e98ddb2` and distinguishes later integration
  corrections from the original caller-test source. It does not rebind old tests
  as fresh execution on the grouped commit.
- **Impact / available boundary:** Future utility additions require coordinated
  edits in shared files and can conflict during integration. The composed routes
  already retain the early utility return without a generation checkpoint and
  their captured-source/cache cleanup. Publishing those workflows does not depend
  on this extraction.
- **Next bounded action:** After the current workflows ship, compare their existing
  admission/dispatch predicates and extract only a demonstrated common decision
  into one owner. Preserve utility-specific validation and captured request data;
  stop at that behavior-preserving extraction rather than designing another
  executor or general graph framework.
- **Acceptance:** Relevant existing API, UI/Library and preparation caller checks
  still cover the same supported/refused routes, early no-checkpoint behavior and
  cleanup. Record the exact before/after source and focused results; preserve all
  assertions and existing model-qualification limits. This refactor cannot close
  separate active acceptance requirements.
