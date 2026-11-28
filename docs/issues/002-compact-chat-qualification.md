# Historical compact Chat qualification failures

- **ID:** RF-ISSUE-002.
- **Status / decision:** Historical failures retained. The 2026-11-09 deferral,
  recorded by `700cc2a935987fe0a595d75ef26171b9d667f32f`, was superseded by the user's
  2026-11-11 [AtomicChat/native default selection](../integrations/atomic-lfm3-default.md).
  That choice is separate from quality qualification.
- **Owner:** Chat/model qualification maintainers.
- **Source:** The [qualification record](../integrations/local-llm-qualification.md)
  and [acceptance case C07](../integrations/remaining-acceptance.md) own the
  behavior rubric and artifact/runtime identities. The old fixed-rubric runs are
  labelled `1546996` and `46c9995`; these historical report references are not a
  claim of fresh qualification on the current application. `1546996` is not
  available in the current Git object database; do not invent its full identity.
- **Evidence / reproduction:** The recorded corrected-template LFM2.5 candidate
  passed 12/28 cases; Qwen3 passed 14/28. The reports retained system-priority,
  persona, tool-eligibility and reviewer-uncertainty failures despite successful
  transport on both engines. Exact prompts/templates and raw-report availability
  remain with the qualification record, not this card. No new model run was made
  to create this issue.
  The later user-selected Atomic artifact has one separate native CPU observation:
  a system request naming Mira returned LFM's identity. Its exact
  [source/template boundary](../integrations/atomic-lfm3-default.md) remains
  distinct from these older model results; two successful responses do not pass
  persona following.
- **Impact / available boundary:** These results do not qualify a universal compact
  default. They also do not establish results for the different Atomic artifact.
  Its LFM3 xllamacpp limitation remains visible. Verified integration fixes can
  ship without presenting those candidates as semantically qualified.
- **Next bounded action:** Only after a relevant source/template/runtime/model
  change and explicit resumption of qualification, recover one original failing
  case and its exact artifacts, compare the matched changed case once, and retain
  both outputs. Stop if identities cannot be recovered or the failure is unchanged;
  do not start a candidate sweep or change defaults as part of that comparison.
- **Acceptance:** A candidate needs the declared complete rubric on both named
  engines with source/artifact-bound evidence before promotion is considered.
  The user has separately resumed the default-selection decision; this does not
  satisfy that rubric. Case R04/R05 lifecycle/physical-memory acceptance remains
  separate; documentation, mocks or a changed selection cannot close it.
