# Private API activity

Settings → Connections → Local API → **Recent API activity** shows actual local
API job states. **Refresh API activity** updates the bounded snapshot; opening
Connections also refreshes it. There is no background poll, remote telemetry,
disk log or model/device probe. Disabled API launches expose no activity snapshot.

`app/api_activity.py` accepts only an allowlisted workload kind, state transitions
and monotonic timestamps. The real `app/api_runtime.py::Job` constructor, run,
terminal and cancellation paths update it; no executor, request copy, stream
buffer or alternate job registry is introduced. UI-owned internal runtime actions
are excluded. API/local API jobs use the same serial worker and its existing
admission, cancellation and resource release behavior.

The process retains at most 128 entries for one hour, pruning on access, and shows
the most recent 20 rows. Counts cover retained entries only, not a lifetime total.
Rows contain workload/state and measured wait/run/total milliseconds. Wait includes
queued job preparation; run covers the actual operation callback. Request parsing,
document extraction before job creation, client transfer and GPU-only time are
outside these measurements. Active durations increase until a terminal state;
post-completion cleanup does not convert completed work into cancellation.
For a running cancellation, the row stays running until the existing Job release
boundary, so callback cleanup after cancellation acknowledgement remains in its
observed run/total duration. Queued cancellation can release immediately. These
times do not measure cleanup outside the API Job callback or physical GPU idleness.

No prompt, document, attachment/source filename, result, model ID/path, owner/session
identifier, bearer key, token count, native error message or request body is
accepted or retained. Private owner UI events return only summary counts and
allowlisted rows. The internal random activity handle is never displayed.

```powershell
python tools/check_cpu.py test_api_activity.py test_ui_api_workspace.py test_api_live_contracts.py
```

These use actual Job state transitions with inert callbacks, bounded history,
running/failed/cancelled states, measured synthetic clocks, disabled API behavior
and private Gradio wiring. They certify source/CPU contracts. Real UI rendering,
copied example payloads and actual model performance remain separate acceptance.
