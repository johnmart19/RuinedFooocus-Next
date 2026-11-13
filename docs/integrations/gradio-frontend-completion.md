# Native Gradio completion and hidden progress

`ruinedfooocus/ui/gradio_frontend.py` adapts the pinned Gradio 6.32.0 frontend
through the public `Blocks.launch(app_kwargs={"middleware": [...]})` hook.
Both owner and guest launchers keep `PrivateResponses` outermost. The adapter
serves corrected bytes for one native asset; it never writes the installed
package, changes import maps or replaces native event dispatch.

The native completion branch sets `stream_state="closed"` for ordinary SSE
events as well as input streams. Native `LoadingStatus` consequently creates
input progress entries for an ordinary completed callback. The single guarded
change sets that state only when `connection_type === "stream"`. Output
completion, actual input streams, validation/error branches, callback data,
queues and concurrency remain native.

The adapter has exact release contracts for these native assets:

| Gradio | Blocks asset         | Original SHA-256                                                   | Corrected SHA-256                                                  |
| ------ | -------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| 6.32.0 | `Blocks-WDOEXx6W.js` | `4b0f4aebda6d990941d1bea997feaa3fbc23f3068334494ff20cf1f6fd37a2cc` | `8f421d2ad9dbd2df54782338601265da5c2d6ea6f6b53e95dac6bebf2d70d0e8` |
| 6.31.0 | `Blocks-D1GWYZdr.js` | `c870d6727be0b3c596cb996e4a7cae42a333ff65bfd17e122c0a560231bc89d1` | `e00fa530ea3f35046f04669a8c9404dea2a9420a633e2216e723be5b1c27c41c` |

Each contract requires exactly one matching completion expression and verifies
the complete corrected hash. The older contract retains compatibility with an
existing 6.31 environment during upgrade; versions never share a minified patch.
An unsupported version, missing asset or changed bytes produces one warning per
process and retains the native asset. Review or remove the adapter when changing
Gradio. GET retains native gzip/Brotli negotiation and its encoded content length.
A weak SHA ETag identifies equivalent corrected content across encoding variants,
with `Vary: Accept-Encoding`. HEAD omits content length and encoding because native
compression determines those while generating GET content, as allowed by
[HTTP HEAD semantics](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.3.2).
Corrected responses always return 200, including stale conditional requests. Owner and
prefixed guest HTML/assets retain `private, no-store` response headers.

Before each launch, `normalize_hidden_progress(root)` sets only native
`BlockFunction` entries with `show_progress="hidden"` and
`show_progress_on is None` to `show_progress_on=[]`, then regenerates native
configuration. Native `None` otherwise resolves to every output even for a
hidden callback. Explicit scopes, full/minimal progress, functions, data,
cancellation and queue ownership are unchanged. This normalization is idempotent
and occurs after callback registration; it does not rewrite dynamically added
events.

The official [6.32.0 wheel](https://pypi.org/project/gradio/6.32.0/) still contains
the ordinary-SSE completion defect; its renamed minified variables require the
separate contract above. Native tracker and ImageEditor fixtures execute that
release's actual code, retaining positive controls for layout snapshots and
image export. An unknown fixture version fails rather than skipping assertions.

Upstream references checked on 2026-11-13:
[Gradio 6.32.0 dependency completion](https://github.com/gradio-app/gradio/blob/gradio%406.32.0/js/core/src/dependency.ts#L551)
and [native stream-state change](https://github.com/gradio-app/gradio/commit/0f889f075fe42409dbdeac510362e8307565e099).
[Gradio 6.32.0 release notes](https://github.com/gradio-app/gradio/releases/tag/gradio%406.32.0)
also document Pillow 13 support. Its wheel permits `pillow>=8,<14`; the pinned
ComfyUI requirements permit `Pillow>=10.1.0`. RF selects Pillow 13.0.0 on Python
3.11+ and Pillow 12.3.0 on Python 3.10, which Pillow 13 no longer supports.
Gradio explicitly converts mode `I` PNG output to `I;16`, retaining the earlier
16-bit clipping behavior without mutating the original image. Regressions also
retain RGBA alpha and text metadata. RF does not add a second image conversion.

Focused CPU validation uses the checkout interpreter:

```sh
python tools/check_cpu.py test_gradio_frontend.py test_guest_localization.py test_ui_runtime.py test_layout_progress_targets.py test_guided_result_ui.py
```

The tests execute the installed native JavaScript completion branch and tracker,
native Gradio metadata and data/error handling, and in-process regular/prefixed
ASGI requests, including raw gzip/Brotli/default/identity GET bytes and HEAD
metadata. They do not launch a listening server or exercise models. These
functional checks do not establish browser responsiveness or a latency cure;
actual candidate-source browser performance needs separate qualification.
