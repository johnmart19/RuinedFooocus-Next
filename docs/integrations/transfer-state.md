# Read-only acquisition state

`PreparationService.transfer_status(request)` reports the captured request's
local transfer state without waiting for its artifact write lease, reading model
contents, hashing model files, starting a transfer or contacting a publisher.
The canonical destination and exact request tuple retain the preparation
service's single-flight ownership. A different request at the same destination
is a conflict; another session's subscriber is not silently retargeted.

Active progress is a small locked snapshot. Cancellation remains cooperative and
reports cancelling while the worker winds down. Missing cache reports absent.
A saved ordinary HTTP partial offers conditional resume only when its bounded
receipt matches the exact URL, expected digest/ETag and expected total size and
the partial does not exceed that total. The next server response must still
validate resume support; partial bytes are not verified model contents.

Stale/malformed/oversized receipts stay untouched during observation. The actual
transport retains its existing stale-partial cleanup and negotiation behavior.
SDK incomplete files are visible as unverified partial cache: the SDK still owns
identity, range negotiation and whether those bytes can be reused. Bounded local
traversal rejects staging paths that escape through symlinks or junctions.
Unknown inspection state is reported explicitly. Offline observation has the
same local-only behavior and does not create cache/model directories.

```sh
python tools/check_cpu.py test_preparation_transfer_status.py test_preparation_service.py test_hub_download.py
```

These CPU checks cover active and cancelled ownership, path aliases, absent and
SDK partial state, stale identities, oversized receipts and unchanged HTTP resume
callers. Live publisher/CDN resume requires a separate source-bound transfer
receipt; the tests do not certify every server's range or access behavior.
