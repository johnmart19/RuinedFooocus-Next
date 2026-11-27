# Bounded child operations

`core/bounded_subprocess.py::run_bounded` owns explicit short-lived diagnostic
and document-parser children. Callers provide trusted argument arrays, a whole
operation deadline, per-query timeout, combined stdout/stderr cap and cancellation
check. Input uses a private pipe. Separate bounded pipe readers prevent stderr or
an unread stdin from blocking the owner; timeout, cancellation, overflow and failure
kill and reap the child. Child stderr is never included in an exception message.

The existing `core/child_process.py` session ownership remains in force. Windows
Job Objects close descendants and optionally enforce a process-memory limit.
POSIX children start in an owned session/process group; cleanup kills that group
even if its immediate parent has exited with descendants holding inherited pipes.
The
document worker additionally sets a POSIX address-space limit before parsing.
Runtime diagnostics use the same output/deadline contract without the document
memory cap, because importing a GPU runtime can reserve larger address spaces.
This helper is resource bounding, not a filesystem or network security sandbox.
Never construct its command from document content or model output.

`test_bounded_subprocess.py` uses real inert Python children for binary pipes,
oversized stderr, hung children, cancellation, expired shared deadlines and private
error handling, including an exited parent with an inert descendant retaining its
pipes. It imports no inference or device API in the children.
