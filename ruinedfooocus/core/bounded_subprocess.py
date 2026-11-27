"""Owned subprocess output, deadline and cancellation for explicit small operations."""

import os
import signal
import subprocess
import threading
import time

from ruinedfooocus.core.child_process import start_process


class BoundedProcessError(RuntimeError):
    """Bounded failure details for trusted adapters; public text stays generic."""

    def __init__(self, returncode, stdout, stderr):
        super().__init__("The subprocess could not complete")
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_bounded(
    command,
    *,
    input_data=None,
    env=None,
    timeout=15,
    deadline=None,
    max_output=64 * 1024,
    check_cancelled=None,
    memory_limit=None,
):
    """Return (stdout, stderr), never retaining more than max_output combined bytes.

    Only trusted callers construct commands. This is process ownership and resource
    bounding, not an OS filesystem/network sandbox. Readers drain separate pipes so
    neither a full stderr nor a child which never reads stdin can defeat the budget.
    """
    if timeout <= 0 or max_output <= 0:
        raise ValueError("Subprocess time and output budgets must be positive")
    end = (
        min(time.monotonic() + timeout, deadline)
        if deadline is not None
        else time.monotonic() + timeout
    )

    def check():
        if check_cancelled is not None:
            check_cancelled()
        if time.monotonic() >= end:
            raise subprocess.TimeoutExpired(command, timeout)

    check()
    process, job = start_process(
        command,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        memory_limit=memory_limit,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    lock = threading.Lock()
    overflow = threading.Event()
    used = 0
    output = [bytearray(), bytearray()]

    def read(pipe, slot):
        nonlocal used
        try:
            while data := pipe.read(4096):
                with lock:
                    remaining = max_output - used
                    output[slot].extend(data[:remaining])
                    used += min(remaining, len(data))
                    if len(data) > remaining:
                        overflow.set()
                if overflow.is_set():
                    return
        except (OSError, ValueError):
            pass  # The owner closes pipes after terminating a failed child.

    def write():
        try:
            process.stdin.write(input_data)
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    readers = [
        threading.Thread(target=read, args=(pipe, slot), daemon=True)
        for slot, pipe in enumerate((process.stdout, process.stderr))
    ]
    threads = readers + (
        [threading.Thread(target=write, daemon=True)] if input_data is not None else []
    )
    try:
        for thread in threads:
            thread.start()
        while process.poll() is None:
            check()
            if overflow.is_set():
                raise ValueError("Subprocess output exceeded its bounded limit")
            try:
                process.wait(timeout=min(0.05, max(0.001, end - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        while any(reader.is_alive() for reader in readers):
            check()
            for reader in readers:
                reader.join(timeout=0.01)
        check()
        if overflow.is_set():
            raise ValueError("Subprocess output exceeded its bounded limit")
        if any(reader.is_alive() for reader in readers):
            raise subprocess.TimeoutExpired(command, timeout)
        if process.returncode:
            # Trusted adapters may classify bounded output; exception text must
            # never echo a command, paths, document content or child stderr.
            raise BoundedProcessError(process.returncode, bytes(output[0]), bytes(output[1]))
        return bytes(output[0]), bytes(output[1])
    finally:
        if os.name != "nt":
            # The immediate process may already have exited while a descendant
            # retains our pipes. Terminate the owned session before closing them.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if job is not None:
            job.close()
        for thread in threads:
            thread.join(timeout=0.5)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
