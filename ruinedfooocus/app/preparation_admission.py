"""Bounded I/O continuations before admission to the serial compute worker."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextvars import copy_context
import threading


class PreparationAdmission:
    def __init__(self, max_workers=2, max_jobs=8):
        if max_workers < 1 or max_jobs < max_workers:
            raise ValueError("Require 1 <= max_workers <= max_jobs")
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="admission")
        self.slots = threading.BoundedSemaphore(max_jobs)
        self._lock = threading.RLock()
        self._pending = {}
        self._closed = False

    @staticmethod
    def _check(cancelled):
        if cancelled.is_set():
            raise CancelledError("Preparation was cancelled")

    def submit(self, prepare, ready, failed, *, cancelled=None):
        cancelled = cancelled if cancelled is not None else threading.Event()

        def run():
            try:
                self._check(cancelled)
                result = prepare()
                self._check(cancelled)
                ready(result)
            except BaseException as error:
                failed(error)

        return self._schedule(run, cancelled, failed)

    def submit_work(self, prepare, *, cancelled=None):
        """Bounded CPU/I/O work uses the existing pool; it never grants compute admission."""
        cancelled = cancelled if cancelled is not None else threading.Event()

        def run():
            self._check(cancelled)
            result = prepare()
            self._check(cancelled)
            return result

        return self._schedule(run, cancelled)

    def _schedule(self, run, cancelled, failed=None):
        context = copy_context()

        def completed(future):
            try:
                if future.cancelled() and failed is not None:
                    failed(CancelledError("Preparation was cancelled"))
            finally:
                with self._lock:
                    self._pending.pop(future, None)
                self.slots.release()

        with self._lock:
            if self._closed:
                raise RuntimeError("Preparation service is closed")
            if not self.slots.acquire(blocking=False):
                raise RuntimeError("Preparation queue is full; retry later.")
            try:
                future = self.executor.submit(context.run, run)
                self._pending[future] = cancelled
                future.add_done_callback(completed)
                return future
            except BaseException:
                self.slots.release()
                raise

    def close(self, *, wait=True):
        with self._lock:
            self._closed = True
            pending = list(self._pending.items())
            for _, cancelled in pending:
                cancelled.set()
        for future, _ in pending:
            future.cancel()
        self.executor.shutdown(wait=wait, cancel_futures=True)


preparation_admission = PreparationAdmission()
