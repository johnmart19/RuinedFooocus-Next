"""Latest-input publication and bounded preparation races use deterministic barriers."""

from concurrent.futures import CancelledError
from contextvars import ContextVar
import asyncio
import hashlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.app.resource_lifecycle import LatestUIRequests


class UIRequestFencingTests(unittest.TestCase):
    def test_two_sessions_remain_independent_and_superseded_result_cannot_publish(self):
        fence = LatestUIRequests()
        pool = PreparationAdmission(max_workers=2, max_jobs=3)
        entered, release = threading.Event(), threading.Event()
        first = fence.begin(("ui", "first"), "library", 1)
        second = fence.begin(("ui", "second"), "library", 1)

        def blocked():
            entered.set()
            self.assertTrue(release.wait(5))
            return fence.publish(first, lambda: "stale first result")

        try:
            old = pool.submit_work(blocked, cancelled=first.cancelled)
            self.assertTrue(entered.wait(5))
            other = pool.submit_work(lambda: fence.publish(second, lambda: "second session"))
            self.assertEqual(other.result(5), "second session")
            newer = fence.begin(first.owner, "library", 2)
            self.assertTrue(first.cancelled.is_set())
            self.assertEqual(fence.publish(newer, lambda: "latest input"), "latest input")
            release.set()
            with self.assertRaises(CancelledError):
                old.result(5)
        finally:
            release.set()
            pool.close()

    def test_out_of_order_acknowledgement_duplicate_work_and_owner_mismatch_are_rejected(self):
        fence = LatestUIRequests()
        owner = ("ui", "one")
        latest = fence.begin(owner, "library", 4)
        stale = fence.begin(owner, "library", 3)
        self.assertTrue(stale.cancelled.is_set())
        self.assertIs(fence.get(owner, "library", 4, start=True), latest)
        for identity, version, start in (
            (owner, 3, False),
            (("ui", "two"), 4, False),
            (owner, 4, True),
        ):
            with self.assertRaises(CancelledError):
                fence.get(identity, "library", version, start=start)
        fence.finish(latest)
        self.assertTrue(fence.current(latest))
        self.assertTrue(fence.cancel(owner, "library"))
        with self.assertRaises(CancelledError):
            fence.publish(latest, lambda: self.fail("Cancelled result published"))

    def test_publication_and_invalidation_have_one_linearization_point(self):
        fence = LatestUIRequests()
        first = fence.begin(("ui", "one"), "library", 1)
        published = []
        fence.publish(first, lambda: published.append("committed snapshot"))
        fence.begin(first.owner, first.channel, 2)
        with self.assertRaises(CancelledError):
            fence.publish(first, lambda: published.append("late snapshot"))
        self.assertEqual(published, ["committed snapshot"])

    def test_work_pool_capacity_context_cancellation_failure_and_shutdown(self):
        pool = PreparationAdmission(max_workers=1, max_jobs=2)
        context = ContextVar("fixture_session", default="missing")
        context.set("captured owner")
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def blocked():
            calls.append(context.get())
            entered.set()
            self.assertTrue(release.wait(5))
            return "finished I/O"

        try:
            active = pool.submit_work(blocked, cancelled=cancelled)
            self.assertTrue(entered.wait(5))
            queued = pool.submit_work(lambda: self.fail("Cancelled queued work ran"))
            with self.assertRaisesRegex(RuntimeError, "full"):
                pool.submit_work(lambda: None)
            self.assertTrue(queued.cancel())
            failed = pool.submit_work(lambda: (_ for _ in ()).throw(OSError("fixture error")))
            cancelled.set()
            release.set()
            with self.assertRaises(CancelledError):
                active.result(5)
            with self.assertRaisesRegex(OSError, "fixture error"):
                failed.result(5)
            self.assertEqual(calls, ["captured owner"])
            self.assertEqual(pool.submit_work(lambda: "retry").result(5), "retry")
        finally:
            release.set()
            pool.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.submit_work(lambda: None)

    def test_shutdown_signals_active_work_and_cancels_queued_work(self):
        pool = PreparationAdmission(max_workers=1, max_jobs=2)
        entered, cancelled = threading.Event(), threading.Event()

        def work():
            entered.set()
            self.assertTrue(cancelled.wait(5))

        active = pool.submit_work(work, cancelled=cancelled)
        self.assertTrue(entered.wait(5))
        queued = pool.submit_work(lambda: self.fail("Shutdown ran queued work"))
        pool.close()
        with self.assertRaises(CancelledError):
            active.result(5)
        self.assertTrue(queued.cancelled())

    def test_owner_prefix_cancellation_invalidates_only_that_login_across_tabs(self):
        fence = LatestUIRequests()
        prefix = ("ui-login", "owner", "opaque login A")
        first = fence.begin(prefix + ("tab one",), "library-view", 1)
        second = fence.begin(prefix + ("tab two",), "library-selection", 2)
        other = fence.begin(("ui-login", "owner", "opaque login B", "tab one"), "library-view", 1)
        self.assertEqual(fence.cancel_owner(prefix), 2)
        for ticket in (first, second):
            with self.assertRaises(CancelledError):
                fence.publish(ticket, lambda: self.fail("Logged-out result was published"))
        self.assertTrue(fence.current(other))

    def test_successful_logout_middleware_cancels_ui_fences_by_authenticated_prefix(self):
        from ruinedfooocus.integrations.web_access import PrivateResponses

        fence = LatestUIRequests()
        prefix = ("ui-login", "owner", hashlib.sha256(b"fixture-login-A").hexdigest())
        first = fence.begin(prefix + ("tab one",), "library-view", 1)
        second = fence.begin(prefix + ("tab two",), "library-selection", 2)
        other = fence.begin(("ui-login", "owner", "different-login", "tab one"), "library-view", 1)
        registry = SimpleNamespace(cancel_owner=Mock())
        server = SimpleNamespace(
            cookie_id="fixture", tokens={"fixture-login-A": "owner"}, auth=True
        )

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def run():
            messages = []

            async def send(message):
                messages.append(message)

            await PrivateResponses(app)(
                {
                    "type": "http",
                    "path": "/logout",
                    "method": "GET",
                    "headers": [(b"cookie", b"access-token-fixture=fixture-login-A")],
                    "app": server,
                },
                None,
                send,
            )
            return messages

        with (
            patch("ruinedfooocus.app.resource_lifecycle.registry", registry),
            patch("ruinedfooocus.app.resource_lifecycle.latest_ui_requests", fence),
        ):
            asyncio.run(run())
        registry.cancel_owner.assert_called_once_with(prefix)
        self.assertTrue(first.cancelled.is_set())
        self.assertTrue(second.cancelled.is_set())
        self.assertTrue(fence.current(other))


if __name__ == "__main__":
    unittest.main()
