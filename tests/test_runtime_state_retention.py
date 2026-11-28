"""Completed worker owners must not retain an unbounded history of source state."""

import unittest
import weakref

from ruinedfooocus.app import resource_lifecycle as life


class CapturedSource:
    pass


class RuntimeStateRetentionTests(unittest.TestCase):
    def test_worker_access_bounds_owner_history_and_releases_evicted_source_state(self):
        state = life.RuntimeState({"pipeline": "shared service cache"})
        jobs = life.JobRegistry(capacity=1, retained=1)

        def complete(index):
            item = jobs.create(owner=("ui", str(index), "browser"))
            source = CapturedSource()
            source_reference = weakref.ref(source)
            item.state["captured_source"] = source
            with life.job_scope(item):
                state["last_image"] = f"{index}.png"
            jobs.finish(item)
            return source_reference

        oldest_source = complete(0)
        for index in range(1, 260):
            newest_source = complete(index)

        self.assertEqual(len(state.owners), 256)
        self.assertNotIn(("ui", "0", "browser"), state.owners)
        self.assertIsNone(oldest_source())
        self.assertIsNotNone(newest_source())
        latest = jobs.get(260, owner=("ui", "259", "browser"))
        self.assertIs(state.owners[latest.owner], latest.state)
        with life.job_scope(latest):
            self.assertEqual(state["last_image"], "259.png")
            self.assertEqual(state["pipeline"], "shared service cache")

    def test_eviction_does_not_clear_a_live_jobs_private_state(self):
        state = life.RuntimeState({})
        jobs = life.JobRegistry(capacity=2, retained=2)
        active = jobs.create(owner=("ui", "active", "browser"))
        source = CapturedSource()
        active.state["captured_source"] = source
        with life.job_scope(active):
            state["last_image"] = "live.png"
        for index in range(260):
            item = jobs.create(owner=("ui", str(index), "browser"))
            with life.job_scope(item):
                state["last_image"] = f"{index}.png"
            jobs.finish(item)

        self.assertNotIn(active.owner, state.owners)
        with life.job_scope(active):
            self.assertEqual(state["last_image"], "live.png")
            self.assertIs(active.state["captured_source"], source)
        self.assertIs(state.owners[active.owner], active.state)
        self.assertEqual(len(state.owners), 256)


if __name__ == "__main__":
    unittest.main()
