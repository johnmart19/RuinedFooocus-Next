"""A superseded search cannot clear the current Chat browsing selection."""

from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

import gradio as gr

from ruinedfooocus.features.chat.model_preparation import ChatModelSession
from ui_test_helpers import function_from


class ChatModelBrowserRaceTests(unittest.TestCase):
    def test_delayed_refresh_or_transfer_projection_cannot_restore_an_old_candidate(self):
        entered, release = threading.Event(), threading.Event()

        class Catalogue:
            def invalidate(self):
                pass

            def __call__(self, source):
                entered.set()
                self.assert_release()
                return {"old": [("Q4", "old.gguf")]}

            def assert_release(self):
                if not release.wait(5):
                    raise AssertionError("Refresh barrier was not released")

        selected = []

        def select(source, model, quant, session, **kwargs):
            selected.append(quant)
            session.select(quant)
            return {"selection": quant}

        refresh = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "refresh_model_browser",
            dict(model_catalogue=Catalogue(), select_source=select, gr=gr),
        )
        session = ChatModelSession()
        session.select("old.gguf")
        session.active = "keep-active.gguf"
        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(refresh, "Local models", "old", "old.gguf", session)
            try:
                self.assertTrue(entered.wait(5))
                session.select("new.gguf")
            finally:
                release.set()
            result = old.result(5)
        self.assertEqual(result, gr.skip())
        self.assertEqual(selected, [])
        self.assertEqual(session.selected, "new.gguf")
        self.assertEqual(session.active, "keep-active.gguf")

    def test_slow_old_search_does_not_clear_a_newer_matching_search(self):
        entered, release = threading.Event(), threading.Event()
        calls = 0

        def catalogue(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                self.assertTrue(release.wait(5))
            return {"Family": [("Q4", "family.gguf")]}

        def clear(session):
            session.select(None)
            return {"selection": None}

        search = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "search_models",
            dict(
                model_catalogue=catalogue,
                model_label=lambda value: value,
                clear_model_selection=clear,
                gr=gr,
                llama_model="model",
                local_picker="picker",
            ),
        )
        session = ChatModelSession()
        session.select("family.gguf")
        session.active = "keep-active.gguf"
        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(search, "Local models", "Family", "family.gguf", "absent", session)
            try:
                self.assertTrue(entered.wait(5))
                current = search("Local models", "Family", "family.gguf", "Family", session)
                self.assertEqual(current["picker"]["value"]["selected"], "Family")
            finally:
                release.set()
            obsolete = old.result(5)
        self.assertEqual(session.selected, "family.gguf")
        self.assertEqual(session.active, "keep-active.gguf")
        self.assertEqual(obsolete, gr.skip())


if __name__ == "__main__":
    unittest.main()
