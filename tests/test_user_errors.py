"""App error messages reach the browser; unexpected failures stay generic."""

import asyncio
import inspect
import unittest

import gradio as gr

from ruinedfooocus.ui.user_errors import as_user_errors, apply


class UserErrorTests(unittest.TestCase):
    def test_app_messages_become_gradio_errors_with_the_same_text(self):
        for error in (
            ValueError("Select an image in Library first."),
            PermissionError("Sign in again."),
            FileNotFoundError("The model is not installed."),
        ):

            def fail(error=error):
                raise error

            with self.subTest(error=type(error).__name__):
                with self.assertRaises(gr.Error) as raised:
                    as_user_errors(fail)()
                self.assertEqual(raised.exception.message, str(error))
                self.assertIs(raised.exception.__cause__, error)

    def test_unexpected_failures_are_not_turned_into_user_messages(self):
        def fail():
            raise KeyError("internal")

        with self.assertRaises(KeyError):
            as_user_errors(fail)()

    def test_generator_async_and_signature_are_preserved(self):
        def stream(text, request: gr.Request = None):
            yield text
            raise ValueError("Stopped: no model.")

        async def answer(text):
            raise ValueError(f"No answer for {text}.")

        async def updates(text):
            yield text
            raise PermissionError("Not yours.")

        wrapped = as_user_errors(stream)
        self.assertTrue(inspect.isgeneratorfunction(wrapped))
        self.assertEqual(inspect.signature(wrapped), inspect.signature(stream))
        values = wrapped("a")
        self.assertEqual(next(values), "a")
        with self.assertRaisesRegex(gr.Error, "no model"):
            next(values)
        self.assertTrue(inspect.iscoroutinefunction(as_user_errors(answer)))
        with self.assertRaisesRegex(gr.Error, "No answer for x"):
            asyncio.run(as_user_errors(answer)("x"))

        async def drain():
            return [item async for item in as_user_errors(updates)("b")]

        self.assertTrue(inspect.isasyncgenfunction(as_user_errors(updates)))
        with self.assertRaisesRegex(gr.Error, "Not yours"):
            asyncio.run(drain())

    def test_real_events_show_the_message_and_keep_request_injection(self):
        seen = {}

        def undo(text, request: gr.Request):
            seen["request"] = request
            raise ValueError("There is no prompt edit to undo.")

        with gr.Blocks() as app:
            box = gr.Textbox()
            button = gr.Button()
            button.click(undo, inputs=box, outputs=box)
        apply(app)
        apply(app)  # applying twice keeps one wrapper
        block_fn = next(iter(app.fns.values()))
        self.assertTrue(block_fn.fn._rf_user_errors)
        self.assertIs(block_fn.fn.__wrapped__, undo)
        with self.assertRaisesRegex(gr.Error, "no prompt edit"):
            asyncio.run(
                app.process_api(
                    block_fn=block_fn,
                    inputs=["draft"],
                    request=gr.Request(session_hash="fixture"),
                )
            )
        self.assertEqual(seen["request"].session_hash, "fixture")


if __name__ == "__main__":
    unittest.main()
