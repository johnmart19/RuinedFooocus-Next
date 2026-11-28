"""Combining neighbouring listeners of one event: batches, order and results."""

import unittest

import gradio as gr

from ruinedfooocus.ui import event_coalescing


class EventCoalescingTests(unittest.TestCase):
    def build(self):
        calls = []
        with gr.Blocks() as app:
            choice = gr.Radio(["a", "b"], value="a")
            other = gr.Textbox("other")
            first, second, third = gr.Textbox(), gr.Textbox(), gr.Textbox()
            chained_out, reader_out, scripted = gr.Textbox(), gr.Textbox(), gr.Textbox()
            pair_a, pair_b, keyed, locale = gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox()

            def one(value):
                calls.append("one")
                return f"one:{value}"

            def two(value, extra):
                calls.append("two")
                return gr.update(value=f"two:{value}:{extra}", visible=value == "a")

            def pair(value):
                calls.append("pair")
                return value.upper(), gr.skip()

            def by_component(value):
                calls.append("keyed")
                return {keyed: f"keyed:{value}"}

            def with_request(value, request: gr.Request):
                calls.append("request")
                return f"{value}:{request}"

            def as_dict(values):
                calls.append("dict")
                return f"dict:{values[choice]}:{values[other]}"

            def reads_first(value):
                return f"reader:{value}"

            def stream(value):
                yield value

            choice.change(one, choice, first)
            gr.on([choice.change, other.change], two, [choice, other], second)
            choice.change(pair, choice, [pair_a, pair_b], queue=False)
            choice.change(by_component, choice, [keyed])
            choice.change(with_request, choice, locale)
            choice.change(as_dict, {choice, other}, third)
            # Not in the first batch: reads what it writes, has JavaScript, is chained, streams.
            choice.change(reads_first, first, reader_out)
            choice.change(one, choice, scripted, js="(value) => value")
            choice.change(lambda value: value, choice, chained_out).then(
                lambda value: value, chained_out, chained_out
            )
            choice.change(stream, choice, chained_out)
            other.change(one, other, first)
        names = dict(
            choice=choice,
            other=other,
            first=first,
            second=second,
            third=third,
            pair_a=pair_a,
            pair_b=pair_b,
            keyed=keyed,
            locale=locale,
        )
        return app, calls, names

    def test_plain_neighbours_are_combined_and_order_is_kept(self):
        app, _, names = self.build()
        choice = names["choice"]
        key = (choice._id, "change")
        before = [function._id for function in app.fns.values() if key in function.targets]
        report = event_coalescing.coalesce(app, [choice, names["other"]])
        self.assertEqual(report, {choice._id: [6], names["other"]._id: [2]})
        after = [function for function in app.fns.values() if key in function.targets]
        combined = after[0]
        # The combined request stands where its first member stood; the rest keep their order.
        self.assertEqual([function._id for function in after[1:]], before[6:])
        self.assertFalse(combined.queue)
        self.assertEqual(combined.trigger_mode, "always_last")
        self.assertEqual(combined.show_progress, "hidden")
        self.assertEqual(len(combined.outputs), 7)
        # A listener is detached only from the events it was combined for.
        two = next(
            function
            for function in app.fns.values()
            if getattr(function.fn, "__name__", "") == "two"
        )
        self.assertEqual(two.targets, [])
        self.assertEqual(app.config["dependencies"][0]["id"], next(iter(app.fns)))

    def test_the_combined_request_returns_what_each_listener_would(self):
        app, calls, names = self.build()
        event_coalescing.coalesce(app, [names["choice"]])
        combined = next(
            function
            for function in app.fns.values()
            if getattr(function.fn, "__name__", "").startswith("combined_")
        )
        values = []
        for component in combined.inputs:
            values.append("b" if component is names["choice"] else "extra")
        result = dict(
            zip(
                [component._id for component in combined.outputs],
                combined.fn("REQUEST", *values),
            )
        )
        self.assertEqual(calls, ["one", "two", "pair", "keyed", "request", "dict"])
        self.assertEqual(result[names["first"]._id], "one:b")
        self.assertEqual(result[names["second"]._id]["value"], "two:b:extra")
        self.assertFalse(result[names["second"]._id]["visible"])
        self.assertEqual(result[names["pair_a"]._id], "B")
        self.assertEqual(result[names["pair_b"]._id], gr.skip())
        self.assertEqual(result[names["keyed"]._id], "keyed:b")
        self.assertEqual(result[names["locale"]._id], "b:REQUEST")
        self.assertEqual(result[names["third"]._id], "dict:b:extra")

    def test_a_listener_that_needs_an_earlier_result_starts_the_next_batch(self):
        with gr.Blocks() as app:
            source, x, y, z, w = (gr.Textbox() for _ in range(5))
            source.change(lambda value: value, source, x)
            source.change(lambda value: value, x, y)
            source.change(lambda value: value, source, z)
            source.change(lambda value: value, y, w)
            source.change(lambda value: value, source, w)
        key = (source._id, "change")
        before = [function._id for function in app.fns.values() if key in function.targets]
        # [x] | [x->y, z] | [y->w] | [w again]: only the middle pair shares a request.
        self.assertEqual(event_coalescing.coalesce(app, [source]), {source._id: [2]})
        after = [function for function in app.fns.values() if key in function.targets]
        self.assertEqual(
            [function._id for function in after],
            [before[0], after[1]._id, before[3], before[4]],
        )
        self.assertEqual(after[1].fn(None, "X", "S"), ["X", "S"])

    def test_few_listeners_are_left_as_they_are(self):
        with gr.Blocks() as app:
            source, target = gr.Textbox(), gr.Textbox()
            source.change(lambda value: value, source, target)
        before = list(app.fns)
        self.assertEqual(event_coalescing.coalesce(app, [source]), {})
        self.assertEqual(list(app.fns), before)

    def test_a_wrong_number_of_results_names_the_listener(self):
        with gr.Blocks() as app:
            source, a, b, c, d = (gr.Textbox() for _ in range(5))

            def short(value):
                return (value,)

            source.change(short, source, [a, b])
            source.change(lambda value: value, source, c)
            source.change(lambda value: value, source, d)
        event_coalescing.coalesce(app, [source])
        combined = next(iter(app.fns.values()))
        with self.assertRaisesRegex(ValueError, "short returned 1 values for 2 outputs"):
            combined.fn(None, "x", "x", "x")


if __name__ == "__main__":
    unittest.main()
