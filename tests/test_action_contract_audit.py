"""Source receipts preserve lexical callbacks and do not count ordinary method calls as UI."""

from pathlib import Path
import tempfile
import unittest

from tools.audit_action_contracts import ROOT, inventory


class ActionContractAuditTests(unittest.TestCase):
    def fixture(self, source):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / "ruinedfooocus/ui/app.py"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return inventory(root)

    def test_ordinary_executor_session_platform_and_load_calls_are_excluded(self):
        document = self.fixture(
            "import gradio as gr\n"
            "pool.submit(work)\nsession.select('model')\nplatform.release()\n"
            "model.load('weight')\nvalues.clear()\n"
            "button.click(fn=run, inputs=prompt, outputs=result)\n"
        )
        self.assertEqual(len(document["actions"]), 1)
        self.assertEqual(document["actions"][0]["binding"], "button.click")
        self.assertEqual(len(document["excluded_method_calls"]), 5)

    def test_lexical_callback_ownership_does_not_resolve_another_nested_render(self):
        document = self.fixture(
            "import gradio as gr\n"
            "def first():\n"
            "    def render(value):\n        return value + 'first'\n"
            "    a.change(render, inputs=a, outputs=b)\n"
            "def second():\n"
            "    def render(value):\n        return value + 'second'\n"
            "    a.change(render, inputs=a, outputs=b)\n"
        )
        first, second = document["actions"]
        self.assertEqual(first["callback_qualified"], "first.render")
        self.assertEqual(second["callback_qualified"], "second.render")
        self.assertNotEqual(first["callback_line"], second["callback_line"])
        self.assertNotEqual(first["callback_sha256"], second["callback_sha256"])

    def test_chained_events_get_distinct_spans_and_loop_helper_expansion_stays_explicit(self):
        document = self.fixture(
            "import gradio as gr\n"
            "for control in controls:\n"
            "    control.click(fn=ack, inputs=a, outputs=b).then(fn=work, inputs=b, outputs=c)\n"
            "localized_output(render, [a], [b])\n"
        )
        rows = document["actions"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["id"] for row in rows}), 3)
        self.assertEqual(
            [row["enclosing_loops"] for row in rows[:2]], [["control in controls"]] * 2
        )
        self.assertEqual(rows[-1]["kind"], "localized-output-helper")
        self.assertEqual((rows[-1]["inputs"], rows[-1]["outputs"]), ("[a]", "[b]"))

    def test_decorated_http_route_has_its_path_and_callback_source(self):
        document = self.fixture(
            "@api.post('/images/generations')\ndef generate(request):\n    return submit(request)\n"
        )
        row = document["actions"][0]
        self.assertEqual(row["kind"], "route")
        self.assertEqual(row["route"], "'/images/generations'")
        self.assertEqual((row["callback"], row["callback_line"]), ("generate", 2))

    def test_shared_registration_options_are_resolved_in_their_lexical_scope(self):
        document = self.fixture(
            "import gradio as gr\n"
            "def create():\n"
            "    options = dict(inputs=[source], outputs=[status], queue=False)\n"
            "    tab.select(show, **options)\n"
        )
        self.assertEqual(len(document["actions"]), 1)
        row = document["actions"][0]
        self.assertEqual(
            (row["inputs"], row["outputs"], row["queue"]), ("[source]", "[status]", "False")
        )
        self.assertEqual(row["option_spreads"], ["options"])

    def test_every_current_source_site_is_grouped_and_source_bound_without_claiming_runtime_count(
        self,
    ):
        document = inventory(ROOT)
        self.assertGreater(len(document["actions"]), 250)
        self.assertFalse(document["unassigned"])
        self.assertEqual(len(document["actions"]), len({row["id"] for row in document["actions"]}))
        for row in document["actions"]:
            self.assertEqual(len(document["source_sha256"][row["owner"]]), 64)
            self.assertNotEqual(row["resource_owner"], "UNREVIEWED")
            self.assertNotEqual(row["persistence"], "UNREVIEWED")
        self.assertIn("never runtime actions", document["scope"])

    def test_new_action_source_without_a_reviewed_group_stays_unassigned(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / "ruinedfooocus/new_feature/ui.py"
        path.parent.mkdir(parents=True)
        path.write_text("import gradio as gr\nbutton.click(run, outputs=result)\n")
        document = inventory(root)
        self.assertEqual(len(document["unassigned"]), 1)
        self.assertEqual(document["actions"][0]["resource_owner"], "UNREVIEWED")


if __name__ == "__main__":
    unittest.main()
