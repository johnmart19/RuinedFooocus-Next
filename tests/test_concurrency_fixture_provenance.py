"""A running fixture keeps its loaded source identity after the checkout changes."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.concurrency_fixture import Fixture, build_app


class FixtureProvenanceTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-fixture-provenance-")
        self.addCleanup(temporary.cleanup)
        fixture = Fixture(Path(temporary.name) / "owned", count=1)
        self.addCleanup(fixture.close)
        return fixture

    def test_receipt_keeps_launch_identity_without_reading_a_changed_checkout(self):
        fixture = self.fixture()
        launch = fixture.receipt()
        identity = launch["source_sha256"].copy()
        launch["source_sha256"]["tools/concurrency_fixture.py"] = "caller mutation"
        with (
            patch(
                "tools.concurrency_fixture.subprocess.check_output",
                side_effect=AssertionError("Checkout HEAD changed"),
            ),
            patch(
                "tools.concurrency_fixture.Path.read_bytes",
                side_effect=AssertionError("Checkout files changed"),
            ),
        ):
            final = fixture.receipt()
        self.assertEqual(final["source_revision"], launch["source_revision"])
        self.assertEqual(final["source_sha256"], identity)

    def test_real_gradio_config_keeps_both_drafts_editable_and_outside_operation_outputs(self):
        app = build_app(self.fixture())
        config = app.get_config_file()
        drafts = {
            component["props"]["label"]: component
            for component in config["components"]
            if component["props"].get("label") in {"Create draft", "Chat draft"}
        }
        self.assertEqual(set(drafts), {"Create draft", "Chat draft"})
        output_ids = {
            identifier
            for dependency in config["dependencies"]
            for identifier in dependency["outputs"]
        }
        for component in drafts.values():
            self.assertIs(component["props"]["interactive"], True)
            self.assertNotIn(component["id"], output_ids)
        export = next(
            dependency
            for dependency in config["dependencies"]
            if dependency["api_name"] == "export_receipt"
        )
        self.assertFalse(export["queue"])

    def test_explicit_receipt_export_retains_frozen_identity(self):
        import json

        fixture = self.fixture()
        launch = fixture.receipt()
        exported = fixture.export_receipt()
        receipt = json.loads(Path(exported["receipt"]).read_text(encoding="utf-8"))
        self.assertEqual(receipt["source_revision"], launch["source_revision"])
        self.assertEqual(receipt["source_sha256"], launch["source_sha256"])


if __name__ == "__main__":
    unittest.main()
