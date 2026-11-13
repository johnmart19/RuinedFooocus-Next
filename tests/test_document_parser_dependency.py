"""The PDF requirement joins the normal package plan without optional extras."""

import json
from pathlib import Path
import unittest

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from ruinedfooocus.packaging.package_plan import read_layer


class DocumentParserDependencyTests(unittest.TestCase):
    def test_declared_parser_and_wheel_contract_match_supported_python(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads(
            (root / "ruinedfooocus/resources/document_parsers.json").read_text(encoding="utf-8")
        )["pdf"]
        layer = read_layer(
            root / "pip/modules.txt",
            identity="rf-overlay",
            source="RuinedFooocus/pip/modules.txt",
            revision="sha256-content",
            license="GPL-3.0",
        )
        declared = [Requirement(row) for row in layer.requirements if row.startswith("pypdf")]
        self.assertEqual(len(declared), 1)
        self.assertEqual(str(declared[0].specifier), "==" + contract["version"])
        self.assertEqual(declared[0].extras, set())
        for python in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(python, SpecifierSet(contract["requires_python"]))
        self.assertTrue(contract["wheel"].endswith("-py3-none-any.whl"))
        self.assertRegex(contract["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(contract["url"].endswith("/" + contract["wheel"]))
        self.assertEqual(contract["extras"], [])


if __name__ == "__main__":
    unittest.main()
