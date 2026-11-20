"""The API extras and the base install name the same releases of shared packages.

A static check of the two requirement files. It does not resolve or install them.
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def pins(path):
    result = {}
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        requirement, _, marker = line.partition(";")
        name, _, version = requirement.partition("==")
        result[(name.strip().lower().replace("_", "-"), marker.strip())] = version.strip()
    return result


class ApiRequirementPinTests(unittest.TestCase):
    def test_api_pins_match_the_base_install_for_every_shared_package(self):
        base, api = pins("pip/modules.txt"), pins("requirements_api_versions.txt")
        self.assertTrue(api)
        base_names = {name for name, _ in base}
        for (name, marker), version in api.items():
            with self.subTest(name=name, marker=marker):
                # Each API package is installed with the interface already; a different
                # pin here would refuse offline starts or change a running dependency.
                self.assertIn(name, base_names)
                self.assertEqual(base.get((name, marker)), version)
        self.assertEqual({name for name, _ in api}, {"fastapi", "pydantic", "python-multipart"})


if __name__ == "__main__":
    unittest.main()
