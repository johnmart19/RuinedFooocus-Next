"""Extra-aware installed requirements checks without running pip."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.packaging.launch_util import requirements_met


class InstalledRequirementTests(unittest.TestCase):
    def check(self, text, versions, dependencies):
        def version(name):
            if name not in versions:
                raise PackageNotFoundError(name)
            return versions[name]

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text(text)
            with (
                patch(
                    "ruinedfooocus.packaging.launch_util.importlib.metadata.version",
                    side_effect=version,
                ),
                patch(
                    "ruinedfooocus.packaging.launch_util.importlib.metadata.requires",
                    side_effect=lambda name: dependencies.get(name, []),
                ),
            ):
                return requirements_met(path)

    def test_missing_or_incompatible_requested_extra_requires_repair(self):
        dependencies = {"ui": ['protocol>=1,<2; extra == "mcp"']}
        self.assertFalse(self.check("ui[mcp]==1", {"ui": "1"}, dependencies))
        self.assertFalse(self.check("ui[mcp]==1", {"ui": "1", "protocol": "2"}, dependencies))
        self.assertTrue(self.check("ui[mcp]==1", {"ui": "1", "protocol": "1.5"}, dependencies))

    def test_unrequested_extra_is_not_required(self):
        self.assertTrue(self.check("ui==1", {"ui": "1"}, {"ui": ['optional; extra == "mcp"']}))

    def test_transitive_extras_cycles_and_repeated_stricter_requirements(self):
        dependencies = {
            "ui": ["protocol[stream]>=1"],
            "protocol": ["ui", 'socket; extra == "stream"'],
        }
        versions = {"ui": "1", "protocol": "1.5", "socket": "1"}
        self.assertTrue(self.check("ui==1", versions, dependencies))
        self.assertFalse(self.check("ui==1\nprotocol>=2", versions, dependencies))
        self.assertFalse(
            self.check("ui==1", {k: v for k, v in versions.items() if k != "socket"}, dependencies)
        )

    def test_inactive_platform_markers_and_local_versions_remain_supported(self):
        self.assertTrue(
            self.check(
                'gpu==2.13\nabsent; python_version < "2"',
                {"gpu": "2.13+rocm10.1"},
                {"gpu": ['absent; python_version < "2"']},
            )
        )


if __name__ == "__main__":
    unittest.main()
