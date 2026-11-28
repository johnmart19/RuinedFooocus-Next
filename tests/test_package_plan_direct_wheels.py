"""Actual offline pip resolution preserves direct wheels and duplicate version pins."""

from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from ruinedfooocus.packaging.package_plan import PackageLayer, resolve_packages


class DirectWheelResolutionTests(unittest.TestCase):
    def test_duplicate_index_pin_and_constraint_reuse_the_explicit_wheel(self):
        with tempfile.TemporaryDirectory() as folder:
            wheel = Path(folder) / "rf_metadata_fixture-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                prefix = "rf_metadata_fixture-1.0.dist-info/"
                archive.writestr(
                    prefix + "METADATA",
                    "Metadata-Version: 2.1\nName: rf-metadata-fixture\nVersion: 1.0\n",
                )
                archive.writestr(
                    prefix + "WHEEL",
                    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                )
                archive.writestr(prefix + "RECORD", "")
            layers = (
                PackageLayer("indexed", ("rf-metadata-fixture==1.0",), "fixture", "1", "MIT", "a"),
                PackageLayer(
                    "prepared",
                    (f"rf-metadata-fixture @ {wheel.as_uri()}",),
                    "fixture",
                    "1",
                    "MIT",
                    "b",
                ),
            )
            observed = []

            def offline(command, **kwargs):
                # No package install, dependency fetching or index/network access.
                self.assertIn("--dry-run", command)
                self.assertIn("--ignore-installed", command)
                self.assertIn("--only-binary=:all:", command)
                command += ["--no-index", "--no-deps", "--disable-pip-version-check"]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    **kwargs,
                )
                observed.append(result)

            artifacts = resolve_packages(
                layers, constraints=("rf-metadata-fixture==1.0",), runner=offline
            )
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0].returncode, 0)
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].version, "1.0")
            self.assertEqual(artifacts[0].source, wheel.as_uri())
            self.assertEqual(len(artifacts[0].sha256), 64)


if __name__ == "__main__":
    unittest.main()
