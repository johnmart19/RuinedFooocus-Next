"""Explicit catalogue metadata has no acquisition/device side effects."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.onnx import windows_ml as winml
from ruinedfooocus.backends.onnx.winml_catalog import catalogue_rows
from ruinedfooocus.packaging import hardware_inventory


class WindowsCatalogueUiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "provider.json"
        (self.root / "catalog-2.3.0").mkdir()
        for patcher in (
            patch.object(winml, "manifest_path", return_value=self.manifest),
            patch.object(winml.platform, "system", return_value="Windows"),
            patch.object(winml.platform, "machine", return_value="AMD64"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def snapshot(self):
        return {
            "catalog_version": "2.3.0",
            "providers": [
                {"name": "MIGraphXExecutionProvider", "certified": True},
                {"name": "AMDGPUExecutionProvider", "certified": True},
                {"name": "OpenVINOExecutionProvider", "certified": False},
            ],
        }

    def test_catalogue_projection_reads_only_publisher_metadata_not_ensure_ready(self):
        providers = [
            SimpleNamespace(
                name=name,
                certification=cert,
                ensure_ready_async=Mock(side_effect=AssertionError("acquisition")),
            )
            for name, cert in (
                ("MIGraphXExecutionProvider", "certified"),
                ("OpenVINOExecutionProvider", "uncertified"),
            )
        ]
        rows = catalogue_rows(providers, "certified")
        self.assertTrue(rows[0]["certified"])
        self.assertFalse(rows[1]["certified"])
        for provider in providers:
            provider.ensure_ready_async.assert_not_called()

    def test_explicit_refresh_uses_isolated_bounded_metadata_command_then_passive_cache(self):
        with patch(
            "ruinedfooocus.backends.onnx.acquisition.run",
            return_value=SimpleNamespace(stdout=json.dumps(self.snapshot())),
        ) as run:
            state = winml.refresh_catalog(check_cancelled=Mock())
        self.assertEqual(run.call_args.kwargs["timeout"], 45)
        self.assertIn("--list-providers", run.call_args.args[0])
        self.assertIn(str(self.root / "catalog-2.3.0"), run.call_args.args[0])
        self.assertFalse(
            next(row for row in state["providers"] if row["name"] == "AMDGPUExecutionProvider")[
                "bridge_compatible"
            ]
        )
        with (
            patch(
                "ruinedfooocus.backends.onnx.acquisition.run",
                side_effect=AssertionError("No process"),
            ),
            patch.object(winml, "install_provider", side_effect=AssertionError("No install")),
            patch.object(winml, "discover_devices", side_effect=AssertionError("No device probe")),
        ):
            self.assertEqual(winml.catalog_status(), state)

    def test_cpu_refresh_and_unknown_cache_do_not_launch_a_process_or_acquire(self):
        with patch(
            "ruinedfooocus.backends.onnx.acquisition.run", side_effect=AssertionError("No process")
        ):
            self.assertEqual(winml.catalog_status()["providers"], [])
            self.assertIn("CPU authority", winml.refresh_catalog(cpu=True)["reason"])
            (self.root / "catalogue.json").write_text("[]")
            self.assertEqual(winml.catalog_status()["providers"], [])

    def test_failed_or_malformed_refresh_preserves_previous_catalogue_bytes(self):
        document = self.snapshot()
        document.update(system="Windows", machine="AMD64", python=list(sys.version_info[:2]))
        path = self.root / "catalogue.json"
        before = json.dumps(document).encode()
        path.write_bytes(before)
        for result in (
            subprocess.TimeoutExpired("metadata", 45),
            SimpleNamespace(stdout='{"catalog_version":"2.3.0","providers":[{}]}'),
        ):
            with (
                self.subTest(result=result),
                patch(
                    "ruinedfooocus.backends.onnx.acquisition.run",
                    side_effect=result if isinstance(result, Exception) else None,
                    return_value=result,
                ),
            ):
                with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                    winml.refresh_catalog()
                self.assertEqual(path.read_bytes(), before)

    def test_hardware_status_uses_acquisition_records_without_workload_hashes_or_queries(self):
        with (
            patch.object(
                winml, "installed_providers", return_value=[{"name": "MIGraphXExecutionProvider"}]
            ),
            patch.object(
                winml, "installed_provider", side_effect=AssertionError("Qualification path")
            ),
            patch(
                "ruinedfooocus.backends.onnx.qualification.current_model",
                side_effect=AssertionError("Model hash/import"),
            ),
            patch(
                "ruinedfooocus.backends.onnx.qualification.driver_identity",
                side_effect=AssertionError("Driver query"),
            ),
            patch.object(winml, "discover_devices", side_effect=AssertionError("Device query")),
        ):
            self.assertEqual(hardware_inventory.windows_ml_provider(), "MIGraphXExecutionProvider")

    def test_passive_acquisition_status_skips_payload_hash_but_runtime_checks_it(self):
        library = self.root / "provider.dll"
        library.write_bytes(b"private fake DLL, never loaded")
        self.manifest.write_text(
            json.dumps(
                {
                    "name": "MIGraphXExecutionProvider",
                    "library_path": str(library),
                    "sha256": "0" * 64,
                }
            )
        )
        with patch.object(winml, "_verified_library", return_value=False) as verify:
            self.assertEqual(hardware_inventory.windows_ml_provider(), "MIGraphXExecutionProvider")
            verify.assert_not_called()
            self.assertEqual(winml.installed_providers(), [])
            verify.assert_called_once()
