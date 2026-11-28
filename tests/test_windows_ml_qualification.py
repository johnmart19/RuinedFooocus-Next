"""Qualification identity and inclusive discovery without GPU/runtime probing."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.onnx import qualification as q, windows_ml as winml


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        library = self.root / "provider.dll"
        library.write_bytes(b"fixture library")
        self.record = dict(
            name="MIGraphXExecutionProvider",
            library_path=str(library),
            sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            catalog_version="2.3.0",
            qualification="acquired_only_not_model_qualified",
        )
        self.manifest = self.root / "provider.json"
        for target, value in (("manifest_path", self.manifest),):
            context = patch.object(winml, target, return_value=value)
            context.start()
            self.addCleanup(context.stop)
        contexts = [
            patch.object(winml.platform, "system", return_value="Windows"),
            patch.object(
                q,
                "current_model",
                return_value=dict(
                    model="bria-rmbg", sha256=q.BRIA_SHA256, preprocessing="native-rembg-2.0.87"
                ),
            ),
            patch.object(
                q,
                "runtime_identity",
                return_value=dict(
                    ort="1.31.1",
                    binaries={"ort.dll": "r" * 64},
                    abi="cp312",
                    python="3.12.10",
                    system="Windows",
                ),
            ),
            patch.object(
                q,
                "driver_identity",
                return_value=dict(
                    build="26100", drivers=[dict(id="physical-pci", version="driver-1")]
                ),
            ),
        ]
        self.mocks = [context.start() for context in contexts]
        for context in contexts:
            self.addCleanup(context.stop)
        self.device = SimpleNamespace(
            ep_name=self.record["name"],
            device=SimpleNamespace(
                type="GPU", vendor_id=4098, device_id=1, metadata={"LUID": "stable"}
            ),
        )
        self.identity = winml.device_identity(self.device)
        self.contract = q.current_contract(self.record, self.identity)
        self.evidence = dict(
            schema=1,
            contract=self.contract,
            cache_identity=q.fingerprint(self.contract),
            qualified=True,
            real_model=True,
            original_bria=True,
            numerical_passed=True,
            visual_passed=True,
            stability_passed=True,
            hidden_fallback_disabled=True,
            assignment=dict(
                scope="first_inference_executed_nodes",
                nodes={self.record["name"]: 10, "CPUExecutionProvider": 1},
            ),
            cold_end_to_end_ms=1000,
            warm_end_to_end_ms=10,
            cpu_end_to_end_ms=100,
        )

    def publish(self, record=None):
        self.manifest.write_text(
            json.dumps({"providers": {self.record["name"]: record or self.record}})
        )

    def test_acquired_and_generic_qualified_flags_never_enable_auto_but_named_discovery_remains(
        self,
    ):
        for extra in (
            {},
            {"qualification": "qualified"},
            {"workload_evidence": {"qualified": True, "cache_identity": "arbitrary"}},
        ):
            self.publish(dict(self.record, **extra))
            self.assertIsNone(winml.installed_provider())
            self.assertIsNotNone(winml.installed_provider(self.record["name"]))
            self.assertEqual(len(winml.installed_providers()), 1)
            with patch.object(winml, "register_provider") as register:
                with self.assertRaisesRegex(RuntimeError, "no matching real-model qualification"):
                    winml.select_installed_device(Mock())
            register.assert_not_called()

    def test_real_evidence_publish_api_enables_exact_auto_and_binds_qualified_device(self):
        self.publish()
        winml.record_qualification(self.record["name"], self.contract, self.evidence)
        self.assertIsNotNone(winml.installed_provider())
        other = SimpleNamespace(
            ep_name=self.device.ep_name,
            device=SimpleNamespace(type="GPU", vendor_id=4098, device_id=2, metadata={}),
        )
        ort = SimpleNamespace(
            get_ep_devices=lambda: [other, self.device],
            OrtHardwareDeviceType=SimpleNamespace(GPU="GPU"),
        )
        with patch.object(winml, "register_provider", return_value=(None, Mock())):
            self.assertIs(winml.select_installed_device(ort)[1], self.device)
        ort.get_ep_devices = lambda: [other]
        with patch.object(winml, "register_provider", return_value=(None, Mock())):
            with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
                winml.select_installed_device(ort)

    def test_every_identity_boundary_rejects_stale_or_relabelled_qualification(self):
        original = dict(self.record, workload_evidence=self.evidence)
        self.publish(original)
        self.assertIsNotNone(winml.installed_provider())
        for key in (
            "model_sha256",
            "preprocessing",
            "shape",
            "dtype",
            "provider",
            "provider_library_sha256",
            "catalog_version",
            "runtime",
            "device",
            "device_type",
            "driver",
            "options",
        ):
            changed = deepcopy(self.evidence)
            changed["contract"][key] = "different"
            changed["cache_identity"] = q.fingerprint(changed["contract"])
            self.publish(dict(self.record, workload_evidence=changed))
            self.assertIsNone(winml.installed_provider(), key)
        self.publish(original)
        self.mocks[2].return_value = dict(ort="1.32", binaries={"ort.dll": "different"})
        self.assertIsNone(winml.installed_provider())
        self.mocks[2].return_value = self.contract["runtime"]
        self.mocks[3].return_value = dict(
            build="26100", drivers=[dict(id="physical-pci", version="driver-2")]
        )
        self.assertIsNone(winml.installed_provider())

    def test_partial_quality_slow_route_or_wrong_real_graph_is_rejected_by_recording_api(self):
        self.publish()
        for key in (
            "real_model",
            "original_bria",
            "numerical_passed",
            "visual_passed",
            "stability_passed",
            "hidden_fallback_disabled",
        ):
            with self.assertRaisesRegex(RuntimeError, "incomplete/stale"):
                winml.record_qualification(
                    self.record["name"], self.contract, dict(self.evidence, **{key: False})
                )
        for changes in (
            {"warm_end_to_end_ms": 100},
            {"assignment": {"scope": "canary", "nodes": {self.record["name"]: 10}}},
        ):
            with self.assertRaisesRegex(RuntimeError, "incomplete/stale"):
                winml.record_qualification(
                    self.record["name"], self.contract, dict(self.evidence, **changes)
                )
        self.mocks[1].return_value = dict(
            model="bria-rmbg", sha256="derived-graph", preprocessing="native-rembg-2.0.87"
        )
        self.assertIsNone(q.current_contract(self.record, self.identity))

    def test_replaced_provider_library_cannot_reuse_qualification_even_with_same_size_and_mtime(
        self,
    ):
        self.publish(dict(self.record, workload_evidence=self.evidence))
        self.assertIsNotNone(winml.installed_provider())
        library = Path(self.record["library_path"])
        stat = library.stat()
        replacement = library.with_name("replacement.dll")
        replacement.write_bytes(b"changed library")
        self.assertEqual(replacement.stat().st_size, stat.st_size)
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(replacement, library)
        self.assertIsNone(q.current_contract(self.record, self.identity))
        self.assertIsNone(winml.installed_provider())


if __name__ == "__main__":
    unittest.main()
