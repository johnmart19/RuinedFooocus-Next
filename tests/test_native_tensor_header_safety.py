"""Malformed saved contexts fail as input errors before native readers or receipts."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import torch

from ruinedfooocus.features.generation import native_tensor_io as native
from ruinedfooocus.features.generation import native_tensor_context as context
from ruinedfooocus.integrations.openai_api import create_apps
import test_native_tensor_context as fixtures
from test_native_tensor_io import mutate_header


class NativeHeaderSafetyTests(unittest.TestCase):
    def setUp(self):
        self.harness = fixtures.SavedNativeWorkflowTests("runTest")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        data = self.harness.ordinary(native_export_conditioning=True)
        condition = [[torch.zeros(1, 2, 768), {}]]
        artifact = native.export_tensors(
            self.harness.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, condition, condition
        )[0]
        self.raw = Path(artifact.path).read_bytes()

    def changed(self, update):
        def mutate(header):
            saved = json.loads(header["__metadata__"][context.MARKER])
            update(saved)
            header["__metadata__"][context.MARKER] = json.dumps(saved)

        return mutate_header(self.raw, mutate)

    def test_nonstring_conditioning_names_return_http400_before_receipt_or_native_read(self):
        backend = NS(
            resolve_model=Mock(return_value=("image/owned", {"family": "SD 1.5"})),
            images=lambda payload, selected: self.harness.ordinary(
                native_positive_input=base64.b64decode(payload["native_positive_input"])
            ),
        )
        app, _ = create_apps(backend)
        with (
            TestClient(app, raise_server_exceptions=False) as client,
            patch.object(native, "_scoped", side_effect=AssertionError("native reader reached")),
            patch.object(context, "verify_receipt", side_effect=AssertionError("receipt reached")),
        ):
            for value in (1, None, True, {}, []):
                raw = self.changed(
                    lambda saved, value=value: saved["entries"][0].update(names=[value])
                )
                response = client.post(
                    "/images/generations",
                    json={"native_positive_input": base64.b64encode(raw).decode()},
                )
                self.assertEqual(response.status_code, 400, response.text)

    def test_malformed_name_option_and_family_structures_raise_valueerror(self):
        cases = (
            lambda saved: saved["entries"][0].update(names=[]),
            lambda saved: saved["entries"][0].update(names=["conditioning", "conditioning"]),
            lambda saved: saved["entries"][0].update(names=["x" * 129]),
            lambda saved: saved["entries"][0].update(names=["conditioning"] * 257),
            lambda saved: saved["entries"][0].update(options=[]),
            lambda saved: saved["entries"].__setitem__(0, None),
            lambda saved: saved.update(version=2.0),
        )
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                context.header(self.changed(update), "conditioning", "SD15")
        for family in (None, 1, True, [], {}):
            with self.subTest(family=family), self.assertRaises(ValueError):
                context.header(self.raw, "conditioning", family)

    def test_saved_replay_family_and_role_types_are_bounded_before_input_access(self):
        for family, role in (
            ([], "latent"),
            ({}, "latent"),
            ("SD15", []),
            ("SD15", {}),
            ("SD15", True),
        ):
            metadata = {"native_tensor_source": {"family": family, "inputs": [{"role": role}]}}
            with (
                self.subTest(family=family, role=role),
                patch.object(native, "_raw", side_effect=AssertionError("input accessed")),
                self.assertRaises(ValueError),
            ):
                native.replay_values(metadata, {})


if __name__ == "__main__":
    unittest.main()
