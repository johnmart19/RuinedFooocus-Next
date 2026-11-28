"""Startup shortcuts: the remembered Torch validation and the single interface build.

Stand-ins replace the child-process probe. These checks do not time a launch.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
import gradio.blocks as blocks

from ruinedfooocus.packaging import runtime_support
from ruinedfooocus.ui import gradio_frontend


class RememberedTorchProbeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.file = self.root / "cache" / "torch-probe.json"
        self.identity = '["rocm", "python", "torch", "2.15.0"]'
        self.bundle = {"hip": "7.17", "versions": {"torch": "2.15.0", "torchvision": "0.29.2"}}
        self.calls = []
        self.result = self.bundle
        self.enterContext(patch.dict(runtime_support._PROBED, clear=True))
        self.enterContext(patch.object(runtime_support, "_probe_cache_file", lambda: self.file))
        self.enterContext(
            patch.object(runtime_support, "_bundle_identity", lambda backend: self.identity)
        )
        self.enterContext(patch.object(runtime_support, "_run_torch_probe", self.probe))

    def probe(self, backend):
        self.calls.append(backend)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def test_a_validated_bundle_is_reused_until_its_packages_change(self):
        self.assertEqual(runtime_support._probe_installed_torch("rocm"), self.bundle)
        self.assertEqual(runtime_support._probe_installed_torch("rocm"), self.bundle)
        self.assertEqual(self.calls, ["rocm"])
        self.assertEqual(json.loads(self.file.read_text()), {self.identity: self.bundle})
        # A new process reads the file instead of importing torch in a child.
        runtime_support._PROBED.clear()
        self.assertEqual(runtime_support._probe_installed_torch("rocm"), self.bundle)
        self.assertEqual(self.calls, ["rocm"])
        # Reinstalled packages have another identity and are validated again.
        self.identity = '["rocm", "python", "torch", "2.16.0"]'
        runtime_support._probe_installed_torch("rocm")
        self.assertEqual(self.calls, ["rocm", "rocm"])
        self.assertEqual(list(json.loads(self.file.read_text())), [self.identity])

    def test_failures_empty_results_and_cuda_are_never_remembered(self):
        self.result = RuntimeError("probe failed")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                runtime_support._probe_installed_torch("rocm")
        self.result = None
        self.assertIsNone(runtime_support._probe_installed_torch("rocm"))
        self.assertIsNone(runtime_support._probe_installed_torch("rocm"))
        self.assertEqual(len(self.calls), 4)
        self.assertFalse(self.file.exists())
        # The CUDA probe also exercises the driver, which changes without the packages.
        self.result = {"cuda": "12.8", "versions": {"torch": "2.15.0"}}
        runtime_support._probe_installed_torch("cuda")
        runtime_support._probe_installed_torch("cuda")
        self.assertEqual(self.calls[-2:], ["cuda", "cuda"])
        self.assertFalse(self.file.exists())

    def test_a_damaged_or_foreign_cache_file_is_ignored(self):
        self.file.parent.mkdir(parents=True)
        for content in ("not json", "[]", json.dumps({self.identity: {"hip": 7}})):
            runtime_support._PROBED.clear()
            self.file.write_text(content)
            self.assertEqual(runtime_support._probe_installed_torch("rocm"), self.bundle)
        self.assertEqual(self.calls, ["rocm"] * 3)


class BundleIdentityTests(unittest.TestCase):
    def test_identity_follows_the_installed_package_records(self):
        identity = runtime_support._bundle_identity("rocm")
        self.assertEqual(identity, runtime_support._bundle_identity("rocm"))
        self.assertNotEqual(identity, runtime_support._bundle_identity("cpu"))
        parts = json.loads(identity)
        self.assertEqual(parts[0], "rocm")
        self.assertIn("torch", parts)
        self.assertIn("torchvision", parts)


class SingleInterfaceBuildTests(unittest.TestCase):
    def tearDown(self):
        gradio_frontend.resume_interface_build()

    def test_layout_contexts_skip_the_rebuild_until_resumed(self):
        original_config, original_app = blocks.Blocks.get_config_file, blocks.App.create_app
        gradio_frontend.defer_interface_build()
        gradio_frontend.defer_interface_build()  # A second call keeps the first originals.
        with gr.Blocks(analytics_enabled=False) as page:
            box = gr.Textbox(label="Draft")
        with page:
            gr.Button("Send").click(lambda value: value, box, box)
        self.assertEqual(page.config, {})
        self.assertIsNone(page.app)
        gradio_frontend.resume_interface_build()
        gradio_frontend.resume_interface_build()
        self.assertIs(blocks.Blocks.get_config_file, original_config)
        self.assertIs(blocks.App.create_app, original_app)
        # What launch does next: the finished page yields its complete configuration.
        config = page.get_config_file()
        kinds = {component["type"] for component in config["components"]}
        self.assertTrue({"button", "textbox"} <= kinds, kinds)
        self.assertEqual(len(config["dependencies"]), 1)
        self.assertIsNotNone(blocks.App.create_app(page, mcp_server=False))


if __name__ == "__main__":
    unittest.main()
