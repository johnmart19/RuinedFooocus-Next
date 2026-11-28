"""Exercise workflow tools across real app imports with exact native CPU fixtures."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.validate_feature_cherry_picks import cpu_test_code


ROOT = Path(__file__).resolve().parents[1]
CHILD = "RF_WORKFLOW_TOOL_ENTRYPOINT_CHILD"


class WorkflowToolEntrypointTests(unittest.TestCase):
    def test_actual_application_workflow_boundaries(self):
        if os.environ.get(CHILD) == "1":
            self.check_application_workflow_boundaries()
            return
        # initialize changes argv/cwd/environment, and native imports retain
        # module state. Give this actual entrypoint its own guarded process.
        with tempfile.TemporaryDirectory(prefix="rf-workflow-entrypoint-") as directory:
            environment = dict(os.environ, RF_DATA_DIR=directory, RF_OFFLINE="1")
            environment[CHILD] = "1"
            result = subprocess.run(
                [sys.executable, "-c", cpu_test_code([Path(__file__).name])],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def check_application_workflow_boundaries(self):
        from tools import comfy_feature_app as adapter
        from tools import validate_comfy_workflows as workflow
        from ruinedfooocus.backends.comfy import comfy_compat

        # Only native initialization is substituted. The CLI must import and
        # parse the real app arguments before handing that object to native code.
        with patch.object(comfy_compat, "initialize_backend", autospec=True) as initialize:
            workflow.initialize(["--cpu", "--settings", "workflow-entrypoint"])
        from ruinedfooocus.app.arguments import args
        import ruinedfooocus.app.state as shared

        initialize.assert_called_once_with(args)
        self.assertTrue(args.cpu)
        self.assertTrue(args.offline)
        self.assertEqual(args.settings, "workflow-entrypoint")
        self.assertEqual(args._device_environment_status["requested"], {})

        workspace = Path(os.environ["RF_DATA_DIR"])
        checkpoint = workspace / "fixture-models" / "owned-fixture.safetensors"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"not model weights; catalog-path fixture only")
        original_directories = shared.models.model_dirs["checkpoints"]
        with self.assertRaisesRegex(RuntimeError, "fixture callback failed"):
            with adapter._catalog({"checkpoints": checkpoint}):
                self.assertEqual(shared.models.get_file("checkpoints", checkpoint.name), checkpoint)
                raise RuntimeError("fixture callback failed")
        self.assertIs(shared.models.model_dirs["checkpoints"], original_directories)

        # The feature validator exports these complete immutable Git snapshots.
        # Never hide an incomplete application closure behind synthetic modules.
        # Ordinary CPU runs use the pinned checkouts under repositories/.
        native = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
        gguf = Path(
            os.environ.get("RF_TEST_GGUF_SOURCE", ROOT / "repositories/molbal_comfyui_gguf")
        )
        self.assertTrue((native / "comfy" / "cli_args.py").is_file())
        self.assertTrue((gguf / "__init__.py").is_file())
        sys.path.insert(0, str(native))
        sys.argv = [sys.argv[0], "--cpu"]
        from comfy.cli_args import args as native_args
        from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management

        native_args.cpu = True
        initialize_cpu_model_management()
        spec = importlib.util.spec_from_file_location(
            "molbal_comfyui_gguf",
            gguf / "__init__.py",
            submodule_search_locations=[str(gguf)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        import ruinedfooocus.features.generation.image_pipeline as application
        import torch
        from PIL import Image

        latent = torch.tensor([[[[0.5]]]], device="cpu")
        pixels = torch.tensor([[[[1.0, 0.0, 0.0]]]], device="cpu")
        positive, negative = [["positive", {}]], [["negative", {}]]
        owner = SimpleNamespace(decode_latent=lambda samples: pixels)

        def process(*, gen_data, callback):
            samples = application.comfy.sample.sample(
                positive=gen_data["positive"], negative=gen_data["negative"]
            )
            owner.decode_latent({"samples": samples})

        owner.process = process
        original_decode = owner.decode_latent
        output = workspace / "artifacts" / "observed.png"
        # The real application module has imported successfully before any
        # sampling operation is mocked; only tiny fixed CPU artifacts follow.
        with patch.object(application.comfy.sample, "sample", return_value=latent) as sample:
            observed = adapter._run(owner, {"positive": positive, "negative": negative}, output)
            sample.assert_called_once()
            self.assertIs(application.comfy.sample.sample, sample)
        self.assertIs(owner.decode_latent, original_decode)
        self.assertIs(observed["latent"], latent)
        self.assertIs(observed["output"], pixels)
        self.assertEqual(observed["conditioning"], {"positive": positive, "negative": negative})
        with Image.open(output) as image:
            self.assertEqual(image.size, (1, 1))
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
