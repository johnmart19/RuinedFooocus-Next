"""Actual owned serializers, worker admission, API media and tensor-only UI result."""

import ast
import base64
from dataclasses import replace
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr
import numpy as np
import safetensors.torch
import torch
from PIL import Image

import ruinedfooocus.app
from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.generation import native_tensor_context as context
from ruinedfooocus.features.generation import native_tensor_io as tensors
from ruinedfooocus.features.generation.image_output_batches import validate_artifacts
from ruinedfooocus.features.image_processing import latent_composition as composition
from ruinedfooocus.integrations.openai_api import create_apps
import test_native_latent_composition_io as fixtures

ROOT = Path(__file__).resolve().parents[1]


def function(file, name, scope):
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, "exec"), scope)
    return scope[name]


class LatentCompositionIntegration(unittest.TestCase):
    def setUp(self):
        fixtures.CompositionTransportTests.setUp(self)
        self.first, self.second = fixtures.CompositionTransportTests.sources(self)
        self.models.get_file.reset_mock()
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                side_effect=AssertionError("checkpoint inspection reached"),
            )
        )

    def capture(self, **values):
        return composition.capture(
            dict(
                latent_composition_primary=self.first,
                latent_composition_secondary=self.second,
                **values,
            ),
            self.manager,
        )

    def payload(self, **values):
        return dict(
            controlnet={"type": composition.KIND},
            model="missing/image",
            latent_composition_primary=base64.b64encode(self.first).decode(),
            latent_composition_secondary=base64.b64encode(self.second).decode(),
            **values,
        )

    def test_full_consumer_returns_one_real_saved_latent_with_replayable_native_context(self):
        data = self.capture(latent_composition_x=8)
        result = composition.process(data, lambda: None)
        self.assertIs(validate_artifacts(data, result), result)
        self.assertEqual(result, [])
        artifact = result.tensor_files[0]
        raw = Path(artifact.path).read_bytes()
        header, saved = context.header(raw, "latent", "SD15")
        self.assertEqual(header["latent_tensor"]["dtype"], "F32")
        expected = torch.zeros((1, 4, 2, 3))
        expected[:, :, :, 1:] = 0.75
        np.testing.assert_array_equal(
            safetensors.torch.load(raw)["latent_tensor"].numpy(), expected.numpy()
        )
        history = saved["layout"]["composition"]
        self.assertEqual(history["original_dtype"], "F16")
        self.assertEqual(history["parameters"]["latent_composition_x"], 8)
        self.assertEqual(history["primary_sha256"], data[tensors.KEY].inputs[0].sha256)
        replay = composition.replay_values(saved, self.payload())
        again = composition.process(
            self.capture(**{key: replay[key] for key in composition.FIELDS}), lambda: None
        )
        self.assertEqual(again.tensor_files[0].sha256, artifact.sha256)
        self.assertEqual(data[tensors.KEY].inputs[0].path.read_bytes(), self.first)
        self.models.get_file.assert_not_called()

    def test_tensor_only_success_requires_completed_request_owned_file_and_no_image(self):
        data = self.capture()
        result = composition.process(data, lambda: None)
        for changed in (
            [],
            tensors.TensorGenerationArtifacts([], ()),
            tensors.TensorGenerationArtifacts(["fake.png"], result.tensor_files),
        ):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                validate_artifacts(data, changed)
        foreign = dict(data, **{tensors.KEY: replace(data[tensors.KEY], directory=self.root)})
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            validate_artifacts(foreign, result)
        Path(result.tensor_files[0].path).write_bytes(b"changed after export")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            validate_artifacts(data, result)

    def test_masked_consumer_serializes_exact_selection_and_refuses_a_different_replay_mask(self):
        mask = Image.new("L", (24, 16), 0)
        mask.paste(255, (0, 0, 8, 16))
        data = self.capture(latent_composition_mode="masked", latent_composition_mask=mask)
        result = composition.process(data, lambda: None)
        raw = Path(result.tensor_files[0].path).read_bytes()
        _, saved = context.header(raw, "latent", "SD15")
        expected = torch.zeros((1, 4, 2, 3))
        expected[:, :, :, 0] = 0.75
        np.testing.assert_array_equal(
            safetensors.torch.load(raw)["latent_tensor"].numpy(), expected.numpy()
        )
        recipe = composition.replay_values(saved, self.payload(latent_composition_mask=mask))
        self.assertEqual(recipe["latent_composition_mode"], "masked")
        with self.assertRaisesRegex(ValueError, "original selection mask"):
            composition.replay_values(
                saved, self.payload(latent_composition_mask=Image.new("L", (24, 16), 255))
            )

    def test_codec16_packed_consumer_keeps_all_sixty_four_channels_and_actual_pixel_offsets(self):
        self.owner.xl_base_patched.unet.model.model_config = type("Flux2", (), {})()
        self.owner.xl_base_patched.vae.latent_channels = 64
        self.owner.xl_base_patched.vae.spacial_compression_decode = lambda: 16
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "Flux2"},
        ):
            first, second = fixtures.CompositionTransportTests.sources(
                self, family="Flux2", channels=64, scale=16
            )
        data = composition.capture(
            dict(
                latent_composition_primary=first,
                latent_composition_secondary=second,
                latent_composition_x=16,
            ),
            self.manager,
        )
        result = composition.process(data, lambda: None)
        raw = Path(result.tensor_files[0].path).read_bytes()
        _, saved = context.header(raw, "latent", "Flux2")
        values = safetensors.torch.load(raw)["latent_tensor"]
        self.assertEqual(tuple(values.shape), (1, 64, 2, 3))
        self.assertEqual(torch.count_nonzero(values[:, :, :, :1]).item(), 0)
        self.assertTrue(torch.all(values[:, :, :, 1:] == 0.75).item())
        self.assertEqual(saved["layout"]["composition"]["native_x"], 8)
        self.assertEqual(saved["layout"]["composition"]["codec_scale"], 16)

    def test_actual_capture_and_execution_callers_bypass_checkpoint_defaults_and_prompt_processing(
        self,
    ):
        shared = NS(path_manager=self.manager, models=NS(offline=True), state={})
        controls = NS(
            get_settings=lambda data: data.get("controlnet") or {"type": composition.KIND}
        )
        scope = dict(shared=shared, control_tools=controls)
        capture = function("ruinedfooocus/app/async_worker.py", "capture_image_data", scope)
        data = capture(
            dict(
                controlnet={"type": composition.KIND},
                latent_composition_primary=self.first,
                latent_composition_secondary=self.second,
                prompt=object(),
                base_model_name="missing",
                loras=object(),
            )
        )
        self.assertNotIn("prompt", data)
        self.assertNotIn("base_model_name", data)
        execute = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            dict(check_interrupt=lambda data: None, control_tools=controls),
        )
        result = execute(data)
        self.assertIsInstance(result, tensors.TensorGenerationArtifacts)
        self.assertEqual(len(result.tensor_files), 1)
        self.models.get_file.assert_not_called()

    def test_actual_api_operation_prepares_and_executes_without_reading_global_generation_settings(
        self,
    ):
        shared = NS(path_manager=self.manager, models=NS(offline=True), state={})
        controls = NS(get_settings=lambda data: data["controlnet"])
        execute = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            dict(check_interrupt=lambda data: None, control_tools=controls),
        )
        prepare = function("ruinedfooocus/app/async_worker.py", "prepare_image_data", {})
        worker = NS(_process=execute, prepare_image_data=prepare, check_interrupt=lambda data: None)
        job = NS(cancelled=threading.Event())
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
        ):
            operation = api_runtime._image_operation(self.payload(), {})
            operation.prepare(job)
            result = operation(job, "owned-api-job")
        self.assertEqual(result, [])
        self.assertEqual(result.tensor_files[0].kind, "latent-composite")
        self.models.get_file.assert_not_called()

    def test_live_api_empty_data_native_tensor_and_owned_media_without_model_resolution(self):
        data = self.capture()
        result = composition.process(data, lambda: None)
        job = NS(events=queue.Queue(), cancelled=threading.Event())
        job.events.put(("result", result))
        backend = NS(
            models=Mock(side_effect=AssertionError("catalogue reached")),
            resolve_model=Mock(side_effect=AssertionError("checkpoint resolution reached")),
            images=Mock(return_value=job),
        )
        api, _ = create_apps(backend)
        app = FastAPI()
        app.mount("/v1", api)
        with TestClient(app) as client:
            response = client.post("/v1/images/generations", json=self.payload())
            self.assertEqual(response.status_code, 200, response.text)
            reply = response.json()
            self.assertEqual(reply["data"], [])
            self.assertEqual(len(reply["native_tensors"]), 1)
            artifact = reply["native_tensors"][0]
            self.assertEqual(artifact["sha256"], result.tensor_files[0].sha256)
            self.assertEqual(
                client.get(artifact["url"]).content, Path(result.tensor_files[0].path).read_bytes()
            )
        backend.models.assert_not_called()
        backend.resolve_model.assert_not_called()

    def test_ui_worker_preserves_tensor_completion_and_retains_existing_image_gallery(self):
        data = self.capture()
        result = composition.process(data, lambda: None)
        publish = Mock()
        ui_process = function(
            "ruinedfooocus/app/async_worker.py",
            "process",
            dict(
                control_tools=NS(get_settings=lambda value: value["controlnet"]),
                _process=Mock(return_value=result),
                add_result=publish,
            ),
        )
        ui_process(dict(data, task_id="ui-job"))
        self.assertIs(publish.call_args.args[2], result)
        update = function(
            "ruinedfooocus/ui/app.py",
            "update_results",
            dict(
                gr=gr,
                run_button="run",
                stop_button="stop",
                progress_html="progress",
                generation_originals="files",
            ),
        )
        updates = update(result)
        self.assertEqual(set(updates), {"run", "stop", "progress", "files"})
        self.assertEqual(updates["files"]["value"], [result.tensor_files[0].path])

    def test_api_refuses_a_different_canvas_and_other_source_operations_without_model_resolution(
        self,
    ):
        shared = NS(path_manager=self.manager, models=NS(offline=True), state={})
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            for changed in (
                dict(size="512x512"),
                dict(input_image="unrelated"),
                dict(native_latent_input="generation"),
                dict(automatic_upscale="4x"),
            ):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    api_runtime._image_operation(self.payload(**changed), {})
        self.models.get_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
