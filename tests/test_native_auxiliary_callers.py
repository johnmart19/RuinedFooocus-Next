"""Native guidance reaches real API, preparation, sampler and replay callers."""

import ast
from inspect import unwrap
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import gradio  # Import before fixtures snapshot sys.modules, including its Pydantic dependencies.
from pydantic import ValidationError
import torch

from ruinedfooocus.features.generation import image_preparation as prep
from ruinedfooocus.features.generation.defaults import STARTUP_DEFAULTS, generation_defaults
from ruinedfooocus.features.image_editing import gligen_layout as gligen
from ruinedfooocus.features.image_editing import native_auxiliary_assets as assets
from ruinedfooocus.features.image_editing import native_hypernetwork as hyper
from ruinedfooocus.features.image_editing import lora_hooks
from ruinedfooocus.integrations.openai_api import ImageRequest
from ruinedfooocus.core.preparation import PreparationBusy
import test_native_auxiliary_guidance as guidance_fixture
from test_model_cache import method
from test_native_model_coverage import PIPELINE
from test_qwen_image21_conditioning import native_class
import test_api_native_inpaint as api_fixture
import test_image_metadata_boundary as metadata_fixture
import test_library_editing_workflow as library_fixture
import test_model_folders as folders_fixture

ROOT = Path(__file__).resolve().parents[1]


class AuxiliaryCallerTests(unittest.TestCase):
    def fixture(self, cls):
        fixture = cls(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_removed_local_guidance_never_enters_a_download_preparer(self):
        files = self.fixture(guidance_fixture.AuxiliaryTests)
        captured = files.capture()
        for role, path in (("gligen", files.gligen_file), ("hypernetwork", files.hyper_file)):
            path.unlink()
            data = dict(captured)
            data[prep.KEY] = prep.ImageAssets(
                tuple(item for item in captured[prep.KEY].artifacts if item.role == role),
                offline=False,
            )
            service = NS(
                submit=Mock(side_effect=AssertionError("local guidance attempted a download"))
            )
            with self.assertRaises(FileNotFoundError) as error:
                prep.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
            self.assertEqual(
                str(error.exception),
                f"Captured image {role} was removed after submission: {path.name}. Choose it again.",
            )
            service.submit.assert_not_called()

    def test_explicit_json_null_clears_saved_guidance_before_capture(self):
        files = self.fixture(guidance_fixture.AuxiliaryTests)
        api = self.fixture(api_fixture.NativeInpaintApiTests)
        sys.modules["ruinedfooocus.app.state"].path_manager.get_model_folders = lambda category: [
            files.root
        ]
        captured = files.capture()
        for prefix in ("gligen", "hypernetwork"):
            with self.subTest(prefix=prefix):
                saved = {"base_model_name": api.selected["name"]}
                saved.update(
                    (key, value) for key, value in captured.items() if key.startswith(prefix + "_")
                )
                with patch.object(api_fixture.api_runtime, "image_defaults", saved):
                    response = api.client.post(
                        "/images/generations",
                        json={
                            "prompt": "owned literal",
                            "size": "512x512",
                            prefix + "_model": None,
                        },
                    )
                self.assertEqual(response.status_code, 200, response.text)
                data = api.prepared[-1]
                self.assertIsNone(data.get(prefix + "_model"))
                self.assertIsNone(data.get(prefix + "_model_signature"))
                self.assertNotIn(gligen.KEY if prefix == "gligen" else hyper.KEY, data)
                self.assertNotIn(prefix, {item.role for item in data[prep.KEY].artifacts})

    def test_configured_roots_register_in_order_without_scanning_or_creating_directories(self):
        from ruinedfooocus.backends.comfy.model_paths import register_image_model_paths
        from ruinedfooocus.core.settings_validation import validate_settings

        folders = self.fixture(folders_fixture.ModelFolderTests)
        paths = folders.paths
        for role in ("gligen", "hypernetworks"):
            first, second = folders.root / (role + "-one"), folders.root / (role + "-two")
            values = validate_settings({"path_" + role: f"{first}\n{second}"})
            paths.paths.update(values)
            self.assertEqual(paths.get_model_folders(role), [first, second])
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
        register = Mock()
        with patch.dict(sys.modules, {"folder_paths": NS(add_model_folder_path=register)}):
            register_image_model_paths(paths)
        for role in ("gligen", "hypernetworks"):
            calls = [call for call in register.call_args_list if call.args[0] == role]
            self.assertEqual(
                [Path(call.args[1]) for call in calls],
                list(reversed(paths.get_model_folders(role))),
            )
            self.assertTrue(all(call.kwargs["is_default"] for call in calls))
        folders.paths.download_file.assert_not_called()

    def test_actual_asgi_generation_captures_guidance_and_rejects_incompatible_requests_cold(self):
        files = self.fixture(guidance_fixture.AuxiliaryTests)
        api = self.fixture(api_fixture.NativeInpaintApiTests)
        state = sys.modules["ruinedfooocus.app.state"]
        state.path_manager.get_model_folders = lambda category: [files.root]
        payload = dict(
            prompt="a literal still-life",
            size="512x512",
            gligen_model=files.gligen_file.name,
            gligen_layout=json.dumps(files.boxes),
            hypernetwork_model=files.hyper_file.name,
            hypernetwork_strength=-0.25,
        )
        with patch.object(
            assets, "safe_native_state", side_effect=AssertionError("cold admission loaded weights")
        ):
            response = api.client.post("/images/generations", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = api.prepared[-1]
        self.assertIsInstance(data[gligen.KEY], gligen.GligenPlan)
        self.assertIsInstance(data[hyper.KEY], hyper.HypernetworkPlan)
        self.assertEqual(
            [box.text for box in data[gligen.KEY].boxes], [box["text"] for box in files.boxes]
        )
        self.assertEqual(
            {item.role for item in data[prep.KEY].artifacts} & {"gligen", "hypernetwork"},
            {"gligen", "hypernetwork"},
        )
        for change in (
            dict(size="510x512"),
            dict(input_image=api_fixture.inline(api.source)),
            dict(automatic_upscale="2× General (RealESRGAN)"),
            dict(controlnet=dict(type="upscale"), input_image=api_fixture.inline(api.source)),
        ):
            with self.subTest(change=change):
                before = api.submit.call_count
                response = api.client.post("/images/generations", json=payload | change)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(api.submit.call_count, before)
        proof = data["hypernetwork_model_signature"]
        second = files.root / "another.pt"
        second.write_bytes(files.hyper_file.read_bytes())
        saved = dict(
            base_model_name=api.selected["name"],
            hypernetwork_model=files.hyper_file.name,
            hypernetwork_model_signature=proof,
        )
        with patch.object(api_fixture.api_runtime, "image_defaults", saved):
            response = api.client.post(
                "/images/generations", json=payload | dict(hypernetwork_model=second.name)
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(api.prepared[-1]["hypernetwork_model_signature"][0], second.name)
        files.hyper_file.write_bytes(b"changed exact source")
        response = api.client.post(
            "/images/generations", json=payload | dict(hypernetwork_model_signature=proof)
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("changed", response.text)
        response = api.client.post(
            "/images/generations", json=payload | dict(gligen_model="missing.safetensors")
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("configured folder", response.text)

    def test_api_schema_strict_bounded_public_fields_and_plain_source_replay(self):
        for fields in (
            dict(gligen_layout="x" * 32769),
            dict(hypernetwork_strength=True),
            dict(hypernetwork_strength=float("nan")),
            dict(hypernetwork_strength=10.1),
            dict(gligen_model_signature=["a.pt", 10, 123, 456]),
        ):
            with self.subTest(fields=list(fields)), self.assertRaises(ValidationError):
                ImageRequest(**fields)
        files = self.fixture(guidance_fixture.AuxiliaryTests)
        data = files.capture()
        for field in gligen.FIELDS + hyper.FIELDS:
            self.assertIn(field, ImageRequest.model_fields)
        replayed = assets.replay_values(data)
        restored = prep.image_control_data(
            dict(prompt=json.dumps(dict(prompt="literal overall", **replayed)))
        )
        for key in gligen.FIELDS + hyper.FIELDS:
            self.assertEqual(restored[key], data[key])
        proof = replayed["gligen_model_signature"]
        proof[1] += 1
        self.assertNotEqual(proof, data["gligen_model_signature"])
        plain = prep.image_control_data(data | dict(prompt='{"prompt":"plain image"}'))
        self.assertIsNone(plain["gligen_model"])
        self.assertIsNone(plain["hypernetwork_model"])
        for feature in (gligen, hyper):
            with self.assertRaisesRegex(ValueError, "original file identity"):
                assets.replay_values({feature.FIELDS[0]: "missing.pt"})

    def test_saved_controls_library_metadata_and_fresh_library_edits_own_guidance(self):
        files = self.fixture(guidance_fixture.AuxiliaryTests)
        data = files.capture()
        saved = generation_defaults(data)
        self.assertEqual(saved["hypernetwork_strength"], -0.5)
        self.assertIsNone(STARTUP_DEFAULTS["gligen_model"])
        self.assertIsNone(STARTUP_DEFAULTS["hypernetwork_model"])
        metadata = dict(
            prompt="literal scene",
            negative="literal negative",
            **{key: data[key] for key in gligen.FIELDS + hyper.FIELDS},
        )
        updates = metadata_fixture.gallery(metadata)
        for key in gligen.FIELDS + hyper.FIELDS:
            self.assertEqual(updates[key]["value"], data[key])
        plain_updates = metadata_fixture.gallery(dict(prompt="plain image"))
        self.assertIsNone(plain_updates["gligen_model"]["value"])
        self.assertIsNone(plain_updates["hypernetwork_model"]["value"])
        library = self.fixture(library_fixture.LibraryRequestTests)
        library.controls.update({key: data[key] for key in gligen.FIELDS + hyper.FIELDS})
        request = library.build()
        self.assertIsNone(request["gligen_model"])
        self.assertIsNone(request["hypernetwork_model"])
        self.assertEqual(library.controls["gligen_model"], files.gligen_file.name)
        # Metadata is public data; no patcher, tensor or absolute file path survives.
        encoded = json.dumps(metadata)
        self.assertNotIn(str(files.root), encoded)
        self.assertNotIn(gligen.KEY, encoded)

    def test_full_production_process_passes_native_boxes_and_request_clone_to_sampler(self):
        from ruinedfooocus.features.image_editing.image_guidance import resolve_guidance
        from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

        files = self.fixture(guidance_fixture.AuxiliaryTests)
        data = files.capture()
        data.update(
            positive_prompt="overall literal scene",
            negative_prompt="literal negative",
            steps=2,
            cfg=4,
            sampler_name="euler",
            scheduler="simple",
            seed=42,
            silent=True,
            show_preview=False,
        )
        owner, native, clip = files.runtime()
        base = owner.xl_base_patched.unet
        base.model = type("BaseModel", (), {"model_config": NS(unet_config={})})()
        owner.xl_base, owner.model_info, owner.known_models = NS(unet=base), {}, {"BaseModel"}
        owner.conditions = {"+": {}, "-": {}}

        def encode(sign, text, *args, **kwargs):
            owner.conditions[sign]["cache"] = [[text, {}]]

        owner.textencode = encode
        owner.xl_base_patched.vae.spacial_compression_decode = lambda: 8
        owner.decode_latent = Mock(return_value=torch.ones(1, 48, 64, 3) * 0.5)
        native.model_management.get_torch_device = lambda: "cpu"
        native.model_management.intermediate_device = lambda: "cpu"
        native.model_management.intermediate_dtype = lambda: torch.float32
        native.utils.ProgressBar = lambda steps: NS(update_absolute=Mock())
        native.utils.PROGRESS_BAR_ENABLED = False

        def sample(**values):
            self.assertIn("gligen", values["positive"][0][1])
            self.assertNotIn("gligen", values["negative"][0][1])
            self.assertIsNot(values["model"], base)
            self.assertEqual(values["positive"][0][0], "overall literal scene")
            self.assertEqual(values["negative"][0][0], "literal negative")
            for path in (files.gligen_file, files.hyper_file):
                with (
                    self.assertRaises(PreparationBusy),
                    prep.preparation_service.artifact_lease(path, write=True, blocking=False),
                ):
                    self.fail("sampling lost its captured guidance source lease")
            return values["latent_image"].clone()

        sampler = Mock(side_effect=sample)
        native.sample = NS(
            sample=sampler, prepare_noise=lambda latent, *args: torch.zeros_like(latent)
        )
        empty = native_class(
            "nodes.py",
            "EmptyLatentImage",
            dict(torch=torch, comfy=native),
            methods=("__init__", "generate"),
        )
        process = method(
            PIPELINE,
            "process",
            dict(
                torch=torch,
                np=np,
                nullcontext=nullcontext,
                random=NS(),
                settings=NS(default_settings={}),
                control_tools=NS(
                    get_settings=lambda data: data.get("controlnet") or {}, controlnet_models={}
                ),
                resolve_guidance=resolve_guidance,
                normalize_reference_images=lambda *a: [],
                request_model=lambda model, *a, **kw: model,
                effective_guidance=lambda *a: {},
                pp=NS(prompt_switch_per_step=lambda *a: []),
                comfy=native,
                EmptyLatentImage=empty,
                SDXL=type("SDXL", (), {}),
                Flux=type("Flux", (), {}),
                fix_empty_latent_channels=lambda model, latent, *a: latent,
                restore_reference_frame=restore_reference_frame,
                reference_tensor=Mock(side_effect=AssertionError("unused reference")),
                path_manager=files.manager,
                shared=NS(),
                lora_hooks=lora_hooks,
            ),
        )
        worker = NS(check_interrupt=Mock())
        unwrap(process).__globals__["worker"] = worker
        with patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": worker}):
            output = process(owner, data)
        sampler.assert_called_once()
        self.assertEqual(output[0].shape, (48, 64, 3))
        self.assertEqual(
            [row["text"] for row in data["gligen_guidance"]["boxes"]],
            [box["text"] for box in files.boxes],
        )
        self.assertEqual(data["hypernetwork_guidance"]["strength"], -0.5)
        self.assertNotIn("patches", base.model_options["transformer_options"])
        self.assertNotIn("gligen", owner.conditions["+"]["cache"][0][1])
        worker.check_interrupt.assert_called()
        # Execute the real worker's metadata conditions, excluding unrelated file output.
        tree = ast.parse((ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
        clauses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            in (
                "gen_data.get('_native_gligen_layout') is not None",
                "gen_data.get('_native_hypernetwork') is not None",
            )
        ]
        scope = dict(gen_data=data, prompt={}, deepcopy=deepcopy)
        exec(
            compile(
                ast.Module(body=clauses, type_ignores=[]),
                "actual auxiliary worker metadata",
                "exec",
            ),
            scope,
        )
        self.assertEqual(scope["prompt"]["gligen_layout"], data["gligen_layout"])
        self.assertEqual(scope["prompt"]["hypernetwork_strength"], -0.5)
        json.dumps(scope["prompt"])
        # A subsequent Off request releases old auxiliary caches and uses the base.
        plain = dict(data)
        for key in (gligen.KEY, hyper.KEY, "gligen_model", "hypernetwork_model"):
            plain.pop(key, None)
        sampler.side_effect = lambda **values: values["latent_image"].clone()
        native.sd.load_gligen.reset_mock()
        native.utils.load_torch_file.reset_mock()
        process(owner, plain)
        self.assertIs(sampler.call_args.kwargs["model"], base)
        self.assertNotIn("gligen", sampler.call_args.kwargs["positive"][0][1])
        self.assertIsNone(owner._native_gligen)
        self.assertIsNone(owner._native_hypernetwork)
        native.sd.load_gligen.assert_not_called()
        native.utils.load_torch_file.assert_not_called()
        native.model_management.unload_model_and_clones.assert_called_once()

    def test_unsupported_combinations_and_standalone_hidden_controls_fail_or_clear(self):
        files = self.fixture(guidance_fixture.AuxiliaryTests)
        for feature, field, overrides in (
            (
                gligen,
                "gligen_model",
                (
                    dict(sdxl_refiner_model="refiner.safetensors"),
                    dict(native_export_conditioning=True),
                    dict(native_positive_input="base64"),
                ),
            ),
            (
                hyper,
                "hypernetwork_model",
                (
                    dict(hypernetwork_strength=float("nan")),
                    dict(image_engine="sdcpp-vulkan"),
                    dict(controlnet={"type": "upscale"}),
                ),
            ),
        ):
            for override in overrides:
                with self.subTest(feature=field, override=override), self.assertRaises(ValueError):
                    feature.capture_assets(
                        files.data | override, files.manager, files.models, "BaseModel"
                    )
            with self.assertRaises(ValueError):
                feature.capture_assets(files.data, files.manager, files.models, "QwenImage")
        scheduled, _ = gligen.capture_assets(
            files.data
            | dict(
                lora_hook_mode="keyframes",
                lora_hook_model="hooks.safetensors",
                lora_hook_clip_strength=1,
                lora_hook_schedule_clip=True,
            ),
            files.manager,
            files.models,
            "BaseModel",
        )
        self.assertIsInstance(scheduled[gligen.KEY], gligen.GligenPlan)
        unchanged, _ = gligen.capture_assets(
            files.data
            | dict(lora_hook_mode="off", lora_hook_clip_strength=1, lora_hook_schedule_clip=True),
            files.manager,
            files.models,
            "BaseModel",
        )
        self.assertIsInstance(unchanged[gligen.KEY], gligen.GligenPlan)
        data = files.capture()
        for kind in ("rembg", "upscale", "detect", "segment", "depth-map"):
            clean = prep.omit_generation_conditioning(data | dict(controlnet={"type": kind}))
            for key in gligen.FIELDS + hyper.FIELDS + (gligen.KEY, hyper.KEY):
                self.assertNotIn(key, clean)


if __name__ == "__main__":
    unittest.main()
