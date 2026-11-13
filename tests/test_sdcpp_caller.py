"""Application caller/receipt checks with tiny files and inert native boundaries."""

import ast
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ruinedfooocus.backends.sdcpp import activation, artifacts, contracts, runtime
from ruinedfooocus.features.generation import sdcpp_pipeline as caller
from ruinedfooocus.features.models.model_identity import file_signature


class SdcppCallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(activation, "root_path", return_value=self.root / "runtime-cache").start()
        patch.object(activation, "_ACTIVE", {}).start()
        self.model_path = self.root / "renamed-local-model.safetensors"
        header = {
            prefix + "fixture": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            for prefix in (
                "model.diffusion_model.",
                "conditioner.embedders.0.",
                "conditioner.embedders.1.",
                "first_stage_model.",
            )
        }
        encoded = json.dumps(header).encode()
        self.model_path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"00")
        self.hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.metadata = {
            "baseModel": "SDXL 1.0",
            "_rf_local_sha256": self.hash,
            "_rf_file_signature": file_signature(self.model_path),
        }
        self.models = SimpleNamespace(
            get_file=Mock(return_value=self.model_path),
            get_models_by_path=Mock(side_effect=lambda *a, **k: dict(self.metadata)),
            get_model_base=lambda info: info["baseModel"],
        )
        self.values = {"image_engine": "auto", "sdcpp_vulkan_device": "Vulkan0"}
        self.data = {
            "task_type": "process",
            "base_model_name": self.model_path.name,
            "prompt": "a blue cat and a red dog",
            "negative": "blur",
            "custom_steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "clip_skip": 1,
            "seed": 42,
            "width": 512,
            "height": 768,
            "loras": [],
            "style_selection": [],
            "controlnet": {},
            "image_component_overrides": {"clip_l": None, "clip_g": None, "vae_sdxl": None},
        }
        self.item = artifacts.ARTIFACTS[1]
        directory = activation.runtime_directory(self.item)
        directory.mkdir(parents=True)
        executable = directory / "sd-cli.exe"
        executable.write_bytes(b"inert test executable, never run")
        files = artifacts._payload(directory)
        (directory / "rf-runtime.json").write_text(json.dumps({"files": files}))
        self.candidate = contracts.RuntimeCandidate(
            executable,
            "vulkan",
            "Vulkan0",
            artifact_sha256=self.item.sha256,
            payload_signatures=(
                tuple(file_signature(executable)),
                tuple(file_signature(directory / "rf-runtime.json")),
            ),
            device_verified=True,
            device_environment=runtime._device_environment(os.environ),
            device_description="fixture physical GPU",
        )
        self.identity = activation.DeviceIdentity(
            "fixture-pci-location", "1.2.3", "fixture-driver-package", "native", True
        )
        self.active = activation.Activation(self.candidate, self.identity)
        activation._ACTIVE[(str(directory), "Vulkan0")] = self.active
        self.model = contracts.with_model_hash(
            contracts.inspect_model(self.model_path, architecture="SDXL 1.0", variant="base"),
            self.hash,
        )
        self.native = Mock(return_value=SimpleNamespace(pipeline_type=["sdxl"]))

    def capture(self, data=None, values=None, **kwargs):
        return caller.capture_image_request(
            self.data if data is None else data,
            self.models,
            self.values if values is None else values,
            launch_args=SimpleNamespace(cpu=False),
            native_gpu_compatible=False,
            **kwargs,
        )

    def accepted(self, operation="txt2img"):
        activation.record_acceptance(
            self.active, self.model, operation, evidence="fixture actual-evidence reference"
        )

    def prepare(self, data=None):
        return caller.prepare_image_request(self.capture(data), check_cancelled=lambda: None)

    def test_capture_copies_nested_request_and_freezes_profile_choices(self):
        data = dict(
            self.data,
            input_image=Image.new("RGB", (512, 768)),
            controlnet={"type": "img2img", "denoise": 0.3},
        )
        result = self.capture(data)
        data["controlnet"]["denoise"] = 0.9
        data["input_image"].putpixel((0, 0), (255, 0, 0))
        self.values["image_engine"] = "native"
        snapshot = result[caller.SNAPSHOT_KEY]
        self.assertEqual(snapshot.engine, "auto")
        self.assertEqual(snapshot.controls["denoise"], 0.3)
        self.assertEqual(snapshot.data["input_image"].getpixel((0, 0)), (0, 0, 0))
        self.models.get_models_by_path.assert_called_once_with(
            "checkpoints", self.model_path, fetch=False
        )

    def test_capture_never_trusts_name_to_classify_standard_base(self):
        self.model_path.rename(self.root / "sd_xl_base_1.0.safetensors")
        self.model_path = self.root / "sd_xl_base_1.0.safetensors"
        self.models.get_file.return_value = self.model_path
        self.metadata["_rf_file_signature"] = file_signature(self.model_path)
        self.data["base_model_name"] = self.model_path.name
        prepared = self.prepare()
        self.assertEqual(prepared[caller.PREPARED_KEY].model.variant, "unknown")
        selected = caller.select_pipeline(prepared, None, self.native)
        self.assertEqual(selected.pipeline_type, ["sdxl"])
        self.assertIn("standard SDXL", prepared["image_engine_route"])

    def test_chat_retargets_reference_without_rereading_captured_engine_or_device(self):
        self.accepted("img2img")
        prepared = self.prepare()
        original = prepared[caller.SNAPSHOT_KEY]
        selected = Image.new("RGB", (512, 768), "blue")
        self.values.update(image_engine="native", sdcpp_vulkan_device="Vulkan9")
        data = dict(prepared, input_image=selected, controlnet={"type": "img2img", "denoise": 0.4})
        with patch.object(activation, "status", side_effect=AssertionError("no activation work")):
            retargeted = caller.retarget_image_request(data)
        snapshot = retargeted[caller.SNAPSHOT_KEY]
        self.assertEqual((snapshot.engine, snapshot.device), (original.engine, original.device))
        self.assertEqual(snapshot.controls, {"type": "img2img", "denoise": 0.4})
        self.assertEqual(retargeted[caller.PREPARED_KEY].input_geometry, (512, 768, "RGB", 1))
        selected.putpixel((0, 0), (255, 0, 0))
        self.assertEqual(snapshot.data["input_image"].getpixel((0, 0)), (0, 0, 255))

    def test_signature_mismatch_does_not_reuse_cached_hash(self):
        self.metadata["_rf_file_signature"][-2] -= 1
        captured = self.capture(values={"image_engine": "sdcpp-vulkan"})
        self.assertEqual(captured[caller.SNAPSHOT_KEY].sha256, "")
        with self.assertRaisesRegex(ValueError, "verified local identity"):
            caller.prepare_image_request(captured, check_cancelled=lambda: None)

    def test_legacy_and_global_cpu_authority_prevents_vulkan(self):
        for policy in ({"image_device": "CPU"}, {"cpu_everywhere": True}, {"image_backend": "cpu"}):
            captured = self.capture(values={**policy, "image_engine": "auto"})
            self.assertEqual(captured[caller.SNAPSHOT_KEY].engine, "native")
            self.assertFalse(caller.needs_preparation(captured))
            with self.assertRaisesRegex(ValueError, "CPU selection"):
                self.capture(values={**policy, "image_engine": "sdcpp-vulkan"})
        automatic = SimpleNamespace(
            cpu=True, _compute_status={"backend": {"cpu_origin": "automatic"}}
        )
        self.assertFalse(caller.cpu_requested({}, automatic))
        cli = SimpleNamespace(cpu=True, _compute_status={"cli": ["--cpu"]})
        self.assertTrue(caller.cpu_requested({}, cli))
        self.assertFalse(
            caller.cpu_requested(
                {"image_device": "CPU", "image_backend": "rocm"}, SimpleNamespace(cpu=False)
            )
        )

    def test_native_gpu_auto_bypasses_optional_activation_and_all_source_gates(self):
        captured = caller.capture_image_request(
            self.data, self.models, self.values, native_gpu_compatible=True
        )
        with patch.object(activation, "status", side_effect=AssertionError("no optional status")):
            self.assertFalse(caller.needs_preparation(captured))
            self.assertIs(
                caller.select_pipeline(captured, None, self.native), self.native.return_value
            )

    def test_known_receipt_routes_normal_job_to_optional_pipeline(self):
        self.accepted()
        data = self.prepare()
        with patch.object(
            runtime, "start_process", side_effect=AssertionError("no native process")
        ):
            selected = caller.select_pipeline(data, None, self.native)
            selected.load_base_model(
                self.model_path.name, hash=self.hash.upper(), component_overrides={"clip_l": None}
            )
            selected.load_loras([])
        self.assertIn("sdcpp", selected.pipeline_type)
        self.native.assert_not_called()

    def test_app_default_clip_skip_maps_to_native_auto_without_changing_metadata(self):
        self.accepted()
        for app_skip in (1, 1.0, None):
            source = dict(self.data)
            if app_skip is None:
                source.pop("clip_skip", None)
            else:
                source["clip_skip"] = app_skip
            data = self.prepare(source)
            request = caller.make_request(data[caller.PREPARED_KEY], data)
            self.assertEqual(request.clip_skip, -1)
            self.assertIn("sdcpp", caller.select_pipeline(data, None, self.native).pipeline_type)
            self.assertEqual(data.get("clip_skip"), app_skip)

    def test_exact_target_restoration_routes_auto_native_and_rejects_explicit_optional_engine(self):
        from ruinedfooocus.features.generation.size_plan import KEY, SizePlan

        self.accepted()
        for plan in (SizePlan.create((511, 767)), SizePlan.create((1024, 1536), 2)):
            data = self.prepare(dict(self.data, **{KEY: plan}))
            request = caller.make_request(data[caller.PREPARED_KEY], data)
            self.assertIn("exact target needs native canvas restoration", request.extra_features)
            with self.assertLogs(level="WARNING"):
                self.assertIs(
                    caller.select_pipeline(data, None, self.native), self.native.return_value
                )
            snapshot = replace(data[caller.SNAPSHOT_KEY], engine="sdcpp-vulkan")
            data[caller.SNAPSHOT_KEY] = snapshot
            data[caller.PREPARED_KEY] = replace(data[caller.PREPARED_KEY], snapshot=snapshot)
            with self.assertRaises(contracts.UnsupportedRequest):
                caller.select_pipeline(data, None, self.native)

    def test_invalid_app_clip_skip_cannot_reuse_the_native_default_sentinel(self):
        self.accepted()
        for value in (-2, -1, 0, None, True, "1", 1.5):
            with self.subTest(value=value):
                data = self.prepare(dict(self.data, clip_skip=value))
                request = caller.make_request(data[caller.PREPARED_KEY], data)
                self.assertIn(
                    "CLIP skip must be a positive integral app value", request.extra_features
                )
                self.assertNotIn(
                    "sdcpp", caller.select_pipeline(data, None, self.native).pipeline_type
                )
                snapshot = replace(data[caller.SNAPSHOT_KEY], engine="sdcpp-vulkan")
                data[caller.SNAPSHOT_KEY] = snapshot
                data[caller.PREPARED_KEY] = replace(data[caller.PREPARED_KEY], snapshot=snapshot)
                with self.assertRaises(contracts.UnsupportedRequest):
                    caller.select_pipeline(data, None, self.native)

    def test_captured_clip_override_is_retained_when_final_data_omits_the_field(self):
        data = self.prepare(dict(self.data, clip_skip=2))
        data.pop("clip_skip")
        request = caller.make_request(data[caller.PREPARED_KEY], data)
        self.assertEqual(request.clip_skip, 2)
        self.assertIn("Explicit CLIP skip is not qualified", contracts.assess(request))

    def test_unqualified_vulkan_falls_back_to_native_with_a_warning(self):
        # Known base provenance alone is not a workload acceptance receipt.
        with patch.object(activation, "known_variant", return_value="base"):
            data = self.prepare()
        with self.assertLogs(level="WARNING") as logs:
            self.assertIs(caller.select_pipeline(data, None, self.native), self.native.return_value)
        self.assertIn("Using CPU fallback", logs.output[0])
        self.assertIn("workload acceptance", logs.output[0])
        self.accepted()
        unknown = activation.Activation(self.candidate, activation.DeviceIdentity())
        with patch.object(activation, "current", return_value=unknown):
            data = self.prepare()
        with self.assertLogs(level="WARNING"):
            self.assertIs(caller.select_pipeline(data, None, self.native), self.native.return_value)

    def test_unsupported_details_keep_auto_native_and_explicit_choice_refuses(self):
        self.accepted()
        for change in (
            {"loras": [["hash", "1.0 - requested.safetensors"]]},
            {"sampler_name": "dpmpp_2m_sde"},
            {"scheduler": "karras"},
            {"inpaint_toggle": True},
            {"controlnet": {"type": "depth"}},
            {"automatic_upscale": "2x"},
            {"clip_skip": 2},
            {"cfg_norm_strength": 1.0},
            {"vae_tiled": True},
            {"sdxl_refiner_model": "refiner.safetensors"},
            {"image_component_overrides": {"vae_sdxl": "replacement.safetensors"}},
        ):
            with self.subTest(change=change):
                data = self.prepare({**self.data, **change})
                self.assertNotIn(
                    "sdcpp", caller.select_pipeline(data, None, self.native).pipeline_type
                )
                snapshot = replace(data[caller.SNAPSHOT_KEY], engine="sdcpp-vulkan")
                data[caller.SNAPSHOT_KEY] = snapshot
                data[caller.PREPARED_KEY] = replace(data[caller.PREPARED_KEY], snapshot=snapshot)
                with self.assertRaises(contracts.UnsupportedRequest):
                    caller.select_pipeline(data, None, self.native)

    def test_final_metadata_features_are_not_dropped_after_capture(self):
        self.accepted()
        data = self.prepare()
        data["inpaint_toggle"] = True
        request = caller.make_request(data[caller.PREPARED_KEY], data)
        self.assertIn("inpaint_toggle", request.extra_features)

    def test_img2img_loopback_and_mismatched_input_stay_on_native_auto(self):
        self.accepted("img2img")
        for controls, size in (
            ({"type": "img2img", "denoise": 0.3, "loopback": True}, (512, 768)),
            ({"type": "img2img", "denoise": 0.3}, (768, 512)),
        ):
            data = self.prepare(
                dict(self.data, controlnet=controls, input_image=Image.new("RGB", size))
            )
            self.assertNotIn("sdcpp", caller.select_pipeline(data, None, self.native).pipeline_type)

    def test_explicit_offline_model_verification_populates_signature_bound_identity(self):
        self.metadata.pop("_rf_local_sha256")
        self.assertFalse(self.capture()[caller.SNAPSHOT_KEY].sha256)
        with patch.object(
            runtime, "start_process", side_effect=AssertionError("no native process")
        ):
            model = activation.verify_model(
                self.model_path, architecture="SDXL 1.0", check_cancelled=lambda: None
            )
        self.assertEqual(model.sha256, self.hash)
        self.assertEqual(model.variant, "unknown")
        self.assertEqual(self.capture()[caller.SNAPSHOT_KEY].sha256, self.hash)
        self.model_path.write_bytes(self.model_path.read_bytes() + b"changed")
        self.assertFalse(activation.cached_model_hash(self.model_path))

    def test_model_change_between_submission_and_preparation_refuses(self):
        captured = self.capture()
        self.model_path.write_bytes(self.model_path.read_bytes() + b"replacement")
        with self.assertRaisesRegex(ValueError, "changed since submission"):
            caller.prepare_image_request(captured, check_cancelled=lambda: None)

    def test_generic_worker_call_requires_06_hooks_and_preserves_final_metadata(self):
        self.accepted()
        data = self.prepare()
        data["positive_prompt"], data["negative_prompt"] = "expanded blue cat", "expanded blur"
        data["task_id"] = 123
        selected = caller.select_pipeline(data, None, self.native)
        with patch.object(caller, "generate") as execute:
            with self.assertRaisesRegex(RuntimeError, "pending 06"):
                selected.process(data)
            execute.assert_not_called()
        hooks = caller.ExecutionHooks(Mock(), Mock(return_value=nullcontext()))
        data[caller.HOOKS_KEY] = hooks
        product = runtime.GenerationResult(
            Image.new("RGB", (512, 768)),
            {"engine": "stable-diffusion.cpp", "request_id": "123"},
            "fixture",
        )
        with patch.object(caller, "generate", return_value=product) as execute:
            images = selected.process(data)
        request = execute.call_args.args[0]
        self.assertEqual(request.prompt, "expanded blue cat")
        self.assertEqual(request.request_id, "123")
        self.assertEqual(images[0].shape, (768, 512, 3))
        self.assertEqual(data["_sdcpp_metadata"], product.metadata)
        hooks.compute_lease_factory.assert_called_once_with(request, selected.candidate)

    def test_img2img_caller_preserves_input_geometry_and_strength(self):
        self.accepted("img2img")
        image = Image.new("RGB", (512, 768), "red")
        data = self.prepare(
            dict(self.data, input_image=image, controlnet={"type": "img2img", "denoise": 0.35})
        )
        data[caller.HOOKS_KEY] = caller.ExecutionHooks(lambda: None, lambda *args: nullcontext())
        selected = caller.select_pipeline(data, None, self.native)
        paths = []

        def execute(request, *args, **kwargs):
            paths.append(request.init_image)
            with Image.open(request.init_image) as prepared:
                self.assertEqual(prepared.size, (512, 768))
                self.assertEqual(prepared.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(request.denoise, 0.35)
            return runtime.GenerationResult(image.copy(), {"operation": "img2img"}, "")

        with patch.object(caller, "generate", side_effect=execute):
            selected.process(data)
        self.assertFalse(paths[0].exists())
        self.assertEqual(image.size, (512, 768))

    def test_status_prepare_and_selection_do_not_install_or_enumerate(self):
        self.accepted()
        with (
            patch.object(activation, "install", side_effect=AssertionError("no install")),
            patch.object(runtime, "verify_device", side_effect=AssertionError("no probe")),
            patch.object(runtime, "start_process", side_effect=AssertionError("no process")),
        ):
            self.assertTrue(activation.status("vulkan")["prepared"])
            data = self.prepare()
            self.assertIn("sdcpp", caller.select_pipeline(data, None, self.native).pipeline_type)

    def test_receipt_matches_driver_payload_model_and_operation(self):
        self.accepted()
        self.assertTrue(activation.qualified(self.active, self.model, "txt2img").qualified)
        for changed in (
            replace(self.active, identity=replace(self.identity, driver_version="changed")),
            replace(
                self.active, candidate=replace(self.candidate, device_description="different GPU")
            ),
        ):
            self.assertFalse(activation.qualified(changed, self.model, "txt2img").qualified)
        self.assertFalse(activation.qualified(self.active, self.model, "img2img").qualified)
        self.assertFalse(
            activation.qualified(
                self.active, replace(self.model, sha256="b" * 64), "txt2img"
            ).qualified
        )

    def test_unknown_identity_cannot_record_automatic_qualification(self):
        with self.assertRaisesRegex(ValueError, "driver identity"):
            activation.record_acceptance(
                replace(self.active, identity=activation.DeviceIdentity()),
                self.model,
                "txt2img",
                evidence="fixture",
            )

    def test_fresh_process_does_not_restore_index_from_receipt(self):
        self.accepted()
        with patch.object(activation, "_ACTIVE", {}):
            self.assertIsNone(activation.current("vulkan", "Vulkan0"))
            with self.assertRaisesRegex(ValueError, "explicitly verify"):
                self.prepare()

    def test_offline_missing_bundle_refuses_acquisition(self):
        with (
            patch.object(activation, "runtime_directory", return_value=self.root / "missing"),
            patch.object(artifacts, "prepare_runtime") as prepare,
        ):
            with self.assertRaisesRegex(RuntimeError, "offline/skip-install"):
                activation.install("vulkan", check_cancelled=lambda: None, offline=True)
            prepare.assert_not_called()

    def test_nonimage_jobs_are_not_captured_or_prepared(self):
        for data in (
            dict(self.data, task_type="llama"),
            dict(self.data, prompt="search: image"),
            dict(self.data, prompt="#!help"),
            dict(self.data, controlnet={"type": "rembg"}),
        ):
            self.assertNotIn(caller.SNAPSHOT_KEY, self.capture(data))
        self.metadata["baseModel"] = "Wan Video"
        self.assertNotIn(caller.SNAPSHOT_KEY, self.capture())

    def test_registry_selects_optional_only_on_nonvideo_image_branch(self):
        source = (
            Path(__file__).resolve().parents[1] / "ruinedfooocus/features/generation/pipelines.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "update"
        )
        state = {"pipeline": None}
        shared = SimpleNamespace(
            models=self.models, settings=SimpleNamespace(default_settings=self.values)
        )
        video = SimpleNamespace(
            pipeline=Mock(return_value=SimpleNamespace(pipeline_type=["wan_video"]))
        )
        scope = {
            "state": state,
            "shared": shared,
            "Path": Path,
            "VIDEO_PIPELINES": {"Wan Video": "wan_video"},
            "controlnet": SimpleNamespace(get_settings=lambda data: {}),
            "selected_pipeline": lambda: state["pipeline"],
            "NoPipeLine": lambda: SimpleNamespace(pipeline_type=[]),
            "image_pipeline": SimpleNamespace(pipeline=self.native),
            "wan_video_pipeline": video,
            "hunyuan_video_pipeline": video,
            "ltx_video_pipeline": video,
            "video_pipeline": video,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        self.accepted()
        self.assertIn("sdcpp", scope["update"](self.prepare()).pipeline_type)
        self.metadata["baseModel"] = "Wan Video"
        self.assertEqual(scope["update"](dict(self.data)).pipeline_type, ["wan_video"])


if __name__ == "__main__":
    unittest.main()
