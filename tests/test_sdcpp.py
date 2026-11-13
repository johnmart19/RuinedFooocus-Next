"""Optional diffusion contract tests with inert CLI processes and tiny local files."""

from concurrent.futures import CancelledError
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image, PngImagePlugin

from ruinedfooocus.backends.sdcpp import artifacts, contracts, runtime
from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.features.models.model_identity import file_signature


class SdcppTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model_path = self.root / "checkpoint.safetensors"
        self.header = {
            name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            for name in (
                "model.diffusion_model.fixture",
                "conditioner.embedders.0.fixture",
                "conditioner.embedders.1.fixture",
                "first_stage_model.fixture",
            )
        }
        self.write_header()
        self.model = contracts.with_model_hash(
            contracts.inspect_model(self.model_path, architecture="SDXL 1.0", variant="base"),
            "a" * 64,
        )
        executable = self.root / "sd-cli.exe"
        executable.write_bytes(b"inert fixture, never executed")
        self.candidate = contracts.RuntimeCandidate(
            executable,
            "vulkan",
            "Vulkan0",
            artifact_sha256=artifacts.ARTIFACTS[1].sha256,
            payload_signatures=(tuple(file_signature(executable)),),
            device_verified=True,
            qualified=True,
            qualification="fixture receipt",
            accepted_models=("a" * 64,),
            accepted_operations=("txt2img", "img2img"),
            device_environment=runtime._device_environment(os.environ),
        )
        self.request = contracts.GenerationRequest(
            self.model,
            'A blue cat, "fine fur"',
            "blur",
            512,
            768,
            1234,
            request_id="owned-request",
        )
        self.service = PreparationService(staging_root=self.root / "cache")
        self.addCleanup(self.service.close)

    def write_header(self):
        data = json.dumps(self.header).encode()
        self.model_path.write_bytes(len(data).to_bytes(8, "little") + data + b"00")

    def generate(self, request=None, **kwargs):
        return runtime.generate(
            request or self.request,
            self.candidate,
            compute_lease=nullcontext(),
            check_cancelled=lambda: None,
            service=self.service,
            **kwargs,
        )

    def fake_process(self, command, **options):
        self.command = command
        self.process_directory = Path(options["cwd"])
        self.assertEqual(options["env"]["GGML_BACKEND_PATH"], str(self.candidate.executable.parent))
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("parameters", "native fixture generation record")
            Image.new("RGB", (self.request.width, self.request.height), "blue").save(
                output,
                pnginfo=metadata,
            )
        options["stdout"].write(b"Vulkan0\tAMD fixture GPU\nUsage: fixture\n")
        process = Mock(returncode=0)
        process.poll.return_value = 0
        self.process, self.job = process, Mock()
        return process, self.job

    def test_source_matrix_and_hash_attachment_are_passive(self):
        with patch.object(runtime, "start_process", side_effect=AssertionError("no process")):
            self.assertEqual(contracts.assess(self.request, self.candidate), ())
            self.assertEqual(self.model.weight_types, ("F16",))
            self.assertTrue(self.model.bundled)
            self.assertFalse(contracts.assess(replace(self.request, scheduler="discrete")))
        with self.assertRaises(ValueError):
            contracts.with_model_hash(self.model, "short")

    def test_unknown_variant_is_not_assumed_to_be_standard_base(self):
        model = contracts.inspect_model(self.model_path, architecture="SDXL 1.0")
        self.assertIn(
            "standard SDXL", ";".join(contracts.assess(replace(self.request, model=model)))
        )

    def test_architecture_format_quantization_and_components_are_exact(self):
        for changes in (
            {"architecture": "Flux.1 D"},
            {"variant": "turbo"},
            {"format": "gguf"},
            {"quantization": "q8_0"},
            {"weight_types": ("BF16",)},
            {"weight_types": ("F8_E4M3",)},
            {"bundled": False},
            {"signature": ()},
        ):
            with self.subTest(changes=changes):
                self.assertTrue(
                    contracts.assess(replace(self.request, model=replace(self.model, **changes)))
                )

    def test_header_rejects_external_components_and_nonfloat_storage(self):
        self.header.pop("first_stage_model.fixture")
        self.header["model.diffusion_model.fixture"]["dtype"] = "I8"
        self.write_header()
        model = contracts.inspect_model(self.model_path, architecture="SDXL 1.0", variant="base")
        reasons = contracts.assess(replace(self.request, model=model))
        self.assertTrue(any("F16/F32" in reason for reason in reasons))
        self.assertTrue(any("bundled" in reason for reason in reasons))

    def test_every_unimplemented_conditioning_feature_is_refused(self):
        for key, value in (
            ("mask", self.root / "mask.png"),
            ("loras", (("lora", 1),)),
            ("embeddings", ("embedding",)),
            ("controlnets", ("control",)),
            ("component_overrides", (("vae_sdxl", "other"),)),
            ("refiner", "refiner"),
            ("extra_features", ("hires", "IP adapter", "preview")),
            ("sampler", "dpmpp_2m_sde"),
            ("scheduler", "karras"),
            ("clip_skip", 2),
            ("batch_size", 2),
            ("operation", "inpaint"),
        ):
            with self.subTest(feature=key):
                self.assertTrue(contracts.assess(replace(self.request, **{key: value})))

    def test_prompt_syntax_is_not_silently_reinterpreted(self):
        self.assertFalse(contracts.assess(replace(self.request, prompt="a black and white cat")))
        for text in (
            "(cat:1.2)",
            "[cat|dog]",
            "<lora:x:1>",
            "embedding:foo",
            "cat AND dog",
            "BREAK",
            "a\0b",
        ):
            for key in ("prompt", "negative_prompt"):
                self.assertTrue(contracts.assess(replace(self.request, **{key: text})))

    def test_numeric_and_geometry_validation_does_not_round_or_rescale(self):
        for changes in (
            {"width": 513},
            {"height": 0},
            {"width": 2**32},
            {"width": True},
            {"steps": 0},
            {"steps": 1.5},
            {"seed": -1},
            {"seed": 2**63},
            {"cfg": float("nan")},
            {"cfg": float("inf")},
            {"cfg": -1},
            {"cfg": True},
            {"denoise": 0},
            {"denoise": 1.1},
            {"denoise": float("nan")},
            {"denoise": 0.4},
            {"init_image": self.root / "init.png"},
            {"operation": "img2img"},
        ):
            with self.subTest(changes=changes):
                self.assertTrue(contracts.assess(replace(self.request, **changes)))

    def test_auto_keeps_native_and_otherwise_chooses_vulkan_before_cpu(self):
        cpu = replace(self.candidate, backend="cpu", device="cpu")
        self.assertIsNone(contracts.choose_runtime(self.request, (), native_gpu_compatible=True))
        self.assertIs(contracts.choose_runtime(self.request, (cpu, self.candidate)), self.candidate)
        self.assertIs(
            contracts.choose_runtime(self.request, (cpu, self.candidate), preference="cpu"), cpu
        )

    def test_pending_compatible_vulkan_never_silently_loses_to_cpu(self):
        cpu = replace(self.candidate, backend="cpu", device="cpu")
        pending = replace(self.candidate, qualified=False)
        with self.assertRaisesRegex(contracts.UnsupportedRequest, "qualify it or select CPU"):
            contracts.choose_runtime(self.request, (cpu, pending))
        self.assertIs(
            contracts.choose_runtime(self.request, (cpu, replace(pending, device_verified=False))),
            cpu,
        )

    def test_device_enumeration_does_not_grant_workload_acceptance(self):
        unqualified = replace(
            self.candidate, qualified=False, qualification="", device_verified=False
        )
        with patch.object(runtime, "start_process", side_effect=self.fake_process):
            verified = runtime.verify_device(unqualified, check_cancelled=lambda: None)
        self.assertTrue(verified.device_verified)
        self.assertFalse(verified.qualified)
        self.assertTrue(contracts.assess(self.request, verified))
        self.assertFalse(contracts.assess(self.request, verified, allow_unqualified=True))
        self.assertEqual(self.command[-1], "--list-devices")
        self.assertFalse(self.process_directory.exists())

    def test_cpu_verification_uses_cpu_bundle_and_never_enumerates_gpu(self):
        cpu = replace(
            self.candidate,
            backend="cpu",
            device="cpu",
            artifact_sha256=artifacts.ARTIFACTS[0].sha256,
        )
        with patch.object(runtime, "start_process", side_effect=self.fake_process):
            self.assertTrue(
                runtime.verify_device(cpu, check_cancelled=lambda: None).device_verified
            )
        self.assertEqual(self.command[-1], "--help")
        with self.assertRaisesRegex(runtime.RuntimeFailure, "differs"):
            runtime.verify_device(
                replace(cpu, artifact_sha256=self.candidate.artifact_sha256),
                check_cancelled=lambda: None,
            )

    def test_exact_device_absence_refuses_without_cpu_retry(self):
        with patch.object(runtime, "start_process", side_effect=self.fake_process) as start:
            with self.assertRaisesRegex(runtime.RuntimeFailure, "exact runtime device"):
                runtime.verify_device(
                    replace(self.candidate, device="Vulkan1"), check_cancelled=lambda: None
                )
        self.assertEqual(start.call_count, 1)

    def test_reverification_clears_old_acceptance_receipts(self):
        with patch.object(runtime, "start_process", side_effect=self.fake_process):
            verified = runtime.verify_device(self.candidate, check_cancelled=lambda: None)
        self.assertFalse(verified.qualified)
        self.assertEqual(verified.accepted_models, ())
        self.assertEqual(verified.accepted_operations, ())
        self.assertEqual(verified.device_description, "AMD fixture GPU")

    def test_file_leases_precede_compute_and_all_release_after_failure(self):
        events = []

        @contextmanager
        def lease(name, **kwargs):
            events.append("enter " + str(name))
            try:
                yield
            finally:
                events.append("leave " + str(name))

        service = Mock()
        service.artifact_lease.side_effect = lease
        with patch.object(runtime, "_run", side_effect=runtime.RuntimeFailure("fixture failure")):
            with self.assertRaisesRegex(runtime.RuntimeFailure, "fixture failure"):
                runtime.generate(
                    self.request,
                    self.candidate,
                    compute_lease=lease("compute"),
                    check_cancelled=lambda: None,
                    service=service,
                )
        enters = [event for event in events if event.startswith("enter")]
        self.assertEqual(enters[-1], "enter compute")
        self.assertEqual(events[len(enters)], "leave compute")
        self.assertEqual(len(events), 2 * len(enters))

    def test_command_preserves_prompts_geometry_seed_and_engine_metadata(self):
        with patch.object(runtime, "start_process", side_effect=self.fake_process):
            result = self.generate()
        for option, expected in (
            ("--prompt", self.request.prompt),
            ("--negative-prompt", "blur"),
            ("--width", "512"),
            ("--height", "768"),
            ("--seed", "1234"),
            ("--backend", "Vulkan0"),
            ("--strength", "1.0"),
            ("--params-backend", "Vulkan0"),
            ("--scheduler", "discrete"),
            ("--auto-fit", "off"),
        ):
            self.assertEqual(self.command[self.command.index(option) + 1], expected)
        self.assertEqual(result.image.size, (512, 768))
        self.assertEqual(result.image.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(result.metadata["prompt"], self.request.prompt)
        self.assertEqual(result.metadata["request_id"], "owned-request")
        self.assertEqual(
            result.metadata["native_metadata"]["parameters"], "native fixture generation record"
        )
        self.assertFalse(self.process_directory.exists())
        self.job.close.assert_called_once()

    def test_img2img_requires_exact_rgb_canvas_and_preserves_strength(self):
        path = self.root / "input.png"
        Image.new("RGB", (512, 768), "red").save(path)
        request = replace(self.request, operation="img2img", init_image=path, denoise=0.35)
        with patch.object(runtime, "start_process", side_effect=self.fake_process):
            result = self.generate(request)
        self.assertEqual(self.command[self.command.index("--strength") + 1], "0.35")
        self.assertEqual(
            self.command[self.command.index("--image-preprocess") + 1], "target=init,mode=none"
        )
        self.assertEqual(result.metadata["denoise"], 0.35)
        self.assertTrue(path.is_file())
        for mode, size in (("RGBA", (512, 768)), ("RGB", (768, 512))):
            Image.new(mode, size).save(path)
            with patch.object(runtime, "start_process") as start:
                with self.assertRaisesRegex(
                    contracts.UnsupportedRequest, "exact requested dimensions"
                ):
                    self.generate(request)
                start.assert_not_called()

    def test_changed_model_or_runtime_refuses_before_launch(self):
        for path in (self.model_path, self.candidate.executable):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with patch.object(runtime, "start_process") as start:
                with self.assertRaisesRegex(runtime.RuntimeFailure, "changed"):
                    self.generate()
                start.assert_not_called()
            # Restoring bytes does not restore the inspection signature. Rebuild
            # only the fixture identity before checking the other file.
            path.write_bytes(original)
            self.model = replace(self.model, signature=tuple(file_signature(self.model_path)))
            self.request = replace(self.request, model=self.model)
            self.candidate = replace(
                self.candidate,
                payload_signatures=(tuple(file_signature(self.candidate.executable)),),
            )

    def test_changed_vulkan_visibility_requires_reverification(self):
        with (
            patch.dict(os.environ, GGML_VK_VISIBLE_DEVICES="1"),
            patch.object(runtime, "start_process") as start,
        ):
            with self.assertRaisesRegex(runtime.RuntimeFailure, "environment changed"):
                self.generate()
            start.assert_not_called()

    def test_cancel_reaps_owned_process_and_job_even_when_terminate_hangs(self):
        running = [True]
        process, job = Mock(returncode=None), Mock()
        process.poll.side_effect = lambda: None if running[0] else -9
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 5), None]
        process.kill.side_effect = lambda: running.__setitem__(0, False)
        calls = [0]

        def cancel():
            calls[0] += 1
            if calls[0] > 1:
                raise CancelledError()

        with patch.object(runtime, "start_process", return_value=(process, job)):
            with self.assertRaises(CancelledError):
                runtime._run(
                    ["inert"], cwd=self.root, environment={}, check_cancelled=cancel, timeout=10
                )
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        job.close.assert_called_once()

    def test_timeout_reaps_child_without_fallback(self):
        running = [True]
        process, job = Mock(returncode=None), Mock()
        process.poll.side_effect = lambda: None if running[0] else -15
        process.terminate.side_effect = lambda: running.__setitem__(0, False)
        with (
            patch.object(runtime, "start_process", return_value=(process, job)) as start,
            patch.object(runtime.time, "monotonic", side_effect=[0, 2]),
        ):
            with self.assertRaisesRegex(runtime.RuntimeFailure, "exceeded"):
                runtime._run(
                    ["inert"],
                    cwd=self.root,
                    environment={},
                    check_cancelled=lambda: None,
                    timeout=1,
                )
        self.assertEqual(start.call_count, 1)
        process.wait.assert_called_once()
        job.close.assert_called_once()

    def test_nonzero_missing_or_wrong_geometry_result_is_not_success(self):
        for kind in ("nonzero", "missing", "wrong size"):

            def start(command, **options):
                process, job = self.fake_process(command, **options)
                output = Path(command[command.index("--output") + 1])
                if kind == "nonzero":
                    process.returncode = 3
                    process.poll.return_value = 3
                elif kind == "missing":
                    output.unlink()
                else:
                    Image.new("RGB", (64, 64)).save(output)
                return process, job

            with (
                self.subTest(kind=kind),
                patch.object(runtime, "start_process", side_effect=start) as launch,
            ):
                with self.assertRaises(runtime.RuntimeFailure):
                    self.generate()
                self.assertEqual(launch.call_count, 1)
                self.assertFalse(self.process_directory.exists())


if __name__ == "__main__":
    unittest.main()
