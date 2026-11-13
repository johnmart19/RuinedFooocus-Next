"""Saved backend choices obey CLI > environment > saved profile > automatic."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.comfy import image_compute
from ruinedfooocus.packaging import backend_preferences as prefs


def launch_args(**values):
    base = dict(cpu=False, directml=None, rocm10=False, cuda_nightly=False, legacy=False)
    return SimpleNamespace(**(base | values))


def apply(values, options=(), environment=None, **args):
    result = launch_args(**args)
    with patch.dict(os.environ, environment or {}, clear=False):
        os.environ.pop("TORCH_PLATFORM", None) if environment is None else None
        image_compute.apply_compute(result, values, options)
    return result


@contextmanager
def saved_profile(values):
    with (
        tempfile.TemporaryDirectory() as folder,
        patch.dict(os.environ, RF_DATA_DIR=folder, TORCH_PLATFORM=""),
        patch.object(sys, "argv", ["RuinedFooocus", "--settings=saved"]),
    ):
        path = Path(folder) / "settings/saved/settings.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(values), encoding="utf-8")
        yield path


class SavedImageBackend(unittest.TestCase):
    def test_legacy_cpu_device_migrates_only_when_no_backend_is_saved(self):
        self.assertEqual(prefs.saved_image_backend({"image_device": "CPU"}), "cpu")
        self.assertEqual(prefs.saved_image_backend({"image_device": "Auto"}), "auto")
        self.assertEqual(prefs.saved_image_backend({}), "auto")
        # An explicit choice in the new tab outranks a stale legacy value.
        self.assertEqual(
            prefs.saved_image_backend({"image_backend": "auto", "image_device": "CPU"}), "auto"
        )
        self.assertEqual(
            prefs.saved_image_backend({"image_backend": "rocm", "image_device": "CPU"}), "rocm"
        )

    def test_cpu_applies_from_either_saved_choice(self):
        for values in (
            {"image_backend": "cpu"},
            {"image_device": "CPU"},
            {"cpu_everywhere": True},
            {"cpu_everywhere": True, "image_backend": "rocm"},
        ):
            with self.subTest(values=values):
                self.assertTrue(apply(values).cpu)
        for values in (
            {},
            {"image_backend": "rocm"},
            {"image_backend": "auto", "image_device": "CPU"},
        ):
            with self.subTest(values=values):
                self.assertFalse(apply(values).cpu)

    def test_cli_environment_then_saved_precedence(self):
        saved = {"image_backend": "cpu"}
        # The environment wins over a saved CPU request.
        result = apply(saved, environment={"TORCH_PLATFORM": "rocm10.1"})
        self.assertFalse(result.cpu)
        # Explicit device-selecting flags keep priority as well.
        for option in ("--directml", "--gpu-only", "--rocm10", "--legacy", "--cuda-nightly"):
            with self.subTest(option=option):
                self.assertFalse(apply(saved, options=(option,)).cpu)
        # An explicit --cpu is applied by the launcher itself, not by the profile.
        self.assertTrue(apply({}, cpu=True).cpu)

    def test_invalid_saved_choices_are_rejected_before_use(self):
        with self.assertRaises(ValueError):
            apply({"image_backend": "tpu"})
        with self.assertRaises(ValueError):
            apply({"cpu_everywhere": "yes"})


class CpuEverywhere(unittest.TestCase):
    def test_every_component_is_forced_but_saved_choices_are_kept(self):
        values = {
            "cpu_everywhere": True,
            "image_backend": "rocm",
            "llama_backend": "Vulkan",
            "rembg_backend": "Windows GPU",
        }
        before = dict(values)
        effective = prefs.effective_backends(values)
        self.assertEqual(
            effective,
            {
                "forced_cpu": True,
                "image": "cpu",
                "llama": "CPU",
                "rembg": "CPU",
                "xllamacpp": "CPU",
            },
        )
        self.assertEqual(values, before)

    def test_turning_the_switch_off_restores_the_saved_choices(self):
        values = {
            "cpu_everywhere": False,
            "image_backend": "rocm",
            "llama_backend": "Vulkan",
            "rembg_backend": "Windows GPU",
        }
        self.assertEqual(
            prefs.effective_backends(values),
            {
                "forced_cpu": False,
                "image": "rocm",
                "llama": "Vulkan",
                "rembg": "Windows GPU",
                "xllamacpp": "Auto",
            },
        )
        self.assertEqual(
            prefs.effective_backends({}),
            {
                "forced_cpu": False,
                "image": "auto",
                "llama": "Auto",
                "rembg": "Auto",
                "xllamacpp": "Auto",
            },
        )


class AuxiliaryCPUPrecedence(unittest.TestCase):
    def test_automatic_image_cpu_never_becomes_global(self):
        for backend in ({}, {"cpu_origin": "automatic"}, {"cpu_scope": "image"}):
            args = SimpleNamespace(cpu=True, _compute_status={"cli": [], "backend": backend})
            self.assertFalse(prefs.force_cpu_for_auxiliary(args))
        self.assertEqual(
            prefs.cpu_selection_scope(automatic_image_cpu=True),
            {"cpu_scope": "image", "cpu_origin": "automatic"},
        )

    def test_cpu_scope_preserves_cli_environment_saved_precedence(self):
        self.assertEqual(
            prefs.cpu_selection_scope(cli_cpu=True, environment={"TORCH_PLATFORM": "xpu"})[
                "cpu_scope"
            ],
            "global",
        )
        self.assertEqual(
            prefs.cpu_selection_scope(
                environment={"TORCH_PLATFORM": "cpu"}, values={"cpu_everywhere": True}
            )["cpu_scope"],
            "image",
        )
        self.assertEqual(
            prefs.cpu_selection_scope(
                environment={"TORCH_PLATFORM": "xpu"}, values={"cpu_everywhere": True}
            )["cpu_scope"],
            "none",
        )
        self.assertEqual(
            prefs.cpu_selection_scope(values={"cpu_everywhere": True})["cpu_scope"], "global"
        )
        self.assertEqual(
            prefs.cpu_selection_scope(values={"image_backend": "cpu", "cpu_everywhere": False})[
                "cpu_scope"
            ],
            "image",
        )

    def test_image_only_cpu_keeps_other_backend_choices_independent(self):
        args = SimpleNamespace(
            cpu=True,
            _compute_status={
                "cli": [],
                "backend": {
                    "saved_image_backend": "cpu",
                    "cpu_applied": True,
                    "cpu_everywhere": False,
                },
            },
        )
        self.assertFalse(prefs.force_cpu_for_auxiliary(args))
        args._compute_status["backend"]["cpu_everywhere"] = True
        self.assertTrue(prefs.force_cpu_for_auxiliary(args))
        args._compute_status["backend"]["cpu_everywhere"] = False
        args._compute_status["cli"] = ["--cpu"]
        self.assertTrue(prefs.force_cpu_for_auxiliary(args))
        self.assertTrue(prefs.force_cpu_for_auxiliary(SimpleNamespace(cpu=True)))
        self.assertFalse(prefs.force_cpu_for_auxiliary(SimpleNamespace(cpu=False)))


class TorchSwitchDecision(unittest.TestCase):
    def decide(self, saved, installed, **options):
        return prefs.torch_switch_decision(
            {"image_backend": saved}, installed, reinstall_pending=False, **options
        )

    def test_matching_automatic_and_cpu_choices_change_nothing(self):
        for saved, installed in (("auto", "rocm"), ("cpu", "rocm"), ("rocm", "rocm")):
            self.assertEqual(self.decide(saved, installed)["action"], "none")

    def test_a_different_gpu_build_waits_for_an_explicit_reinstall_request(self):
        decision = self.decide("cuda", "rocm")
        self.assertEqual(decision["action"], "pending")
        self.assertIn("Hardware & backends", decision["note"])
        applied = prefs.torch_switch_decision(
            {"image_backend": "cuda"}, "rocm", reinstall_pending=True
        )
        self.assertEqual(applied["action"], "apply")
        self.assertEqual(self.decide("rocm", None)["action"], "pending")

    def test_explicit_startup_choices_override_the_saved_backend(self):
        decision = self.decide("cuda", "rocm", overrides=("TORCH_PLATFORM",))
        self.assertEqual(decision["action"], "overridden")
        self.assertIn("TORCH_PLATFORM", decision["note"])

    def test_cli_overrides_are_reported_by_name(self):
        self.assertEqual(prefs.cli_overrides(launch_args(), {}), [])
        self.assertEqual(
            prefs.cli_overrides(launch_args(directml=-1, rocm10=True), {"TORCH_PLATFORM": "cpu"}),
            ["directml", "rocm10", "TORCH_PLATFORM"],
        )
        self.assertEqual(prefs.cli_overrides(launch_args(directml=0), {}), ["directml"])
        self.assertEqual(prefs.cli_overrides(launch_args(gpu_device_id=0), {}), ["gpu_device_id"])
        self.assertEqual(
            prefs.cli_overrides(launch_args(oneapi_device_selector=""), {}),
            ["oneapi_device_selector"],
        )

    def test_applying_a_pending_switch_exports_the_resolved_platform(self):
        args = launch_args(_compute_status={})
        with (
            saved_profile({"image_backend": "rocm"}) as profile,
            patch.object(prefs, "installed_torch_kind", return_value="cuda"),
            patch(
                "ruinedfooocus.packaging.rocm_installer.automatic_rocm_plan",
                return_value=SimpleNamespace(platform="rocm10.1"),
            ) as producer,
        ):
            # The explicit launch profile must win over the saved next launch.
            settings = profile.parent.parent
            (settings / "startup.json").write_text('{"profile":"other"}', encoding="utf-8")
            (settings / "other").mkdir()
            (settings / "other/settings.json").write_text(
                '{"image_backend":"xpu"}', encoding="utf-8"
            )
            prefs.apply_saved_image_backend(args, "Windows", reinstall=True)
            self.assertEqual(os.environ["TORCH_PLATFORM"], "rocm10.1")
            producer.assert_called_once()
            producer.reset_mock()
            os.environ.pop("TORCH_PLATFORM")
            decision = prefs.apply_saved_image_backend(args, "Windows", reinstall=False)
            self.assertEqual(decision["action"], "pending")
            self.assertNotIn("TORCH_PLATFORM", os.environ)
            producer.assert_not_called()
        self.assertEqual(args._compute_status["backend"]["action"], "pending")

    def test_real_saved_cpu_choices_do_not_acquire_gpu_runtimes(self):
        for values, expected in (
            ({"image_backend": "cpu"}, "cpu"),
            ({"image_device": "CPU"}, "cpu"),
            ({"cpu_everywhere": True, "image_backend": "rocm"}, "cpu"),
            ({"image_backend": "auto", "image_device": "CPU"}, "auto"),
        ):
            with (
                self.subTest(values=values),
                saved_profile(values),
                patch.object(prefs, "installed_torch_kind", return_value="cuda"),
                patch("ruinedfooocus.packaging.rocm_installer.automatic_rocm_plan") as producer,
            ):
                decision = prefs.apply_saved_image_backend(
                    launch_args(_compute_status={}), "Windows", reinstall=True
                )
                self.assertEqual((decision["action"], decision["backend"]), ("none", expected))
                self.assertFalse(os.environ.get("TORCH_PLATFORM"))
                producer.assert_not_called()

    def test_real_saved_xpu_and_matching_directml_choices_are_preserved(self):
        for backend in ("xpu", "directml"):
            with (
                self.subTest(backend=backend),
                saved_profile({"image_backend": backend}),
                patch.object(prefs, "installed_torch_kind", return_value="cuda"),
                patch(
                    "ruinedfooocus.packaging.runtime_support.installed_directml_matches_requirements",
                    return_value=True,
                ) as matching_directml,
            ):
                decision = prefs.apply_saved_image_backend(
                    launch_args(_compute_status={}), "Windows", reinstall=True
                )
                self.assertEqual((decision["action"], decision["backend"]), ("apply", backend))
                self.assertEqual(os.environ["TORCH_PLATFORM"], backend)
                if backend == "directml":
                    matching_directml.assert_called_once_with("Windows")
                else:
                    matching_directml.assert_not_called()

    def test_real_saved_backend_keeps_cli_and_environment_override_priority(self):
        with (
            saved_profile({"image_backend": "rocm"}),
            patch.object(prefs, "installed_torch_kind", return_value="cuda"),
            patch("ruinedfooocus.packaging.rocm_installer.automatic_rocm_plan") as producer,
        ):
            for options, environment in (({"rocm10": True}, ""), ({}, "xpu")):
                with self.subTest(options=options, environment=environment):
                    os.environ["TORCH_PLATFORM"] = environment
                    decision = prefs.apply_saved_image_backend(
                        launch_args(_compute_status={}, **options), "Windows", reinstall=True
                    )
                    self.assertEqual(
                        (decision["action"], decision["backend"]), ("overridden", "rocm")
                    )
                    self.assertEqual(os.environ["TORCH_PLATFORM"], environment)
            producer.assert_not_called()

    def test_backend_hook_keeps_full_profile_validation_before_inspection(self):
        for invalid in ({"grid_columns": 0}, {"cpu_everywhere": "yes"}, {"image_backend": "tpu"}):
            with (
                self.subTest(invalid=invalid),
                saved_profile({"image_backend": "rocm", **invalid}) as profile,
                patch.object(prefs, "installed_torch_kind") as inspect,
            ):
                original = profile.read_bytes()
                with self.assertRaises(ValueError):
                    prefs.apply_saved_image_backend(
                        launch_args(_compute_status={}), "Windows", reinstall=True
                    )
                inspect.assert_not_called()
                self.assertEqual(profile.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
