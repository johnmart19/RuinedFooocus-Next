"""Installer probes use mocked children and never load a GPU runtime."""

import json
import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace

import ruinedfooocus.packaging.runtime_support as runtime_support


class CapabilityPolicy(unittest.TestCase):
    """Exact mocked artifacts/devices; these fixtures never qualify real hardware."""

    def setUp(self):
        self.target = runtime_support.RuntimeTarget(
            "Windows",
            "AMD64",
            "3.12.10",
            "cp312",
            frozenset({"cp312-cp312-win_amd64", "py3-none-any"}),
        )
        self.adapters = [
            {"adapter_id": "pci:0", "vendor": "nvidia", "kind": "gpu", "location": "native"},
            {"adapter_id": "pci:1", "vendor": "intel", "kind": "gpu", "location": "native"},
        ]
        self.artifact = runtime_support.RuntimeArtifact(
            "torch",
            "2.14.0",
            "torch-2.14.0-cp312-cp312-win_amd64.whl",
            True,
            ">=3.11",
            source="https://vendor.example/cu128",
        )
        self.candidate = runtime_support.RuntimeCandidate(
            "cu128",
            "image",
            ("Windows", "Linux"),
            adapter_id="pci:0",
            vendor="nvidia",
            hardware_supported=True,
            driver_supported=True,
            artifacts=(self.artifact,),
            required_packages=("torch",),
            closure_complete=True,
            native_payload=True,
            python_validated=True,
            installed=True,
            evidence_current=True,
            device_visible=True,
            workload_qualified=True,
        )

    def decide(self, **changes):
        return runtime_support.capability_decision(
            self.target, replace(self.candidate, **changes), self.adapters
        )

    def test_exact_qualified_route_is_usable_without_runtime_import_or_process(self):
        with patch.object(
            runtime_support.subprocess, "run", side_effect=AssertionError("No probes")
        ):
            row = self.decide()
        self.assertTrue(row["eligible"])
        self.assertTrue(row["auto_eligible"])
        self.assertEqual(row["state"], "qualified")

    def test_presence_installed_visibility_and_qualification_are_distinct(self):
        for changes, state in (
            ({"hardware_supported": None}, "unknown"),
            ({"python_validated": None}, "unknown"),
            ({"installed": False}, "installable"),
            ({"evidence_current": False}, "installed"),
            ({"device_visible": False}, "installed"),
            ({"workload_qualified": None}, "usable"),
            ({"hardware_supported": False}, "unavailable"),
        ):
            with self.subTest(changes=changes):
                row = self.decide(**changes)
                self.assertEqual(row["state"], state)
                self.assertFalse(row["auto_eligible"])
                self.assertTrue(row["reasons"])

    def test_unknown_inventory_is_not_absent_hardware(self):
        unknown = runtime_support.capability_decision(self.target, self.candidate)
        absent = runtime_support.capability_decision(self.target, self.candidate, [])
        self.assertEqual(unknown["state"], "unknown")
        self.assertEqual(absent["state"], "unavailable")

    def test_python_branch_free_threaded_abi_and_platform_tags(self):
        for version, abi, tags in (
            ("3.10.16", "cp310", {"cp310-cp310-win_amd64"}),
            ("3.14.0", "cp314t", {"cp314-cp314t-win_amd64"}),
            ("3.12.10", "cp312", {"cp312-cp312-manylinux_2_28_x86_64"}),
            ("3.12.10", "cp312", {"cp312-cp312-win_arm64"}),
        ):
            with self.subTest(version=version, abi=abi, tags=tags):
                target = replace(
                    self.target, python_version=version, python_abi=abi, wheel_tags=frozenset(tags)
                )
                row = runtime_support.capability_decision(target, self.candidate, self.adapters)
                self.assertFalse(row["eligible"])
                self.assertEqual(row["state"], "unavailable")

    def test_ordinary_cpython_311_through_314_use_their_exact_wheel(self):
        for minor in (11, 12, 13, 14):
            abi = f"cp3{minor}"
            tag = f"{abi}-{abi}-win_amd64"
            target = replace(
                self.target,
                python_version=f"3.{minor}.0",
                python_abi=abi,
                wheel_tags=frozenset({tag}),
            )
            artifact = replace(self.artifact, filename=f"torch-2.14.0-{tag}.whl")
            self.assertTrue(
                runtime_support.capability_decision(
                    target, replace(self.candidate, artifacts=(artifact,)), self.adapters
                )["eligible"]
            )

    def test_legacy_nvidia_recipe_does_not_inherit_modern_architecture_qualification(self):
        modern = self.decide(
            backend="cuda", platform="cu132", recipe_id="torch-modern", hardware_supported=False
        )
        legacy = self.decide(backend="cuda", platform="cu126", recipe_id="torch-legacy")
        self.assertEqual(
            runtime_support.select_capability([modern, legacy], "image")["platform"], "cu126"
        )

    def test_backend_family_and_exact_recipe_survive_catalogue_and_platform_selection(self):
        from ruinedfooocus.packaging.hardware_inventory import image_backend_choices
        from ruinedfooocus.packaging.support_matrix import support_rows

        modern = self.decide(
            backend="cuda", platform="cu132", recipe_id="torch-modern", hardware_supported=False
        )
        legacy = self.decide(backend="cuda", platform="cu126", recipe_id="torch-legacy")
        decisions = [modern, legacy]
        self.assertEqual(
            image_backend_choices(self.adapters, decisions=decisions), ["auto", "cuda"]
        )
        selected = runtime_support.select_capability(decisions, "image", requested="cuda")
        self.assertEqual(
            (selected["resolved"], selected["platform"], selected["recipe_id"]),
            ("cuda", "cu126", "torch-legacy"),
        )
        self.assertEqual(
            runtime_support.select_torch_platform(
                None, "Windows", decisions=decisions, requested="cuda"
            ),
            "cu126",
        )
        self.assertEqual(
            runtime_support.select_torch_platform(
                None, "Windows", decisions=decisions, requested="torch-legacy"
            ),
            "cu126",
        )
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            runtime_support.select_torch_platform(
                None, "Windows", decisions=decisions, requested="cu132"
            )
        self.assertEqual(support_rows(decisions=[legacy])[0]["platform"], "cu126")
        self.assertEqual(support_rows(decisions=[legacy])[0]["recipe_id"], "torch-legacy")
        normalized = self.decide(backend="cu126")
        self.assertEqual((normalized["backend"], normalized["platform"]), ("cuda", "cu126"))
        self.assertEqual(
            image_backend_choices(self.adapters, decisions=[normalized]), ["auto", "cuda"]
        )
        self.assertEqual(self.decide(backend="cu132", platform="cu126")["state"], "unavailable")

    def test_abstract_cuda_cannot_be_used_as_an_exact_installer_platform(self):
        decision = self.decide(backend="cuda", platform=None)
        self.assertEqual(decision["state"], "unknown")
        with self.assertRaisesRegex(RuntimeError, "Exact runtime platform is unknown"):
            runtime_support.select_torch_platform(
                None, "Windows", decisions=[decision], requested="cuda"
            )

    def test_missing_transitive_wheels_conflicts_and_wrong_sources(self):
        for artifact, extra in (
            (replace(self.artifact, available=False), {}),
            (replace(self.artifact, requirements=("torchvision==0.29.0",)), {}),
            (replace(self.artifact, filename="torch-2.15.0-cp312-cp312-win_amd64.whl"), {}),
            (self.artifact, {"artifact_sources": ("https://other.example/index",)}),
        ):
            with self.subTest(artifact=artifact, extra=extra):
                self.assertEqual(
                    self.decide(artifacts=(artifact,), **extra)["state"], "unavailable"
                )
        self.assertEqual(
            self.decide(artifacts=(replace(self.artifact, available=None),))["state"], "unknown"
        )

    def test_dependency_markers_follow_target_and_extras_not_the_host(self):
        torch = replace(
            self.artifact,
            requirements=('linux-lib==1; sys_platform == "linux"', 'cuda-lib==1; extra == "cuda"'),
        )
        self.assertTrue(self.decide(artifacts=(torch,))["eligible"])
        self.assertFalse(self.decide(artifacts=(replace(torch, extras=("cuda",)),))["eligible"])
        companion = runtime_support.RuntimeArtifact(
            "cuda-lib", "2", "cuda_lib-2-py3-none-any.whl", True, source="https://vendor.example"
        )
        self.assertFalse(
            self.decide(artifacts=(replace(torch, extras=("cuda",)), companion))["eligible"]
        )

    def test_referenced_unknown_marker_metadata_blocks_dependency_closure(self):
        artifact = replace(
            self.artifact, requirements=('release-lib==1; platform_release == "6.8.0"',)
        )
        with patch(
            "packaging.markers.default_environment",
            return_value={"platform_release": "other-host-release"},
        ):
            row = self.decide(artifacts=(artifact,))
        self.assertEqual(row["state"], "unknown")
        self.assertTrue(
            any("needs target metadata: platform_release" in reason for reason in row["reasons"])
        )
        candidate = replace(self.candidate, artifacts=(artifact,))
        matching = replace(self.target, marker_environment={"platform_release": "6.8.0"})
        self.assertEqual(
            runtime_support.capability_decision(matching, candidate, self.adapters)["state"],
            "unavailable",
        )
        dependency = runtime_support.RuntimeArtifact(
            "release-lib",
            "1",
            "release_lib-1-py3-none-any.whl",
            True,
            source="https://vendor.example",
        )
        self.assertTrue(
            runtime_support.capability_decision(
                matching, replace(candidate, artifacts=(artifact, dependency)), self.adapters
            )["eligible"]
        )
        other = replace(self.target, marker_environment={"platform_release": "6.9.0"})
        with patch(
            "packaging.markers.default_environment", return_value={"platform_release": "6.8.0"}
        ):
            self.assertTrue(
                runtime_support.capability_decision(other, candidate, self.adapters)["eligible"]
            )

    def test_literal_marker_variable_names_are_not_unresolved_variables(self):
        artifact = replace(
            self.artifact, requirements=('release-lib==1; os_name == "platform_release"',)
        )
        self.assertTrue(self.decide(artifacts=(artifact,))["eligible"])
        artifact = replace(
            self.artifact, requirements=('release-lib==1; "6.8.0" == platform_release',)
        )
        self.assertEqual(self.decide(artifacts=(artifact,))["state"], "unknown")

    def test_wheel_availability_does_not_imply_publisher_python_validation(self):
        row = self.decide(python_validated=False)
        self.assertEqual(row["state"], "unknown")
        self.assertIn("Publisher has not validated this Python interpreter", row["reasons"])

    def test_native_executable_does_not_require_python_wheel_tags(self):
        candidate = replace(
            self.candidate,
            backend="Vulkan",
            operation="llama",
            native_executable=True,
            artifacts=(),
            required_packages=(),
        )
        target = replace(
            self.target, python_version="3.14.0", python_abi="cp314t", wheel_tags=frozenset()
        )
        self.assertTrue(
            runtime_support.capability_decision(target, candidate, self.adapters)["usable"]
        )

    def test_native_plugin_payload_is_required_even_for_generic_python_wheel(self):
        artifact = runtime_support.RuntimeArtifact(
            "hip-ep",
            "0.5.2",
            "hip_ep-0.5.2-py3-none-any.whl",
            True,
            source="https://vendor.example",
        )
        row = self.decide(
            backend="hip-plugin",
            artifacts=(artifact,),
            required_packages=("hip-ep",),
            native_payload=False,
        )
        self.assertFalse(row["eligible"])
        self.assertIn("Native runtime payload is missing", row["reasons"])

    def test_generic_plugin_cannot_hide_incompatible_ort_binding_abi(self):
        plugin = runtime_support.RuntimeArtifact(
            "hip-ep",
            "0.5.2",
            "hip_ep-0.5.2-py3-none-any.whl",
            True,
            requirements=("onnxruntime==1.30.0",),
            source="https://vendor.example",
        )
        binding = runtime_support.RuntimeArtifact(
            "onnxruntime",
            "1.30.0",
            "onnxruntime-1.30.0-cp314-cp314-win_amd64.whl",
            True,
            source="https://vendor.example",
        )
        self.assertFalse(
            self.decide(artifacts=(plugin, binding), required_packages=("hip-ep", "onnxruntime"))[
                "eligible"
            ]
        )

    def test_wsl_host_presence_requires_guest_evidence_and_separate_recipe(self):
        target = replace(
            self.target,
            system="Linux",
            machine="x86_64",
            environment="wsl",
            wheel_tags=frozenset({"cp312-cp312-manylinux_2_28_x86_64"}),
        )
        adapter = self.adapters[0] | {"location": "host", "guest_available": None}
        candidate = replace(
            self.candidate,
            native_executable=True,
            artifacts=(),
            required_packages=(),
            environments=("wsl",),
        )
        self.assertEqual(
            runtime_support.capability_decision(target, candidate, [adapter])["state"], "unknown"
        )
        acquisition = runtime_support.capability_decision(
            target, replace(candidate, installed=False, device_visible=None), [adapter]
        )
        self.assertEqual(acquisition["state"], "installable")
        self.assertFalse(acquisition["usable"])
        self.assertFalse(acquisition["qualified"])
        self.assertTrue(
            runtime_support.capability_decision(
                target, candidate, [adapter | {"guest_available": True}]
            )["usable"]
        )
        self.assertFalse(
            runtime_support.capability_decision(
                target,
                replace(candidate, environments=("native",)),
                [adapter | {"guest_available": True}],
            )["eligible"]
        )
        self.assertFalse(
            runtime_support.capability_decision(
                replace(target, environment="native"), candidate, [adapter]
            )["eligible"]
        )

    def test_npu_vendor_and_gpu_vendor_are_not_interchangeable(self):
        adapter = self.adapters[1] | {"kind": "npu"}
        candidate = replace(self.candidate, adapter_id="pci:1", vendor="intel", backend="xpu")
        self.assertFalse(
            runtime_support.capability_decision(self.target, candidate, [adapter])["eligible"]
        )
        candidate = replace(candidate, backend="torch_npu", vendor="huawei", device_kind="npu")
        self.assertFalse(
            runtime_support.capability_decision(self.target, candidate, [adapter])["eligible"]
        )

    def test_auto_tries_all_qualified_gpus_before_installed_cpu(self):
        cpu = self.decide(backend="cpu", adapter_id=None, device_kind="cpu")
        unsupported = self.decide(hardware_supported=False)
        intel = self.decide(backend="xpu", adapter_id="pci:1", vendor="intel")
        result = runtime_support.select_capability([cpu, unsupported, intel], "image")
        self.assertEqual(result["resolved"], "xpu")
        self.assertIsNone(
            runtime_support.select_capability([cpu, unsupported, intel], "image", requested="cuda")[
                "resolved"
            ]
        )
        self.assertEqual(
            runtime_support.select_capability([cpu, unsupported], "image")["resolved"], "cpu"
        )

    def test_cpu_fallback_explains_gpu_rejection(self):
        cpu = self.decide(backend="cpu", adapter_id=None, device_kind="cpu")
        unsupported = self.decide(hardware_supported=False)
        self.assertIn(
            "Device architecture is unsupported by this route",
            runtime_support.select_capability([cpu, unsupported], "image")["reasons"],
        )

    def test_opt_in_provider_is_never_auto_even_when_qualified(self):
        experimental = self.decide(backend="HIP-plugin", opt_in=True)
        self.assertIsNone(runtime_support.select_capability([experimental], "image")["resolved"])
        self.assertEqual(
            runtime_support.select_capability([experimental], "image", requested="HIP-plugin")[
                "resolved"
            ],
            "HIP-plugin",
        )

    def test_duplicate_adapter_selection_is_exact(self):
        first = self.decide()
        second = first | {"adapter_id": "pci:another"}
        result = runtime_support.select_capability(
            [first, second], "image", requested="cuda", requested_adapter_id="pci:another"
        )
        self.assertEqual(result["adapter_id"], "pci:another")

    def test_cpu_torch_does_not_disable_vulkan_and_cpu_scope_is_local(self):
        vulkan = self.decide(
            backend="Vulkan",
            operation="llama",
            native_executable=True,
            artifacts=(),
            required_packages=(),
        )
        cpu = self.decide(backend="cpu", operation="llama", adapter_id=None, device_kind="cpu")
        self.assertEqual(
            runtime_support.select_capability([cpu, vulkan], "llama", cpu_scope="image")[
                "resolved"
            ],
            "Vulkan",
        )
        self.assertEqual(
            runtime_support.select_capability([cpu, vulkan], "llama", cpu_scope="global")[
                "resolved"
            ],
            "cpu",
        )

    def test_global_cpu_overrides_a_saved_gpu_adapter_selection(self):
        gpu = self.decide()
        cpu = self.decide(backend="cpu", adapter_id=None, device_kind="cpu")
        self.assertEqual(
            runtime_support.select_capability(
                [gpu, cpu], "image", cpu_scope="global", requested_adapter_id="pci:0"
            )["resolved"],
            "cpu",
        )

    def test_rdna35_apu_and_v620_tuning_do_not_apply_to_gfx1100(self):
        target = replace(self.target, system="Linux")
        gpu = {
            "vendor": "amd",
            "architecture": "gfx1100",
            "product": "RX 7900 XTX",
            "integrated": False,
        }
        for guide in ("rdna35-apu-memory", "v620-sriov"):
            self.assertFalse(runtime_support.amd_guide_applicable(target, gpu, guide))
        self.assertTrue(
            runtime_support.amd_guide_applicable(
                target, gpu | {"architecture": "gfx1151", "integrated": True}, "rdna35-apu-memory"
            )
        )
        self.assertFalse(
            runtime_support.amd_guide_applicable(
                replace(target, environment="wsl"),
                gpu | {"architecture": "gfx1151", "integrated": True},
                "rdna35-apu-memory",
            )
        )

    def test_mixed_vendor_legacy_selector_tries_compatible_groups(self):
        amd, intel = SimpleNamespace(vendor_id="1002"), SimpleNamespace(vendor_id="8086")

        def choose(gpus):
            if len(gpus) > 1:
                raise NotImplementedError("multiple vendors")
            return "cpu" if gpus[0].vendor_id == "1002" else "xpu"

        runtime = SimpleNamespace(platform_detection=SimpleNamespace(get_torch_platform=choose))
        self.assertEqual(runtime_support._platform_for(runtime, [amd, intel]), "xpu")

    def test_mixed_vendor_cpu_or_directml_result_still_tries_native_alternatives(self):
        amd, intel = SimpleNamespace(vendor_id="1002"), SimpleNamespace(vendor_id="8086")
        for fallback in ("cpu", "directml"):

            def choose(group):
                return "xpu" if len(group) == 1 and group[0].vendor_id == "8086" else fallback

            runtime = SimpleNamespace(platform_detection=SimpleNamespace(get_torch_platform=choose))
            self.assertEqual(runtime_support._platform_for(runtime, [amd, intel]), "xpu")


class InstalledRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.version = patch.object(
            runtime_support.importlib.metadata, "version", return_value="2.9.1"
        )
        self.version.start()
        self.addCleanup(self.version.stop)

    def test_valid_bundle_survives_native_library_output(self):
        bundle = {"cuda": "12.8", "versions": {"torch": "2.9.1+cu128"}}
        child = SimpleNamespace(
            returncode=0, stdout="native library notice\n" + json.dumps(bundle), stderr=""
        )
        with patch.object(runtime_support.subprocess, "run", return_value=child):
            self.assertEqual(runtime_support.inspect_installed_cuda("Windows"), bundle)

    def test_unusable_cuda_probe_does_not_abort_startup(self):
        for output in ("", "native library notice", "[]", "{}", '{"cuda": 128, "versions": {}}'):
            with (
                self.subTest(output=output),
                patch.object(
                    runtime_support.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout=output, stderr=""),
                ),
            ):
                self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))
        for error in (OSError("cannot launch child"), subprocess.TimeoutExpired("probe", 90)):
            with (
                self.subTest(error=error),
                patch.object(runtime_support.subprocess, "run", side_effect=error),
            ):
                self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))

    def test_unusable_rocm_probe_requires_explicit_repair(self):
        child = SimpleNamespace(returncode=0, stdout="not JSON", stderr="")
        with patch.object(runtime_support.subprocess, "run", return_value=child):
            with self.assertRaisesRegex(RuntimeError, "has not replaced them"):
                runtime_support.inspect_installed_rocm("Windows")
            self.assertIsNone(runtime_support.inspect_installed_rocm("Windows", repair=True))

    def test_explicit_backend_override_prevents_other_gpu_probes(self):
        with (
            patch.dict(os.environ, {"TORCH_PLATFORM": "cpu"}),
            patch.object(runtime_support.subprocess, "run") as run,
        ):
            self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))
            self.assertIsNone(runtime_support.inspect_installed_rocm("Windows"))
            run.assert_not_called()

    def test_missing_runtime_is_not_a_probe_error(self):
        with patch.object(
            runtime_support.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="null\n", stderr=""),
        ):
            self.assertIsNone(runtime_support.inspect_installed_cpu())
            self.assertIsNone(runtime_support.inspect_installed_cuda("Windows"))
            self.assertIsNone(runtime_support.inspect_installed_rocm("Windows"))

    def test_supported_xpu_is_not_replaced_by_cpu_and_invalid_mps_still_is(self):
        for backend, platform, expected in (
            ("xpu", "Linux", "xpu"),
            ("xpu", "Windows", "xpu"),
            ("mps", "Linux", "cpu"),
            ("mps", "Darwin", "mps"),
            ("rocm5.5", "Linux", "cpu"),
            ("rocm10.1", "Windows", "rocm10.1"),
        ):
            with (
                self.subTest(backend=backend, platform=platform),
                patch.dict(os.environ, TORCH_PLATFORM=backend),
            ):
                self.assertEqual(runtime_support.select_torch_platform(None, platform), expected)

    def test_fresh_automatic_directml_does_not_downgrade_modern_torch(self):
        runtime = SimpleNamespace(
            device_db=SimpleNamespace(get_gpus=lambda: []),
            platform_detection=SimpleNamespace(get_torch_platform=lambda _: "directml"),
        )
        with (
            patch("ruinedfooocus.packaging.cuda_selection.nvidia_info", return_value=(None, None)),
            patch.object(
                runtime_support, "installed_directml_matches_requirements", return_value=False
            ) as installed,
        ):
            self.assertEqual(runtime_support.select_torch_platform(runtime, "Windows"), "cpu")
            installed.assert_called_once_with("Windows")
            with patch.dict(os.environ, TORCH_PLATFORM="directml"):
                self.assertEqual(runtime_support.select_torch_platform(None, "Windows"), "directml")
            installed.assert_called_once()
        with (
            patch("ruinedfooocus.packaging.cuda_selection.nvidia_info", return_value=(None, None)),
            patch.object(
                runtime_support, "installed_directml_matches_requirements", return_value=True
            ),
        ):
            self.assertEqual(runtime_support.select_torch_platform(runtime, "Windows"), "directml")

    def test_existing_directml_requires_its_declared_torch_companions_and_numpy_abi(self):
        versions = {
            "torch-directml": "0.2.5.dev240914",
            "torch": "2.4.1",
            "torchvision": "0.19.1",
            "numpy": "1.26.4",
        }
        with (
            patch.object(
                runtime_support.importlib.metadata, "version", side_effect=versions.__getitem__
            ),
            patch.object(
                runtime_support.importlib.metadata,
                "requires",
                return_value=["torch==2.4.1", "torchvision==0.19.1"],
            ),
        ):
            self.assertTrue(runtime_support.installed_directml_matches_requirements("Windows"))
            versions["torch"] = "2.15.0+cu132"
            self.assertFalse(runtime_support.installed_directml_matches_requirements("Windows"))
            versions["torch"] = "2.4.1"
            versions["numpy"] = "2.5.0"
            self.assertFalse(runtime_support.installed_directml_matches_requirements("Windows"))
            versions["numpy"] = "1.26.4"
            with patch.object(runtime_support.platform, "release", return_value="6.8.0-generic"):
                self.assertFalse(runtime_support.installed_directml_matches_requirements("Linux"))
            with patch.object(
                runtime_support.platform, "release", return_value="6.6.87.2-microsoft-standard-WSL2"
            ):
                self.assertTrue(runtime_support.installed_directml_matches_requirements("Linux"))
        with patch.object(
            runtime_support.importlib.metadata,
            "version",
            side_effect=runtime_support.importlib.metadata.PackageNotFoundError,
        ):
            self.assertFalse(runtime_support.installed_directml_matches_requirements("Windows"))


if __name__ == "__main__":
    unittest.main()
