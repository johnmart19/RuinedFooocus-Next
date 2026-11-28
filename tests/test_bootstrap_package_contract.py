"""The real bootstrap caller must admit an image-only package closure, without installs."""

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.packaging import package_plan as plans
from ruinedfooocus.packaging.runtime_support import RuntimeArtifact, RuntimeTarget


class BootstrapPackageContractTests(unittest.TestCase):
    def setUp(self):
        self.target = RuntimeTarget(
            "Windows", "AMD64", "3.12.10", "cp312", frozenset({"cp312-cp312-win_amd64"})
        )
        self.artifacts = tuple(
            RuntimeArtifact(
                name,
                version,
                f"{name}-{version}-cp312-cp312-win_amd64.whl",
                True,
                ">=3.11",
                requirements=requirements,
                source=f"https://download.pytorch.org/whl/cpu/{name}.whl",
                sha256="a" * 64,
            )
            for name, version, requirements in (
                ("torch", "2.14.0", ()),
                ("torchvision", "0.29.0", ("torch==2.14.0",)),
            )
        )

    def bootstrap_plan(self, artifacts, target=None):
        # Importing bootstrap launches the application. Execute its real function
        # in isolation, replacing external work and stopping at the plan boundary.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "ruinedfooocus/app/bootstrap.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_environment"
        )
        namespace = dict(
            os=os,
            sys=sys,
            Path=Path,
            version=SimpleNamespace(version="fixture"),
            REINSTALL_ALL=False,
            REINSTALL_TORCH=False,
            TORCH_FROZEN=False,
            args=SimpleNamespace(cpu=True, cuda_nightly=False, rocm10=False, api=False),
            git_repos=[{"name": "ComfyUI", "hash": "pin"}],
            inspect_installed_rocm=Mock(return_value=None),
            inspect_installed_cuda=Mock(return_value=None),
            inspect_installed_cpu=Mock(return_value=None),
            select_torch_platform=Mock(return_value="cpu"),
            preserve_installed_cuda=Mock(return_value=False),
            preserve_installed_rocm=Mock(return_value=False),
            automatic_rocm_plan=Mock(return_value=None),
            configure_torch_allocator=Mock(),
            installed_torch_constraints=Mock(return_value=None),
            clone_git_repos=Mock(),
            torch_install_commands=Mock(
                return_value=[
                    ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]
                ]
            ),
            run=Mock(side_effect=AssertionError("No commands")),
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "bootstrap-contract", "exec"),
            namespace,
        )
        original = plans.bootstrap_plan
        observed = {}

        class Planned(Exception):
            pass

        def capture(*args, **kwargs):
            observed["candidate"] = kwargs["candidate"]
            observed["plan"] = original(*args, **kwargs)
            raise Planned

        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder)
            core = fixture / "repositories/ComfyUI"
            core.mkdir(parents=True)
            (core / "requirements.txt").write_text("torch\ntorchvision\n")
            (fixture / "pip").mkdir()
            (fixture / "pip/modules.txt").write_text("")
            namespace.update(script_path=fixture, repo_dir=lambda name: core)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.dict(sys.modules, {"torchruntime": SimpleNamespace()}),
                patch.object(sys, "_rf_bootstrap_checked", True, create=True),
                patch("platform.system", return_value="Windows"),
                patch("platform.machine", return_value="AMD64"),
                patch("ruinedfooocus.core.dependency_cache.needed", return_value=False),
                patch("ruinedfooocus.packaging.backend_preferences.apply_saved_image_backend"),
                patch(
                    "ruinedfooocus.packaging.runtime_bundles.python_bundle_supported",
                    return_value=True,
                ),
                patch(
                    "ruinedfooocus.packaging.hardware_inventory.gpu_inventory",
                    side_effect=AssertionError("No GPU queries"),
                ),
                patch.object(plans, "current_target", return_value=target or self.target),
                patch.object(plans, "resolve_packages", return_value=artifacts),
                patch.object(plans, "bootstrap_plan", side_effect=capture),
                patch.object(
                    plans.subprocess, "check_output", side_effect=["pin", b"torch\ntorchvision\n"]
                ),
                patch.object(
                    plans.subprocess, "run", side_effect=AssertionError("No child process")
                ),
                patch.object(plans, "execute_plan", side_effect=AssertionError("No installation")),
            ):
                try:
                    namespace["prepare_environment"]()
                except Planned:
                    pass
        self.assertIn("plan", observed)
        return observed["candidate"], observed["plan"]

    def test_real_bootstrap_admits_torch_and_torchvision_without_torchaudio(self):
        candidate, plan = self.bootstrap_plan(self.artifacts)
        self.assertEqual(candidate.required_packages, ("torch", "torchvision"))
        self.assertEqual(candidate.operation, "image")
        self.assertTrue(plan.ready, plan.conflicts)
        self.assertTrue(json.loads(plan.decision_json)["eligible"])
        self.assertEqual({item.name for item in plan.artifacts}, {"torch", "torchvision"})

    def test_real_bootstrap_keeps_image_package_and_target_compatibility_gates(self):
        for artifacts, target, reason in (
            (self.artifacts[:1], self.target, "missing torchvision"),
            (
                (self.artifacts[0], replace(self.artifacts[1], requirements=("torch==2.13.0",))),
                self.target,
                "torch==2.13.0",
            ),
            (
                (replace(self.artifacts[0], requires_python=">=3.13"), self.artifacts[1]),
                self.target,
                "requires Python >=3.13",
            ),
            (
                self.artifacts,
                replace(
                    self.target,
                    python_abi="cp314t",
                    wheel_tags=frozenset({"cp314-cp314t-win_amd64"}),
                ),
                "no wheel matching cp314t",
            ),
            (self.artifacts, replace(self.target, machine="ARM64"), "not published for ARM64"),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, reason):
                self.bootstrap_plan(artifacts, target)


if __name__ == "__main__":
    unittest.main()
