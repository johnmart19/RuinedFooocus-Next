"""No-install tests for a frozen publisher base and RF overlay closure."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import ast
from types import SimpleNamespace
from unittest.mock import patch

from ruinedfooocus.packaging import package_plan as plans
from ruinedfooocus.packaging.install_lock import installation_lock
from ruinedfooocus.packaging.runtime_support import RuntimeArtifact, RuntimeTarget


def artifact(name, version="1.0", requirements=(), **kwargs):
    return RuntimeArtifact(
        name,
        version,
        f"{name}-{version}-py3-none-any.whl",
        True,
        requirements=requirements,
        source=f"https://vendor.test/wheels/{name}.whl",
        sha256="a" * 64,
        **kwargs,
    )


def layer(identity, *requirements):
    return plans.PackageLayer(
        identity, requirements, "https://publisher.test", "revision", "MIT", "a" * 64
    )


class PackagePlanTests(unittest.TestCase):
    def test_application_filelock_branch_respects_published_python_floor(self):
        # Published filelock 4.1 supports Python 3.10; 4.2 requires Python 3.11.
        # The former unmarked 4.2.9 overlay could not produce a Python 3.10 plan.
        from packaging.requirements import Requirement

        root = Path(__file__).resolve().parents[1]
        overlay = plans.read_layer(
            root / "pip/modules.txt",
            identity="rf-overlay",
            source="RuinedFooocus/pip/modules.txt",
            revision="content-bound",
            license="GPL-3.0",
        )
        overlay = replace(
            overlay,
            requirements=tuple(
                raw for raw in overlay.requirements if Requirement(raw).name == "filelock"
            ),
        )
        for python, version, floor in (
            ("3.10.0", "4.1.0", ">=3.10"),
            ("3.11.0", "4.2.10", ">=3.11"),
            ("3.12.10", "4.2.10", ">=3.11"),
        ):
            with self.subTest(python=python):
                target = replace(
                    self.target,
                    python_version=python,
                    marker_environment={
                        "python_version": ".".join(python.split(".")[:2]),
                        "python_full_version": python,
                    },
                )
                result = plans.plan_packages(
                    target,
                    {"eligible": True},
                    (),
                    (overlay,),
                    (artifact("filelock", version, requires_python=floor),),
                    environment=sys.prefix,
                    recipe="qualified-python-branch",
                )
                self.assertTrue(result.ready, result.conflicts)

    def test_nightly_bootstrap_freezes_and_resolves_prerelease_policy(self):
        from ruinedfooocus.packaging.runtime_support import RuntimeCandidate

        target = plans.current_target()
        index = "https://download.pytorch.org/whl/nightly/cu134"
        report = {
            "install": [
                {
                    "metadata": {"name": name, "version": version},
                    "download_info": {
                        "url": f"{index.replace('download.pytorch.org', 'download-r2.pytorch.org')}/{name}-{version}-py3-none-any.whl",
                        "archive_info": {"hashes": {"sha256": "a" * 64}},
                    },
                }
                for name, version in (
                    ("torch", "2.16.0.dev20261109"),
                    ("torchvision", "0.30.0.dev20261109"),
                )
            ]
        }

        def run(command, **kwargs):
            self.assertIn("--pre", command)
            self.assertIn("--dry-run", command)
            Path(command[command.index("--report") + 1]).write_text(json.dumps(report))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "repositories/ComfyUI").mkdir(parents=True)
            (root / "repositories/ComfyUI/requirements.txt").write_text("torch\n")
            (root / "pip").mkdir()
            (root / "pip/modules.txt").write_text("")
            candidate = RuntimeCandidate(
                "cpu",
                "image",
                (target.system,),
                machines=(target.machine,),
                environments=(target.environment,),
                device_kind="cpu",
                native_payload=True,
                python_validated=True,
            )
            with (
                patch.object(plans.subprocess, "check_output", side_effect=["pin", b"torch\n"]),
                patch.object(plans.subprocess, "run", side_effect=run),
            ):
                result = plans.bootstrap_plan(
                    root,
                    "pin",
                    "cu134-nightly",
                    [["--pre", "torch", "torchvision", "--index-url", index]],
                    candidate=candidate,
                )
        recipe = next(item for item in result.layers if item.identity.startswith("torch-recipe:"))
        self.assertTrue(result.ready, result.conflicts)
        self.assertTrue(recipe.allow_prereleases)
        self.assertEqual(
            recipe.artifact_sources, ("https://download-r2.pytorch.org/whl/nightly/cu134",)
        )
        stable_policy = replace(
            result, layers=tuple(replace(item, allow_prereleases=False) for item in result.layers)
        )
        self.assertNotEqual(result.digest, stable_policy.digest)

    def test_core_api_gate_preserves_only_explicitly_qualified_legacy_exceptions(self):
        with self.assertRaisesRegex(RuntimeError, "Core API requires Torch >= 2.7"):
            plans.require_core_torch_compatibility("2.6.0+cu124")
        plans.require_core_torch_compatibility("2.14.1+cu130")
        plans.require_core_torch_compatibility(
            "2.6.0+cu124", exceptions=("core-torch-floor:2.6.0+cu124",)
        )

    def test_bootstrap_skip_install_checks_core_floor_without_mutation_or_gpu_import(self):
        # Importing bootstrap starts the app. Execute its real entry function
        # alone to exercise the skip-install branch without those side effects.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "ruinedfooocus/app/bootstrap.py").read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_environment"
        )
        namespace = {"sys": sys, "version": SimpleNamespace(version="fixture")}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "bootstrap-fixture", "exec"),
            namespace,
        )
        with (
            patch.object(plans.importlib.metadata, "version", return_value="2.6.0+cu124"),
            patch.object(plans.subprocess, "run", side_effect=AssertionError("Mutation/GPU query")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Core API requires Torch >= 2.7"):
                namespace["prepare_environment"](skip_install=True)

    def test_satisfied_preserved_stack_cannot_short_circuit_core_floor(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "ruinedfooocus/app/bootstrap.py").read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_environment"
        )
        gate = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "preserve"
        )
        needs = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "needs_packages"
                for target in node.targets
            )
        )
        self.assertLess(gate.lineno, needs.lineno)
        namespace = dict(
            preserve=True,
            installed_torch={"versions": {"torch": "2.6.0+cu124"}},
            require_core_torch_compatibility=plans.require_core_torch_compatibility,
            REINSTALL_ALL=False,
            REINSTALL_TORCH=False,
            args=SimpleNamespace(api=False),
            base_file="satisfied-base",
            modules_file="satisfied-overlay",
            requirements_met=lambda path: True,
        )
        with self.assertRaisesRegex(RuntimeError, "Core API requires Torch >= 2.7"):
            exec(
                compile(
                    ast.Module(body=[gate, needs], type_ignores=[]),
                    "bootstrap-preserved-fixture",
                    "exec",
                ),
                namespace,
            )

    def test_overlay_classification_covers_every_requirement_once(self):
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name

        root = Path(__file__).resolve().parents[1]
        classification = json.loads((root / "pip/modules.layers.json").read_text())
        shared, additions = map(set, (classification["shared"], classification["additions"]))
        self.assertFalse(shared & additions)
        actual = {
            canonicalize_name(Requirement(line.split(" #", 1)[0]).name)
            for line in (root / "pip/modules.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(actual, shared | additions)

    def setUp(self):
        self.target = RuntimeTarget(
            "Windows",
            "AMD64",
            "3.12.10",
            "cp312",
            frozenset(("py3-none-any", "cp312-cp312-win_amd64")),
            marker_environment={"python_version": "3.12", "python_full_version": "3.12.10"},
        )

    def plan(self, base=(), overlay=(), artifacts=(), **kwargs):
        return plans.plan_packages(
            self.target,
            {"eligible": True},
            base,
            overlay,
            artifacts,
            environment=sys.prefix,
            recipe="qualified-recipe",
            **kwargs,
        )

    def test_combined_shared_bounds_prevent_overlay_replacing_base(self):
        base = layer("base", "Pillow>=10.1", "pydantic~=2.0")
        overlay = layer("rf", "Pillow==12.3", "pydantic==2.12.5")
        closure = (artifact("Pillow", "12.3"), artifact("pydantic", "2.12.5"))
        result = self.plan((base,), (overlay,), closure)
        self.assertTrue(result.ready, result.conflicts)
        before = result.digest
        self.assertEqual(result.layers[0], base)
        self.assertEqual(result.digest, before)
        for changed in (replace(closure[0], version="13.0"), replace(closure[1], version="3.0")):
            wrong = tuple(changed if item.name == changed.name else item for item in closure)
            self.assertFalse(self.plan((base,), (overlay,), wrong).ready)

    def test_transitive_mcp_gradio_and_directml_conflicts_are_rejected(self):
        cases = [
            (artifact("gradio", requirements=("Pillow<13",)), artifact("Pillow", "13.0")),
            (artifact("mcp", requirements=("pydantic<=2.12.5",)), artifact("pydantic", "2.13")),
            (artifact("scipy", requirements=("numpy<2",)), artifact("numpy", "2.0")),
            (artifact("kornia", requirements=("torch>=2.5.1",)), artifact("torch", "2.4.1")),
            (artifact("torchvision", requirements=("torch==2.14.1",)), artifact("torch", "2.15")),
        ]
        for closure in cases:
            with self.subTest(closure=closure):
                self.assertFalse(self.plan(artifacts=closure).ready)

    def test_missing_transitive_wheel_hash_and_unknown_capability_fail_closed(self):
        for closure in [
            (artifact("torch", "2.14", ("missing>=1",)),),
            (replace(artifact("demo"), sha256=""),),
            (replace(artifact("demo"), available=None),),
        ]:
            self.assertFalse(self.plan(artifacts=closure).ready)
        result = plans.plan_packages(
            self.target,
            {"eligible": False, "reasons": ("driver unknown",)},
            (),
            (),
            (artifact("demo"),),
            environment=sys.prefix,
            recipe="r",
        )
        self.assertIn("driver unknown", result.conflicts)

    def test_vendor_index_python_abi_architecture_and_guest_tags_are_exact(self):
        recipe = replace(layer("recipe", "torch==2.14"), artifact_index="https://amd.test/rocm")
        self.assertFalse(self.plan((recipe,), artifacts=(artifact("torch", "2.14"),)).ready)
        wheel = artifact("demo", requires_python="<3.12")
        self.assertFalse(self.plan(artifacts=(wheel,)).ready)
        for filename in (
            "demo-1.0-cp313-cp313t-win_amd64.whl",
            "demo-1.0-cp312-cp312-win_arm64.whl",
            "demo-1.0-cp312-cp312-manylinux_2_35_x86_64.whl",
        ):
            self.assertFalse(
                self.plan(artifacts=(replace(artifact("demo"), filename=filename),)).ready
            )

    def test_missing_target_marker_is_not_filled_from_the_host(self):
        base = layer("base", 'demo; platform_release == "some kernel"')
        result = self.plan((base,), artifacts=(artifact("demo"),))
        self.assertFalse(result.ready)
        self.assertIn("marker metadata missing", result.conflicts[0])

    def test_core_api_floor_requires_an_explicit_qualified_exception(self):
        closure = (artifact("torch", "2.6.0"),)
        self.assertFalse(self.plan(artifacts=closure).ready)
        self.assertTrue(self.plan(artifacts=closure, exceptions=("core-torch-floor:2.6.0",)).ready)

    def test_resolver_is_single_wheel_only_dry_run(self):
        def run(command, **kwargs):
            self.assertIn("--dry-run", command)
            self.assertIn("--ignore-installed", command)
            self.assertIn("--only-binary=:all:", command)
            Path(command[command.index("--report") + 1]).write_text(json.dumps({"install": []}))

        with patch.object(plans.subprocess, "run", side_effect=run) as execute:
            self.assertEqual(plans.resolve_packages((layer("base", "demo"),)), ())
            execute.assert_called_once()

    def test_vendor_selector_accepts_only_the_exact_prepared_wheel(self):
        source = "file:///C:/cache/sdk/rocm-10.1.0-py3-none-any.whl"
        recipe = replace(
            layer("recipe", "rocm==10.1.0"),
            artifact_index="https://stable.repo.amd.com/rocm/whl-next/",
            artifact_sources=(source,),
        )
        selector = replace(artifact("rocm", "10.1.0"), source=source)
        self.assertTrue(self.plan((recipe,), artifacts=(selector,)).ready)
        other = replace(selector, source="file:///C:/other/rocm-10.1.0-py3-none-any.whl")
        self.assertFalse(self.plan((recipe,), artifacts=(other,)).ready)

    def test_pytorch_download_host_keeps_channel_and_host_boundaries(self):
        recipe = replace(
            layer("recipe", "torch==2.14"),
            artifact_index="https://download.pytorch.org/whl/cu126",
            artifact_sources=("https://download-r2.pytorch.org/whl/cu126",),
        )
        wheel = artifact("torch", "2.14")
        for source, accepted in (
            ("https://download-r2.pytorch.org/whl/cu126/torch.whl", True),
            ("https://download.pytorch.org/whl/cu126/torch.whl", True),
            ("https://download-r2.pytorch.org/whl/cpu/torch.whl", False),
            ("https://download-r2.pytorch.org/whl/cu126-evil/torch.whl", False),
            ("https://download-r2.pytorch.org.evil/whl/cu126/torch.whl", False),
            ("http://download-r2.pytorch.org/whl/cu126/torch.whl", False),
            ("http://download.pytorch.org/whl/cu126/torch.whl", False),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    self.plan((recipe,), artifacts=(replace(wheel, source=source),)).ready,
                    accepted,
                )

    def test_executor_rejects_changed_digest_target_and_imported_native_before_mutation(self):
        result = self.plan(artifacts=(artifact("demo"),))
        with patch.object(plans.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "changed or unresolved"):
                plans.execute_plan(result, "bad")
            with self.assertRaisesRegex(RuntimeError, "different interpreter"):
                plans.execute_plan(result, result.digest)
            result = replace(result, target_json=plans.target_identity(plans.current_target()))
            native = replace(result, artifacts=(artifact("torch", "999.0"),))
            with (
                patch.dict(sys.modules, torch=object()),
                self.assertRaisesRegex(RuntimeError, "restart"),
            ):
                plans.execute_plan(native, native.digest)
            execute.assert_not_called()

    def test_failed_staging_never_runs_environment_install(self):
        result = replace(
            self.plan(artifacts=(artifact("demo"),)),
            target_json=plans.target_identity(plans.current_target()),
        )

        def fail(command, **kwargs):
            self.assertIn("download", command)
            raise subprocess.CalledProcessError(1, command)

        with (
            patch.dict(sys.modules),
            patch.object(plans.subprocess, "run", side_effect=fail) as execute,
        ):
            for name in ("torch", "xllamacpp", "onnxruntime", "torch_directml"):
                sys.modules.pop(name, None)
            with self.assertRaises(subprocess.CalledProcessError):
                plans.execute_plan(result, result.digest)
            execute.assert_called_once()

    def test_success_installs_only_frozen_local_artifacts_and_preserves_base(self):
        base = layer("base", "demo==1.0")
        result = replace(
            self.plan((base,), artifacts=(artifact("demo"),)),
            target_json=plans.target_identity(plans.current_target()),
        )
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if "install" in command:
                self.assertIn("--no-index", command)
                locked = Path(command[command.index("-r") + 1]).read_text()
                self.assertIn("file:", locked)
                self.assertIn("--hash=sha256:" + "a" * 64, locked)
                self.assertNotIn("https://", locked)

        with patch.object(plans.importlib.metadata, "version", return_value="1.0"):
            applied = plans.execute_plan(result, result.digest, runner=run)
        self.assertEqual(len(calls), 3)
        self.assertEqual(applied.layers[0], base)
        self.assertTrue(applied.restart_required)
        self.assertEqual(result.layers[0], base)

    def test_failed_application_reports_repair_without_claiming_rollback(self):
        result = replace(
            self.plan(artifacts=(artifact("demo"),)),
            target_json=plans.target_identity(plans.current_target()),
        )

        def run(command, **kwargs):
            if "install" in command:
                raise subprocess.CalledProcessError(1, command)

        with self.assertRaisesRegex(RuntimeError, "rollback is unavailable"):
            plans.execute_plan(result, result.digest, runner=run)

    def test_environment_lock_excludes_explicit_and_first_use_installers(self):
        acquired = threading.Event()
        with tempfile.TemporaryDirectory() as folder:

            def contender():
                with installation_lock(folder):
                    acquired.set()

            with installation_lock(folder):
                thread = threading.Thread(target=contender)
                thread.start()
                self.assertFalse(acquired.wait(0.05))
            thread.join(2)
            self.assertTrue(acquired.is_set())

    def test_processes_with_different_data_profiles_share_environment_lock(self):
        script = (
            "from ruinedfooocus.packaging.install_lock import installation_lock\n"
            "import sys\n"
            "try:\n"
            "    with installation_lock(sys.argv[1], timeout=.2): print('ACQUIRED')\n"
            "except RuntimeError: print('BLOCKED')\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "shared-environment"
            env = os.environ.copy() | {"RF_DATA_DIR": str(Path(folder) / "different-profile")}
            with installation_lock(target):
                result = subprocess.run(
                    [sys.executable, "-c", script, str(target)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=Path(__file__).resolve().parents[1],
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "BLOCKED", result.stderr)
            result = subprocess.run(
                [sys.executable, "-c", script, str(target)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=Path(__file__).resolve().parents[1],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "ACQUIRED", result.stderr)
