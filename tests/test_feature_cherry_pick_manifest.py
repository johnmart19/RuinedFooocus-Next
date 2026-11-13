"""Validate prerequisite closure ordering without modifying a repository."""

import ast
import os
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.validate_feature_cherry_picks import (
    cpu_test_code,
    cpu_test_patterns,
    dependency_order,
    load_manifest,
    module_exports,
    python_contracts,
    native_comfy_snapshot,
    native_cpu_test_files,
)


class FeatureDependencyTests(unittest.TestCase):
    def source_fixture(self, source, **modules):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        files = {"consumer": source, **modules}
        for name, text in files.items():
            path = root / "ruinedfooocus" / (name.replace(".", "/") + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    def test_local_import_contract_rejects_relative_root_and_package_missing_names(self):
        sources = (
            "from .absent import value\n",
            "from . import absent\n",
            "from ..outside import value\n",
            "from ruinedfooocus import absent\n",
            "from ruinedfooocus.folder import absent\n",
            "from .folder import absent\n",
        )
        for source in sources:
            root = self.source_fixture(source, **{"folder.__init__": "VALUE = 1\n"})
            for options in ({}, dict(allow_deferred=True, deferred_imports=[])):
                with self.subTest(source=source):
                    with self.assertRaises(ValueError):
                        python_contracts(root, **options)

    def test_local_import_contract_accepts_package_exports_and_namespace_submodules(self):
        root = self.source_fixture(
            "from .folder import VALUE\nfrom .namespace import child\n"
            "from ruinedfooocus import folder\n",
            **{"folder.__init__": "VALUE = 1\n", "namespace.child": "VALUE = 2\n"},
        )
        self.assertEqual(python_contracts(root), 3)
        (root / "ruinedfooocus/consumer.py").write_text(
            "from .folder import absent\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "has no absent"):
            python_contracts(root)

    def test_local_import_contract_checks_tools_namespace_exports(self):
        root = self.source_fixture("from tools import missing\n")
        tools = root / "tools"
        tools.mkdir()
        (tools / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "has no missing"):
            python_contracts(root)
        (root / "ruinedfooocus/consumer.py").write_text(
            "from tools.helper import VALUE\nfrom tools import helper\n", encoding="utf-8"
        )
        self.assertEqual(python_contracts(root), 2)

    def test_local_import_contract_respects_regular_modules_before_namespace_directories(self):
        root = self.source_fixture(
            "from .conflict import child\n",
            **{"conflict": "VALUE = 1\n", "conflict.child": "VALUE = 2\n"},
        )
        with self.assertRaisesRegex(ValueError, "has no child"):
            python_contracts(root)

    def test_local_import_contract_preserves_deferred_relative_diagnostics(self):
        root = self.source_fixture("def optional():\n    from .absent import value\n")
        with self.assertRaisesRegex(ValueError, "missing ruinedfooocus.absent"):
            python_contracts(root)
        deferred = []
        python_contracts(root, allow_deferred=True, deferred_imports=deferred)
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["scope"], "optional")
        self.assertEqual(deferred[0]["error"], "missing ruinedfooocus.absent")

    def test_source_contracts_exclude_nested_environments_and_node_modules(self):
        root = self.source_fixture("VALUE = 1\n")
        for relative in (
            "tools/format/.venv/vendor.py",
            "tools/nested/venv/vendor.py",
            "tools/format/node_modules/vendor.py",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("invalid Python dependency source", encoding="utf-8")
        self.assertEqual(python_contracts(root), 1)

    def test_resource_contract_checks_qualified_and_relative_helpers(self):
        sources = (
            "from ruinedfooocus.core import paths as bundled\nvalue = bundled.asset_path('absent.png')\n",
            "import ruinedfooocus.core.paths as bundled\nvalue = bundled.asset_path('absent.png')\n",
            "import ruinedfooocus.core.paths\nvalue = ruinedfooocus.core.paths.resource_path('absent.json')\n",
            "from .core import paths as bundled\nvalue = bundled.resource_path('absent.json')\n",
            "from .core.paths import asset_path as bundled\nvalue = bundled('absent.png')\n",
        )
        for source in sources:
            root = self.source_fixture(
                source,
                **{"core.paths": "def asset_path(name): pass\ndef resource_path(name): pass\n"},
            )
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "missing bundled resource"):
                    python_contracts(root)
                for relative in ("ui/assets/absent.png", "resources/absent.json"):
                    path = root / "ruinedfooocus" / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("bundled", encoding="utf-8")
                self.assertEqual(python_contracts(root), 2)

    def test_missing_eager_imports_fail_in_both_full_and_closure_checks(self):
        sources = (
            "from ruinedfooocus.absent import value\n",
            "import ruinedfooocus.absent\n",
            "class Eager:\n    from ruinedfooocus.absent import value\n",
            "if False:\n    from ruinedfooocus.absent import value\n",
            "try:\n    import ruinedfooocus.absent\nexcept ImportError:\n    pass\n",
        )
        for source in sources:
            root = self.source_fixture(source)
            for options in ({}, dict(allow_deferred=True, deferred_imports=[])):
                with self.subTest(source=source, options=options):
                    with self.assertRaisesRegex(ValueError, "missing ruinedfooocus.absent"):
                        python_contracts(root, **options)

    def test_deferred_imports_require_explicit_mode_and_keep_source_identity(self):
        root = self.source_fixture(
            "def optional():\n    from ruinedfooocus.absent import value\n"
            "async def inspect():\n    import ruinedfooocus.other\n"
        )
        with self.assertRaisesRegex(ValueError, "missing ruinedfooocus.absent"):
            python_contracts(root)
        deferred = []
        self.assertEqual(python_contracts(root, allow_deferred=True, deferred_imports=deferred), 1)
        self.assertEqual(
            deferred,
            [
                dict(
                    path="ruinedfooocus/consumer.py",
                    line=2,
                    scope="optional",
                    error="missing ruinedfooocus.absent",
                ),
                dict(
                    path="ruinedfooocus/consumer.py",
                    line=4,
                    scope="inspect",
                    error="missing ruinedfooocus.other",
                ),
            ],
        )
        with self.assertRaisesRegex(ValueError, "require a report collector"):
            python_contracts(root, allow_deferred=True)

    def test_named_exports_remain_checked_in_eager_and_deferred_scopes(self):
        root = self.source_fixture(
            "def optional():\n    from ruinedfooocus.present import absent\n",
            present="value = 1\n",
        )
        with self.assertRaisesRegex(ValueError, "ruinedfooocus.present has no absent"):
            python_contracts(root)
        deferred = []
        python_contracts(root, allow_deferred=True, deferred_imports=deferred)
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["error"], "ruinedfooocus.present has no absent")
        (root / "ruinedfooocus/consumer.py").write_text(
            "from ruinedfooocus.present import absent\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "ruinedfooocus.present has no absent"):
            python_contracts(root, allow_deferred=True, deferred_imports=[])

    def test_nested_class_bodies_keep_their_enclosing_execution_scope(self):
        root = self.source_fixture(
            "def optional():\n    class Deferred:\n        import ruinedfooocus.absent\n"
            "class Eager:\n    def method(self):\n        import ruinedfooocus.other\n"
        )
        deferred = []
        python_contracts(root, allow_deferred=True, deferred_imports=deferred)
        self.assertEqual(
            [(row["line"], row["scope"]) for row in deferred],
            [(3, "optional.Deferred"), (6, "Eager.method")],
        )

    def test_decorator_default_and_base_dependencies_remain_eager(self):
        suffixes = (
            "@dependency\ndef optional():\n    pass\n",
            "def optional(value=dependency):\n    pass\n",
            "class Eager(dependency):\n    pass\n",
        )
        for suffix in suffixes:
            root = self.source_fixture("from ruinedfooocus.absent import dependency\n" + suffix)
            with self.subTest(source=suffix):
                with self.assertRaisesRegex(ValueError, "missing ruinedfooocus.absent"):
                    python_contracts(root, allow_deferred=True, deferred_imports=[])

    def test_resolved_function_local_imports_need_no_deferred_exception(self):
        root = self.source_fixture(
            "def optional():\n    from ruinedfooocus.present import value\n",
            present="value = 1\n",
        )
        deferred = []
        self.assertEqual(python_contracts(root), 2)
        self.assertEqual(python_contracts(root, allow_deferred=True, deferred_imports=deferred), 2)
        self.assertEqual(deferred, [])

    def test_bundled_resource_contracts_remain_strict_in_deferred_functions(self):
        root = self.source_fixture(
            "from ruinedfooocus.core.paths import asset_path\n"
            "def optional():\n    return asset_path('missing.html')\n",
            **{"core.paths": "def asset_path(name): pass\n"},
        )
        with self.assertRaisesRegex(ValueError, "missing bundled resource"):
            python_contracts(root, allow_deferred=True, deferred_imports=[])

    def test_native_snapshot_uses_consumer_pin_not_mutable_vendor_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = root / "repositories/ComfyUI"
            vendor.mkdir(parents=True)
            subprocess.run(["git", "init", "--quiet", str(vendor)], check=True)
            (vendor / "native.py").write_text("PINNED = True\n")
            subprocess.run(["git", "-C", str(vendor), "add", "native.py"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(vendor),
                    "-c",
                    "user.name=Validation",
                    "-c",
                    "user.email=validation@localhost",
                    "-c",
                    "commit.gpgSign=false",
                    "commit",
                    "--quiet",
                    "-m",
                    "Native fixture",
                ],
                check=True,
            )
            revision = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
            ).strip()
            (vendor / "native.py").write_text("PINNED = False\n")
            (vendor / "private.txt").write_text("not in the native source")
            scratch = root / "scratch"
            bootstrap = scratch / "ruinedfooocus/app/bootstrap.py"
            bootstrap.parent.mkdir(parents=True)
            bootstrap.write_text(f"git_repos = [{{'name': 'ComfyUI', 'hash': '{revision}'}}]")
            snapshot, actual = native_comfy_snapshot(root, scratch)
            self.assertEqual(actual, revision)
            self.assertEqual((snapshot / "native.py").read_text(), "PINNED = True\n")
            self.assertFalse((snapshot / "private.txt").exists())
            bootstrap.unlink()
            snapshot, actual = native_comfy_snapshot(root, scratch, revision=revision)
            self.assertEqual(actual, revision)
            self.assertEqual((snapshot / "native.py").read_text(), "PINNED = True\n")
            with self.assertRaisesRegex(ValueError, "exact commit"):
                native_comfy_snapshot(root, scratch)
            with self.assertRaisesRegex(ValueError, "exact commit"):
                native_comfy_snapshot(root, scratch, revision="main")
            with self.assertRaisesRegex(RuntimeError, "Pinned ComfyUI source is unavailable"):
                native_comfy_snapshot(root, scratch, revision="0" * 40)
            tree_revision = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", revision + "^{tree}"], text=True
            ).strip()
            with self.assertRaisesRegex(ValueError, "requires an exact commit object"):
                native_comfy_snapshot(root, scratch, revision=tree_revision)
            bootstrap.write_text(f"git_repos = [{{'name': 'ComfyUI', 'hash': '{tree_revision}'}}]")
            with self.assertRaisesRegex(ValueError, "requires an exact commit object"):
                native_comfy_snapshot(root, scratch)
            bootstrap.write_text(f"git_repos = [{{'name': 'ComfyUI', 'hash': '{revision}'}}]")
            with self.assertRaisesRegex(ValueError, "does not match the consumer"):
                native_comfy_snapshot(root, scratch, revision="0" * 40)
            bootstrap.write_text("git_repos = [{'name': 'ComfyUI', 'hash': 'main'}]")
            with self.assertRaisesRegex(ValueError, "exact commit"):
                native_comfy_snapshot(root, scratch)
            bootstrap.write_text(f"git_repos = [{{'name': 'ComfyUI', 'hash': '{revision}'}}]")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(vendor),
                    "-c",
                    "user.name=Validation",
                    "-c",
                    "user.email=validation@localhost",
                    "-c",
                    "commit.gpgSign=false",
                    "commit",
                    "--quiet",
                    "-a",
                    "-m",
                    "Replacement source",
                ],
                check=True,
            )
            replacement = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
            ).strip()
            subprocess.run(["git", "-C", str(vendor), "replace", revision, replacement], check=True)
            snapshot, actual = native_comfy_snapshot(root, scratch)
            self.assertEqual(actual, revision)
            self.assertEqual((snapshot / "native.py").read_text(), "PINNED = True\n")

    def test_descriptor_native_pins_are_immutable_when_declared(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for field in ("native_comfy_revision", "native_gguf_revision"):
                entry = dict(id="native", subject="Native", **{field: "a" * 40})
                path.write_text(json.dumps(dict(base="upstream", features=[entry])))
                self.assertEqual(load_manifest(path)["features"], [entry])
                for value in ("main", "a" * 39, "A" * 40, None):
                    entry[field] = value
                    path.write_text(json.dumps(dict(base="upstream", features=[entry])))
                    with self.subTest(field=field, value=value):
                        with self.assertRaisesRegex(ValueError, "requires an exact commit pin"):
                            load_manifest(path)

    def test_cpu_patterns_never_discover_video_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            for name in (
                "test_image.py",
                "test_video_app.py",
                "test_ltx_native.py",
                "test_hunyuan_i2v.py",
            ):
                (root / "tests" / name).write_text("raise AssertionError('not imported')")
            self.assertEqual(cpu_test_patterns(root, ["test_*.py"]), ["test_image.py"])
            for pattern in ("test_video_*.py", "../test_image.py", "test_missing.py"):
                with self.assertRaises(ValueError):
                    cpu_test_patterns(root, [pattern])

    def test_geometry_and_color_source_supplements_only_follow_selected_cpu_suites(self):
        post = "comfy_extras/nodes_post_processing.py"
        geometry = "comfy_extras/color_util.py"
        finishing = ["comfy_extras/nodes_text_overlay.py", "comfy_extras/nodes_morphology.py"]
        for name in ("test_image_geometry_tools.py", "test_image_crop_to_mask.py"):
            self.assertEqual(native_cpu_test_files([name]), [post, geometry])
        for name in (
            "test_image_adjustment_integration.py",
            "test_image_plan_callers.py",
            "test_image_residual_ui_api.py",
        ):
            self.assertEqual(native_cpu_test_files([name]), [post, geometry, *finishing])
        for name in ("test_image_color_tools.py", "test_image_color_finishing.py"):
            self.assertEqual(native_cpu_test_files([name]), [post, *finishing])
        self.assertEqual(native_cpu_test_files(["test_image_filters.py"]), [post])
        self.assertEqual(native_cpu_test_files(["test_image_geometry.py", "test_fixture.py"]), [])
        self.assertEqual(
            native_cpu_test_files(["test_moge_geometry_outputs.py", "test_mediapipe_face_mask.py"]),
            [],
        )

    def test_source_contracts_require_bundled_resources_with_their_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = root / "ruinedfooocus/core/paths.py"
            paths.parent.mkdir(parents=True)
            paths.write_text("def asset_path(name): pass\ndef resource_path(*parts): pass\n")
            consumer = root / "ruinedfooocus/consumer.py"
            consumer.write_text(
                "from ruinedfooocus.core.paths import asset_path as asset, resource_path\n"
                "template = asset('runner.html')\nlanguages = resource_path('language')\n"
            )
            with self.assertRaisesRegex(ValueError, "missing bundled resource"):
                python_contracts(root)
            assets = root / "ruinedfooocus/ui/assets"
            assets.mkdir(parents=True)
            (assets / "runner.html").write_text("<html></html>")
            (root / "ruinedfooocus/resources/language").mkdir(parents=True)
            self.assertEqual(python_contracts(root), 2)

    def test_split_registrations_preserve_real_dependencies_and_legacy_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features").mkdir()
            entries = [
                dict(id="base", subject="Base", requires=[]),
                dict(id="image", subject="Image", requires=["base"]),
            ]
            for entry in entries:
                (root / "features" / (entry["id"] + ".json")).write_text(json.dumps(entry))
            path = root / "manifest.json"
            path.write_text(json.dumps(dict(base="upstream", feature_directory="features")))
            self.assertEqual(load_manifest(path)["features"], entries)
            path.write_text(json.dumps(dict(base="upstream", features=entries)))
            self.assertEqual(load_manifest(path)["features"], entries)

    def test_registration_rejects_duplicate_identity_missing_dependency_and_wrong_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features").mkdir()
            path = root / "manifest.json"
            path.write_text(json.dumps(dict(base="upstream", feature_directory="features")))
            entry = dict(id="base", subject="Base", requires=[])
            (root / "features/base.json").write_text(json.dumps(entry))
            extra = root / "features/extra.json"
            for invalid in (
                entry,
                dict(id="extra", subject="Base", requires=[]),
                dict(id="extra", subject="Extra", requires=["absent"]),
            ):
                extra.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_registration_directory_must_stay_inside_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(dict(base="upstream", feature_directory="..")))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_closure_omits_unrelated_features_and_orders_shared_dependencies_once(self):
        features = [
            dict(id="base"),
            dict(id="image", requires=["base"]),
            dict(id="chat", requires=["base"]),
            dict(id="api", requires=["image", "chat"]),
            dict(id="video", requires=["image"]),
        ]
        self.assertEqual(dependency_order(features, ["api"]), ["base", "image", "chat", "api"])

    def test_missing_dependency_and_cycle_are_rejected(self):
        with self.assertRaises(ValueError):
            dependency_order([dict(id="api", requires=["absent"])], ["api"])
        with self.assertRaises(ValueError):
            dependency_order([dict(id="a", requires=["b"]), dict(id="b", requires=["a"])], ["a"])

    def test_named_import_check_excludes_nested_and_comprehension_bindings(self):
        tree = ast.parse("""import os as operating_system
if True:
    PUBLIC = 1
def outer():
    local_only = 1
    def nested():
        pass
class Exported:
    class_member = 1
values = [temporary for temporary in []]
""")
        self.assertEqual(
            module_exports(tree), {"operating_system", "PUBLIC", "outer", "Exported", "values"}
        )

    def test_empty_test_discovery_cannot_pass(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        with (
            patch.object(unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()),
            patch.dict(sys.modules, {"torch": fake}),
            patch.dict(os.environ),
            patch.object(sys, "argv", ["cpu-check"]),
        ):
            with self.assertRaisesRegex(ValueError, "No CPU tests discovered"):
                exec(cpu_test_code(["test_missing.py"]), {})

    def test_cpu_checks_reject_an_initialized_gpu_before_discovery(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with (
            patch.object(unittest.defaultTestLoader, "discover") as discover,
            patch.dict(sys.modules, {"torch": fake}),
            patch.dict(os.environ),
            patch.object(sys, "argv", ["cpu-check"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "must not initialize a GPU"):
                exec(cpu_test_code(["test_example.py"]), {})
            discover.assert_not_called()

    def test_cpu_checks_prohibit_hardware_queries_during_discovery(self):
        for name in (
            "init",
            "is_available",
            "device_count",
            "current_device",
            "get_device_properties",
            "get_device_name",
            "get_device_capability",
            "mem_get_info",
        ):
            with self.subTest(query=name):
                fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))

                def discover(*args, **kwargs):
                    return getattr(fake.cuda, name)()

                with (
                    patch.object(unittest.defaultTestLoader, "discover", side_effect=discover),
                    patch.dict(sys.modules, {"torch": fake}),
                    patch.dict(os.environ),
                    patch.object(sys, "argv", ["cpu-check"]),
                ):
                    with self.assertRaisesRegex(RuntimeError, "prohibited in CPU feature checks"):
                        exec(cpu_test_code(["test_example.py"]), {})


if __name__ == "__main__":
    unittest.main()
