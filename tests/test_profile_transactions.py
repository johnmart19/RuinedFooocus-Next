"""Private profiles only: persistence failures, startup precedence and CPU policy."""

import argparse
import ast
import base64
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.profile_store import (
    profile_folder,
    profile_name,
    read_profile,
    read_profile_data,
    read_profile_pair,
    selected_profile,
    use_next_launch,
)
from ruinedfooocus.core.settings import SettingsManager, prospective_profile
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.backends.comfy.comfy_args import add_comfy_arguments
from ruinedfooocus.backends.comfy.image_compute import (
    apply_compute,
    finalize_compute,
    explicit_options,
    load_compute,
)

ROOT = Path(__file__).resolve().parents[1]


class ProfileTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(
            os.environ, RF_DATA_DIR=self.temp.name, RF_OFFLINE="0", TORCH_PLATFORM=""
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_known_invalid_fields_do_not_rewrite_the_source(self):
        path = profile_folder() / "settings.json"
        path.parent.mkdir()
        for key, value in (
            ("image_number", True),
            ("startup_port", False),
            ("local_model_metadata", "false"),
            ("image_number_max", 0),
        ):
            raw = json.dumps({key: value})
            path.write_text(raw, encoding="utf-8-sig")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                SettingsManager()
            self.assertEqual(path.read_text(encoding="utf-8-sig"), raw)

    def test_raw_profile_data_does_not_replace_application_validation(self):
        path = profile_folder() / "settings.json"
        for feature_values in ({"grid_columns": 0}, {"image_number": -1}):
            with self.subTest(values=feature_values):
                values = {"startup_offline": False, **feature_values}
                save_json(path, values)
                original = path.read_bytes()
                self.assertEqual(read_profile_data(path), values)
                with self.assertRaises(ValueError):
                    read_profile(path)
                self.assertEqual(path.read_bytes(), original)

    def test_large_integer_seed_survives_loading_default_migration_and_form_validation(self):
        seed = 2**63 - 1
        path = profile_folder() / "settings.json"
        save_json(path, {"seed": seed})
        manager = SettingsManager(defaults={"prompt": ""})
        self.assertEqual(manager.default_settings["seed"], seed)
        self.assertEqual(json.loads(path.read_text())["seed"], seed)
        self.assertEqual(validate_settings({"seed": seed})["seed"], seed)
        save_json(path, {"seed": str(seed)})
        self.assertEqual(read_profile(path)["seed"], seed)
        self.assertEqual(validate_settings({"seed": str(seed)})["seed"], seed)
        for invalid in (f"{seed}.5", "1.5", "NaN", "Infinity", "-Infinity", "1e999999999"):
            with self.subTest(seed=invalid), self.assertRaisesRegex(ValueError, "seed"):
                validate_settings({"seed": invalid})

    def test_bom_objects_and_unknown_values_without_backend_imports(self):
        path = profile_folder() / "settings.json"
        path.parent.mkdir()
        data = {
            "image_number": 0,
            "seed_random": False,
            "startup_port": None,
            "custom": {"v": ""},
            "ui_language": "sv",
        }
        path.write_text(json.dumps(data), encoding="utf-8-sig")
        with patch.dict(
            sys.modules, {"torch": None, "comfy": None, "ruinedfooocus.app.arguments": None}
        ):
            self.assertEqual(read_profile(path), data)
        path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            read_profile(path)

    def test_failure_on_either_document_retains_disk_memory_identity_and_notifications(self):
        manager = SettingsManager("old", defaults={"prompt": "old"}, on_change=Mock())
        folder = profile_folder("copy")
        folder.mkdir()
        originals = {
            "settings.json": b'\xef\xbb\xbf{"prompt":"previous"}',
            "paths.json": b'{"custom":"old"}',
        }
        for name, content in originals.items():
            (folder / name).write_bytes(content)
        for failing in ("settings.json", "paths.json"):

            def write(path, data):
                if Path(path).name == failing:
                    raise OSError("private injected failure")
                return save_json(path, data)

            with (
                self.subTest(failing=failing),
                patch("ruinedfooocus.core.profile_store.save_json", side_effect=write),
            ):
                with self.assertRaises(OSError):
                    manager.save_profile({"prompt": "new"}, {"custom": "new"}, profile="copy")
            self.assertEqual(manager.name, "old")
            self.assertEqual(manager.default_settings, {"prompt": "old"})
            manager._on_change.assert_not_called()
            for name, content in originals.items():
                self.assertEqual((folder / name).read_bytes(), content)
            self.assertFalse((folder / ".profile-save.json").exists())

    def test_first_save_failure_removes_new_partial_documents(self):
        manager = SettingsManager()

        def write(path, data):
            if Path(path).name == "paths.json":
                raise OSError("second document")
            save_json(path, data)

        with patch("ruinedfooocus.core.profile_store.save_json", side_effect=write):
            with self.assertRaises(OSError):
                manager.save_profile({"prompt": "new"}, {}, profile="new")
        self.assertFalse((profile_folder("new") / "settings.json").exists())
        self.assertFalse((profile_folder("new") / "paths.json").exists())

    def test_restart_recovers_interrupted_save_before_reading(self):
        folder = profile_folder("restart")
        folder.mkdir(parents=True)
        original = b'\xef\xbb\xbf{"prompt":"old","ui_language":"sv"}'
        save_json(
            folder / ".profile-save.json",
            {
                "version": 1,
                "before": {
                    "settings.json": base64.b64encode(original).decode(),
                    "paths.json": None,
                },
            },
        )
        save_json(folder / "settings.json", {"prompt": "uncommitted"})
        save_json(folder / "paths.json", {"path_outputs": "uncommitted"})
        self.assertEqual(SettingsManager("restart").default_settings["prompt"], "old")
        self.assertEqual((folder / "settings.json").read_bytes(), original)
        self.assertFalse((folder / "paths.json").exists())
        self.assertFalse((folder / ".profile-save.json").exists())

    def test_success_publishes_both_documents_before_notification(self):
        manager = SettingsManager()
        published = []
        manager._on_change = lambda: published.append(
            (manager.name, read_profile(profile_folder("new") / "settings.json"), published[0])
        )
        manager.save_profile(
            {"ui_language": "sv", "prompt": ""},
            {"custom_path": "keep"},
            profile="new",
            on_publish=lambda: published.append("paths published"),
        )
        self.assertEqual(
            published[1], ("new", {"ui_language": "sv", "prompt": ""}, "paths published")
        )
        self.assertEqual(SettingsManager("new").default_settings, manager.default_settings)
        self.assertEqual(
            json.loads((profile_folder("new") / "paths.json").read_text()), {"custom_path": "keep"}
        )

    def test_form_routing_preserves_preferences_unknown_fields_zero_false_and_blank(self):
        current, paths = {"custom": 1}, {"custom_path": "old"}
        values = {
            "ui_settings_name": "new",
            "ui_language": "sv",
            "seed": 0,
            "save_metadata": False,
            "meta_comment": "",
            "path_outputs": "new",
            "optional": None,
        }
        settings, result_paths = prospective_profile(current, paths, values, {"path_outputs"})
        self.assertEqual(
            settings,
            {
                "custom": 1,
                "ui_language": "sv",
                "seed": 0,
                "save_metadata": False,
                "meta_comment": "",
            },
        )
        self.assertEqual(result_paths, {"custom_path": "old", "path_outputs": "new"})
        self.assertEqual(current, {"custom": 1})

    def test_profile_names_and_next_launch_remain_distinct(self):
        for invalid in ("..", "../other", "NUL", "con.txt", "name.", " name", "name "):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                profile_name(invalid)
        with self.assertRaisesRegex(ValueError, "Save this profile"):
            use_next_launch("missing")
        manager = SettingsManager("copy", defaults={"prompt": "saved"})
        use_next_launch("copy")
        self.assertEqual(selected_profile(), "copy")
        self.assertIsNone(selected_profile("default"))
        self.assertEqual(selected_profile("explicit"), "explicit")
        self.assertEqual(manager.name, "copy")

    def test_saved_startup_preferences_apply_to_parser_and_updater_with_cli_priority(self):
        from ruinedfooocus.packaging.startup_options import offline_startup, skip_dependency_install

        manager = SettingsManager(
            "saved",
            defaults={
                "startup_browser": False,
                "startup_port": 8123,
                "startup_offline": True,
                "startup_skip_dependency_install": True,
                "ui_language": "sv",
            },
        )
        use_next_launch(manager.name)
        for argv, expected_port, expected_locale in (
            ([], 8123, "sv"),
            (["--port=9000", "--language=en"], 9000, "en"),
        ):
            with (
                patch.object(sys, "argv", ["arguments.py", *argv]),
                patch.dict(sys.modules, {"torch": None, "comfy": None}),
            ):
                args = runpy.run_path(str(ROOT / "ruinedfooocus/app/arguments.py"))["args"]
            self.assertEqual(
                (args.settings, args.port, args.language), ("saved", expected_port, expected_locale)
            )
            self.assertTrue(args.nobrowser and args.offline and args.skip_dependency_install)
        self.assertTrue(offline_startup([]))
        self.assertTrue(skip_dependency_install([]))
        self.assertTrue(offline_startup(["--settings=default", "--of"]))
        self.assertTrue(skip_dependency_install(["--settings=default", "--sk"]))
        self.assertFalse(offline_startup(["--settings=default"]))
        self.assertFalse(offline_startup(["--sett=default"]))
        self.assertFalse(offline_startup(["--se=default"]))

    def test_connection_preferences_validate_without_devices_and_cli_wins(self):
        values = {"startup_api": True, "startup_mcp": True, "startup_listen": "127.0.0.1"}
        manager = SettingsManager(defaults=values)
        for argv, expected in (([], "127.0.0.1"), (["--listen=::1"], "::1")):
            with (
                patch.object(sys, "argv", ["arguments.py", *argv]),
                patch.dict(sys.modules, {"torch": None, "comfy": None}),
            ):
                args = runpy.run_path(str(ROOT / "ruinedfooocus/app/arguments.py"))["args"]
            self.assertTrue(args.api and args.mcp)
            self.assertEqual(args.listen, expected)
            self.assertIsNone(args.auth)
            self.assertIsNone(args.api_key)
        for invalid in (False, "https://example.com", "0.0.0.0:8123", "localhost/../../"):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "startup_listen"):
                validate_settings({"startup_listen": invalid})
        self.assertIsNone(validate_settings({"startup_listen": ""})["startup_listen"])
        self.assertEqual(validate_settings({"startup_listen": "::"})["startup_listen"], "::")

    def test_saved_connections_retain_lan_auth_api_key_and_guest_guards(self):
        from ruinedfooocus.integrations.web_access import validate_web_access

        manager = SettingsManager(
            defaults={"startup_api": True, "startup_mcp": True, "startup_listen": "0.0.0.0"}
        )
        with patch.object(sys, "argv", ["arguments.py"]):
            args = runpy.run_path(str(ROOT / "ruinedfooocus/app/arguments.py"))["args"]
        with patch.dict(os.environ, {"RF_UI_AUTH": "", "RF_API_KEY": ""}):
            with self.assertRaisesRegex(SystemExit, "requires a UI login"):
                validate_web_access(args)
            args.auth = "private-owner/private-password"
            validate_web_access(args)
            tree = ast.parse(
                (ROOT / "ruinedfooocus/integrations/openai_api.py").read_text(encoding="utf-8")
            )
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "validate_options"
            )
            scope = {"os": os}
            exec(
                compile(ast.Module(body=[function], type_ignores=[]), "api_options", "exec"), scope
            )
            with self.assertRaisesRegex(ValueError, "requires --api-key"):
                scope["validate_options"](args)
            args.api_key = "private-api-key"
            self.assertEqual(scope["validate_options"](args), "private-api-key")
            args.guest = True
            with self.assertRaisesRegex(SystemExit, "Guest mode"):
                validate_web_access(args)
        self.assertNotIn("private", (profile_folder() / "settings.json").read_text())

    def test_early_reader_rejects_invalid_compute_fields_without_backends(self):
        path = profile_folder() / "settings.json"
        for key, value in (
            ("image_device", "Unknown"),
            ("image_reserve_vram", True),
            ("startup_api", "true"),
        ):
            save_json(path, {key: value})
            with patch.dict(
                sys.modules,
                {"torch": None, "comfy": None, "ruinedfooocus.backends.comfy.image_compute": None},
            ):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    read_profile(path)

    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--settings")
        parser.add_argument("--preview-method", default="auto")
        add_comfy_arguments(parser)
        return parser

    def test_cpu_environment_and_later_fallback_suppress_only_saved_precision(self):
        parser = self.parser()
        for argv, environment in (([], "cpu"), ([], ""), (["--fp16-unet"], "")):
            with (
                self.subTest(argv=argv, env=environment),
                patch.dict(os.environ, TORCH_PLATFORM=environment),
            ):
                args = parser.parse_args(argv)
                apply_compute(
                    args,
                    {"image_unet_precision": "BF16", "image_text_precision": "FP16"},
                    explicit_options(parser, argv),
                )
                finalize_compute(args, cpu=True)
                self.assertFalse(args.bf16_unet)
                self.assertFalse(args.fp16_text_enc)
                self.assertEqual(args.fp16_unet, bool(argv))

    def test_cpu_preference_respects_environment_and_explicit_repairs(self):
        for options, environment, expected in (
            ([], "", True),
            (["--rocm10"], "", False),
            (["--legacy"], "", False),
            (["--cuda-nightly"], "", False),
            (["--oneapi-device-selector"], "", False),
            ([], "cu128", False),
        ):
            with (
                self.subTest(options=options, env=environment),
                patch.dict(os.environ, TORCH_PLATFORM=environment),
            ):
                args = self.parser().parse_args([])
                apply_compute(args, {"image_device": "CPU"}, options)
                self.assertEqual(args.cpu, expected)

    def test_new_native_controls_keep_opposing_cli_and_auto(self):
        parser = self.parser()
        argv = ["--disable-mmap", "--disable-cuda-malloc"]
        args = parser.parse_args(argv)
        apply_compute(
            args,
            {
                "image_mmap": "Enabled",
                "image_allocator": "Async",
                "image_deterministic": "Enabled",
                "image_smart_memory": "Disabled",
                "image_pinned_memory": "Disabled",
            },
            explicit_options(parser, argv),
        )
        self.assertTrue(args.disable_mmap and args.disable_cuda_malloc)
        self.assertFalse(args.mmap_torch_files or args.cuda_malloc)
        self.assertTrue(
            args.deterministic and args.disable_smart_memory and args.disable_pinned_memory
        )
        self.assertIsNone(validate_settings({"startup_port": None})["startup_port"])

    def test_settings_preset_discovery_uses_read_only_instance_api(self):
        source = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "obp_choices"
        )
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "ui_settings", "exec"), scope)
        names = scope["obp_choices"]()
        self.assertIn("Standard", names)
        self.assertEqual(names, sorted(names, key=str.casefold))
        self.assertFalse((Path(self.temp.name) / "random_prompt").exists())

    def test_production_save_callbacks_keep_save_as_and_next_launch_separate(self):
        source = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        names = {"save_clicked", "save_as_clicked", "use_next_launch_clicked"}
        functions = [
            node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        manager = SettingsManager("current", defaults={"custom": "preserve"})
        paths = SimpleNamespace(
            paths={"custom_path": "preserve"},
            DEFAULT_PATHS={"path_outputs": "old"},
            name="current",
            set_settings_path=Mock(),
        )
        scope = {
            "state": {
                "setting_name": [
                    "ui_settings_name",
                    "ui_language",
                    "local_model_metadata",
                    "path_outputs",
                    "seed",
                ]
            },
            "settings": manager,
            "path_manager": paths,
            "validate_settings": validate_settings,
            "gr": SimpleNamespace(Error=ValueError, Info=Mock()),
            "models": SimpleNamespace(offline=False),
            "performance_settings": SimpleNamespace(
                model_recommendations=SimpleNamespace(offline=False)
            ),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), "ui_settings", "exec"), scope)
        launch = SimpleNamespace(
            args=SimpleNamespace(settings="current", _runtime_profile="current", offline=False)
        )
        values = ("copy", "sv", True, "private-output", 0)
        original_base = {
            "profile": "current",
            "settings": dict(manager.default_settings),
            "paths": dict(paths.paths),
        }
        with patch.dict(sys.modules, {"ruinedfooocus.app.arguments": launch}):
            scope["save_clicked"](*values)
            self.assertEqual(manager.name, "current")
            self.assertFalse((profile_folder("copy") / "settings.json").exists())
            scope["save_as_clicked"](*values)
            self.assertEqual(manager.name, "copy")
            # A second browser can keep its own Save target after this browser
            # used Save as. The global manager name must not redirect its save.
            scope["save_clicked"](*values, profile="current")
            self.assertEqual(manager.name, "current")
            self.assertTrue((profile_folder("copy") / "settings.json").exists())
            self.assertIsNone(selected_profile())
            result = scope["use_next_launch_clicked"]("copy")
            self.assertIn("explicit --settings", result)
            self.assertEqual(selected_profile(), "copy")
            # Another browser changes the global manager to an unrelated profile.
            # Save merges this browser's own latest disk pair, not that manager.
            other = SettingsManager("other", defaults={"custom": "other", "only_other": True})
            scope["settings"] = other
            save_json(
                profile_folder("current") / "settings.json",
                {"custom": "latest own", "only_own": True},
            )
            message, snapshot = scope["save_clicked"](
                *values, profile="current", base=original_base, return_snapshot=True
            )
            own, own_paths = read_profile_pair("current")
            self.assertEqual(own["custom"], "latest own")
            self.assertTrue(own["only_own"])
            self.assertNotIn("only_other", own)
            self.assertEqual(own_paths["custom_path"], "preserve")
            self.assertEqual(snapshot["profile"], "current")
            self.assertEqual(original_base["settings"]["custom"], "preserve")
            before = read_profile_pair("current")
            with patch.object(other, "save_profile", side_effect=OSError("write denied")):
                with self.assertRaisesRegex(ValueError, "write denied"):
                    scope["save_clicked"](
                        *values, profile="current", base=original_base, return_snapshot=True
                    )
            self.assertEqual(read_profile_pair("current"), before)
        self.assertEqual(manager.default_settings["ui_language"], "sv")
        self.assertEqual(paths.paths, {"custom_path": "preserve", "path_outputs": "private-output"})

    def test_explicit_cpu_provenance_includes_precision_skipped_at_load(self):
        parser = self.parser()
        save_json(profile_folder() / "settings.json", {"image_unet_precision": "FP16"})
        args = parser.parse_args(["--cpu"])
        load_compute(args, parser, ["--cpu"])
        finalize_compute(args)
        self.assertEqual(args._compute_status["requested"], {"image_unet_precision": "FP16"})
        self.assertEqual(args._compute_status["cli"], ["--cpu"])
        self.assertEqual(args._compute_status["cpu_suppressed"], ["fp16_unet"])
        self.assertTrue(args._compute_status["effective"]["cpu"])
        self.assertFalse(args._compute_status["effective"]["fp16_unet"])

    def test_status_excludes_credentials_and_records_suppressed_flags(self):
        parser = self.parser()
        path = profile_folder() / "settings.json"
        save_json(path, {"image_unet_precision": "FP16", "image_allocator": "Async"})
        args = parser.parse_args([])
        args.auth, args.api_key = "private/password", "private-secret"
        load_compute(args, parser, [])
        finalize_compute(args, cpu=True)
        status = json.dumps(args._compute_status)
        self.assertNotIn("private", status)
        self.assertEqual(args._compute_status["cpu_suppressed"], ["fp16_unet", "cuda_malloc"])
        self.assertFalse(args._compute_status["effective"]["cuda_malloc"])


if __name__ == "__main__":
    unittest.main()
