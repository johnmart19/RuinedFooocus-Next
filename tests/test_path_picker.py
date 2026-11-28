"""Owner folder selection and persistence, with native dialogs always mocked."""

import ast
import asyncio
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
from starlette.requests import Request

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.features.prompting.wildcards import wildcard_roots
from ruinedfooocus.ui import path_picker as picker
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]
NATIVE_CHOOSE = picker.native_choose


class PathPickerTests(unittest.TestCase):
    def test_path_help_expands_base_after_message_registration(self):
        base = Path("owned root Ω")
        text = picker.path_help("en", True, base)
        self.assertIn(str(base), text)
        self.assertNotIn("{base}", text)
        self.assertNotIn("__i18n__", text)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-picker-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        env = patch.dict(os.environ, RF_DATA_DIR=str(self.root), RF_UI_AUTH="")
        env.start()
        self.addCleanup(env.stop)
        available = patch.object(picker, "native_available", return_value=False)
        self.available = available.start()
        self.addCleanup(available.stop)
        native = patch.object(
            picker, "native_choose", side_effect=AssertionError("real native dialog")
        )
        self.native = native.start()
        self.addCleanup(native.stop)
        self.context = SimpleNamespace(auth=None, guest=False)

    def folder(self, name):
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def request(self, *, remote=False, username=None, origin=True, extra=()):
        host = "server.example" if remote else "127.0.0.1"
        headers = [(b"host", (host + ":7860").encode())]
        if origin:
            headers.append((b"origin", ("http://" + host + ":7860").encode()))
        headers.extend(extra)
        raw = Request(
            {
                "type": "http",
                "scheme": "http",
                "method": "POST",
                "root_path": "",
                "path": "/gradio_api/queue/join",
                "query_string": b"",
                "headers": headers,
                "server": (host, 7860),
                "client": ("192.0.2.5" if remote else "127.0.0.1", 12345),
            }
        )
        return gr.Request(raw, username=username)

    def build(self, key="path_checkpoints", value=None, context=None):
        with gr.Blocks() as app:
            field = picker.create_path_control(
                key, "Folders", value or [str(self.root)], context=context or self.context
            )
        return app, field

    def invoke(self, app, button, inputs, *, request=None, state=None):
        from ruinedfooocus.core.translation_manager import catalogue

        def display_text(value):
            if hasattr(value, "key"):
                return catalogue().translate(value.key.removeprefix("ruinedfooocus."), locale="en")
            if isinstance(value, str) and value.startswith("__i18n__"):
                return display_text(SimpleNamespace(key=json.loads(value[8:])["key"]))
            return value

        target = next(
            component
            for component in app.blocks.values()
            if isinstance(component, gr.Button) and display_text(component.value) == button
        )
        index = next(
            index
            for index, function in app.fns.items()
            if (target._id, "click") in function.targets
        )
        result = asyncio.run(
            app.process_api(
                index,
                inputs=inputs,
                state=state or SessionState(app),
                request=request or self.request(),
            )
        )["data"]
        return [display_text(value) for value in result]

    def test_local_owner_request_and_authentication_are_checked_on_server(self):
        self.assertTrue(picker.authorize(self.request(), self.context))
        for request, context in (
            (None, self.context),
            (gr.Request(username="owner"), SimpleNamespace(auth="owner/secret", guest=False)),
            (self.request(), SimpleNamespace(auth=None, guest=True)),
            (self.request(remote=True), self.context),
            (
                self.request(remote=True, username="other"),
                SimpleNamespace(auth="owner/secret", guest=False),
            ),
        ):
            with self.subTest(request=request), self.assertRaises(PermissionError):
                picker.authorize(request, context)
        authenticated = SimpleNamespace(auth="owner/secret", guest=False)
        self.assertFalse(
            picker.authorize(self.request(remote=True, username="owner"), authenticated)
        )
        self.assertFalse(
            picker.authorize(
                self.request(username="owner", extra=[(b"x-forwarded-for", b"192.0.2.1")]),
                authenticated,
            )
        )
        self.assertFalse(picker.authorize(self.request(origin=False), self.context))
        self.assertTrue(
            picker.authorize(
                self.request(username="власник"),
                SimpleNamespace(auth="власник/secret", guest=False),
            )
        )

    def test_cross_origin_requests_are_rejected_before_filesystem_access(self):
        request = self.request(extra=[(b"sec-fetch-site", b"cross-site")])
        with patch.object(picker.os, "scandir") as scan:
            with self.assertRaises(PermissionError):
                picker.authorize(request, self.context)
            scan.assert_not_called()

    def test_remote_and_headless_select_open_lazy_host_browser_without_native_dialog(self):
        self.folder("child Ω")
        for context, request in (
            (self.context, self.request()),
            (
                SimpleNamespace(auth="owner/secret", guest=False),
                self.request(remote=True, username="owner"),
            ),
        ):
            app, _ = self.build(context=context)
            result = self.invoke(app, "Select folder", [str(self.root), "0"], request=request)
            self.assertTrue(result[3]["visible"])
            self.assertEqual(Path(result[4]), self.root)
            self.assertIn(
                ("child Ω", str(self.root / "child Ω")),
                [tuple(item) for item in result[5]["choices"]],
            )
        self.native.assert_not_called()

    def test_guest_and_unauthed_remote_events_cannot_list_directories(self):
        for context, request in (
            (SimpleNamespace(auth=None, guest=True), self.request()),
            (self.context, self.request(remote=True)),
        ):
            app, _ = self.build(context=context)
            with patch.object(picker, "directory_choices") as listing:
                with self.assertRaises(gr.Error):
                    self.invoke(app, "Select folder", [str(self.root), "0"], request=request)
                listing.assert_not_called()
        self.native.assert_not_called()

    def test_native_choose_updates_draft_and_cancel_preserves_it(self):
        chosen = self.folder("native café")
        self.available.return_value = True
        self.native.side_effect = None
        self.native.return_value = str(chosen)
        app, field = self.build()
        result = self.invoke(app, "Open host dialog", [str(self.root), "0", None])
        self.assertEqual(result[0], str(chosen))
        self.assertFalse(result[3]["visible"])
        self.assertIn("restart", result[2])
        self.native.assert_called_once_with(self.root)
        self.assertEqual(field.value, str(self.root))
        self.native.return_value = None
        result = self.invoke(app, "Open host dialog", [str(self.root), "0", None])
        self.assertEqual(result[0], gr.skip())
        self.assertIn("cancelled", result[2])

    def test_remote_owner_never_opens_a_native_dialog_even_when_desktop_exists(self):
        self.available.return_value = True
        app, _ = self.build(context=SimpleNamespace(auth="owner/secret", guest=False))
        result = self.invoke(
            app,
            "Open host dialog",
            [str(self.root), "0", None],
            request=self.request(remote=True, username="owner"),
        )
        self.assertTrue(result[3]["visible"])
        self.native.assert_not_called()

    def test_unavailable_native_dialog_falls_back_to_host_browser(self):
        self.available.return_value = True
        self.native.side_effect = OSError("no desktop")
        app, _ = self.build()
        result = self.invoke(app, "Open host dialog", [str(self.root), "0", None])
        self.assertTrue(result[3]["visible"])
        self.assertIn("Native dialog unavailable", result[2])

    def test_explicit_host_browser_is_available_even_with_a_native_desktop(self):
        self.available.return_value = True
        app, _ = self.build()
        result = self.invoke(app, "Select folder", [str(self.root), "0"])
        self.assertTrue(result[3]["visible"])
        self.native.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows native command contract")
    def test_windows_native_command_is_bounded_and_path_is_data(self):
        initial = self.folder("spaces Ω ' quote")
        for code, output in ((0, str(initial)), (0, "")):
            with (
                self.subTest(code=code, output=output),
                patch.object(
                    picker.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=code, stdout=output),
                ) as run,
                patch.object(picker.shutil, "which", return_value="powershell.exe"),
            ):
                self.assertEqual(NATIVE_CHOOSE(initial), output or None)
                command = run.call_args.args[0]
                self.assertIn("-STA", command)
                self.assertNotIn(str(initial), command[-1])
                self.assertIn("ShowNewFolderButton = $false", command[-1])
                self.assertEqual(
                    run.call_args.kwargs["env"]["RF_FOLDER_PICKER_START"], str(initial)
                )
                self.assertEqual(run.call_args.kwargs["timeout"], 120)
                self.assertEqual(
                    run.call_args.kwargs["creationflags"], picker.subprocess.CREATE_NO_WINDOW
                )

        with (
            patch.object(
                picker.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")
            ),
            patch.object(picker.shutil, "which", return_value="powershell.exe"),
        ):
            with self.assertRaises(OSError):
                NATIVE_CHOOSE(initial)

    def test_browser_navigation_use_and_cancel_change_only_draft(self):
        child = self.folder("child Ω")
        app, field = self.build()
        state = SessionState(app)
        self.invoke(app, "Select folder", [str(self.root), "0"], state=state)
        result = self.invoke(app, "Open subfolder", [str(child)], state=state)
        self.assertEqual(Path(result[0]), child)
        result = self.invoke(app, "Parent folder", [str(child)], state=state)
        self.assertEqual(Path(result[0]), self.root)
        result = self.invoke(
            app, "Use folder", [str(self.root), "0", str(child), None], state=state
        )
        self.assertEqual(result[0], str(child))
        result = self.invoke(app, "Cancel", [], state=state)
        self.assertFalse(result[0]["visible"])
        self.assertEqual(field.value, str(self.root))
        self.assertFalse((self.root / "settings").exists())

    def test_multi_root_add_replace_remove_and_reorder_keep_priority(self):
        first, second, third = (self.folder(name) for name in ("one", "two space", "три"))
        initial = "\n".join((str(first), str(second)))
        value, index = picker.apply_choice(
            "path_checkpoints", initial, "0", str(third), append=True
        )
        self.assertEqual(picker.path_lines(value), [str(first), str(second), str(third)])
        self.assertEqual(index, "2")
        value, index = picker.edit_order(value, index, "up")
        self.assertEqual(picker.path_lines(value), [str(first), str(third), str(second)])
        value, index = picker.edit_order(value, "0", "remove")
        self.assertEqual(picker.path_lines(value), [str(third), str(second)])
        value, index = picker.apply_choice("path_checkpoints", value, "1", str(first))
        self.assertEqual(picker.path_lines(value), [str(third), str(first)])
        self.assertEqual(picker.edit_order("", None, "remove"), ("", "0"))

    def test_real_add_and_order_events_preserve_browser_session_mode(self):
        child = self.folder("child")
        app, _ = self.build()
        state = SessionState(app)
        self.invoke(app, "Add folder", [str(self.root), "0"], state=state)
        result = self.invoke(
            app, "Use folder", [str(self.root), "0", str(child), None], state=state
        )
        self.assertEqual(picker.path_lines(result[0]), [str(self.root), str(child)])
        result = self.invoke(app, "Move up", [result[0], "1"], state=state)
        self.assertEqual(picker.path_lines(result[0]), [str(child), str(self.root)])
        result = self.invoke(app, "Move down", [result[0], "0"], state=state)
        self.assertEqual(picker.path_lines(result[0]), [str(self.root), str(child)])
        result = self.invoke(app, "Remove folder", [result[0], "1"], state=state)
        self.assertEqual(result[0], str(self.root))

    def test_cancel_or_use_finishes_add_mode_before_a_native_replacement(self):
        added, replacement = self.folder("added"), self.folder("replacement")
        self.available.return_value = True
        self.native.side_effect = None
        self.native.return_value = str(replacement)
        for finish in ("Cancel", "Use folder"):
            with self.subTest(finish=finish):
                app, _ = self.build()
                state = SessionState(app)
                draft = str(self.root)
                self.invoke(app, "Add folder", [draft, "0"], state=state)
                if finish == "Cancel":
                    self.invoke(app, finish, [], state=state)
                    expected = [str(replacement)]
                else:
                    draft = self.invoke(app, finish, [draft, "0", str(added), None], state=state)[0]
                    expected = [str(replacement), str(added)]
                result = self.invoke(app, "Open host dialog", [draft, "0", None], state=state)
                self.assertEqual(picker.path_lines(result[0]), expected)

    def test_single_root_picker_replaces_the_scalar_and_keeps_advanced_editing(self):
        child = self.folder("single Ω")
        app, field = self.build(key="path_outputs", value=str(self.root))
        self.assertTrue(field.interactive)
        result = self.invoke(app, "Select folder", [str(self.root), str(self.root)])
        self.assertTrue(result[3]["visible"])
        result = self.invoke(app, "Use folder", [str(self.root), str(self.root), str(child), None])
        self.assertEqual(result[0], str(child))
        self.assertEqual(result[1]["value"], str(child))

    def test_unchanged_legacy_namespace_is_retained_but_new_bad_paths_fail_before_save(self):
        wrong = "/mnt/c/old" if os.name == "nt" else r"C:\old"
        self.assertEqual(
            picker.validate_path_settings({"path_clip": [wrong]}, {"path_clip": [wrong]}),
            {"path_clip": [wrong]},
        )
        with self.assertRaises(ValueError):
            picker.validate_path_settings({"path_clip": [wrong]}, {"path_clip": ["../models/clip"]})

    def test_duplicates_and_aliases_are_rejected_without_rewriting_original(self):
        first = self.folder("one")
        self.folder("one/child")
        for duplicate in (str(first), str(first / "child" / "..")):
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "already"):
                picker.apply_choice("path_clip", str(first), "0", duplicate, append=True)
        self.assertEqual(
            picker.apply_choice("path_clip", str(first), "0", str(first)), (str(first), "0")
        )

    def test_symlink_or_junction_alias_uses_consumer_resolution_without_recursive_browsing(self):
        target = self.folder("target")
        link = self.root / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            if os.name != "nt":
                self.skipTest(f"Host cannot create the owned symlink fixture: {error}")
            # Both junction names stay in this test's owned temporary root.
            # Python rmtree unlinks junctions rather than traversing targets.
            self.assertTrue(link.parent.resolve().is_relative_to(self.root))
            self.assertTrue(target.resolve().is_relative_to(self.root))
            result = picker.subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:RF_PICKER_TEST_LINK -Target $env:RF_PICKER_TEST_TARGET | Out-Null",
                ],
                env=dict(
                    os.environ, RF_PICKER_TEST_LINK=str(link), RF_PICKER_TEST_TARGET=str(target)
                ),
                capture_output=True,
                timeout=15,
                check=False,
                creationflags=picker.subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode:
                self.skipTest("Host cannot create the owned junction fixture.")
            self.assertTrue(link.is_junction())
        self.assertTrue(picker.existing_directory("path_clip", str(link)).samefile(target))
        with patch.object(picker.os, "scandir", wraps=os.scandir) as scan:
            choices, _ = picker.directory_choices(self.root)
        self.assertIn(("link", str(link)), choices)
        scan.assert_called_once_with(self.root)
        with self.assertRaisesRegex(ValueError, "already"):
            picker.apply_choice("path_clip", str(target), "0", str(link), append=True)

    def test_inaccessible_and_non_directory_choices_fail_without_creation(self):
        missing = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            picker.existing_directory("path_clip", str(missing))
        self.assertFalse(missing.exists())
        file = self.root / "file"
        file.write_text("fixture", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "folder"):
            picker.existing_directory("path_clip", str(file))
        with patch.object(picker.os, "scandir", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                picker.existing_directory("path_clip", str(self.root))
        with patch.object(picker.os, "access", return_value=False):
            with self.assertRaisesRegex(PermissionError, "writable"):
                picker.existing_directory("path_outputs", str(self.root))

    def test_listing_is_shallow_and_bounded_including_symlink_directories(self):
        for number in range(5):
            self.folder(f"folder{number}/nested")
        with (
            patch.object(picker, "MAX_ENTRIES", 2),
            patch.object(picker.os, "scandir", wraps=os.scandir) as scan,
        ):
            choices, limited = picker.directory_choices(self.root)
        self.assertEqual(len(choices), 2)
        self.assertTrue(limited)
        scan.assert_called_once_with(self.root)
        self.assertTrue(all("nested" not in name for name, _ in choices))

    def test_relative_roots_match_model_wildcard_and_archive_consumers(self):
        paths = PathManager.__new__(PathManager)
        paths.paths = {"path_clip": ["../models/encoders Ω"]}
        self.assertEqual(
            picker.resolve_path("path_clip", "../models/encoders Ω"),
            paths.get_model_folders("clip")[0],
        )
        self.assertEqual(
            picker.resolve_path("path_wildcards", "wildcards Ω"), wildcard_roots(["wildcards Ω"])[0]
        )
        self.assertEqual(
            picker.resolve_path("archive_folders", "archive Ω"), Path("archive Ω").resolve()
        )
        self.assertEqual(
            picker.validate_path_settings({"path_clip": ["../models/encoders Ω"]}, {}),
            {"path_clip": ["../models/encoders Ω"]},
        )

    def test_wrong_host_namespaces_and_control_characters_are_rejected(self):
        for value in (r"C:\models", r"\\server\share", "C:relative"):
            with self.subTest(linux=value), self.assertRaises(ValueError):
                picker.validate_namespace(value, windows=False)
        for value in (
            "/mnt/c/models",
            "/home/user/models",
            "C:relative",
            "bad\npath",
            "file:///C:/models",
            "https://host/folder",
        ):
            with self.subTest(windows=value), self.assertRaises(ValueError):
                picker.validate_namespace(value, windows=True)
        for value in (r"C:\models Ω", r"\\server\share", "../models"):
            self.assertEqual(picker.validate_namespace(value, windows=True), value)
        self.assertEqual(
            picker.validate_namespace("/mnt/c/models Ω", windows=False), "/mnt/c/models Ω"
        )

    def test_settings_register_all_ten_picker_fields_once_with_manual_editing(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "create_path_control"
                for child in ast.walk(node)
            )
        )
        archive = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "archive_folders"
                for target in node.targets
            )
        )
        registered = {}

        def register(key, field):
            self.assertNotIn(key, registered)
            registered[key] = field

        scope = {
            "path_manager": SimpleNamespace(
                paths=PathManager.DEFAULT_PATHS.copy(), DEFAULT_PATHS=PathManager.DEFAULT_PATHS
            ),
            "t": lambda value: value,
            "create_path_control": picker.create_path_control,
            "add_setting": register,
            "settings": SimpleNamespace(default_settings={}),
        }
        with gr.Blocks():
            exec(
                compile(
                    ast.Module(body=[loop, archive], type_ignores=[]), "production paths", "exec"
                ),
                scope,
            )
            register("archive_folders", scope["archive_folders"])
        self.assertEqual(set(registered), picker.PATH_KEYS)
        self.assertTrue(
            all(
                isinstance(field, gr.Textbox) and field.interactive for field in registered.values()
            )
        )

    def callback(self, manager, paths, keys):
        scope = {
            "state": {"setting_name": keys},
            "settings": manager,
            "path_manager": paths,
            "validate_settings": validate_settings,
            "gr": SimpleNamespace(Error=ValueError, Info=Mock()),
            "models": SimpleNamespace(offline=False),
            "performance_settings": SimpleNamespace(
                model_recommendations=SimpleNamespace(offline=False)
            ),
        }
        return function_from("ruinedfooocus/ui/ui_settings.py", "save_clicked", scope)

    def manager(self):
        manager = SettingsManager("picker", defaults={"archive_folders": []})
        paths = PathManager.__new__(PathManager)
        paths.paths = PathManager.DEFAULT_PATHS.copy()
        paths.name = "picker"
        paths.set_settings_path("picker")
        paths.save_paths()
        return manager, paths

    def test_profile_save_reload_preserves_roots_and_native_model_output_consumers(self):
        manager, paths = self.manager()
        first, second, output, archive, wildcard = (
            self.folder(name) for name in ("first Ω", "second", "output", "archive", "wildcards")
        )
        (second / "encoder.safetensors").write_bytes(b"private lookup fixture")
        chosen, _ = picker.apply_choice("path_clip", str(first), "0", str(second), append=True)
        callback = self.callback(
            manager, paths, ["path_clip", "path_outputs", "archive_folders", "path_wildcards"]
        )
        launch = SimpleNamespace(args=SimpleNamespace(settings="picker", offline=True))
        with patch.dict(sys.modules, {"ruinedfooocus.app.arguments": launch}):
            callback(chosen, str(output), str(archive), str(wildcard))
        reloaded = PathManager.__new__(PathManager)
        reloaded.set_settings_path("picker")
        reloaded.paths = reloaded.load_paths()
        self.assertEqual(reloaded.get_model_folders("clip"), [first, second])
        self.assertEqual(
            reloaded.get_folder_file_path("clip", "encoder.safetensors"),
            second / "encoder.safetensors",
        )
        self.assertEqual(reloaded.get_model_paths()["temp_outputs_path"], output)
        self.assertEqual(wildcard_roots(reloaded.paths["path_wildcards"])[0], wildcard)
        restored = SettingsManager("picker")
        self.assertEqual(restored.default_settings["archive_folders"], [str(archive)])
        # Exercise Library's real scan callback against the reloaded settings.
        (archive / "archive.png").write_bytes(b"private enumeration fixture")
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE images(fullpath text, path text, json text)")
        scope = {
            "Path": Path,
            "os": os,
            "settings": restored,
            "TimeIt": lambda _: nullcontext(),
            "get_png_metadata": lambda _: {},
            "json": json,
        }
        scan = function_from(
            "ruinedfooocus/features/gallery/imagebrowser.py", "update_images", scope
        )
        add = function_from("ruinedfooocus/features/gallery/imagebrowser.py", "add_image", scope)
        browser = SimpleNamespace(path_manager=reloaded, sql_conn=db)
        browser.add_image = lambda *args: add(browser, *args)
        with patch("builtins.print"):
            self.assertIn("Found 1 images", scan(browser))
        self.assertEqual(
            db.execute("SELECT fullpath FROM images").fetchone()[0], str(archive / "archive.png")
        )

    def test_profile_write_failure_keeps_memory_disk_and_draft_recoverable(self):
        manager, paths = self.manager()
        original = manager.settings_path.read_bytes(), paths.settings_path.read_bytes()
        old_paths = dict(paths.paths)
        chosen = self.folder("new output")
        callback = self.callback(manager, paths, ["path_outputs"])
        with patch.object(manager, "save_profile", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(ValueError, "write failed"):
                callback(str(chosen))
        self.assertEqual(
            (manager.settings_path.read_bytes(), paths.settings_path.read_bytes()), original
        )
        self.assertEqual(paths.paths, old_paths)
        self.assertEqual(manager.name, "picker")
        self.assertTrue(chosen.is_dir())


if __name__ == "__main__":
    unittest.main()
