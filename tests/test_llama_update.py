"""Hermetic tests for explicit updates of official native llama.cpp packages."""

from contextlib import nullcontext
from concurrent.futures import CancelledError
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from ruinedfooocus.backends.llama import llama_installer, llama_server, llama_update, runtime_probe


class Response:
    def __init__(self, body=b"", status=200, headers=None):
        self.body = body
        self.status_code = status
        self.headers = headers or {}
        self.is_redirect = status == 302

    def iter_content(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


class LlamaUpdateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, RF_DATA_DIR=str(self.root), LLAMA_SERVER=""))
        self.enterContext(patch.object(llama_update.platform, "system", return_value="Windows"))
        self.enterContext(patch.object(llama_update.platform, "machine", return_value="AMD64"))
        self.offline = self.enterContext(
            patch.object(llama_update, "skip_dependency_install", return_value=False)
        )
        self.get = self.enterContext(patch.object(llama_update.requests, "get"))
        self.probe = self.enterContext(patch.object(llama_update, "_verify_executable"))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("llama-server.exe", b"fixture executable")
        self.archive = buffer.getvalue()

    def release(self, tag, backend="CPU", **overrides):
        name = llama_installer.package_name(
            "Windows", "AMD64", backend == "Vulkan", backend=backend, version=tag
        )
        asset = {
            "name": name,
            "state": "uploaded",
            "browser_download_url": f"{llama_update._DOWNLOAD}/{tag}/{name}",
            "digest": "sha256:" + hashlib.sha256(self.archive).hexdigest(),
            "size": len(self.archive),
        }
        asset.update(overrides)
        return {"tag_name": tag, "draft": False, "prerelease": True, "assets": [asset]}

    def candidate(self, tag="b12462", backend="CPU"):
        return llama_update._candidate(self.release(tag, backend), backend)

    def test_check_selects_highest_verified_compatible_build(self):
        releases = [
            self.release("v0.9.0"),
            self.release("b12465", backend="Vulkan"),
            self.release("b12464", digest=None),
            self.release("b12463", browser_download_url="https://example.test/unknown.zip"),
            self.release("b12460"),
            self.release("b12462"),
            self.release("b12109"),
        ]
        self.get.return_value = nullcontext(Response(json.dumps(releases).encode()))
        selected, candidate = llama_update.check_update("CPU")
        self.assertEqual(selected, llama_installer.VERSION)
        self.assertEqual(candidate, self.candidate())
        self.assertTrue(candidate.prerelease)
        self.assertEqual(self.get.call_args.args[0], llama_update._RELEASES)
        self.assertFalse(self.get.call_args.kwargs["allow_redirects"])

    def test_offline_custom_and_auto_checks_make_no_request(self):
        self.offline.return_value = True
        with self.assertRaisesRegex(RuntimeError, "offline"):
            llama_update.check_update("CPU")
        with self.assertRaisesRegex(RuntimeError, "offline"):
            llama_update.install_update("CPU", self.candidate(), Mock())
        self.offline.return_value = False
        with self.assertRaisesRegex(RuntimeError, "explicit CPU or Vulkan"):
            llama_update.check_update("Auto")
        with patch.dict(os.environ, LLAMA_SERVER="custom.exe"):
            with self.assertRaisesRegex(RuntimeError, "custom binary"):
                llama_update.check_update("CPU")
        self.get.assert_not_called()

    def test_cancelled_release_check_never_reaches_network(self):
        with self.assertRaises(CancelledError):
            llama_update.check_update("CPU", check_cancelled=Mock(side_effect=CancelledError()))
        self.get.assert_not_called()

    def test_update_verifies_and_selects_while_rollback_preserves_cache(self):
        candidate = self.candidate()
        self.get.side_effect = [
            nullcontext(
                Response(
                    status=302,
                    headers={"Location": "https://release-assets.githubusercontent.com/a.zip"},
                )
            ),
            nullcontext(Response(self.archive)),
        ]
        progress = Mock()
        self.assertEqual(llama_update.install_update("CPU", candidate, progress), candidate.tag)
        tag, executable = llama_update.selected_build("CPU")
        self.assertEqual(tag, candidate.tag)
        self.assertEqual(executable.read_bytes(), b"fixture executable")
        self.probe.assert_called_once()
        progress.assert_called_with(1, "Ready for the next Chat load")
        self.assertEqual(llama_update.selected_build("Vulkan"), (llama_installer.VERSION, None))
        self.assertEqual(llama_update.use_pinned_build("CPU"), llama_installer.VERSION)
        self.assertEqual(llama_update.selected_build("CPU"), (llama_installer.VERSION, None))
        self.assertTrue(executable.is_file())
        self.assertEqual(self.get.call_count, 2)

    def test_existing_explicit_older_verified_selection_survives_pin_update(self):
        candidate = self.candidate()
        self.get.return_value = nullcontext(Response(self.archive))
        llama_update.install_update("CPU", candidate, Mock())
        with patch.object(llama_installer, "VERSION", "b99999"):
            self.assertEqual(llama_update.selected_build("CPU")[0], candidate.tag)
        self.get.reset_mock()
        self.assertTrue(llama_update.selected_build("CPU")[1].is_file())
        self.get.assert_not_called()

    def test_failed_update_preserves_working_selection_and_rejects_redirect(self):
        candidate = self.candidate()
        self.get.return_value = nullcontext(Response(self.archive))
        llama_update.install_update("CPU", candidate, Mock())
        before = llama_update.selected_build("CPU")
        newer = self.candidate("b12463")
        self.get.return_value = nullcontext(Response(b"x" * len(self.archive)))
        with self.assertRaisesRegex(RuntimeError, "published size and SHA256"):
            llama_update.install_update("CPU", newer, Mock())
        self.assertEqual(llama_update.selected_build("CPU"), before)
        self.get.reset_mock()
        self.get.return_value = nullcontext(
            Response(status=302, headers={"Location": "https://example.test/archive.zip"})
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected redirect"):
            llama_update.install_update("CPU", newer, Mock())
        self.get.assert_called_once()
        self.assertEqual(llama_update.selected_build("CPU"), before)

    def test_cancel_transfer_and_pointer_publication_preserve_selection(self):
        candidate = self.candidate()
        self.get.return_value = nullcontext(Response(self.archive))
        llama_update.install_update("CPU", candidate, Mock())
        before = llama_update._pointer("CPU").read_bytes()
        for phase in ("downloading_runtime", "staging_runtime"):
            cancelled = False

            def progress(event):
                nonlocal cancelled
                if event["phase"] == phase:
                    cancelled = True

            def check():
                if cancelled:
                    raise CancelledError()

            with self.subTest(phase=phase), self.assertRaises(CancelledError):
                llama_update.install_update(
                    "CPU",
                    self.candidate("b12463"),
                    Mock(),
                    progress_callback=progress,
                    check_cancelled=check,
                )
            self.assertEqual(llama_update._pointer("CPU").read_bytes(), before)
            self.assertEqual(llama_update.selected_build("CPU")[0], candidate.tag)

    def test_cancel_cached_update_verification_preserves_selection_and_cache(self):
        candidate = self.candidate()
        self.get.return_value = nullcontext(Response(self.archive))
        llama_update.install_update("CPU", candidate, Mock())
        pointer = llama_update._pointer("CPU")
        original_pointer = pointer.read_bytes()
        newer = self.candidate("b12463")
        # An earlier cancelled selection may leave a complete verified candidate cached.
        directory = llama_installer.package_directory(newer.name, version=newer.tag)
        directory.mkdir(parents=True)
        executable = directory / "llama-server.exe"
        executable.write_bytes(b"inert cached update fixture")
        manifest = llama_installer._runtime_manifest(directory, newer.name, version=newer.tag)
        marker = directory / "rf-runtime.json"
        marker.write_text(json.dumps(manifest | {"archives": [newer.archive()]}))
        original_manifest = marker.read_bytes()
        events = []

        def check():
            if events:
                raise CancelledError()

        self.get.reset_mock()
        self.probe.reset_mock()
        with self.assertRaises(CancelledError):
            llama_update.install_update(
                "CPU", newer, Mock(), progress_callback=events.append, check_cancelled=check
            )
        self.assertEqual(events[0]["phase"], "verifying_runtime")
        self.assertEqual(
            events[0]["runtime"], {"tag": newer.tag, "backend": "CPU", "package": newer.name}
        )
        self.assertEqual(pointer.read_bytes(), original_pointer)
        self.assertEqual(marker.read_bytes(), original_manifest)
        self.assertEqual(executable.read_bytes(), b"inert cached update fixture")
        self.get.assert_not_called()
        self.probe.assert_not_called()
        # Retry reuses the same cache without a transfer and selects it only after verification.
        self.assertEqual(llama_update.install_update("CPU", newer, Mock()), newer.tag)
        self.assertEqual(llama_update.selected_build("CPU")[0], newer.tag)
        self.get.assert_not_called()

    def test_failed_probe_or_tampered_cache_uses_pin(self):
        candidate = self.candidate()
        self.get.return_value = nullcontext(Response(self.archive))
        self.probe.side_effect = RuntimeError("fixture probe failed")
        with self.assertRaisesRegex(RuntimeError, "fixture probe failed"):
            llama_update.install_update("CPU", candidate, Mock())
        self.assertEqual(llama_update.selected_build("CPU"), (llama_installer.VERSION, None))
        self.probe.side_effect = None
        llama_update.install_update("CPU", candidate, Mock())
        _, executable = llama_update.selected_build("CPU")
        executable.write_bytes(b"tampered")
        self.assertEqual(llama_update.selected_build("CPU"), (llama_installer.VERSION, None))

    def test_candidate_cannot_change_backend_or_asset_origin(self):
        candidate = self.candidate()
        for backend, selection in (
            ("Vulkan", candidate),
            ("CPU", replace(candidate, url="https://example.test/archive.zip")),
        ):
            with self.subTest(backend=backend, url=selection.url):
                with self.assertRaisesRegex(RuntimeError, "Check for a compatible update"):
                    llama_update.install_update(backend, selection, Mock())
        self.get.assert_not_called()

    def test_archive_path_cannot_escape_staging(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../escape.exe", b"fixture")
        self.archive = buffer.getvalue()
        self.get.return_value = nullcontext(Response(self.archive))
        with self.assertRaisesRegex(RuntimeError, "unsafe path"):
            llama_update.install_update("CPU", self.candidate(), Mock())
        self.assertFalse(list(self.root.rglob("escape.exe")))
        self.assertEqual(llama_update.selected_build("CPU"), (llama_installer.VERSION, None))
        self.probe.assert_not_called()

    @unittest.skipUnless(hasattr(tarfile, "data_filter"), "Python needs safe tar extraction")
    def test_linux_cpu_selects_and_extracts_official_tar_family(self):
        with patch.object(llama_update.platform, "system", return_value="Linux"):
            buffer = io.BytesIO()
            binary = b"fixture ELF"
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                item = tarfile.TarInfo("bin/llama-server")
                item.mode = 0o755
                item.size = len(binary)
                archive.addfile(item, io.BytesIO(binary))
            self.archive = buffer.getvalue()
            release = self.release("b12462")
            name = "llama-b12462-bin-ubuntu-x64.tar.gz"
            release["assets"][0].update(
                name=name, browser_download_url=f"{llama_update._DOWNLOAD}/b12462/{name}"
            )
            candidate = llama_update._candidate(release, "CPU")
            self.assertEqual(candidate.name, name)
            self.get.return_value = nullcontext(Response(self.archive))
            llama_update.install_update("CPU", candidate, Mock())
            tag, executable = llama_update.selected_build("CPU")
            self.assertEqual(tag, candidate.tag)
            self.assertEqual(executable.read_bytes(), binary)

    def test_only_explicit_backend_uses_the_updated_executable(self):
        with (
            patch.object(
                llama_update, "selected_build", return_value=("b12462", Path("updated"))
            ) as selected,
            patch.object(llama_server, "server_path", return_value=Path("pinned")),
            patch.object(llama_server, "runtime_environment", return_value={}),
        ):
            self.assertEqual(llama_server.resolve_runtime(False, "Auto")[0], Path("pinned"))
            selected.assert_not_called()
            self.assertEqual(llama_server.resolve_runtime(False, "CPU")[0], Path("updated"))
            selected.assert_called_once_with("CPU")

    def test_failed_vulkan_initialization_retries_the_pinned_backend(self):
        with (
            patch.object(llama_update, "selected_build", return_value=("b12462", Path("updated"))),
            patch.object(llama_server, "server_path", return_value=Path("pinned")),
            patch.object(llama_server, "runtime_environment", return_value={}),
            patch.object(
                llama_server,
                "_probe_runtime",
                side_effect=[llama_server.RuntimeUnavailable(), "Vulkan"],
            ),
        ):
            self.assertEqual(
                llama_server.resolve_runtime(True, "Vulkan")[:4],
                (Path("pinned"), {}, True, "Vulkan"),
            )

    def test_vulkan_device_crash_stops_without_trying_a_second_binary(self):
        with (
            patch.object(llama_update, "selected_build", return_value=("b12462", Path("updated"))),
            patch.object(llama_server, "server_path") as pinned,
            patch.object(llama_server, "runtime_environment", return_value={}),
            patch.object(
                llama_server,
                "_probe_runtime",
                side_effect=llama_server.RuntimeVerificationError("fixture device failure"),
            ),
        ):
            with self.assertRaisesRegex(
                llama_server.RuntimeVerificationError, "fixture device failure"
            ):
                llama_server.resolve_runtime(True, "Vulkan")
        pinned.assert_not_called()


class RuntimeVerificationTests(unittest.TestCase):
    def test_windows_prerelease_reports_the_build_after_semver_on_stderr(self):
        import subprocess

        candidate = llama_update.PrebuiltUpdate("CPU", "b12462", "fixture.zip", "", "", 1, True)
        output = (
            "version: 0.9.0-dev (build 12462, commit 15759e0ca)\n"
            "built with Clang 20.1.8 for Windows x86_64\n"
        )
        with (
            patch.object(llama_installer, "runtime_environment", return_value={}),
            patch.object(llama_update.subprocess, "run") as run,
        ):
            run.return_value = subprocess.CompletedProcess([], 0, "", output)
            llama_update._verify_executable(Path("fixture"), candidate)
            for invalid in (output.replace("12462", "12463"), "version: 0.9.0-dev\n"):
                with self.subTest(output=invalid):
                    run.return_value = subprocess.CompletedProcess([], 0, "", invalid)
                    with self.assertRaisesRegex(RuntimeError, "did not report"):
                        llama_update._verify_executable(Path("fixture"), candidate)

    def test_staged_binary_must_report_expected_build_and_backend(self):
        import subprocess

        candidate = llama_update.PrebuiltUpdate("Vulkan", "b12462", "fixture.zip", "", "", 1, True)
        with (
            patch.object(llama_installer, "runtime_environment", return_value={}),
            patch.object(
                llama_update.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "", "version: 12462 (abc)"),
            ) as run,
            patch.object(runtime_probe, "probe_runtime", return_value="Vulkan") as probe,
        ):
            llama_update._verify_executable(Path("fixture"), candidate)
            self.assertEqual(run.call_args.args[0], ["fixture", "--version"])
            probe.assert_called_once_with(Path("fixture"), {}, "Vulkan")
            probe.return_value = None
            with self.assertRaisesRegex(RuntimeError, "usable Vulkan"):
                llama_update._verify_executable(Path("fixture"), candidate)
            run.return_value = subprocess.CompletedProcess([], 0, "", "version: 12460 (abc)")
            with self.assertRaisesRegex(RuntimeError, "did not report"):
                llama_update._verify_executable(Path("fixture"), candidate)

    def test_callback_probe_is_independent_of_server_and_keeps_runtime_identity(self):
        import subprocess

        candidate = llama_update.PrebuiltUpdate("Vulkan", "b12462", "fixture.zip", "", "", 1, True)
        events = []
        check = Mock()
        with (
            patch.object(llama_installer, "runtime_environment", return_value={}),
            patch.object(
                llama_installer,
                "_run_cancellable",
                return_value=subprocess.CompletedProcess([], 0, "version: 12462 (abc)", ""),
            ),
            patch.object(
                runtime_probe, "run_bounded", return_value=(b"Vulkan0: inert device", b"")
            ) as run,
            patch.object(
                llama_server,
                "_probe_runtime",
                side_effect=AssertionError("later server dependency"),
            ),
        ):
            llama_update._verify_executable(
                Path("fixture"), candidate, progress_callback=events.append, check_cancelled=check
            )
            self.assertIs(run.call_args.kwargs["check_cancelled"], check)
            self.assertEqual(run.call_args.args[0], ["fixture", "--list-devices"])
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event["phase"] == "probing_device" for event in events))
            self.assertTrue(
                all(
                    event["runtime"]
                    == {"tag": candidate.tag, "backend": "Vulkan", "package": candidate.name}
                    for event in events
                )
            )
            run.side_effect = CancelledError()
            with self.assertRaises(CancelledError):
                llama_update._verify_executable(
                    Path("fixture"),
                    candidate,
                    progress_callback=events.append,
                    check_cancelled=check,
                )


class UpdateControlsTests(unittest.TestCase):
    def test_hardware_actions_keep_selection_and_progress_session_local(self):
        import gradio as gr
        from ui_test_helpers import function_from
        from ruinedfooocus.core.translation_manager import localize

        create = function_from(
            "ruinedfooocus/ui/ui_hardware.py",
            "create_llama_update_controls",
            {"gr": gr, "t": str, "localize": localize, "locale_of": lambda request: "en"},
        )
        candidate = llama_update.PrebuiltUpdate("CPU", "b12462", "fixture.zip", "", "", 1, True)
        with (
            patch.object(llama_update, "configured_build", return_value=llama_installer.VERSION),
            patch.object(
                llama_update, "check_update", return_value=(llama_installer.VERSION, candidate)
            ) as check,
            patch.object(llama_update, "install_update", return_value=candidate.tag) as install,
            gr.Blocks(analytics_enabled=False) as app,
        ):
            backend = gr.Dropdown(choices=["CPU", "Vulkan"], value="CPU")
            cpu = gr.Checkbox(value=True)
            create(backend, cpu, "CPU")
            callbacks = {event.fn.__name__: event for event in app.fns.values()}
            self.assertEqual(callbacks["check_update"].api_visibility, "private")
            self.assertEqual(callbacks["install_update"].api_visibility, "private")
            check.assert_not_called()
            install.assert_not_called()
            message, stored, button = callbacks["check_update"].fn("CPU", True, None)
            self.assertIn("prerelease", message)
            self.assertEqual(stored, candidate)
            self.assertTrue(button["interactive"])
            message, stored, button = callbacks["install_update"].fn(
                "CPU", True, stored, None, Mock()
            )
            self.assertIn("next llama.cpp process", message)
            self.assertIsNone(stored)
            self.assertFalse(button["interactive"])
            self.assertEqual(install.call_args.args[:2], ("CPU", candidate))
            message, _, _ = callbacks["check_update"].fn("Vulkan", True, None)
            self.assertIn("CPU-only mode", message)
            check.assert_called_once_with("CPU")


if __name__ == "__main__":
    unittest.main()
