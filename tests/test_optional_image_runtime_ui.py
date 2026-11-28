"""Optional Settings actions stay passive until explicit, owned admission."""

from concurrent.futures import Future, CancelledError
import copy
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.ui import optional_image_runtime as optional
from ruinedfooocus.core.translation_manager import localize, resolve_locale, localized_output
from ui_test_helpers import function_from


class OptionalImageUiTests(unittest.TestCase):
    def test_optional_controls_construct_without_actions_and_register_exact_preferences(self):
        import gradio as gr

        controls = {}
        create = function_from(
            "ruinedfooocus/ui/ui_hardware.py",
            "create_optional_image_controls",
            {
                "gr": gr,
                "t": lambda value: value,
                "settings": SimpleNamespace(default_settings={}),
                "add_setting": lambda name, control: controls.update({name: control}),
                "skip_dependency_install": lambda: True,
                "localized_output": localized_output,
                "localize": localize,
                "resolve_locale": resolve_locale,
            },
        )
        with (
            patch.object(
                optional.activation,
                "status",
                return_value={"available": False, "reason": "Unprepared"},
            ),
            patch.object(
                optional.activation, "verify", side_effect=AssertionError("No probe while building")
            ),
            patch.object(
                optional.activation,
                "install",
                side_effect=AssertionError("No install while building"),
            ),
            gr.Blocks() as app,
        ):
            cpu = gr.Checkbox(False)
            image = gr.Dropdown(["auto", "cpu"], value="auto")
            create({}, cpu, image)
        self.assertEqual(controls["image_engine"].value, "auto")
        self.assertEqual(controls["sdcpp_vulkan_device"].value, "Vulkan0")
        self.assertGreaterEqual(len(app.fns), 4)

    def test_passive_status_does_not_verify_hash_acquire_or_restore_ordinal_identity(self):
        state = {"available": True, "prepared": True, "devices": (), "identity_complete": False}
        with (
            patch.object(optional.activation, "status", return_value=state),
            patch.object(
                optional.activation, "current", side_effect=AssertionError("active restore")
            ),
            patch.object(optional.activation, "install", side_effect=AssertionError("acquisition")),
            patch.object(optional.activation, "verify", side_effect=AssertionError("device query")),
        ):
            text = optional.status_text(values={})
        self.assertIn("Requested image engine: auto", text)
        self.assertIn("Cached manifest present; verify before use", text)
        self.assertIn("not establish a physical", text)
        self.assertIn("do not qualify", text)

    def test_explicit_cpu_and_exact_backend_name_are_authoritative(self):
        for values in ({"cpu_everywhere": True}, {"image_backend": "cpu"}, {"image_device": "CPU"}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "CPU authority"):
                optional.backend_request("vulkan", values)
        args = SimpleNamespace(cpu=True, _compute_status={"cli": ["--cpu"]})
        with self.assertRaises(ValueError):
            optional.backend_request("vulkan", {}, launch_args=args)
        for device in ("0", "GPU0", "Vulkan-1", "Vulkan0 extra"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                optional.backend_request("vulkan", {}, device=device)
        self.assertEqual(optional.backend_request("cpu", {"cpu_everywhere": True}), "cpu")
        fallback = SimpleNamespace(
            cpu=True, _compute_status={"backend": {"cpu_origin": "automatic"}}
        )
        self.assertEqual(optional.backend_request("vulkan", {}, launch_args=fallback), "Vulkan0")

    def test_install_uses_bounded_io_admission_and_forwards_offline_without_compute(self):
        captured = {}

        def submit(prepare, ready, failed):
            captured["prepare"] = prepare
            ready(prepare())
            future = Future()
            future.set_result(None)
            return future

        session = optional.RuntimeActionSession()
        with patch.object(optional.activation, "install", return_value="prepared") as install:
            result = session.install(
                "cpu",
                "Vulkan0",
                {},
                offline=True,
                skip_install=True,
                admission=SimpleNamespace(submit=submit),
            )
        self.assertEqual(result.result(), "prepared")
        self.assertTrue(install.call_args.kwargs["offline"])
        self.assertTrue(install.call_args.kwargs["skip_dependency_install"])
        install.call_args.kwargs["check_cancelled"]()
        session.finish()
        self.assertFalse(session.active)

    def test_cancel_and_duplicate_action_are_scoped_to_this_browser(self):
        first = optional.RuntimeActionSession()
        second = copy.deepcopy(first)
        admission = SimpleNamespace(submit=Mock(return_value=Future()))
        first.install("cpu", "Vulkan0", {}, offline=False, skip_install=False, admission=admission)
        with self.assertRaisesRegex(ValueError, "already pending"):
            first.install(
                "cpu", "Vulkan0", {}, offline=False, skip_install=False, admission=admission
            )
        self.assertTrue(first.active)
        first.cancel()
        with self.assertRaises(CancelledError):
            first.check()
        self.assertFalse(second.cancelled.is_set())
        self.assertFalse(second.active)
        first.finish()

    def test_verify_is_admitted_to_sdcpp_and_cancellation_prevents_native_work(self):
        from ruinedfooocus.app import api_runtime

        job = SimpleNamespace(cancelled=threading.Event())
        verified = SimpleNamespace(
            candidate=SimpleNamespace(device="cpu", device_description="CPU-only bundle")
        )

        def submit(operation):
            self.assertEqual(operation.resource_group, "sdcpp")
            job.value = operation(job, 1)
            return job

        with (
            patch.object(api_runtime, "submit", side_effect=submit),
            patch.object(
                api_runtime, "wait_for_result", side_effect=lambda job, timeout: job.value
            ),
            patch.object(optional.activation, "verify", return_value=verified) as verify,
        ):
            result = optional.RuntimeActionSession().verify("cpu", "Vulkan0", {})
        self.assertIn("qualification remains separate", result)
        self.assertEqual(verify.call_args.args, ("cpu", "cpu"))
        operation = optional.AdmittedOperation(Mock(), "rembg")
        job.cancelled.set()
        with self.assertRaises(CancelledError):
            operation(job, 1)
        operation.operation.assert_not_called()

    def test_cpu_verify_reaches_real_runtime_with_its_canonical_device_name(self):
        from ruinedfooocus.backends.sdcpp import artifacts, runtime
        from ruinedfooocus.backends.sdcpp.contracts import RuntimeCandidate

        item = artifacts.ARTIFACTS[0]
        session = optional.RuntimeActionSession()

        def open_runtime(directory, artifact, *, device):
            return RuntimeCandidate(
                Path(directory) / "sd-cli.exe",
                artifact.backend,
                device,
                artifact_sha256=artifact.sha256,
            )

        with (
            patch.object(
                session,
                "run_admitted",
                side_effect=lambda operation, *a, **k: operation(lambda: None),
            ),
            patch.object(optional.activation, "artifact", return_value=item),
            patch.object(
                optional.activation, "runtime_directory", return_value=Path("owned-runtime")
            ),
            patch.object(artifacts, "open_runtime", side_effect=open_runtime),
            patch.object(runtime, "_unchanged"),
            patch.object(runtime, "_environment", return_value={}),
            patch.object(runtime, "_run", return_value="Usage: sd-cli") as run,
            patch.dict(optional.activation._ACTIVE, {}, clear=True),
        ):
            result = session.verify("cpu", "Vulkan0", {"cpu_everywhere": True})
        self.assertIn("Verified backend cpu", result)
        self.assertEqual(run.call_args.args[0][-1], "--help")

    def test_old_preparation_keeps_its_cancel_token_after_a_new_action_starts(self):
        pending = []

        def submit(prepare, ready, failed):
            pending.append(prepare)
            return Future()

        session = optional.RuntimeActionSession()
        session.install(
            "cpu",
            "Vulkan0",
            {},
            offline=False,
            skip_install=False,
            admission=SimpleNamespace(submit=submit),
        )
        session.cancel()
        session.finish()
        session.begin()
        with patch.object(
            optional.activation,
            "install",
            side_effect=AssertionError("Old action must stay cancelled"),
        ):
            with self.assertRaises(CancelledError):
                pending[0]()
        session.finish()

    def test_busy_install_callback_does_not_clear_another_active_action(self):
        import gradio as gr

        session = SimpleNamespace(
            install=Mock(side_effect=ValueError("already pending")), finish=Mock()
        )
        callback = function_from(
            "ruinedfooocus/ui/ui_hardware.py",
            "optional_install",
            {
                "requested_values": lambda *args: {},
                "settings": SimpleNamespace(default_settings={}),
                "skip_dependency_install": lambda: True,
                "gr": gr,
                "localize": localize,
                "resolve_locale": resolve_locale,
            },
        )
        self.assertIn(
            "already pending", list(callback("cpu", "Vulkan0", False, "auto", session))[0]
        )
        session.finish.assert_not_called()
