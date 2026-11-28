"""Runtime buttons accept offered modes without acquiring unrelated providers."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ui_test_helpers import function_from
from ruinedfooocus.backends.onnx import onnx_installer, windows_ml
from ruinedfooocus.core.translation_manager import localize, resolve_locale
from ruinedfooocus.packaging import gpu_installer

SOURCE = "ruinedfooocus/ui/ui_hardware.py"


class RuntimeButtonUiTests(unittest.TestCase):
    def setUp(self):
        self.check = Mock()
        self.session = SimpleNamespace(run_admitted=Mock(side_effect=lambda fn: fn(self.check)))
        self.scope = {
            "gr": gr,
            "locale_of": lambda request: "en",
            "resolve_locale": resolve_locale,
            "localize": localize,
            "auxiliary_cpu": bool,
            "skip_dependency_install": lambda: False,
            "background_values": ["Auto", "CPU", "CUDA", "MIGraphX", "OpenVINO", "Windows GPU"],
            "background_status_text": Mock(return_value="Runtime status"),
            "gpus": [{"vendor": "amd"}],
            "validate_windows_ml_choice": lambda provider, kind, gpus: kind,
            "provider_state": lambda *args, **kwargs: ({"choices": ["Auto"]}, "status"),
        }
        install_winml = function_from(SOURCE, "install_winml_provider", self.scope)
        self.scope["winml_controls"] = {"install": install_winml}
        self.install = function_from(SOURCE, "install_selected_background", self.scope)

    def test_auto_cpu_and_cpu_authority_are_status_checks_without_acquisition(self):
        for choice, forced in (("Auto", False), ("CPU", False), ("Windows GPU", True)):
            with self.subTest(choice=choice, forced=forced):
                result = list(self.install(choice, forced, self.session))
                self.assertEqual(result[-1][1], "Runtime status")
                self.assertTrue(result[-1][0]["interactive"])
                self.scope["background_status_text"].assert_called_with("en", forced, choice)
        self.session.run_admitted.assert_not_called()

    def test_llama_auto_returns_first_load_guidance_without_starting_an_installer(self):
        scope = dict(self.scope, llama_installer=SimpleNamespace(canonical_backend=lambda c: c))
        install = function_from(SOURCE, "install_selected_llama", scope)
        result = list(install("Auto", None))
        self.assertIn("first Chat load", result[-1][1])
        self.assertTrue(result[-1][0]["interactive"])

    def test_chat_runtime_offline_checks_report_policy_without_installing(self):
        scope = dict(
            self.scope,
            skip_dependency_install=lambda: True,
            llama_installer=SimpleNamespace(
                canonical_backend=lambda c: c, validate_cuda_backend=Mock()
            ),
            llama_values=["Auto", "CPU", "Vulkan"],
            llama_installer_installed=lambda: [],
            xllama_values=["Auto", "CPU", "Vulkan"],
        )
        llama = function_from(SOURCE, "install_selected_llama", scope)
        xllama = function_from(SOURCE, "install_selected_xllama", scope)
        with patch.object(gpu_installer, "ensure_xllamacpp") as acquire:
            for result in (list(llama("CPU", None)), list(xllama("Auto", False, None))):
                self.assertIn("disabled", result[-1][1])
                self.assertTrue(result[-1][0]["interactive"])
        acquire.assert_not_called()

    def test_native_modes_use_only_their_isolated_installer_and_checker(self):
        for choice in ("CUDA", "MIGraphX", "OpenVINO"):
            with (
                self.subTest(choice=choice),
                patch.object(
                    onnx_installer, "install_native_runtime", return_value="Installed fixture"
                ) as install,
            ):
                result = list(self.install(choice, False, self.session))
                install.assert_called_once_with(choice, check_cancelled=self.check)
                self.assertFalse(result[0][0]["interactive"])
                self.assertTrue(result[-1][0]["interactive"])
                self.scope["background_status_text"].assert_called_with("en", False, choice)

    def test_windows_ml_uses_selected_provider_device_and_same_admission(self):
        with (
            patch.object(windows_ml, "install_provider", return_value="Provider ready") as install,
            patch.object(windows_ml, "refresh_catalog") as refresh,
            patch.object(onnx_installer, "install_native_runtime") as native,
        ):
            result = list(
                self.install("Windows GPU", False, self.session, "MIGraphXExecutionProvider", "GPU")
            )
        install.assert_called_once_with(
            "MIGraphXExecutionProvider", "GPU", check_cancelled=self.check
        )
        refresh.assert_called_once_with(check_cancelled=self.check)
        native.assert_not_called()
        self.assertEqual(result[-1][1], "Provider ready")
        self.assertTrue(result[-1][0]["interactive"])

    def test_offline_modes_return_status_without_installing_or_raising(self):
        self.scope["skip_dependency_install"] = lambda: True
        for choice in ("CUDA", "Windows GPU"):
            with self.subTest(choice=choice):
                result = list(self.install(choice, False, self.session))
                self.assertIn("disabled", result[-1][1])
                self.assertTrue(result[-1][0]["interactive"])
        self.session.run_admitted.assert_not_called()

    def test_provider_failure_restores_button_and_reports_reason(self):
        with patch.object(windows_ml, "install_provider", side_effect=RuntimeError("fixture")):
            result = list(self.install("Windows GPU", False, self.session, "Auto", "GPU"))
        self.assertIn("fixture", result[-1][1])
        self.assertTrue(result[-1][0]["interactive"])

    def test_unavailable_backend_does_not_start_provider_work(self):
        self.scope["background_values"] = ["Auto", "CPU"]
        with self.assertRaisesRegex(gr.Error, "not available"):
            list(self.install("Windows GPU", False, self.session))
        self.session.run_admitted.assert_not_called()

    def test_windows_controls_expose_the_same_live_inputs_and_install_callback(self):
        settings = {}
        scope = dict(
            self.scope,
            t=lambda text: text,
            add_setting=lambda key, control: settings.update({key: control}),
            localized_output=lambda *args, **kwargs: None,
            windows_ml_choices=lambda *args: [("Automatic", "Auto")],
            refresh_support_table=lambda: "Support",
        )
        create = function_from(SOURCE, "create_windows_ml_controls", scope)
        with patch.object(
            windows_ml, "catalog_status", return_value={"reason": "Local", "providers": []}
        ):
            with gr.Blocks() as app:
                controls = create(
                    {},
                    [],
                    gr.Markdown(),
                    gr.Checkbox(),
                    gr.Dropdown(choices=["Windows GPU"]),
                    gr.State("owned"),
                )
        self.assertIs(controls["provider"], settings["rembg_winml_provider"])
        self.assertIs(controls["device_type"], settings["rembg_winml_type"])
        install = next(fn for fn in app.fns.values() if fn.fn is controls["install"])
        self.assertEqual(install.inputs[:2], [controls["provider"], controls["device_type"]])

    def test_gradio_keeps_live_provider_inputs_and_request_injection(self):
        seen = []

        def install(provider, kind, cpu, choice, session, request):
            seen.append((provider, kind, cpu, choice, session, request.session_hash))
            yield gr.update(interactive=True), "Ready", gr.skip()

        self.scope["winml_controls"] = {"install": install}
        with gr.Blocks() as app:
            choice = gr.Dropdown(choices=["Windows GPU"])
            cpu = gr.Checkbox()
            session = gr.Textbox()
            provider = gr.Textbox()
            kind = gr.Textbox()
            button = gr.Button()
            status = gr.Markdown()
            button.click(
                self.install,
                inputs=[choice, cpu, session, provider, kind],
                outputs=[button, status],
            )
        fn = next(iter(app.fns.values()))

        async def click():
            result = await app.process_api(
                block_fn=fn,
                inputs=["Windows GPU", False, "owned", "MIGraphXExecutionProvider", "GPU"],
                request=gr.Request(session_hash="button-fixture"),
            )
            with self.assertRaises(StopAsyncIteration):
                await anext(result["iterator"])
            return result

        result = asyncio.run(click())
        self.assertEqual(result["data"][1], "Ready")
        self.assertEqual(
            seen,
            [("MIGraphXExecutionProvider", "GPU", False, "Windows GPU", "owned", "button-fixture")],
        )
