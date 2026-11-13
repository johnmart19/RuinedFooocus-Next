"""Hardware controls distinguish requested intent, local catalogue and execution."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ui_test_helpers import function_from
from ruinedfooocus.core.translation_manager import localize, resolve_locale

SOURCE = "ruinedfooocus/ui/ui_hardware.py"


class HardwareProviderUiTests(unittest.TestCase):
    def choices(self):
        return function_from(SOURCE, "windows_ml_choices", {"t": lambda text: text})

    def test_only_certified_bridge_compatible_local_candidates_are_offered(self):
        rows = [
            {
                "name": "MIGraphXExecutionProvider",
                "certified": True,
                "bridge_compatible": True,
                "device_types": ["GPU"],
            },
            {
                "name": "AMDGPUExecutionProvider",
                "certified": True,
                "bridge_compatible": False,
                "device_types": ["GPU"],
            },
            {
                "name": "VitisAIExecutionProvider",
                "certified": True,
                "bridge_compatible": True,
                "device_types": ["NPU"],
            },
            {
                "name": "OpenVINOExecutionProvider",
                "certified": False,
                "bridge_compatible": True,
                "device_types": ["CPU", "GPU", "NPU"],
            },
        ]
        choices = self.choices()(rows, [{"vendor": "amd", "name": "AMD GPU"}])
        self.assertEqual([value for _, value in choices], ["Auto", "MIGraphXExecutionProvider"])
        remembered = self.choices()(rows, [], "AMDGPUExecutionProvider")
        self.assertIn("Unavailable saved choice", remembered[-1][0])
        self.assertEqual(remembered[-1][1], "AMDGPUExecutionProvider")
        self.assertEqual(self.choices()([], []), [("Automatic qualified provider", "Auto")])

    def test_cpu_class_candidates_do_not_include_gpu_or_npu_only_providers(self):
        rows = [
            {
                "name": "OpenVINOExecutionProvider",
                "certified": True,
                "bridge_compatible": True,
                "device_types": ["CPU", "GPU", "NPU"],
            },
            {
                "name": "MIGraphXExecutionProvider",
                "certified": True,
                "bridge_compatible": True,
                "device_types": ["GPU"],
            },
        ]
        self.assertEqual(
            [value for _, value in self.choices()(rows, [], device_type="CPU")],
            ["Auto", "OpenVINOExecutionProvider"],
        )

    def test_cpu_authority_never_submits_accelerated_discovery_or_install(self):
        gr = SimpleNamespace(update=lambda **kwargs: kwargs, skip=lambda: "skip", Request=object)
        scope = {
            "gr": gr,
            "auxiliary_cpu": lambda requested: requested,
            "skip_dependency_install": lambda: False,
            "localize": localize,
            "resolve_locale": resolve_locale,
        }
        session = SimpleNamespace(run_admitted=Mock(side_effect=AssertionError("No GPU work")))
        refresh = function_from(SOURCE, "refresh_winml_devices", dict(scope))
        self.assertIn("no GPU/NPU", refresh("Auto", "Auto", "GPU", True, "Auto", session)[1])
        self.assertIn("no GPU/NPU", refresh("Auto", "Auto", "GPU", False, "CPU", session)[1])
        install = function_from(SOURCE, "install_winml_provider", dict(scope))
        self.assertIn(
            "no provider acquisition", list(install("Auto", "GPU", True, "Auto", session))[0][1]
        )
        session.run_admitted.assert_not_called()

    def test_automatic_device_class_does_not_infer_npu_from_vendor(self):
        validate = function_from(
            SOURCE, "validate_windows_ml_choice", {"windows_ml_choices": self.choices()}
        )
        self.assertEqual(validate("Auto", "Auto", [{"vendor": "amd"}]), "GPU")
        self.assertEqual(validate("Auto", "Auto", []), "CPU")
        self.assertEqual(validate("Auto", "NPU", []), "NPU")
