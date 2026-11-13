"""Saved engine choices must stay distinct from native device identities."""

import unittest

from ruinedfooocus.core.settings_schema import validate_known


class ImageEngineSettingsTests(unittest.TestCase):
    def test_valid_choices_preserve_explicit_intent_without_inserting_defaults(self):
        self.assertEqual(validate_known({}), {})
        for engine in ("auto", "native", "sdcpp-vulkan", "sdcpp-cpu"):
            values = {
                "image_engine": engine,
                "image_backend": "rocm",
                "sdcpp_vulkan_device": "Vulkan12",
                "cpu_everywhere": True,
            }
            self.assertEqual(validate_known(values), values)

    def test_invalid_engine_does_not_silently_become_native_or_cpu(self):
        for engine in (None, "Vulkan", "directml", "automatic", False):
            with self.subTest(engine=engine), self.assertRaisesRegex(ValueError, "image_engine"):
                validate_known({"image_engine": engine})

    def test_backend_ordinals_are_not_torch_or_physical_adapter_ids(self):
        for device in (None, 0, "0", "cuda:0", "Vulkan-1", "Vulkan0\n", "Vulkan 0", "Auto"):
            with (
                self.subTest(device=device),
                self.assertRaisesRegex(ValueError, "sdcpp_vulkan_device"),
            ):
                validate_known({"sdcpp_vulkan_device": device})


if __name__ == "__main__":
    unittest.main()
