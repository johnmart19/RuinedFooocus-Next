"""GPU resets require a fresh device context; OOM and network timeouts do not."""

import unittest

from ruinedfooocus.core.task_errors import describe_error


class TaskErrorTests(unittest.TestCase):
    def test_fatal_cuda_hip_and_directml_failures_request_an_app_restart(self):
        for message in (
            "CUDA error: unspecified launch failure",
            "hipErrorLaunchFailure",
            "hipErrorIllegalAddress",
            "CUDA error: an illegal memory access was encountered",
            "CUDA error: device-side assert triggered",
            "DXGI_ERROR_DEVICE_REMOVED",
            "DXGI_ERROR_DEVICE_HUNG",
        ):
            with self.subTest(message=message):
                result = describe_error(RuntimeError(message))
                self.assertIn("Restart RuinedFooocus before generating again.", result)
                self.assertIn(message, result)
                self.assertNotIn("You can retry", result)

    def test_oom_diagnostic_footer_does_not_misclassify_it_as_a_device_assert(self):
        error = RuntimeError(
            "CUDA out of memory. Device-side assertion tracking was not enabled by user."
        )
        result = describe_error(error)
        self.assertIn("not enough memory", result)
        self.assertNotIn("Restart", result)

    def test_cpu_and_network_errors_keep_their_actionable_messages(self):
        self.assertIn("not enough memory", describe_error(MemoryError("out of memory")))
        self.assertIn("connection and retry", describe_error(TimeoutError("Connection timed out")))
        self.assertIn("choose another model", describe_error(RuntimeError("unsupported model")))

    def test_validation_errors_show_their_actionable_message_without_model_retry_advice(self):
        for message in (
            "Add an image in Image tools to remove its background.",
            "Image-to-image denoise must be greater than 0 and at most 1.",
            "Unsupported image architecture: unknown",
        ):
            with self.subTest(message=message):
                self.assertEqual(describe_error(ValueError(message)), message)
        self.assertIn("Restart RuinedFooocus", describe_error(ValueError("hipErrorLaunchFailure")))


if __name__ == "__main__":
    unittest.main()
