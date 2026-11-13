from ruinedfooocus.core.paths import asset_path, resource_path
from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary

"""Lazy session reuse must preserve model outputs and transparent-input behavior."""
import ast
import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock

import PIL.Image


class BackgroundSessionTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(
            Path("ruinedfooocus/features/image_processing/rembg_pipeline.py").read_text()
        )
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        self.rembg = Mock()
        self.worker = Mock()
        self.args = SimpleNamespace(cpu=False)
        self.identity = Mock(return_value="request-a")
        self.get_settings = Mock(
            side_effect=lambda data: data.get("controlnet") or data.get("preset_control", {})
        )
        namespace = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            "rembg": self.rembg,
            "worker": self.worker,
            "PIL": PIL,
            "Any": Any,
            "args": self.args,
            "force_cpu_for_auxiliary": force_cpu_for_auxiliary,
            "new_session": self.rembg.new_session,
            "session_request_identity": self.identity,
            "get_settings": self.get_settings,
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "rembg_pipeline.py", "exec"),
            namespace,
        )
        self.pipeline = namespace["pipeline"]()

    def test_repeated_requests_reuse_session_but_run_inference_for_each_image(self):
        first = PIL.Image.new("RGB", (4, 4), "red")
        second = PIL.Image.new("RGB", (4, 4), "blue")
        self.pipeline.remove_background(first)
        self.pipeline.remove_background(second)
        self.rembg.new_session.assert_called_once_with(cpu=False)
        self.assertEqual(self.rembg.remove.call_count, 2)
        self.assertIs(self.rembg.remove.call_args_list[0].args[0], first)
        self.assertIs(self.rembg.remove.call_args_list[1].args[0], second)
        self.assertIs(
            self.rembg.remove.call_args.kwargs["session"], self.rembg.new_session.return_value
        )

    def test_transparent_input_is_unchanged_without_loading_model_unless_forced(self):
        image = PIL.Image.new("RGBA", (4, 4), (255, 0, 0, 0))
        for source in (
            image,
            image.convert("LA"),
            image.convert("RGBa"),
            image.quantize(method=PIL.Image.Quantize.FASTOCTREE),
        ):
            with self.subTest(mode=source.mode):
                self.assertIs(self.pipeline.remove_background(source), source)
        self.rembg.new_session.assert_not_called()
        self.rembg.remove.assert_not_called()
        self.pipeline.remove_background(image, force=True)
        self.rembg.new_session.assert_called_once()

    def test_supplied_session_and_options_are_preserved(self):
        session = object()
        image = PIL.Image.new("RGB", (4, 4))
        self.pipeline.remove_background(image, rembg_session=session, alpha_matting=True)
        self.rembg.new_session.assert_not_called()
        self.rembg.remove.assert_called_once_with(image, session=session, alpha_matting=True)

    def test_provider_request_change_releases_old_session_before_loading_new(self):
        first, second = Mock(), Mock()
        self.rembg.new_session.side_effect = [first, second]
        self.pipeline.remove_background(PIL.Image.new("RGB", (2, 2)))
        self.identity.return_value = "request-b"
        self.pipeline.remove_background(PIL.Image.new("RGB", (2, 2)))
        first.close.assert_called_once_with()
        self.assertIs(self.pipeline._rembg_session, second)

    def test_explicit_cpu_mode_does_not_select_an_installed_gpu_provider(self):
        self.args.cpu = True
        self.pipeline.remove_background(PIL.Image.new("RGB", (4, 4)))
        self.rembg.new_session.assert_called_once_with(cpu=True)

    def test_image_only_cpu_keeps_background_provider_independent(self):
        self.args.cpu = True
        self.args._compute_status = {
            "backend": {"saved_image_backend": "cpu", "cpu_applied": True},
            "cli": [],
        }
        self.pipeline.remove_background(PIL.Image.new("RGB", (4, 4)))
        self.rembg.new_session.assert_called_once_with(cpu=False)

    def test_native_cutout_keeps_straight_rgb_and_source_pixels(self):
        import rembg

        self.rembg.remove.side_effect = rembg.remove
        source = PIL.Image.new("RGB", (3, 1), (200, 100, 50))
        mask = PIL.Image.new("L", source.size)
        mask.putdata([0, 128, 255])
        session = SimpleNamespace(predict=lambda *args, **kwargs: [mask])
        original = source.tobytes()
        result = self.pipeline.remove_background(source, rembg_session=session)
        self.assertEqual(result.size, source.size)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(list(result.getdata()), [(200, 100, 50, a) for a in (0, 128, 255)])
        self.assertEqual(source.tobytes(), original)
        background = PIL.Image.new("RGBA", source.size, "white")
        composite = PIL.Image.alpha_composite(background, result)
        self.assertEqual(composite.getpixel((1, 0)), (227, 177, 152, 255))

    def test_explicit_cutout_and_matting_options_keep_native_precedence(self):
        source = PIL.Image.new("RGB", (3, 1))
        session = object()
        for options in (
            {"putalpha": False},
            {"alpha_matting": True},
            {"vitmatte": True},
            {"decontaminate": True},
        ):
            with self.subTest(options=options):
                self.rembg.remove.reset_mock()
                self.pipeline.remove_background(source, rembg_session=session, **options)
                self.rembg.remove.assert_called_once_with(source, session=session, **options)

    def test_removal_reports_source_dimensions_without_resizing(self):
        image = PIL.Image.new("RGBA", (320, 240))
        data = self.pipeline.parse_gen_data(
            {"input_image": image, "image_number": 3, "custom_width": 1024, "custom_height": 1024}
        )
        self.assertEqual((data["width"], data["height"]), image.size)
        self.assertIs(data["input_image"], image)
        self.assertEqual(data["image_number"], 1)

    def test_cancelled_request_never_loads_session(self):
        self.worker.check_interrupt.side_effect = RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.pipeline.process({"task_id": 1, "input_image": PIL.Image.new("RGB", (4, 4))})
        self.rembg.new_session.assert_not_called()
        self.worker.add_result.assert_not_called()

    def test_missing_input_fails_before_progress_session_or_success_callback(self):
        for request in ({"task_id": 1, "input_image": None}, {"task_id": 1}):
            callback = Mock()
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, "image"):
                self.pipeline.process(request, callback)
            callback.assert_not_called()
        self.get_settings.assert_not_called()
        self.worker.add_result.assert_not_called()
        self.rembg.new_session.assert_not_called()
        self.rembg.remove.assert_not_called()

    def test_force_option_uses_explicit_api_then_ui_then_preset_defaults(self):
        image = PIL.Image.new("RGB", (4, 4))
        self.pipeline.remove_background = Mock(return_value=image)
        for extra, force in (
            ({}, False),
            ({"rembg_force": True}, True),
            ({"preset_control": {"force": True}}, True),
            ({"preset_control": {"force": True}, "rembg_force": False}, False),
            ({"controlnet": {"type": "rembg", "force": True}, "rembg_force": False}, True),
            ({"controlnet": {"type": "rembg", "force": False}, "rembg_force": True}, False),
        ):
            callback = Mock()
            with self.subTest(extra=extra):
                result = self.pipeline.process(
                    dict(task_id=1, input_image=image, **extra), callback
                )
                self.assertEqual(result, [image])
                self.assertIs(self.pipeline.remove_background.call_args.args[0], image)
                self.assertIs(self.pipeline.remove_background.call_args.kwargs["force"], force)
                callback.assert_called_once_with(1, 0, 0, 1, image)

    def test_model_errors_propagate_without_success_callback_and_clear_cancel_hook(self):
        image = PIL.Image.new("RGB", (4, 4))
        error = RuntimeError("mask model failed")
        self.rembg.remove.side_effect = error
        callback = Mock()
        with self.assertRaises(RuntimeError) as raised:
            self.pipeline.process({"task_id": 1, "input_image": image}, callback)
        self.assertIs(raised.exception, error)
        callback.assert_not_called()
        self.assertIsNone(self.rembg.new_session.return_value.check_cancelled)

    def test_api_can_preserve_png_alpha_without_changing_normal_rgb_inputs(self):
        from ruinedfooocus.app.api_runtime import decode_image
        from ruinedfooocus.integrations.openai_api import ImageRequest

        self.assertEqual(
            ImageRequest(input_image="base64", controlnet={"type": "rembg"}).controlnet.type,
            "rembg",
        )
        self.assertFalse(
            ImageRequest(input_image="base64", controlnet={"type": "rembg"}).controlnet.force
        )
        self.assertTrue(
            ImageRequest(
                input_image="base64", controlnet={"type": "rembg", "force": True}
            ).controlnet.force
        )
        image = PIL.Image.new("RGBA", (4, 4), (200, 100, 50, 42))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        self.assertEqual(decode_image(encoded).mode, "RGB")
        preserved = decode_image(encoded, preserve_alpha=True)
        self.assertEqual(preserved.mode, "RGBA")
        self.assertEqual(preserved.tobytes(), image.tobytes())


if __name__ == "__main__":
    unittest.main()
