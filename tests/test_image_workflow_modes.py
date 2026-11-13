from ruinedfooocus.core.paths import asset_path, resource_path

"""Image tool arbitration, preview ownership and early routing checks; no models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_powerup_family_controls import functions

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


class ImageWorkflowModeTests(unittest.TestCase):
    def test_depth_map_keeps_its_upload_and_hides_sampling_controls(self):
        scope, _ = functions()
        scope["controlnet"].cn_options["Depth map"] = {"type": "depth-map"}
        image, strength, mask, *_ = scope["input_context"](
            True,
            "Depth map",
            "Depth",
            "selected",
            None,
            "2× General (RealESRGAN)",
            "Input image only",
        )
        self.assertEqual(image, {"visible": True, "label": "Input image"})
        self.assertEqual(strength, {"visible": "hidden"})
        self.assertEqual(mask, {"visible": False, "value": False})
        self.assertIn(("Depth map", "Depth map"), scope["powerup_choices"](sdxl=False))

    def test_standalone_tools_disable_inpainting_without_clearing_the_upload(self):
        scope, _ = functions()
        scope["controlnet"].cn_options["Restore face"] = {"type": "upscale"}
        for selection in ("Remove background", "Restore face"):
            for preset, kind in (
                (selection, "Img2img"),
                ("New...", scope["controlnet"].cn_options[selection]["type"]),
            ):
                with self.subTest(selection=preset, kind=kind):
                    image, strength, toggle, options, rembg_force = scope["input_context"](
                        True, preset, kind, "selected"
                    )
                    self.assertEqual(toggle, {"visible": False, "value": False})
                    self.assertTrue(image["visible"])
                    self.assertNotIn("value", image)
                    self.assertEqual(strength["visible"], "hidden")
                    self.assertEqual(options["visible"], "hidden")
                    self.assertEqual(
                        rembg_force["visible"],
                        True
                        if scope["controlnet"].cn_options[selection]["type"] == "rembg"
                        else "hidden",
                    )

    def test_input_only_upscale_hides_ignored_controls_and_generated_upscale_keeps_inpaint(self):
        scope, _ = functions()
        for target, allowed in (("Input image only", False), ("Generated image", True)):
            values = scope["input_context"](
                True, "Img2Img", "Img2img", "selected", None, "2× General (RealESRGAN)", target
            )
            self.assertEqual(values[2]["visible"], allowed)
            self.assertEqual(values[3]["visible"], True if allowed else "hidden")
            if not allowed:
                self.assertFalse(values[0]["visible"])
                self.assertEqual(values[2]["value"], False)
        self.assertTrue(
            scope["input_context"](
                True, "None", "Img2img", "selected", None, "Off", "Input image only"
            )[2]["visible"]
        )

    def test_model_change_context_disables_unsupported_inpainting_and_keeps_control_guides(self):
        scope, models = functions()
        values = scope["input_context"](True, "Canny", "Canny", "selected")
        self.assertEqual(values[0], {"visible": True, "label": "Control guide"})
        self.assertTrue(values[2]["visible"])
        self.assertNotIn("value", values[2])
        models.detect_checkpoint_base.return_value = "QwenImage"
        values = scope["input_context"](True, "None", "Img2img", "selected")
        self.assertEqual(values[2], {"visible": False, "value": False})
        self.assertTrue(values[0]["visible"])

    def test_exactly_one_preview_is_visible_for_each_edit_inpaint_and_upscale_mode(self):
        run = function(
            "ruinedfooocus/ui/app.py",
            "switch_upscale_view",
            dict(gr=SimpleNamespace(update=lambda **kw: kw), t=lambda s: s),
        )
        for scale in ("Off", "2× General (RealESRGAN)"):
            for target in ("Generated image", "Input image only"):
                for inpaint in (False, True):
                    for editing in (False, True):
                        with self.subTest(
                            scale=scale, target=target, inpaint=inpaint, editing=editing
                        ):
                            result = run(scale, target, inpaint, editing)
                            self.assertEqual(
                                result[0]["visible"], "hidden" if scale == "Off" else True
                            )
                            self.assertEqual(
                                sum(update["visible"] is True for update in result[1:4]),
                                int(editing),
                            )
                            input_only = scale != "Off" and target == "Input image only"
                            self.assertEqual(
                                result[1]["visible"], editing and not input_only and not inpaint
                            )
                            self.assertEqual(
                                result[2]["visible"] is True, editing and not input_only and inpaint
                            )
                            self.assertEqual(result[3]["visible"], editing and input_only)
                            self.assertTrue(all("value" not in update for update in result[1:4]))
                            self.assertEqual(
                                result[4]["value"], "Upscale" if input_only else "Generate"
                            )
                            self.assertEqual(
                                result[5],
                                {"value": "Generated image"}
                                if scale == "Off" and target != "Generated image"
                                else {},
                            )

    def route(self, control, inpaint):
        state = {"pipeline": None}
        factories = {
            name: SimpleNamespace(pipeline=Mock(return_value=SimpleNamespace(pipeline_type=[kind])))
            for name, kind in (
                ("image_pipeline", "sdxl"),
                ("upscale_pipeline", "upscale"),
                ("rembg_pipeline", "rembg"),
                ("llama_pipeline", "llama"),
            )
        }
        models = SimpleNamespace(
            get_file=lambda *args: Path("model.safetensors"),
            get_models_by_path=lambda *args: {},
            get_model_base=lambda *args: "SDXL",
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            state=state,
            shared=SimpleNamespace(models=models),
            Path=Path,
            VIDEO_PIPELINES={},
            controlnet=SimpleNamespace(get_settings=lambda data: control),
            selected_pipeline=lambda: state["pipeline"],
            NoPipeLine=lambda: SimpleNamespace(pipeline_type=[]),
            **factories,
        )
        return (
            function("ruinedfooocus/features/generation/pipelines.py", "update", scope),
            state,
            factories,
        )

    def test_edit_shortcut_uses_session_input_instead_of_shared_global_state(self):
        toggle = function("ruinedfooocus/ui/app.py", "toggle_edit_mode", {})
        self.assertFalse(toggle(True))
        self.assertFalse(toggle(True))  # Another session starts independently.
        self.assertTrue(toggle(False))

    def test_incompatible_requests_fail_before_a_pipeline_or_model_is_created(self):
        for kind in ("upscale", "rembg", "RemBG"):
            run, state, factories = self.route({"type": kind}, True)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "inpainting"):
                run(
                    dict(
                        prompt="Repair the mask",
                        base_model_name="model.safetensors",
                        inpaint_toggle=True,
                    )
                )
            self.assertIsNone(state["pipeline"])
            for factory in factories.values():
                factory.pipeline.assert_not_called()

    def test_standalone_tools_and_inpaint_with_guides_keep_their_normal_routes(self):
        for kind, inpaint, expected in (
            ("upscale", False, "upscale"),
            ("rembg", False, "rembg"),
            ("canny", True, "sdxl"),
            ("depth", True, "sdxl"),
            ("img2img", True, "sdxl"),
            ("", True, "sdxl"),
        ):
            run, _, _ = self.route({"type": kind}, inpaint)
            with self.subTest(kind=kind, inpaint=inpaint):
                self.assertEqual(
                    run(
                        dict(
                            prompt="Repair the mask",
                            base_model_name="model.safetensors",
                            inpaint_toggle=inpaint,
                        )
                    ).pipeline_type,
                    [expected],
                )

    def test_image_mode_validation_does_not_block_the_chat_route(self):
        run, _, _ = self.route({"type": "rembg"}, True)
        self.assertEqual(run(dict(task_type="llama", inpaint_toggle=True)).pipeline_type, ["llama"])


if __name__ == "__main__":
    unittest.main()
