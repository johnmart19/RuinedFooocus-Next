"""Native UnCLIP image guidance keeps L/H towers and negative conditioning distinct."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.image_editing.native_conditioning import unclip_conditioning
from ruinedfooocus.features.models.image_components import native_edit_supported, IMAGE_COMPONENTS
from test_native_model_coverage import native_method, PIPELINE, route_info
import test_model_cache as cache


class NativeUnclipTests(unittest.TestCase):
    def test_actual_missing_text_codec_fallback_keeps_bundled_vision(self):
        fixture = cache.ModelCacheTests("test_component_choices_reject_malformed_or_unknown_slots")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        checkpoint = fixture.file("checkpoint.safetensors")
        fixture.file("encoder.safetensors")
        fixture.file("vae.safetensors")
        load, owner, loader, _, _ = fixture.base_loader(checkpoint)
        unet = loader.return_value[0]
        vision = object()
        loader.return_value = (unet, None, None, vision)
        owner.load_base_model.__func__.__wrapped__.__globals__[
            "load_diffusion_model"
        ].return_value = unet
        load()
        self.assertIs(owner.xl_base.clip_vision, vision)
        self.assertTrue(loader.call_args.kwargs["output_clipvision"])

    def native_nodes(self):
        def values(conds, updates, append=False):
            return [
                [
                    tensor,
                    dict(
                        meta,
                        **{
                            key: meta.get(key, []) + value if append else value
                            for key, value in updates.items()
                        },
                    ),
                ]
                for tensor, meta in conds
            ]

        return NS(
            CLIPVisionEncode=native_method("nodes.py", "CLIPVisionEncode", "encode", {}),
            unCLIPConditioning=native_method(
                "nodes.py",
                "unCLIPConditioning",
                "apply_adm",
                dict(node_helpers=NS(conditioning_set_values=values)),
            ),
        )

    def test_actual_native_nodes_preserve_l_h_width_and_unmodified_text(self):
        for family, width in (("SD21UnclipL", 768), ("SD21UnclipH", 1024)):
            pixels = torch.zeros(1, 8, 8, 3)
            encoded = NS(image_embeds=torch.zeros(1, width))
            vision = NS(encode_image=Mock(return_value=encoded))
            model = NS(model_config=type(family, (), {})())
            original = [["text", {"pooled_output": "pooled", "start_percent": 0.25}]]
            with patch.dict("sys.modules", {"nodes": self.native_nodes()}):
                result = unclip_conditioning(original, model, vision, pixels)
            vision.encode_image.assert_called_once_with(pixels, crop=True)
            self.assertNotIn("unclip_conditioning", original[0][1])
            self.assertEqual(result[0][1]["start_percent"], 0.25)
            entry = result[0][1]["unclip_conditioning"][0]
            self.assertIs(entry["clip_vision_output"], encoded)
            self.assertEqual((entry["strength"], entry["noise_augmentation"]), (1, 0))

    def test_wrong_or_missing_vision_is_rejected_without_adm_attachment(self):
        model = NS(model_config=type("SD21UnclipL", (), {})())
        with patch.dict("sys.modules", {"nodes": self.native_nodes()}):
            with self.assertRaisesRegex(ValueError, "bundled vision"):
                unclip_conditioning([], model, None, None)
            wrong = NS(encode_image=lambda *a, **k: NS(image_embeds=torch.zeros(1, 1024)))
            with self.assertRaisesRegex(ValueError, "width"):
                unclip_conditioning([], model, wrong, None)

    def test_actual_pipeline_branch_guides_positive_schedule_only(self):
        tree = ast.parse(Path(PIPELINE).read_text(encoding="utf-8"))
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and ast.unparse(n.test)
            == "unclip_model and input_image is not None and (control_type == 'edit')"
        )
        guide = Mock(return_value="guided schedule")
        scope = dict(
            unclip_model=True,
            input_image="source",
            control_type="edit",
            self=NS(xl_base_patched=NS(unet=NS(model="model"), clip_vision="vision")),
            switched_prompt="schedule",
            positive_conditioning="positive",
            negative_conditioning="negative",
            input_images=1,
        )
        with patch(
            "ruinedfooocus.features.image_editing.native_conditioning.unclip_conditioning", guide
        ):
            exec(compile(ast.Module(body=[branch], type_ignores=[]), PIPELINE, "exec"), scope)
        self.assertEqual(scope["switched_prompt"], "guided schedule")
        self.assertEqual(scope["negative_conditioning"], "negative")
        self.assertEqual(scope["input_images"], 0)
        guide.assert_called_once_with("schedule", "model", "vision", "source")

    def test_route_and_loader_request_native_bundled_vision(self):
        self.assertIn("SD21UNCLIP", route_info())
        self.assertEqual(IMAGE_COMPONENTS["SD21UNCLIP"]["clip_names"], ["clip_sd4x_h"])
        self.assertTrue(native_edit_supported("SD21UnclipL"))
        self.assertTrue(native_edit_supported("SD21UnclipH"))
        tree = ast.parse(Path(PIPELINE).read_text(encoding="utf-8"))
        call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "load_checkpoint_guess_config"
        )
        self.assertTrue(
            ast.literal_eval(next(k.value for k in call.keywords if k.arg == "output_clipvision"))
        )


if __name__ == "__main__":
    unittest.main()
