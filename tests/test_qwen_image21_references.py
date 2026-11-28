"""Exercise Qwen Image 2.1 reference conditioning through the pinned node."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame
from ruinedfooocus.features.image_editing.image_guidance import normalize_reference_images
from ruinedfooocus.features.image_editing.qwen_image21_conditioning import QwenImage21References
from test_api_image_edits import queued_data
from test_qwen_image21_conditioning import MODULE, native_function, runtime


def native_references():
    modules, upscale = runtime()
    node = modules["comfy_extras.nodes_qwen"].TextEncodeQwenImage21
    node.execute.__func__.__globals__["node_helpers"] = SimpleNamespace(
        conditioning_set_values=native_function("node_helpers.py", "conditioning_set_values", {})
    )
    return modules, upscale


def vae_fixture():
    return SimpleNamespace(
        latent_channels=64,
        latent_dim=2,
        output_channels=4,
        spacial_compression_encode=lambda: 16,
        spacial_compression_decode=lambda: 16,
        encode=Mock(
            side_effect=lambda pixels: torch.zeros(
                (1, 64, pixels.shape[1] // 16, pixels.shape[2] // 16)
            )
        ),
    )


def clip_fixture():
    return SimpleNamespace(
        tokenize=Mock(side_effect=lambda text, **values: dict(text=text, **values)),
        encode_from_tokens_scheduled=Mock(
            side_effect=lambda tokens: [
                [tokens["text"], {"image_slots": list(range(len(tokens["images"])))}]
            ]
        ),
    )


class QwenImage21ReferencesTests(unittest.TestCase):
    def test_rgba_source_uses_native_first_reference_grid_and_preserves_exact_frame(self):
        source = Image.new("RGBA", (81, 67), (40, 90, 170, 64))
        original = source.tobytes()
        vae, clip = vae_fixture(), clip_fixture()
        refs = QwenImage21References(vae, source, [], (128, 128), "preserve")
        modules, upscale = native_references()
        with (
            patch.dict("sys.modules", modules),
            patch(MODULE + ".warm_patched_text_encoder") as warm,
        ):
            positive, negative, latent = refs.conditioning(clip, "change it", "")
        warm.assert_called_once_with(clip)
        upscale.assert_not_called()
        self.assertEqual((refs.frame["width"], refs.frame["height"]), source.size)
        self.assertEqual(latent["samples"].shape, (1, 64, 6, 6))
        self.assertFalse(latent["samples"].any())
        pixels = vae.encode.call_args.args[0]
        self.assertEqual(pixels.shape, (1, 96, 96, 4))
        encoded = restore_reference_frame(pixels, refs.frame)
        self.assertEqual(encoded.shape, (1, 67, 81, 4))
        torch.testing.assert_close(encoded[0, 0, 0], torch.tensor(source.getpixel((0, 0))) / 255)
        self.assertEqual(source.tobytes(), original)
        for call in clip.tokenize.call_args_list:
            self.assertFalse(call.kwargs["keep_vision"])
            self.assertTrue(call.kwargs["prevent_empty_text"])
            rgb = call.kwargs["images"][0]
            torch.testing.assert_close(rgb, pixels[..., :3] * pixels[..., 3:] + 1 - pixels[..., 3:])
        self.assertEqual((positive[0][0], negative[0][0]), ("change it", ""))
        self.assertEqual(positive[0][1]["image_slots"], [0])
        self.assertIs(
            positive[0][1]["reference_latents"][0], negative[0][1]["reference_latents"][0]
        )

    def test_sixteen_reference_order_and_request_local_vae_cache_across_prompt_switches(self):
        primary = Image.new("RGBA", (64, 32), (1, 0, 0, 255))
        additional = [Image.new("RGBA", (32, 64), (value, 0, 0, 255)) for value in range(2, 17)]
        vae, clip = vae_fixture(), clip_fixture()
        refs = QwenImage21References(vae, primary, additional, (64, 64), "preserve")
        modules, upscale = native_references()
        with patch.dict("sys.modules", modules), patch(MODULE + ".warm_patched_text_encoder"):
            first = refs.conditioning(clip, "first", "")
            switched = refs.conditioning(clip, "second", "")
            separate = QwenImage21References(vae, primary, [], (64, 64), "preserve")
            next_job = separate.conditioning(clip, "next job", "")
        upscale.assert_not_called()
        self.assertEqual(vae.encode.call_count, 17)
        for call in clip.tokenize.call_args_list[:4]:
            reds = [round(pixel[0, 0, 0, 0].item() * 255) for pixel in call.kwargs["images"]]
            self.assertEqual(reds, list(range(1, 17)))
        first_refs = first[0][0][1]["reference_latents"]
        switch_refs = switched[0][0][1]["reference_latents"]
        self.assertEqual(len(first_refs), 16)
        self.assertTrue(all(before is after for before, after in zip(first_refs, switch_refs)))
        self.assertIsNot(first_refs[0], next_job[0][0][1]["reference_latents"][0])
        self.assertEqual(first[2]["samples"].shape, (1, 64, 2, 4))

    def test_explicit_size_and_exif_orientation_keep_alpha_through_framing(self):
        source = Image.new("RGBA", (96, 48), (4, 9, 20, 128))
        source.getexif()[274] = 6
        for mode in ("pad", "crop"):
            with self.subTest(mode=mode):
                refs = QwenImage21References(vae_fixture(), source, [], (101, 79), mode)
                self.assertEqual(
                    (refs.frame["source_width"], refs.frame["source_height"]), (48, 96)
                )
                pixels = restore_reference_frame(refs.images["image_1"], refs.frame)
                self.assertEqual(pixels.shape, (1, 79, 101, 4))
                self.assertAlmostEqual(pixels[0, 39, 50, 3].item(), 128 / 255)
                if mode == "pad":
                    self.assertEqual(pixels[0, 39, 0, 3], 0)

    def test_reference_contract_rejects_wrong_codec_or_text_encoder_and_invalid_inputs(self):
        source = Image.new("RGBA", (64, 64), (0, 0, 0, 128))
        vae = vae_fixture()
        for primary, extras, message in (
            (None, [], "must be an image"),
            (source, [source] * 16, "at most 16"),
            (Image.new("RGB", (4097, 1)), [], "4096 pixels"),
            (source, ["/arbitrary/reference.png"], "must be an image"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                QwenImage21References(vae, primary, extras, (128, 128), "preserve")
        vae.encode.assert_not_called()
        with (
            patch.object(vae, "output_channels", 3),
            self.assertRaisesRegex(ValueError, "RGBA VAE"),
        ):
            QwenImage21References(vae, source, [], (64, 64), "preserve")

        clip = clip_fixture()
        clip.encode_from_tokens_scheduled.side_effect = lambda tokens: [[tokens["text"], {}]]
        refs = QwenImage21References(vae, source, [], (64, 64), "preserve")
        modules, _ = native_references()
        with (
            patch.dict("sys.modules", modules),
            patch(MODULE + ".warm_patched_text_encoder"),
            self.assertRaisesRegex(ValueError, "required image slots"),
        ):
            refs.conditioning(clip, "change it", "")

    def test_shared_reference_normalization_keeps_alpha_only_for_qwen_image21(self):
        source = Image.new("RGB", (64, 64))
        extra = Image.new("RGBA", (64, 64), (120, 100, 20, 74))
        qwen = normalize_reference_images([(extra, "caption")], source, "QwenImage21")
        self.assertEqual(qwen[0].getpixel((0, 0)), extra.getpixel((0, 0)))
        self.assertEqual(normalize_reference_images([extra], source, "Fibo2")[0].mode, "RGB")
        self.assertEqual(normalize_reference_images([extra], source, "QwenImage")[0].mode, "RGB")
        with self.assertRaisesRegex(ValueError, "supported native edit"):
            normalize_reference_images([extra], source, "SDXL")

    def test_pipeline_uses_native_reference_latent_and_keeps_scheduled_conditions_local(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "ruinedfooocus/features/generation/image_pipeline.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        pair = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "textencode_pair"
        )
        branches = {
            ast.unparse(node.test): node for node in ast.walk(process) if isinstance(node, ast.If)
        }
        prepare = branches["qwen_reference_edit"]
        encode = branches["image_edit and input_images > 0"].orelse
        select_latent = branches["edit_latent is not None and (not img2img_mode)"]
        vae, clip = vae_fixture(), clip_fixture()
        owner = SimpleNamespace(
            conditions=None, xl_base_patched=SimpleNamespace(vae=vae, clip=clip)
        )
        primary = Image.new("RGBA", (81, 67), (40, 90, 170, 64))
        extra = Image.new("RGBA", (64, 32), (0, 80, 20, 230))
        scope = dict(
            self=owner,
            tensor_conditions={},
            path_manager=SimpleNamespace(
                model_paths={"embedding_path": [], "cache_path": source.parent}
            ),
            architecture="QwenImage21",
            qwen_reference_edit=True,
            gen_data=dict(width=104, height=80, steps=3),
            input_image=primary,
            additional_references=[extra],
            guidance={"image_resize_mode": "pad"},
            callback=None,
            clip_skip=1,
            positive_prompt="red",
            negative_prompt="",
            img2img_mode=False,
            clean_prompt_cond_caches=lambda: {
                key: {"text": None, "cache": None} for key in ("+", "-", "switch")
            },
            pp=SimpleNamespace(prompt_switch_per_step=lambda *_: ["red", "blue", "red"]),
            conditioning_set_values=lambda cond, values: [
                [text, metadata | values] for text, metadata in cond
            ],
            switched_prompt=[],
        )

        def run(nodes):
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)

        run([pair])
        owner.textencode_pair = MethodType(scope[pair.name], owner)
        modules, _ = native_references()
        with patch.dict("sys.modules", modules), patch(MODULE + ".warm_patched_text_encoder"):
            run([prepare, *encode, select_latent])
        self.assertEqual(vae.encode.call_count, 2)
        self.assertEqual([row[0] for row in scope["switched_prompt"]], ["red", "blue", "red"])
        self.assertTrue(
            all(len(row[1]["reference_latents"]) == 2 for row in scope["switched_prompt"])
        )
        self.assertIsNone(scope["denoise"])
        self.assertEqual(scope["latent"]["samples"].shape, (1, 64, 6, 8))
        frame = scope["reference_frame"]
        self.assertEqual((frame["width"], frame["height"]), (104, 80))
        self.assertEqual((scope["gen_data"]["width"], scope["gen_data"]["height"]), (128, 96))
        self.assertIsNone(owner.conditions["+"]["cache"])
        self.assertIsNone(owner.conditions["-"]["cache"])

    def test_api_qwen21_default_edit_freezes_alpha_and_keeps_explicit_img2img_separate(self):
        source = Image.new("RGBA", (81, 67), (40, 90, 170, 64))

        def passthrough(data, *_, **__):
            return data

        with (
            patch(
                "ruinedfooocus.features.models.component_profiles.snapshot_request",
                side_effect=passthrough,
            ),
            patch(
                "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                side_effect=passthrough,
            ),
        ):
            decode_calls = []
            data = queued_data(
                "QwenImage21",
                source_image=source,
                decode_calls=decode_calls,
                payload_options={"size": "104x80", "reference_images": ["extra"]},
            )
            self.assertEqual(data["controlnet"], {"type": "edit"})
            self.assertEqual(data["image_resize_mode"], "pad")
            self.assertEqual((data["custom_width"], data["custom_height"]), (104, 80))
            self.assertEqual([row["preserve_alpha"] for row in decode_calls], [True, True])
            self.assertEqual(data["input_image"].getpixel((0, 0)), source.getpixel((0, 0)))
            self.assertEqual(data["reference_images"][0].getpixel((0, 0)), source.getpixel((0, 0)))
            self.assertIsNot(data["reference_images"][0], source)

            ordinary = queued_data("QwenImage21", controlnet={"type": "img2img", "denoise": 0.42})
            self.assertEqual(ordinary["controlnet"], {"type": "img2img", "denoise": 0.42})
            with self.assertRaisesRegex(ValueError, "native image edit operation"):
                queued_data(
                    "QwenImage21",
                    controlnet={"type": "img2img"},
                    payload_options={"reference_images": ["extra"]},
                )
            with self.assertRaisesRegex(ValueError, "supported native edit workflow"):
                queued_data("SDXL", payload_options={"reference_images": ["extra"]})
            plus = queued_data(
                "QwenImage",
                payload_options={
                    "reference_images": ["extra"],
                    "native_edit_recipe": "qwen-edit-2511",
                },
            )
            self.assertEqual(plus["native_edit_recipe"], "qwen-edit-2511")
