"""Production source/hint/VAE branches with actual native conditioning, no weights."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ruinedfooocus.features.generation.size_plan import SizePlan
from ruinedfooocus.features.image_editing import guide_handoffs as guides, native_control_recipes
from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame
import test_guide_handoffs as handoff_fixtures
import test_image_controlnet_switch as control_fixtures

ROOT = Path(__file__).resolve().parents[1]


def execute(nodes, scope):
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(ROOT / "ruinedfooocus/features/generation/image_pipeline.py"),
            "exec",
        ),
        scope,
    )


class GuidedPipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = handoff_fixtures.GuideHandoffTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.native_fixture = control_fixtures.ControlNetSwitchTests()
        self.native_fixture.setUp()
        self.addCleanup(self.native_fixture.doCleanups)
        _, _, self.handoff = self.fixture.adopt()
        path = ROOT / "ruinedfooocus/features/generation/image_pipeline.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        self.reference = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.If)
            and any(
                isinstance(part, ast.Name) and part.id == "guided_edit"
                for part in ast.walk(node.test)
            )
            and any(
                isinstance(part, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "requested_size"
                    for target in part.targets
                )
                for part in node.body
            )
        )
        wanted = ast.parse("controlnet['type'] != 'None' and input_images > 0", mode="eval").body
        self.control = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(wanted)
        )
        self.latent = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "img2img_mode"
        )
        self.alpha = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "guided_edit"
            and any(
                isinstance(part, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "decoded_latent"
                    for target in part.targets
                )
                for part in node.body
            )
        )
        native = ast.parse((ROOT / "repositories/ComfyUI/nodes.py").read_text(encoding="utf-8"))
        node = next(
            row for row in native.body if isinstance(row, ast.ClassDef) and row.name == "VAEEncode"
        )
        namespace = {}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "pinned-native-VAEEncode", "exec"),
            namespace,
        )
        self.vae_node = namespace["VAEEncode"]

    def scope(self, denoise=0.35, target=(7, 5), mode="preserve"):
        captured, _ = self.fixture.capture(self.handoff, control_guide_denoise=denoise)
        request = captured[guides.KEY]
        encoded = []

        class VAE:
            def encode(self, pixels):
                encoded.append(pixels.clone())
                return torch.zeros((1, 4, pixels.shape[1] // 8, pixels.shape[2] // 8))

            def spacial_compression_decode(self):
                return 8

        copies = []
        control = control_fixtures.Control(copies)
        owner = NS(
            xl_controlnet=control,
            xl_base_patched=NS(vae=VAE()),
            refresh_controlnet=lambda **kwargs: None,
        )
        captured.update(width=target[0], height=target[1], image_resize_mode=mode)
        scope = dict(
            self=owner,
            gen_data=captured,
            guided_request=request,
            guided_edit=True,
            guide_handoffs=guides,
            input_image=guides.request_image(request.source),
            guidance=dict(image_resize_mode=mode),
            size_plan=None,
            SizePlan=SizePlan,
            architecture="SDXL",
            input_images=1,
            callback=None,
            controlnet=request.controls(),
            control_guide=None,
            control_source_alpha=None,
            native_control_recipes=native_control_recipes,
            named_source_mask=None,
            positive_conditioning=[[torch.zeros(1), {}]],
            negative_conditioning=[[torch.zeros(1), {}]],
            switched_prompt=[],
            img2img_mode=False,
            ControlNetApplyAdvanced=self.native_fixture.native,
            VAEEncode=self.vae_node,
            worker=NS(check_interrupt=lambda data: None),
            torch=torch,
            np=np,
        )
        return scope, encoded, copies

    def test_actual_control_branch_consumes_guide_and_native_vae_encodes_original(self):
        for denoise in (0, 0.35, 1):
            with self.subTest(denoise=denoise):
                scope, encoded, copies = self.scope(denoise)
                execute(self.reference.body, scope)
                original, guide = scope["input_image"].clone(), scope["control_guide"].clone()
                self.assertFalse(torch.equal(original, guide))
                execute([self.control], scope)
                self.assertTrue(scope["img2img_mode"])
                self.assertTrue(copies)
                torch.testing.assert_close(copies[0].hint, guide.movedim(-1, 1), rtol=0, atol=0)
                execute(self.latent.body, scope)
                self.assertEqual(len(encoded), 1)
                torch.testing.assert_close(encoded[0], original, rtol=0, atol=0)
                self.assertFalse(torch.equal(encoded[0], guide))
                self.assertEqual(scope["denoise"], denoise)
                self.assertEqual(scope["gen_data"]["control_guide"]["denoise"], denoise)

    def test_actual_source_and_hint_share_override_geometry_and_final_alpha(self):
        scope, _, copies = self.scope(target=(13, 17), mode="pad")
        execute(self.reference.body, scope)
        self.assertEqual(scope["input_image"].shape, scope["control_guide"].shape)
        self.assertEqual(
            (scope["reference_frame"]["content_width"], scope["reference_frame"]["content_height"]),
            (13, 9),
        )
        execute([self.control], scope)
        self.assertEqual(tuple(copies[0].hint.shape[-2:]), tuple(scope["input_image"].shape[1:3]))
        scope["decoded_latent"] = restore_reference_frame(
            scope["input_image"], scope["reference_frame"]
        )
        execute([self.alpha], scope)
        expected = restore_reference_frame(scope["control_source_alpha"], scope["reference_frame"])
        torch.testing.assert_close(scope["decoded_latent"][..., 3:], expected, rtol=0, atol=0)
        self.assertEqual(tuple(scope["decoded_latent"].shape[1:3]), (17, 13))

    def test_ordinary_depth_control_generation_does_not_become_source_editing(self):
        scope, encoded, copies = self.scope()
        execute(self.reference.body, scope)
        original = scope["input_image"].clone()
        scope.update(
            guided_edit=False,
            guided_request=None,
            control_guide=None,
            gen_data={},
            controlnet=dict(
                type="depth", depth_preprocess="none", strength=0.8, start=0.1, stop=0.9
            ),
        )
        execute([self.control], scope)
        self.assertFalse(scope["img2img_mode"])
        self.assertEqual(encoded, [])
        torch.testing.assert_close(copies[0].hint, original.movedim(-1, 1), rtol=0, atol=0)
        torch.testing.assert_close(scope["input_image"], original, rtol=0, atol=0)
        self.assertNotIn("control_guide", scope["gen_data"])

    def test_auto_guide_uses_source_frame_and_manual_target_still_wins(self):
        scope, _, _ = self.scope(target=(1024, 1024))
        scope["gen_data"]["resolution_auto"] = True
        explicit_draft = scope["gen_data"]["custom_width"], scope["gen_data"]["custom_height"]
        execute(self.reference.body, scope)
        self.assertEqual(scope["gen_data"]["_image_requested_size"], (7, 5))
        self.assertEqual(
            (scope["reference_frame"]["width"], scope["reference_frame"]["height"]), (7, 5)
        )
        self.assertEqual(
            (scope["gen_data"]["custom_width"], scope["gen_data"]["custom_height"]), explicit_draft
        )
        scope, _, _ = self.scope(target=(13, 17), mode="pad")
        scope["gen_data"]["resolution_auto"] = False
        execute(self.reference.body, scope)
        self.assertEqual(scope["gen_data"]["_image_requested_size"], (13, 17))
        self.assertEqual(
            (scope["reference_frame"]["width"], scope["reference_frame"]["height"]), (13, 17)
        )

    def test_auto_source_cannot_bypass_existing_sdxl_canvas_bounds(self):
        scope, _, _ = self.scope(target=(1024, 1024))
        request = scope["guided_request"]
        # Only size metadata is varied: this regression allocates no large pixels.
        for size, expected in (
            ((4096, 4096), (2048, 2048)),
            ((16384, 512), (2048, 64)),
            ((4096, 64), (2048, 64)),
        ):
            with self.subTest(size=size):
                oversized = replace(request, source=replace(request.source, size=size))
                data = dict(scope["gen_data"], resolution_auto=True, **{guides.KEY: oversized})
                plan = guides.automatic_source_plan(data)
                self.assertEqual(plan, dict(target=expected, mode="pad"))
                data["resolution_auto"] = False
                self.assertIsNone(guides.automatic_source_size(data))
        # Exercise the selected fit/pad geometry on a small equivalent frame.
        scope, _, _ = self.scope(target=(1024, 1024))
        scope["gen_data"]["resolution_auto"] = True
        with patch.object(
            guides, "automatic_source_plan", return_value=dict(target=(64, 64), mode="pad")
        ):
            execute(self.reference.body, scope)
        self.assertEqual(scope["reference_frame"]["mode"], "pad")
        self.assertEqual(scope["input_image"].shape, scope["control_guide"].shape)
        self.assertEqual(tuple(scope["input_image"].shape[1:3]), (64, 64))


if __name__ == "__main__":
    unittest.main()
