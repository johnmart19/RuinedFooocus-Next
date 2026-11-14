"""Native scheduled CLIP lists and interval adapters, without model execution."""

from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.image_editing import clip_hook_conditioning as intervals
from ruinedfooocus.features.image_editing import lora_hooks
from ruinedfooocus.features.image_editing import reference_adapters
from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS
import test_native_lora_hooks as hook_fixture
import test_native_lora_hook_callers as caller_fixture
import test_reference_adapters as reference_fixture
import test_qwen_image21_conditioning as qwen_fixture
from test_native_model_coverage import native_method


class SpecializedClipHookTests(unittest.TestCase):
    def fixture(self, cls):
        value = cls()
        value.setUp()
        self.addCleanup(value.doCleanups)
        return value

    def setUp(self):
        self.hooks = self.fixture(hook_fixture.NativeLoraHookTests)
        self.native = self.hooks.native
        self.clones, self.loads = [], []
        self.interrupted = Mock()
        clones, native = self.clones, self.native

        class Patcher(hook_fixture.Patcher):
            load_device = "cpu"

            def clone(self, disable_dynamic=False):
                # This fixture is already a non-dynamic CLIP patcher, as returned
                # by native SetClipHooks. Its clone shares the same text model.
                value = super().clone(disable_dynamic=False)
                value.__class__ = Patcher
                value.forced_hooks = self.forced_hooks.clone() if self.forced_hooks else None
                clones.append(value)
                return value

        class TextModel:
            def reset_clip_options(self):
                pass

            def set_clip_options(self, options):
                pass

            def encode_token_weights(self, tokens):
                values = self.active.forced_hooks
                strength = sum(hook.strength_clip for hook in values.hooks) if values else 0
                # An inert numeric encoder exercises native CLIP's direct method.
                # No neural layer, loaded model weights, device probe or inference.
                return torch.full((1, 77, 2048), 1 + strength), torch.tensor([strength])

        def load_models(patchers, **kwargs):
            for patcher in patchers:
                patcher.model.active = patcher
                self.loads.append(patcher)

        management = NS(
            load_models_gpu=load_models,
            cuda_device_context=lambda _: nullcontext(),
            throw_exception_if_processing_interrupted=self.interrupted,
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {"comfy.hooks": native, "comfy.model_management": management},
            )
        )
        clip_type = hook_fixture.native_module(
            "comfy/sd.py", {"model_management": management}, {"CLIP"}
        ).CLIP
        self.clip = clip_type(no_init=True)
        self.clip.patcher = Patcher(TextModel())
        self.clip.cond_stage_model = self.clip.patcher.model
        self.clip.tokenizer = NS(
            tokenize_with_weights=lambda text, *_, **__: {
                name: [[(i, 1, i) for i in range(77)]] for name in ("l", "g")
            }
        )
        self.clip.layer_idx, self.clip.tokenizer_options = None, {}
        self.clip.apply_hooks_to_conds = None
        self.clip.use_clip_schedule = True
        group = native.HookGroup()
        for strength, rows in ((0.5, ((0, 1), (0.5, 0))), (-0.25, ((0.25, 2), (0.75, 0)))):
            hook = native.WeightHook(strength_model=0, strength_clip=strength)
            hook.weights = {"shared-original-source": torch.ones(1)}
            for start, value in rows:
                hook.hook_keyframe.add(native.HookKeyframe(value, start))
            group.add(hook)
        self.clip.patcher.forced_hooks = group

    def direct(self, clip):
        cond, pooled = clip.encode_from_tokens(clip.tokenize("literal"), return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def test_off_and_constant_take_original_single_call_without_clone_or_native_import(self):
        callback = Mock(return_value=[["numeric", {}]])
        for clip in (object(), NS(use_clip_schedule=False, patcher=Mock())):
            result = intervals.direct_conditioning(clip, callback)
            self.assertIs(result, callback.return_value)
        self.assertEqual(callback.call_count, 2)
        self.assertEqual(self.clones, [])
        self.assertEqual(self.loads, [])

    def test_native_ranges_direct_clip_body_and_original_keyframes_are_preserved(self):
        group = self.clip.patcher.forced_hooks
        before = [(hook.hook_keyframe, hook.strength, hook.weights) for hook in group.hooks]
        result = intervals.direct_conditioning(self.clip, self.direct)
        self.assertEqual(
            [(row[1]["clip_start_percent"], row[1]["clip_end_percent"]) for row in result],
            [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)],
        )
        self.assertEqual([row[1]["pooled_output"].item() for row in result], [0.25, 0, -0.5, 0])
        self.assertEqual(len(self.loads), 4)
        self.assertTrue(all(value.clean_count == 1 for value in self.clones))
        for hook, (frames, strength, weights) in zip(group.hooks, before, strict=True):
            self.assertIs(hook.hook_keyframe, frames)
            self.assertEqual(hook.strength, strength)
            self.assertIs(hook.weights, weights)
        self.assertTrue(self.clip.use_clip_schedule)
        self.assertEqual(self.clip.patcher.clean_count, 0)

    def test_native_gligen_multiple_boxes_intersect_existing_clip_and_prompt_intervals(self):
        node = native_method("nodes.py", "GLIGENTextBoxApply", "append", {})()
        tensor, control, mask = torch.zeros(1), object(), torch.ones(1)
        conditioning = [
            [tensor, {"start_percent": 0.25, "end_percent": 0.75, "control": control, "mask": mask}]
        ]

        def apply(clip, selected):
            return node.append(selected, clip, "owned-gligen", "literal box", 64, 80, 8, 16)[0]

        first = intervals.direct_conditioning(self.clip, apply, conditioning)
        second = intervals.direct_conditioning(self.clip, apply, first)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertNotIn("gligen", conditioning[0][1])
        self.assertEqual(len(self.loads), 4)
        for item, pooled in zip(second, (0, -0.5), strict=True):
            self.assertIs(item[0], tensor)
            self.assertIs(item[1]["control"], control)
            self.assertIs(item[1]["mask"], mask)
            self.assertEqual(item[1]["start_percent"], 0.25)
            self.assertEqual(item[1]["end_percent"], 0.75)
            boxes = item[1]["gligen"][2]
            self.assertEqual(len(boxes), 2)
            self.assertEqual([box[0].item() for box in boxes], [pooled, pooled])
            self.assertEqual(boxes[0][1:], (10, 8, 2, 1))

    def test_native_encode_error_and_interval_cancel_clean_the_current_clone(self):
        def fail(clip):
            self.direct(clip)
            raise RuntimeError("cancelled direct node")

        with self.assertRaisesRegex(RuntimeError, "cancelled direct node"):
            intervals.direct_conditioning(self.clip, fail)
        self.assertEqual(len(self.clones), 1)
        self.assertEqual(self.clones[0].clean_count, 1)
        self.interrupted.side_effect = [None, RuntimeError("cancelled after encode")]
        with self.assertRaisesRegex(RuntimeError, "cancelled after encode"):
            intervals.direct_conditioning(self.clip, self.direct)
        self.assertEqual(self.clones[-1].clean_count, 1)
        self.assertTrue(self.clip.use_clip_schedule)

    def test_actual_photomaker_caller_uses_native_identity_node_for_every_clip_interval(self):
        fixture = self.fixture(reference_fixture.ReferenceAdapterTests)
        data = fixture.data("photomaker")
        data["positive_prompt"] = "[portrait of person photomaker|photo of person photomaker]"
        owner = fixture.owner("photomaker")
        owner.xl_base_patched.clip = self.clip

        class Identity:
            load_device = "cpu"

            def __call__(self, id_pixel_values, prompt_embeds, class_tokens_mask):
                return prompt_embeds + id_pixel_values.mean() * class_tokens_mask.unsqueeze(-1)

        identity = Identity()
        node = native_method(
            "comfy_extras/nodes_photomaker.py",
            "PhotoMakerEncode",
            "execute",
            dict(
                torch=torch,
                comfy=NS(clip_vision=NS(clip_preprocess=lambda image: image)),
                io=NS(NodeOutput=lambda *v: v),
            ),
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "comfy_extras.nodes_photomaker": NS(PhotoMakerEncode=node),
                    "node_helpers": NS(conditioning_set_values=reference_fixture.native_values()),
                },
            ),
            patch.object(reference_adapters, "_photomaker", return_value=identity),
            patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder"),
        ):
            positive = [[torch.zeros(1, 77, 2048), {}]]
            result = reference_adapters.apply_reference(
                owner, positive, data, fixture.manager, fixture.pixels
            )
        self.assertEqual(len(result), 8)
        self.assertEqual(len(self.loads), 8)
        self.assertEqual(
            [item[1]["pooled_output"].item() for item in result], [0.25, 0, -0.5, 0] * 2
        )
        self.assertEqual([item[1]["start_percent"] for item in result], [0] * 4 + [0.5] * 4)
        self.assertTrue(all("clip_start_percent" in item[1] for item in result))
        self.assertTrue(torch.equal(positive[0][0], torch.zeros_like(positive[0][0])))
        self.assertTrue(all(value.clean_count == 1 for value in self.clones))

    def test_request_scope_tracks_direct_clones_and_restores_owner_after_photo_failure(self):
        data = self.hooks.capture(
            lora_hook_clip_strength=0.5,
            lora_hook_schedule_clip=True,
            reference_adapter="photomaker",
        )
        owner, original = self.hooks.owner()
        original.clip = self.clip
        original_group = self.clip.patcher.forced_hooks
        with (
            self.assertRaisesRegex(RuntimeError, "direct failure"),
            lora_hooks.request_scope(owner, data, self.hooks.manager, Mock()),
        ):

            def fail(clip):
                self.direct(clip)
                raise RuntimeError("direct failure")

            intervals.direct_conditioning(owner.xl_base_patched.clip, fail)
        self.assertIs(owner.xl_base_patched, original)
        self.assertIs(original.clip.patcher.forced_hooks, original_group)
        self.assertFalse(hasattr(owner, "_request_lora_hooks"))
        self.assertTrue(
            all(not clone.callbacks and clone.clean_count >= 1 for clone in self.clones)
        )

    def test_every_reviewed_external_clip_family_captures_constant_and_scheduled_targets(self):
        # Bernini applies its own fixed LoRA pair and MiniMax's graph rejects
        # LoRAs, so neither goes through the shared CLIP hooks.
        own_graph = {"BerniniImage", "MiniMaxH3Image"}
        expected = {
            name
            for name, config in IMAGE_COMPONENTS.items()
            if config["clip_names"] and name not in own_graph
        }
        self.assertEqual(lora_hooks.CLIP_FAMILIES, expected)
        for family in sorted(expected):
            self.hooks.family = family
            for scheduled in (False, True):
                with self.subTest(family=family, scheduled=scheduled):
                    data = self.hooks.capture(
                        lora_hook_clip_strength=0.5, lora_hook_schedule_clip=scheduled
                    )
                    self.assertEqual(data[lora_hooks.KEY].family, family)
                    self.assertTrue(lora_hooks.capabilities(family)["clip"])
        for family in ("HiDreamO1", "SenseNovaU15"):
            self.hooks.family = family
            self.assertFalse(lora_hooks.capabilities(family)["clip"])
            with self.assertRaisesRegex(ValueError, "CLIP strength to 0"):
                self.hooks.capture(lora_hook_clip_strength=0.5)

    def test_public_api_modern_family_replays_same_fields_without_private_plan_export(self):
        fixture = self.fixture(caller_fixture.LoraHookCallerTests)
        api = fixture.api()
        state = sys.modules["ruinedfooocus.app.state"]
        state.models.get_model_base = lambda _: "Flux"
        response = api.client.post(
            "/images/generations",
            json=dict(
                prompt="literal",
                size="512x512",
                lora_hook_mode="keyframes",
                lora_hook_model="schedule.safetensors",
                lora_hook_clip_strength=0.5,
                lora_hook_schedule_clip=True,
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = api.prepared[-1]
        self.assertEqual(data[lora_hooks.KEY].family, "Flux")
        from ruinedfooocus.features.gallery.image_metadata import render_values

        metadata = render_values(data)
        self.assertNotIn(lora_hooks.KEY, metadata)
        self.assertEqual(lora_hooks.replay_values(metadata)["lora_hook_clip_strength"], 0.5)

    def test_native_qwen_pair_retains_every_scheduled_clip_entry_and_pooled_metadata(self):
        modules, _ = qwen_fixture.runtime()
        clip, mask = qwen_fixture.clip_fixture()
        pooled = [torch.tensor([0.5]), torch.tensor([-0.5])]
        clip.encode_from_tokens_scheduled.side_effect = lambda tokens: [
            [
                tokens["text"],
                dict(
                    attention_mask=mask,
                    pooled_output=pooled[index],
                    clip_start_percent=index / 2,
                    clip_end_percent=(index + 1) / 2,
                ),
            ]
            for index in range(2)
        ]
        from ruinedfooocus.features.image_editing.qwen_image21_conditioning import (
            text_to_image_conditioning,
        )

        with (
            patch.dict(sys.modules, modules),
            patch(qwen_fixture.MODULE + ".warm_patched_text_encoder"),
        ):
            positive, negative, latent = text_to_image_conditioning(
                clip, "literal positive", "literal negative", 96, 64
            )
        for text, conditions in (("literal positive", positive), ("literal negative", negative)):
            self.assertEqual(len(conditions), 2)
            self.assertEqual([item[0] for item in conditions], [text, text])
            for index, item in enumerate(conditions):
                self.assertIs(item[1]["attention_mask"], mask)
                self.assertIs(item[1]["pooled_output"], pooled[index])
                self.assertEqual(item[1]["clip_start_percent"], index / 2)
                self.assertEqual(item[1]["clip_end_percent"], (index + 1) / 2)
        self.assertEqual(tuple(latent["samples"].shape), (1, 64, 4, 6))


if __name__ == "__main__":
    unittest.main()
