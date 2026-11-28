"""Actual GLIGEN capture and native box conditioning with request CLIP hooks."""

import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

from ruinedfooocus.core.preparation import PreparationBusy
from ruinedfooocus.features.generation import image_preparation as prep
from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY
from ruinedfooocus.features.image_editing import gligen_layout as gligen, lora_hooks as hooks
import test_native_auxiliary_guidance as auxiliary
import test_specialized_lora_clip as specialized
import test_api_native_inpaint as api_fixture


class GligenHookSchedulingTests(unittest.TestCase):
    def fixture(self, cls):
        value = cls()
        value.setUp()
        self.addCleanup(value.doCleanups)
        return value

    def setUp(self):
        self.special = self.fixture(specialized.SpecializedClipHookTests)
        self.files = self.fixture(auxiliary.AuxiliaryTests)
        self.files.checkpoint = self.special.hooks.checkpoint
        self.special.hooks.family = "BaseModel"

    def capture(self):
        data = self.files.capture(
            hypernetwork_model=None,
            lora_hook_mode="keyframes",
            lora_hook_model="schedule.safetensors",
            lora_hook_clip_strength=0.5,
            lora_hook_schedule_clip=True,
        )
        data, items = hooks.capture_assets(
            data, self.special.hooks.manager, self.special.hooks.models, {}
        )
        self.assertEqual(data[gligen.KEY].checkpoint_signature, data[hooks.KEY].checkpoint)
        self.assertEqual(len(items), 1)
        return data

    def runtime(self, width=64):
        get_attr = sys.modules["comfy.utils"].get_attr
        owner, native, original_clip = self.files.runtime()
        native.utils.get_attr = get_attr
        native.model_management.throw_exception_if_processing_interrupted = self.special.interrupted
        clip = self.special.clip
        encode = clip.cond_stage_model.encode_token_weights
        clip.cond_stage_model.encode_token_weights = lambda tokens: (
            lambda pair: (pair[0], pair[1].expand(1, width))
        )(encode(tokens))
        clip.tokenizer.tokenize_with_weights = Mock(wraps=clip.tokenizer.tokenize_with_weights)
        owner.xl_base_patched.clip = clip
        return owner, native, original_clip

    def test_actual_auxiliary_capture_and_native_box_caller_match_hook_intervals(self):
        data = self.capture()
        owner, native, _ = self.runtime()
        tensor, control, mask = torch.zeros(1), object(), torch.ones(1)
        source = [
            [
                tensor,
                dict(
                    clip_start_percent=0,
                    clip_end_percent=0.5,
                    start_percent=0.25,
                    end_percent=0.75,
                    control=control,
                    mask=mask,
                ),
            ],
            [
                tensor,
                dict(
                    clip_start_percent=0.5,
                    clip_end_percent=1,
                    start_percent=0.25,
                    end_percent=0.75,
                    control=control,
                    mask=mask,
                ),
            ],
        ]
        result = gligen.apply(owner, source, data, self.files.manager, Mock())
        self.assertEqual(len(result), 2)
        self.assertTrue(all("gligen" not in item[1] for item in source))
        self.assertEqual(
            [(row[1]["clip_start_percent"], row[1]["clip_end_percent"]) for row in result],
            [(0.25, 0.5), (0.5, 0.75)],
        )
        self.assertEqual(
            [
                call.args[0]
                for call in self.special.clip.tokenizer.tokenize_with_weights.call_args_list
            ],
            [self.files.boxes[0]["text"]] * 2 + [self.files.boxes[1]["text"]] * 2,
        )
        for row, strength in zip(result, (0, -0.5), strict=True):
            self.assertIs(row[0], tensor)
            self.assertIs(row[1]["control"], control)
            self.assertIs(row[1]["mask"], mask)
            self.assertEqual(len(row[1]["gligen"][2]), 2)
            for box, captured in zip(row[1]["gligen"][2], self.files.boxes, strict=True):
                self.assertTrue(torch.equal(box[0], torch.full((1, 64), strength)))
                self.assertEqual(
                    box[1:],
                    (
                        captured["height"] // 8,
                        captured["width"] // 8,
                        captured["y"] // 8,
                        captured["x"] // 8,
                    ),
                )
        self.assertEqual(data["gligen_guidance"]["boxes"], self.files.boxes)
        self.assertEqual(len(self.special.loads), 4)
        self.assertTrue(all(clone.clean_count == 1 for clone in self.special.clones))
        native.sd.load_gligen.assert_called_once()

    def test_pooled_width_error_keeps_original_input_and_cleans_interval_clone(self):
        data = self.capture()
        owner, _, _ = self.runtime(width=32)
        source = [[torch.zeros(1), {}]]
        with self.assertRaisesRegex(ValueError, "pooled width"):
            gligen.apply(owner, source, data, self.files.manager, Mock())
        self.assertEqual(source[0][1], {})
        self.assertEqual(len(self.special.clones), 1)
        self.assertEqual(self.special.clones[0].clean_count, 1)

    def test_interval_cancel_propagates_and_cleans_native_clone(self):
        data = self.capture()
        owner, _, _ = self.runtime()
        self.special.interrupted.side_effect = [None, RuntimeError("cancelled native interval")]
        with self.assertRaisesRegex(RuntimeError, "cancelled native interval"):
            gligen.apply(owner, [[torch.zeros(1), {}]], data, self.files.manager, Mock())
        self.assertEqual(len(self.special.clones), 1)
        self.assertEqual(self.special.clones[0].clean_count, 1)

    def test_constant_off_auxiliary_caller_keeps_its_two_original_encodes(self):
        data = self.files.capture(hypernetwork_model=None)
        owner, _, clip = self.files.runtime()
        source = [[torch.zeros(1), {"owned": True}]]
        result = gligen.apply(owner, source, data, self.files.manager, Mock())
        self.assertEqual(clip.encode_from_tokens.call_count, 2)
        self.assertEqual(len(result), 1)
        self.assertNotIn("clip_start_percent", result[0][1])
        self.assertNotIn("gligen", source[0][1])

    def test_public_api_capture_reaches_actual_native_boxes_in_the_owned_hook_scope(self):
        api = self.fixture(api_fixture.NativeInpaintApiTests)
        state = sys.modules["ruinedfooocus.app.state"]
        state.path_manager.get_model_folders = lambda category: (
            [self.files.root]
            if category == "gligen"
            else self.special.hooks.manager.get_model_folders(category)
        )
        response = api.client.post(
            "/images/generations",
            json=dict(
                prompt="literal scene",
                size="512x512",
                gligen_model=self.files.gligen_file.name,
                gligen_layout=json.dumps(self.files.boxes),
                lora_hook_mode="keyframes",
                lora_hook_model=self.special.hooks.lora.name,
                lora_hook_model_strength=0,
                lora_hook_clip_strength=0.5,
                lora_hook_schedule_clip=True,
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = api.prepared[-1]
        self.assertIsInstance(data[gligen.KEY], gligen.GligenPlan)
        self.assertIsInstance(data[hooks.KEY], hooks.HookPlan)
        self.assertEqual(data[gligen.KEY].checkpoint_signature, data[hooks.KEY].checkpoint)
        self.assertTrue({"gligen", "lora-hook"} <= {item.role for item in data[prep.KEY].artifacts})
        # The process resolves its captured canvas before conditioning; the API
        # preparation deliberately retains the immutable SizePlan instead.
        data["width"], data["height"] = data[SIZE_KEY].canvas
        auxiliary_owner, native, _ = self.runtime()
        native.hooks = self.special.native
        owner, _ = self.special.hooks.owner()
        original = owner.StableDiffusionModel(
            auxiliary_owner.xl_base_patched.unet,
            auxiliary_owner.xl_base_patched.vae,
            self.special.clip,
            None,
        )
        owner.xl_base_patched = original
        owner.base_file_signature = data[hooks.KEY].checkpoint
        previous_conditions = owner.conditions
        positive = [[torch.zeros(1), {}]]
        with hooks.request_scope(owner, data, state.path_manager, Mock()):
            self.assertIsNone(owner.conditions)
            self.assertIs(owner.xl_base_patched.unet, original.unet)
            self.assertIsNot(owner.xl_base_patched.clip, original.clip)
            result = gligen.apply(owner, positive, data, state.path_manager, Mock())
            self.assertEqual(len(result), 2)
            self.assertEqual([item[1]["clip_start_percent"] for item in result], [0, 0.5])
            for item, value in zip(result, (0.5, 0), strict=True):
                self.assertEqual(len(item[1]["gligen"][2]), 2)
                self.assertTrue(
                    all(
                        torch.equal(box[0], torch.full((1, 64), value))
                        for box in item[1]["gligen"][2]
                    )
                )
            with self.assertRaises(PreparationBusy):
                with self.special.hooks.service.artifact_lease(
                    self.special.hooks.lora, write=True, blocking=False
                ):
                    pass
        self.assertIs(owner.xl_base_patched, original)
        self.assertIs(owner.conditions, previous_conditions)
        self.assertNotIn("gligen", positive[0][1])
        self.assertTrue(all(not clone.callbacks for clone in self.special.clones))
        with self.special.hooks.service.artifact_lease(
            self.special.hooks.lora, write=True, blocking=False
        ):
            pass
        from ruinedfooocus.features.gallery.image_metadata import render_values

        exported = render_values(data)
        self.assertNotIn(gligen.KEY, exported)
        self.assertNotIn(hooks.KEY, exported)
        self.assertEqual(hooks.replay_values(exported)["lora_hook_clip_strength"], 0.5)
        self.assertEqual(exported["gligen_layout"], data["gligen_layout"])
        json.dumps(exported)
