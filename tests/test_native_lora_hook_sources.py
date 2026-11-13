"""Actual native combined/model-as-LoRA/float-list hook bodies and admission."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.core.preparation import PreparationBusy
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_editing import lora_hooks, lora_hook_sources
from ruinedfooocus.features.gallery.image_metadata import render_values
from ruinedfooocus.integrations.openai_api import ImageRequest
import test_api_native_inpaint as api_fixture
import test_native_lora_hooks as fixtures
from test_native_model_coverage import NATIVE
from test_native_variant_defaults import write_header


class CombinedHookSourceTests(unittest.TestCase):
    def setUp(self):
        self.files = fixtures.NativeLoraHookTests()
        self.files.setUp()
        self.addCleanup(self.files.doCleanups)
        self.root = self.files.root
        (self.root / "checkpoints").mkdir()
        self.source = write_header(self.root / "checkpoints/blend.safetensors", {"opaque": [1]})
        self.files.key_weight.return_value = (torch.zeros(2, 2), None, None)

    def capture(self, **values):
        return self.files.capture(**values)

    def test_native_combine_2_4_8_preserves_source_order_independent_keyframes_and_strengths(self):
        for count, cls in (
            (2, "CombineHooks"),
            (3, "CombineHooksFour"),
            (4, "CombineHooksFour"),
            (5, "CombineHooksEight"),
            (8, "CombineHooksEight"),
        ):
            rows = [
                dict(
                    type="lora",
                    name="schedule.safetensors",
                    model_strength=index + 1,
                    keyframes=[
                        dict(start=0, strength=1),
                        dict(start=(index + 1) / 10, strength=-index),
                    ],
                )
                for index in range(count)
            ]
            data = self.capture(lora_hook_model="", lora_hook_sources=json.dumps(rows))
            native = self.files.nodes
            with patch.object(native, cls, wraps=getattr(native, cls)) as combine:
                group, _ = lora_hooks.build_hooks(data[lora_hooks.KEY], native)
            self.assertEqual(combine.call_count, 1)
            self.assertEqual(len(group.hooks), count)
            self.assertEqual(
                [hook._strength_model for hook in group.hooks], list(range(1, count + 1))
            )
            self.assertEqual(
                [hook.hook_keyframe.keyframes[-1].strength for hook in group.hooks],
                [-index for index in range(count)],
            )
            self.assertEqual(len({hook.hook_ref for hook in group.hooks}), count)
            self.assertEqual(len({id(hook.hook_keyframe) for hook in group.hooks}), count)

    def test_native_float_list_generator_owns_spacing_endpoints_and_guarantees(self):
        data = self.capture(
            lora_hook_mode="floats",
            lora_hook_strengths="[1,-2,0.5]",
            lora_hook_start=0.2,
            lora_hook_end=0.8,
        )
        _, frames = lora_hooks.build_hooks(data[lora_hooks.KEY], self.files.nodes)
        self.assertEqual([frame.strength for frame in frames.keyframes], [1, -2, 0.5])
        for frame, percent in zip(frames.keyframes, (0.2, 0.5, 0.8), strict=True):
            self.assertAlmostEqual(frame.start_percent, percent)
        self.assertEqual([frame.guarantee_steps for frame in frames.keyframes], [1, 0, 0])
        data = self.capture(
            lora_hook_mode="floats",
            lora_hook_strengths="[0.75]",
            lora_hook_start=0.2,
            lora_hook_end=0.8,
        )
        _, single = lora_hooks.build_hooks(data[lora_hooks.KEY], self.files.nodes)
        self.assertEqual(len(single.keyframes), 1)
        self.assertEqual(single.keyframes[0].strength, 0.75)
        self.assertAlmostEqual(single.keyframes[0].start_percent, 0.2)

    def test_duplicate_source_files_have_one_lease_and_one_native_raw_weight_load(self):
        data = self.capture(
            lora_hook_sources=json.dumps(
                [
                    dict(
                        type="lora",
                        name="schedule.safetensors",
                        model_strength=-0.5,
                        mode="floats",
                        strengths=[0, 1, 0],
                    )
                ]
            )
        )
        owner, original = self.files.owner()
        with patch.object(assets, "prepared_asset", wraps=assets.prepared_asset) as lease:
            with lora_hooks.request_scope(owner, data, self.files.manager, Mock()):
                self.assertEqual(lease.call_count, 1)
                self.assertEqual(self.files.loader.call_count, 1)
                self.assertEqual(len(owner._request_lora_hooks.hooks), 2)
                self.assertEqual(
                    data["native_lora_hooks"]["native_nodes"][-2:],
                    ["CombineHooks2", "SetModelHooksOnCond"],
                )
                with (
                    self.assertRaises(PreparationBusy),
                    self.files.service.artifact_lease(self.files.lora, write=True, blocking=False),
                ):
                    pass
        self.assertIs(owner.xl_base_patched, original)

    def test_native_checkpoint_as_lora_and_model_only_capture_both_targets_without_source_mutation(
        self,
    ):
        before_file = self.source.read_bytes()
        before_model = self.files.source_model_weights["model"].clone()
        for clip_strength in (0, -0.75):
            self.files.checkpoint_loader.reset_mock()
            data = self.capture(
                lora_hook_source_type="checkpoint",
                lora_hook_model="blend.safetensors",
                lora_hook_model_strength=0.25,
                lora_hook_clip_strength=clip_strength,
            )
            owner, original = self.files.owner()
            with lora_hooks.request_scope(owner, data, self.files.manager, Mock()):
                self.files.checkpoint_loader.assert_called_once_with(
                    str(self.source), output_vae=True, output_clip=True, embedding_directory=[]
                )
                hook = owner._request_lora_hooks.hooks[0]
                self.assertFalse(hook.need_weight_init)
                self.assertNotIn("model_sampling.sigma", hook.weights)
                self.assertEqual(hook.weights["model"][0], "model_as_lora")
                self.assertIs(hook.weights["model"][1][0], self.files.source_model_weights["model"])
                metadata = data["native_lora_hooks"]
                self.assertEqual(metadata["clip_patches"], int(clip_strength != 0))
                self.assertEqual(metadata["sources"][0]["type"], "checkpoint")
                expected = (
                    "CreateHookModelAsLora" if clip_strength else "CreateHookModelAsLoraModelOnly"
                )
                self.assertIn(expected, metadata["native_nodes"])
            self.assertIs(owner.xl_base_patched, original)
        self.assertEqual(self.source.read_bytes(), before_file)
        self.assertTrue(torch.equal(self.files.source_model_weights["model"], before_model))

    def test_native_model_as_lora_difference_preserves_source_and_uses_baseline_weight(self):
        tree = ast.parse((NATIVE / "comfy/lora.py").read_text(encoding="utf-8"))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "calculate_weight"
        )
        scope = dict(
            torch=torch,
            comfy=NS(
                model_management=NS(
                    cast_to_device=lambda value, device, dtype, **_: value.to(
                        device=device, dtype=dtype
                    )
                )
            ),
            weight_adapter=NS(WeightAdapterBase=type("Adapter", (), {})),
        )
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "native model-as-LoRA difference", "exec"
            ),
            scope,
        )
        source = self.files.source_model_weights["model"]
        before = source.clone()
        baseline = torch.full_like(source, 0.4)
        result = scope["calculate_weight"](
            [(0.25, ("model_as_lora", (source,)), 1.0, None, None)],
            baseline.clone(),
            "model",
            original_weights={"model": [(baseline, None)]},
        )
        self.assertTrue(torch.allclose(result, torch.full_like(result, 0.8)))
        self.assertTrue(torch.equal(source, before))
        self.assertTrue(torch.equal(baseline, torch.full_like(baseline, 0.4)))

    def test_checkpoint_weight_shape_mismatch_refuses_before_encoding_and_releases_lease(self):
        data = self.capture(lora_hook_source_type="checkpoint", lora_hook_model="blend.safetensors")
        owner, original = self.files.owner()
        self.files.source_model_weights["model"] = torch.zeros(3, 3)
        with (
            self.assertRaisesRegex(ValueError, "different weight shape"),
            lora_hooks.request_scope(owner, data, self.files.manager, Mock()),
        ):
            self.fail("incompatible source was admitted")
        self.assertIs(owner.xl_base_patched, original)
        with self.files.service.artifact_lease(self.source, write=True, blocking=False):
            pass

    def test_checkpoint_source_without_requested_clip_refuses_before_native_registration(self):
        data = self.capture(
            lora_hook_source_type="checkpoint",
            lora_hook_model="blend.safetensors",
            lora_hook_clip_strength=0.5,
        )
        model = self.files.checkpoint_loader.return_value[0]
        self.files.checkpoint_loader.return_value = (model, None, None)
        owner, original = self.files.owner()
        with (
            self.assertRaisesRegex(ValueError, "no CLIP weights"),
            lora_hooks.request_scope(owner, data, self.files.manager, Mock()),
        ):
            self.fail("source without a CLIP target was admitted")
        self.assertIs(owner.xl_base_patched, original)

    def test_shared_checkpoint_loader_reuses_native_source_weights_for_full_and_model_only_hooks(
        self,
    ):
        data = self.capture(
            lora_hook_source_type="checkpoint",
            lora_hook_model="blend.safetensors",
            lora_hook_sources=json.dumps(
                [
                    dict(
                        type="checkpoint",
                        name="blend.safetensors",
                        clip_strength=0.5,
                        model_strength=-0.25,
                    )
                ]
            ),
        )
        owner, _ = self.files.owner()
        with lora_hooks.request_scope(owner, data, self.files.manager, Mock()):
            self.assertEqual(self.files.checkpoint_loader.call_count, 1)
            hooks = owner._request_lora_hooks.hooks
            self.assertEqual(len(hooks), 2)
            self.assertIs(hooks[0].weights["model"][1][0], hooks[1].weights["model"][1][0])

    def test_cancelled_mixed_sources_restore_owner_and_release_all_file_leases(self):
        data = self.capture(
            lora_hook_sources=json.dumps(
                [
                    dict(
                        type="checkpoint",
                        name="blend.safetensors",
                        model_strength=0.25,
                        clip_strength=0.5,
                    )
                ]
            )
        )
        owner, original = self.files.owner()
        conditions = owner.conditions
        with (
            self.assertRaisesRegex(RuntimeError, "cancelled"),
            lora_hooks.request_scope(owner, data, self.files.manager, Mock()),
        ):
            for path in (self.files.lora, self.source):
                with (
                    self.assertRaises(PreparationBusy),
                    self.files.service.artifact_lease(path, write=True, blocking=False),
                ):
                    pass
            raise RuntimeError("cancelled")
        self.assertIs(owner.xl_base_patched, original)
        self.assertIs(owner.conditions, conditions)
        self.assertEqual(original.unet.hook_patches, {})
        self.assertEqual(original.clip.patcher.hook_patches, {})
        for path in (self.files.lora, self.source):
            with self.files.service.artifact_lease(path, write=True, blocking=False):
                pass

    def test_changed_second_source_fails_before_any_native_weight_load(self):
        data = self.capture(
            lora_hook_sources=json.dumps([dict(type="checkpoint", name="blend.safetensors")])
        )
        owner, original = self.files.owner()
        self.source.write_bytes(b"external source replacement")
        with (
            self.assertRaisesRegex(ValueError, "changed after submission"),
            lora_hooks.request_scope(owner, data, self.files.manager, Mock()),
        ):
            self.fail("changed queued source was admitted")
        self.files.loader.assert_not_called()
        self.files.checkpoint_loader.assert_not_called()
        self.assertIs(owner.xl_base_patched, original)
        with self.files.service.artifact_lease(self.files.lora, write=True, blocking=False):
            pass

    def test_each_nonzero_source_must_map_even_when_another_source_matches(self):
        other = write_header(self.root / "loras/unmatched.safetensors", {"opaque": [1]})
        data = self.capture(lora_hook_sources=json.dumps([dict(type="lora", name=other.name)]))
        owner, original = self.files.owner()
        self.files.loader.side_effect = [{"matching": torch.ones(1)}, {"unmatched": torch.ones(1)}]
        self.files.native_lora.load_lora = lambda weights, key_map, **_: (
            {} if "unmatched" in weights else {key: ("native",) for key in key_map}
        )
        with (
            self.assertRaisesRegex(ValueError, "source 2 has no matching"),
            lora_hooks.request_scope(owner, data, self.files.manager, Mock()),
        ):
            self.fail("an inert second source was admitted")
        self.assertIs(owner.xl_base_patched, original)

    def test_clip_only_does_not_attach_zero_model_hooks_or_clone_the_diffusion_patcher(self):
        data = self.capture(
            lora_hook_model_strength=0, lora_hook_clip_strength=0.5, lora_hook_schedule_clip=True
        )
        owner, original = self.files.owner()
        positive, negative = [["positive", {}]], [["negative", {}]]
        with lora_hooks.request_scope(owner, data, self.files.manager, Mock()):
            self.assertIs(owner.xl_base_patched.unet, original.unet)
            self.assertIsNone(owner._request_lora_hooks)
            pos, neg = lora_hooks.conditioning(owner, positive, negative)
            self.assertIs(pos, positive)
            self.assertIs(neg, negative)
            self.assertNotIn("SetModelHooksOnCond", data["native_lora_hooks"]["native_nodes"])
            self.assertEqual(data["native_lora_hooks"]["model_patches"], 0)

    def test_model_only_sources_do_not_add_unneeded_scheduled_clip_encodings(self):
        data = self.capture(
            lora_hook_model_strength=0,
            lora_hook_clip_strength=0.5,
            lora_hook_schedule_clip=True,
            lora_hook_sources=json.dumps(
                [
                    dict(
                        type="lora",
                        name="schedule.safetensors",
                        model_strength=1,
                        clip_strength=0,
                        mode="floats",
                        strengths=[1, 0, 1, 0, 1],
                    )
                ]
            ),
        )
        owner, _ = self.files.owner()
        with lora_hooks.request_scope(owner, data, self.files.manager, Mock()):
            self.assertEqual(len(owner._request_lora_hooks.hooks), 1)
            self.assertEqual(len(owner.xl_base_patched.clip.patcher.forced_hooks.hooks), 1)
            self.assertEqual(
                len(owner.xl_base_patched.clip.patcher.forced_hooks.get_hooks_for_clip_schedule()),
                2,
            )

    def test_active_scheduled_clip_predicate_ignores_off_model_only_and_checks_extra_sources(self):
        self.assertFalse(
            lora_hooks.scheduled_clip_requested(
                dict(lora_hook_mode="off", lora_hook_schedule_clip=True, lora_hook_clip_strength=1)
            )
        )
        self.assertFalse(
            lora_hooks.scheduled_clip_requested(
                dict(
                    lora_hook_mode="keyframes",
                    lora_hook_schedule_clip=True,
                    lora_hook_clip_strength=0,
                )
            )
        )
        values = dict(
            lora_hook_mode="keyframes",
            lora_hook_schedule_clip=True,
            lora_hook_clip_strength=0,
            lora_hook_sources=json.dumps(
                [dict(type="lora", name="schedule.safetensors", clip_strength=-1)]
            ),
        )
        self.assertTrue(lora_hooks.scheduled_clip_requested(values))
        captured = self.capture(
            gligen_model="layout.pt",
            **{key: value for key, value in values.items() if key != "lora_hook_mode"},
        )
        self.assertTrue(captured[lora_hooks.KEY].settings()["lora_hook_schedule_clip"])
        self.assertEqual(len(captured[lora_hooks.KEY].sources), 2)

    def test_bounded_sources_and_float_values_reject_unsafe_or_inert_requests(self):
        for values in (
            dict(lora_hook_sources="{}"),
            dict(lora_hook_sources=json.dumps([dict(type="lora", name="owned.pt")] * 8)),
            dict(lora_hook_sources='[{"type":"lora","name":"../outside.pt"}]'),
            dict(lora_hook_sources='[{"type":"unknown","name":"owned.pt"}]'),
            dict(lora_hook_sources='[{"type":"lora","name":"owned.pt","unknown":1}]'),
            dict(lora_hook_strengths="[]"),
            dict(lora_hook_strengths="[true]"),
            dict(lora_hook_strengths="[NaN]"),
            dict(lora_hook_strengths="[21]"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.capture(**values)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            self.capture(lora_hook_model_strength=0, lora_hook_clip_strength=0)

    def test_card_source_copy_preserves_original_cards_and_order_and_saved_values_are_json_only(
        self,
    ):
        cards = [("one.png", "0.75 - detail.safetensors"), ("two.png", "-0.5 - other.sft")]
        before = deepcopy(cards)
        text = lora_hook_sources.selected_sources(cards)
        self.assertEqual(cards, before)
        rows = json.loads(text)
        self.assertEqual([row["name"] for row in rows], ["detail.safetensors", "other.sft"])
        self.assertEqual([row["model_strength"] for row in rows], [0.75, -0.5])
        saved = lora_hooks.replay_values(
            dict(lora_hook_mode="off", lora_hook_model="", lora_hook_sources=text)
        )
        self.assertEqual(saved["lora_hook_mode"], "off")
        self.assertEqual(json.loads(saved["lora_hook_sources"]), rows)
        exported = render_values(dict(saved, prompt="literal", _lora_hook_plan=object()))
        self.assertNotIn(lora_hooks.KEY, exported)
        json.dumps(exported)

    def test_typed_public_api_captures_checkpoint_and_multiple_native_sources(self):
        api = api_fixture.NativeInpaintApiTests()
        api.setUp()
        self.addCleanup(api.doCleanups)
        state = sys.modules["ruinedfooocus.app.state"]
        state.path_manager.get_model_folders = self.files.manager.get_model_folders
        row = dict(
            type="checkpoint",
            name="blend.safetensors",
            model_strength=-0.25,
            clip_strength=0.5,
            mode="floats",
            strengths=[0, 1, 0],
        )
        payload = dict(
            prompt="literal",
            size="512x512",
            lora_hook_mode="keyframes",
            lora_hook_model="schedule.safetensors",
            lora_hook_sources=json.dumps([row]),
        )
        request = ImageRequest(**payload)
        self.assertEqual(request.lora_hook_sources, payload["lora_hook_sources"])
        response = api.client.post("/images/generations", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        captured = api.prepared[-1]
        self.assertEqual(len(captured[lora_hooks.KEY].sources), 2)
        self.assertEqual(
            [source.category for source in captured[lora_hooks.KEY].sources],
            ["loras", "checkpoints"],
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in captured[assets.KEY].artifacts
                    if item.role.startswith("lora-hook")
                ]
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
