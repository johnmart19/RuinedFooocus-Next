"""Pinned native hook/keyframe bodies and request cancellation/restoration contracts."""

import ast
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
import enum
import itertools
import logging
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ruinedfooocus.app import resource_lifecycle as lifecycle
from ruinedfooocus.core.preparation import PreparationBusy, PreparationService
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_editing import lora_hooks as hooks
from ruinedfooocus.features.models.model_identity import file_signature
from test_native_model_coverage import NATIVE
from test_native_variant_defaults import write_header


def native_module(path, namespace, names=None):
    tree = ast.parse((NATIVE / path).read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and (names is None or node.name in names)
    ]
    module = ModuleType(path)
    module.__dict__.update(namespace)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), path, "exec"),
        module.__dict__,
    )
    return module


class Patcher:
    def __init__(self, model=None):
        self.model = model or object()
        self.callbacks, self.hook_patches, self.hook_backup = {}, {}, {}
        self.forced_hooks = None
        self.clean_count = 0
        self.patches = {"static-card-and-prompt": [0.4]}

    def add_callback_with_key(self, event, key, callback):
        self.callbacks.setdefault((event, key), []).append(callback)

    def remove_callbacks_with_key(self, event, key):
        self.callbacks.pop((event, key), None)

    def clone(self, disable_dynamic=False):
        new = Patcher(object() if disable_dynamic else self.model)
        new.callbacks = {key: list(value) for key, value in self.callbacks.items()}
        new.hook_patches = {key: dict(value) for key, value in self.hook_patches.items()}
        new.hook_backup = self.hook_backup
        new.patches = {key: list(value) for key, value in self.patches.items()}
        for callbacks in self.callbacks.values():
            for callback in callbacks:
                callback(self, new)
        return new

    def add_hook_patches(self, hook, patches, strength_patch):
        self.hook_patches[hook.hook_ref] = dict(patches)
        return list(patches)

    def register_all_hook_patches(self, hook_group, target):
        registered = self.native.HookGroup()
        for hook in hook_group.hooks:
            hook.add_hook_patches(self, {}, target, registered)
        return registered

    def clean_hooks(self):
        self.hook_backup.clear()
        self.clean_count += 1


class Clip:
    def __init__(self, patcher=None):
        self.patcher = patcher or Patcher()
        self.cond_stage_model = self.patcher.model
        self.use_clip_schedule = False

    def clone(self, disable_dynamic=False):
        return Clip(self.patcher.clone(disable_dynamic=disable_dynamic))


class NativeLoraHookTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        (self.root / "loras").mkdir()
        self.lora = write_header(
            self.root / "loras/schedule.safetensors", {"lora_unet_a.lora_up.weight": [1, 1]}
        )
        self.checkpoint = write_header(self.root / "checkpoint.safetensors", {"opaque": [1]})
        self.manager = NS(
            DOWNLOADABLE_FILES={}, get_model_folders=lambda category: [self.root / category]
        )
        self.family = "SDXL"
        self.models = NS(
            get_file=lambda *_: self.checkpoint,
            get_models_by_path=lambda *_, **__: {},
            get_model_base=lambda _: self.family,
        )
        self.native_lora = NS(
            model_lora_keys_unet=lambda *_: {"model": "weight"},
            model_lora_keys_clip=lambda *_: {"clip": "weight"},
            load_lora=lambda weights, key_map, **_: {key: ("native-patch",) for key in key_map},
        )
        self.native = native_module(
            "comfy/hooks.py",
            dict(
                enum=enum,
                itertools=itertools,
                math=math,
                torch=torch,
                np=np,
                logging=logging,
                comfy=NS(lora=self.native_lora),
            ),
        )
        Patcher.native = self.native
        self.loader = Mock(return_value={"raw-native-lora": torch.zeros(1)})
        self.key_weight = Mock(return_value=(NS(tensor_type=None), None, None))
        folder_paths = NS(
            get_full_path=lambda category, name: str(self.root / category / name),
            get_full_path_or_raise=lambda category, name: str(self.root / category / name),
            get_folder_paths=lambda category: [],
        )
        self.source_model_weights = {
            "model": torch.full((2, 2), 2.0),
            "model_sampling.sigma": torch.zeros(1),
        }
        self.source_clip_weights = {"clip": torch.full((2, 2), 3.0)}
        self.checkpoint_loader = Mock(
            return_value=(
                NS(model=NS(state_dict=lambda: dict(self.source_model_weights))),
                NS(patcher=NS(model=NS(state_dict=lambda: dict(self.source_clip_weights)))),
                object(),
            )
        )
        comfy = NS(
            hooks=self.native,
            utils=NS(load_torch_file=self.loader),
            sd=NS(load_checkpoint_guess_config=self.checkpoint_loader),
        )
        self.nodes = native_module(
            "comfy_extras/nodes_hooks.py",
            dict(comfy=comfy, folder_paths=folder_paths, logging=logging, Iterable=Iterable),
            {
                "CreateHookLora",
                "CreateHookLoraModelOnly",
                "CreateHookModelAsLora",
                "CreateHookModelAsLoraModelOnly",
                "CreateHookKeyframesFromFloats",
                "CombineHooks",
                "CombineHooksFour",
                "CombineHooksEight",
                "CreateHookKeyframe",
                "CreateHookKeyframesInterpolated",
                "SetHookKeyframes",
                "SetClipHooks",
                "SetModelHooksOnCond",
            },
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy": comfy,
                    "comfy.patcher_extension": NS(CallbacksMP=NS(ON_CLONE="on_clone")),
                    "comfy.model_patcher": NS(get_key_weight=self.key_weight),
                    "comfy.utils": NS(get_attr=lambda *_: NS()),
                    "comfy_extras": NS(nodes_hooks=self.nodes),
                    "comfy_extras.nodes_hooks": self.nodes,
                    "folder_paths": folder_paths,
                },
            )
        )
        self.enterContext(patch.object(lifecycle, "require_compute_owner"))
        self.service = PreparationService(max_workers=1)
        self.addCleanup(self.service.close)
        prepared_asset = assets.prepared_asset

        @contextmanager
        def prepared(data, role, name, manager, category):
            with prepared_asset(data, role, name, manager, category, service=self.service) as path:
                yield path

        self.enterContext(patch.object(assets, "prepared_asset", side_effect=prepared))

    def capture(self, **values):
        data, items = hooks.capture_assets(
            dict(
                base_model_name="checkpoint.safetensors",
                lora_hook_mode="keyframes",
                lora_hook_model="schedule.safetensors",
            )
            | values,
            self.manager,
            self.models,
            {},
        )
        data[assets.KEY] = assets.ImageAssets(items, True)
        return data

    def owner(self):
        class Model:
            def __init__(self, unet, vae, clip, clip_vision):
                self.unet, self.vae, self.clip, self.clip_vision = unet, vae, clip, clip_vision

        original = Model(Patcher(), object(), Clip(), object())
        owner = NS(
            StableDiffusionModel=Model,
            xl_base_patched=original,
            conditions={"+": {"text": "prior plain prompt", "cache": ["plain"]}},
            base_file_signature=file_signature(self.checkpoint),
        )
        return owner, original

    def test_off_does_no_file_inventory_native_hook_or_cache_work(self):
        manager, models = Mock(), Mock()
        data, items = hooks.capture_assets(
            dict(
                lora_hook_mode="off",
                lora_hook_model="stale-or-outside.pt",
                native_lora_hooks={"old": True},
            ),
            manager,
            models,
            {},
        )
        self.assertNotIn("native_lora_hooks", data)
        self.assertEqual(items, ())
        manager.assert_not_called()
        models.assert_not_called()
        owner, original = self.owner()
        conditions = owner.conditions
        with hooks.request_scope(owner, data, manager, Mock()):
            self.assertIs(owner.xl_base_patched, original)
            self.assertIs(owner.conditions, conditions)
        self.loader.assert_not_called()

    def test_native_manual_keyframes_preserve_signed_strengths_and_guarantee_steps(self):
        data = self.capture(
            lora_hook_keyframes='[{"start":0,"strength":-1},{"start":0.4,"strength":2},{"start":1,"strength":0}]',
            lora_hook_model_strength=-2,
        )
        group, frames = hooks.build_hooks(data[hooks.KEY], self.nodes)
        self.loader.assert_called_once_with(str(self.lora), safe_load=True)
        self.assertEqual(
            [(f.start_percent, f.strength, f.guarantee_steps) for f in frames.keyframes],
            [(0, -1, 1), (0.4, 2, 1), (1, 0, 1)],
        )
        self.assertEqual(group.hooks[0].strength_model, 2)
        self.assertIsNot(group.hooks[0].hook_keyframe, frames)
        group.hooks[0].initialize_timesteps(
            NS(model_sampling=NS(percent_to_sigma=lambda p: 10 * (1 - p)))
        )
        keyframes = group.hooks[0].hook_keyframe
        keyframes.prepare_current_keyframe(10, {"sample_sigmas": torch.tensor([10.0, 0.0])})
        self.assertEqual(keyframes.strength, -1)
        keyframes.prepare_current_keyframe(6, {"sample_sigmas": torch.tensor([10.0, 0.0])})
        self.assertEqual(keyframes.strength, 2)
        keyframes.prepare_current_keyframe(0, {"sample_sigmas": torch.tensor([10.0, 0.0])})
        self.assertEqual(keyframes.strength, 0)

    def test_native_interpolation_all_four_modes_and_exact_endpoints(self):
        for interpolation in hooks.INTERPOLATIONS:
            data = self.capture(
                lora_hook_mode="interpolated",
                lora_hook_interpolation=interpolation,
                lora_hook_start=0.1,
                lora_hook_end=0.9,
                lora_hook_strength_start=2,
                lora_hook_strength_end=0.5,
                lora_hook_keyframe_count=7,
            )
            _, frames = hooks.build_hooks(data[hooks.KEY], self.nodes)
            self.assertEqual(len(frames.keyframes), 7)
            self.assertAlmostEqual(frames.keyframes[0].start_percent, 0.1)
            self.assertAlmostEqual(frames.keyframes[-1].start_percent, 0.9)
            expected = self.native.InterpolationMethod.get_weights(2, 0.5, 7, interpolation)
            self.assertEqual([float(f.strength) for f in frames.keyframes], expected.tolist())
            self.assertEqual([f.guarantee_steps for f in frames.keyframes], [1, 0, 0, 0, 0, 0, 0])

    def test_model_only_clones_preserve_static_patches_and_hold_file_lease_until_scope_finishes(
        self,
    ):
        data = self.capture()
        owner, original = self.owner()
        conditions = owner.conditions
        with hooks.request_scope(owner, data, self.manager, Mock()):
            self.assertIsNot(owner.xl_base_patched, original)
            self.assertIsNot(owner.xl_base_patched.unet, original.unet)
            self.assertEqual(owner.xl_base_patched.unet.patches, original.unet.patches)
            self.assertIs(owner.xl_base_patched.clip, original.clip)
            self.assertIsNone(owner.conditions)
            with (
                self.assertRaises(PreparationBusy),
                self.service.artifact_lease(self.lora, write=True, blocking=False),
            ):
                pass
            self.assertEqual(data["native_lora_hooks"]["target"], "primary-checkpoint")
            self.assertEqual(data["native_lora_hooks"]["model_patches"], 1)
        self.assertIs(owner.xl_base_patched, original)
        self.assertIs(owner.conditions, conditions)
        self.assertEqual(original.unet.hook_patches, {})
        self.assertEqual(original.unet.callbacks, {})
        with self.service.artifact_lease(self.lora, write=True, blocking=False):
            pass

    def test_cancel_during_encoding_cleans_every_native_descendant_and_retry_is_fresh(self):
        data = self.capture(lora_hook_clip_strength=0.7, lora_hook_schedule_clip=True)
        owner, original = self.owner()
        conditions = owner.conditions
        descendants = []
        with (
            self.assertRaisesRegex(RuntimeError, "cancelled"),
            hooks.request_scope(owner, data, self.manager, Mock()),
        ):
            descendants.extend([owner.xl_base_patched.unet, owner.xl_base_patched.clip.patcher])
            descendants.append(owner.xl_base_patched.unet.clone(disable_dynamic=True))
            descendants.append(owner.xl_base_patched.clip.clone().patcher)
            for descendant in descendants:
                descendant.hook_backup["weight"] = "unpatched-original"
            owner.conditions = {"+": {"cache": ["temporary scheduled text"]}}
            raise RuntimeError("cancelled")
        self.assertIs(owner.xl_base_patched, original)
        self.assertIs(owner.conditions, conditions)
        self.assertFalse(hasattr(owner, "_request_lora_hooks"))
        self.assertTrue(
            all(p.clean_count == 1 and not p.hook_backup and not p.callbacks for p in descendants)
        )
        self.assertIsNone(original.clip.patcher.forced_hooks)
        self.assertEqual(original.clip.patcher.hook_patches, {})
        with hooks.request_scope(owner, data, self.manager, Mock()):
            self.assertIsNone(owner.conditions)
            self.assertTrue(owner.xl_base_patched.clip.use_clip_schedule)
        self.assertIs(owner.conditions, conditions)

    def test_secondary_model_never_registers_primary_checkpoint_hooks(self):
        data = self.capture()
        owner, _ = self.owner()
        with hooks.request_scope(owner, data, self.manager, Mock()):
            group = owner._request_lora_hooks
            secondary = Patcher()
            secondary.register_all_hook_patches(
                group, self.native.create_target_dict(self.native.EnumWeightTarget.Model)
            )
            self.assertEqual(secondary.hook_patches, {})
            delegate = owner.xl_base_patched.unet.clone(disable_dynamic=True)
            delegate.register_all_hook_patches(
                group, self.native.create_target_dict(self.native.EnumWeightTarget.Model)
            )
            self.assertTrue(delegate.hook_patches)

    def test_constant_clip_does_not_remove_model_keyframes(self):
        data = self.capture(lora_hook_clip_strength=-0.25)
        owner, _ = self.owner()
        with hooks.request_scope(owner, data, self.manager, Mock()):
            self.assertFalse(owner.xl_base_patched.clip.use_clip_schedule)
            self.assertEqual(len(owner._request_lora_hooks.hooks[0].hook_keyframe.keyframes), 2)
            self.assertEqual(
                len(
                    owner.xl_base_patched.clip.patcher.forced_hooks.hooks[0].hook_keyframe.keyframes
                ),
                0,
            )

    def test_native_clip_encoder_retains_all_keyframe_intervals_and_resets_after_encoding(self):
        data = self.capture(
            lora_hook_mode="interpolated",
            lora_hook_clip_strength=0.4,
            lora_hook_schedule_clip=True,
            lora_hook_start=0.2,
            lora_hook_strength_start=2,
            lora_hook_end=0.8,
            lora_hook_strength_end=0,
            lora_hook_keyframe_count=3,
        )
        owner, _ = self.owner()
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        native_clip = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CLIP")
        method = next(
            n
            for n in native_clip.body
            if isinstance(n, ast.FunctionDef) and n.name == "encode_from_tokens_scheduled"
        )
        namespace = dict(
            model_management=NS(
                cuda_device_context=lambda *_: nullcontext(),
                throw_exception_if_processing_interrupted=Mock(),
            ),
            ProgressBar=lambda *_: NS(update=Mock()),
        )
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), "pinned CLIP scheduled encode", "exec"
            ),
            namespace,
        )
        with hooks.request_scope(owner, data, self.manager, Mock()):
            clip = owner.xl_base_patched.clip
            clip.layer_idx = None
            clip.patcher.load_device = "cpu"
            multipliers = []
            clip.patcher.patch_hooks = lambda group: multipliers.append(
                None if group is None else group.hooks[0].strength
            )
            clip.cond_stage_model = NS(
                reset_clip_options=Mock(),
                set_clip_options=Mock(),
                encode_token_weights=lambda *_: (torch.ones(1), torch.ones(1)),
            )
            clip.load_model = Mock()
            clip.add_hooks_to_dict = Mock()
            encoded = namespace["encode_from_tokens_scheduled"](
                clip, {"literal": [1]}, add_dict={"width": 512}, show_pbar=False
            )
            ranges = [(row[1]["clip_start_percent"], row[1]["clip_end_percent"]) for row in encoded]
            self.assertEqual(len(encoded), 4)
            self.assertAlmostEqual(ranges[0][0], 0)
            self.assertAlmostEqual(ranges[0][1], 0.2)
            self.assertAlmostEqual(ranges[-1][0], 0.8)
            self.assertAlmostEqual(ranges[-1][1], 1)
            self.assertTrue(all(row[1]["width"] == 512 for row in encoded))
            self.assertEqual(multipliers, [None, 1, 2, 1, 0])
            self.assertEqual(clip.patcher.forced_hooks.hooks[0].strength, 2)

    def test_native_condition_attachment_keeps_prompt_intervals_control_and_prior_hooks(self):
        data = self.capture()
        owner, _ = self.owner()
        earlier = self.native.create_hook_lora({}, 1, 0)
        control = object()
        positive = [
            [
                torch.ones(1),
                {
                    "control": control,
                    "start_percent": 0.3,
                    "end_percent": 0.6,
                    "hooks": earlier,
                    "gligen": ("native",),
                },
            ]
        ]
        negative = [[torch.zeros(1), {"pooled_output": torch.zeros(1)}]]
        with hooks.request_scope(owner, data, self.manager, Mock()):
            pos, neg = hooks.conditioning(owner, positive, negative)
            self.assertEqual(len(pos[0][1]["hooks"].hooks), 2)
            self.assertIs(pos[0][1]["control"], control)
            self.assertEqual(pos[0][1]["start_percent"], 0.3)
            self.assertEqual(pos[0][1]["end_percent"], 0.6)
            self.assertEqual(pos[0][1]["gligen"], ("native",))
            self.assertIs(neg[0][1]["hooks"], owner._request_lora_hooks)
            self.assertIs(positive[0][1]["hooks"], earlier)

    def test_empty_native_patch_mapping_rejects_before_encode_and_restores_owner(self):
        for target in ("model", "clip"):
            data = self.capture(
                lora_hook_model_strength=1 if target == "model" else 0,
                lora_hook_clip_strength=1 if target == "clip" else 0,
            )
            owner, original = self.owner()
            with patch.object(self.native_lora, "load_lora", return_value={}):
                with (
                    self.assertRaisesRegex(ValueError, "no matching"),
                    hooks.request_scope(owner, data, self.manager, Mock()),
                ):
                    self.fail("inert native hook was admitted")
            self.assertIs(owner.xl_base_patched, original)

    def test_packed_gguf_targets_require_an_adapter_but_floating_gguf_targets_are_admitted(self):
        data = self.capture()
        owner, original = self.owner()
        self.key_weight.return_value = (NS(tensor_type=NS(name="Q4_K")), None, None)
        with (
            self.assertRaisesRegex(ValueError, "quantized-hook adapter"),
            hooks.request_scope(owner, data, self.manager, Mock()),
        ):
            self.fail("packed native hook accepted without an adapter")
        self.assertIs(owner.xl_base_patched, original)
        self.key_weight.return_value = (NS(tensor_type=NS(name="F16")), None, None)
        with hooks.request_scope(owner, data, self.manager, Mock()):
            self.assertEqual(data["native_lora_hooks"]["model_patches"], 1)

    def test_pinned_native_hook_patch_and_cleanup_restore_shared_original_weight(self):
        tree = ast.parse((NATIVE / "comfy/model_patcher.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelPatcher")
        names = {
            "patch_hook_weight_to_device",
            "unpatch_hooks",
            "clean_hooks",
            "clear_cached_hook_weights",
        }
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        weight = torch.nn.Parameter(torch.full((2, 2), 0.4))
        original = weight.detach().clone()
        model = NS(weight=weight)
        comfy = NS(
            hooks=self.native,
            utils=NS(
                get_attr=lambda obj, key: getattr(obj, key),
                copy_to_param=lambda obj, key, value: getattr(obj, key).data.copy_(value),
                string_to_seed=lambda *_: 0,
            ),
            model_management=NS(
                cast_to_device=lambda value, device, dtype, copy: value.to(
                    device=device, dtype=dtype, copy=copy
                )
            ),
            lora=NS(calculate_weight=lambda patches, value, *_, **__: value + patches),
            float=NS(stochastic_rounding=lambda value, dtype, **_: value.to(dtype)),
        )
        namespace = dict(
            comfy=comfy,
            torch=torch,
            MemoryCounter=object,
            get_key_weight=lambda *_: (weight, None, None),
        )
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])),
                "pinned native hook patch/cleanup",
                "exec",
            ),
            namespace,
        )
        patcher = NS(
            model=model,
            hook_backup={},
            hook_mode=self.native.EnumHookMode.MinVram,
            offload_device="cpu",
            cached_hook_patches={},
            current_hooks=object(),
            use_ejected=lambda: nullcontext(),
        )
        patcher.unpatch_hooks = lambda: namespace["unpatch_hooks"](patcher)
        patcher.patch_hooks = lambda _: patcher.unpatch_hooks()
        patcher.clear_cached_hook_weights = lambda: namespace["clear_cached_hook_weights"](patcher)
        namespace["patch_hook_weight_to_device"](
            patcher,
            self.native.HookGroup(),
            {"weight": torch.ones_like(weight)},
            "weight",
            {"weight": original},
            object(),
        )
        self.assertTrue(torch.allclose(weight, original + 1))
        self.assertIn("weight", patcher.hook_backup)
        namespace["clean_hooks"](patcher)
        self.assertTrue(torch.equal(model.weight, original))
        self.assertEqual(patcher.hook_backup, {})
        self.assertEqual(patcher.cached_hook_patches, {})

    def test_changed_checkpoint_or_lora_and_native_path_mismatch_fail_before_loading(self):
        data = self.capture()
        owner, _ = self.owner()
        self.lora.write_bytes(b"replaced")
        with (
            self.assertRaisesRegex(ValueError, "changed after submission"),
            hooks.request_scope(owner, data, self.manager, Mock()),
        ):
            pass
        self.loader.assert_not_called()
        data = self.capture()
        self.checkpoint.write_bytes(b"replaced checkpoint")
        with (
            self.assertRaisesRegex(ValueError, "checkpoint changed"),
            hooks.request_scope(owner, data, self.manager, Mock()),
        ):
            pass
        self.loader.assert_not_called()

    def test_capture_owns_schedule_and_model_only_admits_new_diffusion_families(self):
        for family in (
            "Flux",
            "QwenImage21",
            "ZImage",
            "Anima",
            "StableCascade_C",
            "HunyuanImage21Refiner",
            "SDXLRefiner",
            "PiD",
            "SenseNovaU15",
        ):
            self.family = family
            data = self.capture()
            self.assertEqual(data[hooks.KEY].family, family)
            self.assertEqual(len(data[assets.KEY].artifacts), 1)
        self.family = "SenseNovaU15"
        with self.assertRaisesRegex(ValueError, "CLIP strength to 0"):
            self.capture(lora_hook_clip_strength=1)

    def test_photomaker_interval_adapter_admits_scheduled_and_constant_clip(self):
        captured = self.capture(
            reference_adapter="photomaker",
            lora_hook_clip_strength=0.5,
            lora_hook_schedule_clip=True,
        )
        self.assertTrue(captured[hooks.KEY].settings()["lora_hook_schedule_clip"])
        self.capture(
            reference_adapter="photomaker",
            lora_hook_clip_strength=0.5,
            lora_hook_schedule_clip=False,
        )
        self.capture(
            reference_adapter="photomaker", lora_hook_clip_strength=0, lora_hook_schedule_clip=True
        )

    def test_bad_keyframes_strengths_paths_counts_and_gguf_adapters_are_rejected(self):
        invalid = [
            dict(lora_hook_model="../outside.pt"),
            dict(lora_hook_model="C:/outside.pt"),
            dict(lora_hook_model_strength=True),
            dict(lora_hook_clip_strength=float("nan")),
            dict(lora_hook_keyframes='[{"start":0.2,"strength":1}]'),
            dict(lora_hook_keyframes='[{"start":0,"strength":1},{"start":0,"strength":2}]'),
            dict(lora_hook_keyframes='[{"start":0,"strength":1,"extra":3}]'),
            dict(lora_hook_keyframe_count=2.5),
            dict(lora_hook_keyframe_count=101),
            dict(lora_hook_end=0),
            dict(lora_hook_schedule_clip="true"),
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                hooks.settings(values)
        (self.root / "loras/adapter.gguf").write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "native LoRA state dictionary"):
            self.capture(lora_hook_model="adapter.gguf")


if __name__ == "__main__":
    unittest.main()
