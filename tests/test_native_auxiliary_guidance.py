"""Real pinned layout and attention functions on tiny CPU state dictionaries."""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import zipfile

import torch

from ruinedfooocus.features.generation import image_preparation as prep
from ruinedfooocus.features.image_editing import gligen_layout as gligen
from ruinedfooocus.features.image_editing import native_auxiliary_assets as assets
from ruinedfooocus.features.image_editing import native_hypernetwork as hyper
from test_qwen_image21_conditioning import native_class, native_function


def gligen_header(query=64, context=64):
    shapes = {
        "position_net.null_positive_feature": (context,),
        "position_net.null_position_feature": (64,),
        "position_net.linears.0.weight": (512, context + 64),
        "position_net.linears.0.bias": (512,),
        "position_net.linears.2.weight": (512, 512),
        "position_net.linears.2.bias": (512,),
        "position_net.linears.4.weight": (context, 512),
        "position_net.linears.4.bias": (context,),
    }
    fuser = {
        "linear.weight": (query, context),
        "linear.bias": (query,),
        "alpha_attn": (),
        "alpha_dense": (),
        "norm1.weight": (query,),
        "norm1.bias": (query,),
        "norm2.weight": (query,),
        "norm2.bias": (query,),
        "attn.to_q.weight": (query, query),
        "attn.to_k.weight": (query, query),
        "attn.to_v.weight": (query, query),
        "attn.to_out.0.weight": (query, query),
        "attn.to_out.0.bias": (query,),
        "ff.net.0.proj.weight": (query * 8, query),
        "ff.net.0.proj.bias": (query * 8,),
        "ff.net.2.weight": (query, query * 4),
        "ff.net.2.bias": (query,),
    }
    shapes.update(("input_blocks.1.1.fuser." + name, shape) for name, shape in fuser.items())
    return {key: dict(shape=list(shape), dtype="F32") for key, shape in shapes.items()}


def header_file(path, header):
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)
    return path


def hyper_state(dimension=4, **flags):
    def layers(multiplier):
        return OrderedDict(
            (
                ("0.weight", torch.eye(dimension) * multiplier),
                ("0.bias", torch.zeros(dimension)),
            )
        )

    return {dimension: [layers(2), layers(3)], "activation_func": "linear", **flags}


class Model:
    offload_device = torch.device("cpu")

    def __init__(self, query=64, context=64):
        attention = NS(
            attn2=NS(to_k=NS(weight=torch.empty(query, context))), disable_self_attn=False
        )
        spatial = type("SpatialTransformer", (), {})()
        spatial.proj_in = NS(weight=torch.empty(query, query, 1, 1))
        spatial.transformer_blocks = [attention]
        self.diffusion = type("UNetModel", (), {})()
        self.diffusion.input_blocks, self.diffusion.middle_block, self.diffusion.output_blocks = (
            [[spatial]],
            [],
            [],
        )
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise AssertionError(name)
        return self.diffusion

    def clone(self):
        result = Model()
        result.diffusion = self.diffusion
        result.model_options = deepcopy(self.model_options)
        return result


native_patcher = native_class(
    "comfy/model_patcher.py",
    "ModelPatcher",
    {},
    methods=("set_model_patch", "set_model_attn1_patch", "set_model_attn2_patch"),
)
for name in ("set_model_patch", "set_model_attn1_patch", "set_model_attn2_patch"):
    setattr(Model, name, getattr(native_patcher, name))


def ordinary_codec():
    return NS(latent_channels=4, downscale_ratio=8, upscale_ratio=8, output_channels=3)


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.checkpoint = self.root / "base.safetensors"
        self.checkpoint.write_bytes(b"owned checkpoint identity")
        self.gligen_file = header_file(self.root / "layout.safetensors", gligen_header())
        self.hyper_file = self.root / "attention.pt"
        torch.save(hyper_state(64), self.hyper_file)
        self.manager = NS(get_model_folders=lambda category: [self.root])
        self.models = NS(get_file=lambda *args: self.checkpoint)
        self.boxes = [
            dict(text="literal [cat:dog:0.5]", x=8, y=0, width=24, height=16),
            dict(text="a blue bowl", x=40, y=24, width=16, height=16),
        ]
        self.data = dict(
            base_model_name=self.checkpoint.name,
            width=64,
            height=48,
            clip_skip=1,
            gligen_model=self.gligen_file.name,
            gligen_layout=json.dumps(self.boxes),
            hypernetwork_model=self.hyper_file.name,
            hypernetwork_strength=-0.5,
        )

    def capture(self, **overrides):
        data = self.data | overrides
        data, first = gligen.capture_assets(data, self.manager, self.models, "BaseModel")
        data, second = hyper.capture_assets(data, self.manager, self.models, "BaseModel")
        data[prep.KEY] = prep.ImageAssets(first + second, offline=True)
        return data

    def runtime(self, pooled=64):
        native = NS(storage=NS(annotate_state_dict=lambda *a: None))
        reader = native_function(
            "comfy/utils.py",
            "load_torch_file",
            dict(torch=torch, MMAP_TORCH_FILES=False, comfy=native),
        )
        native.utils = NS(load_torch_file=Mock(wraps=reader))
        load_hyper = native_function(
            "comfy_extras/nodes_hypernetwork.py",
            "load_hypernetwork_patch",
            dict(torch=torch, logging=logging, comfy=native),
        )
        native.sd = NS(load_gligen=Mock(return_value=NS(model=object())))
        native.model_management = NS(unload_model_and_clones=Mock())
        textbox = native_class("nodes.py", "GLIGENTextBoxApply", {}, methods=("append",))
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy": native,
                    "comfy.utils": native.utils,
                    "comfy.sd": native.sd,
                    "comfy.model_management": native.model_management,
                    "comfy_extras.nodes_hypernetwork": NS(load_hypernetwork_patch=load_hyper),
                    "nodes": NS(GLIGENTextBoxApply=textbox),
                },
            )
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.yaml_recipes.apply_clip_recipe",
                side_effect=lambda clip, *args: clip,
            )
        )
        self.enterContext(
            patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder")
        )
        clip = NS(
            tokenize=Mock(side_effect=lambda text: text),
            encode_from_tokens=Mock(
                side_effect=lambda text, **kw: (torch.zeros(1, 2, pooled), torch.ones(1, pooled))
            ),
        )
        owner = NS(xl_base_patched=NS(unet=Model(), vae=ordinary_codec(), clip=clip))
        return owner, native, clip

    def test_off_never_resolves_reads_or_clones_an_auxiliary(self):
        forbidden = NS(
            get_file=Mock(side_effect=AssertionError("lookup")),
            get_model_folders=Mock(side_effect=AssertionError("scan")),
        )
        for feature in (gligen, hyper):
            output, captured = feature.capture_assets({}, forbidden, forbidden, "Flux")
            self.assertEqual((output, captured), ({}, ()))
        owner = NS()
        with assets.sampling_scope(owner, {}, forbidden):
            self.assertFalse(vars(owner))

    def test_capture_freezes_order_geometry_public_identity_and_exact_artifacts(self):
        data = self.capture()
        self.assertEqual([box.metadata() for box in data[gligen.KEY].boxes], self.boxes)
        self.assertEqual(data[hyper.KEY].strength, -0.5)
        self.assertEqual(
            [item.role for item in data[prep.KEY].artifacts], ["gligen", "hypernetwork"]
        )
        self.assertEqual(data["gligen_model_signature"][0], self.gligen_file.name)
        self.assertTrue(all(isinstance(v, str) for v in data["gligen_model_signature"][2:]))
        self.assertNotIn(str(self.root), json.dumps(data[gligen.KEY].asset.metadata()))
        self.boxes[0]["text"] = "changed caller data"
        self.assertEqual(data[gligen.KEY].boxes[0].text, "literal [cat:dog:0.5]")
        self.assertEqual(self.capture()["width"], 64)

    def test_actual_native_textbox_keeps_all_conditioning_intervals_and_literal_boxes(self):
        data = self.capture(hypernetwork_model=None)
        owner, native, clip = self.runtime()
        source = [
            [torch.zeros(1, 1, 64), {"timestep_start": 20.0}],
            [torch.ones(1, 1, 64), {"timestep_end": 10.0}],
        ]
        positive = gligen.apply(owner, source, data, self.manager, lambda: None)
        self.assertEqual(
            [call.args[0] for call in clip.tokenize.call_args_list],
            [row["text"] for row in self.boxes],
        )
        self.assertTrue(
            all(
                call.kwargs == {"return_pooled": "unprojected"}
                for call in clip.encode_from_tokens.call_args_list
            )
        )
        for original, conditioned in zip(source, positive):
            self.assertIs(original[0], conditioned[0])
            self.assertNotIn("gligen", original[1])
            kind, model, params = conditioned[1]["gligen"]
            self.assertEqual(kind, "position")
            self.assertIs(model, native.sd.load_gligen.return_value)
            self.assertEqual([row[1:] for row in params], [(2, 3, 0, 1), (2, 2, 3, 5)])
        # The actual native position normalizer uses this exact latent canvas.
        # Avoid evaluating module constructors/device selection: execute only its pure CPU body.
        position = native_class(
            "comfy/gligen.py", "Gligen", dict(torch=torch), methods=("set_position",)
        )
        consumer = NS(max_objs=30, key_dim=64, _set_position=Mock(return_value="native callback"))
        self.assertEqual(
            position.set_position(consumer, (1, 4, 6, 8), positive[0][1]["gligen"][2], "cpu"),
            "native callback",
        )
        boxes, masks, pooled = consumer._set_position.call_args.args
        self.assertTrue(
            torch.equal(
                boxes[0, :2], torch.tensor([[0.125, 0, 0.5, 1 / 3], [0.625, 0.5, 0.875, 5 / 6]])
            )
        )
        self.assertEqual(masks.sum().item(), 2)
        self.assertEqual(tuple(pooled.shape), (1, 30, 64))
        gligen.apply(owner, source, data, self.manager, lambda: None)
        native.sd.load_gligen.assert_called_once_with(str(self.gligen_file))

    def test_gligen_runtime_rejects_wrong_pooled_unet_and_codec_before_sampling(self):
        data = self.capture(hypernetwork_model=None)
        owner, native, _ = self.runtime(pooled=32)
        with self.assertRaisesRegex(ValueError, "pooled width"):
            gligen.apply(owner, [[torch.zeros(1), {}]], data, self.manager, lambda: None)
        gligen.clear_cache(owner)
        owner.xl_base_patched.unet = Model(128, 64)
        native.sd.load_gligen.reset_mock()
        with self.assertRaisesRegex(ValueError, "do not match"):
            gligen.apply(owner, [], data, self.manager, lambda: None)
        native.sd.load_gligen.assert_not_called()
        owner.xl_base_patched.vae.latent_channels = 16
        with self.assertRaisesRegex(ValueError, "codec"):
            gligen.apply(owner, [], data, self.manager, lambda: None)
        native.sd.load_gligen.assert_not_called()

    def test_gligen_cache_rejects_a_changed_source_then_reloads_explicit_reselection(self):
        data = self.capture(hypernetwork_model=None)
        owner, native, _ = self.runtime()
        gligen.apply(owner, [], data, self.manager, lambda: None)
        header_file(self.gligen_file, gligen_header() | {"__metadata__": {"revision": "changed"}})
        with self.assertRaisesRegex(ValueError, "changed"):
            gligen.apply(owner, [], data, self.manager, lambda: None)
        reselected = self.capture(hypernetwork_model=None)
        gligen.apply(owner, [], reselected, self.manager, lambda: None)
        self.assertEqual(native.sd.load_gligen.call_count, 2)
        native.model_management.unload_model_and_clones.assert_called_once()

    def test_actual_native_hypernetwork_math_signed_strength_and_clone_isolation(self):
        data = self.capture(gligen_model=None)
        owner, native, _ = self.runtime()
        base = owner.xl_base_patched.unet
        patched = hyper.patch(owner, base, data, self.manager, lambda: None)
        callbacks = patched.model_options["transformer_options"]["patches"]
        self.assertNotIn("patches", base.model_options["transformer_options"])
        self.assertIs(callbacks["attn1_patch"][0], callbacks["attn2_patch"][0])
        q, k, v = (torch.ones(1, 2, 64) * n for n in (1, 2, 3))
        actual = callbacks["attn2_patch"][0](q, k, v, {})
        self.assertIs(actual[0], q)
        self.assertTrue(torch.equal(actual[1], torch.zeros_like(k)))
        self.assertTrue(torch.equal(actual[2], v * -0.5))
        # Reuse immutable weights while retaining each request's own signed strength.
        other = dict(data)
        other[hyper.KEY] = replace(data[hyper.KEY], strength=0.25)
        positive = hyper.patch(owner, base, other, self.manager, lambda: None)
        new_callback = positive.model_options["transformer_options"]["patches"]["attn2_patch"][0]
        self.assertEqual(callbacks["attn2_patch"][0].strength, -0.5)
        self.assertTrue(torch.equal(new_callback(q, k, v, {})[1], k * 1.5))
        self.assertTrue(
            all(
                call.kwargs.get("safe_load") is True
                for call in native.utils.load_torch_file.call_args_list
            )
        )
        self.assertEqual(native.utils.load_torch_file.call_count, 2)

    def test_native_activation_layernorm_and_dropout_schema_and_bad_activation_refusal(self):
        owner, native, _ = self.runtime()
        state = hyper_state(
            4,
            activation_func="relu",
            is_layer_norm=True,
            use_dropout=True,
            activate_output=True,
            last_layer_dropout=True,
        )
        for network in state[4]:
            network["1.weight"], network["1.bias"] = torch.ones(4), torch.zeros(4)
        self.assertEqual(hyper.validate_state(state, {4}), (4,))
        torch.save(state, self.hyper_file)
        from comfy_extras.nodes_hypernetwork import load_hypernetwork_patch

        loaded = load_hypernetwork_patch(str(self.hyper_file), 1)
        self.assertTrue(any(isinstance(m, torch.nn.LayerNorm) for m in loaded.hypernet[4][0]))
        self.assertTrue(any(isinstance(m, torch.nn.Dropout) for m in loaded.hypernet[4][0]))
        state["activation_func"] = "invented"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            hyper.validate_state(state, {4})
        for bad in (
            {"4": state[4], 4: state[4]},
            hyper_state(8),
            hyper_state(4, use_dropout="false"),
        ):
            with self.assertRaises(ValueError):
                hyper.validate_state(bad, {4})
        self.assertFalse(hasattr(owner, "_native_hypernetwork"))

    def test_bad_native_loader_result_and_changed_asset_never_return_unmodified_success(self):
        data = self.capture(gligen_model=None)
        owner, _, _ = self.runtime()
        with (
            patch.object(
                sys.modules["comfy_extras.nodes_hypernetwork"],
                "load_hypernetwork_patch",
                return_value=None,
            ),
            self.assertRaisesRegex(ValueError, "no compatible"),
        ):
            hyper.patch(owner, owner.xl_base_patched.unet, data, self.manager, lambda: None)
        self.hyper_file.write_bytes(b"changed after admission")
        with self.assertRaisesRegex(ValueError, "changed"):
            hyper.patch(owner, owner.xl_base_patched.unet, data, self.manager, lambda: None)
        with self.assertRaisesRegex(ValueError, "changed"):
            hyper.capture_assets(data, self.manager, self.models, "BaseModel")

    def test_sampling_scope_holds_both_sources_and_offloads_on_error_or_cancel(self):
        data = self.capture()
        active = set()

        @contextmanager
        def lease(data, role, name, manager, category):
            self.assertNotIn(role, active)
            active.add(role)
            try:
                yield self.root / name
            finally:
                active.remove(role)

        owner = NS(_native_hypernetwork=(None, NS(to=Mock()), None, "cpu"))
        with patch.object(prep, "prepared_asset", lease):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with assets.sampling_scope(owner, data, self.manager):
                    self.assertEqual(active, {"gligen", "hypernetwork"})
                    raise RuntimeError("cancelled")
        self.assertFalse(active)
        owner._native_hypernetwork[1].to.assert_called_once_with("cpu")

    def test_bounded_layout_header_paths_and_native_weights_only_reader(self):
        for value in (
            "[" * 20000 + "]" * 20000,
            "[]",
            json.dumps([self.boxes[0] | {"x": True}]),
            json.dumps([self.boxes[0] | {"width": 17}]),
            json.dumps(self.boxes * 16),
        ):
            with self.subTest(value=value[:30]), self.assertRaises(ValueError):
                gligen.parse_layout(value)
        data = self.capture()
        with self.assertRaisesRegex(ValueError, "inside"):
            gligen.validate_canvas(data[gligen.KEY], 48, 48)
        for name in ("../outside.pt", "C:/outside.pt", "/outside.pt", "x.gguf"):
            with self.assertRaises(ValueError):
                assets.selected_asset(self.manager, "hypernetwork", name)
        header = gligen_header()
        header.pop("position_net.linears.0.bias")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            gligen.header_profile(header)
        nested = '{"meta":' + "[" * 20000 + "]" * 20000 + "}"
        self.gligen_file.write_bytes(len(nested).to_bytes(8, "little") + nested.encode())
        with self.assertRaisesRegex(ValueError, "header"):
            gligen.capture_assets(self.data, self.manager, self.models, "BaseModel")
        self.runtime()
        torch.save({"metadata": "safe", "bad": [1]}, self.hyper_file)
        with patch.object(torch, "load", wraps=torch.load) as loader:
            assets.safe_native_state(self.hyper_file)
        self.assertIs(loader.call_args.kwargs["weights_only"], True)
        torch.save(
            {"untrusted": Path("unused"), "metadata": "weights-only must reject"}, self.hyper_file
        )
        with self.assertRaisesRegex(ValueError, "weights-only"):
            assets.safe_native_state(self.hyper_file)
        oversized = gligen_header(query=4096)
        oversized.update(
            (key.replace("input_blocks.1.", "input_blocks.2."), value)
            for key, value in list(oversized.items())
            if ".fuser." in key
        )
        with self.assertRaisesRegex(ValueError, "parameter budget"):
            gligen.header_profile(oversized)

    def test_discovery_never_deserializes_and_oversized_zip_is_rejected_cold(self):
        with patch.object(
            assets, "safe_native_state", side_effect=AssertionError("browsing weights")
        ):
            self.assertIn(self.hyper_file.name, assets.local_choices(self.manager, "hypernetwork"))
        with zipfile.ZipFile(self.hyper_file, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr("archive/data.pkl", b"x" * (16 * 1024**2 + 1))
        with self.assertRaisesRegex(ValueError, "oversized Torch"):
            hyper.capture_assets(self.data, self.manager, self.models, "BaseModel")

    def test_archive_directory_is_bounded_before_constructing_zip_entries(self):
        with zipfile.ZipFile(self.hyper_file, "w") as output:
            for index in range(4097):
                output.writestr(f"entry-{index}", b"")
        with (
            patch.object(
                zipfile, "ZipFile", side_effect=AssertionError("unbounded ZIP allocation")
            ),
            self.assertRaisesRegex(ValueError, "directory bounds"),
        ):
            hyper.capture_assets(self.data, self.manager, self.models, "BaseModel")


if __name__ == "__main__":
    unittest.main()
