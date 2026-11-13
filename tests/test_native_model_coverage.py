"""Ordinary native image routes use pinned components, conditioning and latents."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.models.image_components import inventory
import test_model_cache as cache_fixtures
from test_model_cache import method

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
PIPELINE = "ruinedfooocus/features/generation/image_pipeline.py"
CONTRACTS = {
    "Chroma": ("CHROMA", "SD3", ["clip_t5", "vae_flux"]),
    "ChromaRadiance": ("CHROMA", "ChromaRadience", ["clip_t5"]),
    "LongCatImage": ("LONGCAT_IMAGE", "SD3", ["clip_qwen25", "vae_flux"]),
    "ZImagePixelSpace": ("LUMINA2", "ChromaRadience", ["clip_qwen3_4b"]),
}


def native_method(filename, class_name, method_name, scope, bases=()):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    node.body = [
        node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    node.body[0].returns = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename, "exec"
        ),
        scope,
    )
    return scope[class_name]


def route_info():
    tree = ast.parse((ROOT / PIPELINE).read_text(encoding="utf-8"))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "known_model_info"
            for target in node.targets
        )
    )
    names = {
        node.attr
        for node in ast.walk(node)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "CLIPType"
    }
    scope = dict(
        comfy=SimpleNamespace(
            sd=SimpleNamespace(CLIPType=SimpleNamespace(**{name: name for name in names}))
        )
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), PIPELINE, "exec"), scope)
    return scope["known_model_info"]


class NativeModelCoverageTests(unittest.TestCase):
    def test_native_family_overrides_stale_label_without_rewriting_publisher_metadata(self):
        lookup = method(
            "ruinedfooocus/features/models/model_handler.py",
            "get_models_by_path",
            dict(json=json, Path=Path),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "local.safetensors"
            model.write_bytes(b"local fixture")
            cached = root / "local.json"
            metadata = dict(baseModel="SDXL 1.0", name="Publisher title", modelId=42)
            cached.write_text(json.dumps(metadata))
            before = cached.read_bytes()
            owner = SimpleNamespace(
                cache_paths={"checkpoints": root}, offline=True, detect_checkpoint_base=Mock()
            )
            for family in CONTRACTS:
                with self.subTest(family=family):
                    owner.detect_checkpoint_base.return_value = family
                    self.assertEqual(
                        lookup(owner, "checkpoints", model), metadata | dict(baseModel=family)
                    )
                    self.assertEqual(cached.read_bytes(), before)
            owner.detect_checkpoint_base.return_value = "SDXL 1.0"
            metadata["baseModel"] = "Illustrious"
            cached.write_text(json.dumps(metadata))
            self.assertEqual(lookup(owner, "checkpoints", model), metadata)

    def test_native_runtime_roles_and_passive_inventory_agree(self):
        get_info = method(
            PIPELINE, "get_clip_and_vae", dict(settings=SimpleNamespace(default_settings={}))
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "unrelated.safetensors"
            header = json.dumps({"weight": {"shape": [1]}}).encode()
            model.write_bytes(len(header).to_bytes(8, "little") + header)
            paths = SimpleNamespace(
                paths={"path_clip": str(root), "path_vae": str(root)},
                get_abspath=Path,
                DOWNLOADABLE_FILES={},
                download_file=Mock(),
            )
            for family, (clip_type, latent, keys) in CONTRACTS.items():
                unet = SimpleNamespace(
                    model=type(family, (), {"model_config": SimpleNamespace(unet_config={})})()
                )
                with self.subTest(family=family):
                    info = get_info(SimpleNamespace(known_model_info=route_info()), unet)
                    self.assertEqual((info["clip_type"], info["latent"]), (clip_type, latent))
                    state = inventory(model, family, {}, paths)
                    self.assertEqual([row["key"] for row in state["rows"]], keys)
                    self.assertEqual(info["clip_keys"], keys[:1])
                    if latent == "ChromaRadience":
                        self.assertEqual(info["vae_name"], "pixel_space")
                        self.assertIn("Pixel-space", state["note"])
                        self.assertFalse(any(row["category"] == "vae" for row in state["rows"]))
            paths.download_file.assert_not_called()

    def test_native_latent_constructors_match_admitted_runtime_shapes(self):
        scope = dict(
            torch=torch,
            comfy=SimpleNamespace(
                model_management=SimpleNamespace(
                    intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32
                )
            ),
            io=SimpleNamespace(NodeOutput=lambda value: (value,)),
        )
        nodes = {
            "ChromaRadience": native_method(
                "comfy_extras/nodes_chroma_radiance.py",
                "EmptyChromaRadianceLatentImage",
                "execute",
                scope,
            ),
            "SD3": native_method(
                "comfy_extras/nodes_sd3.py", "EmptySD3LatentImage", "execute", scope
            ),
        }
        # Execute the actual production selection branch with pinned native CPU constructors.
        tree = ast.parse((ROOT / PIPELINE).read_text(encoding="utf-8"))
        match = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Match)
            and isinstance(node.subject, ast.Name)
            and node.subject.id == "latent_type"
        )
        for family, (_, latent_type, _) in CONTRACTS.items():
            with self.subTest(family=family):
                scope.update(
                    latent_type=latent_type,
                    gen_data=dict(width=64, height=48),
                    EmptyChromaRadianceLatentImage=nodes["ChromaRadience"],
                    EmptySD3LatentImage=nodes["SD3"],
                )
                exec(compile(ast.Module(body=[match], type_ignores=[]), PIPELINE, "exec"), scope)
                expected = (1, 3, 48, 64) if latent_type == "ChromaRadience" else (1, 16, 6, 8)
                self.assertEqual(tuple(scope["latent"]["samples"].shape), expected)

    def test_longcat_native_metadata_survives_production_text_encoding(self):
        scope = {}
        encode = native_method("nodes.py", "CLIPTextEncode", "encode", scope)()
        conditioning = [
            [torch.ones(1, 2, 3), dict(pooled_output="native-pooled", attention_mask="native-mask")]
        ]
        clip = SimpleNamespace(
            tokenize=Mock(return_value={"qwen25_7b": [[(151644, 1)]]}),
            encode_from_tokens_scheduled=Mock(return_value=conditioning),
        )
        run = method(
            PIPELINE,
            "textencode",
            dict(
                CLIPTextEncode=lambda: encode,
                warm_patched_text_encoder=Mock(),
                path_manager=SimpleNamespace(
                    model_paths={"embedding_path": [], "cache_path": ROOT / "tmp"}
                ),
            ),
        )
        owner = SimpleNamespace(
            conditions={"positive": dict(text=None)}, xl_base_patched=SimpleNamespace(clip=clip)
        )
        with patch(
            "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
            side_effect=lambda text, *args: text,
        ):
            run(owner, "positive", 'a sign "LongCat"', 1)
        self.assertIs(owner.conditions["positive"]["cache"], conditioning)
        clip.tokenize.assert_called_once_with('a sign "LongCat"')
        clip.encode_from_tokens_scheduled.assert_called_once_with({"qwen25_7b": [[(151644, 1)]]})

    def test_production_guidance_branch_preserves_native_cfg_at_sampler_handoff(self):
        tree = ast.parse((ROOT / PIPELINE).read_text(encoding="utf-8"))
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "cfg" for target in child.targets
                )
                and isinstance(child.value, ast.Constant)
                and child.value.value == 1.0
                for child in node.body
            )
        )
        sample = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "comfy.sample.sample"
        )

        class Flux:
            pass

        for family in ("LongCatImage", "Chroma", "ChromaRadiance", "Flux"):
            with self.subTest(family=family):
                model = Flux() if family == "Flux" else type(family, (Flux,), {})()
                positive, negative = [["positive", {}]], [["negative", {}]]
                sampler = Mock(return_value="sampled")
                scope = dict(
                    Flux=Flux,
                    architecture=family,
                    native_plan=None,
                    ordered_family=None,
                    self=SimpleNamespace(
                        xl_base=SimpleNamespace(unet=SimpleNamespace(model=model))
                    ),
                    controlnet={},
                    positive_cond=positive,
                    negative_conditioning=negative,
                    cfg=2.5,
                    conditioning_set_values=lambda cond, values: [
                        [item[0], item[1] | values] for item in cond
                    ],
                    comfy=SimpleNamespace(
                        sample=SimpleNamespace(sample=sampler),
                        utils=SimpleNamespace(PROGRESS_BAR_ENABLED=True),
                    ),
                    sampling_model="native-model",
                    noise="noise",
                    gen_data=dict(steps=12),
                    sampler_name="euler",
                    scheduler="simple",
                    fixed_sigmas=None,
                    latent_image="latent",
                    denoise=None,
                    noise_mask=None,
                    callback_function=None,
                    seed=7,
                )
                exec(
                    compile(ast.Module(body=[branch, sample], type_ignores=[]), PIPELINE, "exec"),
                    scope,
                )
                call = sampler.call_args.kwargs
                self.assertEqual(call["cfg"], 1.0 if family == "Flux" else 2.5)
                self.assertEqual(
                    call["positive"][0][1], {"guidance": 2.5} if family == "Flux" else {}
                )
                self.assertIs(call["negative"], negative)

    def test_native_chroma_default_guidance_and_longcat_removal_are_distinct_from_cfg(self):
        class Flux:
            def extra_conds(self, **kwargs):
                return dict(guidance="inherited")

        scope = dict(
            Flux=Flux,
            torch=torch,
            comfy=SimpleNamespace(conds=SimpleNamespace(CONDRegular=lambda tensor: tensor)),
        )
        chroma = native_method(
            "comfy/model_base.py", "Chroma", "extra_conds", scope, bases=("Flux",)
        )
        longcat = native_method(
            "comfy/model_base.py", "LongCatImage", "extra_conds", scope, bases=("Flux",)
        )
        self.assertEqual(chroma().extra_conds()["guidance"].tolist(), [0.0])
        self.assertNotIn("guidance", longcat().extra_conds(guidance=2.5))

    def test_pixel_vae_native_conversion_uses_no_external_weight_file(self):
        # Reuse the real production loader fixture, replacing only native weight I/O.
        fixture = cache_fixtures.ModelCacheTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        checkpoint = fixture.file("checkpoint.safetensors")
        fixture.file("encoder.safetensors")
        for family in ("ChromaRadiance", "ZImagePixelSpace"):
            with self.subTest(family=family):
                load, owner, _, backend, _ = fixture.base_loader(checkpoint, bare=True)
                owner.get_clip_and_vae.return_value.update(
                    vae_name="pixel_space", vae_key="pixel_space"
                )
                load()
                self.assertIsNone(owner.load_error)
                state = backend.sd.VAE.call_args.kwargs["sd"]
                self.assertEqual(list(state), ["pixel_space_vae"])
                backend.utils.load_torch_file.assert_not_called()
        tree = ast.parse((NATIVE / "comfy/pixel_space_convert.py").read_text())
        scope = dict(torch=torch)
        exec(compile(tree, "native_pixel_vae", "exec"), scope)
        conversion = scope["PixelspaceConversionVAE"]()
        native_sd = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        branch = next(
            node
            for node in ast.walk(native_sd)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Constant)
            and node.test.left.value == "pixel_space_vae"
        )
        native_vae = SimpleNamespace()
        scope.update(
            self=native_vae,
            comfy=SimpleNamespace(
                pixel_space_convert=SimpleNamespace(
                    PixelspaceConversionVAE=scope["PixelspaceConversionVAE"]
                )
            ),
        )
        exec(
            compile(
                ast.Module(body=branch.body, type_ignores=[]), "native_pixel_vae_branch", "exec"
            ),
            scope,
        )
        self.assertIsInstance(native_vae.first_stage_model, scope["PixelspaceConversionVAE"])
        self.assertEqual(
            (
                native_vae.downscale_ratio,
                native_vae.upscale_ratio,
                native_vae.latent_channels,
                native_vae.latent_dim,
            ),
            (1, 1, 3, 2),
        )
        pixels = torch.rand(1, 3, 8, 12)
        self.assertTrue(torch.equal(conversion.decode(conversion.encode(pixels)), pixels))


if __name__ == "__main__":
    unittest.main()
