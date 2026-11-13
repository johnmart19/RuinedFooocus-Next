"""Pinned model-patch method bodies, immutable capture and full-call file leases."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from ruinedfooocus.app import resource_lifecycle as lifecycle
from ruinedfooocus.core.preparation import PreparationBusy, PreparationService
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_editing import model_patches as patches
from ruinedfooocus.features.models.model_identity import file_signature
from test_native_model_coverage import native_method
from test_native_variant_defaults import write_header


class NativeModel:
    def __init__(self, config):
        self.model = NS(model_config=config)
        self.patches = {"existing": "preserved"}

    def clone(self):
        result = NativeModel(self.model.model_config)
        result.patches = dict(self.patches)
        return result

    def get_model_object(self, name):
        return NS(percent_to_sigma=lambda percent: (1 - percent) * 10)

    def set_model_post_input_patch(self, value):
        self.patches["post"] = value

    def set_model_attn1_patch(self, value):
        self.patches["attn1"] = value

    def set_model_attn2_patch(self, value):
        self.patches["attn2"] = value

    def set_model_patch(self, value, name):
        self.patches[name] = value

    def set_model_noise_refiner_patch(self, value):
        self.patches["refiner"] = value

    def set_model_double_block_patch(self, value):
        self.patches["double"] = value


class Record:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class NativeModelPatchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.manager = NS(
            get_model_folders=lambda category: [self.root / category], DOWNLOADABLE_FILES={}
        )
        for category in ("model_patches", "clip_vision"):
            (self.root / category).mkdir()
        self.checkpoint = write_header(self.root / "checkpoint.safetensors", {"opaque": [1]})
        self.family = "Anima"
        self.models = NS(
            get_file=lambda *_: self.checkpoint,
            get_models_by_path=lambda *_, **__: {},
            get_model_base=lambda _: self.family,
        )
        self.guide = Image.new("RGB", (8, 8), "red")
        self.headers = {
            "anima-lllite": {
                "lllite_conditioning1.conv1.weight": [2, 4, 4, 4],
                "lllite_dit_blocks_0_self_attn_q_proj.down.weight": [2, 8],
            },
            "qwen-diffsynth": {
                "controlnet_blocks.0.y_rms.weight": [3072],
                "img_in.weight": [3072, 68],
            },
            "zimage-fun": {"control_all_x_embedder.2-1.weight": [3840, 132]},
            "uso-style": {
                "feature_embedder.mid_layer_norm.bias": [1152],
                "feature_embedder.mid_projection.weight": [3072, 1152],
                "feature_embedder.mid_embedding_linear.0.weight": [64, 729],
            },
        }
        for recipe, header in self.headers.items():
            write_header(self.root / f"model_patches/{recipe}.safetensors", header)
        write_header(
            self.root / "clip_vision/vision.safetensors",
            {
                "vision_model.embeddings.patch_embedding.weight": [1152, 3, 14, 14],
                "vision_model.embeddings.position_embedding.weight": [729, 1152],
                "vision_model.encoder.layers.26.layer_norm1.weight": [1152],
            },
        )
        self.service = PreparationService(staging_root=self.root / "staging")
        self.addCleanup(self.service.close)

    def capture(self, recipe="anima-lllite", **updates):
        self.family = patches.RECIPES[recipe][0]
        data = dict(
            base_model_name="checkpoint",
            model_patch_recipe=recipe,
            model_patch_model=recipe + ".safetensors",
            model_patch_vision="vision.safetensors",
            model_patch_image=self.guide,
            model_patch_strength=1,
            model_patch_start=0,
            model_patch_stop=1,
        )
        data.update(updates)
        result, files = patches.capture_assets(data, self.manager, self.models, {})
        result[assets.KEY] = assets.ImageAssets(files, True)
        return result

    def owner(self, recipe):
        name = {"uso-style": "Flux"}.get(recipe, patches.RECIPES[recipe][0])
        config = type(name, (), {"unet_config": dict(in_channels=64, hidden_size=3072)})()
        model = NativeModel(config)
        owner = NS(
            base_file_signature=file_signature(self.checkpoint), xl_base_patched=NS(unet=model)
        )
        return owner, model

    @contextmanager
    def native(self):
        Anima, Qwen, ZImage, USO = (
            type(name, (), {})
            for name in (
                "AnimaLLLite",
                "QwenImageBlockWiseControlNet",
                "ZImage_Control",
                "SigLIPMultiFeatProjModel",
            )
        )
        native_types = {
            "anima-lllite": Anima,
            "qwen-diffsynth": Qwen,
            "zimage-fun": ZImage,
            "uso-style": USO,
        }
        for value in native_types.values():
            value.cond_in_channels = 4
            value.additional_in_dim = 17
        comfy = NS(
            ldm=NS(
                anima=NS(
                    lllite=NS(
                        AnimaLLLite=Anima,
                        AnimaLLLitePatch=Record,
                        AnimaLLLiteAttentionPatch=Record,
                        AnimaLLLiteMLPPatch=Record,
                    )
                ),
                lumina=NS(controlnet=NS(ZImage_Control=ZImage)),
                qwen_image21=NS(model=NS(QwenImage21FunControl=type("Qwen21", (), {}))),
            )
        )
        scope = dict(
            comfy=comfy,
            torch=torch,
            ZImageControlPatch=Record,
            DiffSynthCnetPatch=Record,
            UsoStyleProjectorPatch=Record,
        )
        methods = {}
        for name, method in (
            ("AnimaLLLiteApply", "apply_patch"),
            ("QwenImageDiffsynthControlnet", "diffsynth_controlnet"),
            ("USOStyleReference", "apply_patch"),
        ):
            methods[name] = native_method("comfy_extras/nodes_model_patch.py", name, method, scope)
        methods["ZImageFunControlnet"] = methods["QwenImageDiffsynthControlnet"]
        loader = Mock(
            side_effect=lambda name: (NS(model=native_types[name.removesuffix(".safetensors")]()),)
        )
        native = NS(
            **methods,
            ModelPatchLoader=lambda: NS(load_model_patch=loader),
            comfy=comfy,
            QwenImageBlockWiseControlNet=Qwen,
            SigLIPMultiFeatProjModel=USO,
        )
        states = torch.zeros((1, 1, 1, 1)).expand(1, 27, 729, 1152)
        vision = NS(
            encode_image=lambda *_args, **_kwargs: NS(
                all_hidden_states=states, penultimate_hidden_states=states[:, -2]
            )
        )
        comfy.clip_vision = NS(load=lambda _: vision)
        encoder = native_method("nodes.py", "CLIPVisionEncode", "encode", {})
        original = assets.prepared_asset

        def leased(*args, **kwargs):
            return original(*args, **kwargs, service=self.service)

        modules = {
            "comfy_extras": NS(nodes_model_patch=native),
            "comfy_extras.nodes_model_patch": native,
            "comfy": comfy,
            "comfy.clip_vision": comfy.clip_vision,
            "folder_paths": NS(
                get_full_path_or_raise=lambda category, name: str(self.root / category / name)
            ),
            "nodes": NS(
                CLIPVisionEncode=encoder,
                VAEEncode=native_method("nodes.py", "VAEEncode", "encode", {}),
                SetLatentNoiseMask=native_method("nodes.py", "SetLatentNoiseMask", "set_mask", {}),
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(assets, "prepared_asset", leased),
            patch.object(lifecycle, "require_compute_owner"),
        ):
            yield loader

    def test_headers_are_distinct_from_generic_controls_and_native_torch_stays_explicit(self):
        for recipe in self.headers:
            self.assertEqual(
                patches.local_choices(self.manager, recipe),
                [(recipe + ".safetensors", "compatible")],
            )
        file = self.root / "model_patches/native.pt"
        file.write_bytes(b"opaque weights-only candidate")
        self.assertEqual(patches.file_contract(file, "anima-lllite"), "unknown")
        self.assertEqual(
            patches.file_contract(file.with_suffix(".gguf"), "anima-lllite"), "incompatible"
        )
        write_header(
            self.root / "model_patches/not-a-patch.safetensors", {"controlnet_blocks.0.weight": [1]}
        )
        self.assertNotIn(
            "not-a-patch.safetensors",
            [name for name, _ in patches.local_choices(self.manager, "anima-lllite")],
        )

    def test_capture_copies_inputs_rejects_wrong_family_missing_mask_and_inactive_api(self):
        captured = self.capture()
        self.guide.paste("blue", (0, 0, 8, 8))
        self.assertEqual(captured["model_patch_image"].getpixel((0, 0)), (255, 0, 0))
        self.family = "QwenImage"
        with self.assertRaisesRegex(ValueError, "checkpoint family"):
            patches.capture_assets(captured, self.manager, self.models, {})
        with self.assertRaisesRegex(ValueError, "source and its mask"):
            self.capture("zimage-fun", model_patch_inpaint_image=self.guide)
        data = self.capture()
        inactive, files = patches.capture_assets(data, self.manager, self.models, {"type": "rembg"})
        self.assertEqual(inactive["model_patch_recipe"], "none")
        self.assertEqual(files, ())
        with self.assertRaisesRegex(ValueError, "own guide"):
            patches.capture_assets(
                dict(data, _api_job=object()), self.manager, self.models, {"type": "control"}
            )

    def test_all_pinned_apply_methods_clone_without_mutating_cached_models_and_keep_white_mask_polarity(
        self,
    ):
        vae = NS(
            latent_channels=16,
            output_channels=3,
            spacial_compression_encode=lambda: 8,
            encode=lambda pixels: torch.zeros((1, 16, pixels.shape[1] // 8, pixels.shape[2] // 8)),
        )
        mask = Image.new("L", (8, 8), 255)
        for recipe in self.headers:
            with self.subTest(recipe=recipe), self.native() as loader:
                changes = {"model_patch_mask": mask} if recipe != "uso-style" else {}
                if recipe == "zimage-fun":
                    changes["model_patch_inpaint_image"] = self.guide
                data = self.capture(recipe, **changes)
                owner, model = self.owner(recipe)
                if recipe == "zimage-fun":
                    latent, pixels, prepared_mask, frame = patches.source_latent(
                        vae, data, (8, 8), "preserve", lambda: None
                    )
                    self.assertIs(latent["noise_mask"]._base, prepared_mask)
                with patches.apply_scope(
                    owner, model, vae, data, self.manager, lambda: None
                ) as output:
                    self.assertIsNot(output, model)
                    self.assertEqual(model.patches, {"existing": "preserved"})
                    self.assertEqual(output.patches["existing"], "preserved")
                    if recipe in ("qwen-diffsynth", "zimage-fun"):
                        native_mask = output.patches["double"].kwargs.get("mask")
                        if native_mask is None:
                            native_mask = output.patches["double"].args[4]
                        self.assertTrue(
                            torch.equal(
                                native_mask[..., 4:12, 4:12],
                                torch.zeros_like(native_mask[..., 4:12, 4:12]),
                            )
                        )
                    with (
                        self.assertRaises(PreparationBusy),
                        self.service.artifact_lease(
                            self.root / "model_patches" / (recipe + ".safetensors"),
                            write=True,
                            blocking=False,
                        ),
                    ):
                        self.fail("writer acquired the patch while sampling held it")
                self.assertEqual(data["native_model_patch"]["recipe"], recipe)
                loader.assert_called_once()

    def test_changed_file_cancellation_and_checkpoint_mismatch_do_not_publish_a_clone(self):
        data = self.capture()
        owner, model = self.owner("anima-lllite")
        vae = NS()
        with self.native() as loader:
            cancel = Mock(side_effect=RuntimeError("cancelled"))
            with (
                self.assertRaisesRegex(RuntimeError, "cancelled"),
                patches.apply_scope(owner, model, vae, data, self.manager, cancel),
            ):
                self.fail("cancelled request applied a patch")
            loader.assert_not_called()
            owner.base_file_signature = ()
            with (
                self.assertRaisesRegex(ValueError, "different checkpoint"),
                patches.apply_scope(owner, model, vae, data, self.manager, lambda: None),
            ):
                self.fail("wrong checkpoint")
            owner.base_file_signature = file_signature(self.checkpoint)
            path = self.root / "model_patches/anima-lllite.safetensors"
            path.write_bytes(path.read_bytes() + b"changed")
            with (
                self.assertRaisesRegex(ValueError, "changed after submission"),
                patches.apply_scope(owner, model, vae, data, self.manager, lambda: None),
            ):
                self.fail("changed patch")
            loader.assert_not_called()

    def test_request_local_patch_objects_differ_while_typed_loaded_weights_are_reused(self):
        data = self.capture()
        owner, model = self.owner("anima-lllite")
        with self.native() as loader:
            with patches.apply_scope(owner, model, NS(), data, self.manager, lambda: None) as first:
                first_patch = first.patches["post"]
            other = self.capture(model_patch_image=Image.new("RGB", (8, 8), "blue"))
            with patches.apply_scope(
                owner, model, NS(), other, self.manager, lambda: None
            ) as second:
                self.assertIsNot(first_patch, second.patches["post"])
                self.assertTrue(
                    torch.equal(first_patch.args[1][0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
                )
                self.assertTrue(
                    torch.equal(
                        second.patches["post"].args[1][0, 0, 0], torch.tensor([0.0, 0.0, 1.0])
                    )
                )
            loader.assert_called_once()

    def test_failed_type_or_cancelled_new_application_never_publishes_loaded_weight_cache(self):
        data = self.capture()
        owner, model = self.owner("anima-lllite")
        with self.native() as loader:
            load = loader.side_effect
            loader.side_effect = lambda _: (NS(model=NS()),)
            with (
                self.assertRaisesRegex(ValueError, "different model-patch type"),
                patches.apply_scope(owner, model, NS(), data, self.manager, lambda: None),
            ):
                self.fail("wrong native type")
            self.assertFalse(hasattr(owner, "native_model_patch_cache"))
            loader.side_effect = load
            cancellation = Mock(side_effect=[None, None, None, RuntimeError("cancelled")])
            with (
                self.assertRaisesRegex(RuntimeError, "cancelled"),
                patches.apply_scope(owner, model, NS(), data, self.manager, cancellation),
            ):
                self.fail("cancelled application")
            self.assertFalse(hasattr(owner, "native_model_patch_cache"))
            self.assertEqual(model.patches, {"existing": "preserved"})
            with patches.apply_scope(
                owner, model, NS(), data, self.manager, lambda: None
            ) as output:
                self.assertIsNot(output, model)
            self.assertTrue(hasattr(owner, "native_model_patch_cache"))


if __name__ == "__main__":
    unittest.main()
