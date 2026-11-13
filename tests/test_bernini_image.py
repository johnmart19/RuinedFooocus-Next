"""One-frame Bernini source admission, preparation and pinned native CPU topology."""

from concurrent.futures import CancelledError
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
import weakref
from unittest.mock import Mock, patch

from PIL import Image
import torch

from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.generation import bernini_image as image
from ruinedfooocus.features.generation import image_preparation as preparation
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.image_scope import require_image_checkpoint
from test_qwen_image21_conditioning import native_class, native_function


class BerniniImageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.rows, self.files, self.catalogue = {}, {}, {}
        for role, category in (
            ("high", "diffusion_models"),
            ("low", "diffusion_models"),
            ("clip", "clip"),
            ("vae", "vae"),
            ("lora", "loras"),
        ):
            filename = role + ".safetensors"
            header = {
                "__metadata__": {"model_type": "bernini_" + role},
                "head.modulation": dict(shape=[1, 2, 5120], dtype="F16", data_offsets=[0, 2]),
                "patch_embedding.weight": dict(
                    shape=[5120, 16, 1, 2, 2], dtype="F16", data_offsets=[0, 2]
                ),
            }
            raw = json.dumps(header, sort_keys=True).encode()
            payload = len(raw).to_bytes(8, "little") + raw + b"payload-" + role.encode()
            target = self.root / category / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            key = category + "/" + filename
            row = dict(
                filename=filename,
                catalogue_key=key,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                header_sha256=hashlib.sha256(raw).hexdigest(),
                url="https://huggingface.co/fixture/resolve/" + "a" * 40 + "/" + filename,
            )
            self.rows[role], self.files[role] = row, target
            self.catalogue[key] = dict(
                filename=filename,
                path="path_" + category,
                url=row["url"],
                size=row["bytes"],
                sha256=row["sha256"],
            )
        self.enterContext(patch.object(image, "sources", return_value=self.rows))
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy.samplers": NS(
                        KSampler=NS(
                            SAMPLERS=["res_multistep", "euler"], SCHEDULERS=["simple", "normal"]
                        )
                    )
                },
            )
        )
        self.manager = NS(
            DOWNLOADABLE_FILES=self.catalogue,
            model_paths={"cache_path": self.root / "cache"},
            get_model_folders=lambda kind: [
                self.root / ("diffusion_models" if kind == "checkpoints" else kind)
            ],
        )

        def request(key, destination=None):
            row = self.catalogue[key]
            return PreparationRequest(
                row["url"],
                destination or self.root / row["path"].removeprefix("path_") / row["filename"],
                expected_size=row["size"],
                sha256=row["sha256"],
            )

        self.manager.preparation_request = request
        self.models = NS(
            get_models_by_path=lambda *a, **k: {"baseModel": image.FAMILY},
            get_model_base=lambda row: row["baseModel"],
            get_file=lambda kind, name: self.files["high"],
        )

    def data(self, **values):
        result = dict(
            base_model_name=str(self.files["high"]),
            prompt="replace the cup",
            negative="blur",
            controlnet=dict(type="edit"),
            input_image=Image.new("RGB", (32, 48), "red"),
            reference_images=[Image.new("RGB", (48, 32), "blue")],
            width=32,
            height=48,
            custom_steps=40,
            cfg=5,
            sampler_name="res_multistep",
            scheduler="simple",
            clip_skip=1,
            bernini_low_model=self.rows["low"]["filename"],
            image_number=1,
            seed=42,
            image_resize_mode="pad",
            image_component_overrides={},
            image_engine="native",
        )
        result.update(values)
        return result

    def resources(self):
        return {
            image.CLIP_ROLE: self.rows["clip"]["filename"],
            image.VAE_ROLE: self.rows["vae"]["filename"],
        }

    def test_header_admission_is_exact_source_bound_and_not_filename_or_shared_wan_shape(self):
        self.assertEqual(image.primary_family(self.files["high"]), image.FAMILY)
        require_image_checkpoint(self.files["high"], image.FAMILY)
        with self.assertRaisesRegex(ValueError, "still images"):
            require_image_checkpoint(self.files["low"])
        renamed = self.files["high"].with_name("official-looking-bernini-high.safetensors")
        renamed.write_bytes(self.files["low"].read_bytes())
        self.assertIsNone(image.primary_family(renamed))
        self.files["high"].write_bytes(self.files["high"].read_bytes() + b"tampered")
        self.assertIsNone(image.primary_family(self.files["high"]))

    def test_capture_preserves_manual_sampling_and_immutable_ordered_references(self):
        data = self.data(
            custom_steps=12,
            cfg=3.25,
            sampler_name="euler",
            scheduler="normal",
            bernini_split_step=4,
        )
        result = image.capture(data, self.files["high"])
        plan = result[image.KEY]
        self.assertEqual(
            (plan.steps, plan.split, plan.cfg, plan.sampler, plan.scheduler),
            (12, 4, 3.25, "euler", "normal"),
        )
        data["input_image"].paste("green", (0, 0, 32, 48))
        self.assertEqual(plan.source.image().getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(plan.references[0].image().getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(result["negative"], "blur")

    def test_replay_requires_original_sources_order_and_settings(self):
        data = image.capture(self.data(), self.files["high"])
        self.assertEqual(image.capture(data, self.files["high"])[image.KEY], data[image.KEY])
        data["reference_images"] = [Image.new("RGB", (48, 32), "green")]
        with self.assertRaisesRegex(ValueError, "original ordered"):
            image.capture(data, self.files["high"])

    def test_masks_media_or_extra_frames_and_invalid_stage_settings_are_refused(self):
        for values in (
            dict(inpaint_toggle=True),
            dict(length=2),
            dict(frames=1),
            dict(cfg=float("nan")),
            dict(bernini_split_step=40),
            dict(bernini_ref_max_size=17),
            dict(reference_images=[Image.new("RGB", (16, 16))] * 9),
            dict(image_engine="sdcpp-vulkan"),
            dict(model_patch_recipe="anima-lllite"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                image.capture(self.data(**values), self.files["high"])

    def test_actual_capture_caller_prepares_both_signed_stages_without_acquiring(self):
        with patch.object(
            PreparationService, "submit", side_effect=AssertionError("capture must not acquire")
        ):
            result = preparation.capture_image_assets(
                self.data(),
                self.manager,
                dict(type="edit"),
                models=self.models,
                values=self.resources(),
                offline=True,
            )
        assets = result[preparation.KEY]
        self.assertEqual(
            [item.role for item in assets.artifacts], ["bernini-high", "bernini-low", "clip", "vae"]
        )
        self.assertTrue(
            all(item.signature is None and item.receipt_required for item in assets.artifacts)
        )
        self.assertTrue(all(item.request.sha256 for item in assets.artifacts))
        self.assertEqual(preparation.image_readiness(result).state, "preparable")

    def test_shared_preparation_verifies_installed_payloads_and_blocks_tampering(self):
        result = preparation.capture_image_assets(
            self.data(),
            self.manager,
            dict(type="edit"),
            models=self.models,
            values=self.resources(),
            offline=True,
        )
        service = PreparationService(
            transport=Mock(side_effect=AssertionError("network")),
            staging_root=self.root / "staging",
        )
        self.addCleanup(service.close)
        prepared = preparation.prepare_image_assets(
            result, service=service, check_cancelled=lambda: None
        )
        image._prepared(prepared)
        self.assertTrue(
            all(item.signature is not None for item in prepared[preparation.KEY].artifacts)
        )
        changed = self.files["low"]
        original = changed.read_bytes()
        changed.write_bytes(original[:-1] + b"X")
        # Header and length still identify a candidate; complete payload SHA is
        # enforced by the existing service before the native loader boundary.
        tampered = preparation.capture_image_assets(
            self.data(),
            self.manager,
            dict(type="edit"),
            models=self.models,
            values=self.resources(),
            offline=True,
        )
        with self.assertRaisesRegex(IOError, "SHA256"):
            preparation.prepare_image_assets(
                tampered, service=service, check_cancelled=lambda: None
            )

    def test_optional_lightx2v_is_a_signed_model_only_asset(self):
        result, assets = image.capture_assets(
            self.data(bernini_lightx2v=True, custom_steps=6, cfg=1),
            self.files["high"],
            self.manager,
            self.resources(),
        )
        self.assertEqual(result[image.KEY].split, 3)
        self.assertEqual(assets[-1].role, "bernini-lightx2v")
        self.assertEqual(assets[-1].request.sha256, self.rows["lora"]["sha256"])

    def test_actual_api_edit_submission_freezes_stage_fields_refs_and_signed_requirements(self):
        import ruinedfooocus.app
        from ruinedfooocus.app import api_runtime as api
        from ruinedfooocus.integrations.openai_api import ImageRequest

        values = self.resources() | {"image_engine": "native", "resolution": "default"}
        self.manager.paths = {
            "path_" + kind: [str(self.root / kind)] for kind in ("clip", "vae", "diffusion_models")
        }
        self.manager.get_abspath = Path
        self.models.get_names = lambda kind: []
        self.models.model_dirs = {"loras": [str(self.root / "loras")]}
        self.models.offline = True
        perf = NS(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *a: ["Own settings"],
            selection_for_model=lambda *a: "Own settings",
            get_perf_options=lambda *a: dict(
                custom_steps=12, cfg=3.25, sampler_name="euler", scheduler="simple", clip_skip=1
            ),
        )
        shared = NS(
            settings=NS(default_settings=values),
            models=self.models,
            path_manager=self.manager,
            performance_settings=perf,
            resolution_settings=NS(
                CUSTOM_RESOLUTION="Custom...",
                selection_for_model=lambda *a: "default",
                aspect_ratios={"default": (32, 48)},
            ),
        )
        state = dict(
            family=image.FAMILY,
            architecture=image.FAMILY,
            rows=[
                dict(
                    key=key,
                    category=role,
                    filename=self.files[role].name,
                    location=str(self.files[role]),
                    status="Installed",
                    compatibility="compatible",
                    download_key=self.rows[role]["catalogue_key"],
                    source=self.rows[role]["url"],
                    published_bytes=self.rows[role]["bytes"],
                )
                for key, role in ((image.CLIP_ROLE, "clip"), (image.VAE_ROLE, "vae"))
            ],
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch(
                "ruinedfooocus.features.models.component_profiles.component_inventory",
                return_value=state,
            ),
            patch.object(
                api,
                "image_defaults",
                dict(base_model_name=str(self.files["high"]), performance_selection="Own settings"),
            ),
            patch.object(
                api,
                "decode_image",
                side_effect=lambda encoded, **kwargs: Image.new(
                    "RGBA", (32, 48), "red" if encoded == "source" else "blue"
                ),
            ),
            patch.object(api, "submit", side_effect=lambda operation: operation),
        ):
            payload = ImageRequest(
                prompt="replace the cup",
                input_image="source",
                reference_images=["ref"],
                size="65x97",
                steps=12,
                cfg=3.25,
                sampler="euler",
                scheduler="simple",
                image_resize_mode="pad",
                bernini_low_model=self.rows["low"]["filename"],
                bernini_split_step=4,
                bernini_high_lora_strength=0,
                bernini_low_lora_strength=0,
            ).model_dump(exclude_none=True, exclude_unset=True)
            operation = api.images(payload, dict(name=str(self.files["high"]), family=image.FAMILY))
            closure = dict(
                zip(
                    operation.__code__.co_freevars,
                    (cell.cell_contents for cell in operation.__closure__),
                )
            )
            data = closure["data"]
        self.assertEqual(
            (data[image.KEY].steps, data[image.KEY].split, data[image.KEY].cfg), (12, 4, 3.25)
        )
        self.assertEqual(len(data[image.KEY].references), 1)
        self.assertEqual(
            (data["bernini_high_lora_strength"], data["bernini_low_lora_strength"]), (0, 0)
        )
        self.assertEqual(
            {item.role for item in data[preparation.KEY].artifacts},
            {"bernini-high", "bernini-low", "clip", "vae"},
        )
        self.assertEqual(data["_effective_image_plan"].family, image.FAMILY)

    def test_unsupported_or_missing_component_overrides_fail_without_download(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            image.capture_assets(
                self.data(image_component_overrides={image.CLIP_ROLE: "wrong.safetensors"}),
                self.files["high"],
                self.manager,
                self.resources(),
            )

    def test_actual_primary_loader_requires_prepared_signatures_and_loads_no_low_stage(self):
        import ruinedfooocus.app

        result, assets = image.capture_assets(
            self.data(), self.files["high"], self.manager, self.resources()
        )
        result[preparation.KEY] = preparation.ImageAssets(
            tuple(
                replace(item, signature=tuple(file_signature(item.request.destination)))
                for item in assets
            ),
            True,
        )
        self.manager.model_paths["embedding_path"] = [self.root / "embeddings"]
        model, clip, vae = object(), object(), object()
        loader = Mock(return_value=model)
        clip_loader = Mock(return_value=clip)
        comfy = NS(
            sd=NS(load_clip=clip_loader, CLIPType=NS(WAN="wan"), VAE=lambda **kwargs: vae),
            utils=NS(load_torch_file=lambda *a, **kw: {}),
        )
        events = []
        comfy.model_management = NS(
            cleanup_models=lambda: events.append("cleanup"),
            soft_empty_cache=lambda: events.append("empty"),
        )
        native_paths = NS(
            add_model_folder_path=Mock(),
            get_folder_paths=lambda kind: [str(self.root / "embeddings")],
        )

        class Owner:
            @staticmethod
            def StableDiffusionModel(unet, vae, clip, vision):
                return NS(unet=unet, vae=vae, clip=clip, clip_vision=vision)

        owner = Owner()
        old = {}

        class Retained:
            def __init__(self):
                self.cycle = self

        for name in (
            "xl_base",
            "xl_base_patched",
            "conditions",
            "cascade_stage_b_cache",
            "cascade_guidance_cache",
            "sdxl_refiner_cache",
            "native_negative_model_cache",
            "reference_adapter_models",
            "_native_gligen",
            "_native_hypernetwork",
        ):
            retained = Retained()
            old[name] = weakref.ref(retained)
            setattr(owner, name, retained)
        del retained

        def first_read(path):
            self.assertIsNone(owner.xl_base)
            self.assertIsNone(owner.xl_base_patched)
            self.assertTrue(all(reference() is None for reference in old.values()))
            self.assertEqual(events[-2:], ["cleanup", "empty"])
            return model

        loader.side_effect = first_read

        def clear_gligen(owner):
            owner._native_gligen = None

        def clear_hypernetwork(owner):
            owner._native_hypernetwork = None

        with (
            patch.dict(
                sys.modules,
                {
                    "comfy": comfy,
                    "comfy.sd": comfy.sd,
                    "comfy.utils": comfy.utils,
                    "comfy.model_management": comfy.model_management,
                    "ruinedfooocus.features.image_editing.gligen_layout": NS(
                        clear_cache=clear_gligen
                    ),
                    "ruinedfooocus.features.image_editing.native_hypernetwork": NS(
                        clear_cache=clear_hypernetwork
                    ),
                    "folder_paths": native_paths,
                    "ruinedfooocus.app.state": NS(path_manager=self.manager),
                    "ruinedfooocus.features.generation.pipeline_utils": NS(
                        load_diffusion_model=loader
                    ),
                },
            ),
            patch.object(ruinedfooocus.app, "state", NS(path_manager=self.manager), create=True),
            patch.object(image, "validate_component"),
        ):
            image.load(owner, result)
            self.assertIs(owner.xl_base.unet, model)
            self.assertEqual(loader.call_args.args, (str(self.files["high"]),))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(
                clip_loader.call_args.kwargs["embedding_directory"], [str(self.root / "embeddings")]
            )
            image.load(owner, result)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(events, ["cleanup", "empty", "cleanup", "empty"])
            bad = dict(result)
            bad[preparation.KEY] = preparation.ImageAssets((), True)
            with self.assertRaisesRegex(ValueError, "missing"):
                image.load(owner, bad)
            self.assertEqual(loader.call_count, 2)

    def test_actual_conditioning_adapter_preserves_negative_prompt_framing_and_reference_order(
        self,
    ):
        import ruinedfooocus.app

        data = self.data(width=35, height=51, clip_skip=2)
        plan = image.capture(data, self.files["high"])[image.KEY]
        node = self.runtime()
        prompts = []

        class Encoder:
            def encode(self, clip, text):
                prompts.append(text)
                return ([[torch.ones(1), {}]],)

        vae = NS(encode=lambda pixels: torch.tensor([float(pixels.mean())]))
        shared = NS(
            path_manager=NS(model_paths={"embedding_path": [], "cache_path": self.root / "cache"})
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "nodes": NS(CLIPTextEncode=Encoder),
                    "comfy_extras.nodes_bernini": NS(BerniniConditioning=node),
                    "ruinedfooocus.app.state": shared,
                },
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch(
                "ruinedfooocus.features.models.yaml_recipes.apply_clip_recipe",
                side_effect=lambda clip, *a: clip,
            ) as layer,
            patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder"),
            patch(
                "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
                side_effect=lambda text, *a: text,
            ),
        ):
            positive, negative, latent, frame = image.condition(plan, object(), vae, data)
        self.assertEqual(
            prompts, [plan.metadata()["prompt_header"] + data["prompt"], data["negative"]]
        )
        self.assertEqual(layer.call_args.args[1], 2)
        self.assertEqual((frame["width"], frame["height"]), (35, 51))
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 1, 8, 6))
        self.assertEqual(len(positive[0][1]["context_latents"]), 2)
        self.assertIs(positive[0][1]["context_latents"], negative[0][1]["context_latents"])

    def runtime(self):
        upscale = native_function("comfy/utils.py", "common_upscale", dict(torch=torch))
        setter = native_function("node_helpers.py", "conditioning_set_values", {})
        comfy = NS(
            utils=NS(common_upscale=upscale), model_management=NS(intermediate_device=lambda: "cpu")
        )
        scope = dict(
            torch=torch,
            comfy=comfy,
            node_helpers=NS(conditioning_set_values=setter),
            io=NS(NodeOutput=lambda *values: values),
        )
        scope["_resize_long_edge"] = native_function(
            "comfy_extras/nodes_bernini.py", "_resize_long_edge", scope
        )
        node = native_class(
            "comfy_extras/nodes_bernini.py", "BerniniConditioning", scope, methods=("execute",)
        )
        return node

    def test_pinned_native_conditioner_produces_one_frame_and_source_then_reference_order(self):
        node = self.runtime()
        seen = []

        def encode(pixels):
            seen.append(pixels.clone())
            return torch.tensor([float(pixels.mean())])

        positive, negative, latent = node.execute(
            [[torch.ones(1), {}]],
            [[torch.zeros(1), {}]],
            NS(encode=encode),
            32,
            48,
            1,
            1,
            source_video=torch.full((1, 48, 32, 3), 0.25),
            reference_images={
                "reference_image_1": torch.full((1, 64, 32, 3), 0.75),
                "reference_image_0": torch.full((1, 32, 64, 3), 0.5),
            },
            ref_max_size=32,
        )
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 1, 6, 4))
        self.assertEqual(torch.count_nonzero(latent["samples"]), 0)
        self.assertEqual([round(float(t.mean()), 2) for t in seen], [0.25, 0.5, 0.75])
        self.assertEqual([tuple(t.shape[1:3]) for t in seen], [(48, 32), (16, 32), (32, 16)])
        self.assertIs(positive[0][1]["context_latents"], negative[0][1]["context_latents"])

    def sampler_runtime(self, events, *, cancel=False):
        split = native_class(
            "comfy_extras/nodes_custom_sampler.py",
            "SplitSigmas",
            dict(io=NS(NodeOutput=lambda *v: v)),
            methods=("execute",),
        )

        def sample(model, noise, cfg, sampler, sigmas, positive, negative, latent, **kwargs):
            events.append(("sample", model, noise.clone(), sigmas.clone(), kwargs["seed"]))
            if kwargs.get("callback"):
                kwargs["callback"](0, latent, latent, len(sigmas) - 1)
            return latent + 1

        memory = NS(unload_model_and_clones=lambda model: events.append(("unload", model)))
        comfy = NS(
            sample=NS(
                sample_custom=sample, prepare_noise=lambda samples, seed: torch.ones_like(samples)
            ),
            utils=NS(PROGRESS_BAR_ENABLED=False),
            model_management=memory,
        )
        modules = {
            "comfy": comfy,
            "comfy.sample": comfy.sample,
            "comfy.utils": comfy.utils,
            "comfy.model_management": memory,
            "comfy_extras.nodes_custom_sampler": NS(
                BasicScheduler=NS(execute=lambda *a: (torch.arange(7, 0, -1).float(),)),
                SplitSigmas=split,
                KSamplerSelect=NS(execute=lambda name: (name,)),
            ),
        }
        return patch.dict(sys.modules, modules)

    def test_stage_topology_high_noise_once_zero_noise_low_sequential_release_and_callbacks(self):
        plan = image.capture(self.data(custom_steps=6, bernini_split_step=3), self.files["high"])[
            image.KEY
        ]
        events, progress = [], []
        with self.sampler_runtime(events):
            result = image.sample_stages(
                plan,
                "high",
                lambda: events.append(("load", "low")) or "low",
                {"samples": torch.zeros(1, 16, 1, 6, 4)},
                [],
                [],
                42,
                lambda *args: progress.append(args),
                lambda: None,
            )
        self.assertEqual(
            [(e[0], e[1]) for e in events],
            [
                ("sample", "high"),
                ("unload", "high"),
                ("load", "low"),
                ("sample", "low"),
                ("unload", "low"),
            ],
        )
        self.assertTrue(torch.all(events[0][2] == 1))
        self.assertEqual(torch.count_nonzero(events[3][2]), 0)
        torch.testing.assert_close(events[0][3][-1], events[3][3][0])
        self.assertEqual((events[0][4], events[3][4]), (42, 0))
        self.assertEqual([p[0] for p in progress], [0, 3])
        self.assertTrue(torch.all(result["samples"] == 2))

    def test_actual_process_offloads_conditioners_reuses_low_lora_consumer_and_publishes_one_still(
        self,
    ):
        import ruinedfooocus.app

        data, assets = image.capture_assets(
            self.data(custom_steps=6), self.files["high"], self.manager, self.resources()
        )
        data[preparation.KEY] = preparation.ImageAssets(
            tuple(
                replace(item, signature=tuple(file_signature(item.request.destination)))
                for item in assets
            ),
            True,
        )
        data["steps"] = 6
        events = []

        class Owner:
            @staticmethod
            def StableDiffusionModel(unet, vae, clip, vision):
                return NS(unet=unet, vae=vae, clip=clip, clip_vision=vision)

            def load_loras(self, rows, *, captured_loras=None):
                events.append(("loras", rows, captured_loras))
                self.xl_base_patched = self.xl_base

            def decode_latent(self, samples, **options):
                self.decoded = samples
                return torch.full((1, 48, 32, 3), 0.5)

        owner = Owner()
        owner.xl_base = owner.StableDiffusionModel(
            "high", NS(patcher="vae"), NS(patcher="text"), None
        )
        owner.xl_base_patched = owner.xl_base
        owner._bernini_plan = data[image.KEY]
        rows = [dict(name="ordinary.safetensors", weight=0.25)]
        owner._bernini_loras = (rows, ("captured-source",))
        shared = NS(path_manager=self.manager)
        frame = dict(width=32, height=48, canvas_width=32, canvas_height=48, left=0, top=0)
        with (
            self.sampler_runtime(events),
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "ruinedfooocus.features.generation.pipeline_utils": NS(
                        load_diffusion_model=lambda path: "low"
                    ),
                },
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(
                image,
                "condition",
                return_value=([], [], {"samples": torch.zeros(1, 16, 1, 6, 4)}, frame),
            ),
        ):
            comfy = sys.modules["comfy"]
            comfy.sd = NS()
            with patch.dict(sys.modules, {"comfy.sd": comfy.sd}):
                output = image.process(owner, data, None, lambda: None)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape[:2], (48, 32))
        self.assertEqual(events[:2], [("unload", "text"), ("unload", "vae")])
        self.assertIn(("loras", rows, ("captured-source",)), events)
        self.assertEqual(len(data["native_bernini_image"]["components"]), 4)
        self.assertEqual((data["width"], data["height"]), (32, 48))
        self.assertTrue(torch.all(owner.decoded["samples"] == 2))

    def test_cancellation_after_high_releases_it_without_loading_low(self):
        plan = image.capture(self.data(custom_steps=6), self.files["high"])[image.KEY]
        events = []

        def check():
            if events and events[0][0] == "sample":
                raise CancelledError()

        with self.sampler_runtime(events), self.assertRaises(CancelledError):
            image.sample_stages(
                plan,
                "high",
                Mock(side_effect=AssertionError("low load")),
                {"samples": torch.zeros(1, 16, 1, 6, 4)},
                [],
                [],
                42,
                None,
                check,
            )
        self.assertEqual(events[-1], ("unload", "high"))


if __name__ == "__main__":
    unittest.main()
