from ruinedfooocus.core.paths import asset_path, resource_path

"""Native preview selection/fallback and image-upscale error boundaries."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

import ruinedfooocus.app

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


class NativeImageHelperTests(unittest.TestCase):
    def test_component_variants_use_native_config_without_reading_state_dicts(self):
        from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS
        from ruinedfooocus.features.models.resource_defaults import resource_name

        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        nodes = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "get_clip_and_vae"
            or isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "known_model_info" for t in n.targets)
        ]
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            comfy=Mock(),
            settings=SimpleNamespace(default_settings={}),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "image_components", "exec"), scope)
        owner = SimpleNamespace(known_model_info=scope["known_model_info"])
        for name, config, family in [
            ("Lumina2", {"dim": 2304}, "Lumina2"),
            ("Lumina2", {"dim": 3840}, "ZImage"),
            ("Lumina2", {"clip_text_dim": 1024}, "NewBieImage"),
            ("Flux2", {"depth_single_blocks": 48}, "Flux2"),
            ("Flux2", {"depth_single_blocks": 24}, "Flux2Klein9B"),
            ("Flux2", {"depth_single_blocks": 20}, "Flux2Klein4B"),
            ("MOSSImage", {}, "MOSSImage"),
        ]:
            with self.subTest(family=family):
                model = type(name, (), {})()
                model.model_config = SimpleNamespace(unet_config=config)
                patcher = SimpleNamespace(
                    model=model,
                    model_state_dict=Mock(side_effect=AssertionError("Must not read tensors")),
                )
                result = scope["get_clip_and_vae"](owner, patcher)
                expected = IMAGE_COMPONENTS[family]
                self.assertEqual(
                    result["clip_names"], [resource_name(key, {}) for key in expected["clip_names"]]
                )
                self.assertEqual(result["vae_name"], resource_name(expected["vae_name"], {}))
                patcher.model_state_dict.assert_not_called()

        # Request-local Automatic resets the family recipe, even with a legacy
        # global choice. Missing keys keep legacy choices for migration.
        scope["settings"].default_settings.update(
            clip_l="legacy-l.safetensors", clip_g="legacy-g.safetensors", vae_sdxl="legacy-vae"
        )
        model = type("SDXL", (), {})()
        model.model_config = SimpleNamespace(unet_config={})
        choices = {"clip_l": "request-l.safetensors", "vae_sdxl": None}
        result = scope["get_clip_and_vae"](owner, SimpleNamespace(model=model), choices)
        self.assertEqual(result["clip_names"], ["request-l.safetensors", "legacy-g.safetensors"])
        self.assertEqual(result["vae_name"], resource_name("vae_sdxl", {}))
        self.assertEqual(result["clip_keys"], ["clip_l", "clip_g"])
        self.assertEqual(result["vae_key"], "vae_sdxl")
        self.assertEqual(scope["settings"].default_settings["vae_sdxl"], "legacy-vae")
        self.assertEqual(choices, {"clip_l": "request-l.safetensors", "vae_sdxl": None})

    def test_silent_sampling_skips_preview_decode_but_keeps_progress_and_callbacks(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        start = next(
            i
            for i, n in enumerate(process.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "previewer" for t in n.targets)
        )
        lcm = next(
            i
            for i, n in enumerate(process.body)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "sampler_name == 'lcm'"
        )
        # Sampling runs inside the model-patch scope that follows the LCM check.
        stop = next(
            i for i in range(lcm, len(process.body)) if isinstance(process.body[i], ast.With)
        )
        # The patch scope rebinds sampling_model, so it is a parameter of the slice.
        method = ast.parse(
            "def sample(self, gen_data, callback=None, sampling_model=sampling_model): pass"
        ).body[0]
        method.body = process.body[start : stop + 1] + ast.parse("return samples").body
        progress, decoder, callback = Mock(), Mock(), Mock()
        load = Mock(return_value=decoder)

        def sample(**kwargs):
            kwargs["callback"](1, "latent", None, 2)
            return "result"

        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            get_previewer=load,
            device="gpu",
            worker=Mock(),
            cfg=4,
            sampler_name="euler",
            guidance={},
            architecture="SDXL",
            checkpoint="checkpoint.safetensors",
            scheduler="simple",
            positive_cond=[],
            negative_conditioning=[],
            latent_image="latent",
            denoise=1,
            force_full_denoise=False,
            noise_mask=None,
            seed=1,
            sampling_model=object(),
            noise=object(),
            fixed_sigmas=None,
            # No refiner hand-off or native sampling plan in this ordinary request.
            native_plan=None,
            positive_prompt="",
            negative_prompt="",
            path_manager=SimpleNamespace(),
            model_patches=SimpleNamespace(
                apply_scope=lambda owner, model, *args: nullcontext(model)
            ),
            comfy=SimpleNamespace(
                sample=SimpleNamespace(sample=sample),
                utils=SimpleNamespace(
                    ProgressBar=lambda steps: progress, PROGRESS_BAR_ENABLED=True
                ),
            ),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                "sampling",
                "exec",
            ),
            scope,
        )
        owner = SimpleNamespace(
            xl_base_patched=SimpleNamespace(
                unet=SimpleNamespace(model=SimpleNamespace(latent_format="format")),
                vae=SimpleNamespace(),
            )
        )
        with patch("ruinedfooocus.features.generation.image_sampling.sample_lcm", sample):
            for sampler_name in ("euler", "lcm"):
                scope["sampler_name"] = sampler_name
                for silent, consumer in ((True, callback), (False, None), (False, callback)):
                    load.reset_mock()
                    decoder.reset_mock()
                    callback.reset_mock()
                    progress.reset_mock()
                    self.assertEqual(
                        scope["sample"](
                            owner,
                            dict(steps=2, task_id="test", silent=silent, image_guidance={}),
                            consumer,
                        ),
                        "result",
                    )
                    self.assertEqual(load.call_count, int(not silent and consumer is not None))
                    self.assertEqual(decoder.preview.call_count, load.call_count)
                    progress.update_absolute.assert_called_once_with(2, 2, None)
                    self.assertEqual(callback.call_count, int(consumer is not None))

    def preview(self, method="Auto", files=()):
        methods = SimpleNamespace(**{n: n for n in ("Auto", "NoPreviews", "TAESD", "Latent2RGB")})
        args = SimpleNamespace(preview_method=method)
        paths = SimpleNamespace(
            add_model_folder_path=Mock(), get_filename_list=Mock(return_value=files)
        )
        native = SimpleNamespace(get_previewer=Mock(), TAEHVPreviewerImpl=type("Temporal", (), {}))
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            args=args,
            LatentPreviewMethod=methods,
            folder_paths=paths,
            latent_preview=native,
            logging=Mock(),
        )
        run = function(
            "ruinedfooocus/features/generation/pipeline_utils.py", "get_previewer", scope
        )
        return run, args, paths, native

    def test_disabled_preview_never_loads_or_registers_a_decoder(self):
        run, _, paths, native = self.preview("NoPreviews")
        self.assertIsNone(run("cpu", SimpleNamespace()))
        native.get_previewer.assert_not_called()
        paths.add_model_folder_path.assert_not_called()

    def test_auto_keeps_native_policy_even_when_a_decoder_is_installed(self):
        for files in ((), ["taesdxl_decoder.pth"]):
            run, args, paths, native = self.preview(files=files)
            selected = SimpleNamespace(decode_latent_to_preview=Mock(return_value="native image"))

            def load(device, latent_format):
                self.assertEqual(args.preview_method, "Auto")
                return selected

            native.get_previewer.side_effect = load
            shared = SimpleNamespace(
                path_manager=SimpleNamespace(model_paths={"vae_approx_path": "custom-previews"})
            )
            with (
                patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
                # Patch the package attribute too: 'import ruinedfooocus.app.state as x' reads it.
                patch.object(ruinedfooocus.app, "state", shared, create=True),
            ):
                previewer = run("cpu", SimpleNamespace(taesd_decoder_name="taesdxl"))
            self.assertIs(previewer, selected)
            self.assertEqual(args.preview_method, "Auto")
            self.assertEqual(previewer.preview("latent", 1, 2), "native image")
            selected.decode_latent_to_preview.assert_called_once_with("latent")
            paths.add_model_folder_path.assert_called_once_with(
                "vae_approx", "custom-previews", is_default=True
            )
            paths.get_filename_list.assert_not_called()

    def test_corrupt_decoder_falls_back_natively_and_restores_user_mode_on_failure(self):
        for fallback_fails in (False, True):
            run, args, _, native = self.preview("TAESD")

            def load(device, latent_format):
                if args.preview_method == "TAESD" or fallback_fails:
                    raise ValueError("bad decoder")
                self.assertEqual(args.preview_method, "Latent2RGB")
                return None

            native.get_previewer.side_effect = load
            shared = SimpleNamespace(
                path_manager=SimpleNamespace(model_paths={"vae_approx_path": "previews"})
            )
            with (
                patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
                # Patch the package attribute too: 'import ruinedfooocus.app.state as x' reads it.
                patch.object(ruinedfooocus.app, "state", shared, create=True),
            ):
                if fallback_fails:
                    with self.assertRaisesRegex(ValueError, "bad decoder"):
                        run("cpu", SimpleNamespace(taesd_decoder_name="taesdxl"))
                else:
                    self.assertIsNone(run("cpu", SimpleNamespace(taesd_decoder_name="taesdxl")))
            self.assertEqual(args.preview_method, "TAESD")
            self.assertEqual(native.get_previewer.call_count, 2)

    def test_upscale_errors_reach_worker_and_success_preserves_rgba(self):
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            torch=torch,
            np=np,
            worker=Mock(),
            controlnet=SimpleNamespace(get_settings=lambda data: {"upscaler": "installed-model"}),
            InterruptibleUpscaler=lambda model, data: model,
        )
        run = function(
            "ruinedfooocus/features/image_processing/upscale_pipeline.py", "process", scope
        )
        owner = SimpleNamespace(load_upscaler_model=Mock(return_value="native model"))
        source = Image.new("RGBA", (3, 2), (255, 128, 0, 64))
        data = dict(input_image=source, task_id=1)
        execute = Mock()
        native = SimpleNamespace(ImageUpscaleWithModel=SimpleNamespace(execute=execute))
        with patch.dict("sys.modules", {"comfy_extras.nodes_upscale_model": native}):
            for error in (ValueError("invalid weights"), torch.OutOfMemoryError("GPU full")):
                execute.side_effect = error
                with self.assertRaises(type(error)):
                    run(owner, data)

            def upscale(model, image):
                self.assertEqual(model, "native model")
                self.assertEqual(tuple(image.shape), (1, 2, 3, 4))
                return (image.repeat_interleave(2, 1).repeat_interleave(2, 2),)

            execute.side_effect = upscale
            result = run(owner, data)[0]
        self.assertEqual(result.shape, (4, 6, 4))
        np.testing.assert_array_equal(result[0, 0], [255, 128, 0, 64])


class ReferenceCanvasTests(unittest.TestCase):
    def test_model_grid_padding_preserves_every_source_pixel(self):
        from ruinedfooocus.features.image_editing.image_geometry import (
            prepare_reference,
            restore_reference_frame,
        )

        source = np.arange(1017 * 1023 * 3, dtype=np.uint8).reshape(1017, 1023, 3)
        pixels, plan = prepare_reference(Image.fromarray(source), (1152, 1152))
        self.assertEqual(tuple(pixels.shape), (1, 1024, 1024, 3))
        restored = restore_reference_frame(pixels, plan)
        self.assertEqual(tuple(restored.shape), (1, 1017, 1023, 3))
        torch.testing.assert_close(restored, torch.from_numpy(source).float().unsqueeze(0) / 255)


if __name__ == "__main__":
    unittest.main()
