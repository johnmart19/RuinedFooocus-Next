from ruinedfooocus.core.paths import asset_path, resource_path

"""Exercise production loader caches with local files and mocked native loaders."""
import ast
import builtins
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import json
import os
import struct
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ruinedfooocus.app
import ruinedfooocus.features.models.model_identity as model_identity
from ruinedfooocus.backends.comfy.component_selection import normalize_component_overrides
from ruinedfooocus.core.config_io import save_json

ROOT = Path(__file__).resolve().parents[1]

# Keep optional codec identities stable while process fixtures isolate sys.modules.
if (ROOT / "ruinedfooocus/features/models/approximate_codecs.py").is_file():
    importlib.import_module("ruinedfooocus.features.models.approximate_codecs")


def method(source, name, scope):
    tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(
        vars(model_identity), asset_path=asset_path, resource_path=resource_path, **scope
    )
    if "lora_hooks" not in namespace and any(
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "lora_hooks"
        for decorator in node.decorator_list
    ):
        from importlib import import_module

        namespace["lora_hooks"] = import_module("ruinedfooocus.features.image_editing.lora_hooks")
    from ruinedfooocus.features.generation.image_preparation import (
        component_read_leases,
        component_file,
    )

    namespace.update(component_read_leases=component_read_leases, component_file=component_file)
    if any(
        isinstance(decorator, ast.Name) and decorator.id == "stage_read_leases"
        for decorator in node.decorator_list
    ):
        namespace["stage_read_leases"] = importlib.import_module(
            "ruinedfooocus.features.generation.cascade_guidance"
        ).stage_read_leases
    exec(compile(ast.Module(body=[node], type_ignores=[]), source, "exec"), namespace)
    return namespace[name]


class Clip:
    pass


class Vae:
    def __init__(self):
        self.first_stage_model = object()
        self.patcher = SimpleNamespace()
        self.throw_exception_if_invalid = Mock()


class ModelCacheTests(unittest.TestCase):
    def test_bundled_sdxl_uses_explicit_pair_and_automatic_restores_bundled_vae(self):
        from ruinedfooocus.features.models import approximate_codecs as codecs
        from test_approximate_codecs import write_component

        checkpoint = self.file("checkpoint.safetensors")
        paths = SimpleNamespace(paths={"path_vae_approx": [str(self.root)]}, get_abspath=Path)
        for role in ("encoder", "decoder"):
            write_component(
                self.root / f"taesdxl_{role}.safetensors", codecs.expected_shapes("taesdxl", role)
            )
        choices = {"vae_sdxl": "approx:taesdxl"}
        data = codecs.capture_request({"image_component_overrides": choices}, paths)
        pair = data[codecs.KEY]
        _, owner, native, backend, _ = self.base_loader(checkpoint)
        bundled_vae = native.return_value[2]
        approximate_vae = Vae()
        with patch(
            "ruinedfooocus.backends.comfy.approximate_codecs.load_paired_vae",
            return_value=approximate_vae,
        ) as load_pair:
            owner.load_base_model(checkpoint.name, component_overrides=choices, prepared_data=data)
            self.assertIs(owner.xl_base.vae, approximate_vae)
            self.assertIsNot(owner.xl_base.vae, bundled_vae)
            self.assertFalse(native.call_args.kwargs["output_vae"])
            self.assertEqual(
                owner.component_file_signatures, [list(item.signature) for item in pair.files]
            )
            load_pair.assert_called_once_with(pair)
            owner.load_base_model(checkpoint.name, component_overrides=choices, prepared_data=data)
            self.assertEqual(load_pair.call_count, 1)
            stamp = pair.encoder.path.stat()
            os.utime(pair.encoder.path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10000000))
            with self.assertRaisesRegex(ValueError, "changed"):
                owner.load_base_model(
                    checkpoint.name, component_overrides=choices, prepared_data=data
                )
            fresh = codecs.capture_request({"image_component_overrides": choices}, paths)
            owner.load_base_model(checkpoint.name, component_overrides=choices, prepared_data=fresh)
            self.assertEqual(load_pair.call_count, 2)
            owner.load_base_model(checkpoint.name)
            self.assertIs(owner.xl_base.vae, bundled_vae)
            self.assertTrue(native.call_args.kwargs["output_vae"])
            self.assertEqual(load_pair.call_count, 2)
        backend.sd.VAE.assert_not_called()

    def test_component_choices_reject_malformed_or_unknown_slots(self):
        for values in ([], "clip_l", {"unknown": "local"}, {"clip_l": ["local"]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_component_overrides(values)
        self.assertEqual(normalize_component_overrides({"clip_l": " "}), {"clip_l": None})

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def file(self, name):
        path = self.root / name
        if path.suffix.casefold() in (".safetensors", ".sft"):
            header = json.dumps({"__metadata__": {"fixture": "inert cache identity"}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header)
        else:
            path.write_bytes(b"local test weights")
        return path

    def replace(self, path):
        before = path.stat()
        path.write_bytes(path.read_bytes() + b"replacement weights")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10000000))

    def base_loader(self, checkpoint, bare=False, defaults=None):
        clip, vae = Clip(), Vae()
        unet = SimpleNamespace(model=type("SDXL", (), {})())
        backend = SimpleNamespace(
            model_base=SimpleNamespace(HunyuanDiT=type("HunyuanDiT", (), {})),
            model_management=Mock(),
            sd=SimpleNamespace(
                load_clip=Mock(return_value=clip),
                VAE=Mock(return_value=vae),
                load_vae_patcher=Mock(),
            ),
            utils=SimpleNamespace(load_torch_file=Mock(return_value=({}, {}))),
        )
        backend.model_management.get_torch_device.return_value = SimpleNamespace(type="cpu")
        files = Mock(get_model_path=Mock(return_value=checkpoint))
        loader = Mock(return_value=(unet, clip, vae, None))
        diffusion = Mock(return_value=unet)
        owner = SimpleNamespace(
            xl_base_hash=None,
            xl_base=None,
            StableDiffusionModel=lambda **values: SimpleNamespace(**values),
            known_models={"SDXL"},
            get_clip_and_vae=Mock(
                return_value=dict(
                    options={},
                    clip_names=["encoder.safetensors"],
                    vae_name="vae.safetensors",
                    clip_type="native",
                    clip_keys=["clip_l", "clip_g"],
                    vae_key="vae_sdxl",
                )
            ),
        )
        import torch

        run = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "load_base_model",
            dict(
                settings=SimpleNamespace(default_settings={} if defaults is None else defaults),
                shared=SimpleNamespace(models=files),
                Path=Path,
                os=os,
                gc=Mock(),
                comfy=backend,
                torch=torch,
                load_checkpoint_guess_config=loader,
                register_embedding_paths=lambda paths: paths,
                load_diffusion_model=diffusion,
                CLIP=Clip,
                VAE=Vae,
                traceback=Mock(),
                path_manager=SimpleNamespace(
                    DOWNLOADABLE_FILES={},
                    get_model_folders=lambda category: [self.root],
                    model_paths={
                        "clip_path": str(self.root),
                        "vae_path": str(self.root),
                        "embedding_path": [str(self.root / "embeddings")],
                    },
                    get_folder_file_path=lambda folder, name, **kwargs: self.root / name,
                ),
            ),
        )
        owner.load_base_model = MethodType(run, owner)
        attention = SimpleNamespace(configure_qwen_attention=Mock())
        context = patch.dict(
            "sys.modules", {"ruinedfooocus.backends.comfy.model_attention": attention}
        )
        context.start()
        self.addCleanup(context.stop)

        def load():
            owner.load_base_model("checkpoint.safetensors", unet_only=bare)

        return load, owner, diffusion if bare else loader, backend, files

    def test_checkpoint_cache_reloads_replaced_weights_but_reuses_unchanged_weights(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, owner, native, _, _ = self.base_loader(checkpoint)
        load()
        load()
        self.assertEqual(native.call_count, 1)
        self.replace(checkpoint)
        load()
        self.assertEqual(native.call_count, 2)

    def test_named_sensenova_bundle_uses_native_diffusion_route_and_all_shard_cache_identity(self):
        import importlib
        from test_sensenova_bundle import write_bundle

        sense = importlib.import_module("ruinedfooocus.features.models.sensenova_bundle")

        index = write_bundle(self.root / "SenseNova")
        bundle = sense.capture_bundle(index)
        _, owner, native, _, _ = self.base_loader(index, bare=True)
        model = type("SenseNovaU15", (), {})()
        model.model_config = SimpleNamespace()
        native.return_value = SimpleNamespace(model=model)
        owner.known_models = {"SenseNovaU15"}
        owner.get_clip_and_vae.return_value = dict(
            options={},
            clip_names=[],
            clip_keys=[],
            vae_name="pixel_space",
            vae_key="pixel_space",
            clip_type="native",
        )
        prepared = {sense.KEY: bundle}
        with patch(
            "ruinedfooocus.backends.comfy.sensenova_clip.load_sensenova_clip", return_value=Clip()
        ) as tokenizer:
            owner.load_base_model(index.name, prepared_data=prepared)
            self.assertIsNotNone(owner.xl_base, owner.load_error)
            native.assert_called_once_with(str(index), sensenova_bundle=bundle)
            self.assertIs(tokenizer.call_args.args[1], bundle)
            owner.load_base_model(index.name, prepared_data=prepared)
            self.assertEqual(native.call_count, 1)
            self.replace(bundle.shards[-1].path)
            with self.assertRaisesRegex(ValueError, "changed"):
                owner.load_base_model(index.name, prepared_data=prepared)
            self.assertEqual(native.call_count, 1)
        with self.assertRaisesRegex(ValueError, "Capture the SenseNova"):
            owner.load_base_model(index.name)

    def test_known_l2p_convolutional_decoder_is_refused_before_native_loader_or_cache(self):
        from test_native_pixel_sources import write_l2p_fixture

        checkpoint = self.root / "renamed-l2p.safetensors"
        write_l2p_fixture(checkpoint)
        load, owner, native, backend, _ = self.base_loader(checkpoint)
        owner.xl_base = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "convolutional pixel decoder"):
            load()
        native.assert_not_called()
        backend.model_management.cleanup_models.assert_not_called()

    def test_checkpoint_and_separate_clip_loaders_receive_embedding_roots(self):
        checkpoint = self.file("checkpoint.safetensors")
        self.file("encoder.safetensors")
        self.file("vae.safetensors")
        for bare in (False, True):
            with self.subTest(bare=bare):
                load, _, native, backend, _ = self.base_loader(checkpoint, bare=bare)
                load()
                call = backend.sd.load_clip.call_args if bare else native.call_args
                self.assertEqual(
                    call.kwargs["embedding_directory"], [str(self.root / "embeddings")]
                )

    def test_diffusion_only_file_loads_external_encoder_and_vae_after_native_detection(self):
        checkpoint = self.file("diffusion.safetensors")
        self.file("encoder.safetensors")
        self.file("vae.safetensors")
        load, owner, native, backend, _ = self.base_loader(checkpoint)
        unet = native.return_value[0]
        native.return_value = unet, None, None, None
        load()
        native.assert_called_once()
        backend.sd.load_clip.assert_called_once()
        backend.sd.VAE.assert_called_once()
        self.assertIs(owner.xl_base.unet, unet)
        self.assertIsNone(owner.load_error)

    def test_sft_checkpoint_and_encoder_keep_native_file_loading_and_cache(self):
        checkpoint = self.file("checkpoint.sft")
        self.file("encoder.SFT")
        self.file("vae.sft")
        _, owner, native, backend, _ = self.base_loader(checkpoint)
        unet = native.return_value[0]
        native.return_value = unet, None, None, None
        owner.get_clip_and_vae.return_value.update(clip_names=["encoder.SFT"], vae_name="vae.sft")
        owner.load_base_model(checkpoint.name)
        owner.load_base_model(checkpoint.name)
        native.assert_called_once()
        self.assertEqual(native.call_args.args[0], str(checkpoint))
        backend.sd.load_clip.assert_called_once()
        self.assertEqual(
            backend.sd.load_clip.call_args.kwargs["ckpt_paths"], [str(self.root / "encoder.SFT")]
        )
        self.assertEqual(backend.utils.load_torch_file.call_args.kwargs, {"return_metadata": True})
        self.assertIsNone(owner.load_error)

    def test_split_image_loader_refuses_mixed_case_gguf_vae_before_weight_readers(self):
        checkpoint = self.file("checkpoint.safetensors")
        for filename in ("image-codec.gguf", "other-codec.GgUf"):
            with self.subTest(filename=filename):
                load, owner, native, backend, _ = self.base_loader(checkpoint, bare=True)
                owner.get_clip_and_vae.return_value["vae_name"] = filename
                load()
                native.assert_called_once()
                self.assertIn("GGUF files are unsupported for still-image VAEs", owner.load_error)
                self.assertIsNone(owner.xl_base)
                backend.sd.load_clip.assert_not_called()
                backend.utils.load_torch_file.assert_not_called()
                backend.sd.VAE.assert_not_called()

    def test_decode_preferences_do_not_reload_weights_but_component_overrides_do(self):
        checkpoint = self.file("checkpoint.safetensors")
        defaults = {"clip_l": "encoder.safetensors", "vae_tile_size": None}
        load, _, native, _, _ = self.base_loader(checkpoint, defaults=defaults)
        load()
        defaults.update(vae_tiled=True, vae_tile_size=256, vae_overlap=0, clip_skip=2)
        load()
        self.assertEqual(native.call_count, 1)
        defaults["clip_l"] = "other-encoder.safetensors"
        load()
        self.assertEqual(native.call_count, 2)

    def test_explicit_bundled_override_is_rejected_without_publishing_a_model(self):
        checkpoint = self.file("checkpoint.safetensors")
        _, owner, native, _, _ = self.base_loader(checkpoint)
        for choice in ({"clip_l": "other.safetensors"}, {"vae_sdxl": "other.safetensors"}):
            with self.subTest(choice=choice):
                with self.assertRaisesRegex(ValueError, "built into this checkpoint"):
                    owner.load_base_model(checkpoint.name, component_overrides=choice)
                self.assertIsNone(owner.xl_base_hash)
                self.assertIsNone(owner.xl_base)
        self.assertEqual(native.call_count, 2)

    def test_explicit_choices_are_copied_cached_and_preserved_by_native_fallback(self):
        checkpoint = self.file("checkpoint.safetensors")
        encoder, _ = self.file("encoder.safetensors"), self.file("vae.safetensors")
        _, owner, native, backend, _ = self.base_loader(checkpoint)
        unet = native.return_value[0]
        native.return_value = unet, None, None, None
        choices = {"clip_l": "encoder.safetensors", "vae_sdxl": None}
        owner.load_base_model(checkpoint.name, component_overrides=choices)
        self.assertEqual(owner.component_overrides, choices)
        self.assertIsNot(owner.component_overrides, choices)
        owner.get_clip_and_vae.assert_called_with(unet, choices, None)
        owner.load_base_model(checkpoint.name, component_overrides=dict(choices))
        self.assertEqual(native.call_count, 1)
        self.replace(encoder)
        owner.load_base_model(checkpoint.name, component_overrides=dict(choices))
        self.assertEqual(native.call_count, 2)
        self.assertEqual(backend.sd.load_clip.call_count, 2)
        choices["clip_l"] = None
        owner.load_base_model(checkpoint.name, component_overrides=choices)
        self.assertEqual(native.call_count, 3)

    def test_invalid_choice_cannot_be_reused_from_an_automatic_cache_entry(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, owner, native, _, _ = self.base_loader(checkpoint, defaults={"clip_l": "same"})
        load()
        with self.assertRaisesRegex(ValueError, "built into this checkpoint"):
            owner.load_base_model(checkpoint.name, component_overrides={"clip_l": "same"})
        self.assertEqual(native.call_count, 2)
        self.assertIsNone(owner.xl_base)

    def test_partial_native_bundle_preserves_clip_and_resolves_only_the_missing_vae(self):
        checkpoint = self.file("checkpoint.safetensors")
        self.file("vae.safetensors")
        _, owner, native, backend, _ = self.base_loader(checkpoint)
        unet, clip, _, _ = native.return_value
        native.return_value = unet, clip, None, None
        choices = {"clip_l": None, "vae_sdxl": "vae.safetensors"}
        owner.load_base_model(checkpoint.name, component_overrides=choices)
        self.assertIs(owner.xl_base.clip, clip)
        backend.sd.load_clip.assert_not_called()
        backend.sd.VAE.assert_called_once()

    def test_missing_or_inapplicable_external_choice_does_not_publish_a_model(self):
        checkpoint = self.file("checkpoint.safetensors")
        _, owner, _, _, _ = self.base_loader(checkpoint, bare=True)
        for choice, message in (
            ({"vae_sdxl": "missing.safetensors"}, "not installed"),
            ({"clip_t5": "t5.safetensors"}, "not a slot"),
        ):
            owner.load_base_model(checkpoint.name, unet_only=True, component_overrides=choice)
            self.assertIsNone(owner.xl_base)
            self.assertIsNone(owner.xl_base_hash)
            self.assertIn(message, owner.load_error)

    def test_removed_checkpoint_cannot_reuse_the_old_model(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, owner, native, _, files = self.base_loader(checkpoint)
        load()
        checkpoint.unlink()
        files.get_model_path.return_value = None
        load()
        self.assertIsNone(owner.xl_base)
        self.assertIsNone(owner.xl_base_hash)

    def test_a_different_resolved_checkpoint_path_invalidates_a_matching_name(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, _, native, _, files = self.base_loader(checkpoint)
        load()
        other = self.file("other.safetensors")
        files.get_model_path.return_value = other
        load()
        self.assertEqual(native.call_count, 2)
        self.assertEqual(native.call_args.args[0], str(other))

    def hash_reader(self):
        owner = SimpleNamespace(cache_paths={"checkpoints": self.root / "cache/checkpoints"})
        run = method(
            "ruinedfooocus/features/models/model_handler.py",
            "model_sha256",
            dict(
                hashlib=hashlib,
                json=json,
                save_json=save_json,
                t=lambda text, mapping: text.format(**mapping),
            ),
        )
        return lambda path: run(owner, path)

    def test_model_hash_cache_reuses_stable_files_and_rehashes_replacements(self):
        path = self.file("checkpoint.safetensors")
        digest = self.hash_reader()
        expected = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        self.assertEqual(digest(path), expected)
        with patch(
            "builtins.open", side_effect=AssertionError("cached weights must not be reread")
        ):
            self.assertEqual(digest(path), expected)
        self.replace(path)
        self.assertEqual(digest(path), hashlib.sha256(path.read_bytes()).hexdigest().upper())

    def test_a_file_changing_during_hashing_is_not_published_as_a_valid_hash(self):
        path = self.file("checkpoint.safetensors")
        original_open = builtins.open
        changed = False

        @contextmanager
        def changing_reader(name, mode):
            nonlocal changed
            with original_open(name, mode) as source:

                def read(size):
                    nonlocal changed
                    block = source.read(size)
                    if block and not changed:
                        self.replace(path)
                        changed = True
                    return block

                yield SimpleNamespace(read=read)

        with patch("builtins.open", changing_reader):
            self.assertIsNone(self.hash_reader()(path))
        self.assertTrue(changed)
        self.assertEqual(list((self.root / "cache/hashes").glob("*.json")), [])

    def test_replacing_an_external_encoder_or_vae_invalidates_the_base_cache(self):
        checkpoint = self.file("checkpoint.safetensors")
        encoder, vae = self.file("encoder.safetensors"), self.file("vae.safetensors")
        load, _, native, backend, _ = self.base_loader(checkpoint, bare=True)
        load()
        load()
        self.assertEqual(native.call_count, 1)
        for file in (encoder, vae):
            self.replace(file)
            load()
        self.assertEqual(native.call_count, 3)
        self.assertEqual(backend.sd.load_clip.call_count, 3)

    def test_failed_component_configuration_does_not_leave_a_successful_cache_entry(self):
        checkpoint = self.file("checkpoint.safetensors")
        load, owner, native, backend, _ = self.base_loader(checkpoint)
        owner.get_clip_and_vae.side_effect = [ValueError("configuration failed"), {"options": {}}]
        with self.assertRaisesRegex(ValueError, "configuration failed"):
            load()
        self.assertIsNone(owner.xl_base_hash)
        load()
        self.assertEqual(native.call_count, 2)
        self.assertEqual(owner.xl_base_hash, "checkpoint.safetensors")

    def test_lora_cache_reloads_replaced_weights_and_starts_from_the_base(self):
        path = self.file("style.safetensors")
        # A compatible LoRA adds native patches; incompatible ones are rejected by load_loras.
        native = Mock(
            return_value=(
                SimpleNamespace(patches={"diffusion_model.w": [object()]}),
                SimpleNamespace(patcher=SimpleNamespace(patches={})),
            )
        )
        reader = Mock(return_value="lora weights")
        resolve = Mock(return_value=path)
        owner = SimpleNamespace(
            xl_base_patched_hash=None,
            xl_base=SimpleNamespace(
                unet="base unet", clip="base clip", vae="vae", clip_vision=None
            ),
            StableDiffusionModel=lambda **values: SimpleNamespace(**values),
        )
        run = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "load_loras",
            dict(
                shared=SimpleNamespace(models=SimpleNamespace(get_model_path=resolve)),
                comfy=SimpleNamespace(
                    utils=SimpleNamespace(load_torch_file=reader),
                    sd=SimpleNamespace(load_lora_for_models=native),
                ),
            ),
        )
        request = [{"name": path.name, "weight": 0.7}]
        run(owner, request)
        run(owner, request)
        self.assertEqual(native.call_count, 1)
        self.replace(path)
        run(owner, request)
        self.assertEqual(native.call_count, 2)
        for call in native.call_args_list:
            self.assertEqual(call.args[:2], ("base unet", "base clip"))
        run(owner, [{"name": "missing.safetensors", "weight": 0}])
        self.assertIs(owner.xl_base_patched, owner.xl_base)
        self.assertEqual(native.call_count, 2)

    def test_automatic_upscale_checks_the_shared_loader_on_each_image(self):
        from ruinedfooocus.features.generation.automatic_upscale import upscale_decoded

        model = SimpleNamespace(scale=2)
        loader = SimpleNamespace(load_upscaler_model=Mock(return_value=model))
        execute = Mock(return_value=["upscaled"])
        native = SimpleNamespace(ImageUpscaleWithModel=SimpleNamespace(execute=execute))
        with patch.dict(
            "sys.modules",
            {
                "ruinedfooocus.features.image_processing.upscale_pipeline": SimpleNamespace(
                    pipeline=lambda: loader, InterruptibleUpscaler=lambda model, data: model
                ),
                "ruinedfooocus.app.async_worker": Mock(),
                "comfy_extras.nodes_upscale_model": native,
            },
        ):
            owner = SimpleNamespace()
            data = dict(automatic_upscale="2× General (RealESRGAN)", task_id=1)
            self.assertEqual(upscale_decoded(owner, "pixels", data), "upscaled")
            self.assertEqual(upscale_decoded(owner, "pixels", data), "upscaled")
        self.assertEqual(loader.load_upscaler_model.call_count, 2)

    def test_controlnet_cache_reloads_replaced_weights(self):
        path = self.file("depth.safetensors")
        native = Mock(return_value=object())
        run = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "refresh_controlnet",
            dict(
                control_tools=SimpleNamespace(
                    model_asset=lambda name, data=None: nullcontext(path)
                ),
                comfy=SimpleNamespace(controlnet=SimpleNamespace(load_controlnet=native)),
            ),
        )
        owner = SimpleNamespace(
            xl_controlnet=None,
            xl_controlnet_hash=None,
            xl_base_patched=SimpleNamespace(unet=object()),
            base_file_signature=("primary A",),
        )
        run(owner, "depth")
        run(owner, "depth")
        self.assertEqual(native.call_count, 1)
        native.assert_called_with(str(path), model=owner.xl_base_patched.unet)
        self.replace(path)
        run(owner, "depth")
        self.assertEqual(native.call_count, 2)
        owner.base_file_signature = ("primary B",)
        run(owner, "depth")
        self.assertEqual(native.call_count, 3)
        owner.xl_base_patched_hash = [("LoRA file", 0.5)]
        run(owner, "depth")
        self.assertEqual(native.call_count, 4)
        owner.xl_base_patched_hash = [("LoRA file", 0.8)]
        run(owner, "depth")
        self.assertEqual(native.call_count, 5)
        owner.xl_base_patched.unet = object()
        run(owner, "depth")
        self.assertEqual(native.call_count, 6)

    def test_upscaler_cache_reloads_replaced_weights(self):
        path = self.file("upscaler.pth")
        native = Mock(return_value=[object()])
        run = method(
            "ruinedfooocus/features/image_processing/upscale_pipeline.py",
            "load_upscaler_model",
            dict(
                path_manager=SimpleNamespace(
                    DOWNLOADABLE_FILES={},
                    get_model_folders=lambda category: [self.root],
                    model_paths={"upscaler_path": str(self.root)},
                ),
                folder_paths=Mock(),
                Path=Path,
                os=os,
            ),
        )
        owner = SimpleNamespace()
        with patch.dict(
            "sys.modules",
            {
                "comfy_extras.nodes_upscale_model": SimpleNamespace(
                    UpscaleModelLoader=SimpleNamespace(execute=native)
                )
            },
        ):
            self.assertIs(run(owner, path.name), run(owner, path.name))
            self.assertEqual(native.call_count, 1)
            self.replace(path)
            run(owner, path.name)
        self.assertEqual(native.call_count, 2)


if __name__ == "__main__":
    unittest.main()
