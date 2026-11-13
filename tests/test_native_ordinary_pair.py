"""Pinned Kandinsky image/Joy contracts without weight loads or device probes."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.backends.comfy.vae_decode import decode_image_vae
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.component_candidates import compatibility
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS
from ruinedfooocus.features.models import component_profiles
from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.generation import image_preparation
from ruinedfooocus.core.preparation import PreparationService
from test_native_model_coverage import NATIVE, PIPELINE, native_method, route_info
from test_model_cache import method
from test_image_preparation import Manager

CONTRACTS = {
    "Kandinsky5Image": ("KANDINSKY5_IMAGE", ["clip_qwen25", "clip_l", "vae_flux"]),
    "JoyImage": ("JOYIMAGE", ["clip_joyimage_8b", "vae_wan"]),
}


def native_function(filename, name, scope):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return scope[name]


class NativeOrdinaryPairTests(unittest.TestCase):
    def test_runtime_inventory_and_captured_component_reads_agree_without_network(self):
        get_info = method(
            PIPELINE, "get_clip_and_vae", dict(settings=SimpleNamespace(default_settings={}))
        )
        for family, (clip_type, keys) in CONTRACTS.items():
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manager = Manager(root)
                model = root / "arbitrary.safetensors"
                header = json.dumps(dict(weight=dict(shape=[1]))).encode()
                model.write_bytes(len(header).to_bytes(8, "little") + header)
                models = SimpleNamespace(
                    get_file=lambda *args: model,
                    get_model_base=lambda value: family,
                    get_models_by_path=lambda *args, **kwargs: dict(baseModel=family),
                )
                paths = {}
                for key in keys:
                    category = "clip" if key.startswith("clip_") else "vae"
                    path = root / category / RESOURCE_DEFAULTS[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"owned installed fixture")
                    paths[key] = path.resolve()
                state = inventory(model, family, {}, manager)
                self.assertEqual([row["key"] for row in state["rows"]], keys)
                self.assertTrue(all(row["status"] == "Installed" for row in state["rows"]))
                runtime = type(family, (), {"model_config": SimpleNamespace(unet_config={})})()
                info = get_info(
                    SimpleNamespace(known_model_info=route_info()), SimpleNamespace(model=runtime)
                )
                self.assertEqual((info["clip_type"], info["latent"]), (clip_type, "SD3"))
                self.assertEqual(info["clip_keys"], keys[:-1])
                request = component_profiles.snapshot_request(
                    dict(base_model_name=model.name), models, {}, manager
                )
                self.assertEqual(request[component_profiles.REQUEST_KEY], dict.fromkeys(keys))
                request = image_preparation.capture_image_assets(
                    request, manager, {}, models=models, values={}, offline=True
                )
                self.assertFalse(image_preparation.needs_image_preparation(request))
                transfer = Mock(side_effect=AssertionError("Installed request cannot transfer"))
                service = PreparationService(transport=transfer, staging_root=root / "staging")
                try:
                    ready = image_preparation.prepare_image_assets(
                        request, check_cancelled=lambda: None, service=service
                    )
                    manager.root = root / "changed-after-submission"
                    for key in keys:
                        category = "clip" if key.startswith("clip_") else "vae"
                        with image_preparation.prepared_asset(
                            ready, category, key, manager, category, service=service
                        ) as actual:
                            self.assertEqual(actual, paths[key])
                    transfer.assert_not_called()
                finally:
                    service._executor.shutdown(wait=True)

    def test_joy_encoder_has_its_own_exact_official_artifact_identity(self):
        name = "qwen3vl_8b_joyimage_edit_int8_convrot.safetensors"
        catalogue = json.loads(
            resource_path("model_catalogue", "native_image_components.json").read_text()
        )
        download = catalogue["clip/" + name]
        annotation = load_registry().annotations_for("clip/" + name, download)
        self.assertEqual(RESOURCE_DEFAULTS["clip_joyimage_8b"], name)
        self.assertNotEqual(RESOURCE_DEFAULTS["clip_qwen3vl_8b"], name)
        self.assertIn("/resolve/8010e7b4c2ac3f762d33775ac5d6a011eea4166c/", download["url"])
        self.assertEqual(
            annotation["artifact"]["sha256"],
            "683de3ab15337afb9dcf19caaabb683a5ce53eb311eb302ff8eeaf4f553ad1a8",
        )
        self.assertEqual(annotation["artifact"]["bytes"], 10064089100)
        self.assertFalse(annotation["artifact"]["full_file_hash_recomputed"])
        self.assertEqual(annotation["artifact"]["availability"], "publisher_metadata_only")
        self.assertFalse(annotation["chat_weight"])
        self.assertEqual(
            compatibility("clip_joyimage_8b", dict(text_type="QWEN3VL_8B")), "compatible"
        )
        self.assertEqual(
            compatibility("clip_joyimage_8b", dict(text_type="QWEN3VL_4B")), "incompatible"
        )

    def test_native_kandinsky_image_dual_clip_preserves_required_pooled_output(self):
        class QwenImageTEModel:
            def encode_token_weights(self, pairs, template_end):
                return "native-qwen-context", None, dict(attention_mask="native-attention")

        scope = dict(QwenImageTEModel=QwenImageTEModel)
        encoder = native_method(
            "comfy/text_encoders/kandinsky5.py",
            "Kandinsky5TEModel",
            "encode_token_weights",
            scope,
            bases=("QwenImageTEModel",),
        )()
        encoder.clip_l = SimpleNamespace(
            encode_token_weights=Mock(return_value=("clip-tokens", "clip-pooled"))
        )
        result = encoder.encode_token_weights(dict(l=["clip-l"], qwen25_7b=["qwen"]))
        self.assertEqual(
            result, ("native-qwen-context", "clip-pooled", dict(attention_mask="native-attention"))
        )
        encoder.clip_l.encode_token_weights.assert_called_once_with(["clip-l"])
        adm = native_method("comfy/model_base.py", "Kandinsky5", "encode_adm", {})()
        self.assertEqual(adm.encode_adm(pooled_output=result[1]), "clip-pooled")

        # The loader's dual branch must select the image template, not video/single-Qwen.
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "clip_type == CLIPType.KANDINSKY5_IMAGE"
        )
        factory = Mock(return_value="native-dual-encoder")
        scope = dict(
            clip_target=SimpleNamespace(),
            clip_data=["qwen-weights", "clip-l-weights"],
            llama_detect=lambda data: dict(dtype_llama="native-dtype"),
            comfy=SimpleNamespace(
                text_encoders=SimpleNamespace(
                    kandinsky5=SimpleNamespace(
                        te=factory, Kandinsky5TokenizerImage="native-image-tokenizer"
                    )
                )
            ),
        )
        exec(
            compile(ast.Module(body=branch.body, type_ignores=[]), "native_dual_clip", "exec"),
            scope,
        )
        self.assertEqual(scope["clip_target"].tokenizer, "native-image-tokenizer")
        factory.assert_called_once_with(dtype_llama="native-dtype")

    def test_native_joy_prefix_and_attention_mask_stay_aligned(self):
        context = torch.arange(40, dtype=torch.float32).reshape(1, 40, 1)
        mask = torch.ones(1, 40, dtype=torch.long)

        class BaseEncoder:
            def encode_token_weights(self, pairs):
                return context, "pooled", dict(attention_mask=mask)

        scope = dict(BaseEncoder=BaseEncoder, JOYIMAGE_DROP_IDX=34)
        encoder = native_method(
            "comfy/text_encoders/joyimage.py",
            "JoyImageTEModel",
            "encode_token_weights",
            scope,
            bases=("BaseEncoder",),
        )()
        encoded, pooled, extra = encoder.encode_token_weights({})
        self.assertTrue(torch.equal(encoded, context[:, 34:]))
        self.assertTrue(torch.equal(extra["attention_mask"], mask[:, 34:]))
        self.assertEqual(pooled, "pooled")

    def test_native_empty_latent_and_image_decode_handle_single_wan_frame(self):
        scope = dict(torch=torch)
        fix = native_function("comfy/sample.py", "fix_empty_latent_channels", scope)
        decode = native_method("nodes.py", "VAEDecode", "decode", {})()
        memory = SimpleNamespace(
            get_total_memory=Mock(side_effect=AssertionError("CPU must not probe"))
        )
        for family, dimensions in (("Kandinsky5Image", 2), ("JoyImage", 3)):
            with self.subTest(family=family):
                latent_format = SimpleNamespace(
                    latent_channels=16,
                    latent_dimensions=dimensions,
                    spacial_downscale_ratio=8,
                    fix_empty_latent=lambda latent: latent,
                )
                model = SimpleNamespace(get_model_object=lambda name: latent_format)
                latent = fix(model, torch.zeros(1, 16, 6, 8), 8)
                expected = (1, 16, 1, 6, 8) if dimensions == 3 else (1, 16, 6, 8)
                self.assertEqual(tuple(latent.shape), expected)
                pixels = torch.ones((1, 1, 48, 64, 3) if dimensions == 3 else (1, 48, 64, 3))
                vae = SimpleNamespace(
                    device=SimpleNamespace(type="cpu"), decode=Mock(return_value=pixels)
                )
                owner = SimpleNamespace(vae=vae, unet=model)
                with patch.dict(
                    "sys.modules",
                    {
                        "comfy": SimpleNamespace(model_management=memory),
                        "nodes": SimpleNamespace(
                            VAEDecode=lambda: decode,
                            VAEDecodeTiled=Mock(side_effect=AssertionError("Untiled CPU decode")),
                        ),
                    },
                ):
                    output = decode_image_vae(owner, dict(samples=latent))
                vae.decode.assert_called_once_with(latent)
                self.assertEqual(tuple(output.shape), (1, 48, 64, 3))
                memory.get_total_memory.assert_not_called()

    def test_production_sampler_keeps_cfg_and_both_native_conditionings(self):
        tree = ast.parse((NATIVE / "comfy/model_base.py").read_text(encoding="utf-8"))
        scope = {}
        # Keep real native inheritance; weight constructors are deliberately absent.
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in (
                "BaseModel",
                "Flux",
                "Kandinsky5",
                "Kandinsky5Image",
                "JoyImage",
            ):
                if node.name == "BaseModel":
                    node.bases = []
                node.body = [ast.copy_location(ast.Pass(), node)]
                node.decorator_list = []
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), "native_inheritance", "exec"),
                    scope,
                )
        production = ast.parse(
            (Path(__file__).resolve().parents[1] / PIPELINE).read_text(encoding="utf-8")
        )
        branch = next(
            node
            for node in ast.walk(production)
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
            for node in ast.walk(production)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "comfy.sample.sample"
        )
        for family in CONTRACTS:
            positive, negative = (
                [["positive", dict(pooled_output="native-pooled")]],
                [["negative", dict(attention_mask="native-mask")]],
            )
            sampler = Mock(return_value="sampled")
            scope.update(
                architecture=family,
                self=SimpleNamespace(
                    xl_base=SimpleNamespace(unet=SimpleNamespace(model=scope[family]()))
                ),
                cfg=3.5,
                positive_cond=positive,
                negative_conditioning=negative,
                comfy=SimpleNamespace(
                    sample=SimpleNamespace(sample=sampler),
                    utils=SimpleNamespace(PROGRESS_BAR_ENABLED=True),
                ),
                sampling_model="native",
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
                compile(ast.Module(body=[branch, sample], type_ignores=[]), PIPELINE, "exec"), scope
            )
            self.assertEqual(sampler.call_args.kwargs["cfg"], 3.5)
            self.assertIs(sampler.call_args.kwargs["positive"], positive)
            self.assertIs(sampler.call_args.kwargs["negative"], negative)


if __name__ == "__main__":
    unittest.main()
