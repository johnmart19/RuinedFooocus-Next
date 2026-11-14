"""Safe native serializers and actual generation/UI/API tensor handoffs on CPU."""

import ast
import base64
from concurrent.futures import CancelledError
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import gradio as gr
from PIL import Image
import numpy as np
import safetensors
import safetensors.torch
import torch

import ruinedfooocus.app
from ruinedfooocus.features.generation import native_tensor_io as native
from test_native_variant_defaults import NATIVE

ROOT = Path(__file__).resolve().parents[1]


def native_serializers():
    """Actual pinned node bodies, with only safe-file and output-path services bound."""
    forbidden_paths = NS(
        get_save_image_path=Mock(side_effect=AssertionError("global output directory")),
        get_full_path_or_raise=Mock(side_effect=AssertionError("global embeddings directory")),
        get_annotated_filepath=Mock(side_effect=AssertionError("global input directory")),
    )

    def load_file(path, **options):
        values = safetensors.torch.load_file(path, device="cpu")
        with safetensors.safe_open(path, framework="pt", device="cpu") as opened:
            metadata = opened.metadata()
        return (values, metadata) if options.get("return_metadata") else values

    scope = dict(
        torch=torch,
        safetensors=safetensors,
        json=json,
        os=os,
        comfy=NS(utils=NS(save_torch_file=safetensors.torch.save_file, load_torch_file=load_file)),
        folder_paths=forbidden_paths,
        args=NS(disable_metadata=False),
        io=NS(NodeOutput=lambda *values: values),
    )
    for filename, names in (
        ("nodes.py", ("SaveLatent", "LoadLatent")),
        ("comfy_extras/nodes_cond.py", ("ConditioningLoader", "SaveConditioning")),
    ):
        tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in names:
                continue
            node.bases = []
            methods = {"save", "load", "execute"}
            node.body = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in methods
            ]
            for item in node.body:
                item.returns = None
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                    filename,
                    "exec",
                ),
                scope,
            )
    return {
        "nodes": NS(SaveLatent=scope["SaveLatent"], LoadLatent=scope["LoadLatent"]),
        "comfy_extras.nodes_cond": NS(
            ConditioningLoader=scope["ConditioningLoader"],
            SaveConditioning=scope["SaveConditioning"],
        ),
    }, forbidden_paths


def latent_file(*, dtype=torch.float16, channels=4, marker=True, metadata=None):
    values = {"latent_tensor": torch.arange(channels * 6, dtype=dtype).view(1, channels, 2, 3) / 17}
    if marker:
        values["latent_format_version_0"] = torch.tensor([])
    return safetensors.torch.save(values, metadata=metadata)


def conditioning_file(width=768, *, dtype=torch.float16, metadata=None, pooled=None):
    tensors = {"conditioning": torch.arange(width * 2, dtype=dtype).view(1, 2, width) / 177}
    if pooled is not None:
        tensors["pooled_output"] = torch.ones(1, pooled, dtype=dtype)
    return safetensors.torch.save(tensors, metadata=metadata)


def mutate_header(raw, change):
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + size])
    change(header)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + raw[8 + size :]


class NativeTensorIOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.manager = NS(model_paths={"cache_path": self.root})
        self.models = NS(get_file=Mock(return_value=self.root / "owned-model.safetensors"))
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value={"native_class": "SD15"},
            )
        )
        self.modules, self.paths = native_serializers()
        self.enterContext(patch.dict(sys.modules, self.modules))
        self.owner = NS(
            xl_base_patched=NS(
                unet=NS(model=NS(model_config=type("SD15", (), {})())),
                vae=NS(
                    latent_channels=4,
                    latent_dim=2,
                    spacial_compression_decode=lambda: 8,
                    first_stage_model=type("AutoencoderKL", (), {})(),
                ),
            )
        )

    def capture(self, **values):
        return native.capture_request(
            dict(
                base_model_name="owned-model.safetensors",
                native_tensor_family="SD15",
                image_number=1,
                base_model_hash="a" * 64,
            )
            | values,
            self.manager,
            self.models,
        )

    def test_off_is_identity_without_model_file_capture_or_clone(self):
        values = {"native_export_latent": False, "native_export_conditioning": False}
        self.assertIs(native.capture_request(values, object(), object()), values)
        self.models.get_file.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_native_latent_input_retains_original_precision_and_legacy_conversion_is_explicit(self):
        for marker in (True, False):
            with self.subTest(marker=marker):
                raw = latent_file(marker=marker)
                data = self.capture(native_latent_input=raw)
                self.assertNotIn("native_latent_input", data)
                self.assertEqual((data["width"], data["height"]), (24, 16))
                item = data[native.KEY].inputs[0]
                self.assertEqual(item.path.read_bytes(), raw)
                self.assertTrue(item.path.is_relative_to(self.root / "native-tensor-jobs"))
                output = native.latent_input(self.owner, data)["samples"]
                original = safetensors.torch.load(raw)["latent_tensor"]
                scale = 1 if marker else 1 / 0.18215
                np.testing.assert_array_equal(output.numpy(), (original.float() * scale).numpy())
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(item.dtype, "F16")
                self.assertEqual(
                    data["native_tensor_source"]["inputs"][0]["native_legacy_scale"], scale
                )
        self.paths.get_annotated_filepath.assert_not_called()

    def test_positive_and_negative_native_inputs_keep_native_options_and_original_dtype(self):
        positive = conditioning_file(metadata={"conditioning_options": '{"guidance":2.5}'})
        negative = conditioning_file(dtype=torch.bfloat16)
        data = self.capture(native_positive_input=positive, native_negative_input=negative)
        values = native.conditioning_inputs(self.owner, data)
        self.assertEqual(set(values), {"positive", "negative"})
        self.assertEqual(values["positive"][0][1], {"guidance": 2.5})
        self.assertEqual(values["positive"][0][0].dtype, torch.float16)
        self.assertEqual(values["negative"][0][0].dtype, torch.bfloat16)
        self.paths.get_full_path_or_raise.assert_not_called()

    def test_safe_header_refuses_pickle_shape_dtype_offsets_batch_and_wrong_family_before_reader(
        self,
    ):
        original = latent_file()
        cases = [
            b"\x80\x04pickle" + b"\0" * 20,
            mutate_header(original, lambda h: h["latent_tensor"].update(dtype="F64")),
            mutate_header(original, lambda h: h["latent_tensor"].update(dtype=[])),
            mutate_header(original, lambda h: h["latent_tensor"].update(shape=[2, 4, 1, 3])),
            mutate_header(original, lambda h: h["latent_tensor"].update(shape=[1, 16, 2, 3])),
            mutate_header(original, lambda h: h["latent_tensor"].update(shape=[1, 4, 1, 2, 3])),
            mutate_header(original, lambda h: h["latent_tensor"].update(data_offsets=[1, 2])),
            original + b"unreferenced",
        ]
        with patch.object(
            safetensors.torch, "load_file", side_effect=AssertionError("reader reached")
        ):
            for raw in cases:
                with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                    self.capture(native_latent_input=raw)
        with self.assertRaisesRegex(ValueError, "family"):
            self.capture(native_latent_input=original, native_tensor_family="SDXL")
        with self.assertRaisesRegex(ValueError, "Declare"):
            self.capture(native_latent_input=original, native_tensor_family="auto")
        with self.assertRaises(ValueError):
            self.capture(native_latent_input="C:/other-session/private.latent")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_conditioning_must_match_encoder_and_pooled_context_and_exclude_executable_options(
        self,
    ):
        for raw, family in (
            (conditioning_file(width=1024), "SD15"),
            (conditioning_file(width=2048), "SDXL"),
            (conditioning_file(metadata={"conditioning_options": '{"hooks":"arbitrary"}'}), "SD15"),
            (conditioning_file(metadata={"conditioning_options": '{"strength":NaN}'}), "SD15"),
        ):
            with self.subTest(family=family), self.assertRaises(ValueError):
                native.tensor_header(raw, "conditioning", family)
        native.tensor_header(conditioning_file(width=2048, pooled=1280), "conditioning", "SDXL")
        with self.assertRaisesRegex(ValueError, "Legacy"):
            native.tensor_header(latent_file(channels=16, marker=False), "latent", "Flux")

    def test_stale_files_nonfinite_values_and_wrong_runtime_codec_are_refused(self):
        data = self.capture(native_latent_input=latent_file())
        item = data[native.KEY].inputs[0]
        item.path.write_bytes(item.path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            native.latent_input(self.owner, data)
        bad = safetensors.torch.save({"conditioning": torch.full((1, 1, 768), float("nan"))})
        data = self.capture(native_positive_input=bad)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            native.conditioning_inputs(self.owner, data)
        self.owner.xl_base_patched.vae.latent_channels = 16
        with self.assertRaisesRegex(ValueError, "codec"):
            native.conditioning_inputs(self.owner, data)

    def test_native_exports_keep_dtype_context_and_never_change_global_native_folders(self):
        data = self.capture(native_export_latent=True, native_export_conditioning=True)
        latent = {"samples": torch.arange(24, dtype=torch.bfloat16).view(1, 4, 2, 3)}
        positive = [[torch.ones(1, 2, 768, dtype=torch.float16), {"guidance": 1.5}]]
        negative = [[torch.zeros(1, 2, 768, dtype=torch.bfloat16), {}]]
        exported = native.export_tensors(self.owner, data, latent, positive, negative)
        self.assertEqual([item.kind for item in exported], ["latent", "positive", "negative"])
        for artifact, dtype in zip(exported, (torch.bfloat16, torch.float16, torch.bfloat16)):
            raw = Path(artifact.path).read_bytes()
            self.assertEqual(artifact.sha256, hashlib.sha256(raw).hexdigest())
            header, context = native.tensor_header(
                raw, "latent" if artifact.kind == "latent" else "conditioning", "SD15"
            )
            self.assertEqual(context["model_sha256"], "a" * 64)
            self.assertEqual(
                next(
                    value
                    for key, value in safetensors.torch.load(raw).items()
                    if key in ("latent_tensor", "conditioning")
                ).dtype,
                dtype,
            )
            self.assertNotIn(str(self.root), json.dumps(header))
        self.paths.get_save_image_path.assert_not_called()
        replay = self.capture(
            native_latent_input=Path(exported[0].path).read_bytes(), native_tensor_family="auto"
        )
        native.latent_input(self.owner, replay)
        replay["base_model_hash"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different model"):
            native.latent_input(self.owner, replay)
        replay.pop("base_model_hash")
        with self.assertRaisesRegex(ValueError, "different model"):
            native.latent_input(self.owner, replay)

    def test_specialized_checkpoints_and_auxiliary_patch_context_are_rejected_before_storage(self):
        with (
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value={"native_class": "SD15", "native_config": {"in_channels": 9}},
            ),
            self.assertRaisesRegex(ValueError, "Specialized"),
        ):
            self.capture(native_export_latent=True)
        for key in ("gligen_model",):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "GLIGEN"):
                self.capture(**{key: "selected.safetensors", "native_export_conditioning": True})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cancelled_inputs_and_incomplete_exports_are_never_published(self):
        data = self.capture(native_latent_input=latent_file())
        cancelled = threading.Event()
        cancelled.set()
        data["_job"] = NS(cancelled=cancelled)
        with patch.object(native, "_scoped", side_effect=AssertionError("reader reached")):
            with self.assertRaises(CancelledError):
                native.latent_input(self.owner, data)
        data = self.capture(native_export_latent=True)

        def check():
            if list(data[native.KEY].directory.iterdir()):
                raise CancelledError("cancelled after native serialization")

        data["_job"] = NS(check=check, cancelled=threading.Event())
        with self.assertRaises(CancelledError):
            native.export_tensors(
                self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
            )
        self.assertEqual(data[native.EXPORT_KEY], [])

    def test_request_ownership_pathless_public_contract_and_replay_require_explicit_inputs(self):
        data = self.capture(native_latent_input=latent_file())
        another = self.capture(native_latent_input=latent_file())
        self.assertNotEqual(data[native.KEY].directory, another[native.KEY].directory)
        source = {
            "native_tensor_source": data["native_tensor_source"],
            "native_export_latent": True,
        }
        with self.assertRaisesRegex(ValueError, "Re-upload"):
            native.replay_values(source, {})
        self.assertEqual(
            native.replay_values(source, {"native_latent_input": latent_file()})[
                "native_export_latent"
            ],
            True,
        )
        from pydantic import ValidationError
        from ruinedfooocus.integrations.openai_api import ImageRequest

        for values in (
            {native.KEY: {"directory": "other-session"}},
            dict(native_export_latent=True, n=5),
            dict(native_export_latent=True, controlnet={"type": "detect"}),
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ImageRequest(**values)

    def test_contextual_generation_ui_registers_binary_uploads_with_exports_off(self):
        from ruinedfooocus.features.generation.ui_native_tensor_io import create_ui

        registered = {}
        shared = NS(add_ctrl=lambda name, control: registered.setdefault(name, control))
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            gr.Blocks(),
        ):
            controls = create_ui()
        self.assertEqual(set(registered), set(native.FIELDS))
        for field in native.INPUTS:
            self.assertEqual(controls[field].type, "binary")
        self.assertFalse(controls["native_export_latent"].value)
        self.assertFalse(controls["native_export_conditioning"].value)

    def test_public_generation_reply_separates_tensor_downloads_and_validates_input_fields(self):
        from ruinedfooocus.integrations.openai_api import create_apps
        from test_api_live_contracts import job

        data = self.capture(native_export_latent=True)
        artifact = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )[0]
        image = self.root / "generated.png"
        Image.new("RGB", (24, 16)).save(image)
        result = native.TensorGenerationArtifacts([str(image)], [artifact])
        backend = NS(
            resolve_model=Mock(return_value=("image/owned", {"family": "SD 1.5"})),
            images=Mock(side_effect=lambda *_: job(("result", result))),
        )
        app, _ = create_apps(backend)
        with TestClient(app) as client:
            response = client.post(
                "/images/generations",
                json=dict(
                    prompt="test",
                    native_export_latent=True,
                    response_format="b64_json",
                    native_latent_input=base64.b64encode(latent_file()).decode(),
                    native_tensor_family="SD15",
                ),
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(body["data"]), 1)
        tensor = body["native_tensors"][0]
        self.assertEqual((tensor["kind"], tensor["family"]), ("latent", "SD15"))
        self.assertEqual(base64.b64decode(tensor["b64_json"]), Path(artifact.path).read_bytes())
        self.assertTrue(backend.images.call_args.args[0]["native_export_latent"])

    def pipeline_nodes(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        loader = next(
            node
            for node in process.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom)
                and child.module == "ruinedfooocus.features.generation.native_tensor_io"
                and any(alias.name == "conditioning_inputs" for alias in child.names)
                for child in node.body
            )
        )
        loop = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.For) and ast.unparse(node.target) == "(sign, role, text)"
        )
        prompt_switch = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "prompt_per_step"
                for target in node.targets
            )
            and isinstance(node.value, ast.IfExp)
        )
        caches = next(
            node
            for node in process.body
            if isinstance(node, ast.If)
            and len(node.body) == 2
            and all(isinstance(child, ast.Assign) for child in node.body)
            and ast.unparse(node.body[0].targets[0]) == "positive_conditioning"
        )
        latent = next(
            node
            for node in process.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "tensor_latent is not None"
        )
        sample = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "comfy.sample.sample"
        )
        start = next(
            index
            for index, node in enumerate(process.body)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "sampled_latent"
        )
        export = next(
            node
            for node in process.body[start:]
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom)
                and child.module == "ruinedfooocus.features.generation.native_tensor_io"
                for child in node.body
            )
        )
        preparation = process.body[start : process.body.index(export)]
        decode = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "self.decode_latent"
        )
        return loader, loop, prompt_switch, caches, latent, sample, preparation, export, decode

    def execute_nodes(self, nodes, scope):
        exec(
            compile(ast.Module(body=list(nodes), type_ignores=[]), "actual_image_pipeline", "exec"),
            scope,
        )

    def test_actual_pipeline_overrides_sampler_inputs_and_exports_final_sample_before_decode(self):
        data = self.capture(
            native_latent_input=latent_file(),
            native_positive_input=conditioning_file(),
            native_export_latent=True,
            native_export_conditioning=True,
            native_latent_denoise=0.37,
        )
        data.update(steps=2, seed=17)
        negative = [[torch.zeros(1, 2, 768), {}]]
        self.owner.conditions = {
            "+": {"text": "old", "cache": None},
            "-": {"text": "old", "cache": None},
        }

        def encode(sign, text, skip, **options):
            self.owner.conditions[sign] = {"text": text, "cache": negative}

        self.owner.textencode = Mock(side_effect=encode)
        self.owner.decode_latent = Mock(return_value=torch.zeros(1, 16, 24, 3))
        scope = dict(
            self=self.owner,
            gen_data=data,
            tensor_conditions={},
            tensor_latent=None,
            positive_prompt="must not encode",
            negative_prompt="encode negative",
            clip_skip=1,
            encode_options={},
            architecture="BaseModel",
            joy_reference_edit=False,
            pp=NS(
                prompt_switch_per_step=Mock(
                    side_effect=AssertionError("re-encoded imported positive")
                )
            ),
        )
        loader, loop, switch, caches, latent, sample, preparation, export, decode = (
            self.pipeline_nodes()
        )
        self.execute_nodes((loader, loop, switch, caches, latent), scope)
        self.owner.textencode.assert_called_once_with("-", "encode negative", 1)
        self.assertIs(scope["positive_conditioning"], scope["tensor_conditions"]["positive"])
        self.assertEqual(scope["prompt_per_step"], [])
        self.assertEqual(scope["denoise"], 0.37)
        sampled = torch.ones_like(scope["latent"]["samples"]) * 0.25
        sampler = Mock(return_value=sampled)
        scope.update(
            comfy=NS(sample=NS(sample=sampler), utils=NS(PROGRESS_BAR_ENABLED=False)),
            sampling_model=object(),
            noise=torch.zeros_like(sampled),
            cfg=1.5,
            sampler_name="euler",
            scheduler="simple",
            positive_cond=scope["positive_conditioning"],
            latent_image=scope["latent"]["samples"],
            fixed_sigmas=None,
            noise_mask=None,
            callback_function=None,
            seed=17,
            guidance={"vae_tiled": False, "vae_tile_size": 512, "vae_overlap": 64},
            cascade_latents=None,
            reference_frame=None,
        )
        self.execute_nodes((sample, *preparation, export, decode), scope)
        self.assertIs(sampler.call_args.kwargs["latent_image"], scope["latent"]["samples"])
        self.assertEqual(sampler.call_args.kwargs["denoise"], 0.37)
        self.assertEqual(len(data[native.EXPORT_KEY]), 3)
        np.testing.assert_array_equal(
            safetensors.torch.load_file(data[native.EXPORT_KEY][0].path)["latent_tensor"].numpy(),
            sampled.numpy(),
        )
        self.assertIs(self.owner.decode_latent.call_args.args[0]["samples"], sampled)
        self.assertEqual(data[native.EXPORT_KEY][0].seed, 17)

    def test_actual_pipeline_off_guards_never_load_capture_or_export_tensors(self):
        loader, *_, export, decode = self.pipeline_nodes()
        with (
            patch.object(
                native, "conditioning_inputs", side_effect=AssertionError("input copy")
            ) as load,
            patch.object(
                native, "export_tensors", side_effect=AssertionError("output copy")
            ) as save,
        ):
            self.execute_nodes((loader, export), dict(gen_data={}, self=object()))
        load.assert_not_called()
        save.assert_not_called()

    def test_export_bounds_run_before_native_cpu_copy_or_file_creation(self):
        data = self.capture(native_export_latent=True)
        with patch.object(
            self.modules["nodes"].SaveLatent, "save", side_effect=AssertionError("copy reached")
        ) as save:
            with self.assertRaisesRegex(ValueError, "before copying"):
                native.export_tensors(
                    self.owner,
                    data,
                    {"samples": torch.empty((1, 4, 4096, 4096), device="meta")},
                    None,
                    None,
                )
        save.assert_not_called()
        self.assertEqual(list(data[native.KEY].directory.iterdir()), [])

    def test_actual_public_runtime_capture_keeps_typed_inputs_and_latent_geometry(self):
        from test_api_image_edits import queued_data

        data = queued_data(
            "SD 1.5",
            payload_options=dict(
                input_image=None,
                size="auto",
                native_latent_input=base64.b64encode(latent_file()).decode(),
                native_tensor_family="SD15",
                native_export_latent=True,
                native_latent_denoise=0.3,
            ),
        )
        self.assertEqual((data["custom_width"], data["custom_height"]), (24, 16))
        self.assertEqual(data["native_latent_denoise"], 0.3)
        self.assertEqual(data[native.KEY].family, "SD15")
        self.assertEqual(data[native.KEY].inputs[0].dtype, "F16")

    def test_actual_library_setup_replay_requires_original_owned_uploads_and_keeps_them(self):
        from test_gallery_actions import functions, Models

        raw = latent_file()
        data = self.capture(native_latent_input=raw)
        scope = functions()
        scope["apply_metadata"] = lambda *_: {}
        metadata = dict(
            prompt="continue",
            base_model_name="new/model.safetensors",
            base_model_hash="a" * 64,
            native_tensor_source=data["native_tensor_source"],
            native_latent_denoise=0.27,
            native_export_latent=True,
        )
        models = Models(self.root / "gallery-models")
        modules = {
            "ruinedfooocus.app": NS(state=NS(models=models, settings=NS(default_settings={}))),
            "ruinedfooocus.core.util": NS(
                get_lora_thumbnail=lambda _: "thumbnail", load_keywords=lambda _: ""
            ),
        }
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(gr.Error, "Re-upload"):
                scope["reuse_setup"](metadata)
            with self.assertRaisesRegex(gr.Error, "identity"):
                scope["reuse_setup"](
                    metadata, tensor_inputs={"native_latent_input": latent_file(marker=False)}
                )
            updates = scope["reuse_setup"](metadata, tensor_inputs={"native_latent_input": raw})
        self.assertNotIn("native_latent_input", updates)
        self.assertIsNone(updates["native_positive_input"]["value"])
        self.assertTrue(updates["native_export_latent"]["value"])
        self.assertEqual(updates["native_latent_denoise"]["value"], 0.27)

    def test_public_tensor_url_retains_real_extension_and_only_contains_generated_artifacts(self):
        from ruinedfooocus.integrations.openai_api import create_apps
        from test_api_live_contracts import job

        data = self.capture(native_export_latent=True)
        artifact = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )[0]
        image = self.root / "url-result.png"
        Image.new("RGB", (24, 16)).save(image)
        result = native.TensorGenerationArtifacts([str(image)], [artifact])
        backend = NS(
            resolve_model=Mock(return_value=("image/owned", {"family": "SD 1.5"})),
            images=Mock(side_effect=lambda *_: job(("result", result))),
        )
        app, _ = create_apps(backend)
        with TestClient(app) as client:
            reply = client.post("/images/generations", json={"native_export_latent": True}).json()
            token = reply["native_tensors"][0]["url"].rsplit("/", 1)[1]
            downloaded = client.get("/images/" + token)
            self.assertEqual(downloaded.status_code, 200)
            self.assertEqual(downloaded.content, Path(artifact.path).read_bytes())
            self.assertIn(".latent", downloaded.headers["content-disposition"])
            self.assertEqual(client.get("/images/guessed-other-session-path").status_code, 404)

    def test_actual_worker_attachment_and_create_downloads_keep_images_and_tensors_separate(self):
        data = self.capture(native_export_latent=True)
        artifacts = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )
        image = self.root / "worker-output.png"
        Image.new("RGB", (24, 16)).save(image)
        tree = ast.parse((ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
        process = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process"
        )
        tensor_return = next(
            node
            for node in process.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Constant) and child.value == "_native_tensor_exports"
                for child in ast.walk(node.test)
            )
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "TensorGenerationArtifacts"
                for child in ast.walk(node)
            )
        )
        self.assertIsInstance(process.body[-1], ast.Return)
        packet = ast.FunctionDef(
            name="packet",
            args=ast.parse("def packet(gen_data, res, stop_batch): pass").body[0].args,
            body=[tensor_return, process.body[-1]],
            decorator_list=[],
        )
        scope = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[packet], type_ignores=[])),
                "actual worker packet",
                "exec",
            ),
            scope,
        )
        result = scope["packet"](data, [str(image)], False)
        self.assertEqual(list(result), [str(image)])
        self.assertEqual(result.tensor_files, tuple(artifacts))
        self.assertFalse(hasattr(scope["packet"](data, [str(image)], True), "tensor_files"))
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        callback = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "update_results"
        )
        namespace = dict(
            gr=gr,
            shared=NS(get_ctrl=lambda name: name),
            editor_workspace=NS(latest_generated="latest_generated", source_input="source_input"),
        )
        for name in (
            "run_button",
            "stop_button",
            "progress_html",
            "main_view",
            "inpaint_toggle",
            "gallery",
            "generation_originals",
            "generation_artifacts",
            "native_main_source",
            "metadata_json",
        ):
            namespace[name] = name
        exec(
            compile(
                ast.Module(body=[callback], type_ignores=[]),
                "actual Create result callback",
                "exec",
            ),
            namespace,
        )
        updates = namespace["update_results"](result)
        self.assertEqual(updates["generation_originals"]["value"], [str(image), artifacts[0].path])
        self.assertEqual(updates["gallery"]["value"], [str(image)])
        self.assertEqual(len(updates["generation_artifacts"]), 1)
        Path(artifacts[0].path).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed before display"):
            namespace["update_results"](result)


if __name__ == "__main__":
    unittest.main()
