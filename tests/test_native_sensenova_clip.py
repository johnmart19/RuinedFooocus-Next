"""Tokenizer-only SenseNova uses native sentinels and frozen reload identities."""

import ast
from concurrent.futures import CancelledError
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import CLIPTokenizer, Qwen2Tokenizer

from ruinedfooocus.backends.comfy.sensenova_clip import load_sensenova_clip
from ruinedfooocus.features.models.model_identity import file_signature
from test_native_model_coverage import NATIVE


def extract_class(filename, name, scope, methods=None, bases=None):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if bases is not None:
        node.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    for method in node.body:
        if isinstance(method, ast.FunctionDef):
            method.returns = None
            for argument in method.args.args + method.args.kwonlyargs:
                argument.annotation = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename, "exec"
        ),
        scope,
    )
    return scope[name]


def native_tokenizer_and_encoder():
    scope = dict(
        torch=torch,
        os=os,
        re=re,
        logging=logging,
        CLIPTokenizer=CLIPTokenizer,
        Qwen2Tokenizer=Qwen2Tokenizer,
        __file__=str(NATIVE / "comfy/sd1_clip.py"),
    )
    tree = ast.parse((NATIVE / "comfy/sd1_clip.py").read_text(encoding="utf-8"))
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in ("parse_parentheses", "token_weights", "escape_important", "unescape_important")
    ]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "native-tokenizer-functions", "exec"),
        scope,
    )
    extract_class("comfy/sd1_clip.py", "SDTokenizer", scope)
    extract_class("comfy/sd1_clip.py", "SD1Tokenizer", scope)
    scope["sd1_clip"] = SimpleNamespace(
        SDTokenizer=scope["SDTokenizer"], SD1Tokenizer=scope["SD1Tokenizer"]
    )
    scope["__file__"] = str(NATIVE / "comfy/text_encoders/sensenova.py")
    source = ast.parse((NATIVE / "comfy/text_encoders/sensenova.py").read_text(encoding="utf-8"))
    values = [n for n in source.body if isinstance(n, ast.Assign) or isinstance(n, ast.FunctionDef)]
    exec(
        compile(ast.Module(body=values, type_ignores=[]), "native-sensenova-prompts", "exec"), scope
    )
    for name in (
        "SenseNovaQwen2Tokenizer",
        "SenseNovaQwenTokenizer",
        "SenseNovaTokenizer",
        "SenseNovaTextEncoder",
    ):
        extract_class("comfy/text_encoders/sensenova.py", name, scope)
    return scope["SenseNovaTokenizer"], scope["SenseNovaTextEncoder"]


def config(tokenizer="native-tokenizer", encoder="native-text-encoder"):
    target = extract_class(
        "comfy/supported_models_base.py", "ClipTarget", {}, methods=("__init__",), bases=()
    )
    scope = dict(
        torch=torch,
        supported_models_base=SimpleNamespace(ClipTarget=target),
        comfy=SimpleNamespace(
            text_encoders=SimpleNamespace(
                sensenova=SimpleNamespace(
                    SenseNovaTokenizer=tokenizer, SenseNovaTextEncoder=encoder
                )
            )
        ),
    )
    native = extract_class(
        "comfy/supported_models.py",
        "SenseNovaU15",
        scope,
        methods=("process_clip_state_dict", "clip_target"),
        bases=(),
    )
    return native()


def modules(constructor):
    native = SimpleNamespace(
        sd=SimpleNamespace(CLIP=constructor),
        utils=SimpleNamespace(
            calculate_parameters=lambda state: sum(x.numel() for x in state.values()),
            load_torch_file=Mock(side_effect=AssertionError("Tokenizer does not read weights")),
        ),
        model_management=SimpleNamespace(throw_exception_if_processing_interrupted=Mock()),
    )
    return {
        "comfy": native,
        "comfy.sd": native.sd,
        "comfy.utils": native.utils,
        "comfy.model_management": native.model_management,
    }


class SenseNovaClipTests(unittest.TestCase):
    def test_actual_native_full_state_sentinel_constructor_and_tokenizer_are_offline(self):
        tokenizer, encoder = native_tokenizer_and_encoder()
        patcher = SimpleNamespace(is_dynamic=lambda: False, set_model_compute_dtype=Mock())
        memory = SimpleNamespace(
            text_encoder_device=lambda: "cpu",
            text_encoder_offload_device=lambda: "cpu",
            text_encoder_dtype=lambda device: torch.float32,
            text_encoder_initial_device=lambda *args: "meta",
            dtype_size=lambda dtype: 4,
            supports_cast=lambda *args: True,
            archive_model_dtypes=Mock(),
            load_models_gpu=Mock(side_effect=AssertionError("No model/GPU loading")),
        )
        scope = dict(
            torch=torch,
            logging=logging,
            model_management=memory,
            comfy=SimpleNamespace(
                model_patcher=SimpleNamespace(
                    ModelPatcher=lambda *args, **kwargs: patcher,
                    CoreModelPatcher=lambda *args, **kwargs: patcher,
                ),
                storage=SimpleNamespace(state_dict_fast_disk=lambda state: None),
                hooks=SimpleNamespace(EnumHookMode=SimpleNamespace(MinVram="native-min-vram")),
            ),
        )
        clip_type = extract_class(
            "comfy/sd.py", "CLIP", scope, methods=("__init__", "load_sd", "tokenize"), bases=()
        )
        dependencies = modules(clip_type)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "identity-only.safetensors"
            checkpoint.write_bytes(b"owned identity fixture; no weights")
            with (
                patch.dict("sys.modules", dependencies),
                patch("socket.create_connection", side_effect=AssertionError("Offline tokenizer")),
            ):
                clip = load_sensenova_clip(config(tokenizer, encoder), file_signature(checkpoint))
                positive = clip.tokenize("a cat")
                negative = clip.tokenize("")
                context, pooled, extra = clip.cond_stage_model.encode_token_weights(positive)
            self.assertIs(clip.patcher, patcher)
            ids = extra["text_input_ids"]
            self.assertEqual(ids.dtype, torch.long)
            self.assertEqual(context.shape, (*ids.shape, 1))
            self.assertEqual(ids.device.type, "cpu")
            self.assertIsNone(pooled)
            self.assertNotEqual(positive, negative)
            self.assertTrue(all(value[0] != 151643 for value in positive["sensenova_u15"][0]))
            dependencies["comfy.utils"].load_torch_file.assert_not_called()
            memory.load_models_gpu.assert_not_called()

    def test_native_sentinel_dictionary_and_disable_dynamic_factory_contract_are_preserved(self):
        constructor = Mock(
            side_effect=lambda *args, **kwargs: SimpleNamespace(
                patcher=SimpleNamespace(get_clone_model_override=lambda: "native-override")
            )
        )
        dependencies = modules(constructor)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "identity-only.safetensors"
            checkpoint.write_bytes(b"owned fixture")
            signature = file_signature(checkpoint)
            with patch.dict("sys.modules", dependencies):
                clip = load_sensenova_clip(config(), signature)
            kwargs = constructor.call_args.kwargs
            self.assertIsInstance(kwargs["state_dict"], dict)
            self.assertEqual(set(kwargs["state_dict"]), {"_sensenova_te_sentinel"})
            self.assertIs(kwargs["tokenizer_data"], kwargs["state_dict"])
            self.assertEqual(kwargs["parameters"], 1)
            native = ast.parse((NATIVE / "comfy/model_patcher.py").read_text(encoding="utf-8"))
            patcher = next(
                n for n in native.body if isinstance(n, ast.ClassDef) and n.name == "ModelPatcher"
            )
            clone = next(
                n for n in patcher.body if isinstance(n, ast.FunctionDef) and n.name == "clone"
            )
            branch = next(n for n in clone.body if isinstance(n, ast.If))
            scope = dict(
                self=SimpleNamespace(
                    is_dynamic=lambda: True, cached_patcher_init=clip.patcher.cached_patcher_init
                ),
                disable_dynamic=True,
                force_deepcopy=False,
                model_override=None,
                ModelPatcher="native-static",
            )
            with patch.dict("sys.modules", dependencies):
                exec(
                    compile(
                        ast.Module(body=[branch], type_ignores=[]), "native-clone-consumer", "exec"
                    ),
                    scope,
                )
            self.assertEqual(scope["model_override"], "native-override")
            self.assertTrue(constructor.call_args.kwargs["disable_dynamic"])
            dependencies["comfy.utils"].load_torch_file.assert_not_called()

    def test_reload_owns_frozen_checkpoint_lease_and_refuses_replacement(self):
        active = []

        @contextmanager
        def lease(path, **kwargs):
            kwargs["cancelled"].is_set()
            active.append(Path(path))
            try:
                yield path
            finally:
                active.remove(Path(path))

        constructor = Mock(
            side_effect=lambda *args, **kwargs: SimpleNamespace(patcher=SimpleNamespace())
        )
        dependencies = modules(constructor)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "identity.safetensors"
            checkpoint.write_bytes(b"owned fixture")
            with patch.dict("sys.modules", dependencies):
                clip = load_sensenova_clip(config(), file_signature(checkpoint))
            reload, args = clip.patcher.cached_patcher_init

            def native(*args, **kwargs):
                self.assertEqual(active, [checkpoint.resolve()])
                return SimpleNamespace(patcher=SimpleNamespace())

            constructor.side_effect = native
            with (
                patch.dict("sys.modules", dependencies),
                patch(
                    "ruinedfooocus.core.preparation.preparation_service",
                    SimpleNamespace(artifact_lease=lease),
                ),
            ):
                reload(*args)
                self.assertFalse(active)
                checkpoint.write_bytes(b"replacement")
                calls = constructor.call_count
                with self.assertRaisesRegex(ValueError, "changed after capture"):
                    reload(*args)
                self.assertEqual(constructor.call_count, calls)

    def test_wrong_config_missing_identity_and_cancellation_refuse_without_weight_reads(self):
        constructor = Mock()
        dependencies = modules(constructor)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "identity.safetensors"
            checkpoint.write_bytes(b"owned fixture")
            with patch.dict("sys.modules", dependencies):
                with self.assertRaisesRegex(ValueError, "identity must be captured"):
                    load_sensenova_clip(config(), None)
                with self.assertRaisesRegex(ValueError, "requires native SenseNovaU15"):
                    load_sensenova_clip(SimpleNamespace(), file_signature(checkpoint))
                dependencies[
                    "comfy.model_management"
                ].throw_exception_if_processing_interrupted.side_effect = CancelledError(
                    "cancelled"
                )
                with self.assertRaises(CancelledError):
                    load_sensenova_clip(config(), file_signature(checkpoint))
            constructor.assert_not_called()
            dependencies["comfy.utils"].load_torch_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
