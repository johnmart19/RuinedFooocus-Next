"""Pinned HyDiT native constructor/tokenizer contracts with no real model weights."""

import ast
from concurrent.futures import CancelledError
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import BertTokenizer, CLIPTokenizer
from sentencepiece import sentencepiece_model_pb2

from ruinedfooocus.backends.comfy.hunyuan_dit import load_hunyuan_dit_clip
from ruinedfooocus.features.models.model_identity import file_signature
from test_native_model_coverage import NATIVE, native_method


def native_class(path, name, scope, *, methods=None, bases=None):
    tree = ast.parse((NATIVE / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    if bases is not None:
        node.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, "exec"),
        scope,
    )
    return scope[name]


def native_function(path, name, scope):
    tree = ast.parse((NATIVE / path).read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def tokenizer_proto():
    """Build a tiny valid fixture directly; no downloaded model or tokenizer training."""
    proto = sentencepiece_model_pb2.ModelProto()
    proto.trainer_spec.model_type = 1
    proto.trainer_spec.unk_id = 2
    proto.trainer_spec.pad_id = 0
    proto.trainer_spec.eos_id = 1
    proto.trainer_spec.bos_id = -1
    for text, kind in (
        ("<pad>", 3),
        ("</s>", 3),
        ("<unk>", 2),
        ("▁", 1),
        ("猫", 1),
        ("c", 1),
        ("a", 1),
        ("t", 1),
    ):
        piece = proto.pieces.add()
        piece.piece, piece.score, piece.type = text, 0, kind
    proto.trainer_spec.vocab_size = len(proto.pieces)
    return torch.tensor(list(proto.SerializeToString()), dtype=torch.uint8)


class HunyuanDiTNativeTests(unittest.TestCase):
    def test_native_detector_keeps_hydit1_size_style_contract_distinct(self):
        scope = {}
        scope["count_blocks"] = native_function("comfy/model_detection.py", "count_blocks", {})
        detect = native_function("comfy/model_detection.py", "detect_unet_config", scope)
        shared = {
            "mlp_t5.0.weight": SimpleNamespace(shape=(1024, 2048)),
            "x_embedder.proj.weight": SimpleNamespace(shape=(1408, 4, 2, 2)),
            **{f"blocks.{index}.fixture": SimpleNamespace(shape=(1,)) for index in range(40)},
        }
        for width, expected in ((1024, "hydit"), (3968, "hydit1")):
            result = detect(
                shared | {"extra_embedder.0.weight": SimpleNamespace(shape=(5632, width))}, ""
            )
            self.assertEqual(result["image_model"], expected)
            self.assertEqual((result["hidden_size"], result["depth"]), (1408, 40))
            self.assertEqual(result.get("size_cond", False), expected == "hydit1")
            self.assertEqual(result.get("use_style_cond", False), expected == "hydit1")

    def test_native_variants_share_runtime_without_flattening_their_discrete_beta_schedule(self):
        factory = Mock(return_value="native-hydit-runtime")
        scope = dict(
            torch=torch,
            supported_models_base=SimpleNamespace(BASE=object),
            latent_formats=SimpleNamespace(SDXL="native-sdxl-latent"),
            model_base=SimpleNamespace(HunyuanDiT=factory),
        )
        first = native_class("comfy/supported_models.py", "HunyuanDiT", scope)
        second = native_class("comfy/supported_models.py", "HunyuanDiT1", scope)
        beta = native_function(
            "comfy/ldm/modules/diffusionmodules/util.py", "make_beta_schedule", dict(torch=torch)
        )
        sampler_type = native_class(
            "comfy/model_sampling.py",
            "ModelSamplingDiscrete",
            dict(torch=torch, make_beta_schedule=beta),
            methods=("__init__", "_register_schedule", "set_sigmas"),
        )
        sigmas = []
        for config_type, linear_end in ((first, 0.018), (second, 0.03)):
            native_config = config_type()
            self.assertEqual(native_config.latent_format, "native-sdxl-latent")
            self.assertEqual(native_config.get_model({}), "native-hydit-runtime")
            self.assertIs(factory.call_args.args[0], native_config)
            sampler = sampler_type(native_config)
            self.assertEqual((sampler.linear_start, sampler.linear_end), (0.00085, linear_end))
            sigmas.append(sampler.sigmas)
        self.assertFalse(torch.allclose(*sigmas))
        self.assertGreater(sigmas[1][-1].item(), sigmas[0][-1].item())

    def tokenizers(self):
        scope = dict(
            torch=torch,
            os=os,
            logging=logging,
            CLIPTokenizer=CLIPTokenizer,
            __file__=str(NATIVE / "comfy/sd1_clip.py"),
        )
        base = native_class("comfy/sd1_clip.py", "SDTokenizer", scope, methods=("__init__",))
        spiece = native_class("comfy/text_encoders/spiece_tokenizer.py", "SPieceTokenizer", scope)
        scope.update(
            sd1_clip=SimpleNamespace(SDTokenizer=base),
            BertTokenizer=BertTokenizer,
            SPieceTokenizer=spiece,
            __file__=str(NATIVE / "comfy/text_encoders/hydit.py"),
        )
        for name in ("HyditBertTokenizer", "MT5XLTokenizer", "HyditTokenizer"):
            native_class("comfy/text_encoders/hydit.py", name, scope)
        return scope["HyditTokenizer"]

    def test_real_native_tokenizer_constructors_are_offline_with_embedded_proto(self):
        tokenizer = self.tokenizers()
        proto = tokenizer_proto()
        with patch(
            "socket.create_connection",
            side_effect=AssertionError("Offline tokenizer cannot connect"),
        ):
            actual = tokenizer(tokenizer_data={"mt5xl.spiece_model": proto})
        self.assertEqual(actual.hydit_clip.embedding_size, 1024)
        self.assertEqual(actual.mt5xl.embedding_size, 2048)
        self.assertTrue(actual.hydit_clip.tokenizer("猫 cat")["input_ids"])
        self.assertTrue(actual.mt5xl.tokenizer("猫 cat")["input_ids"])
        self.assertTrue(torch.equal(actual.state_dict()["mt5xl.spiece_model"], proto))
        with self.assertRaises((TypeError, ValueError)):
            tokenizer(tokenizer_data={})

    def test_actual_pinned_split_loader_omits_proto_while_bundle_prefix_preserves_it(self):
        captured = Mock(return_value="native-clip")
        scope = dict(
            CLIPType=SimpleNamespace(STABLE_DIFFUSION=0, HUNYUAN_DIT=1, SD3=2),
            CLIP=captured,
            comfy=SimpleNamespace(
                utils=SimpleNamespace(calculate_parameters=lambda state: 0),
                text_encoders=SimpleNamespace(
                    hydit=SimpleNamespace(
                        HyditModel="native-model", HyditTokenizer="native-tokenizer"
                    ),
                    long_clipl=SimpleNamespace(
                        model_options_long_clip=lambda state, data, options: (data, options)
                    ),
                ),
            ),
        )
        load = native_function("comfy/sd.py", "load_text_encoder_state_dicts", scope)
        states = [dict(bert="weight"), {"spiece_model": tokenizer_proto()}]
        self.assertEqual(load(states, clip_type=1), "native-clip")
        self.assertEqual(captured.call_args.kwargs["tokenizer_data"], {})
        self.assertIs(captured.call_args.kwargs["state_dict"], states)
        # Bundled native construction instead passes its entire namespaced clip state.
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        bundled = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_state_dict_guess_config"
        )
        calls = [
            node
            for node in ast.walk(bundled)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "CLIP"
        ]
        self.assertTrue(
            any(
                any(
                    arg.arg == "tokenizer_data" and ast.unparse(arg.value) == "clip_sd"
                    for arg in call.keywords
                )
                for call in calls
            )
        )

    def test_native_text_model_forwards_both_masks_and_secondary_conditioning(self):
        scope = {}
        model_type = native_class(
            "comfy/text_encoders/hydit.py",
            "HyditModel",
            scope,
            methods=("encode_token_weights",),
            bases=(),
        )
        context, mt5 = torch.ones(1, 3, 1024), torch.ones(1, 4, 2048)
        bert_mask, mt5_mask = torch.tensor([[1, 1, 0]]), torch.tensor([[1, 1, 1, 0]])
        model = model_type()
        model.hydit_clip = SimpleNamespace(
            encode_token_weights=Mock(
                return_value=(context, "pooled", dict(attention_mask=bert_mask))
            )
        )
        model.mt5xl = SimpleNamespace(
            encode_token_weights=Mock(return_value=(mt5, None, dict(attention_mask=mt5_mask)))
        )
        encoded, pooled, extra = model.encode_token_weights(
            dict(hydit_clip="bert-tokens", mt5xl="mt5-tokens")
        )
        self.assertIs(encoded, context)
        self.assertEqual(pooled, "pooled")
        self.assertIs(extra["attention_mask"], bert_mask)
        self.assertIs(extra["conditioning_mt5xl"], mt5)
        self.assertIs(extra["attention_mask_mt5xl"], mt5_mask)

    def test_native_sampler_provides_actual_nonsquare_geometry_without_a_new_patch(self):
        class BaseModel:
            def extra_conds(self, **kwargs):
                return {}

        scope = dict(
            torch=torch,
            BaseModel=BaseModel,
            comfy=SimpleNamespace(conds=SimpleNamespace(CONDRegular=lambda value: value)),
        )
        model = native_method(
            "comfy/model_base.py", "HunyuanDiT", "extra_conds", scope, bases=("BaseModel",)
        )()
        encode = native_function("comfy/samplers.py", "encode_model_conds", {})
        noise = torch.zeros(1, 4, 64, 96)
        values = dict(
            model_conds={},
            cross_attn="bert",
            attention_mask="bert-mask",
            conditioning_mt5xl="mt5",
            attention_mask_mt5xl="mt5-mask",
        )
        output = encode(model.extra_conds, [values], noise, "cpu", "positive")[0]["model_conds"]
        torch.testing.assert_close(
            output["image_meta_size"], torch.tensor([[512.0, 768.0, 512.0, 768.0, 0.0, 0.0]])
        )
        self.assertEqual(output["encoder_hidden_states_t5"], "mt5")
        self.assertEqual(output["text_embedding_mask"], "bert-mask")
        self.assertEqual(output["text_embedding_mask_t5"], "mt5-mask")

    def fixtures(self, root):
        paths = [(root / "bert.safetensors").resolve(), (root / "mt5.safetensors").resolve()]
        for path in paths:
            path.write_bytes(b"owned identity fixture")
        states = [
            {
                "bert.encoder.layer.0.attention.self.query.weight": torch.empty(
                    1024, 1024, device="meta"
                )
            },
            {
                "encoder.block.23.layer.1.DenseReluDense.wi_1.weight": torch.empty(
                    5120, 2048, device="meta"
                ),
                "spiece_model": tokenizer_proto(),
            },
        ]
        constructor = Mock(
            side_effect=lambda *args, **kwargs: SimpleNamespace(patcher=SimpleNamespace())
        )
        read = Mock(
            side_effect=lambda path, **kwargs: (
                states[paths.index(Path(path))],
                {"path": str(path)},
            )
        )
        native = SimpleNamespace(
            sd=SimpleNamespace(CLIP=constructor),
            utils=SimpleNamespace(
                load_torch_file=read,
                calculate_parameters=lambda state: 42,
                convert_old_quants=Mock(
                    side_effect=lambda state, **kwargs: (state, kwargs["metadata"])
                ),
            ),
            model_management=SimpleNamespace(throw_exception_if_processing_interrupted=Mock()),
        )
        modules = {
            "comfy": native,
            "comfy.sd": native.sd,
            "comfy.utils": native.utils,
            "comfy.model_management": native.model_management,
        }
        return paths, states, constructor, read, modules

    def test_adapter_reads_each_state_once_and_reload_keeps_only_frozen_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths, states, constructor, read, modules = self.fixtures(root)
            with patch.dict("sys.modules", modules):
                clip = load_hunyuan_dit_clip(
                    paths, "native-target", model_options=dict(load_device="cpu")
                )
            self.assertEqual(read.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs == dict(safe_load=True, return_metadata=True)
                    for call in read.call_args_list
                )
            )
            self.assertEqual(modules["comfy.utils"].convert_old_quants.call_count, 2)
            kwargs = constructor.call_args.kwargs
            self.assertIs(kwargs["state_dict"][0], states[0])
            self.assertIs(kwargs["state_dict"][1], states[1])
            self.assertIs(kwargs["tokenizer_data"]["mt5xl.spiece_model"], states[1]["spiece_model"])
            self.assertEqual(kwargs["parameters"], 84)
            reload, args = clip.patcher.cached_patcher_init
            self.assertEqual(args[0], tuple(str(path.resolve()) for path in paths))
            self.assertEqual(args[1], tuple(tuple(file_signature(path)) for path in paths))
            active = []

            @contextmanager
            def lease(path, **kwargs):
                kwargs["cancelled"].is_set()
                active.append(str(path))
                try:
                    yield path
                finally:
                    active.remove(str(path))

            reloaded_patcher = SimpleNamespace()

            def construct(*args, **kwargs):
                self.assertEqual(active, [str(path.resolve()) for path in paths])
                return SimpleNamespace(patcher=reloaded_patcher)

            constructor.side_effect = construct
            service = SimpleNamespace(artifact_lease=lease)
            with (
                patch.dict("sys.modules", modules),
                patch("ruinedfooocus.core.preparation.preparation_service", service),
            ):
                self.assertIs(reload(*args), reloaded_patcher)
                self.assertFalse(active)
                paths[1].write_bytes(b"replacement")
                before = read.call_count
                with self.assertRaisesRegex(ValueError, "changed after capture"):
                    reload(*args)
                self.assertEqual(read.call_count, before)

    def test_adapter_executes_actual_native_clip_constructor_and_hydit_weight_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, states, _, read, modules = self.fixtures(Path(folder))
            memory = SimpleNamespace(
                text_encoder_device=lambda: "cpu",
                text_encoder_offload_device=lambda: "cpu",
                text_encoder_dtype=lambda device: torch.float32,
                text_encoder_initial_device=lambda *args: "meta",
                dtype_size=lambda dtype: 4,
                archive_model_dtypes=Mock(),
                load_models_gpu=Mock(side_effect=AssertionError("No model/device loading")),
            )
            stage_type = native_class(
                "comfy/text_encoders/hydit.py",
                "HyditModel",
                {},
                methods=("load_sd",),
                bases=(),
            )
            stage = stage_type()
            stage.dtypes = set()
            stage.modules = lambda: []
            stage.hydit_clip = SimpleNamespace(load_sd=Mock(return_value=([], [])))
            stage.mt5xl = SimpleNamespace(load_sd=Mock(return_value=([], [])))
            patcher = SimpleNamespace(set_model_compute_dtype=Mock(), is_dynamic=lambda: False)
            scope = dict(
                torch=torch,
                logging=logging,
                model_management=memory,
                comfy=SimpleNamespace(
                    model_patcher=SimpleNamespace(
                        ModelPatcher=lambda *args, **kwargs: patcher,
                        CoreModelPatcher=lambda *args, **kwargs: patcher,
                    ),
                    storage=SimpleNamespace(state_dict_fast_disk=lambda states: None),
                    hooks=SimpleNamespace(EnumHookMode=SimpleNamespace(MinVram="native-min-vram")),
                ),
            )
            clip_type = native_class(
                "comfy/sd.py", "CLIP", scope, methods=("__init__", "load_sd"), bases=()
            )
            modules["comfy.sd"].CLIP = clip_type
            target = SimpleNamespace(
                params={}, clip=lambda **kwargs: stage, tokenizer=self.tokenizers()
            )
            with patch.dict("sys.modules", modules):
                clip = load_hunyuan_dit_clip(paths, target)
            self.assertIs(clip.patcher, patcher)
            self.assertTrue(clip.tokenizer.mt5xl.tokenizer("cat")["input_ids"])
            stage.hydit_clip.load_sd.assert_called_once_with(states[0])
            stage.mt5xl.load_sd.assert_called_once_with(states[1])
            self.assertEqual(read.call_count, 2)
            memory.load_models_gpu.assert_not_called()

    def test_actual_native_clone_consumer_forwards_disable_dynamic_through_reload_factory(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, _, constructor, _, modules = self.fixtures(Path(folder))
            constructor.side_effect = lambda *args, **kwargs: SimpleNamespace(
                patcher=SimpleNamespace(get_clone_model_override=lambda: "native-clone-override")
            )
            with patch.dict("sys.modules", modules):
                clip = load_hunyuan_dit_clip(paths, "native-target")
            owner = SimpleNamespace(
                is_dynamic=lambda: True, cached_patcher_init=clip.patcher.cached_patcher_init
            )
            tree = ast.parse((NATIVE / "comfy/model_patcher.py").read_text(encoding="utf-8"))
            patcher = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelPatcher"
            )
            clone = next(
                n for n in patcher.body if isinstance(n, ast.FunctionDef) and n.name == "clone"
            )
            branch = next(n for n in clone.body if isinstance(n, ast.If))
            scope = dict(
                self=owner,
                disable_dynamic=True,
                force_deepcopy=False,
                model_override=None,
                ModelPatcher="native-nondynamic-type",
            )
            with patch.dict("sys.modules", modules):
                exec(
                    compile(
                        ast.Module(body=[branch], type_ignores=[]), "native-clone-consumer", "exec"
                    ),
                    scope,
                )
            self.assertEqual(scope["model_override"], "native-clone-override")
            self.assertTrue(constructor.call_args.kwargs["disable_dynamic"])

    def test_native_per_file_quant_metadata_and_custom_operations_guard_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, states, _, _, modules = self.fixtures(Path(folder))
            normalize = modules["comfy.utils"].convert_old_quants
            with patch.dict("sys.modules", modules):
                load_hunyuan_dit_clip(paths, "native-target")
            self.assertEqual(normalize.call_count, 2)
            for call, state, path in zip(normalize.call_args_list, states, paths):
                self.assertIs(call.args[0], state)
                self.assertEqual(
                    call.kwargs, dict(model_prefix="", metadata={"path": str(path.resolve())})
                )
            normalize.reset_mock()
            with patch.dict("sys.modules", modules):
                load_hunyuan_dit_clip(
                    paths, "native-target", model_options={"custom_operations": object()}
                )
            normalize.assert_not_called()

    def test_adapter_refuses_missing_wrong_type_or_wrong_shape_tokenizer_before_constructor(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, states, constructor, _, modules = self.fixtures(Path(folder))
            for proto in (
                None,
                "external/path",
                torch.ones(4),
                torch.zeros(2, 2, dtype=torch.uint8),
            ):
                states[1]["spiece_model"] = proto
                with (
                    patch.dict("sys.modules", modules),
                    self.assertRaisesRegex(ValueError, "embedded byte tokenizer"),
                ):
                    load_hunyuan_dit_clip(paths, "native-target")
            states[1]["spiece_model"] = tokenizer_proto()
            states[0]["bert.encoder.layer.0.attention.self.query.weight"] = torch.empty(
                768, 768, device="meta"
            )
            with (
                patch.dict("sys.modules", modules),
                self.assertRaisesRegex(ValueError, "dimensions"),
            ):
                load_hunyuan_dit_clip(paths, "native-target")
            constructor.assert_not_called()

    def test_adapter_native_cancellation_and_during_load_mutation_propagate_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, _, constructor, read, modules = self.fixtures(Path(folder))
            modules[
                "comfy.model_management"
            ].throw_exception_if_processing_interrupted.side_effect = CancelledError("cancelled")
            with patch.dict("sys.modules", modules), self.assertRaises(CancelledError):
                load_hunyuan_dit_clip(paths, "native-target")
            read.assert_not_called()
            modules[
                "comfy.model_management"
            ].throw_exception_if_processing_interrupted.side_effect = None
            constructor.side_effect = lambda *args, **kwargs: (
                paths[0].write_bytes(b"mutated"),
                SimpleNamespace(patcher=SimpleNamespace()),
            )[1]
            with (
                patch.dict("sys.modules", modules),
                self.assertRaisesRegex(ValueError, "changed after capture"),
            ):
                load_hunyuan_dit_clip(paths, "native-target")
            self.assertEqual(read.call_count, 2)

    def test_adapter_rejects_wrong_count_duplicate_roles_and_nonstate_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, states, constructor, read, modules = self.fixtures(Path(folder))
            for invalid in (paths[:1], paths + paths[:1], [paths[0], paths[0]]):
                with patch.dict("sys.modules", modules), self.assertRaises(ValueError):
                    load_hunyuan_dit_clip(invalid, "native-target")
            read.assert_not_called()
            states[1] = {}
            with (
                patch.dict("sys.modules", modules),
                self.assertRaisesRegex(ValueError, "one Chinese RoBERTa"),
            ):
                load_hunyuan_dit_clip(paths, "native-target")
            read.side_effect = lambda *args, **kwargs: ("not a tensor state dictionary", {})
            with (
                patch.dict("sys.modules", modules),
                self.assertRaisesRegex(ValueError, "state dictionaries"),
            ):
                load_hunyuan_dit_clip(paths, "native-target")
            constructor.assert_not_called()

    def test_current_request_cancellation_refuses_before_any_native_read(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, _, constructor, read, modules = self.fixtures(Path(folder))
            job = SimpleNamespace(cancelled=SimpleNamespace(is_set=lambda: True))
            with (
                patch.dict("sys.modules", modules),
                patch("ruinedfooocus.app.resource_lifecycle.current_job", return_value=job),
                self.assertRaises(CancelledError),
            ):
                load_hunyuan_dit_clip(paths, "native-target")
            read.assert_not_called()
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
