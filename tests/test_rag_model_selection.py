"""Three pinned encoders share one immutable Chat retrieval/preparation contract."""

import asyncio
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
import numpy as np

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.core.settings_schema import validate_known
from ruinedfooocus.features.chat import rag_preparation as rag, ui_rag_settings as ui
from ruinedfooocus.features.chat.model_preparation import ModelPreparation, capture_chat_request


class RagModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(
            patch.dict(os.environ, {"RF_DATA_DIR": str(self.root), "RF_OFFLINE": "0"})
        )
        self.snapshot = self.enterContext(
            patch.object(
                rag, "snapshot_download", side_effect=rag.LocalEntryNotFoundError("not cached")
            )
        )
        self.launch = SimpleNamespace(cpu=False, offline=False, directml=None)

    def complete(self, prepared):
        artifacts = []
        for item in prepared.artifacts:
            content = item.destination.name.encode()
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            item.destination.write_bytes(content)
            artifacts.append(
                replace(
                    item, expected_size=len(content), sha256=hashlib.sha256(content).hexdigest()
                )
            )
        return replace(prepared, artifacts=tuple(artifacts))

    def test_all_three_manifests_pin_native_safe_closures_and_keep_default_location(self):
        self.assertEqual(len(rag.model_specs()), 3)
        for spec in rag.model_specs():
            with self.subTest(model=spec.model_id):
                self.assertEqual(
                    validate_known({"rag_embedding_model": spec.model_id}),
                    {"rag_embedding_model": spec.model_id},
                )
                prepared = rag.capture_embedding_model(
                    {"rag_embedding_model": spec.model_id}, launch_args=self.launch
                )
                self.assertEqual(prepared.settings["encoder"], spec.identity)
                self.assertEqual(len(prepared.artifacts), 8)
                self.assertTrue(all(spec.revision in item.url for item in prepared.artifacts))
                self.assertTrue(prepared.model.is_relative_to(self.root / "models/rag_embeddings"))
                self.assertTrue(prepared.settings["gpu"])
                self.assertEqual(spec.identity["max_length"], 512)
                self.assertEqual(spec.identity["pooling"], "mean")
                self.assertTrue(spec.identity["normalize"])
                self.assertEqual(spec.identity["dimensions"], 384)
        default = rag.capture_embedding_model({}, launch_args=self.launch)
        self.assertEqual(default.model, self.root / "models/rag_embeddings" / rag.MODEL_REVISION)
        with self.assertRaisesRegex(ValueError, "listed option"):
            validate_known({"rag_embedding_model": "arbitrary/remote-code"})
        self.snapshot.reset_mock()
        with self.assertRaisesRegex(ValueError, "listed"):
            rag.capture_rag([["text", "document"]], model_id="arbitrary/remote-code")
        self.snapshot.assert_not_called()

    def test_actual_chat_capture_freezes_selected_model_policy_before_settings_change(self):
        selected = rag.RAG_EMBEDDING_MODELS[1]
        values = {"rag_embedding_model": selected, "rag_embedding_device": "CPU"}
        chat = ModelPreparation(self.root / "chat.gguf", (), "{}")
        result = capture_chat_request(
            {"model": str(chat.model), "embed": '[["text","document"]]', "_chat_preparation": chat},
            values,
            SimpleNamespace(),
            {},
            launch_args=self.launch,
        )
        prepared = result["_rag_preparation"]
        values.update(rag_embedding_model=rag.MODEL_ID, rag_embedding_device="Auto")
        self.assertEqual(prepared.settings["encoder"], rag.model_spec(selected).identity)
        self.assertFalse(prepared.settings["gpu"])
        detached = prepared.settings
        detached["encoder"]["configs"].clear()
        self.assertTrue(prepared.settings["encoder"]["configs"])
        self.snapshot.assert_called_with(
            selected, revision=rag.model_spec(selected).revision, local_files_only=True
        )

    def test_each_encoding_contract_field_invalidates_the_same_source_and_file_index(self):
        prepared = self.complete(
            rag.capture_rag([["text", "document"]], cache_root=self.root / "same-files")
        )
        original = json.loads(rag.index_identity([["text", "document"]], prepared))
        for field, value in {
            "model_id": rag.RAG_EMBEDDING_MODELS[1],
            "revision": "f" * 40,
            "dimensions": 768,
            "pooling": "other",
            "normalize": False,
            "max_length": 256,
            "configs": {},
            "manifest_sha256": "0" * 64,
        }.items():
            policy = deepcopy(prepared.settings)
            policy["encoder"][field] = value
            changed = replace(prepared, settings_json=json.dumps(policy))
            with self.subTest(field=field):
                self.assertNotEqual(
                    json.loads(rag.index_identity([["text", "document"]], changed)), original
                )
        options = rag.runtime_options(prepared)
        self.assertEqual(options["maxlength"], 512)
        self.assertEqual(options["vectors"], {"local_files_only": True, "trust_remote_code": False})

    def test_real_settings_model_event_prepares_the_selected_frozen_plan(self):
        register = Mock()
        settings = SimpleNamespace(default_settings={"rag_embedding_model": rag.MODEL_ID})
        with gr.Blocks(analytics_enabled=False) as app:
            with gr.Tab("Local Chat") as tab:
                ui.create_rag_settings(tab, settings, self.launch, register)
        self.addCleanup(app.close)
        selected = rag.RAG_EMBEDDING_MODELS[2]
        plan = self.complete(
            rag.capture_embedding_model({"rag_embedding_model": selected}, launch_args=self.launch)
        )
        events = {fn.fn.__name__: fn for fn in app.fns.values()}
        state = SessionState(app)
        request = gr.Request(session_hash="rag-model-selection")
        with (
            patch.object(rag, "capture_embedding_model", return_value=plan) as capture,
            patch.object(rag, "prepare_rag", return_value=plan) as prepare,
            patch("ruinedfooocus.app.resource_lifecycle.service_pipeline", return_value=None),
        ):
            result = asyncio.run(
                app.process_api(
                    events["refresh_model"], [None, selected], state=state, request=request
                )
            )
            self.assertIn(selected, result["data"][0])
            prepare.assert_not_called()
            running = asyncio.run(
                app.process_api(
                    events["prepare_model"], [None, selected], state=state, request=request
                )
            )
            self.assertFalse(running["data"][-1]["interactive"])
            finished = asyncio.run(
                app.process_api(
                    events["prepare_model"],
                    [None, selected],
                    state=state,
                    request=request,
                    iterator=running["iterator"],
                )
            )
            self.assertTrue(finished["data"][-1]["interactive"])
            prepare.assert_called_once()
            self.assertIs(prepare.call_args.args[0], plan)
            self.assertTrue(
                all(
                    call.args[0]["rag_embedding_model"] == selected
                    for call in capture.call_args_list
                )
            )
            self.assertEqual(settings.default_settings["rag_embedding_model"], rag.MODEL_ID)

    def test_native_factory_keeps_512_instead_of_deriving_shorter_reference_length(self):
        native = Path(importlib.util.find_spec("txtai").origin).parent
        path = native / "models/pooling/factory.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        create = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "create"
        )
        create.decorator_list = []
        pooled = Mock()
        derive = Mock(side_effect=AssertionError("Must not opt into the shorter ST recommendation"))
        scope = dict(
            os=os,
            PoolingFactory=SimpleNamespace(maxlength=derive, method=lambda path: "meanpooling"),
            MeanPooling=pooled,
        )
        exec(compile(ast.Module(body=[create], type_ignores=[]), str(path), "exec"), scope)
        for length in (None, 512):
            scope["create"](dict(path=str(self.root), maxlength=length))
            self.assertEqual(pooled.call_args.args[3], length)
        derive.assert_not_called()
        base = native / "models/pooling/base.py"
        assigned = next(
            node
            for node in ast.walk(ast.parse(base.read_text(encoding="utf-8")))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "maxlength"
                for target in node.targets
            )
        )
        for length in (None, 512):
            owner = SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=512))
            exec(
                compile(ast.Module(body=[assigned], type_ignores=[]), str(base), "exec"),
                dict(self=owner, maxlength=length),
            )
            self.assertEqual(owner.maxlength, 512)

    def test_native_l2_normalization_is_shared_for_query_and_document_vectors(self):
        source = Path(importlib.util.find_spec("txtai").origin).parent / "vectors/base.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        normalize = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "normalize"
        )
        normalize.decorator_list = []
        scope = dict(np=np)
        exec(compile(ast.Module(body=[normalize], type_ignores=[]), str(source), "exec"), scope)
        query = scope["normalize"](None, np.array([[3.0, 4.0]], dtype=np.float32))
        document = scope["normalize"](None, np.array([[6.0, 8.0]], dtype=np.float32))
        np.testing.assert_allclose(query, document)
        np.testing.assert_allclose(np.linalg.norm(query, axis=1), [1.0])
        callers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node is not normalize
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "normalize"
                for call in ast.walk(node)
            )
        ]
        self.assertTrue(callers, "Native vector encoding must retain normalization")
