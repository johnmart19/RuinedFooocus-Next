"""Pinned embedding settings, explicit preparation and the native CPU resolver."""

import ast
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.core.settings_schema import validate_known
from ruinedfooocus.features.chat import rag_preparation as rag
from ruinedfooocus.features.chat.model_preparation import ModelPreparation
from ruinedfooocus.features.chat import ui_rag_settings as ui


class RagSettingsTests(unittest.TestCase):
    def test_cpu_override_and_global_policy_are_captured_without_device_probe(self):
        settings = {"rag_embedding_device": "CPU"}
        self.assertEqual(validate_known(settings), settings)
        launch = SimpleNamespace(cpu=False, offline=False, directml=None)
        with (
            patch.object(
                rag, "snapshot_download", side_effect=rag.LocalEntryNotFoundError("local only")
            ),
            patch.dict("os.environ", {"RF_OFFLINE": "0"}),
        ):
            captured = rag.capture_embedding_model(settings, launch_args=launch)
            settings["rag_embedding_device"] = "Auto"
            automatic = rag.capture_embedding_model(settings, launch_args=launch)
        self.assertEqual(captured.settings, {"gpu": False, "offline": False})
        self.assertEqual(automatic.settings, {"gpu": True, "offline": False})
        self.assertEqual(
            rag.embedding_device_policy(settings | {"cpu_everywhere": True}, launch),
            ("CPU", "global"),
        )
        self.assertEqual(
            rag.embedding_device_policy(
                settings, SimpleNamespace(cpu=True, _compute_status={"cli": ["--cpu"]})
            ),
            ("CPU", "global"),
        )
        self.assertEqual(
            rag.embedding_device_policy(settings, SimpleNamespace(cpu=False, directml=0)),
            ("CPU", "directml"),
        )
        with self.assertRaisesRegex(ValueError, "listed option"):
            validate_known({"rag_embedding_device": "GPU"})

    def test_cpu_placement_short_circuits_installed_txtai_accelerator_probe(self):
        source = Path(importlib.util.find_spec("txtai").origin).parent / "models/models.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        device = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "deviceid"
        )
        device.decorator_list = []
        probes = Mock(side_effect=AssertionError("CPU placement must not probe an accelerator"))
        from torch import device as torch_device

        scope = dict(
            torch=SimpleNamespace(device=torch_device),
            Models=SimpleNamespace(hasaccelerator=probes),
        )
        exec(compile(ast.Module(body=[device], type_ignores=[]), str(source), "exec"), scope)
        with tempfile.TemporaryDirectory() as directory:
            prepared = ModelPreparation(Path(directory), (), '{"gpu":false,"offline":true}')
            options = rag.runtime_options(prepared)
        self.assertIsNone(options["gpu"])
        self.assertEqual(scope["deviceid"](options["gpu"]), -1)
        probes.assert_not_called()

    def test_status_checks_size_only_and_index_identity_keeps_placement_and_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = tuple(PreparationRequest("", root / f"file-{i}", 3) for i in range(3))
            files[0].destination.write_bytes(b"yes")
            files[1].destination.write_bytes(b"no")
            prepared = ModelPreparation(root, files, '{"gpu":false,"offline":true}')
            with patch.object(
                Path, "open", side_effect=AssertionError("Passive status opened a file")
            ):
                self.assertEqual(ui.embedding_file_status(prepared), (1, 1))
            files[1].destination.write_bytes(b"yes")
            files[2].destination.write_bytes(b"yes")
            source = [["text", "owned fixture"]]
            cpu = json.loads(rag.index_identity(source, prepared))
            automatic = json.loads(
                rag.index_identity(source, replace(prepared, settings_json='{"gpu":true}'))
            )
        self.assertEqual(cpu["dimensions"], rag.MODEL_DIMENSIONS)
        self.assertFalse(cpu["gpu"])
        self.assertTrue(automatic["gpu"])

    def test_owner_preparation_cancel_is_local_and_session_is_reusable(self):
        first = ui.EmbeddingPreparationSession()
        second = deepcopy(first)
        with first.preparing() as one, second.preparing() as two:
            with self.assertRaisesRegex(ValueError, "already running"):
                with first.preparing():
                    self.fail("Duplicate preparation")
            first.cancel()
            self.assertTrue(one.is_set())
            self.assertFalse(two.is_set())
            self.assertTrue(first.running)
        self.assertFalse(first.running)
        with first.preparing() as again:
            self.assertFalse(again.is_set())

    def test_observed_device_comes_only_from_an_existing_txtai_index(self):
        import torch

        self.assertIsNone(ui.initialized_device(None))
        self.assertIsNone(ui.initialized_device(SimpleNamespace(embeddings=object())))
        index = SimpleNamespace(
            model=SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
        )
        self.assertEqual(ui.initialized_device(SimpleNamespace(embeddings=index)), "cpu")

    def test_rag_controls_register_one_real_preference_and_private_events(self):
        register = Mock()
        with gr.Blocks(analytics_enabled=False) as app:
            with gr.Tab("Local Chat") as tab:
                ui.create_rag_settings(
                    tab, SimpleNamespace(default_settings={}), SimpleNamespace(), register
                )
        register.assert_called_once()
        self.assertEqual(register.call_args.args[0], "rag_embedding_device")
        radio = register.call_args.args[1]
        self.assertEqual([choice[1] for choice in radio.choices], ["Auto", "CPU"])
        events = app.get_config_file()["dependencies"]
        self.assertEqual(len(events), 4)
        self.assertTrue(all(event["api_visibility"] == "private" for event in events))

    def test_settings_status_is_passive_and_prepare_uses_the_same_captured_model(self):
        launch = SimpleNamespace(cpu=False, directml=None)
        settings = SimpleNamespace(default_settings={"rag_embedding_device": "CPU"})
        with gr.Blocks(analytics_enabled=False) as app:
            with gr.Tab("Local Chat") as tab:
                ui.create_rag_settings(tab, settings, launch, Mock())
        callbacks = {function.fn.__name__: function.fn for function in app.fns.values()}
        session = ui.EmbeddingPreparationSession()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_file = PreparationRequest("", root / "model.safetensors", 3)
            offline = ModelPreparation(root, (model_file,), '{"gpu":false,"offline":true}')
            online = replace(offline, settings_json='{"gpu":false,"offline":false}')
            with (
                patch.object(rag, "capture_embedding_model", side_effect=[offline, online, online]),
                patch.object(rag, "prepare_rag", return_value=online) as prepare,
            ):
                status, placement, button, _ = callbacks["refresh_status"](session, gr.Request())
                self.assertIn("Offline mode", status)
                self.assertIn("CPU", placement)
                self.assertFalse(button["interactive"])
                prepare.assert_not_called()

                progress = callbacks["prepare_model"](session, gr.Request())
                self.assertTrue(next(progress)[-1]["interactive"])
                callbacks["cancel"](session)
                self.assertIn("cancelled", next(progress)[0])
                self.assertFalse(session.running)
                self.assertIs(prepare.call_args.args[0], online)
                self.assertTrue(prepare.call_args.kwargs["cancelled"].is_set())
                with self.assertRaises(StopIteration):
                    next(progress)
