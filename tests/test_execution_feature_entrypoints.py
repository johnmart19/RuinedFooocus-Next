"""Real worker/helper and XLlamacpp entrypoints; external native work is inert."""

from contextlib import ExitStack
from importlib import machinery, metadata, util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_chat_feature_entrypoints import tiny_gguf


class InertXllamaServer:
    def __init__(self, params):
        self.params = params

    def handle_chat_completions(self, parameters, callback):
        callback(
            {
                "choices": [
                    {
                        "delta": {"content": '{"addition":"soft diffuse light"}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


class ExecutionFeatureEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        # Closures expose complete pinned exports beside their thin AST fixtures.
        native = Path(os.environ.get("RF_TEST_COMFY_SOURCE", root / "repositories" / "ComfyUI"))
        gguf = Path(
            os.environ.get("RF_TEST_GGUF_SOURCE", root / "repositories" / "molbal_comfyui_gguf")
        )
        if not (native / "comfy" / "options.py").is_file():
            raise AssertionError(f"Complete pinned ComfyUI source required: {native}")
        if not (gguf / "__init__.py").is_file():
            raise AssertionError(f"Complete pinned ComfyUI-GGUF source required: {gguf}")
        sys.path.insert(0, str(native))
        from comfy.cli_args import args
        from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management

        args.cpu = True
        initialize_cpu_model_management()
        spec = util.spec_from_file_location(
            "molbal_comfyui_gguf",
            gguf / "__init__.py",
            submodule_search_locations=[str(gguf)],
        )
        module = util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # No RF module replacement: these imports must resolve the complete real closure.
        from ruinedfooocus.app import api_runtime, async_worker, resource_lifecycle, state
        from ruinedfooocus.features.chat import llama_pipeline
        from ruinedfooocus.features.prompting import local_model

        cls.api, cls.worker, cls.life, cls.shared = (
            api_runtime,
            async_worker,
            resource_lifecycle,
            state,
        )
        cls.pipeline, cls.helpers = llama_pipeline, local_model

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.model = tiny_gguf(self.folder / "request.gguf")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.dict(os.environ, RF_OFFLINE="1", RF_DATA_DIR=str(self.folder))
        )
        self.stack.enter_context(
            patch.dict(
                self.shared.settings.default_settings,
                {
                    "llama_localfile": str(self.model),
                    "llama_mmproj": "None",
                    "llm_runtime": "xllamacpp",
                    "cpu_everywhere": True,
                    "xllamacpp_backend": "CPU",
                    "llm_n_ctx": 512,
                    "llm_hp_max_tokens": 32,
                },
                clear=True,
            )
        )
        self.stack.enter_context(
            patch.dict(self.shared.path_manager.paths, path_llm=[str(self.folder)], path_vision=[])
        )
        self.engine = ModuleType("xllamacpp")
        self.engine.__spec__ = machinery.ModuleSpec("xllamacpp", loader=None)
        self.engine.CommonParams = lambda: SimpleNamespace(
            model=SimpleNamespace(path=""), mmproj=SimpleNamespace(path="")
        )
        self.engine.Server = InertXllamaServer
        self.engine.get_device_info = Mock(side_effect=AssertionError("CPU must not query GPUs"))
        self.stack.enter_context(patch.dict(sys.modules, xllamacpp=self.engine))

        # Actual installer identity/selection code reads an owned external-wheel RECORD.
        record = self.folder / "xllamacpp-2026.10.11870.dist-info" / "RECORD"
        record.parent.mkdir()
        record.write_text("", encoding="utf-8")
        distribution = SimpleNamespace(
            version="2026.10.11870", locate_file=lambda path: self.folder / path
        )
        entry = metadata.PackagePath("xllamacpp-2026.10.11870.dist-info/RECORD")
        entry.dist, entry.hash = distribution, None
        distribution.files = [entry]
        original = metadata.distribution
        self.stack.enter_context(
            patch.object(
                metadata,
                "distribution",
                side_effect=lambda name: distribution if name == "xllamacpp" else original(name),
            )
        )
        from ruinedfooocus.packaging import gpu_installer

        self.installer = gpu_installer
        gpu_installer.record_xllamacpp_backend("https://pypi.org/simple")

    def test_prompt_helper_reaches_actual_chat_worker_bindings(self):
        prepared = self.helpers.capture_helper(
            self.shared.settings.default_settings, self.shared.path_manager
        )
        self.assertEqual(prepared.model, self.model.resolve())
        data = {self.helpers.PREPARATION_KEY: prepared}
        self.helpers.prepare_prompt_data(data, cancelled=threading.Event())
        prompt = "Two blue cups, literal 0007."

        def operation(job, task_id):
            self.assertEqual(self.life.current_job().owner, job.owner)
            return self.helpers.run_prompt(prompt, instruction="Preserve the supplied scene.")

        operation.owner, operation.resource_group = ("api", "entrypoint-helper"), "chat"
        job = self.api.Job(operation)
        item = self.life.registry.create(owner=job.owner, cancelled=job.cancelled)
        try:
            with (
                patch.object(self.worker, "worker_thread", threading.current_thread()),
                self.life.job_scope(item),
                self.life.service_scope("chat", self.shared.state, item=item),
                self.helpers.prompt_preparation_scope(item, data),
            ):
                job.run(item.task_id)
            self.assertEqual(
                self.api.wait_for_result(job, timeout=3), prompt + "\nsoft diffuse light"
            )
            self.assertNotIn(self.helpers.PREPARATION_KEY, item.state)
            self.assertFalse(self.life.admission.held)
        finally:
            self.life.registry.finish(item)
        self.engine.get_device_info.assert_not_called()

    def test_xllamacpp_selection_reaches_declared_installer(self):
        prepared = self.helpers.capture_helper(
            self.shared.settings.default_settings, self.shared.path_manager
        )
        item = self.life.registry.create(owner=("api", "entrypoint-load"))
        pipeline = self.pipeline.pipeline()
        try:
            with (
                self.life.job_scope(item),
                self.life.service_scope("chat", self.shared.state, item=item),
            ):
                pipeline.load_base_model(str(self.model), prepared=prepared)
                self.assertEqual(pipeline.runtime_name, "xllamacpp")
                self.assertEqual(pipeline.runtime_label, "CPU")
                params = pipeline.llm.client.params
                self.assertEqual(Path(params.model.path).resolve(), self.model.resolve())
                self.assertEqual(params.n_gpu_layers, 0)
                self.assertTrue(params.no_kv_offload)
                self.assertFalse(params.mmproj_use_gpu)
        finally:
            pipeline.unload()
            self.life.registry.finish(item)
        self.engine.get_device_info.assert_not_called()

    def test_missing_xllamacpp_is_rejected_offline_before_acquisition(self):
        original = metadata.distribution

        def missing(name):
            if name == "xllamacpp":
                raise metadata.PackageNotFoundError(name)
            return original(name)

        with patch.object(metadata, "distribution", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "XLlamacpp is missing"):
                self.installer.ensure_xllamacpp("cpu", platform.system(), choice="CPU")
        self.engine.get_device_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
