"""CPU evidence for strict native construction and honest execution partitions."""

import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ruinedfooocus.backends.onnx import execution, model_contracts, onnx_runtime
from ruinedfooocus.backends.onnx.onnx_worker import CANARY
from ruinedfooocus.app import resource_lifecycle as life


class ExecutionTests(unittest.TestCase):
    def test_global_cpu_short_circuits_provider_discovery(self):
        import onnxruntime as ort

        with (
            patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.app.arguments": SimpleNamespace(
                        args=SimpleNamespace(cpu=True, directml=None)
                    )
                },
            ),
            patch.object(
                ort,
                "get_available_providers",
                side_effect=AssertionError("No provider discovery on explicit CPU"),
            ),
            patch.object(onnx_runtime, "create_mask_session") as cpu,
        ):
            onnx_runtime.new_session(preference="Auto")
        cpu.assert_called_once_with(providers=["CPUExecutionProvider"])

    def test_unsupported_real_factory_cannot_bypass_strict_gpu_construction(self):
        factory = Mock()
        with patch.dict("sys.modules", {"rembg": SimpleNamespace(new_session=factory)}):
            with self.assertRaisesRegex(RuntimeError, "no audited strict-construction API"):
                model_contracts.create_mask_session(providers=["CUDAExecutionProvider"])
        factory.assert_not_called()

    def test_native_model_methods_options_and_globals_survive_strict_construction(self):
        import onnxruntime as ort
        import rembg

        native = next(
            cls
            for cls in rembg.new_session.__globals__["sessions_class"]
            if cls.name() == "bria-rmbg"
        )
        constructor, factory, predict = native.__init__, rembg.new_session, native.predict
        options = ort.SessionOptions()
        options.inter_op_num_threads = 7
        inner = Mock()
        with (
            patch.object(native, "download_models", return_value=Path("fixture.onnx")) as download,
            patch.object(ort, "InferenceSession", return_value=inner) as create,
            patch.dict("os.environ", {"OMP_NUM_THREADS": "2"}),
        ):
            result = model_contracts.create_mask_session(sess_opts=options, providers=[])
        create.assert_called_once_with(
            "fixture.onnx", sess_options=options, providers=[], enable_fallback=False
        )
        inner.disable_fallback.assert_called_once_with()
        self.assertIsInstance(result, native)
        self.assertEqual(result.model_name, "bria-rmbg")
        self.assertIs(result.predict.__func__, predict)
        self.assertIs(native.__init__, constructor)
        self.assertIs(rembg.new_session, factory)
        self.assertEqual(options.inter_op_num_threads, 7)
        self.assertEqual(options.intra_op_num_threads, 2)
        download.assert_called_once_with()

    def test_initialization_failure_has_no_hidden_constructor_retry(self):
        import onnxruntime as ort
        import rembg

        native = next(
            cls
            for cls in rembg.new_session.__globals__["sessions_class"]
            if cls.name() == "bria-rmbg"
        )
        with (
            patch.object(native, "download_models", return_value="fixture.onnx"),
            patch.object(
                ort, "InferenceSession", side_effect=RuntimeError("EP init failed")
            ) as create,
        ):
            with self.assertRaisesRegex(RuntimeError, "EP init failed"):
                model_contracts.create_mask_session(providers=["CUDAExecutionProvider"])
        self.assertEqual(create.call_count, 1)

    def test_real_cpu_canary_profile_reports_executed_nodes(self):
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        with execution.profile_options(options):
            session = ort.InferenceSession(
                base64.b64decode(CANARY),
                sess_options=options,
                providers=["CPUExecutionProvider"],
                enable_fallback=False,
            )
            session.disable_fallback()
            graph = execution.graph_assignment(session, "CPUExecutionProvider")
            session.run(None, {"x": np.ones((2, 2), dtype=np.float32)})
            evidence = execution.assignment(session, "CPUExecutionProvider")
        self.assertGreater(evidence["nodes"]["CPUExecutionProvider"], 0)
        self.assertGreater(graph["nodes"]["CPUExecutionProvider"], 0)
        self.assertFalse(evidence["all_graph_nodes"])

    def profile(self, providers):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        filename = Path(temporary.name) / "profile.json"
        filename.write_text(
            json.dumps([{"cat": "Node", "args": {"provider": p}} for p in providers])
        )
        return SimpleNamespace(end_profiling=lambda: str(filename))

    def test_no_gpu_assignment_is_failure_and_cpu_partition_is_not_hidden_retry(self):
        with self.assertRaisesRegex(execution.AssignmentError, "No optimized graph nodes"):
            execution.graph_assignment(
                SimpleNamespace(
                    get_provider_graph_assignment_info=lambda: [
                        SimpleNamespace(ep_name="CPUExecutionProvider", get_nodes=lambda: ["node"])
                    ]
                ),
                "CUDAExecutionProvider",
            )
        with self.assertRaisesRegex(execution.AssignmentError, "No executed nodes"):
            execution.assignment(self.profile(["CPUExecutionProvider"]), "CUDAExecutionProvider")
        evidence = execution.assignment(
            self.profile(["CPUExecutionProvider", "CUDAExecutionProvider"]), "CUDAExecutionProvider"
        )
        self.assertEqual(evidence["cpu_partitions"], 1)
        self.assertEqual(evidence["nodes"]["CUDAExecutionProvider"], 1)
        self.assertIn("CPU partitions: 1", execution.assignment_status(evidence))

    def test_native_run_fallback_is_disabled_before_predict_and_recovery_is_visible(self):
        inner, gpu, cpu = Mock(), Mock(), Mock()
        gpu.inner_session = inner
        inner.get_providers.return_value = ["CUDAExecutionProvider"]
        calls = []
        inner.disable_fallback.side_effect = lambda: calls.append("disabled")
        gpu.predict.side_effect = lambda *a: calls.append("predict") or ["mask"]
        session = onnx_runtime.NativeMaskSession(lambda: gpu, "CUDAExecutionProvider")
        item = life.JobRegistry().create()
        with (
            life.job_scope(item),
            life.admission.lease(item),
            patch.object(onnx_runtime, "release_torch_weights"),
            patch.object(
                onnx_runtime,
                "finish_assignment",
                side_effect=execution.AssignmentError("no assigned GPU nodes"),
            ),
            patch.object(session, "_cpu_session", return_value=cpu),
        ):
            session.predict(Image.new("RGB", (2, 2)))
            session.predict(Image.new("RGB", (2, 2)))
        self.assertEqual(calls, ["disabled", "predict"])
        self.assertTrue(session.disabled)
        self.assertIn("CPU recovery:", session.runtime_status)
        self.assertEqual(cpu.predict.call_count, 2)

    def test_cache_identity_requires_complete_contract_and_changes_with_every_boundary(self):
        contract = dict(
            model_sha256="model",
            preprocessing="native-rembg-2.0.87",
            shape=[1, 3, 1024, 1024],
            dtype="float32",
            provider="plugin",
            runtime="ort",
            device="ep-id",
            driver="driver",
            options={},
        )
        key = execution.cache_identity(**contract)
        self.assertIsNotNone(key)
        for field in contract:
            incomplete = dict(contract)
            incomplete.pop(field)
            self.assertIsNone(execution.cache_identity(**incomplete))
            changed = dict(contract, **{field: "different"})
            self.assertNotEqual(key, execution.cache_identity(**changed))


if __name__ == "__main__":
    unittest.main()
