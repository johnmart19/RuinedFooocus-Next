"""Execute changed native functions without loading accelerators or weights."""

import ast
import builtins
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.backends.comfy.cpu_startup import (
    cpu_discovery_disabled,
    initialize_cpu_model_management,
)


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def native_node(path, kind, name, scope):
    tree = ast.parse((UPSTREAM / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, kind) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class NativeUpdateTests(unittest.TestCase):
    def combined(self, quant_format):
        quantized = type("QuantizedTensor", (), {})
        mixed = type("MixedPrecisionOp", (), {})
        gather = Mock(return_value=object())
        scope = dict(
            QuantizedTensor=quantized,
            MixedPrecisionOp=mixed,
            comfy=SimpleNamespace(memory_management=SimpleNamespace(cat_gathered=gather)),
            comfy_aimdo=SimpleNamespace(torch=SimpleNamespace(aimdo_to_tensor=Mock())),
            run_every_op=lambda: None,
        )
        cls = native_node(Path("comfy/ops.py"), ast.ClassDef, "CombinedLinear", scope)
        linears = [mixed(), mixed()]
        for index, linear in enumerate(linears):
            linear._v_weight = quantized()
            linear._v = (None, 256 * index, 256)
            linear.quant_format = quant_format
            linear._full_precision_mm = False
            linear.comfy_force_forward = False
            linear.bias = None
        return cls(linears), gather

    def test_fp8_stays_in_the_native_split_path_without_consuming_resident_weights(self):
        for quant_format in ("float8_e4m3fn", "float8_e5m2", "unknown"):
            with self.subTest(quant_format=quant_format):
                combined, gather = self.combined(quant_format)
                weights = [linear._v_weight for linear in combined.linears]
                combined.prepare(torch.device("cpu"))
                gather.assert_not_called()
                self.assertIs(combined.forward(torch.ones(1, 2)), NotImplemented)
                self.assertEqual([s._v_weight for s in combined.linears], weights)

    def test_supported_int8_combination_and_live_full_precision_flag_are_preserved(self):
        for quant_format in ("int8_tensorwise", "asym_w4a8_int8", "w6a8_int8"):
            with self.subTest(quant_format=quant_format):
                combined, gather = self.combined(quant_format)
                combined.prepare(torch.device("cpu"))
                gather.assert_called_once()
                self.assertIs(combined._v_weight, gather.return_value)
                self.assertTrue(all(s._v_weight is None for s in combined.linears))
                self.assertFalse(combined._full_precision_mm)
                combined.linears[0]._full_precision_mm = True
                self.assertTrue(combined._full_precision_mm)

    def test_zero_size_pin_never_allocates_registers_or_changes_pin_budget(self):
        management = SimpleNamespace(DISABLE_PINNED_MEMORY=False)
        memory = SimpleNamespace(vram_aligned_size=Mock(return_value=0))
        scope = dict(
            get_pin=lambda *args: None,
            comfy=SimpleNamespace(model_management=management, memory_management=memory),
        )
        run = native_node(Path("comfy/pinned_memory.py"), ast.FunctionDef, "pin_memory", scope)
        for size in (None, 0):
            hostbuf, stack, split, pinned, counter, buckets = Mock(), [], [0], [0], [0], {}
            module = SimpleNamespace(
                _pin_state={"weights": (hostbuf, stack, split, pinned, counter, buckets)},
                weight=None,
                bias=None,
            )
            self.assertIsNone(run(module, size=size))
            hostbuf.extend.assert_not_called()
            self.assertEqual((stack, split, pinned, counter, buckets), ([], [0], [0], [0], {}))


class NativeCpuDiscoveryTests(unittest.TestCase):
    def test_cpu_intent_is_required_and_an_existing_gpu_manager_requires_restart(self):
        for cpu in (False, True):
            cli = SimpleNamespace(args=SimpleNamespace(cpu=cpu))
            manager = SimpleNamespace(cpu_state="gpu", CPUState=SimpleNamespace(CPU="cpu"))
            original = torch.cuda.is_available
            with (
                patch.dict(sys.modules, {"comfy.cli_args": cli}),
                patch(
                    "ruinedfooocus.backends.comfy.cpu_startup.importlib.import_module",
                    side_effect=[SimpleNamespace(), manager],
                ) as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "restart" if cpu else "finalized CPU"):
                    initialize_cpu_model_management()
                self.assertEqual(load.call_count, 2 if cpu else 0)
                self.assertIs(torch.cuda.is_available, original)

    def test_complete_native_cpu_import_is_probe_free_in_an_isolated_process(self):
        script = """
import sys
import os
from pathlib import Path
upstream = Path(os.environ.get('RF_TEST_COMFY_SOURCE', Path.cwd() / 'repositories/ComfyUI'))
sys.path.insert(0, str(upstream))
sys.argv = [sys.argv[0], '--cpu']
import torch
from functools import wraps
def forbidden(*args, **kwargs):
    raise AssertionError('CPU import attempted accelerator discovery')
for backend in (torch.cuda, torch.xpu):
    for name in ('_lazy_init', 'init', 'is_available', 'device_count', 'current_device',
                 'get_device_properties', 'get_device_name', 'get_device_capability', 'mem_get_info'):
        if hasattr(backend, name):
            setattr(backend, name, forbidden)
@wraps(torch.backends.mps.is_available)
def no_mps():
    return forbidden()
torch.backends.mps.is_available = no_mps
from comfy.cli_args import args
args.cpu = True
from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management
management = initialize_cpu_model_management()
assert management.cpu_state == management.CPUState.CPU
assert management.get_torch_device() == torch.device('cpu')
assert not management.xpu_available and not management.npu_available and not management.mlu_available
assert torch.cuda.is_available is forbidden and torch.xpu.device_count is forbidden
from torch._dynamo.variables.torch import constant_fold_functions
assert torch.cuda.is_available in constant_fold_functions
assert not torch.cuda.is_initialized()
assert (torch.tensor([2.0]) * 3).item() == 6

# Execute upstream's own FP8 split-forward regression on actual CPU tensors.
# Only the native residency address boundary is mocked, as in upstream's test.
import ast
from contextlib import ExitStack
from unittest.mock import patch
import comfy.ops
import comfy.lora
import comfy.model_patcher
import comfy.model_prefetch
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor
from ruinedfooocus.backends.comfy.cpu_startup import cpu_discovery_disabled
source = upstream / 'tests-unit/comfy_test/combined_linear_test.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ('_gathered', 'test_combined_linear_preserves_fp8_split_forward')]
for node in nodes:
    node.decorator_list = []
scope = dict(torch=torch, comfy=comfy, QUANT_ALGOS=QUANT_ALGOS, QuantizedTensor=QuantizedTensor)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
for quant_format in ('float8_e4m3fn', 'float8_e5m2'):
    # Kitchen lazily queries hardware for custom-op dispatch even on CPU input.
    # Numerical checks run inside CPU isolation too; this is separate evidence
    # from the completed native startup import above, whose guards are restored.
    with cpu_discovery_disabled(), ExitStack() as restore:
        class MonkeyPatch:
            def setattr(self, owner, name, value):
                restore.enter_context(patch.object(owner, name, value))
        scope['test_combined_linear_preserves_fp8_split_forward'](MonkeyPatch(), quant_format)
assert not torch.cuda.is_initialized()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def discovery(self):
        """Execute the actual early discovery region, before native args.cpu handling."""
        tree = ast.parse((UPSTREAM / "comfy/model_management.py").read_text(encoding="utf-8"))
        start = next(
            i
            for i, n in enumerate(tree.body)
            if isinstance(n, ast.Try) and "torch.xpu.device_count()" in ast.unparse(n)
        )
        end = next(
            i
            for i, n in enumerate(tree.body)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "args.cpu"
        )
        scope = dict(torch=torch, CPUState=SimpleNamespace(MPS="mps"))
        exec(
            compile(ast.Module(body=tree.body[start:end], type_ignores=[]), "discovery", "exec"),
            scope,
        )
        return scope

    def test_cpu_import_skips_native_optional_device_probes_and_restores_torch(self):
        forbidden = Mock(side_effect=AssertionError("Accelerator probe during CPU startup"))
        original_import = builtins.__import__
        extensions = []

        def track_import(name, *args, **kwargs):
            if name in ("torch_npu", "torch_mlu"):
                extensions.append(name)
            return original_import(name, *args, **kwargs)

        with (
            patch.object(torch.xpu, "device_count", forbidden),
            patch.object(torch.xpu, "is_available", forbidden),
            patch.object(torch.backends.mps, "is_available", forbidden),
            patch("builtins.__import__", track_import),
        ):
            with cpu_discovery_disabled():
                result = self.discovery()
                self.assertFalse(torch.cuda.is_available())
                self.assertEqual(torch.cuda.device_count(), 0)
                # CPU tensor execution is unaffected by discovery isolation.
                self.assertEqual((torch.tensor([2]) + 3).item(), 5)
            self.assertIs(torch.xpu.device_count, forbidden)
            self.assertIs(torch.backends.mps.is_available, forbidden)
        forbidden.assert_not_called()
        self.assertEqual(extensions, ["torch_npu", "torch_mlu"])
        for name in ("xpu_available", "npu_available", "mlu_available"):
            self.assertFalse(result[name])

    def test_failure_restores_cached_extensions_and_all_temporary_functions(self):
        cached = SimpleNamespace()
        original_count = torch.xpu.device_count
        original_cuda = torch.cuda.is_available
        with patch.dict(sys.modules, {"torch_npu": cached}):
            before = {name: sys.modules.get(name) for name in ("torch_npu", "torch_mlu")}
            with self.assertRaisesRegex(RuntimeError, "import failed"):
                with cpu_discovery_disabled():
                    self.assertIsNone(sys.modules["torch_npu"])
                    raise RuntimeError("import failed")
            self.assertIs(sys.modules["torch_npu"], cached)
            self.assertEqual({name: sys.modules.get(name) for name in before}, before)
        self.assertIs(torch.xpu.device_count, original_count)
        self.assertIs(torch.cuda.is_available, original_cuda)


if __name__ == "__main__":
    unittest.main()
