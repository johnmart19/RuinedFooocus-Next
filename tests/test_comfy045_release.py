"""Exact release requirements and changed native CPU residency/logits contracts."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ruinedfooocus.packaging import package_plan as plans
from ruinedfooocus.packaging.runtime_support import RuntimeCandidate
from tools.audit_native_model_coverage import read_official_templates


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
PIN = "989d93d0d968e704c589ce0146bb18c77f6b50cc"


class Comfy045ReleaseTests(unittest.TestCase):
    def test_exact_release_pin_and_native_template_base_remain_one_contract(self):
        tree = ast.parse((ROOT / "ruinedfooocus/app/bootstrap.py").read_text(encoding="utf-8"))
        repositories = ast.literal_eval(
            next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "git_repos"
                    for target in node.targets
                )
            )
        )
        self.assertEqual(next(row["hash"] for row in repositories if row["name"] == "ComfyUI"), PIN)
        version = ast.parse((SOURCE / "comfyui_version.py").read_text(encoding="utf-8"))
        self.assertEqual(ast.literal_eval(version.body[0].value), "0.45.0")
        layer = plans.read_layer(
            SOURCE / "requirements.txt",
            identity="comfy-core",
            source="https://github.com/Comfy-Org/ComfyUI",
            revision=PIN,
            license="GPL-3.0",
        )
        self.assertIn("comfyui-workflow-templates==0.11.105", layer.requirements)
        self.assertIn("comfy-kitchen==0.2.41", layer.requirements)
        self.assertIn("comfy-aimdo==0.6.0", layer.requirements)
        self.assertFalse(
            any(
                line.startswith("comfyui-workflow-templates")
                for line in (ROOT / "pip/modules.txt").read_text(encoding="utf-8").splitlines()
            )
        )

    def test_actual_package_planner_consumes_the_release_base_without_installing(self):
        raw = (SOURCE / "requirements.txt").read_bytes()
        target = plans.current_target()
        candidate = RuntimeCandidate("cpu", "packages", (target.system,), device_kind="cpu")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            core = root / "repositories/ComfyUI"
            core.mkdir(parents=True)
            (core / "requirements.txt").write_bytes(raw)
            (root / "pip").mkdir()
            (root / "pip/modules.txt").write_text("", encoding="utf-8")
            ready = SimpleNamespace(ready=True)
            with (
                patch.object(plans.subprocess, "check_output", side_effect=[PIN, raw]),
                patch.object(plans, "resolve_packages", return_value=()) as resolve,
                patch.object(plans, "capability_decision", return_value={}),
                patch.object(plans, "plan_packages", return_value=ready) as combine,
                patch.object(
                    plans.subprocess, "run", side_effect=AssertionError("No package installer")
                ),
            ):
                self.assertIs(
                    plans.bootstrap_plan(root, PIN, "cpu", (), candidate=candidate), ready
                )
            layers = resolve.call_args.args[0]
            self.assertEqual(layers[0].identity, "comfy-core")
            self.assertEqual(layers[0].revision, PIN)
            self.assertEqual(layers[0].digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(layers[1].requirements, ())
            self.assertIn(
                "comfyui-workflow-templates==0.11.105", combine.call_args.args[2][0].requirements
            )

    def test_official_recipe_reader_keeps_reviewed_target_data_separate_from_live_packages(self):
        index = json.loads(
            (ROOT / "ruinedfooocus/resources/model_metadata/native_coverage/index.json").read_text()
        )
        contract = index["source"]["workflow_templates"]
        target = contract["reviewed_release_target"]
        self.assertEqual(target["packages"]["comfyui_workflow_templates"], "0.11.105")
        self.assertEqual(target["recipe_counts"]["recipes"], 672)
        self.assertTrue(target["recipe_assets_match_manifest"])
        self.assertEqual(
            {row["name"] for row in contract["prepared_data_changes"]},
            {
                "comfyui-workflow-templates",
                "comfyui-workflow-templates-core",
                "comfyui-workflow-templates-json",
                "comfyui-workflow-templates-media-assets-02",
            },
        )
        raw = b'{"nodes":[{"id":1,"type":"MiniMaxH3ImageConditioning"}]}'
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            packages = Path(folder)
            templates = packages / "comfyui_workflow_templates_json/templates"
            templates.mkdir(parents=True)
            (templates / "image_minimax_h3_image_t2i.json").write_bytes(raw)
            core = packages / "comfyui_workflow_templates_core"
            core.mkdir()
            manifest = json.dumps(
                {
                    "templates": [
                        {
                            "assets": [
                                {"filename": "image_minimax_h3_image_t2i.json", "sha256": digest}
                            ]
                        }
                    ]
                }
            ).encode()
            (core / "manifest.json").write_bytes(manifest)
            for name, version in target["packages"].items():
                metadata = packages / f"{name}-{version}.dist-info"
                metadata.mkdir()
                (metadata / "METADATA").write_text(f"Version: {version}\n")
            documents, identity = read_official_templates(templates)
        self.assertEqual(documents, {"image_minimax_h3_image_t2i.json": raw})
        self.assertEqual(identity["packages"], target["packages"])
        self.assertEqual(identity["manifest_sha256"], hashlib.sha256(manifest).hexdigest())
        self.assertTrue(identity["recipe_assets_match_manifest"])

    def llama(self):
        tree = ast.parse((SOURCE / "comfy/text_encoders/llama.py").read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("penalty_active", "apply_penalty", "generate_dtype")
        ]
        base = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BaseGenerate"
        )
        base.body = [
            node
            for node in base.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("processed_probs", "sample_token")
        ]
        nodes.append(base)
        management = SimpleNamespace(
            should_use_bf16=lambda device: device == "source-supported-bf16"
        )
        scope = dict(torch=torch, comfy=SimpleNamespace(model_management=management))
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "release native logits", "exec"), scope
        )
        return scope

    def test_native_greedy_penalties_cover_history_and_preallocated_mask(self):
        scope = self.llama()
        sampler = scope["BaseGenerate"]()
        for repetition, presence in ((2.0, 0.0), (1.0, 2.0)):
            for mask in (None, torch.tensor([True, False, False])):
                with self.subTest(repetition=repetition, presence=presence, mask=mask):
                    logits = torch.tensor([[4.0, 3.0, 1.0]])
                    token = sampler.sample_token(
                        logits,
                        0.0,
                        0,
                        1.0,
                        0.0,
                        repetition,
                        [0],
                        None,
                        do_sample=False,
                        presence_penalty=presence,
                        penalty_mask=mask,
                    )
                    self.assertEqual(token.item(), 1)
        token = sampler.sample_token(
            torch.tensor([[4.0, 3.0, 1.0]]), 0.0, 0, 1.0, 0.0, 1.0, [0], None, do_sample=False
        )
        self.assertEqual(token.item(), 0)

    def test_native_generation_dtype_uses_the_existing_device_policy(self):
        choose = self.llama()["generate_dtype"]
        self.assertEqual(choose("cpu"), torch.float32)
        self.assertEqual(choose("source-supported-bf16"), torch.bfloat16)

    def test_actual_upstream_new_combined_linear_cases_on_cpu(self):
        script = r"""
import os, sys, ast
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

source = Path(os.environ['RF_TEST_COMFY_SOURCE'])
sys.path.insert(0, str(source))
sys.argv = [sys.argv[0], '--cpu']
import torch
from functools import wraps

def forbidden(*args, **kwargs):
    raise AssertionError('Release CPU audit attempted accelerator discovery')

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
from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management, cpu_discovery_disabled
management = initialize_cpu_model_management()
assert management.get_torch_device() == torch.device('cpu')
import comfy.ops
import comfy.lora
import comfy.model_patcher
import comfy.model_prefetch
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor
native_path = source / 'tests-unit/comfy_test/combined_linear_test.py'
tree = ast.parse(native_path.read_text(encoding='utf-8'))
names = ('_gathered', 'test_combined_linear_declines_unprefetched_vbar',
         'test_single_module_cast_reloads_combined_members')
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == 3
for node in nodes:
    node.decorator_list = []
scope = dict(torch=torch, comfy=comfy, QUANT_ALGOS=QUANT_ALGOS, QuantizedTensor=QuantizedTensor)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native_path), 'exec'), scope)
with cpu_discovery_disabled():
    scope['test_combined_linear_declines_unprefetched_vbar']()
    with ExitStack() as restore:
        class MonkeyPatch:
            def setattr(self, owner, name, value):
                restore.enter_context(patch.object(owner, name, value))
        scope['test_single_module_cast_reloads_combined_members'](MonkeyPatch())
assert torch.cuda.is_available is forbidden and torch.xpu.device_count is forbidden
assert not torch.cuda.is_initialized()
print('actual native CombinedLinear cases: 2 passed; accelerator initialization: absent')
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("actual native CombinedLinear cases: 2 passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
