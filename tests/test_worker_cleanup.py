"""Cleanup belongs to the actual job and never performs a discovery probe."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from ruinedfooocus.app import resource_lifecycle as life


class WorkerCleanupTests(unittest.TestCase):
    def test_worker_has_no_unconditional_native_eviction_or_cuda_probe(self):
        tree = ast.parse(Path("ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
        worker = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "worker"
        )
        calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in ("soft_empty_cache", "unload_all_models", "device_count", "is_available")
                for node in calls
            )
        )

    def test_only_the_actual_owner_can_reset_its_native_interrupt(self):
        memory = SimpleNamespace(interrupt_current_processing=Mock(), soft_empty_cache=Mock())
        jobs = life.JobRegistry()
        a, b = jobs.create(), jobs.create()
        a.state["native_interrupt"] = True
        with patch.dict("sys.modules", {"comfy.model_management": memory}):
            with life.job_scope(b), self.assertRaises(life.OwnershipError):
                life.cleanup_interrupt(a)
            with life.job_scope(a):
                life.cleanup_interrupt(a)
                life.cleanup_interrupt(a)
        memory.interrupt_current_processing.assert_called_once_with(False)
        memory.soft_empty_cache.assert_not_called()
