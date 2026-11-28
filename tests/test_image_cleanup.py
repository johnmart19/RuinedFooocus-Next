from ruinedfooocus.core.paths import asset_path, resource_path

"""Cleanup regressions using the pinned backend's pure helper bodies; no GPU."""
import ast
import os
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class ImageCleanupTests(unittest.TestCase):
    def test_prompt_ranges_use_native_conditioning_without_pooled_output(self):
        native = function("repositories/ComfyUI/node_helpers.py", "conditioning_set_values", {})
        entries = [[object(), {"pooled_output": object(), "other": 1}], [object(), {"other": 2}]]
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "conditioning_set_values"
            and any(
                isinstance(target, ast.Name) and target.id == "positive_switch"
                for target in node.targets
            )
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            conditioning_set_values=native,
            positive_switch=entries,
            start_perc=0.25,
            end_perc=0.75,
            i=1,
        )
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "prompt_range", "exec"), scope)
        self.assertEqual(
            scope["positive_switch"],
            native(entries, {"start_percent": 0.25, "end_percent": 0.75, "_rf_prompt_step": 1}),
        )
        for original, ranged in zip(entries, scope["positive_switch"]):
            self.assertIs(original[0], ranged[0])
            self.assertNotIn("start_percent", original[1])
            self.assertEqual(ranged[1]["start_percent"], 0.25)
            self.assertEqual(ranged[1]["end_percent"], 0.75)
            self.assertEqual(ranged[1]["_rf_prompt_step"], 1)

    def test_conditioning_cache_entries_and_requests_are_independent(self):
        factory = function(
            "ruinedfooocus/features/generation/pipeline_utils.py", "clean_prompt_cond_caches", {}
        )
        first, second = factory(), factory()
        first["+"]["cache"] = "encoded prompt"
        self.assertIsNone(first["-"]["cache"])
        self.assertIsNone(first["switch"]["cache"])
        self.assertIsNone(second["+"]["cache"])

    def test_wildcard_discovery_does_not_modify_settings_and_follows_changed_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            (first / "cleanup_token.txt").write_text("old-value", encoding="utf-8")
            (second / "cleanup_token.txt").write_text("new-value", encoding="utf-8")
            manager = SimpleNamespace(paths={"path_wildcards": [str(first)]})
            discover = function(
                "ruinedfooocus/core/util.py",
                "get_wildcard_files",
                dict(path_manager=manager, Path=Path),
            )
            expand = function(
                "ruinedfooocus/features/prompting/prompt_processing.py",
                "process_wildcards",
                dict(path_manager=manager, os=os, random=random, re=re),
            )
            self.assertIn("cleanup_token", discover())
            discover()
            self.assertEqual(manager.paths["path_wildcards"], [str(first)])
            self.assertEqual(expand("__cleanup_token__"), "old-value")
            manager.paths["path_wildcards"] = [str(second)]
            self.assertEqual(expand("__cleanup_token__"), "new-value")
            self.assertEqual(expand("__cleanup_token__", [str(first)]), "old-value")
            self.assertEqual(manager.paths["path_wildcards"], [str(second)])

    def test_scalar_wildcard_root_is_one_directory_and_matches_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            user = root / "wildcards"
            user.mkdir()
            (user / "scalar_root_subject.txt").write_text("blue cup", encoding="utf-8")
            manager = SimpleNamespace(paths={"path_wildcards": "wildcards"})
            discover = function(
                "ruinedfooocus/core/util.py",
                "get_wildcard_files",
                dict(path_manager=manager, Path=Path),
            )
            with patch.dict(os.environ, RF_DATA_DIR=str(root)):
                self.assertIn("scalar_root_subject", discover())
            self.assertEqual(manager.paths["path_wildcards"], "wildcards")


if __name__ == "__main__":
    unittest.main()
