"""Exercise local model discovery without publisher or application bootstrap."""

import ast
from pathlib import Path
import runpy
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class LocalModelRefreshTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        folders = {}
        for key in ("modelfile_path", "lorafile_path", "embedding_path", "inbox_path"):
            folders[key] = self.root / key
            folders[key].mkdir()
        folders["cache_path"] = self.root / "cache"
        self.manager = SimpleNamespace(model_paths=folders, get_model_folders=lambda _: [])
        self.state = SimpleNamespace(
            path_manager=self.manager,
            settings=SimpleNamespace(),
            shared_cache={},
            translate=lambda text, **_: text,
        )
        self.patch = patch.dict(sys.modules, {"ruinedfooocus.app.state": self.state})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.Models = runpy.run_path(str(ROOT / "ruinedfooocus/features/models/model_handler.py"))[
            "Models"
        ]

    def test_constructor_and_refresh_are_local_and_preserve_selection_source(self):
        local = self.manager.model_paths["modelfile_path"] / "local.safetensors"
        local.write_bytes(b"not loaded")
        with patch.object(threading, "Thread") as thread:
            models = self.Models()
            self.assertEqual(models.get_names("checkpoints"), [local.name])
            (local.parent / "new.gguf").write_bytes(b"not loaded")
            models.update_all_models()
            self.assertEqual(models.get_names("checkpoints"), [local.name, "new.gguf"])
            thread.assert_not_called()

    def test_slow_artwork_worker_does_not_block_names_or_local_refresh(self):
        models = self.Models()
        entered, release = threading.Event(), threading.Event()
        models._update_models = Mock(side_effect=lambda *_: (entered.set(), release.wait(2)))
        worker = threading.Thread(target=models.civit_update_worker, args=("checkpoints", []))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            models.civit_update_worker("checkpoints", [])
            self.assertEqual(models.get_names("checkpoints"), [])
            models.update_all_models()
            self.assertEqual(models._update_models.call_count, 1)
        finally:
            release.set()
            worker.join(2)
        models.civit_update_worker("checkpoints", [])
        self.assertEqual(models._update_models.call_count, 1)

    def test_artwork_scan_cannot_clear_local_inventory(self):
        local = self.manager.model_paths["modelfile_path"] / "kept.safetensors"
        local.write_bytes(b"not loaded")
        models = self.Models()
        models._update_models("checkpoints", [])
        self.assertEqual(models.get_names("checkpoints"), [local.name])

    def test_ui_load_starts_existing_workers_and_returns_while_artwork_waits(self):
        models = self.Models()
        for directory in models.cache_paths.values():
            directory.mkdir(parents=True, exist_ok=True)
        release = threading.Event()
        started = threading.Event()
        workers = []
        models._update_models = Mock(side_effect=lambda *_: (started.set(), release.wait(2)))
        thread_type = threading.Thread

        def make_thread(**kwargs):
            worker = thread_type(**kwargs)
            workers.append(worker)
            return worker

        try:
            with patch.object(threading, "Thread", side_effect=make_thread):
                self.assertIsNone(models.start_metadata_updates())
            self.assertTrue(started.wait(1))
            self.assertEqual(len(workers), 3)
            self.assertTrue(any(worker.is_alive() for worker in workers))
            tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text())
            load = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "main_tabs.load"
                and n.args
                and ast.unparse(n.args[0]) == "shared.models.start_metadata_updates"
            )
            options = {item.arg: ast.literal_eval(item.value) for item in load.keywords}
            self.assertFalse(options["queue"])
            self.assertEqual(options["outputs"], [])
            self.assertEqual(options["show_progress"], "hidden")
        finally:
            release.set()
            for worker in workers:
                worker.join(2)
            self.assertEqual(models._update_models.call_count, 3)
            self.assertFalse(any(worker.is_alive() for worker in workers))

    def test_backend_checks_are_once_per_process_with_import_paths_retained(self):
        source = ast.parse((ROOT / "ruinedfooocus/app/bootstrap.py").read_text())
        node = next(
            n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "clone_git_repos"
        )
        namespace = dict(
            _checked_git_repos=set(),
            git_repos=[
                dict(path="fixture", hash="pin", name="Fixture", url="none", add_path="fixture")
            ],
            repo_dir=lambda p: p,
            Path=Path,
            script_path=self.root,
            dir_repos="repositories",
            sys=SimpleNamespace(path=[]),
        )
        clone = Mock()
        with patch("ruinedfooocus.packaging.launch_util.git_clone", clone):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "bootstrap_fixture", "exec"),
                namespace,
            )
            namespace["clone_git_repos"]()
            namespace["clone_git_repos"]()
        self.assertEqual(clone.call_count, 1)
        self.assertEqual(len(namespace["sys"].path), 1)

    def test_refresh_callback_preserves_filters_selection_and_prompt(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "model_refresh_clicked"
        )
        decorator = node.decorator_list[0]
        options = {item.arg: item.value for item in decorator.keywords}
        self.assertEqual(ast.literal_eval(options["show_progress"]), "hidden")
        self.assertFalse(ast.literal_eval(options["queue"]))
        outputs = {ast.unparse(value) for value in options["outputs"].elts}
        self.assertTrue(
            outputs.isdisjoint(
                {"prompt", "negative_prompt", "modelfilter", "lorafilter", "adapters.selected"}
            )
        )
        node.decorator_list = []
        names = Mock()
        names.get_names.return_value = ["selected"]
        discover = Mock(return_value=(["local adapter"], "Ready"))
        filter_models = Mock(return_value=("gallery", "records"))
        namespace = dict(
            shared=SimpleNamespace(models=names),
            update_model_filter=filter_models,
            adapters=SimpleNamespace(
                discover=discover, catalog="adapter_catalog", status="adapter_status"
            ),
            model_gallery="gallery",
            model_gallery_records="records",
            model_metadata_refresh="refresh_signal",
            style_selection="styles",
            preset_gallery="presets",
            base_model="base",
            gr=SimpleNamespace(update=lambda **kwargs: kwargs),
            load_styles=lambda: {"style": {}},
            path_manager=SimpleNamespace(get_presets=lambda: []),
            checkpoint_choices=lambda choices, selected, metadata: choices,
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "refresh_callback", "exec"), namespace
        )
        result = namespace["model_refresh_clicked"](
            [], "selected", "keep search", "keep adapter search", 7
        )
        filter_models.assert_called_once_with("keep search")
        names.update_all_models.assert_called_once_with()
        self.assertEqual(
            set(result),
            {
                "gallery",
                "records",
                "adapter_catalog",
                "adapter_status",
                "styles",
                "presets",
                "base",
                "refresh_signal",
            },
        )
        self.assertEqual(result["refresh_signal"], 8)


if __name__ == "__main__":
    unittest.main()
