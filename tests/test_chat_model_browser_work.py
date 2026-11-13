"""Chat search work stays independent of the number of installed GGUF files."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy
import statistics
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def catalogue_fixture(count=160):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        key, value = b"general.architecture", b"llama"
        header = (
            struct.pack("<4sIQQ", b"GGUF", 3, 0, 1)
            + struct.pack("<Q", len(key))
            + key
            + struct.pack("<IQ", 8, len(value))
            + value
        )
        for index in range(count):
            kind = "llm" if index % 2 else "vision"
            path = root / kind / f"nested-{index % 8}" / f"Family{index:03d}-Q4_K_M.gguf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(header)
        manager = SimpleNamespace(
            paths={"path_llm": [str(root / "llm")], "path_vision": [str(root / "vision")]},
            get_abspath=Path,
            get_folder_list=lambda _: ["Missing-Q4_K_M.gguf"],
            DOWNLOADABLE_FILES={
                "llm/Missing-Q4_K_M.gguf": {
                    "path": "path_llm",
                    "filename": "Missing-Q4_K_M.gguf",
                }
            },
        )
        settings = SimpleNamespace(default_settings={})
        with patch.dict(
            "sys.modules",
            {"ruinedfooocus.app.state": SimpleNamespace(path_manager=manager, settings=settings)},
        ):
            module = runpy.run_path(str(ROOT / "ruinedfooocus/features/chat/llama_models.py"))
        yield root, module, manager, settings


def search_callback(catalogue, module):
    return function_from(
        "ruinedfooocus/features/chat/ui.py",
        "search_models",
        dict(
            model_catalogue=catalogue,
            model_label=module["model_label"],
            clear_model_selection=lambda _: {},
            gr=SimpleNamespace(update=lambda **values: values),
            llama_model="model",
            local_picker="picker",
        ),
    )


class ChatModelBrowserWorkTests(unittest.TestCase):
    def test_bulk_search_reuses_inventory_without_scans_or_header_reads(self):
        from ruinedfooocus.features.chat import gguf_metadata
        from ruinedfooocus.features.models.gguf_metadata import read_scalar_strings
        import gguf

        measurements = []
        with catalogue_fixture() as (_, module, _, _):
            cache_type = module["ModelCatalogueCache"]
            for _ in range(5):
                catalogue = function_from(
                    "ruinedfooocus/features/chat/ui.py",
                    "chat_model_catalogue",
                    dict(ModelCatalogueCache=cache_type, model_catalogue=module["model_catalogue"]),
                )
                catalogue = catalogue()
                search = search_callback(catalogue, module)
                gguf_metadata._role_metadata.cache_clear()
                started = time.perf_counter()
                for source in ("Local models", "Downloadable"):
                    search(source, None, None, "")
                cold_seconds = time.perf_counter() - started
                rglob = Path.rglob
                with (
                    patch.object(Path, "rglob", autospec=True, side_effect=rglob) as scans,
                    patch(
                        "ruinedfooocus.features.models.gguf_metadata.read_scalar_strings",
                        wraps=read_scalar_strings,
                    ) as headers,
                    patch.object(gguf, "GGUFReader", side_effect=AssertionError("Tensor scan")),
                ):
                    started = time.perf_counter()
                    for index in range(20):
                        search(
                            ("Local models", "Downloadable")[index % 2],
                            None,
                            None,
                            ("Family", "Q4", "gguf", "")[index % 4],
                        )
                    measurements.append(
                        dict(
                            warm_seconds=time.perf_counter() - started,
                            cold_seconds=cold_seconds,
                            scans=scans.call_count,
                            header_reads=headers.call_count,
                        )
                    )
        evidence = ROOT / "tmp/chat-model-browser-perf/after.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(
            json.dumps(
                dict(
                    fixture_files=160,
                    fixture_tensor_count=0,
                    queries_per_trial=20,
                    trials=measurements,
                    median_warm_seconds=statistics.median(
                        row["warm_seconds"] for row in measurements
                    ),
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual([row["scans"] for row in measurements], [0] * 5)
        self.assertEqual([row["header_reads"] for row in measurements], [0] * 5)

    def test_visible_refresh_discovers_external_files_without_picking_another_quant(self):
        with catalogue_fixture(2) as (root, module, _, _):
            cache = module["ModelCatalogueCache"](module["model_catalogue"])
            initial = cache("Local models")
            original = next(iter(initial.values()))[0][1]
            extra = root / "llm/Added-Q8_0.gguf"
            extra.write_bytes(Path(original).read_bytes())
            self.assertNotIn("Added", cache("Local models"))
            from ruinedfooocus.features.chat.model_preparation import ChatModelSession

            session = ChatModelSession()
            selected = Mock(side_effect=lambda *args, **kwargs: kwargs["catalogue"])
            refresh = function_from(
                "ruinedfooocus/features/chat/ui.py",
                "refresh_model_browser",
                dict(model_catalogue=cache, select_source=selected),
            )
            refreshed = refresh("Local models", "Family001", original, session)
            self.assertIn("Added", refreshed)
            selected.assert_called_once_with(
                "Local models", "Family001", original, session, catalogue=refreshed
            )

    def test_verified_hub_gguf_commit_invalidates_warmed_chat_without_selecting_it(self):
        with catalogue_fixture(2) as (root, module, manager, _):
            manager.get_model_folders = lambda kind: [root / kind]
            manager.model_paths = {"cache_path": root / "cache"}
            cache = module["ModelCatalogueCache"](module["model_catalogue"])
            self.assertIn("Missing", cache("Downloadable"))
            original = next(iter(cache("Local models").values()))[0][1]
            target = root / "llm/Missing-Q4_K_M.gguf"
            target.write_bytes(Path(original).read_bytes())
            self.assertNotIn("Missing", cache("Local models"))
            models = SimpleNamespace(update_all_models=Mock(), revision=0)
            poll = function_from(
                "ruinedfooocus/features/models/ui_acquisition.py",
                "poll",
                dict(Path=Path, manager=manager, models=models, gr=SimpleNamespace(update=dict)),
            )
            job = SimpleNamespace(
                active=True,
                poll=lambda: ("Ready", False, True),
                drain_verified=lambda: [(object(), target, {})],
            )
            with (
                patch(
                    "ruinedfooocus.features.models.model_identity.record_prepared_native_source",
                    return_value=None,
                ),
                patch.dict(
                    "sys.modules",
                    {
                        "ruinedfooocus.features.chat.llama_models": SimpleNamespace(
                            invalidate_model_catalogue=module["invalidate_model_catalogue"]
                        )
                    },
                ),
            ):
                poll(job, 0)
            self.assertIn("Missing", cache("Local models"))
            self.assertNotIn("Missing", cache("Downloadable"))
            models.update_all_models.assert_called_once()

    def test_import_and_remove_invalidate_other_browsing_views(self):
        with catalogue_fixture(2) as (root, module, _, settings):
            settings.save_settings = Mock()
            cache = module["ModelCatalogueCache"](module["model_catalogue"])
            initial = cache("Local models")
            original = next(iter(initial.values()))[0][1]
            imported = root / "Imported-Q4_K_M.gguf"
            imported.write_bytes(Path(original).read_bytes())
            with patch("ruinedfooocus.features.chat.sampling_sources.prepare_sampling_source"):
                family, quant = module["import_model"](imported)
            self.assertEqual(family, "Imported")
            self.assertIn("Imported", cache("Local models"))
            module["remove_local_model"](quant, delete_file=True)
            self.assertNotIn("Imported", cache("Local models"))
            self.assertTrue(imported.is_file())

    def test_root_and_hidden_configuration_changes_invalidate_the_snapshot(self):
        with catalogue_fixture(2) as (_, module, manager, settings):
            cache = module["ModelCatalogueCache"](module["model_catalogue"])
            before = cache("Local models")
            path = next(iter(before.values()))[0][1]
            settings.default_settings["llama_hidden_models"] = [path]
            after = cache("Local models")
            self.assertNotEqual(before, after)
            manager.paths["path_llm"] = []
            manager.paths["path_vision"] = []
            self.assertEqual(cache("Local models"), {})

    def test_concurrent_cold_callers_coalesce_and_discard_a_writer_superseded_scan(self):
        entered, release = threading.Event(), threading.Event()
        with catalogue_fixture(0) as (_, module, _, _):
            calls = 0

            def load(_):
                nonlocal calls
                calls += 1
                if calls == 1:
                    entered.set()
                    self.assertTrue(release.wait(5))
                    return {"old": []}
                return {"current": []}

            cache = module["ModelCatalogueCache"](load)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(cache, "Local models")
                try:
                    self.assertTrue(entered.wait(5))
                    second = pool.submit(cache, "Local models")
                    module["invalidate_model_catalogue"]()
                finally:
                    release.set()
                self.assertEqual(first.result(5), {"current": []})
                self.assertEqual(second.result(5), {"current": []})
            self.assertEqual(calls, 2)

    def test_download_completion_invalidates_presence_without_activating_the_model(self):
        with catalogue_fixture(2) as (root, module, _, _):
            cache = module["ModelCatalogueCache"](module["model_catalogue"])
            self.assertIn("Missing", cache("Downloadable"))
            original = next(iter(cache("Local models").values()))[0][1]
            target = root / "llm/Missing-Q4_K_M.gguf"
            transfer = SimpleNamespace(
                done=lambda: True,
                result=lambda: target.write_bytes(Path(original).read_bytes()),
            )
            session = SimpleNamespace(
                select=lambda _: 1,
                start=lambda _: (1, transfer),
                current=lambda _: True,
                selected="Missing-Q4_K_M.gguf",
                active="keep-active.gguf",
            )
            download = function_from(
                "ruinedfooocus/features/chat/ui.py",
                "download_model",
                dict(
                    capture_model=lambda *args, **kwargs: SimpleNamespace(model=target),
                    settings=SimpleNamespace(default_settings={}),
                    path_manager=object(),
                    local_models=lambda: {},
                    preparation_status="status",
                    llama_download="progress",
                    action_updates=lambda *args: {},
                    gr=SimpleNamespace(update=lambda **kwargs: kwargs),
                    CancelledError=RuntimeError,
                    describe_error=str,
                ),
            )
            with patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.features.chat.llama_models": SimpleNamespace(
                        invalidate_model_catalogue=module["invalidate_model_catalogue"]
                    )
                },
            ):
                list(download("Missing-Q4_K_M.gguf", session))
            self.assertNotIn("Missing", cache("Downloadable"))
            self.assertEqual(session.active, "keep-active.gguf")


if __name__ == "__main__":
    unittest.main()
