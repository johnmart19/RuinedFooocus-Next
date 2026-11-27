import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from ruinedfooocus.core import download_sync
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.models.acquisition import Catalogue
from ruinedfooocus.features.models import popular_models


class DownloadSystemsSyncTests(unittest.TestCase):
    def _complete_transport(self, payload=b"model-weights"):
        def transport(request, staging, report, cancelled):
            part = staging / "artifact.part"
            part.write_bytes(payload)
            report(len(payload), len(payload), 0)
            return part

        return transport

    def test_completion_listener_fires_on_publish_only_and_isolates_errors(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = PreparationService(
                max_workers=1,
                max_jobs=2,
                transport=self._complete_transport(b"abc123"),
                staging_root=root / "staging",
            )
            self.addCleanup(service.close)
            seen = []

            def broken_listener(*_args):
                raise RuntimeError("observer failed")

            def recorder(request, destination, signature):
                seen.append((request.destination, destination, signature))

            service.add_completion_listener(broken_listener)
            service.add_completion_listener(recorder)

            dest = root / "models" / "demo.safetensors"
            req = PreparationRequest("https://example.test/demo.safetensors", dest)
            self.assertEqual(service.submit(req).result(5), dest.resolve())
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0][0], dest.resolve())
            self.assertEqual(seen[0][1], dest.resolve())

            # Re-submitting an already-installed file verifies it without firing publish sync.
            self.assertEqual(service.submit(req).result(5), dest.resolve())
            self.assertEqual(len(seen), 1)

    def test_active_transfers_reports_in_flight_job(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            started, release = threading.Event(), threading.Event()

            def blocking_transport(request, staging, report, cancelled):
                part = staging / "artifact.part"
                part.write_bytes(b"1234")
                report(2, 4, 10)
                started.set()
                self.assertTrue(release.wait(5))
                return part

            service = PreparationService(
                max_workers=1,
                max_jobs=2,
                transport=blocking_transport,
                staging_root=root / "staging",
            )
            self.addCleanup(service.close)
            dest = root / "models" / "active.gguf"
            sub = service.submit(
                PreparationRequest("https://example.test/active.gguf", dest, expected_size=4)
            )
            self.assertTrue(started.wait(5))
            active = service.active_transfers()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["destination"], str(dest.resolve()))
            self.assertEqual(active[0]["state"], "running")
            self.assertEqual(active[0]["received_bytes"], 2)
            self.assertEqual(active[0]["total_bytes"], 4)
            release.set()
            self.assertEqual(sub.result(5), dest.resolve())
            self.assertEqual(service.active_transfers(), [])

    def test_catalogue_auto_invalidates_when_another_downloader_publishes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = root / "checkpoints"
            ckpt_dir.mkdir()
            manager = SimpleNamespace(
                DOWNLOADABLE_FILES={
                    "checkpoints/sample.safetensors": {
                        "path": "path_checkpoints",
                        "filename": "sample.safetensors",
                        "url": "https://example.test/sample.safetensors",
                        "size": 5,
                    }
                },
                get_model_folders=lambda kind: [ckpt_dir] if kind == "checkpoints" else [],
            )
            empty_registry = SimpleNamespace(annotations_for=lambda key, info: None)
            catalogue = Catalogue(manager, registry=empty_registry)

            before = catalogue.records(include_local=False)
            self.assertEqual(len(before), 1)
            self.assertEqual(before[0]["status"], "Available")

            # Simulate another surface (Popular now / Create / Components) publishing the file.
            dest = ckpt_dir / "sample.safetensors"
            dest.write_bytes(b"hello")
            download_sync.notify_artifact_published(dest)

            after = catalogue.records(include_local=False)
            self.assertEqual(len(after), 1)
            self.assertEqual(after[0]["status"], "Installed")

    def test_notify_artifact_published_syncs_chat_and_tracked_models_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_dir = root / "llm"
            cache_dir = root / "cache" / "embeds"
            llm_dir.mkdir(parents=True)
            cache_dir.mkdir(parents=True)
            gguf = llm_dir / "demo.gguf"
            gguf.write_bytes(b"GGUF")
            embed = cache_dir / "page.html"
            embed.write_bytes(b"<html></html>")

            fake_models = MagicMock()
            fake_models.model_dirs = {"checkpoints": [root / "checkpoints"]}
            fake_manager = SimpleNamespace(
                model_paths={"cache_path": root / "cache"},
                get_model_folders=lambda kind: [llm_dir] if kind == "llm" else [],
            )
            fake_shared = SimpleNamespace(models=fake_models, path_manager=fake_manager)

            import ruinedfooocus.app.state as shared_state
            import ruinedfooocus.features.chat.llama_models as llama_models

            with (
                patch.object(shared_state, "models", fake_models),
                patch.object(shared_state, "path_manager", fake_manager),
                patch.object(llama_models, "invalidate_model_catalogue") as invalidate_chat,
            ):
                download_sync.notify_artifact_published(embed)
                invalidate_chat.assert_not_called()
                fake_models.update_all_models.assert_not_called()

                download_sync.notify_artifact_published(gguf)
                invalidate_chat.assert_called_once()
                fake_models.update_all_models.assert_not_called()

            ckpt_dir = root / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            ckpt = ckpt_dir / "model.safetensors"
            ckpt.write_bytes(b"weights")
            with (
                patch.object(shared_state, "models", fake_models),
                patch.object(shared_state, "path_manager", fake_manager),
                patch.object(llama_models, "invalidate_model_catalogue") as invalidate_chat,
            ):
                download_sync.notify_artifact_published(ckpt)
                invalidate_chat.assert_not_called()
                fake_models.update_all_models.assert_called_once()

    def test_popular_and_catalogue_deduplicate_same_in_flight_transfer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_dir = root / "llm"
            llm_dir.mkdir()
            payload = b"quantized-weights"
            sha = hashlib.sha256(payload).hexdigest()
            started, release = threading.Event(), threading.Event()
            calls = []

            def transport(request, staging, report, cancelled):
                calls.append(request)
                part = staging / "artifact.part"
                part.write_bytes(payload)
                started.set()
                self.assertTrue(release.wait(5))
                return part

            service = PreparationService(
                max_workers=1,
                max_jobs=2,
                transport=transport,
                staging_root=root / "staging",
            )
            self.addCleanup(service.close)

            url = "https://huggingface.co/org/repo/resolve/abc/demo.gguf"
            entry = {
                "source": "Hugging Face",
                "category": "Chat models",
                "file": "demo.gguf",
                "url": url,
                "sha256": sha,
                "bytes": len(payload),
            }
            manager = SimpleNamespace(model_paths={"llm_path": llm_dir})

            with patch("ruinedfooocus.features.models.popular_models.offline", return_value=False):
                dest, pop_sub = popular_models.start_download(entry, manager, service=service)
            self.assertTrue(started.wait(5))

            # Submitting the matching Catalogue PreparationRequest joins the same in-flight job.
            cat_request = PreparationRequest(
                url,
                dest,
                expected_size=len(payload),
                sha256=sha,
                model_root=llm_dir,
            )
            cat_sub = service.submit(cat_request)
            release.set()
            self.assertEqual(pop_sub.result(5), dest.resolve())
            self.assertEqual(cat_sub.result(5), dest.resolve())
            self.assertEqual(len(calls), 1)

    def test_load_file_from_url_uses_preparation_service_and_respects_offline(self):
        from ruinedfooocus.core.util import load_file_from_url

        with TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "models"
            with patch("ruinedfooocus.features.models.ui_model_components.offline", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "Offline mode"):
                    load_file_from_url(
                        "https://example.test/weights.safetensors",
                        model_dir=str(target_dir),
                    )

            service = PreparationService(
                max_workers=1,
                max_jobs=1,
                transport=self._complete_transport(b"ok"),
                staging_root=Path(tmp) / "staging",
            )
            self.addCleanup(service.close)
            download_sync.install_default_sync(service)
            before_sync = download_sync.sync_version()

            with (
                patch("ruinedfooocus.features.models.ui_model_components.offline", return_value=False),
                patch("ruinedfooocus.core.preparation.preparation_service", service),
            ):
                saved = load_file_from_url(
                    "https://example.test/weights.safetensors",
                    model_dir=str(target_dir),
                )
            self.assertTrue(Path(saved).is_file())
            self.assertEqual(Path(saved).read_bytes(), b"ok")
            self.assertGreater(download_sync.sync_version(), before_sync)


if __name__ == "__main__":
    unittest.main()
