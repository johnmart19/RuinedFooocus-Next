"""Installed Chat variants and job actions use local presence, not filtered ghosts."""

from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.chat.model_browser import model_action_state
from ruinedfooocus.features.chat.model_preparation import ChatModelSession, capture_model
from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.chat.gguf_metadata import role_metadata, is_projector

ROOT = Path(__file__).resolve().parents[1]


class ChatCatalogueTests(unittest.TestCase):
    def test_real_gguf_scalar_reader_uses_installed_api_and_releases_its_mapping(self):
        import gguf
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "projector.gguf"
            writer = gguf.GGUFWriter(path, "clip")
            try:
                writer.add_string("general.type", "projector")
                writer.add_tensor("owned_fixture", np.zeros((1,), dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
            finally:
                writer.close()
            self.assertEqual(
                role_metadata(path), {"general.architecture": "clip", "general.type": "projector"}
            )
            self.assertTrue(is_projector(path))
            moved = path.with_name("released.gguf")
            path.rename(moved)
            moved.unlink()

    def catalogue(self, root):
        manager = NS(
            paths={"path_llm": [str(root / "llm")], "path_vision": [str(root / "vision")]},
            get_abspath=Path,
            get_folder_list=lambda _: [
                "Example-Q4_K_M.gguf",
                "Example-Q8_0.gguf",
                "Missing-Q4_K_M.gguf",
            ],
            DOWNLOADABLE_FILES={
                "llm/" + name: dict(path="path_llm", filename=name)
                for name in ("Example-Q4_K_M.gguf", "Example-Q8_0.gguf", "Missing-Q4_K_M.gguf")
            },
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.app.state": NS(
                        path_manager=manager, settings=NS(default_settings={})
                    )
                },
            ),
            patch(
                "ruinedfooocus.features.chat.gguf_metadata.is_projector",
                side_effect=lambda p: "mmproj" in p.stem.lower(),
            ),
        ):
            return runpy.run_path(str(ROOT / "ruinedfooocus/features/chat/llama_models.py"))

    def test_installed_lists_all_local_quantizations_and_imported_vision_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in (
                "llm/Example-Q4_K_M.gguf",
                "llm/nested/Example-Q8_0.gguf",
                "vision/imported/CustomVision-Q5_K_M.gguf",
                "vision/custom-mmproj-F16.gguf",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"owned fixture")
            module = self.catalogue(root)
            installed = module["model_catalogue"]("Local models")
            all_values = [value for choices in installed.values() for _, value in choices]
            self.assertEqual(len(all_values), 3)
            self.assertTrue(all(Path(value).is_file() for value in all_values))
            self.assertEqual(len(installed["Example"]), 2)
            self.assertIn("CustomVision", installed)
            downloadable = module["model_catalogue"]("Downloadable")
            self.assertEqual([value for _, value in downloadable["Example"]], ["Example-Q8_0.gguf"])
            self.assertIn("Missing", downloadable)

    def test_filtering_preserves_unavailable_draft_without_changing_active_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            module = self.catalogue(Path(folder))
            session = ChatModelSession()
            session.active = "loaded-other.gguf"
            catalogue = {"Example": [("Q4", "Example-Q4_K_M.gguf")]}
            choices, selected, available = module["selection_choices"](
                catalogue, "Example", "stale.gguf"
            )
            self.assertEqual(choices[0], catalogue["Example"][0])
            self.assertEqual(choices[1][1], "stale.gguf")
            self.assertIn("Unavailable", choices[1][0])
            self.assertEqual(selected, "stale.gguf")
            self.assertFalse(available)
            self.assertEqual(catalogue, {"Example": [("Q4", "Example-Q4_K_M.gguf")]})
            self.assertEqual(
                module["selection_choices"]({}, "Example", "stale.gguf")[1:],
                ("stale.gguf", False),
            )
            self.assertEqual(
                module["selection_choices"](catalogue, "Example", "stale.gguf", preserve=False),
                (catalogue["Example"], "Example-Q4_K_M.gguf", True),
            )
            self.assertEqual(
                module["selection_choices"]({}, "Example", "stale.gguf", preserve=False),
                ([], None, False),
            )
            session.select(None)
            self.assertEqual(session.active, "loaded-other.gguf")

    def test_custom_same_basename_does_not_hide_missing_canonical_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "vision/Example-Q4_K_M.gguf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"custom payload, not verified catalogue artifact")
            module = self.catalogue(root)
            installed = module["model_catalogue"]("Local models")
            downloadable = module["model_catalogue"]("Downloadable")
            self.assertEqual(installed["Example"][0][1], str(path.resolve()))
            self.assertIn("Example-Q4_K_M.gguf", [value for _, value in downloadable["Example"]])
            manager = module["path_manager"]
            canonical = root / "llm/Example-Q4_K_M.gguf"
            manager.get_model_folders = lambda kind: [root / kind]
            manager.preparation_request = lambda key, destination=None: PreparationRequest(
                "https://fixture.invalid/model.gguf", destination or canonical
            )
            manager.get_folder_file_path = Mock(return_value=canonical)
            local = {path.name: str(path.resolve())}
            request = capture_model(path.name, {}, manager, local=local)
            self.assertEqual(request.model, canonical.resolve())
            self.assertFalse(request.present)
            self.assertTrue(request.artifacts[0].url)
            explicit = capture_model(str(path.resolve()), {}, manager, local=local)
            self.assertEqual(explicit.model, path.resolve())
            self.assertEqual(explicit.artifacts[0].url, "")
            self.assertEqual(module["resolve_model"](path.name), canonical)
            manager.get_folder_file_path.assert_called_once_with("llm", path.name, progress=None)

    def test_one_actual_truncated_gguf_does_not_abort_local_catalogue_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            broken = root / "llm/broken.gguf"
            broken.parent.mkdir()
            broken.write_bytes(b"GGUF")
            self.assertFalse(is_projector(broken))
            module = self.catalogue(root)
            # Override only this test's projector stub with the real bounded reader.
            module["local_models"].__globals__["is_projector"] = is_projector
            self.assertEqual(module["local_models"](), {"broken.gguf": str(broken.resolve())})
            self.assertIn("broken", module["model_catalogue"]("Local models"))

    def test_scalar_projector_metadata_is_bounded_closed_and_cached_by_file_identity(self):
        import gguf
        from ruinedfooocus.features.models.gguf_metadata import read_scalar_strings

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "custom.gguf"
            writer = gguf.GGUFWriter(path, "clip")
            writer.add_string("general.type", "x" * 4097)
            writer.add_array("tokenizer.ggml.tokens", ["token"] * 10_000)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            with (
                patch.object(gguf, "GGUFReader", side_effect=AssertionError("Full model scan")),
                patch(
                    "ruinedfooocus.features.models.gguf_metadata.read_scalar_strings",
                    wraps=read_scalar_strings,
                ) as reader,
            ):
                self.assertEqual(role_metadata(path), {"general.architecture": "clip"})
                self.assertTrue(is_projector(path))
                reader.assert_called_once()
                path.write_bytes(path.read_bytes() + b"changed")
                self.assertTrue(is_projector(path))
                self.assertEqual(reader.call_count, 2)

    def test_model_actions_show_download_only_for_missing_selected_payload_and_owned_cancel(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(
                Path, "open", side_effect=AssertionError("Action rendering must not read weights")
            ),
        ):
            root = Path(folder)
            present = root / "local.gguf"
            # Presence is stat-only; create before the read guard using touch.
            present.touch()
            downloads = {
                "llm/local.gguf": dict(path="path_llm", filename="local.gguf"),
                "llm/missing.gguf": dict(path="path_llm", filename="missing.gguf"),
            }
            roots = {"path_llm": [root]}
            self.assertEqual(
                model_action_state(str(present), downloads),
                dict(download_visible=False, cancel_visible=False, load_available=True),
            )
            self.assertFalse(
                model_action_state("local.gguf", downloads, roots=roots)["download_visible"]
            )
            self.assertTrue(
                model_action_state(
                    "local.gguf", downloads, roots={"path_llm": [root / "elsewhere"]}
                )["download_visible"]
            )
            self.assertTrue(model_action_state("missing.gguf", downloads)["download_visible"])
            self.assertFalse(
                model_action_state(str(root / "missing.gguf"), downloads)["download_visible"]
            )
            self.assertFalse(model_action_state(None, downloads)["download_visible"])
            transfer = NS(done=Mock(return_value=False))
            session = NS(transfer=transfer, active="unrelated-active.gguf")
            self.assertEqual(
                model_action_state("missing.gguf", downloads, session),
                dict(download_visible=False, cancel_visible=True, load_available=False),
            )
            transfer.done.return_value = True
            self.assertEqual(
                model_action_state("missing.gguf", downloads, session),
                dict(download_visible=True, cancel_visible=False, load_available=False),
            )


if __name__ == "__main__":
    unittest.main()
