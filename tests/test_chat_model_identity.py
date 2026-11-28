"""Local GGUF discovery never substitutes another same-named model or projector."""

from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class ChatModelIdentityTests(unittest.TestCase):
    def test_reviewed_weights_are_selectable_without_exposing_projectors_or_downloading(self):
        import json

        catalogue = {}
        for path in (ROOT / "ruinedfooocus/resources/model_catalogue").glob("*.json"):
            catalogue.update(json.loads(path.read_text(encoding="utf-8")))
        manager = SimpleNamespace(
            paths={"path_llm": []},
            get_abspath=Path,
            get_folder_list=lambda kind: [
                entry["filename"] for entry in catalogue.values() if entry["path"] == "path_" + kind
            ],
            DOWNLOADABLE_FILES=catalogue,
            get_folder_file_path=Mock(side_effect=AssertionError("Unexpected download")),
        )
        with patch.dict(
            "sys.modules",
            {
                "ruinedfooocus.app.state": SimpleNamespace(
                    path_manager=manager, settings=SimpleNamespace(default_settings={})
                )
            },
        ):
            module = runpy.run_path(str(ROOT / "ruinedfooocus/features/chat/llama_models.py"))
        choices = module["model_catalogue"]("Downloadable")
        selected = {value for variants in choices.values() for _, value in variants}
        additions = json.loads(
            (ROOT / "ruinedfooocus/resources/model_catalogue/llm_reviewed.json").read_text()
        )
        for key, entry in additions.items():
            self.assertEqual(entry["filename"] in selected, key.startswith("llm/"))
        self.assertIn(module["DEFAULT_MODEL"], selected)
        self.assertIn(module["GPT_OSS_MODEL"], selected)

    def test_user_selected_default_is_downloadable_from_its_pinned_publisher(self):
        import json

        with patch.dict(
            "sys.modules",
            {"ruinedfooocus.app.state": SimpleNamespace(path_manager=None, settings=None)},
        ):
            module = runpy.run_path(str(ROOT / "ruinedfooocus/features/chat/llama_models.py"))
        name = module["DEFAULT_MODEL"]
        catalog = json.loads(
            (ROOT / "ruinedfooocus/resources/model_catalogue/llm.json").read_text(encoding="utf-8")
        )
        entry = catalog["llm/" + name]
        self.assertEqual(name, "LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf")
        self.assertIn(
            "https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF/resolve/", entry["url"]
        )
        self.assertNotIn("/resolve/main/", entry["url"])
        self.assertIn("llm/LFM3-1.7B-Q4_K_M.gguf", catalog)
        self.assertIn("llm/" + module["GPT_OSS_MODEL"], catalog)

    def test_duplicate_basenames_have_distinct_choices_and_require_explicit_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = [Path(folder) / name for name in ("one", "two")]
            for root in roots:
                root.mkdir()
                (root / "example.Q4_K_M.gguf").write_bytes(b"fixture")
                (root / "mmproj-example.gguf").write_bytes(b"projector fixture")
            settings = SimpleNamespace(default_settings={})
            manager = SimpleNamespace(
                paths={"path_llm": list(map(str, roots))},
                get_abspath=Path,
                get_folder_list=lambda kind: ["example.Q4_K_M.gguf"],
                DOWNLOADABLE_FILES={},
                get_folder_file_path=Mock(side_effect=AssertionError("Unexpected download")),
            )
            with patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.app.state": SimpleNamespace(
                        path_manager=manager, settings=settings
                    )
                },
            ):
                module = runpy.run_path(str(ROOT / "ruinedfooocus/features/chat/llama_models.py"))
            found = module["local_models"]()
            expected = {str((root / "example.Q4_K_M.gguf").resolve()) for root in roots}
            self.assertEqual(set(found.values()), expected)
            with self.assertRaisesRegex(ValueError, "Multiple local models"):
                module["resolve_model"]("example.Q4_K_M.gguf")
            for path in expected:
                self.assertEqual(module["resolve_model"](path), Path(path))
            choices = module["model_catalogue"]("Local models")["example"]
            self.assertEqual({value for _, value in choices}, expected)
            self.assertEqual(len({label for label, _ in choices}), 2)
            settings.default_settings["llama_hidden_models"] = [
                str((roots[0] / "example.Q4_K_M.gguf").resolve())
            ]
            self.assertEqual(
                module["resolve_model"]("example.Q4_K_M.gguf"),
                (roots[1] / "example.Q4_K_M.gguf").resolve(),
            )
