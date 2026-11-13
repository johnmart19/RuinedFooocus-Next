"""The selected compact default binds a pinned artifact to existing native callers."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.app import api_runtime
import ruinedfooocus.app as app_modules
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.features.chat.model_profiles import (
    chat_load_profile,
    initialized_reasoning,
    ProfiledChatClient,
)
from ruinedfooocus.features.models.model_capabilities import (
    RuntimeIdentity,
    task_eligibility,
)
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.packaging.backend_preferences import effective_backends
from ruinedfooocus.packaging.gpu_installer import selected_llm_runtime

ROOT = Path(__file__).resolve().parents[1]
NAME = "LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf"
REVISION = "3ad712542eb9de349dbd28f2739b69c925cdec7d"
SHA256 = "b22df65673b54ed326f18f74826d76f3b65dadc237ffd6306cd52be04544827d"


def model_functions(manager=None, settings=None):
    with patch.dict(
        sys.modules,
        {
            "ruinedfooocus.app.state": SimpleNamespace(
                path_manager=manager,
                settings=settings or SimpleNamespace(default_settings={}),
            )
        },
    ):
        return runpy.run_path(str(ROOT / "ruinedfooocus/features/chat/llama_models.py"))


class AtomicLfm3DefaultTests(unittest.TestCase):
    def setUp(self):
        self.downloads = {}
        for path in resource_path("model_catalogue").glob("*.json"):
            self.downloads.update(json.loads(path.read_text(encoding="utf-8")))
        self.artifact = load_registry().annotations_for(
            "llm/" + NAME, self.downloads["llm/" + NAME]
        )

    def test_default_resolution_uses_exact_pinned_acquisition_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = PathManager.__new__(PathManager)
            manager.DOWNLOADABLE_FILES = self.downloads
            manager.get_model_download_folder = lambda kind: Path(folder)
            request = manager.preparation_request("llm/" + NAME)
            self.assertEqual(request.destination, (Path(folder) / NAME).resolve())
            self.assertEqual(request.expected_size, 1132208000)
            self.assertEqual(request.sha256, SHA256)
            self.assertEqual(
                request.url,
                f"https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF/resolve/{REVISION}/{NAME}",
            )
            manager.get_folder_file_path = Mock(return_value=request.destination)
            models = model_functions(manager)
            self.assertEqual(models["DEFAULT_MODEL"], NAME)
            self.assertEqual(models["resolve_model"](NAME), request.destination)
            manager.get_folder_file_path.assert_called_once_with("llm", NAME, progress=None)
            self.assertIn("Default", models["model_label"]("LFM3-1.7B-Uncensored-AD"))
            self.assertNotIn("Default", models["model_label"]("LFM3-1.7B"))

    def test_fresh_native_auto_and_saved_explicit_choices_share_actual_api_selection(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"RF_DATA_DIR": folder}),
        ):
            settings = SettingsManager()
            shared = SimpleNamespace(settings=settings)
            defaults = model_functions()["DEFAULT_MODEL"]
            with (
                patch.dict(
                    sys.modules,
                    {
                        "ruinedfooocus.app.state": shared,
                        "ruinedfooocus.features.chat.llama_models": SimpleNamespace(
                            DEFAULT_MODEL=defaults
                        ),
                    },
                ),
                patch.object(app_modules, "state", shared, create=True),
            ):
                self.assertEqual(selected_llm_runtime(), "llama.cpp")
                self.assertEqual(effective_backends(settings.default_settings)["llama"], "Auto")
                self.assertEqual(
                    api_runtime.configured_chat_model(), ("chat/" + NAME, {"path": NAME})
                )
                saved = {
                    "llama_localfile": "custom.gguf",
                    "llm_runtime": "xllamacpp",
                    "sentinel": {"keep": [0, False]},
                }
                path = Path(folder) / "settings/settings.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(saved), encoding="utf-8")
                settings.load_settings()
                self.assertEqual(selected_llm_runtime(), "xllamacpp")
                self.assertEqual(
                    api_runtime.configured_chat_model(),
                    ("chat/custom.gguf", {"path": "custom.gguf"}),
                )
                self.assertEqual(json.loads(path.read_text()), saved)

    def test_source_identity_keeps_native_and_embedded_support_distinct(self):
        self.assertEqual(self.artifact["profile_id"], "language:atomic_lfm3_1_7b_uncensored")
        self.assertEqual(self.artifact["architecture"], "lfm3")
        self.assertEqual(self.artifact["artifact"]["sha256"], SHA256)
        native = task_eligibility(self.artifact, RuntimeIdentity("llama.cpp", "b12110"))
        self.assertEqual(native.status, "source_supported")
        embedded = task_eligibility(self.artifact, RuntimeIdentity("xllamacpp", "b11870"))
        self.assertEqual(embedded.status, "unsupported")
        self.assertIn("LFM3", embedded.reason)
        profile = load_registry().profile(self.artifact["profile_id"])
        self.assertFalse(profile["capabilities"]["tools"]["quantized_app_quality_verified"])
        self.assertEqual(profile["identity"]["artifact_revision"], REVISION)

    def test_existing_lfm3_load_contract_and_explicit_sampling_preserve_prompt(self):
        settings = {"llm_runtime": "llama.cpp", "llm_sampling": {"temperature": 0}}
        before = deepcopy(settings)
        profile = chat_load_profile(self.artifact, settings)
        self.assertIn("--reasoning-budget", profile.native_defaults)
        self.assertEqual(
            profile.native_defaults[profile.native_defaults.index("--reasoning-budget") + 1], "0"
        )
        self.assertEqual(profile.settings, before)
        capability = initialized_reasoning(
            self.artifact, settings, "{{ enable_thinking }}", initialized=True
        )
        self.assertIsNone(capability["control"])
        self.assertIn("LFM3", capability["reason"])
        client = Mock()
        adapter = ProfiledChatClient(client, self.artifact, settings)
        request = {
            "messages": [
                {"role": "user", "content": "Keep <lora:sample:0.7> and Café-Ж; do not draw."}
            ]
        }
        original = deepcopy(request)
        adapter.handle_chat_completions(request)
        sent = client.handle_chat_completions.call_args.args[0]
        self.assertEqual(sent["messages"], original["messages"])
        self.assertEqual(sent["temperature"], 0)
        self.assertNotIn("tools", sent)
        self.assertEqual(request, original)


if __name__ == "__main__":
    unittest.main()
