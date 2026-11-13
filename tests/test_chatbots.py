from ruinedfooocus.core.paths import asset_path, resource_path

"""Bundled chat bot definitions load and match what the Chat bots tab and context indexer accept."""
import ast
import asyncio
import json
import hashlib
import re
import sys
import types
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from PIL import Image
import gradio as gr
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


class SavedAssistantTests(unittest.TestCase):
    def setUp(self):
        from ruinedfooocus.features.chat import assistants

        self.assistants = assistants
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.object(
            assistants, "data_path", side_effect=lambda *parts: self.root.joinpath(*parts).resolve()
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        roots = mock.patch.object(
            assistants,
            "content_roots",
            return_value=(
                resource_path("assistants"),
                self.root / "chatbots",
                self.root / "assistants",
            ),
        )
        roots.start()
        self.addCleanup(roots.stop)

    def test_unicode_name_rename_and_restart_keep_identity_without_optional_fields(self):
        identity, revision = self.assistants.save_assistant("Кіт 猫", "Be a cat.")
        path = self.assistants.user_assistant_path(identity)
        self.assertTrue(path.is_file())
        self.assertIn(("Кіт 猫", identity), self.assistants.assistant_choices())
        info = self.assistants.assistant_info(identity)
        self.assertEqual(info["avatar"], asset_path("chat_normal.png"))
        self.assertEqual(info["chatstart"], [])
        self.assertEqual(info["system"], "Be a cat.")
        renamed, _ = self.assistants.save_assistant(
            "Renamed", "New instruction", identity, revision
        )
        self.assertEqual(renamed, identity)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["name"], "Renamed")
        self.assertIn(("Renamed", identity), self.assistants.assistant_choices())
        self.assertEqual(self.assistants.assistant_identity(str(path.parent)), identity)
        self.assertEqual(
            self.assistants.assistant_info(str(path.parent))["system"], "New instruction"
        )

    def test_general_rename_restart_and_one_read_conflict_token(self):
        info, identity, revision = self.assistants.assistant_edit_snapshot("__normal__")
        self.assertEqual((identity, info["system"]), ("__normal__", ""))
        self.assistants.save_assistant("Renamed General", "", identity, revision)
        self.assertEqual(self.assistants.assistant_choices()[0], ("Renamed General", "__normal__"))
        path = self.root / "assistants/.general/info.json"
        old = path.read_bytes()
        changed = json.dumps({"name": "Other session", "system": ""}).encode()
        read = Path.read_bytes

        def race(candidate):
            value = read(candidate)
            if candidate.resolve() == path.resolve():
                path.write_bytes(changed)
            return value

        with (
            mock.patch.object(self.assistants, "assistant_identity", return_value="__normal__"),
            mock.patch.object(Path, "read_bytes", race),
        ):
            snapshot, identity, token = self.assistants.assistant_edit_snapshot("__normal__")
        self.assertEqual(snapshot["name"], "Renamed General")
        self.assertEqual(token, hashlib.sha256(old).hexdigest())
        self.assertEqual(path.read_bytes(), changed)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.assistants.save_assistant("Stale", "", identity, token)
        self.assertEqual(self.assistants.assistant_info("__normal__")["name"], "Other session")

    def test_avatar_fallback_save_failure_clear_and_delete_preserve_other_files(self):
        source = Image.new("RGBA", (7, 5), (30, 90, 150, 80))
        original = source.tobytes()
        identity, revision = self.assistants.save_assistant("Image", "", avatar=source)
        path = self.assistants.user_assistant_path(identity)
        avatar = self.assistants.assistant_info(identity)["avatar"]
        self.assertEqual(avatar.parent, path.parent)
        with Image.open(avatar) as saved:
            self.assertEqual(
                (saved.mode, saved.size, saved.tobytes()), (source.mode, source.size, original)
            )
        self.assertEqual(source.tobytes(), original)
        old = path.read_bytes()
        with mock.patch.object(self.assistants, "save_json", side_effect=OSError("failed publish")):
            with self.assertRaises(OSError):
                self.assistants.save_assistant(
                    "Changed", "", identity, revision, avatar=Image.new("RGB", (3, 4))
                )
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(list(path.parent.glob("avatar-*.png")), [avatar])
        _, revision = self.assistants.save_assistant("Image", "", identity, revision, avatar=None)
        self.assertEqual(
            self.assistants.assistant_info(identity)["avatar"], asset_path("chat_normal.png")
        )
        unrelated = path.parent / "keep.txt"
        unrelated.write_text("Unrelated")
        self.assistants.delete_assistant(identity, revision)
        self.assertFalse(path.exists())
        self.assertFalse(avatar.exists())
        self.assertEqual(unrelated.read_text(), "Unrelated")

    def test_selected_saved_avatar_passes_real_gradio_cache_boundary_outside_allowed_roots(self):
        from gradio.context import LocalContext
        from gradio.exceptions import InvalidPathError
        from gradio.processing_utils import async_move_files_to_cache

        identity, _ = self.assistants.save_assistant(
            "External portrait", "", avatar=Image.new("RGBA", (73, 61), (20, 90, 160, 120))
        )
        info = self.assistants.assistant_info(identity)
        original = info["avatar"].read_bytes()
        portrait = function_from("ruinedfooocus/features/chat/ui.py", "assistant_avatar", {})
        namespace = {
            "gr": gr,
            "assistant_info": self.assistants.assistant_info,
            "assistant_avatar": portrait,
            **{
                name: name
                for name in (
                    "llama_chat",
                    "llama_history",
                    "llama_msg",
                    "llama_avatar",
                    "llama_system",
                    "llama_embed",
                )
            },
        }
        select = function_from(
            "ruinedfooocus/features/chat/ui.py", "llama_select_assistant", namespace
        )
        output = select(identity)["llama_avatar"]["value"]
        self.assertIsInstance(output, Image.Image)
        self.assertEqual((output.size, output.mode), ((73, 61), "RGBA"))
        source, temporary = self.root / "source", self.root / "system-temp"
        source.mkdir()
        temporary.mkdir()
        with gr.Blocks() as app:
            image = gr.Image(type="pil")
        image.GRADIO_CACHE = str(temporary / "gradio")
        app.has_launched = True
        token = LocalContext.blocks.set(app)
        try:
            with (
                mock.patch("gradio.processing_utils.os.getcwd", return_value=str(source)),
                mock.patch(
                    "gradio.processing_utils.tempfile.gettempdir", return_value=str(temporary)
                ),
            ):
                with self.assertRaises(InvalidPathError):
                    asyncio.run(
                        async_move_files_to_cache(
                            image.postprocess(info["avatar"]).model_dump(), image, postprocess=True
                        )
                    )
                served = asyncio.run(
                    async_move_files_to_cache(
                        image.postprocess(output).model_dump(), image, postprocess=True
                    )
                )
        finally:
            LocalContext.blocks.reset(token)
        self.assertTrue(Path(served["path"]).is_file())
        self.assertNotEqual(Path(served["path"]), info["avatar"])
        self.assertEqual(app.allowed_paths, [])
        self.assertEqual(info["avatar"].read_bytes(), original)

    def test_delete_protects_general_and_guest_and_failed_startup_repair(self):
        for value in ("__normal__", None, "", "unknown"):
            with self.assertRaises(ValueError):
                self.assistants.delete_assistant(value)
        identity, revision = self.assistants.save_assistant("Keep", "Unchanged")
        with self.assertRaises(PermissionError):
            self.assistants.delete_assistant(identity, revision, guest=True)
        with self.assertRaises(OSError):
            self.assistants.delete_assistant(
                identity, revision, before_delete=mock.Mock(side_effect=OSError("profile failed"))
            )
        self.assertEqual(self.assistants.assistant_info(identity)["system"], "Unchanged")

    def test_bundled_delete_hides_only_catalogue_entry(self):
        identity = next(
            value for _, value in self.assistants.assistant_choices() if value != "__normal__"
        )
        path = Path(identity) / "info.json"
        raw = path.read_bytes()
        self.assistants.delete_assistant(identity)
        self.assertNotIn(identity, {value for _, value in self.assistants.assistant_choices()})
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(self.assistants.assistant_choices()[0][1], "__normal__")

    def test_confirm_delete_repairs_only_this_profile_and_keeps_conversation_outputs(self):
        identity, revision = self.assistants.save_assistant("Delete", "Private instruction")
        settings = SimpleNamespace(
            default_settings={"llama_assistant": identity, "unrelated": "keep"},
            name="private",
            save_profile=mock.Mock(),
        )
        names = (
            "pending_assistant_delete",
            "assistant_delete_dialog",
            "assistant_status",
            "llama_assistants",
            "llama_avatar",
            "llama_system",
            "llama_embed",
        )
        confirm = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "confirm_assistant_delete",
            {
                "gr": SimpleNamespace(update=lambda **kwargs: kwargs, EventData=object),
                "settings": settings,
                "path_manager": SimpleNamespace(paths={}),
                "delete_assistant": self.assistants.delete_assistant,
                "assistant_info": self.assistants.assistant_info,
                "assistant_avatar": function_from(
                    "ruinedfooocus/features/chat/ui.py", "assistant_avatar", {}
                ),
                "gr_llama_get_assistants": lambda selected: {"value": selected},
                "translate_ui": lambda value: value,
                **{name: name for name in names},
            },
        )
        with mock.patch.dict(
            sys.modules,
            {"ruinedfooocus.app.arguments": SimpleNamespace(args=SimpleNamespace(guest=False))},
        ):
            result = confirm(
                {"id": identity, "revision": revision},
                identity,
                SimpleNamespace(_data={"action": "delete"}),
            )
        self.assertEqual(result["llama_assistants"]["value"], "__normal__")
        self.assertEqual(result["llama_system"], "")
        self.assertTrue(
            set(result).isdisjoint(
                {"llama_history", "llama_chat", "llama_msg", "llama_active_model"}
            )
        )
        settings.save_profile.assert_called_once_with(
            {"llama_assistant": "__normal__", "unrelated": "keep"}, {}, profile="private"
        )
        self.assertEqual(
            settings.default_settings, {"llama_assistant": "__normal__", "unrelated": "keep"}
        )

    def test_registered_page_load_reads_current_profile_and_refresh_repairs_removed_persona(self):
        from gradio.state_holder import SessionState
        from ruinedfooocus.core.config_io import save_json

        identity, revision = self.assistants.save_assistant(
            "Old startup", "Old instructions", avatar=Image.new("RGB", (7, 5), "purple")
        )
        profile = self.root / "settings/owned/settings.json"
        save_json(profile, {"llama_assistant": identity})
        settings = SimpleNamespace(
            settings_path=profile,
            default_settings={"llama_assistant": identity, "unrelated": "keep"},
        )
        with gr.Blocks() as app:
            scope = {
                "gr": gr,
                "settings": settings,
                "assistant_info": self.assistants.assistant_info,
                "assistant_choices": self.assistants.assistant_choices,
                "translate_ui": lambda value: value,
                "app_llama_chat": app,
            }
            scope.update(
                llama_assistants=gr.Dropdown(["__normal__", identity], value=identity),
                assistant_delete=gr.Button("Delete"),
                llama_avatar=gr.Image(),
                llama_system=gr.Textbox(value="Old instructions"),
                llama_embed=gr.Textbox(value="[]"),
                llama_chat=gr.Chatbot(value=[]),
                llama_history=gr.State([]),
                llama_msg=gr.Textbox(value="draft"),
            )
            for name in (
                "assistant_avatar",
                "gr_llama_get_assistants",
                "llama_select_assistant",
                "refresh_assistant_catalogue",
                "load_assistant_session",
            ):
                function_from("ruinedfooocus/features/chat/ui.py", name, scope)
            tree = ast.parse(
                (ROOT / "ruinedfooocus/features/chat/ui.py").read_text(encoding="utf-8")
            )
            binding = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "load"
                and node.value.args
                and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id == "load_assistant_session"
            )
            exec(
                compile(ast.Module(body=[binding], type_ignores=[]), "assistant-page-load", "exec"),
                scope,
            )
        self.assertTrue(
            any(target[1] == "load" for target in app.config["dependencies"][0]["targets"])
        )

        def load():
            result = asyncio.run(app.process_api(0, [], state=SessionState(app)))
            return dict(zip(app.fns[0].outputs, result["data"]))

        self.assertEqual(load()[scope["llama_system"]]["value"], "Old instructions")
        self.assistants.delete_assistant(identity, revision)
        info, general, token = self.assistants.assistant_edit_snapshot("__normal__")
        self.assistants.save_assistant("UI General test", "", general, token)
        save_json(profile, {"llama_assistant": "__normal__"})
        self.assertEqual(scope["llama_assistants"].value, identity)
        restored = load()
        self.assertEqual(restored[scope["llama_assistants"]]["value"], "__normal__")
        self.assertIn(
            ["UI General test", "__normal__"], restored[scope["llama_assistants"]]["choices"]
        )
        self.assertEqual(restored[scope["llama_avatar"]]["label"], "UI General test")
        self.assertEqual(restored[scope["llama_system"]]["value"], "")
        self.assertFalse(restored[scope["assistant_delete"]]["interactive"])
        refreshed = scope["refresh_assistant_catalogue"](identity)
        self.assertEqual(refreshed[scope["llama_system"]], "")
        self.assertEqual(refreshed[scope["llama_avatar"]]["label"], "UI General test")
        self.assertTrue(
            set(refreshed).isdisjoint(
                {scope["llama_chat"], scope["llama_history"], scope["llama_msg"]}
            )
        )
        profile.unlink()
        self.assertEqual(load()[scope["llama_assistants"]]["value"], "__normal__")
        self.assertEqual(
            settings.default_settings, {"llama_assistant": identity, "unrelated": "keep"}
        )

    def test_save_as_and_duplicate_names_create_distinct_ids(self):
        first, _ = self.assistants.save_assistant("Same", "First")
        second, _ = self.assistants.save_assistant("Same", "Second")
        self.assertNotEqual(first, second)
        self.assertEqual(self.assistants.assistant_info(first)["system"], "First")

    def test_stale_edit_cannot_overwrite_newer_content(self):
        identity, revision = self.assistants.save_assistant("Cat", "Original")
        draft, opened, token = self.assistants.assistant_edit_snapshot(identity)
        self.assertEqual((draft["system"], opened, token), ("Original", identity, revision))
        self.assistants.save_assistant("Cat", "Other client", identity, revision)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.assistants.save_assistant("Cat", "Stale", identity, revision)
        self.assertEqual(self.assistants.assistant_info(identity)["system"], "Other client")

    def test_write_failure_preserves_old_definition_and_revision(self):
        identity, revision = self.assistants.save_assistant("Cat", "Original")
        path = self.assistants.user_assistant_path(identity)
        old = path.read_bytes()
        with mock.patch("ruinedfooocus.core.config_io.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.assistants.save_assistant("Cat", "Unsaved", identity, revision)
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(self.assistants.assistant_revision(identity), revision)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_guest_invalid_ids_and_id_collision_never_write(self):
        with self.assertRaises(PermissionError):
            self.assistants.save_assistant("Guest", "", guest=True)
        for identity in (
            "user:../outside",
            "user:UPPER",
            "user:a/b",
            "user:a\\b",
            "",
            str(self.root),
            "__normal__",
        ):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.assistants.save_assistant("Bad", "", identity)
        identity, revision = self.assistants.save_assistant("Existing", "Keep")
        uuid = types.SimpleNamespace(hex=identity.removeprefix("user:assistant-"))
        with mock.patch.object(self.assistants.uuid, "uuid4", return_value=uuid):
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.assistants.save_assistant("Collision", "Replace")
        self.assertEqual(self.assistants.assistant_revision(identity), revision)

    def test_edit_preserves_character_fields_and_bundled_entries_are_copy_only(self):
        identity, _ = self.assistants.save_assistant("Cat", "Original")
        path = self.assistants.user_assistant_path(identity)
        path.write_text(
            json.dumps(
                {
                    "name": "Cat",
                    "system": "Original",
                    "greeting": "Meow",
                    "embed": [["text", "Fact"]],
                }
            ),
            encoding="utf-8",
        )
        self.assistants.save_assistant(
            "Cat", "Changed", identity, self.assistants.assistant_revision(identity)
        )
        self.assertEqual(
            self.assistants.assistant_info(identity)["chatstart"],
            [{"role": "assistant", "content": "Meow"}],
        )
        self.assertEqual(json.loads(path.read_text())["embed"], [["text", "Fact"]])
        bundled = next(
            value
            for _, value in self.assistants.assistant_choices()
            if value not in (identity, "__normal__")
        )
        info, editable, revision = self.assistants.assistant_edit_snapshot(bundled)
        self.assertTrue(info["name"])
        self.assertIsNone(editable)
        self.assertIsNone(revision)

    def test_arbitrary_json_is_not_loadable_as_an_assistant(self):
        path = self.root / "private.json"
        path.write_text(json.dumps({"name": "Secret", "system": "Private"}))
        self.assertEqual(self.assistants.assistant_info(str(path))["system"], "")


class AssistantSessionTests(unittest.TestCase):
    def test_save_refreshes_catalogue_without_switching_persona_or_erasing_drafts(self):
        namespace = {
            "save_assistant": mock.Mock(return_value=("user:saved", "revision")),
            "gr_llama_get_assistants": lambda selected: {"selected": selected},
            **{
                name: name
                for name in (
                    "llama_assistants",
                    "assistant_draft",
                    "assistant_version",
                    "assistant_status",
                )
            },
        }
        save = function_from("ruinedfooocus/features/chat/ui.py", "save_assistant_draft", namespace)
        with mock.patch.dict(
            sys.modules,
            {"ruinedfooocus.app.arguments": SimpleNamespace(args=SimpleNamespace(guest=False))},
        ):
            result = save("Name", "System", None, None, "__normal__")
        self.assertEqual(
            set(result),
            {"llama_assistants", "assistant_draft", "assistant_version", "assistant_status"},
        )
        self.assertEqual(result["llama_assistants"]["selected"], "__normal__")
        self.assertEqual(result["assistant_draft"], "user:saved")
        namespace["save_assistant"].side_effect = OSError("disk full")
        with mock.patch.dict(
            sys.modules,
            {"ruinedfooocus.app.arguments": SimpleNamespace(args=SimpleNamespace(guest=False))},
        ):
            self.assertEqual(
                save("Name", "System", None, None, "__normal__"), {"assistant_status": "disk full"}
            )

    def test_startup_choice_uses_profile_transaction_and_guest_cannot_publish(self):
        settings = SimpleNamespace(
            default_settings={"unrelated": "keep"}, name="owned", save_profile=mock.Mock()
        )
        namespace = {
            "settings": settings,
            "path_manager": SimpleNamespace(paths={"path_llm": ["models/llm"]}),
            "assistant_choices": lambda: [("Saved", "user:saved")],
            "translate_ui": lambda text: text,
        }
        save = function_from(
            "ruinedfooocus/features/chat/ui.py", "save_startup_assistant", namespace
        )
        launch = SimpleNamespace(guest=False)
        with mock.patch.dict(
            sys.modules, {"ruinedfooocus.app.arguments": SimpleNamespace(args=launch)}
        ):
            save("user:saved")
            settings.save_profile.assert_called_once_with(
                {"unrelated": "keep", "llama_assistant": "user:saved"},
                {"path_llm": ["models/llm"]},
                profile="owned",
            )
            self.assertEqual(settings.default_settings, {"unrelated": "keep"})
            settings.save_profile.reset_mock()
            launch.guest = True
            self.assertIn("Guest", save("user:saved"))
            settings.save_profile.assert_not_called()


class ChatbotDefinitionTests(unittest.TestCase):
    def bots(self):
        folders = sorted(
            path
            for path in (ROOT / "ruinedfooocus/resources/assistants").iterdir()
            if path.is_dir()
        )
        self.assertTrue(folders)
        return [
            (folder, json.loads((folder / "info.json").read_text(encoding="utf-8")))
            for folder in folders
        ]

    def test_each_bot_has_a_name_greeting_system_prompt_and_avatar(self):
        names = set()
        for folder, info in self.bots():
            with self.subTest(bot=folder.name):
                for key in ("name", "greeting", "system"):
                    self.assertIsInstance(info.get(key), str)
                    self.assertTrue(info[key].strip())
                self.assertTrue((folder / "avatar.png").is_file())
                self.assertNotIn(info["name"].casefold(), names)
                names.add(info["name"].casefold())

    def test_bundled_bots_work_offline_from_local_sources(self):
        for folder, info in self.bots():
            for source in info.get("embed", []):
                with self.subTest(bot=folder.name, source=source):
                    self.assertEqual(len(source), 2)
                    # Bundled bots must not depend on another repository or the network.
                    self.assertIn(source[0], ("text", "file"))
                    if source[0] == "file":
                        self.assertTrue((ROOT / source[1]).is_file())


def index_sources():
    from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary

    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
    )
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "index_sources")
    scope = {
        "asset_path": asset_path,
        "resource_path": resource_path,
        "Path": Path,
        "re": re,
        "force_cpu_for_auxiliary": force_cpu_for_auxiliary,
        "__file__": str(ROOT / "ruinedfooocus/features/chat/llama_pipeline.py"),
        "load_file_from_url": None,
        "url_to_filename": None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "llama_pipeline", "exec"), scope)
    return scope["index_sources"]


class ContextSourceTests(unittest.TestCase):
    def run_index(self, sources):
        indexed = []

        class Embeddings:
            def __init__(self, **kwargs):
                pass

            def index(self, documents):
                indexed.extend(documents)

            def close(self):
                pass

        stubs = {
            "txtai": types.SimpleNamespace(Embeddings=Embeddings),
            "ruinedfooocus.app.arguments": types.SimpleNamespace(
                args=SimpleNamespace(cpu=True, directml=None)
            ),
        }
        owner = SimpleNamespace()
        from ruinedfooocus.app import resource_lifecycle as life
        from ruinedfooocus.features.chat import rag_preparation

        item = life.JobRegistry().create()
        prepared = SimpleNamespace(model=ROOT, artifacts=(), settings={"gpu": False})
        with (
            mock.patch.dict(sys.modules, stubs),
            mock.patch.object(
                rag_preparation, "runtime_options", return_value={"content": True, "gpu": False}
            ),
            mock.patch.object(life, "admission", life.FairAdmission()),
            life.job_scope(item),
            life.admission.lease(item),
        ):
            index_sources()(owner, sources, prepared_model=prepared)
            owner._rag_leases.close()
        return indexed

    def test_file_sources_are_read_from_the_app_folder_and_split_into_passages(self):
        docs = self.run_index(
            [["file", "ruinedfooocus/resources/assistants/rf_support_troll/knowledge.md"]]
        )
        self.assertGreater(len(docs), 5)
        self.assertTrue(any("MetadataIncompleteBuffer" in doc for doc in docs))

    def test_file_sources_outside_the_app_folder_or_missing_are_rejected(self):
        for path in ("../outside.md", "chatbots/missing.md"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_index([["file", path]])


if __name__ == "__main__":
    unittest.main()
