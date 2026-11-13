"""Checkpoint choices stay local, immutable at submission and conservative in UI."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ruinedfooocus.app
from ruinedfooocus.features.models import component_profiles as profiles
from ruinedfooocus.features.models import component_candidates as candidates
from ruinedfooocus.core.settings_schema import validate_known


def header(path, tensors):
    payload = json.dumps(tensors).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(payload).to_bytes(8, "little") + payload)
    return path


class ComponentProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = header(self.root / "a.safetensors", {})
        self.b = header(self.root / "b.safetensors", {})
        self.paths = SimpleNamespace(
            paths={
                "path_clip": [str(self.root / "encoders")],
                "path_vae": [str(self.root / "vae")],
            },
            get_abspath=lambda value: Path(value).resolve(),
            DOWNLOADABLE_FILES={},
            EXTENSIONS=[".safetensors", ".gguf"],
        )
        self.values = {"clip_l": "legacy.safetensors", "theme": "keep", "prompt": "keep"}

    def test_checkpoint_scopes_keep_legacy_choices_and_null_resets_only_one_role(self):
        values = profiles.update_choices(
            self.values, self.a, profiles.checkpoint_identity(self.a), {"clip_l": None}
        )
        values = profiles.update_choices(
            values, self.b, profiles.checkpoint_identity(self.b), {"clip_l": "second.safetensors"}
        )
        self.assertEqual(values["clip_l"], "legacy.safetensors")
        self.assertEqual(values["theme"], "keep")
        self.assertEqual(
            profiles.freeze_choices(self.a, "SDXL", values, self.paths)["clip_l"], None
        )
        self.assertEqual(
            profiles.freeze_choices(self.b, "SDXL", values, self.paths)["clip_l"],
            "second.safetensors",
        )
        self.assertNotIn(profiles.PROFILE_KEY, self.values)
        self.assertEqual(validate_known(values), values)

    def test_sft_submission_freezes_the_same_component_roles(self):
        model = header(self.root / "image.SFT", {})
        models = SimpleNamespace(
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
            get_model_base=lambda value: value["baseModel"],
            get_file=lambda *args: model,
        )
        result = profiles.snapshot_request({"base_model_name": model.name}, models, {}, self.paths)
        self.assertEqual(
            result[profiles.REQUEST_KEY], {"clip_l": None, "clip_g": None, "vae_sdxl": None}
        )

    def test_literal_library_json_cannot_replace_the_explicit_checkpoint_or_components(self):
        lookup = Mock(return_value={"baseModel": "SDXL"})
        models = SimpleNamespace(
            get_models_by_path=lookup,
            get_model_base=lambda value: value["baseModel"],
            get_file=lambda *args: self.a,
        )
        data = dict(
            base_model_name=self.a.name,
            prompt=json.dumps(
                {
                    "prompt": "nested",
                    "base_model_name": self.b.name,
                    "image_component_overrides": {"clip_l": "wrong.safetensors"},
                }
            ),
            _metadata_prompt=True,
            image_component_overrides={"clip_l": None},
        )
        result = profiles.snapshot_request(data, models, {}, self.paths)
        self.assertEqual(result["base_model_name"], self.a.name)
        self.assertIsNone(result[profiles.REQUEST_KEY]["clip_l"])
        self.assertEqual(lookup.call_args.args[1], self.a.name)

    def test_roles_follow_architecture_and_bundles_are_locked(self):
        expected = {
            "SDXL 1.0": {"clip_l", "clip_g", "vae_sdxl"},
            "Flux.1 D": {"clip_l", "clip_t5", "vae_flux"},
            "Anima": {"clip_qwen3_06b", "vae_qwen_image"},
            "HiDream O1": set(),
        }
        for family, keys in expected.items():
            self.assertEqual(
                set(profiles.freeze_choices(self.a, family, self.values, self.paths)), keys
            )
        bundled = header(
            self.root / "bundle.safetensors",
            {"conditioner.embedders.0.x": {}, "first_stage_model.x": {}},
        )
        self.assertTrue(
            all(
                value is None
                for value in profiles.freeze_choices(
                    bundled, "SDXL", self.values, self.paths
                ).values()
            )
        )
        with self.assertRaisesRegex(ValueError, "built into"):
            profiles.freeze_choices(
                bundled, "SDXL", self.values, self.paths, {"clip_l": "other.safetensors"}
            )

    def test_replacement_changes_scope_and_replay_survives_change(self):
        mapping, scope = profiles.replay_choices(
            self.a, self.values, {"clip_l": "recorded.safetensors"}
        )
        self.assertEqual(
            profiles.session_choices(self.a, self.values, mapping, scope), (mapping, scope)
        )
        self.assertEqual(profiles.session_choices(self.b, self.values, mapping, scope)[0], {})
        header(self.a, {"new": {"shape": [1]}})
        self.assertNotEqual(profiles.checkpoint_identity(self.a), scope)
        with self.assertRaisesRegex(ValueError, "changed"):
            profiles.update_choices(self.values, self.a, scope, mapping)

    def models(self):
        return SimpleNamespace(
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
            get_model_base=lambda value: value["baseModel"],
            get_file=lambda *args: self.a,
        )

    def test_snapshot_includes_inherited_globals_and_copies_partial_mapping(self):
        mapping = {"clip_g": None}
        request = {"base_model_name": "a.safetensors", "image_component_overrides": mapping}
        snapshot = profiles.snapshot_request(request, self.models(), self.values, self.paths)
        self.values["clip_l"] = "later.safetensors"
        mapping["clip_g"] = "later.safetensors"
        self.assertEqual(
            snapshot[profiles.REQUEST_KEY],
            {"clip_l": "legacy.safetensors", "clip_g": None, "vae_sdxl": None},
        )
        self.assertIsNot(snapshot[profiles.REQUEST_KEY], mapping)

    def test_api_and_tool_prompts_skip_metadata_parsing_and_keep_explicit_choices(self):
        for boundary in ({"_api_job": object()}, {"task_type": "tool_call"}):
            with self.subTest(boundary=tuple(boundary)):
                request = dict(
                    boundary,
                    base_model_name="a.safetensors",
                    prompt=json.dumps(
                        {"prompt": "literal", "base_model_name": "missing.safetensors"}
                    ),
                    image_component_overrides={"clip_l": "chosen.safetensors", "clip_g": None},
                )
                with patch("ruinedfooocus.features.gallery.image_metadata.read_metadata") as read:
                    snapshot = profiles.snapshot_request(
                        request, self.models(), self.values, self.paths
                    )
                read.assert_not_called()
                self.assertEqual(snapshot["base_model_name"], request["base_model_name"])
                self.assertEqual(snapshot["prompt"], request["prompt"])
                self.assertEqual(
                    snapshot[profiles.REQUEST_KEY],
                    {"clip_l": "chosen.safetensors", "clip_g": None, "vae_sdxl": None},
                )

    def test_json_prompt_uses_recorded_mapping_before_queue(self):
        data = {
            "base_model_name": "b.safetensors",
            "prompt": json.dumps(
                {
                    "Prompt": "saved",
                    "base_model_name": "a.safetensors",
                    profiles.REQUEST_KEY: {"clip_l": None},
                }
            ),
        }
        snapshot = profiles.snapshot_request(data, self.models(), self.values, self.paths)
        self.assertEqual(snapshot["base_model_name"], "a.safetensors")
        self.assertIsNone(snapshot[profiles.REQUEST_KEY]["clip_l"])

    def test_preset_model_mode_applies_recorded_or_saved_choices_for_actual_model(self):
        values = profiles.update_choices(
            self.values,
            self.a,
            profiles.checkpoint_identity(self.a),
            {"clip_l": "saved-a.safetensors"},
        )
        values = profiles.update_choices(
            values, self.b, profiles.checkpoint_identity(self.b), {"clip_l": "saved-b.safetensors"}
        )
        models = self.models()
        models.get_file = lambda kind, name: self.a if name == "a" else self.b
        original = {"base_model_name": "b", profiles.REQUEST_KEY: {"clip_l": "stale-a.safetensors"}}
        request = profiles.apply_preset_choices(original, {}, models, values)
        self.assertEqual(request[profiles.REQUEST_KEY], {"clip_l": "saved-b.safetensors"})
        snapshot = profiles.snapshot_request(request, models, values, self.paths)
        self.assertEqual(snapshot[profiles.REQUEST_KEY]["clip_l"], "saved-b.safetensors")
        request = profiles.apply_preset_choices(
            dict(original, base_model_name="a"),
            {profiles.REQUEST_KEY: {"clip_l": None}},
            models,
            values,
        )
        self.assertEqual(request[profiles.REQUEST_KEY], {"clip_l": None})
        self.assertEqual(request[profiles.SCOPE_KEY], profiles.checkpoint_identity(self.a))
        self.assertEqual(original[profiles.REQUEST_KEY], {"clip_l": "stale-a.safetensors"})

    def test_stale_ui_scope_is_ignored_but_explicit_api_and_metadata_choices_survive(self):
        request = {
            "base_model_name": "a",
            profiles.REQUEST_KEY: {"clip_l": None},
            profiles.SCOPE_KEY: profiles.checkpoint_identity(self.b),
        }
        self.assertEqual(
            profiles.snapshot_request(request, self.models(), self.values, self.paths)[
                profiles.REQUEST_KEY
            ]["clip_l"],
            "legacy.safetensors",
        )
        request.pop(profiles.SCOPE_KEY)
        self.assertIsNone(
            profiles.snapshot_request(request, self.models(), self.values, self.paths)[
                profiles.REQUEST_KEY
            ]["clip_l"]
        )
        request[profiles.SCOPE_KEY] = profiles.checkpoint_identity(self.b)
        request["prompt"] = json.dumps({"Prompt": "saved", profiles.REQUEST_KEY: {"clip_l": None}})
        self.assertIsNone(
            profiles.snapshot_request(request, self.models(), self.values, self.paths)[
                profiles.REQUEST_KEY
            ]["clip_l"]
        )

    def test_auxiliary_and_command_requests_do_not_require_image_components(self):
        models = SimpleNamespace(
            get_models_by_path=Mock(side_effect=AssertionError("not an image load"))
        )
        for data in (
            {"task_type": "llama"},
            {"controlnet": {"type": "rembg"}},
            {"controlnet": {"type": "upscale"}},
            {"prompt": "search: apples"},
            {"prompt": "#!command"},
        ):
            request = dict(data, base_model_name="missing.safetensors")
            self.assertEqual(profiles.snapshot_request(request, models, {}, self.paths), request)

    def test_bad_settings_rejected_without_tensor_imports(self):
        for value in ([], {"bad": {}}, {"a" * 64: {"unknown": "x"}}):
            with self.assertRaises(ValueError):
                validate_known({profiles.PROFILE_KEY: value})

    def test_pickle_snapshot_retains_legacy_bundle_behavior_without_reading_weights(self):
        checkpoint = self.root / "legacy.ckpt"
        checkpoint.write_bytes(b"inert pickle substitute; must not be opened")
        models = self.models()
        models.get_file = lambda *args: checkpoint
        with patch.object(
            profiles, "component_inventory", side_effect=AssertionError("no safe header")
        ):
            result = profiles.snapshot_request(
                {"base_model_name": "legacy.ckpt", profiles.REQUEST_KEY: None},
                models,
                self.values,
                self.paths,
            )
            self.assertNotIn(profiles.REQUEST_KEY, result)
            explicit = {"clip_l": "explicit.safetensors"}
            result = profiles.snapshot_request(
                {"base_model_name": "legacy.ckpt", profiles.REQUEST_KEY: explicit},
                models,
                self.values,
                self.paths,
            )
        self.assertEqual(result[profiles.REQUEST_KEY], explicit)
        self.assertIsNot(result[profiles.REQUEST_KEY], explicit)
        self.assertEqual(self.values["clip_l"], "legacy.safetensors")

    def test_local_candidates_use_native_shape_classifier_and_custom_roots(self):
        header(
            self.root / "encoders" / "nested" / "renamed.safetensors",
            {"text_model.encoder.layers.0.mlp.fc1.weight": {"shape": [3072, 768]}},
        )
        header(self.root / "encoders" / "misleading-clip-l.safetensors", {"other": {"shape": [1]}})
        detector = lambda tensors: (
            SimpleNamespace(name="CLIP_L")
            if "text_model.encoder.layers.0.mlp.fc1.weight" in tensors
            else SimpleNamespace(name="CLIP_G")
        )
        with patch.dict("sys.modules", {"comfy.sd": SimpleNamespace(detect_te_model=detector)}):
            rows = candidates.installed_candidates("clip_l", self.paths)
            self.assertEqual(
                {row["name"]: row["compatibility"] for row in rows},
                {
                    "misleading-clip-l.safetensors": "incompatible",
                    "nested/renamed.safetensors": "compatible",
                },
            )
            self.assertEqual(
                candidates.validate_local_choice(
                    "clip_l", "nested/renamed.safetensors", self.paths
                ),
                "nested/renamed.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "different"):
                candidates.validate_local_choice(
                    "clip_l", "misleading-clip-l.safetensors", self.paths, allow_unknown=True
                )
        with patch.dict("sys.modules", {"comfy.sd": None}):
            with self.assertRaisesRegex(ValueError, "unverified"):
                candidates.validate_local_choice("clip_l", "nested/renamed.safetensors", self.paths)
            self.assertEqual(
                candidates.validate_local_choice(
                    "clip_l", "nested/renamed.safetensors", self.paths, allow_unknown=True
                ),
                "nested/renamed.safetensors",
            )
        with self.assertRaisesRegex(ValueError, "no longer installed"):
            candidates.validate_local_choice("clip_l", "gone.safetensors", self.paths)

    def test_candidate_headers_invalidate_on_replacement(self):
        path = header(self.root / "x.safetensors", {"a": {"shape": [1]}})
        self.assertEqual(candidates.header_shapes(path), {"a": (1,)})
        header(path, {"different": {"shape": [2]}})
        self.assertEqual(candidates.header_shapes(path), {"different": (2,)})

    def test_vision_roles_use_vision_roots_instead_of_text_encoder_roots(self):
        self.paths.paths["path_clip_vision"] = [str(self.root / "vision")]
        header(self.root / "vision" / "vision.safetensors", {})
        header(self.root / "encoders" / "text.safetensors", {})
        with patch.dict("sys.modules", {"comfy.sd": None}):
            for role in ("clip_vision", "clip_hunyuan_vision"):
                with self.subTest(role=role):
                    self.assertEqual(
                        [row["name"] for row in candidates.installed_candidates(role, self.paths)],
                        ["vision.safetensors"],
                    )

    def test_saving_one_role_preserves_another_sessions_saved_role(self):
        import runpy
        from ruinedfooocus.features.models import ui_model_components as ui

        identity = profiles.checkpoint_identity(self.a)
        self.values = profiles.update_choices(
            self.values, self.a, identity, {"clip_g": "newer-session.safetensors"}
        )
        settings = SimpleNamespace(default_settings=self.values, name="profile")

        def save_profile(values, paths, *, profile):
            settings.default_settings = values

        settings.save_profile = Mock(side_effect=save_profile)
        shared = SimpleNamespace(settings=settings, path_manager=self.paths)
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ui, "selected", return_value=(self.a, "SDXL")),
        ):
            namespace = runpy.run_path(
                str(
                    Path(__file__).resolve().parents[1]
                    / "ruinedfooocus/features/models/ui_component_overrides.py"
                )
            )
            choices, scope = namespace["save_choice"](
                "model", "clip_l", "", False, {"clip_g": "older-session.safetensors"}, identity
            )
        self.assertEqual(
            profiles.saved_choices(settings.default_settings, self.a),
            {"clip_g": "newer-session.safetensors", "clip_l": None},
        )
        self.assertEqual(choices["clip_g"], "older-session.safetensors")
        self.assertEqual(scope, identity)
        self.assertEqual(settings.default_settings["clip_l"], "legacy.safetensors")
        settings.save_profile.assert_called_once_with(
            settings.default_settings, self.paths.paths, profile="profile"
        )


class ComponentDownloadTests(unittest.TestCase):
    def test_component_panel_constructs_complete_state_and_event_outputs(self):
        import gradio as gr
        from ruinedfooocus.features.models import ui_model_components as ui

        controls = {}
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={}),
            path_manager=SimpleNamespace(),
            add_ctrl=lambda key, control: controls.update({key: control}),
        )
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            gr.Blocks(analytics_enabled=False) as demo,
        ):
            base = gr.Dropdown(choices=["fixture.safetensors"])
            metadata_refresh = gr.Number(value=0, visible=False)
            ui.create_ui(base, model_refresh=metadata_refresh)
        self.assertEqual(set(controls), {"image_component_overrides", "_image_component_scope"})
        config = demo.get_config_file()
        refreshes = [item for item in config["dependencies"] if len(item["outputs"]) == 7]
        self.assertGreaterEqual(len(refreshes), 4)
        self.assertTrue(
            any(
                [metadata_refresh._id, "change"] in [list(target) for target in item["targets"]]
                for item in refreshes
            )
        )
        explicit_refresh = next(fn for fn in demo.fns.values() if fn.fn is ui.refresh_selected_role)
        self.assertEqual(len(explicit_refresh.inputs), 4)
        self.assertIs(explicit_refresh.inputs[-1], explicit_refresh.outputs[-1])

    def test_download_uses_preparation_destination_and_indeterminate_progress(self):
        from ruinedfooocus.features.models import ui_model_components as ui

        row = {
            "download_available": True,
            "download_key": "clip_l",
            "filename": "local.safetensors",
            "location": "owned/local.safetensors",
        }
        manager = SimpleNamespace(prepare_file=Mock())
        progress = Mock()

        def prepare(key, progress, destination):
            progress(4, None, 0)
            return SimpleNamespace(result=lambda: None, cancel=Mock())

        manager.prepare_file.side_effect = prepare
        shared = SimpleNamespace(
            path_manager=manager, settings=SimpleNamespace(default_settings={})
        )
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ui, "selected", return_value=("checkpoint", "SDXL")),
            patch.object(ui, "session_choices", return_value=({}, "identity")),
            patch.object(ui, "component_inventory", return_value={"rows": [row]}),
            patch.object(ui, "offline", return_value=False),
            patch.object(ui, "refresh", side_effect=lambda *args: args),
        ):
            result = ui.download_selected(
                "model", {}, "identity", SimpleNamespace(session_hash="session-a"), progress
            )
        self.assertIn("Downloaded", result[-1])
        manager.prepare_file.assert_called_once()
        self.assertEqual(manager.prepare_file.call_args.kwargs["destination"], row["location"])
        self.assertTrue(
            any(
                call.args[0] is None and "total unknown" in call.kwargs.get("desc", "")
                for call in progress.call_args_list
            )
        )

    def test_cancel_only_owns_current_session_subscription(self):
        import threading
        from ruinedfooocus.features.models import ui_model_components as ui

        a = {"cancelled": threading.Event(), "handle": Mock()}
        b = {"cancelled": threading.Event(), "handle": Mock()}
        with patch.dict(ui._downloads, {"session-a": a, "session-b": b}, clear=True):
            ui.cancel_download(SimpleNamespace(session_hash="session-a"))
        self.assertTrue(a["cancelled"].is_set())
        a["handle"].cancel.assert_called_once()
        self.assertFalse(b["cancelled"].is_set())
        b["handle"].cancel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
