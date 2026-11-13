"""Catalogue UI uses existing cards, owner Settings and explicit activation."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

import test_model_acquisition as fixtures
from ruinedfooocus.features.models.ui_acquisition import create_ui, detail_rows

ROOT = Path(__file__).resolve().parents[1]


class AcquisitionUiTests(unittest.TestCase):
    setUp = fixtures.AcquisitionTests.setUp
    installed = fixtures.AcquisitionTests.installed

    def test_actual_view_offline_filter_selection_and_job_are_independent(self):
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(
                    self.manager, SimpleNamespace(update_all_models=Mock()), offline=lambda: True
                )
            result = ui.refresh_rows("", "All", "discover", "All", "All", None, 1)
            self.assertEqual(len(result[0]), 2)
            details = next(
                fn.fn
                for fn in app.fns.values()
                if getattr(fn.fn, "__name__", "") == "selected_details"
            )
            table, text, download, activate = details("checkpoints/a.safetensors", result[0])
            self.assertFalse(download["interactive"])
            self.assertFalse(activate["interactive"])
            self.assertIn("Available", text)
            self.assertIn(["License", "Test license"], table)
            filtered = next(
                fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "filter_rows"
            )
            self.assertEqual(filtered("", "All", "discover", "All", "All", None, 1)[-1], gr.skip())
            start = next(
                fn.fn
                for fn in app.fns.values()
                if getattr(fn.fn, "__name__", "") == "start_download"
            )
            with self.assertRaisesRegex(gr.Error, "Offline"):
                start("checkpoints/a.safetensors", Mock(), True, "text", 0, False)
            actions = next(
                fn.fn
                for fn in app.fns.values()
                if getattr(fn.fn, "__name__", "") == "select_record"
            )
            with self.assertRaises(gr.Error):
                actions(result[0], "text", True, gr.EventData(None, {"key": "stale"}))
            clicked = actions(
                result[0], "text", True, gr.EventData(None, {"key": "checkpoints/a.safetensors"})
            )
            self.assertEqual(clicked[0], "checkpoints/a.safetensors")
            self.assertIn("Available", clicked[2])
            self.assertTrue(clicked[1])
            # Timer and cancel do not write selection, filters or the active model.
            for fn in app.fns.values():
                if getattr(fn.fn, "__name__", "") in ("poll", "cancel_download"):
                    self.assertNotIn(ui.selection, fn.outputs)

    def test_activation_progress_uses_existing_catalogue_status_and_keeps_data_outputs(self):
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(self.manager, SimpleNamespace(), offline=lambda: True)
                self.assertIsInstance(ui.status, gr.Markdown)
                self.assertTrue(any(ui.status in function.outputs for function in app.fns.values()))
                base = gr.Textbox("Current model")
                overrides, component_scope, chat_selection = (gr.State(None) for _ in range(3))
                destination = gr.State("create")
                model_destination = gr.State(None)
                action = gr.State("generate")
                tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
                call = next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "click"
                    and isinstance(node.func.value, ast.Attribute)
                    and ast.unparse(node.func.value) == "app_settings.acquisition.activate"
                )
                namespace = dict(
                    app_settings=SimpleNamespace(acquisition=ui),
                    activate_destination=Mock(return_value={base: "Chosen model"}),
                    base_model=base,
                    component_overrides=overrides,
                    component_scope=component_scope,
                    app_llama_chat=SimpleNamespace(catalogue_selection_inputs=[chat_selection]),
                    activation_outputs=[base, overrides, component_scope],
                    shell=SimpleNamespace(
                        outputs=[destination], model_destinations=model_destination
                    ),
                    editor_workspace=SimpleNamespace(action=action),
                )
                dependency = eval(
                    compile(ast.Expression(call), "activation binding", "eval"), namespace
                )
            self.assertTrue(dependency["backend_fn"])
            self.assertEqual(dependency["show_progress"], "hidden")
            self.assertEqual(dependency["show_progress_on"], [ui.status._id])
            self.assertEqual(
                dependency["outputs"],
                [
                    control._id
                    for control in (
                        base,
                        overrides,
                        component_scope,
                        destination,
                        model_destination,
                        action,
                    )
                ],
            )

    def test_real_navigation_and_activation_callbacks_do_not_output_prompts(self):
        source = (ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        activation = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "activate_catalogue_checkpoint"
        )
        local = self.installed("checkpoints", "a.safetensors")
        models = SimpleNamespace(get_names=lambda _: ["a.safetensors"], get_file=lambda *_: local)
        scope = dict(
            gr=gr,
            chat_catalogue_outputs=[],
            tool_catalogue_outputs=[],
            path_manager=self.manager,
            shared=SimpleNamespace(models=models),
            app_settings=SimpleNamespace(
                acquisition=SimpleNamespace(
                    catalogue=self.catalogue,
                    checkpoint_name=__import__(
                        "ruinedfooocus.features.models.acquisition", fromlist=["checkpoint_name"]
                    ).checkpoint_name,
                    checkpoint_choices=lambda names, current: [(name, name) for name in names],
                )
            ),
        )
        exec(
            compile(
                ast.Module(body=[activation], type_ignores=[]),
                str(ROOT / "ruinedfooocus/ui/app.py"),
                "exec",
            ),
            scope,
        )
        row = self.catalogue.record("checkpoints/a.safetensors")
        result = scope["activate_catalogue_checkpoint"](row["key"], [row])
        self.assertEqual(result[0]["value"], "a.safetensors")
        self.assertEqual(result[1]["selected"], "create")
        self.assertEqual(result[2]["selected"], "generation")
        self.assertNotIn("prompt", ast.unparse(activation))
        local.write_bytes(b"edit")
        with self.assertRaisesRegex(gr.Error, "changed"):
            scope["activate_catalogue_checkpoint"](row["key"], [row])
        settings = (ROOT / "ruinedfooocus/ui/ui_settings.py").read_text()
        self.assertIn('id="models"', settings)
        self.assertIn('id="settings"', source)

    def test_metadata_rows_distinguish_size_mismatch_and_unknown_compatibility(self):
        self.installed("checkpoints", "a.safetensors", b"wrong-size")
        table, status, download, activate = detail_rows(
            self.catalogue.record("checkpoints/a.safetensors")
        )
        self.assertIn("size mismatch", status)
        self.assertFalse(download)
        self.assertFalse(activate)
        self.assertIn(["Memory suitability", "No workflow memory estimate recorded."], table)

    def test_selected_status_uses_card_title_and_retains_exact_file_in_details(self):
        row = self.catalogue.record("checkpoints/a.safetensors")
        row["label"] = "crystal_Dream_v1.safetensors"
        table, status, _, _ = detail_rows(row)
        self.assertTrue(status.startswith("**crystal Dream v1**"))
        self.assertIn(["File", row["destination"]], table)
        row["label"] = "Qwen Image 2.1 Turbo"
        self.assertTrue(detail_rows(row)[1].startswith("**Qwen Image 2\\.1 Turbo**"))

    def test_installed_component_use_changes_only_its_current_checkpoint_session_choice(self):
        from ruinedfooocus.features.models import ui_model_components

        source = ROOT / "ruinedfooocus/ui/app.py"
        activation = next(
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "activate_catalogue_checkpoint"
        )
        checkpoint = self.installed("checkpoints", "a.safetensors")
        encoder = self.installed("clip", "encoder.safetensors")
        row = self.catalogue.record("clip/encoder.safetensors")
        use = Mock(return_value=({"clip_l": "encoder.safetensors"}, "checkpoint-scope"))
        values = {"prompt": "keep exactly"}
        scope = dict(
            gr=gr,
            chat_catalogue_outputs=[],
            tool_catalogue_outputs=[],
            path_manager=self.manager,
            shared=SimpleNamespace(
                models=SimpleNamespace(), settings=SimpleNamespace(default_settings=values)
            ),
            app_settings=SimpleNamespace(
                acquisition=SimpleNamespace(catalogue=self.catalogue, use_component=use)
            ),
        )
        exec(compile(ast.Module(body=[activation], type_ignores=[]), str(source), "exec"), scope)
        with patch.object(ui_model_components, "selected", return_value=(checkpoint, "SDXL")):
            result = scope["activate_catalogue_checkpoint"](
                row["key"],
                [row],
                "a.safetensors",
                "clip_l",
                {"clip_g": "keep.gguf"},
                "checkpoint-scope",
            )
        self.assertEqual(result[0], gr.skip())
        self.assertEqual(
            (result[1]["selected"], result[2]["selected"]), ("models", "create-models")
        )
        self.assertEqual(result[3:5], ({"clip_l": "encoder.safetensors"}, "checkpoint-scope"))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(value == gr.skip() for value in result[5:]))
        use.assert_called_once_with(
            row,
            "clip_l",
            checkpoint,
            "SDXL",
            values,
            self.manager,
            {"clip_g": "keep.gguf"},
            "checkpoint-scope",
        )
        self.assertEqual(values, {"prompt": "keep exactly"})
        encoder.write_bytes(b"changed")
        with self.assertRaisesRegex(gr.Error, "changed"):
            scope["activate_catalogue_checkpoint"](row["key"], [row], "a.safetensors", "clip_l")
        self.assertEqual(use.call_count, 1)

    def test_installed_chat_activation_routes_to_candidate_adapter_without_loading(self):
        source = ROOT / "ruinedfooocus/ui/app.py"
        activation = next(
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "activate_catalogue_checkpoint"
        )
        path = self.installed("checkpoints", "a.safetensors")
        row = {
            **self.catalogue.record("checkpoints/a.safetensors"),
            "task": "Chat",
            "category": "llm",
        }
        select = Mock(return_value={"candidate": str(path), "source": "Local models"})
        session = SimpleNamespace(active="keep loaded model")
        scope = dict(
            gr=gr,
            shared=SimpleNamespace(models=SimpleNamespace()),
            app_settings=SimpleNamespace(
                acquisition=SimpleNamespace(catalogue=SimpleNamespace(record=lambda _: row))
            ),
            app_llama_chat=SimpleNamespace(select_catalogue_file=select),
            chat_catalogue_outputs=["candidate", "source"],
            tool_catalogue_outputs=[],
        )
        exec(compile(ast.Module(body=[activation], type_ignores=[]), str(source), "exec"), scope)
        result = scope["activate_catalogue_checkpoint"](row["key"], [row], chat_session=session)
        self.assertEqual(result[1]["selected"], "chat")
        self.assertEqual(result[5:7], (str(path), "Local models"))
        self.assertEqual(len(result), 7)
        self.assertTrue(all(value == gr.skip() for value in result[7:]))
        select.assert_called_once_with(path, session)
        self.assertEqual(session.active, "keep loaded model")

    def test_selected_action_states_keep_other_installed_models_usable_and_offer_resume(self):
        from ruinedfooocus.features.models.acquisition import AcquisitionSession, DownloadPlan

        self.installed("checkpoints", "b.safetensors")
        models = SimpleNamespace(
            update_all_models=Mock(), detect_checkpoint_base=Mock(return_value="SDXL")
        )
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(self.manager, models, offline=lambda: True)
            rows = ui.refresh_rows("", "All", "on-device", "All", "All", None, 1)[0]
            missing = self.catalogue.record("checkpoints/a.safetensors")
            rows.append(missing)
            details = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "selected_details")
            job = AcquisitionSession()
            job.plan = DownloadPlan("Model A", (), selection_key=missing["key"])
            job.active, job.state = True, "running"
            self.assertFalse(details(missing["key"], rows, job=job)[2]["visible"])
            local_b = next(row for row in rows if row["key"].startswith("local:"))
            state = details(local_b["key"], rows, job=job)
            self.assertTrue(state[3]["visible"])
            self.assertFalse(state[2]["visible"])
            job.cancel()
            self.assertEqual(details(missing["key"], rows, job=job)[2]["value"], "Retry download")
            with patch(
                "ruinedfooocus.features.models.ui_acquisition.preparation_service.transfer_status",
                return_value=dict(
                    state="partial",
                    received_bytes=2,
                    resumable=True,
                    reason="Partial bytes not verified.",
                ),
            ):
                state = details(missing["key"], rows, job=job)
            self.assertEqual(state[2]["value"], "Resume download")
            self.assertFalse(state[2]["interactive"])
            self.assertIn("Partial bytes not verified.", state[1])

    def test_custom_html_contains_localized_text_and_preserves_publisher_decimal_versions(self):
        from ruinedfooocus.features.models.ui_acquisition import card_data
        from ruinedfooocus.core.translation_manager import localize

        row = self.catalogue.record("checkpoints/a.safetensors")
        row["label"] = "Qwen Image 2.1 Turbo"
        for locale in ("en", "uk", "de", "nonexistent"):
            payload, page = card_data([row], row["key"], self.manager, locale=locale)
            self.assertNotIn("__i18n__", json.dumps(payload))
            self.assertEqual(payload["heading"], localize("Models and downloads", locale))
            self.assertEqual(payload["defaultPreviewLabel"], localize("Default preview", locale))
            self.assertEqual(payload["cards"][0]["label"], "Qwen Image 2.1 Turbo")
            self.assertEqual(page, 1)
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                create_ui(self.manager, SimpleNamespace(), offline=lambda: True)
        table = next(block for block in app.blocks.values() if isinstance(block, gr.Dataframe))
        self.assertNotIn("__i18n__", json.dumps(table.headers))

    def test_artwork_completion_sends_only_identity_fenced_preview_patches(self):
        from ruinedfooocus.features.models.catalogue_previews import preview_identity

        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(self.manager, SimpleNamespace(), offline=lambda: True)
        self.addCleanup(ui.preview_cache.close)
        row = self.catalogue.record("checkpoints/a.safetensors")
        row["annotation"]["artifact"].update(
            pinned_url="https://huggingface.co/owner/model/resolve/" + "a" * 40 + "/weight.gguf",
            sha256="b" * 64,
        )
        identity = preview_identity(row)
        path = ui.preview_cache.directory / (identity["key"] + ".png")
        path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), "blue").save(path)
        ui.preview_cache.revision = 1
        refresh = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "refresh_artwork")
        revision, payload = refresh(0, [row], row["key"], 1)
        self.assertEqual(revision, 1)
        self.assertEqual(set(payload), {"previewPatch"})
        self.assertEqual(payload["previewPatch"][0]["artworkKey"], identity["key"])
        self.assertEqual(payload["previewPatch"][0]["key"], row["key"])
        self.assertEqual(refresh(1, [row], row["key"], 1), (gr.skip(), gr.skip()))

    def test_download_completion_hands_verified_signature_to_proof_cache_without_rehashing(self):
        from ruinedfooocus.core.cache_contract import file_identity
        from ruinedfooocus.features.models import model_identity

        models = SimpleNamespace(update_all_models=Mock(), revision=0)
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(self.manager, models, offline=lambda: True)
        self.addCleanup(ui.preview_cache.close)
        path = self.installed("checkpoints", "a.safetensors")
        request = self.manager.preparation_request("checkpoints/a.safetensors")
        signature = file_identity(path)
        job = SimpleNamespace(
            active=True,
            poll=lambda: ("Ready", False, True),
            drain_verified=lambda: [(request, path, signature)],
        )
        poll = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "poll")
        with (
            patch.object(
                model_identity, "record_prepared_native_source", return_value={"baseModel": "SDXL"}
            ) as record,
            patch.object(
                model_identity, "cached_file_sha256", side_effect=AssertionError("no caller rehash")
            ),
        ):
            result = poll(job, 0)
        record.assert_called_once_with(
            path, request, signature, self.manager.model_paths["cache_path"]
        )
        self.assertEqual(result[0], "Ready")
        self.assertEqual(models.revision, 1)
        models.update_all_models.assert_called_once()
        with patch.object(
            model_identity, "record_prepared_native_source", side_effect=OSError("cache full")
        ):
            result = poll(job, 0)
        self.assertIn("Downloaded files are retained", result[0])
        self.assertEqual(path.read_bytes(), b"unit")

    def test_real_card_event_updates_visible_details_and_native_local_activation_atomically(self):
        import asyncio
        from gradio.state_holder import SessionState
        from ruinedfooocus.features.models.ui_acquisition import card_data

        local = self.installed("checkpoints", "UI ONLY Lens.safetensors")
        models = SimpleNamespace(
            update_all_models=Mock(), detect_checkpoint_base=Mock(return_value="SDXL")
        )
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(self.manager, models, offline=lambda: True)
            refreshed = ui.refresh_rows("Lens", "All", "on-device", "All", "All", None, 1)
            row = refreshed[0][0]
            state = SessionState(app)
            state[ui.records._id] = refreshed[0]
            index, event = next(
                (index, fn)
                for index, fn in app.fns.items()
                if getattr(fn.fn, "__name__", "") == "select_record"
            )

            async def choose():
                result = await app.process_api(
                    index,
                    [None, "text", True, None, None, 1],
                    state=state,
                    event_data=gr.EventData(None, {"key": row["key"]}),
                )
                self.assertIn("SDXL (local header)", result["data"][2])
                self.assertTrue(result["data"][4]["interactive"])
                self.assertTrue(result["data"][1]["data"])

            asyncio.run(choose())
            models.detect_checkpoint_base.assert_called_with(local)
            cards, _ = card_data(refreshed[0], row["key"], self.manager)
            self.assertEqual(cards["cards"][0]["state"], "Custom")

    def test_discover_and_on_device_keep_the_exact_selection_across_search_and_refresh(self):
        import asyncio
        from gradio.state_holder import SessionState

        key = "checkpoints/a.safetensors"
        custom = self.installed("checkpoints", "other/a.safetensors")
        with (
            patch(
                "ruinedfooocus.features.models.acquisition.load_registry",
                return_value=self.registry,
            ),
            patch.object(
                self.manager,
                "prepare_file",
                side_effect=AssertionError("browsing must not prepare a model"),
            ),
            patch(
                "ruinedfooocus.features.models.ui_acquisition.preparation_service.submit",
                side_effect=AssertionError("browsing must not subscribe to a transfer"),
            ),
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(self.manager, SimpleNamespace(), offline=lambda: False)
            details = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "selected_details")
            discovered = ui.refresh_rows("", "All", "discover", "All", "All", key, 1)
            self.assertEqual(
                {card["key"] for card in discovered[1]["cards"]},
                set(self.manager.DOWNLOADABLE_FILES),
            )
            local = ui.refresh_rows("", "All", "on-device", "All", "All", key, 1)
            self.assertEqual([card["key"] for card in local[1]["cards"]], ["local:" + str(custom)])
            self.assertEqual(local[2], key)
            table, status, download, activate = details(key, local[0])
            self.assertIn("outside the current list", status)
            self.assertIn(["File", str(self.root / "checkpoints/a.safetensors")], table)
            self.assertTrue(download["visible"])
            self.assertFalse(activate["interactive"])
            self.installed("checkpoints", "a.safetensors")
            installed = ui.refresh_rows("", "All", "on-device", "All", "All", key, 1)
            self.assertEqual(installed[2], key)
            self.assertEqual(
                {card["key"] for card in installed[1]["cards"]}, {key, "local:" + str(custom)}
            )
            filtered = ui.refresh_rows("nonexistent", "All", "on-device", "All", "All", key, 99)
            self.assertEqual((filtered[2], filtered[3], filtered[4]), (key, 1, "0 matching files"))
            self.assertEqual(filtered[1]["cards"], [])
            self.assertTrue(details(key, filtered[0])[3]["interactive"])
            state = SessionState(app)
            state[ui.selection._id] = key
            index = next(index for index, fn in app.fns.items() if fn.fn.__name__ == "filter_rows")
            event = asyncio.run(
                app.process_api(
                    index,
                    ["nonexistent", "All", "on-device", "All", "All", None, 99, 1],
                    state=state,
                )
            )
            self.assertEqual(state[ui.selection._id], key)
            self.assertEqual(event["data"][1]["cards"], [])
            self.assertFalse(details(key, installed[0], include_companions=False)[2]["visible"])
            self.assertTrue(details(key, installed[0], include_companions=True)[2]["visible"])
            self.installed("clip", "encoder.safetensors")
            installed = ui.refresh_rows("", "All", "on-device", "All", "All", key, 1)
            self.assertFalse(details(key, installed[0])[2]["visible"])

    def test_previews_keep_same_stem_variants_distinct(self):
        self.installed("checkpoints", "a.safetensors")
        variant = self.installed("checkpoints", "a.gguf")
        legacy = variant.with_suffix(".preview.png")
        exact = variant.with_name(variant.name + ".preview.png")
        cache = self.root / "cache/checkpoints/a.png"
        cache.parent.mkdir(parents=True)
        for path in (legacy, exact, cache):
            Image.new("RGB", (8, 8), "red").save(path)
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False):
                ui = create_ui(self.manager, SimpleNamespace(), offline=lambda: True)
            result = ui.refresh_rows("", "All", "on-device", "All", "All", None, 1)
        cards = {card["key"]: card for card in result[1]["cards"]}
        fallback = cards["checkpoints/a.safetensors"]["preview"]["path"]
        self.assertEqual(Path(fallback).parent.name, "default-previews")
        self.assertEqual(cards["checkpoints/a.safetensors"]["fallback"]["path"], fallback)
        self.assertTrue(cards["checkpoints/a.safetensors"]["preview_is_default"])
        with Image.open(fallback) as image:
            self.assertEqual(image.size, (512, 512))
        self.assertEqual(cards["local:" + str(variant)]["preview"]["path"], str(exact))
        self.assertFalse(cards["local:" + str(variant)]["preview_is_default"])

    def test_real_artifact_planning_schema_is_shown_without_claiming_device_fit(self):
        registry = fixtures.packaged_registry()
        artifact = next(
            row for row in registry.entries() if row.get("hardware", {}).get("planning_profiles")
        )
        row = self.catalogue.record("checkpoints/a.safetensors")
        row["annotation"] = dict(artifact)
        row["profile"] = {}
        fields = dict(detail_rows(row)[0])
        self.assertIn("Unmeasured planning envelope", fields["Memory suitability"])
        self.assertIn("context", fields["Memory suitability"])
        self.assertIn("estimated full-offload VRAM", fields["Memory suitability"])
        for estimate in artifact["hardware"]["planning_profiles"]:
            self.assertIn(
                f"context {estimate['context_tokens']} tokens", fields["Memory suitability"]
            )


if __name__ == "__main__":
    unittest.main()
