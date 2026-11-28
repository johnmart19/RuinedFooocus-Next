"""Unified card callbacks exercise real Gradio wiring without app/GPU startup."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image
from ruinedfooocus.core.translation_manager import localize

from ruinedfooocus.features.models import ui_adapters
from ruinedfooocus.features.models.embedding_browser import (
    embedding_spans,
    remove_embedding,
    scan_embeddings,
)


class AdapterBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "style.safetensors").write_bytes(b"private LoRA fixture")
        self.embeddings = self.root / "embeddings"
        self.embeddings.mkdir()
        (self.embeddings / "detail.pt").write_bytes(b"private embedding fixture")
        self.preview = self.root / "fixture.png"
        Image.new("RGB", (8, 8)).save(self.preview)
        models = SimpleNamespace(
            get_names=lambda category: ["style.safetensors"],
            get_file=lambda category, name: self.root / name,
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.core.util": SimpleNamespace(
                        _get_model_thumbnail=lambda *args, **kwargs: None
                    )
                },
            ),
            patch.object(ui_adapters, "t", side_effect=lambda text: text),
        ):
            with gr.Blocks() as self.demo:
                self.prompt = gr.Textbox(value="Keep  two cups.")
                self.negative = gr.Textbox(value="(embedding:detail:0.8) noise")
                self.controls = ui_adapters.create_ui(
                    self.prompt,
                    self.negative,
                    models=models,
                    directories=lambda: [self.embeddings],
                    cache_dir=self.root / "cache",
                    defaults=[("style.safetensors", 0.7)],
                    thumbnail=lambda name: str(self.preview),
                    keywords=lambda name: "trigger",
                )
        self.records = self.controls.catalog.value
        self.gallery = [(None, "0.7 - style.safetensors")]

    def callback(self, name):
        fn = next(fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", None) == name)
        event = next(event for event in self.demo.config["dependencies"] if event["id"] == fn._id)
        return fn.fn, event

    def test_one_shared_available_browser_and_hidden_native_transport(self):
        self.assertFalse(self.controls.transport.visible)
        self.assertEqual(self.controls.transport.value[0]["caption"], "0.7 - style.safetensors")
        self.assertEqual({card["kind"] for card in self.records}, {"LoRA", "Embedding"})
        captions = [item["caption"] for item in self.controls.available.value]
        self.assertTrue(any("[LoRA]" in text for text in captions))
        self.assertTrue(any("[Embedding]" in text for text in captions))
        dropdowns = [block for block in self.demo.blocks.values() if isinstance(block, gr.Dropdown)]
        self.assertEqual(len(dropdowns), 1)
        self.assertEqual(dropdowns[0].label, "Embedding target")
        self.assertEqual(
            [choice[1] for choice in dropdowns[0].choices], ["Auto", "Positive", "Negative"]
        )
        # Both active kinds coexist at startup, including weighted alias replay.
        self.assertEqual(len(self.controls.active.value), 2)

    def test_render_derives_selected_from_replayed_loras_and_both_prompts(self):
        render, _ = self.callback("render")
        available, active = render(
            self.records,
            "",
            "All",
            self.gallery,
            "embedding:detail.pt exact",
            "(embedding:detail:1.2)",
        )
        self.assertEqual(len(available["value"]), 2)
        self.assertEqual(len(active["value"]), 2)
        self.assertTrue(any("Positive + Negative" in caption for _, caption in active["value"]))
        filtered, active = render(self.records, "detail.pt", "Embedding", self.gallery, "", "")
        self.assertEqual(len(filtered["value"]), 1)
        self.assertEqual(len(active["value"]), 1)
        cleared, _ = render(self.records, "", "All", [], "", "")
        self.assertEqual(len(cleared["value"]), 2)

    def test_select_rejects_stale_caption_and_embedding_never_changes_lora_weight(self):
        select, _ = self.callback("select_card")
        caption = next(
            item["caption"]
            for item in self.controls.available.value
            if "[Embedding]" in item["caption"]
        )
        evt = gr.SelectData(None, {"index": 1, "value": {"caption": caption}, "selected": True})
        selection = select(self.records, self.gallery, evt)
        self.assertEqual(selection[0], ("Embedding", "detail.pt"))
        self.assertFalse(selection[2]["visible"])
        # Initial selected cards and cards rerendered after replay accept the same identity.
        active_caption = next(
            item["caption"]
            for item in self.controls.active.value
            if "[Embedding]" in item["caption"]
        )
        active_evt = gr.SelectData(
            None, {"index": 1, "value": {"caption": active_caption}, "selected": True}
        )
        self.assertEqual(select(self.records, self.gallery, active_evt)[0], selection[0])
        render, _ = self.callback("render")
        _, active_update = render(self.records, "", "All", self.gallery, "embedding:detail", "")
        rendered_caption = next(
            caption for _, caption in active_update["value"] if "[Embedding]" in caption
        )
        rendered_evt = gr.SelectData(
            None, {"index": 1, "value": {"caption": rendered_caption}, "selected": True}
        )
        self.assertEqual(select(self.records, self.gallery, rendered_evt)[0], selection[0])
        set_lora, _ = self.callback("update_weight")
        with self.assertRaisesRegex(gr.Error, "only to LoRAs"):
            set_lora(self.records, selection[0], self.gallery, 1)
        stale = gr.SelectData(None, {"index": 0, "value": {"caption": "stale"}, "selected": True})
        with self.assertRaises(gr.Error):
            select(self.records, self.gallery, stale)

    def test_contextual_activation_and_removal_preserve_manual_prompt_text(self):
        add, event = self.callback("activate_selected")
        remove, removal = self.callback("remove_selected")
        current = " Keep  two cups.\n<lora:unrelated:0.8>"
        key = ("Embedding", "detail.pt")
        result = add(self.records, key, self.gallery, current, "noise", {}, {}, 1)
        value = result[1]
        self.assertEqual(value, "detail " + current)
        self.assertEqual(
            add(self.records, key, self.gallery, value, "noise", {}, result[3], 1)[1], gr.skip()
        )
        self.assertEqual(
            remove(self.records, key, self.gallery, value, "noise", {}, result[3], 1)[1], current
        )
        manual = "(embedding:detail:0.8) more  text"
        self.assertEqual(
            remove(self.records, key, self.gallery, manual, "noise", {}, {}, 1)[1], gr.skip()
        )
        self.assertEqual(
            event["outputs"][:3], [self.controls.transport._id, self.prompt._id, self.negative._id]
        )
        self.assertEqual(removal["outputs"], event["outputs"])
        (self.embeddings / "detail.pt").unlink()
        self.assertEqual(
            remove(self.records, key, self.gallery, value, "noise", {}, result[3], 1)[1], current
        )
        self.assertEqual(
            remove(self.records, key, self.gallery, value, "noise", {}, {}, 1)[1], gr.skip()
        )

    def test_lora_weight_and_delete_preserve_native_pipeline_contract(self):
        set_lora, _ = self.callback("activate_selected")
        delete, _ = self.callback("remove_selected")
        key = ("LoRA", "style.safetensors")
        result = set_lora(self.records, key, self.gallery, "", "", {}, {}, 1.2)[0]
        self.assertEqual(result["value"], [(str(self.preview), "1.2 - style.safetensors")])
        self.assertEqual(
            delete(self.records, key, result["value"], "", "", {}, {}, 1.2)[0]["value"], []
        )
        with self.assertRaises(gr.Error):
            set_lora(self.records, key, [], "", "", {}, {}, float("nan"))

    def test_refresh_is_explicit_and_deleted_card_cannot_be_reused(self):
        refresh, _ = self.callback("refresh_files")
        (self.embeddings / "new.pt").write_bytes(b"new fixture")
        records, message, selected = refresh()
        self.assertIsNone(selected)
        self.assertEqual(len(records), 3)
        self.assertIn("2 embeddings", message)

    def test_visible_metadata_search_retains_focus_without_changing_activation(self):
        entry = next(card for card in self.records if card["kind"] == "LoRA")
        entry["metadata"]["trainedWords"] = ["etched glass"]
        render, _ = self.callback("html_render")
        key = (entry["kind"], entry["name"])
        available, active = render(self.records, "etched glass", "All", self.gallery, "", "", key)
        self.assertEqual([card["key"] for card in available["cards"]], [entry["caption"]])
        self.assertEqual(available["selected"], entry["caption"])
        self.assertEqual(active["selected"], entry["caption"])
        self.assertEqual(active["cards"][0]["state"], "Active | 0.7")
        self.assertEqual(available["defaultPreviewLabel"], "Default preview")
        self.assertTrue(available["cards"][0]["preview_is_default"])
        self.assertTrue(Path(available["cards"][0]["fallback"]["path"]).is_file())
        self.assertEqual(self.gallery, [(None, "0.7 - style.safetensors")])
        # Focus is projected after both prompt replay and direct card selection.
        updates = [fn for fn in self.demo.fns.values() if fn.fn.__name__ == "html_render"]
        self.assertTrue(all(self.controls.selected in fn.inputs for fn in updates))
        events = [
            event
            for event in self.demo.config["dependencies"]
            if event["id"] in {fn._id for fn in updates}
        ]
        self.assertTrue(
            any(
                [self.controls.selected._id, "change"]
                in [list(target) for target in event["targets"]]
                for event in events
            )
        )

    def test_native_refresh_clears_stale_details_actions_and_embedding_target(self):
        import asyncio
        from gradio.state_holder import SessionState

        state = SessionState(self.demo)
        state[self.controls.selected._id] = ("Embedding", "detail.pt")
        state[self.controls.destinations._id] = {"detail.pt": "Negative"}

        async def refresh_and_clear():
            refresh, _ = self.callback("refresh_files")
            index = next(index for index, fn in self.demo.fns.items() if fn.fn is refresh)
            await self.demo.process_api(index, [], state=state)
            self.assertIsNone(state[self.controls.selected._id])
            clear, _ = self.callback("clear_selection")
            index = next(index for index, fn in self.demo.fns.items() if fn.fn is clear)
            result = await self.demo.process_api(index, [None], state=state)
            self.assertEqual(result["data"][0], "")
            self.assertFalse(result["data"][1]["visible"])
            self.assertTrue(all(not update["interactive"] for update in result["data"][2:]))
            restore, _ = self.callback("restore_target")
            index = next(index for index, fn in self.demo.fns.items() if fn.fn is restore)
            result = await self.demo.process_api(index, [None, None], state=state)
            self.assertEqual(result["data"][0]["value"], "Auto")
            self.assertFalse(result["data"][0]["visible"])
            # Per-resource target preferences survive refreshing the file list.
            self.assertEqual(state[self.controls.destinations._id], {"detail.pt": "Negative"})
            self.assertEqual(clear(("LoRA", "style.safetensors")), (gr.skip(),) * 4)

        asyncio.run(refresh_and_clear())

    def test_unqueued_card_selection_returns_its_embedding_target_without_state_followup(self):
        import asyncio
        from gradio.state_holder import SessionState

        (self.embeddings / "texture.pt").write_bytes(b"second embedding fixture")
        records = self.controls.refresh_files()[0]
        embeddings = {card["name"]: card for card in records if card["kind"] == "Embedding"}
        state = SessionState(self.demo)
        state[self.controls.catalog._id] = records
        state[self.controls.destinations._id] = {"detail.pt": "Negative"}

        async def select():
            for name in ("interact", "select_card"):
                callback, event = self.callback(name)
                index = next(index for index, fn in self.demo.fns.items() if fn.fn is callback)
                self.assertFalse(event["queue"])
                self.assertIn(self.controls.destinations._id, event["inputs"])
                for resource, expected in (
                    ("detail.pt", "Negative"),
                    ("texture.pt", "Auto"),
                    ("detail.pt", "Negative"),
                ):
                    entry = embeddings[resource]
                    inputs = [None, self.controls.transport.value, None]
                    event_data = gr.SelectData(
                        None,
                        {
                            "index": 0,
                            "value": {"caption": "[Embedding] " + entry["caption"]},
                            "selected": True,
                        },
                    )
                    if name == "interact":
                        inputs = [
                            None,
                            self.controls.transport.value,
                            "prompt",
                            "negative",
                            None,
                            None,
                        ]
                        event_data = gr.EventData(
                            None, {"key": entry["caption"], "action": "select"}
                        )
                    result = await self.demo.process_api(
                        index, inputs, state=state, event_data=event_data
                    )
                    # Unqueued /run does not dispatch downstream State.change in the browser.
                    # The selection response must carry this embedding's target itself.
                    self.assertEqual(result["data"][3].get("value"), expected)
                    self.assertTrue(result["data"][3]["visible"])

        asyncio.run(select())

    def test_native_card_projection_keeps_focused_resource_after_prompt_replay(self):
        import asyncio
        from gradio.state_holder import SessionState

        entry = next(card for card in self.records if card["kind"] == "Embedding")
        key = (entry["kind"], entry["name"])
        state = SessionState(self.demo)
        state[self.controls.catalog._id] = self.records
        state[self.controls.selected._id] = key
        render, _ = self.callback("html_render")
        index = next(index for index, fn in self.demo.fns.items() if fn.fn is render)

        async def replay_prompt():
            result = await self.demo.process_api(
                index,
                [None, "", "All", self.controls.transport.value, "detail exact", "noise", None],
                state=state,
            )
            available, active = result["data"]
            self.assertEqual(available["selected"], entry["caption"])
            self.assertEqual(active["selected"], entry["caption"])
            self.assertEqual(
                next(card["state"] for card in active["cards"] if card["key"] == entry["caption"]),
                "Positive",
            )
            self.assertEqual(state[self.controls.selected._id], key)

        asyncio.run(replay_prompt())

    def test_default_preview_label_uses_plain_localized_text_in_actual_card_payloads(self):
        render, _ = self.callback("html_render")
        for locale in ("en", "uk", "de"):
            request = SimpleNamespace(headers={"accept-language": locale}, cookies={})
            available, active = render(
                self.records, "", "All", self.gallery, "", "", request=request
            )
            for payload in (available, active):
                label = payload["defaultPreviewLabel"]
                self.assertEqual(label, localize("Default preview", locale))
                self.assertNotIn("__i18n__", label)
                self.assertNotIn("ruinedfooocus.", label)

    def test_negative_embedding_card_goes_to_negative_prompt(self):
        (self.embeddings / "lazyneg.safetensors").write_bytes(b"negative fixture")
        records = self.controls.refresh_files()[0]
        interact, _ = self.callback("interact")
        entry = next(card for card in records if card["name"] == "lazyneg.safetensors")
        result = interact(
            records,
            self.gallery,
            "positive",
            "noise",
            {},
            {},
            gr.EventData(None, {"key": entry["caption"], "action": "activate"}),
        )
        self.assertEqual(result[-3], gr.skip())
        self.assertEqual(result[-2], "lazyneg noise")

    def test_direct_activation_uses_hidden_identity_and_bare_trigger(self):
        interact, event = self.callback("interact")
        entry = next(card for card in self.records if card["kind"] == "Embedding")
        result = interact(
            self.records,
            self.gallery,
            "exact  prompt",
            "negative",
            {},
            {},
            gr.EventData(None, {"key": entry["caption"], "action": "activate"}),
        )
        self.assertEqual(result[-3], "detail exact  prompt")
        self.assertEqual(result[-4], gr.skip())
        # Double-click sends select then activate; neither may be dropped.
        self.assertEqual(event["trigger_mode"], "multiple")
        lora = next(card for card in self.records if card["kind"] == "LoRA")
        result = interact(
            self.records,
            [],
            "positive",
            "negative",
            {},
            {},
            gr.EventData(None, {"key": lora["caption"], "action": "activate"}),
        )
        self.assertIn("0.7", str(self.gallery))
        self.assertEqual(result[-4]["value"][0][1], "1 - style.safetensors")
        with self.assertRaises(gr.Error):
            interact(
                self.records,
                [],
                "p",
                "n",
                {},
                {},
                gr.EventData(None, {"key": "stale", "action": "activate"}),
            )

    def test_precise_removal_preserves_ambiguous_syntax_and_other_references(self):
        entry = next(card["embedding"] for card in self.records if card["kind"] == "Embedding")
        prompt = '"embedding:detail" embedding:detail-extra embedding:detail,more EMBEDDING:detail'
        self.assertFalse(embedding_spans(prompt, entry))
        self.assertEqual(remove_embedding(prompt, entry), prompt)
        prompt = "(embedding:detail.pt:0.7) embedding:detail, next (other:1.4)"
        self.assertEqual(remove_embedding(prompt, entry), " , next (other:1.4)")

    def test_mixed_embeddings_keep_independent_targets_and_session_ownership(self):
        (self.embeddings / "lazyneg.pt").write_bytes(b"negative fixture")
        (self.embeddings / "texture.pt").write_bytes(b"positive fixture")
        records = self.controls.refresh_files()[0]
        activate, _ = self.callback("activate_selected")
        remove, _ = self.callback("remove_selected")
        choose, _ = self.callback("remember_target")
        restore, _ = self.callback("restore_target")
        choices = choose(("Embedding", "detail.pt"), "Negative", {})
        choices = choose(("Embedding", "texture.pt"), "Positive", choices)
        self.assertEqual(restore(("Embedding", "detail.pt"), choices)["value"], "Negative")
        self.assertEqual(restore(("Embedding", "lazyneg.pt"), choices)["value"], "Auto")
        positive, negative, owned = "manual  prose", "noise", {}
        for name in ("detail.pt", "lazyneg.pt", "texture.pt"):
            result = activate(
                records, ("Embedding", name), self.gallery, positive, negative, choices, owned, 1
            )
            positive = positive if result[1] == gr.skip() else result[1]
            negative = negative if result[2] == gr.skip() else result[2]
            owned = result[3]
        self.assertEqual(positive, "texture manual  prose")
        self.assertEqual(negative, "lazyneg detail noise")
        # Another session has no insertion receipt and cannot delete authored text.
        self.assertEqual(
            remove(
                records,
                ("Embedding", "detail.pt"),
                self.gallery,
                positive,
                negative,
                choices,
                {},
                1,
            )[2],
            gr.skip(),
        )
        result = remove(
            records, ("Embedding", "detail.pt"), self.gallery, positive, negative, choices, owned, 1
        )
        self.assertEqual(result[2], "lazyneg noise")
        self.assertEqual(
            remove(
                records,
                ("Embedding", "detail.pt"),
                self.gallery,
                positive,
                result[2],
                choices,
                result[3],
                1,
            )[2],
            gr.skip(),
        )

    def test_active_embedding_target_moves_only_its_owned_reference(self):
        (self.embeddings / "texture.pt").write_bytes(b"another embedding")
        records = self.controls.refresh_files()[0]
        activate, _ = self.callback("activate_selected")
        first = activate(
            records, ("Embedding", "detail.pt"), self.gallery, "Keep  two cups.", "noise", {}, {}, 1
        )
        second = activate(
            records, ("Embedding", "texture.pt"), self.gallery, first[1], "noise", {}, first[3], 1
        )
        moved = activate(
            records,
            ("Embedding", "detail.pt"),
            self.gallery,
            second[1],
            "noise",
            {"detail.pt": "Negative"},
            second[3],
            1,
        )
        self.assertEqual(moved[:3], (gr.skip(), "texture Keep  two cups.", "detail noise"))
        self.assertNotIn(("detail.pt", "Positive"), moved[3])
        self.assertEqual(moved[3][("detail.pt", "Negative")], "detail ")
        self.assertEqual(moved[3][("texture.pt", "Positive")], "texture ")
        repeated = activate(
            records,
            ("Embedding", "detail.pt"),
            self.gallery,
            moved[1],
            moved[2],
            {"detail.pt": "Negative"},
            moved[3],
            1,
        )
        self.assertEqual(repeated[:3], (gr.skip(), gr.skip(), gr.skip()))
        edited = activate(
            records,
            ("Embedding", "detail.pt"),
            self.gallery,
            "manual " + second[1],
            "noise",
            {"detail.pt": "Negative"},
            second[3],
            1,
        )
        self.assertEqual(edited[:3], (gr.skip(), gr.skip(), gr.skip()))
        self.assertEqual(edited[3], second[3])

    def test_manual_alias_is_never_relocated_by_auto_and_display_is_not_keyword_transport(self):
        activate, _ = self.callback("activate_selected")
        result = activate(
            self.records,
            ("Embedding", "detail.pt"),
            self.gallery,
            "detail manual",
            "noise",
            {"detail.pt": "Negative"},
            {},
            1,
        )
        self.assertEqual(result[1:3], (gr.skip(), gr.skip()))
        next(card for card in self.records if card["kind"] == "LoRA")["metadata"][
            "trainedWords"
        ] = []
        display, _ = self.callback("active_triggers")
        text = display(self.records, self.gallery, "detail manual", "noise")
        self.assertIn("detail (Positive): detail", text)
        self.assertIn("style (Active | 0.7): trigger", text)
        raw, _ = self.callback("lora_keywords")
        self.assertEqual(raw(self.records, self.gallery), "trigger")
        self.assertFalse(self.controls.keywords.visible)

    def test_replaced_lora_card_cannot_activate_stale_resource(self):
        activate, _ = self.callback("activate_selected")
        (self.root / "style.safetensors").write_bytes(b"different replacement fixture")
        with self.assertRaisesRegex(gr.Error, "changed or was removed"):
            activate(self.records, ("LoRA", "style.safetensors"), [], "", "", {}, {}, 1)

    def test_deleted_active_lora_can_be_removed_without_loading_its_file(self):
        remove, _ = self.callback("remove_selected")
        (self.root / "style.safetensors").unlink()
        gallery = [*self.gallery, ("other.png", "0.4 - other.safetensors")]
        result = remove(
            self.records, ("LoRA", "style.safetensors"), gallery, "keep", "negative", {}, {}, 1
        )
        self.assertEqual(result[0]["value"][0][1], "0.4 - other.safetensors")
        self.assertEqual(len(result[0]["value"]), 1)
        self.assertEqual(result[1:3], (gr.skip(), gr.skip()))


if __name__ == "__main__":
    unittest.main()
