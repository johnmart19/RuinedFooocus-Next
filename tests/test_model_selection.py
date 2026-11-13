"""Friendly checkpoint captions never replace the file identity used by generation."""

import unittest
from unittest.mock import patch

import gradio as gr

from ruinedfooocus.features.models.ui_download import download_default

from ruinedfooocus.features.models.selection import (
    checkpoint_choices,
    checkpoint_records,
    model_label,
    primary_checkpoint_selection,
    selected_checkpoint,
)


class ModelSelectionTests(unittest.TestCase):
    def test_missing_selection_is_retained_and_newly_installed_one_has_no_duplicate(self):
        self.assertEqual(
            checkpoint_choices([], "default.safetensors"),
            [("default (not installed)", "default.safetensors")],
        )
        self.assertEqual(
            checkpoint_choices(["default.safetensors"], "default.safetensors"),
            [("default", "default.safetensors")],
        )
        names = ["other.safetensors"]
        checkpoint_choices(names, "missing.safetensors")
        self.assertEqual(names, ["other.safetensors"])

    def test_offline_download_is_rejected_before_loading_state_or_fetching(self):
        with patch("ruinedfooocus.features.models.ui_download.offline", return_value=True):
            with self.assertRaisesRegex(gr.Error, "Offline mode"):
                download_default()

    def test_publisher_title_and_version_make_janku_readable(self):
        metadata = {
            "model": {"name": "✨ JANKU Trained + Chenkin & NoobAI + RouWei Illustrious XL ✨"},
            "name": "v7.77",
        }
        filename = "JANKUTrainedChenkinNoobai_v777.safetensors"
        self.assertEqual(model_label(filename, metadata), "JANKU v7.77")
        choices = checkpoint_choices([filename], filename, lambda _: metadata)
        self.assertEqual(choices, [("JANKU v7.77", filename)])

    def test_unknown_and_missing_models_remain_selectable_without_guessing_identity(self):
        self.assertEqual(
            checkpoint_choices(["my_model.safetensors"]), [("my model", "my_model.safetensors")]
        )
        choices = checkpoint_choices([], "folder/missing.safetensors")
        self.assertEqual(choices, [("missing (not installed)", "folder/missing.safetensors")])

    def test_filename_hash_suffix_is_not_a_visible_title(self):
        self.assertEqual(model_label("my_model_8d59ab04.safetensors"), "my model")
        self.assertEqual(model_label("qwen_2.1_Q4_K_M.gguf"), "qwen 2.1 Q4 K M")

    def test_same_title_versions_and_filenames_stay_distinguishable(self):
        names = ["a/same.safetensors", "b/same.safetensors"]
        metadata = lambda _: {"model": {"name": "Anima Mayhem"}, "name": "2.9B v1.0"}
        records = checkpoint_records(names, metadata)
        self.assertNotEqual(records[0]["label"], records[1]["label"])
        for record in records:
            self.assertNotIn("same.safetensors", record["caption"])
        self.assertEqual([r["name"] for r in records], names)
        self.assertEqual(checkpoint_records(names, metadata, "2.9b"), records)
        self.assertEqual(checkpoint_records(names, metadata, None), records)
        self.assertEqual(checkpoint_records(names, metadata, "b/same"), [records[1]])

    def test_retired_media_is_not_reintroduced_by_an_existing_selection(self):
        metadata = lambda name: {"baseModel": "LTXV 2.5"} if name == "video" else {}
        choices = checkpoint_choices(["video", "unknown.safetensors"], "video", metadata)
        self.assertIn(("unknown", "unknown.safetensors"), choices)
        self.assertEqual(choices, [("unknown", "unknown.safetensors")])

    def test_retired_startup_choice_uses_first_image_or_empty_state_without_saving(self):
        metadata = lambda name: {"baseModel": "LTXV" if name == "first.safetensors" else "SDXL"}
        names = ["first.safetensors", "image.safetensors"]
        selected = primary_checkpoint_selection(names[0], metadata, names=names)
        self.assertEqual(selected, names[1])
        self.assertIsNone(primary_checkpoint_selection(names[0], metadata, names=names[:1]))
        with gr.Blocks(analytics_enabled=False):
            picker = gr.Dropdown(
                choices=checkpoint_choices(names, selected, metadata), value=selected
            )
        self.assertEqual(picker.value, names[1])

    def test_gallery_selection_uses_snapshot_and_rejects_a_stale_event(self):
        records = checkpoint_records(
            ["actual.safetensors"], lambda _: {"model": {"name": "Human name v1"}}
        )
        caption = records[0]["caption"]
        self.assertEqual(selected_checkpoint(records, 0, caption), "actual.safetensors")
        for index, caption in (
            (1, caption),
            (-1, caption),
            (0, "Other name"),
            (True, caption),
        ):
            with self.assertRaisesRegex(ValueError, "list changed"):
                selected_checkpoint(records, index, caption)

    def test_identical_titles_on_successive_snapshots_do_not_retarget_a_queued_click(self):
        metadata = lambda _: {"model": {"name": "Same title"}, "name": "v1"}
        old = checkpoint_records(["first.safetensors"], metadata)
        new = checkpoint_records(["second.safetensors"], metadata)
        self.assertEqual(old[0]["label"], new[0]["label"])
        with self.assertRaisesRegex(ValueError, "list changed"):
            selected_checkpoint(new, 0, old[0]["caption"])


if __name__ == "__main__":
    unittest.main()
