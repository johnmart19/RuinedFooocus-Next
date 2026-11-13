"""Local selection and checkpoint-tool contracts without starting an app server."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import gradio as gr
from safetensors import SafetensorError
from PIL import Image
from ruinedfooocus.core.translation_manager import localize
from ruinedfooocus.features.models.selection import (
    checkpoint_records,
    model_label,
    selected_checkpoint,
)

from ui_test_helpers import function_from


class LocalModelSelectionTests(unittest.TestCase):
    def test_local_browser_has_no_default_acquisition_or_second_checkpoint_picker(self):
        with tempfile.TemporaryDirectory() as directory:
            preview = Path(directory) / "preview.png"
            Image.new("RGB", (8, 8)).save(preview)
            models = SimpleNamespace(
                cache_paths={"checkpoints": Path(directory) / "cache/checkpoints"},
                get_names=lambda category: ["local.safetensors"],
                get_models_by_path=lambda category, name, fetch=False: {},
                get_file=lambda category, name: "local.safetensors",
            )
            create = function_from(
                "ruinedfooocus/features/models/ui_browser.py",
                "create_ui",
                {
                    "gr": gr,
                    "Path": Path,
                    "__file__": str(
                        Path(__file__).resolve().parents[1]
                        / "ruinedfooocus/features/models/ui_browser.py"
                    ),
                    "checkpoint_records": checkpoint_records,
                    "model_label": model_label,
                    "selected_checkpoint": selected_checkpoint,
                    "get_checkpoint_thumbnail": lambda name: str(preview),
                    "localize": localize,
                    "localized_output": Mock(),
                    "resolve_locale": lambda **kwargs: "en",
                    "create_components_ui": Mock(),
                    "create_checkpoint_ui": Mock(),
                },
            )
            with gr.Blocks() as demo:
                base = gr.Dropdown(choices=["local.safetensors"], value="local.safetensors")
                refresh = gr.Number(value=0, visible=False)
                _, _, selection, _ = create(base, models, model_refresh=refresh)
            self.assertEqual(
                sum(isinstance(block, gr.Dropdown) for block in demo.blocks.values()), 1
            )
            self.assertFalse(
                any(
                    "Download" in str(getattr(block, "value", "")) for block in demo.blocks.values()
                )
            )
            functions = {index: getattr(fn.fn, "__name__", "") for index, fn in demo.fns.items()}
            refreshed = {
                functions[item["id"]]
                for item in demo.config["dependencies"]
                if [refresh._id, "change"] in [list(target) for target in item["targets"]]
            }
            self.assertTrue({"refresh", "describe"}.issubset(refreshed))
            identity = next(
                block
                for block in demo.blocks.values()
                if getattr(block, "elem_id", None) == "selected-checkpoint-identity"
            )
            self.assertEqual(identity.value, "local")
            query = next(
                block
                for block in demo.blocks.values()
                if getattr(block, "elem_id", None) == "model-search"
            )
            self.assertLess(identity._id, query._id)
            visible = next(
                block
                for block in demo.blocks.values()
                if getattr(block, "elem_id", None) == "checkpoint-cards"
            )
            self.assertEqual(visible.value["cards"][0]["kind"], "")
            projected_cards = next(
                fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "visible_cards_for"
            )
            for locale in ("en", "uk", "de"):
                payload = projected_cards(selection.value, base.value, locale=locale)
                self.assertEqual(
                    payload["defaultPreviewLabel"], localize("Default preview", locale)
                )
                self.assertNotIn("__i18n__", payload["defaultPreviewLabel"])
                self.assertNotIn("ruinedfooocus.", payload["defaultPreviewLabel"])
            # Filtering has no output to the persistent selected-identity control.
            self.assertTrue(
                all(
                    identity not in fn.outputs
                    for fn in demo.fns.values()
                    if fn.fn.__name__ == "refresh"
                )
            )

    def test_same_local_card_is_a_noop_and_removed_file_is_rejected(self):
        models = SimpleNamespace(get_file=Mock(return_value=Path("local.safetensors")))
        select = function_from(
            "ruinedfooocus/features/models/ui_browser.py",
            "choose_card",
            {"gr": gr, "models": models},
        )
        records = [{"caption": "verified identity", "name": "local.safetensors"}]
        event = gr.EventData(None, {"key": "verified identity"})
        self.assertEqual(select(records, "local.safetensors", event), gr.skip())
        self.assertEqual(select(records, "other.safetensors", event)["value"], "local.safetensors")
        models.get_file.return_value = None
        with self.assertRaisesRegex(gr.Error, "changed"):
            select(records, "local.safetensors", event)

    def test_selected_model_details_are_local_only(self):
        metadata = Mock(return_value={"baseModel": "Flux", "model": {"name": "Local title"}})
        describe = function_from(
            "ruinedfooocus/features/models/ui_browser.py",
            "describe",
            {
                "models": SimpleNamespace(get_models_by_path=metadata),
                "model_label": lambda name, info: "Friendly model",
            },
        )
        self.assertIn("Family: Flux", describe("local.safetensors"))
        metadata.assert_called_once_with("checkpoints", "local.safetensors", fetch=False)


class CheckpointToolAvailabilityTests(unittest.TestCase):
    def test_unknown_checkpoint_empty_role_disables_choices_without_inventory(self):
        choose = function_from(
            "ruinedfooocus/features/models/ui_component_overrides.py",
            "role_choices",
            {"gr": gr},
        )
        result = choose("unknown.ckpt", None, {"retained": "choice"}, "scope")
        self.assertEqual(result["choices"], [])
        self.assertIsNone(result["value"])
        self.assertFalse(result["interactive"])

    def test_inspect_also_enables_useful_conversion_for_initial_selection(self):
        selection = Mock(return_value=("header report", {"interactive": True}))
        inspect = function_from(
            "ruinedfooocus/features/models/ui_checkpoint_tools.py",
            "inspect_selected",
            {
                "gr": gr,
                "selection_report": selection,
                "Path": Path,
                "NATIVE_TORCH_SUFFIXES": (".ckpt",),
                "model_refresh": None,
            },
        )
        self.assertEqual(inspect("local.safetensors"), selection.return_value)
        selection.assert_called_once()
        self.assertEqual(selection.call_args.args, ("local.safetensors",))
        self.assertTrue(selection.call_args.kwargs["verify_source"])
        self.assertTrue(callable(selection.call_args.kwargs["on_verified"]))

    def test_fp16_action_tracks_useful_saving_for_the_selected_file(self):
        inspect = Mock(return_value={"dtypes": {"F32": 100}, "saving_bytes": 50})
        run = Mock(return_value="header report")
        selected = Mock(return_value=(Path("local.safetensors"), {}))
        report = function_from(
            "ruinedfooocus/features/models/ui_checkpoint_tools.py",
            "selection_report",
            {
                "gr": gr,
                "tools": SimpleNamespace(inspect_checkpoint=inspect),
                "_selected": selected,
                "run_action": run,
                "SafetensorError": SafetensorError,
                "NATIVE_TORCH_SUFFIXES": (".ckpt",),
                "is_safetensors": lambda name: str(name).endswith(".safetensors"),
            },
        )
        self.assertTrue(report("local.safetensors")[1]["interactive"])
        for dtypes, saving in (
            ({"BF16": 100}, 0),
            ({"F8_E4M3": 100}, 0),
            ({"F32": 1, "BF16": 1000}, 0),
        ):
            inspect.return_value = {"dtypes": dtypes, "saving_bytes": saving}
            self.assertFalse(report("local.safetensors")[1]["interactive"])
        inspect.side_effect = ValueError("unsupported format")
        self.assertFalse(report("local.gguf")[1]["interactive"])
        self.assertFalse(report(None)[1]["interactive"])


if __name__ == "__main__":
    unittest.main()
