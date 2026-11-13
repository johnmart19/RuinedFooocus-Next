"""Exact WeVisOne acquisition and actual utility/Hub consumers, without inference."""

from pathlib import Path
import ast
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
from PIL import Image
from safetensors.numpy import save_file

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.acquisition import CATEGORIES, Catalogue
from ruinedfooocus.features.models.component_candidates import (
    TEXT_TYPES,
    classify_shapes,
    compatibility,
)
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_inbox import InboxSorter
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models import native_text_identity


KEY = "clip/wevisone_vl_2b_int8_convrot.safetensors"
SIZE = 2890564784
SHA = "88f3b1be21ff0828f20f0a711a14e00c55f401bceafad3c8943b7ce2c577a03e"
REVISION = "135355350443103e34089dd39264fe0651691929"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


class WeVisOneCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = {
            "path_" + category: str(self.root / category) for category in CATEGORIES
        }
        self.manager.model_paths = {"cache_path": self.root / "cache"}
        self.manager.DOWNLOADABLE_FILES = json.loads(
            resource_path("model_catalogue", "native_wevisone.json").read_text(encoding="utf-8")
        )
        self.registry = load_registry()
        self.catalogue = Catalogue(self.manager, self.registry)
        self.enterContext(
            patch.object(native_text_identity, "_source", return_value=NATIVE / "comfy/sd.py")
        )

    def file(self, name="renamed.safetensors", *, wevis=True):
        path = self.root / "clip" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            "model.visual.merger.linear_fc2.weight": np.zeros((2048, 1), dtype=np.float16),
            "model.visual.deepstack_merger_list.0.norm.weight": np.zeros(1, dtype=np.float16),
        }
        if wevis:
            tensors["mask_decoder.query_builder.content_proj.weight"] = np.zeros(
                (1, 1), dtype=np.float16
            )
        save_file(tensors, path)
        return path

    def test_registered_tuple_license_role_and_qualification_are_exact(self):
        source = self.manager.DOWNLOADABLE_FILES[KEY]
        annotation = self.registry.annotations_for(KEY, source)
        profile = self.registry.profile(annotation["profile_id"])
        self.assertEqual(source["path"], "path_clip")
        self.assertIn("/resolve/" + REVISION + "/text_encoders/", source["url"])
        self.assertEqual((source["size"], source["sha256"]), (SIZE, SHA))
        self.assertEqual(
            (annotation["artifact"]["bytes"], annotation["artifact"]["sha256"]), (SIZE, SHA)
        )
        self.assertEqual(annotation["role"], "image_utility_encoder")
        self.assertEqual(annotation["component_role"], "wevisone-clip")
        self.assertEqual(annotation["native_text_type"], "WEVISONE_2B")
        self.assertEqual(tuple(annotation["utility_kinds"]), ("wevisone-detect", "wevisone-matte"))
        self.assertFalse(
            annotation["chat_weight"]
            or annotation["uses_chat_runtime"]
            or annotation["auto_recipe_supported"]
        )
        self.assertEqual(profile["license"]["value"], "Apache-2.0")
        self.assertIn(REVISION, profile["license"]["source"])
        self.assertEqual(profile["standalone_vram_status"], "profile_or_workflow_dependent")
        self.assertFalse(profile["qualification"]["automatic_recipe"])
        rows = {row["id"]: row for row in self.registry.native_dispositions(kind="te_models")}
        self.assertEqual(tuple(rows["WEVISONE_2B"]["available_artifact_ids"]), (annotation["id"],))
        self.assertFalse(rows["WEVISONE_4B"]["available_artifact_ids"])

    def test_hub_acquisition_captures_only_the_exact_needed_utility_file(self):
        row = self.catalogue.record(KEY)
        self.assertEqual((row["task"], row["category_label"]), ("Image tools", "Image tools"))
        self.assertEqual(row["category"], "clip")
        self.assertIn("WeVisOne matting", row["support"])
        self.assertTrue(row["allowed"])
        self.assertFalse(row["can_activate"])
        plan = self.catalogue.capture(KEY, reserve_gib=0)
        self.assertEqual(len(plan.artifacts), 1)
        request = plan.artifacts[0]
        self.assertEqual((request.expected_size, request.sha256), (SIZE, SHA))
        self.assertEqual(request.destination, self.root / "clip" / Path(KEY).name)
        self.manager.paths["path_clip"] = str(self.root / "changed")
        self.assertEqual(request.destination.parent, self.root / "clip")

    def test_changed_publisher_tuple_is_not_silently_presented_as_reviewed(self):
        self.manager.DOWNLOADABLE_FILES[KEY]["url"] += "?different=1"
        row = self.catalogue.record(KEY)
        self.assertFalse(row["allowed"])
        self.assertIn("no longer matches", row["reason"])

    def test_renamed_native_file_reaches_real_hub_tool_activation_and_utility_role_capture(self):
        from ruinedfooocus.features.image_editing.tool_activation import (
            activation_values,
            installed_tool,
        )
        from ruinedfooocus.features.image_processing import wevisone

        path = self.file()
        row = dict(
            local=str(path),
            local_signature=file_signature(path),
            category="clip",
            size_matches=True,
        )
        target = installed_tool(row, self.manager)
        self.assertEqual(
            target, dict(kind="wevisone-matte", resource="wevisone_model", value=str(path))
        )
        values = activation_values(
            "wevisone-matte",
            target,
            presets={"Matting": {"type": "wevisone-matte"}},
            custom="Custom...",
        )
        self.assertEqual(values["wevisone_model"], str(path))
        native_signatures = tuple(
            tuple(file_signature(NATIVE / name)) for name in wevisone.NATIVE_FILES
        )
        with patch.object(wevisone, "native_source_signatures", return_value=native_signatures):
            captured, (artifact,) = wevisone.capture_assets(
                dict(
                    controlnet={"type": "wevisone-matte"},
                    input_image=Image.new("RGB", (8, 6)),
                    wevisone_model=values["wevisone_model"],
                ),
                self.manager,
            )
        self.assertEqual((artifact.role, artifact.name), ("wevisone-clip", str(path)))
        self.assertEqual(captured[wevisone.KEY].native_type, "WEVISONE_2B")
        self.assertNotIn("image_component_overrides", captured)

    def test_generic_qwen_and_every_diffusion_encoder_slot_remain_separate(self):
        from ruinedfooocus.features.image_editing.tool_activation import installed_tool
        from ruinedfooocus.features.models.model_identity import cached_safetensors_header

        path = self.file()
        header = cached_safetensors_header(path)
        classification = classify_shapes({key: tuple(row["shape"]) for key, row in header.items()})
        self.assertEqual(classification["text_role"], "image_utility_encoder")
        for key in (*TEXT_TYPES, "clip_unregistered"):
            self.assertEqual(compatibility(key, classification), "incompatible", key)
        generic = self.file("generic.safetensors", wevis=False)
        row = dict(
            local=str(generic),
            local_signature=file_signature(generic),
            category="clip",
            size_matches=True,
        )
        self.assertIsNone(installed_tool(row, self.manager))

    def test_real_inbox_native_reader_sorts_a_renamed_utility_without_publisher_or_runtime(self):
        path = self.file()
        inbox = self.root / "inbox"
        inbox.mkdir()
        source = inbox / "unrelated.sft"
        path.rename(source)
        sorter = InboxSorter(self.manager, offline=True)
        with patch("ruinedfooocus.features.models.model_sources.civitai_metadata") as remote:
            self.assertEqual(sorter.scan([inbox]), 0)
            self.assertEqual(sorter.scan([inbox]), 1)
            remote.assert_not_called()
        self.assertFalse(source.exists())
        self.assertTrue((self.root / "clip" / source.name).is_file())

    def test_actual_hub_details_offer_utility_use_without_an_unrelated_checkpoint_warning(self):
        from ruinedfooocus.features.models.ui_acquisition import create_ui

        path = self.file()
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(
                    self.manager, SimpleNamespace(update_all_models=Mock()), offline=lambda: True
                )
            rows = ui.catalogue.records()
            row = next(row for row in rows if row.get("local") == str(path))
            details = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "selected_details")
            _, text, download, activate = details(row["key"], rows)
        self.assertTrue(activate["visible"] and activate["interactive"])
        self.assertEqual(activate["value"], "Use installed file in Image tools")
        self.assertFalse(download["visible"])
        self.assertNotIn("No compatible external role on the selected checkpoint", text)

    def test_actual_hub_activation_dispatches_reader_proven_clip_to_image_tools(self):
        from ruinedfooocus.features.image_editing.tool_activation import activation_values
        from ruinedfooocus.features.models.acquisition import inspected_local

        path = self.file()
        row = next(row for row in self.catalogue.records() if row.get("local") == str(path))
        source = ROOT / "ruinedfooocus/ui/app.py"
        activation = next(
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "activate_catalogue_checkpoint"
        )
        use_component = Mock(
            side_effect=AssertionError("Utility cannot consume a diffusion encoder slot")
        )
        activate = Mock(
            side_effect=lambda kind, resource, **_: activation_values(
                kind,
                resource,
                presets={"Matting": {"type": "wevisone-matte"}},
                custom="Custom...",
            )
        )
        values = {"prompt": "keep exactly", "clip_l": "unchanged.safetensors"}
        scope = dict(
            gr=gr,
            path_manager=self.manager,
            shared=SimpleNamespace(
                models=SimpleNamespace(), settings=SimpleNamespace(default_settings=values)
            ),
            app_settings=SimpleNamespace(
                acquisition=SimpleNamespace(
                    catalogue=self.catalogue,
                    inspected_local=inspected_local,
                    use_component=use_component,
                )
            ),
            ui_controlnet=SimpleNamespace(activate_tool=activate),
            chat_catalogue_outputs=[],
            tool_catalogue_outputs=["wevisone_model", "cn_selection"],
        )
        exec(compile(ast.Module(body=[activation], type_ignores=[]), str(source), "exec"), scope)
        result = scope["activate_catalogue_checkpoint"](
            row["key"], [row], "keep-checkpoint.safetensors"
        )
        self.assertEqual(result[0], gr.skip())
        self.assertEqual(result[2]["selected"], "image-tools")
        self.assertEqual((result[3], result[4]), (gr.skip(), gr.skip()))
        self.assertEqual(result[5:7], (str(path), "Matting"))
        self.assertEqual(len(result), 7)
        self.assertEqual(values, {"prompt": "keep exactly", "clip_l": "unchanged.safetensors"})
        use_component.assert_not_called()
        self.assertEqual(activate.call_args.args[0], "wevisone-matte")
        self.assertEqual(activate.call_args.kwargs, {"checkpoint": "keep-checkpoint.safetensors"})


if __name__ == "__main__":
    unittest.main()
