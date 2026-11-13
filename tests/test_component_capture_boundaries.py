"""Real submission/component capture shares the API/tool literal prompt boundary."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.model_folders import model_folders
from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS
from ui_test_helpers import function_from


FAMILIES = ("HunyuanDiT", "Kandinsky5Image", "Omnigen2")


class ComponentCaptureBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.models_by_name = {}
        for family in (*FAMILIES, "SDXL"):
            path = self.root / (family + ".safetensors")
            path.write_bytes((2).to_bytes(8, "little") + b"{}")
            self.models_by_name[path.name] = (family, path)
        self.lookup = Mock(
            side_effect=lambda category, name, **kwargs: {"baseModel": self.models_by_name[name][0]}
        )
        models = SimpleNamespace(
            offline=True,
            get_models_by_path=self.lookup,
            get_model_base=lambda info: info["baseModel"],
            get_file=lambda category, name: self.models_by_name[name][1],
        )
        paths = {"path_checkpoints": [str(self.root)], "path_clip": [], "path_vae": []}
        manager = SimpleNamespace(
            paths=paths,
            get_abspath=Path,
            get_model_folders=lambda category: model_folders(paths, category, Path),
            DOWNLOADABLE_FILES={},
        )
        self.apply = Mock(side_effect=lambda data, family: dict(data))
        self.capture = function_from(
            "ruinedfooocus/app/async_worker.py",
            "capture_image_data",
            {
                "shared": SimpleNamespace(
                    models=models,
                    path_manager=manager,
                    resolution_settings=object(),
                    performance_settings=SimpleNamespace(apply=self.apply),
                ),
                "settings": SimpleNamespace(default_settings={}),
                "control_tools": SimpleNamespace(get_settings=lambda data: data["controlnet"]),
            },
        )
        # Keep actual metadata parsing, component inventory and request freezing.
        # Acquisition/other engines are outside this authority regression.
        for target in (
            "ruinedfooocus.features.prompting.local_model.capture_prompt_preparation",
            "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
            "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
        ):
            self.enterContext(patch(target, side_effect=lambda data, *args, **kwargs: data))

    def roles(self, family, selected):
        contract = IMAGE_COMPONENTS[family]
        result = {key: None for key in (*contract["clip_names"], contract["vae_name"])}
        result[contract["clip_names"][0]] = selected
        return result

    def test_api_and_tool_capture_preserve_native_roles_and_explicit_choices(self):
        prompt = json.dumps(
            {
                "prompt": "literal JSON request",
                "base_model_name": "not-installed.safetensors",
                "image_component_overrides": {"clip_l": "prompt-owned.safetensors"},
            }
        )
        for family in FAMILIES:
            for boundary in ({"_api_job": object()}, {"task_type": "tool_call"}):
                with self.subTest(family=family, boundary=tuple(boundary)):
                    choices = self.roles(family, "request-owned.safetensors")
                    request = dict(
                        boundary,
                        base_model_name=family + ".safetensors",
                        prompt=prompt,
                        controlnet={"type": "img2img"},
                        image_component_overrides=choices,
                    )
                    captured = self.capture(request)
                    self.assertEqual(captured["base_model_name"], request["base_model_name"])
                    self.assertEqual(captured["prompt"], prompt)
                    self.assertEqual(captured["image_component_overrides"], choices)
                    self.assertIsNot(captured["image_component_overrides"], choices)
                    self.assertEqual(self.apply.call_args.args[1], family)
                    self.assertTrue(captured["_sampling_captured"])
                    self.assertEqual(request["image_component_overrides"], choices)
                    if "_api_job" in boundary:
                        self.assertIs(captured["_api_job"], boundary["_api_job"])
        self.assertNotIn(
            "not-installed.safetensors", [call.args[1] for call in self.lookup.call_args_list]
        )

    def test_ui_metadata_replay_changes_checkpoint_and_components_together(self):
        recorded = self.roles("SDXL", "recorded.safetensors")
        prompt = json.dumps(
            {
                "prompt": "saved UI prompt",
                "base_model_name": "SDXL.safetensors",
                "image_component_overrides": recorded,
            }
        )
        for family in FAMILIES:
            with self.subTest(family=family):
                captured = self.capture(
                    {
                        "base_model_name": family + ".safetensors",
                        "prompt": prompt,
                        "controlnet": {"type": "img2img"},
                        "image_component_overrides": self.roles(family, "old.safetensors"),
                    }
                )
                self.assertEqual(captured["base_model_name"], "SDXL.safetensors")
                self.assertEqual(captured["image_component_overrides"], recorded)
                self.assertEqual(self.apply.call_args.args[1], "SDXL")


if __name__ == "__main__":
    unittest.main()
