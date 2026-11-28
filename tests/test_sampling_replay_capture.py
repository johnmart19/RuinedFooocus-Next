"""The existing metadata importer runs before an immutable sampling plan."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.generation.request_metadata import (
    capture_sampling_replay,
    metadata_intent,
)
from ruinedfooocus.features.generation.size_plan import SizePlan
from ui_test_helpers import function_from


class SamplingReplayCaptureTests(unittest.TestCase):
    def setUp(self):
        self.lookup = Mock(return_value={"baseModel": "QwenImage"})
        self.models = SimpleNamespace(
            get_models_by_path=self.lookup,
            get_model_base=lambda info: info["baseModel"],
            get_file=lambda *args: None,
        )
        self.recorded = dict(
            prompt="recorded prompt",
            negative="recorded negative",
            steps=17,
            cfg=4.5,
            base_model_name="saved.safetensors",
            loras=[["recorded", "0.5 - style.safetensors"]],
            width=504,
            height=256,
            size_plan=SizePlan.create((1001, 503), 2).metadata(),
            sampling_shift=None,
            effective_sampling_shift=3.1,
            effective_sampling_shift_mode="flow",
        )
        self.request = dict(
            prompt=json.dumps(self.recorded),
            base_model_name="current.safetensors",
            custom_steps=25,
            cfg=7,
            negative="current negative",
            loras=[],
            performance_selection="Model Defaults",
            style_selection=["current style"],
        )

    def test_sampling_replay_freezes_recorded_settings_and_geometry_before_queue(self):
        result = capture_sampling_replay(self.request, self.models)
        self.assertEqual((result["custom_steps"], result["cfg"]), (17, 4.5))
        self.assertEqual(result["negative"], "recorded negative")
        self.assertEqual(result["loras"], self.recorded["loras"])
        self.assertEqual(result["prompt"], "recorded prompt")
        self.assertEqual((result["width"], result["height"]), (1001, 503))
        self.assertEqual(result["aspect_ratios_selection"], "Custom...")
        self.assertEqual(result["performance_selection"], "Own settings")
        self.assertEqual(result["sampling_shift"], 3.1)
        self.assertTrue(result["_metadata_prompt"])
        self.assertEqual(self.request["custom_steps"], 25)

    def test_api_and_tool_json_stay_literal(self):
        for boundary in ({"_api_job": object()}, {"task_type": "tool_call"}):
            for prompt in (self.request["prompt"], [self.request["prompt"]]):
                with self.subTest(boundary=tuple(boundary), prompt=prompt):
                    request = self.request | boundary | {"prompt": prompt}
                    self.assertEqual(capture_sampling_replay(request, self.models), request)
                    self.assertEqual(metadata_intent(request), (request, None))
        self.lookup.assert_not_called()

    def test_ui_single_document_list_replays_before_control_and_sampling_capture(self):
        request = self.request | {"prompt": [self.request["prompt"]]}
        with patch(
            "ruinedfooocus.features.generation.image_preparation.image_control_data",
            side_effect=lambda data: dict(data),
        ) as controls:
            intent, imported = metadata_intent(request)
        self.assertEqual(controls.call_args.args[0]["prompt"], self.request["prompt"])
        self.assertEqual(intent["base_model_name"], "saved.safetensors")
        self.assertEqual(imported, self.recorded)
        self.assertEqual(
            capture_sampling_replay(intent, self.models),
            capture_sampling_replay(self.request, self.models),
        )

    def test_ordinary_prompt_lists_remain_unchanged(self):
        for prompt in (["one"], ["one", "two"], [self.request["prompt"], "two"]):
            request = self.request | {"prompt": prompt}
            self.assertEqual(metadata_intent(request), (request, None))
            self.assertEqual(capture_sampling_replay(request, self.models), request)
        self.lookup.assert_not_called()

    def test_replay_is_idempotent_without_late_lookup_or_default_reapplication(self):
        result = capture_sampling_replay(self.request, self.models)
        self.lookup.reset_mock()
        self.assertEqual(capture_sampling_replay(result, self.models), result)
        self.lookup.assert_not_called()

    def test_explicit_importer_delegates_the_same_policy_and_preserves_mutation_contract(self):
        importer = function_from(
            "ruinedfooocus/features/prompting/prompt_processing.py", "process_metadata", {}
        )
        request = dict(self.request)
        with patch.dict(
            "sys.modules", {"ruinedfooocus.app.state": SimpleNamespace(models=self.models)}
        ):
            result = importer(request)
        self.assertIs(result, request)
        self.assertEqual(result, capture_sampling_replay(self.request, self.models))

    def test_invalid_recorded_prompt_fails_before_model_lookup(self):
        with self.assertRaisesRegex(ValueError, "text prompt"):
            capture_sampling_replay({"prompt": json.dumps({"prompt": 9})}, self.models)
        self.lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
