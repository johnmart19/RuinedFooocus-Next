"""PNG replay binds components to its effective model before worker admission."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image

from ruinedfooocus.features.models import component_profiles as profiles
from ui_test_helpers import function_from


class PresetComponentCallbacks(unittest.TestCase):
    def exercise(self, preset, *, mode=4, missing=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model-b.safetensors"
            checkpoint.write_bytes(b"{}")
            values = profiles.update_choices(
                {}, checkpoint, profiles.checkpoint_identity(checkpoint), {"clip_l": "saved-b"}
            )
            metadata = {"software": "RuinedFooocus", "base_model_name": "b", "loras": []}
            metadata.update(preset)
            from PIL.PngImagePlugin import PngInfo

            info = PngInfo()
            info.add_text("parameters", json.dumps(metadata))
            source = root / "preset.png"
            Image.new("RGB", (1, 1)).save(source, pnginfo=info)
            controls = {
                "base_model_name": "a",
                "preset_selection": str(source),
                "image_number": 1,
                "image_component_overrides": {"clip_l": "session-a"},
            }
            submitted, warning = Mock(return_value=1), Mock()
            shared = SimpleNamespace(
                models=SimpleNamespace(get_file=lambda *args: None if missing else checkpoint),
                settings=SimpleNamespace(default_settings=values),
                state={},
            )
            callback = function_from(
                "ruinedfooocus/ui/app.py",
                "generate_clicked",
                {
                    "state": {"ctrls_name": list(controls)},
                    "generation_task": "generation-task-state",
                    "settings": {"preset_mode": mode},
                    "shared": shared,
                    "Image": Image,
                    "json": json,
                    "append_work": submitted,
                    "ui_controlnet": SimpleNamespace(
                        tool_kind=lambda selection, control_type: (
                            (control_type or "").lower() if selection == "Custom..." else None
                        )
                    ),
                    "update_clicked": lambda: {},
                    "update_results": lambda _: {},
                    "worker": SimpleNamespace(task_result=lambda _: ("results", [])),
                    "gr": SimpleNamespace(Warning=warning),
                    "traceback": SimpleNamespace(print_exc=lambda: None),
                    "time": SimpleNamespace(sleep=lambda _: None),
                },
            )
            list(callback(*controls.values()))
            return submitted, warning

    def test_model_replay_uses_saved_choices_when_old_png_has_no_mapping(self):
        submitted, warning = self.exercise({})
        data = submitted.call_args.args[0]
        self.assertEqual(data["base_model_name"], "b")
        self.assertEqual(data[profiles.REQUEST_KEY], {"clip_l": "saved-b"})
        warning.assert_not_called()

    def test_recorded_automatic_override_survives_replay(self):
        submitted, _ = self.exercise({profiles.REQUEST_KEY: {"clip_l": None}})
        self.assertEqual(submitted.call_args.args[0][profiles.REQUEST_KEY], {"clip_l": None})

    def test_non_model_preset_mode_keeps_current_model_and_components(self):
        submitted, _ = self.exercise({"width": 832, "height": 1216}, mode=1)
        data = submitted.call_args.args[0]
        self.assertEqual(data["base_model_name"], "a")
        self.assertEqual(data[profiles.REQUEST_KEY], {"clip_l": "session-a"})
        self.assertEqual((data["width"], data["height"]), (832, 1216))

    def test_missing_preset_model_never_submits_partial_setup(self):
        submitted, warning = self.exercise({}, missing=True)
        submitted.assert_not_called()
        self.assertIn("no longer installed", warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
