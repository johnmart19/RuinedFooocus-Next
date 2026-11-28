"""Ready-use selectors reuse trusted capture without acquiring during UI browsing."""

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import utility_defaults as defaults
from ruinedfooocus.features.image_processing import native_seedvr2


class UtilityDefaultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.DOWNLOADABLE_FILES = {}
        for path in resource_path("model_catalogue").glob("*.json"):
            self.DOWNLOADABLE_FILES.update(json.loads(path.read_text(encoding="utf-8")))
        for entry in self.DOWNLOADABLE_FILES.values():
            (self.root / entry["path"].removeprefix("path_")).mkdir(exist_ok=True)
        self.model_paths = {}
        self.image = Image.new("RGBA", (8, 8), (10, 20, 30, 128))

    def get_model_folders(self, category):
        return (self.root / category,)

    def preparation_request(self, key):
        entry = self.DOWNLOADABLE_FILES[key]
        return PreparationRequest(
            entry["url"],
            self.root / entry["path"].removeprefix("path_") / entry["filename"],
            entry.get("size"),
            entry.get("sha256"),
        )

    def capture(self, kind, *, offline=True, **values):
        return assets.capture_image_assets(
            dict(
                input_image=self.image,
                seedvr2_scale=1,
                segment_positive_points='[{"x":2,"y":2}]',
                **values,
            ),
            self,
            {"type": kind},
            offline=offline,
        )

    def test_all_defaults_bind_exact_registered_roles_without_acquiring(self):
        with patch.object(assets.preparation_service, "submit") as submit:
            for kind, fields in defaults.PREFERENCES.items():
                with self.subTest(kind=kind):
                    data = self.capture(kind)
                    items = data[assets.KEY].artifacts
                    self.assertEqual(len(items), len(fields))
                    for field, (key, _, _) in fields.items():
                        request = self.preparation_request(key)
                        self.assertEqual(data[field], str(request.destination))
                        item = next(row for row in items if row.request == request)
                        self.assertIsNone(item.signature)
                        self.assertFalse(item.request.destination.exists())
            submit.assert_not_called()

    def test_offline_failure_names_missing_action_component_without_acquiring(self):
        for kind in defaults.PREFERENCES:
            with self.subTest(kind=kind):
                data = self.capture(kind)
                service = Mock()
                status = assets.image_readiness(data)
                self.assertEqual(status.state, "blocked")
                self.assertTrue(
                    any(
                        row.destination.name in " ".join(status.reasons)
                        for row in status.requirements
                    )
                )
                with self.assertRaisesRegex(ValueError, "Offline mode"):
                    assets.prepare_image_assets(data, check_cancelled=Mock(), service=service)
                service.submit.assert_not_called()

    def test_online_run_reaches_existing_acquisition_only_after_capture(self):
        with patch.dict(os.environ, {"RF_OFFLINE": "0", "HF_HUB_OFFLINE": "0"}):
            data = self.capture("restore-seedvr2", offline=False)
        service = Mock()
        service.submit.side_effect = RuntimeError("owned fake acquisition")
        with self.assertRaisesRegex(RuntimeError, "owned fake acquisition"):
            assets.prepare_image_assets(data, check_cancelled=Mock(), service=service)
        self.assertEqual(service.submit.call_args.args[0], data[assets.KEY].artifacts[0].request)

    def test_compatible_local_codec_wins_and_wrong_header_is_not_a_default(self):
        root = self.root / "vae"
        (root / "aaa-wrong.safetensors").write_bytes(b"not a safe tensor")
        path = root / "local-codec.safetensors"
        header = json.dumps(
            {
                native_seedvr2.CODEC_KEY: dict(
                    dtype="F16", shape=[4, 1, 1, 1, 1], data_offsets=[0, 0]
                )
            }
        ).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header)
        selected = defaults.default_components(self, "restore-seedvr2")
        self.assertEqual(selected["seedvr2_vae"], str(path))
        choices = defaults.component_choices(self, "restore-seedvr2", "seedvr2_vae")
        self.assertEqual(choices[0][1], str(path))
        self.assertNotIn(str(root / "aaa-wrong.safetensors"), [value for _, value in choices])

    def test_explicit_invalid_selection_is_preserved_and_capture_rejects_it(self):
        selected = defaults.default_components(
            self, "depth-map", dict(native_vision_model="../outside.safetensors")
        )
        self.assertEqual(selected["native_vision_model"], "../outside.safetensors")
        with self.assertRaises(ValueError):
            self.capture("depth-map", **selected)
        self.assertEqual(defaults.default_components(self, "image-transform"), {})
        self.assertEqual(defaults.default_components(self, "moge-depth"), {})

    def test_changed_registry_tuple_cannot_become_a_first_use_default(self):
        key = defaults.PREFERENCES["pose-map"]["image_utility_model"][0]
        self.DOWNLOADABLE_FILES[key] = dict(
            self.DOWNLOADABLE_FILES[key], url="https://example.invalid/untrusted"
        )
        self.assertEqual(defaults.component_choices(self, "pose-map", "image_utility_model"), [])
        with self.assertRaises(ValueError):
            self.capture("pose-map")

    def test_context_switch_replaces_known_other_tool_but_retains_unknown_missing_path(self):
        old = defaults.default_components(self, "face-landmarks")["image_utility_model"]
        choices = defaults.component_choices(self, "pose-map", "image_utility_model")
        new = defaults.picker_value(self, "pose-map", "image_utility_model", old, choices)
        self.assertEqual(new, choices[0][1])
        self.assertNotEqual(old, new)
        missing = str(self.root / "checkpoints/custom-missing.safetensors")
        self.assertEqual(
            defaults.picker_value(self, "pose-map", "image_utility_model", missing, choices),
            missing,
        )

    def test_real_gradio_pickers_start_ready_and_face_pose_switch_is_compatible(self):
        import asyncio
        import sys
        from types import SimpleNamespace
        import gradio as gr
        from gradio.state_holder import SessionState
        import ruinedfooocus.app
        from ruinedfooocus.features.image_processing import (
            ui_depth_utilities,
            ui_native_vision,
            ui_native_seedvr2,
        )

        controls = {}
        shared = SimpleNamespace(
            path_manager=self,
            settings=SimpleNamespace(default_settings={}),
            add_ctrl=lambda key, control: controls.update({key: control}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(assets.preparation_service, "submit") as submit,
        ):
            for kind, owner in (
                ("depth-map", ui_native_vision),
                ("restore-seedvr2", ui_native_seedvr2),
                ("face-landmarks", ui_depth_utilities),
            ):
                with self.subTest(kind=kind), gr.Blocks(analytics_enabled=False) as demo:
                    selection, tool, refresh = (
                        gr.Textbox(value=kind),
                        gr.Textbox(value=kind),
                        gr.Number(value=0),
                    )
                    image = gr.Image(type="pil", image_mode="RGBA")
                    if owner is ui_native_seedvr2:
                        owner.create_ui(selection, tool, refresh, lambda selection, tool: tool)
                    else:
                        owner.create_ui(
                            selection, tool, image, refresh, lambda selection, tool: tool
                        )
                    for field, value in defaults.default_components(self, kind).items():
                        self.assertEqual(controls[field].value, value)
                if kind == "face-landmarks":
                    index = next(
                        index
                        for index, fn in demo.fns.items()
                        if fn.fn.__name__ == "utility_context"
                    )
                    result = asyncio.run(
                        demo.process_api(
                            index,
                            [
                                "pose-map",
                                "pose-map",
                                controls["image_utility_model"].value,
                                None,
                                None,
                                None,
                            ],
                            state=SessionState(demo),
                        )
                    )["data"]
                    expected = defaults.default_components(self, "pose-map")["image_utility_model"]
                    self.assertEqual(result[2]["value"], expected)
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
