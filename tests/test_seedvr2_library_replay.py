"""Typed Library/Hub restoration callers retain the selected original and assets."""

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.gallery.actions import source_record, source_path
from ruinedfooocus.features.gallery.editing_workflow import build_library_request
from ruinedfooocus.features.gallery.seedvr2_replay import (
    prepare_record,
    replay_settings,
    restoration_metadata,
)
from ruinedfooocus.features.generation.image_preparation import capture_image_assets
from ruinedfooocus.features.image_processing import native_seedvr2 as native
from ruinedfooocus.features.image_editing.tool_activation import activation_values, installed_tool
from ruinedfooocus.features.gallery.image_metadata import read_metadata
from ruinedfooocus.features.gallery.index_store import MEDIA_SUFFIXES
import test_native_seedvr2 as fixtures
from test_utility_sampling_capture import function

ROOT = Path(__file__).resolve().parents[1]


class SeedVR2LibraryReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeSeedVR2Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.manager = self.fixture.root, self.fixture.manager
        self.manager.model_paths["cache_path"] = self.root / "cache"
        self.metadata = dict(
            controlnet={"type": native.KIND},
            seedvr2_scale=1,
            seedvr2_color="none",
            seed=17,
            native_seedvr2_source=dict(
                native="SeedVR2",
                model=self.fixture.model.name,
                codec=self.fixture.codec.name,
                model_bytes=self.fixture.model.stat().st_size,
                codec_bytes=self.fixture.codec.stat().st_size,
                model_sha256=hashlib.sha256(self.fixture.model.read_bytes()).hexdigest(),
                codec_sha256=hashlib.sha256(self.fixture.codec.read_bytes()).hexdigest(),
            ),
        )
        self.path = self.root / "selected-original.png"
        info = PngInfo()
        info.add_text("parameters", json.dumps(self.metadata))
        self.fixture.image.save(self.path, pnginfo=info)
        self.record = source_record(self.path)
        self.presets = {"Restore still image (SeedVR2)": {"type": native.KIND}}
        self.resets = dict(
            input_image=None,
            reference_images=[],
            inpaint_toggle=False,
            _native_main_source=None,
            _generation_artifacts=[],
        )

    def test_replay_matches_both_component_hashes_without_a_prompt_or_checkpoint(self):
        self.assertIsNone(read_metadata(self.metadata))
        values = replay_settings(self.metadata, self.manager)
        self.assertEqual(
            (values["seedvr2_model"], values["seedvr2_vae"]),
            (str(self.fixture.model), str(self.fixture.codec)),
        )
        self.assertEqual(
            (values["seedvr2_scale"], values["seedvr2_color"], values["seed"]), (1, "none", 17)
        )
        prepared = prepare_record(
            self.record,
            self.metadata,
            self.manager,
            presets=self.presets,
            custom="Custom...",
            resets=self.resets,
        )
        np.testing.assert_array_equal(
            np.asarray(prepared["input_image"]), np.asarray(self.fixture.image)
        )
        self.assertEqual(prepared["cn_selection"], "Restore still image (SeedVR2)")
        self.assertNotIn("prompt", prepared)
        self.assertNotIn("base_model_name", prepared)
        self.assertIsNone(prepared["_native_main_source"])
        self.assertNotEqual(prepared["input_image"], Image.new("RGBA", (1, 1)))

    def test_ambiguous_basename_wrong_hash_and_changed_source_are_refused(self):
        duplicate = self.root / "checkpoints" / self.fixture.model.name
        duplicate.write_bytes(self.fixture.model.read_bytes().replace(b"F16", b"F32"))
        self.assertEqual(
            replay_settings(self.metadata, self.manager)["seedvr2_model"], str(self.fixture.model)
        )
        duplicate.write_bytes(self.fixture.model.read_bytes())
        with self.assertRaisesRegex(ValueError, "uniquely matched"):
            replay_settings(self.metadata, self.manager)
        duplicate.unlink()
        bad = json.loads(json.dumps(self.metadata))
        bad["native_seedvr2_source"]["codec_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "uniquely matched"):
            replay_settings(bad, self.manager)
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_record(
                self.record,
                self.metadata,
                self.manager,
                presets=self.presets,
                custom="Custom...",
                resets=self.resets,
            )

    def test_actual_library_availability_and_setup_callback_bypass_generation_resolution(self):
        buttons = ["prompt", "setup", "edit", "inpaint", "upscale", "rembg", "export", "run"]
        availability = function(
            "ruinedfooocus/features/gallery/ui.py",
            "availability",
            dict(
                read_metadata=read_metadata,
                action_buttons=buttons,
                navigation_controls=[],
                gr=gr,
                Path=Path,
                MEDIA_SUFFIXES=MEDIA_SUFFIXES,
            ),
        )
        self.assertTrue(availability(self.record, self.metadata)["setup"]["interactive"])
        self.assertFalse(availability(self.record, self.metadata)["prompt"]["interactive"])
        shared = NS(path_manager=self.manager)
        guard, forbidden = Mock(), Mock(side_effect=AssertionError("generation setup resolution"))
        navigate = Mock(
            side_effect=lambda updates, tab=None, *, reset_source=False: dict(
                updates, selected_tab=tab
            )
        )
        callback = function(
            "ruinedfooocus/features/gallery/ui.py",
            "use_setup",
            dict(
                validate_view_record=guard,
                source_path=source_path,
                gr=gr,
                shared=shared,
                get_ctrl=lambda name: name,
                REFERENCE_RESETS=self.resets,
                reuse_setup=forbidden,
                reset_preset=forbidden,
                navigate=navigate,
            ),
        )
        import ruinedfooocus.features.image_editing as editing

        controls = NS(cn_options=self.presets, NEWCN="Custom...")
        with (
            patch.dict(sys.modules, {"ruinedfooocus.features.image_editing.controlnet": controls}),
            patch.object(editing, "controlnet", controls, create=True),
        ):
            updates = callback(
                self.record, self.metadata, "unused", gr.Request(session_hash="owner")
            )
        self.assertEqual(updates["selected_tab"], "image-tools")
        self.assertEqual(updates["seedvr2_vae"]["value"], str(self.fixture.codec))
        self.assertEqual(updates["input_image"]["value"].size, (11, 7))
        forbidden.assert_not_called()
        self.assertEqual(guard.call_count, 2)
        navigate.assert_called_once()
        self.assertIs(navigate.call_args.kwargs["reset_source"], True)

    def test_direct_library_generation_uses_original_and_typed_seed_settings(self):
        settings = replay_settings(self.metadata, self.manager)
        data = build_library_request(
            dict(settings, input_image=Image.new("RGB", (1, 1)), loras=["hidden"]),
            self.record,
            prompt="#!not executable",
            negative="unused",
            checkpoint=None,
            action=native.KIND,
            resets=self.resets,
            features={},
            presets=self.presets,
            custom="Custom...",
            tool_resource=settings["seedvr2_model"],
        )
        self.assertNotIn("base_model_name", data)
        self.assertEqual(data["prompt"], "")
        data = capture_image_assets(data, self.manager, {"type": native.KIND}, offline=True)
        np.testing.assert_array_equal(
            np.asarray(data["input_image"]), np.asarray(self.fixture.image)
        )
        self.assertEqual((data["custom_width"], data["custom_height"]), (11, 7))
        self.assertEqual(data["seedvr2_vae"], str(self.fixture.codec))
        self.assertEqual(data["loras"], [])

    def test_hub_typed_activation_routes_seed_model_without_loading_a_generation_checkpoint(self):
        row = dict(
            category="diffusion_models",
            local=True,
            path=str(self.fixture.model),
            local_path=str(self.fixture.model),
            signature=self.fixture.model.stat().st_size,
        )
        with patch(
            "ruinedfooocus.features.models.acquisition_actions.validate_local_record",
            return_value=self.fixture.model,
        ):
            resource = installed_tool(row, self.manager)
        self.assertEqual(
            resource,
            dict(kind=native.KIND, resource="seedvr2_model", value=str(self.fixture.model)),
        )
        values = activation_values(native.KIND, resource, presets=self.presets, custom="Custom...")
        self.assertEqual(values["seedvr2_model"], str(self.fixture.model))
        self.assertNotIn("cn_type", values)

    def test_installed_library_run_callback_submits_seed_controls_with_request_owner(self):
        import test_library_editing_ui as library_fixtures
        from gradio.helpers import special_args

        fixture = library_fixtures.LibraryEditingUiTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.policy.controlnet.cn_options.update(self.presets)
        fixture.policy.checkpoint_features = Mock(side_effect=AssertionError("generation lookup"))
        arguments = fixture.arguments()
        arguments[4], arguments[5], arguments[9] = None, native.KIND, str(self.fixture.model)
        arguments.extend([str(self.fixture.codec), 1.25, "none"])
        callback = fixture.functions["run"].fn
        inputs, *_ = special_args(
            callback, arguments, request=gr.Request(session_hash="seed-owner")
        )
        stream = callback(*inputs)
        next(stream)
        next(stream)
        stream.close()
        data = fixture.captured[0]
        self.assertEqual(fixture.validate.call_args.args[1].session_hash, "seed-owner")
        self.assertEqual(
            (data["seedvr2_model"], data["seedvr2_vae"]),
            (str(self.fixture.model), str(self.fixture.codec)),
        )
        self.assertEqual(data["seedvr2_scale"], 1.25)
        self.assertEqual(data["input_image"].size, (73, 121))
        self.assertEqual(data["input_image"].getpixel((72, 120))[3], 127)
        self.assertEqual(data["prompt"], "")
        self.assertNotIn("base_model_name", data)
        fixture.policy.checkpoint_features.assert_not_called()


if __name__ == "__main__":
    unittest.main()
