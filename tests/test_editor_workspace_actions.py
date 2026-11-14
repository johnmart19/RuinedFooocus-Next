"""Real P3/P4 capture from the canonical controls; no inference or acquisition."""

from concurrent.futures import CancelledError
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.app.resource_lifecycle import ui_request_owner
from ruinedfooocus.features.gallery.editor_session import EditorSessionStore
from ruinedfooocus.features.gallery.ui_workspace import EditorWorkspace


class EditorWorkspaceActionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.request = gr.Request(session_hash="workspace-owner")
        self.owner = ui_request_owner(self.request)
        self.store = EditorSessionStore(self.root / "sessions")
        self.source = self.image("source.png", "RGBA", (12, 24, 48, 127))
        self.snapshot = self.store.start(self.source, owner=self.owner)
        with gr.Blocks(analytics_enabled=False):
            self.workspace = EditorWorkspace(store=self.store)
        self.values = dict(
            _editor_context=self.workspace.public_context(self.snapshot),
            _editor_action="prompt-edit",
            prompt="literal instruction",
            negative="no blur",
            base_model_name="fixture.safetensors",
            image_engine="native",
            loras=[["detail.safetensors", 0]],
            cn_strength=0.37,
            _editor_denoise=0.37,
            custom_width=73,
            custom_height=121,
            _editor_framing="preserve",
            image_component_overrides={"vae": "selected-vae.safetensors"},
        )

    def image(self, name, mode, color, size=(73, 121)):
        path = self.root / name
        Image.new(mode, size, color).save(path)
        return path

    def test_literal_edit_uses_existing_controls_and_detached_p4_ticket(self):
        with patch(
            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
            return_value="SD1",
        ):
            data = self.workspace.build_request(self.values, request=self.request)
        step = data["_image_workflow_step"]
        self.values["loras"][0][1] = 0.9
        self.assertEqual(step.parameters["prompt"], "literal instruction")
        self.assertEqual(step.parameters["negative"], "no blur")
        self.assertEqual(step.parameters["loras"], [["detail.safetensors", 0]])
        self.assertEqual(step.parameters["denoise"], 0.37)
        self.assertEqual(
            step.parameters["image_component_overrides"], {"vae": "selected-vae.safetensors"}
        )
        self.assertEqual(data["input_image"].getpixel((0, 0)), (12, 24, 48, 127))
        self.assertFalse(data["obp_assume_direct_control"])
        self.assertTrue(data["_metadata_prompt"])
        self.assertEqual(data["cn_selection"], "None")
        self.assertEqual(
            self.store.snapshot(self.snapshot.session_id, owner=self.owner).revision,
            self.snapshot.revision,
        )

    def test_uploaded_png16_and_native_original_keep_saved_prompt_context(self):
        import json
        import numpy as np

        from ruinedfooocus.features.generation.request_metadata import capture_source_generation
        from ruinedfooocus.features.image_processing.native_image_outputs import (
            NativeImageOutput,
            result_presentation,
        )
        from ruinedfooocus.features.prompting.prompt_processing import process_prompt

        uploaded = self.root / "uploaded-16bit.png"
        source_prompt = "A woman with a blue scarf, café sign 0007."
        metadata = PngInfo()
        metadata.add_text(
            "parameters",
            json.dumps(
                dict(
                    Prompt=source_prompt,
                    Negative="old negative",
                    base_model_name="old-model.safetensors",
                    _api_job="untrusted saved job",
                    task_type="llama",
                    input_image="unrequested-source.png",
                )
            ),
        )
        Image.fromarray(np.full((121, 73), 32768, dtype=np.uint16)).save(uploaded, pnginfo=metadata)
        original = uploaded.read_bytes()
        preview = self.image("uploaded-preview.png", "RGB", (10, 20, 30))
        native = result_presentation([NativeImageOutput(str(uploaded), str(preview))])[2][0]
        instruction = "  Make her more detailed.  "
        for marker in (None, native):
            with self.subTest(native_original=marker is not None):
                snapshot = self.workspace.adopt_image(
                    str(preview if marker else uploaded), native_source=marker, owner=self.owner
                )
                self.assertEqual(snapshot.source_generation.get("prompt"), source_prompt)
                self.assertEqual(snapshot.original.frame.dtype, "uint16")
                self.assertEqual(Path(snapshot.original.path).read_bytes(), original)
                values = dict(
                    self.values,
                    prompt=instruction,
                    _editor_context=self.workspace.public_context(snapshot),
                )
                with patch(
                    "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                    return_value="SDXL",
                ):
                    data = self.workspace.build_request(values, request=self.request)
                data = capture_source_generation(data, store=self.store, owner=self.owner)
                self.assertEqual(
                    process_prompt([], instruction, data["negative"], data),
                    (source_prompt + "\n" + instruction, "no blur"),
                )
                self.assertEqual(data["raw_prompt"], instruction)
                self.assertEqual(data["base_model_name"], "fixture.safetensors")
                self.assertEqual(data["input_image"].size, (73, 121))
                for key in ("_api_job", "task_type", "input_image"):
                    self.assertNotIn(key, data["_source_generation"])
                self.assertEqual(uploaded.read_bytes(), original)

    def test_run_adopts_visible_image_and_keeps_instruction_model_and_manual_frame(self):
        visible = self.image("visible.png", "RGBA", (90, 80, 70, 120))
        values = dict(
            self.values,
            _editor_context=None,
            main_view=str(visible),
            input_image=Image.new("RGB", (8, 9), "red"),
            resolution_auto=False,
            _editor_framing="pad",
            custom_width=96,
            custom_height=128,
        )
        with patch(
            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
            return_value="SDXL",
        ):
            data = self.workspace.build_request(values, request=self.request)
        step = data["_image_workflow_step"]
        self.assertNotEqual(step.ticket.session_id, self.snapshot.session_id)
        self.assertEqual(Path(step.ticket.source.path).read_bytes(), visible.read_bytes())
        self.assertEqual(data["input_image"].getpixel((0, 0)), (90, 80, 70, 120))
        self.assertEqual(data["prompt"], values["prompt"])
        self.assertEqual(data["base_model_name"], values["base_model_name"])
        self.assertEqual(
            (step.parameters["target_width"], step.parameters["target_height"]), (96, 128)
        )
        self.assertEqual(self.store.snapshot(step.ticket.session_id, owner=self.owner).steps, ())
        self.assertEqual(visible.read_bytes(), Path(step.ticket.source.path).read_bytes())

    def test_latest_input_upload_wins_over_stale_visible_preview(self):
        values = dict(
            self.values,
            _editor_context=None,
            main_view=str(self.source),
            input_image=Image.new("RGBA", (37, 59), (100, 110, 120, 130)),
            _editor_source_input="input",
            resolution_auto=True,
        )
        with patch(
            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
            return_value="SDXL",
        ):
            data = self.workspace.build_request(values, request=self.request)
        self.assertEqual(data["input_image"].size, (37, 59))
        self.assertEqual(data["input_image"].getpixel((0, 0)), (100, 110, 120, 130))
        self.assertEqual((data["custom_width"], data["custom_height"]), (37, 59))

    def test_quick_tools_adopt_visible_source_instead_of_stale_hidden_input(self):
        visible = self.image("quick-visible.png", "RGBA", (70, 80, 90, 140))
        for tool in ("Face landmarks", "Whole-body pose", "Detect objects", "Segment by points"):
            with self.subTest(tool=tool):
                data = self.workspace.build_request(
                    dict(
                        self.values,
                        _editor_destination="editor",
                        _editor_action="generate",
                        _editor_context=None,
                        main_view=str(visible),
                        cn_selection=tool,
                        input_image=Image.new("RGB", (4, 5), "red"),
                    ),
                    request=self.request,
                )
                self.assertEqual(data["input_image"].getpixel((0, 0)), (70, 80, 90, 140))
                self.assertEqual(data["main_view"].size, (73, 121))
                self.assertEqual(data["editor_context"], data["_editor_context"])
                self.assertEqual(data["cn_selection"], tool)
                self.assertIsNone(data["_native_main_source"])

    def test_branding_placeholder_is_not_an_editor_source(self):
        from ruinedfooocus.core.paths import asset_path

        with self.assertRaisesRegex(ValueError, "Choose or upload"):
            self.workspace.build_request(
                dict(
                    self.values, _editor_context=None, main_view=str(asset_path("init_image.png"))
                ),
                request=self.request,
            )

    def test_edit_strength_is_independent_of_controlnet_and_retains_explicit_override(self):
        with patch(
            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
            return_value="SDXL",
        ):
            for strength in (1.0, 1.5):
                values = dict(self.values, cn_strength=strength)
                values.pop("_editor_denoise")
                data = self.workspace.build_request(values, request=self.request)
                self.assertEqual(data["_image_workflow_step"].parameters["denoise"], 0.3)
            data = self.workspace.build_request(
                dict(self.values, _editor_denoise=0.47, cn_strength=1.5), request=self.request
            )
        self.assertEqual(data["_image_workflow_step"].parameters["denoise"], 0.47)

    def test_upscale_defaults_to_native_two_times_and_preserves_explicit_model_target(self):
        data = self.workspace.build_request(
            dict(
                self.values,
                _editor_action="upscale",
                automatic_upscale="Off",
                custom_width=512,
                custom_height=768,
            ),
            request=self.request,
        )
        self.assertEqual(
            data["_image_workflow_step"].parameters["upscaler"], "2× General (RealESRGAN)"
        )
        self.assertEqual((data["custom_width"], data["custom_height"]), (146, 242))
        data = self.workspace.build_request(
            dict(
                self.values,
                _editor_action="upscale",
                automatic_upscale="4× UltraSharp",
                resolution_auto=False,
                custom_width=219,
                custom_height=363,
            ),
            request=self.request,
        )
        self.assertEqual(data["_image_workflow_step"].parameters["upscaler"], "4× UltraSharp")
        self.assertEqual((data["custom_width"], data["custom_height"]), (219, 363))

    def test_compose_accepts_background_without_intermediate_history_step(self):
        background = self.image("background.png", "RGBA", (40, 60, 200, 255))
        data = self.workspace.build_request(
            dict(self.values, _editor_action="compose", _editor_background=str(background)),
            request=self.request,
        )
        step = data["_image_workflow_step"]
        self.assertEqual(step.ticket.source.artifact_id, self.snapshot.current.artifact_id)
        self.assertIn("background_id", step.parameters)
        self.assertNotIn("foreground_id", step.parameters)
        self.assertEqual(len(step.ticket.additional_images), 1)
        self.assertEqual(self.store.snapshot(self.snapshot.session_id, owner=self.owner).steps, ())

    def test_fresh_upload_and_mask_are_owned_together_on_run(self):
        mask = self.image("fresh-mask.png", "L", 100)
        data = self.workspace.build_request(
            dict(
                self.values,
                _editor_context=None,
                main_view=str(self.source),
                _editor_action="inpaint",
                _editor_mask=str(mask),
            ),
            request=self.request,
        )
        step = data["_image_workflow_step"]
        self.assertEqual(step.mask.source_id, step.ticket.source.artifact_id)
        self.assertEqual(data["inpaint_view"]["layers"][0].getchannel("A").getpixel((0, 0)), 100)

    def test_reusing_gradio_upload_cache_does_not_mutate_owned_original_or_break_undo(self):
        from gradio.processing_utils import move_files_to_cache

        metadata = PngInfo()
        metadata.add_text("parameters", '{"prompt":"literal requested detail","seed":17}')
        Image.new("RGBA", (73, 121), (12, 24, 48, 127)).save(self.source, pnginfo=metadata)
        original_bytes = self.source.read_bytes()
        with gr.Blocks(analytics_enabled=False):
            preview = gr.Image(type="filepath", format="png")
        cache = str(self.root / "gradio-cache")

        def file_value(control):
            value = move_files_to_cache(
                control.postprocess(str(self.source)), control, postprocess=True
            )
            return value.model_dump() if hasattr(value, "model_dump") else value

        with (
            patch.object(preview, "GRADIO_CACHE", cache),
            patch.object(self.workspace.background, "GRADIO_CACHE", cache),
            patch(
                "ruinedfooocus.core.cache_contract.configured_cache_root",
                return_value=self.root / "owned-cache",
            ),
        ):
            cached = Path(file_value(preview)["path"])
            os.utime(cached, ns=(cached.stat().st_atime_ns, 1_000_000_000))
            before = cached.stat()
            original = self.workspace.adopt_image(str(cached), owner=self.owner)
            ticket = self.store.capture(
                original.session_id,
                original.revision,
                "process-image",
                source_id=original.current.artifact_id,
                owner=self.owner,
            )
            result = self.image("processed.png", "RGBA", (40, 60, 80, 127))
            done = self.store.complete(ticket, result, owner=self.owner)
            self.assertEqual(Path(file_value(self.workspace.background)["path"]), cached)
            self.assertNotEqual(cached.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(cached.read_bytes(), original_bytes)
            self.store.snapshot(done.session_id, owner=self.owner)
            undone = self.store.undo(done.session_id, done.revision, owner=self.owner)

        self.assertEqual(undone.current.artifact_id, original.original.artifact_id)
        self.assertNotEqual(Path(undone.current.path), cached)
        self.assertEqual(Path(undone.current.path).read_bytes(), original_bytes)
        with Image.open(undone.current.path) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((0, 0)), (12, 24, 48, 127))
            self.assertEqual(
                image.info["parameters"], '{"prompt":"literal requested detail","seed":17}'
            )
        self.assertEqual(self.source.read_bytes(), original_bytes)

    def test_borrowed_ui_source_replacement_during_copy_is_rejected(self):
        from ruinedfooocus.features.models.model_identity import file_signature

        captured = []

        def replace_source(path):
            signature = file_signature(path)
            if not captured:
                captured.append(signature)
                self.image("source.png", "RGBA", (90, 100, 110, 120))
            return signature

        owned = self.root / "owned-cache"
        with (
            patch("ruinedfooocus.core.cache_contract.configured_cache_root", return_value=owned),
            patch(
                "ruinedfooocus.features.models.model_identity.file_signature",
                side_effect=replace_source,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "source changed during capture"):
                self.workspace.persist_input(self.source)
        self.assertEqual(list(owned.rglob("source.png")), [])

    def test_failed_image_and_background_admission_remove_only_their_new_input_copy(self):
        owned = self.root / "owned-cache"
        malformed = self.root / "not-an-image.png"
        malformed.write_bytes(b"not an image")
        source_bytes = self.source.read_bytes()
        with patch("ruinedfooocus.core.cache_contract.configured_cache_root", return_value=owned):
            retained = self.workspace.adopt_image(self.source, owner=self.owner)
            retained_path = Path(retained.current.path)
            with self.assertRaises(OSError):
                self.workspace.adopt_image(malformed, owner=self.owner)
            self.assertEqual({path for path in owned.rglob("*") if path.is_file()}, {retained_path})
            with patch.object(self.store, "_save", side_effect=OSError("manifest write failed")):
                for admit in (
                    lambda: self.workspace.adopt_image(self.source, owner=self.owner),
                    lambda: self.workspace.build_request(
                        dict(
                            self.values,
                            _editor_action="select-background",
                            _editor_background=str(self.source),
                        ),
                        request=self.request,
                    ),
                ):
                    with self.assertRaisesRegex(OSError, "manifest write failed"):
                        admit()
                    self.assertEqual(
                        {path for path in owned.rglob("*") if path.is_file()}, {retained_path}
                    )
        self.assertEqual(retained_path.read_bytes(), source_bytes)
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(malformed.read_bytes(), b"not an image")

    def test_inpaint_captures_exact_owned_grayscale_mask_and_inversion_once(self):
        path = self.image("mask.png", "L", 64)
        mask = self.store.bind_mask(
            self.snapshot.session_id, path, self.snapshot.current.artifact_id, owner=self.owner
        )
        values = dict(
            self.values,
            _editor_action="inpaint",
            _editor_mask_invert=True,
            _editor_context=dict(self.values["_editor_context"], mask_id=mask.artifact_id),
        )
        data = self.workspace.build_request(values, request=self.request)
        step = data["_image_workflow_step"]
        self.assertTrue(step.mask.inverted)
        self.assertEqual(step.mask.source_frame_id, self.snapshot.current.frame_id)
        self.assertEqual(data["inpaint_view"]["layers"][0].getchannel("A").getpixel((0, 0)), 191)
        self.assertEqual(
            len(self.store.snapshot(self.snapshot.session_id, owner=self.owner).steps), 0
        )
        values["_editor_context"] = dict(self.values["_editor_context"])
        with self.assertRaisesRegex(ValueError, "grayscale selection mask"):
            self.workspace.build_request(values, request=self.request)
        cutout = self.image("cutout.png", "RGBA", (255, 255, 255, 128))
        with self.assertRaises(ValueError):
            self.store.bind_mask(
                self.snapshot.session_id,
                cutout,
                self.snapshot.current.artifact_id,
                owner=self.owner,
            )

    def test_foreign_or_changed_source_rejects_before_step_capture(self):
        with self.assertRaises(PermissionError):
            self.workspace.build_request(
                dict(self.values, _editor_action="remove-background"),
                request=gr.Request(session_hash="other-owner"),
            )
        self.image("source.png", "RGBA", (8, 8, 8, 255))
        with self.assertRaises(ValueError):
            self.workspace.build_request(
                dict(self.values, _editor_action="remove-background"), request=self.request
            )

    def test_generate_keeps_existing_random_profile_and_adapter_controls(self):
        values = dict(
            self.values,
            _editor_action="generate",
            seed_random=True,
            obp_assume_direct_control=True,
            reference_adapter="redux",
            reference_adapter_strength=0.6,
        )
        data = self.workspace.build_request(values, request=self.request)
        self.assertEqual(data, values)
        self.assertNotIn("_image_workflow_step", data)

    def test_readiness_reports_real_offline_missing_role_without_preparing_or_running(self):
        from ruinedfooocus.core.preparation import PreparationRequest
        from ruinedfooocus.features.generation.image_preparation import (
            ImageArtifact,
            ImageAssets,
            KEY,
        )
        from ruinedfooocus import app

        captured = []

        def capture(data):
            captured.append(data)
            missing = ImageArtifact(
                "background",
                "local-background.safetensors",
                PreparationRequest("", self.root / "missing.safetensors"),
                None,
            )
            return dict(data, **{KEY: ImageAssets((missing,), offline=True)})

        worker = SimpleNamespace(capture_image_data=capture)
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.async_worker": worker}),
            patch.object(app, "async_worker", worker, create=True),
        ):
            text = self.workspace.readiness(
                dict(self.values, _editor_action="remove-background"), request=self.request
            )
        self.assertIn("blocked", text)
        self.assertIn("background: local-background.safetensors", text)
        self.assertIn("Offline", text)
        self.assertEqual(len(captured), 1)
        with self.assertRaises(CancelledError):
            self.store.validate(captured[0]["_editor_operation"], owner=self.owner)
        self.assertEqual(
            len(self.store.snapshot(self.snapshot.session_id, owner=self.owner).steps), 0
        )

    def test_existing_painted_mask_becomes_owned_coverage_without_using_source_alpha(self):
        layer = Image.new("RGBA", (73, 121), (255, 255, 255, 64))
        target = self.root / "painted-mask.png"

        def persist(mask):
            mask.save(target)
            return target

        with patch.object(self.workspace, "persist_input", side_effect=persist):
            data = self.workspace.build_request(
                dict(self.values, _editor_action="inpaint", inpaint_view={"layers": [layer]}),
                request=self.request,
            )
        self.assertEqual(data["inpaint_view"]["layers"][0].getchannel("A").getpixel((0, 0)), 64)
        self.assertEqual(data["main_view"].getchannel("A").getpixel((0, 0)), 127)
        self.assertEqual(
            data["_image_workflow_step"].mask.source_id, self.snapshot.current.artifact_id
        )


if __name__ == "__main__":
    unittest.main()
