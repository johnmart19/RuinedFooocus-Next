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
