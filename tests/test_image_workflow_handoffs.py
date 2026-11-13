"""Typed native image handoffs against the actual artifact-backed editor store."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.gallery.editor_session import EditorSessionStore
from ruinedfooocus.features.image_editing import image_workflows as workflows
from ruinedfooocus.features.image_editing.inpaint_utils import prepare_inpaint, source_and_mask
from ruinedfooocus.features.image_editing.mask_handoffs import (
    capture_mask_handoff,
    inpaint_inputs,
)
from ruinedfooocus.features.image_processing import image_composition, sam_prompts


class ImageWorkflowHandoffTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.store = EditorSessionStore(self.root / "sessions")
        self.owner = ("fixture", "one")
        self.other_owner = ("fixture", "two")
        self.image = Image.new("RGBA", (24, 16), (200, 100, 40, 128))
        self.source = self.save(self.image, "original")
        self.snapshot = self.store.start(
            self.source, provenance={"prompt": "original prompt", "seed": 17}, owner=self.owner
        )
        self.mask_image = Image.new("L", self.image.size, 0)
        self.mask_image.paste(128, (4, 4, 12, 12))
        self.mask = self.store.bind_mask(
            self.snapshot.session_id,
            self.save(self.mask_image, "mask"),
            self.snapshot.current.artifact_id,
            owner=self.owner,
        )

    def save(self, image, name):
        path = self.root / (name + ".png")
        image.save(path)
        return path

    def handoff(self, **values):
        return capture_mask_handoff(
            self.store,
            self.snapshot.session_id,
            self.snapshot.revision,
            self.snapshot.current.artifact_id,
            self.mask.artifact_id,
            owner=self.owner,
            choice="union",
            **values,
        )

    def capture(self, action, parameters=None, *, snapshot=None, **values):
        snapshot = snapshot or self.snapshot
        return workflows.capture_step(
            self.store,
            snapshot.session_id,
            snapshot.revision,
            action,
            parameters or {},
            owner=self.owner,
            **values,
        )

    def complete(self, step, image, name, execution=None):
        return self.store.complete(
            step.ticket,
            self.save(image, name),
            execution=execution or {"kind": "processing"},
            owner=self.owner,
        )

    def test_native_soft_mask_is_selection_independent_of_cutout_alpha(self):
        scalar = torch.full((1, 16, 24), 0.5, device="cpu")
        mask = sam_prompts.render(
            scalar, self.image, cutout=False, precision=False, check_cancelled=lambda: None
        )[0]
        cutout = sam_prompts.render(
            scalar, self.image, cutout=True, precision=False, check_cancelled=lambda: None
        )[0]
        np.testing.assert_array_equal(np.array(mask), 128)
        np.testing.assert_array_equal(np.array(cutout.getchannel("A")), 64)
        ref = self.store.bind_mask(
            self.snapshot.session_id,
            self.save(mask, "native-mask"),
            self.snapshot.current.artifact_id,
            owner=self.owner,
        )
        handoff = capture_mask_handoff(
            self.store,
            self.snapshot.session_id,
            self.snapshot.revision,
            self.snapshot.current.artifact_id,
            ref.artifact_id,
            owner=self.owner,
            choice="instance",
            instance_index=1,
            instance_count=3,
        )
        request = inpaint_inputs(self.store, handoff, owner=self.owner)
        source, selection = source_and_mask(
            request["main_view"], request["inpaint_view"], preserve_alpha=True
        )
        self.assertEqual(source.getchannel("A").getextrema(), (128, 128))
        self.assertEqual(selection.getextrema(), (128, 128))
        self.assertEqual(handoff.metadata()["instance_index"], 1)

    def test_cutout_cannot_bind_or_masquerade_as_a_selection_mask(self):
        cutout = self.save(self.image, "cutout")
        with self.assertRaisesRegex(ValueError, "mask|grayscale"):
            self.store.bind_mask(
                self.snapshot.session_id,
                cutout,
                self.snapshot.current.artifact_id,
                owner=self.owner,
            )
        ref = self.store.bind_image(self.snapshot.session_id, cutout, owner=self.owner)
        with self.assertRaisesRegex(ValueError, "selection mask|mask"):
            capture_mask_handoff(
                self.store,
                self.snapshot.session_id,
                self.snapshot.revision,
                self.snapshot.current.artifact_id,
                ref.artifact_id,
                owner=self.owner,
                choice="union",
            )

    def test_white_black_and_stored_or_requested_inversion_apply_once(self):
        for stored, requested, expected in (
            (False, False, 128),
            (True, False, 127),
            (True, True, 128),
        ):
            ref = self.store.bind_mask(
                self.snapshot.session_id,
                self.mask.path,
                self.snapshot.current.artifact_id,
                inverted=stored,
                owner=self.owner,
            )
            handoff = capture_mask_handoff(
                self.store,
                self.snapshot.session_id,
                self.snapshot.revision,
                self.snapshot.current.artifact_id,
                ref.artifact_id,
                choice="union",
                invert=requested,
                owner=self.owner,
            )
            alpha = inpaint_inputs(self.store, handoff, owner=self.owner)["inpaint_view"]["layers"][
                0
            ].getchannel("A")
            self.assertEqual(alpha.getpixel((5, 5)), expected)
            self.assertEqual(alpha.getpixel((0, 0)), 255 if stored ^ requested else 0)

    def test_instance_choice_and_empty_masks_reject_before_preparation(self):
        for values in (
            dict(choice="automatic"),
            dict(choice="union", instance_index=0),
            dict(choice="instance", instance_index=None),
            dict(choice="instance", instance_index=2, instance_count=2),
            dict(choice="union", instance_count=True),
            dict(choice="union", invert="yes"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                capture_mask_handoff(
                    self.store,
                    self.snapshot.session_id,
                    self.snapshot.revision,
                    self.snapshot.current.artifact_id,
                    self.mask.artifact_id,
                    owner=self.owner,
                    **values,
                )
        ref = self.store.bind_mask(
            self.snapshot.session_id,
            self.save(Image.new("L", self.image.size), "empty"),
            self.snapshot.current.artifact_id,
            owner=self.owner,
        )
        empty = replace(self.handoff(), mask_id=ref.artifact_id, mask_frame_id=ref.frame_id)
        with patch(
            "ruinedfooocus.core.preparation.PreparationService.submit",
            side_effect=AssertionError("prepared"),
        ):
            with self.assertRaisesRegex(ValueError, "empty"):
                self.capture("inpaint", mask=empty)
        self.assertEqual(self.store.snapshot(self.snapshot.session_id, owner=self.owner).steps, ())

    def test_foreign_owner_same_size_different_source_and_changed_mask_are_rejected(self):
        with self.assertRaises(PermissionError):
            inpaint_inputs(self.store, self.handoff(), owner=self.other_owner)
        foreign = self.store.bind_image(
            self.snapshot.session_id, self.save(self.image, "another-source"), owner=self.owner
        )
        wrong = self.store.bind_mask(
            self.snapshot.session_id, self.mask.path, foreign.artifact_id, owner=self.owner
        )
        with self.assertRaisesRegex(ValueError, "exact|frame"):
            capture_mask_handoff(
                self.store,
                self.snapshot.session_id,
                self.snapshot.revision,
                self.snapshot.current.artifact_id,
                wrong.artifact_id,
                owner=self.owner,
                choice="union",
            )
        handoff = self.handoff()
        Image.new("L", self.image.size, 255).save(self.mask.path)
        with self.assertRaisesRegex(ValueError, "changed"):
            inpaint_inputs(self.store, handoff, owner=self.owner)

    def test_wrong_geometry_and_stale_revision_fail_before_compute(self):
        with self.assertRaisesRegex(ValueError, "frame|dimensions|match"):
            self.store.bind_mask(
                self.snapshot.session_id,
                self.save(Image.new("L", (12, 16), 255), "wrong-size"),
                self.snapshot.current.artifact_id,
                owner=self.owner,
            )
        step = self.capture(
            "prompt-edit", {"prompt": "first", "negative": "noise", "denoise": 0.42}
        )
        changed = self.complete(step, self.image, "edit")
        with self.assertRaisesRegex(ValueError, "changed|revision"):
            inpaint_inputs(self.store, self.handoff(), owner=self.owner)
        self.assertEqual(changed.revision, self.snapshot.revision + 1)

    def test_inpaint_captures_prompts_negative_loras_denoise_and_owned_masks(self):
        values = dict(
            prompt="replace sky",
            negative="noise",
            loras=[["id", "0.7 - lora"]],
            inpaint_denoise=0.37,
            inpaint_mask_grow=5,
        )
        plan = {"model": "captured", "components": [{"role": "vae", "id": "v1"}]}
        step = self.capture("inpaint", values, mask=self.handoff(), effective_plan=plan)
        values["loras"][0][1] = "9 - changed"
        plan["components"][0]["id"] = "changed"
        request = workflows.worker_request(step, self.store, owner=self.owner)
        self.assertEqual(request["prompt"], "replace sky")
        self.assertEqual(request["negative"], "noise")
        self.assertEqual(request["loras"], [["id", "0.7 - lora"]])
        self.assertEqual(request["inpaint_denoise"], 0.37)
        self.assertEqual(request["inpaint_mask_grow"], 5)
        self.assertEqual(step.ticket.effective_plan["components"][0]["id"], "v1")
        self.assertEqual(step.ticket.mask_id, self.mask.artifact_id)
        self.assertEqual(request[workflows.EDITOR_KEY], step.ticket)
        self.assertEqual(request["main_view"].getchannel("A").getextrema(), (128, 128))

    def test_inpaint_geometry_reuses_the_existing_identical_mask_transform(self):
        step = self.capture(
            "inpaint",
            {"target_width": 32, "target_height": 32, "image_resize_mode": "pad"},
            mask=self.handoff(),
        )
        request = workflows.worker_request(step, self.store, owner=self.owner)
        source, mask = source_and_mask(
            request["main_view"], request["inpaint_view"], preserve_alpha=True
        )
        pixels, coverage, plan = prepare_inpaint(source, mask, (32, 32), "pad")
        self.assertEqual(tuple(pixels.shape[1:3]), tuple(coverage.shape[1:3]))
        self.assertEqual(plan["mode"], "pad")
        self.assertEqual(coverage[0, 0].count_nonzero().item(), 0)

    def test_edit_size_changes_need_explicit_pad_or_crop_and_valid_denoise(self):
        for values in (
            {"target_width": 32},
            {"denoise": 0},
            {"denoise": float("nan")},
            {"image_resize_mode": "stretch"},
            {"target_width": True},
            {"input_image": "transient"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.capture("prompt-edit", values)

    def test_alpha_hook_preserves_byte_and_float_outputs_and_rejects_wrong_frames(self):
        step = self.capture("prompt-edit", {"prompt": "retouch", "denoise": 0.2})
        data = workflows.worker_request(step, self.store, owner=self.owner)
        output = workflows.finish_worker_output(
            data, Image.new("RGB", self.image.size), store=self.store, owner=self.owner
        )
        self.assertEqual(output.getchannel("A").getextrema(), (128, 128))
        precision = np.full((16, 24, 3), 0.1234567, dtype=np.float32)
        result = workflows.finish_worker_output(data, precision, store=self.store, owner=self.owner)
        np.testing.assert_array_equal(result[..., :3], precision)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[..., 3], 128 / 255)
        native_rgba = Image.new("RGBA", self.image.size, (30, 60, 90, 64))
        self.assertIs(
            workflows.finish_worker_output(data, native_rgba, store=self.store, owner=self.owner),
            native_rgba,
        )
        with self.assertRaisesRegex(ValueError, "frame"):
            workflows.finish_worker_output(
                data, Image.new("RGB", (12, 16)), store=self.store, owner=self.owner
            )
        self.assertIs(workflows.finish_worker_output({}, precision), precision)

    def test_alpha_pad_and_deliberate_flatten_are_explicit(self):
        step = self.capture(
            "prompt-edit", {"image_resize_mode": "pad", "target_width": 32, "target_height": 32}
        )
        result = workflows.finish_pixels(
            step, self.store, Image.new("RGB", (32, 32)), owner=self.owner
        )
        self.assertEqual(result.getchannel("A").getpixel((0, 0)), 0)
        self.assertEqual(result.getchannel("A").getpixel((16, 16)), 128)
        flat = self.capture("prompt-edit", {"alpha_policy": "flatten-black"})
        request = workflows.worker_request(flat, self.store, owner=self.owner)
        self.assertEqual(request["input_image"].mode, "RGB")
        self.assertEqual(request["input_image"].getpixel((0, 0)), (100, 50, 20))
        paint = self.capture("inpaint", {"alpha_policy": "flatten-black"}, mask=self.handoff())
        request = workflows.worker_request(paint, self.store, owner=self.owner)
        self.assertEqual(request["main_view"].mode, "RGB")
        self.assertEqual(request["main_view"].getpixel((0, 0)), (100, 50, 20))

    def test_exact_upscale_target_and_source_alpha_reach_native_capture(self):
        from ruinedfooocus.features.generation.size_plan import KEY, capture_size_plan

        step = self.capture(
            "upscale",
            {"upscaler": "RealESRGAN_x2plus.pth", "target_width": 61, "target_height": 43},
        )
        data = workflows.worker_request(step, self.store, owner=self.owner)
        data = capture_size_plan(data, None, controls=data["controlnet"])
        self.assertEqual(data[KEY].target, (61, 43))
        self.assertEqual(data[KEY].scale, 2)
        self.assertEqual(data["input_image"].getchannel("A").getextrema(), (128, 128))
        self.assertEqual(data["loras"], [])
        self.assertNotIn("base_model_name", data)
        with self.assertRaisesRegex(ValueError, "target"):
            self.capture("upscale", {"upscaler": "RealESRGAN_x2plus.pth"})

    def test_scalar_presets_exclude_uploads_masks_paths_graphs_and_nonfinite_values(self):
        for values in (
            {"input_image": "base64"},
            {"mask": "image_123"},
            {"model_path": "C:/weights/model"},
            {"steps": ["repeat"]},
            {"background_mode": "C:/private"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                workflows.preset_values("background-replacement", values)
        for value in (None, True, float("nan"), "C:/private"):
            with self.assertRaises(ValueError):
                workflows.preset_values("portrait-enhancement", {"denoise": value})
        values = workflows.preset_values("portrait-enhancement", {"denoise": 0.23})
        self.assertEqual(json.loads(json.dumps(values))["denoise"], 0.23)

    def test_noop_background_skip_does_not_add_a_completed_operation(self):
        cursor = workflows.begin_workflow(
            self.store, self.snapshot.session_id, "background-replacement", owner=self.owner
        )
        skipped = workflows.skip_background_removal(self.store, cursor, owner=self.owner)
        self.assertEqual(skipped.next_action, "select-background")
        self.assertEqual(skipped.foreground_id, self.snapshot.current.artifact_id)
        after = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(after.revision, self.snapshot.revision)
        self.assertEqual(after.steps, ())

    def test_complete_background_chain_keeps_each_source_target_and_lineage(self):
        cursor = workflows.begin_workflow(
            self.store, self.snapshot.session_id, "background-replacement", owner=self.owner
        )
        cursor = workflows.skip_background_removal(self.store, cursor, owner=self.owner)
        background_image = Image.new("RGBA", (32, 32), (0, 0, 240, 255))
        background = self.store.bind_image(
            self.snapshot.session_id, self.save(background_image, "background"), owner=self.owner
        )
        selected = self.capture(
            "select-background", {"background_id": background.artifact_id}, cursor=cursor
        )
        path, execution = workflows.selected_background(selected, self.store, owner=self.owner)
        selected_snapshot = self.store.complete(
            selected.ticket, path, execution=execution, owner=self.owner
        )
        cursor = workflows.advance_workflow(cursor, selected, selected_snapshot)
        self.assertEqual(selected.parameters["target_width"], 32)
        compose = self.capture(
            "compose",
            {
                "composition_mode": "SRC_OVER",
                "composition_x": 2,
                "composition_y": 3,
                "composition_width": 24,
                "composition_height": 16,
            },
            snapshot=selected_snapshot,
            cursor=cursor,
        )
        request = workflows.worker_request(compose, self.store, owner=self.owner)
        with (
            patch(
                "ruinedfooocus.features.gallery.editor_session.default_store",
                return_value=self.store,
            ),
            patch("ruinedfooocus.app.resource_lifecycle.current_owner", return_value=self.owner),
        ):
            captured = image_composition.capture(request, "image-composite")
        output = image_composition.process(captured, lambda: None)
        self.assertEqual(output.size, (32, 32))
        self.assertEqual(output.getpixel((0, 0)), (0, 0, 240, 255))
        self.assertEqual(output.getpixel((2, 3)), (100, 50, 140, 255))
        composed = self.complete(compose, output, "composed")
        cursor = workflows.advance_workflow(cursor, compose, composed)
        edit = self.capture(
            "prompt-edit",
            {"prompt": "polish", "negative": "noise", "denoise": 0.28},
            snapshot=composed,
            cursor=cursor,
        )
        edited = self.complete(edit, output, "edited", {"kind": "fixture", "computed": False})
        cursor = workflows.advance_workflow(cursor, edit, edited)
        upscale = self.capture(
            "upscale",
            {"upscaler": "RealESRGAN_x2plus.pth", "target_width": 64, "target_height": 64},
            snapshot=edited,
            cursor=cursor,
        )
        enlarged = self.complete(
            upscale,
            output.resize((64, 64)),
            "upscaled-fixture",
            {
                "kind": "fixture",
                "computed": False,
                "geometry": {
                    "policy": "explicit-fixture-resize",
                    "source": [32, 32],
                    "result": [64, 64],
                },
            },
        )
        cursor = workflows.advance_workflow(cursor, upscale, enlarged)
        self.assertIsNone(cursor.next_action)
        self.assertEqual(
            [row["operation"] for row in enlarged.steps],
            ["select-background", "compose", "prompt-edit", "upscale"],
        )
        self.assertEqual(enlarged.source_generation, {"prompt": "original prompt", "seed": 17})
        self.assertEqual(enlarged.steps[1]["source_id"], selected_snapshot.current.artifact_id)
        self.assertEqual(
            enlarged.steps[1]["additional_image_ids"], [self.snapshot.current.artifact_id]
        )
        self.assertNotIn("steps", enlarged.steps[1]["execution"])
        self.assertEqual(enlarged.steps[2]["parameters"]["denoise"], 0.28)

    def test_generated_background_captures_current_resources_and_no_source_conditioning(self):
        cursor = workflows.begin_workflow(
            self.store,
            self.snapshot.session_id,
            "background-replacement",
            owner=self.owner,
            overrides={"background_mode": "generate"},
        )
        cursor = workflows.skip_background_removal(self.store, cursor, owner=self.owner)
        step = self.capture(
            "generate-background",
            {
                "prompt": "studio",
                "negative": "blur",
                "base_model_name": "current-model",
                "loras": [["l", "0.5 - selected"]],
                "target_width": 64,
                "target_height": 64,
            },
            cursor=cursor,
            effective_plan={"selected": "current-model"},
        )
        data = workflows.worker_request(step, self.store, owner=self.owner)
        self.assertIsNone(data["input_image"])
        self.assertFalse(data["inpaint_toggle"])
        self.assertEqual(data["controlnet"], {})
        self.assertEqual(data["negative"], "blur")
        self.assertEqual(step.ticket.effective_plan, {"selected": "current-model"})

    def test_failed_canceled_replaced_and_stale_steps_never_advance_history(self):
        cursor = workflows.begin_workflow(
            self.store, self.snapshot.session_id, "portrait-enhancement", owner=self.owner
        )
        failed = self.capture("prompt-edit", cursor=cursor)
        self.store.fail(failed.ticket, owner=self.owner)
        after = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        with self.assertRaises(ValueError):
            workflows.advance_workflow(cursor, failed, after)
        canceled = self.capture("prompt-edit", cursor=cursor)
        request = workflows.worker_request(canceled, self.store, owner=self.owner)
        self.store.cancel(canceled.ticket, owner=self.owner)
        with self.assertRaises(CancelledError):
            workflows.finish_worker_output(request, self.image, store=self.store, owner=self.owner)
        stale = self.capture("prompt-edit", cursor=cursor)
        current = self.capture("prompt-edit", cursor=cursor)
        with self.assertRaises(CancelledError):
            workflows.worker_request(stale, self.store, owner=self.owner)
        self.store.cancel(current.ticket, owner=self.owner)
        self.assertEqual(self.store.snapshot(self.snapshot.session_id, owner=self.owner).steps, ())

    def test_two_owners_keep_mask_source_and_cancellation_isolated(self):
        second = self.store.start(self.save(self.image, "second"), owner=self.other_owner)
        second_mask = self.store.bind_mask(
            second.session_id, self.mask.path, second.current.artifact_id, owner=self.other_owner
        )
        other_handoff = capture_mask_handoff(
            self.store,
            second.session_id,
            second.revision,
            second.current.artifact_id,
            second_mask.artifact_id,
            owner=self.other_owner,
            choice="instance",
            instance_index=0,
        )
        first_step = self.capture("inpaint", mask=self.handoff())
        second_step = workflows.capture_step(
            self.store,
            second.session_id,
            second.revision,
            "inpaint",
            {},
            owner=self.other_owner,
            mask=other_handoff,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            requests = list(
                executor.map(
                    lambda pair: workflows.worker_request(pair[0], self.store, owner=pair[1]),
                    ((first_step, self.owner), (second_step, self.other_owner)),
                )
            )
        self.assertNotEqual(
            requests[0][workflows.EDITOR_KEY].source_id, requests[1][workflows.EDITOR_KEY].source_id
        )
        self.store.cancel(first_step.ticket, owner=self.owner)
        self.store.validate(second_step.ticket, owner=self.other_owner)
        with self.assertRaises(PermissionError):
            workflows.worker_request(second_step, self.store, owner=self.owner)

    def test_cancel_at_publication_keeps_source_and_step_count(self):
        step = self.capture("inpaint", mask=self.handoff())

        def canceled():
            raise CancelledError("fixture canceled before terminal publication")

        with self.assertRaises(CancelledError):
            self.store.complete(
                step.ticket,
                self.save(self.image, "unpublished"),
                owner=self.owner,
                check_cancelled=canceled,
            )
        after = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(after.current.artifact_id, self.snapshot.current.artifact_id)
        self.assertEqual(after.steps, ())

    def test_native_edit_control_uses_the_captured_family_and_preserves_denoise_intent(self):
        for family, name, moss, expected in (
            ("SDXL", "ordinary", "Disabled", "img2img"),
            ("QwenImage21", "qwen", "Disabled", "edit"),
            ("Flux2", "flux", "Disabled", "edit"),
            ("Flux", "flux-kontext", "Disabled", "edit"),
            ("MOSSImage", "moss", "In-Context", "edit"),
        ):
            step = self.capture(
                "prompt-edit",
                {"base_model_name": name, "moss_image_edit_format": moss, "denoise": 0.41},
                effective_plan={"family": family},
            )
            request = workflows.worker_request(step, self.store, owner=self.owner)
            self.assertEqual(request["controlnet"]["type"], expected)
            self.assertEqual(step.ticket.parameters["denoise"], 0.41)
            if expected == "img2img":
                self.assertEqual(request["controlnet"]["denoise"], 0.41)
        with self.assertRaisesRegex(ValueError, "family"):
            self.capture("prompt-edit", {"base_model_name": "unknown"})

    def test_final_effective_plan_supersedes_draft_ticket_before_preparation(self):
        step = self.capture("inpaint", {"inpaint_denoise": 0.33}, mask=self.handoff())
        request = workflows.worker_request(step, self.store, owner=self.owner)
        metadata = {
            "family": "SDXL",
            "checkpoint": {"identity": "captured"},
            "components": [{"key": "vae", "path": "captured-local-resource"}],
        }
        bound = workflows.bind_effective_plan(request, metadata, store=self.store, owner=self.owner)
        self.assertNotEqual(bound[workflows.EDITOR_KEY].operation_id, step.ticket.operation_id)
        self.assertEqual(bound[workflows.EDITOR_KEY].source_id, step.ticket.source_id)
        self.assertEqual(bound[workflows.EDITOR_KEY].mask_id, step.ticket.mask_id)
        self.assertEqual(bound[workflows.EDITOR_KEY].parameters, step.ticket.parameters)
        metadata["checkpoint"]["identity"] = "later-change"
        self.assertEqual(
            bound[workflows.EDITOR_KEY].effective_plan["checkpoint"]["identity"], "captured"
        )
        with self.assertRaises(CancelledError):
            self.store.validate(step.ticket, owner=self.owner)
        self.store.validate(bound[workflows.EDITOR_KEY], owner=self.owner)
        ordinary = {}
        self.assertIs(workflows.bind_effective_plan(ordinary, None), ordinary)

    def test_pure_execution_metadata_has_no_fictitious_diffusion_and_exact_output_frame(self):
        step = self.capture(
            "remove-background", {"rembg_native_model": "selected-birefnet", "rembg_force": True}
        )
        request = workflows.worker_request(step, self.store, owner=self.owner)
        self.assertEqual(request["controlnet"], {"type": "rembg", "force": True})
        self.assertEqual(request["input_image"].getchannel("A").getextrema(), (128, 128))
        execution = workflows.execution_metadata(
            step,
            self.image.size,
            actual={
                "steps": 20,
                "cfg": 4,
                "prompt": "inherited",
                "native": "RemoveBackground.execute",
            },
        )
        self.assertEqual(execution["kind"], "processing")
        self.assertEqual(execution["native"], "RemoveBackground.execute")
        self.assertNotIn("steps", execution)
        self.assertNotIn("prompt", execution)
        with self.assertRaisesRegex(ValueError, "target"):
            workflows.finish_pixels(step, self.store, Image.new("RGBA", (25, 16)), owner=self.owner)

    def test_precision_original_requires_explicit_library_working_copy(self):
        path = self.root / "precision.tiff"
        Image.fromarray(np.full((16, 24), 0.375, dtype=np.float32)).save(path)
        source = self.store.start(path, owner=self.owner)
        with self.assertRaisesRegex(ValueError, "working copy"):
            self.capture("prompt-edit", snapshot=source)

    def test_cursor_preset_intent_is_applied_and_explicit_step_values_win(self):
        cursor = workflows.begin_workflow(
            self.store,
            self.snapshot.session_id,
            "portrait-enhancement",
            owner=self.owner,
            overrides={"denoise": 0.23, "image_resize_mode": "pad"},
        )
        step = self.capture("prompt-edit", cursor=cursor)
        request = workflows.worker_request(step, self.store, owner=self.owner)
        self.assertEqual(request["controlnet"]["denoise"], 0.23)
        self.assertEqual(request["image_resize_mode"], "pad")
        self.assertEqual(step.ticket.parameters["workflow"]["preset"]["denoise"], 0.23)
        override = self.capture(
            "prompt-edit", {"denoise": 0.41, "image_resize_mode": "crop"}, cursor=cursor
        )
        request = workflows.worker_request(override, self.store, owner=self.owner)
        self.assertEqual(request["controlnet"]["denoise"], 0.41)
        self.assertEqual(request["image_resize_mode"], "crop")

    def test_compose_cannot_silently_lose_a_requested_flatten_policy(self):
        foreground = self.store.bind_image(
            self.snapshot.session_id, self.save(self.image, "owned-foreground"), owner=self.owner
        )
        with self.assertRaisesRegex(ValueError, "flattened owned image"):
            self.capture(
                "compose",
                {"foreground_id": foreground.artifact_id, "alpha_policy": "flatten-black"},
            )
        self.assertEqual(self.store.snapshot(self.snapshot.session_id, owner=self.owner).steps, ())

    def test_generated_background_uses_existing_exact_target_padding_and_restoration(self):
        from ruinedfooocus.features.generation.size_plan import KEY, capture_size_plan

        step = self.capture(
            "generate-background", {"prompt": "studio", "target_width": 61, "target_height": 43}
        )
        request = workflows.worker_request(step, self.store, owner=self.owner)
        captured = capture_size_plan(request, None, controls=request["controlnet"])
        self.assertEqual(captured[KEY].target, (61, 43))
        self.assertEqual(captured[KEY].canvas, (64, 48))
        decoded = torch.full((1, 48, 64, 3), 0.375, device="cpu")
        restored = captured[KEY].restore_canvas(decoded)
        result = captured[KEY].finish(restored)[0].numpy()
        output = workflows.finish_worker_output(
            captured, result, store=self.store, owner=self.owner
        )
        self.assertEqual(output.shape, (43, 61, 3))


if __name__ == "__main__":
    unittest.main()
