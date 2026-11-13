"""Real local artifacts and caller DTOs for bounded, nondestructive editor history."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ruinedfooocus.app import resource_lifecycle as lifecycle
from ruinedfooocus.features.gallery import editor_session as editor
from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.gallery.editing_workflow import (
    build_library_request,
    editor_source_record,
    open_editor,
)


class EditorSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.owner = ("fixture", "owner-a")
        self.other = ("fixture", "owner-b")
        self.store = editor.EditorSessionStore(self.root / "history", max_steps=3)
        self.original = self.image("original.png")
        self.original_bytes = self.original.read_bytes()
        self.provenance = dict(
            prompt="rendered instruction",
            raw_prompt="requested instruction",
            negative="literal negative",
            seed=123456789,
            denoise=0.37,
            loras=[["detail.safetensors", 0.75]],
            base_model_name="odd-model.gguf",
            base_model_hash="a" * 64,
            image_component_overrides={"vae": "own-vae.safetensors"},
            size_plan={"owner": "replay", "requested": [73, 121], "effective": [73, 121]},
        )
        self.snapshot = self.store.start(
            self.original, provenance=self.provenance, owner=self.owner
        )

    def image(self, name, *, size=(73, 121), mode="RGBA", color=None):
        path = self.root / name
        Image.new(mode, size, color or ((20, 40, 60, 127) if mode == "RGBA" else 127)).save(path)
        return path

    def context(self, snapshot=None, **values):
        snapshot = snapshot or self.snapshot
        return dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
            **values,
        )

    def capture(self, snapshot=None, *, operation="process-image", **values):
        return self.store.capture(
            **self.context(snapshot),
            operation=operation,
            parameters={"amount": 0.5},
            owner=self.owner,
            **values,
        )

    def complete(self, ticket, name="result.png", **kwargs):
        result = self.image(name)
        return self.store.complete(ticket, result, owner=self.owner, **kwargs)

    def test_real_operation_preserves_original_alpha_odd_size_and_inherited_provenance(self):
        parameters = {"prompt": "own requested prompt", "denoise": 0.25}
        ticket = self.store.capture(
            **self.context(),
            operation="color-adjust",
            parameters=parameters,
            effective_plan={"family": "test", "components": ["vae"]},
            owner=self.owner,
        )
        parameters["prompt"] = "later mutation"
        self.provenance["loras"].clear()
        result = self.root / "adjusted.png"
        with Image.open(self.original) as source:
            changed = source.copy()
            changed.putpixel((72, 120), (200, 100, 50, 127))
            changed.save(result)
        self.store.start_operation(ticket, owner=self.owner)
        snapshot = self.store.complete(
            ticket,
            result,
            execution={"runtime": "Pillow", "operation": "pixel adjustment"},
            owner=self.owner,
        )
        self.assertEqual(snapshot.original.artifact_id, self.snapshot.original.artifact_id)
        self.assertEqual(snapshot.current.frame.size, (73, 121))
        self.assertEqual(snapshot.current.frame.alpha, "straight")
        self.assertEqual(snapshot.source_generation["prompt"], "rendered instruction")
        self.assertEqual(snapshot.source_generation["loras"], [["detail.safetensors", 0.75]])
        self.assertEqual(snapshot.steps[0]["parameters"]["prompt"], "own requested prompt")
        self.assertNotIn("sampling", snapshot.steps[0]["execution"])
        self.assertEqual(snapshot.steps[0]["source_id"], self.snapshot.current.artifact_id)
        self.assertEqual(snapshot.steps[0]["result_id"], snapshot.current.artifact_id)
        self.assertEqual(self.original.read_bytes(), self.original_bytes)
        public = snapshot.to_dict()
        self.assertNotIn("path", public["current"])
        self.assertNotIn(str(self.root), json.dumps(public))
        self.assertEqual(snapshot.original.sha256, hashlib.sha256(self.original_bytes).hexdigest())

    def test_saved_reopen_replays_without_pixels_or_regeneration(self):
        done = self.complete(self.capture())
        second_store = editor.EditorSessionStore(self.store.root, max_steps=3)
        reopened = second_store.reopen(done.session_id, owner=self.owner)
        self.assertEqual(done.to_dict(), reopened.to_dict())
        self.assertFalse(
            any(
                isinstance(value, (Image.Image, bytes, np.ndarray)) for value in vars(done).values()
            )
        )
        self.assertEqual(
            second_store.artifact(done.session_id, done.current.artifact_id, owner=self.owner).path,
            str(self.root / "result.png"),
        )

    def test_public_dto_redacts_nested_plan_paths_without_rewriting_literal_prompts(self):
        local_path = str(self.original)
        provenance = dict(
            prompt=f"literal instruction mentioning {local_path}", model_recipe_path=local_path
        )
        session = self.store.start(self.original, provenance=provenance, owner=self.owner)
        plan = dict(
            checkpoint={"path": local_path, "sha256": "b" * 64, "signature": [local_path, 1, 2, 3]},
            components=[{"path": local_path, "name": "own-vae.safetensors"}],
        )
        ticket = self.store.capture(
            **self.context(session), operation="edit", effective_plan=plan, owner=self.owner
        )
        done = self.complete(ticket)
        public = done.to_dict()
        self.assertEqual(public["source_generation"]["prompt"], provenance["prompt"])
        self.assertNotIn("model_recipe_path", public["source_generation"])
        checkpoint = public["steps"][0]["effective_plan"]["checkpoint"]
        self.assertNotIn("path", checkpoint)
        self.assertEqual(checkpoint["signature"], [1, 2, 3])
        self.assertEqual(checkpoint["sha256"], "b" * 64)
        self.assertEqual(done.steps[0]["effective_plan"], plan)
        self.assertEqual(done.source_generation, provenance)

    def test_undo_redo_and_editing_an_old_head_branches_without_deleting_results(self):
        first = self.complete(self.capture(), "first.png")
        second = self.complete(self.capture(first), "second.png")
        earlier = self.store.undo(second.session_id, second.revision, owner=self.owner)
        self.assertEqual(earlier.current.artifact_id, first.current.artifact_id)
        redone = self.store.redo(earlier.session_id, earlier.revision, owner=self.owner)
        self.assertEqual(redone.current.artifact_id, second.current.artifact_id)
        earlier = self.store.undo(redone.session_id, redone.revision, owner=self.owner)
        branch = self.complete(self.capture(earlier), "branch.png")
        self.assertEqual(branch.steps[-1]["parent_id"], first.current.artifact_id)
        with self.assertRaisesRegex(ValueError, "redo"):
            self.store.redo(branch.session_id, branch.revision, owner=self.owner)
        self.assertTrue((self.root / "second.png").exists())
        self.assertEqual(self.original.read_bytes(), self.original_bytes)

    def test_foreign_owner_session_and_typed_handles_fail_before_preparation(self):
        second = self.store.start(self.original, owner=self.other)
        preparation = Mock(side_effect=AssertionError("preparation was reached"))
        with patch.object(editor.preparation_service, "submit", preparation):
            with self.assertRaises(PermissionError):
                self.store.snapshot(self.snapshot.session_id, owner=self.other)
            with self.assertRaises(ValueError):
                self.store.artifact(
                    second.session_id, self.snapshot.current.artifact_id, owner=self.other
                )
            with self.assertRaises(PermissionError):
                self.store.validate(self.capture(), owner=self.other)
            with self.assertRaises(ValueError):
                self.store.capture(
                    **self.context(),
                    operation="edit",
                    mask_id=self.snapshot.current.artifact_id,
                    owner=self.owner,
                )
        preparation.assert_not_called()

    def test_source_replacement_invalidates_capture_and_late_completion(self):
        ticket = self.capture()
        self.image("original.png", color=(1, 2, 3, 4))
        for action in (lambda: self.capture(), lambda: self.complete(ticket)):
            with self.assertRaisesRegex(ValueError, "changed or was removed"):
                action()
        manifest = json.loads(self.store._path(self.snapshot.session_id).read_text())
        self.assertEqual(manifest["steps"], [])

    def test_replaced_ticket_and_changed_revision_publish_no_success(self):
        old = self.capture()
        current = self.capture()
        with self.assertRaises(CancelledError):
            self.complete(old, "late.png")
        done = self.complete(current)
        with self.assertRaises(CancelledError):
            self.complete(current, "duplicate.png")
        self.assertEqual(len(done.steps), 1)
        with self.assertRaises(CancelledError):
            self.store.capture(**self.context(), operation="edit", owner=self.owner)

    def test_cancel_and_failure_append_no_step_and_keep_original(self):
        for terminal in (self.store.cancel, self.store.fail):
            ticket = self.capture()
            self.assertTrue(terminal(ticket, owner=self.owner))
            self.assertFalse(terminal(ticket, owner=self.owner))
            with self.assertRaises(CancelledError):
                self.complete(ticket)
            self.assertEqual(self.store.snapshot(ticket.session_id, owner=self.owner).steps, ())
        self.assertEqual(self.original.read_bytes(), self.original_bytes)

    def test_existing_job_cancellation_fences_history_commit(self):
        ticket = self.capture()
        job = lifecycle.registry.create(owner=self.owner)
        self.addCleanup(lifecycle.registry.finish, job)
        with lifecycle.job_scope(job):
            lifecycle.registry.cancel(job.task_id, owner=self.owner)
            with self.assertRaises(CancelledError):
                self.complete(ticket)
        self.assertEqual(self.store.snapshot(ticket.session_id, owner=self.owner).steps, ())

    def test_two_completions_race_to_exactly_one_success(self):
        ticket = self.capture()
        paths = [self.image("race-a.png"), self.image("race-b.png")]
        barrier = threading.Barrier(2)

        def commit(path):
            barrier.wait(timeout=5)
            try:
                return self.store.complete(ticket, path, owner=self.owner)
            except CancelledError:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(commit, paths))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(len(self.store.snapshot(ticket.session_id, owner=self.owner).steps), 1)

    def test_cancel_during_result_inspection_prevents_a_successful_history_step(self):
        ticket = self.capture()
        path = self.image("cancel-race.png")
        inspecting, release = threading.Event(), threading.Event()
        original_ref = self.store._ref

        def delayed_ref(*args, **kwargs):
            inspecting.set()
            if not release.wait(timeout=5):
                raise AssertionError("Cancellation race fixture timed out")
            return original_ref(*args, **kwargs)

        with (
            patch.object(self.store, "_ref", side_effect=delayed_ref),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            future = executor.submit(self.store.complete, ticket, path, owner=self.owner)
            self.assertTrue(inspecting.wait(timeout=5))
            self.assertTrue(self.store.cancel(ticket, owner=self.owner))
            release.set()
            with self.assertRaises(CancelledError):
                future.result(timeout=5)
        self.assertEqual(self.store.snapshot(ticket.session_id, owner=self.owner).steps, ())

    def test_atomic_write_failure_does_not_publish_a_completed_step(self):
        ticket = self.capture()
        with patch.object(editor, "save_json", side_effect=OSError("owned write failure")):
            with self.assertRaises(OSError):
                self.complete(ticket)
        self.assertEqual(self.store.snapshot(ticket.session_id, owner=self.owner).steps, ())
        self.assertEqual(len(self.complete(ticket).steps), 1)

    def test_masks_bind_one_exact_oriented_source_and_reject_cutouts_or_geometry(self):
        small = self.image("wrong-mask.png", size=(72, 121), mode="L")
        for path in (small, self.original):
            with self.assertRaisesRegex(ValueError, "single-channel.*source frame"):
                self.store.bind_mask(
                    self.snapshot.session_id,
                    path,
                    self.snapshot.current.artifact_id,
                    owner=self.owner,
                )
        mask_path = self.image("mask.png", mode="L")
        mask = self.store.bind_mask(
            self.snapshot.session_id,
            mask_path,
            self.snapshot.current.artifact_id,
            inverted=True,
            owner=self.owner,
        )
        self.assertEqual(mask.source_frame_id, self.snapshot.current.frame_id)
        self.assertEqual((mask.purpose, mask.channel, mask.inverted), ("selection-mask", "L", True))
        ticket = self.capture(mask_id=mask.artifact_id)
        self.assertEqual(ticket.mask, mask)
        other_image = self.store.bind_image(
            self.snapshot.session_id, self.image("same-frame-other-image.png"), owner=self.owner
        )
        wrong_mask = self.store.bind_mask(
            self.snapshot.session_id, mask_path, other_image.artifact_id, owner=self.owner
        )
        with self.assertRaisesRegex(ValueError, "different image frame"):
            self.capture(mask_id=wrong_mask.artifact_id)

    def test_composition_mask_binds_overlay_and_additional_handles_stay_owned(self):
        overlay = self.store.bind_image(
            self.snapshot.session_id, self.image("overlay.png", size=(17, 31)), owner=self.owner
        )
        mask = self.store.bind_mask(
            self.snapshot.session_id,
            self.image("overlay-mask.png", size=(17, 31), mode="L"),
            overlay.artifact_id,
            owner=self.owner,
        )
        ticket = self.capture(
            operation="image-composite",
            additional_image_ids=[overlay.artifact_id],
            mask_id=mask.artifact_id,
        )
        self.assertEqual(ticket.additional_images, (overlay,))
        with self.assertRaisesRegex(ValueError, "different image frame"):
            self.capture(
                operation="inpaint",
                additional_image_ids=[overlay.artifact_id],
                mask_id=mask.artifact_id,
            )

    def test_geometry_alpha_and_precision_conversion_are_deliberate(self):
        ticket = self.capture()
        enlarged = self.image("larger.png", size=(146, 242))
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.store.complete(ticket, enlarged, owner=self.owner)
        done = self.store.complete(
            ticket,
            enlarged,
            execution={
                "geometry": {
                    "policy": "explicit 2x resize",
                    "source": [73, 121],
                    "result": [146, 242],
                }
            },
            owner=self.owner,
        )
        ticket = self.capture(done)
        opaque = self.image("opaque.png", size=(146, 242), mode="RGB", color=(3, 4, 5))
        with self.assertRaisesRegex(ValueError, "alpha policy"):
            self.store.complete(ticket, opaque, owner=self.owner)
        done = self.store.complete(
            ticket, opaque, execution={"alpha_policy": "explicit opaque flatten"}, owner=self.owner
        )
        self.assertEqual(done.current.frame.alpha, "none")

    def test_native_uint16_source_identity_is_not_converted_or_overwritten(self):
        path = self.root / "precision.png"
        values = np.arange(73 * 121, dtype=np.uint16).reshape(121, 73)
        Image.fromarray(values).save(path)
        original = path.read_bytes()
        session = self.store.start(path, owner=self.owner)
        self.assertEqual(session.original.frame.dtype, "uint16")
        self.assertEqual(session.original.frame.size, (73, 121))
        self.assertEqual(editor.describe_frame(path), session.original.frame)
        with self.assertRaisesRegex(ValueError, "8-bit single-channel"):
            self.store.bind_mask(
                session.session_id, path, session.current.artifact_id, owner=self.owner
            )
        ticket = self.capture(session)
        with self.assertRaisesRegex(ValueError, "precision"):
            self.complete(ticket, "uint8.png")
        self.assertEqual(path.read_bytes(), original)

    def test_expiry_prunes_only_private_manifests_and_keeps_user_artifacts(self):
        now = [100.0]
        store = editor.EditorSessionStore(
            self.root / "expires", max_sessions=1, ttl_seconds=10, clock=lambda: now[0]
        )
        initial = store.start(self.original, owner=self.owner)
        now[0] = 111.0
        with self.assertRaisesRegex(ValueError, "expired"):
            store.snapshot(initial.session_id, owner=self.owner)
        current = store.start(self.original, owner=self.owner)
        self.assertNotEqual(current.session_id, initial.session_id)
        self.assertFalse(store._path(initial.session_id).exists())
        self.assertEqual(self.original.read_bytes(), self.original_bytes)

    def test_bounded_history_retains_original_and_never_deletes_outputs(self):
        store = editor.EditorSessionStore(self.root / "bounded", max_steps=2)
        snapshot = store.start(self.original, owner=self.owner)
        for index in range(24):
            ticket = store.capture(**self.context(snapshot), operation="filter", owner=self.owner)
            path = self.image(f"bounded-{index}.png")
            snapshot = store.complete(ticket, path, owner=self.owner)
        value = json.loads(store._path(snapshot.session_id).read_text())
        self.assertEqual(len(snapshot.steps), 2)
        self.assertLessEqual(len(value["artifacts"]), store.max_artifacts)
        self.assertIn(snapshot.original.artifact_id, value["artifacts"])
        self.assertEqual(len(list(self.root.glob("bounded-*.png"))), 24)
        self.assertLess(store._path(snapshot.session_id).stat().st_size, editor.MAX_MANIFEST_BYTES)

    def test_unknown_schema_is_not_migrated_or_overwritten(self):
        path = self.store._path(self.snapshot.session_id)
        value = json.loads(path.read_text())
        value["schema_version"] = 999
        path.write_text(json.dumps(value), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "matching application version"):
            self.store.reopen(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(before, path.read_bytes())

    def test_strict_json_context_and_immutable_ticket_reject_forgery(self):
        data = dict(
            prompt="requested",
            raw_prompt="rendered",
            negative="negative",
            seed=42,
            loras=[["detail", 0.3]],
            image_number=1,
            input_image=object(),
        )
        captured = editor.capture_request(
            data, self.context(), "edit", owner=self.owner, store=self.store
        )
        ticket = captured[editor.KEY]
        self.assertNotIn("input_image", ticket.parameters)
        self.assertEqual(
            editor.validate_request(captured, owner=self.owner, store=self.store), ticket
        )
        for context in (
            self.context() | {"path": "outside.png"},
            self.context() | {"owner": self.other},
        ):
            with self.assertRaises(ValueError):
                editor.capture_request(data, context, "edit", owner=self.owner, store=self.store)
        with self.assertRaises(ValueError):
            editor.capture_request(
                data | {"image_number": 2},
                self.context(),
                "edit",
                owner=self.owner,
                store=self.store,
            )
        with self.assertRaises(CancelledError):
            self.store.validate(replace(ticket, operation="forged"), owner=self.owner)
        editor.validate_request(captured, owner=self.owner, store=self.store, start=True)
        with self.assertRaises(CancelledError):
            editor.validate_request(captured, owner=self.owner, store=self.store, start=True)
        editor.cancel_request(captured, owner=self.owner, store=self.store)
        self.assertEqual(self.store.snapshot(ticket.session_id, owner=self.owner).steps, ())

    def test_final_effective_plan_retickets_atomically_before_start(self):
        draft = self.capture()
        plan = {"family": "fixture", "operation": "process-image", "size": {"owner": "manual"}}
        data = editor.bind_effective_plan(
            {editor.KEY: draft}, plan, owner=self.owner, store=self.store
        )
        final = data[editor.KEY]
        self.assertNotEqual(final.operation_id, draft.operation_id)
        self.assertEqual(final.parameters, draft.parameters)
        self.assertEqual(final.source, draft.source)
        self.assertEqual(final.effective_plan, plan)
        self.assertIs(
            editor.bind_effective_plan(data, plan, owner=self.owner, store=self.store)[editor.KEY],
            final,
        )
        with self.assertRaises(CancelledError):
            self.store.validate(draft, owner=self.owner)
        self.store.start_operation(final, owner=self.owner)
        with self.assertRaises(CancelledError):
            editor.bind_effective_plan(
                data, {"family": "later"}, owner=self.owner, store=self.store
            )
        done = self.complete(final)
        self.assertEqual(done.steps[-1]["effective_plan"], plan)

    def test_library_caller_resolves_current_owned_source_and_preserves_request(self):
        opened = open_editor(
            source_record(self.original), self.provenance, owner=self.owner, store=self.store
        )
        context = self.context(opened)
        controls = dict(
            loras=[["detail", 0.5]],
            image_component_overrides={"vae": "explicit"},
            base_model_name="model",
            cn_strength=0.37,
            image_resize_mode="pad",
            moss_image_slg=True,
        )
        request = build_library_request(
            controls,
            source_record(self.original),
            prompt="own edit",
            negative="own negative",
            checkpoint="model",
            action="edit",
            resets={},
            features={"native_edit": False},
            presets={"Img2Img": {"type": "img2img"}},
            custom="Custom...",
            editor_context=context,
            editor_store=self.store,
            owner=self.owner,
        )
        self.assertEqual(request["input_image"].size, (73, 121))
        self.assertEqual(request["input_image"].getpixel((0, 0)), (20, 40, 60, 127))
        self.assertEqual(request[editor.KEY].parameters["loras"], controls["loras"])
        self.assertEqual(request[editor.KEY].parameters["cn_strength"], 0.37)
        self.assertEqual(request[editor.KEY].parameters["image_resize_mode"], "pad")
        self.assertTrue(request[editor.KEY].parameters["moss_image_slg"])
        with self.assertRaises(PermissionError):
            editor_source_record(context, owner=self.other, store=self.store)

    def test_owned_library_inpaint_requires_mask_id_and_uses_its_authoritative_pixels(self):
        arguments = dict(
            controls={},
            record=source_record(self.original),
            prompt="owned mask edit",
            negative="",
            checkpoint="model",
            action="inpaint",
            resets={},
            features={"inpaint": True},
            presets={},
            custom="Custom...",
            editor_context=self.context(),
            editor_store=self.store,
            owner=self.owner,
        )
        with patch(
            "ruinedfooocus.features.gallery.editing_workflow.working_source",
            side_effect=AssertionError("source preparation was reached"),
        ):
            with self.assertRaisesRegex(ValueError, "owned selection mask"):
                build_library_request(**arguments)
        mask_path = self.root / "owned-brush.png"
        selection = Image.new("L", (73, 121), 0)
        selection.putpixel((2, 3), 255)
        selection.save(mask_path)
        mask = self.store.bind_mask(
            self.snapshot.session_id, mask_path, self.snapshot.current.artifact_id, owner=self.owner
        )
        arguments["editor_context"] = self.context(mask_id=mask.artifact_id)
        arguments["mask"] = {"layers": [Image.new("RGBA", (1, 1), "white")]}
        request = build_library_request(**arguments)
        self.assertEqual(
            request["inpaint_view"]["layers"][0].getchannel("A").getbbox(), (2, 3, 3, 4)
        )
        self.assertEqual(request[editor.KEY].mask_id, mask.artifact_id)

    def test_owned_composition_ignores_unbound_pixels_and_clears_execution_sampling(self):
        from ruinedfooocus.features.image_processing import image_composition as composition

        overlay = self.store.bind_image(
            self.snapshot.session_id,
            self.image("composition-overlay.png", size=(17, 31)),
            owner=self.owner,
        )
        ticket = self.capture(
            operation="image-composite", additional_image_ids=[overlay.artifact_id]
        )
        data = {
            editor.KEY: ticket,
            "composition_images": [Image.new("RGBA", (1, 1), "red")],
            "prompt": "unrelated diffusion",
            "seed": 333,
            "_image_workflow_step": "owned workflow marker",
        }
        with (
            patch.object(editor, "default_store", return_value=self.store),
            patch.object(lifecycle, "current_owner", return_value=self.owner),
        ):
            captured = composition.capture(data, "image-composite")
        self.assertIs(captured[editor.KEY], ticket)
        self.assertEqual(captured["_image_workflow_step"], "owned workflow marker")
        self.assertEqual(
            [image.size for image in composition.captured(captured).images], [(73, 121), (17, 31)]
        )
        self.assertFalse(captured["_sampling_captured"])
        self.assertEqual(captured["prompt"], "")
        self.assertEqual(captured["loras"], [])
        self.assertEqual(ticket.parameters, {"amount": 0.5})


if __name__ == "__main__":
    unittest.main()
