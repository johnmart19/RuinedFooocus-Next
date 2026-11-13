"""Native SAM writer, owned terminal and real Editor selection continuations."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.image_editing.inpaint_utils import source_and_mask
from ruinedfooocus.features.image_editing.mask_handoffs import (
    capture_mask_handoff,
    inpaint_inputs,
)
from ruinedfooocus.features.image_processing.native_image_outputs import (
    latest_generated_result,
    result_presentation,
)
import test_guided_result_ui as guided_fixture
import test_image_plan_callers as publication_fixture
import test_sam_prompt_utilities as sam_fixture


class SamEditorWorkerTests(unittest.TestCase):
    def setUp(self):
        self.publication = publication_fixture.OwnedEditorPublicationTests()
        self.publication.setUp()
        self.addCleanup(self.publication.doCleanups)
        self.sam = sam_fixture.SamPromptTests()
        self.sam.setUp()
        self.addCleanup(self.sam.doCleanups)
        self.render, self.forbidden = self.sam.worker()
        self.sam.image.save(self.publication.source)
        self.original = self.publication.source.read_bytes()
        self.store, self.owner = self.publication.store, self.publication.owner
        self.publication.snapshot = self.store.start(
            self.publication.source, provenance={"prompt": "original recipe"}, owner=self.owner
        )
        self.snapshot = self.publication.snapshot

    def run_native(self, *, output="mask", threshold=0.5, individual=True):
        data = self.sam.request(
            segment_text="cat:2",
            segment_individual_masks=individual,
            segment_refine_iterations=0,
            segment_output=output,
            vision_threshold=threshold,
        )
        data.update(
            task_id=self.publication.item.task_id,
            task_type="tool_call",
            silent=True,
            index=(0, 1),
        )
        result = self.render(data)
        captured = self.publication.request("segment")
        data["_editor_operation"] = captured["_editor_operation"]
        self.publication.scope["validate_editor_data"](data, start=True)
        self.publication.scope["stage_editor_result"](data, result)
        self.forbidden.assert_not_called()
        return data, result

    def publish(self, result):
        self.publication.scope["add_result"](self.publication.item.task_id, "results", result)
        return self.store.snapshot(self.snapshot.session_id, owner=self.owner)

    def test_actual_native_multiple_outputs_commit_receipt_and_keep_photo_until_selection(self):
        _, result = self.run_native()
        self.assertEqual(len(result), 2)
        self.assertEqual(
            self.store.snapshot(self.snapshot.session_id, owner=self.owner).revision,
            self.snapshot.revision,
        )
        completed = self.publish(result)
        self.assertEqual(completed.current, self.snapshot.current)
        self.assertEqual(completed.revision, self.snapshot.revision + 1)
        self.assertEqual(completed.source_generation, {"prompt": "original recipe"})
        self.assertEqual(len(completed.derived_results["result_ids"]), 2)
        self.assertEqual(completed.derived_results["source_id"], self.snapshot.current.artifact_id)
        self.assertEqual(completed.derived_results["output"], "mask")
        self.assertEqual(self.publication.source.read_bytes(), self.original)
        records = result_presentation(result)[2]
        self.assertTrue(all(row["output_role"] == "mask" for row in records))
        self.assertIsNone(latest_generated_result(records))
        for index, row in enumerate(records):
            metadata = json.loads(row["metadata"])
            self.assertEqual(metadata["native_output_group"]["index"], index)
            source = metadata["native_image_output"]["native_vision_source"]
            self.assertEqual(source["instance_index"], index)
            self.assertEqual(source["mask_role"], "inpaint-selection")
        selected, mask = self.store.use_derived_selection(
            completed.session_id, completed.revision, [1], owner=self.owner
        )
        handoff = capture_mask_handoff(
            self.store,
            selected.session_id,
            selected.revision,
            selected.current.artifact_id,
            mask.artifact_id,
            owner=self.owner,
            choice="instance",
            instance_count=2,
            instance_index=1,
        )
        request = inpaint_inputs(self.store, handoff, owner=self.owner)
        photo, scalar = source_and_mask(
            request["main_view"], request["inpaint_view"], preserve_alpha=True
        )
        self.assertEqual(photo.getchannel("A").getextrema(), (128, 128))
        with Image.open(result[1]) as native:
            np.testing.assert_array_equal(np.asarray(scalar), np.asarray(native.convert("L")))
        self.assertEqual(selected.current, self.snapshot.current)
        with self.assertRaises(CancelledError):
            self.store.use_derived_selection(
                completed.session_id, completed.revision, [0], owner=self.owner
            )

    def test_actual_native_zero_match_completes_without_erasing_photo(self):
        _, result = self.run_native(threshold=1)
        self.assertEqual(len(result), 0)
        completed = self.publish(result)
        self.assertEqual(completed.current, self.snapshot.current)
        self.assertEqual(completed.revision, self.snapshot.revision + 1)
        self.assertEqual(completed.derived_results["result_ids"], [])
        self.assertEqual(
            self.publication.scope["editor_result"](self.publication.item.task_id)["status"],
            "completed",
        )
        self.assertEqual(self.publication.source.read_bytes(), self.original)

    def test_actual_native_union_mask_retains_its_l_mode_and_original_photo(self):
        _, result = self.run_native(individual=False)
        self.assertEqual(len(result), 1)
        completed = self.publish(result)
        selected, mask = self.store.use_derived_selection(
            completed.session_id, completed.revision, [0], owner=self.owner
        )
        with Image.open(result[0]) as native, Image.open(mask.path) as selection:
            self.assertEqual(selection.mode, "L")
            np.testing.assert_array_equal(np.asarray(selection), np.asarray(native.convert("L")))
        self.assertEqual(selected.current, self.snapshot.current)

    def test_cutouts_keep_photo_and_cannot_be_reinterpreted_as_selection_alpha(self):
        _, result = self.run_native(output="cutout")
        completed = self.publish(result)
        self.assertEqual(completed.current, self.snapshot.current)
        with self.assertRaisesRegex(ValueError, "grayscale mask"):
            self.store.use_derived_selection(
                completed.session_id, completed.revision, [0], owner=self.owner
            )

    def test_cancelled_owned_job_cannot_commit_a_staged_native_bundle(self):
        _, result = self.run_native()
        self.publication.registry.cancel(self.publication.item.task_id, owner=self.owner)
        with self.assertRaises(CancelledError):
            self.publication.scope["publish_editor_result"](self.publication.item)
        current = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(current.revision, self.snapshot.revision)
        self.assertIsNone(current.derived_results)

    def test_changed_generated_artifact_cannot_commit_a_staged_bundle(self):
        _, result = self.run_native()
        with Path(result[0]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed before publication"):
            self.publish(result)
        current = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(current.revision, self.snapshot.revision)
        self.assertIsNone(current.derived_results)

    def test_terminal_uses_frozen_results_without_decoding_or_hashing_under_registry_lock(self):
        _, result = self.run_native()
        with (
            patch.object(self.store, "_ref", side_effect=AssertionError("terminal decoded image")),
            patch(
                "ruinedfooocus.features.gallery.editor_session.hashlib.file_digest",
                side_effect=AssertionError("terminal rehashed image"),
            ),
        ):
            completed = self.publish(result)
        self.assertEqual(completed.current, self.snapshot.current)
        self.assertEqual(len(completed.derived_results["result_ids"]), 2)

    def test_replaced_ticket_cannot_publish_staged_native_results(self):
        _, result = self.run_native()
        self.publication.request("segment")
        with self.assertRaises(CancelledError):
            self.publish(result)
        current = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(current.revision, self.snapshot.revision)
        self.assertIsNone(current.derived_results)

    def test_selection_rejects_foreign_changed_empty_and_nonexistent_instances(self):
        _, result = self.run_native()
        completed = self.publish(result)
        with self.assertRaises(PermissionError):
            self.store.use_derived_selection(
                completed.session_id, completed.revision, [0], owner=("ui", "other")
            )
        for indices in ([], [2], [True], [0, 0]):
            with self.assertRaises(ValueError):
                self.store.use_derived_selection(
                    completed.session_id, completed.revision, indices, owner=self.owner
                )
        Image.new("L", self.sam.image.size, 255).save(result[0])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.use_derived_selection(
                completed.session_id, completed.revision, [0], owner=self.owner
            )

    def test_selection_cannot_apply_while_another_operation_is_pending(self):
        _, result = self.run_native()
        completed = self.publish(result)
        self.store.capture(
            completed.session_id,
            completed.revision,
            "inpaint",
            source_id=completed.current.artifact_id,
            owner=self.owner,
        )
        with self.assertRaisesRegex(ValueError, "for this image first"):
            self.store.use_derived_selection(
                completed.session_id, completed.revision, [0], owner=self.owner
            )

    def test_native_results_enforce_frame_precision_empty_and_bounded_count(self):
        ticket = self.publication.request("segment")["_editor_operation"]
        wrong = self.publication.folder / "wrong-frame.png"
        Image.new("L", (41, 20), 255).save(wrong)
        with self.assertRaisesRegex(ValueError, "exact source image frame"):
            self.store.capture_derived_results(ticket, [wrong], owner=self.owner)
        with self.assertRaisesRegex(ValueError, "artifact limit"):
            self.store.capture_derived_results(ticket, [wrong] * 65, owner=self.owner)
        for mode, dtype, message in (("I;16", np.uint16, "8-bit"), ("L", np.uint8, "empty")):
            path = self.publication.folder / (mode.replace(";", "-") + ".png")
            Image.fromarray(np.zeros((20, 40), dtype=dtype), mode).save(path)
            refs = self.store.capture_derived_results(ticket, [path], owner=self.owner)
            completed = self.store.complete_derived_results(
                ticket, refs, output="mask", owner=self.owner
            )
            with self.assertRaisesRegex(ValueError, message):
                self.store.use_derived_selection(
                    completed.session_id, completed.revision, [0], owner=self.owner
                )
            ticket = self.store.capture(
                completed.session_id,
                completed.revision,
                "segment",
                source_id=completed.current.artifact_id,
                owner=self.owner,
            )


class SamEditorUiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = guided_fixture.GuidedResultUiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.case, self.workspace, self.store = (
            self.fixture.case,
            self.fixture.workspace,
            self.fixture.store,
        )
        self.helper = self.workspace.guided_results

    def result(self, session, *, output="mask", count=2):
        source = self.store.start(self.fixture.original, owner=session.owner)
        ticket = self.store.capture(
            source.session_id,
            source.revision,
            "segment",
            source_id=source.current.artifact_id,
            owner=session.owner,
        )
        paths = []
        for index in range(count):
            path = self.case.root / (session.request.session_hash + f"-object-{index}.png")
            pixels = np.zeros((13, 17), dtype=np.uint8)
            pixels[:, index * 8 : index * 8 + 8] = 64 + index * 64
            Image.fromarray(pixels, "L").save(path)
            paths.append(path)
        refs = self.store.capture_derived_results(ticket, paths, owner=session.owner)
        completed = self.store.complete_derived_results(
            ticket, refs, output=output, owner=session.owner
        )
        session.state[self.workspace.context._id] = self.workspace.public_context(completed)
        self.case.fence.begin(session.owner, self.workspace.channel, 1)
        return completed, paths

    def test_registered_cta_selects_one_instance_or_a_soft_union_and_builds_real_inpaint(self):
        for chosen in (["1"], ["0", "1"]):
            with self.subTest(chosen=chosen):
                session = self.case.session("sam-objects-" + str(len(chosen)))
                completed, paths = self.result(session)
                self.case.call(session, self.fixture.projection)
                self.assertEqual(self.case.value(session, self.helper.objects), ["0", "1"])
                self.case.call(session, self.fixture.event, {self.helper.objects: chosen})
                context = self.case.value(session, self.workspace.context)
                self.assertEqual(context["source_id"], completed.current.artifact_id)
                self.assertEqual(self.case.value(session, self.workspace.action), "inpaint")
                values = self.case.request_values(session, "inpaint")
                values.update(
                    _editor_mask_choice=self.case.value(session, self.workspace.mask_choice),
                    _editor_mask_index=self.case.value(session, self.workspace.mask_index),
                    _editor_mask_count=self.case.value(session, self.workspace.mask_count),
                    resolution_auto=True,
                )
                request = self.workspace.build_request(values, request=session.request)
                photo, selection = source_and_mask(
                    request["main_view"], request["inpaint_view"], preserve_alpha=True
                )
                self.assertEqual(photo.getchannel("A").getextrema(), (117, 117))
                self.assertEqual((request["custom_width"], request["custom_height"]), (17, 13))
                self.assertEqual(self.case.value(session, self.case.controls["custom_width"]), 512)
                self.assertEqual(self.case.value(session, self.case.controls["custom_height"]), 768)
                expected = np.zeros((13, 17), dtype=np.uint8)
                for index in chosen:
                    with Image.open(paths[int(index)]) as mask:
                        expected = np.maximum(expected, np.asarray(mask))
                np.testing.assert_array_equal(np.asarray(selection), expected)
                self.assertEqual(self.fixture.original.read_bytes(), self.fixture.original_bytes)
                self.case.run_request.assert_not_called()

    def test_zero_matches_and_cutouts_do_not_offer_selection_cta(self):
        for output, count in (("mask", 0), ("cutout", 2)):
            session = self.case.session(f"sam-hidden-{output}-{count}")
            completed, _ = self.result(session, output=output, count=count)
            updates = self.case.call(session, self.fixture.projection)
            self.assertFalse(updates[self.helper.button]["visible"])
            self.assertFalse(updates[self.helper.objects]["visible"])
            self.assertEqual(
                self.case.value(session, self.workspace.context)["source_id"],
                completed.current.artifact_id,
            )

    def test_mask_preparation_releases_shared_locks_and_cancelled_or_stale_union_never_binds(self):
        for invalidate in ("cancel", "supersede"):
            with self.subTest(invalidate=invalidate):
                session = self.case.session("sam-preparation-" + invalidate)
                completed, paths = self.result(session)
                self.case.call(session, self.fixture.projection)
                entered, release = threading.Event(), threading.Event()
                original_open = Image.open

                def blocked_open(path, *args, **kwargs):
                    if Path(path) == paths[0]:
                        entered.set()
                        if not release.wait(5):
                            raise AssertionError("Mask preparation barrier timed out")
                    return original_open(path, *args, **kwargs)

                def invalidate_request():
                    other = self.case.fence.begin(
                        ("ui", "unrelated-" + invalidate), "another-channel", 1
                    )
                    self.assertTrue(self.case.fence.current(other))
                    self.case.fence.cancel(other.owner, other.channel)
                    # A snapshot also needs the store lock, independently of the UI fence.
                    self.assertEqual(
                        self.store.snapshot(completed.session_id, owner=session.owner).revision,
                        completed.revision,
                    )
                    if invalidate == "cancel":
                        self.case.fence.cancel(session.owner, self.workspace.channel)
                    else:
                        self.case.fence.begin(session.owner, self.workspace.channel, 3)

                with patch.object(Image, "open", side_effect=blocked_open):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        older = executor.submit(self.case.call, session, self.fixture.event)
                        try:
                            self.assertTrue(entered.wait(3), "Mask preparation did not start")
                            executor.submit(invalidate_request).result(timeout=2)
                        finally:
                            release.set()
                        older.result(timeout=5)
                current = self.store.snapshot(completed.session_id, owner=session.owner)
                self.assertEqual(current.revision, completed.revision)
                self.assertEqual(current.current, completed.current)
                self.assertEqual(current.derived_results, completed.derived_results)
                self.assertEqual(
                    self.case.value(session, self.workspace.context)["revision"],
                    completed.revision,
                )
                self.assertNotIn("mask_id", self.case.value(session, self.workspace.context))


class LatestCannyResultTests(unittest.TestCase):
    def test_finished_canny_images_are_latest_eligible_but_technical_maps_are_not(self):
        fixture = publication_fixture.OwnedEditorPublicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        cases = [
            ({"controlnet": {"type": "canny"}, "prompt": "finished image"}, "image"),
            ({"controlnet": {"type": "canny"}, "output_role": "guide"}, "guide"),
            ({"controlnet": {"type": "detect"}, "native_vision_source": {}}, "guide"),
            ({"native_vision_source": {"native": "RT_DETR_v4"}}, "guide"),
            ({"image_utility_source": {"operation": "face-landmarks"}}, "guide"),
            ({"controlnet": {"type": "segment"}, "segment_output": "mask"}, "mask"),
            ({"controlnet": {"type": "depth-map"}}, "guide"),
        ]
        for index, (metadata, role) in enumerate(cases):
            path = fixture.folder / f"latest-role-{index}.png"
            info = PngInfo()
            info.add_text("parameters", json.dumps(metadata))
            Image.new("RGB", (17, 23), "blue").save(path, pnginfo=info)
            records = result_presentation([str(path)])[2]
            self.assertEqual(records[0]["output_role"], role)
            self.assertEqual(latest_generated_result(records) is not None, role == "image")


if __name__ == "__main__":
    unittest.main()
