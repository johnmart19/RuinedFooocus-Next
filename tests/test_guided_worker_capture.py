"""Actual GUI build and worker capture reach plan admission with separate pixels."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ruinedfooocus.app.resource_lifecycle import registry, job_scope
from ruinedfooocus.features.gallery import editor_session
from ruinedfooocus.features.image_editing import guide_handoffs as guides
import test_guided_result_ui as ui_fixtures
import test_effective_image_plan as plan_fixtures
from test_utility_sampling_capture import function


class ReachedPlanAdmission(Exception):
    pass


class GuidedWorkerCaptureTests(unittest.TestCase):
    def test_gui_owned_guide_reaches_actual_worker_plan_boundary_with_original_source(self):
        ui = ui_fixtures.GuidedResultUiTests()
        ui.setUp()
        self.addCleanup(ui.doCleanups)
        session = ui.case.session("guide-worker-capture")
        ui.result(session)
        ui.case.call(session, ui.event)
        data = ui.captured_request(session)
        packet = guides.captured(data)
        original = np.asarray(guides.request_image(packet.source)).copy()
        guide = np.asarray(guides.request_image(packet.guide)).copy()
        # A late widget mutation must not replace either captured input.
        data["input_image"] = guides.request_image(packet.guide)
        captured = self.worker_capture(ui, session, data, packet=packet)
        self.assertIs(guides.captured(captured), packet)
        np.testing.assert_array_equal(np.asarray(captured["input_image"]), original)
        self.assertFalse(np.array_equal(original[..., :3], guide))
        self.assertEqual(captured["controlnet"], packet.controls())
        self.assertEqual(captured[guides.DENOISE_FIELD], 0.35)
        self.assertEqual(captured["prompt"], data["prompt"])
        self.assertEqual(captured["negative"], data["negative"])

    def test_fresh_create_after_guided_editor_is_unconditioned_at_worker_and_real_plan(self):
        ui = ui_fixtures.GuidedResultUiTests()
        ui.setUp()
        self.addCleanup(ui.doCleanups)
        session = ui.case.session("guide-worker-fresh-create")
        ui.result(session, "normal")
        ui.case.call(session, ui.event)
        values = ui.request_values(session, destination="create")
        values["controlnet"] = {"type": "control", "model": "normal.safetensors"}
        data = ui.workspace.build_request(values, request=session.request)
        captured = self.worker_capture(ui, session, data)
        self.assertIsNone(guides.captured(captured))
        self.assertIsNone(captured["input_image"])
        controls = function(
            "ruinedfooocus/features/image_editing/controlnet.py",
            "get_settings",
            dict(NEWCN="Custom...", controlnet_settings={}),
        )(captured)
        self.assertFalse(controls)
        self.assertNotIn("cn_model", captured)
        fixture = plan_fixtures.EffectiveImagePlanTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from ruinedfooocus.features.models import effective_image_plan as plans

        request = dict(fixture.request, **captured)
        request["controlnet"] = controls
        request["base_model_name"] = fixture.path.name
        # The UI event fixture intentionally carries an opaque legacy "vae"
        # slot to test transport. Use the plan fixture's real component slots
        # at this separate trained-artifact boundary.
        request["image_component_overrides"] = {}
        planned = fixture.capture(request)
        plan = plans.get_effective_image_plan(planned)
        self.assertIsNone(guides.captured(planned))
        self.assertEqual(plan.operation, "generate")
        self.assertNotIn("control_guide", plan.metadata()["native_facts"])
        self.assertEqual(values["cn_model"], "normal.safetensors")

    def worker_capture(self, ui, session, data, *, packet=None):
        models = NS(
            offline=True,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
            get_model_base=lambda row: row["baseModel"],
        )
        paths = NS(paths={}, model_paths={}, DOWNLOADABLE_FILES={}, get_abspath=Path)
        performance = NS(apply=Mock(side_effect=AssertionError("Explicit sampling recipe changed")))
        resolutions = NS(CUSTOM_RESOLUTION="Custom...", aspect_ratios={"fixture": (64, 64)})
        control = NS(
            get_settings=function(
                "ruinedfooocus/features/image_editing/controlnet.py",
                "get_settings",
                dict(NEWCN="Custom...", controlnet_settings={}),
            )
        )
        scope = dict(
            shared=NS(path_manager=paths, models=models),
            settings=NS(default_settings={}),
            control_tools=control,
            validate_editor_data=lambda value: editor_session.validate_request(
                value, owner=session.owner, store=ui.store
            ),
        )
        capture = function("ruinedfooocus/app/async_worker.py", "capture_image_data", scope)
        data.update(
            task_type="process",
            _sampling_captured=True,
            _image_capture_models=models,
            _image_capture_paths=paths,
            _image_capture_settings={},
            _image_capture_performance=performance,
            _image_capture_resolutions=resolutions,
            custom_width=packet.source.size[0] if packet is not None else 64,
            custom_height=packet.source.size[1] if packet is not None else 64,
            aspect_ratios_selection="Custom...",
            resolution_auto=False,
            size_mode="target",
        )
        observed = []
        managers = []
        original_verify = type(packet.compatibility).verify if packet is not None else None

        def verify_receipt(receipt, manager, family, controls, spec):
            managers.append(manager)
            self.assertIs(manager, paths)
            return original_verify(receipt, manager, family, controls, spec)

        def plan_boundary(value, *args, **kwargs):
            observed.append(value)
            raise ReachedPlanAdmission()

        # The trained model/component boundaries are fixture-injected. The GUI
        # callback, owned handoff, worker function, capture hook and size planning
        # are production code; guide-plan fingerprinting has its own direct suite.
        with (
            patch(
                "ruinedfooocus.features.gallery.editor_session.default_store", return_value=ui.store
            ),
            patch(
                "ruinedfooocus.features.generation.image_preparation.capture_pending_image_selection",
                return_value=None,
            ),
            patch(
                "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                side_effect=lambda value, *args, **kwargs: dict(value),
            ),
            patch(
                "ruinedfooocus.features.models.component_profiles.snapshot_request",
                side_effect=lambda value, *args, **kwargs: dict(value),
            ),
            patch(
                "ruinedfooocus.features.models.effective_image_plan.capture_effective_image_plan",
                side_effect=plan_boundary,
            ),
            patch.dict(sys.modules, {"ruinedfooocus.features.image_editing.controlnet": control}),
            patch.object(type(packet.compatibility), "verify", new=verify_receipt)
            if packet is not None
            else nullcontext(),
        ):
            parent = registry.create(owner=session.owner)
            try:
                with job_scope(parent), self.assertRaises(ReachedPlanAdmission):
                    capture(data)
            finally:
                registry.finish(parent)
        self.assertEqual(len(observed), 1)
        if packet is not None:
            self.assertTrue(managers)
        return observed[0]


if __name__ == "__main__":
    unittest.main()
