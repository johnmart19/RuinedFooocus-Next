"""Actual Gradio map-to-editor actions use owned files and retain the user's draft."""

from concurrent.futures import CancelledError
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.features.image_editing import guide_handoffs
import test_editor_workspace_events as editor_fixture
from ui_test_helpers import function_from


@dataclass(frozen=True)
class UiControlReceipt:
    controls_json: str
    recipe: str = "ordinary"
    union_type: str = "auto"
    invert_channels: tuple = ()

    @property
    def controls(self):
        return json.loads(self.controls_json)

    def verify(self, manager, family, controls, spec):
        if family != "SDXL" or controls != self.controls:
            raise ValueError("The fixture control contract changed")
        return self

    def metadata(self):
        return dict(family="SDXL", fixture=True, runtime_qualified=False)


class GuidedResultUiTests(unittest.TestCase):
    def setUp(self):
        self.case = editor_fixture.EditorWorkspaceEventTests()
        self.plan = Mock(side_effect=self.control_plan)
        bind = editor_fixture.EditorWorkspace.bind

        def bind_with_guides(workspace, controls, **kwargs):
            controls.update(
                cn_selection=gr.Dropdown(choices=["None", "Custom..."], value="None"),
                cn_type=gr.Dropdown(choices=["Depth", "Control"], value="Depth"),
                cn_model=gr.Dropdown(
                    choices=["depth.safetensors", "normal.safetensors", "wrong.safetensors"],
                    value="depth.safetensors",
                ),
                cn_model_signature=gr.State(None),
                cn_depth_preprocess=gr.Dropdown(choices=["auto", "none"], value="auto"),
                control_recipe=gr.Dropdown(choices=["ordinary", "union"], value="ordinary"),
                control_union_type=gr.Dropdown(choices=["auto", "depth", "normal"], value="auto"),
                cn_strength=gr.Slider(minimum=0, maximum=2, value=0.71),
                cn_start=gr.Slider(minimum=0, maximum=1, value=0.14),
                cn_stop=gr.Slider(minimum=0, maximum=1, value=0.86),
                cn_edge_low=gr.State(0.2),
                cn_edge_high=gr.State(0.8),
                cn_upscale=gr.State("None"),
            )
            self.tools = gr.Tabs()
            with self.tools:
                gr.Tab("Generation", id="generation")
                gr.Tab("More tools", id="image-tools")
            controls.update(
                _editor_guide_handoff=workspace.guided_results.handoff,
                control_guide_denoise=workspace.guided_results.denoise,
            )
            kwargs.update(
                guide_checkpoint=controls["base_model_name"],
                guide_control_plan=self.plan,
                create_tabs=self.tools,
            )
            return bind(workspace, controls, **kwargs)

        with patch.object(editor_fixture.EditorWorkspace, "bind", new=bind_with_guides):
            self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.workspace = self.case.workspace
        self.store = self.case.store
        self.controls = self.case.controls
        self.helper = self.workspace.guided_results
        self.original = self.case.root / "guide-original.png"
        Image.new("RGBA", (17, 13), (19, 47, 81, 117)).save(self.original)
        self.original_bytes = self.original.read_bytes()
        self.event = self.case.event(self.helper.button)
        self.projection = self.case.app.fns[self.helper.projection["id"]]

    @staticmethod
    def control_plan(spec, checkpoint, model, signature):
        if checkpoint != "model" or model != spec.kind + ".safetensors":
            raise ValueError("Select the matching explicit fixture control.")
        values = dict(
            cn_selection="Custom...",
            cn_type="Depth" if spec.kind == "depth" else "Control",
            cn_model=model,
            cn_model_signature=signature,
            cn_depth_preprocess="none",
            control_recipe="ordinary" if spec.kind == "depth" else "union",
            control_union_type="auto" if spec.kind == "depth" else "normal",
        )
        return values

    def result(self, session, kind="depth", *, recognized=True):
        snapshot = self.store.start(self.original, owner=session.owner)
        operation = "face-landmarks" if kind == "selection" else "moge-depth"
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            operation,
            source_id=snapshot.current.artifact_id,
            owner=session.owner,
        )
        result = self.case.root / (session.request.session_hash + "-" + kind + ".png")
        Image.new("RGB", (17, 13), (23, 129, 241) if kind == "normal" else (127, 127, 127)).save(
            result
        )
        info = dict(operation=operation, width=17, height=13)
        if recognized:
            if kind == "selection":
                info["mask_role"] = "inpaint-selection"
            elif kind == "normal":
                info.update(
                    moge_output="normal",
                    convention="native-opengl",
                    projection="perspective",
                    normal_frame="native-camera",
                    normalization="native-normal*0.5+0.5",
                )
            else:
                info.update(moge_output="depth", convention="white-near", projection="perspective")
        completed = self.store.complete(
            ticket,
            result,
            execution=dict(alpha_policy="explicit guide output", image_utility_source=info),
            owner=session.owner,
        )
        session.state[self.workspace.context._id] = self.workspace.public_context(completed)
        self.case.fence.begin(session.owner, self.workspace.channel, 1)
        session.values[self.controls["prompt"]._id] = "Keep this literal prompt 猫 {draft}"
        session.values[self.controls["negative"]._id] = "Keep current exclusions"
        session.values[self.controls["custom_width"]._id] = 111
        session.values[self.controls["custom_height"]._id] = 73
        session.values[self.controls["cn_model"]._id] = (
            kind + ".safetensors" if kind != "selection" else "depth.safetensors"
        )
        return snapshot, completed

    def request_values(self, session, *, destination="editor"):
        """The actual GUI controls, before the explicit destination projection."""
        values = {
            name: self.case.value(session, control) for name, control in self.controls.items()
        }
        values.update(
            _editor_destination=destination,
            _editor_context=self.case.value(session, self.workspace.context),
            _editor_action="generate",
        )
        return values

    def captured_request(self, session):
        """Real UI values/build/capture; the trained artifact identity is fixture-owned."""
        values = self.request_values(session)
        get_settings = function_from(
            "ruinedfooocus/features/image_editing/controlnet.py",
            "get_settings",
            dict(controlnet_settings={}, NEWCN="Custom..."),
        )
        control_module = SimpleNamespace(get_settings=get_settings)
        import ruinedfooocus.features.image_editing as editing

        def compatibility(manager, family, controls, spec):
            normalized = dict(controls, recipe="ordinary", union_type="auto")
            return UiControlReceipt(json.dumps(normalized, sort_keys=True))

        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.features.image_editing.controlnet": control_module}
            ),
            patch.object(editing, "controlnet", control_module, create=True),
            patch.object(self.case.shared, "path_manager", object(), create=True),
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                return_value="SDXL",
            ),
            patch(
                "ruinedfooocus.features.image_editing.guide_control_compat.validate_guide_control",
                side_effect=compatibility,
            ),
        ):
            return self.workspace.build_request(values, request=session.request)

    def test_depth_and_normal_callbacks_restore_original_and_attach_separate_owned_guide(self):
        for kind in ("depth", "normal"):
            with self.subTest(kind=kind):
                session = self.case.session("guided-" + kind)
                original, result = self.result(session, kind)
                updates = self.case.call(session, self.event)
                handoff = self.case.value(session, self.helper.handoff)
                self.assertIsInstance(handoff, guide_handoffs.ResultGuideHandoff)
                self.assertEqual(handoff.guide_id, result.current.artifact_id)
                context = self.case.value(session, self.workspace.context)
                self.assertEqual(context["source_id"], original.current.artifact_id)
                self.assertEqual(context["revision"], result.revision + 1)
                self.assertNotIn("guide_id", context)
                self.assertEqual(context["additional_image_ids"], [result.current.artifact_id])
                with Image.open(
                    self.case.value(session, self.case.remote["preview"])["path"]
                ) as preview:
                    self.assertEqual(preview.convert("RGBA").getpixel((0, 0)), (19, 47, 81, 117))
                with Image.open(
                    self.case.value(session, self.controls["input_image"])["path"]
                ) as guide:
                    self.assertEqual(
                        guide.convert("RGB").getpixel((0, 0)),
                        (23, 129, 241) if kind == "normal" else (127, 127, 127),
                    )
                self.assertEqual(
                    self.case.value(session, self.controls["prompt"]),
                    "Keep this literal prompt 猫 {draft}",
                )
                self.assertEqual(
                    self.case.value(session, self.controls["negative"]), "Keep current exclusions"
                )
                self.assertEqual(
                    [
                        self.case.value(session, self.controls[name])
                        for name in ("custom_width", "custom_height")
                    ],
                    [111, 73],
                )
                self.assertEqual(
                    self.case.value(session, self.controls["cn_depth_preprocess"]), "none"
                )
                self.assertEqual(
                    self.case.value(session, self.controls["control_recipe"]),
                    "ordinary" if kind == "depth" else "union",
                )
                self.assertEqual(
                    [
                        self.case.value(session, self.controls[name])
                        for name in ("cn_strength", "cn_start", "cn_stop")
                    ],
                    [0.71, 0.14, 0.86],
                )
                self.assertEqual(updates[self.tools]["selected"], "image-tools")
        self.assertEqual(self.original.read_bytes(), self.original_bytes)
        self.case.run_request.assert_not_called()

    def test_face_selection_reuses_the_existing_registered_inpaint_handoff(self):
        session = self.case.session("guided-face")
        original, completed = self.result(session, "selection")
        self.case.call(session, self.event)
        context = self.case.value(session, self.workspace.context)
        self.assertEqual(context["source_id"], original.current.artifact_id)
        self.assertIn("mask_id", context)
        self.assertEqual(self.case.value(session, self.workspace.action), "inpaint")
        self.assertIsNone(self.case.value(session, self.helper.handoff))
        self.plan.assert_not_called()
        self.assertEqual(self.original.read_bytes(), self.original_bytes)
        self.case.run_request.assert_not_called()

    def test_incompatible_control_is_rejected_before_history_or_draft_changes(self):
        session = self.case.session("guided-incompatible")
        _, completed = self.result(session)
        with self.assertRaisesRegex(gr.Error, "matching explicit"):
            self.case.call(session, self.event, {self.controls["cn_model"]: "wrong.safetensors"})
        current = self.store.snapshot(completed.session_id, owner=session.owner)
        self.assertEqual(current.revision, completed.revision)
        self.assertEqual(current.current.artifact_id, completed.current.artifact_id)
        self.assertEqual(
            self.case.value(session, self.controls["prompt"]), "Keep this literal prompt 猫 {draft}"
        )
        self.assertIsNone(self.case.value(session, self.helper.handoff))

    def test_plan_cannot_rewrite_prompt_or_model_controls(self):
        session = self.case.session("guided-plan-boundary")
        _, completed = self.result(session)
        self.plan.side_effect = None
        self.plan.return_value = {"prompt": "replace user draft"}
        with self.assertRaisesRegex(gr.Error, "unsupported fields"):
            self.case.call(session, self.event)
        self.assertEqual(
            self.store.snapshot(completed.session_id, owner=session.owner).revision,
            completed.revision,
        )
        self.assertEqual(
            self.case.value(session, self.controls["prompt"]), "Keep this literal prompt 猫 {draft}"
        )

    def test_registered_projection_has_only_its_source_and_guide_outputs(self):
        expected = {
            *self.helper.controls,
            self.workspace.context,
            self.workspace.notice,
            self.workspace.mask,
            self.controls["image_source"],
            self.controls["input_image"],
            *(
                self.case.remote[name]
                for name in ("preview", "native_source", "inpaint_toggle", "inpaint")
            ),
        }
        self.assertEqual(set(self.projection.outputs), expected)
        self.assertEqual(len(self.projection.outputs), 14)
        projection = next(
            event
            for event in self.case.app.config["dependencies"]
            if event["id"] == self.projection._id
        )
        self.assertEqual(projection["show_progress"], "hidden")
        self.assertEqual(projection["show_progress_on"], [])
        self.assertEqual(set(self.event.outputs), set(self.helper.outputs))
        action = next(
            event for event in self.case.app.config["dependencies"] if event["id"] == self.event._id
        )
        self.assertEqual(action["show_progress"], "hidden")
        self.assertEqual(action["show_progress_on"], [self.helper.status._id])
        self.assertTrue(set(self.workspace.outputs).issubset(self.event.outputs))
        planned = self.control_plan(
            SimpleNamespace(kind="depth"), "model", "depth.safetensors", None
        )
        self.assertTrue(
            {
                self.tools,
                self.workspace.mask_choice,
                self.workspace.mask_index,
                self.workspace.mask_count,
                *(self.controls[name] for name in planned),
            }.issubset(self.event.outputs)
        )
        self.assertTrue(
            {
                self.tools,
                self.controls["prompt"],
                self.controls["base_model_name"],
                self.controls["custom_width"],
                self.controls["custom_height"],
            }.isdisjoint(self.projection.outputs)
        )

    def test_native_status_snapshot_does_not_export_or_upload_the_image_editor(self):
        config = next(
            row for row in self.case.app.config["dependencies"] if row["id"] == self.projection._id
        )
        payload = {"editorProjection": config | {"editorId": self.case.remote["inpaint"]._id}}
        script = Path(__file__).with_name("test_layout_progress_targets.js")
        assets = Path(gr.__file__).parent / "templates/frontend/assets"
        result = subprocess.run(
            ["node", str(script), str(assets), gr.__version__],
            input=json.dumps(payload),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        proof = json.loads(result.stdout)["imageEditor"]
        self.assertGreater(proof["before"]["exports"], 0)
        self.assertGreater(proof["before"]["uploads"], 0)
        self.assertEqual(proof["after"], {"exports": 0, "uploads": 0})

    def test_library_mask_visibility_has_no_editor_progress_and_keeps_data_update(self):
        fixture = self.case.legacy
        editor = fixture.editor
        bindings = [
            fn
            for fn in fixture.app.fns.values()
            if fn.inputs == [editor.operation] and fn.outputs == [editor.mask]
        ]
        self.assertEqual(len(bindings), 1)
        function = bindings[0]
        config = next(
            row for row in fixture.app.config["dependencies"] if row["id"] == function._id
        )
        self.assertEqual(config["show_progress"], "hidden")
        self.assertEqual(config["show_progress_on"], [])
        self.assertIsInstance(editor.mask, gr.ImageEditor)
        state = SessionState(fixture.app)
        for operation in ("inpaint", "edit"):
            result = asyncio.run(fixture.app.process_api(function._id, [operation], state=state))
            self.assertEqual(
                result["data"], [dict(__type__="update", visible=operation == "inpaint")]
            )

    def test_ordinary_source_projection_only_hides_guide_controls(self):
        session = self.case.session("ordinary-projection")
        source = self.store.start(self.original, owner=session.owner)
        context = self.workspace.public_context(source)
        session.state[self.workspace.context._id] = context
        self.case.fence.begin(session.owner, self.workspace.channel, 1)
        before = {
            name: self.case.value(session, self.controls[name])
            for name in ("prompt", "negative", "base_model_name", "custom_width", "custom_height")
        }
        projected = self.case.call(session, self.projection)
        self.assertEqual(len(projected), 14)
        self.assertFalse(projected[self.helper.button]["visible"])
        self.assertFalse(projected[self.helper.status]["visible"])
        self.assertFalse(projected[self.helper.objects]["visible"])
        self.assertEqual(self.case.value(session, self.workspace.context), context)
        for control, update in projected.items():
            if control not in self.helper.controls and not control.stateful:
                self.assertEqual(update, gr.skip())
        self.assertEqual(
            {name: self.case.value(session, self.controls[name]) for name in before}, before
        )
        self.assertEqual(self.original.read_bytes(), self.original_bytes)
        self.plan.assert_not_called()
        self.case.run_request.assert_not_called()

    def test_projection_uses_owned_provenance_and_retains_explicit_denoise(self):
        session = self.case.session("guided-projection")
        self.result(session, "normal")
        projected = self.case.call(session, self.projection)
        self.assertTrue(projected[self.helper.button]["visible"])
        self.assertNotIn("value", projected[self.helper.denoise])
        self.plan.assert_not_called()
        self.case.call(session, self.event)
        projected = self.case.call(session, self.projection, {self.helper.denoise: 0.61})
        self.assertFalse(projected[self.helper.button]["visible"])
        self.assertTrue(projected[self.helper.denoise]["visible"])
        self.assertEqual(self.case.value(session, self.helper.denoise), 0.61)
        other = self.store.start(self.original, owner=session.owner)
        projected = self.case.call(
            session, self.projection, {self.workspace.context: self.workspace.public_context(other)}
        )
        self.assertFalse(projected[self.helper.button]["visible"])
        self.assertIsNone(self.case.value(session, self.helper.handoff))

    def test_registered_projection_formats_guide_status_for_each_request_locale(self):
        from ruinedfooocus.core import translation_manager as translations

        manager = translations.catalogue()
        server_language = manager.language
        template = (
            "{guide}\n\nSelect a compatible local ControlNet in More tools before using this map."
        )
        # Interleave clients to catch accidental dependence on shared language.
        for locale in ("en", "uk", "en", *manager.manifest["locales"]):
            for kind in ("normal", "depth"):
                with self.subTest(locale=locale, kind=kind):
                    session = self.case.session("guide-status-" + locale + "-" + kind)
                    session.request = gr.Request(
                        session_hash=session.request.session_hash,
                        cookies={translations.LOCALE_COOKIE: locale},
                        headers={"accept-language": "en"},
                    )
                    self.result(session, kind)
                    projected = self.case.call(session, self.projection)
                    label = (
                        "Normal map · editing guide"
                        if kind == "normal"
                        else "Depth map · editing guide"
                    )
                    expected = manager.translate(
                        template,
                        {"guide": manager.translate(label, locale=locale)},
                        locale=locale,
                    )
                    actual = projected[self.helper.status]["value"]
                    self.assertEqual(actual, expected)
                    self.assertNotIn("{guide}", actual)
                    self.assertNotIn("__i18n__", actual)
                    self.assertEqual(
                        self.case.value(session, self.controls["prompt"]),
                        "Keep this literal prompt 猫 {draft}",
                    )
        self.assertEqual(manager.language, server_language)
        self.plan.assert_not_called()

    def test_stale_and_foreign_results_cannot_restore_another_source(self):
        session = self.case.session("guided-owner")
        _, completed = self.result(session)
        foreign = self.case.session("guided-foreign")
        with self.assertRaises(PermissionError):
            self.case.call(
                foreign,
                self.event,
                {self.workspace.context: self.workspace.public_context(completed)},
            )
        self.store.undo(completed.session_id, completed.revision, owner=session.owner)
        with self.assertRaises(CancelledError):
            self.case.call(session, self.event)

    def test_changed_source_during_projection_cannot_clear_a_newer_guide_state(self):
        session = self.case.session("guided-race")
        _, completed = self.result(session)
        newer_guide = {"newer_guide": "keep this server state"}
        session.state[self.helper.handoff._id] = newer_guide
        describe = guide_handoffs.describe_result_guide

        def delayed(*args, **kwargs):
            spec = describe(*args, **kwargs)
            self.store.undo(completed.session_id, completed.revision, owner=session.owner)
            return spec

        with patch.object(guide_handoffs, "describe_result_guide", side_effect=delayed):
            projected = self.case.call(session, self.projection)
        self.assertEqual(self.case.value(session, self.helper.handoff), newer_guide)
        for control, value in projected.items():
            if not control.stateful:
                self.assertNotIn("value", value)

    def test_newer_editor_intent_fences_a_late_projection_even_with_same_source(self):
        session = self.case.session("guided-intent-race")
        self.result(session)
        newer_guide = {"newer_guide": "keep this state"}
        session.state[self.helper.handoff._id] = newer_guide
        describe = guide_handoffs.describe_result_guide

        def delayed(*args, **kwargs):
            spec = describe(*args, **kwargs)
            self.case.fence.begin(session.owner, self.workspace.channel, 2)
            return spec

        with patch.object(guide_handoffs, "describe_result_guide", side_effect=delayed):
            projected = self.case.call(session, self.projection)
        self.assertEqual(self.case.value(session, self.helper.handoff), newer_guide)
        self.assertTrue(
            all(
                "value" not in value for control, value in projected.items() if not control.stateful
            )
        )

    def test_another_editor_action_clears_guide_foreground_role_without_history_changes(self):
        session = self.case.session("guided-other-action")
        self.result(session)
        self.case.call(session, self.event)
        context = self.case.value(session, self.workspace.context)
        before = self.store.snapshot(context["session_id"], owner=session.owner)
        self.case.call(session, self.projection, {self.workspace.action: "prompt-edit"})
        context = self.case.value(session, self.workspace.context)
        self.assertEqual(context["additional_image_ids"], [])
        self.assertIsNone(self.case.value(session, self.helper.handoff))
        self.assertEqual(
            self.store.snapshot(context["session_id"], owner=session.owner).revision,
            before.revision,
        )
        with Image.open(self.case.value(session, self.controls["input_image"])["path"]) as source:
            self.assertEqual(source.convert("RGBA").getpixel((0, 0)), (19, 47, 81, 117))

    def test_actual_gui_values_build_a_frozen_request_before_worker_admission(self):
        session = self.case.session("guided-capture")
        self.result(session)
        self.case.call(session, self.event)
        request = self.captured_request(session)
        captured = request[guide_handoffs.KEY]
        self.assertIsInstance(captured, guide_handoffs.GuideRequest)
        self.assertNotIn(guide_handoffs.HANDOFF_KEY, request)
        self.assertEqual(
            guide_handoffs.request_image(captured.source).getpixel((0, 0)), (19, 47, 81, 117)
        )
        self.assertEqual(
            guide_handoffs.request_image(captured.guide).getpixel((0, 0)), (127, 127, 127)
        )
        self.assertEqual(captured.denoise, 0.35)
        self.assertEqual(request["controlnet"]["depth_preprocess"], "none")
        self.assertEqual(request["controlnet"]["strength"], 0.71)
        self.assertEqual(request["prompt"], "Keep this literal prompt 猫 {draft}")
        self.assertEqual([request[key] for key in ("custom_width", "custom_height")], [111, 73])
        self.case.run_request.assert_not_called()

    def test_explicit_control_verification_does_not_hold_the_shared_ui_publication_lock(self):
        session = self.case.session("guided-preflight-lock")
        _, completed = self.result(session)
        entered, release = threading.Event(), threading.Event()

        def verify(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Explicit control verification was not released")
            return self.control_plan(*args)

        self.plan.side_effect = verify
        with ThreadPoolExecutor(max_workers=2) as pool:
            operation = pool.submit(self.case.call, session, self.event)
            try:
                self.assertTrue(entered.wait(3))
                invalidation = pool.submit(
                    self.case.fence.begin, session.owner, self.workspace.channel, 4
                )
                self.assertEqual(invalidation.result(timeout=2).version, 4)
            finally:
                release.set()
            operation.result(timeout=5)
        self.assertEqual(
            self.store.snapshot(completed.session_id, owner=session.owner).revision,
            completed.revision,
        )
        self.assertIsNone(self.case.value(session, self.helper.handoff))

    def test_fresh_create_drops_the_derived_control_request_and_retains_editor_widgets(self):
        session = self.case.session("guided-fresh-create")
        self.result(session, "normal")
        self.case.call(session, self.event)
        values = self.request_values(session, destination="create")
        original = dict(values)
        # Even an explicit dictionary from a prior captured request must not
        # override the fresh Create intent through get_settings' priority rule.
        values["controlnet"] = {"type": "control", "model": "normal.safetensors"}
        fresh = self.workspace.build_request(values, request=session.request)
        self.assertIsNone(fresh["input_image"])
        self.assertIsNone(fresh["editor_context"])
        self.assertIsNone(fresh[guide_handoffs.HANDOFF_KEY])
        self.assertNotIn(guide_handoffs.KEY, fresh)
        for name in (
            "controlnet",
            "cn_model",
            "cn_model_signature",
            "cn_depth_preprocess",
            "control_recipe",
            "control_union_type",
            "control_guide_denoise",
        ):
            self.assertNotIn(name, fresh)
        self.assertEqual(fresh["cn_selection"], "None")
        self.assertIsNone(fresh["cn_type"])
        self.assertEqual(values["cn_model"], "normal.safetensors")
        for name, control in self.controls.items():
            if name == "_editor_destination":
                continue
            self.assertEqual(self.case.value(session, control), original[name])
        ordinary = dict(original, _editor_guide_handoff=None)
        ordinary_create = self.workspace.build_request(ordinary, request=session.request)
        self.assertEqual(ordinary_create["cn_model"], ordinary["cn_model"])
        self.assertEqual(ordinary_create["control_recipe"], ordinary["control_recipe"])


if __name__ == "__main__":
    unittest.main()
