"""Run outputs are declared before workspace callbacks are bound in the real app."""

import asyncio
from concurrent.futures import CancelledError
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

import test_editor_workspace_events as fixtures
from ui_test_helpers import function_from


class EditorRunOutputTests(unittest.TestCase):
    def setUp(self):
        self.ui = fixtures.EditorWorkspaceEventTests()
        self.ui.setUp()
        self.addCleanup(self.ui.doCleanups)

    def test_early_declared_reference_gallery_delivers_for_all_run_entries_and_terminals(self):
        ui = self.ui
        with ui.app:
            ui.controls.update(
                _editor_context=ui.workspace.context,
                _editor_action=ui.workspace.action,
                _editor_framing=ui.workspace.framing,
                main_view=ui.remote["preview"],
            )
            controls = ui.workspace.controls
            ui.workspace.controls = {}
            try:
                # Production declares this event before launch_app calls bind().
                outputs = ui.production_run_outputs()
            finally:
                ui.workspace.controls = controls
            self.assertIn(ui.controls["reference_images"], outputs)
            run = function_from(
                "ruinedfooocus/ui/app.py",
                "generate_clicked",
                {
                    "editor_workspace": ui.workspace,
                    "state": {"ctrls_name": list(ui.controls)},
                },
            )
            entries = []
            for name in ("button", "prompt-submit", "prompt-tool"):
                control = gr.Textbox() if name == "prompt-submit" else gr.Button(name)
                method = control.submit if name == "prompt-submit" else control.click
                method(
                    run,
                    inputs=[*ui.controls.values(), ui.workspace.version],
                    outputs=outputs,
                    js=ui.workspace.version_js,
                )
                entries.append((name, control, "submit" if name == "prompt-submit" else "click"))
        for name, control, event in entries:
            for mode in ("owned-copy-result", "empty-failed", "cancelled-result"):
                with self.subTest(entry=name, terminal=mode):
                    session = ui.session(name + "-" + mode)
                    ui.open_library(session)
                    before = ui.value(session, ui.workspace.context).copy()
                    ui.worker_mode = mode

                    async def journey():
                        with patch(
                            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                            return_value="SDXL",
                        ):
                            function = ui.event(control, event)
                            result, _ = await ui.api(session, function)
                            await ui.finish(session, function, result)

                    asyncio.run(journey())
                    context = ui.value(session, ui.workspace.context)
                    expected = before["revision"] + (mode == "owned-copy-result")
                    self.assertEqual(context["revision"], expected)
                    self.assertEqual(ui.value(session, ui.controls["prompt"]), ui.saved["prompt"])

    def test_generator_close_retains_generator_exit_if_worker_cleanup_cancels(self):
        def stream(data):
            try:
                yield {}
            finally:
                raise CancelledError("Owned worker was stopped while closing")

        self.ui.run_request.side_effect = stream
        with self.assertLogs("ruinedfooocus.features.gallery.ui_workspace", level="WARNING"):
            with self.assertRaises(GeneratorExit):
                with self.ui.workspace._request_stream({}) as source:
                    next(source)
                    raise GeneratorExit()

    def test_upload_success_mirrors_after_source_origin_update_without_a_change_event(self):
        from gradio.processing_utils import move_files_to_cache

        ui = self.ui
        session = ui.session("upload-success-mirror")
        path = ui.root / "upload-mirror.png"
        Image.new("RGBA", (71, 117), (61, 81, 101, 121)).save(path)
        value = ui.plain(
            move_files_to_cache(
                ui.remote["preview"].postprocess(str(path)), ui.remote["preview"], postprocess=True
            )
        )
        session.values[ui.remote["preview"]._id] = value
        source = ui.event(ui.remote["preview"], "upload", "source_changed")
        ui.call(session, source)
        child = next(
            dep
            for dep in ui.app.config["dependencies"]
            if dep["trigger_after"] == source._id
            and getattr(ui.app.fns[dep["id"]].fn, "__name__", "") == "preview_source"
        )
        self.assertTrue(child["trigger_only_on_success"])
        ui.call(session, ui.app.fns[child["id"]])
        shown = ui.value(session, ui.controls["input_image"])
        with Image.open(shown["path"]) as image:
            self.assertEqual(image.size, (71, 117))
            self.assertEqual(image.getpixel((0, 0)), (61, 81, 101, 121))
        self.assertEqual(ui.value(session, ui.workspace.source_input), "preview")
        self.assertIsNone(ui.value(session, ui.workspace.context))
        ui.run_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
