"""Main image uploads import prompt text without publishing a full PNG recipe."""

import ast
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image, PngImagePlugin

from ruinedfooocus.features.image_processing import interrogate

ROOT = Path(__file__).resolve().parents[1]


def upload_fixture():
    """Register the production upload decorator and callback without app startup."""
    source = ROOT / "ruinedfooocus/ui/app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    callback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_images_handler"
    )
    with gr.Blocks() as app:
        controls = {
            "main_view": gr.Image(type="filepath", image_mode=None),
            "prompt": gr.Textbox(),
            "gallery": gr.Gallery(),
            "generation_artifacts": gr.State([]),
            "native_main_source": gr.State(None),
            "generation_originals": gr.File(file_count="multiple", visible=False),
            "image_source": gr.State(None),
            "destination": gr.State("create"),
        }
        scope = dict(
            controls,
            gr=gr,
            Image=Image,
            metadata_prompt=interrogate.metadata_prompt,
            editor_workspace=SimpleNamespace(destination=controls["destination"]),
            shared=SimpleNamespace(get_ctrl=controls.__getitem__),
        )
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), scope)
    event = next(
        event
        for event in app.config["dependencies"]
        if event["targets"] == [(controls["main_view"]._id, "upload")]
    )
    return app, event, controls


class MainImageUploadTests(unittest.TestCase):
    def look(self, parameters, prompt=""):
        image = SimpleNamespace(info={"parameters": parameters})
        return interrogate.look(image, prompt, None)

    def test_structured_prompts_prefer_rendered_then_original_and_preserve_text(self):
        for metadata, expected in (
            ({"Prompt": "  rendered\ntext  ", "raw_prompt": "original"}, "  rendered\ntext  "),
            ({"PROMPT": "rendered"}, "rendered"),
            ({"raw_prompt": "original"}, "original"),
            ({"Prompt": None, "RAW_PROMPT": "original"}, "original"),
            ({"Prompt": "", "raw_prompt": "original"}, ""),
        ):
            with (
                self.subTest(metadata=metadata),
                patch.object(interrogate, "queued_caption") as caption,
            ):
                self.assertEqual(self.look(json.dumps(metadata)), expected)
                caption.assert_not_called()

    def test_nonempty_drafts_are_exact_and_explicit_caption_commands_still_win(self):
        for prompt in ("my draft", "  draft\nwith spaces : ", " "):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.look('{"Prompt":"saved"}', prompt), prompt)
        for command in interrogate.looks:
            with (
                self.subTest(command=command),
                patch.object(interrogate, "queued_caption", return_value="caption") as caption,
            ):
                self.assertEqual(self.look('{"Prompt":"saved"}', f" {command}: "), "caption")
                caption.assert_called_once()
                self.assertEqual(caption.call_args.args[1], command)

    def test_free_text_parameters_are_preserved_verbatim(self):
        for parameters in ("  a fox\nNegative prompt: blur\nSteps: 20  ", "{unfinished text"):
            with (
                self.subTest(parameters=parameters),
                patch.object(interrogate, "queued_caption") as caption,
            ):
                self.assertEqual(self.look(parameters), parameters)
                caption.assert_not_called()

    def test_nonprompt_or_invalid_metadata_uses_existing_caption_route(self):
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={"interrogator": "clip"})
        )
        for parameters in (None, "", "{}", "[]", "null", '{"Prompt":42}', '{"raw_prompt":[]}'):
            with (
                self.subTest(parameters=parameters),
                patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
                patch.object(interrogate, "queued_caption", return_value="caption") as caption,
            ):
                self.assertEqual(self.look(parameters), "caption")
                self.assertEqual(caption.call_args.args[1], "clip")

    def test_registered_upload_callback_serializes_prompt_and_clears_stale_sources(self):
        app, event, controls = upload_fixture()
        self.assertEqual(
            event["inputs"],
            [controls[name]._id for name in ("main_view", "prompt", "destination")],
        )
        self.assertEqual(
            event["outputs"],
            [
                controls[name]._id
                for name in (
                    "prompt",
                    "gallery",
                    "generation_artifacts",
                    "native_main_source",
                    "generation_originals",
                    "image_source",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "original.png"
            # Create with an empty prompt reads the saved prompt text only; a
            # draft, or an upload into the Editor, leaves the prompt unchanged.
            # Uploading never starts a caption job.
            kept = None
            for parameters, draft, destination, expected in (
                (
                    json.dumps(
                        {"Prompt": "rendered fox", "raw_prompt": "original fox", "width": 832}
                    ),
                    "",
                    "create",
                    "rendered fox",
                ),
                ('{"Prompt":"saved"}', "  current\ndraft  ", "create", kept),
                ('{"Prompt":"saved"}', "", "editor", kept),
                ('{"raw_prompt":"original fox"}', "", "create", "original fox"),
                ("  free text\nparameters  ", "", "create", "  free text\nparameters  "),
                (None, "", "create", ""),
            ):
                metadata = PngImagePlugin.PngInfo()
                if parameters is not None:
                    metadata.add_text("parameters", parameters)
                Image.new("RGB", (19, 23)).save(original, pnginfo=metadata)
                cached = app.move_resource_to_block_cache(str(original))
                original_bytes = original.read_bytes()
                with (
                    self.subTest(draft=draft),
                    patch.object(interrogate, "queued_caption") as caption,
                ):
                    caption.return_value = "caption"
                    state = SessionState(app)
                    for name in ("generation_artifacts", "native_main_source", "image_source"):
                        state[controls[name]._id] = ["stale"]
                    state[controls["destination"]._id] = destination
                    result = asyncio.run(
                        app.process_api(
                            event["id"], [{"path": cached}, draft, destination], state=state
                        )
                    )
                    projected = dict(zip(event["outputs"], result["data"]))
                    prompt = projected[controls["prompt"]._id]
                    if expected is kept:
                        self.assertNotIn("value", prompt)
                    else:
                        self.assertEqual(prompt, expected)
                    caption.assert_not_called()
                    self.assertEqual(len(projected[controls["gallery"]._id]), 1)
                    self.assertEqual(original.read_bytes(), original_bytes)
                    self.assertEqual(state[controls["generation_artifacts"]._id], [])
                    self.assertIsNone(state[controls["native_main_source"]._id])
                    self.assertIsNone(state[controls["image_source"]._id])
                    self.assertEqual(
                        projected[controls["generation_originals"]._id],
                        gr.update(value=[], visible=False),
                    )


if __name__ == "__main__":
    unittest.main()
