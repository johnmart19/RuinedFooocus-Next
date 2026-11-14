"""Explicit caption action, immutable source capture and final browser fencing."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

from ruinedfooocus.app.state import settings
from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.features.prompting.ui_caption import add_describe_controls, describe_image


class PromptCaptionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.root / "current.png"
        Image.new("RGB", (73, 121), (20, 40, 60)).save(self.path)
        self.enterContext(patch.dict(settings.default_settings, interrogator="brainblip"))
        self.caption = self.enterContext(
            patch("ruinedfooocus.features.image_processing.interrogate.queued_caption")
        )

    def test_only_explicit_button_captions_selected_image_and_undo_keeps_manual_edits(self):
        self.caption.return_value = "Exact description café 0007."
        original = "  Original draft.  "
        with gr.Blocks(analytics_enabled=False) as app:
            prompt = gr.Textbox(value=original)
            image = gr.Image(value=str(self.path), type="filepath")
            marker = gr.State(None)
            describe, _ = add_describe_controls(prompt, image, marker)
        self.caption.assert_not_called()
        callbacks = {f.fn.__name__: f for f in app.fns.values() if f.fn is not None}
        guard = {"text": original, "revision": 0, "serial": 1}
        result = callbacks["describe_current"].fn(original, str(self.path), None, guard)
        self.assertEqual(
            result["updates"][:2],
            [self.caption.return_value, (original, self.caption.return_value)],
        )
        pixels, kind, _ = self.caption.call_args.args
        self.assertEqual(kind, "brainblip")
        self.assertEqual(pixels.size, (73, 121))
        self.assertEqual(pixels.getpixel((0, 0))[:3], (20, 40, 60))
        self.assertIn('"image": true', callbacks["describe_current"].js)
        dependency = app.config["dependencies"][callbacks["describe_current"]._id]
        self.assertEqual(dependency["targets"], [(describe._id, "click")])
        history = result["updates"][1]
        undone = callbacks["undo_description"].fn(self.caption.return_value, history, guard)
        self.assertEqual(undone["updates"][:2], [original, None])
        with self.assertRaises(gr.Error):
            callbacks["undo_description"].fn("newer manual edit", history, guard)

    def test_active_input_upload_takes_precedence_and_caption_receives_a_frozen_copy(self):
        self.caption.return_value = "upload description"
        with gr.Blocks(analytics_enabled=False) as app:
            prompt = gr.Textbox(value="draft")
            preview = gr.Image(value=str(self.path), type="filepath")
            upload = gr.Image(type="pil")
            marker, kind = gr.State(None), gr.State("input")
            add_describe_controls(prompt, preview, marker, input_source=upload, source_kind=kind)
        callback = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == "describe_current")
        pixels = Image.new("RGBA", (17, 19), (80, 100, 120, 127))
        callback.fn("draft", str(self.path), None, pixels, "input", {})
        captured = self.caption.call_args.args[0]
        self.assertEqual(captured.size, (17, 19))
        self.assertEqual(captured.getpixel((0, 0)), (80, 100, 120, 127))
        kind_change = next(
            item for item in app.config["dependencies"] if (kind._id, "change") in item["targets"]
        )
        self.assertFalse(kind_change["backend_fn"])
        self.assertIn("rfPromptContext.revision++", kind_change["js"])
        self.assertIsNot(captured, pixels)
        pixels.putpixel((0, 0), (0, 0, 0, 0))
        self.assertEqual(captured.getpixel((0, 0)), (80, 100, 120, 127))

    def test_missing_or_cached_placeholder_never_queues_a_caption(self):
        cached = self.root / "cached-placeholder.png"
        shutil.copyfile(asset_path("init_image.png"), cached)
        for image in (None, asset_path("init_image.png"), cached):
            with self.subTest(image=image), self.assertRaises(gr.Error):
                describe_image("draft", image)
        self.caption.assert_not_called()

    def test_file_change_during_caption_and_empty_result_reject_publication(self):
        def changed(*args):
            Image.new("RGB", (73, 121), (80, 90, 100)).save(self.path)
            return "old image description"

        self.caption.side_effect = changed
        with self.assertRaisesRegex(ValueError, "changed"):
            describe_image("draft", self.path)
        self.caption.side_effect, self.caption.return_value = None, ""
        with self.assertRaisesRegex(gr.Error, "empty"):
            describe_image("draft", self.path)

    def test_native_marker_uses_original_pixels_and_rejects_another_preview(self):
        from ruinedfooocus.features.image_processing.native_image_outputs import (
            NativeImageOutput,
            result_presentation,
        )

        preview = self.root / "preview.png"
        Image.new("RGB", (73, 121), (200, 180, 160)).save(preview)
        record = result_presentation([NativeImageOutput(str(self.path), str(preview))])[2][0]
        original = self.path.read_bytes()
        self.caption.return_value = "original description"
        describe_image("draft", preview, record)
        self.assertEqual(self.caption.call_args.args[0].getpixel((0, 0))[:3], (20, 40, 60))
        self.assertEqual(self.path.read_bytes(), original)
        other = self.root / "other.png"
        Image.new("RGB", (73, 121), (1, 2, 3)).save(other)
        self.caption.reset_mock()
        with self.assertRaisesRegex(gr.Error, "original image changed"):
            describe_image("draft", other, record)
        self.caption.assert_not_called()

    def test_browser_fence_tracks_only_caption_source_and_live_prompt(self):
        script = r"""const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const input = {value:'draft',addEventListener(name,fn){this[name]=fn;}};
const selected = {currentSrc:'image-A',src:'image-A'};
const unrelated = {currentSrc:'gallery-A'};
const image = {querySelectorAll(selector){return selector==='img'?[selected]:[]},addEventListener(){}};
const scope = {document:{getElementById(id){return id==='prompt'?{querySelector:()=>input}:id==='image'?image:null}}};
vm.runInNewContext(source+';globalThis.fence=fence;',scope);
const config={prompt:0,history_input:null,history_output:1,contexts:[{id:'image',image:true}],discard_context_stale:true};
const capture=()=>scope.fence.capture('prompt','describe',['draft',null,{}],config).at(-1);
const publish=guard=>scope.fence.publish('prompt',{guard,updates:['caption',['draft','caption']],stale:null},2,config);
let guard=capture(); selected.currentSrc='image-B';
assert.equal(publish(guard)[0].value,undefined);
guard=capture(); input.value='newer'; input.input();
assert.equal(publish(guard)[0].value,undefined);
input.value='draft'; input.input(); guard=capture(); image.rfPromptContext.revision++;
assert.equal(publish(guard)[0].value,undefined);
guard=capture(); unrelated.currentSrc='gallery-B';
assert.equal(publish(guard)[0],'caption');
"""
        result = subprocess.run(
            ["node", "-e", script],
            input=json.dumps(asset_path("prompt_draft_fence.js").read_text(encoding="utf-8")),
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
