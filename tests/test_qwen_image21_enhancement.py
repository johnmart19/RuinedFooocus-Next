from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import ruinedfooocus.app
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

from ruinedfooocus.features.image_editing import ui_guidance
from ruinedfooocus.features.prompting import qwen_image21_enhancement as pe, ui_rewrite


class QwenEnhancementTests(unittest.TestCase):
    def test_extracts_only_the_final_rewrite_and_its_ratio(self):
        self.assertEqual(
            pe.parse_answer('```json\n{"rewritten_prompt":" A bird ","wh_ratio":"3:2"}\n```', 0),
            pe.Rewrite("A bird", "3:2"),
        )
        self.assertEqual(
            pe.parse_answer(
                '{"rewritten_prompt":"Move <image1> onto <image2>","wh_ratio":"",'
                '"ratio_follow":"<image2>"}',
                2,
            ),
            pe.Rewrite("Move <image1> onto <image2>", "", 2),
        )
        for bad in (
            "",
            '{"rewritten_prompt":"x","wh_ratio":"0:1"}',
            '{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"<image1>"}',
            '{"rewritten_prompt":"x","ratio_follow":"<image2>"}',
            '{"rewritten_prompt":"<image2>","wh_ratio":"1:1"}',
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pe.parse_answer(bad, 1)

    def test_reference_order_and_alpha_are_preserved_in_rgb_previews(self):
        source = Image.new("RGBA", (2048, 1024), (1, 2, 3, 0))
        extra = Image.new("RGB", (16, 24), "red")
        primary, second = pe.reference_images(source, [(extra, "label")])
        self.assertEqual((primary.mode, primary.size), ("RGB", (1024, 512)))
        self.assertEqual(primary.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(second.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(source.size, (2048, 1024))
        with self.assertRaisesRegex(ValueError, "up to 10"):
            pe.reference_images(source, [extra] * 10)

    def test_pinned_node_receives_ordered_images_and_official_sampling(self):
        clip_type = SimpleNamespace(QWEN_IMAGE="edit", STABLE_DIFFUSION="text")
        clip = Mock()
        sd = SimpleNamespace(CLIPType=clip_type, load_clip=Mock(return_value=clip))
        utils = SimpleNamespace(PROGRESS_BAR_HOOK=None, set_progress_bar_global_hook=Mock())
        execute = Mock(return_value=SimpleNamespace(result=("final answer", "reasoning")))
        modules = {
            "comfy": SimpleNamespace(sd=sd, utils=utils),
            "comfy_extras.nodes_textgen": SimpleNamespace(
                TextGenerateV2=SimpleNamespace(execute=execute)
            ),
        }
        check = Mock()
        refs = (Image.new("RGB", (4, 5), "red"), Image.new("RGB", (6, 7), "blue"))
        with patch.dict(sys.modules, modules):
            result = pe.generate_answer(
                "edit.safetensors", "system", "edit", refs, check_cancelled=check
            )
        self.assertEqual(result, "final answer")
        sd.load_clip.assert_called_once_with(["edit.safetensors"], clip_type="edit")
        arguments = execute.call_args.kwargs
        self.assertIs(arguments["clip"], clip)
        self.assertTrue(arguments["thinking"])
        self.assertEqual(arguments["system_prompt"], "system")
        self.assertEqual(list(arguments["images"]), ["image1", "image2"])
        self.assertEqual(tuple(arguments["images"]["image2"].shape), (1, 7, 6, 3))
        self.assertEqual(arguments["sampling_mode"]["top_k"], 20)
        self.assertEqual(utils.set_progress_bar_global_hook.call_args.args, (None,))
        self.assertGreaterEqual(check.call_count, 2)

    def test_worker_verifies_the_system_prompt_and_releases_native_models_on_error(self):
        from ruinedfooocus.app import api_runtime, resource_lifecycle, state
        from ruinedfooocus.core.preparation import PreparationRequest

        events = []
        worker = SimpleNamespace(check_interrupt=lambda _: None)
        models = SimpleNamespace(
            unload_all_models=lambda: events.append("unload"),
            soft_empty_cache=lambda: events.append("empty"),
        )
        job = SimpleNamespace()

        def submit(operation):
            self.assertEqual(operation.resource_group, "image")
            events.append("prepare")
            operation.prepare(job)
            return operation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            weights = root / pe.weight_filename("t2i")
            weights.write_bytes(b"fixture")
            system = root / "system_prompt.txt"
            system.write_bytes(b"system")
            request = PreparationRequest(
                "https://example.invalid/system_prompt.txt",
                system,
                6,
                sha256(b"system").hexdigest(),
            )
            manager = SimpleNamespace(
                get_model_folders=lambda _: [root],
                preparation_request=Mock(return_value=PreparationRequest("", weights, 7)),
            )
            with (
                patch.object(state, "path_manager", manager),
                patch.object(api_runtime, "submit", side_effect=submit),
                patch.object(api_runtime, "wait_for_result", side_effect=lambda op: op(job, 1)),
                patch.object(
                    resource_lifecycle,
                    "require_compute_owner",
                    side_effect=lambda: events.append("owned"),
                ),
                patch.object(pe, "system_prompt_request", return_value=request),
                patch.object(
                    pe, "generate_answer", side_effect=RuntimeError("native failure")
                ) as generate,
                patch.dict(
                    sys.modules,
                    {
                        "ruinedfooocus.app.async_worker": worker,
                        "comfy": SimpleNamespace(model_management=models),
                    },
                ),
                # "from ruinedfooocus.app import async_worker" reads the package
                # attribute, so patch it as well as sys.modules.
                patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    pe.rewrite_prompt("A bird")
                self.assertEqual(events, ["prepare", "owned", "unload", "empty"])
                self.assertEqual(generate.call_args.args[:3], (weights, "system", "A bird"))
                system.write_bytes(b"tamper")
                with self.assertRaisesRegex(OSError, "SHA256"):
                    pe.rewrite_prompt("A bird")
                self.assertEqual(generate.call_count, 1)

    def test_ui_applies_reviewed_text_only_to_the_same_prompt_and_references(self):
        edit = SimpleNamespace(tool_kind=lambda selected, kind: kind)
        with (
            patch.object(ui_guidance, "family_for_checkpoint", return_value="QwenImage21"),
            patch.object(
                pe, "rewrite_prompt", return_value=pe.Rewrite('Retain "A".', "", 1)
            ) as rewrite,
            patch.dict(sys.modules, {"ruinedfooocus.features.image_editing.ui": edit}),
            gr.Blocks() as demo,
        ):
            prompt = gr.Textbox()
            checkpoint = gr.Textbox(value="qwen")
            bind = ui_rewrite.add_qwen_rewrite_controls(prompt, checkpoint)
            bind(
                gr.Image(type="pil"),
                gr.Gallery(type="pil"),
                gr.Dropdown(),
                gr.Dropdown(),
                gr.Checkbox(),
            )
            callbacks = {
                event.fn.__name__: event.fn for event in demo.fns.values() if event.fn is not None
            }
            context = ("qwen", "native", "edit", False, Image.new("RGB", (8, 8), "blue"), [])
            proposal, ratio, ticket, _ = callbacks["preview_qwen"]('Keep "A".', 4096, *context)
            self.assertEqual(proposal["value"], 'Retain "A".')
            self.assertEqual(ratio["value"], "Follow reference image 1")
            self.assertEqual(len(rewrite.call_args.args[1]), 1)
            self.assertEqual(
                callbacks["apply_qwen"]('Keep "A".', ticket, *context, {})["updates"][0],
                'Retain "A".',
            )
            context[4].putpixel((0, 0), (255, 0, 0))
            with self.assertRaisesRegex(gr.Error, "context changed"):
                callbacks["apply_qwen"]('Keep "A".', ticket, *context, {})
