"""Image-to-Prompt styles, VLM/Florence routing, output cleaning, and UI controls."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

from ruinedfooocus.features.image_processing import interrogate
from ruinedfooocus.features.prompting import image_to_prompt as itp
from ruinedfooocus.features.prompting.ui_caption import add_describe_controls, describe_image


class ImageToPromptTests(unittest.TestCase):
    def test_vlm_messages_support_all_styles_and_extra_instructions(self):
        payload = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}
        for style in itp.PROMPT_STYLE_KEYS:
            messages = itp.vlm_messages(payload, style=style, extra_instructions="focus on outfit")
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            text = messages[1]["content"][0]["text"]
            self.assertIn("focus on outfit", text)
            self.assertIs(messages[1]["content"][1], payload)

    def test_clean_prompt_output_strips_boilerplate_and_deduplicates_tags(self):
        raw = '```\nPrompt: "The image shows 1girl, solo, anime girl, 1girl, long white hair, red eyes."\n```'
        self.assertEqual(
            itp.clean_prompt_output(raw, style="tags"),
            "1girl, solo, anime girl, long white hair, red eyes",
        )
        prose = "The image depicts a serene mountain lake at golden hour — crisp reflections."
        self.assertEqual(
            itp.clean_prompt_output(prose, style="detailed"),
            "a serene mountain lake at golden hour - crisp reflections.",
        )

    def test_resolve_vision_model_prefers_stronger_installed_vlm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            smol = root / "SmolVLM-256M-Instruct-Q8_0.gguf"
            smol_mm = root / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
            qwen = root / "Qwen3.5-4B-Q4_K_M.gguf"
            qwen_mm = root / "mmproj-Qwen3.5-4B-Q4_K_M.gguf"
            for path, size in ((smol, 100), (smol_mm, 50), (qwen, 400), (qwen_mm, 200)):
                path.write_bytes(b"x" * size)
            manager = SimpleNamespace(
                paths={"path_vision": [str(root)]},
                DOWNLOADABLE_FILES={},
                get_abspath=lambda folder: Path(folder),
            )
            local = {smol.name: str(smol), qwen.name: str(qwen)}
            chosen = itp.resolve_vision_model("Auto", local=local, manager=manager)
            self.assertEqual(Path(chosen).name, qwen.name)
            self.assertEqual(
                itp.resolve_vision_model(str(smol), local=local, manager=manager), str(smol)
            )

    def test_auto_look_delegates_to_vlm_or_falls_back_to_florence(self):
        image = Image.new("RGB", (32, 32), (10, 20, 30))
        with (
            patch(
                "ruinedfooocus.features.prompting.image_to_prompt.resolve_vision_model",
                return_value=None,
            ),
            patch.object(interrogate, "florence_look", return_value="florence tags") as florence,
            patch.object(interrogate, "vlm_look") as vlm,
        ):
            self.assertEqual(
                interrogate.auto_look(image, "auto", None, style="tags"), "florence tags"
            )
            florence.assert_called_once()
            vlm.assert_not_called()

        with (
            patch(
                "ruinedfooocus.features.prompting.image_to_prompt.resolve_vision_model",
                return_value="/models/Qwen3.5-4B.gguf",
            ),
            patch.object(interrogate, "vlm_look", return_value="vlm tags") as vlm,
        ):
            self.assertEqual(interrogate.auto_look(image, "auto", None, style="tags"), "vlm tags")
            vlm.assert_called_once()

    def test_describe_image_passes_engine_style_and_options_to_queued_caption(self):
        image = Image.new("RGB", (40, 40), (100, 120, 140))
        with patch("ruinedfooocus.features.image_processing.interrogate.queued_caption") as queued:
            queued.return_value = "1girl, solo, red eyes"
            desc, history, _ = describe_image(
                "old",
                image,
                engine="auto",
                style="tags",
                extra_instructions="keep it crisp",
                vision_model="/models/vlm.gguf",
                detail="high",
            )
        self.assertEqual(desc, "1girl, solo, red eyes")
        self.assertEqual(history, ("old", "1girl, solo, red eyes"))
        self.assertEqual(queued.call_args.args[1], "auto")
        self.assertEqual(
            queued.call_args.kwargs,
            {
                "style": "tags",
                "extra_instructions": "keep it crisp",
                "vision_model": "/models/vlm.gguf",
                "detail": "high",
            },
        )

    def test_studio_controls_include_append_and_generate_handoff(self):
        with gr.Blocks(analytics_enabled=False) as app:
            prompt = gr.Textbox(value="base")
            preview = gr.Image(type="filepath")
            run_event = gr.Number(value=0)
            add_describe_controls(prompt, preview, allow_upload=True, run_event=run_event)
        callbacks = {f.fn.__name__: f for f in app.fns.values() if f.fn is not None}
        self.assertIn("append_generated_prompt", callbacks)
        self.assertIn("apply_and_generate", callbacks)

        combined, history, _ = callbacks["append_generated_prompt"].fn("base", "1girl, solo")
        self.assertEqual(combined, "base, 1girl, solo")
        self.assertEqual(history, ("base", "base, 1girl, solo"))

        applied, next_counter = callbacks["apply_and_generate"].fn("base", "1girl, solo", 3)
        self.assertEqual(applied, "1girl, solo")
        self.assertEqual(next_counter, 4)


if __name__ == "__main__":
    unittest.main()
