"""Owned Editor source conditioning through real capture and prompt processing."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.gallery.editor_session import EditorSessionStore, KEY as EDITOR_KEY
from ruinedfooocus.features.gallery.image_metadata import generation_provenance
from ruinedfooocus.features.generation.request_metadata import capture_source_generation
from ruinedfooocus.features.image_editing import edit_prompt, image_workflows as workflows
from ruinedfooocus.features.prompting.prompt_processing import process_prompt


class EditorPromptRoutingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.owner = ("editor-prompt", "one")
        self.store = EditorSessionStore(self.root / "sessions")
        self.source = self.root / "source.png"
        Image.new("RGBA", (73, 121), (20, 40, 60, 127)).save(self.source)
        self.base = "A woman with a blue scarf, café sign 0007."
        self.provenance = dict(
            prompt=self.base,
            raw_prompt="source request",
            negative="OLD negative must not leak",
            base_model_name="OLD model.safetensors",
            style_selection=["OLD style"],
            auto_negative=True,
            loras=[["OLD lora.safetensors", 1.0]],
        )
        self.snapshot = self.store.start(self.source, provenance=self.provenance, owner=self.owner)

    def capture(self, instruction, *, family="SDXL", snapshot=None, action="prompt-edit"):
        snapshot = snapshot or self.snapshot
        parameters = dict(
            prompt=instruction,
            negative="current negative",
            base_model_name="selected-model.safetensors",
            loras=[],
            target_width=73,
            target_height=121,
        )
        mask = None
        if action == "inpaint":
            from ruinedfooocus.features.image_editing.mask_handoffs import capture_mask_handoff

            path = self.root / "selection.png"
            Image.new("L", (73, 121), 255).save(path)
            bound = self.store.bind_mask(
                snapshot.session_id, path, snapshot.current.artifact_id, owner=self.owner
            )
            mask = capture_mask_handoff(
                self.store,
                snapshot.session_id,
                snapshot.revision,
                snapshot.current.artifact_id,
                bound.artifact_id,
                owner=self.owner,
                choice="union",
            )
            parameters.update(inpaint_denoise=0.61, inpaint_mask_grow=4)
        else:
            parameters["denoise"] = 0.37
        step = workflows.capture_step(
            self.store,
            snapshot.session_id,
            snapshot.revision,
            action,
            parameters,
            owner=self.owner,
            mask=mask,
            family=family,
            effective_plan={"family": family},
        )
        data = workflows.worker_request(step, self.store, owner=self.owner)
        data.update(
            _metadata_prompt=True,
            _apply_prompt_processing=False,
            style_selection=[],
            auto_negative=False,
            obp_assume_direct_control=False,
            obp_image_prompt=False,
        )
        return capture_source_generation(data, store=self.store, owner=self.owner)

    def render(self, data):
        with (
            patch(
                "ruinedfooocus.features.prompting.prompt_processing.load_styles",
                side_effect=AssertionError("No inherited style processing"),
            ),
            patch(
                "ruinedfooocus.features.prompting.prompt_processing.build_dynamic_prompt",
                side_effect=AssertionError("No random prompt replacement"),
            ),
            patch(
                "ruinedfooocus.features.prompting.prompt_enhancement.enhance_prompt",
                side_effect=AssertionError("No hidden LLM call"),
            ),
        ):
            return process_prompt([], data["prompt"], data["negative"], data)

    def metadata(self, data, positive, negative):
        return edit_prompt.attach_metadata(
            data,
            dict(
                Prompt=positive,
                Negative=negative,
                raw_prompt=data["raw_prompt"],
                base_model_name=data["base_model_name"],
            ),
        )

    def test_plain_models_keep_instruction_source_pixels_dimensions_and_current_settings(self):
        instruction = "  Make her more detailed.\nKeep café 0007 exactly!  "
        for family in ("SDXL", "Anima"):
            with self.subTest(family=family):
                data = self.capture(instruction, family=family)
                source_recipe = deepcopy(data["_source_generation"])
                positive, negative = self.render(data)
                self.assertEqual(positive, self.base + "\n" + instruction)
                self.assertEqual(data["prompt"], instruction)
                self.assertEqual(data["raw_prompt"], instruction)
                self.assertEqual(negative, "current negative")
                self.assertEqual(data["controlnet"], {"type": "img2img", "denoise": 0.37})
                self.assertEqual((data["custom_width"], data["custom_height"]), (73, 121))
                self.assertEqual(data["input_image"].getpixel((0, 0)), (20, 40, 60, 127))
                self.assertEqual(data["base_model_name"], "selected-model.safetensors")
                self.assertEqual(data["loras"], [])
                self.assertEqual(data["style_selection"], [])
                self.assertFalse(data["auto_negative"])
                self.assertEqual(data["_source_generation"], source_recipe)
                metadata = self.metadata(data, positive, negative)
                self.assertEqual(metadata["raw_prompt"], instruction)
                self.assertEqual(metadata[edit_prompt.METADATA_KEY]["source_prompt"], self.base)
                self.assertEqual(metadata[edit_prompt.METADATA_KEY]["instruction"], instruction)

    def test_native_editors_receive_only_exact_instruction(self):
        for family in ("QwenImage21", "Flux2", "SDXL_instructpix2pix"):
            with self.subTest(family=family):
                instruction = "  Make her more detailed.\nKeep café 0007!  "
                data = self.capture(instruction, family=family)
                self.assertEqual(data["controlnet"], {"type": "edit"})
                self.assertEqual(self.render(data), (instruction, "current negative"))
                self.assertEqual(data["raw_prompt"], instruction)
                descriptor = data[edit_prompt.REQUEST_KEY]
                self.assertEqual(descriptor["source_prompt"], "")
                self.assertEqual(descriptor["previous_instructions"], [])

    def test_ordinary_masked_editor_retains_context_and_owned_mask_transport(self):
        for family in ("SDXL", "SD15", "Flux"):
            with self.subTest(family=family):
                instruction = "Make her face more detailed, keep café 0007."
                data = self.capture(instruction, family=family, action="inpaint")
                ticket = data[EDITOR_KEY]
                positive, negative = self.render(data)
                self.assertEqual(positive, self.base + "\n" + instruction)
                self.assertEqual(negative, "current negative")
                self.assertEqual(data["raw_prompt"], instruction)
                self.assertTrue(data["inpaint_toggle"])
                self.assertEqual(data["inpaint_denoise"], 0.61)
                self.assertEqual(data["inpaint_mask_grow"], 4)
                self.assertEqual(ticket.mask.source_image_id, ticket.source_id)
                self.assertEqual(ticket.mask.frame.size, ticket.source.frame.size)
                self.assertEqual(data[edit_prompt.REQUEST_KEY]["mode"], "inpaint")

    def test_qwen_masked_instruction_and_unknown_inpaint_do_not_borrow_context(self):
        instruction = "  Change only the painted face.  "
        for family in ("QwenImage21", "Unknown"):
            with self.subTest(family=family):
                data = self.capture(instruction, family=family, action="inpaint")
                self.assertEqual(self.render(data), (instruction, "current negative"))
                self.assertEqual(data["prompt"], instruction)
                self.assertTrue(data["inpaint_toggle"])

    def test_chain_carries_recent_literal_edits_without_nesting_combined_prompts(self):
        snapshot = self.snapshot
        instructions = [f"literal edit {index}" for index in range(8)]
        for index, instruction in enumerate(instructions):
            data = self.capture(
                instruction,
                snapshot=snapshot,
                action="inpaint" if index % 2 else "prompt-edit",
            )
            positive, negative = self.render(data)
            expected = instructions[max(0, index - 3) : index + 1]
            self.assertEqual(positive, "\n".join([self.base, *expected]))
            self.assertEqual(positive.count(self.base), 1)
            metadata = self.metadata(data, positive, negative)
            descriptor = generation_provenance(metadata)[edit_prompt.METADATA_KEY]
            self.assertEqual(descriptor["source_prompt"], self.base)
            self.assertEqual(descriptor["previous_instructions"], expected[:-1])
            self.assertEqual(descriptor["instruction"], instruction)
            output = self.root / f"result-{index}.png"
            Image.new("RGBA", (73, 121), (index, 40, 60, 127)).save(output)
            snapshot = self.store.complete(
                data[EDITOR_KEY], output, execution=metadata, owner=self.owner
            )

    def test_missing_empty_or_oversize_source_never_replaces_or_truncates_instruction(self):
        instruction = "  Exact current instruction café 0007.  " + "é" * 5000
        for source in ({}, {"prompt": ""}, {"prompt": "x" * 16385}):
            with self.subTest(source_length=len(source.get("prompt", ""))):
                data = self.capture(instruction)
                data["_source_generation"] = source
                self.assertEqual(self.render(data), (instruction, "current negative"))
                descriptor = data[edit_prompt.REQUEST_KEY]
                self.assertEqual(descriptor["instruction"], instruction)
                self.assertEqual(
                    bool(descriptor["context_omission"]), len(source.get("prompt", "")) > 16384
                )

    def test_instruction_carry_budget_drops_whole_old_entries(self):
        data = self.capture("new instruction")
        data["_source_generation"][edit_prompt.METADATA_KEY] = dict(
            schema=1,
            mode="img2img",
            source_prompt=self.base,
            previous_instructions=["old " * 800],
            instruction="recent " * 400,
        )
        positive, _ = self.render(data)
        descriptor = data[edit_prompt.REQUEST_KEY]
        self.assertEqual(descriptor["previous_instructions"], ["recent " * 400])
        self.assertEqual(positive, self.base + "\n" + "recent " * 400 + "\nnew instruction")

    def test_inherited_inline_models_are_inert_but_current_directives_remain_explicit(self):
        from ruinedfooocus.features.prompting.prompt_processing import parse_loras

        instruction = "Exact current detail <lora:current:0.5> embedding:current"
        data = self.capture(instruction)
        source = "Woman <lora:old:0.8> <facerestore> embedding:old <style:Old>"
        previous = "Blue scarf <lora:previous:0.4> (embedding:older:0.8) lazyneg"
        data["_source_generation"][edit_prompt.METADATA_KEY] = dict(
            schema=1,
            mode="img2img",
            source_prompt=source,
            previous_instructions=[],
            instruction=previous,
        )
        positive, negative = self.render(data)
        selected, _, _ = parse_loras(positive, negative)
        self.assertEqual([value["name"] for value in selected], ["current"])
        self.assertNotIn("<facerestore>", positive)
        self.assertNotIn("embedding:old", positive)
        self.assertNotIn("<style:Old>", positive)
        self.assertIn("lazyneg", positive)
        self.assertTrue(positive.endswith(instruction))
        self.assertEqual(data["raw_prompt"], instruction)
        descriptor = data[edit_prompt.REQUEST_KEY]
        self.assertEqual(descriptor["source_prompt"], source)
        self.assertEqual(descriptor["previous_instructions"], [previous])

    def test_new_generation_after_editor_does_not_use_or_emit_source_context(self):
        data = self.capture("Editor instruction")
        self.render(data)
        data.pop(workflows.KEY)
        data.pop(EDITOR_KEY)
        data["prompt"] = "An unrelated new image, literal 0007."
        positive, negative = self.render(data)
        self.assertEqual(positive, data["prompt"])
        self.assertNotIn(edit_prompt.REQUEST_KEY, data)
        self.assertNotIn(edit_prompt.METADATA_KEY, edit_prompt.attach_metadata(data, {}))


if __name__ == "__main__":
    unittest.main()
