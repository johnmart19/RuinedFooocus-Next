from ruinedfooocus.core.paths import asset_path, resource_path

"""Prompt preservation and scheduling without importing models or shared UI state."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.prompting.prompt_scheduling import prompt_switch_per_step
from ruinedfooocus.features.image_editing.shift_attention import shift_attention

ROOT = Path(__file__).resolve().parents[1]


def function_from(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


class PromptPreservationTests(unittest.TestCase):
    def test_plain_invalid_and_escaped_brackets_use_native_tokenization(self):
        for text in (
            "a [red fox]",
            "text [label: value]",
            "[a:b:no]",
            "[]",
            r"\[red|blue\]",
            "A [red",
            "A ]red",
            '{"tags": [1, 2]}',
            "[a:b:NaN]",
            "[a:b:inf]",
            "[a:b:-2]",
            "[What?]",
            r"[red\|blue]",
            r"[red\:2]",
            r"[red\~blue]",
            r"[red\/blue]",
            r"[red\?blue]",
            r"\\\[red|blue]",
        ):
            with self.subTest(text=text):
                self.assertEqual(prompt_switch_per_step(text, 4), [])

    def test_schedules_leave_other_text_and_unmatched_brackets_intact(self):
        self.assertEqual(
            prompt_switch_per_step("[cat|dog] [label] [oops", 4),
            ["cat [label] [oops", "dog [label] [oops"] * 2,
        )
        self.assertEqual(
            prompt_switch_per_step("[name: value] [a|b]", 2), ["[name: value] a", "[name: value] b"]
        )

    def test_step_fraction_start_end_and_alternating_schedules(self):
        cases = {
            "[cat:dog:2]": ["cat", "cat", "dog", "dog"],
            "[cat:dog:.5]": ["cat", "cat", "dog", "dog"],
            "[cat::2]": ["cat", "cat", "", ""],
            "[cat:2]": ["", "", "cat", "cat"],
            "[cat:.25::.75]": ["", "cat", "cat", ""],
            "[a|b:c|d:.5]": ["a", "b", "c", "d"],
            "[a|]": ["a", "", "a", ""],
            "[[a|b]|c]": ["a", "c", "a", "c"],
            "[cat~dog]": ["cat", "cat", "dog", "dog"],
            "[a^b]": ["a", "a", "b", "b"],
            "[a/b]": ["a", "a", "b", "b"],
            r"[a\b]": ["a", "a", "b", "b"],
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(prompt_switch_per_step(prompt, 4), expected)
        with patch(
            "ruinedfooocus.features.prompting.prompt_scheduling.random.randrange", return_value=1
        ):
            self.assertEqual(prompt_switch_per_step("[a?b]", 2), ["b", "b"])
        self.assertEqual(prompt_switch_per_step("[cat:dog:.5]", 3), ["cat", "dog", "dog"])
        self.assertEqual(prompt_switch_per_step(r"\\[a|b]", 2), [r"\\a", r"\\b"])

    def test_literal_numeric_ranges_are_not_replaced_by_attention_weights(self):
        for text in (
            "A 5~10 cm cup",
            "version 1.0~2.0",
            r"\(cat:0~1)",
            "(red:1..2~3)",
            "[red~blue]",
            "ratio:1~2)",
            "(cat:0~)",
        ):
            self.assertEqual(shift_attention(text, 0.5), text)
        for distance, expected in ((0, ".5"), (0.5, "1.0"), (1, "1.5")):
            value = shift_attention("A (cat:0.5~1.5) and 5~10 cups.", distance)
            self.assertEqual(value, f"A (cat:{float(expected)}) and 5~10 cups.")
        self.assertEqual(shift_attention("(cat:0~2~0)", 0.5), "(cat:2.0)")

    def test_lora_controls_accept_signed_weights_and_keep_malformed_text(self):
        parse = function_from(
            "ruinedfooocus/features/prompting/prompt_processing.py", "parse_loras", {"re": re}
        )
        positive = "cat <lora:detail:-0.5> <lora:bad:1..2> <lora:literal>"
        loras, prompt, negative = parse(positive, "<lora:other:+.25> dog")
        self.assertEqual(
            [(row["name"], row["weight"]) for row in loras],
            [("detail", -0.5), ("literal", 1.0), ("other", 0.25)],
        )
        self.assertEqual(prompt, "cat  <lora:bad:1..2> ")
        self.assertEqual(negative, " dog")

    def test_encoder_keeps_native_text_and_reuses_only_identical_conditioning(self):
        encode = Mock(side_effect=lambda **kwargs: [kwargs["text"]])
        last_layer = Mock()

        class NativeClip:
            def __init__(self, origin=None):
                self.origin = origin or self

            def clone(self):
                return NativeClip(self.origin)

            def clip_layer(self, layer):
                last_layer(self.origin, layer)

        owner = SimpleNamespace(
            conditions={"+": {"text": None, "cache": None}},
            xl_base_patched=SimpleNamespace(clip=NativeClip()),
        )
        encode_text = function_from(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "textencode",
            {
                "CLIPTextEncode": lambda: SimpleNamespace(encode=encode),
                "CLIPSetLastLayer": lambda: SimpleNamespace(set_last_layer=last_layer),
                "warm_patched_text_encoder": lambda clip: False,
                # No installed embeddings: bare words stay unchanged.
                "path_manager": SimpleNamespace(
                    model_paths={"embedding_path": [], "cache_path": Path("missing-cache")}
                ),
            },
        )
        prompt = " ,  café\n[red cup] <lora:literal>  , "
        for clip_skip in (1, 1, 2):
            encode_text(owner, "+", prompt, clip_skip)
            self.assertEqual(owner.conditions["+"]["cache"], prompt)
        self.assertEqual(encode.call_count, 2)
        last_layer.assert_called_once_with(owner.xl_base_patched.clip, -2)

    def test_pipeline_reuses_schedule_encodings_without_timestep_gaps(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        branch = next(
            n
            for n in process.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "image_edit and input_images > 0"
        )
        owner = SimpleNamespace(
            conditions={key: {"cache": [[key, {}]]} for key in ("+", "-", "switch")}
        )

        def encode(key, text, clip_skip):
            owner.conditions[key]["cache"] = [[text, {}]]

        owner.textencode = Mock(side_effect=encode)
        helper_tree = ast.parse(
            (ROOT / "repositories/ComfyUI/node_helpers.py").read_text(encoding="utf-8")
        )
        copier = next(
            n
            for n in helper_tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "conditioning_set_values"
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            self=owner,
            callback=None,
            architecture="SDXL",
            positive_prompt="[red|blue]",
            negative_prompt="",
            clip_skip=1,
            switched_prompt=[],
            gen_data={"steps": 30},
            pp=SimpleNamespace(prompt_switch_per_step=prompt_switch_per_step),
        )
        exec(
            compile(
                ast.Module(body=[copier, *branch.orelse], type_ignores=[]),
                "pipeline_schedule",
                "exec",
            ),
            scope,
        )
        switches = [call for call in owner.textencode.call_args_list if call.args[0] == "switch"]
        self.assertEqual(len(switches), 2)
        rows = scope["switched_prompt"]
        self.assertEqual([row[0] for row in rows], ["red", "blue"] * 15)
        self.assertEqual(rows[0][1]["start_percent"], 0)
        self.assertEqual(rows[-1][1]["end_percent"], 1)
        for left, right in zip(rows, rows[1:]):
            self.assertEqual(left[1]["end_percent"], right[1]["start_percent"])
        self.assertEqual(owner.conditions["switch"]["cache"], [["blue", {}]])


if __name__ == "__main__":
    unittest.main()
