from ruinedfooocus.core.paths import asset_path, resource_path

"""Short/large prompt transport and preprocessing, without loading models."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from pydantic import ValidationError
from ruinedfooocus.integrations.openai_api import ImageRequest, MAX_IMAGE_PROMPT_CHARS, create_apps
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
LONG = "A red fox beside a blue teapot. " * 2100 + " FINAL_VIOLET_UMBRELLA"


def function_from(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    if name == "process_prompt":
        policy = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "use_random_prompt"
        )
        exec(compile(ast.Module(body=[policy], type_ignores=[]), path, "exec"), namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class PromptLengthTests(unittest.TestCase):
    def test_image_schema_preserves_short_unicode_and_ultra_large_prompts(self):
        for prompt in ("f", "fox", "\u732b", LONG, "z" * MAX_IMAGE_PROMPT_CHARS):
            with self.subTest(characters=len(prompt)):
                body = ImageRequest(prompt=prompt, negative_prompt=prompt)
                self.assertEqual(body.model_dump()["prompt"], prompt)
                self.assertEqual(body.model_dump()["negative_prompt"], prompt)
        self.assertEqual(ImageRequest(prompt="").prompt, "")

    def test_api_rejects_over_limit_inputs_instead_of_truncating_them(self):
        for field in ("prompt", "negative_prompt"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ImageRequest(**{field: "z" * (MAX_IMAGE_PROMPT_CHARS + 1)})

    def test_http_validation_delivers_large_prompt_to_backend_unchanged(self):
        submit = Mock(side_effect=RuntimeError("inert queue boundary"))
        backend = SimpleNamespace(resolve_model=lambda *args: ("image/example", {}), images=submit)
        api, tools = create_apps(backend, "secret")
        app = FastAPI()
        app.mount("/v1", api)
        with TestClient(app) as client:
            response = client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer secret"},
                json={"prompt": LONG, "negative_prompt": LONG},
            )
        self.assertEqual(response.status_code, 429)
        payload = submit.call_args.args[0]
        self.assertEqual(payload["prompt"], LONG)
        self.assertEqual(payload["negative_prompt"], LONG)

    def test_preprocessing_keeps_short_and_long_text_without_expansion(self):
        expand = Mock(side_effect=AssertionError("Unexpected prompt expansion"))
        process = function_from(
            "ruinedfooocus/features/prompting/prompt_processing.py",
            "process_prompt",
            {
                "re": re,
                "load_styles": lambda: {},
                "build_dynamic_prompt": expand,
                "build_dynamic_negative": expand,
                "apply_style": lambda styles, prompt, negative, keywords, **kwargs: (
                    prompt,
                    negative,
                ),
            },
        )
        for prompt in ("f", "fox", "\u732b", LONG):
            with self.subTest(characters=len(prompt)):
                self.assertEqual(
                    process([], prompt, LONG, {"obp_assume_direct_control": False}), (prompt, LONG)
                )
        expand.assert_not_called()

    def test_image_prompt_requires_its_own_random_prompt_opt_in(self):
        expand = Mock(return_value="random replacement")
        process = function_from(
            "ruinedfooocus/features/prompting/prompt_processing.py",
            "process_prompt",
            {
                "re": re,
                "load_styles": lambda: {},
                "build_dynamic_prompt": expand,
                "build_dynamic_negative": expand,
                "apply_style": lambda styles, prompt, negative, keywords, **kwargs: (
                    prompt,
                    negative,
                ),
            },
        )
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/prompting/prompt_processing.py").read_text(
                encoding="utf-8"
            )
        )
        fields = {
            node.slice.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "gen_data"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        config = dict.fromkeys(fields, "")
        config.update(
            obp_assume_direct_control=True,
            obp_image_prompt=False,
            input_image=object(),
            controlnet={"type": "img2img"},
        )
        for prompt in ("", "Keep the black hair and neon street."):
            self.assertEqual(process([], prompt, "", config), (prompt, ""))
        config.update(input_image=None, inpaint_toggle=True)
        self.assertEqual(process([], "Keep the face.", "", config), ("Keep the face.", ""))
        expand.assert_not_called()
        config["obp_image_prompt"] = True
        self.assertEqual(process([], "Keep the face.", "", config), ("random replacement", ""))
        expand.assert_called_once()

    def test_encoder_cache_handles_long_to_short_and_changes_at_the_tail(self):
        encode = Mock(side_effect=lambda **kwargs: [kwargs["text"]])
        pipeline = SimpleNamespace(
            conditions={"+": {"text": None, "cache": None}},
            xl_base_patched=SimpleNamespace(clip=object()),
        )
        textencode = function_from(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "textencode",
            {
                "re": re,
                "CLIPTextEncode": lambda: SimpleNamespace(encode=encode),
                "warm_patched_text_encoder": lambda clip: False,
                "path_manager": SimpleNamespace(
                    model_paths={"embedding_path": [], "cache_path": Path("missing-cache")}
                ),
            },
        )
        for prompt in (LONG, "f", LONG.replace("VIOLET", "ORANGE")):
            textencode(pipeline, "+", prompt, 1)
            self.assertEqual(pipeline.conditions["+"]["cache"], prompt)
        self.assertEqual(encode.call_count, 3)


if __name__ == "__main__":
    unittest.main()
