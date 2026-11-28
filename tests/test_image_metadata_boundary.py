from ruinedfooocus.core.paths import asset_path, resource_path

"""Imported image settings are renderer data, never command or job-control instructions."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def replay(parameters, **request):
    importer = function(
        "ruinedfooocus/features/prompting/prompt_processing.py", "process_metadata", {"json": json}
    )
    models = SimpleNamespace(
        get_models_by_path=lambda *a, **k: {},
        get_model_base=lambda info: "SDXL",
        get_file=lambda *a: None,
    )
    with patch.dict("sys.modules", {"ruinedfooocus.app.state": SimpleNamespace(models=models)}):
        return importer(dict(base_model_name="model.safetensors", prompt=parameters, **request))


def route(data):
    state = {"pipeline": None}
    names = (
        "image_pipeline",
        "hashbang_pipeline",
        "search_pipeline",
        "template_pipeline",
        "llama_pipeline",
    )
    factories = {
        name: SimpleNamespace(pipeline=Mock(return_value=SimpleNamespace(pipeline_type=[name])))
        for name in names
    }
    models = SimpleNamespace(
        get_file=lambda *a: Path("model.safetensors"),
        get_models_by_path=lambda *a: {},
        get_model_base=lambda *a: "SDXL",
    )
    scope = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        state=state,
        shared=SimpleNamespace(models=models),
        Path=Path,
        VIDEO_PIPELINES={},
        controlnet=SimpleNamespace(get_settings=lambda job: job.get("controlnet") or {}),
        selected_pipeline=lambda: state["pipeline"],
        NoPipeLine=lambda: SimpleNamespace(pipeline_type=[]),
        **factories,
    )
    run = function("ruinedfooocus/features/generation/pipelines.py", "update", scope)
    selected = run({"base_model_name": "model.safetensors", **data}).pipeline_type[0]
    return selected, factories


def gallery(metadata):
    from ruinedfooocus.features.image_editing.image_guidance import (
        capabilities,
        resolve_guidance,
        restore_guidance_metadata,
    )

    tree = ast.parse((ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8"))
    mapping = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "METADATA_CONTROLS" for t in n.targets)
    )
    run = function(
        "ruinedfooocus/features/gallery/ui.py",
        "apply_metadata",
        dict(
            json=json,
            gr=SimpleNamespace(update=lambda **kw: kw, Error=ValueError, Info=Mock()),
            get_ctrl=lambda name: name,
            METADATA_CONTROLS=ast.literal_eval(mapping),
            capabilities=capabilities,
            resolve_guidance=resolve_guidance,
            restore_guidance_metadata=restore_guidance_metadata,
        ),
    )
    with (
        patch.dict(
            "sys.modules",
            {
                "ruinedfooocus.features.image_editing.ui_guidance": SimpleNamespace(
                    family_for_checkpoint=lambda name: "SDXL"
                )
            },
        ),
        patch("ruinedfooocus.features.generation.automatic_upscale.choices", return_value=["Off"]),
    ):
        return run(metadata, "model.safetensors", "")


class MetadataBoundaryTests(unittest.TestCase):
    def test_gallery_replay_keeps_prompts_literal_until_the_user_changes_them(self):
        for text in ("#!clean cache", "search: synthetic", "ruinedfooocuslogo", "first --- second"):
            with self.subTest(prompt=text):
                updates = gallery({"prompt": text})
                request = {
                    name: value["value"] if isinstance(value, dict) else value
                    for name, value in updates.items()
                }
                request["image_number"] = 2
                queue = Mock(return_value=23)
                submit = function(
                    "ruinedfooocus/ui/app.py",
                    "append_work",
                    dict(
                        use_random_prompt=function(
                            "ruinedfooocus/features/prompting/prompt_processing.py",
                            "use_random_prompt",
                            {},
                        ),
                        get_promptlist=function(
                            "ruinedfooocus/features/prompting/prompt_processing.py",
                            "get_promptlist",
                            {},
                        ),
                        worker=SimpleNamespace(add_task=queue),
                        ui_controlnet=SimpleNamespace(tool_kind=lambda *_: None),
                    ),
                )
                submit(request)
                queued = queue.call_args.args[0]
                self.assertEqual((queued["prompt"], queued["image_total"]), (text, 2))
                self.assertIs(queued.get("_metadata_prompt"), True)
                self.assertNotIn("_imported_prompt", queued)
                self.assertEqual(route(queued)[0], "image_pipeline")
                submit(dict(request, prompt="#!help"))
                self.assertEqual(route(queue.call_args.args[0])[0], "hashbang_pipeline")
        for prompt in (None, 123, ["not", "text"]):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "text prompt"):
                gallery({"prompt": prompt})

    def test_every_worker_exported_render_field_can_be_imported(self):
        from ruinedfooocus.features.gallery.image_metadata import read_metadata, render_values

        tree = ast.parse((ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
        producer = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process"
        )
        keys = set()
        for node in ast.walk(producer):
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets):
                    mappings = (
                        (node.value.body, node.value.orelse)
                        if isinstance(node.value, ast.IfExp)
                        else (node.value,)
                    )
                    for mapping in mappings:
                        if isinstance(mapping, ast.Dict):
                            keys.update(
                                k.value.lower() for k in mapping.keys if isinstance(k, ast.Constant)
                            )
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and ast.unparse(target.value) == "prompt"
                        and isinstance(target.slice, ast.Constant)
                    ):
                        keys.add(target.slice.value.lower())
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "prompt.update":
                keys.update(keyword.arg for keyword in node.keywords if keyword.arg)
        self.assertTrue({"prompt", "negative", "steps", "width", "height", "seed"} <= keys)
        metadata = read_metadata(dict.fromkeys(keys, "synthetic"))
        # Runtime observations and executed guide identities are diagnostic output;
        # a guide requires explicit current image bytes rather than metadata replay.
        self.assertEqual(
            set(render_values(metadata)),
            keys - {"image_engine", "image_engine_details", "control_guide"},
        )
        self.assertNotIn(
            "control_guide",
            render_values(
                {"prompt": "scene", "control_guide": {"image": "old-guide.png", "kind": "depth"}}
            ),
        )

    def test_png_import_keeps_command_like_prompts_literal(self):
        for prompt in (
            "#!clean cache",
            "#!branch\nsynthetic",
            "#!hf_login",
            "search: synthetic",
            "ruinedfooocuslogo",
        ):
            with self.subTest(prompt=prompt):
                info = PngInfo()
                info.add_text("parameters", json.dumps({"Prompt": prompt, "steps": 8}))
                buffer = io.BytesIO()
                Image.new("RGB", (2, 2)).save(buffer, format="PNG", pnginfo=info)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    imported = replay(image.info["parameters"])
                self.assertEqual(imported["prompt"], prompt)
                selected, factories = route(imported)
                self.assertEqual(selected, "image_pipeline")
                for name in (
                    "hashbang_pipeline",
                    "search_pipeline",
                    "template_pipeline",
                    "llama_pipeline",
                ):
                    factories[name].pipeline.assert_not_called()

    def test_metadata_cannot_replace_job_state_or_uploaded_images(self):
        image, mask, refs, api_job = object(), object(), [object()], object()
        original = dict(
            task_id=23,
            task_type="process",
            index=(0, 1),
            image_total=2,
            image_number=2,
            generate_forever=False,
            input_image=image,
            main_view=image,
            inpaint_view=mask,
            reference_images=refs,
            upscale_upload=image,
            _api_job=api_job,
            _loopback_image=image,
        )
        payload = dict(
            prompt="Keep the source.",
            cfg=5,
            seed=17,
            task_id=999,
            task_type="llama",
            index=[99, 99],
            image_total=999999,
            image_number=999999,
            generate_forever=True,
            input_image=None,
            main_view=None,
            inpaint_view=None,
            reference_images=["unrequested.png"],
            upscale_upload=None,
            _api_job="fake",
            _loopback_image=None,
            _metadata_prompt=False,
            _editor_prompt_metadata={"instruction": "unrequested runtime descriptor"},
            unrelated_worker_flag=True,
        )
        imported = replay(json.dumps(payload), **original)
        for key, value in original.items():
            with self.subTest(key=key):
                self.assertIs(imported[key], value)
        self.assertEqual((imported["cfg"], imported["seed"]), (5, 17))
        self.assertIs(imported["_metadata_prompt"], True)
        self.assertNotIn("unrelated_worker_flag", imported)
        self.assertNotIn("_editor_prompt_metadata", imported)

    def test_imported_metadata_cannot_add_a_chat_route_or_clear_its_origin(self):
        imported = replay(
            json.dumps({"prompt": "#!help", "task_type": "llama", "_metadata_prompt": False})
        )
        self.assertNotIn("task_type", imported)
        self.assertIs(imported["_metadata_prompt"], True)
        self.assertEqual(route(imported)[0], "image_pipeline")

    def test_json_prompt_delimiters_do_not_split_the_metadata_document(self):
        split = function(
            "ruinedfooocus/features/prompting/prompt_processing.py",
            "get_promptlist",
            {"json": json},
        )
        for value in (
            {"prompt": 'first --- second\n"quoted"', "steps": 8},
            json.dumps({"prompt": 'first --- second\n"quoted"', "steps": 8}),
        ):
            with self.subTest(kind=type(value).__name__):
                self.assertEqual(split({"prompt": value}), [value])
        self.assertEqual(split({"prompt": "first---second"}), ["first---second"])
        self.assertEqual(split({"prompt": "first\n---\nsecond"}), ["first", "second"])

    def test_submit_and_prompt_policy_preserve_explicit_metadata(self):
        policy = function(
            "ruinedfooocus/features/prompting/prompt_processing.py", "use_random_prompt", {}
        )
        split = function(
            "ruinedfooocus/features/prompting/prompt_processing.py",
            "get_promptlist",
            {"json": json},
        )
        queue = Mock(return_value=23)
        submit = function(
            "ruinedfooocus/ui/app.py",
            "append_work",
            dict(
                use_random_prompt=policy,
                get_promptlist=split,
                worker=SimpleNamespace(add_task=queue),
                ui_controlnet=SimpleNamespace(tool_kind=lambda *_: None),
            ),
        )
        text = json.dumps({"prompt": "first --- second", "steps": 8})
        request = dict(
            prompt=text,
            image_number=2,
            obp_assume_direct_control=True,
            obp_image_prompt=True,
            input_image=object(),
        )
        submit(request)
        queued = queue.call_args.args[0]
        self.assertEqual(queued["prompt"], text)
        self.assertEqual(queued["image_total"], 2)
        imported = replay(
            queued["prompt"],
            obp_assume_direct_control=True,
            obp_image_prompt=True,
            input_image=request["input_image"],
        )
        self.assertFalse(policy(imported, imported["prompt"]))
        self.assertEqual(request["prompt"], text)

    def test_explicit_main_commands_keep_their_existing_handlers(self):
        for prompt, expected in (
            ("#!help", "hashbang_pipeline"),
            ("search: synthetic", "search_pipeline"),
            ("ruinedfooocuslogo", "template_pipeline"),
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(route({"prompt": prompt})[0], expected)

    def test_non_metadata_json_stays_text_and_invalid_metadata_prompt_is_rejected(self):
        literal = '{"steps":8,"scene":"blue sky"}'
        self.assertEqual(replay(literal, custom_steps=20)["custom_steps"], 20)
        for prompt in (None, 123, ["not", "text"], {"nested": "text"}):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "text prompt"):
                replay(json.dumps({"prompt": prompt, "steps": 8}))

    def test_renderer_settings_and_effective_observations_are_retained(self):
        settings = dict(
            prompt="Keep the source.",
            negative="noise",
            steps=8,
            cfg=5,
            seed=17,
            width=32,
            height=48,
            loras=[],
            cfg_norm_strength=1.5,
            effective_sampling_shift=3.1,
            effective_sampling_shift_mode="flow",
            controlnet={"type": "img2img", "denoise": 0.25},
            cn_strength=0.25,
            image_geometry={"width": 32, "height": 48},
            software="RuinedFooocus",
            image_edit_prompt=dict(
                schema=1,
                mode="img2img",
                operation="prompt-edit",
                source_prompt="Original scene café 0007.",
                previous_instructions=["Keep the blue scarf."],
                instruction="  Make her more detailed.  ",
                context_omission=None,
            ),
        )
        imported = replay(json.dumps(settings))
        for key, value in settings.items():
            with self.subTest(key=key):
                self.assertEqual(imported[key], value)


if __name__ == "__main__":
    unittest.main()
