from ruinedfooocus.core.paths import asset_path, resource_path

"""PNG dimensions must replay the sampled image, without applying upscale twice."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.generation.image_output_batches import (
    ImageOutputBatch,
    OutputPlan,
    PLAN_KEY,
    with_output_plan,
)
from ruinedfooocus.features.image_editing.image_guidance import (
    capabilities,
    resolve_guidance,
    restore_guidance_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def saved_parameters(
    selected,
    native=None,
    scale=2,
    metadata=None,
    *,
    output_plan=None,
    output_index=0,
    generation_index=0,
):
    tree = ast.parse((ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process")
    loop = next(
        n
        for n in ast.walk(process)
        if isinstance(n, ast.For) and ast.unparse(n.iter) == "enumerate(imgs)"
    )
    start = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "PngInfo"
    )
    width, height = native or selected
    output = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)
    data = dict(
        cfg=1,
        sampler_name="euler",
        scheduler="simple",
        clip_skip=1,
        base_model_name="model.safetensors",
        base_model_hash="hash",
    )
    if native:
        data.update(width=width, height=height)
    if scale > 1:
        data.update(
            automatic_upscale_model=f"RealESRGAN_x{scale}plus.pth",
            automatic_upscale=f"{scale}× General (RealESRGAN)",
        )
    data.update(metadata or {})
    imgs = (output,)
    if output_plan is not None:
        data = with_output_plan(data, output_plan)
        imgs = ImageOutputBatch(
            outputs=(output,) * output_plan.max_per_generation,
            plan=data[PLAN_KEY],
            metadata=tuple(
                dict(instance_index=index) for index in range(output_plan.max_per_generation)
            ),
        )
    scope = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        gen_data=data,
        width=selected[0],
        height=selected[1],
        imgs=imgs,
        ImageOutputBatch=ImageOutputBatch,
        x=imgs[output_index],
        output_index=output_index,
        i=generation_index,
        p_txt="Keep the original framing.",
        n_txt="",
        steps=8,
        seed=71,
        start_step=0,
        denoise=0.25,
        used_loras=[],
        settings=SimpleNamespace(default_settings={}),
        pipeline=SimpleNamespace(pipeline_type=["sdxl"]),
        # The worker sets this for every sampling generation (not image utilities).
        sampling=True,
    )
    exec(
        compile(ast.Module(body=loop.body[start:end], type_ignores=[]), "output_metadata", "exec"),
        scope,
    )
    info = PngInfo()
    info.add_text("parameters", json.dumps(scope["prompt"]))
    buffer = io.BytesIO()
    Image.fromarray(output).save(buffer, format="PNG", pnginfo=info)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return json.loads(image.info["parameters"]), image.size


def replay(parameters):
    run = function(
        "ruinedfooocus/features/prompting/prompt_processing.py", "process_metadata", {"json": json}
    )
    models = SimpleNamespace(
        get_models_by_path=lambda *a, **k: {},
        get_model_base=lambda info: "SDXL",
        get_file=lambda *a: None,
    )
    with patch.dict("sys.modules", {"ruinedfooocus.app.state": SimpleNamespace(models=models)}):
        return run({"prompt": json.dumps(parameters)})


def gallery(parameters):
    tree = ast.parse((ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8"))
    controls = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "METADATA_CONTROLS" for t in n.targets)
    )
    scope = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        json=json,
        gr=SimpleNamespace(update=lambda **kw: kw, Info=Mock()),
        get_ctrl=lambda name: name,
        METADATA_CONTROLS=ast.literal_eval(controls),
        capabilities=capabilities,
        resolve_guidance=resolve_guidance,
        restore_guidance_metadata=restore_guidance_metadata,
    )
    run = function("ruinedfooocus/features/gallery/ui.py", "apply_metadata", scope)
    shared = SimpleNamespace(
        path_manager=SimpleNamespace(upscaler_filenames=[], DOWNLOADABLE_FILES={})
    )
    with patch.dict(
        "sys.modules",
        {
            "ruinedfooocus.app.state": shared,
            "ruinedfooocus.features.image_editing.ui_guidance": SimpleNamespace(
                family_for_checkpoint=lambda name: "SDXL"
            ),
        },
    ):
        return run(parameters, "model.safetensors", None)


class UpscaleMetadataTests(unittest.TestCase):
    def test_native_plans_preserve_each_output_index_and_sampling_metadata_through_png_replay(self):
        from ruinedfooocus.features.generation.native_sampling import plan_from_recipe
        from ruinedfooocus.features.models.resource_defaults import load_family_recipe

        plan = OutputPlan("fixture-instances", 2, 2)
        sampling = plan_from_recipe(load_family_recipe("native-flux2-dev"))
        for index in range(2):
            with self.subTest(index=index):
                meta, size = saved_parameters(
                    (16, 12),
                    (9, 16),
                    scale=2,
                    metadata=dict(native_sampling=sampling.metadata()),
                    output_plan=plan,
                    output_index=index,
                    generation_index=3,
                )
                self.assertEqual(
                    meta["native_output_group"],
                    dict(kind="fixture-instances", index=index, count=2, generation_index=3),
                )
                self.assertEqual(meta["native_image_output"], dict(instance_index=index))
                self.assertEqual(meta["native_sampling"], sampling.metadata())
                self.assertNotIn(PLAN_KEY, meta)
                self.assertEqual((meta["width"], meta["height"]), size)
                self.assertEqual((meta["generation_width"], meta["generation_height"]), (9, 16))
                imported = replay(meta)
                self.assertEqual((imported["width"], imported["height"]), (9, 16))

    def test_target_plan_png_gallery_and_json_round_trip_preserve_final_intent(self):
        from ruinedfooocus.features.generation.size_plan import SizePlan, KEY, capture_size_plan

        for scale in (2, 4):
            plan = SizePlan.create((120, 68), scale, alignment=16)
            meta, size = saved_parameters(
                plan.content, plan.content, scale, dict(size_plan=plan.metadata())
            )
            self.assertEqual(size, plan.target)
            self.assertEqual(meta["size_plan"], plan.metadata())
            imported, updates = replay(meta), gallery(meta)
            self.assertEqual((imported["width"], imported["height"]), plan.target)
            self.assertEqual(
                (updates["custom_width"]["value"], updates["custom_height"]["value"]), plan.target
            )
            self.assertEqual(updates["size_mode"]["value"], "target")
            recaptured = capture_size_plan(imported, None)
            self.assertEqual(recaptured[KEY].content, plan.content)
            self.assertEqual(recaptured[KEY].target, plan.target)

    def test_png_reports_actual_generation_size_when_source_aspect_differs_from_ui(self):
        for selected, native, scale in (
            ((16, 12), (9, 16), 2),
            ((8, 12), (12, 8), 4),
            ((16, 16), (7, 13), 2),
        ):
            with self.subTest(selected=selected, native=native, scale=scale):
                meta, size = saved_parameters(selected, native, scale)
                self.assertEqual((meta["width"], meta["height"]), size)
                self.assertEqual((meta["generation_width"], meta["generation_height"]), native)
                self.assertEqual(meta["Prompt"], "Keep the original framing.")

    def test_json_import_uses_generation_size_in_both_worker_and_custom_resolution(self):
        parameters = {
            "prompt": "Keep the original framing.",
            "width": 18,
            "height": 32,
            "generation_width": 9,
            "generation_height": 16,
            "automatic_upscale": "2× General (RealESRGAN)",
        }
        data = replay(parameters)
        self.assertEqual((data["width"], data["height"]), (9, 16))
        self.assertEqual((data["custom_width"], data["custom_height"]), (9, 16))
        self.assertEqual(data["automatic_upscale_target"], "Generated image")
        self.assertEqual(data["automatic_upscale"], parameters["automatic_upscale"])
        self.assertEqual(data["prompt"], parameters["prompt"])

    def test_saved_png_gallery_and_json_replay_agree_and_do_not_enlarge_twice(self):
        for scale in (2, 4):
            with self.subTest(scale=scale):
                meta, size = saved_parameters((16, 12), (9, 16), scale)
                imported, updates = replay(meta), gallery(meta)
                self.assertEqual((imported["width"], imported["height"]), (9, 16))
                self.assertEqual(
                    (updates["custom_width"]["value"], updates["custom_height"]["value"]), (9, 16)
                )
                self.assertEqual((imported["width"] * scale, imported["height"] * scale), size)

    def test_legacy_metadata_and_text_to_image_keep_their_existing_size(self):
        for native in (None, (16, 12)):
            meta, size = saved_parameters((16, 12), native)
            data = replay(meta)
            self.assertEqual((data["width"], data["height"]), (16, 12))
            self.assertEqual((data["custom_width"], data["custom_height"]), (16, 12))
            self.assertEqual(size, (32, 24))
        legacy = replay({"prompt": "Original", "width": 16, "height": 12})
        self.assertEqual((legacy["width"], legacy["height"]), (16, 12))
        self.assertEqual(legacy["automatic_upscale"], "Off")
        plain, size = saved_parameters((16, 12), (9, 16), scale=1)
        self.assertEqual((plain["width"], plain["height"]), size)
        self.assertNotIn("generation_width", plain)


if __name__ == "__main__":
    unittest.main()
