"""Non-sampling utilities do not depend on a selected checkpoint's Auto recipe."""

import ast
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
from ruinedfooocus.features.models.lora_sources import CapturedPromptLoras
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_editing import image_workflows as workflows
import test_model_recommendations as recommendations

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


class ReachedUtility(Exception):
    pass


class ReachedUtilityProcess(BaseException):
    pass


class UtilitySamplingCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.performance = recommendations.RecommendationTests()._performance(self.root / "cache")
        self.performance.model_recommendations.offline = True
        self.values = {}
        self.family = "SDXLRefiner"
        self.models = SimpleNamespace(
            offline=True,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": self.family},
            get_model_base=lambda value: value["baseModel"],
            detect_checkpoint_base=lambda path: self.family,
            get_file=Mock(side_effect=AssertionError("Utility resolved image checkpoint")),
        )
        (self.root / "RealESRGAN_x2plus.pth").write_bytes(b"owned local upscale fixture")
        self.manager = SimpleNamespace(
            DOWNLOADABLE_FILES={},
            paths={},
            get_abspath=Path,
            get_model_folders=lambda kind: [self.root],
        )
        self.shared = SimpleNamespace(
            models=self.models,
            performance_settings=self.performance,
            path_manager=self.manager,
            resolution_settings=object(),
        )
        self.controls = SimpleNamespace()
        self.controls.get_settings = function(
            "ruinedfooocus/features/image_editing/controlnet.py",
            "get_settings",
            {
                "NEWCN": "Custom...",
                "controlnet_settings": json.loads(
                    resource_path("defaults", "powerup.default").read_text(encoding="utf-8")
                ),
            },
        )
        self.scope = {
            "shared": self.shared,
            "settings": SimpleNamespace(default_settings=self.values),
            "control_tools": self.controls,
        }
        self.capture = function(
            "ruinedfooocus/app/async_worker.py", "capture_image_data", self.scope
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
                    ),
                    "ruinedfooocus.features.image_editing.controlnet": self.controls,
                },
            )
        )

    def request(self, controls):
        return dict(
            task_type="process",
            prompt="Keep the supplied image.",
            base_model_name="unqualified.safetensors",
            performance_selection="Model Defaults",
            custom_steps=12,
            cfg=3.0,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=1,
            loras=[],
            image_number=1,
            silent=True,
            **controls,
        )

    def process(self, data, expected):
        utility = SimpleNamespace(
            pipeline_type=[expected],
            parse_gen_data=lambda value: value,
            load_base_model=Mock(side_effect=ReachedUtility()),
        )
        factory = SimpleNamespace(pipeline=Mock(return_value=utility))
        state = {"pipeline": None}
        route_scope = {
            "state": state,
            "shared": self.shared,
            "Path": Path,
            "controlnet": self.controls,
            "upscale_pipeline": factory,
            "rembg_pipeline": factory,
        }
        function("ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", route_scope)
        update = function("ruinedfooocus/features/generation/pipelines.py", "update", route_scope)
        process = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            {
                **self.scope,
                "check_interrupt": lambda value: None,
                "process_metadata": lambda value: value,
                "pipelines": SimpleNamespace(update=update),
                "get_checkpoint_hashes": Mock(
                    side_effect=AssertionError("Utility hashed diffusion checkpoint")
                ),
            },
        )
        with self.assertRaises(ReachedUtility):
            process(data)
        self.assertEqual(state["pipeline"].pipeline_type, [expected])
        utility.load_base_model.assert_called_once()

    def test_utility_dispatch_ignores_free_prompt_commands_and_missing_checkpoint_hash(self):
        for kind in ("rembg", "upscale"):
            for prompt in ("#! touch prohibited", "search: ignored", "ruinedfooocuslogo"):
                with self.subTest(kind=kind, prompt=prompt):
                    data = self.request({"controlnet": {"type": kind}})
                    data["prompt"] = prompt
                    self.assertNotIn("base_model_hash", data)
                    self.process(data, kind)
        self.models.get_file.assert_not_called()

    def test_utility_reaches_own_process_without_generation_hash_dimensions_prompt_or_loras(self):
        from PIL import Image

        for kind, own_model in (
            ("rembg", "owned-background.safetensors"),
            ("upscale", "owned-upscaler.pth"),
        ):
            with self.subTest(kind=kind):
                utility = SimpleNamespace(
                    pipeline_type=[kind],
                    parse_gen_data=lambda data: data,
                    load_base_model=Mock(),
                    process=Mock(side_effect=ReachedUtilityProcess()),
                )
                forbidden = Mock(side_effect=AssertionError("Utility used generation resources"))
                process = function(
                    "ruinedfooocus/app/async_worker.py",
                    "_process",
                    {
                        **self.scope,
                        "check_interrupt": lambda data: None,
                        "process_metadata": forbidden,
                        "process_prompt": forbidden,
                        "parse_loras": forbidden,
                        "get_checkpoint_hashes": forbidden,
                        "pipelines": SimpleNamespace(update=lambda data: utility),
                        "TimeIt": nullcontext,
                        "resource_path": resource_path,
                        "random": random,
                        "model_management": SimpleNamespace(
                            InterruptProcessingException=InterruptedError
                        ),
                    },
                )
                data = self.request({"controlnet": {"type": kind}})
                data.update(
                    input_image=Image.new("RGB", (64, 48)),
                    prompt="#! ignored",
                    custom_steps=object(),
                    style_selection=object(),
                    negative=object(),
                    loras=object(),
                    seed=7,
                )
                if kind == "rembg":
                    data["rembg_native_model"] = own_model
                else:
                    data["controlnet"]["upscaler"] = own_model
                with self.assertRaises(ReachedUtilityProcess):
                    process(data)
                utility.load_base_model.assert_called_once_with(own_model, hash=None)
                utility.process.assert_called_once()
                executed = utility.process.call_args.kwargs["gen_data"]
                self.assertEqual(
                    (executed["steps"], executed["width"], executed["height"]), (1, 64, 48)
                )
                self.assertEqual(
                    (executed["positive_prompt"], executed["negative_prompt"]), ("", "")
                )
                forbidden.assert_not_called()
        self.models.get_file.assert_not_called()

    def metadata_process(self, data):
        selected = Mock(side_effect=ReachedUtility())
        metadata = function(
            "ruinedfooocus/features/prompting/prompt_processing.py",
            "process_metadata",
            {},
        )
        process = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            {
                **self.scope,
                "check_interrupt": lambda value: None,
                "process_metadata": metadata,
                "pipelines": SimpleNamespace(update=selected),
            },
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": self.shared}),
            self.assertRaises(ReachedUtility),
        ):
            process(dict(data))  # image_stage gives the worker a per-execution copy.
        return selected.call_args.args[0]

    def assert_unqualified_capture(self, data):
        # Isolate recipe refusal from unrelated sampling component inventory.
        # Utility captures keep the real snapshot caller and its no-checkpoint guard.
        with (
            patch(
                "ruinedfooocus.features.models.component_profiles.snapshot_request",
                side_effect=lambda data, *args: data,
            ),
            self.assertRaises(UnavailableModelRecipe),
        ):
            self.capture(data)

    def test_actual_capture_and_native_dispatch_ignore_unqualified_recipe_for_utilities(self):
        for family in ("SDXLRefiner", "Chroma"):
            self.family = family
            for controls, expected in (
                ({"controlnet": {"type": "rembg"}}, "rembg"),
                (
                    {"controlnet": {"type": "upscale", "upscaler": "RealESRGAN_x2plus.pth"}},
                    "upscale",
                ),
                (
                    {
                        "automatic_upscale": "2× General (RealESRGAN)",
                        "automatic_upscale_target": "Input image only",
                    },
                    "upscale",
                ),
            ):
                with (
                    self.subTest(family=family, controls=controls),
                    patch.object(
                        self.performance,
                        "apply",
                        wraps=self.performance.apply,
                    ) as apply,
                ):
                    original = self.request(controls)
                    captured = self.capture(original)
                    self.assertIs(captured["_sampling_captured"], False)
                    self.assertNotIn("_sampling_captured", original)
                    self.assertEqual(captured["controlnet"]["type"], expected)
                    self.assertEqual(captured["custom_steps"], 12)
                    self.process(captured, expected)
                    self.process(original, expected)  # legacy unmarked utility fallback
                    apply.assert_not_called()

    def test_dedicated_upload_is_authoritative_only_for_implicit_create_upscale(self):
        from PIL import Image

        from ruinedfooocus.features.generation.ui_request import build_ui_request

        source = Image.new("RGBA", (3, 2), "red")
        upload = Image.new("RGBA", (13, 17), "yellow")
        for controls in (None, {}, {"type": "img2img"}, {"type": "upscale"}):
            with self.subTest(controls=controls):
                values = dict(
                    controlnet=controls,
                    input_image=source,
                    upscale_upload=upload,
                    automatic_upscale="2× General (RealESRGAN)",
                    automatic_upscale_target="Input image only",
                )
                result = build_ui_request(values, models=self.models, settings={})
                expected = source if controls == {"type": "upscale"} else upload
                self.assertIs(result["input_image"], expected)
                self.assertIs(values["input_image"], source)
        # A missing explicit source must fail, never borrow an unrelated upload.
        values.update(input_image=None, controlnet={"type": "upscale"})
        self.assertIsNone(build_ui_request(values, models=self.models, settings={})["input_image"])

    def test_editor_upscale_survives_actual_ui_submission_capture_and_cpu_processing(self):
        import hashlib

        import numpy as np
        from PIL import Image
        import torch

        from ruinedfooocus.features.gallery.editor_session import EditorSessionStore
        from ruinedfooocus.features.generation.size_plan import KEY

        source = Image.fromarray(
            np.array(
                [
                    [[255, 0, 0, 0], [0, 255, 0, 128], [0, 0, 255, 255]],
                    [[255, 0, 0, 0], [0, 255, 0, 128], [0, 0, 255, 255]],
                ],
                dtype=np.uint8,
            )
        )
        before = source.tobytes()
        path = self.root / "editor-source.png"
        source.save(path)
        store = EditorSessionStore(self.root / "editor")
        owner = ("fixture", "upscale")
        snapshot = store.start(path, owner=owner)
        self.scope["validate_editor_data"] = lambda data: store.validate(
            data["_editor_operation"], owner=owner
        )
        self.shared.settings = SimpleNamespace(default_settings=self.values)
        self.shared.state = {}
        process = function(
            "ruinedfooocus/features/image_processing/upscale_pipeline.py",
            "process",
            dict(
                np=np,
                torch=torch,
                worker=SimpleNamespace(add_result=Mock(), check_interrupt=Mock()),
                controlnet=self.controls,
                InterruptibleUpscaler=lambda model, data: model,
            ),
        )

        class CPUModel:
            scale = 2
            patcher = SimpleNamespace(load_device=torch.device("cpu"))

            def __call__(self, pixels):
                self.assert_cpu(pixels)
                return pixels.repeat_interleave(self.scale, 2).repeat_interleave(self.scale, 3)

            assert_cpu = staticmethod(lambda pixels: self.assertEqual(pixels.device.type, "cpu"))

        tiles = []

        def tiled_scale(pixels, upscale, **options):
            tiles.append(options["tile_x"])
            if options["tile_x"] == 512:
                raise MemoryError("Exercise native smaller-tile fallback on CPU")
            return upscale(pixels)

        def raise_non_oom(error):
            if not isinstance(error, MemoryError):
                raise error

        management = SimpleNamespace(
            load_models_gpu=Mock(),
            intermediate_device=lambda: torch.device("cpu"),
            intermediate_dtype=lambda: torch.float32,
            raise_non_oom=raise_non_oom,
        )
        native_scope = dict(
            torch=torch,
            model_management=management,
            comfy=SimpleNamespace(
                model_management=management,
                utils=SimpleNamespace(
                    get_tiled_scale_steps=lambda *args, **kwargs: 1,
                    ProgressBar=lambda _: None,
                    tiled_scale=tiled_scale,
                    common_upscale=lambda pixels, width, height, *_: (
                        torch.nn.functional.interpolate(
                            pixels, size=(height, width), mode="bilinear", align_corners=False
                        )
                    ),
                ),
            ),
            io=SimpleNamespace(ComfyNode=object, NodeOutput=lambda pixels: (pixels,)),
        )
        native_source = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
        tree = ast.parse(
            (native_source / "comfy_extras/nodes_upscale_model.py").read_text(encoding="utf-8")
        )
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ImageUpscaleWithModel"
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "native CPU upscaler", "exec"),
            native_scope,
        )
        native = SimpleNamespace(ImageUpscaleWithModel=native_scope["ImageUpscaleWithModel"])
        pipeline = SimpleNamespace(load_upscaler_model=lambda *_: CPUModel())
        for upload in (None, Image.new("RGBA", (13, 17), "yellow")):
            for target in ((6, 4), (7, 5)):
                with self.subTest(upload=upload is not None, target=target):
                    step = workflows.capture_step(
                        store,
                        snapshot.session_id,
                        snapshot.revision,
                        "upscale",
                        dict(
                            upscaler="2× General (RealESRGAN)",
                            target_width=target[0],
                            target_height=target[1],
                        ),
                        owner=owner,
                    )
                    request = dict(
                        workflows.worker_request(step, store, owner=owner), upscale_upload=upload
                    )
                    admitted = []

                    def admit(data):
                        captured = self.capture(data)
                        self.assertIsInstance(captured["input_image"], Image.Image)
                        self.assertEqual(captured["input_image"].mode, "RGBA")
                        self.assertEqual(captured["input_image"].size, source.size)
                        self.assertEqual(captured["input_image"].tobytes(), before)
                        self.assertEqual(captured[KEY].target, target)
                        captured["task_id"] = 1
                        result = process(pipeline, captured)[0]
                        self.assertEqual(result.shape, (target[1], target[0], 4))
                        self.assertEqual(int(result[..., 3].min()), 0)
                        # Native bilinear alpha can round down one byte after an odd fit.
                        self.assertAlmostEqual(int(result[..., 3].max()), 255, delta=1)
                        self.assertTrue(np.any((result[..., 3] > 0) & (result[..., 3] < 254)))
                        self.assertEqual(captured["upscale_source_size"], [3, 2])
                        self.assertEqual(
                            captured["upscale_source"]["pixels_sha256"],
                            hashlib.sha256(before).hexdigest(),
                        )
                        self.assertEqual(captured["input_image"].tobytes(), before)
                        admitted.append(captured)
                        return 1

                    warning = Mock()
                    scope = dict(
                        shared=self.shared,
                        settings={},
                        ui_controlnet=SimpleNamespace(
                            tool_kind=lambda *_: None, family_for_checkpoint=None
                        ),
                        generation_task=object(),
                        main_view=object(),
                        update_clicked=lambda: {},
                        update_results=lambda _: {},
                        worker=SimpleNamespace(
                            add_task=admit,
                            task_result=lambda _: ("results", []),
                            interrupt_processing=Mock(),
                        ),
                        gr=SimpleNamespace(Warning=warning),
                        time=SimpleNamespace(sleep=lambda _: None),
                    )
                    function("ruinedfooocus/ui/app.py", "append_work", scope)
                    generate = function("ruinedfooocus/ui/app.py", "generate_request", scope)
                    with (
                        patch(
                            "ruinedfooocus.app.resource_lifecycle.current_owner", return_value=owner
                        ),
                        patch(
                            "ruinedfooocus.features.gallery.editor_session.default_store",
                            return_value=store,
                        ),
                        patch.dict(sys.modules, {"comfy_extras.nodes_upscale_model": native}),
                    ):
                        list(generate(request))
                    warning.assert_not_called()
                    self.assertEqual(len(admitted), 1)
                    self.assertEqual(request["input_image"].tobytes(), before)
                    self.assertEqual(store.snapshot(snapshot.session_id, owner=owner), snapshot)
        self.assertEqual(tiles, [512, 256] * 4)

    def test_generation_stays_strict_and_utility_retarget_requires_explicit_recapture(self):
        utility = self.capture(self.request({"controlnet": {"type": "rembg"}}))
        retargeted = {**utility, "controlnet": {"type": "img2img", "denoise": 0.3}}
        process = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            {
                **self.scope,
                "check_interrupt": lambda value: None,
                "process_metadata": lambda value: value,
                "pipelines": SimpleNamespace(
                    update=Mock(side_effect=AssertionError("Reached pipeline"))
                ),
            },
        )
        with self.assertRaisesRegex(ValueError, "new generation submission"):
            process(retargeted)
        with self.assertRaisesRegex(ValueError, "changed after plan capture"):
            self.capture(retargeted)
        # A new explicit submission owns a new plan; the prior utility plan is inert.
        retargeted.pop("_effective_image_plan")
        self.assert_unqualified_capture(retargeted)
        legacy = self.request({"controlnet": {"type": "img2img", "denoise": 0.3}})
        with self.assertRaises(UnavailableModelRecipe):
            process(legacy)
        # Explicit submission recapture with manual settings produces a sampling recipe.
        self.models.get_file.side_effect = None
        path = self.root / "unqualified.safetensors"
        path.write_bytes((2).to_bytes(8, "little") + b"{}")
        self.models.get_file.return_value = path
        with (
            patch.object(
                assets,
                "capture_image_assets",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch(
                "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
                side_effect=lambda data, *args, **kwargs: data,
            ),
        ):
            manual = self.capture({**retargeted, "performance_selection": "Own settings"})
        self.assertIs(manual["_sampling_captured"], True)
        self.assertEqual((manual["custom_steps"], manual["cfg"]), (12, 3.0))

    def test_api_and_tool_json_prompts_stay_literal_through_capture_assets_and_execution(self):
        prompt = json.dumps({"prompt": "replacement", "controlnet": {"type": "img2img"}, "cfg": 99})
        for source in ({"_api_job": object()}, {"task_type": "tool_call"}):
            with (
                self.subTest(source=source),
                patch.object(self.performance, "apply", wraps=self.performance.apply) as apply,
            ):
                request = self.request({"controlnet": {"type": "rembg"}})
                request.update(source, prompt=prompt)
                captured = self.capture(request)
                self.assertIs(captured["_sampling_captured"], False)
                self.assertEqual(captured["controlnet"], {"type": "rembg"})
                executed = self.metadata_process(captured)
                self.assertEqual((executed["prompt"], executed["cfg"]), (prompt, 3.0))
                self.assertEqual(executed["controlnet"], {"type": "rembg"})
                if "_api_job" in source:
                    self.assertIs(captured["_api_job"], source["_api_job"])
                apply.assert_not_called()
                # Utility-looking literal JSON cannot bypass sampling qualification.
                request.update(
                    controlnet={"type": "img2img"},
                    prompt=json.dumps(
                        {
                            "prompt": "replacement",
                            "controlnet": {"type": "rembg"},
                        }
                    ),
                )
                self.assert_unqualified_capture(request)

    def test_metadata_controls_have_the_same_authority_at_capture_and_execution(self):
        request = self.request({"controlnet": {"type": "img2img"}})
        request["prompt"] = json.dumps(
            {"prompt": "restored", "controlnet": {"type": "rembg"}, "steps": 17}
        )
        with patch.object(self.performance, "apply", wraps=self.performance.apply) as apply:
            captured = self.capture(request)
            self.assertIs(captured["_sampling_captured"], False)
            with patch.object(self.models, "get_file", return_value=None):
                for data in (captured, request):
                    executed = self.metadata_process(data)
                    self.assertEqual(executed["prompt"], "restored")
                    self.assertEqual(executed["controlnet"], {"type": "rembg"})
                    self.assertEqual(executed["custom_steps"], 17)
            apply.assert_not_called()
        # Imported sampling settings cannot retain an old utility bypass.
        captured["prompt"] = json.dumps({"prompt": "new scene", "controlnet": {"type": "img2img"}})
        with self.assertRaisesRegex(ValueError, "new generation submission"):
            self.metadata_process(captured)
        self.assert_unqualified_capture(captured)

    def test_imported_metadata_resets_input_only_upscale_before_recipe_classification(self):
        request = self.request(
            {
                "automatic_upscale": "2× General (RealESRGAN)",
                "automatic_upscale_target": "Input image only",
            }
        )
        request["prompt"] = json.dumps({"prompt": "restored scene"})
        # process_metadata resets this target to Generated image; it must sample.
        self.assert_unqualified_capture(request)
        # This request now samples; the utility-only model-access guard does not apply.
        with (
            patch.object(self.models, "get_file", return_value=None),
            self.assertRaises(UnavailableModelRecipe),
        ):
            self.metadata_process(request)

    def test_imported_utility_metadata_never_resolves_generation_models_or_prompts(self):
        request = self.request({"controlnet": {"type": "img2img"}})
        request["prompt"] = json.dumps(
            {"prompt": "restored", "controlnet": {"type": "rembg"}, "steps": 17}
        )
        with (
            patch.object(
                self.models,
                "get_models_by_path",
                side_effect=AssertionError("Utility read checkpoint metadata"),
            ),
            patch(
                "ruinedfooocus.features.prompting.local_model.capture_prompt_preparation",
                side_effect=AssertionError("Utility prepared a prompt model"),
            ),
        ):
            captured = self.capture(request)
            for data in (request, captured):
                executed = self.metadata_process(data)
                self.assertEqual(executed["controlnet"], {"type": "rembg"})
                self.assertEqual(executed["custom_steps"], 17)
        self.models.get_file.assert_not_called()

    def test_imported_checkpoint_intent_selects_defaults_before_sampling_fields_win(self):
        request = self.request({"controlnet": {"type": "img2img"}})
        request["prompt"] = json.dumps(
            dict(prompt="restored", base_model_name="imported.safetensors", steps=17, cfg=4)
        )
        observed = []

        def defaults(data, family):
            observed.append((data["base_model_name"], family))
            return dict(data, custom_steps=25, cfg=8)

        with (
            patch.object(self.performance, "apply", side_effect=defaults),
            patch.object(self.models, "get_file", return_value=None),
        ):
            executed = self.metadata_process(request)
        self.assertEqual(observed, [("imported.safetensors", self.family)])
        self.assertEqual(executed["base_model_name"], "imported.safetensors")
        self.assertEqual((executed["custom_steps"], executed["cfg"]), (17, 4))
        observed.clear()
        checkpoint = self.root / "imported.safetensors"
        header = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
        checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        with (
            patch.object(self.performance, "apply", side_effect=defaults),
            patch.object(self.models, "get_file", return_value=checkpoint),
            patch(
                "ruinedfooocus.features.prompting.local_model.capture_prompt_preparation",
                side_effect=lambda data, *args: data,
            ),
            patch(
                "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch(
                "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
                side_effect=lambda data, *args, **kwargs: data,
            ),
        ):
            for prompt in (request["prompt"], [request["prompt"]]):
                with self.subTest(prompt_list=isinstance(prompt, list)):
                    observed.clear()
                    captured = self.capture(dict(request, prompt=prompt))
                    self.assertIs(captured["_sampling_captured"], True)
                    self.assertIsInstance(
                        captured["_effective_image_plan"].lora_prompts[0], CapturedPromptLoras
                    )
                    self.assertEqual(
                        captured["_effective_image_plan"].recipe.sampling["custom_steps"], 17
                    )
                    self.assertEqual(captured["_effective_image_plan"].recipe.sampling["cfg"], 4)
                    executed = self.metadata_process(captured)
                    self.assertEqual(observed, [("imported.safetensors", self.family)])
                    self.assertEqual((executed["custom_steps"], executed["cfg"]), (17, 4))

    def test_generation_defaults_still_apply_before_imported_sampling_values(self):
        request = self.request({"controlnet": {"type": "img2img"}})
        request["prompt"] = json.dumps({"prompt": "restored scene", "steps": 17, "cfg": 4})
        with (
            patch.object(
                self.performance,
                "apply",
                side_effect=lambda data, family: dict(data, custom_steps=25, cfg=8),
            ) as apply,
            patch.object(self.models, "get_file", return_value=None),
        ):
            executed = self.metadata_process(request)
        apply.assert_called_once()
        self.assertEqual((executed["custom_steps"], executed["cfg"]), (17, 4))

    def test_effective_metadata_controls_select_the_owned_compute_service(self):
        observed = []
        job = SimpleNamespace(state={})

        @contextmanager
        def service_scope(group, state, *, item):
            observed.append([group, None])
            yield

        stage = function(
            "ruinedfooocus/app/async_worker.py",
            "image_stage",
            {
                "wraps": wraps,
                "current_job": lambda: None,
                "control_tools": self.controls,
                "shared": SimpleNamespace(state={}),
                "service_scope": service_scope,
                "cleanup_interrupt": lambda item: None,
            },
        )
        # Observe native_active inside the operation, before the stage clears it.
        run = stage(lambda data: observed[-1].__setitem__(1, data["_job"].state["native_active"]))
        with patch(
            "ruinedfooocus.features.prompting.local_model.prompt_preparation_scope",
            side_effect=lambda *args: nullcontext(),
        ):
            for raw, imported in (("rembg", "img2img"), ("img2img", "rembg")):
                run(
                    dict(
                        _job=job,
                        task_type="process",
                        controlnet={"type": raw},
                        prompt=json.dumps({"prompt": "restored", "controlnet": {"type": imported}}),
                    )
                )
        self.assertEqual(observed, [["image", True], ["rembg", False]])
