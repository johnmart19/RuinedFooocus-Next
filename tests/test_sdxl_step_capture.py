"""Staged SDXL freezes the same effective step count as the worker, without models."""

import ast
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio  # Load transport dependencies before the API fixture's module snapshot.

from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.generation import sdxl_refiner as refiner
from test_image_preparation import Manager, function
from test_model_cache import method
from test_native_variant_defaults import write_header
import test_api_native_inpaint as api_fixture

ROOT = Path(__file__).resolve().parents[1]


def worker_steps(data):
    """Execute the actual worker's final normalization, without loading a pipeline."""
    tree = ast.parse((ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
    process = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_process"
    )
    index = next(
        index
        for index, node in enumerate(process.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "steps" for target in node.targets)
    )
    scope = dict(gen_data=dict(data), sampling=True)
    exec(
        compile(
            ast.Module(body=process.body[index : index + 2], type_ignores=[]),
            "actual worker step normalization",
            "exec",
        ),
        scope,
    )
    return scope["gen_data"]


class SdxlStepCaptureTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        (self.root / "checkpoints").mkdir()
        self.base = write_header(
            self.root / "checkpoints/base.safetensors",
            {
                "conditioner.embedders.0.fixture": [1],
                "conditioner.embedders.1.fixture": [1],
                "first_stage_model.fixture": [1],
            },
        )
        self.refiner = write_header(
            self.root / "checkpoints/refiner.safetensors", {"conditioner.embedders.0.fixture": [1]}
        )
        families = {self.base.name: "SDXL", self.refiner.name: "SDXLRefiner"}
        self.manager = Manager(self.root)
        self.manager.model_paths = {"embedding_path": [], "cache_path": self.root}
        self.manager.EXTENSIONS = (".safetensors",)
        self.models = NS(
            offline=True,
            get_file=lambda category, name: (
                self.root / category / name
                if name and (self.root / category / name).is_file()
                else None
            ),
            detect_checkpoint_base=lambda path: families.get(Path(path).name),
            get_models_by_path=lambda category, name, **kwargs: {
                "baseModel": families.get(Path(name).name)
            },
            get_model_base=lambda info: info["baseModel"],
        )
        self.api = api_fixture.NativeInpaintApiTests()
        self.api.setUp()
        self.addCleanup(self.api.doCleanups)
        self.api.selected.update(name=self.base.name, path=str(self.base), family="SDXL")
        self.shared = sys.modules["ruinedfooocus.app.state"]
        self.shared.models, self.shared.path_manager = self.models, self.manager
        self.shared.resolution_settings.aspect_ratios = {}
        self.enterContext(
            patch(
                "ruinedfooocus.features.generation.sdcpp_pipeline.initialized_native_gpu",
                return_value=False,
            )
        )
        recipe = dict(custom_steps=50, cfg=8, sampler_name="euler", scheduler="simple", clip_skip=1)
        performance = self.shared.performance_settings
        performance.default_settings = dict(recipe)
        performance.get_perf_options = lambda *args: dict(recipe)
        performance.choices_for_model = lambda *args: ["Own settings", "Model Defaults"]
        performance.selection_for_model = lambda family, name, current: current or "Own settings"
        apply = method(
            "ruinedfooocus/features/models/performance.py",
            "apply",
            {"MODEL_DEFAULT_SELECTIONS": ("Model Defaults",)},
        )
        performance.apply = MethodType(apply, performance)

    def defaults(self):
        return dict(
            base_model_name=self.base.name,
            performance_selection="Own settings",
            custom_steps=30,
            cfg=3,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=2,
        )

    def post(self, **overrides):
        payload = dict(
            prompt="saved literal scene",
            input_image=api_fixture.inline(self.api.source),
            size="832x1216",
            sdxl_refiner_model=self.refiner.name,
            sdxl_refiner_switch_step=24,
        )
        with patch.object(api_runtime, "image_defaults", self.defaults()):
            return self.api.client.post("/images/generations", json=payload | overrides)

    def test_saved_api_30_steps_and_switch24_reach_capture_without_step_override(self):
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        captured = self.api.prepared[-1]
        self.assertEqual(captured["custom_steps"], 30)
        self.assertEqual(captured[refiner.KEY].steps, 30)
        self.assertEqual(captured[refiner.KEY].switch_step, 24)
        self.assertEqual(captured["cfg"], 3)
        self.assertEqual(captured["clip_skip"], 2)
        normalized = worker_steps(captured)
        self.assertEqual(normalized["steps"], captured[refiner.KEY].steps)
        self.assertEqual(refiner.settings(normalized), (30, 24, None))

    def test_explicit_api_steps_win_saved_controls_and_invalid_switch_is_not_silently_moved(self):
        response = self.post(steps=28)
        self.assertEqual(response.status_code, 200, response.text)
        captured = self.api.prepared[-1]
        self.assertEqual(captured["custom_steps"], 28)
        self.assertEqual(captured[refiner.KEY].steps, 28)
        self.assertEqual(captured[refiner.KEY].switch_step, 24)
        self.api.submit.reset_mock()
        response = self.post(steps=20)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("switch step", response.text)
        self.api.submit.assert_not_called()

    def test_ui_capture_uses_actual_performance_precedence_before_freezing_refiner(self):
        scope = dict(
            shared=self.shared,
            settings=self.shared.settings,
            control_tools=NS(get_settings=lambda data: data.get("controlnet") or {}),
        )
        capture = function("capture_image_data", scope)
        for selection, expected in (("Own settings", 30), ("Model Defaults", 50)):
            with self.subTest(selection=selection):
                data = dict(
                    self.defaults(),
                    task_type="process",
                    prompt="literal scene",
                    performance_selection=selection,
                    steps=4,
                    custom_width=512,
                    custom_height=512,
                    sdxl_refiner_model=self.refiner.name,
                    sdxl_refiner_switch_step=24,
                )
                before = dict(data)
                captured = capture(data)
                self.assertEqual(data, before)
                self.assertEqual(captured["custom_steps"], expected)
                self.assertEqual(captured[refiner.KEY].steps, expected)
                self.assertEqual(worker_steps(captured)["steps"], expected)

    def test_canonical_custom_steps_win_stale_alias_and_invalid_control_cannot_fall_back(self):
        data = dict(
            base_model_name=self.base.name,
            sdxl_refiner_model=self.refiner.name,
            sdxl_refiner_switch_step=24,
            custom_steps=30,
            steps=4,
        )
        captured, _ = refiner.capture_assets(data, self.manager, self.models)
        self.assertEqual(captured[refiner.KEY].steps, 30)
        self.assertEqual(data["steps"], 4)
        self.assertEqual(data["custom_steps"], 30)
        for invalid in (None, True, 1, 201, 30.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "2 through 200"):
                refiner.settings(data | dict(custom_steps=invalid, steps=30))
        self.assertEqual(
            refiner.settings(dict(steps=30, sdxl_refiner_switch_step=24)), (30, 24, None)
        )


if __name__ == "__main__":
    unittest.main()
