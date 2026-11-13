"""Native output counts and completion proofs stay local to the captured request."""

from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import pathlib
import random
import tempfile
import traceback
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.core.task_errors import describe_error
from ruinedfooocus.features.generation.image_output_batches import (
    PLAN_KEY,
    OutputPlan,
    ImageOutputBatch,
    ImageBatchArtifacts,
    validate_artifacts,
    with_output_plan,
)
from test_image_preparation import function


class OutputBatchTests(unittest.TestCase):
    def test_plan_separates_repetitions_from_fixed_and_variable_native_counts(self):
        data = with_output_plan(dict(image_number=2), OutputPlan("qwen-layers", 3, 3))
        result = ImageBatchArtifacts([f"layer-{n}.png" for n in range(6)], data[PLAN_KEY], 2)
        self.assertIs(validate_artifacts(data, result), result)
        with self.assertRaises(RuntimeError):
            validate_artifacts(data, ["one.png", "two.png"])
        with self.assertRaises(ValueError):
            ImageOutputBatch((1, 2), data[PLAN_KEY])
        with self.assertRaises(ValueError):
            OutputPlan("unbounded", 0, 65)

    def test_saved_or_forged_completion_and_old_request_proofs_cannot_admit_empty_failure(self):
        from pydantic import ValidationError
        from ruinedfooocus.integrations.openai_api import ImageRequest

        with self.assertRaises(ValidationError):
            ImageRequest(prompt="x", **{PLAN_KEY: {"kind": "sam-instances", "completed": True}})
        first = with_output_plan(dict(image_number=1), OutputPlan("sam-instances", 0, 64))
        previous = ImageBatchArtifacts([], first[PLAN_KEY], 1)
        second = with_output_plan(
            dict(first, native_output_group={"kind": "sam-instances", "count": 0}), first[PLAN_KEY]
        )
        self.assertNotEqual(first[PLAN_KEY], second[PLAN_KEY])
        for result in (None, [], previous):
            with self.subTest(result=type(result).__name__), self.assertRaises(RuntimeError):
                validate_artifacts(second, result)
        with self.assertRaises(RuntimeError):
            validate_artifacts(
                dict(image_number=1, **{PLAN_KEY: {"kind": "sam-instances", "completed": True}}), []
            )
        with self.assertRaises(RuntimeError):
            validate_artifacts(dict(image_number=1), [])
        completed = ImageBatchArtifacts([], second[PLAN_KEY], 1)
        self.assertIs(validate_artifacts(second, completed), completed)

    def test_batch_copies_metadata_and_rejects_incomplete_per_image_records(self):
        metadata = {"instance_index": 0, "source": {"sha256": "abc"}}
        plan = OutputPlan("sam-instances", 0, 4)
        batch = ImageOutputBatch(("image",), plan, (metadata,))
        metadata["source"]["sha256"] = "changed"
        self.assertEqual(batch.metadata[0]["source"]["sha256"], "abc")
        with self.assertRaises(TypeError):
            batch.metadata[0]["instance_index"] = 9
        with self.assertRaises(ValueError):
            ImageOutputBatch((1, 2), plan, ({"index": 0},))

    def test_capture_strips_display_claims_and_previous_private_plans(self):
        from ruinedfooocus.features.generation.image_preparation import capture_image_assets

        original = with_output_plan(
            dict(
                prompt="#!help",
                native_output_group={"count": 0},
                native_image_output={"completed": True},
            ),
            OutputPlan("old", 0, 1),
        )
        captured = capture_image_assets(original, NS(), {})
        for key in (PLAN_KEY, "native_output_group", "native_image_output"):
            self.assertNotIn(key, captured)
            self.assertIn(key, original)

    def worker(self, folder, count, *, fail=False):
        root = Path(folder)
        plan = OutputPlan("sam-instances", 0, 4)
        source = Image.new("RGBA", (16, 8), (20, 80, 160, 128))
        data = with_output_plan(
            dict(
                input_image=source,
                controlnet={"type": "segment"},
                native_vision_model="model.safetensors",
                native_vision_source={"native": "SAM3", "name": "model.safetensors"},
                prompt="",
                style_selection=[],
                task_id=1,
                task_type="tool_call",
                silent=True,
                index=(0, 1),
                _api_job=NS(events=NS(put=Mock())),
            ),
            plan,
        )

        def process(gen_data, callback):
            if fail:
                raise RuntimeError("owned pipeline failure")
            return ImageOutputBatch(
                tuple(source.copy() for _ in range(count)),
                gen_data[PLAN_KEY],
                tuple({"instance_index": index, "instance_count": count} for index in range(count)),
            )

        pipe = NS(pipeline_type=["native-vision"], process=process)
        controls = NS(get_settings=lambda values: values.get("controlnet", {}))
        forbidden = Mock(side_effect=AssertionError("generation dependency accessed"))
        shared = NS(
            models=NS(get_models_by_path=forbidden, get_model_base=forbidden),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=object(),
            path_manager=NS(model_paths={"temp_outputs_path": root}),
            shared_cache={},
        )
        sequence = itertools.count()
        render = function(
            "_process",
            dict(
                image_stage=lambda fn: fn,
                shared=shared,
                settings=NS(default_settings={}),
                control_tools=controls,
                check_interrupt=lambda value: None,
                process_metadata=forbidden,
                pipelines=NS(update=lambda values: pipe),
                get_checkpoint_hashes=forbidden,
                process_prompt=forbidden,
                shift_attention=forbidden,
                parse_loras=forbidden,
                merge_loras=forbidden,
                TimeIt=lambda value: nullcontext(),
                resource_path=resource_path,
                random=random,
                Path=Path,
                pathlib=pathlib,
                Image=Image,
                PngInfo=PngInfo,
                json=json,
                traceback=traceback,
                model_management=NS(
                    InterruptProcessingException=type("Cancelled", (Exception,), {})
                ),
                describe_error=describe_error,
                generate_temp_filename=lambda **kwargs: str(
                    root / f"output-{next(sequence)}.{kwargs['extension']}"
                ),
            ),
        )
        result = render(data)
        forbidden.assert_not_called()
        return data, result

    def test_actual_worker_preserves_each_instance_and_records_output_index(self):
        with tempfile.TemporaryDirectory() as folder:
            data, result = self.worker(folder, 2)
            self.assertIsInstance(result, ImageBatchArtifacts)
            self.assertEqual(len(result), 2)
            validate_artifacts(data, result)
            for index, path in enumerate(result):
                with Image.open(path) as image:
                    self.assertEqual(image.mode, "RGBA")
                    metadata = json.loads(image.info["parameters"])
                self.assertEqual(
                    metadata["native_output_group"],
                    dict(kind="sam-instances", index=index, count=2, generation_index=0),
                )
                self.assertEqual(
                    metadata["native_image_output"], dict(instance_index=index, instance_count=2)
                )

    def test_actual_worker_proves_completed_zero_matches_but_never_swallowed_pipeline_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            data, result = self.worker(folder, 0)
            self.assertIsInstance(result, ImageBatchArtifacts)
            self.assertEqual(result, [])
            validate_artifacts(data, result)
        with tempfile.TemporaryDirectory() as folder:
            data, result = self.worker(folder, 0, fail=True)
            self.assertNotIsInstance(result, ImageBatchArtifacts)
            with self.assertRaises(RuntimeError):
                validate_artifacts(data, result)


if __name__ == "__main__":
    unittest.main()
