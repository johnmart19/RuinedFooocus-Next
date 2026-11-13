"""Submission and publication contracts through real API/worker capture owners."""

from concurrent.futures import CancelledError
from contextlib import nullcontext
import copy
import json
import importlib
import random
import pathlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.app import api_runtime
from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.features.gallery import editor_session
from ruinedfooocus.features.models.effective_image_plan import KEY, get_effective_image_plan
from ruinedfooocus.core.paths import resource_path
from test_api_image_edits import queued_data
from ui_test_helpers import function_from


WORKER = "ruinedfooocus/app/async_worker.py"


class EffectivePlanCallerTests(unittest.TestCase):
    def test_api_captures_manual_recipe_components_and_exact_target_before_assets(self):
        observed = []
        image_preparation = importlib.import_module(
            "ruinedfooocus.features.generation.image_preparation"
        )

        original = image_preparation.capture_image_assets

        def capture(data, *args, **kwargs):
            observed.append(get_effective_image_plan(data))
            return original(data, *args, **kwargs)

        with patch.object(image_preparation, "capture_image_assets", side_effect=capture):
            data = queued_data(
                "SDXL",
                payload_options={
                    "input_image": None,
                    "size": "1919x1079",
                    "negative_prompt": "No text; preserve the frame.",
                    "steps": 13,
                    "cfg": 3.5,
                    "seed": 123,
                },
            )
        plan = get_effective_image_plan(data)
        self.assertEqual(observed[-1].metadata(), plan.metadata())
        self.assertEqual(plan.recipe.sampling["custom_steps"], 13)
        self.assertEqual(plan.recipe.sampling["cfg"], 3.5)
        self.assertEqual((plan.size.width, plan.size.height), (1919, 1079))
        self.assertEqual(data["negative"], "No text; preserve the frame.")
        self.assertEqual(data["seed"], 123)

    def test_negative_lora_and_geometry_changes_cannot_reuse_a_captured_plan(self):
        data = queued_data("SDXL", payload_options={"negative_prompt": "No letters."})
        plan = get_effective_image_plan(data)
        # The inert caller's temporary file is gone; keep identity checks separate.
        with patch.object(type(plan), "verify_files"):
            plan.verify_request(data)
            for changes in (
                {"negative": "Different exclusion"},
                {"loras": [("", "0.7 - changed.safetensors")]},
                {"custom_width": 777},
                {"controlnet": {"type": "img2img", "denoise": 0.81}},
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "changed"):
                    plan.verify_request(dict(data, **changes))

    def test_worker_keeps_a_snapshot_of_manual_size_and_profile_values(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "custom.safetensors"
            header = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
            checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
            models = SimpleNamespace(
                offline=True,
                get_models_by_path=lambda *a, **k: {"baseModel": "SDXL"},
                get_model_base=lambda row: row["baseModel"],
                get_file=lambda *a: checkpoint,
            )
            resolutions = SimpleNamespace(
                CUSTOM_RESOLUTION="Custom...", aspect_ratios={"Manual landscape": (711, 513)}
            )
            values = {"meta_comment": "captured", "image_engine": "auto"}
            performance = SimpleNamespace(apply=lambda data, family: dict(data))
            manager = SimpleNamespace(
                paths={},
                DOWNLOADABLE_FILES={},
                get_abspath=Path,
                get_model_folders=lambda category: [Path(folder)],
            )
            scope = dict(
                shared=SimpleNamespace(
                    models=models,
                    path_manager=manager,
                    resolution_settings=resolutions,
                    performance_settings=performance,
                ),
                settings=SimpleNamespace(default_settings=values),
                control_tools=SimpleNamespace(
                    get_settings=lambda data: data.get("controlnet") or {}
                ),
                validate_editor_data=lambda data, **kwargs: None,
            )
            capture = function_from(WORKER, "capture_image_data", scope)
            request = dict(
                task_type="tool_call",
                base_model_name=checkpoint.name,
                prompt="Exact framing",
                negative="No words",
                controlnet={},
                aspect_ratios_selection="Manual landscape",
                performance_selection="Own settings",
                custom_steps=12,
                cfg=3.5,
                sampler_name="euler",
                scheduler="normal",
                clip_skip=1,
                loras=[],
            )
            with (
                patch(
                    "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                    side_effect=lambda data, *a, **k: data,
                ),
                patch(
                    "ruinedfooocus.features.prompting.local_model.capture_prompt_preparation",
                    side_effect=lambda data, *a, **k: data,
                ),
                patch(
                    "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
                    side_effect=lambda data, *a, **k: data,
                ),
            ):
                result = capture(request)
            values["meta_comment"] = "later edit"
            resolutions.aspect_ratios["Manual landscape"] = (2048, 2048)
            self.assertEqual(result["_image_capture_settings"]["meta_comment"], "captured")
            self.assertEqual(
                result["_image_capture_resolutions"].aspect_ratios["Manual landscape"], (711, 513)
            )
            self.assertEqual((result[KEY].size.width, result[KEY].size.height), (711, 513))
            self.assertNotIn(KEY, request)


class OwnedEditorPublicationTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = editor_session.EditorSessionStore(self.folder / "sessions")
        self.owner = ("ui", "integration-owner")
        self.source = self.folder / "source.png"
        Image.new("RGBA", (17, 23), (3, 7, 11, 80)).save(self.source)
        self.snapshot = self.store.start(self.source, owner=self.owner)
        self.registry = life.JobRegistry()
        self.item = self.registry.create(owner=self.owner)
        self.enterContext(life.job_scope(self.item))
        self.enterContext(patch.object(editor_session, "default_store", return_value=self.store))
        self.enterContext(patch.object(life, "registry", self.registry))
        self.scope = dict(
            registry=self.registry,
            current_job=life.current_job,
            json=json,
            check_interrupt=lambda data: self.item.check(),
            api_jobs={},
        )
        for name in (
            "validate_editor_data",
            "stage_editor_result",
            "publish_editor_result",
            "discard_editor_result",
            "add_result",
            "editor_result",
        ):
            function_from(WORKER, name, self.scope)

    def request(self, operation="fixture-process"):
        context = dict(
            session_id=self.snapshot.session_id,
            revision=self.snapshot.revision,
            source_id=self.snapshot.current.artifact_id,
        )
        return editor_session.capture_request(
            {"prompt": "Keep the border", "image_number": 1, "_sampling_captured": False},
            context,
            operation,
            owner=self.owner,
            store=self.store,
        )

    def output(self):
        path = self.folder / "result.png"
        info = PngInfo()
        info.add_text(
            "parameters", json.dumps({"software": "RuinedFooocus", "width": 17, "height": 23})
        )
        Image.new("RGBA", (17, 23), (5, 9, 13, 80)).save(path, pnginfo=info)
        return str(path)

    def test_editor_context_resolves_owned_pixels_and_rejects_an_accompanying_upload(self):
        context = dict(
            session_id=self.snapshot.session_id,
            revision=self.snapshot.revision,
            source_id=self.snapshot.current.artifact_id,
        )
        bound = api_runtime.bind_editor_source({}, context, owner=self.owner)
        self.assertEqual(bound["input_image"].size, (17, 23))
        self.assertEqual(bound["input_image"].getpixel((0, 0)), (3, 7, 11, 80))
        with self.assertRaisesRegex(ValueError, "own image"):
            api_runtime._editor_payload(
                {"editor_context": context, "input_image": "foreign upload"}, owner=self.owner
            )
        with self.assertRaises(PermissionError):
            api_runtime.bind_editor_source({}, context, owner=("ui", "other-owner"))

    def test_result_is_only_staged_until_the_existing_terminal_publication(self):
        data = self.request()
        self.scope["validate_editor_data"](data, start=True)
        result = self.output()
        self.scope["stage_editor_result"](data, [result])
        self.assertEqual(
            self.store.snapshot(self.snapshot.session_id, owner=self.owner).revision,
            self.snapshot.revision,
        )
        self.scope["add_result"](self.item.task_id, "results", [result])
        current = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(current.revision, self.snapshot.revision + 1)
        self.assertEqual(len(current.steps), 1)
        self.assertEqual(current.steps[0]["execution"]["kind"], "processing")
        self.scope["add_result"](self.item.task_id, "results", [result])
        self.assertEqual(len(self.item.events), 1)
        self.assertEqual(self.scope["editor_result"](self.item.task_id)["status"], "completed")

    def test_cancelled_and_replaced_results_do_not_append_history(self):
        data = self.request()
        self.scope["validate_editor_data"](data, start=True)
        self.scope["stage_editor_result"](data, [self.output()])
        self.registry.cancel(self.item.task_id, owner=self.owner)
        with self.assertRaises(CancelledError):
            self.scope["publish_editor_result"](self.item)
        self.scope["discard_editor_result"](self.item, status="cancelled")
        self.assertEqual(self.store.snapshot(self.snapshot.session_id, owner=self.owner).steps, ())
        self.assertEqual(self.scope["editor_result"](self.item.task_id)["status"], "cancelled")

    def test_native_background_output_publishes_without_diffusion_metadata_or_lookup(self):
        source_generation = {
            "prompt": "Original generation",
            "base_model_name": "original",
            "seed": 17,
        }
        self.snapshot = self.store.start(
            self.source, provenance=source_generation, owner=self.owner
        )
        output = self.folder / "native-cutout.png"
        pixels = Image.new("RGBA", (17, 23), (5, 9, 13, 80))
        forbidden = Mock(side_effect=AssertionError("Utility used diffusion resources"))
        pipeline = SimpleNamespace(
            pipeline_type=["sdxl"],
            parse_gen_data=lambda data: data,
            load_base_model=Mock(),
            process=Mock(return_value=[pixels]),
        )
        data = self.request(operation="remove-background")
        data.update(
            task_type="tool_call",
            task_id=self.item.task_id,
            _job=self.item,
            silent=True,
            input_image=Image.open(self.source).copy(),
            controlnet={"type": "rembg"},
            rembg_native_model="owned-background.safetensors",
            native_background_source={
                "model": "owned-background.safetensors",
                "runtime": "inert CPU fixture",
            },
        )
        self.assertNotIn("base_model_name", data)
        self.assertNotIn("base_model_hash", data)
        process = function_from(
            WORKER,
            "_process",
            dict(
                self.scope,
                shared=SimpleNamespace(
                    state={},
                    shared_cache={},
                    models=SimpleNamespace(get_file=forbidden),
                    path_manager=SimpleNamespace(model_paths={"temp_outputs_path": self.folder}),
                ),
                settings=SimpleNamespace(default_settings={}),
                control_tools=SimpleNamespace(
                    get_settings=lambda value: value.get("controlnet") or {}
                ),
                pipelines=SimpleNamespace(update=lambda value: pipeline),
                process_metadata=forbidden,
                process_prompt=forbidden,
                parse_loras=forbidden,
                get_checkpoint_hashes=forbidden,
                get_lora_hashes=forbidden,
                generate_temp_filename=lambda **kwargs: str(output),
                TimeIt=nullcontext,
                resource_path=resource_path,
                random=random,
                Path=Path,
                pathlib=pathlib,
                Image=Image,
                PngInfo=PngInfo,
                model_management=SimpleNamespace(InterruptProcessingException=InterruptedError),
            ),
        )
        self.scope["validate_editor_data"](data, start=True)
        outputs = process(data)
        self.scope["stage_editor_result"](data, outputs)
        self.assertEqual(
            self.store.snapshot(self.snapshot.session_id, owner=self.owner).revision,
            self.snapshot.revision,
        )
        self.scope["add_result"](self.item.task_id, "results", outputs)
        current = self.store.snapshot(self.snapshot.session_id, owner=self.owner)
        self.assertEqual(current.revision, self.snapshot.revision + 1)
        self.assertEqual(current.source_generation, source_generation)
        self.assertEqual(len(current.steps), 1)
        execution = current.steps[0]["execution"]
        self.assertEqual(execution["kind"], "processing")
        self.assertEqual(execution["native_background_source"], data["native_background_source"])
        for key in (
            "base_model_name",
            "base_model_hash",
            "cfg",
            "sampler_name",
            "scheduler",
            "steps",
            "loras",
        ):
            self.assertNotIn(key, execution)
        with Image.open(output) as result:
            self.assertEqual(
                (result.mode, result.size, result.getpixel((0, 0))),
                ("RGBA", (17, 23), (5, 9, 13, 80)),
            )
            saved = json.loads(result.info["parameters"])
            self.assertNotIn("base_model_hash", saved)
            self.assertNotIn("base_model_name", saved)
        pipeline.load_base_model.assert_called_once_with("owned-background.safetensors", hash=None)
        pipeline.process.assert_called_once()
        forbidden.assert_not_called()
        self.assertEqual(self.scope["editor_result"](self.item.task_id)["status"], "completed")


class APIWorkerFailureTests(unittest.TestCase):
    def test_actual_pipeline_exception_has_exactly_one_api_terminal(self):
        pipeline = SimpleNamespace(
            parse_gen_data=lambda data: data,
            load_base_model=Mock(),
            process=Mock(side_effect=RuntimeError("inert native failure")),
        )
        selected = Mock(return_value=pipeline)
        scope = dict(
            check_interrupt=lambda data: None,
            control_tools=SimpleNamespace(get_settings=lambda data: data.get("controlnet") or {}),
            shared=SimpleNamespace(state={}, models=Mock()),
            pipelines=SimpleNamespace(update=selected),
            TimeIt=nullcontext,
            resource_path=resource_path,
            random=random,
            model_management=SimpleNamespace(InterruptProcessingException=InterruptedError),
            traceback=SimpleNamespace(print_exc=Mock()),
            describe_error=str,
        )
        process = function_from(WORKER, "_process", scope)

        def run(job, task_id):
            return process(
                dict(
                    _api_job=job,
                    task_type="tool_call",
                    task_id=task_id,
                    silent=True,
                    _bound_image_tool=True,
                    prompt="#! untrusted model instruction",
                    controlnet={"type": "upscale", "upscaler": "inert-upscaler"},
                    input_image=Image.new("RGB", (64, 48)),
                    image_number=1,
                    _sampling_captured=False,
                    loras=[],
                    seed=1,
                )
            )

        job = api_runtime.Job(run)
        job.run(1)
        self.assertTrue(job.finished)
        self.assertEqual(list(job.events.queue), [("error", "inert native failure")])
        pipeline.process.assert_called_once()
        self.assertEqual(selected.call_args.args[0]["prompt"], "")
        self.assertEqual(
            pipeline.process.call_args.kwargs["gen_data"]["prompt"],
            "#! untrusted model instruction",
        )
        self.assertFalse(job._terminal("result", []))


if __name__ == "__main__":
    unittest.main()
