from ruinedfooocus.core.paths import asset_path, resource_path

"""Synthetic checks: never load models or execute local command handlers."""

import ast
import queue
import threading
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ruinedfooocus.app as app_modules
import ruinedfooocus.features.generation as generation_modules
import ruinedfooocus.app.api_runtime as api_runtime


ROOT = Path(__file__).resolve().parents[1]


def isolated_function(filename, name, namespace):
    # The GPU worker and pipeline registry import/start application services.
    # Execute their real function bodies against inert dependencies instead.
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class PromptBoundaryTests(unittest.TestCase):
    def select_pipeline(self, data):
        factories = {
            name: SimpleNamespace(pipeline=lambda name=name: SimpleNamespace(pipeline_type=[name]))
            for name in (
                "image_pipeline",
                "hashbang_pipeline",
                "search_pipeline",
                "template_pipeline",
            )
        }
        models = SimpleNamespace(
            get_file=lambda *args: Path("synthetic.safetensors"),
            get_models_by_path=lambda *args: None,
            get_model_base=lambda *args: "SDXL",
        )
        namespace = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            **factories,
            "state": {"pipeline": None},
            "Path": Path,
            "shared": SimpleNamespace(models=models),
            "NoPipeLine": lambda: SimpleNamespace(pipeline_type=[]),
            "controlnet": SimpleNamespace(get_settings=lambda data: {}),
        }
        isolated_function(
            "ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", namespace
        )
        update = isolated_function(
            "ruinedfooocus/features/generation/pipelines.py", "update", namespace
        )
        return update(dict(base_model_name="synthetic.safetensors", **data)).pipeline_type[0]

    def test_tool_and_api_prompts_cannot_select_command_handlers(self):
        for prompt in ("#!clean cache", "#!branch\nsynthetic", "search: all:", "ruinedfooocuslogo"):
            for source in (
                {"task_type": "tool_call"},
                {"task_type": "external_api", "_api_job": object()},
            ):
                with self.subTest(prompt=prompt, source=source):
                    self.assertEqual(
                        self.select_pipeline(dict(prompt=prompt, **source)), "image_pipeline"
                    )

    def test_main_commands_still_select_their_handlers(self):
        for prompt, expected in (
            ("#!help", "hashbang_pipeline"),
            ("search: synthetic", "search_pipeline"),
            ("ruinedfooocuslogo", "template_pipeline"),
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    self.select_pipeline({"task_type": "process", "prompt": prompt}), expected
                )

    def test_tool_and_api_json_prompts_cannot_override_generation_settings(self):
        from ruinedfooocus.features.image_editing.image_guidance import FIELD_SCHEMA

        class ReachedPipeline(Exception):
            pass

        captured = []

        def stop_at_pipeline(data):
            captured.append(dict(data))
            raise ReachedPipeline

        parse_metadata = isolated_function(
            "ruinedfooocus/features/prompting/prompt_processing.py",
            "process_metadata",
            {"json": __import__("json")},
        )
        metadata = Mock(side_effect=parse_metadata)
        shared = SimpleNamespace(
            models=SimpleNamespace(
                get_models_by_path=lambda *args: None, get_model_base=lambda *args: "SDXL"
            ),
            performance_settings=SimpleNamespace(apply=lambda data, family: data),
        )
        process = isolated_function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            {
                "check_interrupt": lambda data: None,
                "shared": shared,
                "process_metadata": metadata,
                "control_tools": SimpleNamespace(
                    get_settings=lambda data: data.get("controlnet") or {}
                ),
                "pipelines": SimpleNamespace(update=stop_at_pipeline),
            },
        )
        with patch.dict(
            sys.modules,
            {
                "ruinedfooocus.app.state": shared,
                "ruinedfooocus.features.image_editing.image_guidance": SimpleNamespace(
                    FIELD_SCHEMA=FIELD_SCHEMA, restore_guidance_metadata=lambda *args, **kwargs: {}
                ),
            },
        ):
            for source in (
                {"task_type": "tool_call"},
                {"_api_job": object()},
                {"task_type": "process"},
            ):
                with self.subTest(source=source), self.assertRaises(ReachedPipeline):
                    process(dict(prompt='{"prompt":"replacement","cfg":99}', cfg=7, **source))
        self.assertEqual([data["cfg"] for data in captured], [7, 7, 99])
        self.assertEqual(metadata.call_count, 1)


class LegacyApiTests(unittest.TestCase):
    def test_legacy_endpoints_use_shared_runtime(self):
        endpoints = {}
        backend = SimpleNamespace(
            configured_chat_model=Mock(return_value=("chat/default", {"kind": "chat"})),
            chat=Mock(return_value="chat job"),
            legacy_images=Mock(return_value="image job"),
            wait_for_result=Mock(
                side_effect=[{"choices": [{"message": {"content": "reply"}}]}, ["result.png"]]
            ),
        )
        register = isolated_function(
            "ruinedfooocus/app/api.py",
            "add_api",
            {
                "gr": SimpleNamespace(
                    api=lambda function, api_name: endpoints.update({api_name: function})
                ),
                "settings": SimpleNamespace(default_settings={}),
                "get_last_image": lambda: "image.png",
            },
        )
        with patch.object(app_modules, "api_runtime", backend):
            register()
        self.assertEqual(endpoints["llama"]("system", "question"), "reply")
        self.assertEqual(
            endpoints["generate_image"]("scene"),
            {"path": "result.png", "meta": {"_type": "gradio.FileData"}},
        )
        backend.legacy_images.assert_called_once_with("scene")
        self.assertEqual(
            backend.chat.call_args.args[0]["messages"],
            [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
        )

    def test_legacy_first_use_download_precedes_compute_admission(self):
        models = SimpleNamespace(
            get_model_path=Mock(return_value=Path("downloaded.safetensors")),
            get_models_by_path=Mock(return_value={}),
            detect_checkpoint_base=Mock(return_value="SDXL"),
            get_model_base=Mock(return_value="Unknown"),
        )
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={"base_model": "downloadable"}), models=models
        )
        operation = Mock(return_value=["result.png"])
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(app_modules, "state", shared, create=True),
            patch.object(api_runtime, "image_defaults", {}),
            patch.object(api_runtime, "submit", side_effect=lambda run: run) as submit,
            patch.object(api_runtime, "_image_operation", return_value=operation) as prepare,
        ):
            run = api_runtime.legacy_images("scene")
            models.get_model_path.assert_not_called()
            job = SimpleNamespace(cancelled=threading.Event())
            with self.assertRaisesRegex(RuntimeError, "preparation"):
                run(job, 42)
            run.prepare(job)
            self.assertEqual(run(job, 42), ["result.png"])
        submit.assert_called_once()
        models.get_model_path.assert_called_once_with("checkpoints", "downloadable")
        prepare.assert_called_once_with(
            {"prompt": "scene"}, {"name": "downloadable", "family": "SDXL"}, {}
        )
        operation.assert_called_once_with(job, 42)

    def test_legacy_chat_keeps_configured_or_default_undownloaded_model(self):
        shared = SimpleNamespace(settings=SimpleNamespace(default_settings={}))
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "ruinedfooocus.features.chat.llama_models": SimpleNamespace(
                        DEFAULT_MODEL="default.gguf"
                    ),
                },
            ),
            patch.object(app_modules, "state", shared, create=True),
        ):
            self.assertEqual(
                api_runtime.configured_chat_model(), ("chat/default.gguf", {"path": "default.gguf"})
            )
            shared.settings.default_settings["llama_localfile"] = "custom.gguf"
            self.assertEqual(
                api_runtime.configured_chat_model(), ("chat/custom.gguf", {"path": "custom.gguf"})
            )

    def test_submission_uses_serial_worker_and_releases_slot_after_failure(self):
        worker = SimpleNamespace(add_task=Mock())
        with patch.object(app_modules, "async_worker", worker, create=True):
            job = api_runtime.submit(lambda job, task_id: task_id)
        worker.add_task.assert_called_once_with({"task_type": "external_api", "_api_job": job})
        job.run(42)
        self.assertEqual(api_runtime.wait_for_result(job), 42)
        worker.add_task.side_effect = RuntimeError("queue failed")
        slots = threading.BoundedSemaphore(1)
        with (
            patch.object(api_runtime, "slots", slots),
            patch.object(app_modules, "async_worker", worker, create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "queue failed"):
                api_runtime.submit(lambda *args: None)
            self.assertTrue(slots.acquire(blocking=False))
            slots.release()

    def test_synchronous_wait_returns_results_and_releases_consumer(self):
        job = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event())
        job.events.put(("chunk", "partial"))
        job.events.put(("result", ["image.png"]))
        self.assertEqual(api_runtime.wait_for_result(job), ["image.png"])
        self.assertTrue(job.cancelled.is_set())

    def test_synchronous_wait_propagates_failures_and_cancels(self):
        for kind, error_type in (("error", RuntimeError), ("invalid_request", ValueError)):
            with self.subTest(kind=kind):
                job = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event())
                job.events.put((kind, "failure"))
                with self.assertRaisesRegex(error_type, "failure"):
                    api_runtime.wait_for_result(job)
                self.assertTrue(job.cancelled.is_set())
        job = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event())
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            api_runtime.wait_for_result(job, timeout=0)
        self.assertTrue(job.cancelled.is_set())


class CustomVisionTests(unittest.TestCase):
    def test_api_reuse_checks_captured_settings_after_live_settings_change(self):
        live = {"llm_n_ctx": 16384}
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings=live),
            path_manager=SimpleNamespace(DOWNLOADABLE_FILES={}),
        )
        pipeline = SimpleNamespace(
            llm=object(),
            model_settings=8192,
            vision_projector=None,
            current_model_settings=lambda model, *, vision, values=None: (
                live if values is None else values
            )["llm_n_ctx"],
            load_base_model=Mock(),
        )
        registry = SimpleNamespace(update=lambda data: pipeline)
        memory = SimpleNamespace(unload_all_models=Mock(), soft_empty_cache=Mock())
        with tempfile.TemporaryDirectory() as temporary:
            weight = Path(temporary) / "custom.gguf"
            weight.write_bytes(b"fixture weights")
            with (
                patch.dict(
                    sys.modules,
                    {
                        "ruinedfooocus.app.state": shared,
                        "ruinedfooocus.features.generation.pipelines": registry,
                        "comfy": SimpleNamespace(model_management=memory),
                        "comfy.model_management": memory,
                    },
                ),
                patch.object(generation_modules, "pipelines", registry, create=True),
                patch.object(app_modules, "state", shared, create=True),
                patch("ruinedfooocus.features.chat.llama_models.local_models", return_value={}),
                patch.object(api_runtime, "submit", side_effect=lambda operation: operation),
                patch.object(api_runtime, "chat_completion", return_value="completed"),
            ):
                operation = api_runtime.chat(
                    {
                        "model": "chat/custom.gguf",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    {"path": str(weight)},
                )
                live["llm_n_ctx"] = 8192
                self.assertEqual(operation(object(), 1), "completed")
        pipeline.load_base_model.assert_called_once()
        self.assertEqual(
            pipeline.load_base_model.call_args.kwargs["prepared"].settings["llm_n_ctx"], 16384
        )

    def run_chat(self, projector, has_images=True):
        pipeline = SimpleNamespace(
            llm=None,
            model_settings=None,
            vision_projector=None,
            current_model_settings=lambda model, **kwargs: (model, kwargs.get("vision", False)),
            unload=Mock(),
        )

        def load(model, *, vision=False, prepared=None, **load_options):
            self.assertEqual(str(prepared.model), model)
            self.assertEqual(prepared.vision, vision)
            self.assertTrue(prepared.present)
            pipeline.llm = object()
            pipeline.vision_projector = projector if vision else None

        pipeline.load_base_model = Mock(side_effect=load)
        registry = SimpleNamespace(update=lambda data: pipeline)
        memory = SimpleNamespace(unload_all_models=Mock(), soft_empty_cache=Mock())
        payload = {
            "model": "chat/custom.gguf",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,synthetic"},
                        }
                    ],
                }
            ],
        }
        if not has_images:
            payload["messages"] = [{"role": "user", "content": "A text-only question."}]
        with tempfile.TemporaryDirectory() as temporary:
            weight = Path(temporary) / "custom.gguf"
            weight.write_bytes(b"fixture weights")
            companion = Path(temporary) / "custom-mmproj.gguf"
            companion.write_bytes(b"fixture projector")
            shared = SimpleNamespace(
                settings=SimpleNamespace(
                    default_settings={"llama_mmproj": str(companion) if projector else "Auto"}
                ),
                path_manager=SimpleNamespace(DOWNLOADABLE_FILES={}),
            )
            self._run_captured_chat(payload, weight, has_images, shared, pipeline, registry, memory)
        return "completed"

    def _run_captured_chat(self, payload, weight, has_images, shared, pipeline, registry, memory):
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "ruinedfooocus.features.generation.pipelines": registry,
                    "comfy": SimpleNamespace(model_management=memory),
                    "comfy.model_management": memory,
                },
            ),
            patch.object(generation_modules, "pipelines", registry, create=True),
            patch.object(app_modules, "state", shared, create=True),
            patch("ruinedfooocus.features.chat.llama_models.local_models", return_value={}),
            patch.object(
                api_runtime, "submit", side_effect=lambda operation: operation(object(), 1)
            ),
            patch.object(api_runtime, "chat_completion", return_value="completed") as completion,
        ):
            result = api_runtime.chat(payload, {"path": str(weight)})
        self.assertEqual(result, "completed")
        pipeline.load_base_model.assert_called_once()
        self.assertEqual(pipeline.load_base_model.call_args.args, (str(weight.resolve()),))
        self.assertEqual(pipeline.load_base_model.call_args.kwargs["vision"], has_images)
        prepared = pipeline.load_base_model.call_args.kwargs["prepared"]
        self.assertEqual(prepared.model, weight.resolve())
        self.assertEqual(
            prepared.projector is not None, has_images and pipeline.vision_projector is not None
        )
        completion.assert_called_once()

    def test_imported_model_with_configured_projector_can_chat(self):
        self.assertEqual(self.run_chat("custom-mmproj.gguf"), "completed")

    def test_text_api_does_not_load_the_configured_projector(self):
        self.assertEqual(self.run_chat("custom-mmproj.gguf", has_images=False), "completed")

    def test_image_chat_without_projector_still_fails(self):
        with self.assertRaisesRegex(ValueError, "matching projector"):
            self.run_chat(None)


if __name__ == "__main__":
    unittest.main()
