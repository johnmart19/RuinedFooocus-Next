"""Prompt-helper selection and I/O stay outside admitted compute; no models run."""

from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from functools import wraps

import ruinedfooocus.app as app_modules
from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.app import api_runtime, resource_lifecycle as life
from ruinedfooocus.features.prompting import local_model as helpers
from ui_test_helpers import function_from


class PromptHelperPreparationTests(unittest.TestCase):
    def test_actual_ui_enhancement_switch_is_the_captured_style_selection(self):
        from ruinedfooocus.features.prompting import ui_styles

        selection, control = Mock(value=[]), Mock()
        ui_styles.bind_subset(selection, control, ui_styles.NEURAL_PROCESSORS)
        choose = control.input.call_args.args[0]
        enabled = choose(["Style: Photo"], True)
        self.assertIn("Local chat model", enabled)
        self.assertTrue(helpers.helper_requested({"style_selection": enabled}))
        disabled = choose(enabled, False)
        self.assertEqual(disabled, ["Style: Photo"])
        self.assertFalse(helpers.helper_requested({"style_selection": disabled}))

    def test_real_image_stage_passes_captured_helper_through_inline_chat_call(self):
        item = life.JobRegistry().create()
        prepared = object()
        worker = SimpleNamespace(worker_thread=threading.current_thread())
        shared = SimpleNamespace(state=life.RuntimeState({"pipeline": None}))
        run = Mock(return_value="enhanced")
        stage = function_from(
            "ruinedfooocus/app/async_worker.py",
            "image_stage",
            {
                "wraps": wraps,
                "current_job": life.current_job,
                "service_scope": life.service_scope,
                "admission": life.admission,
                "shared": shared,
                "control_tools": SimpleNamespace(get_settings=lambda data: {}),
                "cleanup_interrupt": life.cleanup_interrupt,
                "check_interrupt": lambda data: None,
            },
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.async_worker": worker,
                    "ruinedfooocus.features.chat.llama_pipeline": SimpleNamespace(run_llama=run),
                },
            ),
            patch.object(app_modules, "async_worker", worker, create=True),
            patch.object(life, "_PIPELINES", {}),
            patch.object(life, "_RESIDENT", None),
            patch.object(life, "_release_native"),
            life.job_scope(item),
        ):
            wrapped = stage(lambda data: helpers.run_prompt("literal"))
            self.assertEqual(wrapped({helpers.PREPARATION_KEY: prepared}), "enhanced")
        self.assertIs(run.call_args.kwargs["prepared"], prepared)
        self.assertNotIn(helpers.PREPARATION_KEY, item.state)
        self.assertFalse(life.admission.held)

    def test_inline_scope_restores_prior_helper_after_failure(self):
        item = life.JobRegistry().create()
        parent, child = object(), object()
        item.state[helpers.PREPARATION_KEY] = parent
        with self.assertRaises(RuntimeError):
            with helpers.prompt_preparation_scope(item, {helpers.PREPARATION_KEY: child}):
                self.assertIs(item.state[helpers.PREPARATION_KEY], child)
                raise RuntimeError("failed child stage")
        self.assertIs(item.state[helpers.PREPARATION_KEY], parent)

    def test_capture_replaces_untrusted_private_payload_only_for_selected_helper(self):
        prepared = object()
        with patch.object(helpers, "capture_helper", return_value=prepared) as capture:
            plain = helpers.capture_prompt_preparation(
                {"prompt": "plain", helpers.PREPARATION_KEY: "forged"}, {}, object()
            )
            self.assertNotIn(helpers.PREPARATION_KEY, plain)
            capture.assert_not_called()
            selected = helpers.capture_prompt_preparation(
                {"style_selection": ["Hyperprompt"]}, {"llm_n_ctx": 512}, "paths"
            )
        self.assertIs(selected[helpers.PREPARATION_KEY], prepared)
        capture.assert_called_once_with({"llm_n_ctx": 512}, "paths")

    def test_helper_transfer_uses_existing_cancellable_chat_preparation(self):
        from concurrent.futures import CancelledError
        from ruinedfooocus.core.preparation import PreparationRequest
        from ruinedfooocus.features.chat import model_preparation

        with tempfile.TemporaryDirectory() as temporary:
            artifact = PreparationRequest(
                "https://example.invalid/inert", Path(temporary) / "missing.gguf"
            )
            prepared = model_preparation.ModelPreparation(artifact.destination, (artifact,), "{}")
            subscription = Mock()
            subscription.done.return_value = False
            cancelled = threading.Event()
            cancelled.set()
            with patch.object(
                model_preparation.preparation_service, "submit", return_value=subscription
            ):
                with self.assertRaises(CancelledError):
                    helpers.prepare_prompt_data(
                        {helpers.PREPARATION_KEY: prepared}, cancelled=cancelled
                    )
            subscription.cancel.assert_called_once()
            self.assertTrue(helpers.needs_prompt_preparation({helpers.PREPARATION_KEY: prepared}))

    def test_intent_detection_is_explicit_and_does_not_expand_prompt_content(self):
        for data in (
            {"style_selection": ["Hyperprompt"]},
            {"style_selection": ["local CHAT model"]},
            {"prompt": "a cat <style:Erniehancer>"},
            {"prompt": ["first", "second <style:Local chat model>"]},
            {"OBP_promptenhance": "superprompt-v1"},
        ):
            self.assertTrue(helpers.helper_requested(data), data)
        for data in (
            {"prompt": "__wildcard__ {not expanded}"},
            {"style_selection": ["Style: Pick Random", "photo-real"]},
            {"OBP_promptenhance": "off"},
            {"_metadata_prompt": True, "style_selection": ["Hyperprompt"]},
        ):
            self.assertFalse(helpers.helper_requested(data), data)

    def test_standalone_rewrite_captures_then_prepares_without_holding_compute(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "A.gguf"
            model.write_bytes(b"inert model identity")
            values = {"llama_localfile": str(model), "llm_n_ctx": 512}
            shared = SimpleNamespace(
                settings=SimpleNamespace(default_settings=values),
                path_manager=SimpleNamespace(DOWNLOADABLE_FILES={}),
            )
            worker = SimpleNamespace(worker_thread=None, check_interrupt=Mock())
            run = Mock(return_value="rewritten")
            with (
                patch.dict(
                    sys.modules,
                    {
                        "ruinedfooocus.app.state": shared,
                        "ruinedfooocus.app.async_worker": worker,
                        "ruinedfooocus.features.chat.llama_models": SimpleNamespace(
                            DEFAULT_MODEL=str(model), local_models=lambda: {}
                        ),
                        "ruinedfooocus.features.chat.llama_pipeline": SimpleNamespace(
                            run_llama=run
                        ),
                    },
                ),
                patch.object(app_modules, "state", shared, create=True),
                patch.object(app_modules, "async_worker", worker, create=True),
                patch.object(api_runtime, "submit", side_effect=lambda operation: operation),
                patch.object(
                    api_runtime, "wait_for_result", side_effect=lambda operation: operation
                ),
            ):
                operation = helpers.run_prompt("literal prompt", instruction="rules")
                values.update(llama_localfile="B.gguf", llm_n_ctx=8192)
                entered, release = threading.Event(), threading.Event()

                def prepare(data, *, cancelled):
                    entered.set()
                    self.assertTrue(release.wait(3))
                    return data

                admission = PreparationAdmission(max_workers=1, max_jobs=2)
                job = SimpleNamespace(cancelled=threading.Event())
                try:
                    with patch.object(helpers, "prepare_prompt_data", side_effect=prepare):
                        future = admission.submit(
                            lambda: operation.prepare(job), lambda value: None, self.fail
                        )
                        self.assertTrue(entered.wait(2))
                        independent = life.JobRegistry().create()
                        with life.admission.lease(independent):
                            self.assertEqual(operation.resource_group, "chat")
                        release.set()
                        future.result(2)
                    self.assertEqual(operation(job, 1), "rewritten")
                finally:
                    release.set()
                    admission.close()
                prepared = run.call_args.kwargs["prepared"]
                self.assertEqual(prepared.model, model.resolve())
                self.assertEqual(prepared.settings["llm_n_ctx"], 512)

    def test_nested_helper_requires_captured_model_and_never_acquires_under_compute(self):
        worker = SimpleNamespace(worker_thread=threading.current_thread())
        run = Mock(return_value="enhanced")
        item = life.JobRegistry().create()
        prepared = object()
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.async_worker": worker,
                    "ruinedfooocus.features.chat.llama_pipeline": SimpleNamespace(run_llama=run),
                },
            ),
            patch.object(app_modules, "async_worker", worker, create=True),
            patch.object(
                helpers, "capture_helper", side_effect=AssertionError("no live selection")
            ),
            life.job_scope(item),
            life.admission.lease(item),
        ):
            with self.assertRaisesRegex(ValueError, "not captured"):
                helpers.run_prompt("literal")
            item.state[helpers.PREPARATION_KEY] = prepared
            self.assertEqual(helpers.run_prompt("literal"), "enhanced")
        self.assertIs(run.call_args.kwargs["prepared"], prepared)


if __name__ == "__main__":
    unittest.main()
