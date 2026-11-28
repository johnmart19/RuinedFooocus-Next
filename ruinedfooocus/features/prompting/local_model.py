"""Serialize prompt assistance with chat and image model operations."""

import threading
import re
from contextlib import contextmanager

PREPARATION_KEY = "_prompt_preparation"
_OBP_HELPERS = {
    "llama",
    "🦙",
    "local chat model",
    "flufferizer",
    "erniehancer",
    "superprompter",
    "superprompt",
    "superprompt-v1",
    "hyperprompt",
}


def helper_requested(data):
    """Inspect explicit intent without expanding styles, wildcards or prompts."""
    if data.get("_metadata_prompt") and not data.get("_apply_prompt_processing"):
        return False
    from ruinedfooocus.features.prompting.style_catalog import STYLE_ALIASES, normalize_selection

    selected = normalize_selection(data.get("style_selection"))
    prompts = data.get("prompt", "")
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts or ():
        if isinstance(prompt, str):
            selected.extend(normalize_selection(re.findall(r"<style:([^>]+)>", prompt)))
    helper_styles = {name.casefold() for name in STYLE_ALIASES}
    return (
        any(name.casefold() in helper_styles for name in selected)
        or not data.get("_metadata_prompt")
        and str(data.get("OBP_promptenhance", "")).lower() in _OBP_HELPERS
    )


def capture_helper(values, manager):
    from ruinedfooocus.features.chat.model_preparation import capture_model
    from ruinedfooocus.features.chat.llama_models import DEFAULT_MODEL, local_models

    return capture_model(
        values.get("llama_localfile") or DEFAULT_MODEL, values, manager, local=local_models()
    )


def capture_prompt_preparation(data, values, manager):
    result = dict(data)
    result.pop(PREPARATION_KEY, None)
    if helper_requested(result):
        result[PREPARATION_KEY] = capture_helper(values, manager)
    return result


def needs_prompt_preparation(data):
    prepared = data.get(PREPARATION_KEY)
    return prepared is not None and not prepared.present


def prepare_prompt_data(data, *, cancelled):
    from ruinedfooocus.features.chat.model_preparation import prepare_chat_request

    prepared = data.get(PREPARATION_KEY)
    if prepared is not None:
        prepare_chat_request({"_chat_preparation": prepared}, cancelled=cancelled)
    return data


@contextmanager
def prompt_preparation_scope(item, data):
    previous = item.state.get(PREPARATION_KEY)
    item.state[PREPARATION_KEY] = data.get(PREPARATION_KEY)
    try:
        yield
    finally:
        if previous is None:
            item.state.pop(PREPARATION_KEY, None)
        else:
            item.state[PREPARATION_KEY] = previous


def run_prompt(prompt, system_file=None, *, instruction=None, check_cancelled=None):
    from ruinedfooocus.app import async_worker as worker

    if threading.current_thread() is not worker.worker_thread:
        from ruinedfooocus.app.api_runtime import submit, wait_for_result
        from ruinedfooocus.app import state as shared

        prepared = capture_helper(shared.settings.default_settings, shared.path_manager)

        def operation(job, task_id):
            def check():
                if check_cancelled:
                    check_cancelled()
                worker.check_interrupt({"_api_job": job})

            from ruinedfooocus.features.chat.llama_pipeline import run_llama

            return run_llama(
                system_file,
                prompt,
                instruction=instruction,
                check_cancelled=check,
                prepared=prepared,
            )

        def prepare(job):
            prepare_prompt_data({PREPARATION_KEY: prepared}, cancelled=job.cancelled)

        operation.prepare = prepare
        operation.resource_group = "chat"

        return wait_for_result(submit(operation))
    from ruinedfooocus.features.chat.llama_pipeline import run_llama
    from ruinedfooocus.app.resource_lifecycle import current_job

    item = current_job()
    prepared = item.state.get(PREPARATION_KEY) if item is not None else None
    if prepared is None:
        raise ValueError(
            "The local prompt helper was not captured for this request. Select its style explicitly and retry."
        )

    return run_llama(
        system_file,
        prompt,
        instruction=instruction,
        check_cancelled=check_cancelled,
        prepared=prepared,
    )
