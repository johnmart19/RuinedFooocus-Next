"""Per-browser explicit runtime diagnostics; metadata only under global CPU authority."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
import subprocess
from threading import Event, Lock

import gradio as gr

from ruinedfooocus.backends.llama.llama_runtime_info import runtime_info
from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary


class RuntimeInfoSession:
    def __init__(self):
        self.lock = Lock()
        self.cancelled = None

    def __deepcopy__(self, memo):
        return type(self)()

    @contextmanager
    def operation(self):
        with self.lock:
            if self.cancelled is not None:
                raise ValueError("A runtime refresh is already running in this session.")
            event = self.cancelled = Event()
        try:
            yield event
        finally:
            with self.lock:
                self.cancelled = None

    def cancel(self):
        with self.lock:
            if self.cancelled is not None:
                self.cancelled.set()

    def refresh(self, values, launch_args):
        from ruinedfooocus.app.preparation_admission import preparation_admission

        with self.operation() as event:

            def check():
                if event.is_set():
                    raise CancelledError("Runtime refresh cancelled.")

            cpu = values.get("cpu_everywhere") is True or force_cpu_for_auxiliary(launch_args)
            return preparation_admission.submit_work(
                lambda: runtime_info(cpu=cpu, check_cancelled=check), cancelled=event
            ).result()


def create_runtime_info_controls(launch_args, profile_base):
    details = gr.Markdown(t("Click Refresh to inspect installed versions and devices."))
    status = gr.Markdown()
    session = gr.State(RuntimeInfoSession(), delete_callback=lambda item: item.cancel())
    with gr.Row():
        refresh = gr.Button(t("Refresh"), size="sm")
        cancel = gr.Button(t("Cancel runtime refresh"), size="sm")
    gr.Markdown(
        t(
            "Detected devices are separate from PyTorch. CPU mode and GPU-layer settings still apply."
        )
    )

    def inspect(item, profile, request: gr.Request):
        locale = resolve_locale(request=request)
        yield gr.skip(), localize("Inspecting installed runtimes…", locale)
        try:
            result = item.refresh(dict(profile.get("settings", {})), launch_args)
        except CancelledError:
            yield gr.skip(), localize("Runtime refresh cancelled.", locale)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            yield (
                gr.skip(),
                localize(
                    "Runtime refresh failed or exceeded its limits. Previous results are retained.",
                    locale,
                ),
            )
        else:
            yield result, ""

    def stop(item, request: gr.Request):
        item.cancel()
        return localize("Stopping runtime refresh…", resolve_locale(request=request))

    refresh_event = refresh.click(
        inspect,
        inputs=[session, profile_base],
        outputs=[details, status],
        api_visibility="private",
        concurrency_id="chat-runtime-inspection",
        concurrency_limit=2,
        show_progress="hidden",
    )
    cancel.click(
        stop,
        inputs=session,
        outputs=status,
        queue=False,
        cancels=[refresh_event],
        api_visibility="private",
        show_progress="hidden",
    )
    return session
