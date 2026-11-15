"""Explicit preview/confirmation for the three storage-owner resets."""

import asyncio
from concurrent.futures import CancelledError
import threading

import gradio as gr

from ruinedfooocus.app.preparation_admission import preparation_admission
from ruinedfooocus.core.maintenance_reset import execute_reset, preview_reset, recheck_folders
from ruinedfooocus.core.translation_manager import (
    localize,
    localized_output,
    resolve_locale,
    translate_ui as t,
)


class ResetSession:
    def __init__(self):
        self.cancelled = threading.Event()

    def __deepcopy__(self, memo):
        return type(self)()

    def begin(self):
        self.cancelled.set()
        self.cancelled = threading.Event()
        return self.cancelled

    def cancel(self):
        self.cancelled.set()


def target_rows(plan, locale=None):
    if plan is None:
        return []
    labels = {
        "model-hashes": localize("Model hash records", locale),
        "library-index": localize("Library index", locale),
        "library-thumbnails": localize("Library thumbnails", locale),
        "startup-checks": localize("Startup verification", locale),
        "output-file": localize("Original output files", locale),
        "output-directory": localize("Subfolders", locale),
        "profile-document": localize("Settings", locale),
    }
    return [
        [
            item.path,
            labels[item.kind],
            0 if item.kind in {"library-index", "output-directory"} else 1,
            f"{0 if item.kind == 'library-index' else item.bytes} B",
            item.rows,
        ]
        for item in plan.targets
    ]


def retained_text(plan, locale=None):
    if plan is None:
        return ""
    if plan.action == "factory-profile":
        return localize(
            "Current profile: {profile}. Other profiles are kept.",
            locale,
            profile=plan.profile or "default",
        )
    return localize("Retention", locale) + ":\n" + "\n".join(plan.retained)


def result_text(result, locale=None):
    if result["complete"] and result.get("profile_reset"):
        return localize(
            "Current profile defaults restored. Restart the app to load them; current unsaved controls are unchanged.",
            locale,
        )
    heading = (
        localize("Completed", locale)
        if result["complete"]
        else localize("Cancelled", locale)
        if result["cancelled"]
        else localize("Failed", locale)
    )
    summary = localize(
        "Reset {count} entries; retained {retained} protected paths.",
        locale,
        count=result["removed"],
        retained=len(result["retained"]),
    )
    return (
        heading
        + "\n\n"
        + summary
        + (
            "\n\n" + "\n".join(error_text(error, locale) for error in result["errors"])
            if result["errors"]
            else ""
        )
    )


def error_text(error, locale=None):
    text = str(error)
    if text.startswith("Linked maintenance target is not eligible: "):
        return localize(
            "Linked maintenance target is not eligible: {path}",
            locale,
            path=text.partition(": ")[2],
        )
    if text.startswith("Unsupported output entry is not eligible: "):
        return localize(
            "Unsupported output entry is not eligible: {path}", locale, path=text.partition(": ")[2]
        )
    # Known service messages are registered by the same catalogue workflow;
    # opaque OS/native details retain the existing diagnostic English fallback.
    return localize(text, locale)


def create_storage_reset(manager, settings):
    with gr.Accordion(t("Folder maintenance"), open=False):
        gr.Markdown(
            t(
                "Clear rebuildable cache resets model hashes, Gallery index, thumbnails and startup checks. Other cache data, models, installations, inputs, resumable work and custom folders are kept."
            )
        )
        gr.Markdown(
            t(
                "Factory reset affects only the current profile. Other profiles are kept. Restart is required to load its factory defaults."
            )
        )
        session = gr.State(ResetSession(), delete_callback=lambda value: value.cancel())
        plan = gr.State(None)
        with gr.Row():
            recheck = gr.Button(t("Recheck folders"))
            cache = gr.Button(t("Clear rebuildable cache"))
            outputs = gr.Button(t("Clear cache and outputs"))
            factory = gr.Button(t("Factory reset settings"))
        table = gr.Dataframe(
            headers=[
                localize("Path"),
                localize("Action"),
                localize("Files"),
                localize("Size"),
                localize("Rows"),
            ],
            datatype=["str", "str", "number", "str", "number"],
            value=[],
            interactive=False,
            wrap=True,
        )
        retained = gr.Textbox(label=t("Retention"), interactive=False, lines=3)
        confirmed = gr.Checkbox(label=t("Confirm permanent deletion"), value=False, visible=False)
        with gr.Row():
            apply = gr.Button(t("Confirm reset"), interactive=False, variant="stop")
            cancel = gr.Button(t("Cancel"))
        status = gr.Markdown()
        localized_output(lambda locale, value: target_rows(value, locale), [plan], [table])
        localized_output(lambda locale, value: retained_text(value, locale), [plan], [retained])
        localized_output(
            lambda locale: gr.update(
                headers=[
                    localize("Path", locale),
                    localize("Action", locale),
                    localize("Files", locale),
                    localize("Size", locale),
                    localize("Rows", locale),
                ]
            ),
            [],
            [table],
        )

        async def prepare(action, control, request):
            locale = resolve_locale(request=request)
            cancelled = control.begin()
            try:
                value = await asyncio.wrap_future(
                    preparation_admission.submit_work(
                        lambda: preview_reset(action, manager, settings, cancelled=cancelled)
                    )
                )
                if control.cancelled is not cancelled or cancelled.is_set():
                    return (gr.skip(),) * 6
                return (
                    value,
                    target_rows(value, locale),
                    retained_text(value, locale),
                    gr.update(value=False, visible=action != "cache"),
                    gr.update(interactive=action == "cache"),
                    localize("Preview", locale),
                )
            except (
                ValueError,
                OSError,
                RuntimeError,
                CancelledError,
                asyncio.CancelledError,
            ) as error:
                if control.cancelled is not cancelled or cancelled.is_set():
                    return (gr.skip(),) * 6
                return (
                    None,
                    [],
                    "",
                    gr.update(value=False, visible=False),
                    gr.update(interactive=False),
                    error_text(error, locale),
                )

        async def preview_cache(control, request: gr.Request = None):
            return await prepare("cache", control, request)

        async def preview_outputs(control, request: gr.Request = None):
            return await prepare("cache-outputs", control, request)

        async def preview_factory(control, request: gr.Request = None):
            return await prepare("factory-profile", control, request)

        async def check_folders(request: gr.Request = None):
            try:
                folders = await asyncio.wrap_future(
                    preparation_admission.submit_work(lambda: recheck_folders(manager, settings))
                )
                return (
                    localize("Completed", resolve_locale(request=request))
                    + "\n\n"
                    + "\n".join(folders)
                )
            except (ValueError, OSError) as error:
                return error_text(error, resolve_locale(request=request))

        async def apply_reset(value, checked, control, request: gr.Request = None):
            locale = resolve_locale(request=request)
            cancelled = control.begin()
            try:
                result = await asyncio.wrap_future(
                    preparation_admission.submit_work(
                        lambda: execute_reset(
                            value, manager, settings, confirmed=checked, cancelled=cancelled
                        )
                    )
                )
                if control.cancelled is not cancelled:
                    return (gr.skip(),) * 4
                return (
                    None,
                    gr.update(interactive=False),
                    gr.update(value=False, visible=False),
                    result_text(result, locale),
                )
            except (
                ValueError,
                OSError,
                RuntimeError,
                CancelledError,
                asyncio.CancelledError,
            ) as error:
                if control.cancelled is not cancelled:
                    return (gr.skip(),) * 4
                return (
                    None,
                    gr.update(interactive=False),
                    gr.update(value=False, visible=False),
                    localize(
                        "Cancelled"
                        if isinstance(error, (CancelledError, asyncio.CancelledError))
                        else "Failed",
                        locale,
                    )
                    + "\n\n"
                    + error_text(error, locale),
                )

        def confirm(value, checked):
            return gr.update(
                interactive=value is not None and (value.action == "cache" or checked is True)
            )

        def cancel_reset(control, request: gr.Request = None):
            control.cancel()
            return (
                None,
                [],
                "",
                gr.update(value=False, visible=False),
                gr.update(interactive=False),
                localize("Cancelled", resolve_locale(request=request)),
            )

        options = dict(
            concurrency_id="cache_maintenance",
            concurrency_limit=1,
            show_progress="minimal",
            show_progress_on=status,
            api_visibility="private",
            queue=False,
        )
        for button, callback in (
            (cache, preview_cache),
            (outputs, preview_outputs),
            (factory, preview_factory),
        ):
            button.click(
                callback,
                inputs=session,
                outputs=[plan, table, retained, confirmed, apply, status],
                **options,
            )
        recheck.click(check_folders, outputs=status, **options)
        apply.click(
            apply_reset,
            inputs=[plan, confirmed, session],
            outputs=[plan, apply, confirmed, status],
            **options,
        )
        confirmed.change(
            confirm,
            inputs=[plan, confirmed],
            outputs=apply,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        cancel.click(
            cancel_reset,
            inputs=session,
            outputs=[plan, table, retained, confirmed, apply, status],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
