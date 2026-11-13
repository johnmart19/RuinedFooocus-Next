"""Owner controls for the pinned Chat embedding model, without loading an index."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
from threading import Event, Lock

import gradio as gr

from ruinedfooocus.core.settings_schema import RAG_EMBEDDING_DEVICES
from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.features.chat import rag_preparation as rag


class EmbeddingPreparationSession:
    """Own only this Settings tab's preparation subscription and cancel signal."""

    def __init__(self):
        self.lock = Lock()
        self.cancelled = None

    def __deepcopy__(self, memo):
        return type(self)()

    @property
    def running(self):
        with self.lock:
            return self.cancelled is not None

    @contextmanager
    def preparing(self):
        with self.lock:
            if self.cancelled is not None:
                raise ValueError("Embedding preparation is already running in this session.")
            cancelled = self.cancelled = Event()
        try:
            yield cancelled
        finally:
            with self.lock:
                self.cancelled = None

    def cancel(self):
        with self.lock:
            if self.cancelled is not None:
                self.cancelled.set()


def embedding_file_status(prepared):
    """Count exact-size files using stat only. Preparation owns checksum reads."""
    complete = invalid = 0
    for item in prepared.artifacts:
        try:
            if item.destination.is_file():
                if item.destination.stat().st_size == item.expected_size:
                    complete += 1
                else:
                    invalid += 1
        except FileNotFoundError:
            continue
        except OSError:
            invalid += 1
    return complete, invalid


def initialized_device(pipeline):
    index = getattr(pipeline, "embeddings", None)
    vectorizer = getattr(index, "model", None)
    pooling = getattr(vectorizer, "model", None)
    device = getattr(pooling, "device", None)
    if getattr(device, "type", None) in ("cpu", "cuda", "mps", "xpu"):
        return str(device)
    return None


def create_rag_settings(tab, settings, launch_args, add_setting):
    with gr.Accordion(t("Documents and RAG"), open=False):
        total_bytes = sum(size for _, size, _ in rag.MODEL_FILES)
        gr.Markdown(
            f"[{rag.MODEL_ID}](https://huggingface.co/{rag.MODEL_ID}/tree/{rag.MODEL_REVISION})"
            f" · {rag.MODEL_DIMENSIONS} dimensions · {total_bytes / 1024**2:.1f} MiB"
        )
        device = gr.Radio(
            choices=list(RAG_EMBEDDING_DEVICES),
            value=settings.default_settings.get("rag_embedding_device", "Auto"),
            label=t("Chat embedding device"),
            info=t(
                "Save to apply to the next grounded Chat request. Auto lets txtai choose the device."
            ),
        )
        add_setting("rag_embedding_device", device)
        status = gr.Markdown()
        placement = gr.Markdown()
        progress_text = gr.Markdown()
        with gr.Row():
            refresh = gr.Button(t("Refresh embedding status"), size="sm")
            prepare = gr.Button(
                t("Prepare or verify embedding model"), size="sm", interactive=False
            )
            cancel = gr.Button(t("Cancel embedding preparation"), size="sm", interactive=False)
        session = gr.State(
            EmbeddingPreparationSession(), delete_callback=lambda item: item.cancel()
        )

    def refresh_status(item, request: gr.Request):
        from ruinedfooocus.app.resource_lifecycle import service_pipeline

        locale = resolve_locale(request=request)
        values = settings.default_settings
        selected, reason = rag.embedding_device_policy(values, launch_args)
        policy = localize("Placement for new requests: {device}.", locale, device=selected)
        if reason == "global":
            policy += " " + localize("Global CPU mode is active.", locale)
        elif reason == "directml":
            policy += " " + localize("DirectML image mode uses CPU for Chat embeddings.", locale)
        observed = initialized_device(service_pipeline("chat"))
        if observed is not None:
            policy += " " + localize(
                "Initialized assistant index: {device}.", locale, device=observed
            )

        model = rag.capture_embedding_model(values, launch_args=launch_args)
        complete, invalid = embedding_file_status(model)
        if invalid:
            message = localize(
                "One or more local embedding files have the wrong size or are unreadable.", locale
            )
        elif complete == len(model.artifacts):
            message = localize("All pinned embedding files are present.", locale)
        else:
            message = localize(
                "Embedding files present: {present} of {total}.",
                locale,
                present=complete,
                total=len(model.artifacts),
            )
        if model.settings["offline"] and complete < len(model.artifacts):
            message += " " + localize("Offline mode prevents downloading missing files.", locale)
        can_prepare = not item.running and (
            complete == len(model.artifacts) or not model.settings["offline"]
        )
        return (
            message,
            policy,
            gr.update(interactive=can_prepare),
            gr.update(interactive=item.running),
        )

    status_outputs = [status, placement, prepare, cancel]

    def show_when_selected(item, request: gr.Request, event: gr.SelectData):
        return (
            refresh_status(item, request)
            if event.selected
            else tuple(gr.skip() for _ in status_outputs)
        )

    events = dict(
        inputs=[session],
        outputs=status_outputs,
        queue=False,
        api_visibility="private",
        show_progress="hidden",
    )
    tab.select(show_when_selected, **events)
    refresh.click(refresh_status, **events)

    def prepare_model(item, request: gr.Request, progress=gr.Progress()):
        locale = resolve_locale(request=request)
        try:
            with item.preparing() as cancelled:
                busy = localize("Preparing embedding model files…", locale)
                yield (
                    busy,
                    gr.skip(),
                    gr.skip(),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                )
                model = rag.capture_embedding_model(
                    settings.default_settings, launch_args=launch_args
                )

                def report(received, total, speed):
                    if total:
                        progress((received, total), desc=busy)

                rag.prepare_rag(model, cancelled=cancelled, progress=report)
                if cancelled.is_set():
                    raise CancelledError()
            message = localize("Pinned embedding files verified.", locale)
        except CancelledError:
            message = localize(
                "Embedding preparation cancelled. Completed files are retained.", locale
            )
        except (OSError, RuntimeError, ValueError):
            message = localize(
                "Embedding preparation failed. Check the local files or connection and try again.",
                locale,
            )
        yield message, *refresh_status(item, request)

    prepare.click(
        prepare_model,
        inputs=[session],
        outputs=[progress_text, *status_outputs],
        api_visibility="private",
        show_progress="hidden",
    )
    cancel.click(
        EmbeddingPreparationSession.cancel,
        inputs=[session],
        queue=False,
        api_visibility="private",
        show_progress="hidden",
    )
