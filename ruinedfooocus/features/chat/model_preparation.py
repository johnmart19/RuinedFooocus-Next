"""Captured Chat artifacts and session selection, without compute or runtime imports."""

from concurrent.futures import CancelledError
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import threading

from ruinedfooocus.core.preparation import PreparationRequest, preparation_service, verify_file


RUNTIME_SETTING_KEYS = (
    "llm_runtime",
    "llama_backend",
    "cpu_everywhere",
    "llama_server_args",
    "llama_moe_cache_cpu_share_enabled",
    "llama_moe_cache_cpu_share",
    "llm_n_ctx",
    "llm_n_predict",
    "llm_n_gpu_layers",
    "llama_mmproj",
    "llm_sampling",
    "llm_use_model_defaults",
    "llm_rewrite_reasoning",
    "llm_chat_reasoning_mode",
    "llm_chat_reasoning_effort",
    "llm_rewrite_reasoning_mode",
    "llm_rewrite_reasoning_effort",
)


def runtime_selection_key(model, values, *, vision=False, projector=None):
    """Passive equality for the selected file and captured load configuration.

    A key identifies a requested configuration, not proof of initialized devices.
    Unknown/missing files receive their own key; no files are opened or acquired.
    """
    from ruinedfooocus.features.models.model_identity import file_signature

    path = Path(model).expanduser().resolve()
    try:
        signature = file_signature(path)
    except OSError:
        signature = [str(path), "missing"]
    selected = projector or values.get("llama_mmproj")
    projector_signature = None
    if selected and str(selected).strip() not in ("Auto", "None"):
        companion = Path(selected).expanduser().resolve()
        try:
            projector_signature = file_signature(companion)
        except OSError:
            projector_signature = [str(companion), "missing"]
    payload = [
        signature,
        bool(vision),
        projector_signature,
        {key: values.get(key) for key in RUNTIME_SETTING_KEYS},
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def initialized_runtime_status():
    """Read the resident Chat owner after a terminal event without loading it."""
    from ruinedfooocus.app.resource_lifecycle import service_pipeline

    owner = service_pipeline("chat")
    if owner is None or not getattr(owner, "llm", None):
        return {"active_model": None, "runtime_snapshot": None, "load_state": None}
    status = owner.runtime_status()
    paths = getattr(owner, "loaded_paths", ())
    return {
        "active_model": str(paths[0]) if paths else None,
        "runtime_snapshot": status["initialized"],
        "load_state": status["load"],
    }


@dataclass(frozen=True)
class ModelPreparation:
    model: Path
    artifacts: tuple[PreparationRequest, ...]
    settings_json: str
    vision: bool = False
    projector: Path | None = None

    @property
    def settings(self):
        return json.loads(self.settings_json)

    @property
    def present(self):
        return all(item.destination.is_file() for item in self.artifacts)


def capture_model(model, values, manager, *, vision=False, local=None):
    """Resolve paths/catalogue now; a later settings edit cannot redirect the job."""

    def artifact(value, kind):
        path = Path(value).expanduser()
        key = next(
            (
                key
                for key in (f"{kind}/{path.name}", f"llm/{path.name}", f"vision/{path.name}")
                if key in manager.DOWNLOADABLE_FILES
            ),
            None,
        )
        if not path.is_absolute():
            if key:
                info = manager.DOWNLOADABLE_FILES[key]
                roots = manager.get_model_folders(info["path"].removeprefix("path_"))
                path = next(
                    (
                        root / info["filename"]
                        for root in roots
                        if (root / info["filename"]).is_file()
                    ),
                    manager.preparation_request(key).destination,
                )
            elif not path.is_file():
                candidates = [
                    Path(item)
                    for item in (local or {}).values()
                    if Path(item).name.casefold() == str(value).casefold()
                ]
                if len(candidates) > 1:
                    raise ValueError(
                        "Multiple local models have this filename. Choose its full path."
                    )
                if candidates:
                    path = candidates[0]
                else:
                    raise ValueError(f"Model file is missing: {value}")
        path = path.resolve()
        if key:
            # An arbitrary custom path sharing a basename is validated by the
            # loader, rather than being asserted to match a catalogue checksum.
            info = manager.DOWNLOADABLE_FILES[key]
            roots = manager.get_model_folders(info["path"].removeprefix("path_"))
            if any(path == (root / info["filename"]).resolve() for root in roots):
                return manager.preparation_request(key, destination=path)
        if not path.is_file():
            raise ValueError(f"Model file is missing: {value}")
        return PreparationRequest("", path)

    weight = artifact(model, "llm")
    artifacts = [weight]
    projector = None
    if vision:
        from ruinedfooocus.features.chat.llama_vision import projector_setting, vision_entry

        selected = projector_setting(values)
        if selected == "Auto":
            selected = vision_entry(manager, weight.destination).get("mmproj")
        if selected and selected != "None":
            companion = artifact(selected, "vision")
            artifacts.append(companion)
            projector = companion.destination
    return ModelPreparation(
        weight.destination,
        tuple(artifacts),
        json.dumps(values, sort_keys=True),
        bool(vision),
        projector,
    )


class ModelTransfer:
    """One consumer's subscriptions, including known/unknown byte progress."""

    def __init__(self, request, *, service=preparation_service):
        self.request = request
        self._lock = threading.RLock()
        self.progress = (0, 0, 0)
        self._subscriptions = []
        try:
            for artifact in request.artifacts:
                if artifact.destination.is_file():
                    # Existing weights can have active read leases. Verification
                    # is another reader, never a replacement/publication writer.
                    with service.artifact_lease(artifact.destination):
                        verify_file(artifact.destination, artifact)
                    size = artifact.destination.stat().st_size
                    self._report(size, size, 0)
                else:
                    self._subscriptions.append(service.submit(artifact, self._report))
        except BaseException:
            self.cancel()
            raise

    def _report(self, received, total, speed):
        with self._lock:
            self.progress = (received, total, speed)

    def done(self):
        return all(item.done() for item in self._subscriptions)

    def result(self):
        try:
            for item in self._subscriptions:
                item.result()
        except BaseException:
            self.cancel()
            raise
        return self.request

    def cancel(self):
        for item in self._subscriptions:
            item.cancel()


class ChatModelSession:
    """Mutable only under this session's lock; transfer completion never activates."""

    def __init__(self):
        self.lock = threading.RLock()
        self.selected = None
        self.revision = 0
        self.transfer = None
        self.active = None

    def __deepcopy__(self, memo):
        # Gradio copies the initial State once for each browser session.
        return type(self)()

    def select(self, model):
        with self.lock:
            if model != self.selected:
                self.selected = model
                self.revision += 1
            return self.revision

    def start(self, request, *, service=preparation_service):
        with self.lock:
            if self.transfer is not None and not self.transfer.done():
                raise ValueError("Cancel the current download before starting another.")
            revision = self.revision
            transfer = ModelTransfer(request, service=service)
            self.transfer = transfer
            return revision, transfer

    def current(self, revision):
        with self.lock:
            return revision == self.revision

    def cancel(self):
        with self.lock:
            if self.transfer:
                self.transfer.cancel()


def validate_message(message, active):
    if not message.strip():
        raise ValueError("Enter a message first.")
    if not active:
        raise ValueError("Load a model before sending a message.")
    return message


def capture_chat_request(data, values, manager, local, *, launch_args=None):
    """Capture Chat UI/API dependencies before worker/global cleanup admission."""
    from copy import deepcopy
    from ruinedfooocus.features.chat.llama_vision import requested_vision

    result = deepcopy(data)
    if result.get("unload"):
        return result
    attachments = result.get("_chat_attachments")
    if attachments is not None:
        from ruinedfooocus.app.resource_lifecycle import current_owner
        from ruinedfooocus.features.chat.attachments import AttachmentTurn

        if not isinstance(attachments, AttachmentTurn):
            raise ValueError("Use Chat to attach local text files.")
        attachments.check(current_owner())
    model = result.get("model") or values.get("llama_localfile")
    if not model:
        from ruinedfooocus.features.chat.llama_models import DEFAULT_MODEL

        model = DEFAULT_MODEL
    vision = requested_vision(
        model,
        values,
        bool(result.get("vision_enabled"))
        or result.get("image_source", "New image") != "New image",
    )
    prepared = result.get("_chat_preparation") or capture_model(
        model,
        values,
        manager,
        vision=vision,
        local=local,
    )
    result["_chat_preparation"] = prepared
    result["model"] = str(prepared.model)
    if result.get("vision_enabled") and result.get("vision_model"):
        result["_review_preparation"] = capture_model(
            result["vision_model"],
            prepared.settings,
            manager,
            vision=True,
            local=local,
        )
        result["vision_model"] = str(result["_review_preparation"].model)
    # URL context is file preparation too, never a transfer under compute.
    from ruinedfooocus.core.paths import data_path
    from ruinedfooocus.core.util import url_to_filename

    sources = json.loads(result.get("embed") or "[]")
    result["_context_preparation"] = tuple(
        PreparationRequest(value, data_path("cache", "embeds", url_to_filename(value)))
        for kind, value in sources
        if kind == "url"
    )
    if (sources or attachments is not None) and not result.get("load_only"):
        from ruinedfooocus.features.chat.rag_preparation import capture_chat_rag

        result["_rag_preparation"] = capture_chat_rag(
            sources or attachments.files, values, launch_args=launch_args
        )
    return result


def chat_artifacts(data):
    result = list(data.get("_context_preparation", ()))
    for key in ("_chat_preparation", "_review_preparation"):
        if key in data:
            result.extend(data[key].artifacts)
    return tuple(dict.fromkeys(result))


def prepare_chat_request(data, progress=None, cancelled=None):
    from ruinedfooocus.features.chat.rag_preparation import prepare_rag

    rag = data.get("_rag_preparation")
    if (
        rag is not None
        and rag.settings["offline"]
        and any(
            not artifact.destination.is_file() for artifact in data.get("_context_preparation", ())
        )
    ):
        raise RuntimeError(
            "Chat context files are missing; prepare them online before using RAG offline."
        )
    attachments = data.get("_chat_attachments")
    if attachments is not None:
        from ruinedfooocus.features.chat.attachment_retrieval import prepare_passages

        def check_attachments():
            attachments.check(data["_job"].owner, cancelled)

        data["_attachment_passages"] = prepare_passages(
            attachments, data["history"][-1]["content"], check_attachments
        )
    prepare_rag(rag, progress=progress, cancelled=cancelled)
    subscriptions = []
    try:
        for artifact in chat_artifacts(data):
            if not artifact.destination.is_file():
                subscriptions.append(preparation_service.submit(artifact, progress))
        for subscription in subscriptions:
            while not subscription.done():
                if cancelled is not None and cancelled.wait(0.05):
                    raise CancelledError()
                if cancelled is None:
                    try:
                        subscription.result(timeout=0.05)
                    except TimeoutError:
                        pass
            subscription.result()
        if data.get("load_only") and data.get("_chat_preparation") is not None:
            from ruinedfooocus.features.chat.sampling_sources import prepare_sampling_source

            prepare_sampling_source(data["_chat_preparation"].model)
        return data
    finally:
        for subscription in subscriptions:
            if not subscription.done():
                subscription.cancel()
