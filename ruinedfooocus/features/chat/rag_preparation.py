"""Capture the existing txtai model's finite local closure before Chat compute."""

from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ruinedfooocus.core.paths import data_path, resource_path
from ruinedfooocus.core.preparation import PreparationRequest, preparation_service, verify_file
from ruinedfooocus.core.settings_schema import RAG_EMBEDDING_DEVICES, RAG_EMBEDDING_MODELS
from ruinedfooocus.features.chat.model_preparation import ModelPreparation, ModelTransfer
from ruinedfooocus.features.models.model_identity import cached_file_sha256, file_signature

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_DIMENSIONS = 384


@dataclass(frozen=True)
class RagModel:
    model_id: str
    label: str
    revision: str
    files: tuple
    contract: str

    @property
    def identity(self):
        return json.loads(self.contract)


@lru_cache(maxsize=1)
def model_specs():
    """Finite bundled manifests only; no discovery, network or model imports."""
    path = resource_path("rag_embedding_models.json")
    if path.stat().st_size > 64 * 1024:
        raise ValueError("Bundled Chat embedding catalogue is too large.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or {row["id"] for row in data["models"]} != set(
        RAG_EMBEDDING_MODELS
    ):
        raise ValueError("Bundled Chat embedding catalogue is incomplete.")
    result = []
    for row in data["models"]:
        identity = dict(row["encoding"])
        manifest = json.dumps(row["files"], sort_keys=True, separators=(",", ":"))
        identity["manifest_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
        result.append(
            RagModel(
                row["id"],
                row["label"],
                row["revision"],
                tuple(tuple(item) for item in row["files"]),
                json.dumps(identity, sort_keys=True),
            )
        )
    return tuple(result)


def model_spec(model_id=MODEL_ID):
    for item in model_specs():
        if item.model_id == model_id:
            return item
    raise ValueError("Choose a listed Chat embedding model.")


def model_choices():
    return [
        (
            f"{item.label} · {sum(size for _, size, _ in item.files) / 1024**2:.0f} MiB",
            item.model_id,
        )
        for item in model_specs()
    ]


def model_description(model_id):
    item = model_spec(model_id)
    return (
        f"[{item.model_id}](https://huggingface.co/{item.model_id}/tree/{item.revision})"
        f" · {item.identity['dimensions']} dimensions · {sum(size for _, size, _ in item.files) / 1024**2:.1f} MiB"
    )


# Public default constants retain the existing migration and caller contract.
MODEL_FILES = model_spec().files


def embedding_device_policy(values, launch_args):
    """Return the effective placement choice and its owner without probing devices."""
    from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary

    saved = values.get("rag_embedding_device", "Auto")
    if saved not in RAG_EMBEDDING_DEVICES:
        raise ValueError("Choose Auto or CPU for Chat text embeddings.")
    if values.get("cpu_everywhere") or force_cpu_for_auxiliary(launch_args):
        return "CPU", "global"
    if saved == "CPU":
        return "CPU", "profile"
    return saved, "profile"


def _chat_preparation_options(values, launch_args):
    if launch_args is None:
        from ruinedfooocus.app.arguments import args as launch_args

    device, _ = embedding_device_policy(values, launch_args)
    return dict(
        cpu=device == "CPU",
        offline=bool(getattr(launch_args, "offline", False)) or os.environ.get("RF_OFFLINE") == "1",
        model_id=model_spec(values.get("rag_embedding_model", MODEL_ID)).model_id,
    )


def capture_embedding_model(values, *, launch_args=None):
    """Capture the pinned model for explicit owner Settings preparation."""
    return _capture_embedding_model(**_chat_preparation_options(values, launch_args))


def capture_chat_rag(sources, values, *, launch_args=None):
    """Use the same captured embedding policy for UI and API Chat."""
    if not sources:
        return None
    return capture_rag(sources, **_chat_preparation_options(values, launch_args))


def capture_rag(
    sources, *, cpu=False, offline=False, cache_root=None, cpu_reason="", model_id=MODEL_ID
):
    """Capture the pinned model when the Chat request includes context."""
    if not sources:
        return None
    return _capture_embedding_model(
        cpu=cpu, offline=offline, cache_root=cache_root, cpu_reason=cpu_reason, model_id=model_id
    )


def _capture_embedding_model(*, cpu, offline, cache_root=None, cpu_reason="", model_id=MODEL_ID):
    """No acquisition/runtime import; reuse an already complete pinned snapshot."""
    spec = model_spec(model_id)
    files = MODEL_FILES if model_id == MODEL_ID else spec.files
    model_root = data_path("models", "rag_embeddings") if cache_root is None else Path(cache_root)
    directory = (
        Path(cache_root)
        if cache_root is not None
        else data_path("models", "rag_embeddings", spec.revision)
        if model_id == MODEL_ID
        else data_path("models", "rag_embeddings", model_id.rsplit("/", 1)[-1], spec.revision)
    )
    if not all((directory / name).is_file() for name, _, _ in files):
        try:
            cached = Path(
                snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True)
            )
        except LocalEntryNotFoundError:
            cached = None
        if cached is not None and all((cached / name).is_file() for name, _, _ in files):
            directory = cached
            model_root = cached
    directory = directory.resolve()
    policy = {"gpu": not bool(cpu), "offline": bool(offline), "encoder": spec.identity}
    if cpu_reason:
        policy["cpu_reason"] = cpu_reason
    return ModelPreparation(
        directory,
        tuple(
            PreparationRequest(
                f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/{name}",
                directory / name,
                size,
                checksum,
                model_root=model_root,
            )
            for name, size, checksum in files
        ),
        json.dumps(policy, sort_keys=True),
    )


def prepare_rag(prepared, *, cancelled=None, progress=None, service=preparation_service):
    """Use the existing transfer/subscription service in the preparation lane."""
    if prepared is None:
        return None
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()
    if prepared.settings["offline"] and not prepared.present:
        raise RuntimeError(
            "The Chat context embedding model is incomplete; prepare it online before using RAG offline."
        )
    missing = []
    for artifact in prepared.artifacts:
        if not artifact.destination.is_file():
            missing.append(artifact)
            continue
        with service.artifact_lease(artifact.destination, cancelled=cancelled):
            verify_file(artifact.destination, replace(artifact, sha256=None), cancelled)
            if (
                artifact.sha256
                and cached_file_sha256(artifact.destination, data_path("cache", "hashes")).lower()
                != artifact.sha256
            ):
                raise IOError("Prepared RAG artifact SHA256 does not match")
    if not missing:
        return prepared
    transfer = ModelTransfer(replace(prepared, artifacts=tuple(missing)), service=service)
    last = None
    try:
        while not transfer.done():
            if progress is not None and transfer.progress != last:
                last = transfer.progress
                progress(*last)
            if cancelled is not None:
                if cancelled.wait(0.05):
                    raise CancelledError()
            else:
                time.sleep(0.05)
        if cancelled is not None and cancelled.is_set():
            raise CancelledError()
        transfer.result()
        if progress is not None:
            progress(*transfer.progress)
        return prepared
    finally:
        if not transfer.done():
            transfer.cancel()


def runtime_options(prepared):
    """Options for the existing txtai.Embeddings, constructed only by its owner."""
    if prepared is None or not prepared.present or not prepared.model.is_dir():
        raise RuntimeError("Chat context embedding preparation is incomplete; retry the request.")
    if not prepared.settings["gpu"] and prepared.settings.get("cpu_reason"):
        from ruinedfooocus.core.device_fallback import warn_cpu_fallback

        warn_cpu_fallback("Chat text embeddings", prepared.settings["cpu_reason"])
    return {
        "content": True,
        "path": str(prepared.model),
        # txtai's native device resolver short-circuits to CPU on None.
        # False still enters its accelerator availability probes.
        "gpu": True if prepared.settings["gpu"] else None,
        "maxlength": prepared.settings.get("encoder", model_spec().identity)["max_length"],
        # txtai forwards vectors to AutoModel.from_pretrained; its top-level
        # device option does not control HFVectors placement. Tokenizer/pooling
        # metadata also resolve within this complete absolute local directory.
        "vectors": {"local_files_only": True, "trust_remote_code": False},
    }


def index_identity(sources, prepared, prepared_sources=None):
    """Stat-only identity: changed context/model files cannot reuse a stale index."""
    if not sources:
        return str(sources)
    if prepared is None:
        raise ValueError("Chat context embedding preparation is missing; retry the request.")
    from ruinedfooocus.features.chat.assistants import context_file

    context_signatures = []
    for kind, value in sources:
        if kind == "file":
            context_signatures.append(file_signature(context_file(value)))
        elif kind == "url":
            path = (prepared_sources or {}).get(value)
            if path is None:
                raise ValueError("Prepare URL context before entering Chat compute.")
            context_signatures.append(file_signature(path))
    return json.dumps(
        {
            "sources": sources,
            "context_signatures": context_signatures,
            "model": str(prepared.model),
            "dimensions": prepared.settings.get("encoder", model_spec().identity)["dimensions"],
            "encoder": prepared.settings.get("encoder", model_spec().identity),
            "gpu": prepared.settings["gpu"],
            "artifacts": [file_signature(artifact.destination) for artifact in prepared.artifacts],
        },
        sort_keys=True,
    )
