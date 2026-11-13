"""Source-qualified task eligibility; never probe devices to render a picker."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import os
import json
from pathlib import Path
from functools import lru_cache

from ruinedfooocus.features.models.model_registry import load_registry, RegistryError


@lru_cache(maxsize=1)
def _native_source_audits():
    """Additive exact-source reviews; never widen an unknown version range."""
    from ruinedfooocus.core.paths import resource_path

    try:
        with resource_path("model_metadata", "native_source_audits.json").open("rb") as stream:
            raw = stream.read(128 * 1024 + 1)
        document = json.loads(raw) if len(raw) <= 128 * 1024 else {}
        return document.get("audits", {}) if document.get("schema_version") == 1 else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _audited_native_build(artifact, runtime, task, source_build):
    if runtime.name != "llama.cpp" or runtime.build is None:
        return False
    audit = _native_source_audits().get(runtime.build, {})
    return (
        audit.get("base_build") == source_build
        and audit.get("scope") == "source_only"
        and audit.get("architectures", {}).get(artifact.get("architecture"), {}).get(task)
        == "source_supported"
    )


@dataclass(frozen=True)
class RuntimeIdentity:
    name: str
    build: str | None
    package: str | None = None
    origin: str = "unknown"


@dataclass(frozen=True)
class Eligibility:
    status: str
    reason: str

    @property
    def allowed(self):
        return self.status != "unsupported"


def configured_runtime(name):
    """Describe configured source/package identity, not device execution support."""
    if name == "llama.cpp":
        if os.environ.get("LLAMA_SERVER"):
            return RuntimeIdentity(name, None, origin="custom binary; build unverified")
        from ruinedfooocus.backends.llama.llama_installer import VERSION

        return RuntimeIdentity(name, VERSION, origin="managed source pin; device unverified")
    if name == "xllamacpp":
        try:
            registry_version = load_registry().provenance["runtime_versions"]["xllamacpp"]
        except RegistryError:
            return RuntimeIdentity(name, None, origin="source metadata unavailable")
        try:
            package = version("xllamacpp")
        except PackageNotFoundError:
            return RuntimeIdentity(
                name,
                registry_version["llama_tag"],
                registry_version["latest_pypi"],
                "installer source pin; not installed",
            )
        build = (
            registry_version["llama_tag"] if package == registry_version["latest_pypi"] else None
        )
        return RuntimeIdentity(
            name, build, package, "installed package declaration; device unverified"
        )
    return RuntimeIdentity(name, None)


def task_eligibility(artifact, runtime, task="text"):
    """Keep source support independent of access, residency, memory and quality."""
    if task not in ("text", "image"):
        raise ValueError("Choose the text or image task.")
    if artifact is None:
        return Eligibility(
            "unknown", "Custom model: source compatibility has not been established."
        )
    if not artifact["chat_weight"]:
        return Eligibility("unsupported", "This component is not a standalone Chat language model.")
    if task == "image" and not artifact["companions"]["image"]:
        return Eligibility("unsupported", "This artifact has no matching image-input projector.")
    backend = {"llama.cpp": "native_llama_cpp", "xllamacpp": "xllamacpp"}.get(runtime.name)
    support = artifact.get("runtime", {}).get(backend, {})
    source_build = support.get("tested_source_build", support.get("bundled_build"))
    if runtime.build is None or (
        runtime.build != source_build
        and not _audited_native_build(artifact, runtime, task, source_build)
    ):
        return Eligibility(
            "unknown",
            f"{runtime.name}: this build is not covered by the source audit ({runtime.origin}).",
        )
    state = support.get("chat" if task == "text" else "image", "unknown")
    if (
        state.startswith(("unsupported", "blocked_"))
        or support.get("status") == "unsupported_architecture"
    ):
        return Eligibility(
            "unsupported",
            support.get("reason", f"{task} input is unsupported by this runtime build."),
        )
    if state == "source_supported":
        return Eligibility(
            "source_supported",
            f"Source support at {runtime.build}; device execution and quality remain unverified.",
        )
    return Eligibility(
        "unknown", support.get("reason", "Task support is not established for this build.")
    )


def architecture_eligibility(architecture, runtime):
    """Check actual local GGUF architecture without trusting a renamed filename."""
    if architecture in ("clip", "mmproj"):
        return Eligibility(
            "unsupported", "A vision projector is not a standalone Chat language model."
        )
    try:
        candidates = [
            row
            for row in load_registry().entries(include_proposed=True)
            if row["chat_weight"] and row.get("architecture") == architecture
        ]
    except RegistryError as error:
        return Eligibility("unknown", str(error))
    results = [task_eligibility(row, runtime) for row in candidates]
    if results and all(row.status == "unsupported" for row in results):
        return results[0]
    if any(row.status == "source_supported" for row in results):
        return Eligibility(
            "source_supported",
            f"Architecture source support at {runtime.build}; artifact and device unverified.",
        )
    return Eligibility("unknown", "Local architecture is not covered by this runtime source audit.")


def catalogue_artifact(name, downloads):
    """A catalogue label is not proof that a same-named local file is identical."""
    name = str(name)
    keys = [name] if name in downloads else ["llm/" + name, "vision/" + name]
    try:
        registry = load_registry()
    except RegistryError:
        return None
    for key in keys:
        if key in downloads:
            try:
                return registry.annotations_for(key, downloads[key])
            except RegistryError:
                return None
    return None


def selection_eligibility(name, downloads, runtime, task="text"):
    try:
        load_registry()
    except RegistryError as error:
        return Eligibility("unknown", str(error))
    return task_eligibility(catalogue_artifact(name, downloads), runtime, task)


def verified_local_artifact(path, downloads):
    """Only exact published content can inherit per-artifact settings/projectors."""
    from ruinedfooocus.core.paths import data_path
    from ruinedfooocus.features.models.model_identity import cached_file_sha256

    path = Path(path)
    row = catalogue_artifact(path.name, downloads)
    if (
        row is None
        or path.stat().st_size != row["artifact"]["bytes"]
        or not row["artifact"]["sha256"]
    ):
        return None
    checksum = cached_file_sha256(path, data_path("cache", "hashes"))
    return row if checksum.lower() == row["artifact"]["sha256"] else None


@lru_cache(maxsize=32)
def _gguf_architecture(path, signature):
    from ruinedfooocus.features.models.gguf_metadata import read_scalar_strings

    return read_scalar_strings(path, ("general.architecture",)).get("general.architecture")


def local_architecture(path):
    from ruinedfooocus.features.models.model_identity import file_signature

    signature = tuple(file_signature(path))
    architecture = _gguf_architecture(str(Path(path).resolve()), signature)
    if signature != tuple(file_signature(path)):
        raise ValueError("Model changed during metadata inspection; retry when stable.")
    return architecture
