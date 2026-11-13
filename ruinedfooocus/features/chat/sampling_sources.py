"""Bounded local Chat source hints; hints never authorize arbitrary URL requests."""

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import re
import time
from urllib.parse import urlparse

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.paths import data_path

MAX_FIELD_BYTES = 16 * 1024
SCHEMA = 2
SOURCE_FIELDS = (
    "general.architecture",
    "general.type",
    "general.name",
    "general.basename",
    "general.finetune",
    "general.size_label",
    "general.source.url",
    "general.source.huggingface.repository",
    "general.base_model.0.repo_url",
    "general.base_model.1.repo_url",
    "general.base_model.2.repo_url",
)
REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def huggingface_repository(value):
    """Only a provider repository identifier, never an executable/network URL."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    if REPOSITORY.fullmatch(value):
        return value
    url = urlparse(value)
    if url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment:
        return None
    parts = url.path.strip("/").split("/")
    repository = "/".join(parts[:2])
    return repository if REPOSITORY.fullmatch(repository) else None


@lru_cache(maxsize=128)
def _gguf_source_metadata(path, signature):
    from ruinedfooocus.features.models.gguf_metadata import read_scalar_strings

    return read_scalar_strings(path, SOURCE_FIELDS, max_string_bytes=MAX_FIELD_BYTES)


def gguf_source_metadata(path):
    """Read selected scalar metadata only, cached by current resolved identity."""
    signature = tuple(file_signature(path))
    result = _gguf_source_metadata(signature[0], signature)
    if signature != tuple(file_signature(path)):
        raise ValueError("Model changed during source inspection; refresh when stable.")
    return dict(result)


def source_repositories(metadata):
    result = []
    for key in SOURCE_FIELDS:
        if key.endswith((".url", ".repo_url", ".repository")):
            repository = huggingface_repository(metadata.get(key))
            if repository and repository not in result:
                result.append(repository)
    return result[:4]


def source_contract(source):
    source = source if isinstance(source, dict) else {}
    return tuple(
        source.get(key)
        for key in (
            "status",
            "sha256",
            "artifact_id",
            "profile_id",
            "variant_repository",
            "variant_revision",
        )
    )


def offline_mode(offline=None):
    from ruinedfooocus.app.arguments import args

    if offline is None:
        import sys

        shared = sys.modules.get("ruinedfooocus.app.state")
        values = getattr(getattr(shared, "settings", None), "default_settings", {})
        offline = bool(values.get("local_model_metadata")) or bool(
            getattr(shared, "models_offline", False)
        )
    return (
        bool(offline)
        or bool(getattr(args, "offline", False))
        or any(
            os.environ.get(name, "").lower() in ("1", "true", "yes")
            for name in ("RF_OFFLINE", "HF_HUB_OFFLINE")
        )
    )


def is_projector(path):
    if "mmproj" in Path(path).stem.casefold():
        return True
    try:
        metadata = gguf_source_metadata(path)
    except (OSError, ValueError, TypeError, ImportError, AttributeError, KeyError):
        return False
    return (
        metadata.get("general.architecture") in ("clip", "mmproj")
        or metadata.get("general.type") == "projector"
    )


def _cache_file(path, cache_dir):
    identity = str(Path(path).resolve())
    return Path(cache_dir or data_path("cache", "chat_sources")) / (
        hashlib.sha256(identity.encode()).hexdigest() + ".json"
    )


def cached_sampling_source(path, *, cache_dir=None):
    """Current cached provenance only; browsing/generation never fetch or hash."""
    try:
        target = _cache_file(path, cache_dir)
        if target.stat().st_size > 256 * 1024:
            return {}
        value = json.loads(target.read_text(encoding="utf-8"))
        if (
            isinstance(value, dict)
            and value.get("schema") == SCHEMA
            and value.get("signature") == file_signature(path)
            and value.get("status")
            in ("reviewed", "reviewed_variant", "source_unverified", "not_found", "offline")
            and isinstance(value.get("repositories", []), list)
            and len(value.get("repositories", [])) <= 4
            and all(huggingface_repository(repo) == repo for repo in value.get("repositories", []))
            and isinstance(value.get("observations", []), list)
            and len(value.get("observations", [])) <= 4
        ):
            if value["status"] == "reviewed_variant" and not verified_sampling_profile(value):
                return {}
            if value["status"] == "reviewed" and reviewed_source_artifact(path, value) is None:
                return {}
            return value
    except (OSError, ValueError, TypeError):
        pass
    return {}


def reviewed_source_artifact(path, source):
    """A renamed GGUF inherits defaults only from a current exact Chat artifact."""
    from ruinedfooocus.features.models.model_registry import load_registry, RegistryError

    if not isinstance(source, dict) or source.get("status") != "reviewed":
        return None
    try:
        signature = file_signature(path)
        if source.get("signature") != signature:
            return None
        for row in load_registry().entries(role="language_model"):
            if (
                row["role"] == "language_model"
                and row["chat_weight"]
                and row["id"] == source.get("artifact_id")
                and row["profile_id"] == source.get("profile_id")
                and row["artifact"]["bytes"] == signature[1]
                and row["artifact"]["sha256"] == source.get("sha256")
            ):
                return row
    except (OSError, RegistryError, KeyError, TypeError):
        pass
    return None


def _pinned_artifact_source(row):
    url = urlparse(row.get("artifact", {}).get("pinned_url") or "")
    parts = url.path.strip("/").split("/")
    if (
        url.scheme == "https"
        and url.netloc == "huggingface.co"
        and len(parts) >= 5
        and parts[2] == "resolve"
        and REPOSITORY.fullmatch("/".join(parts[:2]))
        and re.fullmatch(r"[a-fA-F0-9]{40}", parts[3])
    ):
        return "/".join(parts[:2]), parts[3].lower()
    return None


def verified_sampling_profile(source):
    """A reviewed variant recipe grants sampling advice, never runtime capabilities."""
    from ruinedfooocus.features.models.model_registry import load_registry, RegistryError

    if not isinstance(source, dict) or source.get("status") != "reviewed_variant":
        return {}
    try:
        signature = source.get("signature")
        if (
            not isinstance(signature, list)
            or len(signature) != 4
            or file_signature(signature[0]) != signature
        ):
            return {}
        if not re.fullmatch(r"[a-f0-9]{64}", source.get("sha256", "")):
            return {}
        registry = load_registry()
        for row in registry.entries(role="language_model"):
            profile = registry.profile(row["profile_id"]) or {}
            if (
                row["chat_weight"]
                and row["profile_id"] == source.get("profile_id")
                and _pinned_artifact_source(row)
                == (source.get("variant_repository"), source.get("variant_revision"))
                and profile.get("identity", {}).get("gguf_architecture")
                == source.get("architecture")
            ):
                return profile
    except (OSError, ValueError, TypeError, RegistryError):
        pass
    return {}


def _refresh_sampling_source(path, *, offline=None, force=False, cache_dir=None):
    """Explicit import/load preparation I/O; never call from generation/rendering.

    Registered identities and exact content found in a reviewed immutable variant
    revision may apply packaged sampling advice. Repository/base_model hints and
    newly fetched generation configuration alone never establish that authority.
    """
    from ruinedfooocus.features.models.model_identity import cached_file_sha256
    from ruinedfooocus.features.models.model_registry import load_registry, RegistryError
    from ruinedfooocus.features.models.model_sources import huggingface_recommendations

    offline = offline_mode(offline)
    cached = cached_sampling_source(path, cache_dir=cache_dir)
    if cached and offline:
        return cached
    checked = cached.get("checked", 0)
    ttl = (
        86400
        if cached.get("status") in ("reviewed", "reviewed_variant", "source_unverified")
        else 300
    )
    if (
        cached
        and not force
        and (offline or type(checked) in (int, float) and 0 <= time.time() - checked < ttl)
    ):
        return cached
    signature = file_signature(path)
    try:
        registry = load_registry()
        weights = [row for row in registry.entries(role="language_model") if row["chat_weight"]]
        candidates = [row for row in weights if row["artifact"]["bytes"] == signature[1]]
    except RegistryError:
        registry, weights = None, []
        candidates = []
    try:
        metadata = gguf_source_metadata(path)
    except (
        OSError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
    ):
        metadata = {}
    repositories = source_repositories(metadata)
    value = dict(
        schema=SCHEMA,
        signature=signature,
        checked=time.time(),
        status="offline" if offline else "not_found",
        repositories=repositories,
    )
    if candidates or (repositories and not offline):
        target = _cache_file(path, cache_dir)
        checksum = cached_file_sha256(path, data_path("cache", "hashes")).lower()
        value["sha256"] = checksum
        matched = [row for row in candidates if row["artifact"]["sha256"] == checksum]
        if matched and len({row["profile_id"] for row in matched}) == 1:
            value.update(
                status="reviewed", artifact_id=matched[0]["id"], profile_id=matched[0]["profile_id"]
            )
        elif not offline:
            observations = []
            variants = []
            for repository in repositories:
                remote = huggingface_recommendations(repository, checksum)
                if remote is not None:
                    # Do not execute cards/templates or apply a quantizer's defaults.
                    from ruinedfooocus.features.chat.model_profiles import (
                        SAMPLERS,
                        validate_sampling,
                    )

                    config = remote.get("generationConfig", {})
                    values = {}
                    if isinstance(config, dict):
                        for name, raw in config.items():
                            key = "repeat_penalty" if name == "repetition_penalty" else name
                            if key in SAMPLERS:
                                try:
                                    values.update(validate_sampling({key: raw}))
                                except ValueError:
                                    pass
                    observations.append(
                        dict(
                            repository=repository,
                            source_url=remote.get("source_url"),
                            generation_config=values,
                        )
                    )
                    url = urlparse(remote.get("source_url") or "")
                    parts = url.path.strip("/").split("/")
                    if (
                        url.scheme == "https"
                        and url.netloc == "huggingface.co"
                        and len(parts) == 5
                        and parts[2] == "blob"
                    ):
                        origin = ("/".join(parts[:2]), parts[3].lower())
                        for row in weights:
                            profile = registry.profile(row["profile_id"]) or {}
                            if _pinned_artifact_source(row) == origin and profile.get(
                                "identity", {}
                            ).get("gguf_architecture") == metadata.get("general.architecture"):
                                variants.append((row["profile_id"], *origin))
            if observations:
                value.update(status="source_unverified", observations=observations)
                if len(set(variants)) == 1:
                    profile, repository, revision = variants[0]
                    value.update(
                        status="reviewed_variant",
                        profile_id=profile,
                        variant_repository=repository,
                        variant_revision=revision,
                        architecture=metadata.get("general.architecture"),
                    )
            elif cached and cached.get("status") in (
                "reviewed",
                "reviewed_variant",
                "source_unverified",
            ):
                # The shared resolver deliberately returns no advice on mismatch,
                # missing metadata or transport failure. None cannot erase a
                # previously current source observation or manufacture defaults.
                value = {**cached, "checked": time.time(), "last_lookup": "unavailable"}
    if signature != file_signature(path):
        raise ValueError("Model changed during source preparation; retry when stable.")
    save_json(_cache_file(path, cache_dir), value)
    return value


def refresh_sampling_source(path, *, offline=None, force=False, cache_dir=None):
    """Serialize one cache identity using the existing artifact lease service.

    The exclusive lease is on metadata, not the active model's weight file.
    Distinct model caches prepare independently. A later failed refresh re-reads
    any successful result before publishing its retained observation.
    """
    from ruinedfooocus.core.preparation import preparation_service

    with preparation_service.artifact_lease(_cache_file(path, cache_dir), write=True):
        return _refresh_sampling_source(path, offline=offline, force=force, cache_dir=cache_dir)


def prepare_sampling_source(path, **options):
    """Best-effort preparation failure never unloads or blocks a usable model."""
    from ruinedfooocus.core.preparation import preparation_service

    try:
        with preparation_service.artifact_lease(path):
            return refresh_sampling_source(path, **options)
    except (
        OSError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
    ):
        cached = cached_sampling_source(path, cache_dir=options.get("cache_dir"))
        return {**cached, "last_lookup": "unavailable"}
