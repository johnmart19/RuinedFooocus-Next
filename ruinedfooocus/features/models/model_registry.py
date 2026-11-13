"""Lazy model annotations, independent of tensor runtimes and download discovery.

Download tuples remain authoritative in model_catalogue. Annotations bind to an
exact tuple, never merely a basename or a familiar model label. Callers can show
RegistryError without changing saved selections or the working download list.
"""

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType

from ruinedfooocus.core.paths import resource_path


class RegistryError(ValueError):
    """Optional model metadata is unavailable or does not match its source."""


def unique_payload_bytes(entries):
    """Sum published payloads by checksum; unknown/conflicting sizes stay unknown."""
    payloads = {}
    for entry in entries:
        artifact = entry["artifact"]
        identity = artifact.get("sha256") or entry["id"]
        size = artifact["bytes"]
        if identity in payloads and payloads[identity] != size:
            return None
        payloads[identity] = size
    return None if any(size is None for size in payloads.values()) else sum(payloads.values())


def download_identity(entry):
    """Bind annotations to the existing downloader contract, including projector."""
    fields = {key: entry[key] for key in ("url", "path", "filename", "mmproj") if key in entry}
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError(f"Duplicate model metadata key: {key}")
        result[key] = value
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _require(condition, message):
    if not condition:
        raise RegistryError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_artifact(entry, profiles):
    """Validate one artifact without loading other artifacts or qualification."""
    _require(isinstance(entry, Mapping), "Invalid model artifact.")
    identity, key = entry.get("id"), entry.get("catalogue_key")
    _require(_text(identity) and _text(key), "Missing model artifact identity.")
    _require(entry.get("state") in ("existing", "proposed"), f"Invalid state: {key}")
    _require(identity == f"{entry['state']}:{key}", f"Inconsistent model identity: {key}")
    _require(
        _text(entry.get("profile_id")) and entry["profile_id"] in profiles,
        f"Unknown model profile: {key}",
    )
    _require(_text(entry.get("role")), f"Missing semantic role: {key}")
    _require(type(entry.get("chat_weight")) is bool, f"Invalid chat role: {key}")
    _require(
        not entry["chat_weight"] or entry["role"] == "language_model",
        f"A component cannot be a Chat language weight: {key}",
    )
    _require(_sha256(entry.get("download_identity")), f"Invalid download identity: {key}")
    artifact = entry.get("artifact")
    _require(isinstance(artifact, Mapping), f"Missing artifact evidence: {key}")
    size, checksum = artifact.get("bytes"), artifact.get("sha256")
    _require(size is None or (type(size) is int and size >= 0), f"Invalid bytes: {key}")
    _require(checksum is None or _sha256(checksum), f"Invalid artifact hash: {key}")
    candidate = entry.get("proposed_download")
    if candidate is not None:
        _require(
            entry["state"] == "proposed"
            and isinstance(candidate, Mapping)
            and all(_text(candidate.get(field)) for field in ("url", "path", "filename"))
            and download_identity(candidate) == entry["download_identity"],
            f"Invalid proposed download: {key}",
        )


def _validate_companions(entry, artifacts):
    dependencies = entry.get("companions", {})
    _require(isinstance(dependencies, Mapping), "Invalid companion mapping.")
    for mode in ("text", "image"):
        companions = dependencies.get(mode, ())
        _require(
            isinstance(companions, (tuple, list))
            and all(
                _text(name) and name in artifacts and name != entry["id"] for name in companions
            )
            and len(companions) == len(set(companions)),
            f"Invalid {mode} companions: {entry['catalogue_key']}",
        )


def _validate_benchmark(record, profiles):
    _require(isinstance(record, Mapping), "Invalid benchmark record.")
    _require(
        _text(record.get("profile_id")) and record["profile_id"] in profiles,
        "Unknown benchmark profile.",
    )
    _require(
        record.get("value") is None
        or (type(record["value"]) in (int, float) and math.isfinite(record["value"])),
        "Invalid numeric benchmark value.",
    )
    _require(
        record.get("higher_is_better") is None or type(record["higher_is_better"]) is bool,
        "Invalid benchmark direction.",
    )


class ModelRegistry:
    """Validated immutable metadata; all picker projections share this instance."""

    def __init__(self, document):
        _require(isinstance(document, dict), "Model metadata must be an object.")
        _require(type(document.get("schema_version")) is int, "Missing model metadata version.")
        _require(document["schema_version"] == 1, "Unsupported model metadata version.")
        profiles = document.get("profiles")
        artifacts = document.get("artifacts")
        benchmarks = document.get("benchmarks")
        _require(isinstance(profiles, dict), "Missing model profiles.")
        _require(isinstance(artifacts, list), "Missing model artifacts.")
        _require(isinstance(benchmarks, list), "Missing model benchmark records.")
        _require(
            all(_text(key) and isinstance(value, dict) for key, value in profiles.items()),
            "Invalid model profile.",
        )
        by_id, by_key = {}, {}
        for entry in artifacts:
            _validate_artifact(entry, profiles)
            identity, key = entry["id"], entry["catalogue_key"]
            _require(
                identity not in by_id and key not in by_key, f"Duplicate model identity: {key}"
            )
            by_id[identity] = entry
            by_key[key] = entry
        for entry in artifacts:
            _validate_companions(entry, by_id)
        alias_groups = document.get("payload_alias_groups", [])
        _require(isinstance(alias_groups, list), "Invalid payload aliases.")
        for group in alias_groups:
            _require(isinstance(group, dict), "Invalid payload alias group.")
            members = group.get("artifact_ids", [])
            _require(
                isinstance(members, list)
                and len(members) > 1
                and all(_text(name) for name in members)
                and len(members) == len(set(members))
                and _sha256(group.get("sha256"))
                and all(
                    name in by_id and by_id[name]["artifact"]["sha256"] == group.get("sha256")
                    for name in members
                ),
                "Invalid shared-payload identity.",
            )
        for record in benchmarks:
            _validate_benchmark(record, profiles)
        self._document = _freeze(document)
        self._by_id = MappingProxyType(
            {entry["id"]: entry for entry in self._document["artifacts"]}
        )
        self._by_key = MappingProxyType(
            {entry["catalogue_key"]: entry for entry in self._document["artifacts"]}
        )

    @property
    def provenance(self):
        return self._document.get("provenance", MappingProxyType({}))

    def entries(self, *, role=None, include_proposed=False):
        return tuple(
            entry
            for entry in self._document["artifacts"]
            if (include_proposed or entry["state"] == "existing")
            and (role is None or entry["role"] == role)
        )

    def artifact(self, identity):
        return self._by_id.get(identity)

    def profile(self, identity):
        return self._document["profiles"].get(identity)

    def benchmarks(self, profile_id):
        return tuple(row for row in self._document["benchmarks"] if row["profile_id"] == profile_id)

    def annotations_for(self, catalogue_key, download_entry):
        entry = self._by_key.get(catalogue_key)
        if entry is not None and download_identity(download_entry) != entry["download_identity"]:
            raise RegistryError(f"Model metadata no longer matches download: {catalogue_key}")
        return entry

    def to_document(self):
        """Materialize legacy schema 1 for audits/migration, including qualification."""
        return _plain(self._document)

    def native_dispositions(self, *, kind="configurations", media=None):
        """Read optional pinned-native rows through the same registry interface."""
        from ruinedfooocus.features.models.native_catalogue import dispositions

        return dispositions(self, getattr(self, "_root", None), kind=kind, media=media)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _read_document(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise RegistryError(f"Model metadata unavailable: {error}") from error


def metadata_digest(document):
    """Stable JSON identity independent of source formatting and object key order."""
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=512)
def _read_shard(path, modified, size, digest):
    try:
        data = Path(path).read_bytes()
        document = json.loads(data, object_pairs_hook=_unique_object)
        _require(metadata_digest(document) == digest, f"Changed model metadata shard: {path}")
        _require(isinstance(document, dict), f"Invalid model metadata shard: {path}")
        _require(
            type(document.get("schema_version")) is int and document["schema_version"] == 1,
            f"Unsupported metadata shard version: {path}",
        )
        return _freeze(document)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise RegistryError(f"Model metadata unavailable: {error}") from error


class _ProfileView(Mapping):
    """Keep legacy profile keys while delaying exact-artifact evidence reads."""

    def __init__(self, registry, identity, profile):
        self.registry, self.identity, self.profile = registry, identity, profile

    def __getitem__(self, key):
        if key == "qualification" and self.identity in self.registry._index["qualification"]:
            record = self.registry._qualification(self.identity)
            if "qualification" in record:
                return record["qualification"]
        return self.profile[key]

    def __iter__(self):
        return iter(self.registry._index["profiles"][self.identity]["keys"])

    def __len__(self):
        return len(self.registry._index["profiles"][self.identity]["keys"])


class IndexedModelRegistry(ModelRegistry):
    """One public registry over small, content-addressed, lazily read resources."""

    def __init__(self, document, root):
        _require(isinstance(document, dict), "Model metadata index must be an object.")
        _require(
            type(document.get("schema_version")) is int and document["schema_version"] == 2,
            "Unsupported model metadata version.",
        )
        for key in ("profiles", "artifacts", "qualification", "shards"):
            _require(isinstance(document.get(key), dict), f"Missing metadata index: {key}")
        _require(isinstance(document.get("benchmark_order"), list), "Missing benchmark order.")
        self._root = Path(root).resolve()
        self._index = _freeze(document)
        self._by_key = {}
        for path, descriptor in document["shards"].items():
            _require(_text(path) and isinstance(descriptor, dict), "Invalid metadata shard.")
            # Validate names without resolving every deferred resource. Actual
            # reads still resolve and check containment, including directory links.
            name = PurePosixPath(path)
            _require(
                not name.is_absolute()
                and ".." not in name.parts
                and "\\" not in path
                and ":" not in path,
                "Metadata shard escapes resource root.",
            )
            _require(_sha256(descriptor.get("sha256")), "Missing metadata shard checksum.")
        for identity, descriptor in document["profiles"].items():
            _require(_text(identity) and isinstance(descriptor, dict), "Invalid profile index.")
            _require(
                _text(descriptor.get("path")) and descriptor["path"] in document["shards"],
                "Missing family shard.",
            )
            keys = descriptor.get("keys")
            _require(
                isinstance(keys, list)
                and all(_text(key) for key in keys)
                and len(keys) == len(set(keys)),
                "Invalid profile fields.",
            )
        for identity, descriptor in document["artifacts"].items():
            _require(isinstance(descriptor, dict), "Invalid artifact index.")
            key, state = descriptor.get("catalogue_key"), descriptor.get("state")
            _require(_text(key) and key not in self._by_key, f"Duplicate model identity: {key}")
            _require(
                state in ("existing", "proposed") and identity == f"{state}:{key}",
                "Invalid artifact identity.",
            )
            _require(
                _text(descriptor.get("profile_id"))
                and descriptor["profile_id"] in document["profiles"],
                "Unknown model profile.",
            )
            _require(_text(descriptor.get("role")), "Missing semantic role.")
            _require(
                _text(descriptor.get("path")) and descriptor["path"] in document["shards"],
                "Missing artifact shard.",
            )
            self._by_key[key] = identity
        for identity, path in document["qualification"].items():
            _require(
                identity in document["profiles"] and _text(path) and path in document["shards"],
                "Invalid qualification reference.",
            )
        _require(
            all(
                _text(identity) and identity in document["qualification"]
                for identity in document["benchmark_order"]
            ),
            "Invalid benchmark reference.",
        )
        groups = document.get("payload_alias_groups", [])
        _require(isinstance(groups, list), "Invalid payload aliases.")
        for group in groups:
            _require(isinstance(group, dict), "Invalid payload alias group.")
            members = group.get("artifact_ids")
            _require(
                isinstance(members, list)
                and len(members) > 1
                and all(_text(name) and name in document["artifacts"] for name in members)
                and len(members) == len(set(members))
                and _sha256(group.get("sha256")),
                "Invalid shared-payload identity.",
            )

    def _shard(self, name):
        descriptor = self._index["shards"][name]
        path = (self._root / name).resolve()
        _require(path.is_relative_to(self._root), "Metadata shard escapes resource root.")
        try:
            stat = path.stat()
        except OSError as error:
            raise RegistryError(f"Model metadata unavailable: {path}") from error
        return _read_shard(str(path), stat.st_mtime_ns, stat.st_size, descriptor["sha256"])

    @property
    def provenance(self):
        return self._index.get("provenance", MappingProxyType({}))

    def entries(self, *, role=None, include_proposed=False):
        shards, result = {}, []
        for identity, descriptor in self._index["artifacts"].items():
            if (include_proposed or descriptor["state"] == "existing") and (
                role is None or descriptor["role"] == role
            ):
                path = descriptor["path"]
                if path not in shards:
                    shards[path] = self._shard(path).get("artifacts")
                result.append(self._entry(identity, shards[path]))
        return tuple(result)

    def _entry(self, identity, entries):
        descriptor = self._index["artifacts"][identity]
        _require(
            isinstance(entries, Mapping) and identity in entries,
            f"Missing model artifact: {identity}",
        )
        entry = entries[identity]
        _validate_artifact(entry, self._index["profiles"])
        _require(entry["id"] == identity, f"Inconsistent artifact identity: {identity}")
        _require(
            all(
                entry.get(key) == descriptor[key]
                for key in ("catalogue_key", "state", "role", "profile_id")
            ),
            f"Inconsistent artifact index: {identity}",
        )
        _validate_companions(entry, self._index["artifacts"])
        for group in self._index.get("payload_alias_groups", ()):
            if identity in group["artifact_ids"]:
                _require(
                    entry["artifact"]["sha256"] == group["sha256"],
                    "Invalid shared-payload identity.",
                )
        return entry

    def artifact(self, identity):
        descriptor = self._index["artifacts"].get(identity)
        return (
            self._entry(identity, self._shard(descriptor["path"]).get("artifacts"))
            if descriptor
            else None
        )

    def profile(self, identity):
        descriptor = self._index["profiles"].get(identity)
        if descriptor is None:
            return None
        profiles = self._shard(descriptor["path"]).get("profiles")
        _require(isinstance(profiles, Mapping), "Invalid family shard.")
        profile = profiles.get(identity)
        _require(isinstance(profile, Mapping), f"Missing model profile: {identity}")
        keys = set(descriptor["keys"])
        has_qualification = "qualification" in keys
        _require(
            not has_qualification or identity in self._index["qualification"],
            f"Missing profile qualification: {identity}",
        )
        _require(
            set(profile) == keys - ({"qualification"} if has_qualification else set()),
            f"Missing or inconsistent profile fields: {identity}",
        )
        return _ProfileView(self, identity, profile)

    def _qualification(self, identity):
        path = self._index["qualification"].get(identity)
        if not path:
            return MappingProxyType({})
        document = self._shard(path)
        _require(
            document.get("profile_id") == identity,
            f"Inconsistent qualification profile: {identity}",
        )
        if "qualification" in self._index["profiles"][identity]["keys"]:
            _require("qualification" in document, f"Missing profile qualification: {identity}")
        records = document.get("benchmarks", ())
        _require(isinstance(records, (tuple, list)), "Invalid benchmark records.")
        for record in records:
            _validate_benchmark(record, self._index["profiles"])
            _require(record["profile_id"] == identity, "Inconsistent benchmark profile.")
        return document

    def benchmarks(self, profile_id):
        return self._qualification(profile_id).get("benchmarks", ())

    def annotations_for(self, catalogue_key, download_entry):
        entry = self.artifact(self._by_key.get(catalogue_key))
        if entry is not None and download_identity(download_entry) != entry["download_identity"]:
            raise RegistryError(f"Model metadata no longer matches download: {catalogue_key}")
        return entry

    def to_document(self):
        cursors, benchmarks = {}, []
        for identity in self._index["benchmark_order"]:
            cursor = cursors.get(identity, 0)
            rows = self.benchmarks(identity)
            _require(cursor < len(rows), "Missing qualification record.")
            benchmarks.append(_plain(rows[cursor]))
            cursors[identity] = cursor + 1
        return {
            "schema_version": 1,
            "provenance": _plain(self.provenance),
            "profiles": {
                identity: _plain(self.profile(identity)) for identity in self._index["profiles"]
            },
            "artifacts": [_plain(row) for row in self.entries(include_proposed=True)],
            "benchmarks": benchmarks,
            "payload_alias_groups": _plain(self._index.get("payload_alias_groups", ())),
        }


@lru_cache(maxsize=2)
def _load(path, modified, size):
    try:
        document = _read_document(path)
        _require(isinstance(document, dict), "Model metadata must be an object.")
        if document.get("schema_version") == 2:
            return IndexedModelRegistry(document, Path(path).parent)
        registry = ModelRegistry(document)
        registry._root = Path(path).parent
        return registry
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise RegistryError(f"Model metadata unavailable: {error}") from error


def load_registry(path=None):
    """Read only packaged metadata; changed files invalidate the bounded cache."""
    path = Path(path) if path is not None else resource_path("model_metadata")
    if path.is_dir():
        path = path / ("index.json" if (path / "index.json").exists() else "catalogue.json")
    try:
        stat = path.stat()
    except OSError as error:
        raise RegistryError(f"Model metadata unavailable: {path}") from error
    return _load(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def registry_document(path=None):
    """Audit-only schema-1 projection of an explicitly selected resource root."""
    document = load_registry(path).to_document()
    ModelRegistry(document)
    return document
