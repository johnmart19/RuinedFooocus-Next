"""Owned artifact-backed image history; execution remains in existing services."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import uuid

from concurrent.futures import CancelledError

from ruinedfooocus.core.cache_contract import configured_cache_root
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.preparation import preparation_service
from ruinedfooocus.features.models.model_identity import file_signature

SCHEMA_VERSION = 1
KEY = "_editor_operation"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 128 * 1024
_LOCK = threading.RLock()
_STORES = {}
_SESSION_ID = re.compile(r"[0-9a-f]{32}")
_ARTIFACT_ID = re.compile(r"(?:image|mask)_[0-9a-f]{32}")


def _json(value, *, limit=MAX_METADATA_BYTES):
    """Accept bounded JSON values, never a mutable pixel/model/runtime object."""

    def check(item, depth=0):
        if depth > 16:
            raise ValueError("Editor metadata is too deeply nested.")
        if item is None or type(item) in (str, int, float, bool):
            return
        if type(item) in (list, tuple):
            if len(item) > 1024:
                raise ValueError("Editor metadata contains too many values.")
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict and len(item) <= 512 and all(type(k) is str for k in item):
            for child in item.values():
                check(child, depth + 1)
            return
        raise ValueError("Editor parameters must contain only bounded JSON values.")

    check(value)
    result = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if len(result.encode("utf-8")) > limit:
        raise ValueError("Editor metadata exceeds its bounded size.")
    return result


def _owner_key(owner=None):
    from ruinedfooocus.app.resource_lifecycle import current_owner

    owner = current_owner() if owner is None else owner
    if not isinstance(owner, tuple) or not owner or any(type(part) is not str for part in owner):
        raise PermissionError("An editor needs an authenticated session owner.")
    return hashlib.sha256(_json(owner).encode("utf-8")).hexdigest()


def _mapping_json(value, label):
    if value is not None and type(value) is not dict:
        raise ValueError(f"Editor {label} must be a JSON mapping.")
    return _json({} if value is None else value)


def _public_metadata(value, key=""):
    """Redact explicit local-path fields while preserving literal prompt text."""
    if isinstance(value, dict):
        return {
            name: _public_metadata(item, name)
            for name, item in value.items()
            if name not in ("path", "paths", "canonical_path", "destination", "owner_key")
            and not name.endswith(("_path", "_paths"))
        }
    if isinstance(value, (list, tuple)):
        values = value
        if (
            "signature" in key
            and values
            and isinstance(values[0], str)
            and Path(values[0]).is_absolute()
        ):
            values = values[1:]
        return [_public_metadata(item, key) for item in values]
    return value


@dataclass(frozen=True)
class ImageFrame:
    width: int
    height: int
    mode: str
    dtype: str
    color_space: str
    alpha: str
    orientation: int = 1

    @property
    def size(self):
        return self.width, self.height


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    path: str
    sha256: str
    signature: tuple
    frame: ImageFrame
    frame_id: str
    source_image_id: str | None = None
    source_frame_id: str | None = None
    purpose: str = "image"
    channel: str = ""
    inverted: bool = False

    @property
    def image_id(self):
        return self.artifact_id if self.kind == "image" else None

    @property
    def mask_id(self):
        return self.artifact_id if self.kind == "mask" else None

    def to_dict(self, *, public=True):
        value = asdict(self)
        value["signature"] = list(self.signature)
        if public:
            value.pop("path")
            value["signature"] = list(self.signature[1:])
        return value


def _artifact(value):
    return ArtifactRef(
        **(value | {"signature": tuple(value["signature"]), "frame": ImageFrame(**value["frame"])})
    )


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    revision: int
    original: ArtifactRef
    current: ArtifactRef
    head_id: str
    source_generation_json: str
    steps_json: str
    expires_at: float
    derived_results_json: str = "null"

    @property
    def source_generation(self):
        return json.loads(self.source_generation_json)

    @property
    def steps(self):
        return tuple(json.loads(self.steps_json))

    @property
    def derived_results(self):
        return json.loads(self.derived_results_json)

    def to_dict(self, *, public=True):
        result = dict(
            schema_version=SCHEMA_VERSION,
            session_id=self.session_id,
            revision=self.revision,
            original=self.original.to_dict(public=public),
            current=self.current.to_dict(public=public),
            head_id=self.head_id,
            source_generation=self.source_generation,
            steps=list(self.steps),
            expires_at=self.expires_at,
            derived_results=self.derived_results,
        )
        return _public_metadata(result) if public else result


@dataclass(frozen=True)
class OperationTicket:
    session_id: str
    revision: int
    owner_key: str
    operation_id: str
    operation: str
    source: ArtifactRef
    mask: ArtifactRef | None
    additional_images: tuple
    parameters_json: str
    effective_plan_json: str

    @property
    def source_id(self):
        return self.source.artifact_id

    @property
    def mask_id(self):
        return self.mask.artifact_id if self.mask is not None else None

    @property
    def parameters(self):
        return json.loads(self.parameters_json)

    @property
    def effective_plan(self):
        return json.loads(self.effective_plan_json)

    def metadata(self):
        return dict(
            operation_id=self.operation_id,
            operation=self.operation,
            source_id=self.source_id,
            mask_id=self.mask_id,
            additional_image_ids=[item.artifact_id for item in self.additional_images],
            parameters=self.parameters,
            effective_plan=self.effective_plan,
        )


@dataclass(frozen=True)
class PreparedDerivedSelection:
    session_id: str
    revision: int
    owner_key: str
    operation_id: str
    source: ArtifactRef
    results: tuple
    mask: ArtifactRef


def _describe(path):
    from PIL import Image
    from ruinedfooocus.features.image_processing.native_image_formats import (
        MAX_BYTES,
        MAX_PIXELS,
        needs_preview,
        read_image,
    )

    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose an existing bounded still-image artifact.")
    if needs_preview(path):
        imported = read_image(path)
        pixels = imported.pixels
        height, width = pixels.shape[:2]
        channels = pixels.shape[2] if len(pixels.shape) == 3 else 1
        mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
        if mode is None:
            raise ValueError("The editor source has an unsupported channel frame.")
        frame = ImageFrame(
            width,
            height,
            mode,
            str(pixels.dtype),
            imported.color_space,
            "straight" if channels == 4 else "none",
        )
    else:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Choose one still image for the editor.")
            orientation = image.getexif().get(274, 1)
            width, height = image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
            alpha = "A" in image.getbands() or "transparency" in image.info
            dtype = "float32" if image.mode == "F" else ("int32" if image.mode == "I" else "uint8")
            frame = ImageFrame(
                width, height, image.mode, dtype, "sRGB", "straight" if alpha else "none"
            )
    if not 0 < frame.width * frame.height <= MAX_PIXELS:
        raise ValueError("The editor source exceeds the still-image pixel limit.")
    return frame


def describe_frame(path):
    """Inspect an existing trusted result without changing precision or inventing policy."""
    path = Path(path).resolve()
    with preparation_service.artifact_lease(path):
        signature = tuple(file_signature(path))
        frame = _describe(path)
        if tuple(file_signature(path)) != signature:
            raise ValueError("The editor artifact changed during frame inspection.")
        return frame


class EditorSessionStore:
    """Small atomic manifests with existing artifact leases, no raster copies or DB."""

    def __init__(
        self, root, *, max_steps=32, max_sessions=64, ttl_seconds=7 * 86400, clock=time.time
    ):
        if type(max_steps) is not int or not 1 <= max_steps <= 64:
            raise ValueError("Editor history needs one through 64 steps.")
        if type(max_sessions) is not int or not 1 <= max_sessions <= 256:
            raise ValueError("Editor session capacity must be one through 256.")
        if not 0 < ttl_seconds <= 30 * 86400:
            raise ValueError("Editor sessions expire within 30 days.")
        self.root = Path(root).resolve()
        self.max_steps, self.max_sessions = max_steps, max_sessions
        self.max_artifacts = max_steps * 4 + 8
        self.ttl_seconds, self.clock = ttl_seconds, clock

    def _path(self, session_id):
        if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Invalid editor session handle.")
        return self.root / (session_id + ".json")

    @contextmanager
    def _edit(self, session_id, owner):
        path = self._path(session_id)
        key = _owner_key(owner)
        with _LOCK, preparation_service.artifact_lease(path, write=True):
            try:
                with path.open("rb") as stream:
                    raw = stream.read(MAX_MANIFEST_BYTES + 1)
                value = json.loads(raw) if len(raw) <= MAX_MANIFEST_BYTES else None
            except (OSError, ValueError, UnicodeError) as error:
                raise ValueError("This editor session is unavailable; reopen the image.") from error
            if (
                not isinstance(value, dict)
                or type(value.get("schema_version")) is not int
                or value["schema_version"] != SCHEMA_VERSION
            ):
                raise ValueError("This editor history requires its matching application version.")
            if value.get("owner_key") != key:
                raise PermissionError("This editor belongs to a different session.")
            if value.get("session_id") != session_id or value.get("expires_at", 0) <= self.clock():
                raise ValueError("This editor session expired; reopen the image.")
            if (
                not isinstance(value.get("artifacts"), dict)
                or len(value["artifacts"]) > self.max_artifacts
            ):
                raise ValueError("This editor history has an incompatible artifact limit.")
            if (
                type(value.get("revision")) is not int
                or not 0 < value["revision"] < 2**53
                or not isinstance(value.get("steps"), list)
                or len(value["steps"]) > self.max_steps
            ):
                raise ValueError("This editor history has an incompatible revision or step limit.")
            yield value

    def _save(self, value):
        _json(value, limit=MAX_MANIFEST_BYTES)
        save_json(self._path(value["session_id"]), value)

    def _ref(self, path, *, kind="image", source=None, inverted=False):
        path = Path(path).resolve()
        with preparation_service.artifact_lease(path):
            signature = tuple(file_signature(path))
            frame = _describe(path)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if tuple(file_signature(path)) != signature:
                raise ValueError("The editor source changed during capture.")
        if kind == "mask" and (
            frame.mode not in ("L", "1")
            or frame.dtype != "uint8"
            or source is None
            or frame.size != source.frame.size
        ):
            raise ValueError(
                "A selection mask must be 8-bit single-channel and match its source frame."
            )
        artifact_id = kind + "_" + uuid.uuid4().hex
        frame_id = hashlib.sha256((artifact_id + _json(asdict(frame))).encode()).hexdigest()
        return ArtifactRef(
            artifact_id,
            kind,
            str(path),
            digest,
            signature,
            frame,
            frame_id,
            source.artifact_id if source else None,
            source.frame_id if source else None,
            "selection-mask" if kind == "mask" else "image",
            "L" if kind == "mask" else "",
            inverted,
        )

    @staticmethod
    def _get(value, artifact_id, *, kind=None):
        if not isinstance(artifact_id, str) or _ARTIFACT_ID.fullmatch(artifact_id) is None:
            raise ValueError("Invalid editor image or mask handle.")
        raw = value["artifacts"].get(artifact_id)
        if raw is None:
            raise ValueError("This editor artifact expired or belongs to another session.")
        ref = _artifact(raw)
        if kind is not None and ref.kind != kind:
            raise ValueError("The editor handle has the wrong image or mask type.")
        try:
            unchanged = tuple(file_signature(ref.path)) == ref.signature
        except OSError:
            unchanged = False
        if not unchanged:
            raise ValueError("The editor source changed or was removed; reopen the image.")
        return ref

    @staticmethod
    def _revision(value, revision):
        if type(revision) is not int or revision != value["revision"]:
            raise CancelledError("A newer editor revision replaced this operation.")

    def _snapshot(self, value):
        return SessionSnapshot(
            value["session_id"],
            value["revision"],
            self._get(value, value["original_id"]),
            self._get(value, value["head_id"]),
            value["head_id"],
            _json(value["source_generation"]),
            _json(value["steps"], limit=MAX_MANIFEST_BYTES),
            value["expires_at"],
            _json(value.get("derived_results")),
        )

    def start(self, path, *, provenance=None, owner=None):
        key, provenance = _owner_key(owner), json.loads(_mapping_json(provenance, "provenance"))
        ref = self._ref(path)
        with _LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            manifests = [
                path for path in self.root.glob("*.json") if _SESSION_ID.fullmatch(path.stem)
            ]
            # Only owned private manifests expire. Source/output files are never removed.
            for path in manifests:
                try:
                    with path.open("rb") as stream:
                        raw = stream.read(MAX_MANIFEST_BYTES + 1)
                    previous = json.loads(raw) if len(raw) <= MAX_MANIFEST_BYTES else {}
                    expired = (
                        previous.get("schema_version") == SCHEMA_VERSION
                        and previous.get("expires_at", 0) <= self.clock()
                    )
                except (OSError, ValueError, UnicodeError, TypeError):
                    expired = False
                if expired:
                    path.unlink(missing_ok=True)
            if sum(path.exists() for path in manifests) >= self.max_sessions:
                raise ValueError("The editor session limit is full; expired histories release it.")
            value = dict(
                schema_version=SCHEMA_VERSION,
                session_id=uuid.uuid4().hex,
                owner_key=key,
                revision=1,
                expires_at=self.clock() + self.ttl_seconds,
                original_id=ref.artifact_id,
                head_id=ref.artifact_id,
                artifacts={ref.artifact_id: ref.to_dict(public=False)},
                source_generation=provenance,
                steps=[],
                redo=[],
                pending=None,
                last_outcome=None,
            )
            self._save(value)
            return self._snapshot(value)

    def snapshot(self, session_id, *, owner=None):
        with self._edit(session_id, owner) as value:
            return self._snapshot(value)

    def reopen(self, session_id, *, owner=None):
        """Reopen the same private manifest; pending work never resumes after restart."""
        with self._edit(session_id, owner) as value:
            result = self._snapshot(value)
            if value["pending"] is not None:
                value["pending"] = None
                self._save(value)
            return result

    def artifact(self, session_id, artifact_id, *, revision=None, owner=None):
        with self._edit(session_id, owner) as value:
            if revision is not None:
                self._revision(value, revision)
            return self._get(value, artifact_id)

    def _bind(self, session_id, path, *, owner, source_id=None, inverted=False):
        with self._edit(session_id, owner) as value:
            if len(value["artifacts"]) >= self.max_artifacts:
                raise ValueError("The editor artifact limit is full; open a new session.")
            source = self._get(value, source_id, kind="image") if source_id else None
            ref = self._ref(
                path, kind="mask" if source else "image", source=source, inverted=inverted
            )
            value["artifacts"][ref.artifact_id] = ref.to_dict(public=False)
            self._save(value)
            return ref

    def bind_image(self, session_id, path, *, owner=None):
        return self._bind(session_id, path, owner=owner)

    def bind_mask(self, session_id, path, image_id, *, inverted=False, owner=None):
        if type(inverted) is not bool:
            raise ValueError("Mask inversion must be explicit true or false.")
        return self._bind(session_id, path, owner=owner, source_id=image_id, inverted=inverted)

    def capture(
        self,
        session_id,
        revision,
        operation,
        parameters=None,
        *,
        source_id,
        mask_id=None,
        additional_image_ids=(),
        effective_plan=None,
        owner=None,
    ):
        if not isinstance(operation, str) or not 0 < len(operation) <= 80:
            raise ValueError("Choose one explicit editor operation.")
        if not isinstance(additional_image_ids, (tuple, list)) or len(additional_image_ids) > 32:
            raise ValueError("An editor operation has at most 32 additional images.")
        if len(additional_image_ids) + 3 + (mask_id is not None) > self.max_artifacts:
            raise ValueError("The editor operation exceeds this session's artifact limit.")
        parameters_json = _mapping_json(parameters, "parameters")
        plan_json = (
            _mapping_json(effective_plan, "effective plan")
            if effective_plan is not None
            else _json(None)
        )
        with self._edit(session_id, owner) as value:
            self._revision(value, revision)
            self._get(value, value["original_id"], kind="image")
            if source_id != value["head_id"]:
                raise CancelledError("The operation must use the current editor image.")
            source = self._get(value, source_id, kind="image")
            extra = tuple(self._get(value, item, kind="image") for item in additional_image_ids)
            mask_source = extra[0] if operation == "image-composite" and extra else source
            mask = self._get(value, mask_id, kind="mask") if mask_id is not None else None
            if mask is not None and (
                mask.source_image_id != mask_source.artifact_id
                or mask.source_frame_id != mask_source.frame_id
                or mask.frame.size != mask_source.frame.size
            ):
                raise ValueError("The selection mask belongs to a different image frame.")
            ticket = OperationTicket(
                session_id,
                revision,
                value["owner_key"],
                uuid.uuid4().hex,
                operation,
                source,
                mask,
                extra,
                parameters_json,
                plan_json,
            )
            value["pending"] = ticket.metadata() | {"revision": revision, "status": "captured"}
            self._save(value)
            return ticket

    def _validate(self, value, ticket):
        self._revision(value, ticket.revision)
        pending = value["pending"]
        expected = ticket.metadata() | {"revision": ticket.revision}
        if (
            pending is None
            or {key: val for key, val in pending.items() if key != "status"} != expected
        ):
            raise CancelledError("The editor operation was canceled or replaced.")
        if value["head_id"] != ticket.source_id:
            raise CancelledError("The editor source changed before this result completed.")
        self._get(value, value["original_id"], kind="image")
        for ref in (ticket.source, ticket.mask, *ticket.additional_images):
            if ref is not None and self._get(value, ref.artifact_id) != ref:
                raise ValueError("The captured editor artifact no longer matches its source.")

    def validate(self, ticket, *, owner=None):
        if not isinstance(ticket, OperationTicket) or ticket.owner_key != _owner_key(owner):
            raise PermissionError("This editor operation belongs to another session.")
        with self._edit(ticket.session_id, owner) as value:
            self._validate(value, ticket)
            return ticket

    def start_operation(self, ticket, *, owner=None):
        self.validate(ticket, owner=owner)
        with self._edit(ticket.session_id, owner) as value:
            self._validate(value, ticket)
            if value["pending"]["status"] != "captured":
                raise CancelledError("This editor operation already started.")
            value["pending"]["status"] = "started"
            self._save(value)
        return ticket

    def bind_effective_plan(self, ticket, plan, *, owner=None):
        """Replace a draft ticket atomically when final request capture supplies P1."""
        if not isinstance(ticket, OperationTicket) or ticket.owner_key != _owner_key(owner):
            raise PermissionError("This editor operation belongs to another session.")
        plan_json = _mapping_json(plan, "effective plan")
        with self._edit(ticket.session_id, owner) as value:
            self._validate(value, ticket)
            if value["pending"]["status"] != "captured":
                raise CancelledError("The editor plan must be bound before its operation starts.")
            if plan_json == ticket.effective_plan_json:
                return ticket
            result = replace(ticket, operation_id=uuid.uuid4().hex, effective_plan_json=plan_json)
            value["pending"] = result.metadata() | {
                "revision": result.revision,
                "status": "captured",
            }
            self._save(value)
            return result

    def _trim(self, value):
        value["steps"] = value["steps"][-self.max_steps :]
        while True:
            protected = {value["original_id"], value["head_id"]}
            derived = value.get("derived_results") or {}
            if derived.get("revision") == value["revision"]:
                protected.update(derived.get("result_ids", ()))
            selection_id = value.get("selection_mask_id")
            selection = value["artifacts"].get(selection_id, {})
            if selection.get("source_image_id") == value["head_id"]:
                protected.add(selection_id)
            for step in value["steps"]:
                protected.update((step["source_id"], step["result_id"]))
                protected.update(step["additional_image_ids"])
                if step["mask_id"] is not None:
                    protected.add(step["mask_id"])
            if len(protected) <= self.max_artifacts:
                break
            if not value["steps"]:
                raise ValueError("The retained results exceed this editor's artifact limit.")
            value["steps"].pop(0)
        for artifact_id in list(value["artifacts"]):
            if len(value["artifacts"]) <= self.max_artifacts:
                break
            if artifact_id not in protected:
                value["artifacts"].pop(artifact_id)

    def complete(self, ticket, result_path, *, execution=None, owner=None, check_cancelled=None):
        from ruinedfooocus.app.resource_lifecycle import current_job, registry

        self.validate(ticket, owner=owner)
        execution = json.loads(_mapping_json(execution, "execution"))
        result = self._ref(result_path)
        if Path(result.path) == Path(ticket.source.path):
            raise ValueError(
                "Editor results must use a new artifact; the source cannot be overwritten."
            )
        source_frame, result_frame = ticket.source.frame, result.frame
        if source_frame.size != result_frame.size:
            geometry = execution.get("geometry")
            if not isinstance(geometry, dict) or (
                not geometry.get("policy")
                or geometry.get("source") != list(source_frame.size)
                or geometry.get("result") != list(result_frame.size)
            ):
                raise ValueError(
                    "Changed output geometry needs its explicit source/result conversion."
                )
        if (
            source_frame.alpha != "none"
            and result_frame.alpha == "none"
            and not execution.get("alpha_policy")
        ):
            raise ValueError("Dropping source alpha needs an explicit alpha policy.")
        if source_frame.dtype != result_frame.dtype and not execution.get("precision_conversion"):
            raise ValueError("Changed output precision needs an explicit conversion record.")
        job = current_job()
        # The existing registry lock makes cancellation versus commit one terminal decision.
        with registry.condition if job is not None else nullcontext():
            if job is not None:
                if _owner_key(job.owner) != ticket.owner_key:
                    raise PermissionError("The compute job does not own this editor operation.")
                job.check()
            if check_cancelled is not None:
                check_cancelled()
            with self._edit(ticket.session_id, owner) as value:
                self._validate(value, ticket)
                value["artifacts"][result.artifact_id] = result.to_dict(public=False)
                step = ticket.metadata() | dict(
                    parent_id=ticket.source_id,
                    result_id=result.artifact_id,
                    execution=execution,
                    completed_at=self.clock(),
                )
                value["steps"].append(step)
                value.update(
                    head_id=result.artifact_id,
                    revision=value["revision"] + 1,
                    redo=[],
                    pending=None,
                    last_outcome={"status": "completed", "operation_id": ticket.operation_id},
                )
                self._trim(value)
                snapshot = self._snapshot(value)
                if check_cancelled is not None:
                    check_cancelled()
                self._save(value)
                return snapshot

    def capture_derived_results(self, ticket, result_paths, *, owner=None):
        """Freeze bounded native results without changing the source or its revision."""
        from ruinedfooocus.features.image_processing.sam_prompts import MAX_MASK_PIXELS

        self.validate(ticket, owner=owner)
        if ticket.operation != "segment" or not isinstance(result_paths, (tuple, list)):
            raise ValueError("Capture native segmentation results for this editor operation.")
        if len(result_paths) > min(64, self.max_artifacts - 2):
            raise ValueError("The segmentation results exceed this editor's artifact limit.")
        if (
            len(result_paths) * ticket.source.frame.width * ticket.source.frame.height
            > MAX_MASK_PIXELS
        ):
            raise ValueError("The segmentation results exceed the native aggregate mask limit.")
        results = tuple(self._ref(path) for path in result_paths)
        for result in results:
            if Path(result.path) == Path(ticket.source.path):
                raise ValueError("A segmentation result cannot replace its source file.")
            if (
                result.frame.size != ticket.source.frame.size
                or result.frame.orientation != ticket.source.frame.orientation
            ):
                raise ValueError("Segmentation results must match the exact source image frame.")
        if len({result.path for result in results}) != len(results):
            raise ValueError("Every segmentation instance needs its own result artifact.")
        self.validate(ticket, owner=owner)
        return results

    def complete_derived_results(
        self, ticket, results, *, output, execution=None, owner=None, check_cancelled=None
    ):
        """Publish native SAM artifacts while keeping the owned photo as the image head."""
        from ruinedfooocus.app.resource_lifecycle import current_job, registry

        if not isinstance(ticket, OperationTicket) or ticket.owner_key != _owner_key(owner):
            raise PermissionError("This segmentation operation belongs to another editor owner.")
        if output not in ("mask", "cutout") or not isinstance(results, (tuple, list)):
            raise ValueError("Choose native segmentation mask or cutout results.")
        if any(not isinstance(result, ArtifactRef) or result.kind != "image" for result in results):
            raise ValueError("Capture every native segmentation result before publication.")
        if ticket.operation != "segment" or len(results) > min(64, self.max_artifacts - 2):
            raise ValueError("The native segmentation result count exceeds this editor's limit.")
        if len({result.path for result in results}) != len(results):
            raise ValueError("Every segmentation instance needs its own result artifact.")
        for result in results:
            if (
                result.frame.size != ticket.source.frame.size
                or result.frame.orientation != ticket.source.frame.orientation
                or Path(result.path) == Path(ticket.source.path)
                or tuple(file_signature(result.path)) != result.signature
            ):
                raise ValueError("A segmentation result changed before publication.")
        execution = json.loads(_mapping_json(execution, "execution"))
        job = current_job()
        with registry.condition if job is not None else nullcontext():
            if job is not None:
                if _owner_key(job.owner) != ticket.owner_key:
                    raise PermissionError(
                        "The compute job does not own these segmentation results."
                    )
                job.check()
            if check_cancelled is not None:
                check_cancelled()
            with self._edit(ticket.session_id, owner) as value:
                self._validate(value, ticket)
                for result in results:
                    value["artifacts"][result.artifact_id] = result.to_dict(public=False)
                revision = value["revision"] + 1
                value.update(
                    revision=revision,
                    pending=None,
                    selection_mask_id=None,
                    derived_results=dict(
                        operation="segment",
                        operation_id=ticket.operation_id,
                        source_id=ticket.source_id,
                        source_frame_id=ticket.source.frame_id,
                        revision=revision,
                        result_ids=[ref.artifact_id for ref in results],
                        output=output,
                        selection_eligible=output == "mask"
                        and ticket.source.frame.dtype == "uint8"
                        and ticket.source.frame.color_space == "sRGB"
                        and all(
                            ref.frame.dtype == "uint8" and ref.frame.color_space == "sRGB"
                            for ref in results
                        ),
                        execution=execution,
                    ),
                    last_outcome={"status": "completed", "operation_id": ticket.operation_id},
                )
                self._trim(value)
                snapshot = self._snapshot(value)
                if check_cancelled is not None:
                    check_cancelled()
                self._save(value)
                return snapshot

    def prepare_derived_selection(
        self, session_id, revision, indices, *, owner=None, check_cancelled=None
    ):
        """Decode/hash/combine selected masks outside shared publication/store locks."""
        import numpy as np
        from PIL import Image, ImageChops

        if (
            not isinstance(indices, (tuple, list))
            or not 1 <= len(indices) <= 64
            or any(type(index) is not int for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Select one or more object masks.")
        with self._edit(session_id, owner) as value:
            self._revision(value, revision)
            derived = value.get("derived_results") or {}
            if (
                value.get("pending") is not None
                or derived.get("operation") != "segment"
                or derived.get("revision") != revision
                or derived.get("source_id") != value["head_id"]
                or derived.get("output") != "mask"
            ):
                raise ValueError(
                    "Run segmentation with grayscale mask output for this image first."
                )
            source = self._get(value, derived["source_id"], kind="image")
            if source.frame_id != derived.get("source_frame_id"):
                raise ValueError("The original image frame changed after segmentation.")
            if source.frame.dtype != "uint8" or source.frame.color_space != "sRGB":
                raise ValueError("Inpainting needs an explicit 8-bit sRGB source and mask.")
            result_ids = derived["result_ids"]
            if any(not 0 <= index < len(result_ids) for index in indices):
                raise ValueError("The object mask results changed; select an existing instance.")
            results = tuple(
                self._get(value, result_ids[index], kind="image") for index in sorted(indices)
            )
            operation_id = derived["operation_id"]
        selection = Image.new("L", source.frame.size)
        for result in results:
            if check_cancelled is not None:
                check_cancelled()
            with preparation_service.artifact_lease(result.path):
                if (
                    result.frame.size != source.frame.size
                    or result.frame.orientation != source.frame.orientation
                    or result.frame.dtype != "uint8"
                    or result.frame.color_space != "sRGB"
                ):
                    raise ValueError("Use exact-frame 8-bit sRGB grayscale masks for inpainting.")
                with Image.open(result.path) as image:
                    pixels = np.asarray(image)
                    if image.mode == "L":
                        scalar = image.copy()
                    elif (
                        image.mode == "RGB"
                        and np.array_equal(pixels[..., 0], pixels[..., 1])
                        and np.array_equal(pixels[..., 0], pixels[..., 2])
                    ):
                        scalar = image.getchannel("R")
                    else:
                        raise ValueError("Use grayscale masks, not cutouts or image alpha.")
                with Path(result.path).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != result.sha256:
                        raise ValueError("An object mask changed after segmentation.")
                self._get(value, result.artifact_id, kind="image")
                # Combine retained soft selections without multiplying by photo alpha.
                selection = ImageChops.lighter(selection, scalar)
        if selection.getbbox() is None:
            raise ValueError("The selection mask is empty; choose another object.")
        path = self.root / "selection-masks" / (uuid.uuid4().hex + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if check_cancelled is not None:
                check_cancelled()
            selection.save(path)
            mask = self._ref(path, kind="mask", source=source)
            if check_cancelled is not None:
                check_cancelled()
            return PreparedDerivedSelection(
                session_id, revision, _owner_key(owner), operation_id, source, results, mask
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def commit_derived_selection(self, prepared, *, owner=None, check_cancelled=None):
        """Revalidate and bind an already prepared mask in one short atomic commit."""
        if not isinstance(prepared, PreparedDerivedSelection) or prepared.owner_key != _owner_key(
            owner
        ):
            raise PermissionError("This prepared selection belongs to another editor owner.")
        if check_cancelled is not None:
            check_cancelled()
        with self._edit(prepared.session_id, owner) as value:
            self._revision(value, prepared.revision)
            derived = value.get("derived_results") or {}
            if (
                value.get("pending") is not None
                or derived.get("revision") != prepared.revision
                or derived.get("operation_id") != prepared.operation_id
                or derived.get("source_id") != value["head_id"]
                or derived.get("output") != "mask"
                or self._get(value, prepared.source.artifact_id, kind="image") != prepared.source
            ):
                raise CancelledError(
                    "The source or segmentation results changed during preparation."
                )
            for result in prepared.results:
                if (
                    result.artifact_id not in derived.get("result_ids", ())
                    or self._get(value, result.artifact_id, kind="image") != result
                ):
                    raise ValueError("A selected object mask changed during preparation.")
            if tuple(file_signature(prepared.mask.path)) != prepared.mask.signature:
                raise ValueError("The prepared selection mask changed before it was attached.")
            mask = prepared.mask
            if (
                mask.kind != "mask"
                or mask.channel != "L"
                or mask.source_image_id != prepared.source.artifact_id
                or mask.source_frame_id != prepared.source.frame_id
                or mask.frame.size != prepared.source.frame.size
                or mask.frame.dtype != "uint8"
            ):
                raise ValueError("The prepared selection does not match its original source frame.")
            value["artifacts"][mask.artifact_id] = mask.to_dict(public=False)
            value.update(
                revision=value["revision"] + 1,
                derived_results=None,
                selection_mask_id=mask.artifact_id,
            )
            self._trim(value)
            snapshot = self._snapshot(value)
            if check_cancelled is not None:
                check_cancelled()
            self._save(value)
            return snapshot, mask

    def discard_derived_selection(self, prepared):
        if (
            isinstance(prepared, PreparedDerivedSelection)
            and Path(prepared.mask.path).parent == self.root / "selection-masks"
        ):
            Path(prepared.mask.path).unlink(missing_ok=True)

    def use_derived_selection(self, session_id, revision, indices, *, owner=None):
        prepared = self.prepare_derived_selection(session_id, revision, indices, owner=owner)
        try:
            return self.commit_derived_selection(prepared, owner=owner)
        except BaseException:
            self.discard_derived_selection(prepared)
            raise

    def _terminal(self, ticket, status, owner):
        if not isinstance(ticket, OperationTicket) or ticket.owner_key != _owner_key(owner):
            raise PermissionError("This editor operation belongs to another session.")
        with self._edit(ticket.session_id, owner) as value:
            pending = value["pending"]
            if pending is None or pending["operation_id"] != ticket.operation_id:
                return False
            value["pending"] = None
            value["last_outcome"] = {"operation_id": ticket.operation_id, "status": status}
            self._save(value)
            return True

    def cancel(self, ticket, *, owner=None):
        return self._terminal(ticket, "canceled", owner)

    def fail(self, ticket, *, owner=None):
        return self._terminal(ticket, "failed", owner)

    def _move(self, session_id, revision, *, redo, owner):
        with self._edit(session_id, owner) as value:
            self._revision(value, revision)
            if redo:
                if not value["redo"]:
                    raise ValueError("There is no retained editor result to redo.")
                target = value["redo"].pop()
            else:
                step = next(
                    (item for item in value["steps"] if item["result_id"] == value["head_id"]), None
                )
                if step is None:
                    raise ValueError("The bounded editor history has no earlier step.")
                target = step["parent_id"]
                value["redo"].append(value["head_id"])
            self._get(value, target, kind="image")
            value.update(head_id=target, revision=value["revision"] + 1, pending=None)
            result = self._snapshot(value)
            self._save(value)
            return result

    def undo(self, session_id, revision, *, owner=None):
        return self._move(session_id, revision, redo=False, owner=owner)

    def use_result_as_mask(self, session_id, revision, result_id, *, owner=None):
        """Attach an exact-byte selection and restore its retained source in one save."""
        import numpy as np
        from PIL import Image

        with self._edit(session_id, owner) as value:
            self._revision(value, revision)
            if value["head_id"] != result_id or value.get("pending") is not None:
                raise CancelledError("The mask result changed or an image action is still pending.")
            step = next(
                (item for item in reversed(value["steps"]) if item["result_id"] == result_id), None
            )
            if (
                step is None
                or step.get("operation") not in ("image-mask", "face-landmarks")
                or step.get("execution", {}).get("image_utility_source", {}).get("mask_role")
                != "inpaint-selection"
            ):
                raise ValueError(
                    "Run an explicit mask adjustment before using its result as a selection."
                )
            source = self._get(value, step["source_id"], kind="image")
            result = self._get(value, result_id, kind="image")
            if result.frame.size != source.frame.size:
                raise ValueError("The mask result must match its original source frame exactly.")
            if (
                result.frame.dtype != "uint8"
                or result.frame.color_space != "sRGB"
                or source.frame.dtype != "uint8"
                or source.frame.color_space != "sRGB"
            ):
                raise ValueError(
                    "Inpainting needs an 8-bit sRGB source and mask. Export an explicit working copy first."
                )
            if len(value["artifacts"]) >= self.max_artifacts:
                raise ValueError("The editor artifact limit is full; open a new session.")
            with Image.open(result.path) as image:
                pixels = np.asarray(image)
                if (
                    pixels.ndim != 3
                    or pixels.shape[-1] not in (3, 4)
                    or not np.array_equal(pixels[..., 0], pixels[..., 1])
                    or not np.array_equal(pixels[..., 0], pixels[..., 2])
                ):
                    raise ValueError(
                        "The mask result must contain matching grayscale RGB channels."
                    )
                selection = Image.fromarray(pixels[..., 0].copy(), "L")
            self._get(value, result_id, kind="image")
            path = self.root / "selection-masks" / (uuid.uuid4().hex + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                selection.save(path)
                mask = self._ref(path, kind="mask", source=source)
                self._get(value, source.artifact_id, kind="image")
                value["artifacts"][mask.artifact_id] = mask.to_dict(public=False)
                value.update(
                    head_id=source.artifact_id,
                    revision=revision + 1,
                    pending=None,
                    redo=[result_id],
                )
                snapshot = self._snapshot(value)
                self._save(value)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            return snapshot, mask

    def redo(self, session_id, revision, *, owner=None):
        return self._move(session_id, revision, redo=True, owner=owner)


def default_store(manager=None):
    root = configured_cache_root(manager) / "editor-sessions"
    with _LOCK:
        key = str(root)
        # Config/profile roots are bounded in-process just like UI request owners.
        if key not in _STORES:
            if len(_STORES) >= 16:
                _STORES.pop(next(iter(_STORES)))
            _STORES[key] = EditorSessionStore(root)
        return _STORES[key]


def request_parameters(data):
    """Capture scalar request intent; actual operation execution is a separate field."""
    from ruinedfooocus.features.gallery.image_metadata import RENDER_FIELDS
    from ruinedfooocus.features.image_editing.image_guidance import FIELD_SCHEMA

    return json.loads(
        _json(
            {
                key: value
                for key, value in data.items()
                if key in RENDER_FIELDS
                or key in FIELD_SCHEMA
                or key.startswith(("cn_", "effective_"))
            }
        )
    )


def capture_request(data, context, operation, *, owner=None, store=None):
    if (
        not isinstance(context, dict)
        or set(context)
        - {
            "session_id",
            "revision",
            "source_id",
            "mask_id",
            "additional_image_ids",
        }
        or not {"session_id", "revision", "source_id"} <= set(context)
    ):
        raise ValueError("Provide only the owned editor session/revision/image/mask handles.")
    count = data.get("image_number", data.get("n", 1))
    if type(count) is not int or count != 1:
        raise ValueError("An editor operation returns exactly one image.")
    selected = store or default_store()
    plan = data.get("_effective_image_plan")
    if plan is not None:
        plan = plan.metadata()
    ticket = selected.capture(
        **context,
        operation=operation,
        parameters=request_parameters(data),
        effective_plan=plan,
        owner=owner,
    )
    return dict(data, **{KEY: ticket})


def validate_request(data, *, owner=None, store=None, start=False):
    ticket = data.get(KEY)
    if ticket is None:
        return None
    selected = store or default_store()
    return (selected.start_operation if start else selected.validate)(ticket, owner=owner)


def bind_effective_plan(data, plan=None, *, owner=None, store=None):
    ticket = data.get(KEY)
    plan = data.get("_effective_image_plan") if plan is None else plan
    if ticket is None or plan is None:
        return data
    metadata = plan if isinstance(plan, dict) else plan.metadata()
    result = (store or default_store()).bind_effective_plan(ticket, metadata, owner=owner)
    return dict(data, **{KEY: result})


def complete_request(
    data, result_path, execution=None, *, owner=None, store=None, check_cancelled=None
):
    ticket = data.get(KEY)
    if ticket is None:
        return None
    return (store or default_store()).complete(
        ticket, result_path, execution=execution, owner=owner, check_cancelled=check_cancelled
    )


def cancel_request(data, *, owner=None, store=None):
    ticket = data.get(KEY)
    return (store or default_store()).cancel(ticket, owner=owner) if ticket is not None else False


def fail_request(data, *, owner=None, store=None):
    ticket = data.get(KEY)
    return (store or default_store()).fail(ticket, owner=owner) if ticket is not None else False
