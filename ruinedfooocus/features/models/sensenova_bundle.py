"""Capture the named SenseNova U1.5 safetensors index without reading weights."""

from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_sensenova_shard_bundle"
INDEX_NAME = "model.safetensors.index.json"
SHARD_NAMES = tuple(f"model-{number:05d}-of-00008.safetensors" for number in range(1, 9))
MAX_INDEX_BYTES = 1024**2
MAX_HEADER_BYTES = 16 * 1024**2
MAX_TENSORS = 100_000
MAX_MODEL_BYTES = 64 * 1024**3
_WIDTHS = {"BF16": 2, "F16": 2, "F32": 4}
VISION = "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"
QUERY = "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"
_active_reads = ContextVar("sensenova_bundle_reads", default=frozenset())


def is_sensenova_index(path):
    return Path(path).name == INDEX_NAME


def is_bundle_shard(path):
    path = Path(path)
    return path.name in SHARD_NAMES and (path.parent / INDEX_NAME).is_file()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate SenseNova index or tensor header key.")
        result[key] = value
    return result


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_unique)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded SenseNova JSON.") from error


@dataclass(frozen=True)
class BundleFile:
    source: Path
    path: Path
    signature: tuple
    header_sha256: str

    def verify(self):
        try:
            unchanged = (
                self.source.resolve(strict=True) == self.path
                and tuple(file_signature(self.path)) == self.signature
            )
        except (OSError, RuntimeError):
            unchanged = False
        if not unchanged:
            raise ValueError("A SenseNova bundle file changed. Refresh and submit again.")


@dataclass(frozen=True)
class Tensor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]


@dataclass(frozen=True)
class SenseNovaBundle:
    index: BundleFile
    shards: tuple[BundleFile, ...]
    tensors: tuple[Tensor, ...]
    metadata: tuple[tuple[str, str], ...]

    @property
    def files(self):
        return (self.index, *self.shards)

    @property
    def identity(self):
        rows = [(item.signature, item.header_sha256) for item in self.files]
        return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()

    def verify(self, check_cancelled=None):
        for item in self.files:
            if check_cancelled is not None:
                check_cancelled()
            item.verify()

    def header(self):
        return {
            item.name: {"shape": list(item.shape), "dtype": item.dtype} for item in self.tensors
        }


def _capture(source, root):
    source = Path(source).absolute()
    try:
        path = source.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Bundle file is unavailable: {source.name}. Install or repair the complete local bundle."
        ) from error
    if not path.is_file() or path.parent != root:
        raise ValueError(
            "SenseNova index and eight shards must be regular files in one local folder."
        )
    return BundleFile(source, path, tuple(file_signature(path)), "")


def _read(item, *, header=False):
    """Read at most bounded JSON and bind the descriptor to the opened file ID."""
    item.verify()
    with item.path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        bound = item.path.stat()
        if (before.st_dev, before.st_ino) != (bound.st_dev, bound.st_ino):
            raise ValueError("SenseNova source changed before inspection.")
        if header:
            prefix = stream.read(8)
            size = int.from_bytes(prefix, "little")
            if len(prefix) != 8 or not 2 <= size <= min(MAX_HEADER_BYTES, item.signature[1] - 8):
                raise ValueError("Invalid or oversized SenseNova safetensors header.")
            raw = stream.read(size)
        else:
            if item.signature[1] > MAX_INDEX_BYTES:
                raise ValueError("SenseNova index exceeds its inspection limit.")
            raw = stream.read(MAX_INDEX_BYTES + 1)
            if len(raw) != item.signature[1]:
                raise ValueError("SenseNova index changed while reading.")
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("SenseNova source changed during inspection.")
    item.verify()
    return _json(raw), hashlib.sha256(raw).hexdigest(), len(raw)


def validate_shard(header, payload_bytes, shard):
    """Validate detached dtype/shape/ranges before any native tensor mapping."""
    if not isinstance(header, dict) or not 1 <= len(header) <= MAX_TENSORS + 1:
        raise ValueError("Invalid SenseNova tensor header.")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(value, str) for value in metadata.values()
    ):
        raise ValueError("SenseNova tensor metadata must contain strings.")
    tensors, ranges = [], []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or len(name) > 4096 or not isinstance(entry, dict):
            raise ValueError("Invalid SenseNova tensor name or descriptor.")
        shape, dtype, offsets = entry.get("shape"), entry.get("dtype"), entry.get("data_offsets")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(type(value) is not int or not 0 <= value <= 2**31 for value in shape)
        ):
            raise ValueError("Invalid SenseNova tensor shape.")
        if (
            not isinstance(dtype, str)
            or dtype not in _WIDTHS
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError("Invalid SenseNova tensor dtype or byte range.")
        start, end = offsets
        if (
            not 0 <= start <= end <= payload_bytes
            or end - start != math.prod(shape) * _WIDTHS[dtype]
        ):
            raise ValueError("SenseNova tensor range does not match its file, dtype and shape.")
        ranges.append((start, end))
        tensors.append(Tensor(name, shard, dtype, tuple(shape), (start, end)))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("SenseNova tensor ranges overlap or leave unindexed payload bytes.")
        cursor = end
    if not tensors or cursor != payload_bytes:
        raise ValueError("SenseNova shard has missing or unindexed tensor payloads.")
    return tuple(tensors), metadata


def capture_bundle(index, *, check_cancelled=None):
    """Capture the reviewed eight-shard layout; index values never become imports."""
    if not is_sensenova_index(index):
        raise ValueError("Select the SenseNova model.safetensors.index.json.")
    root = Path(index).absolute().parent.resolve(strict=True)
    captured = _capture(index, root)
    document, digest, _ = _read(captured)
    captured = BundleFile(captured.source, captured.path, captured.signature, digest)
    if not isinstance(document, dict) or set(document) != {"metadata", "weight_map"}:
        raise ValueError("SenseNova index requires metadata and weight_map objects.")
    weights, info = document["weight_map"], document["metadata"]
    if (
        not isinstance(weights, dict)
        or not 1 <= len(weights) <= MAX_TENSORS
        or not isinstance(info, dict)
    ):
        raise ValueError("Invalid SenseNova index tensor map.")
    if any(
        not isinstance(name, str) or not name or len(name) > 4096 or shard not in SHARD_NAMES
        for name, shard in weights.items()
    ):
        raise ValueError("SenseNova index must reference only its eight named local shards.")
    if set(weights.values()) != set(SHARD_NAMES):
        raise ValueError("SenseNova index must include all eight named shards.")
    total = info.get("total_size")
    if type(total) is not int or not 0 < total <= MAX_MODEL_BYTES:
        raise ValueError("Invalid SenseNova index total_size.")
    shards, tensors, names, metadata, total_bytes = [], [], set(), None, 0
    for name in SHARD_NAMES:
        if check_cancelled is not None:
            check_cancelled()
        item = _capture(Path(index).absolute().parent / name, root)
        header, digest, size = _read(item, header=True)
        payload_bytes = item.signature[1] - 8 - size
        rows, declarations = validate_shard(header, payload_bytes, name)
        for row in rows:
            if row.name in names or weights.get(row.name) != name:
                raise ValueError("SenseNova shard keys duplicate or disagree with the index.")
            names.add(row.name)
        if metadata is not None and metadata != declarations:
            raise ValueError("SenseNova shards have conflicting native metadata.")
        metadata = declarations
        shards.append(BundleFile(item.source, item.path, item.signature, digest))
        tensors.extend(rows)
        total_bytes += payload_bytes
    if names != set(weights) or total_bytes != total:
        raise ValueError("SenseNova index keys or total_size do not match all eight shards.")
    shapes = {row.name: row.shape for row in tensors}
    if (
        not shapes.get(VISION)
        or not shapes.get(QUERY)
        or shapes[VISION][0] != 1024
        or shapes[QUERY][0] != 4096
    ):
        raise ValueError("The named shard bundle does not match native SenseNovaU15.")
    bundle = SenseNovaBundle(
        captured, tuple(shards), tuple(tensors), tuple(sorted((metadata or {}).items()))
    )
    bundle.verify(check_cancelled)
    return bundle


def discover_bundles(roots):
    """Keep a named index visible; metadata distinguishes ready and unknown bundles."""
    seen = set()
    for root in roots:
        root = Path(root)
        for index in root.rglob(INDEX_NAME):
            relative = str(index.relative_to(root))
            if relative in seen:
                continue
            seen.add(relative)
            if not index.is_file():
                continue
            inspect_bundle(index)
            yield index


def _inventory_key(index):
    paths = (index, *(index.parent / name for name in SHARD_NAMES))
    rows = []
    for path in paths:
        try:
            rows.append(tuple(file_signature(path)))
        except (OSError, RuntimeError):
            rows.append(None)
    return tuple(rows)


@lru_cache(maxsize=32)
def _inspect_bundle(index, identity):
    try:
        return capture_bundle(index), None
    except ValueError as error:
        return None, str(error)
    except OSError:
        return (
            None,
            "The local bundle cannot be read. Check the index and all eight local shard files.",
        )


def inspect_bundle(index):
    """Cache one detached inspection, invalidated by any source or missing-file change."""
    index = Path(index).absolute()
    try:
        # Normalize directory aliases (including Windows 8.3 paths), retaining
        # the index leaf so an escaping file symlink cannot change its bundle root.
        index = index.parent.resolve(strict=True) / index.name
    except (OSError, RuntimeError):
        return None, "The local bundle folder is unavailable. Refresh and inspect it again."
    before = _inventory_key(index)
    result = _inspect_bundle(str(index), before)
    if _inventory_key(index) != before:
        return None, "The local bundle changed during inspection. Refresh and inspect it again."
    return result


def bundle_metadata(index):
    metadata = {
        "rf_display_name": Path(index).parent.name,
        "rf_source": "Local SenseNova U1.5 shard bundle",
        "baseModel": "SenseNovaU15",
        "model": {"type": "Checkpoint"},
    }
    bundle, error = inspect_bundle(index)
    if bundle is not None:
        metadata.update(
            _rf_file_signature=list(bundle.index.signature), _rf_bundle_identity=bundle.identity
        )
    else:
        metadata["baseModel"] = "Unknown"
        metadata["rf_source"] = "Local shard index (unavailable)"
        metadata["_rf_bundle_error"] = error
    return metadata


def bundle_inventory(bundle, choices=None):
    if any((choices or {}).values()):
        raise ValueError(
            "SenseNova bundles use their native tokenizer and pixel codec; clear component overrides."
        )
    return dict(
        family="SenseNovaU15",
        architecture="SenseNovaU15",
        rows=[],
        missing_payload_bytes=0,
        bundle_identity=bundle.identity,
        note="Pixel-space model: native tokenizer and no external VAE. Eight local shards stay fixed.",
    )


@contextmanager
def read_scope(bundle, *, cancelled=None):
    from ruinedfooocus.core.preparation import preparation_service

    identity = bundle.identity
    if identity in _active_reads.get():
        bundle.verify()
        yield
        bundle.verify()
        return
    with ExitStack() as scope:
        for item in sorted(bundle.files, key=lambda item: str(item.path)):
            scope.enter_context(preparation_service.artifact_lease(item.path, cancelled=cancelled))
        bundle.verify()
        token = _active_reads.set(_active_reads.get() | {identity})
        try:
            yield
            bundle.verify()
        finally:
            _active_reads.reset(token)
