"""Explicit full native approximate codecs, bound to two current local sources."""

from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

from ruinedfooocus.core.model_folders import model_folders
from ruinedfooocus.core.model_formats import NATIVE_STATE_DICT_SUFFIXES, is_safetensors
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_native_approximate_codec"
PREFIX = "approx:"
# These are the native codec slots, including normalization and spatial layout.
# Equal latent counts in other families do not establish interchangeable codecs.
ALIASES = {
    "taesd": "vae_sd",
    "taesdxl": "vae_sdxl",
    "taesd3": "vae_sd3",
    "taef1": "vae_flux",
    "taef2": "vae_flux2",
    "taeqi2_1": "vae_qwen_image21",
}
CHANNELS = dict(taesd=4, taesdxl=4, taesd3=16, taef1=16, taef2=128, taeqi2_1=64)
MAX_FILE_BYTES = 512 * 1024**2
MAX_HEADER_BYTES = 1024**2
_WIDTHS = {"F32": 4, "F16": 2, "BF16": 2}
_reads = ContextVar("approximate_codec_reads", default=frozenset())


def is_choice(value):
    return isinstance(value, str) and value.startswith(PREFIX)


def choice_alias(key, value):
    if not is_choice(value):
        return None
    alias = value[len(PREFIX) :]
    if ALIASES.get(alias) != key:
        raise ValueError("This approximate codec does not match the checkpoint's VAE contract.")
    return alias


@lru_cache(maxsize=1)
def _contracts():
    return json.loads(resource_path("approximate_codec_shapes.json").read_text(encoding="utf-8"))


def expected_shapes(alias, role):
    return {name: tuple(shape) for name, shape in _contracts()[str(CHANNELS[alias])][role].items()}


@dataclass(frozen=True)
class CodecFile:
    role: str
    source: Path
    path: Path
    signature: tuple
    header_sha256: str | None = None

    def verify(self):
        try:
            same = (
                self.source.resolve(strict=True) == self.path
                and tuple(file_signature(self.path)) == self.signature
            )
        except (OSError, RuntimeError):
            same = False
        if not same:
            raise ValueError("An approximate codec file changed. Refresh and submit again.")


@dataclass(frozen=True)
class ApproximateCodec:
    alias: str
    slot: str
    encoder: CodecFile
    decoder: CodecFile

    @property
    def files(self):
        return (self.encoder, self.decoder)

    @property
    def identity(self):
        rows = [(item.role, item.signature, item.header_sha256) for item in self.files]
        return hashlib.sha256(json.dumps([self.alias, self.slot, rows]).encode()).hexdigest()

    def verify(self):
        for item in self.files:
            item.verify()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate approximate codec tensor header key.")
        result[key] = value
    return result


@lru_cache(maxsize=64)
def _header_contract(alias, item):
    """Validate the complete native key/shape/range contract without tensor reads."""
    item.verify()
    with item.path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        current = item.path.stat()
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Approximate codec source changed before inspection.")
        prefix = stream.read(8)
        size = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 2 <= size <= min(MAX_HEADER_BYTES, item.signature[1] - 8):
            raise ValueError("Invalid or oversized approximate codec header.")
        raw = stream.read(size)
        try:
            header = json.loads(raw, object_pairs_hook=_unique)
        except (UnicodeError, RecursionError) as error:
            raise ValueError("Invalid approximate codec header JSON.") from error
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Approximate codec source changed during inspection.")
    if not isinstance(header, dict) or len(header) > 512:
        raise ValueError("Invalid approximate codec tensor header.")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("Invalid approximate codec metadata.")
    expected = expected_shapes(alias, item.role)
    if header.keys() != expected.keys():
        raise ValueError("The approximate codec needs the complete native encoder and decoder.")
    ranges = []
    payload = item.signature[1] - 8 - size
    for name, shape in expected.items():
        row = header[name]
        if (
            not isinstance(row, dict)
            or row.get("shape") != list(shape)
            or any(type(n) is not int for n in row["shape"])
        ):
            raise ValueError("Approximate codec tensor shapes do not match the native alias.")
        dtype, offsets = row.get("dtype"), row.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or dtype not in _WIDTHS
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
        ):
            raise ValueError("Invalid approximate codec tensor dtype or range.")
        start, end = offsets
        if not 0 <= start < end <= payload or end - start != math.prod(shape) * _WIDTHS[dtype]:
            raise ValueError("Invalid approximate codec tensor payload range.")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Approximate codec tensor ranges overlap or leave missing bytes.")
        cursor = end
    if cursor != payload:
        raise ValueError("Approximate codec has unindexed tensor bytes.")
    item.verify()
    return hashlib.sha256(raw).hexdigest()


def _available(paths):
    """Match native sorted basename and configured-root priority for paired aliases."""
    found = {}
    for root in model_folders(paths.paths, "vae_approx", paths.get_abspath):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in NATIVE_STATE_DICT_SUFFIXES:
                found.setdefault(path.name, (path, root.resolve()))
    return found


def _capture_pair(key, value, paths):
    alias = choice_alias(key, value)
    if alias is None:
        raise ValueError("Choose an explicit full approximate codec alias.")
    found, files = _available(paths), []
    for role in ("encoder", "decoder"):
        name = next((name for name in sorted(found) if name.startswith(f"{alias}_{role}.")), None)
        if name is None:
            raise ValueError(
                f"Approximate codec {alias} needs both local encoder and decoder files."
            )
        source, root = found[name]
        path = source.resolve(strict=True)
        if path.parent != root:
            raise ValueError("Approximate codec sources must stay inside their configured folder.")
        signature = tuple(file_signature(path))
        if not 0 < signature[1] <= MAX_FILE_BYTES:
            raise ValueError("Approximate codec source exceeds the native pair inspection limit.")
        item = CodecFile(role, source.absolute(), path, signature)
        if is_safetensors(path):
            item = replace(item, header_sha256=_header_contract(alias, item))
        item.verify()
        files.append(item)
    if files[0].path == files[1].path:
        raise ValueError("An approximate codec needs distinct encoder and decoder sources.")
    return ApproximateCodec(alias, key, *files)


def capture_pair(key, value, paths):
    try:
        return _capture_pair(key, value, paths)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            "An approximate codec source is unavailable. Refresh and choose again."
        ) from error


def inventory_row(key, value, paths):
    alias = choice_alias(key, value)
    try:
        pair = capture_pair(key, value, paths)
        state = "compatible" if all(item.header_sha256 for item in pair.files) else "unknown"
        files = [
            dict(
                role=item.role,
                filename=item.source.name,
                location=str(item.path),
                bytes=item.signature[1],
            )
            for item in pair.files
        ]
        reason = ""
    except (OSError, ValueError, RuntimeError) as error:
        pair, files, state, reason = None, [], "incompatible", str(error)
    return dict(
        key=key,
        filename=value,
        category="vae",
        approximate_codec=alias,
        files=files,
        override=True,
        status="Installed" if pair else "Missing",
        location="\n".join(item["location"] for item in files),
        source="",
        bytes=sum(item["bytes"] for item in files) if pair else None,
        download_key=None,
        download_available=False,
        published_bytes=None,
        compatibility=state,
        compatibility_reason=reason,
        scaled_fp8=False,
        codec_identity=pair.identity if pair else None,
    )


def installed_pairs(key, paths):
    return [
        dict(
            name=PREFIX + alias,
            path="",
            format="native-pair",
            compatibility=row["compatibility"],
            reason=row["compatibility_reason"],
        )
        for alias, slot in ALIASES.items()
        if slot == key
        and (row := inventory_row(key, PREFIX + alias, paths))["status"] == "Installed"
    ]


def model_capabilities(family, paths):
    """Expose installed full pairs only for the model's exact native VAE slot."""
    from ruinedfooocus.features.models.image_components import ALIASES as FAMILIES, IMAGE_COMPONENTS

    slot = IMAGE_COMPONENTS.get(FAMILIES.get(family, family), {}).get("vae_name")
    if paths is None or slot not in ALIASES.values():
        return []
    rows = []
    for alias, key in ALIASES.items():
        if key != slot:
            continue
        row = inventory_row(key, PREFIX + alias, paths)
        if row["status"] == "Installed":
            rows.append(
                dict(
                    slot=key,
                    choice=PREFIX + alias,
                    compatibility=row["compatibility"],
                    identity=row["codec_identity"],
                    local_bytes=row["bytes"],
                    files=[
                        {field: item[field] for field in ("role", "filename", "bytes")}
                        for item in row["files"]
                    ],
                )
            )
    return rows


def capture_request(data, paths, *, previous=None):
    """Never trust a client-provided private capture; bind the actual selection."""
    from ruinedfooocus.backends.comfy.component_selection import normalize_component_overrides

    result = dict(data)
    choices = normalize_component_overrides(result.get("image_component_overrides"))
    selected = [(key, value) for key, value in choices.items() if is_choice(value)]
    if isinstance(previous, ApproximateCodec) and selected == [
        (previous.slot, PREFIX + previous.alias)
    ]:
        result[KEY] = previous
    if isinstance(result.get(KEY), ApproximateCodec):
        captured_pair(result, choices)
        return result
    result.pop(KEY, None)
    if len(selected) > 1:
        raise ValueError("Select one approximate codec for the checkpoint's VAE role.")
    if selected:
        result[KEY] = capture_pair(*selected[0], paths)
    return result


def captured_pair(data, choices, slot=None):
    selected = [(key, value) for key, value in choices.items() if is_choice(value)]
    pair = (data or {}).get(KEY)
    if not selected:
        if pair is not None:
            raise ValueError("The captured approximate codec does not match this request.")
        return None
    if len(selected) != 1 or not isinstance(pair, ApproximateCodec):
        raise ValueError("Capture both approximate codec files before native loading.")
    key, value = selected[0]
    if key != pair.slot or choice_alias(key, value) != pair.alias or slot not in (None, key):
        raise ValueError("The captured approximate codec does not match this VAE role.")
    pair.verify()
    return pair


@contextmanager
def read_scope(pair, cancelled=None):
    from ruinedfooocus.core.preparation import preparation_service

    identity = pair.identity
    if identity in _reads.get():
        pair.verify()
        yield
        pair.verify()
        return
    with ExitStack() as stack:
        for item in pair.files:
            stack.enter_context(preparation_service.artifact_lease(item.path, cancelled=cancelled))
        pair.verify()
        token = _reads.set(_reads.get() | {identity})
        try:
            yield
            pair.verify()
        finally:
            _reads.reset(token)
