"""Inspect GGUF logical shapes and native detection metadata without tensor data."""

import json
import math
import os
import struct

MAX_METADATA_BYTES = 16 * 1024**2
MAX_HEADER_BYTES = 128 * 1024**2
MAX_FIELDS = 100_000
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


class _HeaderStream:
    """GGUF metadata/descriptors only; never allocate a vocabulary or tensor view."""

    def __init__(self, stream, path):
        self.stream = stream
        self.stat = os.fstat(stream.fileno())
        self.path_stat = os.stat(path)
        self.position = 0
        self.limit = min(self.stat.st_size, MAX_HEADER_BYTES)
        self.order = "<"
        self.captured = 0
        if self.take(4) != b"GGUF":
            raise ValueError("Invalid GGUF magic")
        version = self.take(4)
        self.order = "<" if struct.unpack("<I", version)[0] in (2, 3) else ">"
        if struct.unpack(self.order + "I", version)[0] not in (2, 3):
            raise ValueError("Unsupported GGUF metadata version")
        self.tensor_count, self.field_count = self.integer("Q"), self.integer("Q")
        if self.tensor_count > MAX_FIELDS or self.field_count > MAX_FIELDS:
            raise ValueError("GGUF contains too many metadata fields or tensors")

    def take(self, size):
        if size < 0 or self.position + size > self.limit:
            raise ValueError("GGUF metadata exceeds its file or inspection limit")
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated GGUF metadata")
        self.position += size
        return data

    def skip(self, size):
        if size < 0 or self.position + size > self.limit:
            raise ValueError("GGUF metadata exceeds its file or inspection limit")
        self.position += size
        self.stream.seek(self.position)

    def integer(self, kind):
        return struct.unpack(self.order + kind, self.take(struct.calcsize(kind)))[0]

    def string(self, maximum, *, skip_oversized=False):
        size = self.integer("Q")
        if size > maximum:
            if skip_oversized:
                self.skip(size)
                return None
            raise ValueError("GGUF metadata string is too large")
        self.captured += size
        if self.captured > MAX_METADATA_BYTES:
            raise ValueError("GGUF detection metadata is too large")
        return self.take(size).decode("utf-8")

    def skip_value(self, kind):
        if kind in _SCALARS:
            self.skip(struct.calcsize(_SCALARS[kind]))
        elif kind == 8:
            self.skip(self.integer("Q"))
        elif kind == 9:
            element, count = self.integer("I"), self.integer("Q")
            if count > 1_000_000:
                raise ValueError("GGUF metadata array is too large")
            if element == 8:
                for _ in range(count):
                    self.skip(self.integer("Q"))
            elif element in _SCALARS:
                self.skip(struct.calcsize(_SCALARS[element]) * count)
            else:
                raise ValueError("Unsupported GGUF array element type")
        else:
            raise ValueError("Unknown GGUF metadata type")

    def fields(self):
        names = set()
        for _ in range(self.field_count):
            key = self.string(4096)
            if key in names:
                raise ValueError("Duplicate GGUF metadata key")
            names.add(key)
            yield key, self.integer("I")

    def unchanged(self, path):
        def identity(stat):
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        # Windows fstat/stat can expose different ctime definitions. Compare each
        # API against itself, and bind the opened file to its path by file ID.
        if (
            identity(self.stat) != identity(os.fstat(self.stream.fileno()))
            or identity(self.path_stat) != identity(os.stat(path))
            or identity(self.stat)[:4] != identity(self.path_stat)[:4]
        ):
            raise OSError("GGUF changed during header inspection; retry when stable")


def read_scalar_strings(path, names, *, max_string_bytes=4096):
    """Read selected GGUF strings while skipping vocabularies and tensor data."""
    wanted = set(names)
    result = {}
    with open(path, "rb") as stream:
        reader = _HeaderStream(stream, path)
        for key, kind in reader.fields():
            if kind == 8:  # STRING
                if key in wanted:
                    value = reader.string(max_string_bytes, skip_oversized=True)
                    if value is not None:
                        result[key] = value
                    if wanted.issubset(result):
                        reader.unchanged(path)
                        return result
                else:
                    reader.skip_value(kind)
            else:
                reader.skip_value(kind)
        reader.unchanged(path)
    return result


def qwen3vl_text_only_variant(path):
    """Inspect the pin's explicit text-only marker without reading tensor data."""
    selected = {}
    fields = {
        "general.architecture": 8,
        "comfy.gguf.qwen3vl.text_only": 7,
        "comfy.gguf.qwen3vl.variant": 8,
    }
    with open(path, "rb") as stream:
        reader = _HeaderStream(stream, path)
        for key, kind in reader.fields():
            if key not in fields:
                reader.skip_value(kind)
            elif kind != fields[key]:
                raise ValueError("Invalid GGUF text-only encoder metadata type.")
            else:
                selected[key] = reader.string(256) if kind == 8 else bool(reader.integer("?"))
        reader.unchanged(path)
    if not selected.get("comfy.gguf.qwen3vl.text_only", False):
        return None
    variant = selected.get("comfy.gguf.qwen3vl.variant")
    if selected.get("general.architecture") != "qwen3vl" or variant not in ("4b", "8b"):
        raise ValueError("Qwen3-VL text-only metadata requires a Qwen3-VL 4b or 8b GGUF.")
    return variant


def _shape(value):
    if not isinstance(value, (list, tuple)) or len(value) > 8:
        raise ValueError("Invalid GGUF logical shape")
    if any(type(size) is not int or not 0 < size <= 2**31 for size in value):
        raise ValueError("Invalid GGUF logical shape")
    return list(value)


def read_gguf_header(path, *, include_storage=False):
    """Return detached shapes; optional adapter/storage fields never read tensor data."""
    from gguf.constants import GGML_QUANT_SIZES

    variants = ("modelspec.variant", "variant", "model_variant", "training_variant")
    with open(path, "rb") as stream:
        reader = _HeaderStream(stream, path)
        selected = {}
        for key, kind in reader.fields():
            expected = (
                8
                if key in ("general.architecture", "config", *variants)
                or include_storage
                and key in ("general.type", "adapter.type")
                or key.startswith("comfy.gguf.quant.")
                else 9
                if key.startswith("comfy.gguf.orig_shape.")
                else 7
                if key == "is_512"
                else 4
                if key == "general.alignment"
                else 6
                if include_storage and key == "adapter.lora.alpha"
                else None
            )
            if expected is None:
                reader.skip_value(kind)
                continue
            if kind != expected:
                raise ValueError("Invalid GGUF metadata field type")
            if kind == 8:
                selected[key] = reader.string(
                    256
                    if key in (*variants, "general.type", "adapter.type")
                    else MAX_METADATA_BYTES
                )
            elif kind == 9:
                element, count = reader.integer("I"), reader.integer("Q")
                if element != 5 or count > 8:
                    raise ValueError("Invalid GGUF original shape metadata")
                selected[key] = [reader.integer("i") for _ in range(count)]
            else:
                selected[key] = reader.integer(_SCALARS[kind])

        alignment = selected.get("general.alignment", 32)
        if alignment == 0 or alignment & (alignment - 1):
            raise ValueError("Invalid GGUF tensor alignment")
        descriptors = []
        header = {}
        for _ in range(reader.tensor_count):
            name = reader.string(4096)
            if name in header:
                raise ValueError("Duplicate GGUF tensor name")
            rank = reader.integer("I")
            if rank > 8:
                raise ValueError("Invalid GGUF logical shape")
            storage = _shape(list(reversed([reader.integer("Q") for _ in range(rank)])))
            dtype, offset = reader.integer("I"), reader.integer("Q")
            quant_size = GGML_QUANT_SIZES.get(dtype)
            if quant_size is None:
                raise ValueError("Unknown GGUF tensor type")
            block, width = quant_size
            if block != 1 and (not storage or storage[-1] % block):
                raise ValueError("GGUF quantized shape does not match its block size")
            descriptors.append((offset, math.prod(storage) * width // block))
            logical = storage
            original = selected.get("comfy.gguf.orig_shape." + name)
            if original is not None:
                logical = _shape(original)
                if math.prod(logical) != math.prod(storage):
                    raise ValueError("GGUF original shape does not match stored element count")
            raw_quant = selected.get("comfy.gguf.quant." + name)
            if raw_quant is not None:
                try:
                    quant = json.loads(raw_quant)
                except (ValueError, RecursionError) as error:
                    raise ValueError("Invalid GGUF quantization metadata") from error
                if not isinstance(quant, dict):
                    raise ValueError("Invalid GGUF quantization metadata")
                fmt = quant.get("format")
                if fmt is not None and not isinstance(fmt, str):
                    raise ValueError("Invalid GGUF quantization metadata format")
                if fmt in ("int4_compact_gemm", "int4_pytorch"):
                    raise ValueError("Q4_PT GGUF files are retired. Reconvert as Q8_CR.")
                if "orig_shape" in quant:
                    original = _shape(quant["orig_shape"])
                    packing = {"int4_cr": 2, "int1_cr": 8}.get(fmt)
                    if packing is None or len(original) != 2 or original[1] % packing:
                        raise ValueError("Unsupported GGUF packed logical shape metadata")
                    if storage != [original[0], original[1] // packing] or logical != storage:
                        raise ValueError(
                            "GGUF packed logical shape conflicts with storage metadata"
                        )
                    logical = original
            header[name] = {"shape": logical}
            if include_storage:
                header[name].update(gguf_type=int(dtype), storage_shape=storage.copy())
        data_start = ((reader.position + alignment - 1) // alignment) * alignment
        for offset, size in descriptors:
            if offset % alignment or data_start + offset + size > reader.stat.st_size:
                raise ValueError("GGUF tensor range exceeds its file or alignment")
        metadata = {
            key: selected[key] for key in (*variants, "config", "is_512") if key in selected
        }
        if include_storage:
            metadata.update(
                (key, selected[key])
                for key in ("general.type", "adapter.type", "adapter.lora.alpha")
                if key in selected
            )
        if metadata:
            header["__metadata__"] = metadata
        reader.unchanged(path)
        return header, selected.get("general.architecture")
