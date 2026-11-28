"""Capture implicit native GGUF vision sources without opening tensor data."""

import ast
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from pathlib import Path
import re
from types import FunctionType, SimpleNamespace

from ruinedfooocus.features.models.gguf_metadata import _HeaderStream
from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_native_gguf_projectors"


def _source():
    from ruinedfooocus.core.paths import APP_ROOT

    return APP_ROOT / "repositories/molbal_comfyui_gguf/loader.py"


def _directory_names(root):
    # Preserve native enumeration order and ambiguity behavior, with bounded work.
    with os.scandir(root) as entries:
        names = []
        for entry in entries:
            if len(names) == 16384:
                raise ValueError("Too many files beside the GGUF text encoder.")
            names.append(entry.name)
        return names


@lru_cache(maxsize=4)
def _resolver(signature):
    """Execute only the pinned resolver prefix, before its first weight read."""
    path = Path(signature[0])
    if tuple(file_signature(path)) != signature:
        raise ValueError("Native GGUF source changed during inspection.")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or tuple(file_signature(path)) != signature:
        raise ValueError("Native GGUF source is oversized or changed.")
    tree = ast.parse(raw, filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("strip_quant_suffix", "gguf_mmproj_loader")
    ]
    if {node.name for node in selected} != {"strip_quant_suffix", "gguf_mmproj_loader"}:
        raise ValueError("The native GGUF projector resolver is unavailable.")
    lookup = next(node for node in selected if node.name == "gguf_mmproj_loader")
    boundaries = [
        index
        for index, node in enumerate(lookup.body)
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "gguf_sd_loader"
            for child in ast.walk(node)
        )
    ]
    if not boundaries:
        raise ValueError("The native GGUF projector read boundary is unavailable.")
    lookup.body = lookup.body[: boundaries[0]] + [
        ast.Return(value=ast.Name(id="target", ctx=ast.Load()))
    ]
    scope = dict(os=SimpleNamespace(path=os.path, listdir=_directory_names), re=re, logging=logging)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["gguf_mmproj_loader"]


def _uses_projector(path):
    selected = {}
    with path.open("rb") as stream:
        reader = _HeaderStream(stream, path)
        for key, kind in reader.fields():
            expected = {
                "general.architecture": 8,
                "comfy.gguf.qwen3vl.text_only": 7,
            }.get(key)
            if expected is None:
                reader.skip_value(kind)
            elif kind != expected:
                raise ValueError("Invalid GGUF text encoder metadata type.")
            else:
                selected[key] = reader.string(256) if kind == 8 else bool(reader.integer("?"))
        reader.unchanged(path)
    arch = selected.get("general.architecture")
    return arch in ("qwen2vl", "qwen35") or (
        arch == "qwen3vl" and not selected.get("comfy.gguf.qwen3vl.text_only", False)
    )


@dataclass(frozen=True)
class GGUFProjector:
    encoder: tuple
    source: tuple
    projector: tuple | None
    selected: str | None

    def verify(self):
        for signature in (self.encoder, self.source, self.projector):
            if signature is not None and tuple(file_signature(signature[0])) != signature:
                raise ValueError("A captured GGUF text encoder or projector changed. Submit again.")


def capture(paths):
    result = []
    for filename in paths:
        path = Path(filename).resolve()
        if path.suffix.casefold() != ".gguf":
            continue
        encoder = tuple(file_signature(path))
        if not _uses_projector(path):
            continue
        source = tuple(file_signature(_source()))
        target = _resolver(source)(str(path))
        item = GGUFProjector(
            encoder,
            source,
            tuple(file_signature(target)) if target else None,
            str(Path(target).absolute()) if target else None,
        )
        item.verify()
        result.append(item)
    return tuple(result)


def captured(data):
    value = (data or {}).get(KEY)
    if value is not None and (
        not isinstance(value, tuple) or any(not isinstance(item, GGUFProjector) for item in value)
    ):
        raise ValueError("Capture typed GGUF projector sources before loading.")
    if value is not None:
        for item in value:
            item.verify()
    return value


def capture_assets(data, artifacts):
    """Freeze local sources now; newly acquired encoders are resolved after preparation."""
    previous = captured(data)
    known = {item.encoder[0] for item in previous or ()}
    paths = [
        item.request.destination
        for item in artifacts
        if item.role == "clip"
        and item.request.destination.is_file()
        and str(item.request.destination.resolve()) not in known
    ]
    result = dict(data)
    result[KEY] = (*previous, *capture(paths)) if previous is not None else capture(paths)
    return result


def unchanged(items):
    """Legacy direct callers take a fresh selection; queued callers use their capture."""
    try:
        return items is None or all(capture([item.encoder[0]]) == (item,) for item in items)
    except (OSError, ValueError):
        return False


def for_paths(paths, items):
    if items is None:
        return capture(paths)
    result = []
    for filename in paths:
        path = Path(filename).resolve()
        if path.suffix.casefold() != ".gguf" or not _uses_projector(path):
            continue
        item = next((item for item in items if item.encoder[0] == str(path)), None)
        if item is None:
            raise ValueError("The GGUF text encoder projector was not captured. Submit again.")
        item.verify()
        result.append(item)
    return tuple(result)


@contextmanager
def read_scope(items):
    """Lease the implicit sidecars; the caller already owns each encoder's lease."""
    from ruinedfooocus.core.preparation import preparation_service
    from ruinedfooocus.app.resource_lifecycle import current_job

    with ExitStack() as stack:
        leased = set()
        for item in items:
            if item.projector is not None and item.projector[0] not in leased:
                stack.enter_context(
                    preparation_service.artifact_lease(
                        Path(item.projector[0]), cancelled=getattr(current_job(), "cancelled", None)
                    )
                )
                leased.add(item.projector[0])
            item.verify()
        yield
        for item in items:
            item.verify()


def bind(native, path, items):
    """Scope native reader globals to the captured selection, never mutate its module."""
    item = next((item for item in items if item.encoder[0] == str(Path(path).resolve())), None)
    if item is None:
        return native
    item.verify()
    projector = native.__globals__["gguf_mmproj_loader"]
    if any(
        str(Path(function.__code__.co_filename).resolve()) != item.source[0]
        for function in (native, projector)
    ):
        raise ValueError("The imported native GGUF reader differs from the captured source.")
    names = [Path(item.selected).name] if item.selected is not None else []
    scope = dict(projector.__globals__)
    scope["os"] = SimpleNamespace(path=os.path, listdir=lambda root: names)
    reader = scope["gguf_sd_loader"]

    def read(filename, *args, **kwargs):
        item.verify()
        if item.projector is None or str(Path(filename).resolve()) != item.projector[0]:
            raise ValueError("Native GGUF projector differs from the captured source.")
        return reader(filename, *args, **kwargs)

    scope["gguf_sd_loader"] = read
    scoped_projector = FunctionType(
        projector.__code__, scope, projector.__name__, projector.__defaults__
    )
    clip_scope = dict(native.__globals__, gguf_mmproj_loader=scoped_projector)
    return FunctionType(native.__code__, clip_scope, native.__name__, native.__defaults__)
