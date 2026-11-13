"""Intrinsic native utility roles, independent of filenames and publisher labels."""

from collections.abc import Mapping
import ast
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

IMAGE_UTILITY_CLASSES = frozenset(
    ("LotusD", "RT_DETR_v4", "SAM3", "SAM31", "DepthAnything3", "SeedVR2")
)
_PREDICATE_FILES = ("model_detection.py", "supported_models.py", "supported_models_base.py")


def _native_source_signatures():
    from ruinedfooocus.core.paths import APP_ROOT
    from ruinedfooocus.features.models.model_identity import file_signature

    base = APP_ROOT / "repositories/ComfyUI/comfy"
    return tuple(tuple(file_signature(base / name)) for name in _PREDICATE_FILES)


@lru_cache(maxsize=8)
def _native_predicates(sources, native_classes=IMAGE_UTILITY_CLASSES):
    """Run pinned shape predicates, without importing native or tensor runtimes.

    The detector body and BASE.matches remain native source. Only the requested
    role configurations are exposed; no model constructor or weight operation runs.
    """
    from ruinedfooocus.features.models.model_identity import file_signature

    trees = {}
    for signature in sources:
        path = Path(signature[0])
        if tuple(file_signature(path)) != signature:
            raise OSError("Native utility predicates changed during inspection.")
        with path.open("rb") as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Oversized native utility predicate source.")
        if tuple(file_signature(path)) != signature:
            raise OSError("Native utility predicates changed during inspection.")
        trees[path.name] = ast.parse(raw, filename=str(path))
    base = Path(sources[0][0]).parent
    wanted = {
        "count_blocks",
        "any_suffix_in",
        "calculate_transformer_depth",
        "detect_unet_config",
        "unet_prefix_from_state_dict",
    }
    functions = [
        node
        for node in trees["model_detection.py"].body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise ValueError("The pinned native utility predicates are unavailable.")
    scope = dict(json=json, math=math, logging=logging)
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(base / "model_detection.py"), "exec"
        ),
        scope,
    )
    classes = {
        node.name: node
        for tree in (trees["supported_models.py"], trees["supported_models_base.py"])
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    def literal(name, key):
        node = classes[name]
        for item in node.body:
            if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == key for target in item.targets
            ):
                return ast.literal_eval(item.value)
        for parent in node.bases:
            inherited = parent.id if isinstance(parent, ast.Name) else getattr(parent, "attr", None)
            if inherited in classes:
                return literal(inherited, key)
        raise ValueError("Missing native utility configuration.")

    match = next(
        node
        for node in classes["BASE"].body
        if isinstance(node, ast.FunctionDef) and node.name == "matches"
    )
    match.decorator_list = []
    exec(
        compile(
            ast.Module(body=[match], type_ignores=[]),
            str(base / "supported_models_base.py"),
            "exec",
        ),
        scope,
    )
    configurations = tuple(
        (
            name,
            SimpleNamespace(
                unet_config=literal(name, "unet_config"),
                required_keys=literal(name, "required_keys"),
            ),
        )
        for name in sorted(native_classes)
    )
    return scope, configurations


@lru_cache(maxsize=256)
def _passive_native(path, signature, native_sources, native_classes):
    from ruinedfooocus.features.models.model_identity import cached_safetensors_header

    if Path(path).suffix.lower() == ".gguf":
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        header, _ = read_gguf_header(path)
    else:
        header = cached_safetensors_header(path)
    if len(header) > 100000:
        return None
    shapes = {}
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(type(size) is not int or not 0 <= size <= 2**31 for size in shape)
        ):
            return None
        shapes[name] = SimpleNamespace(shape=tuple(shape), ndim=len(shape))
    scope, configurations = _native_predicates(native_sources, native_classes)
    prefixes = dict.fromkeys((scope["unet_prefix_from_state_dict"](shapes), ""))
    for prefix in prefixes:
        try:
            config = scope["detect_unet_config"](shapes, prefix, metadata=None)
            if not isinstance(config, dict):
                continue
            for name, contract in configurations:
                if scope["matches"](contract, config, shapes, unet_key_prefix=prefix):
                    return name
            # A primary non-utility configuration must not be reinterpreted from
            # unrelated auxiliary keys under a later prefix.
            return None
        except (
            AttributeError,
            IndexError,
            KeyError,
            NameError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            # Non-utility branches may need actual tensor operations. They are
            # deliberately outside this passive utility-only reader.
            return None
    return None


def passive_native_class(path, native_classes, *, include_gguf=False):
    """Current bounded header -> requested native role; never starts a runtime."""
    from ruinedfooocus.features.models.model_identity import file_signature

    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".sft") and not (
        include_gguf and path.suffix.lower() == ".gguf"
    ):
        return None
    try:
        before = tuple(file_signature(path))
        native_sources = _native_source_signatures()
        result = _passive_native(
            str(path.resolve()), before, native_sources, frozenset(native_classes)
        )
        return (
            result
            if before == tuple(file_signature(path))
            and native_sources == _native_source_signatures()
            else None
        )
    except (OSError, SyntaxError, ValueError, TypeError, KeyError, StopIteration):
        return None


def passive_utility_file(path):
    return passive_native_class(path, IMAGE_UTILITY_CLASSES)


def is_image_utility(metadata):
    """Consume the current, signature-bound role emitted by Models metadata."""
    if not isinstance(metadata, Mapping) or metadata.get("rf_model_role") != "image_utility_model":
        return False
    proof = metadata.get("_rf_intrinsic_utility")
    if not isinstance(proof, Mapping) or proof.get("native_class") not in IMAGE_UTILITY_CLASSES:
        return False
    signature = proof.get("signature")
    return (
        isinstance(signature, (list, tuple))
        and len(signature) == 4
        and isinstance(signature[0], str)
        and type(signature[1]) is int
        and type(signature[2]) is int
        and type(signature[3]) is int
    )
