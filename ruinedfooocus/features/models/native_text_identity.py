"""Passive names for newly distinct native encoder roles, never model loading."""

import ast
from enum import Enum
from functools import lru_cache
from types import SimpleNamespace

NEW_NATIVE_TEXT_TYPES = frozenset({"WEVISONE_2B", "WEVISONE_4B", "QWEN3VL_2B"})
WEVISONE_TYPES = frozenset({"WEVISONE_2B", "WEVISONE_4B"})


def _source():
    from ruinedfooocus.core.paths import APP_ROOT

    return APP_ROOT / "repositories/ComfyUI/comfy/sd.py"


@lru_cache(maxsize=4)
def _detector(signature):
    from pathlib import Path
    from ruinedfooocus.features.models.model_identity import file_signature

    path = Path(signature[0])
    if tuple(file_signature(path)) != signature:
        raise OSError("Native encoder source changed during inspection.")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or tuple(file_signature(path)) != signature:
        raise ValueError("Native encoder source is oversized or changed.")
    tree = ast.parse(raw, filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("TEModel", "detect_te_model")
    ]
    if {node.name for node in selected} != {"TEModel", "detect_te_model"}:
        raise ValueError("The native encoder header contract is unavailable.")
    scope = dict(Enum=Enum)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope["detect_te_model"]


def passive_native_text_type(shapes, *, allowed_types=NEW_NATIVE_TEXT_TYPES):
    """Return only reviewed new native enum names; other families stay unchanged."""
    from ruinedfooocus.features.models.model_identity import file_signature

    if not isinstance(shapes, dict) or len(shapes) > 100000:
        return None
    values = {}
    for key, shape in shapes.items():
        if (
            not isinstance(key, str)
            or not isinstance(shape, (list, tuple))
            or len(shape) > 8
            or any(type(size) is not int or not 0 <= size <= 2**31 for size in shape)
        ):
            return None
        values[key] = SimpleNamespace(shape=tuple(shape), ndim=len(shape))
    try:
        path = _source()
        before = tuple(file_signature(path))
        result = _detector(before)(values)
        name = getattr(result, "name", None)
        return name if name in allowed_types and tuple(file_signature(path)) == before else None
    except (
        OSError,
        SyntaxError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        NameError,
        StopIteration,
    ):
        return None
