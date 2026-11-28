"""Pinned mask bodies following input devices; native SciPy grow remains on CPU."""

import ast
import __future__
from functools import lru_cache
from pathlib import Path

FILES = ("comfy_extras/nodes_mask.py",)


def source_signatures():
    from ruinedfooocus.core.paths import APP_ROOT
    from ruinedfooocus.features.models.model_identity import file_signature

    return tuple(tuple(file_signature(APP_ROOT / "repositories/ComfyUI" / name)) for name in FILES)


@lru_cache(maxsize=4)
def _tree(signatures):
    from ruinedfooocus.features.models.model_identity import file_signature

    path = Path(signatures[0][0])
    if tuple(file_signature(path)) != signatures[0]:
        raise ValueError("The native mask source changed. Submit the mask again.")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or tuple(file_signature(path)) != signatures[0]:
        raise ValueError("The native mask source is oversized or changed.")
    return ast.parse(raw, filename=str(path))


def operations(signatures):
    from types import SimpleNamespace as NS

    import numpy as np
    import scipy.ndimage
    import torch

    if source_signatures() != signatures:
        raise ValueError("The native mask source changed. Submit the mask again.")
    scope = dict(np=np, scipy=scipy, torch=torch, IO=NS(NodeOutput=lambda *values: values))
    result = {}
    for name in (
        "GrowMask",
        "FeatherMask",
        "MaskComposite",
        "ImageToMask",
        "ImageColorToMask",
        "ThresholdMask",
        "InvertMask",
        "MaskToImage",
    ):
        cls = next(node for node in _tree(signatures).body if getattr(node, "name", None) == name)
        method = next(node for node in cls.body if getattr(node, "name", None) == "execute")
        method = ast.FunctionDef(**{**method.__dict__, "decorator_list": []})
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                FILES[0],
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            scope,
        )
        result[name] = scope["execute"]
    return NS(**result)
