"""Pure pinned image operations, without importing dataset or model runtimes."""

import ast
from enum import Enum
from functools import lru_cache
from pathlib import Path

FILES = (
    "comfy_extras/nodes_dataset.py",
    "comfy_extras/nodes_compositing.py",
    "comfy_extras/nodes_mask.py",
    "node_helpers.py",
    "comfy/utils.py",
)


def source_signatures():
    from ruinedfooocus.core.paths import APP_ROOT
    from ruinedfooocus.features.models.model_identity import file_signature

    root = APP_ROOT / "repositories/ComfyUI"
    return tuple(tuple(file_signature(root / name)) for name in FILES)


@lru_cache(maxsize=4)
def _trees(signatures):
    from ruinedfooocus.features.models.model_identity import file_signature

    result = {}
    for name, signature in zip(FILES, signatures):
        path = Path(signature[0])
        if tuple(file_signature(path)) != signature:
            raise OSError("The native image operation source changed during capture.")
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024 or tuple(file_signature(path)) != signature:
            raise ValueError("The native image operation source is oversized or changed.")
        result[name] = ast.parse(raw, filename=str(path))
    return result


def _node(tree, name, *, cls=None):
    body = tree.body
    if cls is not None:
        body = next(
            node.body for node in body if isinstance(node, ast.ClassDef) and node.name == cls
        )
    node = next(node for node in body if getattr(node, "name", None) == name)
    if isinstance(node, ast.FunctionDef):
        # This is an extracted method, with its native cls argument kept intact.
        node = ast.FunctionDef(**{**node.__dict__, "decorator_list": []})
    return node


def modes():
    """Native finite mode names are safe to read before importing Torch."""
    signatures = source_signatures()
    node = _node(_trees(signatures)[FILES[1]], "PorterDuffMode")
    scope = dict(Enum=Enum)
    exec(compile(ast.Module(body=[node], type_ignores=[]), FILES[1], "exec"), scope)
    return tuple(value.name for value in scope["PorterDuffMode"])


def operations(signatures, check_cancelled):
    """A fresh request namespace; cached code never owns images or callbacks."""
    import logging
    import math
    from types import SimpleNamespace as NS

    import numpy as np
    from PIL import Image
    import torch

    if source_signatures() != signatures:
        raise ValueError("The native image operations changed. Submit the images again.")
    trees = _trees(signatures)
    scope = dict(Enum=Enum, Image=Image, np=np, torch=torch, logging=logging, math=math)
    output = NS(NodeOutput=lambda *values: values)
    scope.update(io=output, IO=output)

    def run(file, name, *, cls=None, alias=None):
        node = _node(trees[file], name, cls=cls)
        exec(compile(ast.Module(body=[node], type_ignores=[]), file, "exec"), scope)
        return scope[alias or name]

    repeat = run(FILES[4], "repeat_to_batch_size")
    alpha_fix = run(FILES[3], "image_alpha_fix")
    scope.update(
        comfy=NS(utils=NS(repeat_to_batch_size=repeat)), node_helpers=NS(image_alpha_fix=alpha_fix)
    )
    run(FILES[1], "PorterDuffMode")
    run(FILES[1], "porter_duff_composite")
    porter = run(FILES[1], "execute", cls="PorterDuffImageComposite")
    run(FILES[2], "composite")
    masked = run(FILES[2], "execute", cls="ImageCompositeMasked")
    to_pil = run(FILES[0], "tensor_to_pil")
    from_pil = run(FILES[0], "pil_to_tensor")

    def checked_to_pil(value):
        check_cancelled()
        return to_pil(value)

    scope["tensor_to_pil"] = checked_to_pil
    grid = run(FILES[0], "_group_process", cls="ImageGridNode")
    return NS(grid=grid, masked=masked, porter=porter, from_pil=from_pil)
