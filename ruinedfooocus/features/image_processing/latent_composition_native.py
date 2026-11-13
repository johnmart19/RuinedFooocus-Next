"""Pinned native latent formulas without importing model or device managers."""

import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

FILES = ("nodes.py", "comfy_extras/nodes_mask.py", "comfy/utils.py")


def _node(tree, name, *, cls=None):
    body = tree.body
    if cls is not None:
        body = next(
            node.body for node in body if isinstance(node, ast.ClassDef) and node.name == cls
        )
    node = deepcopy(next(node for node in body if getattr(node, "name", None) == name))
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    return node


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
            raise ValueError("The native latent operation source changed during capture.")
        with path.open("rb") as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024 or tuple(file_signature(path)) != signature:
            raise ValueError("The native latent operation source is oversized or changed.")
        result[name] = ast.parse(raw, filename=str(path))
    return result


def operations(signatures):
    """Fresh namespaces retain the exact native formulas and no request tensors."""
    from types import SimpleNamespace as NS

    import torch

    if source_signatures() != signatures:
        raise ValueError("The native latent operations changed. Submit the inputs again.")
    trees = _trees(signatures)

    def run(scope, file, name, *, cls=None):
        node = _node(trees[file], name, cls=cls)
        exec(compile(ast.Module(body=[node], type_ignores=[]), file, "exec"), scope)
        return scope[name]

    plain_scope = dict(torch=torch)
    plain = run(plain_scope, FILES[0], "composite", cls="LatentComposite")
    masked_scope = dict(torch=torch, IO=NS(NodeOutput=lambda *values: values))
    repeat = run(masked_scope, FILES[2], "repeat_to_batch_size")
    masked_scope["comfy"] = NS(utils=NS(repeat_to_batch_size=repeat))
    run(masked_scope, FILES[1], "composite")
    masked = run(masked_scope, FILES[1], "execute", cls="LatentCompositeMasked")
    return NS(plain=plain, masked=masked)
