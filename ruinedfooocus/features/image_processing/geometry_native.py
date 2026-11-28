"""Pinned pure geometry bodies following their request-local tensor devices."""

import ast
import __future__
from enum import Enum
from functools import lru_cache
from pathlib import Path

FILES = (
    "comfy_extras/nodes_images.py",
    "comfy_extras/nodes_post_processing.py",
    "comfy/utils.py",
    "comfy_extras/color_util.py",
)


def source_signatures():
    from ruinedfooocus.core.paths import APP_ROOT
    from ruinedfooocus.features.models.model_identity import file_signature

    return tuple(tuple(file_signature(APP_ROOT / "repositories/ComfyUI" / name)) for name in FILES)


@lru_cache(maxsize=4)
def _trees(signatures):
    from ruinedfooocus.features.models.model_identity import file_signature

    result = {}
    for name, signature in zip(FILES, signatures):
        path = Path(signature[0])
        if tuple(file_signature(path)) != signature:
            raise ValueError("The native geometry source changed. Submit the image again.")
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024 or tuple(file_signature(path)) != signature:
            raise ValueError("The native geometry source is oversized or changed.")
        result[name] = ast.parse(raw, filename=str(path))
    return result


def operations(signatures):
    import math
    from types import SimpleNamespace as NS

    import torch

    if source_signatures() != signatures:
        raise ValueError("The native geometry source changed. Submit the image again.")
    trees = _trees(signatures)
    # RF owns the output writer. The native crop's graph preview must not write
    # another copy of the source; only its preview transport is suppressed.
    output = NS(NodeOutput=lambda *values, **kwargs: values)
    scope = dict(torch=torch, math=math, Enum=Enum, io=output, IO=output)
    scope["UI"] = NS(PreviewImage=lambda image: None)

    def run(file, name, cls=None):
        body = trees[file].body
        if cls:
            body = next(
                node.body for node in body if isinstance(node, ast.ClassDef) and node.name == cls
            )
        node = next(node for node in body if getattr(node, "name", None) == name)
        if isinstance(node, ast.FunctionDef):
            node = ast.FunctionDef(**{**node.__dict__, "decorator_list": []})
        code = compile(
            ast.Module(body=[node], type_ignores=[]),
            file,
            "exec",
            flags=__future__.annotations.compiler_flag,
        )
        exec(code, scope)
        return scope[name]

    common_upscale = run(FILES[2], "common_upscale")
    scope["comfy"] = NS(utils=NS(common_upscale=common_upscale))
    run(FILES[1], "ResizeType")
    for name in (
        "is_image",
        "init_image_mask_input",
        "finalize_image_mask_input",
        "scale_dimensions",
    ):
        run(FILES[1], name)
    resize = run(FILES[1], "execute", "ResizeImageMaskNode")
    crop = run(FILES[0], "execute", "ImageCropV2")
    bounding_box = run(FILES[0], "execute", "BoundingBox")
    rotate = run(FILES[0], "execute", "ImageRotate")
    flip = run(FILES[0], "execute", "ImageFlip")
    run(FILES[0], "PadType")
    run(FILES[0], "PadStyle")
    run(FILES[3], "hex_to_rgb")
    for name in ("split_padding", "mirror_indices", "pad_mirror", "feather_weight", "pad_color"):
        run(FILES[0], name)
    pad = run(FILES[0], "execute", "PadImageNode")
    return NS(
        resize=resize, crop=crop, bounding_box=bounding_box, rotate=rotate, flip=flip, pad=pad
    )
