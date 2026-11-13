"""Pinned color-finishing classes without server imports or device discovery."""

import ast
import __future__
from functools import lru_cache
from pathlib import Path

FILES = (
    "comfy_extras/nodes_post_processing.py",
    "node_helpers.py",
    "comfy_extras/nodes_text_overlay.py",
    "comfy_extras/nodes_morphology.py",
    "comfy_extras/nodes_images.py",
)


def source_signatures():
    from ruinedfooocus.core.paths import APP_ROOT
    from ruinedfooocus.features.models.model_identity import file_signature

    return tuple(tuple(file_signature(APP_ROOT / "repositories/ComfyUI" / name)) for name in FILES)


@lru_cache(maxsize=4)
def _trees(signatures):
    from ruinedfooocus.features.models.model_identity import file_signature

    result = []
    for signature in signatures:
        path = Path(signature[0])
        if tuple(file_signature(path)) != signature:
            raise ValueError("The native color source changed. Submit the images again.")
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024 or tuple(file_signature(path)) != signature:
            raise ValueError("The native color source is oversized or changed.")
        result.append(ast.parse(raw, filename=str(path)))
    return result


def operations(signatures, check_cancelled, *, policy, transfer=False, finishing=None):
    import math
    from types import SimpleNamespace as NS

    import numpy as np
    from PIL import Image
    import torch

    if source_signatures() != signatures:
        raise ValueError("The native color source changed. Submit the images again.")
    trees = _trees(signatures)
    output = NS(NodeOutput=lambda *v: v)
    scope = dict(np=np, Image=Image, torch=torch, math=math, io=output, IO=output)
    if transfer:
        import kornia

        scope["kornia"] = kornia

    def compile_nodes(nodes, filename):
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                filename,
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            scope,
        )

    helper = next(
        node for node in trees[1].body if getattr(node, "name", None) == "image_alpha_fix"
    )
    compile_nodes([helper], FILES[1])
    scope["node_helpers"] = NS(image_alpha_fix=scope["image_alpha_fix"])

    class Progress:
        def __init__(self, count):
            check_cancelled()

        def update(self, count):
            check_cancelled()

    def reject_resize(*args, **kwargs):
        raise ValueError("Blend images must share one frame; resize explicitly first.")

    # The job's configured native device/intermediate policy is injected without
    # importing device management while constructing this pure namespace.
    # Pixel algorithms below remain the captured upstream class bodies.
    scope["comfy"] = NS(
        model_management=NS(
            get_torch_device=lambda: policy.device,
            intermediate_device=lambda: policy.intermediate_device,
            intermediate_dtype=lambda: policy.intermediate_dtype,
        ),
        utils=NS(ProgressBar=Progress, common_upscale=reject_resize),
    )

    def compile_class(index, name):
        node = next(node for node in trees[index].body if getattr(node, "name", None) == name)
        node = ast.ClassDef(
            **{
                **node.__dict__,
                "bases": [],
                "keywords": [],
                "decorator_list": [],
                "body": [
                    part for part in node.body if getattr(part, "name", None) != "define_schema"
                ],
            }
        )
        compile_nodes([node], FILES[index])

    names = ["Blend", "Quantize", *(("ColorTransfer",) if transfer else ())]
    for name in names:
        compile_class(0, name)
    if finishing == "text-overlay":
        from PIL import ImageColor, ImageDraw, ImageFont

        scope.update(
            PILImage=Image, ImageColor=ImageColor, ImageDraw=ImageDraw, ImageFont=ImageFont
        )
        compile_class(2, "TextOverlay")
        names.append("TextOverlay")
    elif finishing == "morphology":
        from kornia import morphology

        for name in (
            "dilation",
            "erosion",
            "opening",
            "closing",
            "gradient",
            "top_hat",
            "bottom_hat",
        ):
            scope[name] = getattr(morphology, name)
        compile_class(3, "Morphology")
        names.append("Morphology")
    elif finishing == "sdr-to-hdr":
        constants = {
            "_REC709_TO_REC2020",
            "_REC2020_LUMA",
            "_SDR_WHITE_NITS",
            "_HLG_PEAK_NITS",
            "_HLG_A",
            "_HLG_B",
            "_HLG_C",
            "_LOTTES_HDR_PARAMETERS",
        }
        functions = {
            "srgb_to_linear",
            "_convert_rgb_primaries",
            "_rgb_luminance",
            "_sdr_to_hdr_lottes",
        }
        body = [
            node
            for node in trees[4].body
            if getattr(node, "name", None) in functions
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets
            )
        ]
        compile_nodes(body, FILES[4])
        compile_class(4, "ImageSDRToHDR")
        names.append("ImageSDRToHDR")
    elif finishing is not None:
        raise ValueError("Choose a supported native finishing operation.")
    return NS(**{name: scope[name] for name in names})
