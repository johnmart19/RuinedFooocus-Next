"""Bounded curve and named-effect recipes for the native GLSL image node.

This module is passive. CPU policy is authoritative; native EGL/GLES imports
occur only on an accelerated request with installed GL packages. CPU/source
validation must not execute the native GL module.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.image_processing import image_histogram
from ruinedfooocus.features.image_processing.native_vision import _number

KINDS = ("image-curves", "glsl-effect")
LABELS = {"image-curves": "RGB curves", "glsl-effect": "GLSL image effect"}
EFFECTS = {"invert": "Invert RGB", "bt709-grayscale": "BT.709 grayscale", "posterize": "Posterize"}
CURVE_FIELDS = ("curve_master", "curve_red", "curve_green", "curve_blue")
DEFAULTS = {
    **{field: [[0.0, 0.0], [1.0, 1.0]] for field in CURVE_FIELDS},
    "curve_interpolation": "monotone_cubic",
    "shader_effect": "invert",
    "shader_amount": 1.0,
    "shader_levels": 8,
}
FIELDS = tuple(DEFAULTS)
KEY = "_image_shader_request"
MAX_PIXELS = 4194304
MAX_DIMENSION = 4096
PROGRAM_SHA256 = {
    "curves": "6938174652569a846460594f13cbcfd51170fc3172d684b5c48b1586a161bd0f",
    "invert": "30fc2a953a5a4b6710868c04fc72a58fcf298c4b0f485b69666e0e4e3e02b112",
    "bt709-grayscale": "3ba2223959ec8289230cb0488295e414699b4733c83b44943737b6fe02bc716c",
    "posterize": "b06a6c5a1c90a9e62622d0edb7e415b7927a5f673cf1c0904c36b167058ea4fb",
}


@dataclass(frozen=True)
class ShaderRequest:
    kind: str
    source: image_histogram.HistogramRequest
    parameters: tuple[tuple[str, object], ...]
    recipe: str
    program: str
    program_sha256: str


def curve_points(value):
    if not isinstance(value, (tuple, list)) or not 2 <= len(value) <= 32:
        raise ValueError("Each curve needs 2 to 32 input/output point pairs.")
    points = []
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each curve row needs one input and one output value.")
        x, y = (_number(item, "Curve coordinate", 0, 1) for item in pair)
        if points and x - points[-1][0] < 0.000001:
            raise ValueError("Curve input values must increase by at least 0.000001.")
        points.append((x, y))
    if points[0][0] != 0 or points[-1][0] != 1:
        raise ValueError("Each curve must start at input 0 and end at input 1.")
    return tuple(points)


def parameters(kind, data):
    if kind == "image-curves":
        interpolation = data.get("curve_interpolation", "monotone_cubic")
        if interpolation not in ("linear", "monotone_cubic"):
            raise ValueError("Choose linear or monotone cubic curve interpolation.")
        return {
            **{field: curve_points(data.get(field, DEFAULTS[field])) for field in CURVE_FIELDS},
            "curve_interpolation": interpolation,
        }
    if kind == "glsl-effect":
        effect = data.get("shader_effect", "invert")
        if not isinstance(effect, str) or effect not in EFFECTS:
            raise ValueError("Choose one of the named GLSL image effects.")
        result = {
            "shader_effect": effect,
            "shader_amount": _number(data.get("shader_amount", 1.0), "Effect amount", 0, 1),
        }
        if effect == "posterize":
            result["shader_levels"] = _number(
                data.get("shader_levels", 8), "Posterize levels", 2, 32, integer=True
            )
        return result
    raise ValueError("Choose RGB curves or a named GLSL image effect.")


def capture_assets(data, manager, models=None):
    from PIL import Image
    from ruinedfooocus.features.image_processing.depth_utilities import kind_from

    kind = kind_from(data)
    if data.get("inpaint_toggle"):
        raise ValueError("Turn off inpainting before applying curves or GLSL image effects.")
    source = image_histogram.capture(data.get("input_image"))
    if max(source.size) > MAX_DIMENSION or source.size[0] * source.size[1] > MAX_PIXELS:
        raise ValueError(
            "Curves and GLSL effects accept one image up to 4 megapixels and 4096 pixels per side."
        )
    options = parameters(kind, data)
    recipe = "curves" if kind == "image-curves" else options["shader_effect"]
    # The path comes only from the bounded enum above, never caller-supplied code.
    program_bytes = (
        resource_path("native_image_effects", recipe + ".frag").read_bytes().replace(b"\r\n", b"\n")
    )
    if (
        len(program_bytes) > 4096
        or hashlib.sha256(program_bytes).hexdigest() != PROGRAM_SHA256[recipe]
    ):
        raise ValueError(
            "The bundled image effect recipe changed. Restore the reviewed recipe before running."
        )
    request = ShaderRequest(
        kind,
        source,
        tuple(options.items()),
        recipe,
        program_bytes.decode("utf-8"),
        hashlib.sha256(program_bytes).hexdigest(),
    )
    result = dict(
        data,
        input_image=Image.frombytes("RGBA", source.size, source.rgba),
        controlnet={"type": kind},
        **options,
    )
    result[KEY] = request
    result.pop("image_utility_source", None)
    return result, ()


def worker_data(data):
    from ruinedfooocus.features.image_processing.image_filters import worker_data as common

    return common(data)


def cpu_mode():
    """Read already-applied policy only; never discover or initialize a device."""
    import os
    import sys

    flags = [
        getattr(sys.modules.get(name), "args", None)
        for name in ("ruinedfooocus.app.arguments", "comfy.cli_args")
    ]
    if (
        "--cpu" in sys.argv
        or os.environ.get("TORCH_PLATFORM", "").lower() == "cpu"
        or any(getattr(args, "cpu", False) for args in flags)
    ):
        return True
    return not any(args is not None for args in flags)


def gl_dependencies_present():
    """Passive package presence only; missing GL must not trigger an import."""
    from importlib.util import find_spec

    try:
        return all(find_spec(name) is not None for name in ("comfy_angle", "OpenGL"))
    except (ImportError, ValueError):
        return False


def gl_pixels(request, tensor, curves, check_cancelled):
    import torch
    from ruinedfooocus.features.image_processing.depth_utilities import native_call

    check_cancelled()
    options = dict(request.parameters)
    # Importing this native module initializes its EGL/GLES runtime. Keep the
    # import deferred; CPU/source tests use the pinned execute body with a traced
    # renderer boundary and never import or execute the GL module itself.
    try:
        from comfy_extras.nodes_glsl import GLSLShader
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "The native GLSL runtime could not initialize. Check its local EGL/GLES dependencies."
        ) from error

    class StillImageShader(GLSLShader):
        @classmethod
        def _build_ui_output(cls, image_list, output_batch):
            # RF owns the preview/output writer; native graph preview saving
            # would create duplicate temporary copies of the source and result.
            return {}

    check_cancelled()
    width, height = request.source.size
    floats = {"u_float0": options["shader_amount"]} if request.kind == "glsl-effect" else {}
    ints = {"u_int0": options["shader_levels"]} if request.recipe == "posterize" else {}
    with torch.inference_mode():
        outputs = native_call(
            check_cancelled,
            StillImageShader.execute,
            request.program,
            {"size_mode": "from_input"},
            {"image0": tensor},
            floats=floats,
            ints=ints,
            bools={},
            curves=curves,
        )
    values = getattr(outputs, "result", outputs)
    if len(values) != 4:
        raise RuntimeError("The native GLSL node returned an unexpected output count.")
    output = values[0]
    if tuple(output.shape) != (1, height, width, 4) or not torch.isfinite(output).all():
        raise RuntimeError("The native GLSL output has invalid dimensions or non-finite pixels.")
    return output[..., :3]


def process(owner, data, manager, check_cancelled):
    import numpy as np
    import torch
    from PIL import Image
    from ruinedfooocus.core.device_fallback import warn_cpu_fallback
    from ruinedfooocus.features.image_processing import image_effect_cpu
    from ruinedfooocus.features.image_processing.depth_utilities import kind_from, native_call
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output

    request = data.get(KEY)
    data.pop("image_utility_source", None)
    if not isinstance(request, ShaderRequest) or request.kind != kind_from(data):
        raise ValueError("Capture the curves or effect source before processing. Submit it again.")
    check_cancelled()
    options = dict(request.parameters)
    curves = {}
    if request.kind == "image-curves":
        from comfy_extras.nodes_curve import CurveEditor

        for index, field in enumerate(CURVE_FIELDS):
            curves[f"u_curve{index}"] = native_call(
                check_cancelled,
                CurveEditor.execute,
                {"points": options[field], "interpolation": options["curve_interpolation"]},
            )[0]
    width, height = request.source.size
    pixels = np.frombuffer(request.source.rgba, dtype=np.uint8).reshape(height, width, 4)
    tensor = torch.from_numpy(pixels[..., :3].copy()).to(dtype=torch.float32).div_(255).unsqueeze(0)
    backend, fallback, output = "cpu", False, None
    with torch.inference_mode():
        if not cpu_mode():
            check_cancelled()
            if gl_dependencies_present():
                try:
                    output = gl_pixels(request, tensor, curves, check_cancelled)
                    backend = "native-gl"
                    operators = ["GLSLShader"]
                except (ImportError, OSError, RuntimeError):
                    check_cancelled()
                    fallback = True
                    warn_cpu_fallback(
                        "curves and named image effects",
                        "the native EGL/GLES operation is unavailable or failed",
                    )
            else:
                fallback = True
                warn_cpu_fallback(
                    "curves and named image effects",
                    "native EGL/GLES packages are unavailable",
                )
        if output is None:
            output, operators = image_effect_cpu.render(request, tensor, curves, check_cancelled)
    check_cancelled()
    if tuple(output.shape) != tuple(tensor.shape) or not torch.isfinite(output).all():
        raise ValueError("The image effect returned invalid dimensions or non-finite pixels.")
    rgb = output[0].detach().float().cpu().numpy().copy()
    alpha = pixels[..., 3:4].astype(np.float32) / 255
    result = np.concatenate((rgb, alpha), axis=-1)
    check_cancelled()
    source = request.source
    data["image_utility_source"] = {
        "operation": request.kind,
        "native_node": operators[0],
        "operators": operators,
        "execution_backend": backend,
        "acceleration_fallback": fallback,
        "native_revision": image_histogram.NATIVE_REVISION,
        "curve_node": "CurveEditor" if curves else None,
        "curve_lut_size": 256 if curves else 0,
        "shader_recipe": request.recipe,
        "shader_sha256": request.program_sha256,
        "output_slot": 0,
        "width": width,
        "height": height,
        "source_sha256": source.source_sha256,
        "source_sha256_layout": "oriented RGBA8 bytes with width and height",
        "input_mode": source.input_mode,
        "has_alpha": source.has_alpha,
        "has_icc_profile": source.has_icc_profile,
        "working_mode": "encoded RGB8 to native float32",
        "color_policy": "no ICC conversion or linearization",
        "alpha": "source bytes preserved",
        **options,
    }
    if advanced_output(data):
        return [result]
    return [Image.fromarray((np.clip(result, 0, 1) * 255).round().astype(np.uint8), "RGBA")]


def output_metadata(data):
    from ruinedfooocus.features.image_processing import image_filters

    return image_filters.output_metadata({**data, image_filters.KEY: data[KEY]})


def replay_values(metadata):
    from ruinedfooocus.features.image_processing.depth_utilities import kind_from

    kind = kind_from(metadata)
    options = parameters(kind, metadata)
    for field in CURVE_FIELDS:
        if field in options:
            options[field] = [list(pair) for pair in options[field]]
    return {
        **deepcopy(DEFAULTS),
        **options,
        "input_image": None,
        "image_source": None,
        "inpaint_toggle": False,
        "cn_selection": LABELS[kind],
        "prompt": "",
        "negative": "",
        "automatic_upscale": "Off",
        "automatic_upscale_target": "Generated image",
        "model_recipe_path": "",
        "reference_adapter": "none",
    }
