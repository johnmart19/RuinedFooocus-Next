"""Captured optional-image requests and the generic worker's native CLI caller."""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import MappingProxyType

from ruinedfooocus.backends.sdcpp import activation
from ruinedfooocus.backends.sdcpp.contracts import (
    GenerationRequest,
    ModelSpec,
    UnsupportedRequest,
    assess,
    choose_runtime,
    inspect_model,
    with_model_hash,
)
from ruinedfooocus.backends.sdcpp.runtime import generate
from ruinedfooocus.core.device_fallback import warn_cpu_fallback
from ruinedfooocus.core.paths import data_path
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.packaging.backend_preferences import saved_cpu_requested

ENGINES = ("auto", "native", "sdcpp-vulkan", "sdcpp-cpu")
SNAPSHOT_KEY = "_sdcpp_snapshot"
PREPARED_KEY = "_sdcpp_prepared"
HOOKS_KEY = "_sdcpp_execution"


@dataclass(frozen=True)
class ImageSnapshot:
    engine: str
    name: str
    path: Path
    signature: tuple
    architecture: str
    sha256: str
    device: str
    data: object
    controls: object
    native_gpu: bool
    cpu_forced: bool


@dataclass(frozen=True)
class PreparedImage:
    snapshot: ImageSnapshot
    model: object
    activations: tuple
    input_geometry: tuple = ()
    input_signature: tuple = ()


@dataclass(frozen=True)
class ExecutionHooks:
    check_cancelled: object
    compute_lease_factory: object


def cpu_requested(values, launch_args=None):
    """Legacy explicit image CPU and global CPU remain authoritative."""
    if saved_cpu_requested(values):
        return True
    if launch_args is None:
        module = sys.modules.get("ruinedfooocus.app.arguments")
        launch_args = getattr(module, "args", None)
    status = getattr(launch_args, "_compute_status", {})
    backend = status.get("backend", {})
    if "--cpu" in status.get("cli", ()) or backend.get("cpu_everywhere"):
        return True
    return bool(getattr(launch_args, "cpu", False) and backend.get("cpu_origin") != "automatic")


def initialized_native_gpu():
    """Read the already initialized manager's state; never call device queries."""
    manager = sys.modules.get("comfy.model_management")
    state = getattr(manager, "cpu_state", None)
    return getattr(state, "name", None) in ("GPU", "MPS")


def _engine(data, values, forced):
    engine = data.get("image_engine", values.get("image_engine", "auto"))
    if engine not in ENGINES:
        raise ValueError("Unknown image engine selection")
    if forced and engine == "sdcpp-vulkan":
        raise ValueError(
            "Explicit CPU selection prevents Vulkan image execution; select native or stable-diffusion.cpp CPU"
        )
    return "native" if forced and engine == "auto" else engine


def _controls(data):
    module = sys.modules.get("ruinedfooocus.features.image_editing.controlnet")
    return deepcopy(
        module.get_settings(data) if module is not None else data.get("controlnet") or {}
    )


def capture_image_request(
    data,
    models,
    values,
    paths=None,
    *,
    launch_args=None,
    native_gpu_compatible=None,
    resolution_settings=None,
):
    """Submission hook, before worker transformations; no acquisition/device work."""
    if data.get("task_type") in ("llama", "chat", "rembg", "upscale") or not data.get(
        "base_model_name"
    ):
        return dict(data)
    prompt = data.get("prompt", "")
    if (
        isinstance(prompt, str)
        and "_api_job" not in data
        and not data.get("_metadata_prompt")
        and data.get("task_type") != "tool_call"
        and (
            prompt.lower() == "ruinedfooocuslogo"
            or prompt.startswith("#!")
            or prompt.lower().startswith("search:")
        )
    ):
        return dict(data)
    forced = cpu_requested(values, launch_args)
    engine = _engine(data, values, forced)
    controls = _controls(data)
    if controls.get("type", "").lower() in ("upscale", "rembg", "detect", "segment", "faceswap"):
        return dict(data)
    path = models.get_file("checkpoints", data["base_model_name"])
    if path is None:
        raise ValueError("The selected checkpoint is no longer installed")
    metadata = models.get_models_by_path("checkpoints", path, fetch=False)
    family = models.get_model_base(metadata)
    from ruinedfooocus.features.models.image_scope import require_image_family

    require_image_family(family)
    # Keep API job/event objects by identity. User-controlled data, uploads and
    # nested settings are copied, preventing later UI edits from changing a job.
    result = {key: value if key.startswith("_") else deepcopy(value) for key, value in data.items()}
    signature = tuple(file_signature(path))
    checksum = (
        metadata.get("_rf_local_sha256", "")
        if metadata.get("_rf_file_signature") == list(signature)
        else ""
    )
    if not checksum:
        checksum = activation.cached_model_hash(path)
    captured = deepcopy({key: value for key, value in result.items() if not key.startswith("_")})
    if resolution_settings is not None:
        selection = captured.get("aspect_ratios_selection")
        if selection == resolution_settings.CUSTOM_RESOLUTION:
            width, height = captured.get("custom_width"), captured.get("custom_height")
        else:
            width, height = resolution_settings.aspect_ratios.get(selection, (None, None))
        if width is not None and height is not None:
            captured.setdefault("width", width)
            captured.setdefault("height", height)
    snapshot = ImageSnapshot(
        engine,
        result["base_model_name"],
        Path(path).resolve(),
        signature,
        family,
        checksum.lower(),
        str(values.get("sdcpp_vulkan_device", "Vulkan0")),
        MappingProxyType(captured),
        MappingProxyType(controls),
        initialized_native_gpu() if native_gpu_compatible is None else bool(native_gpu_compatible),
        forced,
    )
    result[SNAPSHOT_KEY] = snapshot
    # Do not accept an API-supplied prepared object or trusted execution hook.
    result.pop(PREPARED_KEY, None)
    result.pop(HOOKS_KEY, None)
    return result


def needs_preparation(data):
    snapshot = data.get(SNAPSHOT_KEY)
    return (
        isinstance(snapshot, ImageSnapshot)
        and snapshot.engine != "native"
        and not (snapshot.engine == "auto" and snapshot.native_gpu)
    )


def retarget_image_request(data):
    """Bind a Chat tool's final reference without rereading engine/device policy."""
    snapshot = data.get(SNAPSHOT_KEY)
    if not isinstance(snapshot, ImageSnapshot):
        return dict(data)
    if data.get("base_model_name") != snapshot.name:
        raise ValueError("A prepared image request cannot change its checkpoint")
    captured = deepcopy({key: value for key, value in data.items() if not key.startswith("_")})
    # Resolution was resolved at submission, before Chat selected its reference.
    for key in ("width", "height"):
        if key in snapshot.data:
            captured.setdefault(key, snapshot.data[key])
    snapshot = replace(
        snapshot, data=MappingProxyType(captured), controls=MappingProxyType(_controls(data))
    )
    result = dict(data)
    result[SNAPSHOT_KEY] = snapshot
    prepared = data.get(PREPARED_KEY)
    if isinstance(prepared, PreparedImage):
        geometry, signature = _input_geometry(snapshot)
        result[PREPARED_KEY] = replace(
            prepared, snapshot=snapshot, input_geometry=geometry, input_signature=signature
        )
    return result


def _input_geometry(snapshot):
    from PIL import Image

    if str(snapshot.controls.get("type") or "").lower() != "img2img":
        return (), ()
    image = snapshot.data.get("input_image")
    if isinstance(image, (str, Path)):
        signature = tuple(file_signature(image))
        with Image.open(image) as source:
            geometry = (*source.size, source.mode, source.getexif().get(274, 1))
        if signature != tuple(file_signature(image)):
            raise ValueError("Input image changed during optional preparation")
        return geometry, signature
    if isinstance(image, Image.Image):
        return (*image.size, image.mode, image.getexif().get(274, 1)), ()
    shape = getattr(image, "shape", ())
    if len(shape) == 3:
        mode = (
            "RGB"
            if shape[2] == 3 and str(getattr(image, "dtype", "")) == "uint8"
            else "unsupported"
        )
        return (shape[1], shape[0], mode, 1), ()
    return (), ()


def prepare_image_request(data, *, check_cancelled, allow_unqualified=False, variant=None):
    """PreparationAdmission hook: inspect existing local files, never install/probe.

    Explicit Settings Install/Verify prepares and activates runtime dependencies.
    A missing local model hash requires its normal explicit metadata/hash refresh;
    this hook does not put a multi-GB hash inside the serial compute worker.
    """
    if not needs_preparation(data):
        return dict(data)
    snapshot = data[SNAPSHOT_KEY]
    check_cancelled()
    if tuple(file_signature(snapshot.path)) != snapshot.signature:
        raise ValueError("Checkpoint changed since submission; submit it again")
    if not snapshot.sha256 and snapshot.engine != "auto":
        raise ValueError(
            "Refresh the checkpoint's verified local identity before optional image preparation"
        )
    if snapshot.architecture != "SDXL 1.0" or snapshot.path.suffix.lower() != ".safetensors":
        model = ModelSpec(
            snapshot.path,
            snapshot.architecture,
            snapshot.path.suffix.removeprefix("."),
            (),
            False,
            snapshot.signature,
            sha256=snapshot.sha256,
        )
        result = dict(data)
        result[PREPARED_KEY] = PreparedImage(snapshot, model, ())
        return result
    model = inspect_model(
        snapshot.path,
        architecture=snapshot.architecture,
        variant=variant
        if allow_unqualified and variant is not None
        else activation.known_variant(snapshot.sha256),
    )
    if snapshot.sha256:
        model = with_model_hash(model, snapshot.sha256)
    backends = (
        ("vulkan",) if snapshot.engine == "auto" else (snapshot.engine.removeprefix("sdcpp-"),)
    )
    active = []
    # Unknown variants and other source mismatches are native-only in Auto;
    # do not force an unrelated GPU activation just to reject this checkpoint.
    if snapshot.engine == "auto" and model.variant != "base":
        result = dict(data)
        result[PREPARED_KEY] = PreparedImage(snapshot, model, ())
        return result
    for backend in backends:
        check_cancelled()
        selected_device = snapshot.device if backend == "vulkan" else "cpu"
        try:
            found = activation.current(backend, selected_device)
        except ValueError:
            if snapshot.engine != "auto":
                raise
            found = None
        if found is not None:
            active.append(found)
        elif snapshot.engine != "auto" or activation.status(backend)["prepared"]:
            raise ValueError(
                "Use Settings > Hardware to explicitly verify the prepared optional image runtime/device first"
            )
    result = dict(data)
    geometry, signature = _input_geometry(snapshot)
    result[PREPARED_KEY] = PreparedImage(snapshot, model, tuple(active), geometry, signature)
    return result


def _nonempty_loras(value):
    result = []
    for item in value or ():
        if isinstance(item, dict):
            if item.get("name") not in (None, "", "None"):
                result.append(item)
        elif len(item) >= 2 and str(item[1]).split(" - ", 1)[-1] != "None":
            result.append(item)
    return tuple(result)


def _enabled(value):
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value not in ("", "None", "Disabled", "Off")
    if isinstance(value, (tuple, list, dict)):
        return bool(len(value))
    return True


def make_request(prepared, data, *, init_image=None):
    """Map final expanded worker values while retaining captured feature intent."""
    snapshot = prepared.snapshot
    original = snapshot.data
    controls = (
        data.get("controlnet") if isinstance(data.get("controlnet"), dict) else snapshot.controls
    )
    extras = []
    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_PLAN_KEY, SizePlan

    size_plan = data.get(SIZE_PLAN_KEY)
    if isinstance(size_plan, SizePlan) and (
        size_plan.scale > 1 or size_plan.canvas != size_plan.content
    ):
        extras.append("exact target needs native canvas restoration")
    kind = str(controls.get("type") or "").lower()
    if kind not in ("", "none", "img2img"):
        extras.append("image tool " + kind)
    if kind == "img2img":
        extras.extend(
            "img2img option " + key
            for key, value in controls.items()
            if key not in ("type", "denoise", "strength") and _enabled(value)
        )
    for key in (
        "inpaint_toggle",
        "facerestore",
        "hires",
        "refiner",
        "sdxl_refiner_model",
        "reference_images",
        "automatic_upscale_model",
        "automatic_upscale",
        "ip_adapter",
        "embeddings",
        "mask",
        "vae_tiled",
        "cfg_norm_pre_cfg",
    ):
        if _enabled(original.get(key)) or _enabled(data.get(key)):
            extras.append(key)
    for key in (
        "sampling_shift",
        "cfg_rescale",
        "image_cfg",
        "slg_scale",
        "cfg_norm_strength",
        "lcm_noise",
        "lcm_noise_end",
        "lcm_noise_clip",
        "vae_tile_size",
        "vae_overlap",
    ):
        if original.get(key) not in (None, "", 0, 0.0) or data.get(key) not in (None, "", 0, 0.0):
            extras.append(key)
    for source in (original, data):
        extras.extend(str(value) for value in source.get("extra_features", ()))
    if original.get("generate_forever"):
        extras.append("generate forever")
    if original.get("style_selection") and not data.get("positive_prompt"):
        extras.append("unexpanded styles")
    for key in ("prompt", "negative"):
        raw = original.get(key, "")
        if isinstance(raw, str) and ("<lora:" in raw.lower() or "embedding:" in raw.lower()):
            extras.append("prompt adapter/embedding directive")
    replacements = tuple(
        (key, value)
        for source in (original, data)
        for key, value in (source.get("image_component_overrides") or {}).items()
        if value not in (None, "")
    )
    width = data.get("width", original.get("width", original.get("custom_width")))
    height = data.get("height", original.get("height", original.get("custom_height")))
    if width is None or height is None:
        raise UnsupportedRequest("Resolve the selected image canvas before optional-engine routing")
    if kind == "img2img" and prepared.input_geometry != (width, height, "RGB", 1):
        extras.append("img2img requires an oriented RGB input at the exact selected canvas")
    # App metadata uses positive layer choices; 1 keeps the model default.
    # The optional CLI uses -1 as its default sentinel, not as an app value.
    app_clip_skip = data.get("clip_skip", original.get("clip_skip", 1))
    valid_clip_skip = (
        type(app_clip_skip) in (int, float) and app_clip_skip >= 1 and app_clip_skip % 1 == 0
    )
    if not valid_clip_skip:
        extras.append("CLIP skip must be a positive integral app value")
    return GenerationRequest(
        prepared.model,
        data.get("positive_prompt", data.get("prompt", "")),
        data.get("negative_prompt", data.get("negative", "")),
        width,
        height,
        data.get("seed", 0) if isinstance(data.get("seed", 0), int) else 0,
        steps=data.get("steps", data.get("custom_steps", 20)),
        cfg=data.get("cfg", 7),
        sampler=data.get("sampler_name", ""),
        scheduler=data.get("scheduler", ""),
        operation="img2img" if kind == "img2img" else "txt2img",
        denoise=float(controls.get("denoise", controls.get("strength", 1)))
        if kind == "img2img"
        else 1.0,
        init_image=init_image if kind == "img2img" else None,
        request_id=str(data.get("task_id", "")),
        clip_skip=-1 if valid_clip_skip and app_clip_skip == 1 else app_clip_skip,
        loras=_nonempty_loras(original.get("loras")),
        component_overrides=replacements,
        extra_features=tuple(extras),
    )


def _route_request(prepared, data):
    # A marker preserves required img2img intent during pure routing. Process
    # creates the real owned RGB file without changing canvas or proportions.
    return replace(make_request(prepared, data, init_image=Path("prepared-init.png")), seed=0)


def select_pipeline(data, current, native_factory, *, values=None):
    """Image registry hook; no downloads, hashing, loading or enumeration."""
    values = values or {}
    snapshot = data.get(SNAPSHOT_KEY)
    prepared = data.get(PREPARED_KEY)
    engine = (
        snapshot.engine
        if isinstance(snapshot, ImageSnapshot)
        else _engine(data, values, cpu_requested(values))
    )

    def native(reason=""):
        if reason:
            data["image_engine_route"] = reason
            warn_cpu_fallback("image generation", reason.removeprefix("Native CPU: "))
        if (
            current is None
            or "sdxl" not in current.pipeline_type
            or "sdcpp" in current.pipeline_type
        ):
            return native_factory()
        return current

    if engine == "native" or (
        engine == "auto"
        and (
            snapshot.native_gpu if isinstance(snapshot, ImageSnapshot) else initialized_native_gpu()
        )
    ):
        return native()
    if not isinstance(prepared, PreparedImage):
        if engine == "auto":
            # An installed Vulkan bundle is pending work, not proof that CPU is
            # the best route. Avoid a hidden compute-time preparation/probe.
            if activation.status("vulkan")["prepared"]:
                return native("Native CPU: the optional Vulkan runtime still needs verification")
            return native("Native CPU: no prepared qualified optional GPU runtime")
        raise ValueError(
            "Optional image submission/preparation hook is unavailable; submit through the prepared image queue"
        )
    request = _route_request(prepared, data)
    source_reasons = assess(request)
    if source_reasons and engine == "auto":
        return native(
            "Native CPU: optional GPU cannot preserve this request: " + "; ".join(source_reasons)
        )
    candidates = tuple(
        activation.qualified(item, prepared.model, request.operation)
        for item in prepared.activations
    )
    if not candidates and engine == "auto":
        return native("Native CPU: no activated compatible optional GPU runtime")
    try:
        selected = choose_runtime(
            request,
            candidates,
            preference="auto" if engine == "auto" else engine.removeprefix("sdcpp-"),
        )
    except UnsupportedRequest as error:
        if engine == "auto":
            return native(f"Native CPU: {error}")
        raise
    if engine == "auto" and selected.backend == "cpu":
        warn_cpu_fallback(
            "image generation (stable-diffusion.cpp)",
            "no qualified Vulkan runtime for this model and operation",
        )
    return pipeline(prepared, selected)


class pipeline:
    pipeline_type = ["sdxl", "sdcpp"]

    def __init__(self, prepared, candidate):
        self.prepared, self.candidate = prepared, candidate

    def load_base_model(self, name, hash=None, component_overrides=None):
        if name != self.prepared.snapshot.name or (
            hash and hash.lower() != self.prepared.model.sha256
        ):
            raise ValueError("Queued optional checkpoint identity changed")
        if tuple(file_signature(self.prepared.model.path)) != self.prepared.model.signature:
            raise ValueError("Queued optional checkpoint file changed")
        if any(value not in (None, "") for value in (component_overrides or {}).values()):
            raise UnsupportedRequest("Optional image execution cannot replace bundled components")

    def load_loras(self, loras):
        if _nonempty_loras(loras):
            raise UnsupportedRequest("LoRA conditioning requires native image routing")

    def process(self, gen_data=None, callback=None):
        import tempfile
        import numpy as np
        from PIL import Image

        hooks = gen_data.get(HOOKS_KEY)
        if not isinstance(hooks, ExecutionHooks):
            raise RuntimeError(
                "Optional image compute admission is pending 06 integration; no native process was started"
            )
        hooks.check_cancelled()
        with tempfile.TemporaryDirectory(prefix="rf-sdcpp-caller-") as directory:
            initial = None
            if str(self.prepared.snapshot.controls.get("type") or "").lower() == "img2img":
                image = self.prepared.snapshot.data.get("input_image")
                if (
                    self.prepared.input_signature
                    and tuple(file_signature(image)) != self.prepared.input_signature
                ):
                    raise ValueError("Prepared input image changed since submission")
                if isinstance(image, (str, Path)):
                    with Image.open(image) as loaded:
                        image = loaded.copy()
                elif isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                if not isinstance(image, Image.Image):
                    raise UnsupportedRequest("Add a prepared RGB image in Image tools")
                initial = Path(directory) / "init.png"
                image.save(initial)
            request = make_request(self.prepared, gen_data, init_image=initial)
            reasons = assess(request, self.candidate)
            if reasons:
                raise UnsupportedRequest("; ".join(reasons))
            result = generate(
                request,
                self.candidate,
                compute_lease=hooks.compute_lease_factory(request, self.candidate),
                check_cancelled=hooks.check_cancelled,
            )
            gen_data["_sdcpp_metadata"] = result.metadata
            gen_data["sampling_denoise"] = request.denoise
            gen_data["image_engine"] = "sdcpp-" + self.candidate.backend
            from ruinedfooocus.features.generation.size_plan import KEY as SIZE_PLAN_KEY, SizePlan

            size_plan = gen_data.get(SIZE_PLAN_KEY)
            if isinstance(size_plan, SizePlan):
                if result.image.size != size_plan.target:
                    raise UnsupportedRequest(
                        "Optional engine output disagrees with the exact target."
                    )
                gen_data["size_plan"] = size_plan.metadata()
            return [np.asarray(result.image).copy()]
