"""Synchronous owned CLI requests; native code owns all sampling and device work."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from ruinedfooocus.core.child_process import start_process
from .contracts import SOURCE_REVISION, UnsupportedRequest, _Cancellation, assess

_EXECUTION = threading.Lock()
_DEVICE_ENVIRONMENT = (
    "GGML_VK_VISIBLE_DEVICES",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "VK_ADD_DRIVER_FILES",
    "VK_LAYER_PATH",
    "VK_INSTANCE_LAYERS",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
)


class RuntimeFailure(RuntimeError):
    """A native failure; callers must not silently retry with another device."""


@dataclass(frozen=True)
class GenerationResult:
    image: object
    metadata: dict
    diagnostic: str


def _environment(candidate):
    from ruinedfooocus.backends.comfy.comfy_compat import child_device_environment

    environment = child_device_environment(service="sdcpp")
    # Dynamic GGML modules belong to the verified immutable bundle. Do not let
    # another application's inherited override add an unverified backend DLL.
    environment["GGML_BACKEND_PATH"] = str(Path(candidate.executable).resolve().parent)
    return environment


def _device_environment(environment):
    return tuple((key, environment.get(key)) for key in _DEVICE_ENVIRONMENT)


def _unchanged(candidate, model=None):
    from ruinedfooocus.features.models.model_identity import files_unchanged

    signatures = candidate.payload_signatures + ((model.signature,) if model is not None else ())
    if not files_unchanged([list(signature) for signature in signatures]):
        raise RuntimeFailure(
            "Prepared runtime/checkpoint changed; explicitly prepare and verify it again"
        )


def _tail(stream, limit=65536):
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - limit))
    return stream.read().decode("utf-8", errors="replace")


def _run(command, *, cwd, environment, check_cancelled, timeout):
    """Retain cancellation even with chatty native output; reap only our child."""
    check_cancelled()
    started = time.monotonic()
    with tempfile.TemporaryFile() as output:
        process, job = start_process(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            while process.poll() is None:
                check_cancelled()
                if time.monotonic() - started >= timeout:
                    raise RuntimeFailure(f"Optional diffusion request exceeded {timeout} seconds")
                time.sleep(0.1)
            check_cancelled()
            diagnostic = _tail(output)
            if process.returncode:
                raise RuntimeFailure(
                    f"Optional diffusion process failed ({process.returncode}); no CPU retry. "
                    + diagnostic
                )
            return diagnostic
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                if job:
                    job.close()


def verify_device(candidate, *, check_cancelled, timeout=30):
    """Explicit active verification only. Enumeration does not qualify a workload.

    CPU uses the separate CPU artifact and --help, never --list-devices. Vulkan
    enumerates the bundle's native namespace (tab-separated name/description).
    Preserve the same inherited Vulkan visibility/driver selection for execution.
    """
    from .artifacts import ARTIFACTS

    matches = [a for a in ARTIFACTS if a.sha256 == candidate.artifact_sha256]
    if len(matches) != 1 or matches[0].backend != candidate.backend:
        raise RuntimeFailure("Selected backend differs from the frozen runtime artifact")
    _unchanged(candidate)
    if candidate.revision != SOURCE_REVISION:
        raise RuntimeFailure("Unexpected optional diffusion source revision")
    environment = _environment(candidate)
    with tempfile.TemporaryDirectory(prefix="rf-sdcpp-device-") as temporary:
        diagnostic = _run(
            [
                str(candidate.executable),
                "--help" if candidate.backend == "cpu" else "--list-devices",
            ],
            cwd=temporary,
            environment=environment,
            check_cancelled=check_cancelled,
            timeout=timeout,
        )
    if candidate.backend == "cpu":
        found = candidate.device == "cpu" and "Usage:" in diagnostic
    else:
        devices = dict(line.split("\t", 1) for line in diagnostic.splitlines() if "\t" in line)
        found = candidate.device in devices
    if not found:
        raise RuntimeFailure("The requested exact runtime device was not verified; no CPU retry")
    return replace(
        candidate,
        device_verified=True,
        device_environment=_device_environment(environment),
        device_description="CPU-only runtime"
        if candidate.backend == "cpu"
        else devices[candidate.device],
        qualified=False,
        qualification="",
        accepted_models=(),
        accepted_operations=(),
    )


def build_command(request, candidate, output, *, init_image=None):
    """Map supported values without shell parsing, prompt rewriting or resizing."""
    command = [
        str(Path(candidate.executable).resolve()),
        "--mode",
        "img_gen",
        "--model",
        str(Path(request.model.path).resolve()),
        "--prompt",
        request.prompt,
        "--negative-prompt",
        request.negative_prompt,
        "--width",
        str(request.width),
        "--height",
        str(request.height),
        "--seed",
        str(request.seed),
        "--steps",
        str(request.steps),
        "--cfg-scale",
        str(request.cfg),
        "--strength",
        str(request.denoise),
        "--sampling-method",
        "euler",
        "--scheduler",
        "discrete",
        "--rng",
        "cpu",
        "--sampler-rng",
        "cpu",
        "--clip-skip",
        "-1",
        "--batch-count",
        "1",
        "--backend",
        candidate.device,
        "--params-backend",
        candidate.device,
        "--auto-fit",
        "off",
        "--flash-attn",
        "off",
        "--conv-direct",
        "off",
        "--log-level",
        "warn",
        "--output",
        str(output),
    ]
    if init_image is not None:
        command += [
            "--init-img",
            str(init_image),
            "--image-preprocess",
            "target=init,mode=none",
        ]
    return command


def generate(
    request,
    candidate,
    *,
    compute_lease,
    check_cancelled,
    timeout=900,
    allow_unqualified=False,
    service=None,
):
    """Execute one admitted request inside the caller's shared compute ownership.

    No model download, server, Torch mutation, native-model unload, auto fallback
    or global visibility changes. The caller releases conflicting image/Chat/ONNX
    resources through its compute_lease before entry. Returned pixels/metadata own
    their memory; all private native files are removed on every exit path.
    """
    from PIL import Image
    from ruinedfooocus.core.preparation import preparation_service
    from ruinedfooocus.features.models.model_identity import file_signature

    reasons = assess(request, candidate, allow_unqualified=allow_unqualified)
    if reasons:
        raise UnsupportedRequest("; ".join(reasons))
    if not isinstance(timeout, (float, int)) or not 0 < timeout < float("inf"):
        raise ValueError("A finite positive process timeout is required")
    environment = _environment(candidate)
    if _device_environment(environment) != candidate.device_environment:
        raise RuntimeFailure("Vulkan device environment changed; verify the exact device again")
    service = service or preparation_service
    with ExitStack() as stack:
        # File leases are acquired before compute admission, matching the common
        # preparation contract. Waiting for downloads while admitted is forbidden.
        for signature in (request.model.signature, *candidate.payload_signatures):
            stack.enter_context(
                service.artifact_lease(signature[0], cancelled=_Cancellation(check_cancelled))
            )
        if request.init_image is not None:
            stack.enter_context(
                service.artifact_lease(request.init_image, cancelled=_Cancellation(check_cancelled))
            )
        check_cancelled()
        _unchanged(candidate, request.model)
        while not _EXECUTION.acquire(timeout=0.1):
            check_cancelled()
        stack.callback(_EXECUTION.release)
        stack.enter_context(compute_lease)
        check_cancelled()
        with tempfile.TemporaryDirectory(prefix="rf-sdcpp-request-") as temporary:
            directory = Path(temporary)
            initial = None
            initial_signature = None
            if request.init_image is not None:
                initial_signature = tuple(file_signature(request.init_image))
                with Image.open(request.init_image) as image:
                    if image.size != (request.width, request.height) or image.mode != "RGB":
                        raise UnsupportedRequest(
                            "Prepare an RGB init image at the exact requested dimensions"
                        )
                    if image.getexif().get(274, 1) != 1:
                        raise UnsupportedRequest(
                            "Apply init-image orientation during explicit preparation"
                        )
                    initial = directory / "init.png"
                    image.save(initial, format="PNG")
                if initial_signature != tuple(file_signature(request.init_image)):
                    raise RuntimeFailure("Init image changed during preparation")
            output = directory / "result.png"
            diagnostic = _run(
                build_command(request, candidate, output, init_image=initial),
                cwd=directory,
                environment=environment,
                check_cancelled=check_cancelled,
                timeout=timeout,
            )
            check_cancelled()
            _unchanged(candidate, request.model)
            if not output.is_file():
                raise RuntimeFailure("Native process returned no image")
            with Image.open(output) as image:
                if image.size != (request.width, request.height):
                    raise RuntimeFailure("Native output dimensions differ from the request")
                image.load()
                result = image.copy()
                native_metadata = dict(image.info)
            metadata = {
                "engine": "stable-diffusion.cpp",
                "engine_revision": SOURCE_REVISION,
                "artifact_sha256": candidate.artifact_sha256,
                "backend": candidate.backend,
                "device": candidate.device,
                "device_description": candidate.device_description,
                "qualification": candidate.qualification,
                "acceptance_run": allow_unqualified,
                "request_id": request.request_id,
                "model": str(request.model.path),
                "model_sha256": request.model.sha256,
                "model_signature": request.model.signature,
                "architecture": request.model.architecture,
                "model_variant": request.model.variant,
                "model_format": request.model.format,
                "quantization": request.model.quantization,
                "weight_types": request.model.weight_types,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "seed": request.seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler": request.sampler,
                "scheduler": request.scheduler,
                "native_scheduler": "discrete",
                "operation": request.operation,
                "denoise": request.denoise,
                "init_image": str(request.init_image) if request.init_image is not None else None,
                "init_image_signature": initial_signature,
                "rng": "cpu",
                "clip_skip": -1,
                "native_metadata": native_metadata,
            }
            return GenerationResult(result, metadata, diagnostic)
