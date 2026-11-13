"""Passive optional-image presentation and explicit owned runtime actions."""

from concurrent.futures import CancelledError, Future
from copy import deepcopy
import re
import threading

from ruinedfooocus.backends.sdcpp import activation
from ruinedfooocus.core.translation_manager import localize
from ruinedfooocus.features.generation.sdcpp_pipeline import cpu_requested

ENGINE_CHOICES = (
    ("Automatic: native GPU, qualified optional Vulkan, then CPU", "auto"),
    ("Native ComfyUI", "native"),
    ("Optional stable-diffusion.cpp Vulkan", "sdcpp-vulkan"),
    ("Optional stable-diffusion.cpp CPU", "sdcpp-cpu"),
)


def requested_values(values, cpu=False, image_backend=None):
    result = deepcopy(values)
    if cpu:
        result["cpu_everywhere"] = True
    if image_backend is not None:
        result["image_backend"] = image_backend
    return result


def backend_request(backend, values, *, device="Vulkan0", launch_args=None):
    if backend not in ("cpu", "vulkan"):
        raise ValueError("Choose CPU or Vulkan runtime package.")
    if backend == "vulkan":
        if cpu_requested(values, launch_args):
            raise ValueError(
                "Explicit CPU authority prevents Vulkan verification. Choose the CPU runtime."
            )
        if not isinstance(device, str) or not re.fullmatch(r"Vulkan[0-9]+", device):
            raise ValueError(
                "Use the exact optional backend name, such as Vulkan0. It is not a physical GPU identity."
            )
    return "cpu" if backend == "cpu" else device


def status_text(engine="auto", device="Vulkan0", *, values=None, launch_args=None, locale=None):
    """Artifact/cache/session facts only; no current(), hashing or device query."""
    values = values or {}
    lines = [localize("Requested image engine: {engine}.", locale, engine=engine)]
    if cpu_requested(values, launch_args):
        lines.append(
            localize(
                "Effective CPU authority is active; optional Vulkan cannot run. Your saved preference is retained.",
                locale,
            )
        )
    for backend in ("vulkan", "cpu"):
        state = activation.status(backend)
        if not state["available"]:
            lines.append(f"{backend}: {state['reason']}")
            continue
        cache = (
            localize("Cached manifest present; verify before use", locale)
            if state["prepared"]
            else localize("Not prepared", locale)
        )
        devices = ", ".join(state.get("devices", ())) or localize("none in this process", locale)
        lines.append(
            localize(
                "{backend}: {cache}; verified backend names: {devices}.",
                locale,
                backend=backend,
                cache=cache,
                devices=devices,
            )
        )
    lines.append(
        localize(
            "Requested Vulkan backend: {device}. Backend ordinals do not establish a physical adapter/driver identity.",
            locale,
            device=device,
        )
    )
    lines.append(
        localize(
            "Install and Verify do not qualify a model or operation. Auto requires matching acceptance evidence before optional execution.",
            locale,
        )
    )
    return "\n\n".join(lines)


class AdmittedOperation:
    def __init__(self, operation, resource_group, cancelled=None):
        self.operation = operation
        self.resource_group = resource_group
        self.cancelled = cancelled or threading.Event()

    def __call__(self, job, task_id):
        def check():
            if self.cancelled.is_set() or job.cancelled.is_set():
                raise CancelledError("This Settings action was cancelled.")

        check()
        result = self.operation(check)
        check()
        return result


class RuntimeActionSession:
    """One browser's action/cancel handle; acquisition uses the bounded I/O pool."""

    def __init__(self):
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.job = None
        self.future = None
        self.active = False
        self.progress = ""

    def __deepcopy__(self, memo):
        return type(self)()

    def begin(self):
        with self.lock:
            if self.active:
                raise ValueError(
                    "An optional runtime action is already pending. Cancel it or wait."
                )
            self.active = True
            self.cancelled = threading.Event()
            self.job = self.future = None
            self.progress = ""

    def check(self):
        if self.cancelled.is_set():
            raise CancelledError("This Settings action was cancelled.")

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            if self.job is not None:
                self.job.cancelled.set()
            if self.future is not None:
                self.future.cancel()

    def finish(self):
        with self.lock:
            self.active = False
            self.job = self.future = None

    def install(
        self, backend, device, values, *, offline, skip_install, admission=None, locale=None
    ):
        backend_request(backend, values, device=device)
        self.begin()
        if admission is None:
            from ruinedfooocus.app.preparation_admission import preparation_admission

            admission = preparation_admission
        result = Future()
        cancelled = self.cancelled

        def check():
            if cancelled.is_set():
                raise CancelledError("This Settings action was cancelled.")

        def progress(received, total, speed):
            if self.cancelled is not cancelled:
                return
            self.progress = (
                localize("Downloading: {percent:.0f}%", locale, percent=received * 100 / total)
                if total
                else localize(
                    "Received {megabytes:.1f} MB", locale, megabytes=(received or 0) / 1000000
                )
            )

        def prepare():
            check()
            return activation.install(
                backend,
                check_cancelled=check,
                offline=offline,
                skip_dependency_install=skip_install,
                progress=progress,
            )

        try:
            self.future = admission.submit(prepare, result.set_result, result.set_exception)
        except BaseException:
            self.finish()
            raise
        return result

    def verify(self, backend, device, values, *, locale=None):
        selected = backend_request(backend, values, device=device)
        verified = self.run_admitted(
            lambda check: activation.verify(backend, selected, check_cancelled=check),
            "sdcpp",
            timeout=120,
        )
        return localize(
            "Verified backend {device}: {description}. Model/operation qualification remains separate.",
            locale,
            device=verified.candidate.device,
            description=verified.candidate.device_description,
        )

    def run_admitted(self, operation, group="rembg", *, timeout=600):
        from ruinedfooocus.app.api_runtime import submit, wait_for_result

        self.begin()
        try:
            self.job = submit(AdmittedOperation(operation, group, self.cancelled))
            return wait_for_result(self.job, timeout=timeout)
        finally:
            self.finish()
