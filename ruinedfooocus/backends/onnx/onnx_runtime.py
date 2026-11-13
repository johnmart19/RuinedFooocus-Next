"""Optional isolated GPU mask inference with a native CPU fallback."""

import json
import sysconfig
import logging
import math
from pathlib import Path
import platform
import queue
import subprocess
import tempfile
import threading
import time

from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary
from ruinedfooocus.features.image_processing.auxiliary_resources import register, using
from ruinedfooocus.backends.comfy.comfy_compat import child_device_environment
from ruinedfooocus.core.child_process import start_process
from ruinedfooocus.backends.onnx.model_contracts import (
    create_mask_session,
    create_audited_mask_session,
)
from ruinedfooocus.backends.onnx.execution import (
    finish_assignment,
    assignment_status,
    discard_profile,
)
from ruinedfooocus.backends.onnx.onnx_installer import (
    ROOT,
    WORKER,
    NATIVE_RUNTIMES,
    native_runtime_available,
    native_runtime_auto_eligible,
    native_runtime_path,
    runtime_available,
    runtime_path,
    worker_command,
)


class GPUWorkerError(RuntimeError):
    pass


_last_status = "No background-removal GPU session has been loaded."


def runtime_status():
    return _last_status


def report_status(message):
    global _last_status
    _last_status = message
    logging.info("Background removal: %s", message)


def session_request_identity(cpu=False, *, values=None):
    """Invalidate live sessions when effective requests or installed records change."""
    if values is None:
        from ruinedfooocus.app.state import settings

        values = settings.default_settings
    from ruinedfooocus.backends.onnx.windows_ml import manifest_path

    records = [manifest_path(), runtime_path() / "verified.json"]
    from ruinedfooocus.backends.onnx.onnx_installer import native_runtime, LEGACY_MIGRAPHX

    records.extend(
        native_runtime_path(name, native_runtime(name)) / "verified.json"
        for name in NATIVE_RUNTIMES
    )
    records.append(native_runtime_path("MIGraphX", LEGACY_MIGRAPHX) / "verified.json")
    identities = []
    for path in records:
        try:
            stat = path.stat()
            identities.append((str(path), stat.st_size, stat.st_mtime_ns))
        except OSError:
            identities.append((str(path), None, None))
    keys = (
        "cpu_everywhere",
        "rembg_backend",
        "rembg_provider",
        "rembg_winml_device",
        "rembg_winml_provider",
        "rembg_winml_type",
    )
    return (cpu, tuple(values.get(key) for key in keys), tuple(identities))


def release_torch_weights():
    from ruinedfooocus.app.resource_lifecycle import release_native_for_external

    release_native_for_external()


class NativeMaskSession:
    """Give installed native CUDA/ROCm sessions the same resource lifetime."""

    resource_group = "rembg"

    def __init__(self, factory, expected_provider=None):
        self.factory = factory
        self.expected_provider = expected_provider
        self.runtime_status = "Not loaded"
        self.session = None
        self.disabled = False
        self.check_cancelled = None
        register(self)

    @using
    def predict(self, image, *args, **kwargs):
        if not self.disabled:
            from ruinedfooocus.app.resource_lifecycle import require_compute_owner

            require_compute_owner()
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            EPFail,
            EngineError,
            Fail,
            RuntimeException,
        )

        if self.check_cancelled:
            self.check_cancelled()
        phase = "initialization" if self.session is None else "inference"
        try:
            if self.session is None:
                if not self.disabled:
                    release_torch_weights()
                self.session = self.factory() if not self.disabled else self._cpu_session()
                self.session.inner_session.disable_fallback()
                if (
                    not self.disabled
                    and self.expected_provider
                    and self.expected_provider not in self.session.inner_session.get_providers()
                ):
                    raise EPFail(
                        "Requested "
                        + self.expected_provider
                        + " was not assigned; refusing silent CPU inference."
                    )
                if not self.disabled:
                    self.runtime_status = (
                        "Initialized: "
                        + (self.expected_provider or "native provider")
                        + "; assignment unverified"
                    )
                    report_status(self.runtime_status)
            phase = "inference"
            result = self.session.predict(image, *args, **kwargs)
            if not self.disabled:
                evidence = finish_assignment(self.session, self.expected_provider)
                if evidence:
                    self.runtime_status = assignment_status(evidence)
                    report_status(self.runtime_status)
        except (EPFail, EngineError, Fail, RuntimeException, RuntimeError) as error:
            if self.disabled:
                raise
            self.close()
            self.disabled = True
            logging.warning("Native GPU background removal failed; retrying on CPU: %s", error)
            self.runtime_status = f"CPU recovery: {phase} failed: {error}"
            report_status(self.runtime_status)
            if self.check_cancelled:
                self.check_cancelled()
            self.session = self._cpu_session()
            result = self.session.predict(image, *args, **kwargs)
        if self.check_cancelled:
            self.check_cancelled()
        return result

    @staticmethod
    def _cpu_session():
        return create_mask_session(providers=["CPUExecutionProvider"])

    def close(self):
        if self.session is not None:
            discard_profile(self.session)
        self.session = None


class IsolatedMaskSession:
    resource_group = "rembg"

    def __init__(
        self,
        device_id=None,
        *,
        provider="DirectML",
        ep_device="Auto",
        ep_name="Auto",
        ep_type="Auto",
    ):
        self.device_id = device_id
        self.provider = provider
        self.ep_device = ep_device
        self.ep_name = ep_name
        self.ep_type = ep_type
        self.runtime_status = "Not loaded"
        self.process = None
        self.job = None
        self.reader = None
        self.log = None
        self.scratch = None
        self.messages = queue.Queue()
        self.check_cancelled = None
        self.disabled = False
        self.cpu_session = None
        self.last_execution = None
        self.initialization_progress = None
        register(self)

    def _read_messages(self, process, messages):
        try:
            while line := process.stdout.readline(65537):
                if len(line) > 65536:
                    messages.put(
                        {
                            "error": "GPU mask worker response exceeded 65536 characters.",
                            "phase": "protocol",
                        }
                    )
                    break
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    # Progress can never grow the pending queue without bound.
                    # Leave ready/result/error delivery on the existing protocol.
                    if value.get("event") == "initialization_progress" and messages.qsize() >= 32:
                        continue
                    messages.put(value)
        except (OSError, ValueError):
            pass
        finally:
            messages.put({"error": "GPU mask worker exited."})

    def _wait(self, timeout, *, phase="response"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.check_cancelled:
                try:
                    self.check_cancelled()
                except BaseException:
                    self.close()
                    raise
            try:
                message = self.messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.get("error"):
                failed_phase = message.get("phase")
                raise GPUWorkerError(
                    (failed_phase + " failed: " if failed_phase else "") + message["error"]
                )
            if message.get("event") == "initialization_progress":
                detail, elapsed = message.get("phase"), message.get("elapsed_seconds")
                if (
                    phase != "initialization"
                    or not isinstance(detail, str)
                    or not 0 < len(detail) <= 160
                    or isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or elapsed > 86400
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise GPUWorkerError("Invalid initialization progress from GPU mask worker.")
                self.initialization_progress = {"phase": detail, "elapsed_seconds": elapsed}
                self.runtime_status = (
                    f"Initializing {self.provider}: {detail} (phase started at {elapsed:.1f}s)."
                )
                report_status(self.runtime_status)
                continue
            return message
        detail = ""
        if phase == "initialization" and self.initialization_progress:
            progress = self.initialization_progress
            detail = (
                f" Last phase: {progress['phase']} (started at {progress['elapsed_seconds']:.1f}s)."
            )
        raise GPUWorkerError(f"GPU mask worker {phase} timed out after {timeout:g}s.{detail}")

    def _start(self):
        if self.ep_type != "CPU":
            release_torch_weights()
        log_path = ROOT / "cache/onnx/worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="worker-", dir=log_path.parent)
        self.messages = queue.Queue()
        self.initialization_progress = None
        if self.provider == "Windows ML":
            from ruinedfooocus.backends.onnx.windows_ml import manifest_path

            command = worker_command(sysconfig.get_path("purelib"), worker=WORKER)
            command += [
                "--provider",
                self.provider,
                "--manifest",
                str(manifest_path()),
                "--ep-device",
                self.ep_device,
            ]
            command += ["--ep-name", self.ep_name, "--ep-type", self.ep_type]
        elif self.provider in NATIVE_RUNTIMES:
            command = worker_command(
                native_runtime_path(self.provider), device_id=self.device_id, worker=WORKER
            )
            command += ["--provider", self.provider, "--ep-type", self.ep_type]
        else:
            command = worker_command(runtime_path(), device_id=self.device_id, worker=WORKER)
        command += ["--profile-root", self.scratch.name]
        try:
            self.log = log_path.open("a", encoding="utf-8")
            self.process, self.job = start_process(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                encoding="utf-8",
                env=child_device_environment(service="onnx-runtime"),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            raise GPUWorkerError(f"Cannot start the GPU mask worker: {error}") from error
        self.reader = threading.Thread(
            target=self._read_messages, args=(self.process, self.messages), daemon=True
        )
        self.reader.start()
        ready = self._wait(180, phase="initialization")
        if not ready.get("ready"):
            raise GPUWorkerError("GPU mask worker did not initialize.")
        self.runtime_status = (
            "Initialized "
            + ready.get("device_type", "Accelerator")
            + ": "
            + ready.get("provider", self.provider)
            + "; assignment unverified"
        )
        self.ready_info = ready
        report_status(self.runtime_status)
        logging.info("Background removal: isolated %s worker ready.", self.runtime_status)

    def _cpu_predict(self, image, *args, **kwargs):
        if self.check_cancelled:
            self.check_cancelled()
        if self.cpu_session is None:
            self.cpu_session = create_mask_session(providers=["CPUExecutionProvider"])
        result = self.cpu_session.predict(image, *args, **kwargs)
        if self.check_cancelled:
            self.check_cancelled()
        return result

    @using
    def predict(self, image, *args, **kwargs):
        # Preserve uncommon model-specific Python options without narrowing the
        # native session contract or silently discarding unsupported arguments.
        self.last_execution = None
        if not self.disabled and not args and not kwargs and self.ep_type != "CPU":
            from ruinedfooocus.app.resource_lifecycle import require_compute_owner

            require_compute_owner()
        if self.disabled or args or kwargs:
            self.close()
            if not self.disabled:
                self.runtime_status = (
                    "CPU recovery: model-specific options require the native CPU session."
                )
                report_status(self.runtime_status)
            return self._cpu_predict(image, *args, **kwargs)
        try:
            if self.process is None:
                self._start()
            jobs = ROOT / "cache/onnx/jobs"
            jobs.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="mask-", dir=jobs) as folder:
                source = Path(folder) / "input.png"
                image.save(source)
                try:
                    self.process.stdin.write(
                        json.dumps({"input": str(source), "output": folder}) + "\n"
                    )
                    self.process.stdin.flush()
                except (OSError, ValueError) as error:
                    raise GPUWorkerError("GPU mask worker connection failed.") from error
                response = self._wait(120, phase="inference")
                self.last_execution = {
                    "assignment": response.get("assignment"),
                    "inference_seconds": response.get("inference_seconds"),
                }
                if response.get("assignment"):
                    self.runtime_status = assignment_status(response["assignment"])
                    report_status(self.runtime_status)
                from PIL import Image

                masks = []
                for filename in response.get("masks", []):
                    path = Path(filename).resolve()
                    if path.parent != Path(folder).resolve():
                        raise GPUWorkerError("GPU mask worker returned an invalid output path.")
                    with Image.open(path) as mask:
                        masks.append(mask.copy())
                if not masks:
                    raise GPUWorkerError("GPU mask worker returned no mask.")
                return masks
        except GPUWorkerError as error:
            self.last_execution = None
            self.close()
            self.disabled = True
            self.runtime_status = "CPU recovery: " + str(error)
            report_status(self.runtime_status)
            logging.warning("GPU background removal unavailable; using CPU: %s", error)
            return self._cpu_predict(image)

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        # A dead worker can leave buffered input that cannot
                        # be flushed. Cleanup must not prevent CPU recovery.
                        pass
        if self.reader is not None:
            self.reader.join(timeout=1)
            self.reader = None
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def new_session(
    cpu=False, preference=None, provider=None, ep_device=None, ep_name=None, ep_type=None
):
    if cpu or preference == "CPU":
        report_status("CPU: native ONNX Runtime (explicit request).")
        return create_mask_session(providers=["CPUExecutionProvider"])
    if preference is None:
        from ruinedfooocus.app.state import settings
        from ruinedfooocus.backends.onnx.windows_ml import provider_preference

        from ruinedfooocus.packaging.backend_preferences import effective_backends

        preference = effective_backends(settings.default_settings)["rembg"]
        provider = provider or provider_preference(settings.default_settings)
        ep_device = ep_device or settings.default_settings.get("rembg_winml_device", "Auto")
        ep_name = ep_name or settings.default_settings.get("rembg_winml_provider", "Auto")
        ep_type = ep_type or settings.default_settings.get("rembg_winml_type", "Auto")
    provider = provider or "Auto"
    if preference == "CPU":
        report_status("CPU: native ONNX Runtime (effective CPU policy).")
        return create_mask_session(providers=["CPUExecutionProvider"])
    if preference not in ("Auto", "Windows GPU", *NATIVE_RUNTIMES):
        raise ValueError("Choose a listed background-removal runtime.")
    from ruinedfooocus.app.arguments import args

    if force_cpu_for_auxiliary(args):
        report_status("CPU: native ONNX Runtime (global CPU scope).")
        return create_mask_session(providers=["CPUExecutionProvider"])
    import onnxruntime as ort

    native_gpu = set(ort.get_available_providers()) & {
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
    }
    if preference in NATIVE_RUNTIMES:
        if native_runtime_available(preference):
            return IsolatedMaskSession(provider=preference)
        logging.warning("%s ONNX runtime is not installed/verified; using CPU.", preference)
        report_status(
            f"CPU recovery: the {preference} ONNX runtime is not installed. "
            "Install it in Settings > Hardware & backends."
        )
        return create_mask_session(providers=["CPUExecutionProvider"])
    from ruinedfooocus.backends.onnx.windows_ml import PROVIDERS, installed_provider

    if provider not in PROVIDERS:
        raise ValueError("Choose Auto, Windows ML or DirectML for the background provider.")
    if provider == "Windows ML" and args.directml is None:
        if ep_type == "CPU" and ep_name in (None, "Auto"):
            report_status("CPU: native ONNX Runtime (built-in provider; no vendor package needed).")
            return create_mask_session(providers=["CPUExecutionProvider"])
        if installed_provider(ep_name or "Auto") is not None:
            return IsolatedMaskSession(
                provider="Windows ML",
                ep_device=ep_device or "Auto",
                ep_name=ep_name or "Auto",
                ep_type=ep_type or "Auto",
            )
        logging.warning("Windows ML provider not installed/verified; using CPU background removal.")
        report_status(
            "CPU recovery: Windows ML vendor Auto has no matching real-model qualification; choose a named vendor for bounded qualification."
            if ep_name in (None, "Auto")
            else "CPU recovery: Windows ML provider is not installed or its recorded library changed."
        )
        return create_mask_session(providers=["CPUExecutionProvider"])

    if preference == "Auto" and provider == "Auto" and native_gpu and args.directml is None:
        selected = next(
            name
            for name in (
                "CUDAExecutionProvider",
                "MIGraphXExecutionProvider",
                "ROCMExecutionProvider",
            )
            if name in native_gpu
        )
        factory = lambda: create_audited_mask_session(
            selected, providers=[selected, "CPUExecutionProvider"]
        )
        return NativeMaskSession(factory, expected_provider=selected)
    if preference == "Auto" and provider == "Auto" and args.directml is None:
        # A vendor build the user installed here was verified on this machine.
        for name in NATIVE_RUNTIMES:
            if native_runtime_auto_eligible(name):
                return IsolatedMaskSession(provider=name)
    if (
        platform.system() == "Windows"
        and (provider == "DirectML" or args.directml is not None)
        and runtime_available()
    ):
        device_id = args.directml if args.directml is not None and args.directml >= 0 else None
        return IsolatedMaskSession(device_id=device_id)
    if preference == "Windows GPU" or provider == "DirectML" or args.directml is not None:
        logging.warning(
            "Optional Windows GPU ONNX runtime is not installed; using native/CPU removal."
        )
        report_status("CPU recovery: the requested Windows GPU runtime is not installed/verified.")
    else:
        from ruinedfooocus.core.device_fallback import warn_cpu_fallback

        warn_cpu_fallback(
            "background removal", "no qualified automatic ONNX GPU provider is available"
        )
        report_status("CPU: native ONNX Runtime (no automatic accelerator selected).")
    return create_mask_session(providers=["CPUExecutionProvider"])
