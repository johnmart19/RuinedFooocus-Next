"""Managed local llama-server with the completion API used by Chat bots."""

import json
import os
from pathlib import Path
import secrets
import re
import shlex
import socket
import subprocess
import sys
import time
import weakref
from concurrent.futures import CancelledError

import requests
from ruinedfooocus.backends.llama.llama_installer import (
    server_path,
    runtime_environment,
    canonical_backend,
    CUDA_BACKEND,
)
from ruinedfooocus.core.child_process import start_process
from ruinedfooocus.core.device_fallback import warn_cpu_fallback
from ruinedfooocus.core.paths import data_path
from ruinedfooocus.backends.llama.runtime_probe import (
    probe_runtime as _probe_runtime,
    RuntimeUnavailable,
    RuntimeVerificationError,
)
from ruinedfooocus.backends.llama.runtime_events import (
    RuntimeLoad,
    file_identity,
    native_request,
    model_header_evidence,
    runtime_identity,
    startup_evidence,
)
from ruinedfooocus.backends.llama.runtime_policy import preferred_backend
from ruinedfooocus.backends.llama.moe_cache import (
    moe_cache_cpu_share_capability,
    public_moe_cache_state,
    warn_unsupported,
)
from ruinedfooocus.features.chat.llama_vision import projector_path


def resolve_runtime(gpu, backend, *, check_cancelled=None, progress_callback=None):
    """Try cached vendor builds, Vulkan, and uncached vendor builds before CPU."""
    system = {"win32": "Windows", "linux": "Linux"}.get(sys.platform, sys.platform)
    automatic = gpu and backend == "Auto"
    preferred = preferred_backend(system, backend, gpu=gpu)
    failures = []
    uncached = []
    callbacks = {}
    if check_cancelled is not None:
        callbacks["check_cancelled"] = check_cancelled
    if progress_callback is not None:
        callbacks["progress_callback"] = progress_callback

    def attempt(choice, acquire, *, executable=None):
        if check_cancelled:
            check_cancelled()
        use_gpu = choice != "CPU"
        try:
            if executable is None:
                executable = server_path(use_gpu, backend=choice, acquire=acquire, **callbacks)
            environment = runtime_environment(use_gpu, backend=choice, acquire=acquire, **callbacks)
        except RuntimeError as error:
            if not automatic:
                raise
            failures.append(f"{choice}: {error}")
            return None
        try:
            label = (
                "CPU"
                if not use_gpu
                else _probe_runtime(
                    executable,
                    environment,
                    None
                    if os.environ.get("LLAMA_SERVER") or choice == "Auto"
                    else choice.split()[0],
                    **callbacks,
                )
            )
        except RuntimeUnavailable as error:
            if not automatic:
                raise
            failures.append(f"{choice}: {error}")
            return None
        if check_cancelled:
            check_cancelled()
        return executable, environment, use_gpu, label

    if automatic and not os.environ.get("LLAMA_SERVER"):
        from ruinedfooocus.packaging.hardware_inventory import detect_gpus, llama_installed

        vendors = {item.get("vendor") for item in detect_gpus()}
        cached = set(llama_installed())
        alternatives = []
        if "nvidia" in vendors:
            alternatives += [CUDA_BACKEND, "CUDA 12.4" if system == "Windows" else "CUDA 12.8"]
        if "intel" in vendors:
            alternatives += ["SYCL"]
        if "amd" in vendors:
            alternatives += ["ROCm"]
        uncached = [choice for choice in alternatives if choice not in cached]
        for choice in alternatives:
            if choice not in cached:
                continue
            result = attempt(choice, False)
            if result and result[3]:
                reason = "; ".join(failures)
                if reason:
                    print(f"Chat Auto selected {result[3]} after {reason}.")
                return (*result, reason)
            if result:
                failures.append(f"{choice}: no visible GPU")
            else:
                # A cached binary can still lack its SDK/WSL bridge. Give its
                # dependency acquisition one online attempt before CPU fallback.
                uncached.append(choice)
    if backend == preferred and backend in ("CPU", "Vulkan") and not os.environ.get("LLAMA_SERVER"):
        from ruinedfooocus.backends.llama.llama_update import selected_build

        tag, updated = selected_build(backend, **callbacks)
        if updated is not None:
            try:
                result = attempt(backend, False, executable=updated)
                if result and result[3]:
                    return (*result, "")
            except RuntimeUnavailable:
                pass
            print(f"Chat could not initialize llama.cpp {tag} ({backend}); using the pinned build.")

    first = attempt(preferred, True)
    if first and first[3]:
        reason = "; ".join(failures)
        if reason:
            print(f"Chat Auto selected {first[3]} after {reason}.")
        return (*first, reason)
    if not automatic:
        raise RuntimeError(
            f"llama.cpp has no usable {preferred} GPU; check its driver or select CPU explicitly"
        )
    if first:
        failures.append(f"{preferred}: no visible GPU")
    for choice in uncached:
        result = attempt(choice, True)
        if result and result[3]:
            reason = "; ".join(failures)
            print(f"Chat Auto selected {result[3]} after {reason}.")
            return (*result, reason)
        if result:
            failures.append(f"{choice}: no visible GPU")
    reason = "; ".join(failures)
    if first:
        warn_cpu_fallback("Chat (llama.cpp)", reason)
        return first[0], first[1], False, "CPU", reason
    cpu = attempt("CPU", True)
    if cpu:
        warn_cpu_fallback("Chat (llama.cpp)", reason)
        return (*cpu, reason)
    raise RuntimeError("No verified cached Chat runtime is usable: " + "; ".join(failures))


class Server:
    def __init__(
        self,
        model,
        settings,
        gpu=True,
        *,
        model_defaults=(),
        check_cancelled=None,
        load_id=None,
        progress_callback=None,
    ):
        self.load = RuntimeLoad(load_id, progress_callback, check_cancelled)
        extra = shlex.split(settings.get("llama_server_args", ""))
        # Reject malformed values before runtime resolution can probe/acquire.
        moe_cache_cpu_share_capability(settings, arguments=extra)
        if not Path(model).is_file():
            raise FileNotFoundError(f"LLM model not found: {model}")
        self.process = None
        self.session = requests.Session()
        self.session.trust_env = False
        token = secrets.token_urlsafe(32)
        self.session.headers["Authorization"] = "Bearer " + token
        with socket.socket() as socket_:
            socket_.bind(("127.0.0.1", 0))
            port = socket_.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        backend = canonical_backend(settings.get("llama_backend", "Auto"))
        try:
            self.load.emit(phase="resolving_runtime", runtime={"backend": backend})
            executable, environment, gpu, self.backend, self.fallback_reason = resolve_runtime(
                gpu, backend, check_cancelled=self.load.check, progress_callback=self.load.emit
            )
            identity = runtime_identity(
                executable,
                self.backend,
                check_cancelled=self.load.check,
                custom=bool(os.environ.get("LLAMA_SERVER")),
            )
            model_identity = file_identity(model)
            header = model_header_evidence(model, check_cancelled=self.load.check)
        except Exception:
            self.session.close()
            raise
        try:
            moe_share = moe_cache_cpu_share_capability(
                {**settings, "llm_n_gpu_layers": settings.get("llm_n_gpu_layers", -1)},
                executable,
                gpu=gpu,
                arguments=extra,
                environment=environment,
                model_is_moe=header.get("moe") if header is not None else None,
            )
        except Exception:
            self.session.close()
            raise
        self.moe_cache_cpu_share = public_moe_cache_state(moe_share)
        warn_unsupported(moe_share, executable)
        extra = list(moe_share["extra_arguments"])
        projector = projector_path(settings)
        if projector:
            extra += ["--mmproj", str(projector)]
        if not gpu:
            extra += [
                "--no-mmproj-offload",
                "--device",
                "none",
                "--n-gpu-layers",
                "0",
                "--no-kv-offload",
                "--no-op-offload",
            ]
        layers = int(settings.get("llm_n_gpu_layers", -1)) if gpu else 0
        gpu_layers = "auto" if layers == -1 else "all" if layers < -1 else str(layers)
        predict = settings.get("llm_n_predict", -1)
        log_options = []
        if identity.get("manifest_verified") and identity.get("tag") in (
            "b12110",
            "b12327",
            "b12366",
        ):
            flags = {token.partition("=")[0] for token in (*model_defaults, *extra)}
            if "LLAMA_ARG_LOG_JSONL" not in environment:
                log_options.append("--log-jsonl")
            if "LLAMA_ARG_LOG_VERBOSITY" not in environment and not flags.intersection(
                {"-lv", "--verbosity", "--log-verbosity", "-v", "--verbose", "--log-disable"}
            ):
                # Native model/buffer information uses trace level, not debug.
                log_options += ["--verbosity", "4"]
        command = [
            str(executable),
            "--model",
            str(Path(model).resolve()),
            "--ctx-size",
            str(int(settings.get("llm_n_ctx", 8192))),
            "--n-predict",
            str(int(-1 if predict is None else predict)),
            "--n-gpu-layers",
            gpu_layers,
            "--parallel",
            "1",
            "--jinja",
            *log_options,
            *moe_share["arguments"],
            *model_defaults,
            *extra,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            token,
            "--no-webui",
        ]
        requested = native_request(
            settings, [*model_defaults, *extra], gpu=gpu, environment=environment
        )
        # Extra arguments follow the defaults, so account for a GPU-layer override.
        layer_flags = re.findall(
            r"(?:--n-gpu-layers|--gpu-layers|-ngl)(?:=|\s+)(\S+)", " ".join(extra)
        )
        if gpu and layer_flags:
            layers = layer_flags[-1]
        self.runtime_label = self.backend
        if str(layers) == "0" and self.backend != "CPU":
            self.runtime_label += "; CPU weights"
        log_path = data_path("cache", "llama.cpp", f"server-{port}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # This input-layer rule is reviewed for these exact native sources.
            # Custom binaries/tensor overrides cannot inherit that attribution.
            host_input = (
                header
                if (
                    identity.get("manifest_verified")
                    and identity.get("tag") in ("b12110", "b12327", "b12366")
                    and header is not None
                    and header.get("architecture") in ("gpt-oss", "lfm3")
                    and not requested.get("tensor_overrides")
                )
                else None
            )
            self.load.emit(phase="loading_model", runtime=identity)
            self.load.check()
            with log_path.open("w", encoding="utf-8") as log:
                self.process, job = start_process(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            self._cleanup = weakref.finalize(self, self._stop, self.process, self.session, job)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                self.load.check()
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with code {self.process.returncode}; see {log_path}."
                    )
                try:
                    if self.session.get(self.url + "/health", timeout=2).ok:
                        self.load.check()
                        if model_identity["signature"] != file_identity(model)["signature"]:
                            raise RuntimeVerificationError(
                                "Model changed while loading; retry when its files are stable."
                            )
                        if (
                            identity.get("signature") is not None
                            and identity["signature"] != file_identity(executable)["signature"]
                        ):
                            raise RuntimeVerificationError(
                                "Native runtime changed while loading; retry when its files are stable."
                            )
                        placement = startup_evidence(log_path, host_input=host_input)
                        model_identity.update(
                            quantization=placement["quantization"],
                            architecture=placement["architecture"],
                        )
                        self.moe_cache_cpu_share.update(
                            applied=bool(moe_share["arguments"])
                            or (moe_share["supported"] and moe_share["source"] != "app_setting"),
                            observed=None,
                            active=False,
                            initialized=bool(placement["moe_cache_buffers"]),
                            initialized_buffers=placement["moe_cache_buffers"],
                            host_experts=placement["host_experts"],
                        )
                        if self.moe_cache_cpu_share["applied"]:
                            self.moe_cache_cpu_share["reason"] = (
                                "The share option was passed; full-cache CPU miss execution has not been observed."
                            )
                        if placement["state"] == "full":
                            self.moe_cache_cpu_share.update(
                                active=False,
                                observed=False,
                                reason="Dormant: initialized offloadable weights are on the GPU.",
                            )
                        self.runtime_snapshot = {
                            "load_id": self.load.load_id,
                            "model_identity": model_identity,
                            "runtime_identity": identity,
                            "requested": requested,
                            "effective": {"context": placement["context"], "backend": self.backend},
                            "placement": placement,
                            "moe_cache": self.moe_cache_cpu_share.copy(),
                            "fallback_reason": self.fallback_reason,
                        }
                        if placement["state"] == "mixed":
                            manual = (
                                requested["gpu_layers"] not in ("auto", "-1")
                                or requested.get("cpu_experts")
                                or int(requested.get("cpu_expert_layers", 0)) > 0
                                or requested.get("tensor_overrides")
                            )
                            self.runtime_snapshot["allocation_reason"] = (
                                "Explicit native allocation controls are present."
                                if manual
                                else "Native automatic fit selected mixed placement within the available model/context/compute memory budget."
                                if requested["fit"] not in ("off", "0", "false")
                                else "Mixed placement was observed; automatic fit is disabled."
                            )
                        self.load.emit(phase="ready", result=self.runtime_snapshot)
                        return
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"llama-server did not finish loading; see {log_path}.")
        except CancelledError:
            self.close()
            self.load.emit(phase="cancelled", message="Model loading was cancelled.")
            raise
        except Exception:
            self.close()
            self.load.emit(phase="failed", message="The selected runtime could not load the model.")
            raise

    @staticmethod
    def _stop(process, session, job=None):
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if job:
                job.close()
            session.close()

    def close(self):
        if hasattr(self, "_cleanup"):
            self._cleanup()
        else:
            self.session.close()

    def _request(self, endpoint, data, callback=None):
        with self.session.post(
            self.url + endpoint, json=data, stream=bool(callback), timeout=(10, 600)
        ) as response:
            if not response.ok:
                template_kwargs = data.get("chat_template_kwargs") or {}
                if (
                    endpoint == "/v1/chat/completions"
                    and response.status_code == 400
                    and data.get("tools")
                    and "tools_in_user_message" not in template_kwargs
                    and "Unable to generate parser" in response.text
                    and "Cannot put tools in the first user message" in response.text
                ):
                    # Some Llama templates fail during the parser's empty-history probe.
                    # Keep tools enabled, but render their definitions in the system message.
                    retry = {
                        **data,
                        "chat_template_kwargs": {
                            **template_kwargs,
                            "tools_in_user_message": False,
                        },
                    }
                    response.close()
                    return self._request(endpoint, retry, callback)
                raise RuntimeError(f"llama-server: {response.status_code}: {response.text[:1000]}")
            if callback is None:
                return response.json()
            for line in response.iter_lines():
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    callback(chunk)

    def handle_chat_completions(self, data, callback=None):
        return self._request("/v1/chat/completions", data, callback)
