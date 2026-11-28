"""One bounded, cancellable native device probe shared by acquisition and Chat."""

import re

from ruinedfooocus.core.bounded_subprocess import BoundedProcessError, run_bounded


class RuntimeVerificationError(RuntimeError):
    """A device/runtime error requiring an explicit recovery decision."""


class RuntimeUnavailable(RuntimeError):
    """A missing executable dependency before device execution."""


def probe_runtime(
    executable, environment, expected=None, *, check_cancelled=None, progress_callback=None
):
    if progress_callback is not None:
        progress_callback(
            {"phase": "probing_device", "message": "Checking the selected runtime's devices."}
        )
    try:
        stdout, _ = run_bounded(
            [str(executable), "--list-devices"],
            env=environment,
            timeout=30,
            max_output=64 * 1024,
            check_cancelled=check_cancelled,
        )
    except OSError as error:
        if error.errno == 2 or getattr(error, "winerror", None) == 126:
            raise RuntimeUnavailable("Native runtime loader dependency is missing") from error
        raise
    except BoundedProcessError as error:
        if error.returncode >= 0 and re.search(
            r"error while loading shared libraries|cannot open shared object file|Vulkan loader.*not found",
            error.stderr.decode("utf-8", errors="replace"),
            re.I,
        ):
            raise RuntimeUnavailable("Native runtime loader dependency is missing") from error
        raise RuntimeVerificationError(
            "llama.cpp device verification failed; select another backend explicitly after checking its driver"
        ) from error
    except RuntimeError as error:
        # A device crash is not evidence for a safe automatic CPU retry.
        raise RuntimeVerificationError(
            "llama.cpp device verification failed; select another backend explicitly after checking its driver"
        ) from error
    names = re.findall(
        r"^\s*(CUDA|ROCm|Vulkan|Metal|SYCL)\d*:",
        stdout.decode("utf-8", errors="replace"),
        re.MULTILINE,
    )
    return (
        "/".join(dict.fromkeys(names))
        if names and (expected is None or expected in names)
        else None
    )
