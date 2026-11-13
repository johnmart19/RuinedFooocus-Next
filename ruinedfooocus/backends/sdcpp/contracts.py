"""Pure request admission, independent of Torch and native device discovery."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re

SOURCE_REVISION = "fc350be2ceeeb859bcc201ac435591ed7d46821d"
RELEASE = "master-1068-fc350be"


class UnsupportedRequest(ValueError):
    """The optional engine cannot preserve this request's complete contract."""


@dataclass(frozen=True)
class ModelSpec:
    path: Path
    architecture: str
    format: str
    weight_types: tuple[str, ...]
    bundled: bool
    signature: tuple = ()
    sha256: str = ""
    quantization: str = "none"
    variant: str = "unknown"


def inspect_model(path, *, architecture, variant="unknown"):
    """Inspect a prepared file using the caller's native architecture classification.

    Call during explicit preparation, not picker rendering. Header inspection is
    bounded; it neither hashes weights nor imports Torch. A native SDXL result
    does not distinguish a distilled/turbo derivative: the caller must retain its
    known variant and unknown variants must remain unknown.
    """
    from ruinedfooocus.features.models.model_identity import (
        file_signature,
        read_safetensors_header,
    )

    path = Path(path).resolve()
    before = tuple(file_signature(path))
    header = read_safetensors_header(path)
    names = [key for key in header if key != "__metadata__"]
    bundled = all(
        any(name.startswith(prefix) for name in names)
        for prefix in (
            "model.diffusion_model.",
            "conditioner.embedders.0.",
            "conditioner.embedders.1.",
            "first_stage_model.",
        )
    )
    types = tuple(sorted({str(header[name].get("dtype", "unknown")) for name in names}))
    if before != tuple(file_signature(path)):
        raise OSError("Checkpoint changed during inspection; retry preparation")
    return ModelSpec(path, architecture, "safetensors", types, bundled, before, variant=variant)


@dataclass(frozen=True)
class GenerationRequest:
    model: ModelSpec
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    operation: str = "txt2img"
    denoise: float = 1.0
    init_image: Path | None = None
    request_id: str = ""
    clip_skip: int = -1
    batch_size: int = 1
    mask: Path | None = None
    loras: tuple = ()
    embeddings: tuple = ()
    controlnets: tuple = ()
    component_overrides: tuple = ()
    refiner: str = ""
    # The caller must enumerate every other requested processing/conditioning
    # feature, including styles not yet expanded, hires, IP adapters and preview.
    extra_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeCandidate:
    executable: Path
    backend: str
    device: str
    revision: str = SOURCE_REVISION
    artifact_sha256: str = ""
    payload_signatures: tuple = ()
    device_verified: bool = False
    qualified: bool = False
    qualification: str = ""
    # Exact model/operation receipts are supplied by explicit acceptance, never
    # manufactured by --list-devices. Empty means no accepted workloads.
    accepted_models: tuple[str, ...] = ()
    accepted_operations: tuple[str, ...] = ()
    device_environment: tuple = ()
    device_description: str = ""


class _Cancellation:
    """Adapt the caller's cancellation callback to preparation's Event protocol."""

    def __init__(self, check):
        self.check = check

    def is_set(self):
        self.check()
        return False


_PROMPT_SYNTAX = re.compile(r"[\[\](){}<>|]|\b(?:BREAK|AND)\b|(?i:embedding:)")


def assess(request, candidate=None, *, allow_unqualified=False):
    """Return all refusal reasons without I/O, downloads or compute discovery."""
    reasons = []
    model = request.model
    if model.architecture != "SDXL 1.0" or model.variant != "base":
        reasons.append("Only standard SDXL 1.0 base is source-supported")
    if model.format != "safetensors" or model.quantization != "none":
        reasons.append("Only unquantized safetensors checkpoints are supported")
    if not model.weight_types or not set(model.weight_types).issubset({"F16", "F32"}):
        reasons.append("Only F16/F32 checkpoint storage is supported")
    if not model.bundled:
        reasons.append("A full checkpoint with bundled CLIP-L, CLIP-G and VAE is required")
    if not model.signature:
        reasons.append("Checkpoint inspection identity is missing")
    if request.operation not in ("txt2img", "img2img"):
        reasons.append("Only txt2img and img2img are supported")
    if request.sampler != "euler" or request.scheduler not in ("normal", "discrete"):
        reasons.append("Only Euler with the normal/discrete scheduler is supported")
    if request.clip_skip != -1:
        reasons.append("Explicit CLIP skip is not qualified")
    if request.batch_size != 1:
        reasons.append("Only single-image requests are supported")
    for field in ("mask", "loras", "embeddings", "controlnets", "component_overrides", "refiner"):
        if getattr(request, field):
            reasons.append(f"Unsupported conditioning: {field}")
    if request.extra_features:
        reasons.append("Unsupported features: " + ", ".join(request.extra_features))
    for field in ("prompt", "negative_prompt"):
        value = getattr(request, field)
        if (
            not isinstance(value, str)
            or len(value) > 8192
            or "\0" in value
            or _PROMPT_SYNTAX.search(value)
        ):
            reasons.append(f"Only plain literal {field} text is qualified")
    if any(
        type(n) is not int or not 64 <= n <= 8192 or n % 64 for n in (request.width, request.height)
    ):
        reasons.append("Dimensions must be multiples of 64 in [64, 8192]; no implicit rounding")
    if type(request.steps) is not int or request.steps < 1:
        reasons.append("Steps must be a positive integer")
    if type(request.seed) is not int or not 0 <= request.seed <= 2**63 - 1:
        reasons.append("Resolve random seeds before submission; expected a signed 64-bit seed")
    if type(request.cfg) not in (int, float) or not math.isfinite(request.cfg) or request.cfg < 0:
        reasons.append("CFG must be finite and nonnegative")
    if (
        type(request.denoise) not in (int, float)
        or not math.isfinite(request.denoise)
        or not 0 < request.denoise <= 1
    ):
        reasons.append("Denoise must be finite and in (0, 1]")
    if request.operation == "txt2img" and (request.init_image is not None or request.denoise != 1):
        reasons.append("txt2img requires no init image and full denoise")
    if request.operation == "img2img" and request.init_image is None:
        reasons.append("img2img requires a prepared init image matching the requested dimensions")
    if candidate is not None:
        if candidate.revision != SOURCE_REVISION or not candidate.payload_signatures:
            reasons.append("Runtime pin/payload identity is unverified")
        if not re.fullmatch(r"[a-f0-9]{64}", candidate.artifact_sha256):
            reasons.append("Runtime archive checksum is missing")
        if candidate.backend not in ("cpu", "vulkan") or not re.fullmatch(
            "cpu" if candidate.backend == "cpu" else r"Vulkan\d+", candidate.device
        ):
            reasons.append("Select an exact CPU or Vulkan runtime device")
        if not candidate.device_verified:
            reasons.append("Runtime device has not been explicitly verified")
        if not allow_unqualified and (
            not candidate.qualified
            or not candidate.qualification
            or not model.sha256
            or model.sha256.lower() not in candidate.accepted_models
            or request.operation not in candidate.accepted_operations
        ):
            reasons.append("This runtime/device/model/operation has no actual acceptance receipt")
    return tuple(reasons)


def choose_runtime(request, candidates, *, preference="auto", native_gpu_compatible=False):
    """Pure optional-engine resolver. None means retain the compatible native GPU.

    CPU is considered only after compatible qualified Vulkan candidates. Runtime
    failures never trigger a CPU retry; this function does not execute candidates.
    """
    if preference not in ("auto", "cpu", "vulkan"):
        raise UnsupportedRequest("Unknown optional image backend preference")
    if preference == "auto" and native_gpu_compatible:
        return None
    failures = list(assess(request))
    for backend in ("vulkan", "cpu") if preference == "auto" else (preference,):
        pending_vulkan = False
        for candidate in candidates:
            if candidate.backend != backend:
                continue
            reasons = assess(request, candidate)
            if not reasons:
                return candidate
            failures.extend(reasons)
            if backend == "vulkan" and not assess(request, candidate, allow_unqualified=True):
                pending_vulkan = True
        if preference == "auto" and pending_vulkan:
            raise UnsupportedRequest(
                "A compatible usable Vulkan candidate needs workload acceptance; "
                "qualify it or select CPU explicitly"
            )
    raise UnsupportedRequest(
        "; ".join(dict.fromkeys(failures or ["No prepared compatible runtime"]))
    )


def with_model_hash(model, sha256):
    """Attach the hash already verified by acquisition/explicit acceptance."""
    if not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError("Expected the prepared model's SHA256")
    return replace(model, sha256=sha256.lower())
