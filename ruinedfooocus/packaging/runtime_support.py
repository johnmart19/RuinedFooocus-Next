"""Select a runtime without importing PyTorch into the installer process."""

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeTarget:
    """Interpreter/environment supplied by the caller, never discovered by this policy.

    ``wheel_tags`` are the exact interpreter's supported tags, in string form.
    Supplying guest tags for WSL is essential: Windows host tags are not usable.
    """

    system: str
    machine: str
    python_version: str
    python_abi: str
    wheel_tags: frozenset[str] = frozenset()
    implementation: str = "cpython"
    environment: str = "native"
    marker_environment: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeArtifact:
    """One exact artifact from a resolved dependency closure or installed manifest."""

    name: str
    version: str
    filename: str | None = None
    available: bool | None = None
    requires_python: str = ""
    requirements: tuple[str, ...] = ()
    source: str = ""
    sha256: str = ""
    extras: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeCandidate:
    """A versioned route, not a vendor-level promise of support.

    Route names distinguish native MIGraphX plugin, legacy integrated MIGraphX,
    Windows ML AMDGPU, experimental HIP and WSL recipes. Their publisher/device
    allowlists and driver checks belong to the recipe producer. Unknown evidence
    stays ``None``; installed package names alone cannot make a route usable.
    ``backend`` is the saved family/engine (cuda, rocm, Vulkan); ``platform``
    identifies its executable/package route (cu126, cu132, rocm10.1), while
    ``recipe_id`` retains the caller's versioned recipe identity. Historical
    exact image backend values are normalized without dropping their route.
    """

    backend: str
    operation: str
    systems: tuple[str, ...]
    machines: tuple[str, ...] = ("x86_64",)
    environments: tuple[str, ...] = ("native",)
    adapter_id: str | None = None
    device_kind: str = "gpu"
    vendor: str | None = None
    hardware_supported: bool | None = None
    driver_supported: bool | None = None
    artifacts: tuple[RuntimeArtifact, ...] = ()
    required_packages: tuple[str, ...] = ()
    artifact_sources: tuple[str, ...] = ()
    closure_complete: bool | None = None
    native_executable: bool = False
    native_payload: bool | None = None
    python_validated: bool | None = None
    installed: bool = False
    evidence_current: bool | None = None
    device_visible: bool | None = None
    workload_qualified: bool | None = None
    active: bool = False
    opt_in: bool = False
    platform: str | None = None
    recipe_id: str | None = None


def _machine_name(machine):
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(machine.lower(), machine.lower())


def _route_identity(candidate):
    """Keep a saved backend family separate from the executable/package route."""
    backend = candidate.backend
    route = candidate.platform or backend
    if candidate.operation == "image":
        if re.fullmatch(r"cu\d+", backend):
            backend = "cuda"
        elif re.fullmatch(r"rocm\d+(?:\.\d+)*", backend):
            backend = "rocm"
        if route in ("cuda", "rocm"):
            route = None
    return backend, route


def _marker_variables(marker):
    """Read variable names from public marker syntax without treating literals as names."""
    tokens = re.findall(
        r""""(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[A-Za-z_][A-Za-z_0-9]*""", str(marker)
    )
    names = {
        "implementation_name",
        "implementation_version",
        "os_name",
        "platform_machine",
        "platform_release",
        "platform_system",
        "platform_version",
        "python_full_version",
        "platform_python_implementation",
        "python_version",
        "sys_platform",
        "extra",
        "extras",
        "dependency_groups",
    }
    return names.intersection(tokens)


def _artifact_reasons(target, candidate):
    """Validate exact wheel tags, Requires-Python and the complete declared closure."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
    from packaging.version import InvalidVersion, Version

    rejected, unknown = [], []
    artifacts = {canonicalize_name(item.name): item for item in candidate.artifacts}
    if len(artifacts) != len(candidate.artifacts):
        rejected.append("Dependency closure contains duplicate package identities")
    if candidate.closure_complete is False:
        rejected.append("Dependency closure is incomplete or conflicting")
    elif candidate.closure_complete is not True:
        unknown.append("Dependency closure has not been verified")
    if not candidate.artifacts and not candidate.native_executable:
        unknown.append("Exact runtime wheel artifacts are unknown")
    for name in candidate.required_packages:
        if canonicalize_name(name) not in artifacts:
            rejected.append(f"Dependency closure is missing {name}")
    # Missing target metadata is unknown, not an empty string that can silently
    # skip a dependency. Check every referenced variable before evaluation so
    # packaging cannot fill missing values from the installer's host environment.
    markers = dict(
        python_version=".".join(target.python_version.split(".")[:2]),
        python_full_version=target.python_version,
        implementation_name=target.implementation,
        platform_python_implementation="CPython"
        if target.implementation == "cpython"
        else target.implementation,
        platform_system=target.system,
        platform_machine=target.machine,
        extra="",
    )
    if target.implementation == "cpython":
        markers["implementation_version"] = target.python_version
    if target.system in ("Windows", "Linux", "Darwin"):
        markers["sys_platform"] = {"Windows": "win32", "Linux": "linux", "Darwin": "darwin"}[
            target.system
        ]
        markers["os_name"] = "nt" if target.system == "Windows" else "posix"
    markers.update(target.marker_environment)
    for artifact in candidate.artifacts:
        name = canonicalize_name(artifact.name)
        if not artifact.source:
            unknown.append(f"Artifact source is unknown for {name}")
        elif candidate.artifact_sources and not any(
            artifact.source == source or artifact.source.startswith(source.rstrip("/") + "/")
            for source in candidate.artifact_sources
        ):
            rejected.append(f"{name} is from the wrong artifact source")
        if artifact.available is False:
            rejected.append(f"No published artifact is available for {name}")
        elif artifact.available is not True:
            unknown.append(f"Artifact availability is unknown for {name}")
        try:
            version = Version(artifact.version)
            if artifact.requires_python and not SpecifierSet(artifact.requires_python).contains(
                target.python_version, prereleases=True
            ):
                rejected.append(f"{name} requires Python {artifact.requires_python}")
            if artifact.filename:
                wheel_name, wheel_version, _, tags = parse_wheel_filename(artifact.filename)
                if canonicalize_name(wheel_name) != name or wheel_version != version:
                    rejected.append(f"Wheel identity does not match {name}=={artifact.version}")
                if not target.wheel_tags:
                    unknown.append("Interpreter wheel tags are unknown")
                elif not target.wheel_tags.intersection(str(tag) for tag in tags):
                    rejected.append(
                        f"{name} has no wheel matching {target.python_abi}/{target.system}/{target.machine}"
                    )
            else:
                unknown.append(f"Exact wheel filename is unknown for {name}")
            for raw in artifact.requirements:
                requirement = Requirement(raw)
                if requirement.marker:
                    missing = sorted(
                        key
                        for key in _marker_variables(requirement.marker)
                        if key not in markers or markers[key] is None
                    )
                    if missing:
                        unknown.append(
                            f"Dependency marker for {name} needs target metadata: {', '.join(missing)}"
                        )
                        continue
                if requirement.marker and not any(
                    requirement.marker.evaluate(markers | {"extra": extra})
                    for extra in ("", *artifact.extras)
                ):
                    continue
                dependency = artifacts.get(canonicalize_name(requirement.name))
                if dependency is None:
                    rejected.append(f"{name} requires missing {requirement}")
                elif requirement.url:
                    if dependency.source != requirement.url:
                        rejected.append(
                            f"{name} requires a different source for {requirement.name}"
                        )
                elif not requirement.specifier.contains(dependency.version, prereleases=True):
                    rejected.append(f"{name} requires {requirement}; found {dependency.version}")
                if dependency is not None and not requirement.extras.issubset(dependency.extras):
                    rejected.append(f"{name} requires unresolved extras for {requirement.name}")
        except (
            InvalidVersion,
            InvalidSpecifier,
            InvalidWheelFilename,
            InvalidRequirement,
            ValueError,
        ) as error:
            rejected.append(f"Invalid artifact metadata for {name}: {error}")
    return rejected, unknown


def capability_decision(target, candidate, adapters=None):
    """Pure installer/runtime/Settings admission contract; performs no discovery.

    ``eligible`` means the exact route can be acquired, ``usable`` additionally
    requires current installed/device evidence. Qualification is independent.
    Auto selection requires qualification; explicit activation may verify an
    eligible installed route without claiming prior workload qualification.
    """
    rejected, unknown = [], []
    backend, route = _route_identity(candidate)
    if (
        candidate.operation == "image"
        and backend != candidate.backend
        and candidate.platform
        and candidate.platform != candidate.backend
    ):
        rejected.append("Exact backend route conflicts with the runtime platform")
    if route is None:
        unknown.append("Exact runtime platform is unknown")
    elif candidate.operation == "image" and backend in ("cuda", "rocm"):
        prefix = r"cu\d+" if backend == "cuda" else r"rocm\d+(?:\.\d+)*"
        if not re.fullmatch(prefix, route):
            rejected.append("Runtime platform does not match the selected backend family")
    if target.system not in candidate.systems:
        rejected.append(f"Route requires {'/'.join(candidate.systems)}")
    if target.environment not in candidate.environments:
        rejected.append(f"Route is not available in {target.environment}")
    if _machine_name(target.machine) not in tuple(
        _machine_name(value) for value in candidate.machines
    ):
        rejected.append(f"Route is not published for {target.machine}")
    adapter = next(
        (item for item in adapters or () if item.get("adapter_id") == candidate.adapter_id), None
    )
    if candidate.device_kind != "cpu":
        if candidate.adapter_id is None:
            unknown.append("Runtime adapter identity is unknown")
        elif adapters is None:
            unknown.append("Adapter inventory is unavailable")
        elif adapter is None:
            rejected.append("Selected adapter is absent from this inventory")
        else:
            if adapter.get("kind", "gpu") != candidate.device_kind:
                rejected.append("Adapter GPU/NPU kind does not match this route")
            if candidate.vendor and adapter.get("vendor") != candidate.vendor:
                rejected.append("Adapter vendor does not match this route")
            if adapter.get("location") == "guest" and target.environment != "wsl":
                rejected.append("Guest adapter does not belong to this native environment")
            if (
                adapter.get("location") == "host"
                and target.system != "Windows"
                and target.environment != "wsl"
            ):
                rejected.append("Windows host adapter does not belong to native Linux")
            if adapter.get("location") == "host" and target.environment == "wsl":
                if adapter.get("guest_available") is False:
                    rejected.append("Host adapter is unavailable in the WSL guest")
                elif candidate.installed and adapter.get("guest_available") is not True:
                    unknown.append("Host presence does not establish WSL guest availability")
        for label, value in (
            ("Device architecture", candidate.hardware_supported),
            ("Driver", candidate.driver_supported),
        ):
            if value is False:
                rejected.append(f"{label} is unsupported by this route")
            elif value is not True:
                unknown.append(f"{label} compatibility has not been verified")
    if candidate.native_payload is False:
        rejected.append("Native runtime payload is missing")
    elif candidate.native_payload is not True:
        unknown.append("Native runtime payload has not been verified")
    artifact_rejected, artifact_unknown = _artifact_reasons(target, candidate)
    rejected += artifact_rejected
    unknown += artifact_unknown
    if (
        candidate.artifacts or not candidate.native_executable
    ) and candidate.python_validated is not True:
        # An available ABI wheel and vendor validation are distinct facts.
        unknown.append(
            "Publisher has not validated this Python interpreter"
            if candidate.python_validated is False
            else "Publisher Python validation is unknown"
        )
    eligible = not rejected and not unknown
    usable = (
        eligible
        and candidate.installed
        and candidate.evidence_current is True
        and (candidate.device_kind == "cpu" or candidate.device_visible is True)
    )
    reasons = rejected + unknown
    if eligible and not candidate.installed:
        reasons.append("Compatible runtime is not installed")
    elif eligible and candidate.evidence_current is not True:
        reasons.append("Installed runtime evidence is stale or unverified")
    elif eligible and candidate.device_kind != "cpu" and candidate.device_visible is not True:
        reasons.append(
            "Runtime device visibility has not been verified"
            if candidate.device_visible is None
            else "Selected adapter is not visible to this runtime"
        )
    qualified = usable and candidate.workload_qualified is True
    if usable and not qualified:
        reasons.append("Selected workload has not been qualified")
    if candidate.opt_in:
        reasons.append("Route requires explicit opt-in")
    state = (
        "unavailable"
        if rejected
        else "unknown"
        if unknown
        else "installable"
        if not candidate.installed
        else "qualified"
        if qualified
        else "usable"
        if usable
        else "installed"
    )
    return dict(
        backend=backend,
        platform=route,
        recipe_id=candidate.recipe_id,
        operation=candidate.operation,
        adapter_id=candidate.adapter_id,
        state=state,
        eligible=eligible,
        installed=candidate.installed,
        usable=usable,
        qualified=qualified,
        active=usable and candidate.active,
        auto_eligible=qualified and not candidate.opt_in,
        opt_in=candidate.opt_in,
        device_kind=candidate.device_kind,
        python_validated=candidate.python_validated,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def select_capability(
    decisions,
    operation,
    *,
    requested="auto",
    requested_adapter_id=None,
    cpu_scope="none",
    preference=(),
):
    """Resolve one operation without rewriting saved requests or selecting unknowns.

    Explicit GPU requests fail visibly. Auto tries all qualified GPU adapters
    before CPU; image-only CPU never affects Chat or ONNX decisions.
    """
    if cpu_scope not in ("none", "image", "global"):
        raise ValueError("CPU scope must be none, image or global")
    forced = cpu_scope == "global" or (cpu_scope == "image" and operation == "image")
    rows = [
        row
        for row in decisions
        if row["operation"] == operation
        and (forced or requested_adapter_id is None or row["adapter_id"] == requested_adapter_id)
    ]
    request = "cpu" if forced else requested
    automatic = request.lower() == "auto"
    if not automatic:
        rows = [
            row
            for row in rows
            if request.lower()
            in {str(row.get(key) or "").lower() for key in ("backend", "platform", "recipe_id")}
        ]
    order = {name.lower(): index for index, name in enumerate(preference)}
    rows.sort(
        key=lambda row: (
            row["device_kind"] == "cpu" if automatic else False,
            min(
                order.get(str(row.get(key) or "").lower(), len(order))
                for key in ("backend", "platform", "recipe_id")
            ),
        )
    )
    selected = (
        next((row for row in rows if row["auto_eligible"]), None)
        if automatic
        else next((row for row in rows if row["usable"]), None)
    )
    if selected:
        reasons = selected["reasons"]
        if automatic and selected["device_kind"] == "cpu":
            reasons = tuple(
                dict.fromkeys(
                    reason
                    for row in rows
                    if row["device_kind"] != "cpu"
                    for reason in row["reasons"]
                )
            ) or ("No qualified GPU runtime is available for this operation",)
    else:
        reasons = tuple(dict.fromkeys(reason for row in rows for reason in row["reasons"])) or (
            "No compatible runtime is available for this operation",
        )
    return dict(
        requested=requested,
        resolved=selected["backend"] if selected else None,
        platform=selected.get("platform") if selected else None,
        recipe_id=selected.get("recipe_id") if selected else None,
        adapter_id=selected["adapter_id"] if selected else None,
        cpu_scope=cpu_scope,
        automatic=automatic,
        decision=selected,
        reasons=reasons,
    )


def amd_guide_applicable(target, adapter, guide):
    """Narrow published tuning applicability; this never changes system settings."""
    if (
        target.system != "Linux"
        or target.environment != "native"
        or adapter.get("vendor") != "amd"
        or adapter.get("kind", "gpu") != "gpu"
    ):
        return False
    if guide == "rdna35-apu-memory":
        return (
            adapter.get("architecture") in ("gfx1150", "gfx1151", "gfx1152")
            and adapter.get("integrated") is True
        )
    if guide == "v620-sriov":
        return (
            adapter.get("product") == "Radeon PRO V620" and adapter.get("virtualization") == "sriov"
        )
    return False


def _read_installed_torch(backend="rocm"):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        import torch
    except Exception:
        if backend in version.lower() or (backend == "cuda" and "+cu" in version.lower()):
            raise
        return None
    runtime = (
        "cpu"
        if backend == "cpu"
        else torch.version.hip
        if backend == "rocm"
        else torch.version.cuda
    )
    if not runtime:
        return None
    if backend == "cpu":
        if (torch.ones((2, 2), device="cpu") @ torch.ones((2, 2), device="cpu"))[0, 0].item() != 2:
            raise RuntimeError("CPU kernel check failed.")
    if backend == "cuda":
        if not torch.cuda.is_available():
            return None
        # Validate kernels too: an import can succeed with unsupported GPU wheels.
        x = torch.ones((16, 16), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        if y[0, 0].item() != 16:
            raise RuntimeError("CUDA kernel check failed.")

    import torchvision  # Validate native extensions before preserving the bundle.

    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision")}
    for name in versions:
        for raw in importlib.metadata.requires(name) or []:
            requirement = Requirement(raw)
            dependency = canonicalize_name(requirement.name)
            if dependency not in versions or (
                requirement.marker and not requirement.marker.evaluate()
            ):
                continue
            if not requirement.specifier.contains(versions[dependency], prereleases=True):
                raise RuntimeError(f"{name} requires {requirement}, found {versions[dependency]}")
    return {"hip" if backend == "rocm" else backend: runtime, "versions": versions}


def _probe_installed_torch(backend):
    """Keep torch imports isolated and normalize child-process probe failures."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from ruinedfooocus.packaging.runtime_support import _read_installed_torch; "
                f"print(json.dumps(_read_installed_torch({backend!r})))",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Installed {backend.upper()} validation could not finish.") from error
    if result.returncode:
        raise RuntimeError(result.stderr or f"Installed {backend.upper()} validation failed.")
    try:
        bundle = json.loads(result.stdout.strip().splitlines()[-1])
        key = "hip" if backend == "rocm" else backend
        if bundle is not None and (
            not isinstance(bundle, dict)
            or not isinstance(bundle.get(key), str)
            or not bundle[key]
            or not isinstance(bundle.get("versions"), dict)
            or not isinstance(bundle["versions"].get("torch"), str)
        ):
            raise ValueError("Invalid runtime bundle")
        return bundle
    except (ValueError, IndexError) as error:
        raise RuntimeError(
            f"Installed {backend.upper()} validation returned invalid output."
        ) from error


def inspect_installed_cpu():
    try:
        return _probe_installed_torch("cpu")
    except RuntimeError:
        return None


def inspect_installed_cuda(os_platform):
    override = os.environ.get("TORCH_PLATFORM", "")
    if os_platform not in ("Windows", "Linux") or (override and not override.startswith("cu")):
        return None
    try:
        if "rocm" in importlib.metadata.version("torch").lower():
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return _probe_installed_torch("cuda")
    except RuntimeError:
        # Allow the installer to repair a broken CUDA environment.
        print("Installed CUDA packages need repair; checking installation requirements.")
        return None


def installed_cuda_platform(bundle):
    major, minor, *_ = bundle["cuda"].split(".")
    return f"cu{major}{minor}"


def inspect_installed_rocm(os_platform, repair=False):
    """Probe in a child process so pip never replaces a loaded torch DLL."""
    override = os.environ.get("TORCH_PLATFORM", "")
    if os_platform not in ("Windows", "Linux") or (override and not override.startswith("rocm")):
        return None
    try:
        if "+cu" in importlib.metadata.version("torch").lower():
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return _probe_installed_torch("rocm")
    except RuntimeError as error:
        if repair:
            print("Installed ROCm packages need repair; checking AMD installation requirements.")
            return None
        raise RuntimeError(
            "Cannot validate the installed ROCm bundle. Repair torch/torchvision; "
            "RuinedFooocus has not replaced them.\n" + str(error)
        ) from error


def installed_rocm_platform(bundle):
    # ROCm release and HIP component versions differ; prefer the wheel's release tag.
    version = bundle["versions"]["torch"]
    match = re.search(r"\+rocm(\d+)\.(\d+)", version)
    if not match:
        match = re.match(r"(\d+)\.(\d+)", bundle["hip"])
    if not match:
        raise RuntimeError(f"Unrecognized ROCm build: {version}")
    return f"rocm{match[1]}.{match[2]}"


def installed_directml_matches_requirements(os_platform):
    """Retain an old explicit environment without importing its binary runtime.

    This checks package declarations, not device execution. Fresh installations
    must not downgrade modern Torch just because PCI detection suggests DML.
    """
    if os_platform != "Windows" and not (
        os_platform == "Linux" and "microsoft" in platform.release().lower()
    ):
        return False
    from packaging.requirements import Requirement, InvalidRequirement
    from packaging.version import Version, InvalidVersion

    try:
        importlib.metadata.version("torch-directml")
        requirements = {}
        for raw in importlib.metadata.requires("torch-directml") or ():
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate():
                requirements[requirement.name.lower().replace("_", "-")] = requirement
        return all(
            name in requirements
            and requirements[name].specifier.contains(
                importlib.metadata.version(name), prereleases=True
            )
            for name in ("torch", "torchvision")
        ) and Version(importlib.metadata.version("numpy")) < Version("2")
    except (importlib.metadata.PackageNotFoundError, InvalidRequirement, InvalidVersion):
        return False


def _platform_for(torchruntime, gpus):
    """Try every vendor group before a mixed list falls back to CPU/DirectML.

    This legacy discovery adapter is not an eligibility certificate; new callers
    pass exact artifact/device decisions to ``select_torch_platform``.
    """
    try:
        fallback = torchruntime.platform_detection.get_torch_platform(gpus)
    except NotImplementedError:
        fallback = "cpu"
    if fallback not in ("cpu", "directml", "rocm5.5"):
        return fallback
    vendors = dict.fromkeys(str(getattr(gpu, "vendor_id", "")).lower() for gpu in gpus)
    for vendor in vendors:
        group = [gpu for gpu in gpus if str(getattr(gpu, "vendor_id", "")).lower() == vendor]
        try:
            selected = torchruntime.platform_detection.get_torch_platform(group)
        except (NotImplementedError, RuntimeError):
            continue
        if selected not in ("cpu", "directml", "rocm5.5"):
            return selected
    return fallback


def wsl_device_infos(torchruntime, gpus):
    """Retain guest evidence; Windows host PCI IDs cannot certify a guest route.

    Host adapters remain visible in ``hardware_inventory``. Explicit guest
    runtime verification can bind them through ``RuntimeCandidate.adapter_id``.
    """
    return gpus


def select_torch_platform(
    torchruntime,
    os_platform,
    installed_rocm=None,
    installed_cuda=None,
    *,
    decisions=None,
    requested="auto",
    cpu_scope="none",
):
    if decisions is not None:
        result = select_capability(decisions, "image", requested=requested, cpu_scope=cpu_scope)
        if result["resolved"] is None:
            raise RuntimeError("; ".join(result["reasons"]))
        if result["platform"] is None:
            raise RuntimeError("Exact runtime platform is unknown")
        return result["platform"]
    override = os.environ.get("TORCH_PLATFORM")
    if override:
        selected = override
    elif installed_rocm:
        selected = installed_rocm_platform(installed_rocm)
    elif installed_cuda:
        selected = installed_cuda_platform(installed_cuda)
    else:
        gpus = torchruntime.device_db.get_gpus()
        if os_platform == "Linux":
            gpus = wsl_device_infos(torchruntime, gpus)
        from ruinedfooocus.packaging.cuda_selection import select_cuda, nvidia_info

        nvidia = any(gpu.vendor_id.lower() == "10de" for gpu in gpus)
        if not nvidia and os_platform in ("Windows", "Linux"):
            # WSL and virtual machines may not expose NVIDIA in the PCI list.
            capability, _ = nvidia_info()
            # An installed driver's CUDA header alone does not prove that an
            # NVIDIA GPU is present (for example after moving a disk to AMD).
            nvidia = capability is not None
        if nvidia:
            try:
                selected = select_cuda(gpus)
            except RuntimeError as error:
                alternatives = [gpu for gpu in gpus if str(gpu.vendor_id).lower() != "10de"]
                selected = _platform_for(torchruntime, alternatives)
                if selected == "cpu":
                    print(f"NVIDIA CUDA unavailable: {error}")
        else:
            selected = _platform_for(torchruntime, gpus)
    if (
        selected == "directml"
        and not override
        and not installed_directml_matches_requirements(os_platform)
    ):
        print(
            "Automatic torch-directml downgrades are disabled; checking native AMD support or using CPU. Use --directml explicitly only in a compatible legacy environment."
        )
        return "cpu"
    if selected == "rocm5.5" or (selected == "mps" and os_platform != "Darwin"):
        return "cpu"
    return selected
