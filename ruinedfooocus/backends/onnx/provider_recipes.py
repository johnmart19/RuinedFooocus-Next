"""Pure, exact AMD ONNX routes. Unknown artifact/device evidence stays unknown."""

from ruinedfooocus.packaging.runtime_support import RuntimeCandidate, capability_decision

MIGRAPHX_RECIPE = "onnx-migraphx-plugin-rocm10.1-ort1.29"
MIGRAPHX_PACKAGES = (
    "onnxruntime-ep-migraphx==1.0.0+rocm10.1.0",
    "onnxruntime==1.29.0",
    "migraphx==2.18.0+rocm10.1.0",
    "migraphx-libs==2.18.0+rocm10.1.0",
)
MIGRAPHX_INDEX = "https://stable.repo.amd.com/rocm/onnxruntime/whl-next/"
MIGRAPHX_LIBS_INDEX = "https://stable.repo.amd.com/rocm/migraphx/whl-next/"
MIGRAPHX_GFX = frozenset(
    (
        "gfx950",
        "gfx942",
        "gfx90a",
        "gfx1200",
        "gfx1201",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
    )
)


def migraphx_candidate(
    target,
    adapter=None,
    *,
    artifacts=(),
    closure_complete=None,
    native_payload=None,
    installed=False,
    evidence_current=None,
    device_visible=None,
    workload_qualified=None,
):
    adapter = adapter or {}
    architecture = adapter.get("gfx") or adapter.get("architecture")
    return RuntimeCandidate(
        backend="MIGraphX",
        operation="onnx",
        systems=("Linux",),
        platform="rocm10.1-plugin",
        recipe_id=MIGRAPHX_RECIPE,
        adapter_id=adapter.get("adapter_id"),
        vendor="amd",
        hardware_supported=(architecture in MIGRAPHX_GFX) if architecture else None,
        driver_supported=adapter.get(
            "rocm10.1_driver_supported", adapter.get("rocm10_1_driver_supported")
        ),
        artifacts=tuple(artifacts),
        required_packages=tuple(p.split("==")[0] for p in MIGRAPHX_PACKAGES),
        artifact_sources=(MIGRAPHX_INDEX, MIGRAPHX_LIBS_INDEX, "https://files.pythonhosted.org/"),
        closure_complete=closure_complete,
        native_payload=native_payload,
        python_validated=target.python_version.split(".")[:2] == ["3", "12"],
        installed=installed,
        evidence_current=evidence_current,
        device_visible=device_visible,
        workload_qualified=workload_qualified,
    )


def migraphx_decision(target, adapter=None, **evidence):
    if adapter:
        adapter = dict(adapter, vendor=str(adapter.get("vendor", "")).lower())
    candidate = migraphx_candidate(target, adapter, **evidence)
    result = capability_decision(target, candidate, [adapter] if adapter else [])
    from packaging.utils import canonicalize_name

    artifacts = {canonicalize_name(a.name): a for a in candidate.artifacts}
    blockers = []
    if not adapter or not adapter.get("onnx_device_id"):
        blockers.append("Provider-specific MIGraphX device identity is unknown")
    for package in MIGRAPHX_PACKAGES:
        name, version = package.split("==")
        artifact = artifacts.get(canonicalize_name(name))
        if artifact is not None and artifact.version != version:
            blockers.append(f"{name} must use the exact recipe version {version}")
        if artifact is not None and (
            len(artifact.sha256) != 64
            or any(c not in "0123456789abcdef" for c in artifact.sha256.lower())
        ):
            blockers.append(f"{name} has no verified artifact SHA-256")
    if blockers:
        result.update(
            eligible=False,
            usable=False,
            auto_eligible=False,
            state="unavailable",
            reasons=result["reasons"] + tuple(blockers),
        )
    return result


def hip_decision(target, adapter=None, *, artifacts=(), native_payload=None):
    # Evaluation only. The released cp314 ORT bindings and supplied SDK targets
    # cannot certify RF's cp312/gfx1100 worker. No source build is offered.
    adapter = adapter or {}
    candidate = RuntimeCandidate(
        backend="HIP-EP",
        operation="onnx",
        systems=("Windows", "Linux"),
        platform="hip-ep-v0.5.2",
        recipe_id="onnx-hip-ep-v0.5.2-evaluation",
        adapter_id=adapter.get("adapter_id"),
        vendor="amd",
        opt_in=True,
        hardware_supported=adapter.get("gfx") == "gfx1151" if adapter.get("gfx") else None,
        artifacts=tuple(artifacts),
        required_packages=("onnxruntime",),
        native_payload=native_payload,
        closure_complete=None,
        python_validated=target.python_abi == "cp314",
    )
    result = capability_decision(target, candidate, [adapter] if adapter else [])
    blockers = list(result["reasons"])
    if target.python_abi != "cp314":
        blockers.append(
            "Released HIP-EP ORT bindings are cp314; this worker has " + target.python_abi
        )
    if adapter.get("gfx") == "gfx1100":
        blockers.append("Published HIP-EP SDK/kernel closure does not establish gfx1100 support")
    result.update(
        eligible=False,
        usable=False,
        auto_eligible=False,
        state="unavailable",
        reasons=tuple(blockers),
    )
    return result
