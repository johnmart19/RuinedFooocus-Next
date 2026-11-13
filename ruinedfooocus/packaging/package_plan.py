"""Pure base/overlay package admission and hash-locked execution.

Resolution is a separate, explicit pip dry run. A PackagePlan contains the full
closure, including already installed packages, so execution never resolves an
RF overlay independently of the publisher base. No compute module is imported.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name

from ruinedfooocus.packaging.runtime_support import (
    RuntimeArtifact,
    RuntimeCandidate,
    RuntimeTarget,
    capability_decision,
)


@dataclass(frozen=True)
class PackageLayer:
    identity: str
    requirements: tuple[str, ...]
    source: str
    revision: str
    license: str
    digest: str
    artifact_index: str = ""
    artifact_sources: tuple[str, ...] = ()
    allow_prereleases: bool = False


@dataclass(frozen=True)
class PackagePlan:
    target_json: str
    layers: tuple[PackageLayer, ...]
    artifacts: tuple[RuntimeArtifact, ...]
    environment: str
    recipe: str
    exceptions: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    restart_required: bool = False
    decision_json: str = "{}"

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def ready(self):
        return not self.conflicts


def require_core_torch_compatibility(version, *, api_floor="2.7", exceptions=()):
    """Check the core API floor even when pip requirements are already satisfied.

    Installed metadata alone cannot qualify an older native API. An exception
    must be supplied explicitly by the independently qualified recipe producer.
    """
    from packaging.version import Version

    if not version:
        raise RuntimeError(
            "Pinned ComfyUI requires an installed Torch runtime; no packages were changed"
        )
    if (
        api_floor
        and Version(version) < Version(api_floor)
        and f"core-torch-floor:{version}" not in exceptions
    ):
        raise RuntimeError(
            f"Core API requires Torch >= {api_floor}; {version} needs an independently qualified legacy exception. "
            "Select a compatible runtime/core environment before starting; no packages were changed."
        )


def current_target():
    """Capture this interpreter; do not substitute host tags for a WSL guest."""
    return RuntimeTarget(
        platform.system(),
        platform.machine(),
        platform.python_version(),
        sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag,
        frozenset(str(tag) for tag in sys_tags()),
        implementation=sys.implementation.name,
        environment="wsl" if "microsoft" in platform.release().lower() else "native",
        marker_environment=default_environment(),
    )


def target_identity(target):
    values = asdict(target)
    values["wheel_tags"] = sorted(values["wheel_tags"])
    return json.dumps(values, sort_keys=True)


def read_layer(path, *, identity, source, revision, license):
    data = Path(path).read_bytes()
    requirements = []
    for line in data.decode("utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith("#"):
            # Nested requirements/options would hide a second source of policy.
            Requirement(line)
            requirements.append(line)
    return PackageLayer(
        identity, tuple(requirements), source, revision, license, hashlib.sha256(data).hexdigest()
    )


def plan_packages(
    target,
    decision,
    pinned_base,
    rf_overlay,
    artifacts,
    *,
    environment,
    recipe,
    exceptions=(),
    api_floor="2.7",
):
    """Validate a resolved combined closure. Unknown evidence cannot be installed.

    ``decision`` is the shared capability result for the selected hardware route;
    it must admit acquisition, not merely enumerate an installed package name.
    Package metadata is independently checked even after that decision succeeds.
    """
    layers = tuple(pinned_base) + tuple(rf_overlay)
    artifacts = tuple(artifacts)
    conflicts = []
    if not decision.get("eligible"):
        conflicts.extend(decision.get("reasons") or ("Runtime capability has not been admitted",))
    by_name = {canonicalize_name(item.name): item for item in artifacts}
    markers = dict(target.marker_environment)
    markers.setdefault("python_version", ".".join(target.python_version.split(".")[:2]))
    markers.setdefault("python_full_version", target.python_version)
    # Validate the root requirements as well as every transitive dependency.
    for layer in layers:
        if not all((layer.source, layer.revision, layer.license, layer.digest)):
            conflicts.append(f"Provenance is incomplete for {layer.identity}")
        for raw in layer.requirements:
            requirement = Requirement(raw)
            from ruinedfooocus.packaging.runtime_support import _marker_variables

            if requirement.marker:
                missing = _marker_variables(requirement.marker) - markers.keys() - {"extra"}
                if missing:
                    conflicts.append(f"{layer.identity}: marker metadata missing {sorted(missing)}")
                    continue
                if not requirement.marker.evaluate(markers | {"extra": ""}):
                    continue
            artifact = by_name.get(canonicalize_name(requirement.name))
            if artifact is None or not requirement.specifier.contains(
                artifact.version, prereleases=True
            ):
                conflicts.append(
                    f"{layer.identity} requires {requirement}; combined closure differs"
                )
            elif requirement.url and artifact.source != requirement.url:
                conflicts.append(
                    f"{layer.identity} requires a different URL for {requirement.name}"
                )
            elif not requirement.extras.issubset(artifact.extras):
                conflicts.append(f"{layer.identity} has unresolved extras for {requirement.name}")
            if artifact and layer.artifact_index:
                from urllib.parse import urlsplit

                expected = urlsplit(layer.artifact_index)
                actual = urlsplit(artifact.source)
                extra_source = any(
                    artifact.source == source
                    or artifact.source.startswith(source.rstrip("/") + "/")
                    for source in layer.artifact_sources
                )
                if not extra_source and (
                    actual.scheme != expected.scheme
                    or actual.netloc != expected.netloc
                    or not actual.path.startswith(expected.path.rstrip("/") + "/")
                ):
                    conflicts.append(
                        f"{layer.identity}: {artifact.name} is from the wrong vendor index"
                    )
    torch = by_name.get("torch")
    if torch:
        try:
            require_core_torch_compatibility(
                torch.version, api_floor=api_floor, exceptions=exceptions
            )
        except RuntimeError as error:
            conflicts.append(str(error))
    closure = capability_decision(
        target,
        RuntimeCandidate(
            backend="cpu",
            operation="packages",
            systems=(target.system,),
            machines=(target.machine,),
            environments=(target.environment,),
            device_kind="cpu",
            artifacts=artifacts,
            closure_complete=True,
            native_payload=True,
            python_validated=True,
        ),
    )
    if not closure["eligible"]:
        conflicts.extend(closure["reasons"])
    for item in artifacts:
        if len(item.sha256) != 64 or any(char not in "0123456789abcdef" for char in item.sha256):
            conflicts.append(f"Missing SHA256 artifact identity for {item.name}")
    return PackagePlan(
        target_identity(target),
        layers,
        artifacts,
        str(Path(environment).resolve()),
        recipe,
        tuple(exceptions),
        tuple(dict.fromkeys(conflicts)),
        decision_json=json.dumps(decision, sort_keys=True),
    )


def artifacts_from_report(report, requirements):
    """Convert a pip --ignore-installed report to immutable wheel identities."""
    extras = {}
    markers = default_environment()
    for raw in requirements:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate(markers | {"extra": ""}):
            continue
        extras.setdefault(canonicalize_name(requirement.name), set()).update(requirement.extras)
    entries = report.get("install", ())
    # Extras requested transitively must be retained for closure marker checks.
    changed = True
    while changed:
        changed = False
        for entry in entries:
            parent = canonicalize_name(entry["metadata"]["name"])
            for raw in entry["metadata"].get("requires_dist", ()):
                requirement = Requirement(raw)
                if requirement.marker and not any(
                    requirement.marker.evaluate(markers | {"extra": extra})
                    for extra in ("", *extras.get(parent, ()))
                ):
                    continue
                requested = extras.setdefault(canonicalize_name(requirement.name), set())
                previous = len(requested)
                requested.update(requirement.extras)
                changed |= len(requested) != previous
    artifacts = []
    for entry in entries:
        metadata, download = entry["metadata"], entry["download_info"]
        url = download["url"]
        artifacts.append(
            RuntimeArtifact(
                metadata["name"],
                metadata["version"],
                unquote(urlsplit(url).path).rsplit("/", 1)[-1],
                True,
                metadata.get("requires_python", ""),
                tuple(metadata.get("requires_dist", ())),
                url,
                download.get("archive_info", {}).get("hashes", {}).get("sha256", ""),
                tuple(sorted(extras.get(canonicalize_name(metadata["name"]), ()))),
            )
        )
    return tuple(artifacts)


def resolve_packages(
    layers, *, index="https://pypi.org/simple", extra_indexes=(), constraints=(), runner=None
):
    """One wheel-only resolution, without changing the target environment."""
    runner = subprocess.run if runner is None else runner
    requirements = tuple(raw for layer in layers for raw in layer.requirements)
    # pip 26.3 can reject an indexed pin plus a same-version constraint when
    # its explicit local wheel appears later. Establish direct sources first;
    # retain every indexed requirement and constraint for normal conflict checks.
    direct = tuple(raw for raw in requirements if Requirement(raw).url)
    indexed = tuple(raw for raw in requirements if not Requirement(raw).url)
    with TemporaryDirectory(prefix="rf-package-plan-") as temporary:
        path, report = Path(temporary) / "requirements.txt", Path(temporary) / "report.json"
        path.write_text("\n".join((*direct, *indexed)) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--only-binary=:all:",
            "--report",
            str(report),
            "-r",
            str(path),
            "--index-url",
            index,
        ]
        if any(layer.allow_prereleases for layer in layers):
            command.append("--pre")
        if constraints:
            constraints_file = Path(temporary) / "constraints.txt"
            constraints_file.write_text("\n".join(constraints) + "\n", encoding="utf-8")
            command += ["--constraint", str(constraints_file)]
        for extra in extra_indexes:
            command += ["--extra-index-url", extra]
        runner(command, check=True)
        return artifacts_from_report(json.loads(report.read_text(encoding="utf-8")), requirements)


def execute_plan(plan, digest, *, runner=None, force=False, interpreter=None):
    """Acquire hash-locked wheels before mutation, then install the same closure.

    Main-environment pip installation cannot promise rollback. Acquisition and
    failed validation leave it intact; a pip failure requires restart/repair and
    is reported as such. Imported native packages cannot be replaced in-process.
    """
    from ruinedfooocus.packaging.install_lock import installation_lock

    runner = subprocess.run if runner is None else runner
    interpreter = str(interpreter or sys.executable)
    main_environment = Path(plan.environment) == Path(sys.prefix).resolve()
    if digest != plan.digest or not plan.ready:
        raise RuntimeError("Package plan is changed or unresolved: " + "; ".join(plan.conflicts))
    if plan.target_json != target_identity(current_target()):
        raise RuntimeError(
            "Package plan belongs to a different interpreter/environment; resolve again"
        )
    if not main_environment:
        result = runner(
            [
                interpreter,
                "-c",
                "import json,sys,sysconfig,platform; print(json.dumps([sys.prefix,platform.python_version(),sysconfig.get_config_var('SOABI') or sys.implementation.cache_tag,platform.machine(),sys.implementation.name]))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        prefix, version, abi, machine, implementation = json.loads(result.stdout)
        target = json.loads(plan.target_json)
        if Path(prefix).resolve() != Path(plan.environment) or (
            version,
            abi,
            machine,
            implementation,
        ) != (
            target["python_version"],
            target["python_abi"],
            target["machine"],
            target["implementation"],
        ):
            raise RuntimeError(
                "Optional interpreter ABI/environment differs from the frozen package plan"
            )
    native_names = {"torch", "xllamacpp", "onnxruntime", "torch-directml", "numpy"}

    def check_imports():
        if not main_environment:
            return
        for artifact in plan.artifacts:
            name = canonicalize_name(artifact.name)
            vendor_library = (
                name.startswith(("rocm-sdk-", "nvidia-", "pytorch-triton", "triton"))
                or name == "rocm"
            )
            if (
                name in native_names
                and name.replace("-", "_") in sys.modules
                or vendor_library
                and "torch" in sys.modules
            ):
                try:
                    same = importlib.metadata.version(artifact.name) == artifact.version
                except importlib.metadata.PackageNotFoundError:
                    same = False
                if force or not same:
                    raise RuntimeError(
                        "Native runtime is imported; restart before applying the package plan"
                    )

    check_imports()
    with (
        installation_lock(plan.environment),
        TemporaryDirectory(prefix="rf-packages-") as temporary,
    ):
        root = Path(temporary)
        locked = root / "locked.txt"
        locked.write_text(
            "\n".join(
                f"{item.name} @ {item.source} --hash=sha256:{item.sha256}"
                for item in plan.artifacts
            )
            + "\n",
            encoding="utf-8",
        )
        wheels = root / "wheels"
        runner(
            [
                interpreter,
                "-m",
                "pip",
                "download",
                "--require-hashes",
                "--no-deps",
                "--only-binary=:all:",
                "--dest",
                str(wheels),
                "-r",
                str(locked),
            ],
            check=True,
        )
        # Use local artifact paths so installation cannot consult a mutable index.
        local = root / "local.txt"
        local.write_text(
            "\n".join(
                f"{item.name} @ {(wheels / item.filename).as_uri()} --hash=sha256:{item.sha256}"
                for item in plan.artifacts
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            interpreter,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--require-hashes",
            "-r",
            str(local),
        ]
        if force:
            command.append("--force-reinstall")
        check_imports()
        try:
            runner(command, check=True)
            runner([interpreter, "-m", "pip", "check"], check=True)
            if main_environment:
                for artifact in plan.artifacts:
                    if importlib.metadata.version(artifact.name) != artifact.version:
                        raise RuntimeError(
                            f"Installed identity differs from the frozen plan: {artifact.name}"
                        )
            else:
                expected = {item.name: item.version for item in plan.artifacts}
                runner(
                    [
                        interpreter,
                        "-c",
                        "import importlib.metadata as m,json; expected=json.loads("
                        + repr(json.dumps(expected))
                        + "); assert all(m.version(name)==version for name,version in expected.items()), 'Installed identities differ from package plan'",
                    ],
                    check=True,
                )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Package application failed; restart and repair this environment. "
                "Automatic in-place rollback is unavailable."
            ) from error
    return replace(plan, restart_required=True)


def save_plan(plan, path):
    """Retain reviewable provenance and exact artifacts, without re-resolution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"digest": plan.digest, "plan": asdict(plan)}, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def preserved_torch_commands(installed):
    """Keep exact installed vendor companions when only adding RF dependencies."""
    commands = []
    versions = installed.get("versions", {})
    for name in (
        "torch",
        "torchvision",
        *sorted(set(versions) - {"torch", "torchvision", "transformers"}),
    ):
        version = versions.get(name)
        if version is None:
            raise RuntimeError(f"Installed vendor stack has no {name} companion")
        local = version.partition("+")[2]
        if local.startswith("rocm") or name.startswith("rocm-sdk-") or name == "rocm":
            index = "https://stable.repo.amd.com/rocm/whl-next/"
        elif local.startswith("cu") or local in ("cpu", "xpu"):
            index = "https://download.pytorch.org/whl/" + local
        else:
            index = "https://pypi.org/simple"
        commands.append([f"{name}=={version}", "--index-url", index])
    return commands


def bootstrap_plan(
    root,
    core_revision,
    recipe,
    commands,
    *,
    candidate,
    adapters=(),
    installed=None,
    api=False,
    admission_evidence=(),
):
    """Resolve the pinned embedded core and classified RF additions together.

    The caller has already selected/validated hardware using the shared runtime
    policy. This function additionally proves exact Python/package admission.
    No legacy API-floor exception is granted implicitly.
    """
    root = Path(root)
    core = root / "repositories/ComfyUI"
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
    if actual != core_revision:
        raise RuntimeError(
            "ComfyUI checkout differs from the pinned package base; update it before installation"
        )
    base_data = subprocess.check_output(
        ["git", "show", f"{core_revision}:requirements.txt"], cwd=core
    )
    base_path = core / "requirements.txt"
    if base_path.read_bytes().replace(b"\r\n", b"\n") != base_data.replace(b"\r\n", b"\n"):
        raise RuntimeError(
            "ComfyUI requirements have local changes; package planning preserves them and stops"
        )
    base = read_layer(
        base_path,
        identity="comfy-core",
        source="https://github.com/Comfy-Org/ComfyUI",
        revision=core_revision,
        license="GPL-3.0",
    )
    overlay = [
        read_layer(
            root / "pip/modules.txt",
            identity="rf-overlay",
            source="RuinedFooocus/pip/modules.txt",
            revision="sha256-content",
            license="GPL-3.0",
        )
    ]
    if api:
        overlay.append(
            read_layer(
                root / "requirements_api_versions.txt",
                identity="rf-api",
                source="RuinedFooocus/requirements_api_versions.txt",
                revision="sha256-content",
                license="GPL-3.0",
            )
        )
    recipe_layers, indexes = [], []
    for number, command in enumerate(commands):
        index = (
            command[command.index("--index-url") + 1]
            if "--index-url" in command
            else "https://pypi.org/simple"
        )
        indexes.append(index)
        roots = []
        skip = False
        for part in command:
            if skip:
                skip = False
                continue
            if part in ("--index-url", "--extra-index-url"):
                skip = True
            elif not part.startswith("--"):
                Requirement(part)
                roots.append(part)
        prereleases = "--pre" in command
        digest = hashlib.sha256(
            json.dumps(
                {"requirements": roots, "allow_prereleases": prereleases}, sort_keys=True
            ).encode()
        ).hexdigest()
        recipe_layers.append(
            PackageLayer(
                f"torch-recipe:{recipe}:{number}",
                tuple(roots),
                index,
                recipe,
                "publisher-distribution",
                digest,
                index,
                ("https://files.pythonhosted.org/packages",)
                if index == "https://pypi.org/simple"
                else ("https://stable.repo.amd.com/rocm/pytorch/whl-next",)
                if index.rstrip("/") == "https://stable.repo.amd.com/rocm/whl-next"
                else ("https://download-r2.pytorch.org" + urlsplit(index).path.rstrip("/"),)
                if urlsplit(index).scheme == "https"
                and urlsplit(index).netloc == "download.pytorch.org"
                and urlsplit(index).path.startswith("/whl/")
                else (),
                allow_prereleases=prereleases,
            )
        )
    if installed:
        roots = tuple(
            f"{name}=={version}" for name, version in installed.get("versions", {}).items()
        )
        overlay.append(
            PackageLayer(
                "preserved-vendor-constraints",
                roots,
                "installed-manifest",
                "exact-versions",
                "publisher-distribution",
                hashlib.sha256("\n".join(roots).encode()).hexdigest(),
            )
        )
    if recipe.startswith("rocm"):
        from ruinedfooocus.packaging.rocm_installer import rocm_metadata_layer
        from ruinedfooocus.packaging.runtime_bundles import bundle_for

        version = (
            installed["versions"]["rocm"]
            if installed and "rocm" in installed.get("versions", {})
            else bundle_for(recipe).torch.partition("+rocm")[2]
        )
        overlay.append(rocm_metadata_layer(version))
        selector_source = Requirement(overlay[-1].requirements[0]).url
        recipe_layers = [
            replace(item, artifact_sources=(*item.artifact_sources, selector_source))
            if any(canonicalize_name(Requirement(raw).name) == "rocm" for raw in item.requirements)
            else item
            for item in recipe_layers
        ]
    layers = (base, *recipe_layers, *overlay)
    artifacts = resolve_packages(
        layers,
        index=indexes[0] if indexes else "https://pypi.org/simple",
        extra_indexes=tuple(dict.fromkeys((*indexes[1:], "https://pypi.org/simple"))),
    )
    target = current_target()
    decision = capability_decision(
        target, replace(candidate, artifacts=artifacts, closure_complete=True), adapters
    )
    decision["observed_adapter_ids"] = tuple(item["adapter_id"] for item in adapters)
    decision["adapter_candidates"] = tuple(admission_evidence)
    result = plan_packages(
        target,
        decision,
        (base, *recipe_layers),
        overlay,
        artifacts,
        environment=sys.prefix,
        recipe=recipe,
    )
    if not result.ready:
        raise RuntimeError(
            "Combined ComfyUI/RF package plan is unavailable: " + "; ".join(result.conflicts)
        )
    return result
