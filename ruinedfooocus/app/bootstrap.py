import os
import sys
import subprocess
import version
import warnings
import re
from pathlib import Path
from tempfile import gettempdir
from ruinedfooocus.packaging.rocm_installer import ROCM_PLATFORM, automatic_rocm_plan
from ruinedfooocus.packaging.runtime_support import (
    inspect_installed_rocm,
    inspect_installed_cuda,
    inspect_installed_cpu,
    installed_cuda_platform,
    select_torch_platform,
)
from ruinedfooocus.packaging.gpu_installer import (
    preserve_installed_rocm,
    preserve_installed_cuda,
    installed_torch_constraints,
    torch_install_commands,
    selected_llm_runtime,
    saved_xllamacpp_backend,
    ensure_xllamacpp,
)
from ruinedfooocus.backends.comfy.comfy_compat import (
    configure_device_environment,
    configure_torch_allocator,
)

# Redirected output uses the console codepage (e.g. cp1251); model text or file
# names it cannot encode must not crash the operation that logs them.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated", module=r"torch(\..*)?"
)

from ruinedfooocus.packaging.launch_util import (
    is_installed,
    run,
    python,
    repo_dir,
    requirements_met,
    script_path,
    dir_repos,
)


git_repos = [
    {
        "name": "ComfyUI",
        "path": "ComfyUI",
        "url": "https://github.com/comfy-org/ComfyUI",
        "hash": "989d93d0d968e704c589ce0146bb18c77f6b50cc",
        "add_path": "ComfyUI",
    },
    {
        "name": "molbal/ComfyUI-GGUF",
        "path": "molbal_comfyui_gguf",
        "url": "https://github.com/molbal/ComfyUI-GGUF",
        "hash": "da5f435cd9a9e54ad27eed417ed72b1f2543ef41",
        "add_path": "",
    },
]


def installed_xpu_version(reinstall=False):
    """The installed torch version when it is the pinned Intel XPU build, else None."""
    import importlib.metadata

    from ruinedfooocus.packaging.runtime_bundles import bundle_for

    try:
        installed = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    keep = not reinstall and installed == f"{bundle_for('xpu').torch}+xpu"
    return installed if keep else None


def report_xpu_device(os_platform):
    """Warn (never fail) when the XPU build cannot see an Intel GPU: usually a driver gap."""
    result = subprocess.run(
        [python, "-c", "import torch; print('RF_XPU=' + str(int(torch.xpu.is_available())))"],
        capture_output=True,
        text=True,
    )
    if "RF_XPU=1" in result.stdout:
        print("Intel XPU device verified")
        return
    from ruinedfooocus.packaging.hardware_inventory import is_wsl

    hint = (
        "Install the Intel GPU driver for WSL on Windows and the Intel compute runtime "
        "(Level Zero) inside WSL."
        if is_wsl()
        else "Install a current Intel graphics driver"
        + (" and the Intel compute runtime (Level Zero)." if os_platform == "Linux" else ".")
    )
    print(f"WARNING: Intel XPU PyTorch is installed but no Intel GPU is available. {hint}")


def prepare_environment(skip_install=False):
    print(f"Python {sys.version}")
    print(f"RuinedFooocus version: {version.version}")

    requirements_file = "requirements_versions.txt"

    modules_file = "pip/modules.txt"

    if skip_install:
        import importlib.metadata
        from ruinedfooocus.packaging.package_plan import require_core_torch_compatibility

        try:
            installed_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            installed_version = None
        require_core_torch_compatibility(installed_version)
        print("Skip pip check.")
    else:
        from ruinedfooocus.core.dependency_cache import needed, record

        maintenance_needed = needed(script_path, "maintenance", force=REINSTALL_ALL)
        if maintenance_needed:
            run(
                f'"{python}" -m pip install --quiet --upgrade pip',
                "Check pip",
                "Couldn't check pip",
                live=False,
            )
        if not getattr(sys, "_rf_bootstrap_checked", False) and needed(
            script_path, "bootstrap", force=REINSTALL_ALL
        ):
            run(
                f'"{python}" -m pip install --quiet {"--force-reinstall " if REINSTALL_ALL else ""}-r "{requirements_file}"',
                "Check pre-requirements",
                "Couldn't check pre-reqs",
                live=False,
            )
        if maintenance_needed:
            run(
                f'"{python}" -m pip uninstall --quiet -y llama-cpp-python',
                "Check for old modules",
                "Couldn't check old modules",
                live=False,
            )
        if maintenance_needed:
            record(script_path, "maintenance")
        if not getattr(sys, "_rf_bootstrap_checked", False):
            record(script_path, "bootstrap")

        import torchruntime
        import platform

        os_platform = platform.system()
        from ruinedfooocus.packaging.backend_preferences import apply_saved_image_backend

        apply_saved_image_backend(args, os_platform, reinstall=REINSTALL_ALL or REINSTALL_TORCH)
        if args.cuda_nightly:
            from ruinedfooocus.packaging.cuda_selection import select_cuda_nightly

            os.environ["TORCH_PLATFORM"] = select_cuda_nightly(
                os_platform, os.environ.get("TORCH_PLATFORM")
            )
        installed_rocm = inspect_installed_rocm(
            os_platform, repair=REINSTALL_ALL or REINSTALL_TORCH or args.rocm10
        )
        installed_cuda = inspect_installed_cuda(os_platform)
        installed_cpu = inspect_installed_cpu() if args.cpu else None
        keep_cpu = bool(installed_cpu and not (REINSTALL_ALL or REINSTALL_TORCH))
        torch_platform = select_torch_platform(
            torchruntime, os_platform, installed_rocm, installed_cuda
        )
        keep_cuda = preserve_installed_cuda(
            installed_cuda,
            torch_platform,
            REINSTALL_ALL or REINSTALL_TORCH or args.rocm10 or args.cuda_nightly,
        )
        keep_rocm = preserve_installed_rocm(
            installed_rocm,
            torch_platform,
            REINSTALL_ALL or REINSTALL_TORCH or args.rocm10,
        )

        rocm_plan = None
        if args.rocm10:
            rocm_plan = automatic_rocm_plan(os_platform, ROCM_PLATFORM, force=True)
            if rocm_plan is None:
                raise RuntimeError("--rocm10 requires a supported AMD GPU on Windows or Linux/WSL.")
            torch_platform = rocm_plan.platform
            os.environ["TORCH_PLATFORM"] = torch_platform
        elif not keep_rocm and not TORCH_FROZEN:
            rocm_plan = automatic_rocm_plan(os_platform, torch_platform)
            if rocm_plan:
                torch_platform = rocm_plan.platform

        # Native image fallback is independent of Chat and optional image engines.
        if torch_platform == "cpu":
            if not args.cpu:
                from ruinedfooocus.core.device_fallback import warn_cpu_fallback

                warn_cpu_fallback(
                    "ComfyUI image processing", "no compatible PyTorch GPU backend was detected"
                )
            args.cpu = True
        elif torch_platform == "directml" and args.directml is None:
            args.directml = -1
        os.environ["TORCH_PLATFORM"] = torch_platform
        configure_torch_allocator(torch_platform, os_platform, args)

        print(f"Torch platform: {os_platform}: {torch_platform}")

    if skip_install:
        print("Skip check of required modules.")
    else:
        os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

        # Resolve the publisher base and RF/API additions before changing Torch.
        keep_xpu = torch_platform == "xpu" and installed_xpu_version(
            REINSTALL_ALL or REINSTALL_TORCH
        )
        installed_torch = installed_torch_constraints()
        preserve = bool(
            installed_torch
            and installed_torch.get("versions", {}).get("torch")
            and (
                TORCH_FROZEN
                or keep_cpu
                or keep_xpu
                or keep_rocm
                or keep_cuda
                or args.cpu
                and not (REINSTALL_ALL or REINSTALL_TORCH)
            )
        )
        from ruinedfooocus.packaging.package_plan import (
            bootstrap_plan,
            execute_plan,
            preserved_torch_commands,
            save_plan,
            require_core_torch_compatibility,
        )
        from ruinedfooocus.packaging.runtime_support import RuntimeCandidate
        from ruinedfooocus.packaging.runtime_bundles import python_bundle_supported
        from ruinedfooocus.packaging.hardware_inventory import gpu_inventory

        core_revision = next(repo["hash"] for repo in git_repos if repo["name"] == "ComfyUI")
        clone_git_repos(offline=False)
        base_file = Path(repo_dir("ComfyUI")) / "requirements.txt"
        if preserve:
            require_core_torch_compatibility(installed_torch["versions"]["torch"])
        needs_packages = (
            REINSTALL_ALL
            or REINSTALL_TORCH
            or not preserve
            or not requirements_met(base_file)
            or not requirements_met(modules_file)
            or args.api
            and not requirements_met("requirements_api_versions.txt")
        )
        if needs_packages:
            commands = (
                preserved_torch_commands(installed_torch)
                if preserve
                else [rocm_plan.args]
                if rocm_plan
                else torch_install_commands(
                    torchruntime, torch_platform, os_platform, nightly=args.cuda_nightly
                )
            )
            cpu_route = torch_platform == "cpu" or preserve
            adapters = [] if cpu_route else gpu_inventory()["adapters"]
            vendor = (
                "nvidia"
                if torch_platform.startswith("cu")
                else "amd"
                if torch_platform.startswith("rocm")
                else "intel"
                if torch_platform == "xpu"
                else None
            )
            matching = [item for item in adapters if not vendor or item.get("vendor") == vendor]
            if not preserve and os_platform == "Linux" and torch_platform in ("rocm10.1", "xpu"):
                from ruinedfooocus.packaging.vendor_admission import linux_image_adapters

                targets = (
                    tuple(re.findall(r"device-(gfx[0-9a-f]+)", " ".join(rocm_plan.args)))
                    if rocm_plan
                    else ()
                )
                matching = linux_image_adapters(matching, amd_targets=tuple(dict.fromkeys(targets)))
                adapters = [item for item in adapters if item.get("vendor") != vendor] + matching
            # CUDA's existing producer validates every selected architecture and
            # driver. AMD/Intel acquisition without equivalent evidence remains
            # unknown; an inventory vendor name cannot establish compatibility.
            hardware = driver = None
            evidence = ()
            selected_adapter = matching[0] if matching else None
            if not preserve and torch_platform.startswith("cu"):
                from ruinedfooocus.packaging.cuda_selection import validate_cuda_runtime

                validate_cuda_runtime(torch_platform, nightly=args.cuda_nightly)
                hardware = driver = True
            elif (
                not preserve and os_platform == "Windows" and torch_platform in ("rocm10.1", "xpu")
            ):
                from ruinedfooocus.packaging.vendor_admission import (
                    combined_evidence,
                    windows_driver_inventory,
                    windows_recipe_evidence,
                )

                evidence = windows_recipe_evidence(
                    torch_platform, matching, windows_driver_inventory()
                )
            elif not preserve and os_platform == "Linux" and torch_platform in ("rocm10.1", "xpu"):
                from ruinedfooocus.packaging.vendor_admission import (
                    combined_evidence,
                    guest_device_inventory,
                    linux_platform_inventory,
                    linux_recipe_evidence,
                    windows_driver_inventory,
                )
                from ruinedfooocus.packaging.hardware_inventory import is_wsl

                guest = is_wsl()
                if (
                    guest
                    and torch_platform == "rocm10.1"
                    and not (
                        args.rocm10 or getattr(args, "_user_torch_platform", None) == "rocm10.1"
                    )
                ):
                    raise RuntimeError(
                        "ROCm 10.1 WSL is a technical preview. Select --rocm10 or TORCH_PLATFORM=rocm10.1 explicitly after reviewing its driver/guest requirements."
                    )
                visible = guest_device_inventory(matching) if guest else None
                if guest:
                    matching = [
                        dict(item, guest_available=(visible or {}).get(item["adapter_id"]))
                        for item in matching
                    ]
                    adapters = [
                        item for item in adapters if item.get("vendor") != vendor
                    ] + matching
                evidence = linux_recipe_evidence(
                    torch_platform,
                    matching,
                    linux_platform_inventory(),
                    wsl=guest,
                    host_inventory=windows_driver_inventory() if guest else None,
                    guest_visible=visible,
                )
            if evidence:
                from ruinedfooocus.packaging.vendor_admission import select_installer_adapter

                selected_adapter, hardware, driver = select_installer_adapter(matching, evidence)
                if selected_adapter and not any(
                    item["adapter_id"] == selected_adapter["adapter_id"] for item in adapters
                ):
                    adapters.append(selected_adapter)
            candidate = RuntimeCandidate(
                backend="cpu" if preserve else torch_platform,
                operation="packages" if preserve else "image",
                systems=(os_platform,),
                machines=(platform.machine(),),
                environments=("native", "wsl"),
                adapter_id=selected_adapter["adapter_id"] if selected_adapter else None,
                device_kind="cpu" if cpu_route else "gpu",
                vendor=vendor,
                hardware_supported=hardware,
                driver_supported=driver,
                required_packages=("torch", "torchvision"),
                native_payload=True,
                python_validated=python_bundle_supported(torch_platform),
                recipe_id=torch_platform,
            )
            plan = bootstrap_plan(
                script_path,
                core_revision,
                torch_platform,
                commands,
                candidate=candidate,
                adapters=adapters,
                installed=installed_torch if preserve else None,
                api=args.api,
                admission_evidence=evidence,
            )
            from ruinedfooocus.core.paths import data_path

            save_plan(plan, data_path("cache", "package-plans", plan.digest + ".json"))
            execute_plan(plan, plan.digest, force=REINSTALL_ALL or REINSTALL_TORCH)
            installed_torch = installed_torch_constraints()
            if torch_platform.startswith("cu"):
                actual = inspect_installed_cuda(os_platform)
                if not actual or installed_cuda_platform(actual) != torch_platform:
                    raise RuntimeError(
                        "CUDA package plan installed; native validation failed. Restart/repair required."
                    )
            if torch_platform.startswith("rocm") and not inspect_installed_rocm(os_platform):
                raise RuntimeError(
                    "ROCm package plan installed; native validation failed. Restart/repair required."
                )
            torchruntime.configure()

        if selected_llm_runtime(args.settings) == "xllamacpp":
            from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary

            ensure_xllamacpp(
                torch_platform,
                os_platform,
                REINSTALL_ALL,
                installed_torch,
                choice="CPU"
                if force_cpu_for_auxiliary(args)
                else saved_xllamacpp_backend(args.settings),
            )

    if not skip_install:
        from ruinedfooocus.packaging.linux_gpu_prerequisites import ensure_gpu_jit_toolchain

        ensure_gpu_jit_toolchain(
            enabled=not args.cpu and torch_platform.startswith(("rocm", "cu", "xpu"))
        )

    if args.api:
        print("Check API dependencies")
        api_requirements = "requirements_api_versions.txt"
        if REINSTALL_ALL or not requirements_met(api_requirements):
            if skip_install:
                if not requirements_met(api_requirements):
                    raise RuntimeError(
                        "API dependencies are missing. Launch once without --offline/--skip-dependency-install to install them."
                    )
            elif not requirements_met(api_requirements):
                raise RuntimeError(
                    "Combined package plan did not satisfy the API dependency closure."
                )

    if not skip_install:
        # Package installation above can change the environment fingerprint.
        record(script_path, "bootstrap")
        record(script_path, "maintenance")


_checked_git_repos = set()


def clone_git_repos(offline=False):
    from ruinedfooocus.packaging.launch_util import git_clone

    for repo in git_repos:
        key = (repo["path"], repo["hash"])
        if not offline and key not in _checked_git_repos:
            # Both preparation and runtime path setup call this function. A
            # network failure must not spend the same deadline twice at launch.
            _checked_git_repos.add(key)
            git_clone(repo["url"], repo_dir(repo["path"]), repo["name"], hash=repo["hash"])
        add_path = str(Path(script_path) / dir_repos / repo["add_path"])
        if add_path not in sys.path:
            sys.path.append(add_path)


from ruinedfooocus.app.arguments import args
from ruinedfooocus.app.startup_port import configure_port
from ruinedfooocus.integrations.web_access import validate_web_access

validate_web_access(args)
configure_port(args)

# Warn if using python 3.10
import platform
import time

py_ver = platform.python_version().split(".")
if int(py_ver[0]) == 3 and int(py_ver[1]) <= 10:
    print("")
    print("WARNING: Python 3.10 is deprecated and support will be removed.")
    print("Install Python 3.11 or newer, delete the venv folder and start RuinedFooocus again.")
    print("See the Install section of readme.md.")
    print("")
    time.sleep(3)
    # input("Press enter to continue...")

from ruinedfooocus.core.paths import startup_marker
from ruinedfooocus.packaging.startup_options import skip_dependency_install

REINSTALL_ALL_PATH = startup_marker("reinstall")
REINSTALL_TORCH_PATH = startup_marker("reinstalltorch")
REINSTALL_ALL = REINSTALL_ALL_PATH.exists()
REINSTALL_TORCH = REINSTALL_TORCH_PATH.exists()
TORCH_FROZEN = startup_marker("freezetorch").exists() and not (REINSTALL_ALL or REINSTALL_TORCH)

configure_device_environment(args)

from ruinedfooocus.backends.comfy.image_compute import finalize_compute

finalize_compute(args)

offline = os.environ.get("RF_OFFLINE") == "1" or args.offline
skip_install = skip_dependency_install()
if args.cpu and args.directml is not None:
    raise RuntimeError("Choose --cpu or --directml, not both.")
if args.directml is not None and args.directml < -1:
    raise RuntimeError(
        "--directml expects a non-negative GPU index, or no value for the default GPU."
    )
if args.cpu or args.directml is not None:
    os.environ["TORCH_PLATFORM"] = "cpu" if args.cpu else "directml"
if os.environ.get("TORCH_PLATFORM") == "cpu":
    args.cpu = True
elif os.environ.get("TORCH_PLATFORM") == "directml" and args.directml is None:
    args.directml = -1

if args.rocm10 and (skip_install or args.cpu or args.directml is not None or TORCH_FROZEN):
    raise RuntimeError(
        "--rocm10 cannot be combined with offline/--skip-dependency-install, CPU, DirectML or freezetorch mode."
    )
if args.cuda_nightly and (
    skip_install or args.rocm10 or args.cpu or args.directml is not None or TORCH_FROZEN
):
    raise RuntimeError(
        "--cuda-nightly cannot be combined with offline/--skip-dependency-install, ROCm, CPU, DirectML or freezetorch mode."
    )

if args.legacy:
    if args.cuda_nightly or args.rocm10 or args.cpu or args.directml is not None:
        raise RuntimeError(
            "--legacy is a CUDA profile and cannot be combined with CPU, DirectML, ROCm or CUDA nightly."
        )
    from ruinedfooocus.packaging.cuda_selection import select_cuda_legacy
    import platform

    os.environ["TORCH_PLATFORM"] = select_cuda_legacy(
        platform.system(),
        os.environ.get("TORCH_PLATFORM"),
        install=not (skip_install or TORCH_FROZEN),
    )

if skip_install:
    print("Dependency installation disabled.")
    from ruinedfooocus.packaging.backend_preferences import apply_saved_image_backend

    apply_saved_image_backend(args, platform.system(), reinstall=False)
    if os.environ.get("TORCH_PLATFORM"):
        print(f"Torch platform: {platform.system()}: {os.environ['TORCH_PLATFORM']}")
    if REINSTALL_ALL or REINSTALL_TORCH:
        print(
            "Reinstall request pending: restart without --offline/--skip-dependency-install to apply it."
        )

prepare_environment(skip_install)
finalize_compute(args)

if not skip_install and REINSTALL_ALL:
    try:
        REINSTALL_ALL_PATH.unlink(missing_ok=True)
    except:
        print("ERROR: Failed to remove 'reinstall'. Pleae remove manually.")
if not skip_install and REINSTALL_TORCH:
    try:
        REINSTALL_TORCH_PATH.unlink(missing_ok=True)
    except:
        print("ERROR: Failed to remove 'reinstalltorch'. Pleae remove manually.")

try:
    clone_git_repos(offline)
except:
    print(f"WARNING: Failed checking git-repos. Trying to start without update.")

# Initialize native memory services before UI model discovery imports backend modules.
from ruinedfooocus.backends.comfy.comfy_compat import initialize_backend

initialize_backend(args)

from ruinedfooocus.core.paths import data_path

gradio_cache = str(data_path("cache", "uploads"))
os.environ["GRADIO_TEMP_DIR"] = gradio_cache
# Delete old data
import shutil

try:
    if args.clean_cache and Path(gradio_cache).is_dir() and not Path(gradio_cache).is_symlink():
        shutil.rmtree(gradio_cache)
except FileNotFoundError:
    pass
except PermissionError:
    pass
except Exception as e:
    print(f"An error occurred: {str(e)}")


def launch_ui():
    print("Starting webui...", flush=True)
    import ruinedfooocus.ui.app as webui


launch_ui()
