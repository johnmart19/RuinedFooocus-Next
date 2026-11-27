"""Settings > Hardware & backends: detected GPUs and one backend choice per component.

Choices are filtered to the detected hardware and operating system. Installing a
missing runtime is always an explicit button, never a side effect of choosing an
option; a changed image backend only takes effect after a restart.
"""

import platform
from pathlib import Path

import gradio as gr

from ruinedfooocus.app.state import add_setting, settings
from ruinedfooocus.core.paths import data_path, startup_marker
from ruinedfooocus.core.translation_manager import (
    localize,
    localized_output,
    resolve_locale,
    translate_ui as t,
)
from ruinedfooocus.packaging import hardware_inventory as inventory
from ruinedfooocus.packaging.backend_preferences import (
    cli_overrides,
    effective_backends,
    saved_image_backend,
)
from ruinedfooocus.packaging.startup_options import skip_dependency_install

# Product names (CUDA, ROCm, Vulkan, Windows ML) stay as written in every language.
IMAGE_LABELS = {
    "auto": t("Automatic (recommended)"),
    "cuda": t("CUDA (NVIDIA)"),
    "rocm": t("ROCm (AMD)"),
    "xpu": t("XPU (Intel)"),
    "cpu": t("CPU"),
}
LLAMA_LABELS = {
    "Auto": t("Automatic (recommended)"),
    "CPU": t("CPU"),
    "SYCL": t("SYCL (Intel)"),
}
BACKGROUND_LABELS = {
    "Auto": t("Automatic (recommended)"),
    "CPU": t("CPU"),
    "CUDA": t("CUDA (NVIDIA)"),
    "MIGraphX": t("MIGraphX (AMD)"),
    "OpenVINO": t("OpenVINO (Intel)"),
    "Windows GPU": t("Windows ML (CPU / GPU / NPU)"),
}
XLLAMA_LABELS = {"Auto": t("Automatic (recommended)"), "CPU": t("CPU")}
SUPPORT_HEADERS = ("Component", "Option", "Status", "Details")
# Words of the support table: built in English by support_matrix, localized per browser.
SUPPORT_TEXTS = (
    t("Component"),
    t("Option"),
    t("Status"),
    t("Details"),
    t("Installed"),
    t("Available"),
    t("Not installed"),
    t("Ready"),
    t("Main"),
    t("Set as main"),
    t("Install"),
    t("Install on restart"),
    t("Image generation"),
    t("Chat & prompt LLM"),
    t("Background removal"),
    t("Uses your detected GPU automatically"),
    t("Optional Vulkan GGUF image engine"),
    t("No supported backends detected."),
    t("Use CPU for everything is enabled — individual GPU selections are paused."),
    t("Active default: {summary}"),
    t("Not available"),
    t("Unknown"),
    t("Any installed build runs on CPU"),
    t("Windows only"),
    t("Linux and WSL only"),
    t("Native Linux only"),
    t("Not for this operating system"),
    t("Needs an NVIDIA GPU"),
    t("Needs an AMD GPU"),
    t("Needs an Intel GPU"),
    t("Needs a GPU"),
    t("Needs a Qualcomm Snapdragon PC"),
    t("Not supported on this PC"),
    t("No official build for this CPU"),
    t("Conflicts with ROCm 10 PyTorch"),
    t("Needs Mesa's Dozen Vulkan driver in WSL"),
)
KIND_NAMES = {"cuda": "CUDA", "rocm": "ROCm", "xpu": "XPU", "cpu": "CPU"}


def labelled(values, labels, saved):
    """Dropdown options; a saved choice this machine lacks stays visible instead of vanishing."""
    options = [(labels.get(value, value), value) for value in values]
    if saved not in values:
        options.append((saved, saved))
    return options


def locale_of(request):
    return resolve_locale(request=request)


def hardware_text(locale=None):
    system = f"{platform.system()} {platform.release()} ({platform.machine()})"
    if inventory.is_wsl():
        system += " · WSL"
    lines = [localize("**Operating system:** {system}", locale, system=system)]
    gpus = inventory.detect_gpus()
    if gpus:
        lines.append(localize("**Detected GPUs:**", locale))
        lines += [
            f"- {gpu['name']} ({inventory.VENDOR_LABELS.get(gpu['vendor'], gpu['vendor'])})"
            for gpu in gpus
        ]
    else:
        lines.append(
            localize(
                "**No GPU detected.** Image generation, Chat and background removal run on the CPU.",
                locale,
            )
        )
    return "\n\n".join(lines)


def support_table(locale=None, refresh=False):
    """The "Supported on this PC" table with headers, status and notes localized.

    Dataframe headers are plain strings (Gradio does not translate them), so the
    headers travel with the data in the browser's language.
    """
    from ruinedfooocus.packaging.support_matrix import support_rows

    gpus = inventory.detect_gpus(refresh=refresh)
    statuses = {
        "Installed (unverified)": "Installed",
        "Usable (workload unverified)": "Available",
        "Qualified": "Available",
    }
    rows = []
    for row in support_rows(gpus):
        note = "; ".join(
            part
            for part in row["note"].split("; ")
            if part != "Exact runtime/device eligibility has not been verified"
        )
        rows.append(
            [
                row["component"],
                row["option"],
                localize(statuses.get(row["status"], row["status"]), locale),
                localize(note, locale) if note else "",
            ]
        )
    return {
        "headers": [localize(header, locale) for header in SUPPORT_HEADERS],
        "data": rows,
    }


def _clean_note(note, locale=None):
    parts = [
        localize(part, locale)
        for part in (note or "").split("; ")
        if part and part != "Exact runtime/device eligibility has not been verified"
    ]
    return "; ".join(parts)


def support_matrix_payload(
    locale=None,
    refresh=False,
    include_unavailable=False,
    cpu_forced=False,
    image_choice="auto",
    image_engine_choice="auto",
    llm_runtime_choice="llama.cpp",
    llama_choice="Auto",
    xllama_choice="Auto",
    bg_choice="Auto",
    bg_provider_choice="Auto",
    winml_provider_choice="Auto",
):
    """Build the interactive 'Supported on this PC' payload without probing devices or installing."""
    from ruinedfooocus.packaging import support_matrix as matrix
    from ruinedfooocus.ui import optional_image_runtime as optional

    gpus = inventory.detect_gpus(refresh=refresh)
    system = platform.system()
    machine = platform.machine()
    os_name = "WSL" if system == "Linux" and inventory.is_wsl() else system
    installed_image = inventory.installed_runtimes(system)
    image_choices = set(inventory.image_backend_choices(gpus, system, installed_image, machine))
    image_kind = installed_image.get("kind")

    def _status_pair(raw_status):
        if raw_status in ("Installed", "Installed (unverified)"):
            return "installed", localize("Installed", locale)
        if raw_status in ("Available", "Usable (workload unverified)", "Qualified", "Installable"):
            return "available", localize("Not installed", locale)
        return "unavailable", localize("Not available", locale)

    hidden_unsupported = 0

    # 1. Image generation (ComfyUI / PyTorch + optional Vulkan engine)
    image_rows = []
    image_auto_main = image_engine_choice in ("auto", "comfy") and image_choice == "auto"
    image_rows.append(
        {
            "id": "image:auto",
            "label": localize("Automatic (recommended)", locale),
            "runtimeLabel": "ComfyUI / PyTorch",
            "detail": localize("Uses your detected GPU automatically", locale),
            "status": "installed",
            "statusLabel": localize("Ready", locale),
            "isMain": image_auto_main,
            "canSetMain": True,
            "canInstall": False,
            "requiresGpu": False,
        }
    )
    for value, label in matrix.IMAGE_LABELS.items():
        if value == "directml":
            continue
        supported = value in image_choices
        if not supported:
            if not include_unavailable:
                hidden_unsupported += 1
                continue
            note = matrix._unavailable_note(
                gpus,
                matrix.IMAGE_VENDORS[value],
                os_name,
                ("Windows", "Linux", "WSL"),
            )
            image_rows.append(
                {
                    "id": f"image:{value}",
                    "label": localize(label, locale),
                    "runtimeLabel": "ComfyUI / PyTorch",
                    "detail": localize(note, locale),
                    "status": "unavailable",
                    "statusLabel": localize("Not available", locale),
                    "isMain": False,
                    "canSetMain": False,
                    "canInstall": False,
                    "requiresGpu": True,
                }
            )
            continue
        is_installed = image_kind == value
        if value == "cpu":
            status_code = "installed"
            status_label = localize("Ready", locale)
            detail = (
                (installed_image.get("torch") or "")
                if is_installed
                else localize("Any installed build runs on CPU", locale)
            )
            can_install = False
        else:
            status_code = "installed" if is_installed else "available"
            status_label = (
                localize("Installed", locale) if is_installed else localize("Not installed", locale)
            )
            detail = (installed_image.get("torch") or "") if is_installed else ""
            can_install = not is_installed
        image_rows.append(
            {
                "id": f"image:{value}",
                "label": localize(label, locale),
                "runtimeLabel": "ComfyUI / PyTorch",
                "detail": detail,
                "status": status_code,
                "statusLabel": status_label,
                "isMain": image_engine_choice in ("auto", "comfy") and image_choice == value,
                "canSetMain": True,
                "canInstall": can_install,
                "installLabel": localize("Install on restart", locale),
                "requiresGpu": value != "cpu",
            }
        )

    if gpus or include_unavailable:
        try:
            opt_state = optional.activation.status()
        except Exception:
            opt_state = {}
        opt_ready = bool(opt_state.get("available") or opt_state.get("prepared"))
        if gpus:
            image_rows.append(
                {
                    "id": "image:sdcpp-vulkan",
                    "label": "Vulkan (GGUF)",
                    "runtimeLabel": "stable-diffusion.cpp",
                    "detail": localize("Optional Vulkan GGUF image engine", locale),
                    "status": "installed" if opt_ready else "available",
                    "statusLabel": (
                        localize("Installed", locale)
                        if opt_ready
                        else localize("Not installed", locale)
                    ),
                    "isMain": image_engine_choice == "sdcpp-vulkan",
                    "canSetMain": True,
                    "canInstall": not opt_ready,
                    "installLabel": localize("Install", locale),
                    "requiresGpu": True,
                }
            )
        elif include_unavailable:
            image_rows.append(
                {
                    "id": "image:sdcpp-vulkan",
                    "label": "Vulkan (GGUF)",
                    "runtimeLabel": "stable-diffusion.cpp",
                    "detail": localize("Needs a GPU", locale),
                    "status": "unavailable",
                    "statusLabel": localize("Not available", locale),
                    "isMain": False,
                    "canSetMain": False,
                    "canInstall": False,
                    "requiresGpu": True,
                }
            )

    if cpu_forced:
        image_summary_name = localize("CPU", locale)
    elif image_engine_choice == "sdcpp-vulkan":
        image_summary_name = "stable-diffusion.cpp · Vulkan (GGUF)"
    else:
        image_summary_name = localize(
            matrix.IMAGE_LABELS.get(image_choice, "Automatic (recommended)"), locale
        )

    # 2. Chat & prompt LLM (llama.cpp + xllamacpp unified!)
    chat_rows = []
    chat_rows.append(
        {
            "id": "llama:Auto",
            "label": localize("Automatic (recommended)", locale),
            "runtimeLabel": "llama.cpp",
            "detail": localize("Uses your detected GPU automatically", locale),
            "status": "installed",
            "statusLabel": localize("Ready", locale),
            "isMain": llm_runtime_choice == "llama.cpp" and llama_choice == "Auto",
            "canSetMain": True,
            "canInstall": False,
            "requiresGpu": False,
        }
    )
    for row in matrix.llama_rows(gpus, system, os_name, machine):
        status_code, status_label = _status_pair(row["status"])
        if status_code == "unavailable":
            if not include_unavailable:
                hidden_unsupported += 1
                continue
        option = row["option"]
        chat_rows.append(
            {
                "id": f"llama:{option}",
                "label": localize(option, locale) if option == "CPU" else option,
                "runtimeLabel": "llama.cpp",
                "detail": _clean_note(row.get("note", ""), locale),
                "status": status_code,
                "statusLabel": status_label,
                "isMain": llm_runtime_choice == "llama.cpp" and llama_choice == option,
                "canSetMain": status_code != "unavailable",
                "canInstall": status_code == "available",
                "installLabel": localize("Install", locale),
                "requiresGpu": option != "CPU",
            }
        )
    for row in matrix.xllama_rows(gpus, system, os_name, machine):
        status_code, status_label = _status_pair(row["status"])
        if status_code == "unavailable":
            if not include_unavailable:
                hidden_unsupported += 1
                continue
        option = row["option"]
        chat_rows.append(
            {
                "id": f"xllama:{option}",
                "label": localize(option, locale) if option == "CPU" else option,
                "runtimeLabel": "xllamacpp",
                "detail": _clean_note(row.get("note", ""), locale),
                "status": status_code,
                "statusLabel": status_label,
                "isMain": llm_runtime_choice == "xllamacpp" and xllama_choice == option,
                "canSetMain": status_code != "unavailable",
                "canInstall": status_code == "available",
                "installLabel": localize("Install", locale),
                "requiresGpu": option != "CPU",
            }
        )

    if cpu_forced:
        chat_summary_name = f"{llm_runtime_choice} · {localize('CPU', locale)}"
    elif llm_runtime_choice == "xllamacpp":
        chat_choice_label = (
            localize("Automatic (recommended)", locale) if xllama_choice == "Auto" else xllama_choice
        )
        chat_summary_name = f"xllamacpp · {chat_choice_label}"
    else:
        chat_choice_label = (
            localize("Automatic (recommended)", locale) if llama_choice == "Auto" else llama_choice
        )
        chat_summary_name = f"llama.cpp · {chat_choice_label}"

    # 3. Background removal (ONNX Runtime + Windows ML)
    bg_rows = []
    bg_rows.append(
        {
            "id": "onnx:Auto",
            "label": localize("Automatic (recommended)", locale),
            "runtimeLabel": "ONNX Runtime",
            "detail": localize("Uses your detected GPU automatically", locale),
            "status": "installed",
            "statusLabel": localize("Ready", locale),
            "isMain": bg_choice == "Auto",
            "canSetMain": True,
            "canInstall": False,
            "requiresGpu": False,
        }
    )
    onnx_label_to_value = {label: value for value, label in matrix.ONNX_LABELS.items()}
    for row in matrix.onnx_rows(gpus, system, os_name, machine):
        value = onnx_label_to_value.get(row["option"], row["option"])
        if value == "Windows GPU":
            continue
        status_code, status_label = _status_pair(row["status"])
        if status_code == "unavailable":
            if not include_unavailable:
                hidden_unsupported += 1
                continue
        is_main = bg_choice == value
        bg_rows.append(
            {
                "id": f"onnx:{value}",
                "label": localize(row["option"], locale),
                "runtimeLabel": "ONNX Runtime",
                "detail": _clean_note(row.get("note", ""), locale),
                "status": status_code,
                "statusLabel": status_label,
                "isMain": is_main,
                "canSetMain": status_code != "unavailable",
                "canInstall": status_code == "available" and value != "CPU",
                "installLabel": localize("Install", locale),
                "requiresGpu": value != "CPU",
            }
        )
    if os_name == "Windows" or include_unavailable:
        for row in matrix.winml_rows(gpus, os_name, machine):
            status_code, status_label = _status_pair(row["status"])
            if status_code == "unavailable":
                if not include_unavailable:
                    hidden_unsupported += 1
                    continue
            provider_name = row["option"] + "ExecutionProvider"
            is_main = (
                bg_choice == "Windows GPU"
                and bg_provider_choice in ("Auto", "Windows ML")
                and winml_provider_choice == provider_name
            )
            bg_rows.append(
                {
                    "id": f"winml:{provider_name}",
                    "label": row["option"],
                    "runtimeLabel": "Windows ML",
                    "detail": _clean_note(row.get("note", ""), locale),
                    "status": status_code,
                    "statusLabel": status_label,
                    "isMain": is_main,
                    "canSetMain": status_code != "unavailable",
                    "canInstall": status_code == "available",
                    "installLabel": localize("Install", locale),
                    "requiresGpu": True,
                }
            )

    if cpu_forced:
        bg_summary_name = localize("CPU", locale)
    elif bg_choice == "Windows GPU" and bg_provider_choice in ("Auto", "Windows ML"):
        if winml_provider_choice and winml_provider_choice != "Auto":
            bg_summary_name = f"Windows ML · {winml_provider_choice.replace('ExecutionProvider', '')}"
        else:
            bg_summary_name = f"Windows ML · {localize('Automatic (recommended)', locale)}"
    elif bg_choice == "Auto":
        bg_summary_name = localize("Automatic (recommended)", locale)
    else:
        bg_summary_name = localize(matrix.ONNX_LABELS.get(bg_choice, bg_choice), locale)

    return {
        "groups": [
            {
                "id": "image",
                "title": localize("Image generation", locale),
                "summary": localize("Active default: {summary}", locale, summary=image_summary_name),
                "rows": image_rows,
            },
            {
                "id": "chat",
                "title": localize("Chat & prompt LLM", locale),
                "summary": localize("Active default: {summary}", locale, summary=chat_summary_name),
                "rows": chat_rows,
            },
            {
                "id": "background",
                "title": localize("Background removal", locale),
                "summary": localize("Active default: {summary}", locale, summary=bg_summary_name),
                "rows": bg_rows,
            },
        ],
        "labels": {
            "main": localize("Main", locale),
            "setMain": localize("Set as main", locale),
            "install": localize("Install", locale),
            "installOnRestart": localize("Install on restart", locale),
            "empty": localize("No supported backends detected.", locale),
            "cpuForced": localize(
                "Use CPU for everything is enabled — individual GPU selections are paused.",
                locale,
            ),
        },
        "cpuForced": bool(cpu_forced),
        "hiddenUnsupportedCount": hidden_unsupported,
    }


def xllama_status_text(locale=None, choice=None, cpu_forced=False):
    installed = inventory.xllamacpp_installed()
    lines = []
    if installed:
        lines.append(
            localize(
                "Installed xllamacpp: {version} ({backend})",
                locale,
                version=installed["version"],
                backend=installed["backend"] or localize("unknown backend", locale),
            )
        )
    else:
        lines.append(localize("xllamacpp is not installed yet.", locale))
    if cpu_forced:
        lines.append(localize("Effective choice: CPU (Use CPU for everything is on).", locale))
    elif installed and choice not in (None, "Auto") and choice != installed["backend"]:
        lines.append(
            localize(
                "The selected backend differs from the installed wheel. Install it now, then restart.",
                locale,
            )
        )
    return "\n\n".join(lines)


def image_status_text(locale=None, choice=None, cpu_forced=False, device_name=""):
    status = inventory.component_status()["image"]
    kind = status["kind"]
    lines = []
    if status["torch"]:
        lines.append(
            localize(
                "Installed PyTorch: {torch} ({kind})",
                locale,
                torch=status["torch"],
                kind=KIND_NAMES.get(kind, localize("other build", locale)),
            )
        )
    else:
        lines.append(localize("PyTorch is not installed yet.", locale))
    if device_name:
        lines.append(localize("Active device: {device}", locale, device=device_name))
    choice = choice or saved_image_backend(settings.default_settings)
    if cpu_forced:
        lines.append(localize("Effective choice: CPU (Use CPU for everything is on).", locale))
    if choice in ("cuda", "rocm", "xpu") and choice != kind:
        lines.append(
            localize(
                "The selected backend differs from the installed build. Save, request the switch below and restart online to install it.",
                locale,
            )
        )
    from ruinedfooocus.app.arguments import args as launch_args

    overrides = cli_overrides(launch_args)
    if (
        launch_args.cpu
        and "--cpu" not in overrides
        and not (cpu_forced or saved_image_backend(settings.default_settings) == "cpu")
    ):
        overrides.append("--cpu")
    if overrides:
        lines.append(
            localize(
                "Startup option {names} overrides the saved image backend.",
                locale,
                names=", ".join(overrides),
            )
        )
    if startup_marker("reinstalltorch").exists():
        lines.append(localize("A torch reinstall is pending for the next online restart.", locale))
    return "\n\n".join(lines)


def create_llama_update_controls(llama_backend, cpu_everywhere, initial_backend):
    from ruinedfooocus.backends.llama import llama_update

    def _auto_update_default():
        try:
            import ruinedfooocus.app.state as shared

            return bool(
                getattr(getattr(shared, "settings", None), "default_settings", {}).get(
                    "llama_auto_update", True
                )
            )
        except Exception:
            return True

    def local_status(backend, locale=None):
        tag = llama_update.configured_build(backend)
        if tag is None:
            return localize(
                "Select Auto, CPU, Vulkan, ROCm, or SYCL to manage official llama.cpp updates.",
                locale,
            )
        return localize("Configured llama.cpp build: {tag}.", locale, tag=tag)

    auto_update_toggle = gr.Checkbox(
        label=t("Auto-update llama.cpp to latest release"),
        value=_auto_update_default(),
        info=t(
            "Automatically check for and install the latest verified official llama.cpp release when online."
        ),
    )
    status = gr.Markdown(local_status(initial_backend))
    selection = gr.State(None)
    with gr.Row():
        check = gr.Button(t("Check for llama.cpp update"), size="sm")
        update = gr.Button(t("Install llama.cpp update"), size="sm", interactive=False)
        rollback = gr.Button(t("Use pinned llama.cpp build"), size="sm")

    def save_auto_update(enabled):
        try:
            import ruinedfooocus.app.state as shared

            if getattr(shared, "settings", None) is not None:
                shared.settings.default_settings["llama_auto_update"] = bool(enabled)
                if hasattr(shared.settings, "save_settings"):
                    shared.settings.save_settings()
        except Exception:
            pass

    auto_update_toggle.change(
        save_auto_update,
        inputs=auto_update_toggle,
        api_visibility="private",
        queue=False,
        show_progress="hidden",
    )

    def checked_backend(backend, cpu):
        if cpu and backend in ("Vulkan", "ROCm", "SYCL"):
            raise RuntimeError("CPU-only mode is active. Select CPU to manage updates.")
        if cpu and backend == "Auto":
            return "CPU"
        return llama_update.resolve_backend(backend)

    def reset(backend, request: gr.Request):
        return local_status(backend, locale_of(request)), None, gr.update(interactive=False)

    def check_update(backend, cpu, request: gr.Request):
        locale = locale_of(request)
        try:
            tag, candidate = llama_update.check_update(checked_backend(backend, cpu))
        except (RuntimeError, OSError, ValueError) as error:
            return str(error), None, gr.update(interactive=False)
        if candidate is None:
            text = localize(
                "Configured build: {tag}. No newer compatible build in the recent releases.",
                locale,
                tag=tag,
            )
        else:
            release_kind = (
                localize("prerelease", locale)
                if candidate.prerelease
                else localize("release", locale)
            )
            text = localize(
                "Configured build: {tag}. Available: {new_tag} ({kind}, {size} MiB).",
                locale,
                tag=tag,
                new_tag=candidate.tag,
                kind=release_kind,
                size=round(candidate.size / 1024**2, 1),
            )
        return text, candidate, gr.update(interactive=candidate is not None)

    def install_update(backend, cpu, candidate, request: gr.Request, progress=gr.Progress()):
        locale = locale_of(request)
        try:
            tag = llama_update.install_update(
                checked_backend(backend, cpu),
                candidate,
                lambda fraction, message: progress(fraction, desc=localize(message, locale)),
            )
        except (RuntimeError, OSError, ValueError) as error:
            return (
                localize("Update failed: {error}", locale, error=str(error)),
                candidate,
                gr.update(interactive=candidate is not None),
            )
        text = localize(
            "Build {tag} is ready for the next llama.cpp process using {backend}. Reload Chat to use it.",
            locale,
            tag=tag,
            backend=backend,
        )
        return text, None, gr.update(interactive=False)

    def use_pin(backend, request: gr.Request):
        try:
            tag = llama_update.use_pinned_build(backend)
        except (RuntimeError, OSError, ValueError) as error:
            return str(error), None, gr.update(interactive=False)
        text = localize(
            "Pinned build {tag} will be used for the next llama.cpp process using {backend}.",
            locale_of(request),
            tag=tag,
            backend=backend,
        )
        return text, None, gr.update(interactive=False)

    outputs = [status, selection, update]
    events = {
        "outputs": outputs,
        "api_visibility": "private",
        "concurrency_id": "backend-install",
        "concurrency_limit": 1,
        "trigger_mode": "once",
        "show_progress": "minimal",
    }
    check.click(check_update, inputs=[llama_backend, cpu_everywhere], **events)
    update.click(install_update, inputs=[llama_backend, cpu_everywhere, selection], **events)
    rollback.click(use_pin, inputs=llama_backend, **events)
    for component in (llama_backend, cpu_everywhere):
        component.change(
            reset,
            inputs=llama_backend,
            outputs=outputs,
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )


def llama_status_text(locale=None, choice=None, cpu_forced=False):
    status = inventory.component_status()["llm"]
    paths = status.get("paths", {})
    lines = []
    if status["server_override"]:
        lines.append(
            localize(
                "LLAMA_SERVER overrides this selection: {path}",
                locale,
                path=status["server_override"],
            )
        )
    if status["installed"]:
        lines.append(
            localize(
                "Installed llama.cpp builds: {names}", locale, names=", ".join(status["installed"])
            )
        )
        effective = (
            "CPU"
            if cpu_forced
            else (choice or settings.default_settings.get("llama_backend", "Auto"))
        )
        selected_path = paths.get(effective) or (
            next(iter(paths.values())) if effective == "Auto" and paths else None
        )
        if selected_path:
            lines.append(localize("Cached server binary: {path}", locale, path=selected_path))
        if effective == "Auto":
            import platform

            from ruinedfooocus.backends.llama.runtime_policy import preferred_backend

            preferred = preferred_backend(platform.system())
            if preferred not in paths:
                # Automatic falls back to what is cached; say what is missing.
                lines.append(
                    localize(
                        "Automatic prefers {backend}, which is not installed yet. Press Install now to add it.",
                        locale,
                        backend=preferred,
                    )
                )
    else:
        lines.append(localize("No llama.cpp build is installed yet.", locale))
    if cpu_forced:
        lines.append(localize("Effective choice: CPU (Use CPU for everything is on).", locale))
    return "\n\n".join(lines)


def background_status_text(locale=None, cpu_forced=False, choice=None):
    from ruinedfooocus.backends.onnx.onnx_runtime import runtime_status

    lines = []
    lines.append(
        localize(
            "Requested background backend: {backend}.",
            locale,
            backend=choice or settings.default_settings.get("rembg_backend", "Auto"),
        )
    )
    status = inventory.component_status()["background"]
    if status["windows_ml"]:
        lines.append(
            localize(
                "Windows ML provider installed: {name}",
                locale,
                name=status["windows_ml"],
            )
        )
    if choice == "CPU" or auxiliary_cpu(cpu_forced):
        lines.append(localize("Effective choice: CPU (explicit CPU authority).", locale))
    last = runtime_status()
    if last == "No background-removal GPU session has been loaded.":
        last = localize("No background-removal GPU session has been loaded.", locale)
    if last:
        lines.append("Last shared runtime status: " + str(last))
    return "\n\n".join(lines) or localize("Background removal has not run in this session.", locale)


def create_hardware_tab(device_name="", launch_args=None, profile_base=None):
    from ruinedfooocus.backends.llama import llama_installer
    from ruinedfooocus.backends.llama.llama_build import build_cuda_runtime
    from ruinedfooocus.backends.llama.llama_runtime_info import runtime_choices

    saved = settings.default_settings
    system = platform.system()
    gpus = inventory.detect_gpus()
    forced = effective_backends(saved)["forced_cpu"]
    installed = inventory.installed_runtimes(system)
    saved_image = saved_image_backend(saved)
    saved_engine = saved.get("image_engine", "auto")
    saved_llm_runtime = saved.get("llm_runtime", "llama.cpp")
    saved_llama = llama_installer.canonical_backend(saved.get("llama_backend", "Auto"))
    saved_xllama = saved.get("xllamacpp_backend", "Auto")
    saved_background = saved.get("rembg_backend", "Auto")
    saved_bg_provider = saved.get("rembg_provider", "Auto")
    saved_winml_provider = saved.get("rembg_winml_provider", "Auto")

    hardware = gr.Markdown(hardware_text(), elem_classes=["rf-status-card"])
    localized_output(hardware_text, [], [hardware])
    with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
        refresh_hardware = gr.Button(
            t("Refresh hardware"), size="sm", min_width=120, elem_classes=["rf-inline-action"]
        )
    cpu_everywhere = gr.Checkbox(
        label=t("Use CPU for everything"),
        value=saved.get("cpu_everywhere", False),
        info=t(
            "Runs image generation, Chat and background removal on the CPU. Image generation changes after a restart; Chat and background removal at their next load. Your individual choices are kept."
        ),
    )
    add_setting("cpu_everywhere", cpu_everywhere)

    assets = Path(__file__).resolve().parent / "assets"
    with gr.Accordion(t("Supported on this PC"), open=True, elem_id="hardware-support"):
        support = gr.HTML(
            value=support_matrix_payload(
                include_unavailable=False,
                cpu_forced=forced,
                image_choice=saved_image,
                image_engine_choice=saved_engine,
                llm_runtime_choice=saved_llm_runtime,
                llama_choice=saved_llama,
                xllama_choice=saved_xllama,
                bg_choice=saved_background,
                bg_provider_choice=saved_bg_provider,
                winml_provider_choice=saved_winml_provider,
            ),
            html_template=(assets / "hardware_matrix.html").read_text(encoding="utf-8"),
            css_template=(assets / "hardware_matrix.css").read_text(encoding="utf-8"),
            js_on_load=(assets / "hardware_matrix.js").read_text(encoding="utf-8"),
            elem_id="hardware-support-matrix",
        )
        matrix_status = gr.Markdown("", visible=False, elem_classes=["rf-status-card"])
        show_unsupported = gr.Checkbox(
            label=t("Show unsupported options for other GPUs/OS"),
            info=t("Reveal backends and packages intended for other GPUs or operating systems."),
            value=False,
        )

    # Image generation options (PyTorch / ComfyUI + optional Vulkan engine)
    with gr.Accordion(t("Image generation (ComfyUI / PyTorch)"), open=False):
        image_values = inventory.image_backend_choices(gpus, system, installed)
        image_backend = gr.Dropdown(
            label=t("Image backend"),
            choices=labelled(image_values, IMAGE_LABELS, saved_image),
            value=saved_image,
            interactive=not forced,
            info=t(
                "Takes effect after a restart. TORCH_PLATFORM and explicit launch options take priority."
            ),
        )
        add_setting("image_backend", image_backend)
        image_status = gr.Markdown(
            image_status_text(device_name=device_name), elem_classes=["rf-status-card"]
        )
        localized_output(
            lambda locale, choice, cpu: image_status_text(locale, choice, cpu, device_name),
            [image_backend, cpu_everywhere],
            [image_status],
        )
        with gr.Row():
            switch_image = gr.Button(
                t("Install / switch on next start"),
                variant="primary",
                size="sm",
                min_width=200,
            )
            cancel_switch = gr.Button(
                t("Cancel pending switch"), variant="stop", size="sm", min_width=160
            )

        def request_switch(choice, cpu, request: gr.Request):
            locale = locale_of(request)
            current = inventory.installed_torch_kind()
            if choice in ("auto", "cpu") or choice == current:
                gr.Info(
                    localize(
                        "No installation is needed for this choice. Save settings and restart to apply it.",
                        locale,
                    )
                )
            elif skip_dependency_install():
                raise gr.Error(
                    localize(
                        "Installation is disabled (offline or --skip-dependency-install). Restart online to switch.",
                        locale,
                    )
                )
            else:
                data_path("reinstalltorch").touch()
                gr.Info(
                    localize(
                        "Switch to {backend} requested. Save settings, then restart online to install it.",
                        locale,
                        backend=KIND_NAMES.get(choice, choice),
                    )
                )
            return image_status_text(locale, choice, cpu, device_name)

        def cancel_pending(choice, cpu, request: gr.Request):
            locale = locale_of(request)
            marker = startup_marker("reinstalltorch")
            writable = data_path("reinstalltorch")
            if marker.exists() or writable.exists():
                marker.unlink(missing_ok=True)
                writable.unlink(missing_ok=True)
                gr.Info(localize("The pending torch switch was cancelled.", locale))
            else:
                gr.Info(localize("No switch is pending.", locale))
            return image_status_text(locale, choice, cpu, device_name)

        def update_image_status(choice, cpu, request: gr.Request):
            return image_status_text(locale_of(request), choice, cpu, device_name)

        image_backend.change(
            update_image_status,
            inputs=[image_backend, cpu_everywhere],
            outputs=image_status,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        cancel_switch.click(
            cancel_pending,
            inputs=[image_backend, cpu_everywhere],
            outputs=image_status,
            show_progress="minimal",
            api_visibility="private",
        )
        optional_controls = create_optional_image_controls(saved, cpu_everywhere, image_backend)

    # Chat and prompt LLM (unified: main runtime + llama.cpp + xllamacpp)
    with gr.Accordion(t("Chat and prompt LLM (llama.cpp)"), open=False):
        llm_runtime = gr.Dropdown(
            label=t("Chat runtime"),
            info=t("Backend used to run local GGUF Chat and Vision models (llama.cpp server or in-process xllamacpp)."),
            choices=runtime_choices(),
            value=saved_llm_runtime,
        )
        add_setting("llm_runtime", llm_runtime)
        runtime_signature = gr.State(str(runtime_choices()))

        def refresh_runtime_label(previous):
            choices = runtime_choices()
            signature = str(choices)
            if signature == previous:
                return gr.skip(), gr.skip()
            return gr.update(choices=choices), signature

        gr.Timer(2).tick(
            refresh_runtime_label,
            inputs=runtime_signature,
            outputs=[llm_runtime, runtime_signature],
            queue=False,
            api_visibility="undocumented",
            show_progress="hidden",
        )
        llama_values = inventory.llm_backend_choices(gpus)
        llama_backend = gr.Dropdown(
            label=t("llama.cpp backend"),
            choices=labelled(llama_values, LLAMA_LABELS, saved_llama),
            value=saved_llama,
            interactive=not forced,
            info=t(
                "Save settings, then load or chat to switch. LLAMA_SERVER overrides this selection."
            ),
        )
        add_setting("llama_backend", llama_backend)
        llama_status = gr.Markdown(
            llama_status_text(cpu_forced=forced), elem_classes=["rf-status-card"]
        )
        localized_output(
            lambda locale, choice, cpu: llama_status_text(locale, choice, cpu),
            [llama_backend, cpu_everywhere],
            [llama_status],
        )
        install_llama = gr.Button(
            t("Install now"),
            variant="primary",
            size="sm",
            min_width=160,
            elem_classes=["rf-inline-action"],
        )

        def install_selected_llama(choice, request: gr.Request, progress=gr.Progress()):
            locale = locale_of(request)
            backend = llama_installer.canonical_backend(choice)
            if backend == "Auto":
                yield (
                    gr.update(interactive=True),
                    localize(
                        "Choose a specific backend to install. Automatic selection installs on the first Chat load.",
                        locale,
                    ),
                )
                return
            if backend not in llama_values:
                raise gr.Error(
                    localize("This backend is not available on the detected hardware.", locale)
                )
            if backend != "CPU":
                try:
                    llama_installer.validate_cuda_backend(backend)
                except Exception as error:
                    raise gr.Error(str(error)) from error
            if backend in llama_installer_installed():
                yield gr.update(interactive=True), llama_status_text(locale, backend)
                gr.Info(localize("{backend} is already installed.", locale, backend=backend))
                return
            if skip_dependency_install():
                yield (
                    gr.update(interactive=True),
                    localize("Installation is disabled by offline/skip-install policy.", locale),
                )
                return
            yield (
                gr.update(interactive=False),
                localize("Downloading llama.cpp {backend}…", locale, backend=backend),
            )
            progress(None, desc=backend)
            try:
                path = llama_installer.server_path(gpu=backend != "CPU", backend=backend)
            except Exception as error:
                yield (
                    gr.update(interactive=True),
                    localize("Installation failed: {error}", locale, error=str(error)),
                )
                return
            gr.Info(localize("Installed llama.cpp {backend}.", locale, backend=backend))
            yield gr.update(interactive=True), llama_status_text(locale, backend) + f"\n\n`{path}`"

        def llama_installer_installed():
            return inventory.llama_installed(system)

        def update_llama_status(choice, cpu, request: gr.Request):
            return llama_status_text(locale_of(request), choice, cpu)

        llama_backend.change(
            update_llama_status,
            inputs=[llama_backend, cpu_everywhere],
            outputs=llama_status,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        if system in ("Windows", "Linux"):
            create_llama_update_controls(llama_backend, cpu_everywhere, saved_llama)
        # Passive OS/vendor eligibility only. The explicit build action validates
        # devices, driver and toolkit; browsing must not run nvidia-smi.
        can_build_cuda = system == "Linux" and inventory.has_vendor(gpus, "nvidia")
        with gr.Accordion(t("Build local CUDA runtime"), open=False, visible=can_build_cuda):
            gr.Markdown(
                t(
                    "Builds the pinned llama.cpp for Linux/WSL and this NVIDIA GPU. "
                    "CUDA/CMake dependencies are installed in a separate cache environment. "
                    "Requires system build tools (Ubuntu: `sudo apt install build-essential git python3-venv`). "
                    "Compilation may take several minutes and use several GB of disk space."
                )
            )
            build_runtime = gr.Button(
                t("Install dependencies and build"),
                variant="primary",
                size="sm",
                min_width=200,
            )
            build_status = gr.Textbox(
                label=t("Build status"),
                info=t("Compiler output for the local llama.cpp CUDA runtime build."),
                lines=6,
                max_lines=12,
                interactive=False,
                visible=False,
            )

            def build_native_runtime():
                yield (
                    gr.update(interactive=False),
                    gr.update(value="Preparing build...", visible=True),
                    gr.skip(),
                )
                try:
                    for message in build_cuda_runtime():
                        yield gr.skip(), gr.update(value=message), gr.skip()
                except Exception as error:
                    yield (
                        gr.update(interactive=True),
                        gr.update(value=str(error)),
                        gr.skip(),
                    )
                else:
                    yield (
                        gr.update(interactive=True),
                        gr.skip(),
                        gr.update(
                            choices=labelled(
                                inventory.llm_backend_choices(gpus), LLAMA_LABELS, saved_llama
                            )
                        ),
                    )

        # Alternative chat runtime (xllamacpp wheels) inside the unified Chat section
        with gr.Accordion(t("Chat runtime (xllamacpp)"), open=False):
            xllama_values = inventory.xllamacpp_backend_choices(gpus, system)
            xllama_backend = gr.Dropdown(
                label=t("xllamacpp backend"),
                choices=labelled(xllama_values, XLLAMA_LABELS, saved_xllama),
                value=saved_xllama,
                interactive=not forced,
                info=t(
                    "Used when Chat runtime is set to xllamacpp. Automatic follows the installed PyTorch build."
                ),
            )
            add_setting("xllamacpp_backend", xllama_backend)
            xllama_status = gr.Markdown(
                xllama_status_text(choice=saved_xllama, cpu_forced=forced),
                elem_classes=["rf-status-card"],
            )
            localized_output(
                lambda locale, choice, cpu: xllama_status_text(locale, choice, cpu),
                [xllama_backend, cpu_everywhere],
                [xllama_status],
            )
            install_xllama = gr.Button(
                t("Install xllamacpp now"),
                variant="primary",
                size="sm",
                min_width=180,
                elem_classes=["rf-inline-action"],
            )

            def install_selected_xllama(choice, cpu, request: gr.Request, progress=gr.Progress()):
                import os

                from ruinedfooocus.packaging.gpu_installer import (
                    ensure_xllamacpp,
                    installed_torch_constraints,
                )

                locale = locale_of(request)
                choice = "CPU" if cpu else choice
                if choice not in xllama_values:
                    raise gr.Error(
                        localize("This backend is not available on the detected hardware.", locale)
                    )
                if skip_dependency_install():
                    yield (
                        gr.update(interactive=True),
                        localize("Installation is disabled by offline/skip-install policy.", locale),
                    )
                    return
                yield (
                    gr.update(interactive=False),
                    localize("Installing xllamacpp {backend}…", locale, backend=choice),
                )
                progress(None, desc=choice)
                try:
                    ensure_xllamacpp(
                        os.environ.get("TORCH_PLATFORM") or ("cpu" if cpu else "vulkan"),
                        system,
                        installed_torch=installed_torch_constraints(),
                        choice=choice,
                    )
                except Exception as error:
                    yield (
                        gr.update(interactive=True),
                        localize("Installation failed: {error}", locale, error=str(error)),
                    )
                    return
                gr.Info(localize("Installed xllamacpp {backend}.", locale, backend=choice))
                yield (
                    gr.update(interactive=True),
                    xllama_status_text(locale, choice, cpu)
                    + "\n\n"
                    + localize("Restart RuinedFooocus if xllamacpp was already loaded.", locale),
                )

            def update_xllama_status(choice, cpu, request: gr.Request):
                return xllama_status_text(locale_of(request), choice, cpu)

            xllama_backend.change(
                update_xllama_status,
                inputs=[xllama_backend, cpu_everywhere],
                outputs=xllama_status,
                show_progress="hidden",
                api_visibility="undocumented",
            )

        if launch_args is not None and profile_base is not None:
            with gr.Accordion(t("Installed chat runtimes"), open=False):
                from ruinedfooocus.features.chat.ui_runtime_info import (
                    create_runtime_info_controls,
                )

                create_runtime_info_controls(launch_args, profile_base)

    # Background removal (ONNX Runtime / rembg)
    with gr.Accordion(t("Background removal (ONNX Runtime)"), open=False):
        from ruinedfooocus.ui.optional_image_runtime import RuntimeActionSession

        background_actions = gr.State(RuntimeActionSession())
        background_values = inventory.onnx_backend_choices(gpus, system)
        background_backend = gr.Dropdown(
            label=t("Background-removal device"),
            choices=labelled(background_values, BACKGROUND_LABELS, saved_background),
            value=saved_background,
            interactive=not forced,
            info=t(
                "Auto uses an installed GPU runtime, otherwise CPU. CUDA, MIGraphX and OpenVINO install into a separate folder; Windows ML offers CPU/GPU/NPU selection on Windows. CPU mode overrides all acceleration choices."
            ),
        )
        add_setting("rembg_backend", background_backend)
        background_status = gr.Markdown(
            background_status_text(cpu_forced=forced), elem_classes=["rf-status-card"]
        )
        localized_output(
            lambda locale, choice, cpu: background_status_text(locale, cpu, choice),
            [background_backend, cpu_everywhere],
            [background_status],
        )

        def update_background_request(choice, cpu, request: gr.Request):
            return background_status_text(locale_of(request), cpu, choice)

        gr.on(
            triggers=[background_backend.input, cpu_everywhere.input],
            fn=update_background_request,
            inputs=[background_backend, cpu_everywhere],
            outputs=background_status,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        install_background = gr.Button(
            t("Install / verify selected ONNX runtime"),
            variant="primary",
            size="sm",
            min_width=240,
            elem_classes=["rf-inline-action"],
        )

        matrix_state_inputs = [
            show_unsupported,
            cpu_everywhere,
            image_backend,
            optional_controls["engine"],
            llm_runtime,
            llama_backend,
            xllama_backend,
            background_backend,
        ]

        def render_matrix_for_state(
            locale,
            include_unsupported,
            cpu,
            img_choice,
            eng_choice,
            chat_rt,
            llama_choice,
            xllama_choice,
            bg_choice,
            bg_prov="Auto",
            winml_prov="Auto",
            *,
            refresh=False,
        ):
            return support_matrix_payload(
                locale=locale,
                refresh=refresh,
                include_unavailable=bool(include_unsupported),
                cpu_forced=bool(cpu),
                image_choice=img_choice or "auto",
                image_engine_choice=eng_choice or "auto",
                llm_runtime_choice=chat_rt or "llama.cpp",
                llama_choice=llama_choice or "Auto",
                xllama_choice=xllama_choice or "Auto",
                bg_choice=bg_choice or "Auto",
                bg_provider_choice=bg_prov or "Auto",
                winml_provider_choice=winml_prov or "Auto",
            )

        def refresh_matrix(*values, request: gr.Request = None):
            return render_matrix_for_state(locale_of(request), *values)

        winml_controls = (
            create_windows_ml_controls(
                saved,
                gpus,
                support,
                cpu_everywhere,
                background_backend,
                background_actions,
                refresh_support=refresh_matrix,
                refresh_inputs=matrix_state_inputs,
            )
            if system == "Windows"
            else None
        )
        if winml_controls is not None:
            matrix_state_inputs.extend(
                [winml_controls["background_provider"], winml_controls["provider"]]
            )

        def install_selected_background(
            choice,
            cpu,
            session,
            provider_name="Auto",
            device_type="Auto",
            request: gr.Request = None,
        ):
            from ruinedfooocus.backends.onnx.onnx_installer import (
                NATIVE_RUNTIMES,
                install_native_runtime,
            )

            locale = locale_of(request)
            if auxiliary_cpu(cpu) or choice in ("Auto", "CPU"):
                yield gr.update(interactive=True), background_status_text(locale, cpu, choice)
                return
            if choice not in background_values:
                raise gr.Error(
                    localize("This backend is not available on the detected hardware.", locale)
                )
            if choice == "Windows GPU" and winml_controls is not None:
                for button, status, _ in winml_controls["install"](
                    provider_name, device_type, cpu, choice, session, request
                ):
                    yield button, status
                return
            if choice not in NATIVE_RUNTIMES:
                raise gr.Error(
                    localize(
                        "Choose CUDA, MIGraphX or OpenVINO to install. CPU needs no installation; Windows ML has its own controls below.",
                        locale,
                    )
                )
            if skip_dependency_install():
                yield (
                    gr.update(interactive=True),
                    localize("Installation is disabled by offline/skip-install policy.", locale),
                )
                return
            yield (
                gr.update(interactive=False),
                localize("Installing the {backend} ONNX runtime…", locale, backend=choice),
            )
            try:
                result = session.run_admitted(
                    lambda check: install_native_runtime(choice, check_cancelled=check)
                )
            except Exception as error:
                yield (
                    gr.update(interactive=True),
                    localize("Installation failed: {error}", locale, error=str(error)),
                )
                return
            yield (
                gr.update(interactive=True),
                result + "\n\n" + background_status_text(locale, cpu, choice),
            )

        install_background.click(
            install_selected_background,
            inputs=[background_backend, cpu_everywhere, background_actions]
            + (
                [winml_controls["provider"], winml_controls["device_type"]]
                if winml_controls is not None
                else []
            ),
            outputs=[install_background, background_status],
            trigger_mode="once",
            show_progress="minimal",
            api_visibility="private",
            concurrency_id="onnx-runtime-install",
            concurrency_limit=1,
        ).then(
            refresh_matrix,
            inputs=matrix_state_inputs,
            outputs=support,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    switch_image.click(
        request_switch,
        inputs=[image_backend, cpu_everywhere],
        outputs=image_status,
        trigger_mode="once",
        show_progress="minimal",
        api_visibility="private",
        concurrency_id="backend-install",
        concurrency_limit=1,
    ).then(
        refresh_matrix,
        inputs=matrix_state_inputs,
        outputs=support,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    install_llama.click(
        install_selected_llama,
        inputs=llama_backend,
        outputs=[install_llama, llama_status],
        trigger_mode="once",
        show_progress="minimal",
        api_visibility="private",
        concurrency_id="backend-install",
        concurrency_limit=1,
    ).then(
        refresh_matrix,
        inputs=matrix_state_inputs,
        outputs=support,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    build_runtime.click(
        build_native_runtime,
        outputs=[build_runtime, build_status, llama_backend],
        trigger_mode="once",
        concurrency_id="native-runtime-build",
        concurrency_limit=1,
        api_visibility="private",
        show_progress="hidden",
    ).then(
        refresh_matrix,
        inputs=matrix_state_inputs,
        outputs=support,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    install_xllama.click(
        install_selected_xllama,
        inputs=[xllama_backend, cpu_everywhere],
        outputs=[install_xllama, xllama_status],
        trigger_mode="once",
        show_progress="minimal",
        api_visibility="private",
        concurrency_id="backend-install",
        concurrency_limit=1,
    ).then(
        refresh_matrix,
        inputs=matrix_state_inputs,
        outputs=support,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    localized_output(
        lambda locale, *vals: render_matrix_for_state(locale, *vals),
        matrix_state_inputs,
        [support],
    )
    gr.on(
        triggers=[control.change for control in matrix_state_inputs],
        fn=refresh_matrix,
        inputs=matrix_state_inputs,
        outputs=support,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    # Interactive "Set as main" & "Install" handler directly from Supported on this PC cards
    matrix_outputs = [
        support,
        matrix_status,
        image_backend,
        optional_controls["engine"],
        llm_runtime,
        llama_backend,
        xllama_backend,
        background_backend,
        image_status,
        llama_status,
        xllama_status,
        background_status,
    ]
    if winml_controls is not None:
        matrix_outputs.extend(
            [winml_controls["background_provider"], winml_controls["provider"]]
        )

    def _matrix_result(
        locale,
        include_unsupported,
        cpu,
        img_choice,
        eng_choice,
        chat_rt,
        llama_choice,
        xllama_choice,
        bg_choice,
        bg_prov,
        winml_prov,
        message,
    ):
        payload = render_matrix_for_state(
            locale,
            include_unsupported,
            cpu,
            img_choice,
            eng_choice,
            chat_rt,
            llama_choice,
            xllama_choice,
            bg_choice,
            bg_prov,
            winml_prov,
        )
        out = [
            payload,
            gr.update(value=message, visible=bool(message)),
            img_choice,
            eng_choice,
            chat_rt,
            llama_choice,
            xllama_choice,
            bg_choice,
            image_status_text(locale, img_choice, cpu, device_name),
            llama_status_text(locale, llama_choice, cpu),
            xllama_status_text(locale, xllama_choice, cpu),
            background_status_text(locale, cpu, bg_choice),
        ]
        if winml_controls is not None:
            out.extend([bg_prov, winml_prov])
        return tuple(out)

    def handle_matrix_action(
        evt: gr.EventData,
        include_unsupported,
        cpu,
        img_choice,
        eng_choice,
        chat_rt,
        llama_choice,
        xllama_choice,
        bg_choice,
        bg_session,
        opt_device,
        opt_session,
        bg_prov="Auto",
        winml_prov="Auto",
        request: gr.Request = None,
    ):
        locale = locale_of(request)
        data = evt._data if isinstance(getattr(evt, "_data", None), dict) else {}
        action = data.get("action")
        row_id = data.get("id") or ""
        if ":" not in row_id or action not in ("set_main", "install"):
            yield tuple(gr.skip() for _ in matrix_outputs)
            return
        prefix, target = row_id.split(":", 1)

        if action == "set_main":
            if cpu:
                yield _matrix_result(
                    locale,
                    include_unsupported,
                    cpu,
                    img_choice,
                    eng_choice,
                    chat_rt,
                    llama_choice,
                    xllama_choice,
                    bg_choice,
                    bg_prov,
                    winml_prov,
                    localize("Disable Use CPU for everything to change default GPU backends.", locale),
                )
                return
            if prefix == "image":
                if target == "sdcpp-vulkan":
                    eng_choice = "sdcpp-vulkan"
                    chosen_label = "stable-diffusion.cpp (Vulkan)"
                else:
                    img_choice = target
                    eng_choice = "auto" if target == "auto" else "comfy"
                    chosen_label = localize(IMAGE_LABELS.get(target, target), locale)
                msg = localize(
                    "Main Image generation backend set to {backend}. Click Save to keep it.",
                    locale,
                    backend=chosen_label,
                )
            elif prefix == "llama":
                chat_rt = "llama.cpp"
                llama_choice = target
                msg = localize(
                    "Main Chat runtime set to {runtime} ({backend}). Click Save to keep it.",
                    locale,
                    runtime="llama.cpp",
                    backend=target,
                )
            elif prefix == "xllama":
                chat_rt = "xllamacpp"
                xllama_choice = target
                msg = localize(
                    "Main Chat runtime set to {runtime} ({backend}). Click Save to keep it.",
                    locale,
                    runtime="xllamacpp",
                    backend=target,
                )
            elif prefix == "onnx":
                bg_choice = target
                msg = localize(
                    "Main Background removal backend set to {backend}. Click Save to keep it.",
                    locale,
                    backend=target,
                )
            elif prefix == "winml":
                bg_choice = "Windows GPU"
                bg_prov = "Windows ML"
                winml_prov = target
                msg = localize(
                    "Main Background removal backend set to Windows ML ({backend}). Click Save to keep it.",
                    locale,
                    backend=target.replace("ExecutionProvider", ""),
                )
            else:
                yield tuple(gr.skip() for _ in matrix_outputs)
                return
            yield _matrix_result(
                locale,
                include_unsupported,
                cpu,
                img_choice,
                eng_choice,
                chat_rt,
                llama_choice,
                xllama_choice,
                bg_choice,
                bg_prov,
                winml_prov,
                msg,
            )
            return

        # action == "install"
        try:
            if prefix == "image":
                if target == "sdcpp-vulkan":
                    for msg in optional_controls["install"](
                        "vulkan", opt_device, cpu, img_choice, opt_session, request
                    ):
                        yield _matrix_result(
                            locale,
                            include_unsupported,
                            cpu,
                            img_choice,
                            eng_choice,
                            chat_rt,
                            llama_choice,
                            xllama_choice,
                            bg_choice,
                            bg_prov,
                            winml_prov,
                            msg,
                        )
                    return
                img_choice = target
                status_msg = request_switch(target, cpu, request)
                yield _matrix_result(
                    locale,
                    include_unsupported,
                    cpu,
                    img_choice,
                    eng_choice,
                    chat_rt,
                    llama_choice,
                    xllama_choice,
                    bg_choice,
                    bg_prov,
                    winml_prov,
                    status_msg,
                )
                return
            if prefix == "llama":
                for _, status_msg in install_selected_llama(target, request):
                    yield _matrix_result(
                        locale,
                        include_unsupported,
                        cpu,
                        img_choice,
                        eng_choice,
                        chat_rt,
                        llama_choice,
                        xllama_choice,
                        bg_choice,
                        bg_prov,
                        winml_prov,
                        status_msg,
                    )
                return
            if prefix == "xllama":
                for _, status_msg in install_selected_xllama(target, cpu, request):
                    yield _matrix_result(
                        locale,
                        include_unsupported,
                        cpu,
                        img_choice,
                        eng_choice,
                        chat_rt,
                        llama_choice,
                        xllama_choice,
                        bg_choice,
                        bg_prov,
                        winml_prov,
                        status_msg,
                    )
                return
            if prefix == "onnx":
                for _, status_msg in install_selected_background(
                    target, cpu, bg_session, request=request
                ):
                    yield _matrix_result(
                        locale,
                        include_unsupported,
                        cpu,
                        img_choice,
                        eng_choice,
                        chat_rt,
                        llama_choice,
                        xllama_choice,
                        bg_choice,
                        bg_prov,
                        winml_prov,
                        status_msg,
                    )
                return
            if prefix == "winml" and winml_controls is not None:
                for _, status_msg, _ in winml_controls["install"](
                    target, "Auto", cpu, "Windows GPU", bg_session, request
                ):
                    yield _matrix_result(
                        locale,
                        include_unsupported,
                        cpu,
                        img_choice,
                        eng_choice,
                        chat_rt,
                        llama_choice,
                        xllama_choice,
                        bg_choice,
                        bg_prov,
                        winml_prov,
                        status_msg,
                    )
                return
        except gr.Error as error:
            yield _matrix_result(
                locale,
                include_unsupported,
                cpu,
                img_choice,
                eng_choice,
                chat_rt,
                llama_choice,
                xllama_choice,
                bg_choice,
                bg_prov,
                winml_prov,
                str(error),
            )

    matrix_click_inputs = [
        show_unsupported,
        cpu_everywhere,
        image_backend,
        optional_controls["engine"],
        llm_runtime,
        llama_backend,
        xllama_backend,
        background_backend,
        background_actions,
        optional_controls["device"],
        optional_controls["session"],
    ]
    if winml_controls is not None:
        matrix_click_inputs.extend(
            [winml_controls["background_provider"], winml_controls["provider"]]
        )

    support.click(
        handle_matrix_action,
        inputs=matrix_click_inputs,
        outputs=matrix_outputs,
        show_progress="minimal",
        api_visibility="private",
        concurrency_id="backend-install",
        concurrency_limit=1,
    )

    else_outputs = [hardware, image_status, llama_status, xllama_status, background_status, support]

    def refresh_all(
        include_unsupported,
        cpu,
        img_choice,
        eng_choice,
        chat_rt,
        llama_choice,
        xllama_choice,
        bg_choice,
        bg_prov="Auto",
        winml_prov="Auto",
        request: gr.Request = None,
    ):
        locale = locale_of(request)
        inventory.detect_gpus(refresh=True)
        return (
            hardware_text(locale),
            image_status_text(locale, img_choice, cpu, device_name),
            llama_status_text(locale, llama_choice, cpu),
            xllama_status_text(locale, xllama_choice, cpu),
            background_status_text(locale, cpu, bg_choice),
            render_matrix_for_state(
                locale,
                include_unsupported,
                cpu,
                img_choice,
                eng_choice,
                chat_rt,
                llama_choice,
                xllama_choice,
                bg_choice,
                bg_prov,
                winml_prov,
                refresh=True,
            ),
        )

    refresh_hardware.click(
        refresh_all,
        inputs=matrix_state_inputs,
        outputs=else_outputs,
        show_progress="minimal",
        api_visibility="private",
    )

    def toggle_cpu(cpu):
        return [gr.update(interactive=not cpu)] * 4

    cpu_everywhere.change(
        toggle_cpu,
        inputs=cpu_everywhere,
        outputs=[image_backend, llama_backend, xllama_backend, background_backend],
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return cpu_everywhere, llm_runtime


def refresh_support_table(request: gr.Request):
    """Read installed state after an explicit action, in this browser's locale."""
    return support_table(locale_of(request))


def create_optional_image_controls(saved, cpu_everywhere, image_backend):
    from ruinedfooocus.ui.optional_image_runtime import (
        ENGINE_CHOICES,
        RuntimeActionSession,
        requested_values,
        status_text,
    )
    import time

    with gr.Accordion(t("Optional image engine"), open=False):
        engine = gr.Dropdown(
            label=t("Image engine"),
            choices=[(t(label), value) for label, value in ENGINE_CHOICES],
            value=saved.get("image_engine", "auto"),
            info=t("Automatic tries native GPU first, then qualified optional Vulkan, then CPU."),
            interactive=True,
        )
        add_setting("image_engine", engine)
        device = gr.Textbox(
            label=t("Optional Vulkan backend name"),
            value=saved.get("sdcpp_vulkan_device", "Vulkan0"),
            info=t(
                "Use the exact backend name checked by Verify. Vulkan0 is not a physical GPU identity."
            ),
        )
        add_setting("sdcpp_vulkan_device", device)
        package = gr.Dropdown(
            label=t("Optional runtime package"),
            info=t("Variant of the optional stable-diffusion.cpp package to install (Vulkan GPU or CPU)."),
            choices=[("Vulkan", "vulkan"), ("CPU", "cpu")],
            value="vulkan",
        )
        action_session = gr.State(RuntimeActionSession())
        status = gr.Markdown(
            status_text(
                saved.get("image_engine", "auto"),
                saved.get("sdcpp_vulkan_device", "Vulkan0"),
                values=saved,
            )
        )
        with gr.Row():
            install_optional = gr.Button(t("Install optional runtime"), size="sm")
            verify_optional = gr.Button(t("Verify optional runtime device"), size="sm")
            refresh_optional = gr.Button(t("Refresh optional runtime status"), size="sm")
            cancel_optional = gr.Button(t("Cancel runtime action"), size="sm")

        def optional_status_text(locale, engine, device, cpu, image_choice):
            return status_text(
                engine,
                device,
                values=requested_values(settings.default_settings, cpu, image_choice),
                locale=locale,
            )

        localized_output(
            optional_status_text, [engine, device, cpu_everywhere, image_backend], [status]
        )

        def optional_status(engine, device, cpu, image_choice, request: gr.Request):
            return optional_status_text(
                resolve_locale(request=request), engine, device, cpu, image_choice
            )

        gr.on(
            triggers=[
                engine.input,
                device.input,
                cpu_everywhere.input,
                image_backend.input,
                refresh_optional.click,
            ],
            fn=optional_status,
            inputs=[engine, device, cpu_everywhere, image_backend],
            outputs=status,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

        def optional_install(
            backend, device, cpu, image_choice, session, request: gr.Request = None
        ):
            started = False
            locale = resolve_locale(request=request)
            try:
                values = requested_values(settings.default_settings, cpu, image_choice)
                future = session.install(
                    backend,
                    device,
                    values,
                    offline=skip_dependency_install(),
                    skip_install=skip_dependency_install(),
                    locale=locale,
                )
                started = True
                yield localize(
                    "Preparing optional runtime; active model compute remains available.", locale
                )
                last = ""
                while not future.done():
                    if session.progress != last:
                        last = session.progress
                        if last:
                            yield last
                    time.sleep(0.1)
                future.result()
                yield localize(
                    "Optional runtime payload prepared. Verify its device explicitly; model/operation acceptance is still required.",
                    locale,
                )
            except Exception as error:
                yield localize(
                    "Optional runtime preparation failed: {error}", locale, error=str(error)
                )
            finally:
                if started:
                    if not future.done():
                        session.cancel()
                    session.finish()

        def optional_verify(
            backend, device, cpu, image_choice, session, request: gr.Request = None
        ):
            locale = resolve_locale(request=request)
            yield localize(
                "Queued for shared compute admission; verification does not generate an image.",
                locale,
            )
            try:
                yield session.verify(
                    backend,
                    device,
                    requested_values(settings.default_settings, cpu, image_choice),
                    locale=locale,
                )
            except Exception as error:
                yield localize(
                    "Optional runtime verification failed: {error}", locale, error=str(error)
                )

        action_inputs = [package, device, cpu_everywhere, image_backend, action_session]
        install_optional.click(
            optional_install,
            inputs=action_inputs,
            outputs=status,
            show_progress="hidden",
            api_visibility="private",
            trigger_mode="once",
        )
        verify_optional.click(
            optional_verify,
            inputs=action_inputs,
            outputs=status,
            show_progress="hidden",
            api_visibility="private",
            trigger_mode="once",
        )
        cancel_optional.click(
            lambda session: session.cancel(),
            inputs=action_session,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    return {
        "engine": engine,
        "device": device,
        "session": action_session,
        "install": optional_install,
    }


def auxiliary_cpu(cpu=False):
    from ruinedfooocus.app.arguments import args
    from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary

    return bool(cpu or force_cpu_for_auxiliary(args))


def windows_ml_choices(providers, gpus, saved="Auto", device_type="Auto"):
    """Only locally offered certified bridge-compatible candidates; retain intent."""
    from ruinedfooocus.packaging.support_matrix import windows_ml_provider_names

    offered = [
        row["name"]
        for row in providers
        if isinstance(row, dict)
        and row.get("certified") is True
        and row.get("bridge_compatible") is True
        and (
            device_type in row.get("device_types", ())
            if device_type != "Auto"
            else any(kind in row.get("device_types", ()) for kind in ("CPU", "GPU"))
        )
    ]
    names = offered if device_type == "CPU" else windows_ml_provider_names(offered, gpus, "Auto")
    choices = [(t("Automatic (recommended)"), "Auto")] + [
        (name.replace("ExecutionProvider", "") + " · Catalogue candidate", name)
        for name in dict.fromkeys(names)
    ]
    if saved != "Auto" and saved not in names:
        choices.append(
            (saved.replace("ExecutionProvider", "") + " · Unavailable saved choice", saved)
        )
    return choices


def validate_windows_ml_choice(provider_name, device_type, gpus):
    from ruinedfooocus.backends.onnx.windows_ml import catalog_status

    if provider_name != "Auto" and provider_name not in {
        value
        for _, value in windows_ml_choices(catalog_status()["providers"], gpus, "Auto", device_type)
    }:
        raise ValueError(
            "This saved provider is unavailable in the certified compatible local catalogue. Refresh the catalogue or choose an offered provider."
        )
    # Auto must not infer an NPU from the CPU/GPU vendor. NPU is explicit.
    return ("GPU" if gpus else "CPU") if device_type == "Auto" else device_type


def create_windows_ml_controls(
    saved,
    gpus,
    support,
    cpu_everywhere,
    background_backend,
    action_session,
    *,
    refresh_support=None,
    refresh_inputs=None,
):
    """Windows-only provider controls, moved from Image runtime; keys are unchanged."""
    from ruinedfooocus.backends.onnx.windows_ml import (
        catalog_status,
        DEVICE_TYPES,
        provider_preference,
    )

    background_provider = gr.Dropdown(
        label=t("Windows background provider"),
        choices=["Windows ML"],
        value="Windows ML",
        visible=False,
    )
    add_setting("rembg_provider", background_provider)
    with gr.Accordion(t("Optional Windows ML providers"), open=False):
        initial_winml_choices = windows_ml_choices(
            catalog_status()["providers"],
            gpus,
            saved.get("rembg_winml_provider", "Auto"),
            saved.get("rembg_winml_type", "Auto"),
        )
        winml_provider = gr.Dropdown(
            label=t("Windows ML execution provider"),
            info=t("Explicit Windows ML vendor provider to use for background removal; unsupported providers fall back to CPU."),
            choices=initial_winml_choices,
            value=saved.get("rembg_winml_provider", "Auto"),
            visible=len(initial_winml_choices) > 1,
        )
        add_setting("rembg_winml_provider", winml_provider)
        winml_type = gr.Dropdown(
            label=t("Windows ML device type"),
            choices=list(DEVICE_TYPES),
            value=saved.get("rembg_winml_type", "Auto"),
            info=t(
                "CPU is always available through native ONNX. A selected vendor provider may expose CPU, GPU or NPU devices; unsupported combinations report a reason and recover on CPU."
            ),
        )
        add_setting("rembg_winml_type", winml_type)
        saved_device = saved.get("rembg_winml_device", "Auto")
        initial_device_choices = [("Auto", "Auto")] + (
            [("Saved device (refresh to resolve)", saved_device)]
            if saved_device != "Auto"
            else []
        )
        winml_device = gr.Dropdown(
            label=t("Windows ML device"),
            choices=initial_device_choices,
            value=saved_device,
            visible=len(initial_device_choices) > 1,
            info=t(
                "Provider device identities are separate from PyTorch indices. Refresh explicitly enumerates provider devices; no device probe runs while browsing Settings."
            ),
        )
        add_setting("rembg_winml_device", winml_device)
        with gr.Row():
            refresh_catalogue = gr.Button(t("Refresh local Windows ML catalogue"), size="sm")
            refresh_winml = gr.Button(t("Refresh Windows ML devices"), size="sm", min_width=180)
            install_winml = gr.Button(
                t("Install / verify Windows ML provider"),
                variant="primary",
                size="sm",
                min_width=220,
            )
            cancel_winml = gr.Button(t("Cancel background runtime action"), size="sm")
        winml_status = gr.Markdown(catalog_status()["reason"])

        def provider_state(
            provider_name,
            device_type,
            cpu,
            background_choice,
            request: gr.Request = None,
            *,
            locale=None,
        ):
            locale = locale or resolve_locale(request=request)
            cache = catalog_status()
            requested = localize(
                "Requested provider: {provider_name}; device class: {device_type}.",
                locale,
                provider_name=provider_name,
                device_type=device_type,
            )
            reason = (
                localize(
                    "CPU authority is active; accelerated discovery/install is disabled.", locale
                )
                if auxiliary_cpu(cpu or background_choice == "CPU")
                else cache["reason"]
            )
            updated_choices = windows_ml_choices(
                cache["providers"], gpus, provider_name, device_type
            )
            return gr.update(
                choices=updated_choices,
                value=provider_name,
                visible=len(updated_choices) > 1,
            ), requested + "\n\n" + reason

        localized_output(
            lambda locale, *values: provider_state(*values, locale=locale),
            [winml_provider, winml_type, cpu_everywhere, background_backend],
            [winml_provider, winml_status],
        )

        gr.on(
            triggers=[
                winml_provider.input,
                winml_type.input,
                cpu_everywhere.input,
                background_backend.input,
            ],
            fn=provider_state,
            inputs=[winml_provider, winml_type, cpu_everywhere, background_backend],
            outputs=[winml_provider, winml_status],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

        def refresh_local_catalogue(
            provider_name, device_type, cpu, background_choice, session, request: gr.Request = None
        ):
            from ruinedfooocus.backends.onnx.windows_ml import refresh_catalog

            locale = resolve_locale(request=request)

            if auxiliary_cpu(cpu or background_choice == "CPU"):
                return provider_state(
                    provider_name, device_type, cpu, background_choice, locale=locale
                )
            try:
                session.run_admitted(
                    lambda check: refresh_catalog(check_cancelled=check), timeout=60
                )
                return provider_state(
                    provider_name, device_type, cpu, background_choice, locale=locale
                )
            except Exception as error:
                return gr.skip(), localize(
                    "Local catalogue unavailable: {error}", locale, error=str(error)
                )

        def refresh_winml_devices(
            current,
            provider_name,
            device_type,
            cpu,
            background_choice,
            session,
            request: gr.Request = None,
        ):
            from ruinedfooocus.backends.onnx.windows_ml import discover_devices

            locale = resolve_locale(request=request)

            if (
                auxiliary_cpu(cpu or background_choice == "CPU")
                or device_type == "CPU"
                and provider_name == "Auto"
            ):
                return (
                    gr.update(),
                    localize(
                        "Built-in CPU is authoritative; no GPU/NPU device probe was started.",
                        locale,
                    ),
                )
            try:
                selected_type = validate_windows_ml_choice(provider_name, device_type, gpus)
                if selected_type == "CPU" and provider_name == "Auto":
                    return gr.update(), localize(
                        "Built-in CPU selected; no vendor device probe was started.", locale
                    )
                rows = session.run_admitted(
                    lambda check: discover_devices(
                        provider_name, selected_type, check_cancelled=check
                    ),
                    timeout=60,
                )
                choices = [("Auto", "Auto")] + [(row["label"], row["id"]) for row in rows]
                if current not in {value for _, value in choices}:
                    choices.append(("Saved device (unavailable)", current))
                return gr.update(
                    choices=choices, value=current, visible=len(choices) > 1
                ), localize(
                    "Device discovery complete; no model loaded.", locale
                )
            except Exception as error:
                return gr.update(), localize(
                    "Device discovery unavailable: {error}", locale, error=str(error)
                )

        def install_winml_provider(
            provider_name, device_type, cpu, background_choice, session, request: gr.Request = None
        ):
            from ruinedfooocus.backends.onnx.windows_ml import install_provider, refresh_catalog

            locale = resolve_locale(request=request)

            if auxiliary_cpu(cpu or background_choice == "CPU"):
                yield (
                    gr.update(interactive=True),
                    localize(
                        "CPU authority is active; no provider acquisition or GPU/NPU probe was started.",
                        locale,
                    ),
                    gr.skip(),
                )
                return
            if skip_dependency_install():
                yield (
                    gr.update(interactive=True),
                    localize("Installation is disabled by offline/skip-install policy.", locale),
                    gr.skip(),
                )
                return
            yield (
                gr.update(interactive=False),
                localize("Queued for explicit Windows ML installation…", locale),
                gr.skip(),
            )
            try:
                selected_type = validate_windows_ml_choice(provider_name, device_type, gpus)

                def install(check):
                    message = install_provider(provider_name, selected_type, check_cancelled=check)
                    if not (provider_name == "Auto" and selected_type == "CPU"):
                        refresh_catalog(check_cancelled=check)
                    return message

                result = session.run_admitted(install)
                choices, _ = provider_state(
                    provider_name, device_type, cpu, background_choice, locale=locale
                )
                yield gr.update(interactive=True), result, choices
            except Exception as error:
                yield (
                    gr.update(interactive=True),
                    localize(
                        "Windows ML unavailable; CPU fallback remains active. {error}",
                        locale,
                        error=str(error),
                    ),
                    gr.skip(),
                )

        refresh_winml.click(
            refresh_winml_devices,
            inputs=[
                winml_device,
                winml_provider,
                winml_type,
                cpu_everywhere,
                background_backend,
                action_session,
            ],
            outputs=[winml_device, winml_status],
            show_progress="minimal",
            api_visibility="private",
            concurrency_id="onnx-runtime-install",
            concurrency_limit=1,
        )
        refresh_fn = refresh_support or refresh_support_table
        refresh_args = (
            [*refresh_inputs, background_provider, winml_provider]
            if refresh_inputs is not None
            else None
        )
        install_winml.click(
            install_winml_provider,
            inputs=[winml_provider, winml_type, cpu_everywhere, background_backend, action_session],
            outputs=[install_winml, winml_status, winml_provider],
            trigger_mode="once",
            show_progress="minimal",
            api_visibility="private",
            concurrency_id="onnx-runtime-install",
            concurrency_limit=1,
        ).then(
            refresh_fn,
            inputs=refresh_args,
            outputs=support,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        refresh_catalogue.click(
            refresh_local_catalogue,
            inputs=[winml_provider, winml_type, cpu_everywhere, background_backend, action_session],
            outputs=[winml_provider, winml_status],
            show_progress="hidden",
            api_visibility="private",
        )
        cancel_winml.click(
            lambda session: session.cancel(),
            inputs=action_session,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    return {
        "background_provider": background_provider,
        "provider": winml_provider,
        "device_type": winml_type,
        "install": install_winml_provider,
    }
