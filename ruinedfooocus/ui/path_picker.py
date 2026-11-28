"""Owner-only host folder selection; browsing changes drafts, never files."""

import os
from pathlib import Path, PureWindowsPath
import platform
import secrets
import shutil
import stat
import subprocess
from urllib.parse import urlsplit

import gradio as gr

from ruinedfooocus.core.paths import data_path, legacy_model_path
from ruinedfooocus.core.translation_manager import localize, localized_output, translate_ui as t

MULTIPLE = frozenset(
    {
        "path_checkpoints",
        "path_diffusion_models",
        "path_clip",
        "path_loras",
        "path_embeddings",
        "path_gligen",
        "path_hypernetworks",
        "path_wildcards",
        "archive_folders",
    }
)
PATH_KEYS = MULTIPLE | {
    "path_inbox",
    "path_geometry_estimation",
    "path_detection",
    "path_model_patches",
    "path_outputs",
}
MAX_ENTRIES = 256
MAX_SCANNED = 2048


def host_namespace():
    if os.name == "nt":
        return "Windows host filesystem"
    if os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower():
        return "WSL Linux server filesystem"
    return "Linux server filesystem" if platform.system() == "Linux" else "Server filesystem"


def path_lines(value):
    values = value.splitlines() if isinstance(value, str) else (value or [])
    if not isinstance(values, (list, tuple)) or any(not isinstance(item, str) for item in values):
        raise ValueError("Enter one folder path per line.")
    return [item.strip() for item in values if item.strip()]


def validate_namespace(value, *, windows=None):
    if not isinstance(value, str) or not value.strip() or len(value) > 32767:
        raise ValueError("Enter a host folder path.")
    if any(ord(character) < 32 for character in value):
        raise ValueError("Folder paths cannot contain control characters.")
    if "://" in value or value.startswith("file:"):
        raise ValueError("Enter a host folder path, not a browser or file URL.")
    parsed = PureWindowsPath(value)
    windows = os.name == "nt" if windows is None else windows
    if windows:
        if (parsed.drive and not parsed.root) or (parsed.root and not parsed.drive):
            raise ValueError(
                "Use a Windows drive or UNC path, or a relative path; Linux paths belong to the Linux server."
            )
    elif parsed.drive or "\\" in value:
        raise ValueError(
            "Use a Linux server path. In WSL, use /mnt/c/... instead of a Windows drive path."
        )
    return value.strip()


def resolve_path(key, value):
    """Match consumers exactly; the three relative path namespaces differ."""
    value = validate_namespace(value)
    if key == "path_wildcards":
        root = Path(value).expanduser()
        return root.resolve() if root.is_absolute() else data_path(root)
    if key == "archive_folders":
        return Path(value).resolve()
    return legacy_model_path(value)


def validate_path_settings(values, previous):
    """Validate edited syntax without creating roots or rewriting old profiles."""
    for key in PATH_KEYS & values.keys():
        value = values[key]
        old = previous.get(key)
        if key in MULTIPLE:
            roots = path_lines(value)
            if roots == path_lines(old):
                continue
            for root in roots:
                validate_namespace(root)
        elif value != old:
            validate_namespace(value)
    return values


def _loopback(host):
    from ipaddress import ip_address

    try:
        return ip_address(host.strip("[]")).is_loopback
    except (ValueError, AttributeError):
        return str(host).lower() == "localhost"


def _origin(url):
    parsed = urlsplit(str(url))
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def authorize(request, context):
    """Only Gradio's server-supplied request identity grants host enumeration."""
    if (
        getattr(context, "guest", False)
        or not isinstance(request, gr.Request)
        or request.request is None
    ):
        raise PermissionError("Host folders are available only in owner Settings.")
    headers = request.headers
    if headers.get("sec-fetch-site") == "cross-site":
        raise PermissionError("Open folder selection from this app's Settings page.")
    origin = headers.get("origin")
    try:
        if origin and _origin(origin) != _origin(request.url):
            raise PermissionError("Open folder selection from this app's Settings page.")
        local = _loopback(request.client.host) and _loopback(request.url.hostname)
    except (ValueError, AttributeError):
        raise PermissionError("Cannot verify the folder selection request.") from None
    forwarded = any(
        headers.get(name)
        for name in ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")
    )
    local = local and not forwarded
    auth = getattr(context, "auth", None) or os.environ.get("RF_UI_AUTH")
    if auth:
        expected = auth.partition("/")[0]
        if not isinstance(request.username, str) or not secrets.compare_digest(
            request.username.encode("utf-8"), expected.encode("utf-8")
        ):
            raise PermissionError("Owner login is required to browse host folders.")
    elif not local:
        raise PermissionError("Owner login is required for remote host folder browsing.")
    # Missing Origin (for example an API request) can use the browser fallback,
    # but cannot open a desktop dialog. Forwarded requests never open dialogs.
    return local and bool(origin) and _loopback(urlsplit(origin).hostname)


def native_available():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Flags(ctypes.Structure):
            _fields_ = [
                ("inherit", wintypes.BOOL),
                ("reserved", wintypes.BOOL),
                ("flags", wintypes.DWORD),
            ]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetProcessWindowStation.restype = wintypes.HANDLE
        user32.GetUserObjectInformationW.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        flags = Flags()
        visible = user32.GetUserObjectInformationW(
            user32.GetProcessWindowStation(), 1, ctypes.byref(flags), ctypes.sizeof(flags), None
        )
        return bool(visible and flags.flags & 1 and shutil.which("powershell.exe"))
    if os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower():
        return False
    return bool(
        (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        and (shutil.which("zenity") or shutil.which("kdialog"))
    )


def native_choose(initial):
    """Run only after the explicit local-owner action and desktop checks."""
    if os.name == "nt":
        script = """$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$folderPicker = New-Object System.Windows.Forms.FolderBrowserDialog
$folderPicker.Description = 'Select RuinedFooocus host folder'
$folderPicker.ShowNewFolderButton = $false
$folderPicker.SelectedPath = $env:RF_FOLDER_PICKER_START
try { if ($folderPicker.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Write($folderPicker.SelectedPath)
} } finally { $folderPicker.Dispose() }
"""
        command = [
            shutil.which("powershell.exe"),
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-Command",
            script,
        ]
        options = {
            "creationflags": subprocess.CREATE_NO_WINDOW,
            "env": dict(os.environ, RF_FOLDER_PICKER_START=str(initial)),
        }
    elif shutil.which("zenity"):
        command = [
            shutil.which("zenity"),
            "--file-selection",
            "--directory",
            "--title=Select server folder",
            "--filename=" + str(initial) + "/",
        ]
        options = {}
    elif shutil.which("kdialog"):
        command = [
            shutil.which("kdialog"),
            "--getexistingdirectory",
            str(initial),
            "--title",
            "Select server folder",
        ]
        options = {}
    else:
        raise OSError("A native folder chooser is unavailable.")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
        **options,
    )
    if result.returncode and not (os.name != "nt" and result.returncode == 1):
        raise OSError("The native folder chooser could not open.")
    if result.returncode == 1 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def existing_directory(key, value):
    path = resolve_path(key, value)
    if not stat.S_ISDIR(path.stat().st_mode):
        raise ValueError("Choose an existing host folder.")
    with os.scandir(path):
        pass
    if key == "path_outputs" and not os.access(path, os.W_OK):
        raise PermissionError("Choose a writable output folder.")
    return path


def directory_choices(path):
    entries = []
    limited = False
    with os.scandir(path) as iterator:
        for index, entry in enumerate(iterator):
            if index >= MAX_SCANNED:
                limited = True
                break
            try:
                if entry.is_dir(follow_symlinks=True):
                    entries.append((entry.name, str(Path(path) / entry.name)))
            except OSError:
                continue
            if len(entries) > MAX_ENTRIES:
                limited = True
                break
    return sorted(entries[:MAX_ENTRIES], key=lambda item: item[0].casefold()), limited


def initial_directory(key, value):
    try:
        candidate = resolve_path(key, value)
    except (ValueError, OSError):
        candidate = data_path()
    for path in (candidate, *candidate.parents):
        try:
            return existing_directory(key, str(path))
        except (ValueError, OSError):
            continue
    raise ValueError("Enter an accessible host folder in the advanced path field.")


def apply_choice(key, value, index, choice, *, append=False):
    chosen = str(existing_directory(key, choice))
    if key not in MULTIPLE:
        return chosen, "0"
    roots = path_lines(value)
    selected = int(index or 0) if key in MULTIPLE else 0
    selected = min(max(selected, 0), max(len(roots) - 1, 0))
    retained = (
        roots if append else [root for number, root in enumerate(roots) if number != selected]
    )

    def same_folder(root):
        candidate = resolve_path(key, root)
        try:
            return candidate.samefile(chosen)
        except OSError:
            return candidate == Path(chosen)

    if any(same_folder(root) for root in retained):
        raise ValueError("That folder is already in this list.")
    if append or not roots:
        roots.append(chosen)
        selected = len(roots) - 1
    else:
        roots[selected] = chosen
    return "\n".join(roots), str(selected)


def edit_order(value, index, action):
    roots = path_lines(value)
    selected = min(max(int(index or 0), 0), max(len(roots) - 1, 0))
    if roots:
        if action == "remove":
            roots.pop(selected)
            selected = min(selected, max(len(roots) - 1, 0))
        else:
            target = max(0, min(selected + (-1 if action == "up" else 1), len(roots) - 1))
            roots[selected], roots[target] = roots[target], roots[selected]
            selected = target
    return "\n".join(roots), str(selected)


def path_help(locale, multiple, base):
    prefix = (
        localize("One host folder per line, in priority order.", locale) + " " if multiple else ""
    )
    return prefix + localize("Relative paths start at: {base}", locale, base=str(base))


def create_path_control(key, label, value, *, context=None):
    """Return the sole persisted field; browser state stays per session/control."""
    multiple = key in MULTIPLE
    roots = path_lines(value)

    def summary(value, index="0"):
        lines = path_lines(value)
        if not multiple:
            return gr.update(value=lines[0] if lines else "")
        index = str(min(max(int(index or 0), 0), max(len(lines) - 1, 0))) if lines else None
        return gr.update(
            choices=[(f"{number + 1}. {path}", str(number)) for number, path in enumerate(lines)],
            value=index,
        )

    def owner(request):
        active = context
        if active is None:
            from ruinedfooocus.app.arguments import args

            active = args
        return authorize(request, active)

    def checked(function):
        def wrapped(request: gr.Request, *values):
            try:
                local = owner(request)
                return function(*values, local=local)
            except (ValueError, OSError, subprocess.TimeoutExpired) as error:
                raise gr.Error(str(error)) from error

        return wrapped

    with gr.Group(elem_classes=["rf-folder-picker"]):
        if multiple:
            selected = gr.Dropdown(
                label=label,
                choices=summary("\n".join(roots))["choices"],
                value="0" if roots else None,
                interactive=True,
                min_width=0,
            )
        else:
            selected = gr.Textbox(
                label=label, value=roots[0] if roots else "", interactive=False, min_width=0
            )
        with gr.Row() as folder_actions:
            select = gr.Button(t("Select folder"), size="sm", min_width=100)
            if multiple:
                add = gr.Button(t("Add folder"), size="sm", min_width=90)
                remove = gr.Button(t("Remove folder"), size="sm", min_width=90)
                up = gr.Button(t("Move up"), size="sm", min_width=80)
                down = gr.Button(t("Move down"), size="sm", min_width=80)
        with gr.Accordion(t("Advanced path"), open=False):
            base = (
                data_path()
                if key == "path_wildcards"
                else (Path.cwd() if key == "archive_folders" else legacy_model_path("."))
            )
            field = gr.Textbox(
                label=label,
                value="\n".join(roots),
                lines=2 if multiple else 1,
                max_lines=4,
                interactive=True,
                info=path_help("en", multiple, base),
            )
            localized_output(
                lambda locale: gr.update(info=path_help(locale, multiple, base)), [], [field]
            )
            native_dialog = gr.Button(t("Open host dialog"), size="sm")
        # The system folder dialog is one click away, next to the in-page browser.
        with folder_actions:
            native_dialog.unrender().render()
        status = gr.Markdown("")
        mode = gr.State("replace")
        with gr.Group(visible=False, elem_classes=["rf-folder-browser"]) as browser:
            gr.Markdown(t(host_namespace()))
            browser_path = gr.Textbox(label=t("Host folder"), interactive=True)
            with gr.Row():
                browse = gr.Button(t("Open path"), size="sm")
                parent = gr.Button(t("Parent folder"), size="sm")
            children = gr.Dropdown(label=t("Subfolders"), choices=[], interactive=True)
            with gr.Row():
                open_child = gr.Button(t("Open subfolder"), size="sm")
                use = gr.Button(t("Use folder"), size="sm", variant="primary")
                cancel = gr.Button(t("Cancel"), size="sm")

    def listing(value):
        path = existing_directory(key, value)
        choices, limited = directory_choices(path)
        message = t("Showing some subfolders. Enter a path to browse others.") if limited else ""
        return str(path), gr.update(choices=choices, value=None), message

    def start(value, index, *, local, append=False):
        lines = path_lines(value)
        number = min(max(int(index or 0), 0), max(len(lines) - 1, 0)) if multiple else 0
        initial = initial_directory(key, lines[number] if lines else str(data_path()))
        fallback = ""
        try:
            can_open_native = local and native_available()
        except OSError:
            can_open_native = False
        if can_open_native:
            try:
                choice = native_choose(initial)
            except (OSError, subprocess.TimeoutExpired):
                fallback = t("Native dialog unavailable. Choose a host folder below.")
            else:
                if choice is None:
                    return (
                        gr.skip(),
                        gr.skip(),
                        t("Folder selection cancelled."),
                        gr.update(visible=False),
                        gr.skip(),
                        gr.skip(),
                        "replace",
                    )
                value, index = apply_choice(key, value, index, choice, append=append)
                return (
                    value,
                    summary(value, index),
                    t("Folder selected. Save and restart to apply."),
                    gr.update(visible=False),
                    gr.skip(),
                    gr.skip(),
                    "replace",
                )
        path, choices, message = listing(str(initial))
        return (
            gr.skip(),
            gr.skip(),
            fallback
            or message
            or (t("Choose a folder to add.") if append else t("Choose a replacement folder.")),
            gr.update(visible=True),
            path,
            choices,
            "add" if append else "replace",
        )

    def select_clicked(value, index, *, local):
        return start(value, index, local=False)

    def add_clicked(value, index, *, local):
        return start(value, index, local=False, append=True)

    def native_clicked(value, index, action, *, local):
        return start(value, index, local=local, append=action == "add")

    def open_path(value, *, local):
        return listing(value)

    def open_parent(value, *, local):
        return listing(str(existing_directory(key, value).parent))

    def use_folder(value, index, chosen, action, *, local):
        value, index = apply_choice(key, value, index, chosen, append=action == "add")
        return (
            value,
            summary(value, index),
            t("Folder selected. Save and restart to apply."),
            gr.update(visible=False),
            "replace",
        )

    field.input(
        lambda value, index: summary(value, index),
        inputs=[field, selected],
        outputs=selected,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    outputs = [field, selected, status, browser, browser_path, children, mode]
    event = dict(show_progress="hidden", api_visibility="undocumented")
    select.click(
        checked(select_clicked),
        inputs=[field, selected],
        outputs=outputs,
        queue=False,
        **event,
    )
    native_dialog.click(
        checked(native_clicked),
        inputs=[field, selected, mode],
        outputs=outputs,
        concurrency_id="host-folder-dialog",
        **event,
    )
    if multiple:
        add.click(
            checked(add_clicked),
            inputs=[field, selected],
            outputs=outputs,
            queue=False,
            **event,
        )
        for button, action in ((remove, "remove"), (up, "up"), (down, "down")):

            def reorder(value, index, action=action):
                value, index = edit_order(value, index, action)
                return value, summary(value, index)

            button.click(
                reorder, inputs=[field, selected], outputs=[field, selected], queue=False, **event
            )
    for button, function, component in (
        (browse, open_path, browser_path),
        (parent, open_parent, browser_path),
        (open_child, open_path, children),
    ):
        button.click(
            checked(function),
            inputs=component,
            outputs=[browser_path, children, status],
            queue=False,
            **event,
        )
    use.click(
        checked(use_folder),
        inputs=[field, selected, browser_path, mode],
        outputs=[field, selected, status, browser, mode],
        queue=False,
        **event,
    )
    cancel.click(
        lambda: (gr.update(visible=False), t("Folder selection cancelled."), "replace"),
        outputs=[browser, status, mode],
        queue=False,
        **event,
    )
    return field
