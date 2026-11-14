"""Fire every side-effect-free UI event on the real app, as a browser would.

Usage (CPU, offline, no port is opened):
    venv/Scripts/python.exe tools/check_ui_events.py [--settings <profile>] [--list-skipped]
        [--actions]

With --actions, install, switch, save, delete, dialog and similar actions are
fired too, under guards: starting installers, git, build tools or PowerShell,
connecting beyond this machine, restarting, and deleting or writing outside the
private data and temp folders all fail the check. Offline, every install action
must refuse with a message. Generation and Chat sends stay excluded.

The app is built by the normal startup (see check_ui_wiring.build_app). Each event
whose callback is not an install, download, delete, file dialog, model load,
generation, restart or similar action is sent through Gradio's own
process_api with its inputs' initial values, exactly like a browser event.
A gr.Error is a deliberate user-facing message and counts as handled; any
other exception is reported as a problem. Exits non-zero when problems exist.
"""

import asyncio
import inspect
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Actions with real side effects outside the private data folder, or that start
# model, generation or install work. Their behavior is covered by focused tests.
SIDE_EFFECTS = re.compile(
    r"install|reinstall|build|rollback|use_pin|download|delete|remove|purge|restart|"
    r"shutdown|import_gguf|dialog|select_folder|browse|generat|launch|send|submit|"
    r"load_model|unload|warm|transfer|acquire|switch|verify|toggle_cpu|interrupt|"
    r"caption|describe|enhance|rewrite|inspect|export|save|cancel|stop|run_|^run$|"
    r"poke|confirm|apply_update|update_llama|refresh_all|trigger_|use_model|"
    r"local_model_action|prepare|start|retry|slideshow|share|with_attachments",
    re.IGNORECASE,
)


# Display/status callbacks whose names contain an action word but only read state.
READ_ONLY = re.compile(
    r"(^|\.)(describe_performance|generation_context|update_llama_status|update_xllama_status|"
    r"update_cancel_action|share_chat_model|refresh_local|refresh_presets)$"
)


# Always excluded: generation and Chat sends start model compute.
COMPUTE = re.compile(r"generate_|send|submit|with_attachments|slideshow", re.IGNORECASE)


# File pickers open the system dialog through PowerShell; that is their purpose.
FILE_DIALOGS = re.compile(r"(^|\.)(import_gguf|select_gguf|native_dialog|open_host_dialog)$")


class BlockedSideEffect(Exception):
    """An event tried to act outside the private test environment."""


def _guard(allowed_roots):
    """Install process, network, exit and file guards for --actions runs."""
    import builtins
    import os
    import shutil
    import socket
    import subprocess

    roots = [Path(root).resolve() for root in allowed_roots]
    blocked_words = (
        "pip",
        "install",
        "cmake",
        "git",
        "nvcc",
        "winget",
        "powershell",
        "curl",
        "wget",
    )

    def inside(path):
        try:
            resolved = Path(path).resolve()
        except (OSError, TypeError, ValueError):
            return False
        return any(resolved == root or root in resolved.parents for root in roots)

    original_popen = subprocess.Popen.__init__

    def popen(self, args, *rest, **kwargs):
        command = " ".join(map(str, args)) if isinstance(args, (list, tuple)) else str(args)
        if any(word in command.lower() for word in blocked_words):
            raise BlockedSideEffect(f"started a process: {command[:120]}")
        return original_popen(self, args, *rest, **kwargs)

    subprocess.Popen.__init__ = popen

    original_connect = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise BlockedSideEffect(f"connected to {host}")
        return original_connect(self, address)

    socket.socket.connect = connect

    def refuse(name):
        def blocked(*args, **kwargs):
            raise BlockedSideEffect(f"called {name}")

        return blocked

    for name in ("execv", "execve", "execl", "_exit"):
        if hasattr(os, name):
            setattr(os, name, refuse("os." + name))

    def guarded_remove(original, name):
        def remove(path, *args, **kwargs):
            if not inside(path):
                raise BlockedSideEffect(f"{name} outside the test folders: {path}")
            return original(path, *args, **kwargs)

        return remove

    os.remove = guarded_remove(os.remove, "remove")
    os.unlink = guarded_remove(os.unlink, "unlink")
    shutil.rmtree = guarded_remove(shutil.rmtree, "rmtree")
    original_unlink = Path.unlink

    def path_unlink(self, missing_ok=False):
        if not inside(self):
            raise BlockedSideEffect(f"unlink outside the test folders: {self}")
        return original_unlink(self, missing_ok=missing_ok)

    Path.unlink = path_unlink
    original_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and any(flag in mode for flag in "wax+"):
            if not inside(file):
                raise BlockedSideEffect(f"wrote outside the test folders: {file}")
        return original_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open


def _unwrap(fn):
    fn = getattr(fn, "__wrapped__", fn)  # ruinedfooocus.ui.user_errors wrapper
    return getattr(fn, "func", fn)  # functools.partial


def _name(fn):
    target = _unwrap(fn)
    return getattr(target, "__qualname__", repr(target))


def _where(fn):
    target = _unwrap(fn)
    try:
        return f"{Path(inspect.getsourcefile(target)).name}:{inspect.getsourcelines(target)[1]}"
    except (OSError, TypeError):
        return "?"


def _payload(block):
    import gradio as gr

    if isinstance(block, gr.State):
        return None
    value = getattr(block, "value", None)
    if callable(value):
        return None
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


def _unreachable(block_fn, app):
    """Events a browser cannot fire with initial values: API-only routes, and
    selections or uploads on a component that holds nothing to select or upload."""
    targets = [(target, event) for target, event in block_fn.targets]
    if not targets or all(target is None and event == "api" for target, event in targets):
        return "API route"
    if all(event in ("select", "upload") for _, event in targets):
        from gradio.components import Component

        values = [
            _payload(app.blocks[t])
            for t, _ in targets
            if t is not None and t in app.blocks and isinstance(app.blocks[t], Component)
        ]
        if not values:
            return None  # tabs and other layout blocks have no value to select
        if all(value in (None, [], {}, "") for value in values):
            return "empty selection or upload"
    return None


def _needs_page_data(block_fn):
    """Callbacks reading custom event data sent by the page's own JavaScript."""
    import gradio as gr
    from gradio.utils import get_type_hints

    if all(event == "select" for _, event in block_fn.targets):
        return False
    try:
        parameters = inspect.signature(block_fn.fn).parameters.values()
        hints = get_type_hints(block_fn.fn)
    except (TypeError, ValueError):
        return False
    for parameter in parameters:
        kind = hints.get(parameter.name, parameter.annotation)
        if isinstance(kind, type) and issubclass(kind, gr.EventData):
            return True
    return False


def _event_data(block_fn, app):
    import gradio as gr

    for target, event in block_fn.targets:
        if event == "select" and target is not None and target in app.blocks:
            # A browser selects a real item: send the first one, as Gradio would.
            items = _payload(app.blocks[target])
            first = items[0] if isinstance(items, list) and items else None
            return gr.SelectData(app.blocks[target], {"index": 0, "value": first, "selected": True})
    return None


async def _fire(app, index, block_fn, state, timeout):
    import gradio as gr

    inputs = [_payload(block) for block in block_fn.inputs]
    event_data = _event_data(block_fn, app)
    iterator = None
    for _ in range(50):
        result = await asyncio.wait_for(
            app.process_api(
                block_fn=block_fn,
                inputs=inputs,
                state=state,
                request=gr.Request(session_hash="ui-event-check"),
                iterator=iterator,
                event_data=event_data,
                session_hash="ui-event-check",
            ),
            timeout,
        )
        if not result.get("is_generating"):
            return
        iterator = result.get("iterator")
        if iterator is None:
            return


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fire every side-effect-free UI event on the real app with initial values."
    )
    parser.add_argument("--settings", help="settings profile to build with (default profile)")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per event")
    parser.add_argument("--list-skipped", action="store_true", help="print skipped callbacks")
    parser.add_argument(
        "--actions", action="store_true", help="also fire guarded side-effect actions"
    )
    options = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools.check_ui_wiring import build_app

    apps, _ = build_app(options.settings)
    import gradio as gr

    if options.actions:
        import os
        import tempfile

        data = os.environ.get("RF_DATA_DIR")
        if not data:
            print("FAILED: --actions needs RF_DATA_DIR set to a private folder")
            return 1
        _guard([data, tempfile.gettempdir()])
    from gradio.state_holder import SessionState

    counts = dict(
        ok=0, rejected=0, problem=0, timeout=0, skipped=0, js=0, page=0, unreachable=0, dialog=0
    )
    problems, skipped = [], set()
    started = time.monotonic()
    for app in apps:
        state = SessionState(app)
        for index, block_fn in list(app.fns.items()):
            if block_fn.fn is None:
                counts["js"] += 1
                continue
            name = _name(block_fn.fn)
            reason = _unreachable(block_fn, app)
            if reason:
                counts["unreachable"] += 1
                skipped.add(f"{name} ({_where(block_fn.fn)}) [{reason}]")
                continue
            if _needs_page_data(block_fn):
                counts["page"] += 1
                skipped.add(f"{name} ({_where(block_fn.fn)}) [page event data]")
                continue
            if COMPUTE.search(name) or (
                SIDE_EFFECTS.search(name) and not READ_ONLY.search(name) and not options.actions
            ):
                counts["skipped"] += 1
                skipped.add(f"{name} ({_where(block_fn.fn)})")
                continue
            try:
                asyncio.run(_fire(app, index, block_fn, state, options.timeout))
                counts["ok"] += 1
            except gr.Error:
                counts["rejected"] += 1
            except BlockedSideEffect as error:
                if FILE_DIALOGS.search(name) and "powershell" in str(error).lower():
                    counts["dialog"] += 1
                    continue
                counts["problem"] += 1
                problems.append(f"{name} ({_where(block_fn.fn)}): blocked: {error}")
            except (asyncio.TimeoutError, TimeoutError):
                counts["timeout"] += 1
                problems.append(f"{name} ({_where(block_fn.fn)}): no result in {options.timeout}s")
            except Exception as error:  # noqa: BLE001 - every unexpected failure is reported
                counts["problem"] += 1
                problems.append(
                    f"{name} ({_where(block_fn.fn)}): {type(error).__name__}: {str(error)[:200]}"
                )
    for line in problems:
        print("PROBLEM", line)
    if options.list_skipped:
        for line in sorted(skipped):
            print("SKIPPED", line)
    print(
        "Fired {ok} events cleanly, {rejected} with user messages, {dialog} opening a file "
        "dialog (blocked), {problem} problems, "
        "{timeout} timeouts; skipped {skipped} side-effect actions, {page} events needing "
        "page data, {unreachable} API routes or empty selections/uploads and {js} "
        "browser-only events".format(**counts)
        + f" in {time.monotonic() - started:.0f}s"
    )
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
