"""Build the real UI without serving it, then check every registered event.

Usage (CPU, offline, no port is opened):
    venv/Scripts/python.exe tools/check_ui_wiring.py [--settings <profile>]

For each Gradio event in every Blocks the app builds, check that its input and
output components belong to that app and that its callback accepts the number
of inputs it receives (Gradio-injected Request, Progress and EventData
parameters excluded). Gradio warnings raised while building are reported.
Exits non-zero when wiring problems are found.
"""

import inspect
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Stop(Exception):
    pass


def _injected(parameter, hints):
    import gradio as gr

    kind = hints.get(parameter.name, parameter.annotation)
    if isinstance(kind, str):
        return any(name in kind for name in ("Request", "EventData", "Progress"))
    if isinstance(parameter.default, gr.Progress):
        return True
    try:
        return isinstance(kind, type) and issubclass(kind, (gr.Request, gr.EventData))
    except TypeError:
        return False


def _arity_problem(fn, count):
    from gradio.utils import get_type_hints

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    hints = get_type_hints(fn)
    params = list(signature.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and not _injected(p, hints)
    ]
    required = [p for p in positional if p.default is p.empty]
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return None if count >= len(required) else f"needs >= {len(required)}, gets {count}"
    if count < len(required) or count > len(positional):
        return f"accepts {len(required)}-{len(positional)}, gets {count}"
    return None


def _check(app):
    problems = []
    ids = set(app.blocks)
    for block_fn in app.fns.values():
        fn = block_fn.fn
        name = getattr(fn, "__qualname__", repr(fn)) if fn else "js-only"
        where = ""
        try:
            where = f" ({Path(inspect.getsourcefile(fn)).name}:{inspect.getsourcelines(fn)[1]})"
        except Exception:
            pass
        for role, items in (("input", block_fn.inputs), ("output", block_fn.outputs)):
            for item in items:
                if item._id not in ids:
                    problems.append(f"{name}{where}: {role} {type(item).__name__} not in app")
        if fn is not None and not block_fn.inputs_as_dict:
            issue = _arity_problem(fn, len(block_fn.inputs))
            if issue:
                problems.append(f"{name}{where}: {issue}")
    return problems


def build_app(settings=None):
    """Run normal CPU startup up to Blocks.launch and return (apps, warnings).

    No port is opened. Callers run in their own process: startup changes the
    working directory, sys.argv and global application state.
    """
    import gradio as gr

    built = []
    original = gr.Blocks.launch

    def stop(self, *args, **kwargs):
        built.append(self)
        raise _Stop

    gr.Blocks.launch = stop
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "launch.py",
        "--cpu",
        "--offline",
        "--skip-dependency-install",
        "--nobrowser",
        *(["--settings", settings] if settings else []),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import runpy

            runpy.run_path(str(ROOT / "launch.py"), run_name="__main__")
        except _Stop:
            pass
        finally:
            gr.Blocks.launch = original
    if not built:
        raise RuntimeError("The UI never reached launch.")
    from ruinedfooocus.app import state as shared

    apps = list({id(a): a for a in [*built, getattr(shared, "gradio_root", None)] if a}.values())
    notes = sorted(
        {
            f"{Path(w.filename).name}:{w.lineno}: {str(w.message)[:200]}"
            for w in caught
            if "gradio" in str(w.filename).lower() or "ruinedfooocus" in str(w.filename).lower()
        }
    )
    return apps, notes


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the UI on CPU without serving it and check every event's wiring."
    )
    parser.add_argument("--settings", help="settings profile to build with (default profile)")
    options = parser.parse_args()
    try:
        apps, notes = build_app(options.settings)
    except RuntimeError as error:
        print("FAILED:", error)
        return 1
    problems, events = [], 0
    for app in apps:
        events += len(app.fns)
        problems += _check(app)
    for line in problems:
        print("PROBLEM", line)
    for line in notes:
        print("WARNING", line)
    print(
        f"Checked {events} events in {len(apps)} Blocks: {len(problems)} wiring problems, "
        f"{len(notes)} distinct warnings"
    )
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
