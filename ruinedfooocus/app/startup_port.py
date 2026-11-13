from ruinedfooocus.core.paths import APP_ROOT

"""Resolve listen-port conflicts before loading models or starting Gradio."""

import os
import socket
from pathlib import Path


def ruinedfooocus_owner(port):
    """Only recognize Python launchers from this checkout, never a process name alone."""
    try:
        import psutil

        owners = {
            c.pid
            for c in psutil.net_connections(kind="tcp")
            if c.status == psutil.CONN_LISTEN and c.laddr.port == port
        }
        if len(owners) != 1 or None in owners or os.getpid() in owners:
            return None
        process = psutil.Process(owners.pop())
        if not process.name().lower().startswith(("python", "pypy")):
            return None
        command = process.cmdline()
        # Inspect only the interpreter's script argument; do not log arguments
        # (which may contain API keys). -c and -m launches are not identifiable.
        index = 1
        while index < len(command) and command[index] in ("-u", "-B", "-E", "-s", "-S"):
            index += 1
        if index >= len(command) or command[index].startswith("-"):
            return None
        script = Path(command[index])
        if not script.is_absolute():
            script = Path(process.cwd()) / script
        root = APP_ROOT
        if script.resolve() not in {
            root / name
            for name in (
                "launch.py",
                "ruinedfooocus/app/bootstrap.py",
                "entry_with_update.py",
                "ruinedfooocus/ui/app.py",
            )
        }:
            return None
        process.create_time()  # Cache identity so terminate() guards against PID reuse.
        return process
    except (ImportError, OSError):
        return None
    except Exception:
        # Process permissions or a disappearing listener must never prevent startup.
        return None


def offer_close_instance(host, port):
    process = ruinedfooocus_owner(port)
    if process is None:
        return False
    print(
        f"RuinedFooocus from this folder is already listening on port {port} (PID {process.pid})."
    )
    print("Closing it stops any active generation and disconnects its users.")
    try:
        answer = input("Close that instance and use this port? [y/N]: ").strip().lower()
    except (EOFError, OSError, KeyboardInterrupt):
        return False
    if answer not in ("y", "yes"):
        return False
    try:
        current = ruinedfooocus_owner(port)
        if current is None or current != process:
            print("The port owner changed; no process was stopped.")
            return False
        process.terminate()
        process.wait(timeout=10)
    except Exception:
        print("Could not close the existing instance. Close its terminal or choose another port.")
        return False
    if available(host, port):
        print(f"Previous instance closed. Reusing port {port}.")
        return True
    print("The port is still unavailable. Choose another port or cancel.")
    return False


def available(host, port):
    try:
        addresses = socket.getaddrinfo(host.strip("[]"), port, type=socket.SOCK_STREAM)
        for family, kind, protocol, _, address in addresses:
            with socket.socket(family, kind, protocol) as probe:
                if os.name == "nt":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                else:
                    # Match the server's POSIX restart behavior. TIME_WAIT
                    # connections are not listeners; live listeners still fail.
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(address)
        return True
    except OSError:
        return False


def configure_port(args):
    host = args.listen or os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    explicit = args.port is not None
    try:
        port = args.port if explicit else int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
        if not 1 <= port <= 65535:
            raise ValueError
    except (ValueError, TypeError):
        print("Invalid listen port. Use --port with a number from 1 to 65535.")
        raise SystemExit(1)
    if available(host, port):
        args.port = port
        return
    if offer_close_instance(host, port):
        args.port = port
        return
    # Preserve Gradio's automatic port selection when --port was not supplied.
    candidate = next(
        (p for p in range(port + 1, min(port + 101, 65536)) if available(host, p)), None
    )
    if not explicit and candidate:
        args.port = candidate
        print(f"Port {port} is unavailable; using port {candidate}.")
        return
    print(
        f"\nWarning: {host}:{port} is occupied or unavailable. Another RuinedFooocus may already be running."
    )
    while True:
        print("[1] Use another port  [Q] Cancel")
        try:
            choice = input("Choose [Q]: ").strip().lower()
            if choice in ("", "q"):
                print("Startup cancelled. The existing server was left running.")
                raise SystemExit(0)
            if choice != "1":
                continue
            entered = input(f"New port{f' [{candidate}]' if candidate else ''}: ").strip()
            selected = int(entered) if entered else candidate
            if selected is None or not 1 <= selected <= 65535 or not available(host, selected):
                print("That port is invalid or unavailable. Choose another.")
                continue
        except ValueError:
            print("Enter a port number from 1 to 65535.")
            continue
        except (EOFError, OSError, KeyboardInterrupt):
            print("\nStartup cancelled. Relaunch with --port <available-port>.")
            raise SystemExit(1)
        args.port = selected
        print(
            f"Starting on port {selected}" + (" with API enabled." if args.api else " without API.")
        )
        return
