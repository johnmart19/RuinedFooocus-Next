#!/usr/bin/env bash
set -e
printf 'Starting RuinedFooocus...\n'
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

startup_arguments=("$@")

install_system_prerequisites() {
    if [[ "${RF_OFFLINE:-0}" == "1" ]]; then
        echo "Missing system prerequisites must be installed before an offline launch: $*" >&2
        exit 1
    fi
    local option
    for option in "${startup_arguments[@]}"; do
        case "$option" in
            --of*|--sk*|--iINSTallLEDmYOwNPaCKaGeS)
                echo "Missing system prerequisites must be installed before an offline launch: $*" >&2
                exit 1
                ;;
        esac
    done
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Install system prerequisites using your package manager: $*" >&2
        exit 1
    fi
    echo "Installing missing system prerequisites with apt-get: $*. Administrator access is required."
    local elevate=()
    if [[ "$EUID" -ne 0 ]]; then
        elevate=(sudo)
    fi
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y "$@"
}

if command -v dpkg-query >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    if [[ "$(dpkg-query -W -f='${Status}' libgomp1 2>/dev/null || true)" != "install ok installed" ]]; then
        install_system_prerequisites libgomp1
    fi
fi

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -x ./venv/bin/python ]]; then
    python_bin="$PWD/venv/bin/python"
else
    python_bin="python3"
    if ! command -v "$python_bin" >/dev/null 2>&1; then
        if [[ -n "${PYTHON:-}" ]]; then
            echo "The selected PYTHON was not found: $PYTHON" >&2
            exit 1
        fi
        install_system_prerequisites python3 python3-venv
    fi
    if ! "$python_bin" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        echo "RuinedFooocus requires Python 3.10 or newer. Set PYTHON to a supported executable." >&2
        exit 1
    fi
    echo "Creating Python environment in ./venv..."
    if ! "$python_bin" -m venv ./venv; then
        if [[ -n "${PYTHON:-}" ]]; then
            echo "The selected PYTHON needs a working venv module: $PYTHON" >&2
            exit 1
        fi
        install_system_prerequisites python3 python3-venv
        "$python_bin" -m venv ./venv
    fi
    python_bin="$PWD/venv/bin/python"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "The selected PYTHON was not found: $python_bin" >&2
    exit 1
fi
exec "$python_bin" -u entry_with_update.py "$@"
