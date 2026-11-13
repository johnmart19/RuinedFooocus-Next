#!/bin/sh
# Quotes, Unicode and newlines stay one argument; Gradio serializes JSON.
# RF_PYTHON selects the existing application interpreter (default: python3).
SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
exec "${RF_PYTHON:-python3}" "$SCRIPT_DIR/api_examples.py" llama "$@"

