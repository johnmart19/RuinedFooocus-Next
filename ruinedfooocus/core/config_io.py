"""Write configuration without truncating the last working file on failure."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def save_json(path, data):
    save_bytes(path, json.dumps(data, indent=2, allow_nan=False).encode("utf-8"))


def save_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
