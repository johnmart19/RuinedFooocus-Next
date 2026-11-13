"""Lightweight profile I/O and recoverable paired-document saves.

The undo journal is durable before either document changes. Its removal is the
commit boundary. A failed/interrupted save is rolled back before either reader
uses the profile. This is recovery, not a claim of filesystem-wide atomicity.
"""

import base64
import json
from pathlib import Path
from threading import RLock

from ruinedfooocus.core.config_io import save_bytes, save_json
from ruinedfooocus.core.paths import data_path

_lock = RLock()


def profile_name(name):
    if name in (None, "", "default"):
        return None
    if not isinstance(name, str) or name != name.strip() or name.endswith("."):
        raise ValueError("Profile name: avoid leading/trailing spaces or trailing dots.")
    if any(c in name for c in '\\/:*?"<>|') or any(ord(c) < 32 for c in name):
        raise ValueError("Profile name: use a folder name without path separators.")
    stem = name.split(".", 1)[0].upper()
    if name in (".", "..") or stem in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError("Profile name: this name is reserved on Windows.")
    return name


def profile_folder(name=None):
    name = profile_name(name)
    return data_path("settings", name) if name else data_path("settings")


def read_object(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid profile JSON: {path}: {error}") from error
    if not isinstance(values, dict):
        raise ValueError(f"Settings must contain a JSON object: {path}")
    return values


def recover_profile(folder):
    with _lock:
        folder = Path(folder)
        journal = folder / ".profile-save.json"
        if not journal.exists():
            return
        undo = read_object(journal)
        if undo.get("version") != 1 or set(undo.get("before", {})) != {
            "settings.json",
            "paths.json",
        }:
            raise ValueError(f"Invalid profile recovery journal: {journal}")
        for name, previous in undo["before"].items():
            target = folder / name
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                save_bytes(target, base64.b64decode(previous, validate=True))
        journal.unlink()


def read_profile_data(path):
    """Recover and read a JSON object without importing application feature validators.

    Startup policy validates its own limited fields. Application settings readers
    must continue to use read_profile for complete known-field validation.
    """
    with _lock:
        recover_profile(Path(path).parent)
        return read_object(path)


def read_profile(path):
    with _lock:
        values = read_profile_data(path)
        from ruinedfooocus.core.settings_schema import validate_known

        return validate_known(values)


def read_profile_pair(profile=None):
    """Recover and read one profile's settings/paths under the existing lock."""
    with _lock:
        folder = profile_folder(profile)
        return read_profile(folder / "settings.json"), read_object(folder / "paths.json")


def save_profile(folder, settings, paths):
    """Commit both documents, or restore their exact previous bytes."""
    with _lock:
        folder = Path(folder)
        recover_profile(folder)
        # Serialize both before creating the journal or touching the documents.
        for values in (settings, paths):
            json.dumps(values, allow_nan=False)
        before = {
            name: base64.b64encode((folder / name).read_bytes()).decode("ascii")
            if (folder / name).exists()
            else None
            for name in ("settings.json", "paths.json")
        }
        journal = folder / ".profile-save.json"
        save_json(journal, {"version": 1, "before": before})
        try:
            save_json(folder / "settings.json", settings)
            save_json(folder / "paths.json", paths)
            journal.unlink()
        except BaseException:
            # If recovery itself fails, leave the journal for the next reader.
            recover_profile(folder)
            raise


def selected_profile(explicit=None):
    if explicit is not None:
        return profile_name(explicit)
    return profile_name(read_object(data_path("settings", "startup.json")).get("profile"))


def use_next_launch(name):
    name = profile_name(name)
    if not (profile_folder(name) / "settings.json").is_file():
        raise ValueError("Save this profile before selecting it for the next launch.")
    save_json(data_path("settings", "startup.json"), {"profile": name})
