"""Application resources and writable data, independent of the launch directory.

The default data root preserves existing clone installations. RF_DATA_DIR allows
an archive/read-only application to keep writable state somewhere else. Legacy
model path values are relative to the former modules/ directory by design.
"""

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = APP_ROOT / "ruinedfooocus"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
ASSET_ROOT = PACKAGE_ROOT / "ui" / "assets"


def source_checkout(root):
    """Git is a file in a worktree, a directory in a conventional clone."""
    return (Path(root) / ".git").exists()


def data_path(*parts):
    configured = os.environ.get("RF_DATA_DIR")
    base = Path(configured).expanduser() if configured else APP_ROOT
    if not base.is_absolute():
        base = APP_ROOT / base
    return base.joinpath(*parts).resolve()


def resource_path(*parts):
    return RESOURCE_ROOT.joinpath(*parts)


def settings_path(profile=None):
    if profile in (None, "", "default"):
        return data_path("settings", "settings.json")
    return data_path("settings", profile, "settings.json")


def asset_path(name):
    return ASSET_ROOT / name


def startup_marker(name):
    """Read legacy clone markers, but create new markers in writable data."""
    writable = data_path(name)
    legacy = APP_ROOT / name
    return legacy if not writable.exists() and legacy.exists() else writable


def legacy_model_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (data_path() / "modules" / path).resolve()


def content_roots(kind, legacy):
    """User entries override bundled entries with the same relative identity.

    New user locations take precedence over the old folder; neither is moved or
    modified by discovery. The bundled folder is always read-only to the loader.
    """
    return (resource_path(kind), data_path(legacy), data_path(kind))
