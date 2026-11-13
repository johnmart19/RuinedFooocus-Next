"""Profile-bound configuration storage without application or UI imports."""

import copy

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.paths import settings_path
from ruinedfooocus.core.profile_store import (
    read_profile,
    profile_name,
    profile_folder,
    save_profile,
)


class SettingsManager:
    def __init__(self, profile=None, *, defaults=None, on_change=None):
        self.name = profile_name(profile)
        self._defaults = copy.deepcopy(defaults or {})
        self._on_change = on_change
        self.set_settings_path(profile)
        self.load_settings()

    def set_settings_path(self, subfolder=None):
        subfolder = profile_name(subfolder)
        self.subfolder = subfolder
        self.settings_path = settings_path(subfolder)

    def load_settings(self):
        path = self.settings_path
        values = read_profile(path)
        changed = False
        if "resolution_auto" in self._defaults and "resolution_auto" not in values:
            # Existing profiles own their saved size; only a fresh profile opts
            # into following model recommendations without an explicit choice.
            values["resolution_auto"] = "resolution" not in values
            changed = True
        for key, value in self._defaults.items():
            if key not in values:
                values[key] = copy.deepcopy(value)
                changed = True
        for key in ("style", "archive_folders"):
            if key in values and not isinstance(values[key], list):
                values[key] = [values[key]]
                changed = True
        self.default_settings = values
        if changed:
            save_json(path, values)

    def save_settings(self):
        save_json(self.settings_path, self.default_settings)
        if self._on_change:
            self._on_change()

    def save_profile(self, values, paths, *, profile=None, on_publish=None):
        """Publish new memory/profile identity only after the paired disk commit."""
        profile = profile_name(profile)
        save_profile(profile_folder(profile), values, paths)
        self.default_settings = copy.deepcopy(values)
        self.name = profile
        self.set_settings_path(profile)
        if on_publish:
            on_publish()
        if self._on_change:
            self._on_change()


def prospective_profile(current, current_paths, values, path_keys):
    """Route only explicit form metadata and known path fields; retain ui_* prefs."""
    prospective, paths = copy.deepcopy(current), copy.deepcopy(current_paths)
    for key, value in values.items():
        if key == "ui_settings_name":
            continue
        target = paths if key in path_keys else prospective
        if value is None:
            target.pop(key, None)
        else:
            target[key] = copy.deepcopy(value)
    return prospective, paths
