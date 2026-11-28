from ruinedfooocus.core.paths import data_path
import json
from pathlib import Path
from ruinedfooocus.core.config_io import save_json
from .data_store import user_file


class OneButtonPresets:
    DEFAULT_OBP_FILE = Path(__file__).with_name("presets") / "obp_presets.default"
    OBP_FILE = None  # Resolve the active RF_DATA_DIR at instance creation.
    CUSTOM_OBP = "Custom..."
    RANDOM_PRESET_OBP = "All (random)..."
    # Renamed bundled presets; saved files and settings keep working.
    RENAMED = {"WIERD HATS": "Weird Hats"}

    def __init__(self):
        self.OBP_FILE = self.OBP_FILE or data_path("random_prompt", "userfiles", "obp_presets.json")
        self.opb_presets = self.load_obp_presets()

    def load_obp_presets(self):
        try:
            default_data = self._load_data(self.DEFAULT_OBP_FILE)
        except (OSError, ValueError):
            print(f"ERROR: Failed to load {self.DEFAULT_OBP_FILE}")
            default_data = {}
        try:
            source = self.OBP_FILE if self.OBP_FILE.exists() else user_file("obp_presets.json")
            data = self._load_data(source) if source else {}
        except (OSError, ValueError):
            print(f"ERROR: Failed to load {self.OBP_FILE}")
            data = {}

        for old, new in self.RENAMED.items():
            if old in data and new not in data:
                data[new] = data.pop(old)

        for name, settings in default_data.items():
            if name not in data:
                data[name] = settings
            elif "enhancement_focus" in settings:
                data[name].setdefault("enhancement_focus", settings["enhancement_focus"])

        # Sanity check
        for name, settings in data.items():
            if settings.get("subject") == "------ all":
                settings["subject"] = "all"
            # Upgrade unchanged bundled anime text, preserving custom preset edits.
            if name in ("Waifu's", "Husbando's"):
                bundled = default_data.get(name, {})
                for key, old in {
                    "prefixprompt": "(((masterpiece))), (((best quality))), anime style, 2d,",
                    "suffixprompt": "key visual",
                }.items():
                    if settings.get(key) == old and key in bundled:
                        settings[key] = bundled[key]

        return data

    def _load_data(self, file_path):
        with open(file_path, encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, dict) or any(
            not isinstance(value, dict) for value in data.values()
        ):
            raise ValueError(f"{file_path}: presets must map names to settings objects")
        return data

    def _save_data(self, file_path, data):
        save_json(file_path, data)

    def save_obp_preset(self, perf_options):
        self._save_data(self.OBP_FILE, perf_options)
        self.opb_presets = self.load_obp_presets()

    def resolve_name(self, name):
        # If both names were customized, a saved selection must keep its own values.
        return name if name in self.opb_presets else self.RENAMED.get(name, name)

    def get_obp_preset(self, name):
        name = self.resolve_name(name)
        # Saved settings or API calls may name a deleted/renamed preset; fall back to Standard.
        if name in self.opb_presets:
            return self.opb_presets[name]
        if name not in ("", self.CUSTOM_OBP, self.RANDOM_PRESET_OBP):
            print(f"One Button preset {name!r} not found, using Standard.")
        return self.opb_presets.get("Standard") or next(iter(self.opb_presets.values()), {})
