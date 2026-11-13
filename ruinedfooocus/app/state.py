from ruinedfooocus.core.translation_manager import TranslationManager
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.features.generation.defaults import STARTUP_DEFAULTS
from ruinedfooocus.features.models.performance import PerformanceSettings
from ruinedfooocus.features.models.resolutions import ResolutionSettings
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.app.arguments import args
import time
import os

gradio_root = None
server_app = None
local_url = "http://127.0.0.1:7860/"
share_url = "http://127.0.0.1:7860/"

translation_manager = TranslationManager()
translation_manager.set_language(args.language)
translate = translation_manager.translate

from ruinedfooocus.app.resource_lifecycle import RuntimeState

state = RuntimeState(
    {
        "preview_image": None,
        "ctrls_name": [],
        "ctrls_obj": [],
        "setting_name": [],
        "setting_obj": [],
        "pipeline": None,
        "last_config": 0.0,
    }
)

wildcards = None


def update_cfg():
    """Notify controls after a configuration change."""
    state["last_config"] = str(time.time())


settings = SettingsManager(
    args.settings,
    defaults={
        **STARTUP_DEFAULTS,
        "rembg_backend": "Auto",
        "theme": "None",
        "OBP_preset": "Standard",
        "hint_chance": 25,
    },
    on_change=update_cfg,
)
path_manager = PathManager()
models_offline = (
    args.offline
    or os.environ.get("RF_OFFLINE") == "1"
    or settings.default_settings.get("local_model_metadata", False)
)
performance_settings = PerformanceSettings(
    cache_dir=path_manager.model_paths["cache_path"] / "checkpoints", offline=models_offline
)
resolution_settings = ResolutionSettings()
shared_cache = {}
from ruinedfooocus.features.models.model_handler import Models

models = Models(offline=models_offline)


def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]


def get_ctrl(name):
    return state["ctrls_obj"][state["ctrls_name"].index(name)]


def add_setting(name, obj):
    state["setting_name"] += [name]
    state["setting_obj"] += [obj]
