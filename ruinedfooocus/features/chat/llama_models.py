"""Chat model choices and local-first model resolution."""

from pathlib import Path
import re
import shutil
import threading
from ruinedfooocus.app.state import path_manager, settings
from ruinedfooocus.features.chat.gguf_metadata import is_projector

GPT_OSS_MODEL = "gpt-oss-20b-MXFP4.gguf"
DEFAULT_MODEL = "LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf"
_catalogue_revision = 0
_catalogue_lock = threading.Lock()


def invalidate_model_catalogue():
    """Known file writes invalidate browsing snapshots, never model verification."""
    global _catalogue_revision
    with _catalogue_lock:
        _catalogue_revision += 1


def _catalogue_context():
    roots = []
    for kind in ("path_llm", "path_vision"):
        folders = path_manager.paths.get(kind, [])
        folders = [folders] if isinstance(folders, str) else folders
        roots.append((kind, tuple(str(path_manager.get_abspath(item)) for item in folders)))
    with _catalogue_lock:
        revision = _catalogue_revision
    return (
        revision,
        id(path_manager),
        tuple(roots),
        tuple(settings.default_settings.get("llama_hidden_models", [])),
        settings.default_settings.get("llama_localfile"),
        tuple(
            (key, value.get("path"), value.get("filename"))
            for key, value in path_manager.DOWNLOADABLE_FILES.items()
            if key.startswith(("llm/", "vision/"))
        ),
    )


class ModelCatalogueCache:
    """One Chat app's three browsing views, refreshed explicitly after file changes."""

    def __init__(self, loader):
        self.loader = loader
        self.lock = threading.RLock()
        self.context = None
        self.views = {}

    def invalidate(self):
        with self.lock:
            self.context = None
            self.views.clear()

    def __call__(self, source=None):
        source = source if source in ("Local models", "Downloadable") else None
        # Coalesce cold callers; a concurrent writer invalidates the result before
        # publication so it cannot become an indefinitely stale browsing snapshot.
        with self.lock:
            while True:
                context = _catalogue_context()
                if self.context != context:
                    self.context = context
                    self.views.clear()
                if source in self.views:
                    return self.views[source]
                catalogue = self.loader(source)
                if context == _catalogue_context():
                    self.views[source] = catalogue
                    return catalogue


def local_models():
    folders = path_manager.paths["path_llm"]
    if isinstance(folders, str):
        folders = [folders]
    roots = [path_manager.get_abspath(folder) for folder in folders]
    files = {}
    hidden = set(settings.default_settings.get("llama_hidden_models", []))
    for root in roots:
        for path in sorted(root.rglob("*.gguf")):
            if path.is_file() and not is_projector(path):
                filename = str(path.resolve())
                if filename not in hidden:
                    files.setdefault(filename, path.name)
    # Imported vision-language GGUF variants belong here too, not just catalogue names.
    vision_folders = path_manager.paths.get("path_vision", [])
    if isinstance(vision_folders, str):
        vision_folders = [vision_folders]
    for folder in vision_folders:
        root = path_manager.get_abspath(folder)
        for path in sorted(root.rglob("*.gguf")):
            filename = str(path.resolve())
            if path.is_file() and filename not in hidden and not is_projector(path):
                files.setdefault(filename, path.name)
    configured = settings.default_settings.get("llama_localfile")
    if configured:
        path = Path(configured).expanduser()
        filename = str(path.resolve())
        if (
            path.is_absolute()
            and path.is_file()
            and path.suffix.lower() == ".gguf"
            and not is_projector(path)
            and filename not in hidden
        ):
            files.setdefault(filename, path.name)
    counts = {}
    for name in files.values():
        counts[name.casefold()] = counts.get(name.casefold(), 0) + 1
    return {
        name if counts[name.casefold()] == 1 else filename: filename
        for filename, name in files.items()
    }


def import_model(filename):
    """Copy a selected GGUF into the app's model folder, keeping the original."""
    path = Path(filename).resolve()
    if not path.is_file():
        raise ValueError("The selected model file is missing.")
    local = local_models()
    existing = next(
        (value for value in local.values() if Path(value).name.casefold() == path.name.casefold()),
        None,
    )
    if existing and str(path) not in local.values():
        raise ValueError("A different model has this filename. Rename one file before importing.")
    if not existing:
        folders = path_manager.paths["path_llm"]
        folder = folders[0] if isinstance(folders, list) else folders
        destination = path_manager.get_abspath(folder).resolve() / path.name
        if destination != path:
            from ruinedfooocus.core.preparation import preparation_service

            destination.parent.mkdir(parents=True, exist_ok=True)
            with preparation_service.artifact_lease(destination, write=True):
                if destination.exists():
                    raise ValueError(
                        "A model with this filename already exists. Rename one file before importing."
                    )
                temporary = destination.with_suffix(".gguf.importing")
                target = temporary.open("xb")
                try:
                    with target, path.open("rb") as source:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    temporary.rename(destination)
                except OSError:
                    temporary.unlink(missing_ok=True)
                    raise
            path = destination
        hidden = settings.default_settings.get("llama_hidden_models", [])
        settings.default_settings["llama_hidden_models"] = [
            item for item in hidden if item != str(path)
        ]
        try:
            settings.save_settings()
        except OSError:
            settings.default_settings["llama_hidden_models"] = hidden
            raise
        finally:
            invalidate_model_catalogue()
    invalidate_model_catalogue()
    for model, choices in model_catalogue().items():
        for _, value in choices:
            if value in (path.name, str(path)):
                from ruinedfooocus.features.chat.sampling_sources import prepare_sampling_source

                prepare_sampling_source(path)
                return model, value
    raise ValueError("Could not add this model to the list.")


def remove_local_model(filename, delete_file=False):
    path = Path(filename).resolve()
    if path.suffix.lower() != ".gguf" or str(path) not in {
        str(Path(item).resolve()) for item in local_models().values()
    }:
        raise ValueError("This model is no longer in the local list. Refresh the list first.")
    if delete_file:
        try:
            from ruinedfooocus.core.preparation import preparation_service

            with preparation_service.artifact_lease(path, write=True):
                path.unlink()
        except PermissionError as error:
            raise ValueError(
                "Cannot delete this file. Load another model or close apps using it, then try again."
            ) from error
    hidden = settings.default_settings.get("llama_hidden_models", [])
    settings.default_settings["llama_hidden_models"] = (
        list(dict.fromkeys([*hidden, str(path)])) if not delete_file else hidden
    )
    try:
        settings.save_settings()
    except OSError as error:
        if delete_file:
            raise ValueError(
                "The file was deleted, but the updated model list could not be saved."
            ) from error
        raise
    finally:
        invalidate_model_catalogue()


def model_catalogue(source=None):
    local = local_models()
    downloadable = path_manager.get_folder_list("llm")
    from ruinedfooocus.features.models.model_registry import load_registry, RegistryError

    try:
        downloadable = list(downloadable) + [
            row["catalogue_key"].removeprefix("vision/")
            for row in load_registry().entries(role="language_model")
            if row["chat_weight"]
            and row["catalogue_key"].startswith("vision/")
            and row["catalogue_key"] in path_manager.DOWNLOADABLE_FILES
        ]
    except RegistryError:
        pass
    from ruinedfooocus.features.chat.model_browser import catalogue_local_file

    roots = {}
    for kind in ("path_llm", "path_vision"):
        folders = path_manager.paths.get(kind, [])
        folders = [folders] if isinstance(folders, str) else folders
        roots[kind] = [path_manager.get_abspath(folder) for folder in folders]
    records = [(name, value, True) for name, value in local.items()]
    records += [
        (name, name, False)
        for name in dict.fromkeys(downloadable)
        if catalogue_local_file(name, path_manager.DOWNLOADABLE_FILES, roots) is None
    ]
    catalogue = {}
    hidden_names = {
        Path(item).name for item in settings.default_settings.get("llama_hidden_models", [])
    }
    for name, value, installed in records:
        if Path(name).name.lower().startswith("mmproj"):
            continue
        if name in hidden_names and not installed:
            continue
        if source == "Local models" and not installed:
            continue
        if source == "Downloadable" and installed:
            continue
        match = re.fullmatch(
            r"(.+?)[.-]((?:I?Q\d|MXFP\d|BF16|F16|F32)[A-Za-z0-9_]*)",
            Path(name).stem,
            flags=re.IGNORECASE,
        )
        model, quant = match.groups() if match else (Path(name).stem, "Original")
        status = "Local" if installed else "Download"
        if installed and name != Path(name).name:
            status += " · " + str(Path(value).parent)
        catalogue.setdefault(model, []).append((f"{quant.upper()} · {status}", value))
    return catalogue


def model_label(name):
    if name == "LFM2.5-1.2B-Instruct":
        return "LFM2.5 1.2B Instruct (candidate)"
    if name == "LFM3-1.7B":
        return "LFM3 1.7B (LiquidAI)"
    if name == "LFM3-1.7B-Uncensored-AD":
        return "AtomicChat LFM3 1.7B Uncensored (Default)"
    if name == "Qwen2.5-7B-Instruct-abliterated-v2":
        return "Qwen 2.5 7B Instruct — abliterated v2"
    if name == "gpt-oss-20b":
        return "GPT-OSS 20B"
    return name.replace("-", " ").replace("_", " ")


def preferred_quant(choices):
    return next((value for label, value in choices if label.startswith("Q4_K_M ")), choices[0][1])


def saved_selection(catalogue, filename):
    """Resolve exact paths first; ambiguous basenames must never select a sibling."""
    for name, choices in catalogue.items():
        if any(value == filename for _, value in choices):
            return name, filename
    if not Path(filename).is_absolute():
        matches = [
            (name, value)
            for name, choices in catalogue.items()
            for _, value in choices
            if Path(value).name == filename
        ]
        if len(matches) == 1:
            return matches[0]
    return Path(filename).stem, filename


def selection_choices(catalogue, model, selected=None, *, preserve=True):
    """Browsing contains actual choices; the active runtime has separate session state."""
    choices = list(catalogue.get(model, []))
    if not preserve:
        selected = preferred_quant(choices) if choices else None
    available = selected is not None and any(value == selected for _, value in choices)
    if not available and selected is not None and preserve:
        # A source/filter refresh must not silently replace a saved absolute path
        # with another file sharing its basename. Keep it visible and unavailable
        # until the user explicitly picks an actual choice.
        choices.append((f"{Path(selected).name} · Unavailable in this view", selected))
    elif not available:
        selected = None
    return choices, selected, available


def resolve_model(name, progress=None):
    path = Path(name).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path
        raise ValueError("The selected local Chat file is missing.")
    for kind in ("llm", "vision"):
        key = kind + "/" + str(name)
        if key in path_manager.DOWNLOADABLE_FILES:
            return path_manager.get_folder_file_path(kind, name, progress=progress)
    if path.is_file():
        return path
    candidates = [
        filename
        for filename in local_models().values()
        if Path(filename).name.casefold() == str(name).casefold()
    ]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple local models have this filename. Select its full path in Chat models."
        )
    if candidates:
        return Path(candidates[0])
    raise ValueError("The selected Chat model is not installed or in the download catalogue.")
