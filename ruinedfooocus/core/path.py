from ruinedfooocus.core.paths import APP_ROOT, data_path, resource_path, legacy_model_path
from pathlib import Path
import json
import os
import errno
import re
from urllib.parse import unquote, urlsplit
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.model_folders import FOLDER_SETTINGS, canonical_model_root, model_folders
from ruinedfooocus.core.model_formats import (
    NATIVE_STATE_DICT_SUFFIXES,
    NATIVE_TORCH_SUFFIXES,
    SAFETENSORS_SUFFIXES,
)


def _registered_pinned_url(info, artifact, file_key):
    original, pinned = info["url"], artifact.get("pinned_url")
    if pinned is None or pinned == original:
        return original
    valid = (
        isinstance(pinned, str)
        and type(artifact.get("bytes")) is int
        and artifact["bytes"] > 0
        and isinstance(artifact.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None
        and not any(ord(char) < 32 or ord(char) == 127 for char in original + pinned)
    )
    if valid:
        source, target = urlsplit(original), urlsplit(pinned)
        source_parts, target_parts = (
            unquote(source.path).split("/"),
            unquote(target.path).split("/"),
        )
        valid = (
            source.scheme == target.scheme == "https"
            and source.netloc == target.netloc == "huggingface.co"
            and not target.query
            and not target.fragment
            and len(source_parts) >= 6
            and len(target_parts) >= 6
            and source_parts[3] == target_parts[3] == "resolve"
            and source_parts[1:3] == target_parts[1:3]
            and source_parts[5:] == target_parts[5:]
            and not any(part in ("", ".", "..") for part in target_parts[1:])
            and re.fullmatch(r"[0-9a-f]{40}", target_parts[4]) is not None
        )
    if not valid:
        raise ValueError(
            f"The pinned publisher source is incompatible with the registered tuple: {file_key}"
        )
    return pinned


class PathManager:
    DEFAULT_PATHS = {
        "path_checkpoints": ["../models/checkpoints/"],
        "path_diffusion_models": ["../models/diffusion_models/"],
        "path_diffusers": "../models/diffusers/",
        "path_diffusers_cache": "../models/diffusers_cache/",
        "path_loras": ["../models/loras/"],
        "path_embeddings": ["../models/embeddings/"],
        "path_controlnet": "../models/controlnet/",
        "path_model_patches": "../models/model_patches/",
        "path_gligen": ["../models/gligen/"],
        "path_hypernetworks": ["../models/hypernetworks/"],
        "path_geometry_estimation": "../models/geometry_estimation/",
        "path_detection": "../models/detection/",
        "path_background_removal": "../models/background_removal/",
        "path_vae_approx": "../models/vae_approx/",
        "path_vae": "../models/vae/",
        "path_preview": "../outputs/preview.jpg",
        "path_faceswap": "../models/faceswap/",
        "path_upscalers": "../models/upscale_models",
        "path_latent_upscalers": "../models/latent_upscale_models",
        "path_outputs": "../outputs/",
        "path_clip": "../models/text_encoders/",
        "path_clip_vision": "../models/clip_vision/",
        "path_style_models": "../models/style_models/",
        "path_photomaker": "../models/photomaker/",
        "path_cache": "../cache/",
        "path_llm": "../models/llm",
        "path_vision": "../models/vision",
        "path_inbox": "../models/inbox",
        "path_presets": "presets",
    }

    EXTENSIONS = [*NATIVE_TORCH_SUFFIXES, *SAFETENSORS_SUFFIXES, ".gguf"]

    # Add a dictionary to store file download information
    DOWNLOADABLE_FILES = {}

    name = None
    settings_path = None
    paths = None

    def __init__(self):
        from ruinedfooocus.app.arguments import args

        self.name = args.settings
        self.set_settings_path(args.settings)
        self.paths = self.load_paths()
        self.model_paths = self.get_model_paths()
        self.upscaler_filenames = self.get_model_filenames(
            self.model_paths["upscaler_path"], extensions=NATIVE_STATE_DICT_SUFFIXES
        )

        pathdb_folder = resource_path("model_catalogue")
        files = os.listdir(pathdb_folder)
        for file in files:
            # Check if the file has a .json extension
            if file.endswith(".json"):
                file_path = os.path.join(pathdb_folder, file)

                try:
                    # Open and read the JSON file
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        self.DOWNLOADABLE_FILES.update(data)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    def set_settings_path(self, subfolder=None):
        self.subfolder = subfolder
        if self.subfolder in [None, "", "default"]:
            path = data_path("settings", "paths.json")
        else:
            path = data_path("settings", self.subfolder, "paths.json")
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path = path

    def load_paths(self):
        from ruinedfooocus.core.profile_store import read_object, recover_profile

        recover_profile(self.settings_path.parent)
        paths = self.DEFAULT_PATHS.copy()
        if self.settings_path.exists():
            paths.update(read_object(self.settings_path))
        for key in self.DEFAULT_PATHS:
            if key not in paths:
                paths[key] = self.DEFAULT_PATHS[key]
        # Fix paths
        for key in ["path_checkpoints", "path_diffusion_models", "path_loras", "path_embeddings"]:
            if key in paths and not isinstance(paths[key], list):  # Some folders should be lists
                paths[key] = [paths[key]]

        save_json(self.settings_path, paths)
        return paths

    def save_paths(self):
        paths = self.paths

        #        for key in newpaths:
        #            if key not in paths:
        #                paths[key] = newpaths[key]
        save_json(self.settings_path, paths)
        return paths

    def get_model_paths(self):
        clip_folders = self.get_model_folders("clip")
        if not clip_folders:
            raise ValueError("path_clip: provide at least one folder.")
        model_roots = list(
            dict.fromkeys(
                self.get_model_folders("checkpoints") + self.get_model_folders("diffusion_models")
            )
        )
        # Keep legacy roots searchable without recreating removed alias folders.
        for category in ("checkpoints", "diffusion_models"):
            for root in self.get_model_folders(category):
                self.get_abspath_folder(canonical_model_root(category, root, self.get_abspath))
        return {
            "modelfile_path": model_roots,
            "diffusers_path": self.get_abspath_folder(self.paths["path_diffusers"]),
            "diffusers_cache_path": self.get_abspath_folder(
                self.paths["path_diffusers_cache"], required=True
            ),
            "lorafile_path": self.get_abspath_folder(self.paths["path_loras"]),
            # Embedding discovery must not create/move user folders.
            "embedding_path": self.get_model_folders("embeddings"),
            "controlnet_path": self.get_abspath_folder(
                self.get_model_download_folder("controlnet")
            ),
            "geometry_estimation_path": self.get_abspath_folder(
                self.paths["path_geometry_estimation"]
            ),
            "detection_path": self.get_abspath_folder(self.paths["path_detection"]),
            "vae_approx_path": self.get_abspath_folder(self.paths["path_vae_approx"]),
            "vae_path": self.get_abspath_folder(self.paths["path_vae"]),
            "temp_outputs_path": self.get_abspath_folder(self.paths["path_outputs"], required=True),
            "temp_preview_path": self.get_abspath(self.paths["path_preview"]),
            "faceswap_path": self.get_abspath_folder(self.paths["path_faceswap"]),
            "upscaler_path": self.get_abspath_folder(self.paths["path_upscalers"]),
            "latent_upscaler_path": self.get_abspath_folder(self.paths["path_latent_upscalers"]),
            # Older callers use this scalar as a cache/download fallback. File
            # lookup and native registration still search every configured root.
            "clip_path": self.get_abspath_folder(self.get_model_download_folder("clip")),
            "clip_vision_path": self.get_abspath_folder(self.paths["path_clip_vision"]),
            "style_models_path": self.get_abspath_folder(self.paths["path_style_models"]),
            "photomaker_path": self.get_abspath_folder(self.paths["path_photomaker"]),
            "cache_path": self.get_abspath_folder(self.paths["path_cache"], required=True),
            "llm_path": self.get_abspath_folder(self.paths["path_llm"]),
            "inbox_path": self.get_abspath_folder(self.paths["path_inbox"]),
            "preset_path": self.get_abspath_folder(
                data_path(self.paths["path_presets"]), required=True
            ),
        }

    def get_model_folders(self, folder):
        return model_folders(self.paths, folder, self.get_abspath)

    def get_model_download_folder(self, folder):
        roots = self.get_model_folders(folder)
        if not roots:
            raise ValueError(
                f"path_{FOLDER_SETTINGS.get(folder, folder)}: provide at least one folder."
            )
        return canonical_model_root(folder, roots[0], self.get_abspath)

    def get_abspath_folder(self, path, required=False):
        def prepare(folder):
            directory = self.get_abspath(folder)
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except OSError as error:
                    if required or error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                        raise
                    # Shared model roots can be read-only. Downloads and output
                    # writes still report their own destination errors when used.
                    print(f"Directory is unavailable for writing: {directory}")
            return directory

        if isinstance(path, list):
            rc = [prepare(folder) for folder in path]
        else:
            rc = prepare(path)
        return rc

    def get_abspath(self, path):
        return legacy_model_path(path)

    def get_model_filenames(self, folder_path, cache=None, isLora=False, *, extensions=None):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            return []
        extensions = self.EXTENSIONS if extensions is None else extensions
        filenames = []
        for path in folder_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                if isLora:
                    txtcheck = path.with_suffix(".txt")
                    if txtcheck.exists():
                        fstats = txtcheck.stat()
                        if fstats.st_size > 0:
                            path = path.with_suffix(f"{path.suffix}")
                filenames.append(str(path.relative_to(folder_path)))
        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        # if it is a plain file. This will sort folders above files in the dropdown
        return sorted(
            filenames,
            key=lambda x: (
                f"0{x.casefold()}" if not str(Path(x).parent) == "." else f"1{x.casefold()}"
            ),
        )

    def get_presets(self):
        presets = {}
        # User presets override the bundled example without copying either file.
        for folder in (
            resource_path("generation_presets"),
            APP_ROOT / "presets",
            data_path(self.paths["path_presets"]),
        ):
            if folder.is_dir():
                for path in folder.rglob("*.png"):
                    presets[path.relative_to(folder)] = path
        names = sorted(presets, key=lambda name: (len(name.parts) == 1, str(name).casefold()))
        return [(presets[name], name.stem) for name in names]

    def get_file_path(self, file_key, default=None, progress=None):
        """
        Get the path for a file, downloading it if it doesn't exist.
        """
        folder, separator, filename = str(file_key).partition("/")
        folder = FOLDER_SETTINGS.get(folder, folder)
        if separator:
            for directory in self.get_model_folders(folder):
                candidate = directory / filename
                if candidate.is_file():
                    return candidate
        if file_key not in self.DOWNLOADABLE_FILES and separator:
            # Older catalogs use aliases (e.g. lcm_lora) instead of folder/name.
            file_key = next(
                (
                    key
                    for key, entry in self.DOWNLOADABLE_FILES.items()
                    if entry["path"] == "path_" + folder and entry["filename"] == filename
                ),
                file_key,
            )
        if file_key not in self.DOWNLOADABLE_FILES:
            return default

        file_info = self.DOWNLOADABLE_FILES[file_key]
        # Catalog aliases (e.g. clip_l or cn_depth) share the same root lookup.
        for directory in self.get_model_folders(file_info["path"].removeprefix("path_")):
            candidate = directory / file_info["filename"]
            if candidate.is_file():
                return candidate
        folder = self.get_model_download_folder(file_info["path"].removeprefix("path_"))
        file_path = folder / file_info["filename"]

        if not file_path.exists():
            self.download_file(file_key, progress=progress, destination=file_path)

        return file_path

    def get_folder_file_path(self, folder, filename, default=None, progress=None):
        return self.get_file_path(
            f"{str(folder)}/{str(filename)}", default=default, progress=progress
        )

    def get_folder_list(self, folder):
        folder = FOLDER_SETTINGS.get(folder, folder)
        result = []
        for file in self.DOWNLOADABLE_FILES:
            if self.DOWNLOADABLE_FILES[file]["path"] == "path_" + folder:
                result.append(self.DOWNLOADABLE_FILES[file]["filename"])
        for root in self.get_model_folders(folder):
            result.extend(
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in self.EXTENSIONS
            )
        return sorted(set(result), key=str.casefold)

    def preparation_request(self, file_key, *, destination=None):
        """Capture catalog metadata and the destination before settings can change."""
        from ruinedfooocus.core.preparation import PreparationRequest

        info = self.DOWNLOADABLE_FILES[file_key].copy()
        from ruinedfooocus.features.models.model_registry import load_registry, RegistryError

        try:
            registry = load_registry()
        except RegistryError:
            registry = None  # Optional metadata cannot remove the original download contract.
        if registry is not None:
            annotation = registry.annotations_for(file_key, info)
            if annotation is not None:
                for field, published in (
                    ("size", annotation["artifact"]["bytes"]),
                    ("sha256", annotation["artifact"]["sha256"]),
                ):
                    explicit = info.get(field)
                    if (
                        explicit is not None
                        and published is not None
                        and (str(explicit).lower() != str(published).lower())
                    ):
                        raise ValueError(f"Conflicting {field} metadata for {file_key}")
                    if explicit is None and published is not None:
                        info[field] = published
                info["url"] = _registered_pinned_url(info, annotation["artifact"], file_key)
        if destination is None:
            folder = self.get_model_download_folder(info["path"].removeprefix("path_"))
            destination = folder / info["filename"]
        return PreparationRequest(
            info["url"], destination, expected_size=info.get("size"), sha256=info.get("sha256")
        )

    def prepare_file(self, file_key, progress=None, *, destination=None):
        """Start preparation without taking a compute slot; return a caller-owned handle."""
        from ruinedfooocus.core.preparation import preparation_service

        request = self.preparation_request(file_key, destination=destination)
        return preparation_service.submit(request, progress)

    def download_file(self, file_key, progress=None, *, destination=None):
        """Synchronous compatibility entry point; asynchronous callers use prepare_file."""
        from ruinedfooocus.core.preparation import preparation_service

        request = self.preparation_request(file_key, destination=destination)
        return preparation_service.submit(request, progress).result()
