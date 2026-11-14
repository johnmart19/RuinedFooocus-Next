"""Sort stable, individually identifiable model files into configured roots."""

import errno
import os
from pathlib import Path
import re
import shutil
import stat
import struct
from tempfile import TemporaryDirectory
import time

from ruinedfooocus.core.model_formats import NATIVE_STATE_DICT_SUFFIXES, is_safetensors
from ruinedfooocus.features.models.model_identity import cached_file_sha256, file_signature
from ruinedfooocus.features.models.model_registry import RegistryError, load_registry


MODEL_SUFFIXES = (*NATIVE_STATE_DICT_SUFFIXES, ".gguf")
SIDECAR_SUFFIXES = (
    ".civitai.info",
    ".json",
    ".txt",
    ".jpeg",
    ".jpg",
    ".png",
    ".gif",
    ".webp",
    ".preview.jpeg",
    ".preview.jpg",
    ".preview.png",
    ".preview.gif",
    ".preview.webp",
)
PUBLISHER_CATEGORIES = {
    "Checkpoint": "checkpoints",
    "LORA": "loras",
    "DoRA": "loras",
    "LoCon": "loras",
    "TextualInversion": "embeddings",
    "Controlnet": "controlnet",
}
LOCAL_UTILITY_CATEGORIES = {
    "DepthAnything3": "geometry_estimation",
    "RT_DETR_v4": "checkpoints",
    "SAM3": "checkpoints",
    "SAM31": "checkpoints",
}
FLAT_CATEGORIES = {"background_removal", "faceswap"}


def _signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError("not a regular local file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _image_category(header, gguf_arch=None):
    from ruinedfooocus.features.models.model_architecture import inspect_tensor_model
    from ruinedfooocus.features.models.image_components import (
        ALIASES,
        GGUF_ARCHITECTURES,
        IMAGE_COMPONENTS,
    )

    from ruinedfooocus.features.models.model_architecture import native_detector_ready

    image = inspect_tensor_model(header, include_config=True) if native_detector_ready() else None
    if image:
        native_class = image["native_class"]
        if native_class == "MiniMaxH3Image":
            return "diffusion_models"
        if native_class in LOCAL_UTILITY_CATEGORIES:
            return LOCAL_UTILITY_CATEGORIES[native_class] if gguf_arch is None else None
        architecture = image["architecture"]
        family = ALIASES.get(architecture, architecture)
        if family in IMAGE_COMPONENTS:
            if family in {"SD_X4Upscaler", "HunyuanImage21Refiner"}:
                return "checkpoints"
            embedded = (*image["vae_prefixes"], *image["text_encoder_prefixes"])
            if any(prefix and name.startswith(prefix) for prefix in embedded for name in header):
                return "checkpoints"
            return "diffusion_models"
    if gguf_arch in GGUF_ARCHITECTURES:
        return "diffusion_models"
    return None


def _local_category(path):
    from ruinedfooocus.features.models.model_identity import cached_safetensors_header

    if path.suffix.lower() == ".gguf":
        from ruinedfooocus.features.models.gguf_metadata import (
            read_gguf_header,
            read_scalar_strings,
        )

        try:
            header, architecture = read_gguf_header(path)
            meta = read_scalar_strings(
                path,
                {"general.type", "adapter.type", "tokenizer.chat_template"},
                max_string_bytes=64 * 1024,
            )
        except (IndexError, OverflowError, struct.error) as error:
            raise ValueError("Invalid GGUF model header") from error
        if meta.get("general.type") == "adapter":
            if meta.get("adapter.type") == "lora" and any(
                name.endswith(".lora_a") and name.removesuffix(".lora_a") + ".lora_b" in header
                for name in header
            ):
                return "loras"
            return None
        if (
            meta.get("general.type", "model") == "model"
            and meta.get("tokenizer.chat_template")
            and "token_embd.weight" in header
            and any(name.startswith("blk.0.") for name in header)
        ):
            return "llm"
        if (
            architecture
            and any(name != "__metadata__" for name in header)
            and meta.get("general.type", "model") == "model"
        ):
            return _image_category(header, gguf_arch=architecture)
        return None
    if not is_safetensors(path):
        return None  # Legacy Torch files require a verified publisher identity.

    from safetensors import SafetensorError, safe_open

    # Validate the data extent as well as the header before moving a partial copy.
    try:
        with safe_open(path, framework="np"):
            pass
    except SafetensorError as error:
        raise ValueError("Invalid or incomplete safetensors file") from error
    header = cached_safetensors_header(path)
    from ruinedfooocus.features.models.native_workflow_admission import planned_image_family

    if planned_image_family(path) == "MiniMaxH3Image":
        return "diffusion_models"
    keys = set(header) - {"__metadata__"}
    if not keys:
        return None
    if any(
        name.endswith(".lora_down.weight")
        and name.removesuffix(".lora_down.weight") + ".lora_up.weight" in keys
        or name.endswith(".lora_A.weight")
        and name.removesuffix(".lora_A.weight") + ".lora_B.weight" in keys
        for name in keys
    ):
        return "loras"
    if keys <= {"clip_l", "clip_g", "emb_params"} and all(
        len(header[key].get("shape", ())) in (1, 2) for key in keys
    ):
        return "embeddings"
    if {"controlnet_cond_embedding.conv_in.weight", "controlnet_down_blocks.0.weight"} <= keys or {
        "control_model.input_hint_block.0.weight",
        "control_model.zero_convs.0.0.weight",
    } <= keys:
        return "controlnet"
    if "bb.layers.1.blocks.0.attn.relative_position_index" in keys:
        from ruinedfooocus.features.image_processing.native_background import validate_model

        validate_model(path)
        return "background_removal"
    if category := _image_category(header):
        return category

    from ruinedfooocus.features.models.component_candidates import classify_shapes

    shapes = {name: tuple(header[name]["shape"]) for name in keys}
    components = classify_shapes(shapes)
    if components.get("vae_channels") and components.get("vae_contract_valid", True):
        return "vae"
    if components.get("text_type") and components.get("text_contract_valid", True):
        return "text_encoders"
    return None


def _publish(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno not in (
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EINVAL,
        ):
            raise
        # Copy across volumes to a hidden staging directory before publishing.
        with TemporaryDirectory(prefix=".rf-inbox-", dir=target.parent) as directory:
            temporary = Path(directory) / "payload"
            shutil.copy2(source, temporary)
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)


def _move_group(files, model_stamp=None):
    stamps = [(source, target, _signature(source)) for source, target in files]
    if model_stamp is not None and stamps[-1][2] != model_stamp:
        raise OSError("model changed before sorting")
    published = []
    try:
        for source, target, _ in stamps:
            _publish(source, target)
            published.append((target, _signature(target)))
        if any(_signature(source) != stamp for source, _, stamp in stamps):
            raise OSError("an inbox file changed during sorting")
    except (OSError, ValueError):
        for target, stamp in reversed(published):
            try:
                if _signature(target)[:2] == stamp[:2]:
                    target.unlink()
            except (OSError, ValueError):
                pass
        raise
    # Every destination is complete before removing any source. If cleanup
    # fails, preserve both copies so a later scan cannot lose data.
    for source, _, stamp in stamps:
        if _signature(source) != stamp:
            raise OSError("an inbox file changed before source cleanup")
        source.unlink()


class InboxSorter:
    def __init__(self, manager, *, offline):
        self.manager = manager
        self.offline = offline
        self.seen = {}
        self.hash_cache = Path(manager.model_paths["cache_path"]) / "hashes"
        self.catalogue_by_size = {}
        try:
            registry = load_registry()
            for key, download in manager.DOWNLOADABLE_FILES.items():
                row = registry.annotations_for(key, download)
                artifact = row["artifact"] if row else {}
                size = artifact.get("bytes") or download.get("size")
                digest = artifact.get("sha256") or download.get("sha256")
                if size and digest and Path(download["filename"]).suffix.lower() in MODEL_SUFFIXES:
                    category = download["path"].removeprefix("path_")
                    suffix = Path(download["filename"]).suffix.lower()
                    self.catalogue_by_size.setdefault(size, {}).setdefault(
                        digest.lower(), set()
                    ).add((category, suffix))
        except RegistryError as error:
            print(f"Model Inbox catalogue unavailable: {error}")

    def _category(self, path, size):
        matches = self.catalogue_by_size.get(size)
        digest = cached_file_sha256(path, self.hash_cache).lower() if matches else None
        if matches and digest in matches:
            categories = {
                category
                for category, suffix in matches[digest]
                if path.suffix.lower() == suffix
                or (is_safetensors(path) and is_safetensors("model" + suffix))
            }
            return next(iter(categories)) if len(categories) == 1 else None
        category = _local_category(path)
        if category or self.offline:
            return category

        from ruinedfooocus.features.models.model_sources import civitai_metadata

        digest = digest or cached_file_sha256(path, self.hash_cache).lower()
        data = civitai_metadata(f"model-versions/by-hash/{digest}") or {}
        model, artifacts = data.get("model"), data.get("files")
        if isinstance(model, dict) and isinstance(artifacts, list):
            for artifact in artifacts:
                hashes = artifact.get("hashes") if isinstance(artifact, dict) else None
                if isinstance(hashes, dict) and str(hashes.get("SHA256", "")).lower() == digest:
                    return PUBLISHER_CATEGORIES.get(model.get("type"))
        return None

    def scan(self, folders):
        moved, current = 0, {}
        now = time.monotonic()
        roots = list(dict.fromkeys(Path(folder).resolve() for folder in folders))
        for root in roots:
            for directory, subdirs, filenames in os.walk(root):
                parent = Path(directory)
                subdirs[:] = [
                    name
                    for name in subdirs
                    if not (parent / name).is_symlink()
                    and not getattr(parent / name, "is_junction", lambda: False)()
                ]
                if any(
                    name.lower() == "model_index.json"
                    or name.lower().endswith((".safetensors.index.json", ".bin.index.json"))
                    for name in filenames
                ):
                    subdirs.clear()
                    continue
                for name in filenames:
                    path = parent / name
                    if path.suffix.lower() not in MODEL_SUFFIXES or re.search(
                        r"-\d+-of-\d+\.[^.]+$", name, re.IGNORECASE
                    ):
                        continue
                    try:
                        stamp = _signature(path)
                        previous, attempted = self.seen.get(path, (None, None))
                        if previous != stamp:
                            current[path] = stamp, None
                            continue
                        if attempted is not None and now - attempted < 300:
                            current[path] = stamp, attempted
                            continue
                        current[path] = stamp, now
                        if stamp[2] == 0:
                            continue
                        before = file_signature(path)
                        category = self._category(path, stamp[2])
                        if not category:
                            if attempted is None:
                                print(f"Model Inbox kept {path.name}: model category is unknown")
                            continue
                        if _signature(path) != stamp or file_signature(path) != before:
                            continue
                        target_root = Path(
                            self.manager.get_model_download_folder(category)
                        ).resolve()
                        relative = (
                            path.name if category in FLAT_CATEGORIES else path.relative_to(root)
                        )
                        target = target_root / relative
                        resolved = target.parent.resolve()
                        if not resolved.is_relative_to(target_root) or any(
                            resolved.is_relative_to(inbox) for inbox in roots
                        ):
                            raise ValueError("destination is outside its category or inside Inbox")
                        sidecars = [
                            (path.with_suffix(suffix), target.with_suffix(suffix))
                            for suffix in SIDECAR_SUFFIXES
                            if os.path.lexists(path.with_suffix(suffix))
                        ]
                        if sidecars and any(
                            other != name
                            and Path(other).stem.casefold() == path.stem.casefold()
                            and Path(other).suffix.lower() in MODEL_SUFFIXES
                            for other in filenames
                        ):
                            raise ValueError("several models share these sidecar filenames")
                        exact_sidecars = [
                            (parent / (name + suffix), target.parent / (name + suffix))
                            for suffix in SIDECAR_SUFFIXES
                            if os.path.lexists(parent / (name + suffix))
                        ]
                        files = [*sidecars, *exact_sidecars, (path, target)]
                        if any(os.path.lexists(destination) for _, destination in files):
                            raise FileExistsError("a destination model or sidecar already exists")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _move_group(files, stamp)
                        moved += 1
                        current.pop(path, None)
                        print(f"Model Inbox moved {path.name} to {target}")
                    except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as error:
                        print(f"Model Inbox kept {path.name}: {error}")
        self.seen = current
        return moved
