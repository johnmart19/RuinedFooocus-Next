import requests
import base64
import hashlib
import shutil
import os
import cv2
import json
import threading
import time
import sys
from pathlib import Path
import numpy as np

from ruinedfooocus.app.state import translate as t
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.core.model_formats import (
    NATIVE_STATE_DICT_SUFFIXES,
    NATIVE_TORCH_SUFFIXES,
    SAFETENSORS_SUFFIXES,
)


class Models:
    def civit_update_worker(self, model_type, folder_paths):
        with self._metadata_lock:
            if (
                model_type in self.civit_workers
                or time.monotonic() - self._metadata_checked.get(model_type, float("-inf")) < 300
            ):
                return
            self.civit_workers.append(str(model_type))
        try:
            self.cache_paths[model_type].mkdir(parents=True, exist_ok=True)
            self._update_models(model_type, folder_paths)
        except Exception as error:
            print(f"Model update failed for {model_type}: {error}")
        finally:
            from ruinedfooocus.app.state import shared_cache

            for key in list(shared_cache):
                if isinstance(key, Path) and key.parent == self.cache_paths[model_type]:
                    shared_cache.pop(key, None)
            self.ready[model_type] = True
            with self._metadata_lock:
                self.civit_workers.remove(str(model_type))
                self._metadata_checked[model_type] = time.monotonic()
            self.revision += 1

    def _update_models(self, model_type, folder_paths):
        if model_type == "embeddings":
            return self._update_embeddings(folder_paths)

        updated = 0

        # Go through and check previews
        seen_names = set()
        for folder in folder_paths:
            for path in folder.rglob("*"):
                if path.suffix.lower() in self.EXTENSIONS:
                    name = str(path.relative_to(folder))
                    if name in seen_names:
                        continue
                    seen_names.add(name)
                    # get file name, add cache path change suffix
                    cache_file = Path(self.cache_paths[model_type] / path.name)
                    from ruinedfooocus.features.models.model_previews import (
                        SUFFIXES,
                        VERSION,
                        is_warning,
                        local_preview,
                        valid_preview,
                    )

                    suffixes = list(SUFFIXES)
                    local = local_preview(
                        path,
                        ambiguous=sum(
                            Path(candidate).with_suffix("").as_posix().casefold()
                            == Path(name).with_suffix("").as_posix().casefold()
                            for candidate in self.get_names(model_type)
                        )
                        > 1,
                    )
                    has_preview = False
                    for suffix in suffixes:
                        thumbcheck = cache_file.with_suffix(suffix)
                        if (
                            thumbcheck.is_file()
                            and valid_preview(thumbcheck)
                            and not is_warning(thumbcheck)
                        ):
                            # Retain valid artwork if its online refresh fails.
                            has_preview = True
                            break

                    if local:
                        if not has_preview:
                            from ruinedfooocus.features.models.model_previews import write_preview

                            write_preview(Path(local).read_bytes(), cache_file)
                        has_preview = True
                    # A display hit must also prevent unnecessary remote artwork
                    # and weight hashing in the background presentation scan.
                    model_data = self.get_models_by_path(
                        model_type, path, fetch=not has_preview, discover_hf=False
                    )

                    # Rebuild legacy low-resolution/animated network thumbnails
                    # once, preserving the old valid file if fetching fails.
                    marker = cache_file.with_suffix(".preview.json")
                    try:
                        preview_state = json.loads(marker.read_text())
                        preview_state = preview_state if isinstance(preview_state, dict) else {}
                    except (OSError, ValueError):
                        preview_state = {}
                    real_artwork = any(
                        isinstance(item, dict)
                        and item.get("url")
                        and "cdn-thumbnails.huggingface.co/social-thumbnails/" not in item["url"]
                        for item in (model_data.get("images") or [])
                    )
                    replace_generated = preview_state.get("generated") and real_artwork
                    if (
                        has_preview
                        and not local
                        and not self.offline
                        and (preview_state.get("version") != VERSION or replace_generated)
                        and model_data.get("images")
                    ):
                        from PIL import Image

                        with Image.open(thumbcheck) as image:
                            legacy = (
                                replace_generated
                                or max(image.size) <= 166
                                or getattr(image, "n_frames", 1) > 1
                            )
                        if legacy:
                            self.get_image(model_data, cache_file)

                    if not has_preview:
                        has_preview = self.copy_embedded_preview(path, cache_file)

                    if not has_preview and not self.offline:
                        # Try existing publisher images first, then make one HF
                        # attempt here. Metadata lookup above suppresses its own
                        # HF discovery, including when Civitai artwork fails.
                        if not self.get_image(model_data, cache_file):
                            from ruinedfooocus.features.models.model_identity import (
                                checkpoint_hashes,
                                verified_native_source,
                            )
                            from ruinedfooocus.features.models.model_sources import (
                                huggingface_preview,
                                huggingface_source,
                            )

                            digest = checkpoint_hashes(model_data).get("SHA256")
                            header = self.read_safetensors_header(path).get("__metadata__", {})
                            source = huggingface_source(
                                model_data, header, verified_native_source(path, model_data)
                            )
                            fallback = huggingface_preview(path, digest, source)
                            if fallback:
                                attempted = {
                                    item.get("url")
                                    for item in (model_data.get("images") or [])[:5]
                                    if isinstance(item, dict) and isinstance(item.get("url"), str)
                                }
                                model_data = model_data | fallback
                                save_json(cache_file.with_suffix(".json"), model_data)
                                fresh_images = [
                                    item
                                    for item in fallback["images"]
                                    if item["url"] not in attempted
                                ]
                                if fresh_images:
                                    self.get_image(
                                        model_data | {"images": fresh_images}, cache_file
                                    )
                        updated += 1

                    txtcheck = cache_file.with_suffix(".txt")
                    if (
                        model_type == "loras"
                        and model_data.get("trainedWords")
                        and not txtcheck.exists()
                    ):
                        print(
                            t(
                                "Get LoRA keywords for {name} ({base} - {type})",
                                mapping={
                                    "name": Path(path).name,
                                    "base": self.get_model_base(model_data),
                                    "type": self.get_model_type(model_data),
                                },
                            )
                        )
                        keywords = self.get_keywords(model_data)
                        with open(txtcheck, "w") as f:
                            f.write(", ".join(keywords))
                        updated += 1

                    try:
                        from ruinedfooocus.features.models.model_identity import checkpoint_hashes

                        digest = checkpoint_hashes(model_data).get("SHA256")
                        if digest:
                            self.model_hash[model_type][digest] = str(path)
                    except (AttributeError, TypeError):
                        pass

        if updated > 0:
            print(t("CivitAI update for {type} done.", mapping={"type": model_type}))

    def _update_embeddings(self, folder_paths):
        """Refresh textual-inversion presentation in the existing background worker.

        The embedding scanner owns names and collisions; do not interpret them as
        checkpoints or import the backend to infer an image model architecture.
        """
        from ruinedfooocus.features.models.embedding_browser import scan_embeddings
        from ruinedfooocus.features.models.resource_cards import resource_preview

        entries, _ = scan_embeddings(folder_paths)
        self.ready["embeddings"] = True
        for entry in entries:
            path = Path(entry.signature[0])
            # Reuse local artwork/cache lookup. PIL neutral markers mean no
            # publisher preview is installed; never persist a fake author image.
            preview = resource_preview(path, "Embedding", self.cache_paths["embeddings"])
            data = self.get_models_by_path("embeddings", path, fetch=not isinstance(preview, str))
            if not isinstance(preview, str) and not self.offline:
                published = data.get("images") or []
                safe_images = [
                    image
                    for image in published
                    if isinstance(image, dict)
                    and image.get("nsfwLevel") == 1
                    and image.get("type", "image") == "image"
                ]
                if safe_images:
                    self.get_image(
                        data | {"images": safe_images}, self.cache_paths["embeddings"] / path.name
                    )

    def get_names(self, model_type):
        return list(self.names[model_type])

    def get_file(self, model_type, name):
        # Search the folders for the model
        try:
            for folder in self.model_dirs[model_type]:
                file = Path(folder) / name
                if file.is_file():
                    return file
        except:
            pass
        return None

    def update_all_models(self):
        """Publish a local inventory only; never hash or fetch publisher artwork."""
        from ruinedfooocus.app.state import path_manager
        from ruinedfooocus.features.models.embedding_browser import scan_embeddings
        from ruinedfooocus.features.models.gguf_adapters import discover_image_loras
        from ruinedfooocus.features.models.local_diffusers import discover_bundles
        from ruinedfooocus.features.models.sensenova_bundle import (
            discover_bundles as sense_bundles,
            is_bundle_shard,
        )

        for model_type, folders in self.model_dirs.items():
            if model_type == "embeddings":
                entries, _ = scan_embeddings(folders)
                names = [entry.name for entry in entries]
            elif model_type == "loras":
                names = discover_image_loras(folders)
            else:
                names = [
                    str(path.relative_to(folder))
                    for folder in folders
                    for path in folder.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in self.EXTENSIONS
                    and not (model_type == "checkpoints" and is_bundle_shard(path))
                ]
                if model_type == "checkpoints":
                    for index in sense_bundles(folders):
                        folder = next(root for root in folders if index.is_relative_to(root))
                        names.append(str(index.relative_to(folder)))
                    names.extend(
                        str(index)
                        for index in discover_bundles(path_manager.get_model_folders("diffusers"))
                    )
            self.names[model_type] = sorted(
                dict.fromkeys(names),
                key=lambda name: (str(Path(name).parent) == ".", name.casefold()),
            )
            self.ready[model_type] = True
        self.revision += 1

    def start_metadata_updates(self):
        """Use existing workers after the UI is ready; repeated clients deduplicate."""
        if self.offline:
            return
        for model_type in ("checkpoints", "loras", "embeddings"):
            threading.Thread(
                target=self.civit_update_worker,
                args=(
                    model_type,
                    self.model_dirs[model_type],
                ),
                daemon=True,
            ).start()

    def start_inbox_sorting(self):
        """Start one process-owned sorter after the server is ready."""
        from ruinedfooocus.app.state import path_manager
        from ruinedfooocus.features.models.model_inbox import InboxSorter

        def run():
            sorter = InboxSorter(path_manager, offline=self.offline)
            while True:
                try:
                    if sorter.scan(self.model_dirs["inbox"]):
                        path_manager.upscaler_filenames = path_manager.get_model_filenames(
                            path_manager.model_paths["upscaler_path"],
                            extensions=NATIVE_STATE_DICT_SUFFIXES,
                        )
                        self.update_all_models()
                except OSError as error:
                    print(f"Model Inbox scan failed: {error}")
                time.sleep(5)

        with self._metadata_lock:
            if self._inbox_thread is None:
                self._inbox_thread = threading.Thread(target=run, name="model-inbox", daemon=True)
                self._inbox_thread.start()

    def __init__(self, offline=False):
        from ruinedfooocus.app.state import path_manager, settings

        self.offline = offline
        self.civit_workers = []
        self._metadata_lock = threading.Lock()
        self._metadata_checked = {}
        self._inbox_thread = None
        self.revision = 0

        self.ready = {
            "checkpoints": False,
            "loras": False,
            "embeddings": False,
            "inbox": False,
        }
        self.names = {
            "checkpoints": [],
            "loras": [],
            "embeddings": [],
            "inbox": [],
        }
        self.model_hash = {
            "checkpoints": {},
            "loras": {},
            "embeddings": {},
            "inbox": {},
        }
        checkpoints = path_manager.model_paths["modelfile_path"]
        checkpoints = checkpoints if isinstance(checkpoints, list) else [checkpoints]
        loras = path_manager.model_paths["lorafile_path"]
        loras = loras if isinstance(loras, list) else [loras]
        embeddings = path_manager.model_paths["embedding_path"]
        embeddings = embeddings if isinstance(embeddings, list) else [embeddings]
        inbox = path_manager.model_paths["inbox_path"]
        inbox = inbox if isinstance(inbox, list) else [inbox]
        self.model_dirs = {
            "checkpoints": checkpoints,
            "loras": loras,
            "embeddings": embeddings,
            "inbox": inbox,
        }
        self.cache_paths = {
            "checkpoints": Path(path_manager.model_paths["cache_path"] / "checkpoints"),
            "loras": Path(path_manager.model_paths["cache_path"] / "loras"),
            "embeddings": Path(path_manager.model_paths["cache_path"] / "embeddings"),
            "inbox": Path(path_manager.model_paths["cache_path"] / "inbox"),
        }

        self.base_url = "https://civitai.com/api/v1/"
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.EXTENSIONS = [*NATIVE_TORCH_SUFFIXES, *SAFETENSORS_SUFFIXES, ".gguf"]

        self.update_all_models()

    def get_file_from_hash(self, model_type, hash):
        return self.model_hash.get(model_type, {}).get(hash, None)

    def get_file_from_name(self, model_type, model_name):
        for folder in self.model_dirs[model_type]:
            path = Path(folder) / model_name
            if path.is_file():
                return path
        return None

    def model_sha256(self, filename):
        from ruinedfooocus.features.models.model_identity import cached_file_sha256

        try:
            return cached_file_sha256(filename, self.cache_paths["checkpoints"].parent / "hashes")
        except OSError as error:
            print(f"model_sha256(): Failed reading {filename}: {error}")
            return None

    def search_civitai_with_hash(self, hash):
        if self.offline:
            return None
        from ruinedfooocus.features.models.model_sources import civitai_metadata

        return civitai_metadata(f"model-versions/by-hash/{hash}") or {
            "files": [{"hashes": {"SHA256": hash}}]
        }

    def get_models_by_path(self, model_type, path, fetch=False, *, discover_hf=True):
        if path in (None, ""):
            return {}
        path = Path(path)
        if not path.is_file():
            path = self.get_file(model_type, path)
        if path is None:
            return {}

        from ruinedfooocus.features.models.local_diffusers import (
            is_diffusers_index,
            bundle_metadata,
        )
        from ruinedfooocus.features.models.sensenova_bundle import (
            is_sensenova_index,
            bundle_metadata as sense_metadata,
        )

        if model_type == "checkpoints" and is_sensenova_index(path):
            return sense_metadata(path)
        if model_type == "checkpoints" and is_diffusers_index(path):
            return bundle_metadata(
                path,
                self.detect_checkpoint_base if "comfy.model_detection" in sys.modules else None,
            )

        json_path = (self.cache_paths[model_type] / path.name).with_suffix(".json")
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}

        signature = file_signature(path)
        previous_signature = data.get("_rf_file_signature")
        if previous_signature is not None and previous_signature != signature:
            # A checkpoint replaced under the same name needs fresh identity.
            data = {}
        if model_type == "checkpoints":
            from ruinedfooocus.features.models.model_identity import read_prepared_native_source

            prepared = read_prepared_native_source(path, self.cache_paths["checkpoints"].parent)
            if prepared:
                data = (data if data.get("_rf_file_signature") == signature else {}) | prepared
        verified_fields = {}
        verified = None
        if model_type == "checkpoints":
            from ruinedfooocus.features.models.model_identity import verified_native_source

            verified = verified_native_source(path, data, current_signature=signature)
            if verified:
                verified_fields = {
                    key: data[key]
                    for key in (
                        "_rf_local_sha256",
                        "_rf_file_signature",
                        "_rf_verified_native_source",
                    )
                }
                verified_fields["baseModel"] = verified["native_family"]

        # Optimized checkpoints have a different hash. Keep downloaded metadata
        # beside them as <checkpoint>.civitai.info to retain the original identity.
        local_metadata = bool(data.get("rf_source"))
        if not data.get("id"):
            try:
                local = json.loads(path.with_suffix(".civitai.info").read_text(encoding="utf-8"))
                if isinstance(local, dict) and local.get("baseModel"):
                    data = local
                    local_metadata = True
            except (OSError, ValueError):
                pass

        if verified_fields:
            data = data | verified_fields

        # Record the actual local file hash separately from all release artifacts.
        # model_sha256 reuses the file-signature cache; never hash on UI selection.
        if fetch and not self.offline and not data.get("_rf_local_sha256"):
            local_hash = self.model_sha256(path)
            if local_hash:
                data["_rf_local_sha256"] = local_hash
                save_json(json_path, data)

        # Retry incomplete metadata during background refresh, never on UI selection.
        needs_metadata = not (data.get("id") or data.get("hf_repo_id")) or not data.get("images")
        if fetch and not self.offline and needs_metadata and (not local_metadata or data.get("id")):
            from ruinedfooocus.features.models.model_identity import checkpoint_hashes

            hash = checkpoint_hashes(data).get("SHA256")
            if not hash and not data.get("id"):
                hash = self.model_sha256(path)
            if hash or data.get("id"):
                from ruinedfooocus.features.models.model_sources import civitai_metadata

                from ruinedfooocus.features.models.model_sources import huggingface_source

                header = (
                    self.read_safetensors_header(path).get("__metadata__", {})
                    if model_type != "embeddings"
                    else {}
                )
                source = huggingface_source(data, header, verified)
                remote = (
                    civitai_metadata(f"model-versions/{data['id']}")
                    if data.get("id")
                    else None
                    if source.startswith("https://huggingface.co/")
                    else self.search_civitai_with_hash(hash)
                )
                if remote:
                    data = data | remote
                if model_type != "embeddings" and discover_hf and hash and not data.get("id"):
                    # Identity discovery must not depend on needing new artwork.
                    from ruinedfooocus.features.models.model_sources import huggingface_preview

                    fallback = huggingface_preview(path, hash, source)
                    if fallback:
                        data = data | fallback
                if hash:
                    # An existing files entry may lack hashes. setdefault on the
                    # outer key alone loses the computed hash and rehashes on
                    # every background refresh when lookup returns no metadata.
                    data["_rf_local_sha256"] = hash
                data["_rf_file_signature"] = signature
                save_json(json_path, data)

        if fetch and data and data.get("_rf_file_signature") != signature:
            # Stamp legacy metadata once without rehashing an unchanged model.
            data["_rf_file_signature"] = signature
            save_json(json_path, data)

        if model_type == "embeddings":
            return data

        from ruinedfooocus.features.models.image_scope import retired_family

        from ruinedfooocus.features.models.utility_models import (
            IMAGE_UTILITY_CLASSES,
            passive_utility_file,
        )
        from ruinedfooocus.features.models.stage_companions import passive_companion_file

        companion = passive_companion_file(path) if model_type == "checkpoints" else None
        from ruinedfooocus.features.models.native_workflow_admission import planned_image_family

        planned = planned_image_family(path) if model_type == "checkpoints" else None
        base = (
            passive_utility_file(path) or companion or planned or self.detect_checkpoint_base(path)
        )

        # These ephemeral role fields may never be inherited from basename JSON.
        data.pop("rf_model_role", None)
        data.pop("_rf_intrinsic_utility", None)
        data.pop("_rf_intrinsic_companion", None)
        try:
            source_unchanged = file_signature(path) == signature
        except OSError:
            source_unchanged = False
        corrected_hunyuan = (
            data.get("baseModel") == "Hunyuan Video"
            and base in ("HunyuanImage21", "HunyuanImage21Refiner")
            and data.get("_rf_file_signature") == signature
            and source_unchanged
        )
        if (
            base
            and source_unchanged
            and (
                retired_family(base)
                or base in IMAGE_UTILITY_CLASSES
                or base in ("Flux2Klein4B", "Flux2Klein9B")
                or base
                in (
                    "Chroma",
                    "ChromaRadiance",
                    "MiniMaxH3Image",
                    "BerniniImage",
                    "LongCatImage",
                    "ZImagePixelSpace",
                    "SDXLRefiner",
                    "HunyuanDiT",
                    "HunyuanDiT1",
                    "SD_X4Upscaler",
                    "Kandinsky5Image",
                    "JoyImage",
                    "SD15_instructpix2pix",
                    "SDXL_instructpix2pix",
                    "Omnigen2",
                    "Lens",
                    "MingImage",
                    "HunyuanImage21",
                    "QwenImage21",
                    "SenseNovaU15",
                    "SD21UnclipL",
                    "SD21UnclipH",
                    "Stable_Cascade_C",
                    "Stable_Cascade_B",
                    "PiD",
                    "HunyuanImage21Refiner",
                )
                or corrected_hunyuan
                or data.get("baseModel") in (None, "", "Unknown", "Other", "Lumina2")
            )
        ):
            data = data | {"baseModel": base}
        if model_type == "checkpoints" and source_unchanged and base in IMAGE_UTILITY_CLASSES:
            data = data | {
                "rf_model_role": "image_utility_model",
                "_rf_intrinsic_utility": {"native_class": base, "signature": list(signature)},
            }
        if companion and source_unchanged:
            data = data | {
                "rf_model_role": "image_stage_companion",
                "_rf_intrinsic_companion": {
                    "native_class": companion,
                    "conditioning": "stable_cascade_prior",
                    "primary_native_class": "Stable_Cascade_C",
                    "signature": list(signature),
                },
            }
        if verified_fields:
            # Publisher presentation may change; explicit verified source identity wins.
            if verified_native_source(path, verified_fields, current_signature=signature):
                data = data | verified_fields
                if fetch:
                    save_json(json_path, data)
            else:
                for key in verified_fields:
                    if key != "baseModel" or data.get(key) == verified_fields[key]:
                        data.pop(key, None)
        return data

    @staticmethod
    def read_safetensors_header(path):
        from ruinedfooocus.features.models.model_identity import cached_safetensors_header

        try:
            return cached_safetensors_header(path)
        except (OSError, ValueError):
            return {}

    # Detection runs on every model lookup; reuse it while the file is unchanged.
    _base_cache = {}

    @staticmethod
    def detect_checkpoint_base(path):
        path = Path(path)
        from ruinedfooocus.features.models.local_diffusers import is_diffusers_index
        from ruinedfooocus.features.models.sensenova_bundle import is_sensenova_index

        if is_diffusers_index(path) or is_sensenova_index(path):
            # Its selected diffusion file can change while model_index stays fixed.
            return Models._detect_checkpoint_base(path)
        try:
            key = (str(path), tuple(file_signature(path)[1:]))
        except OSError:
            return None
        if key in Models._base_cache:
            return Models._base_cache[key]
        base = Models._detect_checkpoint_base(path)
        if base is not None:  # Unknown results may change once the backend is installed.
            if len(Models._base_cache) >= 512:
                Models._base_cache.clear()
            Models._base_cache[key] = base
        return base

    @staticmethod
    def _detect_checkpoint_base(path):
        from ruinedfooocus.features.models.image_scope import unsupported_tensor_family
        from ruinedfooocus.features.models.local_diffusers import is_diffusers_index, capture_bundle
        from ruinedfooocus.features.models.sensenova_bundle import (
            is_sensenova_index,
            capture_bundle as capture_sense_bundle,
        )

        if is_sensenova_index(path):
            try:
                capture_sense_bundle(path)
                return "SenseNovaU15"
            except (OSError, ValueError):
                return None
        if is_diffusers_index(path):
            if "comfy.model_detection" not in sys.modules:
                return None
            try:
                return Models.detect_checkpoint_base(capture_bundle(path).diffusion.path)
            except (OSError, ValueError):
                return None

        if path.suffix.lower() == ".gguf":
            from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

            try:
                tensors, architecture = read_gguf_header(path)
                excluded = unsupported_tensor_family(tensors)
                if excluded:
                    return excluded
                family = {
                    "wan": "Wan Video",
                    "hyvid": "Hunyuan Video",
                    "minimax_h3": "MiniMaxH3",
                }.get(architecture)
                if family and architecture != "hyvid":
                    return family
                from ruinedfooocus.features.models.model_architecture import (
                    detect_tensor_architecture,
                )

                detected = detect_tensor_architecture(tensors)
                if architecture == "hyvid":
                    return (
                        detected
                        if detected in ("HunyuanImage21", "HunyuanImage21Refiner")
                        else family
                    )
                return detected
            except (OSError, ValueError, IndexError):
                return None
        # Inspect tensor names, not the filename or optimizer's family label.
        tensors = Models.read_safetensors_header(path)
        from ruinedfooocus.features.generation.bernini_image import primary_family

        if family := primary_family(path):
            return family
        excluded = unsupported_tensor_family(tensors)
        if excluded:
            return excluded
        if (
            "conditioner.embedders.1.model.text_projection" in tensors
            and "model.diffusion_model.input_blocks.0.0.weight" in tensors
        ):
            return "SDXL 1.0"
        prefix = "model.diffusion_model."
        for anima_prefix in (prefix, "net.", ""):
            if (
                anima_prefix + "blocks.0.mlp.layer1.weight" in tensors
                and anima_prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight" in tensors
            ):
                return "Anima"
        if (
            prefix + "double_blocks.0.img_attn.qkv.weight" in tensors
            and prefix + "single_blocks.0.linear1.weight" in tensors
            and tensors.get(prefix + "img_in.weight", {}).get("shape") == [3072, 64]
        ):
            return "Flux.1 D" if prefix + "guidance_in.in_layer.weight" in tensors else "Flux.1 S"
        try:
            from ruinedfooocus.features.models.model_architecture import detect_tensor_architecture

            return detect_tensor_architecture(tensors)
        except ImportError:
            return None  # Backend is not installed during first-run setup.

    @staticmethod
    def copy_embedded_preview(model_path, cache_path):
        metadata = Models.read_safetensors_header(model_path).get("__metadata__", {})
        thumbnail = metadata.get("modelspec.thumbnail", "")
        if not isinstance(thumbnail, str) or not thumbnail.startswith("data:image/"):
            return False
        try:
            data = base64.b64decode(thumbnail.split(",", 1)[1], validate=True)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.imwrite(str(cache_path.with_suffix(".jpeg")), image)
        except (ValueError, IndexError, cv2.error):
            pass
        return False

    def get_model_path(self, model_type, name, hash=None, default=None):
        from ruinedfooocus.app.state import path_manager, settings

        # Look through folders for the filename
        filename = self.get_file(model_type, name)

        # Try looking for model using the hash
        if filename is None and hash is not None:
            filename = self.get_file_from_hash(model_type, hash)
            if filename is not None:
                print(f"INFO: Found {filename} from hash")

        # Download the selected catalog entry for checkpoints and LoRAs.
        # Do not replace a missing custom model with an unrelated default.
        if filename is None and model_type in ("checkpoints", "loras"):
            from ruinedfooocus.app.arguments import args

            if not args.offline and os.environ.get("RF_OFFLINE") != "1":
                filename = path_manager.get_folder_file_path(model_type, name)
                if filename is not None:
                    threading.Thread(
                        target=self.civit_update_worker,
                        args=(model_type, self.model_dirs[model_type]),
                        daemon=True,
                    ).start()

        # If we don't have a filename, get the default.
        if filename is None and default is not None:
            name = path_manager.get_folder_file_path(
                model_type,
                default,
            )
            filename = self.get_file(model_type, name)
            if filename is not None:
                print(f"INFO: Using {filename} (default)")

        if filename is None:
            print(f"Could not find file: {name}")
            if hash is not None and hash != "None":
                print(f"    SHA256: {hash}")
                data = self.search_civitai_with_hash(hash)
                if data is not None and data.get("modelId", None) is not None:
                    print(f"    Download link: https://civitai.com/models/{data.get('modelId')}")
                print(f"    Search here: https://civitaiarchive.com/sha256/{hash}")

        return filename

    def get_keywords(self, model):
        keywords = model.get("trainedWords", [""])
        return keywords

    def get_model_base(self, model):
        return (model or {}).get("baseModel", "Unknown")

    def get_model_type(self, model):
        res = (model or {}).get("model", None)
        if res is not None:
            res = res.get("type", "Unknown")
        else:
            res = "Unknown"
        return res

    def get_image(self, model, path):
        from ruinedfooocus.app.state import settings
        from ruinedfooocus.features.models.model_previews import write_preview
        from ruinedfooocus.features.models import model_sources
        from urllib.parse import urlparse

        if self.offline:
            return False

        caption = (
            Path(path).stem
            if "caption" in settings.default_settings.get("model_preview", "").split(",")
            else ""
        )
        previews = model.get("images") or []
        if not isinstance(previews, list):
            return False
        attempted = set()
        for preview in previews[:5]:
            if (
                not isinstance(preview, dict)
                or not isinstance(preview.get("url"), str)
                or preview["url"] in attempted
                or preview.get("type", "image") != "image"
                or "cdn-thumbnails.huggingface.co/social-thumbnails/" in preview["url"]
            ):
                continue
            attempted.add(preview["url"])
            host = ""
            try:
                host = urlparse(preview["url"]).hostname or ""
                if host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co")):
                    data = model_sources.hf_metadata.read(preview["url"], 12 * 1024 * 1024)
                else:
                    with self.session.get(preview["url"], timeout=(5, 20), stream=True) as response:
                        response.raise_for_status()
                        data = bytearray()
                        started = time.monotonic()
                        for chunk in response.iter_content(65536):
                            if (
                                len(data) + len(chunk) > 64 * 1024 * 1024
                                or time.monotonic() - started > 60
                            ):
                                raise ValueError("Preview exceeds the download size or time limit")
                            data.extend(chunk)
                write_preview(bytes(data), path, caption)
                return True
            except Exception as error:
                if model_sources.hf_metadata.cooling_down():
                    break
                if not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))):
                    print(f"Could not update preview for {Path(path).stem}: {error}")
        return False
