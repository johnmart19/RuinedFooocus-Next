"""Local catalogue projection and captured subscriptions to shared preparation.

The packaged download tuples and ModelRegistry remain the authorities. Browsing
does not hash/load weights, contact a publisher, or change the selected model.
"""

from concurrent.futures import CancelledError
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import shutil
import threading

from ruinedfooocus.core.preparation import PreparationRequest, preparation_service
from ruinedfooocus.features.models.model_registry import RegistryError, load_registry
from ruinedfooocus.features.models.model_registry import unique_payload_bytes
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.stage_companions import is_companion_annotation

CATEGORIES = {
    "checkpoints": "Image models",
    "diffusion_models": "Image models",
    "llm": "Chat models",
    "vision": "Vision / projectors",
    "clip": "Text encoders",
    "clip_vision": "Vision encoders",
    "vae": "VAEs",
    "loras": "LoRAs",
    "embeddings": "Embeddings",
    "controlnet": "ControlNet",
    "model_patches": "Model patches",
    "upscalers": "Upscalers",
    "latent_upscalers": "Latent upscalers",
    "geometry_estimation": "Depth",
    "background_removal": "Background removal",
    "faceswap": "Face restoration",
}


def size_label(value):
    return "Unknown" if value is None else f"{value / 1024**3:.2f} GiB"


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _target(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if Path(filename).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError("Catalogue filename must stay inside its model folder.")
    return path


def _space(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def workflow(annotation, profile):
    """Report source consumers separately from hardware/model qualification."""
    role = annotation.get("role", "unknown") if annotation else "unknown"
    consumers = tuple(profile.get("consumers", ()))
    from ruinedfooocus.features.models.image_scope import require_image_family, MESSAGE

    if annotation:
        for key in ("native_family", "architecture"):
            require_image_family(annotation.get(key))
    if role == "audio_video_diffusion_checkpoint":
        raise ValueError(MESSAGE)
    if role == "bundle_manifest":
        return (
            "Bundle",
            "Use the matching native bundle workflow; this entry is not a weight.",
            False,
        )
    if annotation and annotation.get("chat_weight"):
        return "Chat", "Chat · source/runtime eligibility is shown in Chat model details.", True
    if is_companion_annotation(annotation):
        return (
            "Component",
            "Cascade Stage B companion · select Stage C as the image model and Stage B in Cascade settings.",
            True,
        )
    if role == "image_diffusion_checkpoint":
        return (
            "Image",
            "Available source · the matching native image workflow is not implemented."
            if annotation.get("auto_recipe_supported") is False
            else ("Create · " + ", ".join(consumers))
            if consumers
            else "Downloadable image file · workflow not recorded.",
            True,
        )
    if role == "image_utility_model":
        return (
            "Image tools",
            "Available source · requires the matching native image utility workflow.",
            True,
        )
    if role == "image_utility_encoder":
        return (
            "Image tools",
            "Image tools · " + ", ".join(consumers)
            if consumers
            else "Available source · requires the matching native image utility workflow.",
            True,
        )
    if role in (
        "pixel_upscaler",
        "latent_upscaler",
        "depth_estimation",
        "background_removal",
        "face_restoration",
    ):
        if role == "face_restoration" and not consumers:
            return (
                "Component",
                "Downloadable restoration weights · callable workflow not recorded.",
                True,
            )
        return (
            "Image tools",
            "Image tools · "
            + CATEGORIES.get(
                {
                    "pixel_upscaler": "upscalers",
                    "latent_upscaler": "latent_upscalers",
                    "depth_estimation": "geometry_estimation",
                    "background_removal": "background_removal",
                    "face_restoration": "faceswap",
                }[role],
                role,
            ),
            True,
        )
    return (
        "Component",
        ("Component for " + ", ".join(consumers))
        if consumers
        else "Workflow compatibility not recorded.",
        True,
    )


class Catalogue:
    def __init__(self, manager, registry=None):
        self.manager = manager
        self.registry = registry
        self._snapshot = None
        self._snapshot_lock = threading.Lock()
        self._snapshot_version = 0

    def invalidate(self):
        with self._snapshot_lock:
            self._snapshot_version += 1
            self._snapshot = None

    def _registry(self):
        return self.registry if self.registry is not None else load_registry()

    def record(self, key):
        info = self.manager.DOWNLOADABLE_FILES[key]
        category = info["path"].removeprefix("path_")
        roots = self.manager.get_model_folders(category)
        candidates = [_target(root, info["filename"]) for root in roots]
        if not candidates:
            raise ValueError("No model folder is configured for this category.")
        local = next((path for path in candidates if path.is_file()), None)
        annotation, profile, metadata_error = None, {}, ""
        try:
            registry = self._registry()
            annotation = _plain(registry.annotations_for(key, info))
            profile = _plain(registry.profile(annotation["profile_id"])) if annotation else {}
        except RegistryError as error:
            metadata_error = str(error)
        artifact = annotation.get("artifact", {}) if annotation else {}
        task, support, allowed = workflow(annotation, profile)
        expected = artifact.get("bytes", info.get("size"))
        signature = file_signature(local) if local else None
        matches = not local or expected is None or signature[1] == expected
        from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS, ALIASES

        consumers = [name.split(":")[0] for name in profile.get("consumers", ())]
        # Native identities belong to the registered artifact; publisher consumer
        # captions are display text and may use a different spelling (Qwen 2.1).
        families = [
            *((annotation or {}).get(key) for key in ("native_family", "architecture")),
            *consumers,
        ]
        callable_image = (
            task == "Image"
            and (annotation or {}).get("auto_recipe_supported") is not False
            and any(ALIASES.get(name, name) in IMAGE_COMPONENTS for name in families if name)
        )
        return dict(
            key=key,
            category=category,
            category_label="Component"
            if (annotation or {}).get("role")
            in ("image_diffusion_companion", "image_stage_companion")
            or is_companion_annotation(annotation)
            else "Image tools"
            if (annotation or {}).get("role") in ("image_utility_model", "image_utility_encoder")
            else CATEGORIES.get(category, category),
            label=annotation.get("label", info["filename"]) if annotation else info["filename"],
            quantization=annotation.get("quantization", {}).get("label", "Unknown")
            if annotation
            else "Unknown",
            format=annotation.get("format", Path(info["filename"]).suffix.lstrip("."))
            if annotation
            else Path(info["filename"]).suffix.lstrip("."),
            task=task,
            support=support,
            allowed=allowed and not metadata_error,
            reason=metadata_error or ("" if allowed else support),
            local=str(local) if local else None,
            local_signature=signature,
            destination=str(candidates[0]),
            status=("Installed" if matches else "Present · size mismatch")
            if local
            else "Available",
            installation="Installed" if local else "Available",
            published_bytes=expected,
            local_bytes=signature[1] if local else None,
            size_matches=matches,
            can_activate=bool(
                local
                and matches
                and (
                    callable_image
                    and category in ("checkpoints", "diffusion_models")
                    or task == "Chat"
                    and category in ("llm", "vision")
                )
            ),
            annotation=annotation,
            profile=profile,
            source=artifact.get("pinned_url") or info["url"],
        )

    def _scan(self, cancelled=None):
        def check():
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("The catalogue request changed")

        records, represented = [], set()
        for key in sorted(self.manager.DOWNLOADABLE_FILES):
            check()
            try:
                record = self.record(key)
            except (OSError, ValueError, KeyError):
                continue
            records.append(record)
            if record["local"]:
                represented.add(str(Path(record["local"]).resolve()))
        from ruinedfooocus.core.path import PathManager

        for kind in CATEGORIES:
            for root in self.manager.get_model_folders(kind):
                check()
                root = Path(root)
                if not root.is_dir():
                    continue
                for path in root.rglob("*"):
                    check()
                    if not path.is_file() or path.suffix.lower() not in PathManager.EXTENSIONS:
                        continue
                    resolved = str(path.resolve())
                    if resolved in represented:
                        continue
                    represented.add(resolved)
                    records.append(
                        dict(
                            key="local:" + resolved,
                            category=kind,
                            category_label=CATEGORIES[kind],
                            label=path.name,
                            quantization="Unknown",
                            format=path.suffix.lstrip("."),
                            task="Local import",
                            support="Custom local file · inspect in its workflow.",
                            allowed=False,
                            reason="Already local; no registered download is required.",
                            local=resolved,
                            destination=resolved,
                            status="Installed · custom",
                            installation="Installed",
                            published_bytes=None,
                            local_bytes=path.stat().st_size,
                            local_signature=file_signature(path),
                            size_matches=True,
                            can_activate=False,
                            annotation=None,
                            profile={},
                            source="Local import",
                        )
                    )
        return records

    def records(
        self,
        *,
        query="",
        category="All",
        status="All",
        task="All",
        quantization="All",
        include_local=True,
        cancelled=None,
    ):
        with self._snapshot_lock:
            version, snapshot = self._snapshot_version, self._snapshot
        if snapshot is None:
            scanned = self._scan(cancelled=cancelled)
            with self._snapshot_lock:
                if version == self._snapshot_version:
                    if self._snapshot is None:
                        self._snapshot = scanned
                    snapshot = self._snapshot
                else:
                    # The caller's UI ticket fences this result; an obsolete
                    # scan must never overwrite a newer shared cache snapshot.
                    snapshot = scanned
        search = query.casefold().strip()
        return [
            row
            for row in snapshot
            if (include_local or not row["key"].startswith("local:"))
            and (category == "All" or row["category_label"] == category)
            and (status == "All" or row["installation"] == status)
            and (task == "All" or row["task"] == task)
            and (quantization == "All" or row["quantization"] == quantization)
            and (
                not search
                or search in (row["label"] + " " + row["key"] + " " + row["support"]).casefold()
            )
        ]

    def capture(
        self, key, *, include_companions=True, mode="text", reserve_gib=100, allow_unknown=False
    ):
        if mode not in ("text", "image"):
            raise ValueError("Choose text or image companions.")
        row = self.record(key)
        if not row["allowed"]:
            raise ValueError(row["reason"])
        registry = self._registry()
        keys = [key]
        if include_companions and row["annotation"]:
            keys.extend(
                registry.artifact(identity)["catalogue_key"]
                for identity in row["annotation"]["companions"].get(mode, ())
            )
        requests, seen = [], set()
        for item in dict.fromkeys(keys):
            record = self.record(item)
            if not record["allowed"]:
                raise ValueError(record["reason"])
            if record["local"]:
                if not record["size_matches"]:
                    raise ValueError(
                        "A local file has a different size; inspect it instead of replacing it."
                    )
                continue
            request = self.manager.preparation_request(item)
            if request.destination in seen:
                continue
            seen.add(request.destination)
            requests.append(request)
        if not requests:
            raise ValueError("All selected files are already installed. Nothing will be replaced.")
        if any(request.expected_size is None for request in requests) and not allow_unknown:
            raise ValueError(
                "The download size is unknown. Enable unknown-size downloads to continue."
            )
        if (
            isinstance(reserve_gib, bool)
            or not isinstance(reserve_gib, (int, float))
            or not 0 <= reserve_gib <= 10000
        ):
            raise ValueError("Enter a disk reserve between 0 and 10000 GiB.")
        # Check cumulative requirements on each target volume, before subscriptions.
        for request in requests:
            needed = sum(
                item.expected_size or 0
                for item in requests
                if item.destination.anchor == request.destination.anchor
            )
            if needed + reserve_gib * 1024**3 > _space(request.destination.parent):
                raise ValueError("The selected downloads would exceed the free-space reserve.")
        return DownloadPlan(row["label"], tuple(requests), selection_key=key)

    def companion_status(self, row, mode="text"):
        if not row["annotation"]:
            return [], None
        registry = self._registry()
        results, missing = [], []
        for identity in row["annotation"]["companions"].get(mode, ()):
            annotation = registry.artifact(identity)
            record = self.record(annotation["catalogue_key"])
            results.append(
                [record["label"], record["status"], size_label(record["published_bytes"])]
            )
            if not record["local"]:
                missing.append(annotation)
        return results, unique_payload_bytes(missing)


@dataclass(frozen=True)
class DownloadPlan:
    label: str
    artifacts: tuple[PreparationRequest, ...]
    selection_key: str = ""


class AcquisitionSession:
    """Session-owned subscribers; browsing another record cannot retarget a job."""

    def __init__(self):
        self.lock = threading.RLock()
        self.plan = None
        self.handles = []
        self.received = {}
        self.message = "No download running."
        self.active = False
        self.starting = False
        self.state = "idle"
        self.verified = []
        self._receipted = set()

    def __deepcopy__(self, memo):
        return AcquisitionSession()

    def snapshot(self):
        with self.lock:
            return dict(
                active=self.active,
                state=self.state,
                selection_key=self.plan.selection_key if self.plan else None,
                label=self.plan.label if self.plan else "",
            )

    def drain_verified(self):
        with self.lock:
            receipts, self.verified = self.verified, []
            return receipts

    def start(self, plan, *, offline=False, service=preparation_service):
        with self.lock:
            if offline:
                raise ValueError("Offline mode: downloads are disabled.")
            if self.active or self.starting:
                raise ValueError("A download is already running in this session.")
            if not plan.artifacts:
                raise ValueError("The download plan has no missing files.")
            self.plan, self.received = plan, {}
            self.verified, self._receipted = [], set()
            self.active = True
            self.starting = True
            self.state = "running"
            self.message = "Downloading " + plan.label
        try:
            for artifact in plan.artifacts:
                handle = service.submit(
                    artifact,
                    lambda count, total, speed, path=artifact.destination: self._progress(
                        path, count, total
                    ),
                )
                with self.lock:
                    cancelled = not self.active
                    if not cancelled:
                        self.handles.append(handle)
                if cancelled:
                    handle.cancel()
                    break
        except BaseException as error:
            self.cancel()
            with self.lock:
                self.state = "failed"
                self.message = "Download failed: " + str(error)
            raise
        finally:
            with self.lock:
                self.starting = False

    def _progress(self, path, count, total):
        with self.lock:
            self.received[path] = (count, total)

    def poll(self):
        cancel = []
        with self.lock:
            if not self.active:
                return self.message, False, False
            message = None
            for handle in self.handles:
                if not handle.done():
                    continue
                try:
                    path = handle.result()
                    signature = getattr(handle, "verified_signature", None)
                    if isinstance(signature, Mapping) and id(handle) not in self._receipted:
                        self._receipted.add(id(handle))
                        self.verified.append((handle.request, path, signature))
                except CancelledError:
                    self.state = "cancelled"
                    message = "Download cancelled. Installed files are retained."
                    break
                except Exception as error:
                    self.state = "failed"
                    message = "Download failed: " + str(error)
                    break
            if message is not None:
                self.message = message
                cancel, self.handles = self.handles, []
                self.active = False
            elif (
                self.starting
                or not self.handles
                or not all(handle.done() for handle in self.handles)
            ):
                count = sum(value[0] for value in self.received.values())
                total = sum(item.expected_size or 0 for item in self.plan.artifacts)
                return (
                    f"Downloading {self.plan.label} · {size_label(count)} / {size_label(total) if all(item.expected_size for item in self.plan.artifacts) else 'unknown total'}",
                    True,
                    False,
                )
            else:
                self.message = (
                    "Ready: " + self.plan.label + ". Select it explicitly in its workflow."
                )
                self.handles = []
                self.active = False
                self.state = "complete"
                return self.message, False, True
        for handle in cancel:
            handle.cancel()
        return message, False, False

    def cancel(self):
        with self.lock:
            handles = self.handles
            self.handles = []
            self.active = False
            self.state = "cancelled"
            self.message = "Download cancelled. Installed files are retained."
        # A progress callback holds the subscription lock before taking ours.
        # Never wait for its cancellation while retaining the session lock.
        for handle in handles:
            handle.cancel()


def inspected_local(row, models):
    """Explicit local-card inspection reuses the model index's native detector."""
    if (
        row.get("local")
        and row["key"].startswith("local:")
        and row["category"] in ("llm", "vision")
    ):
        from ruinedfooocus.features.chat.llama_models import local_models

        path = Path(row["local"])
        if (
            path.is_file()
            and file_signature(path) == row.get("local_signature")
            and any(Path(value).resolve() == path.resolve() for value in local_models().values())
        ):
            return {
                **row,
                "can_activate": True,
                "task": "Chat",
                "support": "Chat · local candidate; runtime eligibility is checked in Chat.",
            }
    if (
        row.get("local")
        and row["key"].startswith("local:")
        and row["category"] in ("checkpoints", "diffusion_models")
    ):
        from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS, ALIASES

        if not Path(row["local"]).is_file() or file_signature(row["local"]) != row.get(
            "local_signature"
        ):
            return {
                **row,
                "can_activate": False,
                "support": "Local file changed. Refresh and select again.",
            }
        detect = getattr(models, "detect_checkpoint_base", None)
        family = detect(Path(row["local"])) if detect is not None else None
        if family and ALIASES.get(family, family) in IMAGE_COMPONENTS:
            return {**row, "can_activate": True, "support": f"Create · {family} (local header)"}
    return row


def checkpoint_name(row, models):
    """Activation is separate from acquisition, and never replaces prompts."""
    if not row.get("can_activate"):
        raise ValueError("Select an installed image checkpoint to use in Create.")
    target = Path(row["local"]).resolve()
    if not target.is_file() or file_signature(target) != row["local_signature"]:
        raise ValueError("The local file changed. Refresh and select it again.")
    for name in models.get_names("checkpoints"):
        candidate = models.get_file("checkpoints", name)
        if candidate is not None and Path(candidate).resolve() == target:
            return name
    raise ValueError(
        "Refresh local files first. A same-name file in another root may have priority."
    )
