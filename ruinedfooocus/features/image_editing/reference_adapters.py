"""Captured local native Style/Redux/PhotoMaker reference conditioning."""

from contextlib import ExitStack
from dataclasses import dataclass
import math
from pathlib import Path

from ruinedfooocus.core.model_formats import SAFETENSORS_SUFFIXES, NATIVE_TORCH_SUFFIXES
from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature

KEY = "_reference_adapter_capture"
FIELDS = (
    "reference_adapter",
    "reference_adapter_model",
    "reference_adapter_vision",
    "reference_adapter_strength",
)
KINDS = ("none", "style", "redux", "photomaker")


def modes_for_family(family):
    from ruinedfooocus.features.models.image_components import ALIASES

    if family == "FluxInpaint":
        return ["none"]
    family = ALIASES.get(family, family)
    return ["none"] + {"BaseModel": ["style"], "SDXL": ["photomaker"], "Flux": ["redux"]}.get(
        family, []
    )


def _shape(header, key):
    value = header.get(key)
    return value.get("shape") if isinstance(value, dict) else None


def compatible_file(mode, role, path):
    path = Path(path)
    if mode not in KINDS[1:] or role not in ("model", "vision"):
        return False
    if mode == "photomaker" and role == "vision":
        return False
    if path.suffix.lower() in NATIVE_TORCH_SUFFIXES:
        # Explicit local selection; native safe-load and strict state validation
        # happen during admitted compute. Browsing never unpickles this file.
        return True
    if path.suffix.lower() not in SAFETENSORS_SUFFIXES:
        return False
    try:
        header = cached_safetensors_header(path)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(header, dict):
        return False
    if role == "vision":
        width, positions, last = (1152, 729, 26) if mode == "redux" else (1024, 257, 23)
        return (
            _shape(header, "vision_model.embeddings.patch_embedding.weight") == [width, 3, 14, 14]
            and _shape(header, "vision_model.embeddings.position_embedding.weight")
            == [positions, width]
            and _shape(header, f"vision_model.encoder.layers.{last}.layer_norm1.weight") == [width]
        )
    if mode == "redux":
        return _shape(header, "redux_up.weight") == [12288, 1152] and _shape(
            header, "redux_down.weight"
        ) == [4096, 12288]
    if mode == "style":
        return _shape(header, "style_embedding") == [1, 8, 1024] and _shape(header, "proj") == [
            1024,
            768,
        ]
    return (
        _shape(header, "visual_projection.weight") == [768, 1024]
        and _shape(header, "visual_projection_2.weight") == [1280, 1024]
        and _shape(header, "fuse_module.mlp1.fc1.weight") == [2048, 4096]
    )


def local_choices(manager, mode, role):
    if (
        mode not in KINDS[1:]
        or role not in ("model", "vision")
        or (mode == "photomaker" and role == "vision")
    ):
        return []
    category = (
        "clip_vision"
        if role == "vision"
        else "photomaker"
        if mode == "photomaker"
        else "style_models"
    )
    result, names = [], set()
    for root in manager.get_model_folders(category):
        root = Path(root).resolve()
        pending, visited = [root], set()
        while pending:
            folder = pending.pop()
            resolved = folder.resolve()
            if not resolved.is_relative_to(root) or resolved in visited:
                continue
            visited.add(resolved)
            try:
                children = sorted(folder.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif path.is_file() and path.resolve().is_relative_to(root):
                    name = path.relative_to(root).as_posix()
                    # Root priority applies even when a later same-name file is compatible.
                    if name in names:
                        continue
                    names.add(name)
                    if compatible_file(mode, role, path):
                        result.append(name)
    return sorted(result, key=str.casefold)


@dataclass(frozen=True)
class CapturedAdapter:
    mode: str
    checkpoint_signature: tuple
    artifacts: tuple
    strength: float


def _local_artifact(manager, mode, role, name):
    from ruinedfooocus.features.generation.image_preparation import _artifact

    category = (
        "clip_vision"
        if role == "vision"
        else "photomaker"
        if mode == "photomaker"
        else "style_models"
    )
    roots = tuple(Path(root).resolve() for root in manager.get_model_folders(category))
    if not isinstance(name, str) or name not in local_choices(manager, mode, role):
        raise ValueError("Select an installed compatible reference adapter and vision encoder.")
    item = _artifact(manager, "reference_" + role, category, name)
    path = item.request.destination.resolve()
    if item.signature is None or not any(path.is_relative_to(root) for root in roots):
        raise ValueError(
            "The selected reference component changed or is outside configured folders."
        )
    if not compatible_file(mode, role, path):
        raise ValueError("The selected reference component is no longer compatible.")
    return item


def omit_inactive_ui_reference(data, controls):
    """Keep inactive UI choices in the profile, outside the queued request."""
    result = dict(data)
    if (
        "_api_job" not in data
        and data.get("task_type") != "tool_call"
        and (
            str((controls or {}).get("type", "")).lower() not in ("", "none", "img2img")
            or data.get("inpaint_toggle")
        )
    ):
        for key in (*FIELDS, "reference_adapter_image", KEY):
            result.pop(key, None)
    return result


def capture_assets(data, manager, models, controls):
    result = dict(data)
    result.pop(KEY, None)
    mode = result.get("reference_adapter") or "none"
    if mode not in KINDS:
        raise ValueError("Choose a supported reference adapter.")
    if mode == "none":
        return result, ()
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Reference adapters require the native image engine.")
    if str((controls or {}).get("type", "")).lower() not in ("", "none", "img2img") or result.get(
        "inpaint_toggle"
    ):
        raise ValueError("Reference adapters support ordinary text-to-image and image-to-image.")
    from PIL import Image

    image = result.get("reference_adapter_image")
    if not isinstance(image, Image.Image):
        raise ValueError("Add the reference adapter's image in Image tools.")
    result["reference_adapter_image"] = image.copy()
    name = result.get("base_model_name")
    checkpoint = models.get_file("checkpoints", name) if models is not None and name else None
    if checkpoint is None:
        raise ValueError("Select an installed checkpoint for the reference adapter.")
    family = models.get_model_base(
        models.get_models_by_path("checkpoints", checkpoint, fetch=False)
    )
    detector = getattr(models, "detect_checkpoint_base", None)
    if callable(detector):
        family = detector(checkpoint) or family
    if mode not in modes_for_family(family):
        raise ValueError(
            "This reference adapter is not compatible with the selected checkpoint family."
        )
    strength = result.get("reference_adapter_strength", 1.0)
    strength = 1.0 if strength is None else strength
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 10:
        raise ValueError("Reference strength must be from zero through ten.")
    if mode == "photomaker" and strength != 1:
        raise ValueError("PhotoMaker uses its native identity encoding without a strength setting.")
    artifacts = [_local_artifact(manager, mode, "model", result.get("reference_adapter_model"))]
    if mode != "photomaker":
        artifacts.append(
            _local_artifact(manager, mode, "vision", result.get("reference_adapter_vision"))
        )
    result[KEY] = CapturedAdapter(
        mode, tuple(file_signature(checkpoint)), tuple(artifacts), float(strength)
    )
    result["image_engine"] = "native"
    return result, tuple(artifacts)


def replay_values(metadata):
    return {
        "reference_adapter": metadata.get("reference_adapter", "none"),
        "reference_adapter_model": metadata.get("reference_adapter_model", ""),
        "reference_adapter_vision": metadata.get("reference_adapter_vision", ""),
        "reference_adapter_strength": metadata.get("reference_adapter_strength", 1.0),
    }


def _native_model(captured, model):
    config = getattr(model, "model_config", None)
    name = type(config).__name__
    allowed = {"style": {"SD15"}, "redux": {"Flux", "FluxSchnell"}, "photomaker": {"SDXL"}}[
        captured.mode
    ]
    matches = (
        bool(allowed.intersection(cls.__name__ for cls in type(config).__mro__))
        if captured.mode == "photomaker"
        else name in allowed
    )
    if not matches or (getattr(config, "inpaint_model", lambda: False)()):
        raise ValueError(
            "The reference adapter requires its ordinary native SD1, Flux or SDXL model."
        )
    if captured.mode == "redux" and config.unet_config.get("in_channels") != 64:
        raise ValueError("Redux requires an ordinary Flux checkpoint.")


def _photomaker(path):
    import folder_paths
    from comfy_extras.nodes_photomaker import PhotoMakerLoader

    # The exact file was captured under configured roots. Later profile edits
    # cannot redirect this queued load through a different live root list.
    folder_paths.add_model_folder_path("photomaker", str(Path(path).parent), is_default=True)
    name = Path(path).name
    if (
        Path(folder_paths.get_full_path_or_raise("photomaker", name)).resolve()
        != Path(path).resolve()
    ):
        raise ValueError(
            "The selected PhotoMaker file changed root priority. Refresh and submit again."
        )
    return PhotoMakerLoader.execute(name)[0]


def apply_reference(owner, positive, data, manager, to_tensor):
    """Return request-local positive conditioning; never mutate cached text/negative."""
    captured = data.get(KEY)
    if captured is None:
        if data.get("reference_adapter") not in (None, "", "none"):
            raise ValueError("Capture the reference adapter before native compute.")
        return positive
    if not isinstance(captured, CapturedAdapter):
        raise ValueError("Capture a valid reference adapter before native compute.")
    _native_model(captured, owner.xl_base_patched.unet.model)
    if tuple(getattr(owner, "base_file_signature", ())) != captured.checkpoint_signature:
        raise ValueError("The reference adapter belongs to another checkpoint. Submit again.")
    from PIL import ImageOps
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.features.models.yaml_recipes import apply_clip_recipe
    from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder

    with ExitStack() as stack:
        if captured.mode == "photomaker" and any(
            item[0].shape[-1] != 2048 or item[0].shape[1] != 77 for item in positive
        ):
            raise ValueError(
                "PhotoMaker requires native dual-CLIP SDXL conditioning and one 77-token prompt segment."
            )
        files = [
            stack.enter_context(
                prepared_asset(
                    data,
                    item.role,
                    item.name,
                    manager,
                    "clip_vision"
                    if item.role == "reference_vision"
                    else "photomaker"
                    if captured.mode == "photomaker"
                    else "style_models",
                )
            )
            for item in captured.artifacts
        ]
        identity = (captured.mode, *(tuple(file_signature(path)) for path in files))
        cached = getattr(owner, "reference_adapter_models", None)
        if cached is None or cached[0] != identity:
            if captured.mode == "photomaker":
                loaded = (_photomaker(files[0]), None)
            else:
                import comfy.sd
                import comfy.clip_vision

                try:
                    loaded = (
                        comfy.sd.load_style_model(str(files[0])),
                        comfy.clip_vision.load(str(files[1])),
                    )
                except (ValueError, RuntimeError, KeyError, TypeError) as error:
                    raise ValueError(
                        "The selected native reference adapter or vision state dictionary "
                        "is invalid or incompatible. " + str(error)
                    ) from error
                if loaded[1] is None:
                    raise ValueError("The selected native vision encoder could not be loaded.")
            owner.reference_adapter_models = (identity, loaded)
        adapter, vision = owner.reference_adapter_models[1]
        pixels = to_tensor(ImageOps.exif_transpose(data["reference_adapter_image"]))
        if captured.mode == "photomaker":
            from comfy_extras.nodes_photomaker import PhotoMakerEncode
            from ruinedfooocus.features.image_editing.clip_hook_conditioning import (
                direct_conditioning,
            )
            from ruinedfooocus.features.prompting.prompt_scheduling import prompt_switch_per_step

            clip = apply_clip_recipe(
                owner.xl_base_patched.clip, data["clip_skip"], getattr(owner, "yaml_recipe", None)
            )
            warm_patched_text_encoder(clip)
            schedule = prompt_switch_per_step(data["positive_prompt"], data["steps"])
            texts = schedule or [data["positive_prompt"]]
            encoded, output = {}, []
            from node_helpers import conditioning_set_values

            for index, text in enumerate(texts):
                from ruinedfooocus.features.models.embedding_browser import native_embedding_text

                text = native_embedding_text(
                    text,
                    manager.model_paths["embedding_path"],
                    manager.model_paths["cache_path"] / "embeddings",
                )
                if text.split(" ").count("photomaker") != 1:
                    raise ValueError(
                        "PhotoMaker needs one standalone photomaker token after the subject word."
                    )
                if text.split(" ").index("photomaker") == 0:
                    raise ValueError(
                        "Put photomaker after the subject word, such as person photomaker."
                    )
                if text not in encoded:
                    encoded[text] = direct_conditioning(
                        clip,
                        lambda current: PhotoMakerEncode.execute(adapter, pixels, current, text)[0],
                    )
                values = (
                    {
                        "start_percent": index / len(texts),
                        "end_percent": (index + 1) / len(texts),
                        "_rf_prompt_step": index,
                    }
                    if schedule
                    else {}
                )
                output += conditioning_set_values(encoded[text], values)
        else:
            from nodes import CLIPVisionEncode, StyleModelApply

            expected = 4096 if captured.mode == "redux" else 768
            if any(item[0].shape[-1] != expected for item in positive):
                raise ValueError(
                    "Reference adapter conditioning width does not match the native checkpoint."
                )
            encoded = CLIPVisionEncode().encode(vision, pixels, "center")[0]
            if encoded.last_hidden_state.shape[-1] != (1152 if captured.mode == "redux" else 1024):
                raise ValueError(
                    "The native vision encoder output does not match this reference adapter."
                )
            output = StyleModelApply().apply_stylemodel(
                positive, adapter, encoded, captured.strength, "multiply"
            )[0]
        import hashlib

        source = ImageOps.exif_transpose(data["reference_adapter_image"]).convert("RGB")
        data["reference_adapter_info"] = {
            "kind": captured.mode,
            "model": captured.artifacts[0].name,
            "vision": captured.artifacts[1].name if len(captured.artifacts) > 1 else None,
            "strength": captured.strength if captured.mode != "photomaker" else None,
            "source_size": list(source.size),
            "source_rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "files": [
                {"name": item.name, "signature": list(item.signature[1:])}
                for item in captured.artifacts
            ],
        }
        if tuple(
            file_signature(captured.checkpoint_signature[0])
        ) != captured.checkpoint_signature or identity != (
            captured.mode,
            *(tuple(file_signature(path)) for path in files),
        ):
            raise ValueError(
                "A reference adapter or checkpoint changed during conditioning. Submit again."
            )
        return output
