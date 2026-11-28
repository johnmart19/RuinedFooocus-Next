"""Explicit local, data-only recipes for ordinary native Stable Diffusion."""

from concurrent.futures import CancelledError
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import yaml

from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES, SAFETENSORS_SUFFIXES
from ruinedfooocus.core.preparation import preparation_service
from ruinedfooocus.features.models.model_identity import file_signature

MAX_BYTES = 64 * 1024
MAX_NODES = 2048
MAX_DEPTH = 32
KEY = "_native_yaml_recipe"
FAMILIES = frozenset({"BaseModel", "SD 1.5", "SD 2.0", "SDXL", "SDXL 1.0", "SDXLRefiner"})
_DATA_TAGS = frozenset(
    "tag:yaml.org,2002:" + name for name in ("map", "seq", "str", "int", "float", "bool", "null")
)


class _DataLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._recipe_depth = self._recipe_nodes = 0

    def compose_node(self, parent, index):
        event = self.peek_event()
        if isinstance(event, yaml.events.AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ValueError("YAML recipe anchors and aliases are not supported.")
        self._recipe_nodes += 1
        self._recipe_depth += 1
        try:
            if self._recipe_nodes > MAX_NODES or self._recipe_depth > MAX_DEPTH:
                raise ValueError("YAML recipe is too complex.")
            node = super().compose_node(parent, index)
            if node.tag not in _DATA_TAGS:
                raise ValueError("YAML recipes accept plain data tags only.")
            return node
        finally:
            self._recipe_depth -= 1

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            if key_node.tag != "tag:yaml.org,2002:str":
                raise ValueError("YAML recipe mapping keys must be text.")
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("YAML recipe mapping keys must be unique.")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class RecipeOverrides:
    prediction: str | None
    clip_layer_idx: int | None


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"YAML recipe {label} must be a mapping.")
    return value


def parse_recipe(payload):
    """Read only the two overrides consumed by pinned comfy.sd.load_checkpoint."""
    if isinstance(payload, str):
        if len(payload) > MAX_BYTES:
            raise ValueError("YAML recipe must be at most 64 KiB of UTF-8 text.")
        payload = payload.encode("utf-8")
    if not isinstance(payload, bytes) or len(payload) > MAX_BYTES:
        raise ValueError("YAML recipe must be at most 64 KiB of UTF-8 text.")
    loader = None
    try:
        loader = _DataLoader(payload.decode("utf-8-sig"))
        document = _mapping(loader.get_single_data(), "document")
    except (yaml.YAMLError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid data-only YAML recipe.") from error
    finally:
        if loader is not None:
            loader.dispose()
    model = _mapping(document.get("model"), "model")
    params = _mapping(model.get("params"), "model.params")
    parameterization = params.get("parameterization")
    if "parameterization" in params and parameterization not in ("eps", "v"):
        raise ValueError("YAML recipe parameterization must be eps or v.")
    stage = _mapping(params.get("cond_stage_config", {}), "cond_stage_config")
    clip_params = _mapping(stage.get("params", {}), "cond_stage_config.params")
    layer = clip_params.get("layer_idx")
    if layer is not None and type(layer) is not int:
        raise ValueError("YAML recipe layer_idx must be an integer.")
    if "parameterization" not in params and "layer_idx" not in clip_params:
        raise ValueError("YAML recipe needs a supported prediction or CLIP-layer field.")
    # The native legacy loader only patches v. An eps declaration does not
    # force an EPS patch over a model's own detected prediction type.
    return RecipeOverrides("v_prediction" if parameterization == "v" else None, layer)


@dataclass(frozen=True)
class RecipeFile:
    source: Path
    path: Path
    signature: tuple

    def verify(self):
        try:
            unchanged = (
                self.source.resolve(strict=True) == self.path
                and tuple(file_signature(self.path)) == self.signature
            )
        except OSError:
            unchanged = False
        if not unchanged:
            raise ValueError("A YAML recipe or checkpoint changed. Refresh and submit again.")


def _capture_file(filename):
    source = Path(filename).expanduser().absolute()
    path = source.resolve(strict=True)
    if not path.is_file():
        raise ValueError("A YAML recipe and its checkpoint must be regular local files.")
    result = RecipeFile(source, path, tuple(file_signature(path)))
    result.verify()
    return result


def _read_recipe(item):
    item.verify()
    with item.path.open("rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("YAML recipe must be at most 64 KiB of UTF-8 text.")
    item.verify()
    return payload


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()


@dataclass(frozen=True)
class CapturedRecipe:
    recipe: RecipeFile
    checkpoint: RecipeFile
    recipe_sha256: str
    overrides: RecipeOverrides
    family: str
    clip_layer_source: str

    @property
    def recipe_clip_layer(self):
        # No numeric Clip Skip heuristic: explicit 1 and every legacy request win.
        return self.overrides.clip_layer_idx if self.clip_layer_source == "recipe" else None

    @property
    def identity(self):
        values = [
            str(self.recipe.source),
            self.recipe.signature,
            self.recipe_sha256,
            str(self.checkpoint.source),
            self.checkpoint.signature,
            self.family,
            self.overrides.prediction,
            self.overrides.clip_layer_idx,
            self.clip_layer_source,
        ]
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

    def verify(self, cancelled=None):
        _check_cancelled(cancelled)
        self.checkpoint.verify()
        payload = _read_recipe(self.recipe)
        if hashlib.sha256(payload).hexdigest() != self.recipe_sha256:
            raise ValueError("The YAML recipe changed. Refresh and submit again.")
        _check_cancelled(cancelled)


def capture_recipe(
    recipe_path,
    checkpoint,
    *,
    family,
    engine="native",
    clip_layer_source="request",
    cancelled=None,
    service=preparation_service,
):
    """Capture installed files only; no downloads, imports from YAML or model reads."""
    if engine != "native" or family not in FAMILIES:
        raise ValueError("YAML recipes require a supported native Stable Diffusion checkpoint.")
    if clip_layer_source not in ("request", "recipe"):
        raise ValueError("Choose request or recipe as the CLIP-layer source.")
    if Path(recipe_path).suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Select a local YAML recipe file.")
    if Path(checkpoint).suffix.lower() not in (*SAFETENSORS_SUFFIXES, *NATIVE_TORCH_SUFFIXES):
        raise ValueError("YAML recipes require a native single-file checkpoint container.")
    _check_cancelled(cancelled)
    recipe = _capture_file(recipe_path)
    checkpoint = _capture_file(checkpoint)
    with ExitStack() as stack:
        for item in (recipe, checkpoint):
            stack.enter_context(service.artifact_lease(item.path, cancelled=cancelled))
        payload = _read_recipe(recipe)
        result = CapturedRecipe(
            recipe,
            checkpoint,
            hashlib.sha256(payload).hexdigest(),
            parse_recipe(payload),
            family,
            clip_layer_source,
        )
        result.verify(cancelled)
        return result


@contextmanager
def recipe_read_scope(captured, *, cancelled=None, service=preparation_service):
    """Retain the existing preparation service's local read leases during load."""
    if not isinstance(captured, CapturedRecipe):
        raise ValueError("Capture the YAML recipe before preparing or loading it.")
    with ExitStack() as stack:
        for item in (captured.recipe, captured.checkpoint):
            stack.enter_context(service.artifact_lease(item.path, cancelled=cancelled))
        captured.verify(cancelled)
        yield captured
        captured.verify(cancelled)


def discover_local_recipes(path_manager):
    """List separate local recipe choices without parsing or attaching sidecars."""
    choices, seen = [], set()
    for category in ("checkpoints", "diffusion_models"):
        for index, root in enumerate(path_manager.get_model_folders(category)):
            root = Path(root).expanduser().resolve()
            if not root.is_dir():
                continue
            # Refuse escaped directories before traversal, including junctions.
            pending, visited, files = [root], set(), []
            while pending:
                directory = pending.pop()
                resolved = directory.resolve()
                if not resolved.is_relative_to(root) or resolved in visited:
                    continue
                visited.add(resolved)
                try:
                    children = sorted(directory.iterdir())
                except OSError:
                    continue
                for child in children:
                    if child.is_dir():
                        pending.append(child)
                    else:
                        files.append(child)
            for path in sorted(files):
                if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file():
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root) or str(resolved) in seen:
                    continue
                seen.add(str(resolved))
                label = f"{category}[{index + 1}]/{path.relative_to(root).as_posix()}"
                choices.append((label, str(path.absolute())))
    return choices


def normalize_policy(data, *, explicit_clip_skip=False):
    """Establish intent before defaults merge; a numeric value is not provenance."""
    result = dict(data)
    source = result.get("clip_layer_source", "request")
    if source not in ("request", "recipe"):
        raise ValueError("Choose request or recipe as the CLIP-layer source.")
    if explicit_clip_skip and source == "recipe":
        raise ValueError("Explicit Clip Skip conflicts with recipe CLIP-layer ownership.")
    result["clip_layer_source"] = "request" if explicit_clip_skip else source
    return result


def replay_values(metadata):
    """Saved selection is a recapture request, never a trusted live object."""
    result = {
        "model_recipe_path": metadata.get("model_recipe_path") or "",
        "model_recipe_sha256": metadata.get("model_recipe_sha256"),
        "model_recipe_overrides": metadata.get("model_recipe_overrides"),
        "clip_layer_source": metadata.get("clip_layer_source", "request"),
    }
    if not result["model_recipe_path"]:
        result.update(
            clip_layer_source="request", model_recipe_sha256=None, model_recipe_overrides=None
        )
    return normalize_policy(result)


def capture_request(data, models, path_manager, controls):
    """Freeze explicit selection after sampling capture, before any queue wait."""
    result = normalize_policy(data)
    result.pop(KEY, None)  # Persisted metadata never authorizes a live capture.
    recipe_path = result.get("model_recipe_path")
    if recipe_path in (None, "", "None"):
        if result["clip_layer_source"] == "recipe":
            raise ValueError("Choose a local YAML recipe before using its CLIP layer.")
        return result
    if not isinstance(recipe_path, str) or len(recipe_path) > 4096:
        raise ValueError("Choose a local YAML recipe from the configured model folders.")
    selected = Path(recipe_path).expanduser().absolute()
    allowed_roots = tuple(
        Path(root).expanduser().resolve()
        for category in ("checkpoints", "diffusion_models")
        for root in path_manager.get_model_folders(category)
    )
    choices = {Path(path).absolute() for _, path in discover_local_recipes(path_manager)}
    if selected not in choices:
        raise ValueError("The selected YAML recipe is missing or outside configured model folders.")
    kind = str((controls or {}).get("type", "")).lower()
    if (
        kind not in ("", "none", "img2img")
        or result.get("inpaint_toggle")
        or result.get("task_type") in ("llama", "chat", "rembg", "upscale")
    ):
        raise ValueError("YAML recipes support ordinary text-to-image and image-to-image only.")
    name = result.get("base_model_name")
    checkpoint = models.get_file("checkpoints", name) if name else None
    if checkpoint is None:
        raise ValueError("Choose an installed checkpoint for this YAML recipe.")
    family = models.get_model_base(
        models.get_models_by_path("checkpoints", checkpoint, fetch=False)
    )
    engine = result.get("image_engine", "native")
    captured = capture_recipe(
        selected,
        checkpoint,
        family=family,
        engine="native" if engine == "auto" else engine,
        clip_layer_source=result["clip_layer_source"],
    )
    if not any(captured.recipe.path.is_relative_to(root) for root in allowed_roots):
        raise ValueError("The YAML recipe location changed outside the configured model folders.")
    expected_digest = result.get("model_recipe_sha256")
    if expected_digest and expected_digest != captured.recipe_sha256:
        raise ValueError(
            "The saved YAML recipe changed. Refresh the recipe selection and submit again."
        )
    result.update(
        model_recipe_path=str(captured.recipe.source),
        model_recipe_sha256=captured.recipe_sha256,
        model_recipe_overrides={
            "prediction": captured.overrides.prediction,
            "clip_layer_idx": captured.overrides.clip_layer_idx,
        },
        image_engine="native",
    )
    result[KEY] = captured
    return result


def captured_recipe(data):
    captured = (data or {}).get(KEY)
    if captured is not None and not isinstance(captured, CapturedRecipe):
        raise ValueError("Capture the selected YAML recipe before native loading.")
    if (data or {}).get("model_recipe_path") not in (None, "", "None") and captured is None:
        raise ValueError("Capture the selected YAML recipe before native loading.")
    return captured


def recipe_identity(data):
    captured = captured_recipe(data)
    return captured.identity if captured is not None else None


def apply_model_recipe(model, captured):
    if captured is None:
        return model
    from comfy.model_sampling import ModelSamplingDiscrete
    from comfy_extras.nodes_model_advanced import ModelSamplingDiscrete as NativePatch

    config = getattr(model.model, "model_config", None)
    allowed = {
        "BaseModel": {"SD15", "SD20"},
        "SD 1.5": {"SD15"},
        "SD 2.0": {"SD20"},
        "SDXL": {"SDXL"},
        "SDXL 1.0": {"SDXL"},
        "SDXLRefiner": {"SDXLRefiner"},
    }
    if (
        type(config).__name__ not in allowed.get(captured.family, ())
        or not isinstance(model.get_model_object("model_sampling"), ModelSamplingDiscrete)
        or config.inpaint_model()
    ):
        raise ValueError("YAML recipe does not match an ordinary native discrete SD checkpoint.")
    return (
        NativePatch().patch(model, sampling="v_prediction", zsnr=False)[0]
        if captured.overrides.prediction == "v_prediction"
        else model
    )


def clip_policy(captured, clip_skip):
    layer = captured.recipe_clip_layer if captured is not None else None
    if layer is None:
        layer = -int(clip_skip) if int(clip_skip) > 1 else None
    return (
        captured.identity if captured is not None else None,
        captured.clip_layer_source if captured is not None else "request",
        layer,
    )


def apply_clip_recipe(clip, clip_skip, captured):
    """Native layer selection is request-local, including signed YAML indices."""
    layer = clip_policy(captured, clip_skip)[2]
    if captured is None and layer is None:
        return clip
    result = clip.clone()
    if layer is not None:
        result.clip_layer(layer)
    return result
