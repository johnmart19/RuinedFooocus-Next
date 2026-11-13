"""Checkpoint recipes from public version metadata, with separate local overrides.

Model descriptions are data: only recognised numeric and sampling fields are read.
Prompts, image pixels, embedded workflows and executable instructions are ignored.
"""

import json
import math
import re
import threading
import time
import sys
from html import unescape
from pathlib import Path

from ruinedfooocus.core.config_io import save_json

FIELDS = ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
RECOMMENDATION_SCHEMA = 6
QWEN_IMAGE21_FAMILIES = ("QwenImage21", "Qwen Image 2.1")
_QWEN_IMAGE21_WORKFLOW = "https://comfy.org/hub/workflows/qwen-image-2-1-image-edit-5fbe136ba6f9/"
_QWEN_IMAGE21_TURBO_WORKFLOW = (
    "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/"
    "image_qwen_image_2_1_turbo_t2i.json"
)
QWEN_IMAGE21_TURBO_SIGMAS = (
    1.0,
    0.978453,
    0.95418,
    0.926626,
    0.89508,
    0.845148,
    0.704534,
    0.414568,
    0.0,
)
FAMILY_PRESETS = {
    "HiDream": "HiDream I1 Full",
    "MOSSImage": "MOSS Image",
    "Anima": "Anima",
    "SDXL 1.0": "SDXL",
    "Illustrious": "Illustrious XL",
    "NoobAI": "NoobAI XL",
    "Pony": "Pony XL",
    "Krea2": "Krea 2",
    "Flux.1 D": "FLUX.1 dev",
    "Flux.1 S": "FLUX.1 schnell",
    "QwenImage": "Qwen Image",
    "Qwen": "Qwen Image",
    "Qwen-Image": "Qwen Image",
    "SD 3.5": "SD3.5",
    "SD 3.5 Large": "SD3.5",
    "MageFlow": "Mage Flow",
    "HiDreamO1": "HiDream O1",
    "MingImage": "Ming Image Design",
    "QwenImage21": "Qwen Image 2.1",
    "Stable_Cascade_C": "Stable Cascade C",
    "StableCascade_C": "Stable Cascade C",
}


def family_recipe(family, *, variant=None, media="image"):
    """Return bounded local geometry/sampling provenance without model/history I/O."""
    from ruinedfooocus.features.models.resource_defaults import load_family_recipe
    from ruinedfooocus.features.models.model_registry import RegistryError

    try:
        return load_family_recipe(family, variant=variant, media=media)
    except RegistryError:
        return {}


def validated(values, samplers, schedulers, *, native_sampling=None):
    if not isinstance(values, dict):
        return {}
    result = {}
    for key, low, high in (("custom_steps", 1, 200), ("cfg", 0, 20), ("clip_skip", 1, 5)):
        value = values.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and low <= value <= high
            and math.isfinite(value)
        ):
            if key == "cfg" or int(value) == value:
                result[key] = value if key == "cfg" else int(value)
    for key, allowed in (("sampler_name", samplers), ("scheduler", schedulers)):
        native = (
            key == "scheduler"
            and native_sampling is not None
            and values.get(key) == native_sampling.scheduler
        )
        if isinstance(values.get(key), str) and (values[key] in allowed or native):
            result[key] = values[key]
    return result


def published_resolution(metadata):
    """Read an explicit, bounded image size; card prose and examples cast no vote."""
    config = metadata.get("generationConfig") if isinstance(metadata, dict) else None
    if not isinstance(config, dict):
        return None
    size = (config.get("width"), config.get("height"))
    return (
        size
        if all(type(value) is int and 256 <= value <= 4096 and value % 2 == 0 for value in size)
        else None
    )


def verified_qwen_image21_variant(metadata):
    from ruinedfooocus.features.models.model_identity import verified_native_source

    signature = metadata.get("_rf_file_signature")
    if not isinstance(signature, list) or len(signature) != 4 or not isinstance(signature[0], str):
        return None
    source = verified_native_source(Path(signature[0]), metadata, current_signature=signature)
    if source and source.get("native_family") == "QwenImage21":
        return source if source.get("behavioral_variant") in ("base", "turbo") else None
    return None


def _selected_checkpoint(checkpoint, metadata):
    """Resolve an existing selection without importing application state."""
    selected = Path(checkpoint) if isinstance(checkpoint, (str, Path)) and checkpoint else None
    if selected is not None and not selected.is_absolute():
        models = getattr(sys.modules.get("ruinedfooocus.app.state"), "models", None)
        selected = models.get_file("checkpoints", checkpoint) if models is not None else None
    if selected is None:
        signature = metadata.get("_rf_file_signature")
        if isinstance(signature, list) and len(signature) == 4 and isinstance(signature[0], str):
            path = Path(signature[0])
            if path.is_absolute() and (not checkpoint or path.name == Path(checkpoint).name):
                selected = path
    return Path(selected) if selected is not None else None


def compatible_qwen_image21_variant(metadata, checkpoint=None):
    """A family recipe crosses quantization; exact-artifact qualification does not."""
    source = verified_qwen_image21_variant(metadata)
    if source:
        return source
    # Native header classification must already be available. Browsing does not
    # initialize the backend or trust a sidecar/filename as a tensor architecture.
    if "comfy.model_detection" not in sys.modules:
        return None
    from ruinedfooocus.features.models.model_architecture import inspect_model_file
    from ruinedfooocus.features.models.model_identity import file_signature

    selected = _selected_checkpoint(checkpoint, metadata)
    if selected is None:
        return None
    try:
        signature = file_signature(selected)
        contract = inspect_model_file(selected)
        if (
            contract.get("architecture") != "QwenImage21"
            or contract.get("native_class") != "QwenImage21"
            or contract.get("native_config", {}).get("image_model") != "qwen_image21"
        ):
            return None
        current = metadata if metadata.get("_rf_file_signature") == signature else {}
        declarations = [contract.get("metadata", {}), current]
        variants = {
            value.strip().casefold()
            for declaration in declarations
            for key in ("modelspec.variant", "variant", "model_variant", "training_variant")
            if isinstance(value := declaration.get(key), str) and value.strip()
        }
        if (
            len(variants) != 1
            or not variants <= {"base", "turbo"}
            or signature != file_signature(selected)
        ):
            return None
        return dict(
            native_family="QwenImage21",
            behavioral_variant=next(iter(variants)),
            recipe_identity_basis="current_native_header_and_explicit_variant",
            artifact_qualified=False,
        )
    except (OSError, ValueError, TypeError, AttributeError, IndexError):
        return None


def _qwen_image21_workflow(source, samplers, schedulers):
    if not source:
        return {}
    recipe = family_recipe("QwenImage21", variant=source["behavioral_variant"])
    return validated(recipe.get("sampling", {}), samplers, schedulers)


def qwen_image21_sigmas(values, schedule="auto"):
    """Resolve the visible schedule choice and qualified automatic workflow."""
    if schedule == "standard":
        return None
    if schedule == "turbo_8":
        expected = dict(steps=8, cfg=1, sampler_name="euler", scheduler="simple")
        if any(values.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Qwen Turbo 8-step requires 8 steps, CFG 1, Euler and the simple scheduler. "
                "Choose Own settings to set these values, or select Auto."
            )
        return QWEN_IMAGE21_TURBO_SIGMAS
    return (
        QWEN_IMAGE21_TURBO_SIGMAS
        if values.get("_model_default_sigmas") == QWEN_IMAGE21_TURBO_SIGMAS
        else None
    )


def extract(metadata, samplers, schedulers, family=None):
    """Prefer structured version settings, then explicit labelled card fields.

    For ranges use the lower recommended value; retain the card as the source.
    Ambiguous sampler alternatives are left for the user to select.
    """
    config = metadata.get("generationConfig") or {}
    if not isinstance(config, dict):
        config = {}
    family = family or metadata.get("baseModel")

    def sdk_guidance(value):
        # O1's SDK skips CFG at <=1; native ComfyUI uses CFG=1 for positive-only.
        if (
            family in ("HiDreamO1", "HiDream O1")
            and type(value) in (int, float)
            and 0 <= value <= 1
        ):
            return 1.0
        return value

    aliases = {
        "steps": "custom_steps",
        "num_inference_steps": "custom_steps",
        "cfgScale": "cfg",
        "guidance_scale": "cfg",
        "sampler": "sampler_name",
        "clipSkip": "clip_skip",
    }
    values = {
        aliases.get(key, key): sdk_guidance(value) if key == "guidance_scale" else value
        for key, value in config.items()
    }
    text = unescape(re.sub(r"<[^>]*>", "\n", str(metadata.get("description") or "")[:200000]))
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text.replace("**", ""))
    for key, label in (
        ("custom_steps", "Steps"),
        ("cfg", r"CFG(?:\s*Scale)?"),
        ("clip_skip", r"Clip\s*Skip"),
    ):
        matches = re.findall(r"(?im)^\s*" + label + r"\s*[:：]\s*(\d+(?:\.\d+)?)", text)
        if len(set(matches)) == 1 and key not in values:
            number = float(matches[0])
            values[key] = number
    # Public cards often publish these settings as literal pipeline arguments.
    # Read numbers only; never evaluate code or guess between conflicting examples.
    for key, argument in (("custom_steps", "num_inference_steps"), ("cfg", "guidance_scale")):
        matches = re.findall(
            r"(?m)^\s*" + argument + r"\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(?:,|$)", text
        )
        numbers = {float(value) for value in matches}
        if len(numbers) == 1 and key not in values:
            value = numbers.pop()
            values[key] = sdk_guidance(value) if argument == "guidance_scale" else value
    # Only inspect sampling-labelled lines, never a random mention in prose.
    lines = re.findall(
        r"(?im)^\s*(?:Sampler(?:\s*/\s*Scheduler)?|Scheduler)\s*[:：]\s*([^\n]+)", text
    )
    candidates = {"sampler_name": set(), "scheduler": set()}
    for line in lines:
        normalized = (
            line.lower()
            .replace("sgm-uniform", "sgm_uniform")
            .replace("res-multistep", "res_multistep")
            .replace("er-sde", "er_sde")
        )
        if re.search(r"\beuler\s+a\b", normalized):
            normalized = re.sub(r"\beuler\s+a\b", "euler_ancestral", normalized)
        for key, allowed in (("sampler_name", samplers), ("scheduler", schedulers)):
            found = [
                name
                for name in allowed
                if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", normalized)
            ]
            candidates[key].update(found)
    for key, found in candidates.items():
        if len(found) == 1 and key not in values:
            values[key] = next(iter(found))
    return validated(values, samplers, schedulers)


def example_settings(metadata, samplers, schedulers):
    """Infer missing fields from a clear majority of this version's examples.

    Read bounded sampling scalars and resource types, never prompts or workflows.
    Require three supporting examples and at least two-thirds agreement among
    examples publishing each field. Missing or invalid values cast no vote.
    """
    from collections import Counter

    images = metadata.get("images")
    if not isinstance(images, list):
        return {}
    votes = {field: [] for field in FIELDS}
    seen = set()
    for image in images[:100]:
        if not isinstance(image, dict) or not isinstance(image.get("meta"), dict):
            continue
        # Repeated entries for the same public image must not amplify its vote.
        identity = image.get("id") or image.get("url")
        if isinstance(identity, (str, int)):
            if identity in seen:
                continue
            seen.add(identity)
        meta = image["meta"]
        # A gallery recipe using extra network weights (notably Lightning)
        # does not describe the bare checkpoint selected by Model Defaults.
        if any(
            isinstance(resources, list)
            and any(
                isinstance(resource, dict)
                and isinstance(resource.get("type"), str)
                and resource["type"].casefold()
                in ("lora", "locon", "dora", "lycoris", "hypernetwork")
                for resource in resources[:100]
            )
            for resources in (meta.get("additionalResources"), meta.get("resources"))
        ):
            continue
        values = {
            target: meta.get(source)
            for source, target in (
                ("steps", "custom_steps"),
                ("cfgScale", "cfg"),
                ("clipSkip", "clip_skip"),
                ("sampler", "sampler_name"),
                ("scheduler", "scheduler"),
            )
        }
        for field, allowed in (("sampler_name", samplers), ("scheduler", schedulers)):
            value = values[field]
            if isinstance(value, str):
                normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
                normalized = {"euler_a": "euler_ancestral"}.get(normalized, normalized)
                values[field] = next((name for name in allowed if name.lower() == normalized), None)
        for field, value in validated(values, samplers, schedulers).items():
            votes[field].append(value)
    result = {}
    for field, values in votes.items():
        if values:
            value, count = Counter(values).most_common(1)[0]
            if count >= 3 and count * 3 >= len(values) * 2:
                result[field] = value
    return result


class ModelRecommendations:
    def __init__(self, cache_dir=None, path=None, offline=False):
        from ruinedfooocus.core.paths import data_path
        from ruinedfooocus.core.cache_contract import configured_cache_root

        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else configured_cache_root() / "checkpoints"
        )
        self.source_cache_root = (
            self.cache_dir.parent if self.cache_dir.name == "checkpoints" else self.cache_dir
        )
        self.path = Path(path) if path is not None else data_path("settings", "model_settings.json")
        self.lock = threading.RLock()
        self.offline = offline

    def family_details(self, checkpoint, *, family=None):
        """Current compatible family geometry with source/qualification kept separate."""
        metadata = self.metadata(checkpoint)
        family = family or metadata.get("baseModel")
        if family in QWEN_IMAGE21_FAMILIES:
            source = compatible_qwen_image21_variant(metadata, checkpoint)
            if not source:
                return {}
            recipe = family_recipe(family, variant=source["behavioral_variant"])
            if recipe:
                recipe["identity_basis"] = source.get(
                    "recipe_identity_basis", "verified_publisher_artifact"
                )
                recipe["artifact_qualified"] = False
            return recipe
        selected = _selected_checkpoint(checkpoint, metadata)
        if selected is None or metadata.get("_rf_stale_native_family"):
            return {}
        from ruinedfooocus.features.models.native_recipes import compatible_native_recipe

        native = compatible_native_recipe(selected, metadata)
        if native:
            return native
        from ruinedfooocus.features.models.model_identity import file_signature
        from ruinedfooocus.features.models.performance import PerformanceSettings

        try:
            if metadata.get("_rf_file_signature") != file_signature(selected):
                return {}
            resolver = PerformanceSettings.__new__(PerformanceSettings)
            resolver.model_recommendations = self
            preset = resolver.family_preset(family, checkpoint)
            return family_recipe(preset) if preset else {}
        except (OSError, ValueError, TypeError):
            return {}

    def metadata(self, checkpoint):
        if not isinstance(checkpoint, str) or not checkpoint:
            return {}
        # The cache is named after the checkpoint stem; never write into models.
        try:
            data = json.loads(
                (self.cache_dir / (Path(checkpoint).stem + ".json")).read_text(encoding="utf-8")
            )
            from ruinedfooocus.features.models.model_identity import read_prepared_native_source

            selected = _selected_checkpoint(checkpoint, data if isinstance(data, dict) else {})
            prepared = (
                read_prepared_native_source(selected, self.source_cache_root)
                if selected is not None
                else {}
            )
            if prepared:
                data = (
                    data
                    if isinstance(data, dict)
                    and data.get("_rf_file_signature") == prepared["_rf_file_signature"]
                    else {}
                ) | prepared
            if isinstance(data, dict) and "_rf_file_signature" in data:
                from ruinedfooocus.features.models.model_identity import file_signature

                selected = Path(checkpoint)
                if not selected.is_absolute():
                    import sys

                    models = getattr(sys.modules.get("ruinedfooocus.app.state"), "models", None)
                    selected = (
                        models.get_file("checkpoints", checkpoint) if models is not None else None
                    )
                    if selected is None and models is None:
                        signature = data["_rf_file_signature"]
                        if isinstance(signature, list) and len(signature) == 4:
                            cached_path = (
                                Path(signature[0]) if isinstance(signature[0], str) else None
                            )
                            if (
                                cached_path is not None
                                and cached_path.is_absolute()
                                and cached_path.name == Path(checkpoint).name
                            ):
                                selected = cached_path
                if selected is None or file_signature(selected) != data["_rf_file_signature"]:
                    return {"_rf_stale_native_family": data.get("baseModel", "Unknown")}
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            from ruinedfooocus.features.models.model_identity import read_prepared_native_source

            selected = _selected_checkpoint(checkpoint, {})
            return (
                read_prepared_native_source(selected, self.source_cache_root)
                if selected is not None
                else {}
            )

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return (
                {key: value for key, value in data.items() if isinstance(value, dict)}
                if isinstance(data, dict)
                else {}
            )
        except (OSError, ValueError):
            return {}

    def identity(self, checkpoint, metadata=None):
        if metadata is None:
            metadata = self.metadata(checkpoint)
        if not metadata.get("id") and metadata.get("hf_repo_id") and self.cached_hash(metadata):
            return f"{checkpoint}:hf:{metadata['hf_repo_id']}:{self.cached_hash(metadata)}"
        return f"{checkpoint}:{metadata.get('id', 'local')}"

    @staticmethod
    def cached_hash(metadata):
        from ruinedfooocus.features.models.model_identity import checkpoint_hashes

        value = checkpoint_hashes(metadata).get("SHA256")
        if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value):
            return value.lower()
        return None

    def refresh(self, checkpoint, samplers, schedulers, force=False):
        if self.offline:
            return "Offline metadata mode; retained local settings."
        metadata = self.metadata(checkpoint)
        if metadata.get("baseModel") in QWEN_IMAGE21_FAMILIES:
            source = compatible_qwen_image21_variant(metadata, checkpoint)
            if _qwen_image21_workflow(source, samplers, schedulers):
                return "Using the verified ComfyUI Qwen Image 2.1 workflow."
            return "Verify the published Qwen Image 2.1 checkpoint to use its Model Defaults."
        version = metadata.get("id")
        has_version = type(version) is int and version > 0
        repository = metadata.get("hf_repo_id")
        sha256 = self.cached_hash(metadata)
        if not has_version and not (repository and sha256):
            return "No public version ID or verified Hugging Face identity available; using local metadata."
        key = self.identity(checkpoint, metadata)
        with self.lock:
            entry = self.load().get(key, {})
        ttl = 86400 if entry.get("status") == "Public version metadata refreshed." else 300
        checked = entry.get("checked", 0)
        if (
            not isinstance(checked, (float, int))
            or isinstance(checked, bool)
            or not 0 <= checked <= time.time() + 86400
        ):
            checked = 0
        if (
            not force
            and entry.get("schema") == RECOMMENDATION_SCHEMA
            and time.time() - checked < ttl
        ):
            return entry.get("status", "Cached recommendation")
        from ruinedfooocus.features.models.model_sources import civitai_metadata

        remote = civitai_metadata(f"model-versions/{version}") if has_version else None
        matched = (
            isinstance(remote, dict) and type(remote.get("id")) is int and remote["id"] == version
        )
        remote_version = remote if matched else None
        recommended = extract(remote, samplers, schedulers) if matched else {}
        source = f"Civitai version {version}" if matched else None
        source_url = None
        if matched and type(remote.get("modelId")) is int and remote["modelId"] > 0:
            source_url = f"https://civitai.com/models/{remote['modelId']}?modelVersionId={version}"
            if set(recommended) != set(FIELDS):
                model = civitai_metadata(f"models/{remote['modelId']}")
                versions = model.get("modelVersions", []) if isinstance(model, dict) else []
                versions = (
                    [item for item in versions if isinstance(item, dict)]
                    if isinstance(versions, list)
                    else []
                )
                families = {
                    item.get("baseModel")
                    for item in versions
                    if isinstance(item.get("baseModel"), str)
                }
                if (
                    isinstance(model, dict)
                    and type(model.get("id")) is int
                    and model["id"] == remote["modelId"]
                    and any(item.get("id") == version for item in versions)
                    and isinstance(remote.get("baseModel"), str)
                    and families == {remote["baseModel"]}
                ):
                    general = extract(model, samplers, schedulers)
                    general = {
                        key: value for key, value in general.items() if key not in recommended
                    }
                    if general:
                        source = (
                            source + "; " if recommended else ""
                        ) + "Civitai creator model card (shared across versions)"
                        recommended = general | recommended
        # Civitai commonly publishes recipes only on version example images.
        # Keep explicit Civitai card settings authoritative; infer missing fields only
        # from the exact version response, never another version's gallery.
        inferred = {}
        if has_version and isinstance(remote_version, dict):
            inferred = {
                key: value
                for key, value in example_settings(remote_version, samplers, schedulers).items()
                if key not in recommended
            }
        if inferred:
            has_explicit = bool(recommended)
            recommended = inferred | recommended
            source = (
                source + "; " if has_explicit and source else ""
            ) + f"Civitai version {version} example consensus (inferred: {', '.join(inferred)})"
            if (
                not source_url
                and type(remote_version.get("modelId")) is int
                and remote_version["modelId"] > 0
            ):
                source_url = f"https://civitai.com/models/{remote_version['modelId']}?modelVersionId={version}"
        if sha256 and not repository and not recommended:
            from ruinedfooocus.features.models.model_sources import huggingface_preview

            discovery = huggingface_preview(checkpoint, sha256)
            if isinstance(discovery, dict):
                repository = discovery.get("hf_repo_id")
        if repository and sha256 and not recommended:
            from ruinedfooocus.features.models.model_sources import huggingface_recommendations

            hf = huggingface_recommendations(repository, sha256)
            if isinstance(hf, dict):
                # Repackaged weights may link to the original creator's card.
                # Follow a bounded set of HF repositories only when each contains
                # exactly the same weights; never trust a link as model identity.
                if not extract(hf, samplers, schedulers, metadata.get("baseModel")):
                    linked = re.findall(
                        r"https://huggingface\.co/([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)",
                        str(hf.get("description") or "")[:200000],
                    )
                    parents = hf.get("source_repositories", [])
                    if isinstance(parents, list):
                        linked = [name for name in parents if isinstance(name, str)] + linked
                    candidates = [
                        name.rstrip(".")
                        for name in dict.fromkeys(linked)
                        if name != repository
                        and name.split("/")[0]
                        not in ("docs", "spaces", "datasets", "blog", "collections")
                    ][:3]
                    for candidate in candidates:
                        creator = huggingface_recommendations(candidate, sha256)
                        if isinstance(creator, dict) and extract(
                            creator, samplers, schedulers, metadata.get("baseModel")
                        ):
                            hf = creator
                            break
                remote = hf
                recommended = extract(hf, samplers, schedulers, metadata.get("baseModel"))
                source, source_url = hf.get("source"), hf.get("source_url")
                matched = True
        status = (
            "Public version metadata refreshed."
            if matched
            else "Lookup unavailable; retained cached settings."
        )
        with self.lock:
            data = self.load()
            entry = data.setdefault(key, data.pop(f"{checkpoint}:local", {}))
            entry.update(checked=time.time(), status=status, schema=RECOMMENDATION_SCHEMA)
            if matched:
                entry.update(
                    recommended=recommended,
                    source=source,
                    source_url=source_url,
                    resolution=published_resolution(remote),
                    resolution_signature=metadata.get("_rf_file_signature"),
                )
            save_json(self.path, data)
        return status

    def cached_resolution(self, checkpoint):
        """Return exact cached publisher size/provenance, with no network lookup."""
        metadata = self.metadata(checkpoint)
        if metadata.get("_rf_stale_native_family"):
            return None, None
        with self.lock:
            entry = self.load().get(self.identity(checkpoint, metadata), {})
        signature = entry.get("resolution_signature")
        if (
            not isinstance(signature, list)
            or len(signature) != 4
            or signature != metadata.get("_rf_file_signature")
        ):
            return None, None
        size = entry.get("resolution")
        if "resolution" in entry and size is None:
            return None, entry.get("source") or "Cached publisher metadata"
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            return None, None
        checked = published_resolution({"generationConfig": dict(zip(("width", "height"), size))})
        return (
            (checked, entry.get("source") or "Cached publisher metadata")
            if checked
            else (None, None)
        )

    def recipe(self, checkpoint, samplers, schedulers, *, family=None):
        details = self.details(checkpoint, samplers, schedulers, family=family)
        values = details["recommended"] | details["override"]
        if details.get("native_sampling_plan") is not None:
            from ruinedfooocus.features.generation.native_sampling import KEY

            values[KEY] = details["native_sampling_plan"]
        status = details["source"]
        if not details["recommended"]:
            status += ": no sampling settings published."
        if details["override"]:
            status += " Saved local settings are active."
        return values, status

    def cached_examples(self, metadata, samplers, schedulers):
        """Infer only from a current, exact local variant's cached public version.

        metadata() has already checked the file signature. The separate local
        checksum must also match a release file, not just its version/name or
        another precision in the same release. Never hash or fetch in this path.
        """
        signature = metadata.get("_rf_file_signature")
        checksum = metadata.get("_rf_local_sha256")
        files = metadata.get("files")
        if not (
            type(metadata.get("id")) is int
            and metadata["id"] > 0
            and isinstance(signature, list)
            and len(signature) == 4
            and isinstance(checksum, str)
            and re.fullmatch(r"[a-fA-F0-9]{64}", checksum)
            and isinstance(files, list)
            and any(
                isinstance(item, dict)
                and isinstance(item.get("hashes"), dict)
                and isinstance(item["hashes"].get("SHA256"), str)
                and item["hashes"]["SHA256"].casefold() == checksum.casefold()
                for item in files[:1000]
            )
        ):
            return {}
        return example_settings(metadata, samplers, schedulers)

    def details(self, checkpoint, samplers, schedulers, *, family=None):
        metadata = self.metadata(checkpoint)
        with self.lock:
            data = self.load()
            entry = data.get(
                self.identity(checkpoint, metadata), data.get(f"{checkpoint}:local", {})
            )
        from ruinedfooocus.features.generation.native_sampling import plan_from_recipe

        native = self.family_details(checkpoint, family=family)
        native_sampling = plan_from_recipe(native)
        override = validated(
            entry.get("override", {}), samplers, schedulers, native_sampling=native_sampling
        )
        if (
            family in QWEN_IMAGE21_FAMILIES
            or metadata.get("baseModel") in QWEN_IMAGE21_FAMILIES
            or metadata.get("_rf_stale_native_family") in QWEN_IMAGE21_FAMILIES
        ):
            source = compatible_qwen_image21_variant(metadata, checkpoint)
            workflow = _qwen_image21_workflow(source, samplers, schedulers)
            family_data = (
                family_recipe("QwenImage21", variant=source["behavioral_variant"]) if source else {}
            )
            turbo = bool(source and source["behavioral_variant"] == "turbo")
            label = "Turbo workflow (fixed 8-step schedule)" if turbo else "base workflow"
            quantization = source.get("quantization", {}).get("label") if source else None
            if quantization and not turbo:
                label += f" ({quantization})"
            return dict(
                recommended=workflow,
                fetched={},
                override=override,
                source=(
                    f"ComfyUI Qwen Image 2.1 {label}"
                    if workflow
                    else "Qwen Image 2.1 checkpoint identity"
                ),
                source_url=(
                    (_QWEN_IMAGE21_TURBO_WORKFLOW if turbo else _QWEN_IMAGE21_WORKFLOW)
                    if workflow
                    else None
                ),
                fixed_sigmas=(
                    QWEN_IMAGE21_TURBO_SIGMAS
                    if turbo
                    and workflow
                    and all(
                        override.get(key, workflow[key]) == workflow[key]
                        for key in ("custom_steps", "cfg", "sampler_name", "scheduler")
                    )
                    else None
                ),
                network_cached=False,
                status="",
                provenance=family_data.get("sampling_provenance", {}),
                geometry_provenance=family_data.get("provenance", {}),
                applicability=family_data.get("applicability", {}),
                qualification=None,
            )
        explicit = extract(metadata, samplers, schedulers)
        inferred = {
            key: value
            for key, value in self.cached_examples(metadata, samplers, schedulers).items()
            if key not in explicit
        }
        local = inferred | explicit
        fetched = validated(entry.get("recommended", {}), samplers, schedulers)
        if metadata.get("_rf_stale_native_family"):
            fetched = {}
            local = {}
        recommended = fetched if "recommended" in entry else local
        baseline = (
            validated(
                native.get("sampling", {}), samplers, schedulers, native_sampling=native_sampling
            )
            if native.get("id", "").startswith("native-") and native.get("auto_recipe_supported")
            else {}
        )
        recommended = baseline | recommended
        source = entry.get("source") or (
            f"Civitai version {metadata['id']}"
            if isinstance(metadata.get("id"), int)
            else "local metadata"
        )
        if inferred and "recommended" not in entry:
            source += f"; cached version example consensus (inferred: {', '.join(inferred)})"
        url = entry.get("source_url")
        if metadata.get("_rf_stale_native_family"):
            source, url = "Stale model metadata ignored", None
        if baseline and not (fetched if "recommended" in entry else local):
            source = native["provenance"]["label"]
            url = native["provenance"]["source_url"]
        if not url and type(metadata.get("modelId")) is int and type(metadata.get("id")) is int:
            url = (
                f"https://civitai.com/models/{metadata['modelId']}?modelVersionId={metadata['id']}"
            )
        return dict(
            recommended=recommended,
            fetched=fetched,
            override=override,
            source=source,
            source_url=url,
            network_cached="recommended" in entry,
            status=entry.get("status", ""),
            native_recipe=bool(baseline),
            native_sampling_plan=native_sampling,
            recipe_requirements=native.get("applicability", {}).get("runtime_requirements", []),
            provenance=native.get("provenance", {}),
            applicability=native.get("applicability", {}),
        )

    def describe(self, checkpoint, options, samplers, schedulers, family_default=True, locale=None):
        from ruinedfooocus.core.translation_manager import localize

        details = self.details(checkpoint, samplers, schedulers)
        source = details["source"]
        url = details["source_url"]
        if isinstance(url, str):
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if url == _QWEN_IMAGE21_WORKFLOW or (
                parsed.scheme == "https" and parsed.hostname in ("civitai.com", "huggingface.co")
            ):
                source = f"[{source}]({url})"
        lines = [source]
        if details["override"]:
            lines += [
                localize(
                    "**Your override is active.** Remove saved settings to use network recommendations.",
                    locale,
                )
            ]
        labels = dict(
            custom_steps=localize("Steps", locale),
            cfg="CFG",
            sampler_name=localize("Sampler", locale),
            scheduler=localize("Scheduler", locale),
            clip_skip=localize("Clip Skip", locale),
            sampling_shift=localize("Sampling shift", locale),
        )
        if details["network_cached"]:
            if details["fetched"]:
                fetched = " · ".join(
                    f"{labels[key]}: {value}" for key, value in details["fetched"].items()
                )
                lines += [localize("**Network settings:** {values}", locale, values=fetched)]
            else:
                lines += [
                    localize(
                        "**Network settings:** none published in the fetched metadata.", locale
                    )
                ]
        elif url != _QWEN_IMAGE21_WORKFLOW:
            lines += [
                localize("**Network settings:** not fetched; using cached model metadata.", locale)
            ]
        if self.offline:
            lines += [localize("Offline mode.", locale)]
        elif details["status"].startswith("Lookup unavailable"):
            lines += [localize("Refresh failed; previous settings retained.", locale)]
        table = [
            "| {} | {} | {} |".format(
                localize("Setting", locale),
                localize("Model default", locale),
                localize("Source", locale),
            ),
            "|---|---|---|",
        ]
        fields = ["custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip"]
        if "sampling_shift" in options:
            fields.append("sampling_shift")
        for key in fields:
            origin = (
                localize("Your override", locale)
                if key in details["override"]
                else localize("Network", locale)
                if key in details["fetched"]
                else details["source"]
                if key in details["recommended"]
                and (url == _QWEN_IMAGE21_WORKFLOW or details.get("native_recipe"))
                else localize("Cached metadata", locale)
                if key in details["recommended"]
                else options["_recipe_baseline"]
                if options.get("_recipe_baseline")
                else localize("Family preset", locale)
                if family_default
                else localize("App fallback", locale)
            )
            table.append(f"| {labels[key]} | {options[key]} | {origin} |")
        return "\n\n".join(lines) + "\n\n" + "\n".join(table)

    def save_override(self, checkpoint, values, samplers, schedulers):
        from ruinedfooocus.features.generation.native_sampling import plan_from_recipe

        plan = plan_from_recipe(self.family_details(checkpoint))
        clean = validated(values, samplers, schedulers, native_sampling=plan)
        if set(clean) != set(FIELDS):
            raise ValueError(
                "Override requires valid steps, CFG, sampler, scheduler and Clip Skip."
            )
        with self.lock:
            data = self.load()
            key = self.identity(checkpoint)
            if key not in data:
                data[key] = data.pop(f"{checkpoint}:local", {})
            data[key]["override"] = clean
            save_json(self.path, data)

    def reset_override(self, checkpoint):
        with self.lock:
            data = self.load()
            for key in (self.identity(checkpoint), f"{checkpoint}:local"):
                data.get(key, {}).pop("override", None)
            save_json(self.path, data)
