from ruinedfooocus.core.paths import data_path, resource_path
import json
from pathlib import Path
import re
import sys
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.models.model_recommendations import (
    QWEN_IMAGE21_FAMILIES,
    compatible_qwen_image21_variant,
)

MODEL_DEFAULT_SELECTIONS = ("Model Defaults", "Model settings", "Model defaults")
FLUX3_FAMILIES = ("Flux3", "Flux.3 Klein")
VARIANT_PRESETS = {
    "Stable Cascade C": {"base": "Stable Cascade C"},
    "HiDream I1 Full": {
        "full": "HiDream I1 Full",
        "dev": "HiDream I1 Dev",
        "fast": "HiDream I1 Fast",
    },
    "HiDream O1": {"full": "HiDream O1", "base": "HiDream O1", "dev": "HiDream O1 Dev"},
    "Qwen Image": {"base": "Qwen Image", "edit": "Qwen Image Edit"},
    "Mage Flow": {"base": "Mage Flow", "turbo": "Mage Flow Turbo"},
    "Lens": {"base": "Lens", "turbo": "Lens Turbo"},
    "Ming Image Design": {"design": "Ming Image Design"},
}
AMBIGUOUS_VARIANT_FAMILIES = (
    "Stable_Cascade_C",
    "StableCascade_C",
    "QwenImage21",
    "HiDream",
    "HiDreamO1",
    "HiDream O1",
    "QwenImage",
    "Qwen",
    "Qwen-Image",
    "Qwen Image",
    "MageFlow",
    "Mage Flow",
    "Lens",
    "MingImage",
)
# These routes require a qualified native variant baseline or a checkpoint recipe.
EXPLICIT_RECIPE_FAMILIES = (
    "PiD",
    "HunyuanImage21Refiner",
    "Stable_Cascade_C",
    "StableCascade_C",
    "SD21UNCLIP",
    "SD21UnclipL",
    "SD21UnclipH",
    "HunyuanImage21",
    "QwenImage21",
    "SenseNovaU15",
    "HunyuanDiT",
    "HunyuanDiT1",
    "Chroma",
    "ChromaRadiance",
    "LongCatImage",
    "ZImagePixelSpace",
    "SDXLRefiner",
    "SD_X4Upscaler",
    "Kandinsky5Image",
    "JoyImage",
    "SD15_instructpix2pix",
    "SDXL_instructpix2pix",
    "Omnigen2",
    "Lens",
    "MingImage",
)
KANDINSKY5_LITE_SHA256 = "be7a9e1f002f3aded7bed95dfe664a762e66ce766876f137444aab13ba48b63f"
OMNIGEN2_FP16_SHA256 = "60dbde45107762d164bac463e1cf365e074b377fa843dc90cb2985fb211cd4de"
# Historical publisher payload identities retained for source/integrity callers.
# Auto admission uses the native configuration, independently of these checksums.
PAYLOAD_RECIPES = {
    "HunyuanDiT": (
        "4fb84f84079cda457d171b3c6b15d1be95b5a3e5d9825703951a99ddf92d1787",
        dict(custom_steps=30, cfg=6, sampler_name="euler", scheduler="sgm_uniform", clip_skip=1),
    ),
    "Kandinsky5Image": (
        KANDINSKY5_LITE_SHA256,
        dict(custom_steps=50, cfg=3.5, sampler_name="euler", scheduler="simple", clip_skip=1),
    ),
    "Omnigen2": (
        OMNIGEN2_FP16_SHA256,
        dict(custom_steps=20, cfg=5, sampler_name="euler", scheduler="simple", clip_skip=1),
    ),
}
NATIVE_BASELINE_CONFIGS = {
    "HunyuanDiT": dict(image_model="hydit"),
    "Kandinsky5Image": dict(image_model="kandinsky5", model_dim=2560, visual_embed_dim=64),
    "Omnigen2": dict(
        image_model="omnigen2", hidden_size=2520, num_layers=32, in_channels=16, patch_size=2
    ),
}


class UnavailableModelRecipe(ValueError):
    """The model can load, but Auto cannot select a supported sampling recipe."""


class PerformanceSettings:
    DEFAULT_PERFORMANCE_FILE = resource_path("defaults", "index.json")
    PERFORMANCE_FILE = data_path("settings", "performance.json")
    CUSTOM_PERFORMANCE = "Own settings"
    MODEL_PERFORMANCE = MODEL_DEFAULT_SELECTIONS[0]
    LEGACY_MODEL_PERFORMANCE = MODEL_DEFAULT_SELECTIONS[1]

    RETIRED_PRESETS = {
        "Speed",
        "Quality",
        "Lcm",
        "Lightning",
        "SD3",
        "Flux",
        "SDXL - Balanced",
        "Illustrious XL - Balanced",
        "Animagine XL 3.1",
        "NoobAI XL - EPS",
        "SDXL Lightning - 2 steps",
        "SDXL Lightning - 4 steps",
        "SDXL Lightning - 8 steps",
        "FLUX.1 dev - 20 steps",
        "SD3.5 Large",
        "SD3.5 Large Turbo",
    }

    default_settings = {
        "custom_steps": 30,
        "cfg": 8,
        "sampler_name": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
        "clip_skip": 1,
    }

    def __init__(self, cache_dir="cache/checkpoints", offline=False):
        from ruinedfooocus.features.models.model_recommendations import ModelRecommendations

        self.model_recommendations = ModelRecommendations(cache_dir=cache_dir, offline=offline)
        self.performance_options = self.load_performance()

    def load_performance(self):
        default_data = self._load_data(self.DEFAULT_PERFORMANCE_FILE)
        data = self._load_data(self.PERFORMANCE_FILE)

        # Retain old recipes for saved selections, but keep them out of the menu.
        self.legacy_options = {
            name: value
            for name, value in data.items()
            if name in self.RETIRED_PRESETS
            or value.get("video_models")
            and not value.get("image_models")
        }
        current = {name: value for name, value in data.items() if name not in self.legacy_options}
        options = {
            name: defaults | current.get(name, {}) for name, defaults in default_data.items()
        }
        for name, defaults in default_data.items():
            options[name]["recommended_resolutions"] = defaults.get("recommended_resolutions", "")
        options.update({name: value for name, value in current.items() if name not in options})
        self._save_data(self.PERFORMANCE_FILE, self.legacy_options | options)
        return options

    def _load_data(self, file_path):
        if file_path == self.DEFAULT_PERFORMANCE_FILE and file_path.name == "index.json":
            from ruinedfooocus.features.models.resource_defaults import performance_defaults

            return performance_defaults(file_path.parent)
        if not file_path.exists():
            save_json(file_path, self._load_data(self.DEFAULT_PERFORMANCE_FILE))
        with open(file_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _save_data(self, file_path, data):
        save_json(file_path, data)

    def save_performance(self, perf_options):
        self._save_data(self.PERFORMANCE_FILE, self.legacy_options | perf_options)
        self.performance_options = self.load_performance()

    def get_perf_options(self, name, checkpoint=None, family=None):
        if name in MODEL_DEFAULT_SELECTIONS and checkpoint:
            samplers, schedulers = self.sampling_choices()

            metadata = self.model_recommendations.metadata(checkpoint)
            family = family or metadata.get("baseModel", "Unknown")
            baseline_family, baseline = self._native_variant_recipe(checkpoint, metadata, family)
            family = baseline_family or family
            native_recipe = self.model_recommendations.family_details(checkpoint, family=family)
            unavailable_native_recipe = bool(native_recipe) and not native_recipe.get(
                "auto_recipe_supported", True
            )
            preset = (
                None if family in QWEN_IMAGE21_FAMILIES else self.family_preset(family, checkpoint)
            )
            fallback = baseline or self.performance_options.get(preset, {})
            # Submission reads captured/local metadata only. Explicit Inspect/refresh
            # owns network metadata; waiting for a generation never polls a publisher.
            fixed_sigmas = None
            if family in QWEN_IMAGE21_FAMILIES:
                details = self.model_recommendations.details(
                    checkpoint, samplers, schedulers, family=family
                )
                recipe = details["recommended"] | details["override"]
                fixed_sigmas = details["fixed_sigmas"]
            else:
                recipe, _ = self.model_recommendations.recipe(checkpoint, samplers, schedulers)
            sampling_keys = {"custom_steps", "cfg", "sampler_name", "scheduler"}
            fallback_label = None
            if (
                not baseline
                and not unavailable_native_recipe
                and not sampling_keys.issubset(recipe)
            ):
                example = self.native_example(family, checkpoint)
                if example:
                    fallback = example["sampling"]
                    fallback_label = (
                        f"Native workflow example for {family}; no publisher recommendation"
                    )
            if not unavailable_native_recipe and not sampling_keys.issubset(fallback | recipe):
                ordinary_family = self._ordinary_native_family(checkpoint, metadata, family)
                if ordinary_family:
                    ordinary_preset = self.family_preset(ordinary_family, checkpoint)
                    family_fallback = self.performance_options.get(ordinary_preset, {})
                    fallback = self.default_settings | family_fallback
                    fallback_label = (
                        f"Family fallback settings for {ordinary_family}; no complete checkpoint recommendation"
                        if family_fallback
                        else "Generic app fallback settings; no complete checkpoint recommendation"
                    )
            if unavailable_native_recipe:
                override = self.model_recommendations.details(
                    checkpoint, samplers, schedulers, family=family
                )["override"]
                if not {"custom_steps", "cfg", "sampler_name", "scheduler"}.issubset(override):
                    raise UnavailableModelRecipe(
                        f"{family} requires its specialized workflow for Model Defaults. "
                        "Save explicit sampling settings or choose Own settings."
                    )
            if (
                family in EXPLICIT_RECIPE_FAMILIES
                or unavailable_native_recipe
                or (family in AMBIGUOUS_VARIANT_FAMILIES and preset is None)
                or (preset is None and family not in FLUX3_FAMILIES)
                or (family == "Unknown" and metadata.get("_rf_stale_native_family"))
            ) and not {
                "custom_steps",
                "cfg",
                "sampler_name",
                "scheduler",
            }.issubset(fallback | recipe):
                if family in QWEN_IMAGE21_FAMILIES:
                    raise UnavailableModelRecipe(
                        f"{family} has no qualified Model Defaults recipe. Use Inspect to verify "
                        "the published checkpoint. Choose Own settings for "
                        "other checkpoints, or save their explicit sampling settings."
                    )
                raise UnavailableModelRecipe(
                    f"{family} has no qualified Model Defaults recipe. Choose Own settings "
                    "with the checkpoint publisher's steps, CFG, sampler and scheduler, "
                    "or save those four sampling settings for this checkpoint."
                )
            if family in FLUX3_FAMILIES and preset is None:
                details = self.model_recommendations.details(checkpoint, samplers, schedulers)
                if set(details["override"]) != set(self.default_settings):
                    raise UnavailableModelRecipe(
                        "FLUX 3 Model Defaults need a signature-matched local header for the distilled "
                        "Klein variant. Base or unrecognized variants need explicit sampling settings."
                    )
            result = self.default_settings | fallback | recipe
            if fixed_sigmas is not None:
                result["_model_default_sigmas"] = fixed_sigmas
            if baseline:
                result["_recipe_baseline"] = f"Compatible native {baseline_family} variant baseline"
            elif fallback_label:
                result["_recipe_baseline"] = fallback_label
            elif (
                native_recipe.get("id", "").startswith("native-") and not unavailable_native_recipe
            ):
                result["_recipe_baseline"] = native_recipe["provenance"]["label"]
            elif family in FLUX3_FAMILIES and preset:
                result["_recipe_baseline"] = "Compatible native FLUX 3 flow-map variant baseline"
            elif preset and (preset in VARIANT_PRESETS or family in AMBIGUOUS_VARIANT_FAMILIES):
                result["_recipe_baseline"] = f"Compatible {preset} baseline"
            elif fallback and not sampling_keys.issubset(recipe):
                result["_recipe_baseline"] = (
                    f"Family fallback settings for {family}; no complete checkpoint recommendation"
                )
            return result
        # Stale/deleted names must return usable settings, never a string.
        options = self.performance_options.get(name, self.legacy_options.get(name, {}))
        return self.default_settings | options

    def control_options(self, name, checkpoint=None, family=None):
        """Keep UI controls constructible; generation still uses strict get_perf_options."""
        try:
            return self.get_perf_options(name, checkpoint, family)
        except UnavailableModelRecipe as error:
            return self.default_settings | {"_recipe_unavailable": str(error)}

    @staticmethod
    def sampling_choices():
        """Use loaded runtime enums, otherwise accepted bundled source constants."""
        current = getattr(sys.modules.get("comfy.samplers"), "KSampler", None)
        if current is not None:
            return tuple(current.SAMPLERS), tuple(current.SCHEDULERS)
        from ruinedfooocus.features.models.native_snapshot import native_sampler_choices

        return native_sampler_choices()

    @staticmethod
    def _native_ready():
        return (
            sys.modules.get("comfy.model_detection") is not None
            and getattr(sys.modules.get("comfy.samplers"), "KSampler", None) is not None
        )

    def _passive_example_contract(self, checkpoint, metadata, family):
        """Reuse the existing native shape predicate cache without a runtime import."""
        from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS
        from ruinedfooocus.features.models.model_identity import (
            cached_safetensors_header,
            file_signature,
            verified_native_source,
        )
        from ruinedfooocus.features.models.model_recommendations import _selected_checkpoint
        from ruinedfooocus.features.models.model_registry import RegistryError
        from ruinedfooocus.features.models.native_snapshot import get_image_family_facts
        from ruinedfooocus.features.models.resource_defaults import native_family_recipes
        from ruinedfooocus.features.models.utility_models import passive_native_class
        from ruinedfooocus.features.generation.native_sampling import SCHEDULERS

        protected = (*EXPLICIT_RECIPE_FAMILIES, *AMBIGUOUS_VARIANT_FAMILIES, *FLUX3_FAMILIES)
        if family in protected or ALIASES.get(family, family) not in IMAGE_COMPONENTS:
            return None, {}
        facts = get_image_family_facts(family)
        if facts is None:
            return None, {}
        try:
            selected = _selected_checkpoint(checkpoint, metadata)
            if selected is None:
                return None, {}
            signature = file_signature(selected)
            if Path(selected).suffix.lower() == ".gguf":
                from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

                header, _ = read_gguf_header(selected)
            else:
                header = cached_safetensors_header(selected)
            native = passive_native_class(selected, facts.native_classes, include_gguf=True)
            source = verified_native_source(selected, metadata, current_signature=signature)
            if (
                native is None
                and source is not None
                and (
                    ALIASES.get(source["native_family"], source["native_family"])
                    == ALIASES.get(family, family)
                    and source.get("native_class") in facts.native_classes
                )
            ):
                native = source["native_class"]
            if native is None:
                return None, {}
            contracts = native_family_recipes(native)
            if (
                len(contracts) > 1
                or any(
                    row["applicability"].get("runtime_requirements")
                    or not row["applicability"].get("auto_recipe_supported", True)
                    or row.get("sampling", {}).get("scheduler") in SCHEDULERS
                    for row in contracts
                )
                or signature != file_signature(selected)
            ):
                return None, {}
            header_metadata = header.get("__metadata__")
            return selected, {
                "native_class": native,
                "metadata": header_metadata if isinstance(header_metadata, dict) else {},
            }
        except (OSError, RegistryError, ValueError, TypeError, AttributeError, KeyError):
            return None, {}

    def native_example(self, family, checkpoint):
        """Only a current ordinary native configuration may inherit a T2I example."""
        from ruinedfooocus.features.models.model_architecture import inspect_model_file
        from ruinedfooocus.features.models.model_recommendations import _selected_checkpoint
        from ruinedfooocus.features.models.resource_defaults import native_workflow_example

        metadata = self.model_recommendations.metadata(checkpoint)
        if self._native_ready():
            current = self._ordinary_native_family(checkpoint, metadata, family, allow_variant=True)
            selected = _selected_checkpoint(checkpoint, metadata) if current else None
            contract = inspect_model_file(selected) if selected is not None else {}
        else:
            current = family
            selected, contract = self._passive_example_contract(checkpoint, metadata, family)
        if selected is None:
            return {}
        samplers, schedulers = self.sampling_choices()
        if not samplers or not schedulers:
            return {}
        from ruinedfooocus.features.models.model_identity import file_signature

        variants = {
            value.strip().casefold()
            for source in (
                contract.get("metadata", {}),
                metadata if metadata.get("_rf_file_signature") == file_signature(selected) else {},
            )
            for key in ("modelspec.variant", "variant", "model_variant", "training_variant")
            if isinstance(value := source.get(key), str) and value.strip()
        }
        if len(variants) > 1:
            return {}
        return native_workflow_example(
            current,
            native_class=contract.get("native_class"),
            variant=next(iter(variants)) if variants else None,
            samplers=samplers,
            schedulers=schedulers,
        )

    def _ordinary_native_family(self, checkpoint, metadata, family, *, allow_variant=False):
        """A missing optional recipe can use defaults only for a connected ordinary image."""
        from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS
        from ruinedfooocus.features.models.model_architecture import inspect_model_file
        from ruinedfooocus.features.models.model_identity import file_signature
        from ruinedfooocus.features.models.model_recommendations import _selected_checkpoint
        from ruinedfooocus.features.models.model_registry import RegistryError
        from ruinedfooocus.features.models.resource_defaults import native_family_recipes
        from ruinedfooocus.features.generation.native_sampling import SCHEDULERS

        protected = (*EXPLICIT_RECIPE_FAMILIES, *AMBIGUOUS_VARIANT_FAMILIES, *FLUX3_FAMILIES)
        if family in protected or not self._native_ready():
            return None
        try:
            selected = _selected_checkpoint(checkpoint, metadata)
            if selected is None:
                return None
            signature = file_signature(selected)
            contract = inspect_model_file(selected)
            native = contract.get("architecture")
            canonical = ALIASES.get(native, native)
            if (
                native in protected
                or canonical not in IMAGE_COMPONENTS
                or not contract.get("native_class")
                or family not in (None, "", "Unknown")
                and ALIASES.get(family, family) != canonical
            ):
                return None
            contracts = native_family_recipes(contract["native_class"])
            if len(contracts) > 1 or any(
                recipe["applicability"].get("runtime_requirements")
                or not recipe["applicability"].get("auto_recipe_supported", True)
                or recipe.get("sampling", {}).get("scheduler") in SCHEDULERS
                for recipe in contracts
            ):
                return None
            declarations = [contract.get("metadata", {})]
            if metadata.get("_rf_file_signature") == signature:
                declarations.append(metadata)
            if not allow_variant and any(
                isinstance(value := declaration.get(key), str) and value.strip()
                for declaration in declarations
                for key in ("modelspec.variant", "variant", "model_variant", "training_variant")
            ):
                return None
            return native if signature == file_signature(selected) else None
        except (OSError, RegistryError, ValueError, TypeError, AttributeError, KeyError):
            return None

    def _native_variant_recipe(self, checkpoint, metadata, family):
        """Use exact native configuration matches, not URLs, precision or hashes."""
        from ruinedfooocus.features.models.model_architecture import inspect_model_file
        from ruinedfooocus.features.models.model_identity import file_signature

        # Other families retain their existing contracts; Unknown may be an unindexed import.
        if family not in (*PAYLOAD_RECIPES, "Unknown", None, "") or not self._native_ready():
            return None, {}
        try:
            selected = Path(checkpoint)
            if not selected.is_absolute():
                models = getattr(sys.modules.get("ruinedfooocus.app.state"), "models", None)
                selected = (
                    models.get_file("checkpoints", checkpoint) if models is not None else None
                )
                if selected is None:
                    return None, {}
            selected = Path(selected)
            signature = file_signature(selected)
            contract = inspect_model_file(selected)
            native = contract.get("architecture")
            if native in (*FLUX3_FAMILIES, *AMBIGUOUS_VARIANT_FAMILIES):
                return native, {}
            if native not in PAYLOAD_RECIPES:
                return (native, {}) if native in EXPLICIT_RECIPE_FAMILIES else (None, {})
            if contract.get("native_class") != native or any(
                contract.get("native_config", {}).get(key) != value
                for key, value in NATIVE_BASELINE_CONFIGS[native].items()
            ):
                return native, {}
            if family in PAYLOAD_RECIPES and family != native:
                return None, {}
            # Only explicit, current behavioral-variant declarations can contradict
            # the normal baseline. Missing source/training metadata is not ambiguity.
            current = metadata if metadata.get("_rf_file_signature") == signature else {}
            declarations = [contract.get("metadata", {}), current]
            incompatible = {
                "distill",
                "distilled",
                "turbo",
                "lightning",
                "refiner",
                "edit",
                "inpaint",
            }
            for declaration in declarations:
                for key in ("modelspec.variant", "variant", "model_variant", "training_variant"):
                    value = declaration.get(key)
                    if isinstance(value, str) and value.strip().lower() in incompatible:
                        return native, {}
            return native, PAYLOAD_RECIPES[native][1].copy()
        except (OSError, ValueError, TypeError, AttributeError, IndexError):
            return None, {}

    def family_preset(self, family, checkpoint):
        from ruinedfooocus.features.models.model_recommendations import (
            FAMILY_PRESETS,
            family_recipe,
        )

        preset = FAMILY_PRESETS.get(family, family)
        if family in QWEN_IMAGE21_FAMILIES:
            metadata = self.model_recommendations.metadata(checkpoint)
            source = compatible_qwen_image21_variant(metadata, checkpoint)
            if source:
                return "Qwen Image 2.1 Turbo" if source["behavioral_variant"] == "turbo" else preset
            return None
        if family in FLUX3_FAMILIES:
            return self._flux3_preset(checkpoint)
        from ruinedfooocus.features.models.model_recommendations import _selected_checkpoint
        from ruinedfooocus.features.models.native_recipes import compatible_native_recipe
        from ruinedfooocus.features.models.resource_defaults import has_native_family_recipes

        metadata = self.model_recommendations.metadata(checkpoint)
        selected = _selected_checkpoint(checkpoint, metadata)
        native = compatible_native_recipe(selected, metadata) if selected is not None else {}
        if native.get("id", "").startswith("native-"):
            return native["id"]
        if preset in VARIANT_PRESETS:
            variant = self._declared_variant(checkpoint, preset)
            if variant is not None:
                return VARIANT_PRESETS[preset].get(variant)
            if family in AMBIGUOUS_VARIANT_FAMILIES:
                return None
        if has_native_family_recipes(family) and not family_recipe(preset).get("legacy"):
            return None
        return preset

    def _declared_variant(self, checkpoint, preset):
        """Use current bounded metadata declarations, never artifact names or dtype."""
        from ruinedfooocus.features.models.model_architecture import inspect_model_file
        from ruinedfooocus.features.models.model_identity import file_signature

        if not self._native_ready():
            return None
        metadata = self.model_recommendations.metadata(checkpoint)
        declarations = []
        try:
            selected = Path(checkpoint)
            if not selected.is_absolute():
                models = getattr(sys.modules.get("ruinedfooocus.app.state"), "models", None)
                selected = (
                    models.get_file("checkpoints", checkpoint) if models is not None else None
                )
                if selected is None:
                    signature = metadata.get("_rf_file_signature")
                    if not isinstance(signature, list) or len(signature) != 4:
                        return None
                    selected = Path(signature[0])
                    if selected.name != Path(checkpoint).name:
                        return None
            signature = file_signature(selected)
            declarations.append(inspect_model_file(selected).get("metadata", {}))
            if metadata.get("_rf_file_signature") == signature:
                declarations.append(metadata)
                from ruinedfooocus.features.models.model_identity import verified_native_source
                from ruinedfooocus.features.models.model_recommendations import FAMILY_PRESETS

                source = verified_native_source(selected, metadata, current_signature=signature)
                if (
                    source
                    and FAMILY_PRESETS.get(source["native_family"], source["native_family"])
                    == preset
                ):
                    declarations.append({"variant": source.get("behavioral_variant")})
        except (OSError, ValueError, TypeError, AttributeError, IndexError):
            return None
        variants = {
            value.strip().casefold()
            for declaration in declarations
            for key in ("modelspec.variant", "variant", "model_variant", "training_variant")
            if isinstance(value := declaration.get(key), str) and value.strip()
        }
        if variants:
            return next(iter(variants)) if len(variants) == 1 else "conflicting"
        # A signature-bound publisher version/model title describes the exact
        # source, unlike the local filename. Read only bounded variant tokens.
        current = metadata if metadata in declarations else {}
        if not (
            (type(current.get("id")) is int and current["id"] > 0)
            or (current.get("hf_repo_id") and self.model_recommendations.cached_hash(current))
        ):
            return None
        model = current.get("model")
        titles = (current.get("name"), model.get("name") if isinstance(model, dict) else None)
        tokens = {
            token
            for title in titles
            if isinstance(title, str) and len(title) <= 256
            for token in re.findall(r"[a-z0-9]+", title.casefold())
        }
        expected = set(VARIANT_PRESETS[preset])
        found = tokens & expected
        if preset == "HiDream O1" and found == {"full", "base"}:
            found = {"full"}
        return next(iter(found)) if len(found) == 1 else "conflicting" if found else None

    def _flux3_preset(self, checkpoint):
        """Recognize the qualified flow-map variant, never infer it from a filename."""
        from ruinedfooocus.features.models.model_architecture import inspect_model_file
        from ruinedfooocus.features.models.model_identity import file_signature

        if not self._native_ready():
            return None
        try:
            selected = Path(checkpoint)
            if not selected.is_absolute():
                models = getattr(sys.modules.get("ruinedfooocus.app.state"), "models", None)
                path = models.get_file("checkpoints", checkpoint) if models is not None else None
                if path is None:
                    # A signature-bound display cache remains usable before the
                    # application model index is available, including CPU callers.
                    signature = self.model_recommendations.metadata(checkpoint).get(
                        "_rf_file_signature"
                    )
                    if not isinstance(signature, list) or len(signature) != 4:
                        return None
                    path = Path(signature[0])
                    if path.name != selected.name or file_signature(path) != signature:
                        return None
                selected = Path(path)
            contract = inspect_model_file(selected)
            if (
                contract.get("native_class") == "Flux3"
                and contract.get("native_config", {}).get("flow_map") is True
                and contract.get("native_variant") == "flow_map"
            ):
                return "FLUX 3 Klein"
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        return None

    def apply(self, gen_data, family=None):
        result = gen_data.copy()
        result.pop("_model_default_sampling_shift", None)
        result.pop("_model_default_guidance", None)
        result.pop("_model_default_sigmas", None)
        result.pop("sampling_defaults_source", None)
        from ruinedfooocus.features.generation.native_sampling import KEY

        result.pop(KEY, None)
        name = result.get("performance_selection")
        if name and name not in (self.CUSTOM_PERFORMANCE, "Custom..."):
            options = self.get_perf_options(name, result.get("base_model_name"), family)
            result.update({key: options[key] for key in self.default_settings})
            if name in MODEL_DEFAULT_SELECTIONS and options.get("_recipe_baseline"):
                result["sampling_defaults_source"] = options["_recipe_baseline"]
            if name in MODEL_DEFAULT_SELECTIONS and options.get("sampling_shift") is not None:
                result["_model_default_sampling_shift"] = options["sampling_shift"]
            if options.get("_model_default_sigmas") is not None:
                result["_model_default_sigmas"] = options["_model_default_sigmas"]
            if options.get(KEY) is not None:
                result[KEY] = options[KEY]
            if name in MODEL_DEFAULT_SELECTIONS and family in ("Lens", "MingImage"):
                result["_model_default_guidance"] = {
                    key: options[key]
                    for key in ("cfg_norm_strength", "model_sampling_flux")
                    if key in options
                }
        return result

    def describe(self, name, locale=None):
        from ruinedfooocus.core.translation_manager import localize

        resolution = self.performance_options.get(name, {}).get("recommended_resolutions", "")
        return localize("Recommended: {sizes}", locale, sizes=resolution) if resolution else ""

    def describe_workflow(
        self, family, checkpoint, selection, steps, cfg, sampler, scheduler, locale=None
    ):
        from ruinedfooocus.core.translation_manager import localize

        values = dict(custom_steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler)
        if selection != self.CUSTOM_PERFORMANCE:
            options = self.control_options(selection, checkpoint, family)
            if options.get("_recipe_unavailable"):
                return options["_recipe_unavailable"]
            values.update(options)
        schedule = values["scheduler"]
        guidance = "CFG"
        text = (
            localize("Steps", locale)
            + f": {values['custom_steps']} · {guidance}: {values['cfg']}  \n"
            + localize("Sampler", locale)
            + f": {values['sampler_name']} · "
            + localize("Schedule", locale)
            + f": {schedule}\n\n"
        )
        options = self.get_perf_options(selection, checkpoint, family)
        text += (
            localize("Clip Skip", locale) + f": {options['clip_skip']}"
            if selection != self.CUSTOM_PERFORMANCE
            else ""
        )
        return text

    def choices_for_model(self, model_base, checkpoint=None):
        return [self.MODEL_PERFORMANCE, self.CUSTOM_PERFORMANCE]

    def selection_for_model(self, family, checkpoint, current):
        return (
            self.CUSTOM_PERFORMANCE
            if current in (self.CUSTOM_PERFORMANCE, "Custom...")
            else self.MODEL_PERFORMANCE
        )
