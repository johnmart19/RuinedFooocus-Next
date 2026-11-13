"""Current source/configuration recipes, independent of exact-file qualification."""

import sys

from ruinedfooocus.features.models.model_registry import RegistryError
from ruinedfooocus.features.models.resource_defaults import (
    load_family_recipe,
    native_family_recipes,
)

_VARIANT_KEYS = ("modelspec.variant", "variant", "model_variant", "training_variant")


def current_native_recipe(path, manager, metadata=None):
    """Share the prepared-source cache boundary with UI/default lookup and loaders."""
    from pathlib import Path
    from ruinedfooocus.features.models.model_recommendations import ModelRecommendations

    values = dict(metadata or {})
    cache = getattr(manager, "model_paths", {}).get("cache_path")
    if cache is not None:
        current = ModelRecommendations(
            cache_dir=Path(cache) / "checkpoints", offline=True
        ).metadata(str(path))
        values.update(current)
    return compatible_native_recipe(path, values)


def compatible_native_recipe(path, metadata, *, root=None):
    """Use an exact source or initialized native header, never a name or dtype.

    A returned source contract can have auto_recipe_supported=False: geometry and
    missing native nodes remain inspectable, but sampling must not use it as Auto.
    """
    from ruinedfooocus.features.models.model_identity import (
        file_signature,
        verified_native_source,
    )

    try:
        signature = file_signature(path)
        if metadata.get("_rf_stale_native_family") or (
            metadata.get("_rf_file_signature") is not None
            and metadata["_rf_file_signature"] != signature
        ):
            return {}
        from ruinedfooocus.features.generation.bernini_image import primary_family

        if primary_family(path):
            recipe = load_family_recipe("native-bernini-image", root=root)
            recipe["identity_basis"] = "exact_publisher_header_then_signed_preparation"
            recipe["artifact_qualified"] = False
            return recipe
        source = verified_native_source(path, metadata, current_signature=signature)
        if source is not None:
            evidence = source.get("evidence", {})
            recipe = load_family_recipe(evidence.get("default_recipe_id"), root=root)
            if recipe:
                recipe["identity_basis"] = "verified_publisher_artifact"
                recipe["artifact_qualified"] = False
                recipe["auto_recipe_supported"] = evidence.get(
                    "auto_recipe_supported",
                    recipe["applicability"].get("auto_recipe_supported", True),
                )
                return recipe if signature == file_signature(path) else {}
        # Browsing/default lookup must not initialize the native runtime.
        if sys.modules.get("comfy.model_detection") is None:
            return {}
        from ruinedfooocus.features.models.model_architecture import inspect_model_file

        contract = inspect_model_file(path)
        candidates = [
            recipe
            for recipe in native_family_recipes(contract.get("native_class"), root=root)
            if recipe["applicability"].get("architecture") == contract.get("architecture")
            and (
                "native_variant" not in recipe["applicability"]
                or recipe["applicability"]["native_variant"] == contract.get("native_variant")
            )
            and all(
                contract.get("native_config", {}).get(key) == value
                for key, value in recipe["applicability"].get("native_config", {}).items()
            )
        ]
        declarations = [contract.get("metadata", {})]
        native_config = contract.get("native_config", {})
        if (
            contract.get("native_class") == "QwenImage"
            and native_config.get("use_additional_t_cond") is True
            and native_config.get("default_ref_method") == "negative_index"
        ):
            # The native detector derives this variant from the actual Layered
            # timestep embedding; it is independent of container and scalar tags.
            declarations.append({"variant": "layered"})
        if metadata.get("_rf_file_signature") == signature:
            declarations.append(metadata)
        variants = {
            value.strip().casefold()
            for declaration in declarations
            for key in _VARIANT_KEYS
            if isinstance(value := declaration.get(key), str) and value.strip()
        }
        if variants:
            candidates = [
                recipe
                for recipe in candidates
                if variants <= set(recipe["applicability"]["behavioral_variants"])
            ]
        else:
            candidates = [
                recipe
                for recipe in candidates
                if recipe["applicability"].get("default_for_native_configuration")
            ]
        if len(candidates) != 1 or signature != file_signature(path):
            return {}
        recipe = candidates[0]
        recipe["identity_basis"] = "compatible_native_configuration_and_current_variant"
        recipe["artifact_qualified"] = False
        recipe["auto_recipe_supported"] = recipe["applicability"].get(
            "auto_recipe_supported", False
        )
        return recipe
    except (OSError, RegistryError, ValueError, TypeError, AttributeError, KeyError):
        return {}
