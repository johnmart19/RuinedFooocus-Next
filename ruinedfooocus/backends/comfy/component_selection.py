"""Pure request-local component choices at the native image loader boundary."""

from collections.abc import Mapping

from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS


def normalize_component_overrides(values):
    """Copy explicit choices; a present empty value resets the family recipe."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("Image component choices must be a mapping of slots to local filenames.")
    choices = {}
    for key, value in values.items():
        if key not in RESOURCE_DEFAULTS:
            raise ValueError(f"Unknown image component slot: {key}.")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Image component {key} must be a local filename or Automatic.")
        choices[key] = value.strip() or None if isinstance(value, str) else None
    return choices


def component_settings(defaults, overrides=None):
    """Keep legacy choices for absent keys without mutating the shared profile."""
    values = {key: value for key, value in defaults.items() if key in RESOURCE_DEFAULTS}
    values.update(normalize_component_overrides(overrides))
    return values


def validate_component_slots(overrides, model_info, *, bundled_clip=False, bundled_vae=False):
    """Bundled components remain native; external choices cannot splice a bundle."""
    from ruinedfooocus.features.models.approximate_codecs import choice_alias

    clips = set(model_info.get("clip_keys", ()))
    vae = model_info.get("vae_key")
    slots = clips | ({vae} if vae != "pixel_space" and vae else set())
    for key, value in overrides.items():
        if key not in slots:
            raise ValueError(
                f"Image component {key} is not a slot in this checkpoint's architecture."
            )
        approximate = choice_alias(key, value)
        if value and (
            (key in clips and bundled_clip) or (key == vae and bundled_vae and not approximate)
        ):
            raise ValueError(
                f"{key} is built into this checkpoint and cannot be replaced. "
                "Reset its checkpoint component choice to Automatic."
            )
