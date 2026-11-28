"""Checkpoint-scoped choices and frozen image requests; no tensor runtime imports."""

from copy import deepcopy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from ruinedfooocus.backends.comfy.component_selection import normalize_component_overrides
from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.features.models.model_identity import file_signature

PROFILE_KEY = "image_component_profiles"
REQUEST_KEY = "image_component_overrides"
SCOPE_KEY = "_image_component_scope"


def component_contract(family, *, native_layout=None):
    """The actual selector's logical types/codecs, independent of payload precision."""
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS
    from ruinedfooocus.features.models.component_candidates import (
        TEXT_TYPES,
        VAE_CHANNELS,
        VAE_LAYOUTS,
    )

    if not isinstance(family, str):
        return {}
    canonical = ALIASES.get(family, family)
    components = IMAGE_COMPONENTS.get(canonical)
    if components is None:
        return {}
    if native_layout is not None:
        if native_layout["family"] != canonical:
            raise ValueError("The component display recipe differs from the checkpoint family.")
        components = dict(
            clip_names=native_layout["clip_roles"], vae_name=native_layout["vae_role"]
        )
    roles = []
    for key in components["clip_names"]:
        roles.append(
            dict(
                key=key,
                role="text_encoder",
                logical_types=sorted(TEXT_TYPES.get(key, ())),
                native_reader="native HunyuanDiT split encoder"
                if key in ("clip_hydit_bert", "clip_hydit_mt5")
                else "native CLIP loader with role-specific tokenizer and GGUF adapter",
                tokenizer_requirement="embedded tokenizer_json required"
                if key in ("clip_lens_gpt_oss", "clip_ming_image")
                else "embedded MT5 SentencePiece required"
                if key == "clip_hydit_mt5"
                else "native family tokenizer; an encoder label does not qualify its tokenizer",
                quantization_policy="logical family/type and native reader compatibility; no filename or dtype-only admission",
            )
        )
    vae = components["vae_name"]
    if vae != "pixel_space":
        roles.append(
            dict(
                key=vae,
                role="vae",
                latent_channels=VAE_CHANNELS.get(vae),
                layout=VAE_LAYOUTS.get(vae),
                native_reader="comfy.sd.VAE native state dictionary",
                gguf_image_supported=False,
                quantization_policy="native codec architecture/layout, not channel count or file label alone",
            )
        )
    return dict(
        family=canonical,
        roles=roles,
        pixel_space=vae == "pixel_space",
        embedded_components_first=True,
        explicit_checkpoint_overrides_first=True,
        unknown_files="visible through explicit advanced choice; not labelled compatible",
        mixed_encoder_policy="GGUF plus scaled FP8 remains incompatible with the pinned mixed reader",
        qualification="A compatible component contract does not certify its bytes, device fit or generated quality.",
    )


def checkpoint_identity(checkpoint):
    from ruinedfooocus.features.models.local_diffusers import is_diffusers_index, capture_bundle
    from ruinedfooocus.features.models.sensenova_bundle import (
        is_sensenova_index,
        capture_bundle as capture_sense_bundle,
    )

    if is_sensenova_index(checkpoint):
        return capture_sense_bundle(checkpoint).identity
    if is_diffusers_index(checkpoint):
        return capture_bundle(checkpoint).identity
    signature = file_signature(checkpoint)
    signature[0] = os.path.normcase(signature[0])
    return hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()


def _bind_capture_scope(result, checkpoint, identity):
    if checkpoint_identity(checkpoint) != identity:
        raise ValueError("The checkpoint changed during component capture. Select it again.")
    result[SCOPE_KEY] = identity
    return result


def validate_profiles(value):
    if not isinstance(value, dict):
        raise ValueError(
            "Image component profiles must map checkpoint identities to component choices."
        )
    result = {}
    for identity, choices in value.items():
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise ValueError("Invalid image checkpoint component identity.")
        result[identity] = normalize_component_overrides(choices)
    return result


def saved_choices(values, checkpoint):
    return validate_profiles(values.get(PROFILE_KEY, {})).get(checkpoint_identity(checkpoint), {})


def update_choices(values, checkpoint, identity, choices):
    """Return a new profile; never alter legacy globals or another checkpoint."""
    if checkpoint_identity(checkpoint) != identity:
        raise ValueError("The checkpoint changed. Refresh components before saving.")
    profiles = validate_profiles(values.get(PROFILE_KEY, {}))
    profiles[identity] = normalize_component_overrides(choices)
    return dict(deepcopy(values), **{PROFILE_KEY: profiles})


def component_inventory(checkpoint, family, values, paths, overrides=None):
    from ruinedfooocus.features.models.image_components import inventory
    from ruinedfooocus.features.models.local_diffusers import (
        is_diffusers_index,
        capture_bundle,
        bundle_inventory,
    )
    from ruinedfooocus.features.models.sensenova_bundle import (
        is_sensenova_index,
        capture_bundle as capture_sense_bundle,
        bundle_inventory as sense_inventory,
    )

    choices = (
        saved_choices(values, checkpoint)
        if overrides is None
        else normalize_component_overrides(overrides)
    )
    if is_sensenova_index(checkpoint):
        return dict(
            sense_inventory(capture_sense_bundle(checkpoint), choices),
            compatibility_contract=component_contract(family),
        )
    if is_diffusers_index(checkpoint):
        return dict(
            bundle_inventory(capture_bundle(checkpoint), family, choices),
            compatibility_contract=component_contract(family),
        )
    state = inventory(checkpoint, family, values, paths, component_overrides=choices)
    return dict(
        state,
        compatibility_contract=component_contract(
            family, native_layout=state.get("native_component_layout")
        ),
    )


def freeze_choices(checkpoint, family, values, paths, overrides=None, *, include_inventory=False):
    """Materialize every actual role, including legacy values, before queuing.

    Null resets to the architecture recipe. All slots are present so the native
    loader cannot inherit a later edit to a profile-wide legacy setting.
    """
    state = component_inventory(checkpoint, family, values, paths, overrides)
    from ruinedfooocus.features.models.component_candidates import validate_inventory

    validate_inventory(state)
    choices = (
        saved_choices(values, checkpoint)
        if overrides is None
        else normalize_component_overrides(overrides)
    )
    frozen = (
        {}
        if state.get("bundle_identity")
        else {
            row["key"]: None
            if row["status"] == "Embedded in checkpoint"
            else choices.get(row["key"], values.get(row["key"]) or None)
            for row in state["rows"]
        }
    )
    if include_inventory:
        from ruinedfooocus.features.models.effective_image_plan import capture_component_inventory

        return frozen, capture_component_inventory(state, choices)
    return frozen


def snapshot_request(data, models, values, paths):
    """Copy image choices at submission; skip non-image jobs."""
    result = dict(data)
    from ruinedfooocus.features.models.effective_image_plan import COMPONENTS_KEY

    result.pop(COMPONENTS_KEY, None)
    from ruinedfooocus.features.models.approximate_codecs import (
        KEY as APPROXIMATE_KEY,
        capture_request as capture_approximate,
    )

    previous_approximate = result.pop(APPROXIMATE_KEY, None)
    if result.get("task_type") in ("llama", "chat", "rembg", "upscale"):
        return result
    from ruinedfooocus.features.gallery.image_metadata import read_metadata

    imported = (
        read_metadata(result.get("prompt"))
        if "_api_job" not in result
        and result.get("task_type") != "tool_call"
        and not result.get("_metadata_prompt")
        else None
    )
    if not imported:
        prompt = result.get("prompt", "")
        if (
            isinstance(prompt, str)
            and result.get("task_type") != "tool_call"
            and "_api_job" not in result
            and not result.get("_metadata_prompt")
            and (
                prompt.lower() == "ruinedfooocuslogo"
                or prompt.startswith("#!")
                or prompt.lower().startswith("search:")
            )
        ):
            return result
        control = result.get("controlnet") or {}
        # Reuse initialized routing without importing its application state.
        module = sys.modules.get("ruinedfooocus.features.image_editing.controlnet")
        if module is not None:
            control = module.get_settings(result)
        if control.get("type", "").lower() in ("rembg", "upscale"):
            result.pop(REQUEST_KEY, None)
            return result
    if imported:
        previous_approximate = None
        result["base_model_name"] = imported.get("base_model_name", result.get("base_model_name"))
        result[REQUEST_KEY] = imported.get(REQUEST_KEY)
        result.pop(SCOPE_KEY, None)
    name = result.get("base_model_name")
    if not name:
        return result
    info = models.get_models_by_path("checkpoints", name, fetch=False)
    family = models.get_model_base(info)
    from ruinedfooocus.features.models.local_diffusers import (
        BUNDLE_KEY,
        is_diffusers_index,
        capture_bundle,
        bundle_inventory,
    )

    result.pop(BUNDLE_KEY, None)
    from ruinedfooocus.features.models.sensenova_bundle import (
        KEY as SENSE_KEY,
        is_sensenova_index,
        capture_bundle as capture_sense_bundle,
        bundle_inventory as sense_inventory,
    )

    result.pop(SENSE_KEY, None)
    if is_sensenova_index(name):
        checkpoint = models.get_file("checkpoints", name)
        if checkpoint is None:
            raise ValueError("The selected SenseNova shard bundle is no longer installed.")
        bundle = capture_sense_bundle(checkpoint)
        if result.get(SCOPE_KEY) and result[SCOPE_KEY] != bundle.identity:
            result.pop(REQUEST_KEY, None)
            result.pop(SCOPE_KEY, None)
        from ruinedfooocus.features.models.effective_image_plan import capture_component_inventory

        state = sense_inventory(bundle, normalize_component_overrides(result.get(REQUEST_KEY)))
        result[COMPONENTS_KEY] = capture_component_inventory(state)
        result[SENSE_KEY] = bundle
        result[REQUEST_KEY] = {}
        result[SCOPE_KEY] = bundle.identity
        return result
    if is_diffusers_index(name):
        checkpoint = models.get_file("checkpoints", name)
        if checkpoint is None:
            raise ValueError("The selected local Diffusers bundle is no longer installed.")
        bundle = capture_bundle(checkpoint)
        if result.get(SCOPE_KEY) and result[SCOPE_KEY] != bundle.identity:
            result.pop(REQUEST_KEY, None)
            result.pop(SCOPE_KEY, None)
        from ruinedfooocus.features.models.effective_image_plan import capture_component_inventory

        state = bundle_inventory(
            bundle, family, normalize_component_overrides(result.get(REQUEST_KEY))
        )
        result[COMPONENTS_KEY] = capture_component_inventory(state)
        result[BUNDLE_KEY] = bundle
        result[REQUEST_KEY] = {}
        result[SCOPE_KEY] = bundle.identity
        return result
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS

    if ALIASES.get(family, family) not in IMAGE_COMPONENTS:
        return result
    checkpoint = models.get_file("checkpoints", name)
    if checkpoint is None:
        raise ValueError("The selected image checkpoint is no longer installed.")
    identity = checkpoint_identity(checkpoint)
    if result.get(SCOPE_KEY) and result[SCOPE_KEY] != identity:
        result.pop(REQUEST_KEY, None)
        result.pop(SCOPE_KEY, None)
    if not is_safetensors(checkpoint) and Path(checkpoint).suffix.lower() != ".gguf":
        from ruinedfooocus.features.models.model_identity import verified_native_source

        if verified_native_source(checkpoint, info) is not None:
            result[REQUEST_KEY], result[COMPONENTS_KEY] = freeze_choices(
                checkpoint, family, values, paths, result.get(REQUEST_KEY), include_inventory=True
            )
            result = _bind_capture_scope(result, checkpoint, identity)
            return capture_approximate(result, paths, previous=previous_approximate)
        # Reading pickle weights to discover bundles is not a browsing action.
        # Retain legacy fallback instead of turning an inherited global into
        # an explicit (and forbidden) bundled replacement.
        if result.get(REQUEST_KEY) is None:
            result.pop(REQUEST_KEY, None)
        else:
            result[REQUEST_KEY] = normalize_component_overrides(result[REQUEST_KEY])
        result[COMPONENTS_KEY] = ()
        result = _bind_capture_scope(result, checkpoint, identity)
        return capture_approximate(result, paths, previous=previous_approximate)
    overrides = result.get(REQUEST_KEY)
    result[REQUEST_KEY], result[COMPONENTS_KEY] = freeze_choices(
        checkpoint, family, values, paths, overrides, include_inventory=True
    )
    result = _bind_capture_scope(result, checkpoint, identity)
    return capture_approximate(result, paths, previous=previous_approximate)


def session_choices(checkpoint, values, choices=None, scope=None):
    """A replay tagged for this exact file survives the subsequent change event."""
    identity = checkpoint_identity(checkpoint)
    return (
        normalize_component_overrides(choices)
        if scope == identity
        else saved_choices(values, checkpoint)
    ), identity


def replay_choices(checkpoint, values, choices=None):
    """Tag Library/preset choices so a following model change preserves them."""
    return (
        saved_choices(values, checkpoint)
        if choices is None
        else normalize_component_overrides(choices)
    ), checkpoint_identity(checkpoint)


def apply_preset_choices(data, preset_data, models, values):
    """Apply only when the caller's preset mode includes model settings."""
    result = dict(data)
    checkpoint = models.get_file("checkpoints", result["base_model_name"])
    if checkpoint is None:
        raise ValueError("The preset's image checkpoint is no longer installed.")
    result[REQUEST_KEY], result[SCOPE_KEY] = replay_choices(
        checkpoint,
        values,
        preset_data.get(REQUEST_KEY),
    )
    return result
