"""Validated native hypernetworks on request-local SD attention callbacks."""

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
import math

from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
    AuxiliaryAsset,
    artifact,
    attention_layout,
    selected_asset,
    require_ordinary_codec,
    safe_native_state,
)

KEY = "_native_hypernetwork"
FIELDS = ("hypernetwork_model", "hypernetwork_model_signature", "hypernetwork_strength")
RESETS = dict(hypernetwork_model=None, hypernetwork_model_signature=None, hypernetwork_strength=1.0)
ACTIVATIONS = frozenset(
    ("linear", "relu", "leakyrelu", "elu", "swish", "tanh", "sigmoid", "softsign", "mish")
)


@dataclass(frozen=True)
class HypernetworkPlan:
    asset: AuxiliaryAsset
    checkpoint_signature: tuple
    strength: float


def validate_state(state, widths):
    """Bound the native nested schema before it constructs dense Linear modules."""
    if not isinstance(state, Mapping) or len(state) > 128:
        raise ValueError("Choose a bounded native nested hypernetwork state dictionary.")
    activation = state.get("activation_func", "linear")
    if not isinstance(activation, str) or activation not in ACTIVATIONS:
        raise ValueError("The native hypernetwork activation is unsupported; no patch was applied.")
    for key in ("is_layer_norm", "use_dropout", "activate_output", "last_layer_dropout"):
        value = state.get(key, False)
        if type(value) not in (bool, int) or value not in (False, True):
            raise ValueError("Native hypernetwork normalization/dropout flags must be booleans.")
    for key in state:
        if type(key) is int:
            continue
        try:
            int(key)
        except (TypeError, ValueError, OverflowError):
            continue
        raise ValueError("Native hypernetwork attention dimensions must use integer keys.")
    dimensions = [key for key in state if type(key) is int]
    if not dimensions or len(dimensions) > 32 or any(not 1 <= key <= 8192 for key in dimensions):
        raise ValueError("The hypernetwork has no bounded native attention dimensions.")
    if not set(dimensions) <= set(widths):
        raise ValueError(
            "Hypernetwork dimensions do not match this UNet's native attention inputs."
        )
    elements = 0
    for dimension in dimensions:
        pair = state[dimension]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each hypernetwork dimension needs separate key and value networks.")
        for weights in pair:
            if not isinstance(weights, Mapping) or not 1 <= len(weights) <= 128:
                raise ValueError("The native hypernetwork layer state is malformed or oversized.")
            if any(not isinstance(key, str) or len(key) > 256 for key in weights):
                raise ValueError("The native hypernetwork layer names are invalid.")
            linears = [key[:-7] for key in weights if key.endswith(".weight")]
            if not linears or set(weights) != {
                f"{key}.{suffix}" for key in linears for suffix in ("weight", "bias")
            }:
                raise ValueError("Native hypernetwork layers require paired weights and biases.")
            incoming, index = dimension, 0
            while index < len(linears):
                name = linears[index]
                weight, bias = weights[name + ".weight"], weights[name + ".bias"]
                if (
                    not hasattr(weight, "is_floating_point")
                    or not weight.is_floating_point()
                    or not hasattr(bias, "is_floating_point")
                    or not bias.is_floating_point()
                ):
                    raise ValueError(
                        "Hypernetwork weights and biases must be native floating tensors."
                    )
                if (
                    weight.ndim != 2
                    or bias.ndim != 1
                    or tuple(weight.shape) != (bias.shape[0], incoming)
                    or not 1 <= weight.shape[0] <= 32768
                ):
                    raise ValueError("The native hypernetwork Linear dimensions are incompatible.")
                incoming = int(weight.shape[0])
                elements += weight.numel() + bias.numel()
                if state.get("is_layer_norm", False):
                    index += 1
                    if index >= len(linears):
                        raise ValueError("Native hypernetwork LayerNorm parameters are missing.")
                    name = linears[index]
                    norm, offset = weights[name + ".weight"], weights[name + ".bias"]
                    if (
                        not hasattr(norm, "is_floating_point")
                        or not hasattr(offset, "is_floating_point")
                        or tuple(norm.shape) != (incoming,)
                        or tuple(offset.shape) != (incoming,)
                        or not norm.is_floating_point()
                        or not offset.is_floating_point()
                    ):
                        raise ValueError(
                            "Native hypernetwork LayerNorm dimensions are incompatible."
                        )
                    elements += norm.numel() + offset.numel()
                index += 1
            if incoming != dimension:
                raise ValueError(
                    "Native hypernetwork output width must equal its attention input width."
                )
    if elements > 128 * 1024**2:
        raise ValueError("The hypernetwork exceeds the bounded dense-parameter budget.")
    return tuple(sorted(dimensions))


def capture_assets(data, manager, models, family):
    from ruinedfooocus.features.image_editing.gligen_layout import FAMILIES
    from ruinedfooocus.features.models.image_components import ALIASES

    result = dict(data)
    previous = result.pop(KEY, None)
    if not result.get("hypernetwork_model"):
        result.pop("hypernetwork_guidance", None)
        return result, ()
    if ALIASES.get(family, family) not in FAMILIES:
        raise ValueError("Hypernetworks require a compatible ordinary native SD image UNet.")
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Hypernetworks require the native image engine.")
    if str((result.get("controlnet") or {}).get("type", "")).lower() not in (
        "",
        "none",
        "edit",
        "img2img",
        "canny",
        "depth",
        "recolour",
        "sketch",
        "control",
    ):
        raise ValueError("Hypernetworks require native SD generation or an SD image edit tool.")
    from ruinedfooocus.features.generation.native_tensor_io import enabled as tensor_io_enabled

    if tensor_io_enabled(result):
        # The callback remains on the rebuilt request model. Numeric files bind
        # its source/strength receipt; no executable callback is serialized.
        result["native_tensor_workflow"] = True
    strength = result.get("hypernetwork_strength", 1.0)
    if (
        type(strength) not in (int, float)
        or not math.isfinite(strength)
        or not -10 <= strength <= 10
    ):
        raise ValueError("Hypernetwork strength must be a finite number from -10 through 10.")
    asset = selected_asset(
        manager,
        "hypernetwork",
        result["hypernetwork_model"],
        result.get("hypernetwork_model_signature"),
    )
    if is_safetensors(asset.signature[0]):
        raise ValueError(
            "Native hypernetworks require their nested Torch state, not flat safetensors weights."
        )
    checkpoint = models.get_file("checkpoints", result.get("base_model_name"))
    plan = HypernetworkPlan(asset, tuple(file_signature(checkpoint)), float(strength))
    if isinstance(previous, HypernetworkPlan) and (
        previous.asset != asset or previous.checkpoint_signature != plan.checkpoint_signature
    ):
        raise ValueError("The queued hypernetwork sources changed. Submit a new request.")
    result.update(
        {
            KEY: plan,
            "image_engine": "native",
            "hypernetwork_model": asset.name,
            "hypernetwork_model_signature": list(asset.public_signature),
            "hypernetwork_strength": plan.strength,
        }
    )
    return result, (artifact(asset),)


def clear_cache(owner):
    cached = getattr(owner, "_native_hypernetwork", None)
    if cached is not None:
        owner._native_hypernetwork = None
        cached[1].to(cached[3])


def offload(owner):
    cached = getattr(owner, "_native_hypernetwork", None)
    if cached is not None:
        cached[1].to(cached[3])


def patch(owner, model, data, manager, check_cancelled):
    from comfy_extras.nodes_hypernetwork import load_hypernetwork_patch
    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    plan = data.get(KEY)
    if (
        not isinstance(plan, HypernetworkPlan)
        or tuple(file_signature(plan.checkpoint_signature[0])) != plan.checkpoint_signature
    ):
        raise ValueError("Capture the current hypernetwork and checkpoint before generation.")
    require_ordinary_codec(owner.xl_base_patched.vae)
    _, _, widths = attention_layout(model)
    check_cancelled()
    with prepared_asset(data, "hypernetwork", plan.asset.name, manager, "hypernetworks") as path:
        cached = getattr(owner, "_native_hypernetwork", None)
        if cached is None or cached[0] != plan.asset.signature:
            clear_cache(owner)
            state = safe_native_state(path)
            dimensions = validate_state(state, widths)
            del state
            check_cancelled()
            native = load_hypernetwork_patch(str(path), 1.0)
            if native is None or set(native.hypernet) != set(dimensions):
                raise ValueError(
                    "The native hypernetwork loader returned no compatible attention patch."
                )
            cached = (plan.asset.signature, native, dimensions, model.offload_device)
            owner._native_hypernetwork = cached
        if not set(cached[2]) <= set(widths):
            raise ValueError("The cached hypernetwork dimensions do not match this UNet.")
        if tuple(file_signature(path)) != plan.asset.signature:
            clear_cache(owner)
            raise ValueError("The hypernetwork changed during loading. Submit the request again.")
    check_cancelled()
    native = copy(cached[1])
    native.strength = plan.strength
    result = model.clone()
    result.set_model_attn1_patch(native)
    result.set_model_attn2_patch(native)
    data["hypernetwork_guidance"] = dict(
        recipe="native-hypernetwork-v1",
        source=plan.asset.metadata(),
        dimensions=list(cached[2]),
        strength=plan.strength,
        sampling_target="primary-model",
    )
    return result
