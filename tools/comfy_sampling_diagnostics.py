"""Scoped observers for direct-node/application sampler comparisons."""

from contextlib import contextmanager
import inspect
from unittest.mock import patch


def _marker(value):
    if callable(value):
        return (
            getattr(value, "__module__", "")
            + "."
            + getattr(value, "__qualname__", type(value).__name__)
        )
    return type(value).__module__ + "." + type(value).__name__


@contextmanager
def observe_sampling(enabled=False):
    """Observe CFGGuider.sample without altering its inputs, callback or result."""
    artifacts = {}
    if not enabled:
        yield artifacts
        return
    from comfy.samplers import CFGGuider

    original = CFGGuider.sample
    signature = inspect.signature(original)

    def observe(guider, *args, **kwargs):
        bound = signature.bind(guider, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        patcher = guider.model_patcher
        sampling = patcher.get_model_object("model_sampling")
        options = guider.model_options
        transformer = options.get("transformer_options", {})
        model = patcher.model
        artifacts.update(
            noise=values["noise"].detach().cpu().clone(),
            initial_latent=values["latent_image"].detach().cpu().clone(),
            sigmas=values["sigmas"].detach().cpu().clone(),
        )
        artifacts["effective"] = {
            "cfg": guider.cfg,
            "seed": values["seed"],
            "sampling_class": type(sampling).__name__,
            "sampling_mro": [kind.__name__ for kind in type(sampling).__mro__],
            "sampling_scalars": {
                key: value
                for key in ("shift", "multiplier", "noise_scale")
                if isinstance((value := getattr(sampling, key, None)), (str, int, float, bool))
            },
            "patcher_class": type(patcher).__name__,
            "model_class": type(model).__name__,
            "model_dtype": str(model.get_dtype()),
            "inference_dtype": str(model.get_dtype_inference()),
            "latent_format": type(model.latent_format).__name__,
            "sampler_function": _marker(values["sampler"].sampler_function),
            "model_option_keys": sorted(str(key) for key in options),
            "model_option_markers": {
                str(key): _marker(value)
                for key, value in options.items()
                if key != "transformer_options"
            },
            "transformer_option_keys": sorted(str(key) for key in transformer),
            "attention_override": _marker(transformer["optimized_attention_override"])
            if "optimized_attention_override" in transformer
            else None,
            "patch_keys": sorted(str(key) for key in getattr(patcher, "patches", {})),
            "object_patch_keys": sorted(str(key) for key in getattr(patcher, "object_patches", {})),
        }
        calls = artifacts.setdefault("model_apply_batches", [])
        first_calls = artifacts.setdefault("first_model_calls", [])
        apply_model = model.apply_model
        apply_signature = inspect.signature(apply_model)

        def observe_apply(*inputs, **apply_kwargs):
            call = apply_signature.bind(*inputs, **apply_kwargs)
            call.apply_defaults()
            data = call.arguments
            transformer_options = data.get("transformer_options", {})
            labels = transformer_options.get("cond_or_uncond")
            calls.append(
                {
                    "input_shape": list(data["x"].shape),
                    "timestep_shape": list(data["t"].shape),
                    "cross_attention_shape": list(data["c_crossattn"].shape)
                    if data.get("c_crossattn") is not None
                    else None,
                    "cond_or_uncond": list(labels) if labels is not None else None,
                }
            )
            snapshot = None
            if len(first_calls) < 2:
                # The harness fingerprints these CPU tensors, so identical
                # inputs and divergent predictions locate a model-evaluation
                # discrepancy rather than a sampler/noise discrepancy.
                snapshot = {
                    "input": data["x"].detach().cpu().clone(),
                    "timestep": data["t"].detach().cpu().clone(),
                }
            result = apply_model(*inputs, **apply_kwargs)
            if snapshot is not None:
                snapshot["prediction"] = result.detach().cpu().clone()
                first_calls.append(snapshot)
            return result

        with patch.object(model, "apply_model", observe_apply):
            return original(guider, *args, **kwargs)

    with patch.object(CFGGuider, "sample", observe):
        yield artifacts
