"""Private numeric file transport for the standalone native composition consumer."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import secrets
import types

from ruinedfooocus.features.models.model_identity import file_signature
from . import native_tensor_context as context
from .native_tensor_io import (
    KEY,
    EXPORT_KEY,
    MAX_BYTES,
    MAX_TOTAL_BYTES,
    MAX_HEADER,
    TensorInput,
    TensorPlan,
    TensorExport,
    _raw,
    _directory,
    _input_file,
    _scoped,
    _check_job,
    assert_composable,
)

OPERATION_KEY = "_native_tensor_operation"
OPERATION = "latent-composition"


def capture_composition_inputs(primary_raw, secondary_raw, manager, *, data=None):
    """Capture exactly two bytes-only saved contexts without model inventory/loading."""
    result = dict(data or {})
    _check_job(result)
    raws = (_raw(primary_raw), _raw(secondary_raw))
    if sum(map(len, raws)) > MAX_TOTAL_BYTES:
        raise ValueError("The two latent inputs together exceed 64 MiB.")
    directory = _directory(manager)
    inputs = []
    for role, raw in zip(("latent", "composition"), raws):
        values, _ = context.raw_header(raw)
        try:
            source = json.loads(values["__metadata__"][context.MARKER])
            family = source["family"]
        except (KeyError, TypeError, ValueError, RecursionError) as error:
            raise ValueError(
                "Standalone composition requires locally receipted saved workflow latents."
            ) from error
        values, saved = context.header(raw, "latent", family)
        context.verify_receipt(raw, saved, directory)
        path = directory / (role + ".latent")
        path.write_bytes(raw)
        row = values["latent_tensor"]
        inputs.append(
            TensorInput(
                role,
                path,
                tuple(file_signature(path)),
                hashlib.sha256(raw).hexdigest(),
                row["dtype"],
                tuple(row["shape"]),
                tuple(sorted(saved.items())),
            )
        )
    assert_composable(*inputs)
    saved = dict(inputs[0].context)
    result[KEY] = TensorPlan(
        saved["family"], directory, tuple(inputs), True, False, context.canonical(saved["capture"])
    )
    result[EXPORT_KEY] = []
    result[OPERATION_KEY] = OPERATION
    result["native_latent_composition_source"] = dict(
        family=saved["family"],
        primary_sha256=inputs[0].sha256,
        secondary_sha256=inputs[1].sha256,
        original_dtype=inputs[0].dtype,
        working_dtype="float32",
        shape=list(inputs[0].shape),
    )
    return result


def _owned(data, item):
    plan = data.get(KEY)
    if (
        data.get(OPERATION_KEY) != OPERATION
        or not isinstance(plan, TensorPlan)
        or len(plan.inputs) != 2
        or item not in plan.inputs
        or item.path.parent.resolve() != plan.directory.resolve()
    ):
        raise ValueError("Capture the two owned latent inputs for this composition request.")
    assert_composable(*plan.inputs)
    return plan


def load_composition_latent(data, item):
    """Use native LoadLatent's explicit F32 working samples; load no companions."""
    import safetensors.torch
    from nodes import LoadLatent

    plan = _owned(data, item)
    with _input_file(item, data) as path:
        raw = path.read_bytes()
        _, saved = context.header(raw, "latent", plan.family)
        context.verify_receipt(raw, saved, plan.directory)
        paths = types.SimpleNamespace(get_annotated_filepath=lambda *_: str(path))
        latent = _scoped(LoadLatent.load, LoadLatent(), paths, path.name)[0]
        extra = safetensors.torch.load_file(str(path), device="cpu")
        if "rf_noise_mask" in extra:
            latent["noise_mask"] = extra["rf_noise_mask"]
        context.finite(latent)
        return latent


def save_composed_latent(data, primary, latent, *, composition):
    """Save one real native operation result with inherited verified provenance."""
    import safetensors.torch
    import torch
    from nodes import SaveLatent

    plan = _owned(data, primary)
    if primary is not plan.inputs[0]:
        raise ValueError("The composed destination must retain the captured primary context.")
    if not isinstance(composition, dict) or composition.get("native_node") not in (
        "LatentComposite",
        "LatentCompositeMasked",
    ):
        raise ValueError("Record the actual bounded native composition operation.")
    try:
        encoded = context.canonical(composition).encode()
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Composition metadata must contain bounded JSON primitives.") from error
    if len(encoded) > 8192:
        raise ValueError("Composition operation metadata exceeds 8 KiB.")
    sample = latent.get("samples") if isinstance(latent, dict) else None
    extras = (
        {"rf_noise_mask": latent["noise_mask"]}
        if isinstance(latent, dict) and "noise_mask" in latent
        else {}
    )
    tensors = [sample, *extras.values()]
    if (
        not isinstance(sample, torch.Tensor)
        or tuple(sample.shape) != primary.shape
        or sample.dtype != torch.float32
        or sample.device.type != "cpu"
        or any(
            not isinstance(t, torch.Tensor)
            or t.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or t.device.type != "cpu"
            for t in tensors
        )
        or sum(t.numel() * t.element_size() for t in tensors) > MAX_BYTES - MAX_HEADER
    ):
        raise ValueError(
            "Composition must preserve the exact native rank-four shape and explicit CPU float32 working dtype before saving."
        )
    context.finite(tensors)
    with _input_file(primary, data) as original:
        raw = original.read_bytes()
        _, saved = context.header(raw, "latent", plan.family)
        context.verify_receipt(raw, saved, plan.directory)
    saved = deepcopy(saved)
    saved["layout"]["composition"] = dict(
        composition,
        primary_sha256=primary.sha256,
        secondary_sha256=plan.inputs[1].sha256,
        original_dtype=primary.dtype,
        working_dtype="F32",
    )
    _check_job(data)
    prefix = "latent-composite-" + secrets.token_hex(12)
    paths = types.SimpleNamespace(
        get_save_image_path=lambda *_: (str(plan.directory), prefix, 0, "", prefix)
    )
    saver = SaveLatent.__new__(SaveLatent)
    saver.output_dir = str(plan.directory)
    _scoped(SaveLatent.save, saver, paths, latent, filename_prefix=prefix)
    path = plan.directory / (prefix + "_00000_.latent")
    parts = [("", path.read_bytes())]
    if extras:
        parts.append(
            (
                "",
                safetensors.torch.save(
                    {key: value.detach().contiguous() for key, value in extras.items()}
                ),
            )
        )
    raw = context.pack(parts, saved)
    context.header(raw, "latent", plan.family)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    digest = hashlib.sha256(raw).hexdigest()
    receipts = context.receipt_root(plan.directory)
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = receipts / (digest + ".json")
    temporary = receipt.with_suffix("." + secrets.token_hex(8) + ".tmp")
    temporary.write_text(context.canonical(saved), encoding="utf-8")
    temporary.replace(receipt)
    _check_job(data)
    result = TensorExport(
        str(path), "latent-composite", tuple(file_signature(path)), digest, plan.family
    )
    data[EXPORT_KEY].append(result)
    return result
