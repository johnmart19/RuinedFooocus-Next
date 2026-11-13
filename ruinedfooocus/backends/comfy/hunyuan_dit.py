"""Native HyDiT split encoders with the MT5 tokenizer omitted by pinned load_clip."""

from contextlib import ExitStack
from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature

MAX_TOKENIZER_BYTES = 16 * 1024 * 1024


def _check_signatures(paths, signatures):
    if any(tuple(file_signature(path)) != signature for path, signature in zip(paths, signatures)):
        raise ValueError(
            "HunyuanDiT text encoders changed after capture. Submit the request again."
        )


def _check_cancelled():
    from comfy.model_management import throw_exception_if_processing_interrupted
    from ruinedfooocus.app.resource_lifecycle import current_job

    throw_exception_if_processing_interrupted()
    job = current_job()
    if job is not None and job.cancelled.is_set():
        from concurrent.futures import CancelledError

        raise CancelledError("HunyuanDiT text encoder loading cancelled.")


class _Cancellation:
    def is_set(self):
        _check_cancelled()
        return False


def _tokenizer_data(states):
    import torch

    bert = [
        state for state in states if "bert.encoder.layer.0.attention.self.query.weight" in state
    ]
    mt5 = [
        state for state in states if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in state
    ]
    if len(bert) != 1 or len(mt5) != 1 or bert[0] is mt5[0]:
        raise ValueError("HunyuanDiT needs one Chinese RoBERTa and one MT5-XL encoder.")
    bert_weight = bert[0]["bert.encoder.layer.0.attention.self.query.weight"]
    mt5_weight = mt5[0]["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
    if (
        not isinstance(bert_weight, torch.Tensor)
        or not isinstance(mt5_weight, torch.Tensor)
        or tuple(bert_weight.shape) != (1024, 1024)
        or tuple(mt5_weight.shape) != (5120, 2048)
    ):
        raise ValueError(
            "The selected HunyuanDiT text encoder dimensions do not match native HyDiT."
        )
    tokenizer = mt5[0].get("mt5xl.spiece_model", mt5[0].get("spiece_model"))
    if (
        not isinstance(tokenizer, torch.Tensor)
        or tokenizer.dtype != torch.uint8
        or tokenizer.ndim != 1
        or tokenizer.device.type != "cpu"
        or not 0 < tokenizer.numel() <= MAX_TOKENIZER_BYTES
    ):
        raise ValueError(
            "HunyuanDiT MT5-XL needs its embedded byte tokenizer (spiece_model); "
            "a generic T5 encoder cannot replace it."
        )
    return {"mt5xl.spiece_model": tokenizer}


def load_hunyuan_dit_clip(
    ckpt_paths,
    clip_target,
    embedding_directory=None,
    model_options=None,
    *,
    signatures=None,
    disable_dynamic=False,
):
    """Read each native state once; caller holds prepared-asset read leases.

    The reload factory owns fresh leases on these same frozen file identities.
    Do not nest another read lease here: a waiting writer could deadlock the
    caller's already-held read leases.
    """
    if not isinstance(ckpt_paths, (tuple, list)) or len(ckpt_paths) != 2:
        raise ValueError("HunyuanDiT requires exactly two local text encoder files.")
    paths = tuple(str(Path(path).resolve()) for path in ckpt_paths)
    if len(set(paths)) != 2:
        raise ValueError("HunyuanDiT requires distinct BERT and MT5-XL encoder files.")
    frozen = (
        tuple(tuple(item) for item in signatures)
        if signatures is not None
        else tuple(tuple(file_signature(path)) for path in paths)
    )
    if len(frozen) != 2 or any(len(item) != 4 for item in frozen):
        raise ValueError("HunyuanDiT encoder identities were not captured correctly.")
    _check_cancelled()
    _check_signatures(paths, frozen)
    import comfy.sd
    import comfy.utils

    options = dict(model_options or {})
    states = []
    for path in paths:
        _check_cancelled()
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        if not isinstance(state, dict):
            raise ValueError(
                "HunyuanDiT text encoders must contain native tensor state dictionaries."
            )
        if options.get("custom_operations") is None:
            state, metadata = comfy.utils.convert_old_quants(
                state, model_prefix="", metadata=metadata
            )
        states.append(state)
    tokenizer_data = _tokenizer_data(states)
    _check_cancelled()
    _check_signatures(paths, frozen)
    clip = comfy.sd.CLIP(
        clip_target,
        embedding_directory=embedding_directory,
        tokenizer_data=tokenizer_data,
        parameters=sum(comfy.utils.calculate_parameters(state) for state in states),
        state_dict=states,
        model_options=options,
        disable_dynamic=disable_dynamic,
    )
    _check_cancelled()
    _check_signatures(paths, frozen)
    clip.patcher.cached_patcher_init = (
        reload_hunyuan_dit_clip_patcher,
        (paths, frozen, clip_target, embedding_directory, options),
    )
    return clip


def reload_hunyuan_dit_clip_patcher(
    paths, signatures, clip_target, embedding_directory, model_options, *, disable_dynamic=False
):
    """Reload exactly the captured native files, refusing mutation or retargeting."""
    from ruinedfooocus.core.preparation import preparation_service

    with ExitStack() as stack:
        for path in paths:
            stack.enter_context(preparation_service.artifact_lease(path, cancelled=_Cancellation()))
        return load_hunyuan_dit_clip(
            paths,
            clip_target,
            embedding_directory,
            model_options,
            signatures=signatures,
            disable_dynamic=disable_dynamic,
        ).patcher
