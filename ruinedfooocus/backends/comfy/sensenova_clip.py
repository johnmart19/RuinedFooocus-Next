"""Construct the native SenseNova text encoder from its frozen checkpoint source."""

from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature


def _check_identity(signature):
    from ruinedfooocus.features.models.sensenova_bundle import SenseNovaBundle

    if isinstance(signature, SenseNovaBundle):
        signature.verify(_check_cancelled)
        return
    if (
        not isinstance(signature, (tuple, list))
        or len(signature) != 4
        or not isinstance(signature[0], str)
    ):
        raise ValueError("SenseNova checkpoint identity must be captured before loading.")
    if tuple(file_signature(signature[0])) != tuple(signature):
        raise ValueError("SenseNova checkpoint changed after capture. Submit the request again.")


def _check_cancelled():
    from comfy.model_management import throw_exception_if_processing_interrupted
    from ruinedfooocus.app.resource_lifecycle import current_job

    throw_exception_if_processing_interrupted()
    job = current_job()
    if job is not None and job.cancelled.is_set():
        from concurrent.futures import CancelledError

        raise CancelledError("SenseNova tokenizer loading cancelled.")


def _checkpoint_state(checkpoint_signature):
    """Native tensor references of the frozen source in the native checkpoint layout."""
    from ruinedfooocus.features.models.sensenova_bundle import SenseNovaBundle

    if isinstance(checkpoint_signature, SenseNovaBundle):
        from ruinedfooocus.backends.comfy.sensenova_bundle import read_bundle_state

        return read_bundle_state(checkpoint_signature)
    import comfy.utils

    return comfy.utils.load_torch_file(checkpoint_signature[0])


def load_sensenova_clip(
    model_config,
    checkpoint_signature,
    embedding_directory=None,
    model_options=None,
    *,
    disable_dynamic=False,
):
    """Build the native text encoder from the checkpoint's own language weights.

    ComfyUI v0.46 moved the SenseNova language and vision-embedding weights out of
    the diffusion model into `text_encoders.`; there is no tokenizer-only sentinel.
    The caller already owns the checkpoint's read lease and loaded native config.
    Only the frozen source signature and small native config enter the reload
    factory; it owns a fresh checkpoint lease if a delegate is requested later.
    """
    _check_cancelled()
    _check_identity(checkpoint_signature)
    if type(model_config).__name__ != "SenseNovaU15":
        raise ValueError("The tokenizer-only CLIP contract requires native SenseNovaU15.")
    import comfy.sd
    import comfy.utils

    state = _checkpoint_state(checkpoint_signature)
    _check_cancelled()
    target = model_config.clip_target(state)
    if target is None:
        raise ValueError(
            "SenseNova U1.5 in ComfyUI v0.46+ requires text encoder weights in the checkpoint; "
            "the tokenizer-only sentinel was removed upstream."
        )
    # Same order as the native checkpoint loader: detect first, then take the encoder keys.
    encoder_state = model_config.process_clip_state_dict(state)
    del state
    options = dict(model_options or {})
    clip = comfy.sd.CLIP(
        target,
        embedding_directory=embedding_directory,
        tokenizer_data=encoder_state,
        state_dict=encoder_state,
        parameters=comfy.utils.calculate_parameters(encoder_state),
        model_options=options,
        disable_dynamic=disable_dynamic,
    )
    _check_cancelled()
    _check_identity(checkpoint_signature)
    clip.patcher.cached_patcher_init = (
        reload_sensenova_clip_patcher,
        (
            model_config,
            tuple(checkpoint_signature)
            if isinstance(checkpoint_signature, (list, tuple))
            else checkpoint_signature,
            embedding_directory,
            options,
        ),
    )
    return clip


class _Cancellation:
    def is_set(self):
        _check_cancelled()
        return False


def reload_sensenova_clip_patcher(
    model_config, checkpoint_signature, embedding_directory, model_options, *, disable_dynamic=False
):
    """Reload the native text encoder only from the frozen checkpoint."""
    from ruinedfooocus.core.preparation import preparation_service
    from ruinedfooocus.features.models.sensenova_bundle import SenseNovaBundle, read_scope

    _check_identity(checkpoint_signature)
    scope = (
        read_scope(checkpoint_signature, cancelled=_Cancellation())
        if isinstance(checkpoint_signature, SenseNovaBundle)
        else preparation_service.artifact_lease(
            Path(checkpoint_signature[0]), cancelled=_Cancellation()
        )
    )
    with scope:
        return load_sensenova_clip(
            model_config,
            checkpoint_signature,
            embedding_directory,
            model_options,
            disable_dynamic=disable_dynamic,
        ).patcher
