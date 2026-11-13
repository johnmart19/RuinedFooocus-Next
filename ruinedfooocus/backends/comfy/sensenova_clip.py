"""Construct the native tokenizer-only SenseNova CLIP for UNet-only routes."""

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


def load_sensenova_clip(
    model_config,
    checkpoint_signature,
    embedding_directory=None,
    model_options=None,
    *,
    disable_dynamic=False,
):
    """Build the native sentinel without reading diffusion or encoder weights.

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

    sentinel = model_config.process_clip_state_dict({})
    options = dict(model_options or {})
    clip = comfy.sd.CLIP(
        model_config.clip_target(),
        embedding_directory=embedding_directory,
        tokenizer_data=sentinel,
        state_dict=sentinel,
        parameters=comfy.utils.calculate_parameters(sentinel),
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
    """Reload the weight-free native tokenizer only for the frozen checkpoint."""
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
