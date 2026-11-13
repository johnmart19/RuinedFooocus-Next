"""Use the pinned native paired-alias loader with a frozen local file namespace."""

from types import FunctionType, SimpleNamespace

from ruinedfooocus.features.models.approximate_codecs import (
    ApproximateCodec,
    expected_shapes,
    read_scope,
)


def _cancelled():
    from concurrent.futures import CancelledError
    from comfy.model_management import throw_exception_if_processing_interrupted
    from ruinedfooocus.app.resource_lifecycle import current_job

    throw_exception_if_processing_interrupted()
    job = current_job()
    if job is not None and job.cancelled.is_set():
        raise CancelledError("Approximate codec loading cancelled.")


class _Cancellation:
    def is_set(self):
        _cancelled()
        return False


def load_paired_vae(pair, device=None):
    if not isinstance(pair, ApproximateCodec):
        raise ValueError("Capture a full approximate codec pair before native loading.")
    import comfy.sd
    import comfy.utils
    from nodes import VAELoader

    files = {item.source.name: item for item in pair.files}

    def read(filename):
        _cancelled()
        item = next((item for item in pair.files if str(item.path) == filename), None)
        if item is None:
            raise ValueError("The native approximate loader requested an uncaptured file.")
        item.verify()
        state = comfy.utils.load_torch_file(filename, safe_load=True)
        expected = expected_shapes(pair.alias, item.role)
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("The approximate codec needs complete native encoder/decoder weights.")
        for name, value in state.items():
            if tuple(getattr(value, "shape", ())) != expected[name] or str(
                getattr(value, "dtype", "")
            ) not in ("torch.float32", "torch.float16", "torch.bfloat16"):
                raise ValueError("Approximate codec weights disagree with the native alias layout.")
        item.verify()
        return state

    def full_path(category, name):
        if category != "vae_approx" or name not in files:
            raise ValueError("The native approximate loader requested an uncaptured source.")
        return str(files[name].path)

    folders = SimpleNamespace(
        get_filename_list=lambda category: sorted(files) if category == "vae_approx" else [],
        get_full_path_or_raise=full_path,
    )
    # Reuse native prefixing and all six normalization contracts. The function's
    # private namespace prevents concurrent jobs from changing folder_paths.
    native = VAELoader.load_taesd
    assemble = FunctionType(
        native.__code__,
        dict(
            native.__globals__,
            folder_paths=folders,
            comfy=SimpleNamespace(
                utils=SimpleNamespace(load_torch_file=read),
                latent_formats=comfy.latent_formats,
            ),
        ),
        native.__name__,
        native.__defaults__,
        native.__closure__,
    )
    with read_scope(pair, cancelled=_Cancellation()):
        _cancelled()
        state = assemble(pair.alias)
        metadata = {"tae_latent_channels": 128} if pair.alias == "taef2" else None
        vae = comfy.sd.VAE(sd=state, metadata=metadata, device=device)
        vae.throw_exception_if_invalid()
        if pair.alias == "taef2":
            # This pinned VAE branch leaves the default8 despite native Flux2's
            # 2x2 latent packing. Keep its crop/tiling geometry with that codec.
            vae.downscale_ratio = vae.upscale_ratio = 16
        pair.verify()
        _cancelled()
    vae.patcher.cached_patcher_init = (load_pair_patcher, (pair, device))
    return vae


def load_pair_patcher(pair, device=None, disable_dynamic=False):
    """Native VAE reload factories accept disable_dynamic; VAE owns its patcher."""
    return load_paired_vae(pair, device=device).patcher
