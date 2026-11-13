"""Load a frozen named shard bundle using native per-file storage and model APIs."""

from ruinedfooocus.features.models.sensenova_bundle import SenseNovaBundle, read_scope


def _check_cancelled():
    from ruinedfooocus.backends.comfy.sensenova_clip import _check_cancelled

    _check_cancelled()


class _Cancellation:
    def is_set(self):
        _check_cancelled()
        return False


def load_bundle_model(bundle, model_options=None, *, disable_dynamic=False):
    """Keep native per-shard tensor storage; never create a merged weight file.

    The unified dictionary contains the native readers' tensor references. Native
    mmap/aimdo settings and storage annotations remain owned by ComfyUI.
    """
    if not isinstance(bundle, SenseNovaBundle):
        raise ValueError("Capture the named SenseNova shard bundle before loading.")
    import comfy.sd
    import comfy.utils

    options = dict(model_options or {})
    expected = {row.name: row for row in bundle.tensors}
    state = {}
    with read_scope(bundle, cancelled=_Cancellation()):
        for shard in bundle.shards:
            _check_cancelled()
            loaded, metadata = comfy.utils.load_torch_file(str(shard.path), return_metadata=True)
            keys = {name for name, row in expected.items() if row.shard == shard.path.name}
            if set(loaded) != keys or dict(metadata or {}) != dict(bundle.metadata):
                raise ValueError("Native SenseNova shard keys or metadata changed after capture.")
            for name, value in loaded.items():
                if name in state or tuple(value.shape) != expected[name].shape:
                    raise ValueError(
                        "Native SenseNova tensor layout disagrees with the captured index."
                    )
                dtype = str(value.dtype).removeprefix("torch.")
                if (
                    dtype
                    != {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[
                        expected[name].dtype
                    ]
                ):
                    raise ValueError("Native SenseNova tensor dtype changed after capture.")
            state.update(loaded)
            del loaded
            bundle.verify(_check_cancelled)
        _check_cancelled()
        model = comfy.sd.load_diffusion_model_state_dict(
            state,
            model_options=options,
            metadata=dict(bundle.metadata),
            disable_dynamic=disable_dynamic,
        )
        if model is None or type(model.model.model_config).__name__ != "SenseNovaU15":
            raise ValueError("The native loader did not classify the bundle as SenseNovaU15.")
        _check_cancelled()
        bundle.verify()
    model.cached_patcher_init = (load_bundle_model, (bundle, options))
    return model
