"""Session input contracts for the native rembg preprocessors we have qualified."""

from importlib.metadata import PackageNotFoundError, version
import inspect
import os
import hashlib
from pathlib import Path


def _strict_factory(factory, qualified, *, injected=False):
    """Versioned construction adapter; all native model methods stay inherited.

    rembg 2.0.87 BaseSession initializes only model_name and inner_session,
    and does not forward enable_fallback. Do that small construction step in
    a subclass rather than mutating rembg/ORT globals. New versions require
    an audit before the GPU path can use this adapter.
    """
    namespace = getattr(factory, "__globals__", {})
    classes = namespace.get("sessions_class")
    if not classes:
        if injected:
            return factory

        def unsupported(*args, **kwargs):
            if kwargs.get("providers") == ["CPUExecutionProvider"]:
                return factory(*args, **kwargs)
            raise RuntimeError("The native rembg factory has no audited strict-construction API.")

        return unsupported

    def adapted(*args, **kwargs):
        import onnxruntime as ort
        from rembg.sessions.base import BaseSession

        bound = inspect.signature(factory).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        name = values["model_name"]
        native = next((cls for cls in classes if cls.name() == name), None)
        if native is None:
            raise ValueError(f"No session class found for model '{name}'")
        if not qualified or native.__init__ is not BaseSession.__init__:
            if kwargs.get("providers") == ["CPUExecutionProvider"]:
                session = factory(*args, **kwargs)
                session.inner_session.disable_fallback()
                return session
            raise RuntimeError("The native rembg constructor needs a strict-fallback API audit.")
        options = values["sess_opts"] or ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            threads = int(os.environ["OMP_NUM_THREADS"])
            if options.inter_op_num_threads == 0:
                options.inter_op_num_threads = threads
            if options.intra_op_num_threads == 0:
                options.intra_op_num_threads = threads
        model_args = values["args"]
        model_kwargs = dict(values["kwargs"])
        providers = model_kwargs.pop("providers", ["CPUExecutionProvider"])

        class StrictSession(native):
            def __init__(self):
                self.model_name = name
                self._rf_model_path = str(
                    self.__class__.download_models(*model_args, **model_kwargs)
                )
                self.inner_session = ort.InferenceSession(
                    self._rf_model_path,
                    sess_options=options,
                    providers=providers,
                    enable_fallback=False,
                )
                self.inner_session.disable_fallback()
                self._warn_on_provider_fallback(providers)

        return StrictSession()

    return adapted


def create_mask_session(*args, _factory=None, **kwargs):
    import rembg

    try:
        qualified = version("rembg") == "2.0.87"
    except PackageNotFoundError:
        qualified = False
    factory = _factory if _factory is not None else rembg.new_session
    parameter = inspect.signature(factory).parameters.get("model_name")
    default = parameter.default if parameter is not None else None
    model_name = kwargs.get("model_name", args[0] if args else default)
    if qualified and model_name == "bria-rmbg":
        import onnxruntime as ort

        options = kwargs.get("sess_opts")
        if options is None:
            options = ort.SessionOptions()
        # BriaRmBgSession.predict in rembg 2.0.87 always normalizes to 1024².
        # Bind its exported symbolic dimensions, not the uploaded image size.
        # Unknown/new preprocessors retain their native dynamic-shape behavior.
        options.add_free_dimension_override_by_name("height", 1024)
        options.add_free_dimension_override_by_name("width", 1024)
        kwargs["sess_opts"] = options
    return _strict_factory(factory, qualified, injected=_factory is not None)(*args, **kwargs)


def model_identity(session):
    path = vars(session).get("_rf_model_path")
    if not path:
        return {"model": session.model_name, "sha256": None}
    with Path(path).open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "model": session.model_name,
        "sha256": checksum,
        "preprocessing": "native-rembg-" + version("rembg"),
    }


def create_audited_mask_session(expected_provider, *, profile_root=None, **kwargs):
    import onnxruntime as ort
    from ruinedfooocus.backends.onnx.execution import profile_options, graph_assignment

    options = kwargs.get("sess_opts")
    if options is None:
        options = ort.SessionOptions()
    directory = profile_options(options, parent=profile_root)
    try:
        session = create_mask_session(
            sess_opts=options, **{k: v for k, v in kwargs.items() if k != "sess_opts"}
        )
        session.inner_session.disable_fallback()
        if expected_provider not in session.inner_session.get_providers():
            raise RuntimeError(
                f"{expected_provider} did not initialize; refusing silent CPU inference."
            )
        session._rf_graph_assignment = graph_assignment(session.inner_session, expected_provider)
        session._rf_profile = directory
        session._rf_assignment = None
        return session
    except BaseException:
        directory.cleanup()
        raise
