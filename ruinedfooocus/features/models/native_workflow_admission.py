"""Keep a newly detected native family distinct from a connected app workflow."""

from functools import lru_cache
from pathlib import Path

from ruinedfooocus.features.models.utility_models import (
    _native_predicates,
    _native_source_signatures,
    passive_native_class,
)

PLANNED_IMAGE_WORKFLOWS = {
    "MiniMaxH3Image": (
        "Capture the MiniMax H3 Image workflow with its PDD sampling, image conditioning "
        "and matching image codec before generation."
    )
}


@lru_cache(maxsize=4)
def _has_pending_class(sources):
    try:
        _native_predicates(sources, frozenset(PLANNED_IMAGE_WORKFLOWS))
        return True
    except (OSError, SyntaxError, ValueError, TypeError, KeyError, StopIteration):
        return False


def planned_image_family(path):
    try:
        if path is None or not _has_pending_class(_native_source_signatures()):
            return None
        from ruinedfooocus.features.models.model_architecture import native_detector_ready

        family = passive_native_class(path, PLANNED_IMAGE_WORKFLOWS)
        if family is None and Path(path).suffix.lower() == ".gguf" and native_detector_ready():
            from ruinedfooocus.features.models.model_architecture import inspect_model_file

            family = inspect_model_file(path).get("native_class")
        return family if family in PLANNED_IMAGE_WORKFLOWS else None
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def require_available_image_workflow(path=None, *, family=None, prepared_data=None, loading=False):
    actual = planned_image_family(path)
    if actual is not None or (isinstance(family, str) and family in PLANNED_IMAGE_WORKFLOWS):
        from ruinedfooocus.features.generation.minimax_image import require_plan, validate_prepared

        if path is None:
            raise ValueError(PLANNED_IMAGE_WORKFLOWS[actual or family])
        require_plan(prepared_data, path)
        if loading:
            validate_prepared(prepared_data)
