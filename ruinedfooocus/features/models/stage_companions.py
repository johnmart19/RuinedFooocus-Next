"""Native stage roles that require another stage's conditioning."""

from collections.abc import Mapping

from ruinedfooocus.features.models.utility_models import passive_native_class

CASCADE_STAGE_B = "Stable_Cascade_B"


def is_companion_annotation(annotation):
    """Expected registered native role; installed identity still needs its header."""
    return (
        isinstance(annotation, Mapping)
        and (
            annotation.get("native_class")
            or annotation.get("native_family")
            or annotation.get("architecture")
        )
        == CASCADE_STAGE_B
    )


def passive_companion_file(path):
    return passive_native_class(path, (CASCADE_STAGE_B,))


def is_stage_companion(metadata):
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("rf_model_role") != "image_stage_companion"
    ):
        return False
    proof = metadata.get("_rf_intrinsic_companion")
    if (
        not isinstance(proof, Mapping)
        or proof.get("native_class") != CASCADE_STAGE_B
        or proof.get("conditioning") != "stable_cascade_prior"
        or proof.get("primary_native_class") != "Stable_Cascade_C"
    ):
        return False
    signature = proof.get("signature")
    return (
        isinstance(signature, (list, tuple))
        and len(signature) == 4
        and isinstance(signature[0], str)
        and all(type(value) is int for value in signature[1:])
    )
