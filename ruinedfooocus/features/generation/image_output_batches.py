"""Native output groups are distinct from repeated generation requests."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from uuid import uuid4

PLAN_KEY = "_native_image_output_plan"


@dataclass(frozen=True)
class OutputPlan:
    kind: str
    min_per_generation: int
    max_per_generation: int
    _request_token: str = field(default_factory=lambda: uuid4().hex, repr=False)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not 1 <= len(self.kind) <= 64:
            raise ValueError("A native output group needs a bounded operation name.")
        if (
            type(self.min_per_generation) is not int
            or type(self.max_per_generation) is not int
            or not 0 <= self.min_per_generation <= self.max_per_generation <= 64
            or self.max_per_generation == 0
        ):
            raise ValueError("Native output groups require a limit from 1 through 64.")

    def validate(self, count, generations=1):
        if type(generations) is not int or not 1 <= generations <= 100:
            raise ValueError("Native output groups require a bounded generation count.")
        if (
            not self.min_per_generation * generations
            <= count
            <= self.max_per_generation * generations
        ):
            raise ValueError("The native image output count differs from the captured request.")


def with_output_plan(data, plan):
    if not isinstance(plan, OutputPlan):
        raise ValueError("Capture a typed native image output plan.")
    return dict(data, **{PLAN_KEY: replace(plan, _request_token=uuid4().hex)})


@dataclass(frozen=True)
class ImageOutputBatch(Sequence):
    outputs: tuple
    plan: OutputPlan
    metadata: tuple = ()

    def __post_init__(self):
        if not isinstance(self.plan, OutputPlan):
            raise ValueError("Capture the native output plan before processing images.")
        object.__setattr__(self, "outputs", tuple(self.outputs))
        self.plan.validate(len(self.outputs))
        if self.metadata and len(self.metadata) != len(self.outputs):
            raise ValueError("Native output metadata must identify every output image.")
        object.__setattr__(
            self, "metadata", tuple(MappingProxyType(deepcopy(dict(row))) for row in self.metadata)
        )

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.outputs[index]


class ImageBatchArtifacts(list):
    """A request-local list with proof that every native batch finished successfully."""

    def __init__(self, outputs, plan, completed_generations, *, tensor_files=()):
        super().__init__(outputs)
        if not isinstance(plan, OutputPlan):
            raise ValueError("A native artifact group requires its captured output plan.")
        plan.validate(len(self), completed_generations)
        self.plan = plan
        self.completed_generations = completed_generations
        self.tensor_files = tuple(tensor_files)


def validate_artifacts(data, result):
    """Allow a completed empty detection, while refusing cancellation/partial failure."""
    from ruinedfooocus.features.image_processing import latent_composition

    if latent_composition.handles((data.get("controlnet") or {}).get("type")):
        return latent_composition.validate_result(data, result)
    plan = data.get(PLAN_KEY)
    count = data.get("image_number", 1)
    if isinstance(plan, OutputPlan):
        if (
            not isinstance(result, ImageBatchArtifacts)
            or result.plan != plan
            or result.completed_generations != count
        ):
            raise RuntimeError(
                "Native image processing did not complete every requested output group."
            )
        plan.validate(len(result), count)
    elif not result or len(result) != count:
        raise RuntimeError("Image generation failed or was cancelled; see the application log.")
    return result
