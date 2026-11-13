"""Source-bound consumers for owned or explicitly declared geometry guides.

Planning may verify a selected file once through the existing SHA cache. A frozen
request rechecks its signature and native header contract, never hashes weights
or prepares a model on the worker path.
"""

from dataclasses import dataclass
import json
import math

from ruinedfooocus.core.cache_contract import configured_cache_root
from ruinedfooocus.features.image_editing.control_candidates import (
    compatibility,
    model_family,
    selected_control,
)
from ruinedfooocus.features.image_editing.native_control_recipes import (
    control_contract,
    recipes_for_family,
    union_types,
)
from ruinedfooocus.features.models.model_identity import (
    cached_file_sha256,
    cached_safetensors_header,
    file_signature,
)

# Semantics belong to these exact publisher payloads, never a filename or an
# SDXL-compatible header alone. Other controls remain usable in the regular UI.
DEPTH_SHA256 = "37e391757e6c0042faa3795174acbe11a31992252058cfb8bbaccf11073567b4"
UNION_SHA256 = "a9e13fd61f3193887791c8a0dd07a07202174dc47d5ddaea94ea1344f07c7467"
SOURCES = {
    DEPTH_SHA256: (395729267, "Control-LoRA Depth rank128", "ordinary", 0),
    UNION_SHA256: (2512030408, "Xinsir SDXL Union 1.0 base6", "union", 6),
}
_SPEC_FIELDS = (
    "kind",
    "convention",
    "projection",
    "producer",
    "source_id",
    "guide_id",
    "width",
    "height",
)
_DEPTH_CONVENTIONS = frozenset(
    ("white-near", "white-near; invalid-black", "white-near; uncovered-native-far-fill")
)


def _spec_values(spec):
    values = tuple(getattr(spec, name, None) for name in _SPEC_FIELDS)
    kind, convention, projection, producer, source_id, guide_id, width, height = values
    if (
        producer not in ("moge-depth", "lotus-depth", "explicit-upload")
        or not source_id
        or not guide_id
    ):
        raise ValueError("Choose an owned MoGe or Lotus guide with its original source.")
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("The guide must retain its exact positive source dimensions.")
    if projection != "perspective":
        raise ValueError(
            "This SDXL guide route requires a perspective map; panorama conditioning is unqualified."
        )
    if producer == "explicit-upload":
        if not all(
            isinstance(identity, str)
            and len(identity) == 64
            and all(char in "0123456789abcdef" for char in identity)
            for identity in (source_id, guide_id)
        ):
            raise ValueError(
                "Capture uploaded source and guide pixels with their exact content hashes."
            )
        if (kind == "depth" and convention == "white-near") or (
            kind == "normal"
            and convention in ("native-opengl", "native-directx", "normalbae-scannet-camera")
        ):
            return values
        raise ValueError(
            "Declare the uploaded map's supported depth or camera-normal convention explicitly."
        )
    if kind == "depth" and convention in _DEPTH_CONVENTIONS:
        return values
    if kind == "normal":
        if producer != "moge-depth" or convention not in ("native-opengl", "native-directx"):
            raise ValueError("Choose an owned MoGe normal with its exact native camera convention.")
        return values
    raise ValueError("The selected result has no supported white-near depth convention.")


def _resolved(manager, family, controls):
    if model_family(family) != "SDXL":
        raise ValueError("This verified geometry-guide route requires an SDXL checkpoint.")
    name = controls.get("model")
    signature = controls.get("model_signature")
    if signature is None:
        raise ValueError(
            "Select an explicit installed ControlNet and refresh its file identity first."
        )
    path, signature, info = selected_control(manager, name, family, signature)
    if compatibility(info, family) != "compatible":
        raise ValueError("The selected control has no verified native SDXL header contract.")
    return path, tuple(signature)


def _normalized(controls, digest, path, signature, kind):
    source = SOURCES.get(digest)
    if source is None or signature[1] != source[0]:
        raise ValueError("Select the verified SDXL rank128 depth or base6 Union control model.")
    _, _, recipe, modes = source
    if kind == "normal" and not modes:
        raise ValueError(
            "Normal guides require the verified SDXL base6 Union model, not the depth-only payload."
        )
    contract = control_contract(cached_safetensors_header(path))
    if contract["union_modes"] != modes or recipe not in recipes_for_family("SDXL"):
        raise ValueError("The selected guide control no longer has its reviewed native mode table.")
    union_type = kind if recipe == "union" else "auto"
    if modes and union_types().index(union_type) - 1 >= modes:
        raise ValueError("The guide type is absent from the selected native Union mode table.")
    strength, start, stop = (
        controls.get(key, default)
        for key, default in (("strength", 1.0), ("start", 0.0), ("stop", 1.0))
    )
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (strength, start, stop)
        )
        or not 0 < strength <= 2
        or not 0 <= start < stop <= 1
    ):
        raise ValueError("Guide control requires 0 < strength ≤ 2 and 0 ≤ start < stop ≤ 1.")
    # This is a completed map, not an image awaiting another depth estimator.
    result = dict(
        controls,
        type="control" if modes else "depth",
        model_signature=list(signature),
        strength=strength,
        start=start,
        stop=stop,
        depth_preprocess="none",
        recipe=recipe,
        union_type=union_type,
    )
    for key, expected in (
        ("type", result["type"]),
        ("depth_preprocess", "none"),
        ("recipe", recipe),
        ("union_type", union_type),
    ):
        if key in controls and controls[key] != expected:
            raise ValueError(
                "The selected guide control type, recipe or preprocessing changed; adopt the guide again."
            )
    return result, recipe, union_type


@dataclass(frozen=True)
class GuideControlReceipt:
    family: str
    model_signature: tuple
    artifact_sha256: str
    source_label: str
    spec_values: tuple
    controls_json: str
    recipe: str
    union_type: str

    @property
    def controls(self):
        return json.loads(self.controls_json)

    @property
    def invert_channels(self):
        # Source-qualified inference: pinned MoGe's outward camera normals,
        # original BAE's FrameNet labels, and DSINE's direct signed evaluation
        # against those same labels. This is not an Xinsir axis declaration.
        if self.spec_values[0] == "normal":
            return {
                "native-opengl": (0,),
                "native-directx": (0, 1),
                "normalbae-scannet-camera": (),
            }[self.spec_values[1]]
        return ()

    def ui_values(self):
        controls = self.controls
        return dict(
            cn_type="Control" if self.recipe == "union" else "Depth",
            cn_model=controls["model"],
            cn_model_signature=list(self.model_signature),
            cn_depth_preprocess="none",
            control_recipe=self.recipe,
            control_union_type=self.union_type,
        )

    def metadata(self):
        normal = self.spec_values[0] == "normal"
        uploaded = self.spec_values[3] == "explicit-upload"
        conversion = (
            (
                "caller_declared_normalbae_passthrough"
                if normal and not self.invert_channels
                else "caller_declared_moge_axes_with_source_derived_framenet_conversion"
                if normal
                else "caller_declared_white_near_disparity"
            )
            if uploaded
            else (
                "inferred_from_pinned_moge_and_bae_dsine_shared_framenet_labels"
                if normal
                else "native_white_near_disparity"
            )
        )
        return dict(
            family=self.family,
            model=self.controls["model"],
            model_signature=list(self.model_signature[1:]),
            sha256=self.artifact_sha256,
            source=self.source_label,
            guide=dict(zip(_SPEC_FIELDS, self.spec_values, strict=True)),
            semantic_convention="normalbae-scannet-camera" if normal else "white-near-disparity",
            provenance="caller-declared-upload" if uploaded else "owned-producer-execution",
            convention_qualification="caller-declared axes; content hashes bind pixels, not their geometric correctness"
            if uploaded
            else "source-qualified native producer convention",
            invert_channels=list(self.invert_channels),
            conversion_evidence=conversion,
            preprocessing="none",
            recipe=self.recipe,
            union_type=self.union_type,
            runtime_qualified=False,
        )

    def verify(self, manager, family, controls, spec):
        if model_family(family) != self.family or _spec_values(spec) != self.spec_values:
            raise ValueError(
                "The selected checkpoint family or geometry guide changed after capture."
            )
        path, signature = _resolved(manager, family, controls)
        normalized, recipe, union_type = _normalized(
            controls, self.artifact_sha256, path, signature, self.spec_values[0]
        )
        if (
            signature != self.model_signature
            or normalized != self.controls
            or (recipe, union_type) != (self.recipe, self.union_type)
        ):
            raise ValueError("The selected guide control or its settings changed after capture.")
        return self


def validate_guide_control(manager, family, controls, spec):
    """Explicit admission; verify exact source once, preserve user control strength."""
    values = _spec_values(spec)
    path, signature = _resolved(manager, family, controls)
    if signature[1] not in {row[0] for row in SOURCES.values()}:
        raise ValueError(
            "Select a reviewed SDXL guide control in the existing ControlNet selector."
        )
    digest = cached_file_sha256(path, configured_cache_root(manager) / "hashes").lower()
    if tuple(file_signature(path)) != signature:
        raise ValueError("The guide control changed during source verification; select it again.")
    normalized, recipe, union_type = _normalized(controls, digest, path, signature, values[0])
    return GuideControlReceipt(
        "SDXL",
        signature,
        digest,
        SOURCES[digest][1],
        values,
        json.dumps(normalized, sort_keys=True, allow_nan=False),
        recipe,
        union_type,
    )


def plan_guide_control(spec, family, manager, cn_model, cn_model_signature):
    """CTA preflight before restoring source/history; no automatic file selection."""
    return validate_guide_control(
        manager, family, dict(model=cn_model, model_signature=cn_model_signature), spec
    )
