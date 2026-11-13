"""One immutable view of the existing image request captures; no execution engine.

Native readers still classify/load files. Component inventory, sampling and
SizePlan remain their owners; this joins their submission-time results for
preparation, history and tool callers without consulting later profile edits.
"""

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_effective_image_plan"
COMPONENTS_KEY = "_captured_image_components"
VERSION = 1
_SAMPLING_KEYS = ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
_INTENT_KEYS = (
    "base_model_name",
    "image_component_overrides",
    "performance_selection",
    *_SAMPLING_KEYS,
    "sampling_defaults_source",
    "negative_prompt",
    "negative",
    "denoise",
    "inpaint_denoise",
    "cn_strength",
    "cn_start",
    "cn_stop",
    "loras",
    "width",
    "height",
    "custom_width",
    "custom_height",
    "aspect_ratios_selection",
    "resolution_auto",
    "size_mode",
    "image_resize_mode",
    "inpaint_toggle",
    "automatic_upscale",
    "automatic_upscale_target",
    "cascade_super_resolution",
    "bernini_low_model",
    "bernini_split_step",
    "bernini_lightx2v",
    "bernini_high_lora_strength",
    "bernini_low_lora_strength",
    "bernini_ref_max_size",
    "native_edit_recipe",
    "krea_style_lora",
    "krea_style_strength",
    "native_negative_model",
    "native_schedule_alpha",
    "native_schedule_beta",
    "native_schedule_mu",
    "native_schedule_std",
    "reference_adapter",
    "reference_adapter_model",
    "reference_adapter_vision",
    "reference_adapter_strength",
    "model_sampling_flux",
    "cfg_norm_strength",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _projector_metadata(items):
    return [
        dict(
            encoder_signature=list(item.encoder),
            reader_signature=list(item.source),
            projector_signature=list(item.projector) if item.projector is not None else None,
            selected_path=item.selected,
            state="selected" if item.projector is not None else "captured-absent",
        )
        for item in items
    ]


def _processing(data):
    return data.get("_sampling_captured") is False or data.get("task_type") in ("rembg", "upscale")


def _replay(data):
    from ruinedfooocus.features.gallery.image_metadata import read_metadata

    return data.get("_metadata_prompt") is True or (
        "_api_job" not in data
        and data.get("task_type") != "tool_call"
        and read_metadata(data.get("prompt")) is not None
    )


@dataclass(frozen=True)
class ImagePlanComponent:
    key: str
    category: str
    path: str
    filename: str
    status: str
    required: bool
    active: bool
    embedded: bool
    explicit: bool
    compatibility: str
    download_key: str | None
    source: str
    expected_bytes: int | None
    signature: tuple | None = None
    acquisition: PreparationRequest | None = None
    catalogue_identity: str | None = None

    def metadata(self):
        result = asdict(self)
        if self.acquisition is not None:
            result["acquisition"]["destination"] = str(self.acquisition.destination)
        return result


@dataclass(frozen=True)
class ImagePlanRecipe:
    ownership: str
    source_kind: str
    label: str
    values_json: str
    provenance_json: str

    @property
    def sampling(self):
        return json.loads(self.values_json)

    def metadata(self):
        return dict(
            ownership=self.ownership,
            source_kind=self.source_kind,
            label=self.label,
            sampling=self.sampling,
            provenance=json.loads(self.provenance_json),
        )


@dataclass(frozen=True)
class ImagePlanSize:
    ownership: str
    width: int | None
    height: int | None
    framing: str
    native_plan: object = None

    def metadata(self):
        return dict(
            ownership=self.ownership,
            width=self.width,
            height=self.height,
            framing=self.framing,
            size_plan=self.native_plan.metadata() if self.native_plan is not None else None,
        )


@dataclass(frozen=True)
class EffectiveImagePlan:
    version: int
    checkpoint_name: str
    checkpoint_path: str
    checkpoint_identity: str
    checkpoint_signature: tuple
    family: str
    native_class: str | None
    variant: str | None
    storage: str
    quantization: tuple[str, ...]
    recognition: str
    operation: str
    components: tuple[ImagePlanComponent, ...]
    recipe: ImagePlanRecipe
    size: ImagePlanSize
    native_facts_json: str
    intent_sha256: str
    projectors: tuple = ()
    lora_sources: tuple = ()
    lora_prompts: tuple = ()

    def loras_for_prompt(self, prompt):
        from ruinedfooocus.features.models.lora_sources import sources_for_prompt

        return sources_for_prompt(prompt, self.lora_sources, self.lora_prompts)

    @property
    def required_components(self):
        components = tuple(row for row in self.components if row.active and row.required)
        # Pending adapters use the existing generic preparation bridge. Their
        # typed owner already selected the destination and trusted request.
        pending_loras = tuple(
            ImagePlanComponent(
                key="lora:" + item.name,
                category="loras",
                path=str(item.path),
                filename=item.name,
                status="Missing",
                required=True,
                active=True,
                embedded=False,
                explicit=True,
                compatibility="service-owned",
                download_key=item.download_key,
                source=item.request.url,
                expected_bytes=item.request.expected_size,
                acquisition=item.request,
            )
            for item in self.lora_sources
            if item.signature is None and item.request is not None
        )
        distinct_loras = {}
        for row in pending_loras:
            if distinct_loras.setdefault(row.key, row) != row:
                raise ValueError("A pending image LoRA has conflicting preparation sources.")
        return components + tuple(distinct_loras.values())

    def metadata(self):
        """Detached JSON provenance; never return a mutable owner of queued intent."""
        return dict(
            version=self.version,
            checkpoint=dict(
                name=self.checkpoint_name,
                path=self.checkpoint_path,
                identity=self.checkpoint_identity,
                signature=list(self.checkpoint_signature),
            ),
            family=self.family,
            native_class=self.native_class,
            variant=self.variant,
            storage=self.storage,
            quantization=list(self.quantization),
            recognition=self.recognition,
            operation=self.operation,
            components=[row.metadata() for row in self.components],
            gguf_projectors=_projector_metadata(self.projectors),
            lora_sources=[item.metadata() for item in self.lora_sources],
            lora_prompts=[item.metadata() for item in self.lora_prompts],
            recipe=self.recipe.metadata(),
            size=self.size.metadata(),
            native_facts=json.loads(self.native_facts_json),
            intent_sha256=self.intent_sha256,
            qualification="Source compatibility; no model/device or generated-quality qualification.",
        )

    def verify_files(self):
        try:
            primary_changed = (
                self.checkpoint_path
                and tuple(file_signature(self.checkpoint_path)) != self.checkpoint_signature
            )
            if self.checkpoint_path and not primary_changed:
                from ruinedfooocus.features.models.component_profiles import checkpoint_identity

                primary_changed = (
                    checkpoint_identity(self.checkpoint_path) != self.checkpoint_identity
                )
        except OSError:
            primary_changed = True
        if primary_changed:
            raise ValueError("The image checkpoint changed after capture. Select it again.")
        for row in self.required_components:
            try:
                changed = (
                    row.signature is not None and tuple(file_signature(row.path)) != row.signature
                )
            except OSError:
                changed = True
            if changed:
                raise ValueError(
                    f"Image component {row.key} changed after capture. Select it again."
                )
        try:
            for item in self.projectors:
                item.verify()
        except OSError as error:
            raise ValueError(
                "A captured GGUF encoder, projector or reader changed. Submit again."
            ) from error
        for item in self.lora_sources:
            item.verify()

    def verify_request(self, data, controls=None):
        self.verify_files()
        controls = controls if controls is not None else data.get("controlnet") or {}
        if self.intent_sha256 != _intent(data, controls):
            raise ValueError(
                "The image request changed after plan capture. Capture the new request again."
            )


def capture_component_inventory(state, choices=None):
    """Freeze the inventory already read by checkpoint-scoped component capture."""
    from ruinedfooocus.features.models.component_candidates import validate_inventory

    validate_inventory(state)
    result = []
    for row in state["rows"]:
        # Approximate codecs already own distinct encoder/decoder files.
        files = row.get("files") or [row]
        for item in files:
            location = str(item.get("location", row.get("location", "")))
            embedded = row.get("status") == "Embedded in checkpoint"
            path = Path(location)
            signature = tuple(file_signature(path)) if location and path.is_file() else None
            result.append(
                ImagePlanComponent(
                    key=row["key"] + (":" + item["role"] if item is not row else ""),
                    category=row["category"],
                    path=location,
                    filename=str(item.get("filename", row.get("filename", ""))),
                    status=row["status"],
                    required=row.get("required", True),
                    active=row.get("active", True),
                    embedded=embedded,
                    explicit=row["key"] in (choices or {}) and not embedded,
                    compatibility=row.get("compatibility", "unknown"),
                    download_key=row.get("download_key"),
                    source=row.get("source", ""),
                    expected_bytes=row.get("published_bytes"),
                    signature=signature,
                )
            )
    return tuple(result)


def get_effective_image_plan(data):
    plan = data.get(KEY)
    if plan is not None and not isinstance(plan, EffectiveImagePlan):
        raise ValueError("An effective image plan must be captured by the image request service.")
    return plan


def _bind_component_acquisitions(components, paths):
    """Bind canonical inventory identity and the actual producer transport once."""
    from ruinedfooocus.features.models.model_registry import download_identity

    bound = []
    for row in components:
        if row.embedded or not row.active or not row.required or row.category == "operation":
            bound.append(row)
            continue
        if not row.download_key:
            request = PreparationRequest("", Path(row.path))
            if row.catalogue_identity is not None or (
                row.acquisition is not None and row.acquisition != request
            ):
                raise ValueError(
                    f"Image component {row.key} local acquisition changed after capture. Submit it again."
                )
            bound.append(row if row.acquisition is not None else replace(row, acquisition=request))
            continue
        entry = paths.DOWNLOADABLE_FILES.get(row.download_key)
        if (
            entry is None
            or row.source
            and row.source != entry.get("url", "")
            or row.category != entry.get("path", "").removeprefix("path_")
            or row.filename != entry.get("filename")
        ):
            raise ValueError(
                f"Image component {row.key} acquisition metadata changed after inventory capture. Submit it again."
            )
        identity = download_identity(entry)
        request = paths.preparation_request(row.download_key, destination=Path(row.path))
        if not isinstance(request, PreparationRequest):
            raise ValueError(
                "Image component acquisition must use an immutable preparation request."
            )
        if (
            row.expected_bytes is not None
            and row.expected_bytes != request.expected_size
            or row.acquisition is not None
            and (row.acquisition != request or row.catalogue_identity != identity)
        ):
            raise ValueError(
                f"Image component {row.key} acquisition metadata changed after capture. Submit it again."
            )
        bound.append(
            row
            if row.acquisition is not None
            else replace(row, acquisition=request, catalogue_identity=identity)
        )
    return components if all(a is b for a, b in zip(components, bound)) else tuple(bound)


def _intent(data, controls):
    values = {key: data[key] for key in _INTENT_KEYS if key in data}
    values["replay_ownership"] = _replay(data)
    guide = _guide_metadata(data)
    if guide is not None:
        values["control_guide"] = guide
    if not _processing(data):
        from ruinedfooocus.features.models import gguf_projectors
        from ruinedfooocus.features.models import lora_sources

        values[gguf_projectors.KEY] = _projector_metadata(gguf_projectors.captured(data) or ())
        values[lora_sources.KEY] = [item.metadata() for item in lora_sources.captured(data) or ()]
        values[lora_sources.PROMPTS_KEY] = [
            item.metadata() for item in lora_sources.captured_prompts(data) or ()
        ]
    values["control"] = {
        key: controls[key]
        for key in (
            "type",
            "denoise",
            "model",
            "control_recipe",
            "inpaint_mode",
            "strength",
            "start",
            "stop",
        )
        if key in controls
    }
    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY

    if data.get(SIZE_KEY) is not None:
        values[SIZE_KEY] = data[SIZE_KEY].metadata()
    return hashlib.sha256(_json(values).encode()).hexdigest()


def _guide_metadata(data):
    """Export technical guide identity; runtime ownership stays with its typed owner."""
    if "_editor_guide_request" not in data:
        return None
    from ruinedfooocus.features.image_editing.guide_handoffs import captured

    request = captured(data)
    if request is None:
        return None
    metadata = request.metadata()
    facts = {
        key: metadata[key]
        for key in (
            "purpose",
            "provenance",
            "preprocessing",
            "source_sha256",
            "guide_sha256",
            "source_signature",
            "guide_signature",
            "source_pixel_sha256",
            "guide_pixel_sha256",
            "alpha_policy",
            "denoise",
        )
        if key in metadata
    }
    spec_fields = ("kind", "convention", "projection", "producer", "width", "height")
    facts["guide"] = {
        key: value for key, value in metadata.get("guide", {}).items() if key in spec_fields
    }
    for field in ("source_pixels", "guide_pixels"):
        facts[field] = {
            key: value
            for key, value in metadata.get(field, {}).items()
            if key in ("size", "dtype", "channels", "color_space")
        }
    control_fields = (
        "family",
        "model",
        "model_signature",
        "sha256",
        "source",
        "semantic_convention",
        "invert_channels",
        "conversion_evidence",
        "preprocessing",
        "recipe",
        "union_type",
        "runtime_qualified",
    )
    control = metadata.get("control", {})
    facts["control"] = {key: value for key, value in control.items() if key in control_fields}
    if "guide" in control:
        facts["control"]["guide"] = {
            key: value for key, value in control["guide"].items() if key in spec_fields
        }
    control_values = metadata.get("controls", {})
    facts["controls"] = {
        key: value
        for key, value in control_values.items()
        if key
        in (
            "type",
            "model",
            "model_signature",
            "strength",
            "start",
            "stop",
            "depth_preprocess",
            "recipe",
            "union_type",
        )
    }
    for settings in (facts["control"], facts["controls"]):
        if "model" in settings:
            settings["model"] = str(settings["model"]).replace("\\", "/").rsplit("/", 1)[-1]
        signature = settings.get("model_signature")
        if signature and isinstance(signature[0], str):
            settings["model_signature"] = signature[1:]
    # Hash the complete frozen execution settings without exporting private paths.
    facts["controls_sha256"] = hashlib.sha256(_json(request.controls()).encode()).hexdigest()
    derived = metadata.get("derived_guide", {})
    facts["derived_guide"] = {
        key: value
        for key, value in derived.items()
        if key
        in (
            "invert_channels",
            "native_conversion",
            "native_revision",
            "applications",
            "spatial_flip",
            "reestimate",
        )
    }
    facts["derived_guide"]["native_sources"] = [
        {key: value for key, value in row.items() if key in ("file", "sha256")}
        for row in derived.get("native_sources", ())
    ]
    return json.loads(_json(facts))


def finalize_prepared_image_plan(original_data, prepared_data):
    """Bind newly resolved projector/adapter sources after owned preparation.

    Original intent, component selection, recipe and geometry remain frozen.
    An added projector needs the original active missing CLIP request and its
    matching prepared signature. Existing selected/absent projector captures
    cannot retarget. The existing GGUF owner resolves and verifies every packet.
    Pending trusted LoRAs are finalized by their existing typed receipt owner.
    """
    from ruinedfooocus.core.cache_contract import canonical_path_key
    from ruinedfooocus.features.generation.image_preparation import KEY as ASSETS_KEY, ImageAssets
    from ruinedfooocus.features.models import gguf_projectors, lora_sources

    previous = get_effective_image_plan(original_data)
    current = get_effective_image_plan(prepared_data)
    if current is not previous:
        raise ValueError("The effective image plan changed during preparation.")
    if previous is None:
        return dict(prepared_data)
    previous.verify_request(original_data)
    if prepared_data.get(COMPONENTS_KEY) != original_data.get(COMPONENTS_KEY):
        raise ValueError("The image component snapshot changed during preparation.")
    if _processing(original_data):
        previous.verify_request(prepared_data)
        return dict(prepared_data)
    shadow = dict(
        prepared_data,
        **{
            gguf_projectors.KEY: previous.projectors,
            lora_sources.KEY: previous.lora_sources,
        },
    )
    previous.verify_request(shadow)
    adapters = lora_sources.bind_prepared(original_data, prepared_data)
    incoming = gguf_projectors.captured(prepared_data) or ()
    known = {canonical_path_key(item.encoder[0]): item for item in previous.projectors}
    updated = {canonical_path_key(item.encoder[0]): item for item in incoming}
    if len(updated) != len(incoming) or any(
        updated.get(key) != item for key, item in known.items()
    ):
        raise ValueError("A captured GGUF projector selection changed during preparation.")
    if incoming == previous.projectors and adapters == previous.lora_sources:
        previous.verify_request(prepared_data)
        return dict(prepared_data)
    before, after = original_data.get(ASSETS_KEY), prepared_data.get(ASSETS_KEY)
    missing = (
        {
            canonical_path_key(item.request.destination)
            for item in before.artifacts
            if item.role == "clip" and item.signature is None
        }
        if isinstance(before, ImageAssets)
        else set()
    )
    receipts = (
        {
            canonical_path_key(item.request.destination): item.signature
            for item in after.artifacts
            if item.role == "clip" and item.signature is not None
        }
        if isinstance(after, ImageAssets)
        else {}
    )
    if any(
        key not in missing or receipts.get(key) != item.encoder
        for key, item in updated.items()
        if key not in known
    ):
        raise ValueError(
            "A new GGUF projector needs its captured missing encoder and prepared receipt."
        )
    result = dict(prepared_data)
    result[lora_sources.KEY] = adapters
    result[KEY] = replace(
        previous,
        projectors=incoming,
        lora_sources=adapters,
        intent_sha256=_intent(result, result.get("controlnet") or {}),
    )
    result[KEY].verify_request(result)
    return result


def _native_facts(family):
    from ruinedfooocus.features.models.native_snapshot import (
        get_image_family_facts,
        load_native_snapshot,
    )

    snapshot = load_native_snapshot()
    facts = get_image_family_facts(family, snapshot=snapshot) if snapshot is not None else None
    return dict(
        snapshot=snapshot.metadata() if snapshot is not None else None,
        family=facts.metadata() if facts is not None else None,
    )


@lru_cache(maxsize=256)
def _storage_facts(path, signature):
    """Storage evidence is header-only and signature-cached; it grants no recipe."""
    from ruinedfooocus.features.models.model_identity import cached_safetensors_header

    suffix = Path(path).suffix.lower()
    if suffix in (".safetensors", ".sft"):
        header = cached_safetensors_header(path)
        result = tuple(
            sorted(
                {
                    row["dtype"]
                    for row in header.values()
                    if isinstance(row, dict) and isinstance(row.get("dtype"), str)
                }
            )
        )
    elif suffix == ".gguf":
        from gguf.constants import GGMLQuantizationType
        from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

        header, _ = read_gguf_header(path, include_storage=True)
        result = tuple(
            sorted(
                {
                    GGMLQuantizationType(row["gguf_type"]).name
                    for row in header.values()
                    if isinstance(row, dict) and "gguf_type" in row
                }
            )
        )
    else:
        header = {}
        result = ()
    if tuple(file_signature(path)) != signature:
        raise ValueError("The image checkpoint changed during storage inspection.")
    metadata = header.get("__metadata__", {})
    return result, _json(
        {
            key: metadata[key]
            for key in ("modelspec.variant", "variant", "model_variant", "training_variant")
            if isinstance(metadata, dict) and isinstance(metadata.get(key), str)
        }
    )


def _size(data, controls, resolutions, replay, *, processing=False):
    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan

    native = data.get(SIZE_KEY)
    if native is not None and not isinstance(native, SizePlan):
        raise ValueError("The image size plan must be captured by the native size owner.")
    source_owned = bool(
        data.get("inpaint_toggle")
        or data.get("input_image") is not None
        and data.get("image_resize_mode", "preserve") == "preserve"
    )
    ownership = (
        "replay"
        if replay
        else "source"
        if source_owned
        else "automatic"
        if data.get("resolution_auto") is True
        else "manual"
    )
    target = native.target if native is not None else None
    if target is None and "_editor_guide_request" in data:
        from ruinedfooocus.features.image_editing.guide_handoffs import automatic_source_size

        target = automatic_source_size(data)
    framing = "source_proportions" if source_owned else data.get("image_resize_mode", "preserve")
    if (
        target is None
        and not processing
        and data.get("width") is not None
        and data.get("height") is not None
    ):
        target = data["width"], data["height"]
    if target is None and not processing:
        target = getattr(resolutions, "aspect_ratios", {}).get(data.get("aspect_ratios_selection"))
    if (
        target is None
        and data.get("custom_width") is not None
        and data.get("custom_height") is not None
    ):
        target = data["custom_width"], data["custom_height"]
    if processing:
        from ruinedfooocus.features.image_processing import image_composition, latent_composition

        composition = image_composition.captured(data) if image_composition.KEY in data else None
        latent = latent_composition.captured(data) if latent_composition.KEY in data else None
        if composition is not None:
            target = composition.output_size
        elif latent is not None:
            target = latent.pixel_size
        elif controls.get("type") == "image-transform":
            from ruinedfooocus.features.image_processing.image_geometry_tools import (
                KEY,
                TransformRequest,
            )

            request = data.get(KEY)
            if not isinstance(request, TransformRequest):
                raise ValueError(
                    "Capture native transform geometry before its effective image plan."
                )
            target = request.output_size
            framing = "native " + dict(request.parameters)["transform_operation"]
        elif native is None and (
            target is None or controls.get("type") not in ("upscale", "restore-seedvr2")
        ):
            source = data.get("input_image")
            size = getattr(source, "size", None)
            if isinstance(size, tuple) and len(size) == 2:
                target = size
        ownership = "operation"
    return ImagePlanSize(
        ownership,
        *(target or (None, None)),
        framing,
        native,
    )


def _recipe(data, family, performance, replay, facts):
    from ruinedfooocus.features.models.performance import MODEL_DEFAULT_SELECTIONS
    from ruinedfooocus.features.generation.native_sampling import KEY as NATIVE_KEY

    selected = data.get("performance_selection")
    automatic = selected in MODEL_DEFAULT_SELECTIONS and not replay
    label = data.get("sampling_defaults_source") or (
        "Captured Model Defaults" if automatic else "Library replay" if replay else "Own settings"
    )
    kind = (
        "native-workflow-example"
        if label.startswith("Native workflow example")
        else "app-or-family-fallback"
        if "fallback" in label.casefold()
        else "compatible-recipe"
        if automatic
        else "replay"
        if replay
        else "manual"
    )
    sampling = {key: data[key] for key in _SAMPLING_KEYS if key in data}
    for key in (
        "_model_default_sampling_shift",
        "_model_default_sigmas",
        "_model_default_guidance",
    ):
        if key in data:
            sampling[key] = data[key]
    if data.get(NATIVE_KEY) is not None:
        sampling["native_sampling"] = data[NATIVE_KEY].metadata()
    recipe = {}
    details = getattr(getattr(performance, "model_recommendations", None), "family_details", None)
    if automatic and callable(details):
        recipe = details(data["base_model_name"], family=family) or {}
    provenance = {
        key: recipe[key]
        for key in ("id", "variant", "provenance", "sampling_provenance", "identity_basis")
        if key in recipe
    }
    provenance["selection"] = selected
    if kind == "native-workflow-example":
        provenance["template_example"] = (facts.get("family") or {}).get("example_recipe")
    return ImagePlanRecipe(
        "automatic" if automatic else "replay" if replay else "manual",
        kind,
        label,
        _json(sampling),
        _json(provenance),
    )


def capture_effective_image_plan(
    data, models, values, paths, *, family=None, controls=None, performance=None, resolutions=None
):
    """Join completed submission captures; never change settings or acquire files.

    A recapture of unchanged intent retains the exact object. Mutating a captured
    request requires a new service-owned capture, rather than accepting stale
    provenance or a client-supplied metadata dictionary as execution authority.
    """
    from ruinedfooocus.features.models.image_scope import require_image_operation

    require_image_operation(data)
    result = dict(data)
    previous = get_effective_image_plan(result)
    controls = controls if controls is not None else result.get("controlnet") or {}
    if result.get("task_type") in ("llama", "chat"):
        result.pop(KEY, None)
        result.pop(COMPONENTS_KEY, None)
        return result
    name = result.get("base_model_name")
    processing = _processing(result)
    if previous is None and not processing and name:
        from ruinedfooocus.features.models import lora_sources

        result = lora_sources.capture_request(result, models, manager=paths)
    intent = _intent(result, controls)
    if previous is not None:
        previous.verify_request(result, controls)
        return result
    if processing:
        from ruinedfooocus.features.generation.image_preparation import (
            KEY as ASSETS_KEY,
            ImageAssets,
        )

        assets = result.get(ASSETS_KEY)
        resources = ()
        if isinstance(assets, ImageAssets):
            resources = tuple(
                ImagePlanComponent(
                    item.role,
                    "operation",
                    str(item.request.destination),
                    item.name,
                    "Installed" if item.signature is not None else "Missing",
                    True,
                    True,
                    False,
                    True,
                    "service-owned",
                    None,
                    item.request.url,
                    item.request.expected_size,
                    item.signature,
                )
                for item in assets.artifacts
            )
        result[KEY] = EffectiveImagePlan(
            VERSION,
            "",
            "",
            "",
            (),
            "",
            None,
            None,
            "",
            (),
            "operation-service",
            str(controls.get("type") or result.get("task_type") or "process").lower(),
            resources,
            ImagePlanRecipe("operation", "operation", "Image operation parameters", "{}", "{}"),
            _size(result, controls, resolutions, False, processing=True),
            _json(dict(snapshot=None, family=None)),
            intent,
        )
        result.pop(COMPONENTS_KEY, None)
        return result
    if not name:
        result.pop(KEY, None)
        return result
    checkpoint = models.get_file("checkpoints", name)
    if checkpoint is None:
        raise ValueError("The selected image checkpoint is no longer installed.")
    checkpoint = Path(checkpoint).resolve()
    from ruinedfooocus.features.models.image_scope import require_image_checkpoint
    from ruinedfooocus.features.models.component_profiles import (
        SCOPE_KEY,
        checkpoint_identity,
        component_inventory,
    )

    metadata = models.get_models_by_path("checkpoints", name, fetch=False)
    family = family or models.get_model_base(metadata)
    require_image_checkpoint(checkpoint, family)
    signature = tuple(file_signature(checkpoint))
    identity = checkpoint_identity(checkpoint)
    native = {}
    if "comfy.model_detection" in sys.modules and checkpoint.suffix.lower() in (
        ".safetensors",
        ".sft",
        ".gguf",
    ):
        from ruinedfooocus.features.models.model_architecture import inspect_model_file

        native = inspect_model_file(checkpoint)
    from ruinedfooocus.features.models.image_components import ALIASES

    actual_family = native.get("architecture")
    if (
        actual_family
        and family not in (None, "", "Unknown")
        and ALIASES.get(actual_family, actual_family) != ALIASES.get(family, family)
    ):
        raise ValueError(
            "The image checkpoint header differs from the selected family. Refresh its metadata."
        )
    family = actual_family or family
    quantization, header_metadata = _storage_facts(str(checkpoint), signature)
    variants = {
        value.strip().casefold()
        for source in (
            json.loads(header_metadata),
            native.get("metadata", {}),
            metadata if metadata.get("_rf_file_signature") == list(signature) else {},
        )
        for key in ("modelspec.variant", "variant", "model_variant", "training_variant")
        if isinstance(value := source.get(key), str) and value.strip()
    }
    components = result.get(COMPONENTS_KEY)
    if components is not None and result.get(SCOPE_KEY) != identity:
        raise ValueError(
            "The image components were captured for another checkpoint. Capture them again."
        )
    if components is None:
        components = capture_component_inventory(
            component_inventory(
                checkpoint, family, values, paths, result.get("image_component_overrides")
            ),
            result.get("image_component_overrides"),
        )
    if not isinstance(components, tuple) or any(
        not isinstance(row, ImagePlanComponent) for row in components
    ):
        raise ValueError("Image components must be captured by the checkpoint inventory owner.")
    components = _bind_component_acquisitions(components, paths)
    result[COMPONENTS_KEY] = components
    replay = _replay(result)
    from ruinedfooocus.features.models import gguf_projectors

    facts = dict(_native_facts(family))
    guide = _guide_metadata(result)
    if guide is not None:
        facts["control_guide"] = guide
    facts["selection"] = dict(
        native_class=native.get("native_class"),
        native_config=native.get("native_config", {}),
        native_variant=native.get("native_variant"),
        variant_declarations=sorted(variants),
    )
    plan = EffectiveImagePlan(
        VERSION,
        str(name),
        str(checkpoint),
        identity,
        signature,
        family,
        native.get("native_class"),
        next(iter(variants))
        if len(variants) == 1
        else native.get("native_variant")
        if not variants
        else None,
        checkpoint.suffix.lower().lstrip("."),
        quantization,
        "native-header" if native.get("native_class") else "initialized-family",
        "inpaint"
        if result.get("inpaint_toggle")
        else "generate"
        if str(controls.get("type") or "").lower() in ("", "none")
        else str(controls["type"]).lower(),
        components,
        _recipe(result, family, performance, replay, facts),
        _size(result, controls, resolutions, replay),
        _json(facts),
        intent,
        projectors=gguf_projectors.captured(result) or (),
        lora_sources=lora_sources.captured(result) or (),
        lora_prompts=lora_sources.captured_prompts(result) or (),
    )
    plan.verify_files()
    result[KEY] = plan
    return result
