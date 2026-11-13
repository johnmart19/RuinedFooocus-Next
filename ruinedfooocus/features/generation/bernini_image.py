"""Source-bound Bernini-R still editing, using the pinned native image graph."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_identity import file_signature

FAMILY = "BerniniImage"
KEY = "_native_bernini_image"
CLIP_ROLE = "clip_bernini_umt5"
VAE_ROLE = "vae_bernini"
FIELDS = (
    "bernini_low_model",
    "bernini_split_step",
    "bernini_lightx2v",
    "bernini_high_lora_strength",
    "bernini_low_lora_strength",
    "bernini_ref_max_size",
)
DEFAULTS = dict(
    bernini_low_model="wan2.2_bernini_r_low_noise_fp8_scaled.safetensors",
    bernini_split_step=None,
    bernini_lightx2v=False,
    bernini_high_lora_strength=3.0,
    bernini_low_lora_strength=1.5,
    bernini_ref_max_size=848,
)
MAX_PIXELS = 16 * 1024 * 1024


@lru_cache(maxsize=1)
def sources():
    """Immutable URLs, full payload hashes and bounded header identities."""
    document = json.loads(
        resource_path("bernini_image_components.json").read_text(encoding="utf-8")
    )
    return document["components"]


@lru_cache(maxsize=128)
def _header_role(signature):
    path, size = Path(signature[0]), signature[1]
    candidates = [(role, row) for role, row in sources().items() if row["bytes"] == size]
    if path.suffix.lower() not in (".safetensors", ".sft") or not candidates:
        return None
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        length = int.from_bytes(raw_size, "little")
        if len(raw_size) != 8 or not 2 <= length <= 1024 * 1024:
            return None
        raw = stream.read(length)
    if len(raw) != length or tuple(file_signature(path)) != signature:
        return None
    digest = hashlib.sha256(raw).hexdigest()
    matches = [role for role, row in candidates if row["header_sha256"] == digest]
    return matches[0] if len(matches) == 1 else None


def header_role(path):
    """A known header is a candidate, never a payload-verification receipt."""
    try:
        return _header_role(tuple(file_signature(path)))
    except (OSError, ValueError, TypeError):
        return None


def primary_family(path):
    return FAMILY if header_role(path) == "high" else None


def require_header(path, role):
    if header_role(path) != role:
        raise ValueError(f"Choose the published Bernini-R {role} component from its image recipe.")
    return tuple(file_signature(path))


def _number(data, key, default, low, high, *, integer=False):
    value = data.get(key)
    value = default if value is None or value == "" else value
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not low <= value <= high
        or integer
        and int(value) != value
    ):
        raise ValueError(f"{key} must be between {low} and {high}.")
    return int(value) if integer else float(value)


@dataclass(frozen=True)
class Reference:
    size: tuple[int, int]
    rgb: bytes

    @classmethod
    def capture(cls, image):
        if not isinstance(image, Image.Image) or not 0 < image.width * image.height <= MAX_PIXELS:
            raise ValueError("Each Bernini reference must be an image of at most 16 megapixels.")
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        base = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        base.alpha_composite(rgba)
        rgb = base.convert("RGB")
        return cls(rgb.size, rgb.tobytes())

    def image(self):
        return Image.frombytes("RGB", self.size, self.rgb)

    def metadata(self):
        return dict(size=list(self.size), rgb_sha256=hashlib.sha256(self.rgb).hexdigest())


@dataclass(frozen=True)
class BerniniImagePlan:
    high: tuple
    source: Reference
    references: tuple[Reference, ...]
    steps: int
    split: int
    cfg: float
    sampler: str
    scheduler: str
    lightx2v: bool
    high_strength: float
    low_strength: float
    ref_max_size: int
    framing: str
    clip_skip: int
    replay_components: tuple = ()

    def metadata(self):
        return dict(
            version=1,
            family=FAMILY,
            recipe="pinned-comfy-blueprint:blueprints/Image Edit (Bernini-R).json",
            latent_layout="1,16,1,H/8,W/8",
            high_sha256=sources()["high"]["sha256"],
            low_sha256=sources()["low"]["sha256"],
            source=self.source.metadata(),
            references=[reference.metadata() for reference in self.references],
            steps=self.steps,
            split_step=self.split,
            cfg=self.cfg,
            sampler=self.sampler,
            scheduler=self.scheduler,
            lightx2v=self.lightx2v,
            high_lora_strength=self.high_strength,
            low_lora_strength=self.low_strength,
            ref_max_size=self.ref_max_size,
            framing=self.framing,
            clip_skip=self.clip_skip,
            prompt_header="You are a helpful assistant specialized in image editing.",
            alpha="RGBA references composited over neutral RGB; opaque RGB output",
        )


def replay_values(metadata):
    return {key: metadata[key] for key in (*FIELDS, "native_bernini_image") if key in metadata}


def capture(data, path):
    from ruinedfooocus.features.generation.image_output_batches import OutputPlan, with_output_plan
    from ruinedfooocus.features.generation.native_tensor_io import enabled

    high = require_header(path, "high")
    result = dict(data)
    repetitions = result.get("image_number", 1)
    if type(repetitions) is not int or not 1 <= repetitions <= 100:
        raise ValueError("Choose 1 through 100 bounded Bernini still-image repetitions.")
    controls = result.get("controlnet") or {}
    if (
        controls.get("type") != "edit"
        or result.get("input_image") is None
        or result.get("inpaint_toggle")
    ):
        raise ValueError(
            "Bernini-R still editing requires Native image edit, one source and no mask."
        )
    if any(result.get(key) for key in ("length", "frames", "reference_video", "source_video")):
        raise ValueError("Bernini-R image editing accepts still images only.")
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Bernini-R still editing requires the native image engine.")
    if (
        enabled(result)
        or any(
            result.get(key)
            for key in (
                "model_recipe_path",
                "sdxl_refiner_model",
                "gligen_model",
                "hypernetwork_model",
            )
        )
        or result.get("reference_adapter") not in (None, "", "none")
        or result.get("automatic_upscale", "Off") not in (None, "Off")
        or result.get("moss_image_slg")
        or result.get("cfg_norm_strength") not in (None, 0, 0.0)
        or result.get("cfg_norm_auto")
        or result.get("cfg_norm_pre_cfg")
        or result.get("sampling_shift") not in (None, "")
        or result.get("native_edit_recipe") not in (None, "", "auto")
    ):
        raise ValueError(
            "Clear alternate image conditioning, model patches and automatic upscaling for Bernini-R."
        )
    if (
        result.get("model_patch_recipe", "none") != "none"
        or result.get("lora_hook_mode", "off") != "off"
        or result.get("control_recipe", "ordinary") != "ordinary"
    ):
        raise ValueError("Clear alternate native patches/hooks before Bernini-R image editing.")
    turbo = result.get("bernini_lightx2v", False)
    if type(turbo) is not bool:
        raise ValueError("Bernini Lightx2v must be enabled or disabled.")
    result["steps"] = result.get("steps", result.get("custom_steps"))
    steps = _number(result, "steps", 6 if turbo else 40, 2, 200, integer=True)
    split = _number(
        result, "bernini_split_step", min(3 if turbo else 20, steps - 1), 1, steps - 1, integer=True
    )
    cfg = _number(result, "cfg", 1 if turbo else 5, 0, 100)
    sampler = result.get("sampler_name") or "res_multistep"
    scheduler = result.get("scheduler") or "simple"
    # Native lists are read only after app startup, never to detect a model.
    from comfy.samplers import KSampler

    if sampler not in KSampler.SAMPLERS or scheduler not in KSampler.SCHEDULERS:
        raise ValueError("Choose a native Bernini sampler and scheduler.")
    refs = result.get("reference_images") or []
    if not isinstance(refs, (list, tuple)) or len(refs) > 8:
        raise ValueError("Bernini-R accepts at most eight ordered reference images.")
    refs = [
        entry[0] if isinstance(entry, (list, tuple)) and len(entry) == 2 else entry
        for entry in refs
    ]
    source = Reference.capture(result["input_image"])
    references = tuple(Reference.capture(ref) for ref in refs)
    if sum(math.prod(ref.size) for ref in (source, *references)) > MAX_PIXELS:
        raise ValueError("Bernini inputs exceed the combined 16-megapixel budget.")
    framing = result.get("image_resize_mode") or "preserve"
    if framing not in ("preserve", "pad", "crop"):
        raise ValueError("Choose a source image framing mode.")
    plan = BerniniImagePlan(
        high,
        source,
        references,
        steps,
        split,
        cfg,
        sampler,
        scheduler,
        turbo,
        _number(result, "bernini_high_lora_strength", 3, -100, 100),
        _number(result, "bernini_low_lora_strength", 1.5, -100, 100),
        _number(result, "bernini_ref_max_size", 848, 16, 8192, integer=True),
        framing,
        _number(result, "clip_skip", 1, 1, 32, integer=True),
    )
    if plan.ref_max_size % 16:
        raise ValueError("Bernini reference size must be a multiple of 16.")
    previous = result.get("native_bernini_image")
    if previous is not None:
        if (
            not isinstance(previous, dict)
            or {key: value for key, value in previous.items() if key != "components"}
            != plan.metadata()
        ):
            raise ValueError(
                "Reattach the original ordered Bernini source/references and settings to replay."
            )
        components = previous.get("components", [])
        if (
            not isinstance(components, list)
            or len(components) not in (0, 4, 5)
            or any(
                not isinstance(row, list)
                or len(row) != 5
                or not isinstance(row[0], str)
                or not isinstance(row[1], str)
                or Path(row[1]).name != row[1]
                or type(row[2]) is not int
                or row[2] < 1
                or any(not isinstance(value, str) or not value.isdecimal() for value in row[3:])
                for row in components
            )
        ):
            raise ValueError("The saved Bernini component identities are malformed.")
        prior_plan = result.get(KEY)
        plan = replace(
            plan,
            replay_components=tuple(tuple(row) for row in components)
            or (prior_plan.replay_components if isinstance(prior_plan, BerniniImagePlan) else ()),
        )
    result[KEY] = plan
    result["native_bernini_image"] = plan.metadata()
    result["steps"], result["cfg"] = steps, cfg
    result.setdefault("custom_steps", steps)
    result["sampler_name"], result["scheduler"] = sampler, scheduler
    return with_output_plan(result, OutputPlan("bernini-image", 1, 1))


def _signed_artifact(manager, role, category, name, source_role):
    from ruinedfooocus.features.generation.image_preparation import _artifact

    row = sources()[source_role]
    if name not in (row["filename"], row["catalogue_key"]):
        supplied = Path(name)
        if not supplied.is_absolute():
            if ".." in supplied.parts:
                raise ValueError("Choose a Bernini stage from the configured model folders.")
            supplied = next(
                (
                    Path(root) / supplied
                    for root in manager.get_model_folders(category)
                    if (Path(root) / supplied).is_file()
                ),
                None,
            )
        if supplied is None or not supplied.is_file():
            raise ValueError(
                "The selected Bernini component is not installed. Choose its registered source or refresh."
            )
        name = str(supplied.resolve())
    item = _artifact(manager, role, category, name, key=row["catalogue_key"])
    if (
        item.request.sha256 != row["sha256"]
        or item.request.expected_size != row["bytes"]
        or item.request.url != row["url"]
    ):
        raise ValueError(
            "The Bernini component catalogue differs from its captured publisher source."
        )
    if item.signature is not None:
        require_header(item.request.destination, source_role)
    # Preparation verifies an already installed payload too; presence/header
    # alone cannot authorize loading these shared Wan architectures.
    return replace(item, signature=None, receipt_required=True)


def capture_assets(data, path, manager, values):
    """Freeze signed requirements; never download or hash weights during capture."""
    from ruinedfooocus.backends.comfy.component_selection import (
        component_settings,
        validate_component_slots,
    )
    from ruinedfooocus.features.models.resource_defaults import resource_name
    from ruinedfooocus.features.models.effective_image_plan import get_effective_image_plan

    result = capture(data, path)
    plan = get_effective_image_plan(result)
    components = {row.key: row for row in plan.components} if plan is not None else {}
    resources = component_settings(values or {}, result.get("image_component_overrides"))
    validate_component_slots(
        result.get("image_component_overrides"), {"clip_keys": [CLIP_ROLE], "vae_key": VAE_ROLE}
    )
    items = [
        _signed_artifact(manager, "bernini-high", "checkpoints", str(path), "high"),
        _signed_artifact(
            manager,
            "bernini-low",
            "diffusion_models",
            result.get("bernini_low_model") or DEFAULTS["bernini_low_model"],
            "low",
        ),
    ]
    for role, category, key, source_role in (
        ("clip", "clip", CLIP_ROLE, "clip"),
        ("vae", "vae", VAE_ROLE, "vae"),
    ):
        selected = resource_name(key, resources)
        row = sources()[source_role]
        component = components.get(key)
        if plan is not None and (component is None or component.acquisition is None):
            raise ValueError("The Bernini component is absent from its captured image plan.")
        local = (
            component.download_key is None
            if component is not None
            else bool((result.get("image_component_overrides") or {}).get(key))
        )
        if component is not None and local and component.acquisition.url:
            raise ValueError(
                "A local Bernini component cannot use a publisher acquisition request."
            )
        if selected == row["filename"] and not local:
            item = _signed_artifact(manager, role, category, selected, source_role)
        else:
            # Explicit local alternatives retain their own native contract and
            # signature; no arbitrary URL becomes an acquisition request.
            from ruinedfooocus.features.generation.image_preparation import _artifact

            item = _artifact(manager, role, category, selected, local_only=True)
            if item.signature is None:
                raise ValueError(
                    "Install an explicit Bernini encoder/codec override locally first."
                )
            validate_component(item.request.destination, source_role)
        if component is not None and (
            item.request != component.acquisition or local and item.signature != component.signature
        ):
            raise ValueError("The Bernini component differs from its captured image plan.")
        items.append(item)
    if result[KEY].lightx2v:
        items.append(
            _signed_artifact(
                manager, "bernini-lightx2v", "loras", sources()["lora"]["filename"], "lora"
            )
        )
    return result, tuple(items)


def validate_component(path, role):
    from ruinedfooocus.features.models.component_candidates import header_shapes, wan21_rgb_contract

    if Path(path).suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError(
            "Bernini encoder/codec overrides require inspected native safetensors files."
        )
    shapes = header_shapes(path)
    if role == "clip":
        valid = shapes.get("shared.weight") == (256384, 4096) and shapes.get(
            "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
        ) == (10240, 4096)
    else:
        valid = wan21_rgb_contract(shapes)
    if not valid:
        raise ValueError(
            "Choose a native UMT5 XXL encoder / 16-channel RGB Wan2.1 codec for Bernini-R."
        )


def require_plan(data, path=None):
    plan = (data or {}).get(KEY)
    if not isinstance(plan, BerniniImagePlan):
        raise ValueError("Capture the signed one-frame Bernini image recipe before loading.")
    if path is not None and tuple(file_signature(path)) != plan.high:
        raise ValueError("The captured Bernini high stage changed. Submit again.")
    return plan


def _prepared(data):
    from ruinedfooocus.features.generation.image_preparation import ImageAssets, KEY as ASSETS_KEY

    assets = data.get(ASSETS_KEY)
    if not isinstance(assets, ImageAssets):
        raise ValueError("Prepare the signed Bernini image components before loading.")
    items = {item.role: item for item in assets.artifacts}
    roles = ["bernini-high", "bernini-low", "clip", "vae"]
    if require_plan(data).lightx2v:
        roles.append("bernini-lightx2v")
    for role in roles:
        item = items.get(role)
        if (
            item is None
            or item.signature is None
            or tuple(file_signature(item.request.destination)) != item.signature
        ):
            raise ValueError("The prepared Bernini image components changed or are missing.")
        source_role = {
            "bernini-high": "high",
            "bernini-low": "low",
            "bernini-lightx2v": "lora",
        }.get(role)
        if source_role:
            row = sources()[source_role]
            if (
                item.request.sha256 != row["sha256"]
                or item.request.expected_size != row["bytes"]
                or item.request.url != row["url"]
            ):
                raise ValueError("The Bernini stage source differs from its signed image recipe.")
    if items["bernini-high"].signature != require_plan(data).high:
        raise ValueError("The prepared Bernini primary differs from its captured request.")
    items = {role: items[role] for role in roles}
    replay = require_plan(data).replay_components
    if replay and replay != tuple(tuple(row) for row in component_metadata(items)):
        raise ValueError("Select the original Bernini component files to replay this image.")
    return items


def component_metadata(items):
    return [
        [
            role,
            Path(item.signature[0]).name,
            item.signature[1],
            str(item.signature[2]),
            str(item.signature[3]),
        ]
        for role, item in items.items()
    ]


def load(owner, data):
    """Load only the primary stage here. Native memory management owns devices."""
    import comfy.sd
    import comfy.utils
    from ruinedfooocus.features.generation.pipeline_utils import load_diffusion_model
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.app.state import path_manager
    from ruinedfooocus.backends.comfy.embedding_paths import register_embedding_paths
    from ruinedfooocus.features.generation.image_model_lifecycle import (
        reset_image_models,
        collect_image_models,
    )

    plan, items = require_plan(data), _prepared(data)
    reset_image_models(owner)
    collect_image_models()
    with ExitStack() as stack:
        paths = {
            role: stack.enter_context(prepared_asset(data, role, item.name, path_manager, category))
            for role, item, category in (
                (role, items[role], category)
                for role, category in (
                    ("bernini-high", "checkpoints"),
                    ("clip", "clip"),
                    ("vae", "vae"),
                )
            )
        }
        require_header(paths["bernini-high"], "high")
        validate_component(paths["clip"], "clip")
        validate_component(paths["vae"], "vae")
        model = load_diffusion_model(str(paths["bernini-high"]))
        clip = comfy.sd.load_clip(
            ckpt_paths=[str(paths["clip"])],
            clip_type=comfy.sd.CLIPType.WAN,
            embedding_directory=register_embedding_paths(
                path_manager.model_paths["embedding_path"],
            ),
        )
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(paths["vae"]), safe_load=True))
    owner.xl_base = owner.StableDiffusionModel(model, vae, clip, None)
    owner.xl_base_patched = owner.xl_base
    owner.xl_base_patched_hash = None
    owner.xl_base_hash = list(plan.high)
    owner.load_error = None
    owner.conditions = None
    owner.model_info = dict(latent="BERNINI_IMAGE")
    owner._bernini_plan = plan
    owner._bernini_loras = ([], None)


def condition(plan, clip, vae, data):
    from nodes import CLIPTextEncode
    from comfy_extras.nodes_bernini import BerniniConditioning
    from ruinedfooocus.features.image_editing.image_geometry import (
        prepare_reference,
        reference_plan,
    )
    from ruinedfooocus.features.models.yaml_recipes import apply_clip_recipe
    from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder
    from ruinedfooocus.features.models.embedding_browser import native_embedding_text
    from ruinedfooocus.app.state import path_manager

    prompt = data.get("positive_prompt", data.get("prompt", ""))
    negative = data.get("negative_prompt", data.get("negative", ""))
    if not isinstance(prompt, str) or not isinstance(negative, str):
        raise ValueError("Bernini positive and negative prompts must be text.")
    clip = apply_clip_recipe(clip, plan.clip_skip, None)
    warm_patched_text_encoder(clip)

    def embeddings(text):
        return native_embedding_text(
            text,
            path_manager.model_paths["embedding_path"],
            path_manager.model_paths["cache_path"] / "embeddings",
        )

    prompt, negative = embeddings(prompt), embeddings(negative)
    header = plan.metadata()["prompt_header"]
    positive = CLIPTextEncode().encode(clip, header + prompt)[0]
    negative = CLIPTextEncode().encode(clip, negative)[0]
    target = (data["width"], data["height"])
    geometry = reference_plan(plan.source.size, target, plan.framing, multiple=16)
    if max(geometry["canvas_width"], geometry["canvas_height"]) > 8192:
        raise ValueError("Bernini's native image canvas supports at most 8192 pixels per side.")
    pixels, frame = prepare_reference(plan.source.image(), target, plan.framing, multiple=16)
    refs = {}
    import numpy as np
    import torch

    for index, ref in enumerate(plan.references):
        refs[f"reference_image_{index}"] = (
            torch.from_numpy(np.asarray(ref.image()).copy()).float().unsqueeze(0) / 255
        )
    positive, negative, latent = BerniniConditioning.execute(
        positive,
        negative,
        vae,
        frame["canvas_width"],
        frame["canvas_height"],
        length=1,
        batch_size=1,
        source_video=pixels,
        reference_images=refs,
        ref_max_size=plan.ref_max_size,
    )
    expected = (1, 16, 1, frame["canvas_height"] // 8, frame["canvas_width"] // 8)
    if tuple(latent["samples"].shape) != expected:
        raise ValueError("Bernini must create exactly one still-image latent.")
    return positive, negative, latent, frame


def sample_stages(
    plan, high, load_low, latent, positive, negative, seed, callback, check_cancelled
):
    """Pinned SplitSigmas topology: high noise once, then zero-noise low continuation."""
    import comfy.sample
    import comfy.utils
    import comfy.model_management as memory
    from comfy_extras.nodes_custom_sampler import BasicScheduler, SplitSigmas, KSamplerSelect

    sigmas = BasicScheduler.execute(high, plan.scheduler, plan.steps, 1.0)[0]
    high_sigmas, low_sigmas = SplitSigmas.execute(sigmas, plan.split)
    sampler = KSamplerSelect.execute(plan.sampler)[0]
    check_cancelled()
    noise = comfy.sample.prepare_noise(latent["samples"], seed)

    def stage(model, samples, noise, sigmas, offset):
        def progress(step, x0, x, count):
            check_cancelled()
            if callback is not None:
                callback(step + offset, x0, x, plan.steps, None)

        return comfy.sample.sample_custom(
            model,
            noise,
            plan.cfg,
            sampler,
            sigmas,
            positive,
            negative,
            samples,
            callback=progress,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=seed if offset == 0 else 0,
        )

    try:
        samples = stage(high, latent["samples"], noise, high_sigmas, 0)
        check_cancelled()
    finally:
        memory.unload_model_and_clones(high)
    low = load_low()
    try:
        check_cancelled()
        import torch

        samples = stage(low, samples, torch.zeros_like(samples), low_sigmas, plan.split)
        check_cancelled()
        if tuple(samples.shape) != tuple(latent["samples"].shape):
            raise ValueError("The Bernini sampler changed the single-image latent layout.")
        return dict(samples=samples)
    finally:
        memory.unload_model_and_clones(low)


def process(owner, data, callback, check_cancelled):
    import comfy.model_management as memory
    import comfy.sd
    import comfy.utils
    from ruinedfooocus.app.state import path_manager
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.features.generation.pipeline_utils import load_diffusion_model
    from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame
    from ruinedfooocus.features.generation.image_output_batches import ImageOutputBatch, PLAN_KEY
    from ruinedfooocus.features.image_processing.native_image_formats import pipeline_images

    plan, items = require_plan(data), _prepared(data)
    for key, value in (
        ("steps", plan.steps),
        ("custom_steps", plan.steps),
        ("cfg", plan.cfg),
        ("sampler_name", plan.sampler),
        ("scheduler", plan.scheduler),
        ("clip_skip", plan.clip_skip),
    ):
        if data.get(key, value) != value:
            raise ValueError("Bernini sampling settings changed after capture. Submit again.")
    if getattr(owner, "_bernini_plan", None) != plan:
        raise ValueError("The loaded Bernini model belongs to another captured image request.")
    with ExitStack() as stack:
        paths = {}
        for role, item in items.items():
            category = (
                "diffusion_models"
                if role == "bernini-low"
                else "loras"
                if role == "bernini-lightx2v"
                else "checkpoints"
                if role == "bernini-high"
                else role
            )
            paths[role] = stack.enter_context(
                prepared_asset(data, role, item.name, path_manager, category)
            )
        # Preserve every pending user LoRA read lease while the shared loader
        # applies it to the low stage, including multiple captured selections.
        from ruinedfooocus.features.generation.image_preparation import KEY as ASSETS_KEY

        for item in data[ASSETS_KEY].artifacts:
            if item.role == "loras":
                stack.enter_context(
                    prepared_asset(data, item.role, item.name, path_manager, "loras")
                )
        check_cancelled()
        positive, negative, latent, frame = condition(
            plan, owner.xl_base_patched.clip, owner.xl_base_patched.vae, data
        )
        memory.unload_model_and_clones(owner.xl_base_patched.clip.patcher)
        memory.unload_model_and_clones(owner.xl_base_patched.vae.patcher)
        high = owner.xl_base_patched.unet
        lora = (
            comfy.utils.load_torch_file(str(paths["bernini-lightx2v"]), safe_load=True)
            if plan.lightx2v
            else None
        )
        if lora is not None:
            high = comfy.sd.load_lora_for_models(high, None, lora, plan.high_strength, 0)[0]

        def load_low():
            require_header(paths["bernini-low"], "low")
            low = load_diffusion_model(str(paths["bernini-low"]))
            # Reuse the actual shared LoRA consumer, including its captured
            # signatures, mismatch rejection and native precision adapters.
            proxy = SimpleNamespace(
                StableDiffusionModel=owner.StableDiffusionModel,
                xl_base=owner.StableDiffusionModel(
                    low, owner.xl_base.vae, owner.xl_base.clip, None
                ),
                xl_base_patched_hash=None,
            )
            rows, captured = owner._bernini_loras
            type(owner).load_loras(proxy, rows, captured_loras=captured)
            low = proxy.xl_base_patched.unet
            if lora is not None:
                low = comfy.sd.load_lora_for_models(low, None, lora, plan.low_strength, 0)[0]
            return low

        samples = sample_stages(
            plan,
            high,
            load_low,
            latent,
            positive,
            negative,
            data["seed"],
            callback,
            check_cancelled,
        )
        check_cancelled()
        decoded = owner.decode_latent(
            samples,
            tiled=data.get("vae_tiled", False),
            tile_size=data.get("vae_tile_size"),
            overlap=data.get("vae_overlap"),
        )
        check_cancelled()
        _prepared(data)
    if tuple(decoded.shape) != (1, frame["canvas_height"], frame["canvas_width"], 3):
        raise ValueError("The Bernini image codec must return one opaque RGB still image.")
    decoded = restore_reference_frame(decoded, frame)
    data["width"], data["height"] = frame["width"], frame["height"]
    data["image_geometry"] = frame
    data["native_bernini_image"] = plan.metadata() | dict(components=component_metadata(items))
    data["bernini_source_frame"] = frame
    return ImageOutputBatch(tuple(pipeline_images(decoded, data)), data[PLAN_KEY])
