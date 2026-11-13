"""Exact official Marigold V2 albedo recipe, with four distinct local roles."""

from contextlib import ExitStack
from functools import lru_cache
from threading import RLock
import json

from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature
from ruinedfooocus.features.image_processing.depth_utilities import cached_model, native_call


def recipe():
    from ruinedfooocus.core.paths import resource_path

    path = resource_path("native_utility_recipes", "marigold-v2-albedo.json")
    return _recipe(str(path), tuple(file_signature(path)))


@lru_cache(maxsize=8)
def _recipe(path, signature):
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


_backbone_proofs = {}
_backbone_proofs_lock = RLock()


def _edit_source(row):
    from ruinedfooocus.features.models.model_identity import is_native_checkpoint_source

    return (
        row is not None
        and row.get("state") == "existing"
        and is_native_checkpoint_source(row)
        and row.get("native_class") == "QwenImage"
        and row.get("native_family") == "QwenImage"
        and row.get("behavioral_variant") == "edit-2509"
    )


def _row_proof(row):
    return {
        "origin": "reviewed_registry",
        "artifact": row["id"],
        "native_family": row["native_family"],
        "behavioral_variant": row["behavioral_variant"],
        "bytes": row["artifact"]["bytes"],
        "sha256": row["artifact"]["sha256"],
        "pinned_url": row["artifact"]["pinned_url"],
    }


def _recipe_proof():
    facts = recipe()
    base = facts["components"]["image_utility_model"]
    return {
        "origin": "reviewed_marigold_recipe",
        "artifact": facts["id"],
        "native_family": base["native_family"],
        "behavioral_variant": base["behavioral_variant"],
        "bytes": base["bytes"],
        "sha256": base["sha256"],
        "source_repository": facts["source_repository"],
        "source_revision": facts["source_revision"],
        "source_filename": base["filename"],
    }


def _proof_current(proof, registry):
    if proof.get("origin") == "reviewed_marigold_recipe":
        return (
            proof == _recipe_proof()
            and proof.get("native_family") == "QwenImage"
            and proof.get("behavioral_variant") == "edit-2509"
        )
    row = registry.artifact(proof.get("artifact"))
    return _edit_source(row) and proof == _row_proof(row)


def backbone_proof(path, *, verify_hash=False):
    """Positive Edit2509 provenance; passive calls never hash model weights."""
    from ruinedfooocus.core.paths import data_path
    from ruinedfooocus.features.models.model_identity import (
        cached_file_sha256,
        read_prepared_native_source,
        verified_native_source,
    )
    from ruinedfooocus.features.models.model_registry import load_registry

    signature = tuple(file_signature(path))
    registry = load_registry()
    with _backbone_proofs_lock:
        cached = dict(_backbone_proofs.get(signature, ()))
    if cached and _proof_current(cached, registry):
        return cached
    metadata = read_prepared_native_source(path, data_path("cache"))
    row = verified_native_source(path, metadata, current_signature=list(signature))
    proof = _row_proof(row) if _edit_source(row) else None
    if proof is None and verify_hash:
        candidates = [
            row
            for row in registry.entries(role="image_diffusion_checkpoint")
            if _edit_source(row) and row["artifact"]["bytes"] == signature[1]
        ]
        official = _recipe_proof()
        if candidates or official["bytes"] == signature[1]:
            checksum = cached_file_sha256(path, data_path("cache")).lower()
            proof = next(
                (_row_proof(row) for row in candidates if row["artifact"]["sha256"] == checksum),
                None,
            )
            if (
                proof is None
                and official["bytes"] == signature[1]
                and official["sha256"] == checksum
                and _proof_current(official, registry)
            ):
                proof = official
    if tuple(file_signature(path)) != signature:
        raise ValueError(
            "The Marigold backbone changed during source verification. Select it again."
        )
    if proof is not None:
        with _backbone_proofs_lock:
            if len(_backbone_proofs) >= 128:
                _backbone_proofs.pop(next(iter(_backbone_proofs)))
            _backbone_proofs[signature] = tuple(proof.items())
    return proof


def component_label(field, path, name, *, locale=None):
    from ruinedfooocus.core.translation_manager import localize

    if field != "image_utility_model":
        return name
    status = (
        localize("Edit2509 source verified", locale)
        if backbone_proof(path)
        else localize("Edit2509 source unverified", locale)
    )
    return f"{name} ({status})"


def verify_captured_request(request):
    from ruinedfooocus.features.models.model_registry import load_registry

    item = next(item for item in request.files if item.field == "image_utility_model")
    proof = dict(item.provenance)
    if (
        tuple(file_signature(item.path)) != item.signature
        or not proof
        or proof.get("bytes") != item.signature[1]
        or not _proof_current(proof, load_registry())
    ):
        raise ValueError(
            "The captured Marigold Edit2509 source proof changed. Select and submit the backbone again."
        )


def verify_component(field, path):
    """Bind indistinguishable albedo/depth/normal tensor layouts to published roles."""
    from ruinedfooocus.core.paths import data_path
    from ruinedfooocus.features.models.model_identity import cached_file_sha256

    if field == "image_utility_model":
        proof = backbone_proof(path, verify_hash=True)
        if proof is None:
            raise ValueError(
                "Marigold albedo needs verified Edit2509 backbone provenance. Choose a reviewed Edit2509 encoding; a QwenImage architecture or filename alone is not proof."
            )
        return proof
    expected = recipe()["components"][field]
    if (
        path.stat().st_size != expected["bytes"]
        or cached_file_sha256(path, data_path("cache")).lower() != expected["sha256"]
    ):
        raise ValueError(
            "Select the source-verified component from the official Marigold V2 albedo recipe."
        )


def validate_model(path):
    from ruinedfooocus.features.models.model_architecture import inspect_model_file

    if inspect_model_file(path).get("native_class") != "QwenImage":
        raise ValueError("Select the Qwen Image Edit model from the Marigold V2 albedo recipe.")


def validate_vae(path):
    header = cached_safetensors_header(path)
    if not all(
        any(key.startswith(prefix) for key in header) for prefix in ("encoder.", "decoder.")
    ):
        raise ValueError("Select the dedicated Marigold V2 albedo VAE.")


def validate_lora(path):
    header = cached_safetensors_header(path)
    if not any(key.endswith((".lora_A.weight", ".lora_down.weight")) for key in header) or not any(
        key.endswith((".lora_B.weight", ".lora_up.weight")) for key in header
    ):
        raise ValueError("Select the dedicated Marigold V2 albedo LoRA.")


def validate_conditioning(path):
    header = cached_safetensors_header(path)
    entry = header.get("conditioning", {})
    shape = entry.get("shape", [])
    if (
        entry.get("dtype") not in ("F16", "BF16", "F32")
        or len(shape) != 3
        or shape[0] != 1
        or not 1 <= shape[1] <= 32768
        or shape[2] != 3584
    ):
        raise ValueError("Select the saved Qwen conditioning from the Marigold V2 albedo recipe.")
    metadata = header.get("__metadata__", {})
    options = metadata.get("conditioning_options", "{}")
    if not isinstance(options, str) or len(options) > 16384:
        raise ValueError("The saved albedo conditioning options are invalid.")
    try:
        options = json.loads(options)
    except (ValueError, RecursionError) as error:
        raise ValueError("The saved albedo conditioning options are invalid.") from error
    if not isinstance(options, dict):
        raise ValueError("The saved albedo conditioning options must be a mapping.")


COMPONENTS = (
    (
        "image_utility_model",
        "marigold-qwen-edit",
        ("diffusion_models", "checkpoints"),
        validate_model,
    ),
    ("image_utility_vae", "marigold-albedo-vae", ("vae",), validate_vae),
    ("image_utility_lora", "marigold-albedo-lora", ("loras",), validate_lora),
    (
        "image_utility_conditioning",
        "marigold-albedo-conditioning",
        ("embeddings",),
        validate_conditioning,
    ),
)


def parameters(data):
    return {}


def render(owner, request, pixels, data, manager, check_cancelled):
    import torch.nn.functional as F
    import comfy.sd
    import folder_paths
    from comfy.model_management import unload_model_and_clones
    from nodes import VAELoader, LoraLoaderModelOnly, VAEEncode, VAEDecode
    from comfy_extras.nodes_cond import ConditioningLoader
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    from comfy_extras.nodes_custom_sampler import (
        DisableNoise,
        BasicGuider,
        KSamplerSelect,
        ManualSigmas,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_marigold import MarigoldV2PostProcess

    files = {item.field: item for item in request.files}

    def load():
        # Cleanup partial loads on a rejected component or cancellation. Native
        # loaders/patchers own device placement, LoRA conversion and memory.
        with ExitStack() as cleanup:
            base = comfy.sd.load_diffusion_model(str(files["image_utility_model"].path))
            if base is None:
                raise ValueError("The native loader could not initialize the Marigold Qwen model.")
            cleanup.callback(unload_model_and_clones, base)
            check_cancelled()
            if type(base.model.model_config).__name__ != "QwenImage":
                raise ValueError("The native albedo model must use the Qwen Image architecture.")
            for field in ("image_utility_vae", "image_utility_lora", "image_utility_conditioning"):
                item = files[field]
                folder_paths.add_model_folder_path(
                    item.category, str(item.path.parent), is_default=True
                )
            vae = VAELoader().load_vae(files["image_utility_vae"].path.name)[0]
            cleanup.callback(unload_model_and_clones, vae.patcher)
            check_cancelled()
            vae.throw_exception_if_invalid()
            if (
                vae.latent_channels != 16
                or vae.spacial_compression_encode() != 8
                or vae.spacial_compression_decode() != 8
                or vae.output_channels != 3
            ):
                raise ValueError(
                    "The native albedo VAE must use the Qwen 16-channel eightfold latent format."
                )
            model = native_call(
                check_cancelled,
                LoraLoaderModelOnly().load_lora_model_only,
                base,
                files["image_utility_lora"].path.name,
                1.0,
            )[0]
            model = native_call(
                check_cancelled,
                ModelSamplingAuraFlow().patch_aura,
                model,
                shift=1.73,
                sampling="img_to_img_velocity",
            )[0]
            conditioning = native_call(
                check_cancelled,
                ConditioningLoader.execute,
                files["image_utility_conditioning"].path.name,
            )[0]
            cleanup.pop_all()
        return (model, vae, conditioning), (base, vae.patcher)

    model, vae, conditioning = cached_model(owner, request, load, check_cancelled)
    width, height = request.size
    # Native Qwen patchification needs an even latent grid: pad to 16, then crop.
    padded = F.pad(
        pixels.movedim(-1, 1), (0, -width % 16, 0, -height % 16), mode="replicate"
    ).movedim(1, -1)
    latent = native_call(check_cancelled, VAEEncode().encode, vae, padded)[0]
    guider = native_call(check_cancelled, BasicGuider.execute, model, conditioning)[0]
    noise = native_call(check_cancelled, DisableNoise.execute)[0]
    sampler = native_call(check_cancelled, KSamplerSelect.execute, "euler")[0]
    sigmas = native_call(check_cancelled, ManualSigmas.execute, "0.5, 0")[0]
    output = native_call(
        check_cancelled, SamplerCustomAdvanced.execute, noise, guider, sampler, sigmas, latent
    )[0]
    decoded = native_call(check_cancelled, VAEDecode().decode, vae, output)[0]
    albedo = native_call(
        check_cancelled, MarigoldV2PostProcess.execute, decoded, prediction="albedo"
    )[0]
    return albedo[:, :height, :width, :], {
        "native": "MarigoldV2",
        "native_output": "IMAGE",
        "result": "srgb-albedo",
        "recipe": recipe()["id"],
        "verified_backbone": dict(files["image_utility_model"].provenance),
        "verified_task_components": {
            entry["role"]: entry["sha256"]
            for field, entry in recipe()["components"].items()
            if field != "image_utility_model"
        },
        "padding": "replicate-to-16-crop",
    }
