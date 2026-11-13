import numpy as np
import torch
import traceback

import ruinedfooocus.features.image_editing.controlnet as control_tools
import ruinedfooocus.app.async_worker as worker
import ruinedfooocus.features.prompting.prompt_processing as pp

from PIL import Image, ImageOps
from ruinedfooocus.features.models.model_identity import file_signature, files_unchanged
from ruinedfooocus.backends.comfy.embedding_paths import register_embedding_paths
from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder
from ruinedfooocus.backends.comfy.model_paths import register_image_model_paths
from ruinedfooocus.features.generation.image_preparation import (
    component_read_leases,
    component_file,
)
from ruinedfooocus.features.generation.cascade_guidance import stage_read_leases

# Known models (from ComfyUI/comfy/model_base.py)
from comfy.model_base import Flux, SDXL
from ruinedfooocus.app.state import path_manager, settings
import ruinedfooocus.app.state as shared
from pathlib import Path
import random

register_image_model_paths(path_manager)

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config

from nodes import (
    CLIPTextEncode,
    CLIPSetLastLayer,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    VAEEncode,
    VAEEncodeForInpaint,
)

from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_flux import FluxKontextImageScale, EmptyFlux2LatentImage
from comfy_extras.nodes_hunyuan import EmptyHunyuanImageLatent
from comfy_extras.nodes_hidream_o1 import EmptyHiDreamO1LatentImage, HiDreamO1ReferenceImages
from comfy_extras.nodes_chroma_radiance import EmptyChromaRadianceLatentImage
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
from comfy_extras.nodes_mage import TextEncodeMageFlowEdit
from comfy_extras.nodes_fibo2 import TextEncodeFibo2
from comfy_extras.nodes_moss_image import EDIT_FORMATS, TextEncodeMOSSImageEdit
from ruinedfooocus.features.image_editing.image_guidance import (
    resolve_guidance,
    request_model,
    effective_guidance,
    normalize_reference_images,
    reference_image_map,
)
from node_helpers import conditioning_set_values

from comfy.sample import fix_empty_latent_channels
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny
from comfy.sd import CLIP, VAE
from ruinedfooocus.backends.comfy.sequence_prefill import install_qwen35_prefill

install_qwen35_prefill()

from ruinedfooocus.features.generation.pipeline_utils import (
    get_previewer,
    clean_prompt_cond_caches,
    load_diffusion_model,
)
from ruinedfooocus.features.image_editing.canny_utils import sanitize_canny_thresholds
from ruinedfooocus.features.image_editing.image_geometry import (
    prepare_reference,
    restore_reference_frame,
)
from ruinedfooocus.features.image_editing import krea_style, lora_hooks


def reference_tensor(image, width=None, height=None):
    pixels = torch.from_numpy(np.array(image.convert("RGB")).astype(np.float32) / 255.0)[None,]
    if width is not None:
        pixels = ImageScaleToTotalPixels().execute(
            image=pixels,
            upscale_method="bicubic",
            megapixels=float(width) * float(height) / (1024 * 1024),
            resolution_steps=1,
        )[0]
    return pixels


def joy_reference_conditioning(clip, vae, source, positive, negative):
    from comfy_extras.nodes_joyimage import TextEncodeJoyImageEdit

    if vae is None:
        raise ValueError("JoyAI Image Edit requires its Wan 2.1 VAE for reference conditioning.")
    images = {"image0": source}
    return (
        TextEncodeJoyImageEdit.execute(clip, positive, vae=vae, images=images)[0],
        TextEncodeJoyImageEdit.execute(clip, negative, vae=vae, images=images)[0],
    )


class pipeline:
    pipeline_type = ["sdxl", "ssd", "sd3", "flux", "flux2", "lumina2"]

    class StableDiffusionModel:
        def __init__(self, unet, vae, clip, clip_vision):
            self.unet = unet
            self.vae = vae
            self.clip = clip
            self.clip_vision = clip_vision

    # Add new model setups here.
    # Simple workflows with "Load models"->"Sample"->"VAE" might work right away.
    # Add new clip/vae models to settings and pathdb-files.
    known_model_info = {
        "BerniniImage": {"latent": "BERNINI_IMAGE", "clip_type": comfy.sd.CLIPType.WAN},
        "MiniMaxH3Image": {"latent": "MINIMAX_IMAGE", "clip_type": comfy.sd.CLIPType.MINIMAX},
        "SD20": {"clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION},
        "PiD": {"latent": "ChromaRadience", "clip_type": comfy.sd.CLIPType.PIXELDIT},
        "HunyuanImage21Refiner": {"clip_type": comfy.sd.CLIPType.HUNYUAN_IMAGE},
        "StableCascade_C": {"latent": "CASCADE_C", "clip_type": comfy.sd.CLIPType.STABLE_CASCADE},
        "SD21UNCLIP": {"clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION},
        "HunyuanImage21": {"latent": "HunyuanImage", "clip_type": comfy.sd.CLIPType.HUNYUAN_IMAGE},
        "QwenImage21": {"latent": "QwenImage21", "clip_type": comfy.sd.CLIPType.QWEN_IMAGE},
        "SenseNovaU15": {
            "latent": "ChromaRadience",
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "Anima": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.COSMOS,
            "shift_setting": "anima_image_shift",
        },
        "AuraFlow": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
            "shift_setting": "auraflow_shift",
        },
        "BaseModel": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "Boogu": {
            "clip_type": comfy.sd.CLIPType.BOOGU,
        },
        "Chroma": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.CHROMA,
        },
        "ChromaRadiance": {
            "latent": "ChromaRadience",
            "clip_type": comfy.sd.CLIPType.CHROMA,
        },
        "CosmosPredict2": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.COSMOS,
        },
        "ErnieImage": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Fibo2": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FIBO2,
            "options": {"has_image_edit": True},
        },
        "Flux": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.FLUX,
        },
        "Flux2": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Flux2Klein4B": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Flux2Klein9B": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX2,
        },
        "Flux3": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.FLUX3,
        },
        "HiDream": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.HIDREAM,
            "shift_setting": "hidream_shift",
        },
        "HiDreamO1": {
            "latent": "HiDreamO1",
            "clip_type": comfy.sd.CLIPType.HIDREAM,
        },
        "HunyuanDiT": {
            "clip_type": comfy.sd.CLIPType.HUNYUAN_DIT,
        },
        "Ideogram4": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.IDEOGRAM4,
        },
        "JoyImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.JOYIMAGE,
        },
        "Kandinsky5Image": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.KANDINSKY5_IMAGE,
        },
        "Krea2": {
            "clip_type": comfy.sd.CLIPType.KREA2,
        },
        "Lens": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.LENS,
        },
        "Lumina2": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.LUMINA2,
            "shift_setting": "lumina2_shift",
        },
        "LongCatImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.LONGCAT_IMAGE,
        },
        "MageFlow": {
            "latent": "MAGE",
            "clip_type": comfy.sd.CLIPType.MAGE,
            "options": {"has_image_edit": True},
        },
        "MOSSImage": {
            "latent": "FLUX2",
            "clip_type": comfy.sd.CLIPType.MOSS_IMAGE,
        },
        "MingImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "NewBieImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.NEWBIE,
            "shift_setting": "newbieimage_shift",
        },
        "Omnigen2": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.OMNIGEN2,
        },
        "PixArt": {
            "clip_type": comfy.sd.CLIPType.PIXART,
        },
        "PixelDiTT2I": {
            "latent": "ChromaRadience",
            "clip_type": comfy.sd.CLIPType.PIXELDIT,
        },
        "QwenImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.QWEN_IMAGE,
            "shift_setting": "qwen_image_shift",
            "options": {"has_image_edit": True},
        },
        "SD3": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.SD3,
            "shift_setting": "sd3_shift",
        },
        "SDXL": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "SDXLRefiner": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "SD15_instructpix2pix": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "SDXL_instructpix2pix": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "SD_X4Upscaler": {
            "clip_type": comfy.sd.CLIPType.STABLE_DIFFUSION,
        },
        "ZImage": {
            "latent": "SD3",
            "clip_type": comfy.sd.CLIPType.LUMINA2,
            "shift_setting": "lumina2_shift",
        },
        "ZImagePixelSpace": {
            "latent": "ChromaRadience",
            "clip_type": comfy.sd.CLIPType.LUMINA2,
        },
    }
    known_models = known_model_info.keys()

    xl_base: StableDiffusionModel = None
    xl_base_hash = None

    xl_base_patched: StableDiffusionModel = None
    xl_base_patched_hash = None

    xl_controlnet: StableDiffusionModel = None
    xl_controlnet_hash = None

    model_info = None
    # Why the last checkpoint load failed, shown to the user instead of a generic error.
    load_error = None

    def get_clip_and_vae(self, unet, component_overrides=None, component_layout=None):
        from ruinedfooocus.features.models.resource_defaults import resource_name
        from ruinedfooocus.backends.comfy.component_selection import component_settings

        resources = component_settings(settings.default_settings, component_overrides)

        unet_type = unet.model.__class__.__name__
        config = unet.model.model_config.unet_config
        if unet_type == "BaseModel" and type(unet.model.model_config).__name__ == "SD20":
            unet_type = "SD20"
        if unet_type == "HunyuanImage21Refiner" and (
            config.get("context_in_dim") != 3584 or config.get("byt5") is not False
        ):
            raise ValueError("This Hunyuan refiner needs a different native text-encoder contract.")
        # ComfyUI already classified these variants before constructing the model.
        if unet_type == "Lumina2":
            if config.get("dim") == 3840:
                unet_type = "ZImage"
            elif "clip_text_dim" in config:
                unet_type = "NewBieImage"
        elif unet_type == "Flux2":
            unet_type = {20: "Flux2Klein4B", 24: "Flux2Klein9B"}.get(
                config.get("depth_single_blocks"), "Flux2"
            )

        if unet_type not in self.known_model_info.keys():
            raise RuntimeError(
                f"No default text encoder/VAE configuration for backend model {unet_type}. "
                "This architecture needs an explicit supported component configuration."
            )

        info = self.known_model_info[unet_type].copy()
        shift = settings.default_settings.get(info.pop("shift_setting", None))
        info["sampling_shift"] = float(shift) if shift not in (None, "") else None
        from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS

        info.update(IMAGE_COMPONENTS[unet_type])
        if component_layout is not None:
            if component_layout.family != unet_type:
                raise ValueError("The loaded model differs from its captured component recipe.")
            info.update(
                clip_names=list(component_layout.clip_roles), vae_name=component_layout.vae_role
            )
            if component_layout.vae_role == "vae_qwen_image_layered":
                info["latent"] = "QWEN_LAYERED"
        info["clip_keys"] = list(info["clip_names"])
        info["vae_key"] = info["vae_name"]
        info["clip_names"] = [resource_name(key, resources) for key in info["clip_names"]]
        if info["vae_name"] != "pixel_space":
            info["vae_name"] = resource_name(info["vae_name"], resources)
        return info

    @component_read_leases
    def load_base_model(
        self,
        name,
        unet_only=False,
        input_unet=None,
        hash=None,
        input_components=None,
        component_overrides=None,
        prepared_data=None,
    ):
        from ruinedfooocus.features.models.image_scope import (
            require_image_operation,
            require_image_checkpoint,
        )

        require_image_operation({"base_model_name": name})
        from ruinedfooocus.backends.comfy.text_encoder_loader import (
            has_gguf_encoder,
            load_mixed_clip,
        )
        from ruinedfooocus.backends.comfy.component_selection import (
            component_settings,
            normalize_component_overrides,
            validate_component_slots,
        )
        from ruinedfooocus.core.model_formats import require_image_vae_format
        from ruinedfooocus.features.models.yaml_recipes import (
            captured_recipe,
            recipe_identity,
            apply_model_recipe,
        )

        yaml_identity = recipe_identity(prepared_data)
        from ruinedfooocus.features.models import gguf_projectors

        projectors = gguf_projectors.captured(prepared_data)
        previous_projectors = getattr(self, "base_gguf_projectors", None)

        embedded = input_components or {}
        choices = normalize_component_overrides(component_overrides)
        from ruinedfooocus.features.models.approximate_codecs import captured_pair

        approximate = captured_pair(prepared_data, choices)
        approximate_identity = approximate.identity if approximate is not None else None
        components = component_settings(settings.default_settings, choices)
        filename = shared.models.get_model_path("checkpoints", name, hash=hash, default=None)
        from ruinedfooocus.features.generation import bernini_image

        self._bernini_plan = None
        self._bernini_loras = None
        if filename is not None and bernini_image.primary_family(filename):
            if input_unet is not None or input_components:
                raise ValueError("Bernini-R requires its captured signed image components.")
            bernini_image.require_plan(prepared_data, filename)
            return bernini_image.load(self, prepared_data)
        if filename is not None:
            require_image_checkpoint(filename)
            from ruinedfooocus.features.models.native_pixel_sources import (
                require_native_pixel_source,
            )

            require_native_pixel_source(filename)
            from ruinedfooocus.features.models.native_workflow_admission import (
                require_available_image_workflow,
            )

            require_available_image_workflow(filename, prepared_data=prepared_data, loading=True)
        recipe = captured_recipe(prepared_data)
        if recipe is not None and (
            filename is None or recipe.checkpoint.path != Path(filename).resolve()
        ):
            raise ValueError(
                "The captured YAML recipe belongs to another checkpoint. Submit it again."
            )
        from ruinedfooocus.features.models.local_diffusers import (
            BUNDLE_KEY,
            LocalDiffusersBundle,
            is_diffusers_index,
        )
        from ruinedfooocus.features.models.sensenova_bundle import (
            KEY as SENSE_KEY,
            SenseNovaBundle,
            is_sensenova_index,
        )

        bundle = None
        sense_bundle = None
        if filename is not None and is_sensenova_index(filename):
            sense_bundle = (prepared_data or {}).get(SENSE_KEY)
            if not isinstance(sense_bundle, SenseNovaBundle):
                raise ValueError("Capture the SenseNova shard bundle before native loading.")
            if sense_bundle.index.source.resolve() != Path(filename).resolve():
                raise ValueError("The captured SenseNova bundle does not match this selection.")
            if input_unet is not None or embedded or any(choices.values()):
                raise ValueError("SenseNova bundle sources and native components stay fixed.")
            sense_bundle.verify()
        if filename is not None and is_diffusers_index(filename):
            bundle = (prepared_data or {}).get(BUNDLE_KEY)
            if not isinstance(bundle, LocalDiffusersBundle):
                raise ValueError("Capture the local Diffusers bundle before native loading.")
            if bundle.index.source.resolve() != Path(filename).resolve():
                raise ValueError("The captured Diffusers bundle does not match this selection.")
            if input_unet is not None or embedded or any(choices.values()):
                raise ValueError("Local Diffusers bundle components are fixed; clear replacements.")
            bundle.verify()
        bundle_identity = (
            bundle.identity
            if bundle is not None
            else sense_bundle.identity
            if sense_bundle is not None
            else None
        )
        signature = file_signature(filename) if filename is not None else None
        from ruinedfooocus.features.models.native_component_layout import (
            KEY as LAYOUT_KEY,
            NativeComponentLayout,
            current_layout,
        )

        layout = (prepared_data or {}).get(LAYOUT_KEY)
        if layout is None and filename is not None:
            layout = current_layout(filename, path_manager)
        if layout is not None:
            if not isinstance(layout, NativeComponentLayout):
                raise ValueError("Capture a typed native component recipe before loading.")
            layout.verify(filename)
        if (
            self.xl_base_hash is not None
            and self.xl_base_hash in (name, hash)
            and approximate_identity == getattr(self, "base_approximate_codec_identity", None)
            and signature is not None
            and signature == getattr(self, "base_file_signature", None)
            and components == getattr(self, "component_settings", None)
            and choices == getattr(self, "component_overrides", None)
            and bundle_identity == getattr(self, "base_bundle_identity", None)
            and yaml_identity == getattr(self, "base_yaml_recipe_identity", None)
            and layout == getattr(self, "base_component_layout", None)
            and (
                gguf_projectors.unchanged(previous_projectors)
                if projectors is None
                else projectors == previous_projectors
            )
            and files_unchanged(getattr(self, "component_file_signatures", None))
        ):
            return
        self.component_settings = components
        self.component_overrides = choices
        self.yaml_recipe = captured_recipe(prepared_data)
        component_files = []

        def selected_component(role, slot, filename, index=None):
            if bundle is not None:
                item = bundle.vae if role == "vae" else bundle.encoders[index]
                role, slot, filename = "bundle", item.role, str(item.path)
            return component_file(prepared_data, role, slot, filename, path_manager)

        from ruinedfooocus.features.generation.image_model_lifecycle import reset_image_models

        reset_image_models(self)

        if filename is None:
            print(f"ERROR: Could not load checkpoint {name}")
            self.load_error = (
                f"Checkpoint not found: {name}. It may have been moved or renamed; "
                "choose another checkpoint in the Models tab."
            )
            return

        if input_unet is None:  # Be quiet if we already loaded a unet
            print(f"Loading base {'unet' if unet_only else 'model'}: {name}")

        from ruinedfooocus.features.generation.image_model_lifecycle import collect_image_models

        collect_image_models(comfy.model_management)

        unet = None

        filename = str(filename)
        embedding_paths = register_embedding_paths(path_manager.model_paths["embedding_path"])
        if (
            sense_bundle is not None
            or bundle is not None
            or Path(filename).suffix.lower() == ".gguf"
            or unet_only
        ):
            with torch.inference_mode():
                try:
                    diffusion_path = filename
                    if bundle is not None:
                        for item in (bundle.index, bundle.diffusion):
                            path = component_file(
                                prepared_data, "bundle", item.role, str(item.path), path_manager
                            )
                            component_files.append(file_signature(path))
                        diffusion_path = str(bundle.diffusion.path)
                    if sense_bundle is not None:
                        component_files.extend(list(item.signature) for item in sense_bundle.files)
                        unet = load_diffusion_model(diffusion_path, sensenova_bundle=sense_bundle)
                    else:
                        unet = load_diffusion_model(diffusion_path, input_unet)

                    # Get text-encoders (clip) and vae to match the unet
                    model_info = self.get_clip_and_vae(unet, choices, layout)
                    if bundle is not None:
                        if len(model_info["clip_names"]) != len(bundle.encoders):
                            raise ValueError(
                                "Bundle encoder count does not match the native image model."
                            )
                        if model_info["vae_name"] == "pixel_space":
                            raise ValueError(
                                "This native model does not use the bundle VAE contract."
                            )
                        model_info = dict(
                            model_info,
                            clip_names=[item.source.name for item in bundle.encoders],
                            vae_name=bundle.vae.source.name,
                        )
                    captured_pair(prepared_data, choices, model_info["vae_key"])
                    validate_component_slots(
                        choices,
                        model_info,
                        bundled_clip="clip" in embedded,
                        bundled_vae="vae" in embedded,
                    )
                    if "vae" not in embedded and model_info["vae_name"] != "pixel_space":
                        require_image_vae_format(model_info["vae_name"])
                    self.model_info = model_info

                    # Load everything...
                    clip_paths = []
                    for index, clip_name in enumerate(
                        [] if "clip" in embedded else model_info["clip_names"]
                    ):
                        clip_paths.append(
                            str(
                                selected_component(
                                    "clip",
                                    model_info["clip_keys"][index],
                                    clip_name,
                                    index,
                                )
                            )
                        )

                    missing = [Path(path).name for path in clip_paths if not Path(path).is_file()]
                    if missing:
                        raise ValueError(
                            f"{name} needs text encoder file(s) that are not installed: "
                            f"{', '.join(missing)}. Download them in Models > Model > Image model components."
                        )
                    print(f"Loading CLIP: {model_info['clip_names']}")
                    component_files.extend(file_signature(path) for path in clip_paths)
                    if "clip" in embedded:
                        clip = embedded["clip"]
                    elif type(unet.model).__name__ == "SenseNovaU15":
                        from ruinedfooocus.backends.comfy.sensenova_clip import (
                            load_sensenova_clip,
                            _Cancellation,
                        )
                        from ruinedfooocus.core.preparation import preparation_service

                        # The native sentinel has no encoder file. Its reload factory
                        # remains bound to the exact diffusion source, including delegates.
                        if sense_bundle is not None:
                            clip = load_sensenova_clip(
                                unet.model.model_config, sense_bundle, embedding_paths
                            )
                        else:
                            with preparation_service.artifact_lease(
                                Path(filename), cancelled=_Cancellation()
                            ):
                                clip = load_sensenova_clip(
                                    unet.model.model_config, signature, embedding_paths
                                )
                    elif (
                        isinstance(unet.model, comfy.model_base.HunyuanDiT)
                        or bundle is not None
                        or not has_gguf_encoder(clip_paths)
                    ):
                        model_options = {}
                        device = comfy.model_management.get_torch_device()
                        if device.type == "cpu":
                            model_options["load_device"] = model_options["offload_device"] = (
                                torch.device("cpu")
                            )
                        if isinstance(unet.model, comfy.model_base.HunyuanDiT):
                            from ruinedfooocus.backends.comfy.hunyuan_dit import (
                                load_hunyuan_dit_clip,
                            )

                            clip = load_hunyuan_dit_clip(
                                clip_paths,
                                unet.model.model_config.clip_target(),
                                embedding_directory=embedding_paths,
                                model_options=model_options,
                                signatures=component_files[-len(clip_paths) :],
                            )
                        else:
                            clip = comfy.sd.load_clip(
                                ckpt_paths=clip_paths,
                                embedding_directory=embedding_paths,
                                clip_type=model_info["clip_type"],
                                model_options=model_options,
                            )
                    else:
                        projectors = gguf_projectors.for_paths(clip_paths, projectors)
                        clip = load_mixed_clip(
                            clip_paths, model_info["clip_type"], projectors=projectors
                        )
                        component_files.extend(
                            list(item.projector)
                            for item in projectors
                            if item.projector is not None
                        )

                    if approximate is not None:
                        from ruinedfooocus.backends.comfy.approximate_codecs import load_paired_vae

                        vae = load_paired_vae(approximate)
                        component_files.extend(list(item.signature) for item in approximate.files)
                    elif "vae" in embedded:
                        vae = embedded["vae"]
                    else:
                        metadata = None
                        if model_info["vae_name"] == "pixel_space":
                            sd = {"pixel_space_vae": torch.tensor(1.0)}
                        else:
                            vae_path = selected_component(
                                "vae",
                                model_info["vae_key"],
                                model_info["vae_name"],
                            )
                            require_image_vae_format(vae_path)
                            if not Path(vae_path).is_file():
                                raise ValueError(
                                    f"{name} needs the VAE file {Path(vae_path).name}, which is not "
                                    "installed. Download it in Models > Model > Image model components."
                                )
                            print(f"Loading VAE: {model_info['vae_name']}")
                            component_files.append(file_signature(vae_path))
                            sd, metadata = comfy.utils.load_torch_file(
                                str(vae_path), return_metadata=True
                            )
                        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                        if model_info["vae_name"] != "pixel_space":
                            vae.patcher.cached_patcher_init = (
                                comfy.sd.load_vae_patcher,
                                (str(vae_path), metadata, None),
                            )
                    vae.throw_exception_if_invalid()
                    if bundle is not None:
                        bundle.verify()

                    clip_vision = embedded.get("clip_vision")
                except Exception as e:
                    unet = None
                    traceback.print_exc()
                    from ruinedfooocus.core.task_errors import describe_error

                    self.load_error = (
                        describe_error(e)
                        if isinstance(e, ValueError)
                        else f"Could not load {name}. {describe_error(e)}"
                    )

        else:
            # The file loader preserves quantization metadata and registers the
            # reload factories required by dynamic patchers and their delegates.
            unet, clip, vae, clip_vision = load_checkpoint_guess_config(
                filename,
                embedding_directory=embedding_paths,
                output_clipvision=True,
                output_vae=approximate is None,
            )
            valid_vae = (
                approximate is None and isinstance(vae, VAE) and vae.first_stage_model is not None
            )
            if not isinstance(clip, CLIP) or not valid_vae:
                embedded = {}
                if isinstance(clip, CLIP):
                    embedded["clip"] = clip
                if valid_vae:
                    embedded["vae"] = vae
                if clip_vision is not None:
                    embedded["clip_vision"] = clip_vision
                self.load_base_model(
                    name,
                    hash=hash,
                    unet_only=True,
                    input_unet=unet,
                    input_components=embedded,
                    component_overrides=choices,
                    prepared_data=prepared_data,
                )
                return
            self.model_info = self.get_clip_and_vae(unet, choices, layout)
            validate_component_slots(
                choices,
                self.model_info,
                bundled_clip=True,
                bundled_vae=True,
            )
            vae.throw_exception_if_invalid()

        if unet == None:
            print(f"Failed to load {name}")
            self.load_error = self.load_error or (
                f"Could not load {name}: its model type was not recognized. "
                "Check that it is an image checkpoint or diffusion model file."
            )
            self.xl_base = None
            self.xl_base_hash = None
            self.xl_base_patched = None
            self.xl_base_patched_hash = None
        else:
            from ruinedfooocus.backends.comfy.model_attention import configure_qwen_attention

            unet = apply_model_recipe(unet, self.yaml_recipe)
            configure_qwen_attention(unet)
            self.xl_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (self.xl_base.unet.model.__class__.__name__ in self.known_models):
                print(
                    f"Model {self.xl_base.unet.model.__class__.__name__} not supported. RuinedFooocus supports {list(self.known_models)} models as the base model."
                )
                self.load_error = (
                    f"{name} is a {self.xl_base.unet.model.__class__.__name__} model, "
                    "which is not supported as an image model."
                )
                self.xl_base = None

            if self.xl_base is not None:
                self.xl_base_patched = self.xl_base
                self.xl_base_patched_hash = None

        if self.xl_base is None:
            return
        if self.model_info is None:
            self.model_info = self.get_clip_and_vae(self.xl_base.unet, choices, layout)
        if self.model_info.get("latent") == "QWEN_LAYERED":
            from ruinedfooocus.features.generation.layered import require_vae

            require_vae(self.xl_base.vae)
        self.xl_base_hash = hash if hash is not None else name
        self.base_approximate_codec_identity = approximate_identity
        self.base_file_signature = signature
        self.base_bundle_identity = bundle_identity
        self.base_yaml_recipe_identity = yaml_identity
        self.base_component_layout = layout
        self.base_gguf_projectors = projectors or ()
        self.component_file_signatures = component_files
        return

    def load_loras(self, loras, *, captured_loras=None):
        if getattr(self, "_bernini_plan", None) is not None:
            self._bernini_loras = ([dict(row) for row in loras], captured_loras)
        from ruinedfooocus.backends.comfy.gguf_lora import apply_gguf_lora, patch_count
        from ruinedfooocus.features.models.lora_names import capture_gguf_lora, is_gguf_lora_name
        from ruinedfooocus.features.models.lora_sources import for_loras

        loaded_loras = []
        loras = sorted(loras, key=lambda x: x["name"].lower())
        resolved = []
        gguf_files = set()
        captured_by_name = (
            {
                (item.name.replace("\\", "/").casefold(), item.weight): item
                for item in for_loras(loras, captured_loras)
            }
            if captured_loras is not None
            else None
        )
        for lora in loras:
            name = lora.get("name", "None")
            weight = lora.get("weight", 0)
            hash = lora.get("hash", None)
            if name == "None":
                continue
            gguf = is_gguf_lora_name(name)
            if gguf and (type(weight) not in (int, float) or not -100 <= weight <= 100):
                raise ValueError("GGUF LoRA strength must be a number from -100 to 100.")
            if weight == 0:
                continue
            item = (
                captured_by_name[(name.replace("\\", "/").casefold(), weight)]
                if captured_by_name is not None
                else None
            )
            if item is not None:
                filename, stamp = item.path, list(item.signature)
            elif gguf:
                filename, stamp = capture_gguf_lora(name, shared.models.model_dirs["loras"])
            else:
                filename = shared.models.get_model_path("loras", name, hash=hash)
                if filename is None:
                    raise FileNotFoundError(f"Could not find image LoRA: {name}")
                if is_gguf_lora_name(filename):
                    raise ValueError("Select a GGUF LoRA by its explicit relative .gguf filename.")
                stamp = file_signature(filename)
            if gguf:
                if stamp[0] in gguf_files:
                    raise ValueError(f"The same GGUF LoRA is selected more than once: {name}")
                gguf_files.add(stamp[0])
            resolved.append((name, weight, filename, stamp, gguf, item))
        signature = [(stamp, weight) for _, weight, _, stamp, _, _ in resolved]
        if self.xl_base_patched_hash == signature:
            return

        model = self.xl_base
        for name, weight, filename, stamp, gguf, item in resolved:
            print(f"Loading LoRA: {name}")
            try:
                if gguf:
                    unet, clip = apply_gguf_lora(filename, stamp, model.unet, model.clip, weight)
                else:
                    if file_signature(filename) != stamp:
                        raise ValueError(
                            "The LoRA changed before loading. Submit the request again."
                        )
                    lora = comfy.utils.load_torch_file(str(filename), safe_load=True)
                    if item is not None:
                        item.verify()
                    if file_signature(filename) != stamp:
                        raise ValueError(
                            "The LoRA changed during loading. Submit the request again."
                        )
                    unet, clip = comfy.sd.load_lora_for_models(
                        model.unet, model.clip, lora, weight, weight
                    )
                if item is not None:
                    item.verify()
                if patch_count(unet, clip) <= patch_count(model.unet, model.clip):
                    raise ValueError(
                        "No compatible weights were applied. Choose a LoRA for this checkpoint family."
                    )
                model = self.StableDiffusionModel(
                    unet=unet,
                    clip=clip,
                    vae=model.vae,
                    clip_vision=model.clip_vision,
                )
                loaded_loras += [(name, weight)]
            except Exception as e:
                raise RuntimeError(f"Could not load image LoRA {name}: {e}") from e
        self.xl_base_patched = model
        self.xl_base_patched_hash = signature
        self.conditions = None

        print(f"LoRAs loaded: {loaded_loras}")
        return

    def refresh_controlnet(self, name=None, gen_data=None):
        with control_tools.model_asset(name, gen_data) as filename:
            signature = file_signature(filename) if filename is not None else None
            patch_signature = (
                id(self.xl_base_patched.unet),
                repr(getattr(self, "xl_base_patched_hash", None)),
            )
            if (
                signature is not None
                and self.xl_controlnet_hash == name
                and self.xl_controlnet is not None
                and signature == getattr(self, "controlnet_file_signature", None)
                and getattr(self, "base_file_signature", None)
                == getattr(self, "controlnet_base_signature", None)
                and patch_signature == getattr(self, "controlnet_patch_signature", None)
            ):
                return

            if filename is None and name in ("canny", "depth", "recolour", "sketch"):
                raise ValueError(f"ControlNet weights are unavailable: {name}")
            if filename is not None:
                self.xl_controlnet = comfy.controlnet.load_controlnet(
                    str(filename), model=self.xl_base_patched.unet
                )
                if self.xl_controlnet is None:
                    raise ValueError(f"ComfyUI could not load ControlNet: {filename}")
                self.xl_controlnet_hash = name
                self.controlnet_file_signature = signature
                self.controlnet_base_signature = getattr(self, "base_file_signature", None)
                self.controlnet_patch_signature = patch_signature
                print(f"ControlNet model loaded: {self.xl_controlnet_hash}")
            if self.xl_controlnet_hash != name:
                self.xl_controlnet = None
                self.xl_controlnet_hash = None

    conditions = None

    def textencode(self, id, text, clip_skip, refiner_size=None):
        from ruinedfooocus.features.models.embedding_browser import native_embedding_text
        from ruinedfooocus.features.models.yaml_recipes import clip_policy, apply_clip_recipe

        text = native_embedding_text(
            text,
            path_manager.model_paths["embedding_path"],
            path_manager.model_paths["cache_path"] / "embeddings",
        )
        update = False
        recipe = getattr(self, "yaml_recipe", None)
        hash = f"{text} {clip_skip} {clip_policy(recipe, clip_skip)}"
        if refiner_size is not None:
            hash = f"{hash} {refiner_size} {id == '-'}"
        if hash != self.conditions[id]["text"]:
            # Keep the model's native CLIP layer intact when switching presets.
            clip = apply_clip_recipe(self.xl_base_patched.clip, clip_skip, recipe)
            # Same prompt, seed and LoRA must not depend on the previous job.
            warm_patched_text_encoder(clip)
            if refiner_size is not None:
                from ruinedfooocus.features.image_editing.native_conditioning import (
                    refiner_text_conditioning,
                )

                self.conditions[id]["cache"] = refiner_text_conditioning(
                    clip, text, *refiner_size, negative=id == "-"
                )
            else:
                self.conditions[id]["cache"] = CLIPTextEncode().encode(clip=clip, text=text)[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    def textencode_pair(self, positive, negative, clip_skip, width, height, *, references=None):
        from ruinedfooocus.features.models.embedding_browser import native_embedding_text
        from ruinedfooocus.features.image_editing.qwen_image21_conditioning import (
            text_to_image_conditioning,
        )

        prompts = [
            native_embedding_text(
                text,
                path_manager.model_paths["embedding_path"],
                path_manager.model_paths["cache_path"] / "embeddings",
            )
            for text in (positive, negative)
        ]
        clip = self.xl_base_patched.clip
        if clip_skip > 1:
            clip = CLIPSetLastLayer().set_last_layer(clip, -clip_skip)[0]
        if references is not None:
            return references.conditioning(clip, *prompts)
        return text_to_image_conditioning(clip, *prompts, width, height)

    def decode_latent(self, samples, **options):
        from ruinedfooocus.backends.comfy.vae_decode import decode_image_vae

        return decode_image_vae(self.xl_base_patched, samples, **options)

    @stage_read_leases
    @torch.inference_mode()
    @lora_hooks.scoped
    @krea_style.scoped
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        if self.xl_base_patched is None:
            if self.load_error:
                raise ValueError(self.load_error)
            raise RuntimeError("Image model is not loaded. Check its required components.")
        architecture = self.xl_base_patched.unet.model.__class__.__name__
        from ruinedfooocus.features.generation import bernini_image

        if getattr(self, "_bernini_plan", None) is not None:
            return bernini_image.process(
                self, gen_data, callback, lambda: worker.check_interrupt(gen_data)
            )
        if (
            getattr(self, "_native_gligen", None) is not None
            and gen_data.get("_native_gligen_layout") is None
        ):
            from ruinedfooocus.features.image_editing.gligen_layout import clear_cache

            clear_cache(self)
        if (
            getattr(self, "_native_hypernetwork", None) is not None
            and gen_data.get("_native_hypernetwork") is None
        ):
            from ruinedfooocus.features.image_editing.native_hypernetwork import clear_cache

            clear_cache(self)
        if architecture == "MiniMaxH3Image":
            from ruinedfooocus.features.generation.minimax_image import process

            return process(self, gen_data, callback, lambda: worker.check_interrupt(gen_data))
        from ruinedfooocus.features.generation import layered

        layered_plan = gen_data.get(layered.KEY)
        layered_model = self.model_info.get("latent") == "QWEN_LAYERED"
        if layered_model != isinstance(layered_plan, layered.LayeredPlan):
            raise ValueError("Capture the Qwen Layered request with its matching RGBA codec.")
        if layered_model:
            layered.require_vae(self.xl_base_patched.vae)
        if architecture == "StableCascade_C":
            from ruinedfooocus.features.generation.cascade import sampling_settings

            sampling_settings(gen_data)
            if gen_data.get("cascade_super_resolution") or gen_data.get("cascade_vision_enabled"):
                from ruinedfooocus.features.generation.cascade_guidance import require_plan

                require_plan(self, gen_data)
        if architecture not in self.known_models:
            raise ValueError(f"Unsupported image architecture: {architecture}")
        if gen_data.get("sampling_shift", settings.default_settings.get("sampling_shift")) in (
            None,
            "",
        ):
            from ruinedfooocus.features.models.performance import MODEL_DEFAULT_SELECTIONS

            gen_data["sampling_shift"] = self.model_info.get("sampling_shift")
            if (
                gen_data["sampling_shift"] in (None, "")
                and gen_data.get("performance_selection") in MODEL_DEFAULT_SELECTIONS
            ):
                gen_data["sampling_shift"] = gen_data.get("_model_default_sampling_shift")
        from ruinedfooocus.features.image_editing.native_conditioning import (
            is_flux_inpaint,
            is_instruct_pix2pix,
            require_inpaint_input,
            require_instruct_edit,
        )

        flux_inpaint = is_flux_inpaint(self.xl_base_patched.unet.model)
        from ruinedfooocus.features.image_editing.native_sd_inpaint import (
            is_native_sd_inpaint,
            require_native_sd_inpaint_source,
        )

        sd_inpaint = is_native_sd_inpaint(self.xl_base_patched.unet.model)
        require_native_sd_inpaint_source(
            self.xl_base_patched.unet.model, gen_data.get("inpaint_toggle")
        )
        instruct_model = is_instruct_pix2pix(self.xl_base_patched.unet.model)
        from ruinedfooocus.features.image_editing.sd4x_conditioning import (
            is_sd4x,
            require_sd4x_source,
            require_sd4x_codec,
        )

        sd4x_model = is_sd4x(self.xl_base_patched.unet.model)
        if sd4x_model:
            require_sd4x_codec(self.xl_base_patched.vae)
        if sd4x_model and gen_data.get("inpaint_toggle"):
            require_sd4x_source(self.xl_base_patched.unet.model, "edit", None, inpainting=True)
        require_inpaint_input(self.xl_base_patched.unet.model, gen_data.get("inpaint_toggle"))
        if instruct_model and gen_data.get("inpaint_toggle"):
            require_instruct_edit(self.xl_base_patched.unet.model, "edit", None, inpainting=True)
        qwen_local_edit = gen_data.get("inpaint_toggle") and architecture == "QwenImage21"
        if gen_data.get("inpaint_toggle"):
            if not (
                qwen_local_edit
                or isinstance(self.xl_base_patched.unet.model, SDXL)
                or flux_inpaint
                or sd_inpaint
            ):
                raise ValueError("Inpainting requires an SDXL or Flux inpaint checkpoint.")
            from ruinedfooocus.features.image_editing.inpaint_utils import (
                load_source,
                outpaint_margins,
                source_and_mask,
            )

            if qwen_local_edit:
                inpaint_source, inpaint_mask = source_and_mask(
                    gen_data.get("main_view"), gen_data.get("inpaint_view"), preserve_alpha=True
                )
            elif any(outpaint_margins(gen_data)):
                inpaint_source, inpaint_mask = load_source(gen_data.get("main_view")), None
            else:
                inpaint_source, inpaint_mask = source_and_mask(
                    gen_data.get("main_view"), gen_data.get("inpaint_view")
                )
        checkpoint = gen_data.get("base_model_name")
        guidance = resolve_guidance(gen_data, settings.default_settings, architecture, checkpoint)
        if sd4x_model:
            guidance = dict(guidance, image_resize_mode="preserve")

        cfg = gen_data["cfg"]
        sampler_name = gen_data["sampler_name"]
        scheduler = gen_data["scheduler"]
        clip_skip = gen_data["clip_skip"]
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)
        if "<facerestore>" in gen_data.get("positive_prompt", ""):
            gen_data["facerestore"] = True
        positive_prompt = gen_data["positive_prompt"].replace("<facerestore>", "")
        negative_prompt = gen_data["negative_prompt"].replace("<facerestore>", "")
        image_edit = self.model_info.get("options", {}).get("has_image_edit", False) or (
            architecture == "MOSSImage" and guidance["moss_image_edit_format"] != "Disabled"
        )
        hidream_edit = architecture == "HiDreamO1"
        unclip_model = architecture == "SD21UNCLIP"
        pid_model = architecture == "PiD"
        refiner_model = architecture == "HunyuanImage21Refiner"
        cascade_model = architecture == "StableCascade_C"
        controlnet = control_tools.get_settings(gen_data)
        control_type = controlnet.get("type", "")
        from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

        named_source = model_patches.has_source(gen_data) or native_control_recipes.has_source(
            gen_data
        )
        qwen_union = control_type == "qwen-union"
        joy_reference_edit = architecture == "JoyImage" and control_type == "edit"
        image_edit = image_edit or joy_reference_edit
        from ruinedfooocus.features.image_editing.ordered_native_references import (
            FLUX2_FAMILIES,
            OrderedNativeReferences,
            current_source,
            recipe_for,
            reference_family,
        )

        ordered_family = reference_family(self.xl_base_patched.unet.model)
        ordered_recipe, ordered_source = None, None
        if control_type == "edit" and ordered_family in (
            "Boogu",
            "Krea2",
            "QwenImage",
            "LongCatImage",
            *FLUX2_FAMILIES,
        ):
            ordered_source = current_source(checkpoint, shared.models, path_manager)
            ordered_recipe = recipe_for(
                ordered_family, gen_data.get("native_edit_recipe", "auto"), ordered_source
            )
            image_edit = False  # The typed paired encoder below owns this request.
        if qwen_local_edit:
            if control_type not in ("", "None", None, "edit", "img2img"):
                raise ValueError(
                    "Qwen Image 2.1 localized editing cannot be combined with another image tool."
                )
            controlnet, control_type = {"type": "edit"}, "edit"
        qwen_reference_edit = architecture == "QwenImage21" and control_type == "edit"
        from ruinedfooocus.features.image_editing import guide_handoffs

        guided_request = gen_data.get(guide_handoffs.KEY)
        guided_edit = isinstance(guided_request, guide_handoffs.GuideRequest)
        control_guide, control_source_alpha = None, None
        input_image = (
            guide_handoffs.request_image(guided_request.source)
            if guided_edit
            else gen_data.get("input_image")
            if control_type not in ("", "None", None)
            else None
        )
        if qwen_local_edit:
            input_image = inpaint_source
        if qwen_union:
            input_image = None
        require_instruct_edit(self.xl_base_patched.unet.model, control_type, input_image)
        require_sd4x_source(self.xl_base_patched.unet.model, control_type, input_image)
        if pid_model or refiner_model:
            if control_type != "edit" or input_image is None or gen_data.get("inpaint_toggle"):
                raise ValueError(
                    "This restoration/refiner checkpoint requires Image tools Edit with a source image and no mask."
                )
        has_reference_input = input_image is not None and (
            architecture != "QwenImage21" or qwen_reference_edit
        )
        additional_references = normalize_reference_images(
            gen_data.get("reference_images") if has_reference_input else None,
            input_image,
            architecture,
            guidance["moss_image_edit_format"],
        )
        gen_data["reference_image_count"] = (1 if input_image is not None else 0) + len(
            additional_references
        )
        if (
            additional_references
            and ordered_family in ("Boogu", "Krea2", "QwenImage", *FLUX2_FAMILIES)
            and control_type != "edit"
        ):
            raise ValueError(
                "Ordered references for this checkpoint require the native image edit operation."
            )
        if control_type == "edit":
            if input_image is None:
                raise ValueError("Add an image in Image tools.")
            native = (
                image_edit
                or ordered_recipe is not None
                or qwen_reference_edit
                or hidream_edit
                or instruct_model
                or sd4x_model
                or unclip_model
                or pid_model
                or refiner_model
                or cascade_model
            )
            if architecture == "Flux":
                name = gen_data.get("base_model_name", "")
                path = shared.models.get_file("checkpoints", name)
                metadata = (
                    shared.models.get_models_by_path("checkpoints", path, fetch=False)
                    if path
                    else {}
                )
                native = "kontext" in f"{name} {shared.models.get_model_base(metadata)}".lower()
                if native:
                    controlnet = dict(controlnet, type="kontext")
                    control_type = "kontext"
            if not native:
                # Saved presets or API requests may ask for native editing on a
                # checkpoint without it; edit by image-to-image instead of failing.
                from ruinedfooocus.features.models.image_components import IMG2IMG_DENOISE

                print(
                    "Native image edit is not available for this checkpoint; using image to image."
                )
                controlnet = {"type": "img2img", "denoise": IMG2IMG_DENOISE}
                control_type = "img2img"
        if control_type in control_tools.controlnet_models and control_type not in (
            "upscale",
            "rembg",
            "faceswap",
            "qwen-union",
        ):
            if gen_data.get("inpaint_toggle") and control_type == "img2img":
                # Inpainting encodes Main with its mask, not the optional tool upload.
                input_image, controlnet, control_type = None, {}, ""
                gen_data["reference_image_count"] = 0
            elif input_image is None and not named_source:
                raise ValueError("Add an image in Image tools.")
            if control_type in ("canny", "depth", "recolour", "sketch", "control"):
                if not controlnet.get("model") and not isinstance(
                    self.xl_base_patched.unet.model, SDXL
                ):
                    raise ValueError("The bundled ControlNet weights require an SDXL checkpoint.")
                start, stop = float(controlnet["start"]), float(controlnet["stop"])
                if not 0 <= start < stop <= 1:
                    raise ValueError("ControlNet requires 0 <= Start < Stop <= 1.")
            elif control_type == "img2img":
                if not 0 < float(controlnet.get("denoise", controlnet.get("strength", 1))) <= 1:
                    raise ValueError("Image-to-image denoise must be greater than 0 and at most 1.")

        device = comfy.model_management.get_torch_device()
        switched_prompt = []
        img2img_mode = control_type == "img2img"
        edit_latent = None
        reference_frame = None
        from ruinedfooocus.features.generation.size_plan import KEY as SIZE_PLAN_KEY, SizePlan

        size_plan = gen_data.get(SIZE_PLAN_KEY)
        if isinstance(size_plan, SizePlan):
            # The loaded native VAE owns spatial compression. Pixel-space
            # families additionally require their native patch alignment.
            compression = self.xl_base_patched.vae.spacial_compression_decode()
            alignment = max(8, int(compression))
            if architecture == "HiDreamO1":
                alignment = 32
            elif architecture == "ChromaRadiance":
                alignment = 16
            elif layered_model:
                alignment = 16
            if architecture == "StableCascade_C":
                # Stage B decodes from its native quarter-resolution canvas.
                alignment = 32 if gen_data.get("cascade_super_resolution") else 8
            size_plan = size_plan.aligned(
                alignment,
                minimum=64 if architecture == "HiDreamO1" else 1,
                maximum=4096 if architecture == "HiDreamO1" else None,
            )
            gen_data[SIZE_PLAN_KEY] = size_plan
            gen_data["width"], gen_data["height"] = size_plan.content
        outpaint_composite = None
        qwen_references = None
        ordered_references = None
        gen_data.pop("image_geometry", None)

        if layered_model:
            # This source is immutable RGBA data captured before the queue. Its
            # reference latent is distinct from the target's layers+1 slots.
            source_frame = layered.source_frame(
                layered_plan,
                gen_data["width"],
                gen_data["height"],
                guidance["image_resize_mode"],
            )
            if source_frame is not None:
                reference_frame = source_frame
                gen_data["width"], gen_data["height"] = (
                    source_frame["canvas_width"],
                    source_frame["canvas_height"],
                )

        # Pre-process input-image
        input_images = 0
        cascade_latents = None
        named_source_latent = None
        named_source_mask = None
        if named_source:
            requested_size = gen_data.setdefault(
                "_image_requested_size", (gen_data["width"], gen_data["height"])
            )
            check_cancelled = lambda: worker.check_interrupt(gen_data)
            source_recipe = (
                native_control_recipes
                if native_control_recipes.has_source(gen_data)
                else model_patches
            )
            named_source_latent = source_recipe.source_latent(
                self.xl_base_patched.vae,
                gen_data,
                requested_size,
                guidance["image_resize_mode"],
                check_cancelled,
            )
            edit_latent, input_image, named_source_mask, reference_frame = named_source_latent
            gen_data["width"], gen_data["height"] = (
                reference_frame["canvas_width"],
                reference_frame["canvas_height"],
            )
            input_images = 1
        elif qwen_reference_edit:
            from ruinedfooocus.features.image_editing.qwen_image21_conditioning import (
                QwenImage21References,
            )

            requested_size = gen_data.setdefault(
                "_image_requested_size", (gen_data["width"], gen_data["height"])
            )
            qwen_references = QwenImage21References(
                self.xl_base_patched.vae,
                input_image,
                additional_references,
                requested_size,
                guidance["image_resize_mode"],
                paint_mask=inpaint_mask if gen_data.get("inpaint_toggle") else None,
                edit_mode=gen_data.get("qwen_local_edit_mode", "paint"),
            )
            reference_frame = qwen_references.frame
            gen_data["width"], gen_data["height"] = (
                reference_frame["canvas_width"],
                reference_frame["canvas_height"],
            )
        elif input_image:
            if cascade_model:
                if gen_data.get("cascade_super_resolution"):
                    from ruinedfooocus.features.generation.cascade_guidance import (
                        super_resolution_source,
                    )

                    input_image, reference_frame, cascade_latents = super_resolution_source(
                        self.xl_base_patched.vae,
                        input_image,
                        gen_data,
                        mode=guidance["image_resize_mode"],
                    )
                else:
                    from ruinedfooocus.features.generation.cascade import (
                        source_latents,
                        source_pixels,
                    )

                    input_image, reference_frame = source_pixels(
                        input_image,
                        gen_data,
                        mode=guidance["image_resize_mode"],
                        to_tensor=reference_tensor,
                    )
                    cascade_latents = source_latents(
                        self.xl_base_patched.vae, input_image, gen_data, frame=reference_frame
                    )
            elif pid_model:
                input_image = reference_tensor(ImageOps.exif_transpose(input_image))
            elif sd4x_model:
                # The native operation samples at source size and decodes at 4×.
                # Preserve its geometry instead of applying the generation budget.
                from ruinedfooocus.features.image_editing.sd4x_conditioning import sd4x_pixels

                input_image = sd4x_pixels(input_image, gen_data)
                gen_data["width"], gen_data["height"] = (
                    int(input_image.shape[2]) * 4,
                    int(input_image.shape[1]) * 4,
                )
            elif ordered_recipe == "krea2-style":
                # Native reference nodes own image sizing independently of the target canvas.
                input_image = reference_tensor(ImageOps.exif_transpose(input_image))
            elif not gen_data.get("inpaint_toggle") and (
                image_edit
                or ordered_recipe is not None
                or refiner_model
                or hidream_edit
                or instruct_model
                or control_type in ("img2img", "canny", "depth", "sketch", "recolour")
                or guided_edit
            ):
                requested_size = gen_data.setdefault(
                    "_image_requested_size", (gen_data["width"], gen_data["height"])
                )
                auto_source = (
                    guide_handoffs.automatic_source_plan(gen_data) if guided_edit else None
                )
                if auto_source is not None:
                    requested_size = gen_data["_image_requested_size"] = auto_source["target"]
                alignment = (
                    size_plan.alignment
                    if isinstance(size_plan, SizePlan)
                    else (32 if architecture == "HiDreamO1" else 16)
                )
                if guided_edit:
                    input_image, control_guide, reference_frame, control_source_alpha = (
                        guide_handoffs.prepare_reference_pair(
                            guided_request,
                            requested_size,
                            auto_source["mode"] if auto_source else guidance["image_resize_mode"],
                            multiple=alignment,
                        )
                    )
                else:
                    input_image, reference_frame = prepare_reference(
                        input_image,
                        requested_size,
                        guidance["image_resize_mode"],
                        multiple=alignment,
                    )
                gen_data["width"], gen_data["height"] = (
                    reference_frame["canvas_width"],
                    reference_frame["canvas_height"],
                )
            else:
                input_image = np.array(input_image.convert("RGB")).astype(np.float32) / 255.0
                input_image = torch.from_numpy(input_image)[None,]
                megapixels = float(gen_data["width"]) * float(gen_data["height"]) / (1024 * 1024)
                input_image = ImageScaleToTotalPixels().execute(
                    image=input_image,
                    upscale_method="bicubic",
                    megapixels=megapixels,
                    resolution_steps=1,
                )[0]
            input_images = 1  # "Counter" for the single input-image we have. (for now)
            if cascade_latents is not None:
                input_images -= 1
        elif architecture == "HiDreamO1" and (gen_data["width"] % 32 or gen_data["height"] % 32):
            from ruinedfooocus.features.image_editing.image_geometry import canvas_plan

            reference_frame = canvas_plan((gen_data["width"], gen_data["height"]), 32)
            gen_data["width"], gen_data["height"] = (
                reference_frame["canvas_width"],
                reference_frame["canvas_height"],
            )

        if isinstance(size_plan, SizePlan) and input_image is None and reference_frame is None:
            reference_frame = dict(
                width=size_plan.content_width,
                height=size_plan.content_height,
                canvas_width=size_plan.canvas_width,
                canvas_height=size_plan.canvas_height,
                left=size_plan.left,
                top=size_plan.top,
            )
            gen_data["width"], gen_data["height"] = size_plan.canvas

        if hidream_edit and input_image is not None and not img2img_mode:
            controlnet = None

        # Text-encoding
        tensor_conditions = {}
        tensor_latent = None
        if "_native_tensor_plan" in gen_data:
            from ruinedfooocus.features.generation.native_tensor_io import (
                conditioning_inputs,
                latent_input,
            )

            tensor_conditions = conditioning_inputs(self, gen_data)
            tensor_latent = latent_input(self, gen_data)
        moss_edit_format = guidance["moss_image_edit_format"]
        moss_edit_strength = guidance["moss_image_edit_strength"]
        if architecture == "MOSSImage" and moss_edit_format != "Disabled":
            if moss_edit_format not in EDIT_FORMATS:
                raise ValueError("Choose a supported MOSS image edit format in Settings.")
            if not np.isfinite(moss_edit_strength) or not 0 <= moss_edit_strength <= 100:
                raise ValueError("MOSS image edit strength must be between 0 and 100.")
        if ordered_recipe is not None:
            ordered_references = OrderedNativeReferences(
                ordered_recipe,
                self.xl_base_patched.vae,
                (
                    input_image,
                    *(
                        reference_tensor(image)
                        if ordered_recipe in ("boogu-edit", "krea2-style")
                        else reference_tensor(image, gen_data["width"], gen_data["height"])
                        for image in additional_references
                    ),
                ),
                ordered_source,
            )
            gen_data["native_edit_recipe"] = ordered_recipe
            gen_data["native_reference_conditioning"] = ordered_references.provenance()
        if ordered_references is not None and ordered_recipe != "flux2-reference":
            self.conditions = clean_prompt_cond_caches()
            clip = self.xl_base_patched.clip
            if clip_skip > 1:
                clip = CLIPSetLastLayer().set_last_layer(clip, -clip_skip)[0]
            positive_conditioning, negative_conditioning, switched_prompt = (
                ordered_references.scheduled_conditioning(
                    clip, positive_prompt, negative_prompt, gen_data["steps"]
                )
            )
            self.conditions["+"]["cache"] = positive_conditioning
            self.conditions["-"]["cache"] = negative_conditioning
        elif image_edit and input_images > 0:
            if callback is not None:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Processing image with text encoding ...", None),
                )
            if not img2img_mode:
                controlnet = None
            if not joy_reference_edit:
                self.conditions = clean_prompt_cond_caches()
            # Edit encoders share the patched text encoder; settle LoRA patches first.
            warm_patched_text_encoder(self.xl_base_patched.clip)

            match self.xl_base_patched.unet.model.__class__.__name__:
                case "JoyImage":
                    positive_conditioning, negative_conditioning = joy_reference_conditioning(
                        self.xl_base_patched.clip,
                        self.xl_base_patched.vae,
                        input_image,
                        positive_prompt,
                        negative_prompt,
                    )
                case "Fibo2":
                    self.conditions["+"]["cache"], self.conditions["-"]["cache"], edit_latent = (
                        TextEncodeFibo2()
                        .execute(
                            clip=self.xl_base_patched.clip,
                            prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            vae=self.xl_base_patched.vae,
                            width=gen_data["width"],
                            height=gen_data["height"],
                            images=reference_image_map(
                                input_image, additional_references, reference_tensor
                            ),
                        )
                        .result
                    )
                case "MOSSImage":
                    for sign, prompt in (("+", positive_prompt), ("-", negative_prompt)):
                        self.conditions[sign]["cache"] = TextEncodeMOSSImageEdit().execute(
                            clip=self.xl_base_patched.clip,
                            prompt=prompt,
                            edit_format=moss_edit_format,
                            image_strength=moss_edit_strength,
                            images=reference_image_map(
                                input_image,
                                additional_references,
                                lambda image: reference_tensor(
                                    image, gen_data["width"], gen_data["height"]
                                ),
                            ),
                            vae=self.xl_base_patched.vae,
                        )[0]
                case "QwenImage":
                    self.conditions["+"]["cache"] = TextEncodeQwenImageEdit().execute(
                        clip=self.xl_base_patched.clip,
                        prompt=positive_prompt,
                        vae=self.xl_base.vae,
                        image=input_image,
                    )[0]
                    self.conditions["-"]["cache"] = TextEncodeQwenImageEdit().execute(
                        clip=self.xl_base_patched.clip,
                        prompt=negative_prompt,
                        vae=self.xl_base.vae,
                        image=input_image,
                    )[0]
                case "MageFlow":
                    self.conditions["+"]["cache"], self.conditions["-"]["cache"], edit_latent = (
                        TextEncodeMageFlowEdit()
                        .execute(
                            clip=self.xl_base_patched.clip,
                            vae=self.xl_base_patched.vae,
                            prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            images=reference_image_map(
                                input_image, additional_references, reference_tensor
                            ),
                            width=gen_data["width"],
                            height=gen_data["height"],
                        )
                        .result
                    )
                case _:
                    print(
                        f"ERROR: Trying to encode with unknown model {self.xl_base_patched.unet.__class__.__name__}. (This should never happen."
                    )
                    raise

            input_images -= 1
        else:
            if callback is not None:
                worker.add_result(
                    gen_data["task_id"], "preview", (-1, f"Processing text encoding ...", None)
                )
            if self.conditions is None:
                self.conditions = clean_prompt_cond_caches()
            # Scheduled text is request-local; release the last job's extra tensor.
            self.conditions["switch"] = {"text": None, "cache": None}
            encode_options = (
                {"refiner_size": (gen_data["width"], gen_data["height"])}
                if architecture == "SDXLRefiner"
                else {}
            )
            if architecture == "QwenImage21":
                pair_options = (
                    {"references": qwen_references} if qwen_references is not None else {}
                )
                positive_conditioning, negative_conditioning, edit_latent = self.textencode_pair(
                    positive_prompt,
                    negative_prompt,
                    clip_skip,
                    gen_data["width"],
                    gen_data["height"],
                    **pair_options,
                )
            else:
                for sign, role, text in (
                    ("+", "positive", positive_prompt),
                    ("-", "negative", negative_prompt),
                ):
                    if role in tensor_conditions:
                        self.conditions[sign] = {"text": None, "cache": tensor_conditions[role]}
                    else:
                        self.textencode(sign, text, clip_skip, **encode_options)

            prompt_per_step = (
                []
                if "positive" in tensor_conditions
                else pp.prompt_switch_per_step(positive_prompt, gen_data["steps"])
            )
            if prompt_per_step:
                encoded_switches = {}
                for i in range(len(prompt_per_step)):
                    text = prompt_per_step[i]
                    if text not in encoded_switches:
                        if architecture == "QwenImage21":
                            encoded_switches[text] = self.textencode_pair(
                                text,
                                negative_prompt,
                                clip_skip,
                                gen_data["width"],
                                gen_data["height"],
                                **pair_options,
                            )[0]
                        else:
                            self.textencode("switch", text, clip_skip, **encode_options)
                            encoded_switches[text] = self.conditions["switch"]["cache"]
                    positive_switch = encoded_switches[text]
                    start_perc = i / len(prompt_per_step)
                    end_perc = (i + 1) / len(prompt_per_step)
                    positive_switch = conditioning_set_values(
                        positive_switch,
                        {
                            "start_percent": start_perc,
                            "end_percent": end_perc,
                            "_rf_prompt_step": i,
                        },
                    )
                    switched_prompt += positive_switch

        # Keep cached text conditioning independent of each request's image/mask.
        if architecture != "QwenImage21" and not joy_reference_edit:
            positive_conditioning = self.conditions["+"]["cache"]
            negative_conditioning = self.conditions["-"]["cache"]
        from ruinedfooocus.features.image_editing.reference_adapters import apply_reference

        reference_positive = apply_reference(
            self,
            switched_prompt if switched_prompt else positive_conditioning,
            gen_data,
            path_manager,
            reference_tensor,
        )
        if switched_prompt:
            switched_prompt = reference_positive
        else:
            positive_conditioning = reference_positive
        if gen_data.get("_cascade_guidance") is not None:
            from ruinedfooocus.features.generation.cascade_guidance import apply_guidance

            guided, negative_conditioning = apply_guidance(
                self,
                switched_prompt if switched_prompt else positive_conditioning,
                negative_conditioning,
                gen_data,
                path_manager,
                cascade_latents.control_hint if cascade_latents is not None else None,
                lambda: worker.check_interrupt(gen_data),
            )
            if switched_prompt:
                switched_prompt = guided
            else:
                positive_conditioning = guided
        if pid_model or refiner_model:
            source_positive = switched_prompt if switched_prompt else positive_conditioning
            if pid_model:
                from ruinedfooocus.features.image_editing.pid_conditioning import pid_conditioning

                source_positive, negative_conditioning, edit_latent = pid_conditioning(
                    self.xl_base_patched.unet.model,
                    source_positive,
                    negative_conditioning,
                    input_image,
                    gen_data,
                    path_manager,
                )
            else:
                from ruinedfooocus.features.image_editing.hunyuan_refiner import (
                    refiner_conditioning,
                )

                source_positive, negative_conditioning, edit_latent = refiner_conditioning(
                    source_positive,
                    negative_conditioning,
                    self.xl_base_patched.vae,
                    input_image,
                    gen_data,
                )
            if switched_prompt:
                switched_prompt = source_positive
            else:
                positive_conditioning = source_positive
            input_images -= 1
        if unclip_model and input_image is not None and control_type == "edit":
            from ruinedfooocus.features.image_editing.native_conditioning import unclip_conditioning

            guided = unclip_conditioning(
                switched_prompt if switched_prompt else positive_conditioning,
                self.xl_base_patched.unet.model,
                self.xl_base_patched.clip_vision,
                input_image,
            )
            if switched_prompt:
                switched_prompt = guided
            else:
                positive_conditioning = guided
            input_images -= 1
        if hidream_edit and input_image is not None:
            images = reference_image_map(input_image, additional_references, reference_tensor)
            positive_conditioning, negative_conditioning = HiDreamO1ReferenceImages.execute(
                positive=positive_conditioning, negative=negative_conditioning, images=images
            )
            if switched_prompt:
                switched_prompt = HiDreamO1ReferenceImages.execute(
                    positive=switched_prompt, negative=self.conditions["-"]["cache"], images=images
                )[0]
            input_images -= 1

        # Controlnet / img2img
        if controlnet is None or not "type" in controlnet:
            controlnet = {}
            controlnet["type"] = "None"

        if controlnet["type"] == "kontext":
            img2img_mode = True

        if controlnet["type"] != "None" and input_images > 0:
            if callback is not None:
                worker.add_result(gen_data["task_id"], "preview", (-1, f"Powering up ...", None))

            self.refresh_controlnet(name=controlnet["type"], gen_data=gen_data)
            control_image = control_guide if guided_edit else input_image
            match controlnet["type"].lower():
                case "canny" if not guided_edit:
                    low, high = sanitize_canny_thresholds(
                        float(controlnet.get("edge_low", 0.0)),
                        float(controlnet.get("edge_high", 1.0)),
                    )
                    control_image = Canny.execute(
                        image=control_image,
                        low_threshold=low,
                        high_threshold=high,
                    )[0]
                case "depth" if not guided_edit:
                    from ruinedfooocus.features.image_editing.depth_control import (
                        prepare_depth,
                        depth_settings,
                    )

                    options = depth_settings(controlnet)
                    control_image = prepare_depth(self, control_image, options, gen_data)
                    gen_data["depth_preprocessing"] = dict(
                        options,
                        resolved_model=getattr(self, "depth_model_name", None)
                        if options["depth_preprocess"] == "auto"
                        else None,
                    )
            if not guided_edit:
                input_image = control_image
            if self.xl_controlnet:
                named_control = native_control_recipes.apply(
                    self,
                    switched_prompt if switched_prompt else positive_conditioning,
                    negative_conditioning,
                    self.xl_controlnet,
                    self.xl_base_patched.vae,
                    control_image,
                    gen_data,
                    lambda: worker.check_interrupt(gen_data),
                    mask=named_source_mask,
                )
                if named_control is None:
                    named_control = ControlNetApplyAdvanced().apply_controlnet(
                        positive=switched_prompt if switched_prompt else positive_conditioning,
                        negative=negative_conditioning,
                        control_net=self.xl_controlnet,
                        image=control_image,
                        strength=float(controlnet["strength"]),
                        start_percent=float(controlnet["start"]),
                        end_percent=float(controlnet["stop"]),
                        vae=self.xl_base_patched.vae,
                    )
                positive_conditioning, negative_conditioning = named_control
                switched_prompt = []  # The native result retains every interval and its ControlNet.

            if controlnet["type"].lower() == "img2img" or guided_edit:
                img2img_mode = True
            if guided_edit:
                gen_data["control_guide"] = guided_request.metadata()

        if sd4x_model:
            from ruinedfooocus.features.image_editing.sd4x_conditioning import sd4x_conditioning

            conditioned, negative_conditioning, edit_latent = sd4x_conditioning(
                switched_prompt if switched_prompt else positive_conditioning,
                negative_conditioning,
                input_image,
            )
            if switched_prompt:
                switched_prompt = conditioned
            else:
                positive_conditioning = conditioned
        if instruct_model:
            from ruinedfooocus.features.image_editing.native_conditioning import (
                instruct_conditioning,
            )

            conditioned, negative_conditioning, edit_latent = instruct_conditioning(
                switched_prompt if switched_prompt else positive_conditioning,
                negative_conditioning,
                self.xl_base_patched.vae,
                input_image,
            )
            if switched_prompt:
                switched_prompt = conditioned
            else:
                positive_conditioning = conditioned
        if layered_model:
            conditioned, negative_conditioning, _ = layered.reference_conditioning(
                layered_plan,
                self.xl_base_patched.vae,
                switched_prompt if switched_prompt else positive_conditioning,
                negative_conditioning,
                gen_data["width"],
                gen_data["height"],
                guidance["image_resize_mode"],
                frame=reference_frame,
            )
            if switched_prompt:
                switched_prompt = conditioned
            else:
                positive_conditioning = conditioned
        if ordered_references is not None and ordered_recipe == "flux2-reference":
            conditioned, negative_conditioning, edit_latent = ordered_references.attach(
                switched_prompt if switched_prompt else positive_conditioning, negative_conditioning
            )
            if switched_prompt:
                switched_prompt = conditioned
            else:
                positive_conditioning = conditioned
        if tensor_latent is not None:
            latent = tensor_latent
            denoise = gen_data["native_latent_denoise"]
            if "_rf_stage_b" in latent:
                from dataclasses import replace
                from ruinedfooocus.features.generation.cascade import CascadeLatents

                cascade_latents = (
                    replace(cascade_latents, stage_c=latent, stage_b=latent["_rf_stage_b"])
                    if cascade_latents is not None
                    else CascadeLatents(latent, latent["_rf_stage_b"])
                )
        elif named_source_latent is not None:
            latent = edit_latent
            denoise_key = (
                "control_inpaint_denoise"
                if native_control_recipes.has_source(gen_data)
                else "model_patch_denoise"
            )
            denoise = float(gen_data.get(denoise_key, 1.0))
        elif cascade_latents is not None:
            from ruinedfooocus.features.generation.cascade import source_settings

            latent = cascade_latents.stage_c
            if gen_data.get("cascade_super_resolution"):
                from ruinedfooocus.features.generation.cascade_guidance import KEY as GUIDANCE_KEY

                denoise = gen_data[GUIDANCE_KEY].denoise
            else:
                denoise = source_settings(gen_data)[1]
        elif edit_latent is not None and not img2img_mode:
            # Native edit references and target must share the encoder's size.
            latent = edit_latent
            denoise = None
        elif img2img_mode:
            # Ordinary batches always start from the supplied reference. Explicit
            # loopback is request-local and cannot reuse another job's output.
            if controlnet.get("loopback", False) and gen_data.get("_loopback_image") is not None:
                input_image = Image.fromarray(gen_data["_loopback_image"]).convert("RGB")
                input_image = np.array(input_image).astype(np.float32) / 255.0
                input_image = torch.from_numpy(input_image)[None,]
            if controlnet["type"].lower() == "kontext":
                input_image = FluxKontextImageScale().execute(input_image)[0]

            latent = VAEEncode().encode(vae=self.xl_base_patched.vae, pixels=input_image)[0]
            # The output keeps the encoded source size; record it for metadata/replay.
            scale = self.xl_base_patched.vae.spacial_compression_decode()
            gen_data["height"], gen_data["width"] = (
                int(side) * scale for side in latent["samples"].shape[-2:]
            )
            denoise = (
                guided_request.denoise
                if guided_edit
                else float(controlnet.get("denoise", controlnet.get("strength", 1)))
            )
        else:
            # Get the correct of latent image to start with
            latent_type = self.model_info.get("latent", None)
            match latent_type:
                case "QWEN_LAYERED":
                    latent = layered.empty_latent(
                        layered_plan,
                        self.xl_base_patched.vae,
                        gen_data["width"],
                        gen_data["height"],
                    )
                case "CASCADE_C":
                    from comfy_extras.nodes_stable_cascade import StableCascade_EmptyLatentImage

                    empty = StableCascade_EmptyLatentImage.execute(
                        gen_data["width"],
                        gen_data["height"],
                        gen_data.get("cascade_compression", 42),
                        1,
                    )
                    latent = empty[0]
                    if "_native_tensor_plan" in gen_data:
                        from ruinedfooocus.features.generation.cascade import CascadeLatents

                        cascade_latents = CascadeLatents(latent, empty[1])
                case "FLUX2":
                    latent = EmptyFlux2LatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case "HiDreamO1":
                    latent = EmptyHiDreamO1LatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case "HunyuanImage":
                    latent = EmptyHunyuanImageLatent().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case "ChromaRadience":
                    latent = EmptyChromaRadianceLatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case "SD3":
                    latent = EmptySD3LatentImage().execute(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
                case "MAGE":
                    zero_latent = torch.zeros(
                        [1, 128, gen_data["height"] // 16, gen_data["width"] // 16],
                        device=comfy.model_management.intermediate_device(),
                    )
                    latent = {"samples": zero_latent, "downscale_ratio_spacial": 16}
                case _:
                    latent = EmptyLatentImage().generate(
                        width=gen_data["width"], height=gen_data["height"], batch_size=1
                    )[0]
            denoise = None

        if gen_data.get("inpaint_toggle") and architecture != "QwenImage21":
            denoise = float(gen_data.get("inpaint_denoise", 1.0))
            if not 0 < denoise <= 1:
                raise ValueError("Inpainting denoise must be greater than 0 and at most 1.")
            mask_grow = int(gen_data.get("inpaint_mask_grow", 20))
            if not 0 <= mask_grow <= 128:
                raise ValueError("Mask expansion must be between 0 and 128 pixels.")
            from ruinedfooocus.features.image_editing.inpaint_utils import (
                prepare_inpaint,
                prepare_outpaint,
            )

            requested_size = gen_data.setdefault(
                "_image_requested_size", (gen_data["width"], gen_data["height"])
            )
            if inpaint_mask is None:
                image, mask, reference_frame = prepare_outpaint(inpaint_source, gen_data)
                outpaint_composite = image, mask
            else:
                image, mask, reference_frame = prepare_inpaint(
                    inpaint_source, inpaint_mask, requested_size, guidance["image_resize_mode"]
                )
            gen_data["width"], gen_data["height"] = (
                reference_frame["canvas_width"],
                reference_frame["canvas_height"],
            )
            if flux_inpaint or sd_inpaint:
                from ruinedfooocus.features.image_editing.native_conditioning import (
                    flux_inpaint_conditioning,
                )

                conditioner = flux_inpaint_conditioning
                if sd_inpaint:
                    from functools import partial
                    from ruinedfooocus.features.image_editing.native_sd_inpaint import (
                        native_sd_inpaint_conditioning,
                    )

                    conditioner = partial(
                        native_sd_inpaint_conditioning, self.xl_base_patched.unet.model
                    )

                conditioned, negative_conditioning, latent = conditioner(
                    switched_prompt if switched_prompt else positive_conditioning,
                    negative_conditioning,
                    self.xl_base_patched.vae,
                    image,
                    mask,
                    grow_mask_by=mask_grow,
                )
                if switched_prompt:
                    switched_prompt = conditioned
                else:
                    positive_conditioning = conditioned
                gen_data["native_inpaint_conditioning"] = {
                    "native_node": "InpaintModelConditioning",
                    "native_config": type(self.xl_base_patched.unet.model.model_config).__name__,
                    "grow_mask_by": mask_grow,
                    "mask_polarity": "white-edit",
                }
            else:
                latent = VAEEncodeForInpaint().encode(
                    vae=self.xl_base_patched.vae, pixels=image, mask=mask, grow_mask_by=mask_grow
                )[0]

        gen_data["sampling_denoise"] = denoise if denoise is not None else 1.0

        if guidance.get("_model_sampling_flux"):
            # Source framing and native edit nodes may have changed the target.
            guidance["_model_sampling_flux"].update(
                width=gen_data["width"], height=gen_data["height"]
            )
        sampling_model = request_model(
            self.xl_base_patched.unet, guidance, architecture, checkpoint=checkpoint
        )
        if layered_model:
            sampling_model = layered.patch_model(sampling_model, guidance)
        if gen_data.get("_native_hypernetwork") is not None:
            from ruinedfooocus.features.image_editing.native_hypernetwork import patch

            sampling_model = patch(
                self,
                sampling_model,
                gen_data,
                path_manager,
                lambda: worker.check_interrupt(gen_data),
            )
        from ruinedfooocus.features.generation.native_sampling import KEY, NativeSampling

        native_plan = gen_data.get(KEY)
        if native_plan is not None:
            from ruinedfooocus.features.generation.native_sampling import patch_model

            if not isinstance(native_plan, NativeSampling):
                raise ValueError("Capture the native sampling recipe before generation.")
            sampling_model = patch_model(native_plan, sampling_model, guidance=guidance)
        if qwen_union:
            from ruinedfooocus.features.image_editing.qwen_image21_union import apply_union

            sampling_model = apply_union(
                sampling_model, self.xl_base_patched.vae, gen_data, path_manager
            )
        effective = effective_guidance(guidance, sampling_model, architecture)
        gen_data.update(effective)
        gen_data["image_guidance"] = effective

        # KSampler

        # The native Chroma and LongCat subclasses use sampler CFG. Chroma's
        # separate distilled guidance defaults to zero; LongCat removes it.
        positive_cond = switched_prompt if switched_prompt else positive_conditioning
        if gen_data.get("_native_gligen_layout") is not None:
            from ruinedfooocus.features.image_editing.gligen_layout import apply

            positive_cond = apply(
                self,
                positive_cond,
                gen_data,
                path_manager,
                lambda: worker.check_interrupt(gen_data),
            )
        from ruinedfooocus.features.prompting.prompt_scheduling import bind_prompt_steps

        fixed_sigmas = None
        if native_plan is not None:
            from ruinedfooocus.features.generation.native_sampling import condition, sigmas

            positive_cond = condition(native_plan, positive_cond)
            fixed_sigmas = sigmas(
                native_plan,
                sampling_model,
                gen_data["steps"],
                gen_data["width"],
                gen_data["height"],
                denoise,
                scheduler,
            )
            effective["effective_sampling_sigmas"] = fixed_sigmas.tolist()
            gen_data["native_sampling"] = native_plan.metadata() | {"scheduler": scheduler}
        elif ordered_references is not None and ordered_recipe == "flux2-reference":
            from ruinedfooocus.features.image_editing.ordered_native_references import flux2_sigmas

            fixed_sigmas = flux2_sigmas(
                gen_data["steps"], gen_data["width"], gen_data["height"], denoise
            )
            effective["effective_sampling_sigmas"] = fixed_sigmas.tolist()
            gen_data["native_reference_conditioning"]["native_scheduler"] = "Flux2Scheduler"
        if guidance.get("qwen_image21_schedule") == "turbo_8" and denoise not in (None, 1.0):
            raise ValueError(
                "Qwen Turbo 8-step requires full denoise. Select Auto for denoise-based image-to-image."
            )
        if architecture == "QwenImage21" and (denoise is None or denoise == 1.0):
            from ruinedfooocus.features.models.model_recommendations import (
                QWEN_IMAGE21_TURBO_SIGMAS,
                qwen_image21_sigmas,
            )

            schedule = qwen_image21_sigmas(gen_data, guidance.get("qwen_image21_schedule", "auto"))
            if (
                schedule is not None
                and gen_data["steps"] == len(QWEN_IMAGE21_TURBO_SIGMAS) - 1
                and sampler_name == "euler"
                and scheduler == "simple"
                and cfg == 1
            ):
                fixed_sigmas = torch.tensor(QWEN_IMAGE21_TURBO_SIGMAS, dtype=torch.float32)
                effective["effective_sampling_sigmas"] = list(QWEN_IMAGE21_TURBO_SIGMAS)
        positive_cond = bind_prompt_steps(
            positive_cond,
            sampling_model,
            gen_data["steps"],
            sampler_name,
            scheduler,
            denoise,
            sigmas=fixed_sigmas,
        )
        if (
            isinstance(self.xl_base.unet.model, Flux)
            and native_plan is None
            and architecture
            not in (
                "Chroma",
                "ChromaRadiance",
                "LongCatImage",
            )
            and ordered_family not in ("Flux2Klein4B", "Flux2Klein9B")
        ):
            if controlnet.get("type", "") == "kontext":
                positive_cond = ReferenceLatent().execute(positive_cond, latent=latent)[0]
            positive_cond = conditioning_set_values(positive_cond, {"guidance": cfg})
            cfg = 1.0

        if "_native_tensor_plan" in gen_data:
            from ruinedfooocus.features.generation.native_tensor_io import apply_conditioning_inputs

            positive_cond, negative_conditioning = apply_conditioning_inputs(
                self,
                gen_data,
                positive_cond,
                negative_conditioning,
                frame=reference_frame,
            )
        positive_cond, negative_conditioning = lora_hooks.conditioning(
            self, positive_cond, negative_conditioning
        )
        latent_image = latent["samples"]
        latent_image = fix_empty_latent_channels(
            sampling_model,
            latent_image,
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        if (
            architecture == "PixArt"
            and round(((latent_image.shape[-2] + latent_image.shape[-1]) // 2) / 64) == 0
        ):
            raise ValueError(
                "PixArt resolution is too small for its positional embeddings. Use 512x512 or a larger supported resolution."
            )

        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        previewer = (
            get_previewer(device, self.xl_base_patched.unet.model.latent_format)
            if callback is not None and not gen_data.get("silent", False)
            else None
        )

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            y = None
            if previewer:
                try:
                    y = previewer.preview(x0, step, total_steps)
                except:
                    y = None
            if callback is not None:
                callback(step, x0, x, total_steps, y)
            pbar.update_absolute(step + 1, total_steps, None)

        if callback is not None:
            worker.add_result(gen_data["task_id"], "preview", (-1, f"Prepare models ...", None))
            worker.add_result(gen_data["task_id"], "preview", (-1, f"Start sampling ...", None))

        noise_options = None
        patch_scope = None
        if (
            gen_data.get("_native_gligen_layout") is not None
            or gen_data.get("_native_hypernetwork") is not None
        ):
            from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
                apply_scope as patch_scope,
            )
        if sampler_name == "lcm":
            from ruinedfooocus.features.generation.image_sampling import (
                sample_lcm,
                lcm_options,
                LCM_FIELDS,
            )

            noise_options = lcm_options(guidance, architecture, checkpoint)
            gen_data["image_guidance"].update(
                {"effective_" + key: noise_options[native] for key, native in LCM_FIELDS.items()}
            )
        with (patch_scope or model_patches.apply_scope)(
            self,
            sampling_model,
            self.xl_base_patched.vae,
            gen_data,
            path_manager,
            lambda: worker.check_interrupt(gen_data),
        ) as sampling_model:
            if gen_data.get("sdxl_refiner_model"):
                from ruinedfooocus.features.generation.sdxl_refiner import sample_handoff

                samples = sample_handoff(
                    self,
                    model=sampling_model,
                    noise=noise,
                    steps=gen_data["steps"],
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive_cond,
                    negative=negative_conditioning,
                    latent_image=latent_image,
                    data=gen_data,
                    manager=path_manager,
                    check_cancelled=lambda: worker.check_interrupt(gen_data),
                    denoise=denoise,
                    noise_mask=noise_mask,
                    noise_options=noise_options,
                    prompts=(positive_prompt, negative_prompt),
                    callback=callback_function,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=seed,
                )
            elif native_plan is not None:
                from ruinedfooocus.features.generation.native_sampling import (
                    negative_model_scope,
                    sample as sample_native,
                )

                with negative_model_scope(
                    self,
                    native_plan,
                    gen_data,
                    path_manager,
                    lambda: worker.check_interrupt(gen_data),
                ) as negative_model:
                    samples = sample_native(
                        native_plan,
                        model=sampling_model,
                        negative_model=negative_model,
                        noise=noise,
                        cfg=cfg,
                        sampler_name=sampler_name,
                        sigmas=fixed_sigmas,
                        positive=positive_cond,
                        negative=negative_conditioning,
                        latent_image=latent_image,
                        noise_options=noise_options,
                        noise_mask=noise_mask,
                        callback=callback_function,
                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                        seed=seed,
                    )
            elif architecture == "Omnigen2":
                from ruinedfooocus.features.generation.image_sampling import sample_omnigen2

                samples = sample_omnigen2(
                    model=sampling_model,
                    noise=noise,
                    steps=gen_data["steps"],
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive_cond,
                    negative=negative_conditioning,
                    latent_image=latent_image,
                    denoise=denoise,
                    noise_options=noise_options,
                    noise_mask=noise_mask,
                    callback=callback_function,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=seed,
                )
            elif sampler_name == "lcm":
                samples = sample_lcm(
                    model=sampling_model,
                    noise=noise,
                    steps=gen_data["steps"],
                    cfg=cfg,
                    scheduler=scheduler,
                    positive=positive_cond,
                    negative=negative_conditioning,
                    latent_image=latent_image,
                    denoise=denoise,
                    noise_options=noise_options,
                    noise_mask=noise_mask,
                    callback=callback_function,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=seed,
                )
            else:
                samples = comfy.sample.sample(
                    model=sampling_model,
                    noise=noise,
                    steps=gen_data["steps"],
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive_cond,
                    negative=negative_conditioning,
                    latent_image=latent_image,
                    denoise=denoise,
                    start_step=0,
                    last_step=gen_data["steps"],
                    sigmas=fixed_sigmas,
                    noise_mask=noise_mask,
                    callback=callback_function,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=seed,
                )

        # VAE
        sampled_latent = latent.copy()
        sampled_latent.pop("downscale_ratio_spacial", None)
        sampled_latent.pop("downscale_ratio_temporal", None)
        sampled_latent["samples"] = samples
        if "_native_tensor_plan" in gen_data and (
            gen_data.get("native_export_latent") or gen_data.get("native_export_conditioning")
        ):
            from ruinedfooocus.features.generation.native_tensor_io import export_tensors

            export_tensors(
                self,
                gen_data,
                sampled_latent,
                positive_cond,
                negative_conditioning,
                stage_b=cascade_latents.stage_b if cascade_latents is not None else None,
                frame=reference_frame,
            )
        if layered_model:
            sampled_latent = layered.layer_latent(layered_plan, sampled_latent)

        if callback is not None:
            worker.add_result(gen_data["task_id"], "preview", (-1, f"VAE decoding ...", None))

        if architecture == "StableCascade_C":
            from ruinedfooocus.features.generation.cascade import decode_stage_b

            decoded_latent = decode_stage_b(
                self,
                sampled_latent,
                gen_data,
                path_manager,
                seed,
                stage_b_latent=cascade_latents.stage_b if cascade_latents is not None else None,
                check_cancelled=lambda: worker.check_interrupt(gen_data),
            )
        else:
            decoded_latent = self.decode_latent(
                sampled_latent,
                tiled=guidance["vae_tiled"],
                tile_size=guidance["vae_tile_size"],
                overlap=guidance["vae_overlap"],
            )
        if sd4x_model:
            from ruinedfooocus.features.image_editing.sd4x_conditioning import restore_sd4x_alpha

            decoded_latent = restore_sd4x_alpha(decoded_latent, gen_data["input_image"])
        # Save feedback only when explicitly requested, before output upscaling.
        if img2img_mode and controlnet.get("loopback", False):
            gen_data["_loopback_image"] = np.clip(
                255.0 * decoded_latent[0].cpu().numpy(), 0, 255
            ).astype(np.uint8)
        if qwen_references is not None:
            decoded_latent = qwen_references.finish(decoded_latent)
        if native_control_recipes.has_source(gen_data):
            decoded_latent = native_control_recipes.finish(
                decoded_latent, gen_data, reference_frame
            )
        else:
            decoded_latent = restore_reference_frame(decoded_latent, reference_frame)
        if guided_edit:
            decoded_latent = guide_handoffs.restore_source_alpha(
                decoded_latent, control_source_alpha, reference_frame
            )
        if reference_frame is not None:
            if outpaint_composite is not None:
                from ruinedfooocus.features.image_editing.inpaint_utils import blend_outpaint

                decoded_latent = blend_outpaint(
                    decoded_latent, *outpaint_composite, reference_frame
                )
            gen_data["width"], gen_data["height"] = (
                reference_frame["width"],
                reference_frame["height"],
            )
            gen_data["image_geometry"] = reference_frame
        from ruinedfooocus.features.generation.automatic_upscale import upscale_decoded

        if layered_model and isinstance(size_plan, SizePlan):
            from ruinedfooocus.features.generation.size_plan import fit_tensor

            decoded_latent = fit_tensor(decoded_latent, size_plan.content)
        decoded_latent = upscale_decoded(self, decoded_latent, gen_data)
        from ruinedfooocus.features.image_processing.native_image_formats import pipeline_images

        images = pipeline_images(decoded_latent, gen_data)

        if callback is not None:
            preview = np.clip(255.0 * decoded_latent[0].cpu().numpy(), 0, 255).astype(np.uint8)
            callback(gen_data["steps"], 0, 0, gen_data["steps"], preview)

        return layered.outputs(gen_data, images) if layered_model else images
