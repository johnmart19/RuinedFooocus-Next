"""Read saved rendering settings without importing worker state or image handles."""

import json
from ruinedfooocus.features.image_editing import model_patches, native_control_recipes
from ruinedfooocus.features.image_editing import lora_hooks

from ruinedfooocus.features.image_processing.still_image_utilities import ADJUSTMENT_DEFAULTS
from ruinedfooocus.features.image_processing.depth_utilities import EXTRA_UTILITY_DEFAULTS

RENDER_FIELDS = frozenset(
    {
        "prompt",
        "grid_columns",
        "grid_cell_width",
        "grid_cell_height",
        "grid_padding",
        "composition_mode",
        "composition_x",
        "composition_y",
        "composition_width",
        "composition_height",
        "native_composition_source",
        "negative",
        "raw_prompt",
        "image_edit_prompt",
        "raw_negative",
        "style_provenance",
        "steps",
        "custom_steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "clip_skip",
        "output_format",
        "output_bit_depth",
        "output_color_space",
        "output_crf",
        "output_source_dtype",
        "model_recipe_path",
        "model_recipe_sha256",
        "model_recipe_overrides",
        "clip_layer_source",
        "reference_adapter",
        "reference_adapter_model",
        "reference_adapter_vision",
        "reference_adapter_strength",
        "reference_adapter_info",
        *ADJUSTMENT_DEFAULTS,
        *EXTRA_UTILITY_DEFAULTS,
        *model_patches.FIELDS,
        *native_control_recipes.FIELDS,
        *lora_hooks.FIELDS,
        "native_lora_hooks",
        "native_model_patch",
        "native_control_recipe",
        "cascade_stage_b",
        "cascade_b_steps",
        "cascade_b_cfg",
        "cascade_b_sampler",
        "cascade_b_scheduler",
        "cascade_stage_b_source",
        "cascade_compression",
        "cascade_source_denoise",
        "native_cascade",
        "cascade_super_resolution",
        "cascade_super_resolution_model",
        "cascade_control_strength",
        "cascade_control_start",
        "cascade_control_end",
        "cascade_control_denoise",
        "cascade_vision_enabled",
        "cascade_vision_model",
        "cascade_vision_strengths",
        "cascade_vision_crop",
        "native_cascade_guidance",
        "sdxl_refiner_model",
        "sdxl_refiner_switch_step",
        "sdxl_refiner_cfg",
        "sdxl_refiner_clip",
        "sdxl_refiner_source",
        "refiner_noise_augmentation",
        "pid_input_vae",
        "pid_latent_format",
        "pid_degrade_sigma",
        "pid_conditioning",
        "width",
        "height",
        "custom_width",
        "custom_height",
        "aspect_ratios_selection",
        "seed",
        "base_model_name",
        "base_model_hash",
        "loras",
        "start_step",
        "denoise",
        "sampling_denoise",
        "controlnet",
        "inpaint_toggle",
        "inpaint_denoise",
        "inpaint_mask_grow",
        "qwen_local_edit_mode",
        "rembg_force",
        "rembg_native_model",
        "native_background_source",
        "native_vision_model",
        "vision_threshold",
        "detect_class",
        "detect_max_count",
        "segment_positive_points",
        "segment_negative_points",
        "segment_refine_iterations",
        "segment_output",
        "segment_text",
        "segment_boxes",
        "segment_individual_masks",
        "segment_max_instances",
        "native_vision_source",
        "seedvr2_model",
        "seedvr2_vae",
        "seedvr2_scale",
        "seedvr2_color",
        "native_seedvr2_source",
        "native_tensor_source",
        "native_tensor_family",
        "native_tensor_workflow",
        "native_latent_denoise",
        "native_export_latent",
        "native_export_conditioning",
        "native_export_handoff",
        "image_utility_source",
        "wevisone_query",
        "wevisone_max_length",
        "wevisone_output",
        "curve_master",
        "curve_red",
        "curve_green",
        "curve_blue",
        "curve_interpolation",
        "shader_effect",
        "shader_amount",
        "shader_levels",
        "filter_radius",
        "filter_sigma",
        "filter_strength",
        "filter_factor",
        "moge_resolution_level",
        "moge_refine_steps",
        "face_detector_variant",
        "face_max_count",
        "face_min_confidence",
        "pose_score_threshold",
        "automatic_upscale",
        "automatic_upscale_target",
        "generation_width",
        "generation_height",
        "size_plan",
        "size_mode",
        "upscale_source_size",
        "upscale_source",
        "upscaler",
        "image_guidance",
        "image_component_overrides",
        "image_geometry",
        "depth_preprocessing",
        "reference_image_count",
        "native_edit_recipe",
        "krea_style_lora",
        "krea_style_strength",
        "native_krea_style",
        "native_reference_conditioning",
        "native_inpaint_conditioning",
        "native_sampling",
        "qwen_layer_count",
        "native_layered_conditioning",
        "native_minimax_image",
        "native_bernini_image",
        "bernini_source_frame",
        "bernini_low_model",
        "bernini_split_step",
        "bernini_lightx2v",
        "bernini_high_lora_strength",
        "bernini_low_lora_strength",
        "bernini_ref_max_size",
        "native_component_layout",
        "gligen_model",
        "gligen_model_signature",
        "gligen_layout",
        "gligen_guidance",
        "hypernetwork_model",
        "hypernetwork_model_signature",
        "hypernetwork_strength",
        "hypernetwork_guidance",
        "native_negative_model",
        "native_schedule_alpha",
        "native_schedule_beta",
        "native_schedule_mu",
        "native_schedule_std",
        "native_output_group",
        "native_image_output",
        "image_source",
        "sampling_shift_supported",
        "comment",
        "software",
        "frames",
        "fps",
    }
)


def _metadata_document(value):
    if isinstance(value, str):
        if not value.lstrip().startswith("{"):
            return None
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return {key.lower(): item for key, item in value.items()}


def generation_provenance(value):
    """Keep a bounded inert rendering recipe, never job controls or source paths."""
    metadata = _metadata_document(value)
    if metadata is None or not isinstance(metadata.get("prompt"), str):
        return None
    from ruinedfooocus.features.gallery.editor_session import request_parameters

    try:
        values = render_values(metadata)
        # Captured source handles/paths are not reusable authority in a derived PNG.
        values = {
            key: item
            for key, item in values.items()
            if key
            not in (
                "image_source",
                "native_composition_source",
                "native_tensor_source",
                "image_utility_source",
            )
            and not key.endswith(("_path", "_paths"))
        }
        return request_parameters(values)
    except (TypeError, ValueError):
        return None


def read_metadata(value):
    metadata = _metadata_document(value)
    if metadata is None:
        return None
    if "prompt" in metadata:
        return metadata
    if metadata.get("kind") == "processing":
        values = generation_provenance(metadata.get("source_generation"))
        if values is None:
            return None
        # Source sampling facts stay nested; replay targets the current artifact
        # without reapplying its source's old size plan or automatic upscale.
        for key in (
            "width",
            "height",
            "custom_width",
            "custom_height",
            "generation_width",
            "generation_height",
            "size_plan",
            "size_mode",
            "aspect_ratios_selection",
            "upscale_source",
            "upscale_source_size",
            "upscaler",
        ):
            values.pop(key, None)
        values.update(automatic_upscale="Off", automatic_upscale_target="Generated image")
        from ruinedfooocus.features.generation.size_plan import dimension
        from ruinedfooocus.features.image_processing.native_image_formats import MAX_PIXELS

        try:
            width, height = (dimension(metadata.get(key)) for key in ("width", "height"))
            if width * height > MAX_PIXELS:
                raise ValueError("Invalid processing output geometry.")
        except (TypeError, ValueError):
            return values
        return dict(values, width=width, height=height)
    return None


def render_values(metadata):
    from ruinedfooocus.features.image_editing.image_guidance import FIELD_SCHEMA
    from ruinedfooocus.features.image_editing.qwen_image21_local_edit import replay_values

    values = {
        key: value
        for key, value in metadata.items()
        if key in RENDER_FIELDS or key in FIELD_SCHEMA or key.startswith(("cn_", "effective_"))
    }
    values.update(replay_values(metadata))
    from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
        replay_values as replay_auxiliary,
    )

    if metadata.get("gligen_model") or metadata.get("hypernetwork_model"):
        values.update(replay_auxiliary(metadata))
    return values


def control_value(key, value):
    if key == "hidream_o1_seam_smoothing":
        if value is None:
            return "auto"
        if value is True:
            return "on"
        if value is False:
            return "off"
    return value


def restore_prompt_provenance(values, imported):
    """Replay a rendered prompt literally until the user requests new processing."""
    from ruinedfooocus.features.prompting.style_catalog import normalize_selection

    if normalize_selection(values.get("style_selection")) or values.get("auto_negative"):
        return
    metadata = read_metadata(imported)
    if metadata is None:
        if isinstance(imported, str) and values.get("prompt") == imported:
            values["_metadata_prompt"] = True
        return
    if (values.get("prompt"), values.get("negative")) != (
        metadata.get("prompt"),
        metadata.get("negative", ""),
    ):
        return
    values["_metadata_prompt"] = True
    for key in ("raw_prompt", "raw_negative", "style_provenance"):
        if key in metadata:
            values[key] = metadata[key]
    if "native_minimax_image" in metadata and values.get("base_model_name") == metadata.get(
        "base_model_name"
    ):
        from copy import deepcopy

        values["native_minimax_image"] = deepcopy(metadata["native_minimax_image"])
    if "native_bernini_image" in metadata and values.get("base_model_name") == metadata.get(
        "base_model_name"
    ):
        from copy import deepcopy

        values["native_bernini_image"] = deepcopy(metadata["native_bernini_image"])
