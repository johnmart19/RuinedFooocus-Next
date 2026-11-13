from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui
import gradio as gr
import json
from pathlib import Path
from PIL import Image
from ruinedfooocus.app.state import translate, get_ctrl
from ruinedfooocus.features.gallery.imagebrowser import (
    ImageBrowser,
    get_png_metadata,
    get_media_metadata,
    format_metadata_string,
)
from ruinedfooocus.features.gallery.actions import select_source, source_path, resolve_setup
from ruinedfooocus.features.gallery.image_metadata import read_metadata
from ruinedfooocus.features.gallery.index_store import MEDIA_SUFFIXES
from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    capabilities,
    restore_guidance_metadata,
    resolve_guidance,
)

browser = ImageBrowser()
t = translate_ui

# One browser-tab counter is shared by navigation, search and refresh acknowledgements.
# It is an ordering token only; authenticated request context owns the server fence.
request_version_js = """(page, term, version) => {
    window.__rfLibraryRequest = (window.__rfLibraryRequest || 0) + 1;
    return [page, term, window.__rfLibraryRequest];
}"""

METADATA_CONTROLS = {
    "grid_columns": "grid_columns",
    "grid_cell_width": "grid_cell_width",
    "grid_cell_height": "grid_cell_height",
    "grid_padding": "grid_padding",
    "composition_mode": "composition_mode",
    "composition_x": "composition_x",
    "composition_y": "composition_y",
    "composition_width": "composition_width",
    "composition_height": "composition_height",
    "prompt": "prompt",
    "negative": "negative",
    "steps": "custom_steps",
    "cfg": "cfg",
    "sampler_name": "sampler_name",
    "scheduler": "scheduler",
    "clip_skip": "clip_skip",
    "gligen_model": "gligen_model",
    "gligen_model_signature": "gligen_model_signature",
    "gligen_layout": "gligen_layout",
    "hypernetwork_model": "hypernetwork_model",
    "hypernetwork_model_signature": "hypernetwork_model_signature",
    "hypernetwork_strength": "hypernetwork_strength",
    "cascade_stage_b": "cascade_stage_b",
    "cascade_b_steps": "cascade_b_steps",
    "cascade_b_cfg": "cascade_b_cfg",
    "cascade_b_sampler": "cascade_b_sampler",
    "cascade_b_scheduler": "cascade_b_scheduler",
    "cascade_compression": "cascade_compression",
    "cascade_source_denoise": "cascade_source_denoise",
    "sdxl_refiner_model": "sdxl_refiner_model",
    "sdxl_refiner_switch_step": "sdxl_refiner_switch_step",
    "sdxl_refiner_cfg": "sdxl_refiner_cfg",
    "sdxl_refiner_clip": "sdxl_refiner_clip",
    "refiner_noise_augmentation": "refiner_noise_augmentation",
    "pid_input_vae": "pid_input_vae",
    "pid_latent_format": "pid_latent_format",
    "pid_degrade_sigma": "pid_degrade_sigma",
    "qwen_local_edit_mode": "qwen_local_edit_mode",
    "width": "custom_width",
    "height": "custom_height",
    "size_mode": "size_mode",
    "seed": "seed",
}


def apply_metadata(metadata, checkpoint, preset):
    if preset:
        raise gr.Error("Clear the active preset in Create before applying image metadata.")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            raise gr.Error("Select an image with valid generation metadata first.") from None
    if not isinstance(metadata, dict):
        raise gr.Error("Select an image with generation metadata first.")
    data = {key.lower(): value for key, value in metadata.items()}
    if data.get("native_composition_source"):
        return composition_setup(data)
    if "prompt" not in data:
        raise gr.Error("This file has no saved prompt.")
    if not isinstance(data["prompt"], str):
        raise gr.Error("Image metadata needs a text prompt.")
    from ruinedfooocus.features.image_processing import still_image_utilities as depth_utilities

    if depth_utilities.is_utility(depth_utilities.kind_from(data)):
        try:
            values = depth_utilities.replay_values(data)
        except ValueError as error:
            raise gr.Error(str(error)) from None
        values["_imported_prompt"] = None
        return {
            get_ctrl(key): gr.update(value=value)
            for key, value in values.items()
            if get_ctrl(key) is not None
        }
    data.setdefault("negative", "")
    if not isinstance(data["negative"], str):
        raise gr.Error("The saved negative prompt must be text.")
    from ruinedfooocus.features.image_editing.qwen_image21_local_edit import (
        replay_values as replay_local_edit,
    )

    try:
        data.update(replay_local_edit(data))
        from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
            replay_values as replay_auxiliary,
        )

        data.update(replay_auxiliary(data))
    except ValueError as error:
        raise gr.Error(str(error)) from None
    # Saved width/height describe the final file; sampling must use the native size.
    from ruinedfooocus.features.generation.size_plan import replay_size

    data["width"], data["height"] = replay_size(data)
    data["size_mode"] = "target" if "size_plan" in data else "legacy_base"
    updates = {
        get_ctrl(target): gr.update(value=data[key])
        for key, target in METADATA_CONTROLS.items()
        if data.get(key) is not None or key.startswith(("gligen_", "hypernetwork_"))
    }
    updates[get_ctrl("_imported_prompt")] = data["prompt"]
    from ruinedfooocus.features.image_editing.control_candidates import (
        replay_values as replay_control,
    )

    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in replay_control(data).items()}
    )
    from ruinedfooocus.features.image_editing.qwen_image21_union import (
        replay_values as replay_union,
    )

    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in replay_union(data).items()}
    )
    from ruinedfooocus.features.models.yaml_recipes import replay_values

    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in replay_values(data).items()}
    )
    from ruinedfooocus.features.image_editing.reference_adapters import (
        replay_values as replay_reference,
    )

    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in replay_reference(data).items()}
    )
    updates[get_ctrl("reference_adapter_image")] = gr.update(value=None)
    from ruinedfooocus.features.generation.cascade_guidance import (
        replay_values as replay_cascade_guidance,
    )

    updates.update(
        {
            get_ctrl(key): gr.update(value=value)
            for key, value in replay_cascade_guidance(data).items()
        }
    )
    updates[get_ctrl("cascade_vision_images")] = gr.update(value=[])
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

    family = family_for_checkpoint(checkpoint)
    guidance = restore_guidance_metadata(data, family=family)
    from ruinedfooocus.features.image_editing import lora_hooks

    try:
        updates.update(
            {
                get_ctrl(key): gr.update(value=value)
                for key, value in lora_hooks.replay_values(data).items()
            }
        )
    except ValueError as error:
        raise gr.Error(str(error)) from None
    if not capabilities(family)["sampling_shift"]:
        guidance["sampling_shift"] = None
    from ruinedfooocus.features.gallery.image_metadata import control_value

    updates.update(
        {
            get_ctrl(key): gr.update(value=control_value(key, value))
            for key, value in resolve_guidance(guidance, {}, family, checkpoint).items()
        }
    )
    updates[get_ctrl("performance_selection")] = gr.update(value="Own settings")
    updates[get_ctrl("style_selection")] = gr.update(value=[])
    updates[get_ctrl("obp_assume_direct_control")] = gr.update(value=False)
    updates[get_ctrl("obp_image_prompt")] = gr.update(value=False)
    updates[get_ctrl("auto_negative")] = gr.update(value=False)
    if data.get("width") and data.get("height"):
        updates[get_ctrl("aspect_ratios_selection")] = gr.update(value="Custom...")
    if data.get("seed") is not None:
        from ruinedfooocus.features.generation.ui_seed import seed_visibility

        updates[get_ctrl("seed_random")] = gr.update(value=False)
        updates[get_ctrl("seed")].update(seed_visibility(False))
    from ruinedfooocus.features.generation.automatic_upscale import choices, PRESETS

    upscale = data.get("automatic_upscale", "Off")
    if upscale not in choices():
        upscale = "Off"
    updates[get_ctrl("automatic_upscale")] = gr.update(choices=choices(), value=upscale)
    updates[get_ctrl("advanced_upscalers")] = gr.update(value=upscale not in PRESETS)
    updates[get_ctrl("automatic_upscale_target")] = gr.update(value="Generated image")
    from ruinedfooocus.features.image_processing.ui_native_formats import (
        replay_values as replay_format,
    )

    updates.update(
        {
            get_ctrl(key): update
            for key, update in replay_format(data).items()
            if get_ctrl(key) is not None
        }
    )
    return updates


REFERENCE_RESETS = {
    "composition_images": [],
    "composition_mask": None,
    "native_latent_input": None,
    "native_positive_input": None,
    "native_negative_input": None,
    "native_tensor_family": "auto",
    "native_tensor_workflow": False,
    "native_latent_denoise": 0.5,
    "native_export_latent": False,
    "native_export_conditioning": False,
    "native_export_handoff": False,
    "image_utility_model": None,
    "image_utility_vae": None,
    "image_utility_lora": None,
    "image_utility_conditioning": None,
    "wevisone_model": None,
    "wevisone_query": "person",
    "wevisone_max_length": 4096,
    "wevisone_output": "cutout",
    "input_image": None,
    "qwen_union_guide": None,
    "qwen_union_source": None,
    "qwen_union_mask": None,
    "reference_images": [],
    "inpaint_view": None,
    "inpaint_toggle": False,
    "outpaint_left": 0,
    "outpaint_top": 0,
    "outpaint_right": 0,
    "outpaint_bottom": 0,
    "upscale_upload": None,
    "cn_selection": "None",
    "obp_assume_direct_control": False,
    "obp_image_prompt": False,
    "reference_adapter_image": None,
    "native_edit_recipe": "auto",
    "krea_style_lora": "krea2_style_reference.safetensors",
    "krea_style_strength": 1.0,
    "model_patch_recipe": "none",
    "model_patch_image": None,
    "model_patch_inpaint_image": None,
    "model_patch_mask": None,
    "control_recipe": "ordinary",
    "control_inpaint_image": None,
    "control_inpaint_mask": None,
    "gligen_model": None,
    "gligen_model_signature": None,
    "gligen_layout": "[]",
    "hypernetwork_model": None,
    "hypernetwork_model_signature": None,
    "hypernetwork_strength": 1.0,
    "lora_hook_mode": "off",
    "segment_positive_points": "[]",
    "segment_negative_points": "[]",
    "_native_main_source": None,
    "_generation_artifacts": [],
    "_generation_originals": [],
    "_generation_gallery": [],
    "image_source": None,
}


def composition_setup(data):
    """Restore scalar recipe only; saved pixel identities are never upload paths."""
    from ruinedfooocus.features.image_processing import image_composition as composition
    from ruinedfooocus.features.image_editing.tool_activation import activation_values
    from ruinedfooocus.features.image_editing import controlnet
    from ruinedfooocus.features.image_processing.ui_native_formats import replay_values

    source = data.get("native_composition_source")
    if not isinstance(source, dict) or not composition.handles(source.get("kind")):
        raise gr.Error("This image has no valid image composition recipe.")
    try:
        values = dict(REFERENCE_RESETS, **composition.replay_values(data))
        values.update(
            activation_values(
                source["kind"], presets=controlnet.cn_options, custom=controlnet.NEWCN
            )
        )
    except ValueError as error:
        raise gr.Error(str(error)) from None
    values.update(preset_selection=None, automatic_upscale="Off")
    updates = {get_ctrl(key): gr.update(value=value) for key, value in values.items()}
    updates.update(
        {
            get_ctrl(key): value
            for key, value in replay_values(data).items()
            if get_ctrl(key) is not None
        }
    )
    return updates


def reuse_prompt(metadata, original=False):
    data = read_metadata(metadata)
    key = "raw_prompt" if original else "prompt"
    negative_key = "raw_negative" if original else "negative"
    if data is None or not isinstance(data.get(key), str):
        raise gr.Error(
            "This image has no original prompt. Choose the rendered prompt instead."
            if original
            else "This image has no saved prompt."
        )
    negative = data.get(negative_key, "")
    if not isinstance(negative, str):
        raise gr.Error("The saved negative prompt must be text.")
    updates = {
        get_ctrl("prompt"): gr.update(value=data[key]),
        get_ctrl("negative"): gr.update(value=negative),
        get_ctrl("_imported_prompt"): data if not original else None,
        get_ctrl("style_selection"): gr.update(value=[]),
        get_ctrl("obp_assume_direct_control"): gr.update(value=False),
        get_ctrl("obp_image_prompt"): gr.update(value=False),
        get_ctrl("auto_negative"): gr.update(value=False),
    }
    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in REFERENCE_RESETS.items()}
    )
    updates[get_ctrl("automatic_upscale_target")] = gr.update(value="Generated image")
    return updates


def reuse_setup(metadata, *, tensor_inputs=None):
    data = read_metadata(metadata)
    if data and data.get("native_composition_source"):
        return composition_setup(data)
    from ruinedfooocus.features.image_processing import still_image_utilities as depth_utilities

    utility = read_metadata(metadata)
    if utility and depth_utilities.is_utility(depth_utilities.kind_from(utility)):
        return apply_metadata(utility, None, None)
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.core.util import get_lora_thumbnail, load_keywords
    from ruinedfooocus.core.settings_validation import validate_settings

    try:
        data, checkpoint, loras = resolve_setup(metadata, shared.models)
        from ruinedfooocus.features.generation.native_tensor_io import (
            replay_values as replay_tensors,
        )

        tensor_values = replay_tensors(data, tensor_inputs or {})
        sampling = {
            target: data.get(f"generation_{key}", data[key])
            if key in ("width", "height")
            else data[key]
            for key, target in METADATA_CONTROLS.items()
            if key in data
        }
        validated = validate_settings(sampling)
        for key, target in METADATA_CONTROLS.items():
            if key in data:
                field = (
                    f"generation_{key}"
                    if key in ("width", "height") and f"generation_{key}" in data
                    else key
                )
                data[field] = validated[target]
    except ValueError as error:
        raise gr.Error(str(error)) from None
    for field in ("sampler_name", "scheduler"):
        if field in data:
            values = [
                choice[1] if isinstance(choice, (tuple, list)) else choice
                for choice in get_ctrl(field).choices
            ]
            if data[field] not in values:
                raise gr.Error(f"The saved {field} is unavailable: {data[field]}")
    updates = apply_metadata(data, checkpoint, None)
    updates[get_ctrl("_imported_prompt")] = data
    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in REFERENCE_RESETS.items()}
    )
    from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
        replay_values as replay_auxiliary,
    )

    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in replay_auxiliary(data).items()}
    )
    from ruinedfooocus.features.image_editing.qwen_image21_union import (
        replay_values as replay_union,
    )

    updates.update(
        {get_ctrl(key): gr.update(value=value) for key, value in replay_union(data).items()}
    )
    from ruinedfooocus.features.image_editing.model_patches import replay_values as replay_patch
    from ruinedfooocus.features.image_editing.native_control_recipes import (
        replay_values as replay_control_recipe,
    )
    from ruinedfooocus.features.image_editing.lora_hooks import replay_values as replay_lora_hooks

    try:
        for replay in (replay_patch, replay_control_recipe, replay_lora_hooks):
            updates.update(
                {get_ctrl(key): gr.update(value=value) for key, value in replay(data).items()}
            )
    except ValueError as error:
        raise gr.Error(str(error)) from None
    recipe = data.get("native_edit_recipe", "auto")
    if recipe == "krea2-style":
        from ruinedfooocus.features.image_editing.krea_style import values as krea_values

        style = krea_values(data)
        updates[get_ctrl("krea_style_lora")] = gr.update(value=style.name)
        updates[get_ctrl("krea_style_strength")] = gr.update(value=style.strength)
    if recipe != "auto":
        from ruinedfooocus.features.image_editing.ui_ordered_references import recipe_choices
        from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

        available_recipes = recipe_choices(family_for_checkpoint(checkpoint))
        if recipe not in {value for _, value in available_recipes}:
            raise gr.Error("The saved native edit recipe is unavailable for this checkpoint.")
        updates[get_ctrl("native_edit_recipe")] = gr.update(choices=available_recipes, value=recipe)
    updates[get_ctrl("preset_selection")] = gr.update(value=None)
    updates[get_ctrl("base_model_name")] = gr.update(value=checkpoint)
    from ruinedfooocus.features.models.component_profiles import replay_choices

    component_choices, component_scope = replay_choices(
        shared.models.get_file("checkpoints", checkpoint),
        shared.settings.default_settings,
        data.get("image_component_overrides"),
    )
    updates[get_ctrl("image_component_overrides")] = component_choices
    updates[get_ctrl("_image_component_scope")] = component_scope
    updates[get_ctrl("loras")] = gr.update(
        value=[(get_lora_thumbnail(name), f"{weight} - {name}") for name, weight in loras]
    )
    updates[get_ctrl("lora_keywords")] = gr.update(
        value=", ".join(keyword for name, _ in loras if (keyword := load_keywords(name)).strip())
    )
    from ruinedfooocus.features.generation.native_tensor_io import INPUTS

    required = {"latent": INPUTS[0], "positive": INPUTS[1], "negative": INPUTS[2]}
    for entry in data.get("native_tensor_source", {}).get("inputs", []):
        updates.pop(get_ctrl(required[entry["role"]]), None)
    updates.update(
        {
            get_ctrl(key): gr.update(value=value)
            for key, value in tensor_values.items()
            if get_ctrl(key) is not None
        }
    )
    return updates


def prepare_image(
    record,
    action,
    checkpoint,
    moss_format,
    upscale,
    current_selection="None",
    control_type="",
    strength=0.25,
):
    from ruinedfooocus.features.image_editing.ui import (
        checkpoint_features,
        edit_tool,
        tool_kind,
        controlnet,
    )
    from ruinedfooocus.features.image_editing.inpaint_utils import editor_payload
    from ruinedfooocus.features.image_editing.tool_activation import activation_values

    try:
        from ruinedfooocus.features.gallery.editing_workflow import working_source

        image, details = working_source(record)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from None
    from ruinedfooocus.features.image_processing.image_composition import handles

    features = {} if handles(action) else checkpoint_features(checkpoint, moss_format)
    values = dict(
        REFERENCE_RESETS,
        main_view=image,
        image_source=details,
        automatic_upscale_target="Generated image",
    )
    if action == "edit":
        tool = edit_tool(features)
        if tool is None:
            raise gr.Error("Select an image checkpoint in Create before editing this image.")
        values.update(
            activation_values(
                "edit" if features.get("native_edit") else "img2img",
                presets=controlnet.cn_options,
                custom=getattr(controlnet, "NEWCN", "Custom..."),
                features=features,
            ),
            input_image=image.copy(),
        )
        if tool_kind(tool, "") == "img2img":
            options = controlnet.cn_options.get(tool, {})
            values["cn_strength"] = (
                strength
                if tool_kind(current_selection, control_type) == "img2img"
                else options.get("denoise", options.get("strength", 0.25))
            )
    elif action == "inpaint":
        if not features["inpaint"]:
            raise gr.Error("Select a checkpoint with a supported inpainting route.")
        values.update(
            inpaint_toggle=True,
            inpaint_view=editor_payload(image, preserve_alpha=bool(features.get("qwen_local"))),
        )
    elif action == "upscale":
        values.update(
            upscale_upload=image.copy(),
            automatic_upscale_target="Input image only",
            automatic_upscale=upscale if upscale != "Off" else "2× General (RealESRGAN)",
        )
    elif action == "rembg":
        values.update(
            activation_values(
                "rembg",
                presets=controlnet.cn_options,
                custom=getattr(controlnet, "NEWCN", "Custom..."),
                features=features,
            ),
            input_image=image.copy(),
        )
    elif action in ("image-grid", "image-composite"):
        values.update(
            activation_values(
                action,
                presets=controlnet.cn_options,
                custom=getattr(controlnet, "NEWCN", "Custom..."),
            ),
            composition_images=[image.copy()],
            composition_mask=None,
        )
    else:
        raise gr.Error("Unknown image action.")
    return {get_ctrl(key): gr.update(value=value) for key, value in values.items()}


def create_image_gallery():
    import asyncio
    from concurrent.futures import CancelledError
    from ruinedfooocus.app.preparation_admission import preparation_admission
    from ruinedfooocus.app.resource_lifecycle import current_owner, latest_ui_requests

    with gr.Blocks() as app_image_browser:
        # Opening Gallery requests its bounded refresh; hidden pages prepare no images.
        page_records = gr.State([])
        selected = gr.State(None)
        view_request = gr.JSON(value={}, visible=False)
        selection_request = gr.JSON(value={}, visible=False)
        request_version = gr.Number(value=0, visible=False, precision=0)
        with gr.Row(elem_id="browser-layout"):
            with gr.Column(scale=2, min_width=0, elem_id="browser-images"):
                gallery = gr.Gallery(
                    buttons=["download", "download_all", "fullscreen"],
                    label=t("Images"),
                    show_label=False,
                    columns=4,
                    fit_columns=False,
                    preview=False,
                    height="var(--rf-gallery-height)",
                    elem_id="output-browser",
                    object_fit="contain",
                    value=[],
                )
                pages = browser.num_images_pages()[1]
                with gr.Row(elem_id="browser-pagination", visible=True) as pagination:
                    previous_page = gr.Button(
                        t("Previous"), scale=0, min_width=80, interactive=False
                    )
                    ib_page = gr.Number(
                        label=t("Page"),
                        value=1,
                        step=1,
                        precision=0,
                        minimum=1,
                        maximum=pages,
                        min_width=80,
                        scale=0,
                        elem_id="browser-page",
                        container=False,
                        show_label=False,
                    )
                    page_total = gr.Markdown(f"1 / {pages}", elem_id="browser-page-total")
                    next_page = gr.Button(t("Next"), scale=0, min_width=80, interactive=pages > 1)
                ib_range = gr.Markdown("", elem_id="browser-image-count")
            with gr.Column(scale=1, min_width=0, elem_id="browser-controls") as browser_controls:
                with gr.Row(elem_id="browser-actions"):
                    # min_width lets the row wrap on phones instead of squeezing labels.
                    update_btn = gr.Button(t("Refresh images"), min_width=120)
                    cancel_refresh_btn = gr.Button(t("Cancel refresh"), min_width=120)
                    continue_btn = gr.Button(translate_ui("Continue in Create"), min_width=120)
                    gr.Button(
                        translate_ui("Slideshow"),
                        link="gradio_api/file=ruinedfooocus/ui/assets/slideshow.html",
                        link_target="_blank",
                        min_width=120,
                        elem_id="browser-slideshow",
                    )
                prompt_version = gr.Radio(
                    [
                        (translate_ui("Rendered prompt"), "Rendered prompt"),
                        (translate_ui("Original prompt"), "Original prompt"),
                    ],
                    value="Rendered prompt",
                    label=translate_ui("Prompt to reuse"),
                )
                with gr.Row(elem_classes=["rf-actions"]):
                    prompt_btn = gr.Button(translate_ui("Use prompt"), interactive=False)
                    setup_btn = gr.Button(translate_ui("Reuse full setup"), interactive=False)
                with gr.Row(elem_classes=["rf-actions"]):
                    open_editor_btn = gr.Button(t("Open in Editor"), interactive=False)
                    edit_btn = gr.Button(translate_ui("Edit image"), interactive=False)
                    inpaint_btn = gr.Button(translate_ui("Inpaint"), interactive=False)
                with gr.Row(elem_classes=["rf-actions"]):
                    upscale_btn = gr.Button(translate_ui("Upscale"), interactive=False)
                    rembg_btn = gr.Button(translate_ui("Remove background"), interactive=False)
                with gr.Row(elem_classes=["rf-actions"]):
                    grid_btn = gr.Button(t("Start image grid"), interactive=False)
                    composite_btn = gr.Button(t("Start image composition"), interactive=False)
                metadata_output = gr.Textbox(
                    label=t("Image Metadata"),
                    interactive=False,
                    lines=8,
                    max_lines=14,
                    elem_id="browser-metadata",
                )
                search_input = gr.Textbox(
                    label=t("Search Metadata"), placeholder=t("Search term"), value=""
                )
                search_btn = gr.Button(t("Search"))
                status_output = gr.Markdown()
                metadata_json = gr.JSON(value={}, visible=False)

                from ruinedfooocus.app import state as shared
                from ruinedfooocus.features.image_processing.ui_native_formats import (
                    create_library_ui,
                )

                export_btn = create_library_ui(selected, shared.path_manager)

        action_buttons = [
            prompt_btn,
            setup_btn,
            edit_btn,
            inpaint_btn,
            upscale_btn,
            rembg_btn,
            grid_btn,
            composite_btn,
            export_btn,
            open_editor_btn,
        ]
        navigation_controls = [update_btn, ib_page, search_btn, search_input]
        refresh_outputs = list(
            dict.fromkeys(
                [
                    gallery,
                    ib_page,
                    status_output,
                    page_records,
                    ib_range,
                    pagination,
                    page_total,
                    previous_page,
                    next_page,
                    selected,
                    metadata_json,
                    metadata_output,
                    *action_buttons,
                    *navigation_controls,
                ]
            )
        )

        def lock_selection(trigger, reset=False):
            # Client-only updates block stale actions before a server round trip.
            gallery_update = {"__type__": "update", "elem_classes": ["rf-library-pending"]}
            if reset:
                gallery_update["selected_index"] = None
                gallery_update["preview"] = False
            locked_controls = [*action_buttons]
            trigger(
                fn=None,
                outputs=[gallery, *locked_controls, metadata_output],
                js=(
                    f"() => [{json.dumps(gallery_update)}, "
                    f"...Array.from({{length: {len(locked_controls)}}}, "
                    "() => ({__type__: 'update', interactive: false})), "
                    "{__type__: 'update', value: 'Loading Library…'}]"
                ),
                queue=False,
                api_visibility="undocumented",
            )

        lock_selection(gallery.select)
        for trigger in (
            update_btn.click,
            ib_page.input,
            previous_page.click,
            next_page.click,
            search_btn.click,
            search_input.submit,
        ):
            lock_selection(trigger, reset=True)

        def availability(record, metadata):
            from ruinedfooocus.features.gallery.seedvr2_replay import restoration_metadata

            data = read_metadata(metadata)
            valid_prompt = data is not None and isinstance(data.get("prompt"), str)
            image = record is not None and Path(record["path"]).suffix.lower() in MEDIA_SUFFIXES
            enabled = [
                valid_prompt,
                bool(data and data.get("native_composition_source"))
                or valid_prompt
                and bool(data.get("base_model_name"))
                or image
                and restoration_metadata(metadata) is not None,
                image,
                image,
                image,
                image,
                image,
                image,
                image,
                image,
            ]
            return {
                button: gr.update(interactive=active)
                for button, active in zip(action_buttons, enabled)
            } | {control: gr.update(interactive=True) for control in navigation_controls}

        def clear_selection(message=""):
            return availability(None, {}) | {
                selected: None,
                metadata_json: {},
                metadata_output: message,
                gallery: gr.update(elem_classes=[]),
            }

        def render_page(page, term, status=None, request=None, cancelled=None):
            locale = resolve_locale(request=request)
            options = {"cancelled": cancelled} if cancelled is not None else {}
            images, text, records, snapshot = browser.load_images(
                page, term, include_paths=True, include_page=True, locale=locale, **options
            )
            count, pages, page = snapshot["count"], snapshot["pages"], snapshot["page"]
            return availability(None, {}) | {
                # Constructor updates preserve explicit None; gr.update drops it.
                gallery: gr.Gallery(
                    value=images, selected_index=None, preview=False, elem_classes=[], render=False
                ),
                ib_page: gr.update(value=page, maximum=pages, interactive=True),
                status_output: gr.update(
                    value=status or localize("Images found: {count}", locale, count=count)
                ),
                page_records: records,
                ib_range: text,
                pagination: gr.update(visible=True),
                page_total: f"{page} / {pages}",
                previous_page: gr.update(interactive=page > 1),
                next_page: gr.update(interactive=page < pages),
                selected: None,
                metadata_json: {},
                metadata_output: "",
            }

        def load_page(page, search, request: gr.Request = None):
            return render_page(page, search, request=request)

        def previous(page, term, request: gr.Request = None):
            return render_page(page - 1, term, request=request)

        def following(page, term, request: gr.Request = None):
            return render_page(page + 1, term, request=request)

        def refresh(search, request: gr.Request = None):
            locale = resolve_locale(request=request)
            return render_page(1, search, browser.update_images(locale=locale), request=request)

        def search(term, request: gr.Request = None):
            return render_page(1, term, request=request)

        def owner(request):
            identity = current_owner()
            if (
                request is not None
                and request.session_hash
                and (identity is None or len(identity) == 3)
            ):
                from ruinedfooocus.integrations.web_access import authenticated_owner

                prefix = authenticated_owner(getattr(request, "request", None))
                if prefix is None:
                    prefix = (
                        identity
                        if identity is not None and identity[0] == "ui-login"
                        else ("ui", str(request.username or ""), "local")
                    )
                identity = prefix + (str(request.session_hash),)
            return identity

        def acknowledge(page, term, version, refresh=False, request=None, *, delta=0):
            ticket = latest_ui_requests.begin(owner(request), "library-view", int(version))
            if not latest_ui_requests.current(ticket):
                return gr.skip()
            try:
                page = int(page or 1)
            except (TypeError, ValueError, OverflowError):
                page = 1
            return {
                "version": ticket.version,
                "page": max(1, page + delta),
                "term": term or "",
                "refresh": refresh,
            }

        def request_refresh(page, term, version, request: gr.Request = None):
            return acknowledge(page, term, version, True, request)

        def request_page(page, term, version, request: gr.Request = None):
            return acknowledge(page, term, version, False, request)

        def request_search(page, term, version, request: gr.Request = None):
            return acknowledge(1, term, version, False, request)

        def request_previous(page, term, version, request: gr.Request = None):
            return acknowledge(page, term, version, False, request, delta=-1)

        def request_next(page, term, version, request: gr.Request = None):
            return acknowledge(page, term, version, False, request, delta=1)

        for trigger, callback in (
            (update_btn.click, request_refresh),
            (ib_page.input, request_page),
            (previous_page.click, request_previous),
            (next_page.click, request_next),
        ):
            trigger(
                callback,
                inputs=[ib_page, search_input, request_version],
                outputs=view_request,
                js=request_version_js,
                queue=False,
                trigger_mode="multiple",
                show_progress="hidden",
                api_visibility="undocumented",
            )
        gr.on(
            triggers=[search_btn.click, search_input.submit],
            fn=request_search,
            inputs=[ib_page, search_input, request_version],
            outputs=view_request,
            js=request_version_js,
            queue=False,
            trigger_mode="multiple",
            show_progress="hidden",
            api_visibility="undocumented",
        )

        async def prepare_view(descriptor, request: gr.Request = None, progress=gr.Progress()):
            ticket = None
            try:
                if not isinstance(descriptor, dict):
                    return gr.skip()
                ticket = latest_ui_requests.get(
                    owner(request), "library-view", descriptor.get("version"), start=True
                )
                locale = resolve_locale(request=request)

                def work():
                    status = descriptor.get("status")
                    if descriptor.get("refresh"):
                        status = browser.update_images(
                            locale=locale,
                            cancelled=ticket.cancelled,
                            progress=lambda count, parsed, reused: progress(
                                None,
                                desc=localize(
                                    "Scanning Library: {count} files ({reused} unchanged)",
                                    locale,
                                    count=count,
                                    reused=reused,
                                ),
                            ),
                            publish=lambda commit: latest_ui_requests.publish(ticket, commit),
                        )
                    updates = render_page(
                        descriptor["page"], descriptor["term"], status, request, ticket.cancelled
                    )
                    updates[page_records] = [
                        record | {"_view_version": ticket.version}
                        for record in updates[page_records]
                    ]
                    return updates

                future = preparation_admission.submit_work(work, cancelled=ticket.cancelled)
                latest_ui_requests.attach(ticket, future)
                result = await asyncio.wrap_future(future)
                return latest_ui_requests.publish(ticket, lambda: result)
            except (CancelledError, asyncio.CancelledError):
                if ticket is not None:
                    ticket.cancelled.set()
                return gr.skip()
            except Exception as error:
                if ticket is not None:
                    message = str(error)
                    try:
                        return latest_ui_requests.publish(
                            ticket,
                            lambda: (
                                clear_selection()
                                | {
                                    status_output: message,
                                    page_records: [],
                                    gallery: gr.Gallery(
                                        value=[],
                                        selected_index=None,
                                        preview=False,
                                        elem_classes=[],
                                        render=False,
                                    ),
                                }
                            ),
                        )
                    except CancelledError:
                        pass
                return gr.skip()
            finally:
                if ticket is not None:
                    latest_ui_requests.finish(ticket)

        view_request.change(
            prepare_view,
            inputs=view_request,
            outputs=refresh_outputs,
            concurrency_id="library_preparation",
            concurrency_limit=2,
            trigger_mode="always_last",
            show_progress="minimal",
            show_progress_on=status_output,
            api_visibility="undocumented",
        )

        def cancel_refresh(page, term, version, request: gr.Request = None):
            latest_ui_requests.cancel(owner(request), "library-view")
            message = localize(
                "Library refresh cancelled. The previous index is still available.",
                resolve_locale(request=request),
            )
            descriptor = acknowledge(page, term, version, False, request)
            if isinstance(descriptor, dict) and "version" in descriptor:
                descriptor["status"] = message
            return descriptor, message

        cancel_refresh_btn.click(
            cancel_refresh,
            inputs=[ib_page, search_input, request_version],
            outputs=[view_request, status_output],
            js=request_version_js,
            queue=False,
            trigger_mode="multiple",
            show_progress="hidden",
            api_visibility="undocumented",
        )

        def select(records, evt: gr.SelectData, descriptor=None, request: gr.Request = None):
            ticket = None
            if descriptor:
                try:
                    ticket = latest_ui_requests.get(
                        owner(request), "library-view", descriptor.get("version")
                    )
                    if any(record.get("_view_version") != ticket.version for record in records):
                        return {}
                except CancelledError:
                    # A newer view owns both its pending class and its controls.
                    return {}
            if not getattr(evt, "selected", True):
                return availability(None, {}) | {
                    selected: None,
                    metadata_json: {},
                    metadata_output: "",
                    gallery: gr.update(elem_classes=[]),
                }
            try:
                record = select_source(records, evt.index, evt.value.get("caption"))
                path = source_path(record)
                data = (
                    get_png_metadata(str(path))
                    if path.suffix.lower() == ".png"
                    else get_media_metadata(path)
                )
                if path.suffix.lower() not in (".exr", ".avif"):
                    with Image.open(path) as image:
                        data.update(width=image.width, height=image.height)
                data["file_path"] = str(path)
                metadata = read_metadata(data.get("parameters")) or read_metadata(data) or data
            except (ValueError, OSError, AttributeError) as error:
                gr.Warning("Could not select this file. Refresh Library and select it again.")
                return availability(None, {}) | {
                    selected: None,
                    metadata_json: {},
                    metadata_output: str(error),
                    gallery: gr.update(elem_classes=[]),
                }
            updates = availability(record, metadata) | {
                selected: record,
                metadata_json: metadata,
                metadata_output: format_metadata_string(data),
                gallery: gr.update(elem_classes=[]),
            }
            return (
                latest_ui_requests.publish(ticket, lambda: updates)
                if ticket is not None
                else updates
            )

        def acknowledge_selection(
            records, descriptor, version, evt: gr.SelectData, request: gr.Request = None
        ):
            if not isinstance(descriptor, dict):
                return gr.skip()
            try:
                view = latest_ui_requests.get(
                    owner(request), "library-view", descriptor.get("version")
                )
                # Delayed page data must not cancel a valid selection worker.
                if any(record.get("_view_version") != view.version for record in records):
                    return gr.skip()
                ticket = latest_ui_requests.publish(
                    view,
                    lambda: latest_ui_requests.begin(
                        owner(request), "library-selection", int(version)
                    ),
                )
                if not latest_ui_requests.current(ticket):
                    return gr.skip()
                return {
                    "version": ticket.version,
                    "view_version": descriptor["version"],
                    "index": evt.index,
                    "value": evt.value,
                    "selected": getattr(evt, "selected", True),
                }
            except CancelledError:
                return gr.skip()

        async def prepare_selection(
            records, descriptor, selected_request, request: gr.Request = None
        ):
            from types import SimpleNamespace

            ticket = None

            def publish(result):
                return latest_ui_requests.publish(
                    ticket,
                    lambda: latest_ui_requests.publish(
                        latest_ui_requests.get(
                            owner(request), "library-view", descriptor["version"]
                        ),
                        lambda: result,
                    ),
                )

            try:
                if (
                    not isinstance(descriptor, dict)
                    or not isinstance(selected_request, dict)
                    or selected_request.get("view_version") != descriptor.get("version")
                ):
                    return gr.skip()
                ticket = latest_ui_requests.get(
                    owner(request), "library-selection", selected_request["version"], start=True
                )
                event = SimpleNamespace(
                    index=selected_request["index"],
                    value=selected_request["value"],
                    selected=selected_request["selected"],
                )
                future = preparation_admission.submit_work(
                    lambda: select(records, event, descriptor, request),
                    cancelled=ticket.cancelled,
                )
                latest_ui_requests.attach(ticket, future)
                result = await asyncio.wrap_future(future)
                if not result:
                    return gr.skip()
                if isinstance(result.get(selected), dict):
                    result[selected] = result[selected] | {"_selection_version": ticket.version}
                return publish(result)
            except (CancelledError, asyncio.CancelledError):
                if ticket is not None:
                    ticket.cancelled.set()
                return gr.skip()
            except Exception as error:
                if ticket is not None:
                    try:
                        return publish(clear_selection(str(error)))
                    except CancelledError:
                        pass
                return gr.skip()
            finally:
                if ticket is not None:
                    latest_ui_requests.finish(ticket)

        gallery.select(
            acknowledge_selection,
            inputs=[page_records, view_request, request_version],
            outputs=selection_request,
            js=request_version_js,
            queue=False,
            trigger_mode="multiple",
            show_progress="hidden",
            api_visibility="undocumented",
        )
        selection_request.change(
            prepare_selection,
            inputs=[page_records, view_request, selection_request],
            outputs=refresh_outputs,
            concurrency_id="library_selection",
            concurrency_limit=2,
            trigger_mode="always_last",
            show_progress="minimal",
            show_progress_on=metadata_output,
            api_visibility="undocumented",
        )

    def bind_actions(
        top_tabs,
        create_tabs,
        reset_preset,
        reset_controls,
        *,
        image_tools_tabs,
        run_request=None,
        stop_request=None,
        run_controls=None,
        editor_workspace=None,
        navigation=None,
        navigation_outputs=(),
        gallery_tab=None,
    ):
        local_editor = None
        lock_selection((gallery_tab or top_tabs.children[2]).select, reset=True)

        source_outputs = (
            [editor_workspace.context, editor_workspace.action, editor_workspace.notice]
            if editor_workspace is not None
            else []
        )

        def navigate(updates, tab="generation", *, destination=None, reset_source=False):
            if reset_source and editor_workspace is not None:
                updates = updates | {
                    editor_workspace.context: None,
                    editor_workspace.action: "generate",
                }
            if navigation is not None:
                return updates | navigation(
                    destination or ("editor" if tab == "image-tools" else "create"), tab
                )
            return updates | {
                top_tabs: gr.update(selected="create"),
                (image_tools_tabs if tab == "image-tools" else create_tabs): gr.update(
                    selected=tab
                ),
            }

        def validate_view_record(record, request):
            ticket = None
            if isinstance(record, dict) and "_view_version" in record:
                try:
                    latest_ui_requests.get(owner(request), "library-view", record["_view_version"])
                    if "_selection_version" in record:
                        ticket = latest_ui_requests.get(
                            owner(request), "library-selection", record["_selection_version"]
                        )
                except CancelledError:
                    raise gr.Error("The image list changed. Select the image again.") from None
            return ticket

        if run_request is not None and run_controls is not None:
            from ruinedfooocus.features.gallery.ui_editing import LibraryEditor

            with browser_controls:
                local_editor = LibraryEditor()
            local_editor.bind(
                selected,
                metadata_json,
                prompt_version,
                validate_record=validate_view_record,
                resets=REFERENCE_RESETS,
                run_request=run_request,
                stop_request=stop_request,
                run_controls=run_controls,
                prompt_button=prompt_btn,
                setup_button=setup_btn,
                setup_values=reuse_setup,
            )
            if editor_workspace is not None:
                editor_workspace.bind_chat_open(local_editor.open_result, local_editor.terminal)

        if editor_workspace is not None:
            editor_workspace.bind_picker(browser)
            editor_workspace.bind_library(
                open_editor_btn,
                selected,
                metadata_json,
                prompt_version,
                validate_record=validate_view_record,
            )

        def use_prompt(record, metadata, version, request: gr.Request = None):
            validate_view_record(record, request)
            try:
                source_path(record)
            except ValueError as error:
                raise gr.Error(str(error)) from None
            return navigate(reuse_prompt(metadata, version == "Original prompt"), reset_source=True)

        def use_setup(
            record, metadata, current_resolution, request: gr.Request = None, *tensor_inputs
        ):
            validate_view_record(record, request)
            try:
                source_path(record)
            except ValueError as error:
                raise gr.Error(str(error)) from None
            from ruinedfooocus.features.gallery.seedvr2_replay import (
                restoration_metadata,
                prepare_record,
            )

            if restoration_metadata(metadata) is not None:
                from ruinedfooocus.features.image_editing import controlnet

                try:
                    values = prepare_record(
                        record,
                        metadata,
                        shared.path_manager,
                        presets=controlnet.cn_options,
                        custom=controlnet.NEWCN,
                        resets=REFERENCE_RESETS,
                    )
                except (ValueError, OSError) as error:
                    raise gr.Error(str(error)) from None
                validate_view_record(record, request)
                return navigate(
                    {
                        get_ctrl(name): gr.update(value=value)
                        for name, value in values.items()
                        if get_ctrl(name) is not None
                    },
                    "image-tools",
                    reset_source=True,
                )
            from ruinedfooocus.features.generation.native_tensor_io import INPUTS

            updates = reuse_setup(metadata, tensor_inputs=dict(zip(INPUTS, tensor_inputs)))
            saved = read_metadata(metadata)
            if saved.get("native_composition_source"):
                return navigate(updates, "image-tools", reset_source=True)
            performance = updates[get_ctrl("performance_selection")]["value"]
            resolution = updates.get(get_ctrl("aspect_ratios_selection"), {}).get(
                "value", current_resolution
            )
            for control, presentation in reset_preset(performance, resolution).items():
                updates[control] = presentation | updates.get(control, {})
            from ruinedfooocus.features.image_processing.depth_utilities import kind_from

            source_bound = kind_from(saved) not in ("", "none") or bool(saved.get("inpaint_toggle"))
            if source_bound and editor_workspace is not None:
                updates[editor_workspace.notice] = t("Add source images and masks before running.")
            return navigate(
                updates,
                destination="editor" if source_bound else "create",
                reset_source=True,
            )

        from ruinedfooocus.features.image_processing.image_composition import (
            FIELDS as COMPOSITION_FIELDS,
            IMAGE_FIELDS as COMPOSITION_IMAGES,
        )
        from ruinedfooocus.features.image_editing.model_patches import FIELDS as PATCH_FIELDS
        from ruinedfooocus.features.image_editing.lora_hooks import FIELDS as LORA_HOOK_FIELDS
        from ruinedfooocus.features.image_editing.native_control_recipes import (
            FIELDS as CONTROL_RECIPE_FIELDS,
        )
        from ruinedfooocus.features.generation.cascade_guidance import (
            FIELDS as CASCADE_FIELDS,
            IMAGE_FIELD as CASCADE_IMAGE_FIELD,
        )

        metadata_names = (
            list(COMPOSITION_FIELDS)
            + list(COMPOSITION_IMAGES)
            + list(METADATA_CONTROLS.values())
            + list(FIELD_SCHEMA)
            + list(PATCH_FIELDS)
            + list(LORA_HOOK_FIELDS)
            + list(CONTROL_RECIPE_FIELDS)
            + list(CASCADE_FIELDS)
            + [
                CASCADE_IMAGE_FIELD,
                "cn_model",
                "cn_model_signature",
                "cn_selection",
                "cn_strength",
                "cn_start",
                "cn_stop",
                "qwen_union_model",
                "qwen_union_model_signature",
                "qwen_union_mode",
                "model_recipe_path",
                "model_recipe_sha256",
                "model_recipe_overrides",
                "clip_layer_source",
                "reference_adapter",
                "reference_adapter_model",
                "reference_adapter_vision",
                "reference_adapter_strength",
                "reference_adapter_image",
                "output_format",
                "output_bit_depth",
                "output_color_space",
                "output_crf",
                "performance_selection",
                "style_selection",
                "obp_assume_direct_control",
                "obp_image_prompt",
                "auto_negative",
                "aspect_ratios_selection",
                "seed_random",
                "automatic_upscale",
                "advanced_upscalers",
                "automatic_upscale_target",
                "_imported_prompt",
            ]
        )
        setup_names = list(
            dict.fromkeys(
                metadata_names
                + list(REFERENCE_RESETS)
                + [
                    "preset_selection",
                    "base_model_name",
                    "loras",
                    "lora_keywords",
                    "image_component_overrides",
                    "_image_component_scope",
                    "main_view",
                    "seedvr2_model",
                    "seedvr2_vae",
                    "seedvr2_scale",
                    "seedvr2_color",
                ]
            )
        )
        prompt_names = list(
            dict.fromkeys(
                [
                    "prompt",
                    "negative",
                    "_imported_prompt",
                    "style_selection",
                    "automatic_upscale_target",
                    "auto_negative",
                ]
                + list(REFERENCE_RESETS)
            )
        )
        if local_editor is None:
            prompt_btn.click(
                use_prompt,
                inputs=[selected, metadata_json, prompt_version],
                outputs=list(
                    dict.fromkeys(
                        [get_ctrl(name) for name in prompt_names]
                        + [top_tabs, create_tabs, *navigation_outputs, *source_outputs]
                    )
                ),
                api_visibility="undocumented",
            )
            setup_btn.click(
                use_setup,
                inputs=[
                    selected,
                    metadata_json,
                    get_ctrl("aspect_ratios_selection"),
                    *(
                        get_ctrl(name)
                        for name in (
                            "native_latent_input",
                            "native_positive_input",
                            "native_negative_input",
                        )
                    ),
                ],
                outputs=list(
                    dict.fromkeys(
                        [get_ctrl(name) for name in setup_names]
                        + reset_controls
                        + [top_tabs, create_tabs, *navigation_outputs, *source_outputs]
                    )
                ),
                api_visibility="undocumented",
            )
        image_names = list(REFERENCE_RESETS) + [
            "main_view",
            "cn_type",
            "cn_strength",
            "automatic_upscale",
            "automatic_upscale_target",
        ]
        for button, kind in (
            (edit_btn, "edit"),
            (inpaint_btn, "inpaint"),
            (upscale_btn, "upscale"),
            (rembg_btn, "rembg"),
            (grid_btn, "image-grid"),
            (composite_btn, "image-composite"),
        ):
            if local_editor is not None and kind in ("edit", "inpaint", "upscale", "rembg"):

                def local_action(record, request: gr.Request = None, kind=kind):
                    validate_view_record(record, request)
                    source_path(record)
                    return kind

                button.click(
                    local_action,
                    inputs=selected,
                    outputs=local_editor.operation,
                    show_progress="hidden",
                    api_visibility="private",
                )
                continue

            def prepare(
                record,
                checkpoint,
                moss_format,
                scale,
                selection,
                control_type,
                strength,
                task_id,
                request: gr.Request = None,
                kind=kind,
            ):
                if task_id is not None and editor_workspace is not None:
                    return {
                        editor_workspace.notice: t(
                            "An image job is still running. Finish it or use Stop, then open this image again."
                        )
                    }
                validate_view_record(record, request)
                updates = prepare_image(
                    record, kind, checkpoint, moss_format, scale, selection, control_type, strength
                )
                return navigate(
                    updates,
                    "generation" if kind == "upscale" else "image-tools",
                    destination="editor",
                    reset_source=True,
                )

            button.click(
                prepare,
                inputs=[
                    selected,
                    get_ctrl("base_model_name"),
                    get_ctrl("moss_image_edit_format"),
                    get_ctrl("automatic_upscale"),
                    get_ctrl("cn_selection"),
                    get_ctrl("cn_type"),
                    get_ctrl("cn_strength"),
                    run_controls["task"] if run_controls is not None else gr.State(None),
                ],
                outputs=list(
                    dict.fromkeys(
                        [get_ctrl(name) for name in image_names]
                        + [top_tabs, create_tabs, image_tools_tabs, *navigation_outputs]
                        + source_outputs
                    )
                ),
                show_progress_on=(
                    local_editor.notice if local_editor is not None else status_output
                ),
                api_visibility="undocumented",
            )
        continue_btn.click(
            lambda: (
                navigation("create") if navigation is not None else gr.update(selected="create")
            ),
            outputs=list(dict.fromkeys(navigation_outputs)) if navigation is not None else top_tabs,
            queue=False,
            api_visibility="undocumented",
        )

    return (
        app_image_browser,
        request_refresh,
        [ib_page, search_input, request_version],
        [view_request],
        bind_actions,
    )
