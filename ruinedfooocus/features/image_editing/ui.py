from ruinedfooocus.core.translation_manager import translate_ui
import ruinedfooocus.features.image_editing.controlnet as controlnet
from ruinedfooocus.features.image_editing.controlnet import (
    load_cnsettings,
    save_cnsettings,
    NEWCN,
)
import gradio as gr
import ruinedfooocus.app.state as shared
from ruinedfooocus.features.models.image_scope import require_image_family
from ruinedfooocus.app.state import add_ctrl, path_manager, translate
from ruinedfooocus.features.image_editing.inpaint_utils import (
    OUTPAINT_DEFAULT_FEATHER,
    OUTPAINT_MAX_FEATHER,
    OUTPAINT_MAX_MARGIN,
    editor_payload,
)
from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint
from ruinedfooocus.features.image_editing.tool_type_dropdown import ToolTypeDropdown

t = translate_ui

SDXL_CONTROLS = frozenset(("canny", "depth", "sketch", "recolour"))


def checkpoint_features(name, moss_format=None):
    """Use header-detected architecture so SDXL metadata aliases stay supported."""
    from ruinedfooocus.features.models.image_components import ALIASES
    from ruinedfooocus.features.image_editing.inpaint_utils import inpaint_supported
    from ruinedfooocus.features.image_editing.native_conditioning import INSTRUCT_FAMILIES

    family = family_for_checkpoint(name)
    require_image_family(family)
    canonical = ALIASES.get(family, family)
    sdxl = canonical in ("SDXL", "SDXL 1.0")
    if moss_format is None:
        moss_format = shared.settings.default_settings.get("moss_image_edit_format", "Disabled")
    from ruinedfooocus.features.models.image_components import native_edit_supported
    from ruinedfooocus.features.image_editing.native_sd_inpaint import local_native_sd_inpaint

    native_edit = native_edit_supported(family, name, moss_format)
    from ruinedfooocus.features.image_editing.control_candidates import model_family

    return {
        "sdxl": sdxl,
        "control": model_family(family) is not None,
        "inpaint": inpaint_supported(family)
        or local_native_sd_inpaint(shared.models.get_file("checkpoints", name)),
        "qwen_local": family == "QwenImage21",
        "qwen_union": canonical == "QwenImage21",
        "native_edit": native_edit,
        "instruct": family in INSTRUCT_FAMILIES,
        "sd4x": family == "SD_X4Upscaler",
        "minimax_image": canonical == "MiniMaxH3Image",
    }


def control_types(features):
    from ruinedfooocus.features.image_processing.image_composition import handles
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        KINDS as DEPTH_UTILITIES,
    )

    return [
        name.capitalize()
        for name in controlnet.controlnet_models
        if name
        not in ("depth-map", "qwen-union", "restore-seedvr2", "latent-compose", *DEPTH_UTILITIES)
        and not handles(name)
        and (features.get("native_edit") or name != "edit")
        and (not features.get("minimax_image") or name != "img2img")
        and (features["sdxl"] or features.get("control") or name not in SDXL_CONTROLS)
        and (features.get("control") or name != "control")
        and (not (features.get("instruct") or features.get("sd4x")) or name != "img2img")
    ]


def input_label(features):
    return (
        "Image to upscale (4×)"
        if features.get("sd4x")
        else "Image edit reference"
        if features["native_edit"]
        else "Image to image"
    )


def image_tool_help(features):
    if features.get("minimax_image"):
        return translate_ui(
            "MiniMax H3 Image uses its PDD schedule, 32B conditioning encoder and image VAE. "
            "Use None for text-to-image or Native image edit with up to four ordered pictures. "
            "Refer to them as <Picture 1>, <Picture 2>. Transparent inputs use a neutral background; outputs are opaque RGB."
        )
    if features.get("sd4x"):
        return translate_ui(
            "This diffusion upscaler requires a source image and Diffusion 4× upscale. "
            "The native output is four times the source width and height; optional upscaling applies afterward."
        )
    if features.get("instruct"):
        return translate_ui(
            "This instruct checkpoint requires Native image edit and a source image. "
            "Adding an image selects Native image edit."
        )
    return translate_ui(
        "None generates a fresh image. Adding an image selects Native image edit when the "
        "checkpoint supports it, otherwise Image to image."
    )


def sd4x_output_info(
    image,
    checkpoint,
    selection=None,
    control_type=None,
    upscale_scale="Off",
    upscale_target="Generated image",
):
    from ruinedfooocus.features.image_processing.image_composition import handles
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        KINDS as DEPTH_UTILITIES,
    )

    enabled = checkpoint_features(checkpoint).get("sd4x", False)
    enabled = (
        enabled
        and not handles(tool_kind(selection, control_type))
        and tool_kind(selection, control_type)
        not in (
            "upscale",
            "rembg",
            "detect",
            "segment",
            "depth-map",
            "restore-seedvr2",
            "latent-compose",
            *DEPTH_UTILITIES,
        )
        and not (upscale_scale != "Off" and upscale_target == "Input image only")
    )
    message = translate_ui("Add a source image to see its four-times output size.")
    if enabled and image is not None:
        from ruinedfooocus.features.image_editing.sd4x_conditioning import sd4x_size

        width, height = sd4x_size(image)
        message = translate_ui(
            "SD4x native size: {width} × {height}, before optional post-upscaling."
        ).format(width=width, height=height)
    return gr.update(visible=enabled, value=message)


def tool_kind(selection, control_type):
    from ruinedfooocus.features.image_processing.still_image_utilities import adjustment_kind

    adjustment = adjustment_kind(selection, control_type)
    if adjustment is not None:
        return adjustment
    return (
        (control_type or "").lower()
        if selection == NEWCN
        else controlnet.cn_options.get(selection, {}).get("type")
    )


def input_context(
    inpainting,
    selection,
    control_type,
    checkpoint,
    moss_format=None,
    upscale_scale="Off",
    upscale_target="Generated image",
    *,
    features=None,
):
    from ruinedfooocus.features.image_processing.image_composition import handles
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        KINDS as DEPTH_UTILITIES,
    )

    if features is None:
        features = checkpoint_features(checkpoint, moss_format)
    kind = tool_kind(selection, control_type)
    composition = handles(kind)
    qwen_local = features.get("qwen_local")
    input_only = (
        kind not in ("depth-map", "restore-seedvr2", "latent-compose", *DEPTH_UTILITIES)
        and not composition
        and upscale_scale != "Off"
        and upscale_target == "Input image only"
    )
    allowed = (
        features["inpaint"]
        and not composition
        and kind
        not in (
            "upscale",
            "rembg",
            "detect",
            "segment",
            "depth-map",
            "latent-compose",
            "qwen-union",
            "restore-seedvr2",
            *DEPTH_UTILITIES,
        )
        and (not qwen_local or kind in (None, "", "img2img", "edit"))
        and not input_only
    )
    active = inpainting and allowed
    strength = not input_only and (
        kind in (*SDXL_CONTROLS, "control")
        and (features["sdxl"] or features.get("control"))
        or kind == "img2img"
        and not active
        or (kind == "qwen-union" and features.get("qwen_union"))
    )
    label = "Control guide" if kind in (*SDXL_CONTROLS, "control") else input_label(features)
    if kind in (
        "upscale",
        "rembg",
        "detect",
        "segment",
        "depth-map",
        "restore-seedvr2",
        "latent-compose",
        *DEPTH_UTILITIES,
    ):
        label = "Input image"
    return [
        gr.update(
            visible=not composition
            and kind not in ("qwen-union", "latent-compose")
            and not input_only
            and (not active or not qwen_local and kind not in (None, "img2img")),
            label=t(label),
        ),
        gr.update(visible=True if strength else "hidden"),
        gr.update(visible=allowed, **({"value": False} if inpainting and not allowed else {})),
        gr.update(visible=True if active else "hidden"),
        gr.update(visible=True if not input_only and kind == "rembg" else "hidden"),
    ]


def powerup_choices(
    sdxl=True,
    native_edit=False,
    inpaint=False,
    instruct=False,
    sd4x=False,
    control=False,
    qwen_local=False,
    qwen_union=False,
    minimax_image=False,
):
    from ruinedfooocus.features.image_processing.image_composition import handles
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        KINDS as DEPTH_UTILITIES,
    )

    choices = [("None", "None")]
    order = {
        "edit": -1,
        "img2img": 0,
        "qwen-union": 1,
        "canny": 1,
        "depth": 2,
        "sketch": 3,
        "recolour": 4,
        "upscale": 5,
        "rembg": 6,
        "detect": 7,
        "segment": 7,
        "depth-map": 7,
        "restore-seedvr2": 7,
        "latent-compose": 7,
    }
    for name, options in sorted(
        controlnet.cn_options.items(),
        key=lambda item: (order.get(item[1]["type"], 8), item[0].casefold()),
    ):
        if name == "Image to image (preserve details)":
            continue  # Use the image-to-image denoise slider instead of a second preset.
        if options["type"] == "upscale" and name.startswith("Upscale"):
            continue  # Image upscaling is selected with the resolution controls.
        if not sdxl and not control and options["type"] in SDXL_CONTROLS:
            continue
        if options["type"] == "control" and not control:
            continue
        if options["type"] == "edit" and not native_edit:
            continue
        if options["type"] == "qwen-union" and not qwen_union:
            continue
        if (instruct or sd4x or minimax_image) and options["type"] == "img2img":
            continue
        label = (
            "Diffusion 4× upscale"
            if sd4x and options["type"] == "edit"
            else "Image to image"
            if name == "Img2Img"
            else t("Qwen Image 2.1 Union")
            if name == "Qwen Image 2.1 Union"
            else t(name)
            if options["type"] in DEPTH_UTILITIES or name == "Compose saved latents"
            else name
        )
        choices.append((label, name))
    return choices + [(NEWCN, NEWCN)]


def preset_for(kind):
    """First saved tool preset of a type, e.g. the stock Img2Img or Native image edit."""
    for name, options in controlnet.cn_options.items():
        if options.get("type") == kind and name != "Image to image (preserve details)":
            return name
    return None


ACTIVATION_CONTROLS = (
    "cn_selection",
    "cn_type",
    "cn_upscale",
    "rembg_native_model",
    "native_vision_model",
    "wevisone_model",
    "seedvr2_model",
    "automatic_upscale_target",
)


def activate_tool(kind, resource=None, *, checkpoint=None, moss_format=None):
    """Set the operation and exact resource together; change callbacks keep values."""
    from ruinedfooocus.features.image_editing.tool_activation import activation_values

    features = (
        checkpoint_features(checkpoint, moss_format)
        if kind in ("img2img", "edit") and checkpoint is not None
        else None
    )
    values = activation_values(
        kind, resource, presets=controlnet.cn_options, custom=NEWCN, features=features
    )
    updates = {}
    for name, value in values.items():
        control = shared.get_ctrl(name)
        if control is None:
            continue
        options = {}
        if name == "cn_selection" and features is not None:
            options["choices"] = powerup_choices(**features)
        elif name == "cn_type" and features is not None:
            options["choices"] = control_types(features)
        elif name == "native_vision_model":
            from ruinedfooocus.features.image_processing.ui_native_vision import model_choices

            # Component.choices is the process layout default, not this
            # browser's latest type-specific choices. Rebuild via the same
            # header-only owner used by the native utility picker.
            options["choices"] = model_choices(shared.path_manager, kind, value)
        elif name == "wevisone_model":
            from ruinedfooocus.features.image_processing.ui_wevisone import choices

            options["choices"] = choices(shared.path_manager, value)
        elif name in ("cn_upscale", "rembg_native_model", "seedvr2_model"):
            choices = list(control.choices)
            if value not in {
                item[1] if isinstance(item, (tuple, list)) else item for item in choices
            }:
                choices.append((value, value))
            options["choices"] = choices
        updates[control] = gr.update(value=value, **options)
    return updates


def edit_tool(features):
    """The tool that edits an uploaded image on this checkpoint, if any."""
    return preset_for("edit") if features["native_edit"] else preset_for("img2img")


def fallback_selection(features, selection, fell_back):
    """Keep editing across checkpoint changes: native edit falls back to image to
    image on other checkpoints, and returns when an edit-capable one is chosen."""
    choices = [value for _, value in powerup_choices(**features)]
    kind = controlnet.cn_options.get(selection, {}).get("type")
    if kind == "edit" and not features["native_edit"]:
        selection, fell_back = preset_for("img2img"), True
    elif fell_back and kind == "img2img" and features["native_edit"]:
        selection, fell_back = preset_for("edit"), False
    elif (features.get("instruct") or features.get("sd4x")) and kind == "img2img":
        selection, fell_back = preset_for("edit"), False
    if selection not in choices:
        selection, fell_back = "None", False
    return selection, fell_back


def add_controlnet_tab(
    main_view,
    inpaint_view,
    prompt,
    image_number,
    run_event,
    base_model,
    preset_selection=None,
    main_model=None,
    upscale_scale=None,
    upscale_target=None,
    model_refresh=None,
):
    initial_features = checkpoint_features(base_model.value)
    with gr.Tab(label=t("Image tools"), id="image-tools"):
        gr.Markdown(
            translate_ui(
                "Choose a tool and add its image. Inpainting uses Main. "
                "Upscale: Generation > Size and upscale."
            )
        )
        with gr.Row():
            cn_selection = gr.Dropdown(
                label=t("Image tool"),
                choices=powerup_choices(**initial_features),
                value="None",
                info=image_tool_help(initial_features),
            )
            add_ctrl("cn_selection", cn_selection)
        # True while Image to image stands in for a native edit the checkpoint lacks.
        edit_fell_back = gr.State(False)
        cn_selection.input(
            lambda: False, outputs=edit_fell_back, api_visibility="undocumented", queue=False
        )

        cn_name = gr.Textbox(
            label=t("Tool preset name"),
            show_label=True,
            placeholder=t("Name"),
            interactive=True,
            visible="hidden",
        )
        cn_save_btn = gr.Button(
            value=t("Save tool preset"),
            visible="hidden",
        )

        type_choices = control_types(initial_features)
        cn_type = ToolTypeDropdown(
            label=t("Type"),
            choices=type_choices
            + [kind.capitalize() for kind in ("image-transform", "image-mask", "image-color")],
            value=type_choices[0],
            visible="hidden",
        )
        add_ctrl("cn_type", cn_type)

        cn_edge_low = gr.Slider(
            label=t("Edge (low)"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.2,
            visible="hidden",
        )
        add_ctrl("cn_edge_low", cn_edge_low)

        cn_edge_high = gr.Slider(
            label=t("Edge (high)"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.8,
            visible="hidden",
        )
        add_ctrl("cn_edge_high", cn_edge_high)

        cn_start = gr.Slider(
            label=t("Control starts at"),
            info=translate_ui("Fraction of sampling: 0 starts immediately."),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            visible="hidden",
        )
        add_ctrl("cn_start", cn_start)

        cn_stop = gr.Slider(
            label=t("Control ends at"),
            info=translate_ui("Fraction of sampling: 1 keeps control until the end."),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=1.0,
            visible="hidden",
        )
        add_ctrl("cn_stop", cn_stop)

        with gr.Group(visible="hidden") as strength_options:
            cn_strength = gr.Slider(
                label=t("Strength"),
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                value=1.0,
                visible="hidden",
            )
        add_ctrl("cn_strength", cn_strength)

        cn_upscaler = gr.Dropdown(
            label=t("Upscaler"),
            show_label=False,
            choices=["None"]
            + sorted(
                set(path_manager.upscaler_filenames)
                | {
                    key
                    for key, entry in path_manager.DOWNLOADABLE_FILES.items()
                    if entry["path"] == "path_upscalers"
                }
                | {
                    value["upscaler"]
                    for value in controlnet.cn_options.values()
                    if value["type"] == "upscale"
                }
            ),
            value="None",
            visible="hidden",
        )
        add_ctrl("cn_upscale", cn_upscaler)

        cn_outputs = [
            cn_name,
            cn_save_btn,
            cn_type,
        ]
        cn_sliders = [
            cn_start,
            cn_stop,
            cn_strength,
            cn_edge_low,
            cn_edge_high,
            cn_upscaler,
        ]
        from ruinedfooocus.features.image_editing.ui_depth_control import (
            create_ui as create_depth_ui,
        )

        depth_controls = create_depth_ui(cn_selection, cn_type, base_model, checkpoint_features)
        from ruinedfooocus.features.image_editing.ui_control_model import (
            create_ui as create_control_model,
        )

        control_model, control_model_signature = create_control_model(
            cn_selection, cn_type, base_model, model_refresh
        )

        from ruinedfooocus.features.image_editing.ui_guidance import (
            create_ui as create_image_guidance,
        )

        with gr.Column(min_width=0, elem_id="image-generation-guidance") as generation_guidance:
            guidance_controls = create_image_guidance(
                base_model,
                preset_selection=preset_selection,
                main_model=main_model,
                model_refresh=model_refresh,
                tool_selection=cn_selection,
                tool_type=cn_type,
                kind_for=tool_kind,
            )
            from ruinedfooocus.features.image_editing.ui_native_workflows import (
                create_ui as create_native_workflows,
            )

            create_native_workflows(
                base_model,
                preset_selection=preset_selection,
                main_model=main_model,
                model_refresh=model_refresh,
            )
            from ruinedfooocus.features.image_editing.ui_reference_adapters import (
                create_ui as create_reference_ui,
            )

            reference_controls = create_reference_ui(
                base_model, main_model=main_model, model_refresh=model_refresh
            )
            from ruinedfooocus.features.image_editing.ui_ordered_references import (
                create_ui as create_ordered_reference_ui,
            )

            create_ordered_reference_ui(base_model, cn_selection, cn_type, model_refresh, tool_kind)
        from ruinedfooocus.features.image_editing.ui_model_patches import (
            create_ui as create_patch_ui,
        )

        create_patch_ui(base_model, cn_selection, cn_type, model_refresh, tool_kind)
        from ruinedfooocus.features.image_editing.ui_lora_hooks import create_ui as create_hook_ui

        create_hook_ui(base_model, cn_selection, cn_type, model_refresh, tool_kind)
        moss_format_control = guidance_controls["moss_image_edit_format"]
        rembg_force = gr.Checkbox(
            label=t("Reprocess transparent images"),
            value=False,
            visible="hidden",
            info=translate_ui("Run removal even when the image already has alpha."),
        )
        add_ctrl("rembg_force", rembg_force)
        from ruinedfooocus.features.image_processing.ui_native_background import (
            create_ui as native_background_ui,
        )

        native_background_ui(cn_selection, cn_type, model_refresh, tool_kind)

        def rembg_preset_changed(selection):
            return gr.update(
                value=bool(controlnet.cn_options.get(selection, {}).get("force", False))
            )

        cn_selection.input(
            rembg_preset_changed,
            inputs=cn_selection,
            outputs=rembg_force,
            api_visibility="undocumented",
        )

        def cn_changed(selection, control_type, checkpoint, moss_format=None, apply_values=True):
            features = checkpoint_features(checkpoint, moss_format)
            if selection != NEWCN:
                options = controlnet.cn_options.get(selection, {})
                kind = options.get("type")
                updates = [gr.update(visible="hidden")] * len(cn_sliders)
                if kind == "img2img":
                    updates[2] = gr.update(
                        visible=True,
                        label=t("Denoise"),
                        info=translate_ui(
                            "Try 0.15–0.35 to preserve details; 0.6 and above allows substantial redraws."
                        ),
                        minimum=0.01,
                        maximum=1.0,
                        value=options.get("denoise", options.get("strength", 0.25)),
                    )
                elif (
                    kind in (*SDXL_CONTROLS, "control")
                    and (features["sdxl"] or features.get("control"))
                    or (kind == "qwen-union" and features.get("qwen_union"))
                ):
                    updates[0] = gr.update(visible=True, value=options.get("start", 0.0))
                    updates[1] = gr.update(visible=True, value=options.get("stop", 1.0))
                    updates[2] = gr.update(
                        visible=True,
                        label=t("Control strength"),
                        info=translate_ui("How strongly the input guides the composition."),
                        minimum=0.0,
                        maximum=2.0,
                        value=options.get("strength", 1.0),
                    )
                    if kind == "canny":
                        updates[3] = gr.update(visible=True, value=options.get("edge_low", 0.2))
                        updates[4] = gr.update(visible=True, value=options.get("edge_high", 0.8))
                elif kind == "upscale":
                    updates[5] = gr.update(visible=True, value=options.get("upscaler", "None"))
                if not apply_values:
                    updates = [
                        {key: value for key, value in update.items() if key != "value"}
                        for update in updates
                    ]
                return [gr.update(visible="hidden")] * len(cn_outputs) + updates
            else:
                return (
                    [gr.update(value="", visible=True)]
                    + [gr.update(visible=True)] * (len(cn_outputs) - 1)
                    + cn_type_changed(
                        control_type, selection, checkpoint, moss_format, apply_values=apply_values
                    )
                )

        # User preset choices load defaults. Programmatic handoffs and checkpoint
        # changes update visibility without replacing deliberate denoise values.
        cn_selection.input(
            cn_changed,
            inputs=[cn_selection, cn_type, base_model, moss_format_control],
            outputs=cn_outputs + cn_sliders,
            api_visibility="undocumented",
        )
        cn_selection.change(
            lambda *values: cn_changed(*values, apply_values=False),
            inputs=[cn_selection, cn_type, base_model, moss_format_control],
            outputs=cn_outputs + cn_sliders,
            api_visibility="undocumented",
        )

        def cn_type_changed(selection, preset, checkpoint, moss_format=None, apply_values=False):
            if preset != NEWCN:
                return cn_changed(preset, selection, checkpoint, moss_format, apply_values=False)[
                    len(cn_outputs) :
                ]
            features = checkpoint_features(checkpoint, moss_format)
            if selection not in control_types(features):
                return [gr.update(visible="hidden")] * len(cn_sliders)
            # cn_start,cn_stop,cn_strength,cn_edge_low,cn_edge_high, cn_upscaler
            slider_states = {
                "canny": [True, True, True, True, True, False],
                "img2img": [False, False, True, False, False, False],
                "edit": [False, False, False, False, False, False],
                "default": [True, True, True, False, False, False],
                "upscale": [False, False, False, False, False, True],
                "rembg": [False, False, False, False, False, False],
                "detect": [False, False, False, False, False, False],
                "segment": [False, False, False, False, False, False],
            }
            if selection.lower() in slider_states:
                show = slider_states[selection.lower()]
            else:
                show = slider_states["default"]

            result = []
            for vis in show:
                result += [gr.update(visible=True if vis else "hidden")]

            if selection.lower() == "img2img":
                result[2] = gr.update(
                    visible=True,
                    label=t("Denoise"),
                    minimum=0.01,
                    maximum=1.0,
                    value=0.25,
                    info=translate_ui(
                        "Lower values preserve the input; higher values allow larger changes."
                    ),
                )
            else:
                result[2] = gr.update(
                    visible=show[2],
                    label=t("Strength"),
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    info=translate_ui("How strongly the input guides the composition."),
                )

            if not apply_values:
                result = [
                    {key: value for key, value in update.items() if key != "value"}
                    for update in result
                ]
            return result

        cn_type.input(
            lambda *values: cn_type_changed(*values, apply_values=True),
            api_visibility="undocumented",
            inputs=[cn_type, cn_selection, base_model, moss_format_control],
            outputs=cn_sliders,
        )
        cn_type.change(
            cn_type_changed,
            api_visibility="undocumented",
            inputs=[cn_type, cn_selection, base_model, moss_format_control],
            outputs=cn_sliders,
        )

        @cn_save_btn.click(
            api_visibility="undocumented",
            inputs=cn_outputs
            + cn_sliders
            + [base_model, moss_format_control, rembg_force, control_model, control_model_signature]
            + list(depth_controls.values()),
            outputs=[cn_selection],
        )
        def cn_save(
            cn_name,
            cn_save_btn,
            cn_type,
            cn_start,
            cn_stop,
            cn_strength,
            cn_edge_low,
            cn_edge_high,
            upscale_model,
            checkpoint,
            moss_format=None,
            rembg_force=False,
            selected_control_model="",
            selected_control_signature=None,
            *depth_values,
        ):
            if cn_type not in control_types(checkpoint_features(checkpoint, moss_format)):
                raise gr.Error("This control type is not supported by the selected checkpoint.")
            if cn_name != "":
                cn_options = load_cnsettings()
                opts = {
                    "type": cn_type.lower(),
                    "start": cn_start,
                    "stop": cn_stop,
                    "strength": cn_strength,
                    "upscaler": upscale_model,
                }
                if cn_type.lower() in (*SDXL_CONTROLS, "control") and selected_control_model:
                    from ruinedfooocus.features.image_editing.control_candidates import (
                        selected_control,
                    )

                    try:
                        selected_control(
                            path_manager,
                            selected_control_model,
                            family_for_checkpoint(checkpoint),
                            selected_control_signature,
                        )
                    except (ValueError, OSError) as error:
                        raise gr.Error(str(error)) from None
                    opts.update(
                        model=selected_control_model, model_signature=selected_control_signature
                    )
                if cn_type.lower() == "canny":
                    opts.update(
                        {
                            "edge_low": cn_edge_low,
                            "edge_high": cn_edge_high,
                        }
                    )
                if cn_type.lower() == "depth":
                    from ruinedfooocus.features.image_editing.depth_control import (
                        DEPTH_DEFAULTS,
                        depth_settings,
                    )

                    opts.update(depth_settings(dict(zip(DEPTH_DEFAULTS, depth_values))))
                if cn_type.lower() == "rembg":
                    opts["force"] = rembg_force
                cn_options[cn_name] = opts
                save_cnsettings(cn_options)
                choices = powerup_choices(**checkpoint_features(checkpoint, moss_format))
                return gr.update(choices=choices, value=cn_name)
            else:
                return gr.update()

        from ruinedfooocus.features.image_editing.ui_qwen_union import create_ui as create_union_ui

        create_union_ui(
            cn_selection, cn_type, base_model, model_refresh, tool_kind, checkpoint_features
        )
        input_image = gr.Image(
            height="clamp(12rem, 30dvh, 24rem)",
            elem_id="image-tool-input",
            buttons=["download", "fullscreen"],
            label=t(input_label(initial_features)),
            type="pil",
            image_mode="RGBA",
            format="png",
            visible=True,
        )
        add_ctrl("input_image", input_image)
        from ruinedfooocus.features.image_processing.ui_native_vision import (
            create_ui as native_vision_ui,
        )

        native_vision_ui(cn_selection, cn_type, input_image, model_refresh, tool_kind)
        from ruinedfooocus.features.image_processing.ui_native_seedvr2 import (
            create_ui as seedvr2_ui,
        )

        seedvr2_ui(cn_selection, cn_type, model_refresh, tool_kind)
        from ruinedfooocus.features.image_processing.ui_depth_utilities import (
            create_ui as depth_utility_ui,
        )

        depth_utility_ui(cn_selection, cn_type, input_image, model_refresh, tool_kind)
        from ruinedfooocus.features.image_processing.ui_wevisone import create_ui as wevisone_ui

        wevisone_ui(cn_selection, cn_type, model_refresh, tool_kind)
        from ruinedfooocus.features.image_processing.ui_image_filters import (
            create_ui as image_filter_ui,
        )

        image_filter_ui(cn_selection, cn_type, tool_kind)
        from ruinedfooocus.features.image_processing.ui_image_adjustments import (
            create_ui as adjustment_ui,
        )

        adjustment_ui(cn_selection, cn_type, tool_kind)
        from ruinedfooocus.features.image_processing.ui_image_shaders import (
            create_ui as image_shader_ui,
        )

        image_shader_ui(cn_selection, cn_type, tool_kind)
        from ruinedfooocus.features.image_processing.ui_image_histogram import (
            create_ui as histogram_ui,
        )

        histogram_ui(input_image)
        from ruinedfooocus.features.image_processing.ui_image_composition import (
            create_ui as composition_ui,
        )

        composition_ui(cn_selection, cn_type, tool_kind)
        from ruinedfooocus.features.image_processing.ui_latent_composition import (
            create_ui as latent_composition_ui,
        )

        latent_composition_ui(cn_selection, cn_type, tool_kind)
        sd4x_output = gr.Markdown(
            translate_ui("Add a source image to see its four-times output size."),
            visible=initial_features.get("sd4x", False),
        )
        size_controls = [
            shared.get_ctrl(name)
            for name in (
                "aspect_ratios_selection",
                "custom_width",
                "custom_height",
                "image_resize_mode",
            )
        ]
        size_controls = [control for control in size_controls if control is not None]
        if initial_features.get("sd4x"):
            for control in size_controls:
                control.interactive = False

        def sd4x_size_changed(
            image,
            checkpoint,
            selection=None,
            control_type=None,
            scale="Off",
            target="Generated image",
        ):
            output = sd4x_output_info(image, checkpoint, selection, control_type, scale, target)
            enabled = output["visible"]
            return [
                output,
                *[gr.update(interactive=not enabled) for _ in size_controls],
            ]

        size_context_inputs = [input_image, base_model, cn_selection, cn_type]
        if upscale_scale is not None and upscale_target is not None:
            size_context_inputs += [upscale_scale, upscale_target]
        gr.on(
            triggers=[control.change for control in size_context_inputs]
            + ([model_refresh.change] if model_refresh is not None else []),
            fn=sd4x_size_changed,
            inputs=size_context_inputs,
            outputs=[sd4x_output, *size_controls],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        with gr.Column(min_width=0, elem_id="image-generation-random") as generation_random:
            random_image_prompt = gr.Checkbox(
                label=t("Generate random prompt"),
                value=False,
                info=translate_ui(
                    "Opt in to replacing the prompt with One Button when an input image is supplied."
                ),
            )
            add_ctrl("obp_image_prompt", random_image_prompt)
            with gr.Group(visible=False) as image_prompt_options:
                gr.Markdown(
                    translate_ui(
                        "Uses the Prompt tools settings; it does not describe the input image. "
                        "Advanced prompt options are under Prompt tools."
                    )
                )
                for name, label in (
                    ("OBP_preset", "One Button Preset"),
                    ("OBP_promptenhance", "Prompt enhancement"),
                    ("OBP_modeltype", "Prompt style"),
                ):
                    source = shared.get_ctrl(name)
                    mirror = gr.Dropdown(choices=source.choices, value=source.value, label=label)
                    # Input fires only for user edits; change also follows preset updates.
                    mirror.input(
                        lambda value: value,
                        inputs=mirror,
                        outputs=source,
                        api_visibility="undocumented",
                    )
                    source.change(
                        lambda value: value,
                        inputs=source,
                        outputs=mirror,
                        api_visibility="undocumented",
                    )
                subject_source = shared.get_ctrl("obp_givensubject")
                subject = gr.Textbox(
                    label=t("Subject"),
                    value=subject_source.value,
                    placeholder=translate_ui("Describe the subject and desired action"),
                )
                subject.input(
                    lambda value: value,
                    inputs=subject,
                    outputs=subject_source,
                    api_visibility="undocumented",
                )
                subject_source.change(
                    lambda value: value,
                    inputs=subject_source,
                    outputs=subject,
                    api_visibility="undocumented",
                )
        random_image_prompt.change(
            lambda enabled: gr.update(visible=enabled),
            inputs=random_image_prompt,
            outputs=image_prompt_options,
            api_visibility="undocumented",
        )

        def inpaint_label(qwen_local):
            return t("Localized edit") if qwen_local else t("Inpainting")

        inpaint_toggle = gr.Checkbox(
            label=inpaint_label(initial_features["qwen_local"]),
            value=False,
            visible=initial_features["inpaint"],
        )

        add_ctrl("inpaint_toggle", inpaint_toggle)
        from ruinedfooocus.features.image_editing.ui_reference_adapters import bind_tool_context

        bind_tool_context(
            reference_controls,
            base_model,
            cn_selection,
            cn_type,
            inpaint_toggle,
            tool_kind,
        )
        with gr.Group(visible="hidden") as inpaint_options:
            with gr.Column(visible=False) as qwen_local_options:
                qwen_local_mode = gr.Radio(
                    choices=[(t("Paint guide"), "paint"), (t("Mask"), "mask")],
                    value="paint",
                    label=t("Localized edit mode"),
                    info=translate_ui(
                        "Paint guide shows Qwen the marked area. Mask also preserves the source "
                        "outside the brush stroke."
                    ),
                )
                gr.Markdown(
                    translate_ui(
                        "Paint over the area on Main, then describe the edit. For example: "
                        "‘Make the jacket in the red area blue.’ Keep the stroke inside the "
                        "area you want to change."
                    )
                )
            add_ctrl("qwen_local_edit_mode", qwen_local_mode)
            inpaint_denoise = gr.Slider(
                label=t("Inpainting denoise"),
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=1.0,
                info=translate_ui(
                    "Amount of regeneration inside the mask. 1 fully redraws the masked area."
                ),
            )
            add_ctrl("inpaint_denoise", inpaint_denoise)
            inpaint_mask_grow = gr.Slider(
                label=t("Mask expansion (pixels)"),
                minimum=0,
                maximum=128,
                step=1,
                value=20,
                info=translate_ui(
                    "Expand the painted mask to blend edits into surrounding pixels."
                ),
            )
            add_ctrl("inpaint_mask_grow", inpaint_mask_grow)
            with gr.Accordion(t("Extend canvas"), open=False) as outpaint_options:
                gr.Markdown(
                    translate_ui(
                        "Add pixels on one or more sides to extend Main using the prompt. Set every side to zero to use the painted mask."
                    )
                )
                with gr.Row():
                    add_ctrl(
                        "outpaint_left",
                        gr.Number(
                            label=t("Left"),
                            minimum=0,
                            maximum=OUTPAINT_MAX_MARGIN,
                            precision=0,
                            value=0,
                        ),
                    )
                    add_ctrl(
                        "outpaint_right",
                        gr.Number(
                            label=t("Right"),
                            minimum=0,
                            maximum=OUTPAINT_MAX_MARGIN,
                            precision=0,
                            value=0,
                        ),
                    )
                with gr.Row():
                    add_ctrl(
                        "outpaint_top",
                        gr.Number(
                            label=t("Top"),
                            minimum=0,
                            maximum=OUTPAINT_MAX_MARGIN,
                            precision=0,
                            value=0,
                        ),
                    )
                    add_ctrl(
                        "outpaint_bottom",
                        gr.Number(
                            label=t("Bottom"),
                            minimum=0,
                            maximum=OUTPAINT_MAX_MARGIN,
                            precision=0,
                            value=0,
                        ),
                    )
                add_ctrl(
                    "outpaint_feather",
                    gr.Slider(
                        label=t("Edge blend (pixels)"),
                        minimum=0,
                        maximum=OUTPAINT_MAX_FEATHER,
                        step=1,
                        value=OUTPAINT_DEFAULT_FEATHER,
                    ),
                )
        context_inputs = [inpaint_toggle, cn_selection, cn_type, base_model, moss_format_control]
        if upscale_scale is not None and upscale_target is not None:
            context_inputs += [upscale_scale, upscale_target]

        def inpaint_context(active, selection, kind, checkpoint, moss_format, *upscale):
            features = checkpoint_features(checkpoint, moss_format)
            updates = input_context(
                active, selection, kind, checkpoint, moss_format, *upscale, features=features
            )
            qwen_local = features["qwen_local"]
            updates[2]["label"] = inpaint_label(qwen_local)
            return [
                *updates,
                gr.update(visible=qwen_local),
                *[gr.update(visible=not qwen_local) for _ in range(3)],
            ]

        gr.on(
            triggers=[component.change for component in context_inputs]
            + ([model_refresh.change] if model_refresh is not None else []),
            fn=inpaint_context,
            inputs=context_inputs,
            outputs=[
                input_image,
                strength_options,
                inpaint_toggle,
                inpaint_options,
                rembg_force,
                qwen_local_options,
                inpaint_denoise,
                inpaint_mask_grow,
                outpaint_options,
            ],
            trigger_mode="always_last",
            concurrency_id="image_input_context",
            concurrency_limit=1,
            api_visibility="undocumented",
            show_progress="hidden",
            show_progress_on=[],
        )

        def model_changed(
            checkpoint, selection, moss_format=None, fell_back=False, current_type=None
        ):
            features = checkpoint_features(checkpoint, moss_format)
            selection, fell_back = fallback_selection(features, selection, fell_back)
            types = control_types(features)
            from ruinedfooocus.features.image_processing.still_image_utilities import (
                ADJUSTMENT_KINDS,
                ADJUSTMENT_LABEL,
            )

            if selection == ADJUSTMENT_LABEL:
                types += [kind.capitalize() for kind in ADJUSTMENT_KINDS]
            selected_type = (
                current_type
                if selection == ADJUSTMENT_LABEL
                and str(current_type or "").lower() in ADJUSTMENT_KINDS
                else types[0]
            )
            return [
                gr.update(
                    choices=powerup_choices(**features),
                    value=selection,
                    info=image_tool_help(features),
                ),
                gr.update(choices=types, value=selected_type),
                fell_back,
            ]

        gr.on(
            triggers=[base_model.change]
            + ([model_refresh.change] if model_refresh is not None else []),
            fn=model_changed,
            inputs=[base_model, cn_selection, moss_format_control, edit_fell_back, cn_type],
            outputs=[cn_selection, cn_type, edit_fell_back],
            api_visibility="undocumented",
            show_progress="hidden",
        )

        def edit_format_changed(moss_format, checkpoint, selection, control_type, fell_back=False):
            features = checkpoint_features(checkpoint, moss_format)
            previous = selection
            selection, fell_back = fallback_selection(features, selection, fell_back)
            kind = tool_kind(selection, control_type)
            denoise = gr.update()
            if kind == "img2img":
                denoise = gr.update(
                    visible=True,
                    label=t("Denoise"),
                    minimum=0.01,
                    maximum=1.0,
                    info=translate_ui(
                        "Lower values preserve the input; higher values allow larger changes."
                    ),
                )
            # Preserve image pixels and the user's current denoise value when
            # switching edit formats; only the tool follows the new capability.
            # Re-send the selection only when the tool changes, so presets stay selected.
            changed = {} if selection == previous else {"value": selection}
            return [gr.update(choices=powerup_choices(**features), **changed), denoise, fell_back]

        moss_format_control.change(
            edit_format_changed,
            inputs=[moss_format_control, base_model, cn_selection, cn_type, edit_fell_back],
            outputs=[cn_selection, cn_strength, edit_fell_back],
            api_visibility="undocumented",
        )

        def image_added(image, selection, checkpoint, moss_format=None):
            # An upload with no tool means "edit this image": pick the best tool.
            if image is None or selection != "None":
                return gr.update()
            tool = edit_tool(checkpoint_features(checkpoint, moss_format))
            if tool is None:
                return gr.update()
            gr.Info(f"Image added: using {'Image to image' if tool == 'Img2Img' else tool}.")
            return gr.update(value=tool)

        input_image.upload(
            image_added,
            inputs=[input_image, cn_selection, base_model, moss_format_control],
            outputs=cn_selection,
            api_visibility="undocumented",
        )
        input_image.upload(
            lambda: None,
            outputs=shared.get_ctrl("image_source"),
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

        @inpaint_toggle.change(
            api_visibility="undocumented",
            inputs=[inpaint_toggle, main_view, base_model],
            outputs=[inpaint_view, inpaint_toggle],
            show_progress_on=inpaint_toggle,
        )
        def inpaint_checked(r, image, checkpoint=None):
            if r:
                try:
                    payload = editor_payload(
                        image,
                        preserve_alpha=checkpoint is not None
                        and checkpoint_features(checkpoint)["qwen_local"],
                    )
                except ValueError as error:
                    gr.Warning(str(error))
                    return {
                        inpaint_view: gr.update(interactive=True),
                        inpaint_toggle: gr.update(value=False),
                    }

                return {
                    inpaint_view: gr.update(
                        interactive=True,
                        value=payload,
                    )
                }
            else:
                return {
                    inpaint_view: gr.update(
                        # Gradio's editor get_data needs its editable canvas even
                        # while hidden; read-only mode removes that canvas.
                        interactive=True,
                    ),
                }

        generation_groups = [generation_guidance, generation_random]
        generation_inputs = [cn_selection, cn_type]
        if upscale_scale is not None and upscale_target is not None:
            generation_inputs += [upscale_scale, upscale_target]

        def generation_context(selection, control_type, scale="Off", target="Generated image"):
            from ruinedfooocus.features.image_processing.image_composition import handles
            from ruinedfooocus.features.image_processing.still_image_utilities import (
                KINDS as DEPTH_UTILITIES,
            )

            active = (
                not handles(tool_kind(selection, control_type))
                and tool_kind(selection, control_type)
                not in (
                    "upscale",
                    "rembg",
                    "detect",
                    "segment",
                    "depth-map",
                    "restore-seedvr2",
                    "latent-compose",
                    *DEPTH_UTILITIES,
                )
                and not (scale != "Off" and target == "Input image only")
            )
            return [gr.update(visible=True if active else "hidden") for _ in generation_groups]

        gr.on(
            triggers=[component.change for component in generation_inputs],
            fn=generation_context,
            inputs=generation_inputs,
            outputs=generation_groups,
            trigger_mode="always_last",
            concurrency_id="image_generation_context",
            concurrency_limit=1,
            api_visibility="undocumented",
        )

    return inpaint_toggle
