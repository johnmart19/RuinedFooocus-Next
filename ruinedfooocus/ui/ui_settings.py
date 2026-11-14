import gradio as gr
from ruinedfooocus.core.settings_validation import validate_settings, MOSS_EDIT_FORMATS
from ruinedfooocus.features.image_editing.image_guidance import STARTUP_SHIFT_LABELS
from ruinedfooocus.core.paths import asset_path, data_path
from ruinedfooocus.features.image_processing.interrogate import looks

from ruinedfooocus.app.state import (
    state,
    add_setting,
    performance_settings,
    resolution_settings,
    path_manager,
    settings,
    models,
)

from ruinedfooocus.core.translation_manager import translate_ui as t

SETTINGS_INDEX = {
    "Models / downloads": "installed, available, catalogue, download, cancel, components, source, license, disk, memory, quantization",
    "Appearance & language": "language, text size, theme, metadata, batch size, LoRA weights, update interval",
    "Profiles & storage": "Save, Save as profile, Use next launch, browser, port, offline, installed packages, paths, archive, library",
    "Hardware & backends": "GPU, CPU, Use CPU for everything, CUDA, ROCm, Vulkan, DirectML, Windows ML, PyTorch, llama.cpp, ONNX, background removal, install",
    "Image runtime": "device, CPU, preview, dynamic VRAM, precision, attention, offload, deterministic, smart memory, pinned memory, mmap, allocator, compiler, CUDA graphs",
    "Local Chat": "llama.cpp, XLlamacpp, backend, sampling, context, GPU layers, history, rewrite reasoning, documents, RAG, embeddings, diagnostics, file presence, sampling recommendations",
    "Generation defaults": "presets, hints, metadata, framing, VAE tiles, MOSS, HiDream, LCM, sampling shifts",
    "Connections": "API, MCP, listen, authentication, guest",
    "Maintenance": "cache, exact cache, hash, model hashes, library index, sizes, cache storage, clear, reinstall, Python packages, Torch",
}
SETTINGS_INDEX["Local Chat"] += ", memory, offload, MoE cache CPU share"


def search_settings(query):
    query = (query or "").strip().casefold()
    if not query:
        return ""
    matches = [
        f"**{section}**: {terms}"
        for section, terms in SETTINGS_INDEX.items()
        if query in (section + " " + terms).casefold()
    ]
    return "\n\n".join(matches) or "No matching setting. Try a shorter term."


def obp_choices():
    from ruinedfooocus.features.prompting.one_button.one_button_presets import OneButtonPresets

    return sorted(OneButtonPresets().opb_presets, key=str.casefold)


def save_clicked(*args, save_as=False, profile=..., base=None, return_snapshot=False):
    from ruinedfooocus.core.settings import prospective_profile
    from ruinedfooocus.core.profile_store import profile_name

    try:
        values = validate_settings(dict(zip(state["setting_name"], args)))
        if "theme" in values:
            from ruinedfooocus.ui.themes import validate_selection

            values["theme"] = validate_selection(
                values["theme"],
                previous=(base["settings"] if base is not None else settings.default_settings).get(
                    "theme"
                ),
            )
        from ruinedfooocus.ui.path_picker import validate_path_settings

        validate_path_settings(
            values,
            {**base["settings"], **base["paths"]}
            if base is not None
            else {**settings.default_settings, **path_manager.paths},
        )
        if save_as and not values.get("ui_settings_name"):
            raise ValueError("Enter a profile name for Save as profile.")
        target = (
            profile_name(values.get("ui_settings_name"))
            if save_as
            else settings.name
            if profile is ...
            else profile_name(profile)
        )
        source_settings = settings.default_settings
        source_paths = path_manager.paths
        if base is not None:
            from ruinedfooocus.core.profile_store import read_profile_pair

            source_profile = base["profile"] if save_as else target
            saved_settings, saved_paths = read_profile_pair(source_profile)
            source_settings = {**base["settings"], **saved_settings}
            source_paths = {**base["paths"], **saved_paths}
        new_settings, new_paths = prospective_profile(
            source_settings,
            source_paths,
            values,
            set(path_manager.DEFAULT_PATHS) | {"path_wildcards"},
        )

        def publish_paths():
            # Derived model folders belong to the active worker until restart.
            path_manager.paths = new_paths
            path_manager.name = target
            path_manager.set_settings_path(target)

        settings.save_profile(new_settings, new_paths, profile=target, on_publish=publish_paths)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    from ruinedfooocus.app.arguments import args as launch_args
    import os

    models.offline = (
        launch_args.offline
        or os.environ.get("RF_OFFLINE") == "1"
        or settings.default_settings.get("local_model_metadata", False)
    )
    performance_settings.model_recommendations.offline = models.offline
    message = (
        f"Saved profile {settings.name or 'default'}. Local metadata preference is active. "
        "Theme previews apply in this browser; the saved theme is used on future launches. "
        "Startup defaults, runtime and folders are pending restart. "
        f"Running profile: {getattr(launch_args, '_runtime_profile', launch_args.settings) or 'default'}. "
        "Use next launch selects the saved profile; an explicit --settings overrides it."
    )
    gr.Info(message)
    if return_snapshot:
        return message, {"profile": target, "settings": new_settings, "paths": new_paths}
    return message


def save_as_clicked(*args):
    return save_clicked(*args, save_as=True)


def use_next_launch_clicked(name):
    from ruinedfooocus.core.profile_store import use_next_launch, profile_name

    try:
        use_next_launch(name)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    return f"Next launch: {profile_name(name) or 'default'}. An explicit --settings remains authoritative."


def create_settings(*, models_in_settings=True):
    with gr.Blocks() as app_settings, gr.Column(elem_id="settings-content", min_width=0):
        from ruinedfooocus.ui.settings_navigation import (
            bind_navigation,
            create_profile_state,
            create_search_results,
            profile_receipt,
        )

        save_target = gr.State(settings.name)
        profile_base = gr.State(
            {
                "profile": settings.name,
                "settings": settings.default_settings,
                "paths": path_manager.paths,
            }
        )
        settings_search = gr.Textbox(
            label=t("Find a setting"),
            placeholder=t("Search options or section names"),
            elem_id="settings-search",
        )
        settings_index = create_search_results()
        with gr.Tabs(elem_id="settings-layout", overflow_behavior="wrap") as settings_tabs:
            with gr.Tab(t("Models / downloads"), id="models") as models_tab:
                from ruinedfooocus.features.models.ui_acquisition import (
                    create_ui as create_acquisition,
                )
                from ruinedfooocus.features.models.ui_model_components import offline

                acquisition = create_acquisition(path_manager, models, offline=offline)
                models_tab.select(
                    acquisition.refresh_rows,
                    inputs=acquisition.refresh_inputs,
                    outputs=acquisition.refresh_outputs,
                    show_progress="hidden",
                    queue=False,
                    trigger_mode="multiple",
                    js=acquisition.refresh_js,
                    api_visibility="undocumented",
                )
            with gr.Tab(t("Appearance & language")):
                from ruinedfooocus.core.translation_manager import locale_choices, locale_change_js

                language = gr.Dropdown(
                    label=t("Language for this browser"),
                    choices=locale_choices(),
                    value="auto",
                    info=t(
                        "Changes display text for this browser. Prompts and model identifiers stay unchanged."
                    ),
                )
                language.change(
                    fn=None,
                    inputs=language,
                    outputs=[],
                    js=locale_change_js(),
                    queue=False,
                    show_progress="hidden",
                )
                app_settings.language_control = language
                profile_language = gr.Dropdown(
                    label=t("Default language for this profile"),
                    choices=locale_choices(),
                    value=settings.default_settings.get("ui_language", "auto"),
                    info=t(
                        "Save to set the startup default. --language takes priority; each browser can choose its own display language."
                    ),
                )
                add_setting("ui_language", profile_language)
                text_size = gr.Slider(
                    label=t("Text size (%)"),
                    minimum=75,
                    maximum=200,
                    step=5,
                    value=100,
                    info=t(
                        "Applies immediately and is remembered by this browser. Browser zoom remains available."
                    ),
                )
                # Native Slider Reset changes the value without an input event.
                text_size.change(
                    fn=None,
                    inputs=text_size,
                    outputs=[],
                    js=asset_path("text_scale.js").read_text(encoding="utf-8"),
                    api_visibility="undocumented",
                    show_progress="hidden",
                )
                app_settings.load(
                    fn=None,
                    outputs=text_size,
                    js="() => Math.max(75, Math.min(200, Number(localStorage.getItem('ruinedfooocus-text-scale')) || 100))",
                    show_progress="hidden",
                )
                with gr.Accordion(t("Preferences"), open=True):
                    local_metadata = gr.Checkbox(
                        label=t("Local model metadata only"),
                        value=settings.default_settings.get("local_model_metadata", False),
                        info=t("Skip online model and artwork lookups."),
                    )
                    add_setting("local_model_metadata", local_metadata)
                    image_number_max = gr.Number(
                        label=t("Maximum batch size"),
                        minimum=1,
                        step=1,
                        value=settings.default_settings.get("image_number_max", 50),
                    )
                    add_setting("image_number_max", image_number_max)
                    with gr.Row():
                        lora_min = gr.Number(
                            label=t("LoRA weight minimum"),
                            value=settings.default_settings.get("lora_min", 0),
                        )
                        lora_max = gr.Number(
                            label=t("LoRA weight maximum"),
                            value=settings.default_settings.get("lora_max", 2),
                        )
                        add_setting("lora_min", lora_min)
                        add_setting("lora_max", lora_max)

                with gr.Accordion(t("Theme and updates"), open=True):
                    from ruinedfooocus.ui.themes import theme_choices, theme_id, resolve_theme

                    saved_theme = settings.default_settings.get("theme")
                    theme = gr.Dropdown(
                        label=t("Theme"),
                        interactive=True,
                        info=t(
                            "Applies at once; Save to keep it. Uses system fonts and your text size preference."
                        ),
                        choices=theme_choices(saved_theme),
                        value="" if saved_theme in (None, "None") else saved_theme,
                    )
                    add_setting("theme", theme)
                    theme_status = gr.Markdown(resolve_theme(saved_theme)[1])
                    with gr.Accordion(t("Add custom theme"), open=False):
                        custom_theme = gr.Textbox(
                            label=t("Gradio theme ID"), placeholder="author/theme"
                        )
                        with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
                            download_theme = gr.Button(t("Download theme"))

                    with gr.Accordion(t("Advanced display settings"), open=False):
                        update_interval = gr.Number(
                            label=t("WebUI update interval"),
                            interactive=True,
                            value=settings.default_settings.get("update_interval", 0.1),
                            minimum=0.01,
                            step=0.01,
                        )
                        add_setting("update_interval", update_interval)

                    theme_css = gr.Textbox(visible=False)
                    theme_active = gr.Checkbox(value=False, visible=False)

                    def theme_changed(value):
                        chosen, warning = resolve_theme(value)
                        return (
                            warning or t("Theme applied to this browser. Save to keep it."),
                            chosen._get_theme_css(),
                            not warning and bool(theme_id(value)),
                        )

                    def download_theme_clicked(value):
                        import os
                        from ruinedfooocus.app.arguments import args as launch_args
                        from ruinedfooocus.ui.themes import acquire_theme

                        try:
                            value = acquire_theme(
                                value,
                                offline=launch_args.offline or os.environ.get("RF_OFFLINE") == "1",
                            )
                        except Exception as error:
                            raise gr.Error(str(error)) from error
                        return gr.update(choices=theme_choices(value), value=theme_id(value)), t(
                            "Theme cached and selected. Save to keep it."
                        )

                    theme.change(
                        theme_changed,
                        inputs=theme,
                        outputs=[theme_status, theme_css, theme_active],
                        show_progress="hidden",
                        api_visibility="undocumented",
                    ).then(
                        # Show the chosen theme at once; Save keeps it for later launches.
                        fn=None,
                        inputs=[theme_css, theme_active],
                        js="(css, themed) => { window.rfApplyTheme(css, themed); }",
                        queue=False,
                        api_visibility="undocumented",
                    )
                    download_theme.click(
                        download_theme_clicked,
                        inputs=custom_theme,
                        outputs=[theme, theme_status],
                        show_progress="minimal",
                        concurrency_id="theme-download",
                        api_visibility="undocumented",
                    )
            with gr.Tab(t("Profiles & storage")):
                from ruinedfooocus.core.settings_schema import STARTUP_CONTROLS

                with gr.Accordion(t("Startup preferences"), open=False):
                    for key, label, default, help_text in STARTUP_CONTROLS:
                        control = gr.Checkbox(
                            label=t(label),
                            value=settings.default_settings.get(key, default),
                            info=t(help_text),
                        )
                        add_setting(key, control)
                    startup_port = gr.Number(
                        label=t("Default listen port"),
                        value=settings.default_settings.get("startup_port") or 0,
                        minimum=0,
                        maximum=65535,
                        step=1,
                        info=t(
                            "0 uses automatic port selection. --port takes priority. Restart required."
                        ),
                    )
                    add_setting("startup_port", startup_port)
                with gr.Accordion(t("Paths"), open=False):
                    from ruinedfooocus.ui.path_picker import create_path_control, host_namespace

                    gr.Markdown(t(host_namespace()))
                    gr.Markdown(
                        t(
                            "Folder changes are saved with the profile and apply after restart. Selecting a folder does not move files."
                        )
                    )
                    for key, label in (
                        ("path_checkpoints", t("Checkpoint Folders")),
                        ("path_diffusion_models", t("Diffusion Model Folders")),
                        ("path_clip", t("Text Encoder Folders")),
                        ("path_loras", t("LoRA Folders")),
                        ("path_embeddings", t("Embedding Folders")),
                        ("path_inbox", t("Inbox Folder")),
                        ("path_geometry_estimation", t("Geometry models folder")),
                        ("path_detection", t("Face detection models folder")),
                        ("path_model_patches", t("Model patches folder")),
                        ("path_gligen", t("GLIGEN model folders")),
                        ("path_hypernetworks", t("Hypernetwork folders")),
                        ("path_outputs", t("Output Folder")),
                        ("path_wildcards", t("Wildcard Folders")),
                    ):
                        value = path_manager.paths.get(
                            key, path_manager.DEFAULT_PATHS.get(key, ["wildcards"])
                        )
                        control = create_path_control(key, label, value)
                        add_setting(key, control)

                with gr.Accordion(t("Image Browser"), open=False):
                    images_per_page = gr.Number(
                        label=t("Images per page"),
                        value=settings.default_settings.get("images_per_page", 100),
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
                    add_setting("images_per_page", images_per_page)
                    archive_folders = create_path_control(
                        "archive_folders",
                        t("Archive Folders"),
                        settings.default_settings.get("archive_folders", []),
                    )
                    add_setting("archive_folders", archive_folders)
            from comfy import memory_management, model_management
            from comfy.cli_args import args as comfy_args
            from ruinedfooocus.backends.comfy.image_compute import CHOICES
            from ruinedfooocus.backends.comfy.runtime_info import runtime_details
            from ruinedfooocus.ui.ui_hardware import create_hardware_tab

            device = model_management.get_torch_device()
            device_name = model_management.get_torch_device_name(device)
            with gr.Tab(t("Hardware & backends")):
                create_hardware_tab(device_name=device_name)
            with gr.Tab(t("Image runtime")):
                from ruinedfooocus.ui.settings_presentation import create_image_runtime_summary

                create_image_runtime_summary(
                    runtime_details(
                        model_management.torch,
                        device,
                        device_name,
                        comfy_args,
                        memory_management,
                    )
                )
                with gr.Accordion(t("Image compute"), open=False):
                    preload_model = gr.Checkbox(
                        label=t("Preload selected image model"),
                        value=settings.default_settings.get("preload_image_model", False),
                        info=t(
                            "Load weights during startup. Off loads on the first generation, keeping startup and model browsing lightweight."
                        ),
                    )
                    add_setting("preload_image_model", preload_model)

                    gr.Markdown(
                        t(
                            "Save and restart. Auto follows ComfyUI; command-line options take priority."
                        )
                    )
                    from ruinedfooocus.backends.comfy.image_compute import ADVANCED_CONTROLS

                    with gr.Accordion(t("Advanced native compute"), open=False):
                        gr.Markdown(
                            t(
                                "Saved preferences apply after restart. Auto keeps native ComfyUI decisions; command-line options take priority. Precision changes can affect quality. Unsupported accelerators are not enabled by Auto."
                            )
                        )
                        for key, label, info in ADVANCED_CONTROLS:
                            control = gr.Dropdown(
                                label=t(label),
                                choices=list(CHOICES[key]),
                                value=settings.default_settings.get(key, "Auto"),
                                info=t(info),
                            )
                            add_setting(key, control)
                    for key, label, info in (
                        (
                            "image_dynamic_vram",
                            "Dynamic VRAM",
                            "Auto fits weights to available GPU memory.",
                        ),
                        (
                            "image_async_offload",
                            "Weight transfer streams",
                            "Overlap RAM-to-GPU transfers with inference.",
                        ),
                        (
                            "image_fast_disk",
                            "Fast-disk loading",
                            "Auto selects disk-backed loading for fast local storage.",
                        ),
                        ("image_vae_device", "VAE device", "Auto uses the GPU when available."),
                        (
                            "image_preview_method",
                            "Sampling preview",
                            "Auto is lightweight; approximate VAE uses extra GPU memory.",
                        ),
                    ):
                        choices = (
                            [
                                ("Auto", "auto"),
                                ("Off", "none"),
                                ("Fast RGB", "latent2rgb"),
                                ("Approximate VAE", "taesd"),
                            ]
                            if key == "image_preview_method"
                            else list(CHOICES[key])
                        )
                        control = gr.Dropdown(
                            label=t(label),
                            choices=choices,
                            value=settings.default_settings.get(key, CHOICES[key][0]),
                            info=t(info),
                        )
                        add_setting(key, control)
                    for key, label, info in (
                        (
                            "image_reserve_vram",
                            "System VRAM reserve (GiB)",
                            "Blank uses the backend OS reserve.",
                        ),
                        (
                            "image_vram_headroom",
                            "Extra VRAM headroom (GiB)",
                            "Blank uses the backend default. Increase if GPU memory pressure causes stalls.",
                        ),
                    ):
                        value = settings.default_settings.get(key)
                        control = gr.Textbox(
                            label=t(label),
                            placeholder=t("Auto"),
                            value="" if value is None else str(value),
                            info=t(info),
                        )
                        add_setting(key, control)

            with gr.Tab(t("Generation defaults")):
                gr.Markdown(
                    t(
                        "Save the current prompt, image model, LoRAs and generation controls using **Create → Save current defaults**. Advanced startup overrides are below."
                    )
                )
                with gr.Accordion(t("Saved settings (PNG)"), open=False):
                    preset_mode = gr.Dropdown(
                        label=t("Reuse from the PNG"),
                        info=t(
                            "Used while a saved PNG preset is active in Create. Your current prompt is kept."
                        ),
                        interactive=True,
                        choices=[
                            (t("Model + LoRAs, sampling and size"), 7),
                            (t("Model + LoRAs and sampling"), 6),
                            (t("Model + LoRAs and size"), 5),
                            (t("Sampling and size"), 3),
                            (t("Model + LoRAs"), 4),
                            (t("Sampling only"), 2),
                            (t("Size only"), 1),
                        ],
                        value=settings.default_settings.get("preset_mode", 7),
                    )
                    add_setting("preset_mode", preset_mode)

                with gr.Accordion(t("One Button Prompt"), open=False):
                    preset_choices = obp_choices()
                    saved_preset = settings.default_settings.get("OBP_preset", "Standard")
                    OBP_preset = gr.Dropdown(
                        label=t("OBP Preset"),
                        choices=preset_choices,
                        value=saved_preset if saved_preset in preset_choices else "Standard",
                        interactive=True,
                    )
                    add_setting("OBP_preset", OBP_preset)
                    hint_chance = gr.Number(
                        label=t("Hint Chance"),
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=settings.default_settings.get("hint_chance", 25),
                    )
                    add_setting("hint_chance", hint_chance)

                with gr.Accordion(t("Image and metadata defaults"), open=False):
                    interrogator = gr.Dropdown(
                        label=t("Default Interrogator"),
                        info=t(
                            "Model used to describe an uploaded image when no prompt is supplied."
                        ),
                        interactive=True,
                        choices=list(looks.keys()),
                        value=settings.default_settings.get("interrogator", "florence"),
                    )
                    add_setting("interrogator", interrogator)
                    save_metadata = gr.Checkbox(
                        label=t("Save Metadata"),
                        value=settings.default_settings.get("save_metadata", True),
                    )
                    add_setting("save_metadata", save_metadata)

                    meta_comment = gr.Textbox(
                        label=t("Metadata Comment (Text added to the metadata)"),
                        interactive=True,
                        value=settings.default_settings.get("meta_comment", ""),
                        lines=5,
                        max_lines=5,
                    )
                    add_setting("meta_comment", meta_comment)

                with gr.Accordion(t("Advanced model defaults"), open=False):
                    shift_controls = []
                    with gr.Accordion(t("Model guidance"), open=False):
                        from ruinedfooocus.features.image_editing.image_geometry import (
                            FRAMING_CHOICES,
                        )

                        framing = gr.Dropdown(
                            label=t("Image framing"),
                            choices=FRAMING_CHOICES,
                            value=settings.default_settings.get("image_resize_mode", "preserve"),
                            info=t(
                                "Startup default. Preserve keeps the full frame; Crop fills the selected size."
                            ),
                        )
                        add_setting("image_resize_mode", framing)
                        vae_tiled = gr.Checkbox(
                            label=t("Always use tiled VAE decoding"),
                            value=settings.default_settings.get("vae_tiled", False),
                            info=t("Startup default. Off uses automatic VRAM fitting."),
                        )
                        add_setting("vae_tiled", vae_tiled)
                        for key, label in (
                            ("vae_tile_size", "VAE tile size"),
                            ("vae_overlap", "VAE tile overlap"),
                        ):
                            value = settings.default_settings.get(key)
                            control = gr.Textbox(
                                label=t(label),
                                placeholder=t("Auto"),
                                value="" if value is None else str(value),
                                info=t(
                                    "Startup default in pixels. Adjust the current image under Image tools > Image guidance."
                                ),
                            )
                            add_setting(key, control)
                        moss_image_slg = gr.Checkbox(
                            label=t("MOSS Image skip-layer guidance"),
                            interactive=True,
                            value=settings.default_settings.get("moss_image_slg", False),
                            info=t(
                                "Startup default. Change the current image under Image tools > Image guidance."
                            ),
                        )
                        add_setting("moss_image_slg", moss_image_slg)
                        from ruinedfooocus.features.image_editing.ui_guidance import (
                            create_slg_controls,
                        )

                        with gr.Accordion(t("Skip-layer options"), open=False):
                            for key, component in create_slg_controls(
                                settings.default_settings
                            ).items():
                                add_setting(key, component)
                        norm_pre = gr.Checkbox(
                            label=t("Normalize before CFG"),
                            value=settings.default_settings.get("cfg_norm_pre_cfg", False),
                            info=t(
                                "Startup default. Used when CFG normalization strength is set; may amplify guidance."
                            ),
                        )
                        add_setting("cfg_norm_pre_cfg", norm_pre)
                        from ruinedfooocus.features.image_editing.ui_guidance import (
                            create_hidream_controls,
                            create_lcm_controls,
                        )

                        with gr.Accordion(t("HiDream O1"), open=False):
                            hidream_controls, _ = create_hidream_controls(settings.default_settings)
                            for key, component in hidream_controls.items():
                                add_setting(key, component)
                        with gr.Accordion(t("LCM noise"), open=False):
                            for key, component in create_lcm_controls(
                                settings.default_settings
                            ).items():
                                add_setting(key, component)
                        moss_edit_format = gr.Dropdown(
                            label=t("MOSS Image edit format"),
                            choices=list(MOSS_EDIT_FORMATS),
                            value=settings.default_settings.get(
                                "moss_image_edit_format", "Disabled"
                            ),
                            info=t(
                                "Startup default. Current-image controls are under Image tools > Image guidance. Requires a matching edit LoRA."
                            ),
                        )
                        add_setting("moss_image_edit_format", moss_edit_format)
                        moss_edit_strength = gr.Slider(
                            label=t("MOSS Image reference attention strength"),
                            minimum=0,
                            maximum=100,
                            step=0.05,
                            value=settings.default_settings.get("moss_image_edit_strength", 1.0),
                            info=t(
                                "1 keeps normal attention to reference images; 0 omits vision conditioning. Applies to ai-toolkit formats."
                            ),
                        )
                        add_setting("moss_image_edit_strength", moss_edit_strength)
                    with gr.Accordion(t("Sampling shift overrides"), open=False):
                        gr.Markdown(
                            t("Leave blank for automatic sampling. Changes apply after restart.")
                        )
                        for key, label in STARTUP_SHIFT_LABELS.items():
                            component = gr.Textbox(
                                label=t(label),
                                placeholder=t("Automatic"),
                                value=str(settings.default_settings[key])
                                if settings.default_settings.get(key) is not None
                                else "",
                                info=t("Auto: model recipe or native shift."),
                            )
                            add_setting(key, component)
                            shift_controls.append(component)
                    reset_shifts = gr.Button(t("Use automatic sampling shifts"), size="sm")
                    reset_shifts.click(
                        lambda: [gr.update(value="")] * len(shift_controls),
                        outputs=shift_controls,
                        api_visibility="undocumented",
                        show_progress="hidden",
                    )
                    gr.Markdown(
                        t(
                            "Reset changes this form. Click Save to keep the change. Component overrides are beside the component inventory in Create > Models."
                        )
                    )
            with gr.Tab(t("Local Chat")) as chat_settings:
                from ruinedfooocus.app.arguments import args as launch_args
                from ruinedfooocus.app.state import get_ctrl
                from ruinedfooocus.features.chat.ui_model_diagnostics import (
                    create_model_diagnostics,
                )
                from ruinedfooocus.features.chat.ui_rag_settings import create_rag_settings

                create_rag_settings(chat_settings, settings, launch_args, add_setting)
                create_model_diagnostics(
                    get_ctrl("chat_selected_quant"),
                    get_ctrl("chat_runtime_status"),
                    get_ctrl("chat_sampling_status"),
                )
                with gr.Accordion(t("Local chat and prompt runtime"), open=False):
                    from ruinedfooocus.backends.llama.llama_runtime_info import runtime_choices

                    llm_runtime = gr.Dropdown(
                        label=t("Chat runtime"),
                        choices=runtime_choices(),
                        value=settings.default_settings.get("llm_runtime", "llama.cpp"),
                    )
                    add_setting("llm_runtime", llm_runtime)
                    runtime_signature = gr.State(str(runtime_choices()))

                    def refresh_runtime_label(previous):
                        choices = runtime_choices()
                        signature = str(choices)
                        if signature == previous:
                            return gr.skip(), gr.skip()
                        return gr.update(choices=choices), signature

                    gr.Timer(2).tick(
                        refresh_runtime_label,
                        inputs=runtime_signature,
                        outputs=[llm_runtime, runtime_signature],
                        queue=False,
                        api_visibility="undocumented",
                        show_progress="hidden",
                    )
                    with gr.Accordion(t("Installed chat runtimes"), open=False):
                        from ruinedfooocus.features.chat.ui_runtime_info import (
                            create_runtime_info_controls,
                        )

                        create_runtime_info_controls(launch_args, profile_base)
                    curr_localfile = settings.default_settings.get("llama_localfile", None)
                    llama_localfile = gr.Textbox(
                        label=t("Custom GGUF path"),
                        interactive=True,
                        info=t(
                            "Optional local file. Chat model and quantization selection is in Chat."
                        ),
                        value=curr_localfile,
                    )
                    add_setting("llama_localfile", llama_localfile)
                    llama_server_args = gr.Textbox(
                        label=t("Extra llama.cpp arguments"),
                        placeholder=t("Optional, e.g. --reasoning off"),
                        value=settings.default_settings.get("llama_server_args", ""),
                    )
                    add_setting("llama_server_args", llama_server_args)
                    llm_n_predict = gr.Number(
                        label=t("Chat output token limit"),
                        info=t(
                            "Blank uses the verified model load default; -1 means no fixed output limit."
                        ),
                        interactive=True,
                        value=settings.default_settings.get("llm_n_predict"),
                        minimum=-1,
                        step=1,
                    )
                    add_setting("llm_n_predict", llm_n_predict)
                    llm_n_ctx = gr.Number(
                        label=t("Chat context tokens"),
                        info=t(
                            "Blank uses the verified model load default; larger contexts use more memory. 0 uses the runtime default."
                        ),
                        interactive=True,
                        value=settings.default_settings.get("llm_n_ctx"),
                        minimum=0,
                        step=1,
                    )
                    add_setting("llm_n_ctx", llm_n_ctx)
                    llm_n_gpu_layers = gr.Number(
                        label=t("Chat GPU layers"),
                        info=t(
                            "-1: automatic VRAM fitting in llama.cpp, full offload in xllamacpp. 0: CPU weights."
                        ),
                        interactive=True,
                        value=settings.default_settings.get("llm_n_gpu_layers", -1),
                        minimum=-1,
                        step=1,
                    )
                    add_setting("llm_n_gpu_layers", llm_n_gpu_layers)
                    from ruinedfooocus.features.chat.ui_moe_cache import create_moe_cache_settings

                    create_moe_cache_settings(
                        chat_settings,
                        settings.default_settings,
                        add_setting,
                        llm_runtime,
                        llama_server_args,
                        profile_base,
                        on_load=app_settings.load,
                    )
                    llm_chat_history = gr.Number(
                        label=t("Messages kept in chat history"),
                        info=t("0 keeps only the current message."),
                        interactive=True,
                        value=settings.default_settings.get("llm_chat_history", 7),
                        minimum=0,
                        step=1,
                    )
                    add_setting("llm_chat_history", llm_chat_history)
                    llm_hp_max_tokens = gr.Number(
                        label=t("Prompt rewrite and enhancer token limit"),
                        interactive=True,
                        value=settings.default_settings.get("llm_hp_max_tokens", 256),
                        minimum=1,
                        step=1,
                        info=t(
                            "Maximum reply for Local prompt rewrite and prompt enhancement. A reasoning model's "
                            "hidden thinking counts toward it; raise it with higher reasoning."
                        ),
                    )
                    add_setting("llm_hp_max_tokens", llm_hp_max_tokens)
                    from ruinedfooocus.features.chat.ui_reasoning_settings import (
                        create_reasoning_settings,
                    )

                    create_reasoning_settings(
                        settings.default_settings,
                        add_setting,
                        get_ctrl("chat_reasoning_capability"),
                    )

                    with gr.Accordion(t("Chat sampling"), open=False):
                        use_model_defaults = gr.Checkbox(
                            label=t("Use verified publisher sampling defaults"),
                            value=settings.default_settings.get("llm_use_model_defaults", True),
                            info=t(
                                "Only exact verified model files receive publisher defaults. Disable to keep runtime defaults. Explicit overrides still apply."
                            ),
                        )
                        add_setting("llm_use_model_defaults", use_model_defaults)
                        gr.Markdown(
                            t(
                                "Optional overrides for Chat, image review and prompt enhancement. Leave blank for automatic; zero is an explicit value. "
                                "Save to apply to the next model request. API request values and native runtime arguments take precedence."
                            )
                        )
                        from ruinedfooocus.features.chat.model_profiles import SAMPLERS

                        saved_sampling = settings.default_settings.get("llm_sampling", {})
                        sampling_controls = []
                        sampler_labels = dict(
                            temperature=t("Temperature"),
                            top_p=t("Top p"),
                            top_k=t("Top k"),
                            min_p=t("Min p"),
                            typical_p=t("Typical p"),
                            repeat_penalty=t("Repeat penalty"),
                            presence_penalty=t("Presence penalty"),
                            frequency_penalty=t("Frequency penalty"),
                        )
                        for name in SAMPLERS:
                            control = gr.Textbox(
                                label=sampler_labels.get(name, name.replace("_", " ").capitalize()),
                                value=str(saved_sampling[name]) if name in saved_sampling else "",
                                placeholder=t("Automatic"),
                                interactive=True,
                            )
                            add_setting("llm_sampling__" + name, control)
                            sampling_controls.append(control)

                        def current_sampling():
                            values = settings.default_settings.get("llm_sampling", {})
                            return [
                                settings.default_settings.get("llm_n_ctx"),
                                settings.default_settings.get("llm_n_predict"),
                                settings.default_settings.get("llm_use_model_defaults", True),
                                *[str(values[name]) if name in values else "" for name in SAMPLERS],
                            ]

                        app_settings.load(
                            current_sampling,
                            outputs=[
                                llm_n_ctx,
                                llm_n_predict,
                                use_model_defaults,
                                *sampling_controls,
                            ],
                            queue=False,
                            api_visibility="undocumented",
                            show_progress="hidden",
                        )

            with gr.Tab(t("Connections")) as connections:
                from ruinedfooocus.ui.ui_api_workspace import create_api_workspace

                create_api_workspace(connections)
                with gr.Accordion(t("Advanced local connections"), open=False):
                    startup_api = gr.Checkbox(
                        label=t("Enable local API after restart"),
                        value=settings.default_settings.get("startup_api", False),
                        info=t(
                            "OpenAI-compatible endpoints. --api takes priority; API keys stay in RF_API_KEY or --api-key."
                        ),
                    )
                    add_setting("startup_api", startup_api)
                    startup_mcp = gr.Checkbox(
                        label=t("Enable MCP after restart"),
                        value=settings.default_settings.get("startup_mcp", False),
                        info=t(
                            "Expose the app's MCP tools. --mcp takes priority. Restart required."
                        ),
                    )
                    add_setting("startup_mcp", startup_mcp)
                    startup_listen = gr.Textbox(
                        label=t("Listen interface after restart"),
                        value=settings.default_settings.get("startup_listen") or "",
                        placeholder=t("Blank uses localhost"),
                        info=t(
                            "Use localhost or an IP address. 0.0.0.0 exposes LAN interfaces and requires RF_UI_AUTH or --auth. API access also requires RF_API_KEY or --api-key. --listen takes priority; GRADIO_SERVER_NAME applies when blank."
                        ),
                    )
                    add_setting("startup_listen", startup_listen)
                gr.Markdown(
                    t(
                        "**Local API and MCP** are startup options. Use `--api` for the OpenAI-compatible API or `--mcp` for MCP. LAN access uses `--listen` with `--auth`; API access beyond localhost also requires `--api-key`. Restart is required. The restricted guest interface uses `--guest`."
                    )
                )

            with gr.Tab(t("Maintenance")):
                from ruinedfooocus.ui.cache_maintenance import create_cache_maintenance

                create_cache_maintenance(path_manager)
                with gr.Accordion(t("Reinstall"), open=False):
                    gr.Markdown(t("Reinstallation runs on the next online restart."))
                    with gr.Row(elem_classes=["rf-actions"]):
                        reinstall_all_btn = gr.Button(t("Reinstall Python packages"), size="sm")
                        reinstall_torch_btn = gr.Button(t("Reinstall Torch"), size="sm")

        profile_status = gr.Markdown("", elem_classes=["rf-settings-profile"])
        profile_area = gr.Column(min_width=0, elem_classes=["rf-settings-profile"])
        with gr.Row(elem_id="settings-actions", elem_classes=["rf-actions"], equal_height=True):
            ui_settings_name = gr.Text(
                label=t("Profile name"),
                interactive=True,
                placeholder=t("Optional"),
                value=settings.name,
            )
            add_setting("ui_settings_name", ui_settings_name)
            save_btn = gr.Button(t("Save"), variant="primary", elem_id="settings-save")
            save_as_btn = gr.Button(t("Save as profile"))
            next_launch_btn = gr.Button(t("Use next launch"))

        setting_controls = state["setting_obj"]
        with profile_area:
            profile_state, profile_receipt_state = create_profile_state(
                settings.name, setting_controls
            )

        def save_form(request: gr.Request, profile, base, *values):
            from ruinedfooocus.ui.settings_navigation import profile_update

            message, snapshot = save_clicked(
                *values, profile=profile, base=base, return_snapshot=True
            )
            receipt = profile_receipt(profile, values)
            return message, profile, profile_update(receipt, request), snapshot, receipt

        def save_as_form(request: gr.Request, base, *values):
            from ruinedfooocus.core.profile_store import profile_name
            from ruinedfooocus.ui.settings_navigation import profile_update

            message, snapshot = save_clicked(*values, save_as=True, base=base, return_snapshot=True)
            profile = profile_name(values[state["setting_name"].index("ui_settings_name")])
            receipt = profile_receipt(profile, values)
            return message, profile, profile_update(receipt, request), snapshot, receipt

        def select_next_launch(request: gr.Request, name, receipt):
            from ruinedfooocus.core.profile_store import profile_name
            from ruinedfooocus.ui.settings_navigation import profile_update

            message = use_next_launch_clicked(name)
            receipt = dict(receipt, next=profile_name(name) or "default")
            return message, profile_update(receipt, request), receipt

        save_btn.click(
            fn=save_form,
            api_visibility="undocumented",
            inputs=[save_target, profile_base, *setting_controls],
            outputs=[
                profile_status,
                save_target,
                profile_state,
                profile_base,
                profile_receipt_state,
            ],
            show_progress="hidden",
        )
        save_as_btn.click(
            save_as_form,
            inputs=[profile_base, *setting_controls],
            outputs=[
                profile_status,
                save_target,
                profile_state,
                profile_base,
                profile_receipt_state,
            ],
            api_visibility="undocumented",
            show_progress="hidden",
        )
        next_launch_btn.click(
            select_next_launch,
            inputs=[ui_settings_name, profile_receipt_state],
            outputs=[profile_status, profile_state, profile_receipt_state],
            api_visibility="undocumented",
            show_progress="hidden",
        )

        bind_navigation(
            app_settings,
            settings_search,
            settings_index,
            SETTINGS_INDEX,
            excluded=() if models_in_settings else (models_tab,),
        )

        # These files are checked in ruinedfooocus/app/bootstrap.py to trigger a --force-reinstall
        def trigger_reinstall_all():
            data_path("reinstall").touch()
            gr.Info(
                "Application Python packages and Torch will be reinstalled on the next online restart, using this Python environment. Overrides freezetorch."
            )

        def trigger_reinstall_torch():
            data_path("reinstalltorch").touch()
            gr.Info(
                "Torch will be reinstalled for the selected GPU runtime on the next online restart. Overrides freezetorch."
            )

        reinstall_all_btn.click(
            fn=trigger_reinstall_all,
            api_visibility="undocumented",
        )
        reinstall_torch_btn.click(
            fn=trigger_reinstall_torch,
            api_visibility="undocumented",
        )

    app_settings.models_tabs = settings_tabs
    app_settings.models_tab = models_tab
    app_settings.acquisition = acquisition
    return app_settings
