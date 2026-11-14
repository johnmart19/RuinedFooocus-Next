from types import SimpleNamespace
from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.core.translation_manager import catalogue, locale_change_js
from ruinedfooocus.core.translation_manager import (
    LOCALE_COOKIE,
    localize,
    localized_output,
    localized_outputs,
    resolve_locale,
)
from ruinedfooocus.core.paths import asset_path
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from ruinedfooocus.app.arguments import args
from ruinedfooocus.backends.comfy.comfy_compat import initialize_backend

torch_platform, os_platform = initialize_backend(args)

from pathlib import Path
import ruinedfooocus.app.state as shared
from ruinedfooocus.app.state import (
    state,
    add_ctrl,
    performance_settings,
    resolution_settings,
    path_manager,
    settings,
)
import time
import json
import gradio as gr
import re
import traceback

import version
from ruinedfooocus.features.models.selection import checkpoint_choices, primary_checkpoint_selection
from ruinedfooocus.features.models.resolution_dropdown import ResolutionDropdown

if args.guest:
    from ruinedfooocus.ui.ui_guest import launch_guest

    launch_guest(args)
    raise SystemExit(0)

import logging

_startup_log_level = logging.getLogger().level
logging.getLogger().setLevel(logging.ERROR)
try:
    import ruinedfooocus.app.async_worker as worker
finally:
    logging.getLogger().setLevel(_startup_log_level)
worker.start_worker()
import ruinedfooocus.ui.html as html
import ruinedfooocus.ui.hints as ui_hints
import ruinedfooocus.features.prompting.ui_random as ui_onebutton
import ruinedfooocus.features.image_editing.ui as ui_controlnet
from ruinedfooocus.features.image_editing.qwen_image21_local_edit import PAINT_COLOR
from ruinedfooocus.app.api import add_api, add_fastapi
from ruinedfooocus.features.image_processing.interrogate import metadata_prompt

# Block the "Token indices sequence length is longer than the specified maximum sequence length for this model" warning
import transformers

transformers.utils.logging.set_verbosity_error()

from comfy.samplers import KSampler
from ruinedfooocus.features.prompting.sdxl_styles import load_styles
from ruinedfooocus.features.prompting import ui_styles
from ruinedfooocus.features.prompting.prompt_processing import get_promptlist, use_random_prompt
from ruinedfooocus.core.util import (
    get_wildcard_files,
    load_keywords,
    get_lora_thumbnail,
)
from ruinedfooocus.features.gallery.imagebrowser import ImageBrowser

import ruinedfooocus.features.gallery.ui as ui_image_gallery
import ruinedfooocus.features.chat.ui as ui_llama_chat
import ruinedfooocus.ui.ui_settings as ui_settings

from PIL import Image

inpaint_toggle = None
shared.shared_cache["browser"] = ImageBrowser()
t = translate_ui
PNG_PRESET_LABEL = "Saved settings (PNG)"


def find_unclosed_markers(s):
    markers = re.findall(r"__", s)
    for marker in markers:
        if s.count(marker) % 2 != 0:
            return s.split(marker)[-1]
    return None


def create_chat_for_workspace(image_controls, workspace):
    return ui_llama_chat.create_chat(
        image_controls, editor_workspace=workspace, prompt_control=shared.get_ctrl("prompt")
    )


def regroup_generation_options(generation_tab, image_number, save_defaults_btn):
    """Fold existing optional panels without recreating their controls or events."""
    # Gradio inserts Form containers around controls, so use the ancestor that
    # actually owns the surrounding sections.
    anchor, holder = image_number, image_number.parent
    while not any(isinstance(child, gr.Accordion) for child in holder.children):
        anchor, holder = holder, holder.parent
    rare_sections = [
        child
        for child in (
            holder.children[holder.children.index(anchor) + 1 :]
            + ([] if holder is generation_tab else list(generation_tab.children))
        )
        if isinstance(child, gr.Accordion) and child.elem_id != "generation-seed"
    ]
    with holder:
        more_section = gr.Accordion(t("More options"), open=False, elem_id="generation-more")
    with more_section:
        for section in dict.fromkeys(rare_sections):
            section.unrender().render()
    with holder:
        save_defaults_btn.unrender().render()
    return more_section


def create_navigation_shell(
    create,
    chat,
    gallery,
    settings_app,
    workspace,
    *,
    model_tab,
    create_tabs,
    image_tools_tabs,
    run_button,
    title,
):
    """Move existing native containers after merging each app exactly once."""

    def page_head(title, text, *classes):
        # Gradio 6's native string formatter replaces one I18nData marker
        # per value. Keep each line reactive in every destination.
        for line in (f"<h1>{title}</h1>", f"<p>{text}</p>"):
            gr.HTML(line, elem_classes=["rf-page-head", *classes], padding=False)

    app = gr.Blocks(title=title, analytics_enabled=False, fill_width=True)
    with app:
        with gr.Tabs(elem_id="app-tabs", overflow_behavior="wrap") as top_tabs:
            with gr.Tab(t("Create"), id="create", elem_id="rf-nav-create") as create_tab:
                pass
            with gr.Tab(t("Editor"), id="editor", elem_id="rf-nav-editor") as editor_tab:
                pass
            with gr.Tab(t("Gallery"), id="library", elem_id="rf-nav-library") as gallery_tab:
                page_head(t("Gallery"), t("Browse your images and reuse their settings."))
                gallery.render()
            with gr.Tab(t("Models"), id="models", elem_id="rf-nav-models") as models_page:
                page_head(t("Models"), t("Pick an installed model or download a new one."))
                model_destinations = gr.Tabs(overflow_behavior="wrap")
            with gr.Tab(t("Chat"), id="chat", elem_id="rf-nav-chat") as chat_tab:
                chat.render()
            with gr.Tab(t("Settings"), id="settings", elem_id="rf-nav-settings") as settings_tab:
                page_head(t("Settings"), t("Profiles, hardware, defaults and maintenance."))
                settings_app.render()
        with gr.Group(elem_id="image-stage") as image_stage:
            for mode, title, text in (
                ("rf-when-create", t("Create images"), t("Generate a new image from a prompt.")),
                (
                    "rf-when-editor",
                    t("Edit images"),
                    t("Change, extend, upscale or clean up an existing image."),
                ),
            ):
                page_head(title, text, mode)
            create.render()
        # All descendants and listeners already belong to this combined Blocks.
        # Native unrender/render changes only the containers' layout parents.
        with models_page, model_destinations:
            model_tab.unrender().render()
            settings_app.models_tab.unrender().render()
        # Inside the Models page the two panels are named by what they hold.
        # A distinct ID keeps the page's own navigation label.
        model_tab.label = t("Installed")
        settings_app.models_tab.label = t("Download")
        settings_app.models_tab.id = "downloads"
        image_tools_tab = next(
            child for child in image_tools_tabs.children if child.id == "image-tools"
        )
        image_tools_tab.visible = False
        image_tools_tab.label = t("More tools")
        workspace.panel.visible = False

        navigation_outputs = [
            top_tabs,
            image_stage,
            workspace.panel,
            image_tools_tab,
            workspace.destination,
            create_tabs,
            run_button,
        ]

        def navigate(destination, inner=None, *, select=True, action=None):
            editing = destination == "editor"
            image = destination in ("create", "editor")
            updates = {
                # A visibility update on this container would also reveal
                # its visible="hidden" descendants; a class leaves them alone.
                image_stage: gr.update(
                    elem_classes=["rf-editor"] if editing else [] if image else ["rf-off"]
                ),
                workspace.panel: gr.update(visible=editing),
                image_tools_tab: gr.update(visible=editing),
                workspace.destination: destination,
            }
            if select:
                updates[top_tabs] = gr.update(selected=destination)
            if image:
                updates[create_tabs] = gr.update(selected=inner or "generation")
            if destination == "create" or editing and action is not None:
                updates[run_button] = gr.update(value=workspace.action_label(action, destination))
            return updates

        navigation_version = gr.Number(0, visible=False, precision=0)
        capture_navigation = "(action, version) => { window.__rfNavigationSerial = (window.__rfNavigationSerial || 0) + 1; return [action, window.__rfNavigationSerial]; }"
        publish_navigation = """(result) => {
            const button = document.getElementById(result.button);
            if (result.version !== window.__rfNavigationSerial || button?.getAttribute('aria-selected') !== 'true')
                return result.updates.map(() => ({__type__: 'update'}));
            return result.updates;
        }"""
        for tab, destination in (
            (create_tab, "create"),
            (editor_tab, "editor"),
            (gallery_tab, "library"),
            (models_page, "models"),
            (chat_tab, "chat"),
            (settings_tab, "settings"),
        ):
            navigation_result = gr.JSON(None, visible=False)

            def select_destination(
                action, version, destination=destination, request: gr.Request = None
            ):
                updates = navigate(destination, select=False, action=action)
                from gradio.i18n import I18nData

                button = updates.get(run_button)
                if button and isinstance(button.get("value"), I18nData):
                    from ruinedfooocus.core.translation_manager import catalogue, resolve_locale

                    updates[run_button] = dict(
                        button,
                        value=catalogue().translate(
                            button["value"].key.removeprefix("ruinedfooocus."),
                            locale=resolve_locale(request=request),
                        ),
                    )
                return dict(
                    version=version,
                    button=f"rf-nav-{destination}-button",
                    updates=[updates.get(control, gr.skip()) for control in navigation_outputs],
                )

            event = tab.select(
                select_destination,
                inputs=[workspace.action, navigation_version],
                outputs=navigation_result,
                js=capture_navigation,
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
                api_visibility="undocumented",
            )
            event.success(
                fn=None,
                inputs=navigation_result,
                outputs=navigation_outputs,
                js=publish_navigation,
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
                api_visibility="private",
            )
        for component in app.blocks.values():
            if isinstance(component, gr.Tabs):
                component.overflow_behavior = "wrap"
    return SimpleNamespace(
        app=app,
        tabs=top_tabs,
        create=create_tab,
        editor=editor_tab,
        gallery=gallery_tab,
        models=models_page,
        chat=chat_tab,
        settings=settings_tab,
        model_destinations=model_destinations,
        image_stage=image_stage,
        outputs=navigation_outputs,
        navigate=navigate,
    )


def launch_app(args):
    from ruinedfooocus.integrations.web_access import PrivateResponses, validate_web_access
    from ruinedfooocus.ui.gradio_frontend import GradioCompletionGuard, normalize_hidden_progress
    from starlette.middleware import Middleware
    from ruinedfooocus.integrations.media_paths import owner_media_paths

    validate_web_access(args)
    inbrowser = not args.nobrowser
    favicon_path = "logo.ico"

    shared.gradio_root.queue(api_open=True)

    from ruinedfooocus.ui.themes import resolve_theme, theme_id

    theme, theme_warning = resolve_theme(settings.get("theme"))
    if theme_warning:
        print(theme_warning)

    # Create the image gallery from the new module
    (
        app_image_browser,
        browser_refresh,
        browser_refresh_inputs,
        browser_refresh_outputs,
        bind_gallery_actions,
    ) = ui_image_gallery.create_image_gallery()
    create_tabs = next(
        component
        for component in shared.gradio_root.blocks.values()
        if isinstance(component, gr.Tabs) and model_tab in component.children
    )
    image_tools_tabs = next(
        component
        for component in shared.gradio_root.blocks.values()
        if isinstance(component, gr.Tabs)
        and any(
            isinstance(child, gr.Tab) and child.id == "image-tools" for child in component.children
        )
    )
    image_keys = {
        "grid_columns",
        "grid_cell_width",
        "grid_cell_height",
        "grid_padding",
        "composition_mode",
        "composition_x",
        "composition_y",
        "composition_width",
        "composition_height",
        "base_model_name",
        "performance_selection",
        "custom_steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "clip_skip",
        "cn_model",
        "cn_model_signature",
        "cn_depth_preprocess",
        "cn_depth_model",
        "cn_depth_lotus_model",
        "cn_depth_lotus_vae",
        "aspect_ratios_selection",
        "custom_width",
        "custom_height",
        "loras",
        "style_selection",
        "negative",
        "qwen_image21_schedule",
        "moss_image_edit_format",
        "moss_image_edit_strength",
        "moss_image_slg",
        "moss_image_slg_double_layers",
        "moss_image_slg_single_layers",
        "moss_image_slg_start",
        "moss_image_slg_end",
        "sampling_shift",
        "cfg_norm_strength",
        "cfg_norm_auto",
        "cfg_norm_pre_cfg",
        "image_resize_mode",
        "hidream_o1_noise_scale",
        "hidream_o1_seam_smoothing",
        "hidream_o1_seam_start",
        "hidream_o1_seam_end",
        "hidream_o1_seam_pattern",
        "hidream_o1_seam_passes",
        "hidream_o1_seam_blend",
        "hidream_o1_seam_strength",
        "lcm_noise",
        "lcm_noise_end",
        "lcm_noise_clip",
        "vae_tiled",
        "vae_tile_size",
        "vae_overlap",
        "cascade_stage_b",
        "bernini_low_model",
        "bernini_split_step",
        "bernini_lightx2v",
        "bernini_high_lora_strength",
        "bernini_low_lora_strength",
        "bernini_ref_max_size",
        "cascade_b_steps",
        "cascade_b_cfg",
        "cascade_b_sampler",
        "cascade_b_scheduler",
        "cascade_compression",
        "cascade_source_denoise",
        "cascade_super_resolution",
        "cascade_super_resolution_model",
        "cascade_control_strength",
        "cascade_control_start",
        "cascade_control_end",
        "cascade_control_denoise",
        "cascade_vision_enabled",
        "cascade_vision_model",
        "cascade_vision_images",
        "cascade_vision_strengths",
        "cascade_vision_crop",
        "sdxl_refiner_model",
        "sdxl_refiner_switch_step",
        "sdxl_refiner_cfg",
        "sdxl_refiner_clip",
        "refiner_noise_augmentation",
        "pid_input_vae",
        "rembg_native_model",
        "native_vision_model",
        "vision_threshold",
        "detect_class",
        "detect_max_count",
        "segment_refine_iterations",
        "segment_output",
        "segment_individual_masks",
        "segment_max_instances",
        "seedvr2_model",
        "seedvr2_vae",
        "seedvr2_scale",
        "seedvr2_color",
        "native_negative_model",
        "qwen_layer_count",
        "qwen_layer_source",
        "gligen_model",
        "gligen_model_signature",
        "gligen_layout",
        "hypernetwork_model",
        "hypernetwork_model_signature",
        "hypernetwork_strength",
        "native_schedule_alpha",
        "native_schedule_beta",
        "native_schedule_mu",
        "native_schedule_std",
        "pid_latent_format",
        "pid_degrade_sigma",
        "model_recipe_path",
        "clip_layer_source",
        "model_recipe_sha256",
        "model_recipe_overrides",
        "reference_adapter",
        "reference_adapter_model",
        "reference_adapter_vision",
        "reference_adapter_strength",
    }
    from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

    image_keys.update((*model_patches.FIELDS, *native_control_recipes.FIELDS))
    from ruinedfooocus.features.image_editing import lora_hooks

    image_keys.update(lora_hooks.FIELDS)
    image_controls = {
        name: control
        for name, control in zip(state["ctrls_name"], state["ctrls_obj"])
        if name in image_keys
    }
    if args.api:
        import ruinedfooocus.app.api_runtime as api_runtime
        import copy

        def api_image_defaults(*values):
            snapshot = dict(zip(image_controls, values))
            snapshot["_image_guidance_checkpoint"] = values[-1]
            api_runtime.image_defaults = copy.deepcopy(snapshot)

        api_image_defaults(
            *(control.value for control in image_controls.values()),
            effective_image_checkpoint.value,
        )
        with shared.gradio_root:
            gr.on(
                triggers=[control.change for control in image_controls.values()]
                + [effective_image_checkpoint.change],
                fn=api_image_defaults,
                inputs=list(image_controls.values()) + [effective_image_checkpoint],
                outputs=[],
                api_visibility="undocumented",
                queue=False,
            )
    app_llama_chat = create_chat_for_workspace(image_controls, editor_workspace)
    app_settings = ui_settings.create_settings(models_in_settings=False)

    shell = create_navigation_shell(
        shared.gradio_root,
        app_llama_chat,
        app_image_browser,
        app_settings,
        editor_workspace,
        model_tab=model_tab,
        create_tabs=create_tabs,
        image_tools_tabs=image_tools_tabs,
        run_button=run_button,
        title="RuinedFooocus " + version.version,
    )
    main_tabs, top_tabs, settings_tab = shell.app, shell.tabs, shell.settings
    with main_tabs:
        shell.models.select(
            app_settings.acquisition.refresh_rows,
            inputs=app_settings.acquisition.refresh_inputs,
            outputs=app_settings.acquisition.refresh_outputs,
            show_progress="hidden",
            queue=False,
            trigger_mode="multiple",
            js=app_settings.acquisition.refresh_js,
            api_visibility="undocumented",
        )
        model_catalogue_button.click(
            lambda: (
                shell.navigate("models")
                | {shell.model_destinations: gr.update(selected="downloads")}
            ),
            outputs=[*shell.outputs, shell.model_destinations],
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
            api_visibility="undocumented",
        ).then(
            app_settings.acquisition.refresh_rows,
            inputs=app_settings.acquisition.refresh_inputs,
            outputs=app_settings.acquisition.refresh_outputs,
            show_progress="hidden",
            queue=False,
            trigger_mode="multiple",
            js=app_settings.acquisition.refresh_js,
            api_visibility="undocumented",
        )
        create_models_button.click(
            lambda: (
                shell.navigate("models")
                | {shell.model_destinations: gr.update(selected="create-models")}
            ),
            outputs=[*shell.outputs, shell.model_destinations],
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
            api_visibility="undocumented",
        )

        component_overrides = shared.get_ctrl("image_component_overrides")
        component_scope = shared.get_ctrl("_image_component_scope")
        app_settings.acquisition.bind_component_context(
            base_model, component_overrides, component_scope
        )
        chat_catalogue_outputs = app_llama_chat.catalogue_selection_outputs
        tool_catalogue_outputs = [
            shared.get_ctrl(name) for name in ui_controlnet.ACTIVATION_CONTROLS
        ]

        def activate_catalogue_checkpoint(
            key,
            rows,
            current_model=None,
            component_key=None,
            overrides=None,
            scope=None,
            chat_session=None,
            request: gr.Request = None,
        ):
            from ruinedfooocus.features.models.ui_requests import require_source

            require_source(key, rows, request)
            row = next((row for row in rows if row["key"] == key), None)
            if row is None:
                raise gr.Error("Select an installed image checkpoint first.")
            # Revalidate the selected row; stale details cannot activate a replaced file.
            if not key.startswith("local:"):
                fresh = app_settings.acquisition.catalogue.record(key)
                if fresh.get("local_signature") != row.get("local_signature"):
                    raise gr.Error("The local file changed. Refresh and select it again.")
                row = fresh
            else:
                row = app_settings.acquisition.inspected_local(row, shared.models)
            if row["task"] == "Chat":
                from ruinedfooocus.features.models.acquisition_actions import validate_local_record

                try:
                    path = validate_local_record(row)
                    updates = app_llama_chat.select_catalogue_file(path, chat_session)
                except (OSError, ValueError) as error:
                    raise gr.Error(str(error)) from None
                return (
                    gr.skip(),
                    gr.update(selected="chat"),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    *(updates.get(control, gr.skip()) for control in chat_catalogue_outputs),
                    *(gr.skip() for _ in tool_catalogue_outputs),
                )
            from ruinedfooocus.features.image_editing.tool_activation import installed_tool

            try:
                tool = installed_tool(row, path_manager)
            except (OSError, ValueError) as error:
                raise gr.Error(str(error)) from None
            if row["category"] in ("clip", "vae", "clip_vision") and tool is None:
                from ruinedfooocus.features.models.ui_model_components import selected

                try:
                    path, family = selected(current_model)
                    choices, identity = app_settings.acquisition.use_component(
                        row,
                        component_key,
                        path,
                        family,
                        shared.settings.default_settings,
                        path_manager,
                        overrides,
                        scope,
                    )
                except (OSError, ValueError) as error:
                    raise gr.Error(str(error)) from None
                return (
                    gr.skip(),
                    gr.update(selected="models"),
                    gr.update(selected="create-models"),
                    choices,
                    identity,
                    *(gr.skip() for _ in chat_catalogue_outputs),
                    *(gr.skip() for _ in tool_catalogue_outputs),
                )
            try:
                if tool is not None:
                    updates = ui_controlnet.activate_tool(
                        tool["kind"], tool, checkpoint=current_model
                    )
                    return (
                        gr.skip(),
                        gr.update(selected="editor"),
                        gr.update(selected="image-tools"),
                        gr.skip(),
                        gr.skip(),
                        *(gr.skip() for _ in chat_catalogue_outputs),
                        *(updates.get(control, gr.skip()) for control in tool_catalogue_outputs),
                    )
            except (OSError, ValueError) as error:
                raise gr.Error(str(error)) from None
            try:
                name = app_settings.acquisition.checkpoint_name(row, shared.models)
            except ValueError as error:
                raise gr.Error(str(error)) from None
            return (
                gr.update(
                    choices=app_settings.acquisition.checkpoint_choices(
                        shared.models.get_names("checkpoints"), name
                    ),
                    value=name,
                ),
                gr.update(selected="create"),
                gr.update(selected="generation"),
                gr.skip(),
                gr.skip(),
                *(gr.skip() for _ in chat_catalogue_outputs),
                *(gr.skip() for _ in tool_catalogue_outputs),
            )

        activation_outputs = [
            base_model,
            top_tabs,
            create_tabs,
            component_overrides,
            component_scope,
            *chat_catalogue_outputs,
            *tool_catalogue_outputs,
        ]

        def activate_destination(*values, request: gr.Request = None):
            result = activate_catalogue_checkpoint(*values, request=request)
            destination = result[1].get("selected")
            inner = result[2].get("selected")
            updates = dict(zip(activation_outputs, result))
            updates.update(
                shell.navigate(destination, inner if destination in ("create", "editor") else None)
            )
            if destination == "models":
                updates[create_tabs] = gr.skip()
                updates[shell.model_destinations] = result[2]
            if destination == "editor":
                updates[editor_workspace.action] = (
                    "generate" if inner == "image-tools" else "prompt-edit"
                )
            return updates

        app_settings.acquisition.activate.click(
            activate_destination,
            inputs=[
                app_settings.acquisition.selection,
                app_settings.acquisition.records,
                base_model,
                app_settings.acquisition.component_role,
                component_overrides,
                component_scope,
                *app_llama_chat.catalogue_selection_inputs,
            ],
            outputs=list(
                dict.fromkeys(
                    [
                        *activation_outputs,
                        *shell.outputs,
                        shell.model_destinations,
                        editor_workspace.action,
                    ]
                )
            ),
            show_progress="hidden",
            show_progress_on=app_settings.acquisition.status,
            api_visibility="undocumented",
        )
        editor_run_controls = dict(
            run=run_button,
            stop=stop_button,
            progress=progress_html,
            preview=main_view,
            inpaint=inpaint_view,
            inpaint_toggle=inpaint_toggle,
            gallery=gallery,
            metadata=metadata_json,
            hint=hint_text,
            task=generation_task,
            originals=generation_originals,
            artifacts=generation_artifacts,
            native_source=native_main_source,
            image_source=shared.get_ctrl("image_source"),
            latest_source=editor_workspace.latest_generated,
        )

        def guide_control_plan(spec, checkpoint, model, signature):
            from ruinedfooocus.features.image_editing.guide_control_compat import plan_guide_control

            selected = plan_guide_control(
                spec,
                ui_controlnet.family_for_checkpoint(checkpoint),
                shared.path_manager,
                model,
                signature,
            )
            return dict(cn_selection=ui_controlnet.NEWCN, **selected.ui_values())

        editor_workspace.bind(
            dict(zip(state["ctrls_name"], state["ctrls_obj"])),
            run_request=generate_request,
            stop_request=stop_clicked,
            run_controls=editor_run_controls,
            top_tabs=top_tabs,
            create_tabs=create_tabs,
            navigation=shell.navigate("editor"),
            guide_checkpoint=effective_image_checkpoint,
            guide_control_plan=guide_control_plan,
        )
        bind_gallery_actions(
            top_tabs,
            create_tabs,
            preset_unselect,
            preset_reset_outputs,
            image_tools_tabs=image_tools_tabs,
            editor_workspace=editor_workspace,
            navigation=shell.navigate,
            navigation_outputs=shell.outputs,
            gallery_tab=shell.gallery,
            run_request=generate_request,
            stop_request=stop_clicked,
            run_controls=dict(
                run=run_button,
                stop=stop_button,
                progress=progress_html,
                preview=main_view,
                inpaint=inpaint_view,
                inpaint_toggle=inpaint_toggle,
                gallery=gallery,
                metadata=metadata_json,
                hint=hint_text,
                task=generation_task,
                originals=generation_originals,
                artifacts=generation_artifacts,
                native_source=native_main_source,
                image_source=shared.get_ctrl("image_source"),
            ),
        )
        shell.gallery.select(
            browser_refresh,
            inputs=browser_refresh_inputs,
            outputs=browser_refresh_outputs,
            js=ui_image_gallery.request_version_js,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        main_tabs.load(
            shared.models.start_metadata_updates,
            inputs=[],
            outputs=[],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        main_tabs.load(
            fn=None,
            inputs=[],
            outputs=[app_settings.language_control],
            js=locale_change_js(
                args.language or settings.get("ui_language", "auto"), initialize=True
            ),
            queue=False,
        )

        # Server-built text follows the browser's language: once on load (cookie or
        # Accept-Language) and again whenever the picker changes.
        def localized_renderer(render, from_picker):
            if from_picker:

                def run(request: gr.Request, language, *values):
                    return render(resolve_locale(language, request), *values)

            else:

                def run(request: gr.Request, *values):
                    return render(resolve_locale(request=request), *values)

            return run

        for render, inputs, outputs in localized_outputs():
            main_tabs.load(
                localized_renderer(render, False),
                inputs=inputs,
                outputs=outputs,
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )
            app_settings.language_control.change(
                localized_renderer(render, True),
                inputs=[app_settings.language_control, *inputs],
                outputs=outputs,
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

    normalize_hidden_progress(main_tabs)
    shared.server_app, shared.local_url, shared.share_url = main_tabs.launch(
        inbrowser=inbrowser,
        server_name=args.listen,
        server_port=args.port,
        share=False,
        auth=(args.auth.split("/", 1) if isinstance(args.auth, str) and "/" in args.auth else None),
        theme=theme,
        i18n=catalogue().native(),
        favicon_path=favicon_path,
        css=html.css,
        head=html.startup_head(
            main_tabs.app_id,
            themed=not theme_warning and bool(theme_id(settings.get("theme"))),
        ),
        js=html.scripts,
        allowed_paths=owner_media_paths(path_manager, shared.models, settings),
        app_kwargs={
            "middleware": [Middleware(PrivateResponses), Middleware(GradioCompletionGuard)]
        },
        enable_monitoring=False,
        pwa=False,
        mcp_server=args.mcp,
        prevent_thread_lock=True,
    )


def update_clicked():
    return {
        run_button: gr.update(interactive=False, visible="hidden"),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True,
            value=html.make_progress_html(0, "Please wait ..."),
        ),
        gallery: gr.update(visible="hidden"),
        main_view: gr.update(visible=True, value=str(asset_path("init_image.png"))),
        inpaint_view: gr.update(
            visible="hidden",
            interactive=True,
        ),
        hint_text: gr.update(visible=True, value=ui_hints.get_hint()),
    }


def update_preview(product):
    percentage, title, image = product
    return {
        run_button: gr.update(interactive=False, visible="hidden"),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(visible=True, value=html.make_progress_html(percentage, title)),
        main_view: (gr.update(visible=True, value=image) if image is not None else gr.update()),
    }


def update_results(product):
    from ruinedfooocus.features.image_processing.native_image_outputs import (
        latest_generated_result,
        result_presentation,
    )

    displays, originals, records, metadata = result_presentation(product)
    tensors = getattr(product, "tensor_files", ())
    if tensors:
        from ruinedfooocus.features.models.model_identity import file_signature

        originals = list(dict.fromkeys([*originals, *(record["path"] for record in records)]))
        for artifact in tensors:
            if tuple(file_signature(artifact.path)) != artifact.signature:
                raise ValueError("A generated native tensor changed before display.")
            originals.append(artifact.path)

    if tensors and not displays:
        # A real tensor-only utility updates its File surface without clearing
        # the user's existing image gallery, source or image metadata.
        return {
            run_button: gr.update(interactive=True, visible=True),
            stop_button: gr.update(interactive=False, visible="hidden"),
            progress_html: gr.update(visible="hidden"),
            generation_originals: gr.update(value=originals, visible=True),
        }

    return {
        run_button: gr.update(interactive=True, visible=True),
        stop_button: gr.update(interactive=False, visible="hidden"),
        progress_html: gr.update(visible="hidden"),
        main_view: gr.update(
            value=displays[0], buttons=["fullscreen"] if originals else ["download", "fullscreen"]
        )
        if displays
        else gr.update(),
        inpaint_toggle: gr.update(value=False),
        gallery: gr.update(
            visible=True,
            selected_index=0,
            allow_preview=True,
            preview=True,
            value=displays,
            buttons=["fullscreen"] if originals else ["download", "fullscreen"],
        ),
        generation_originals: gr.update(value=originals, visible=bool(originals)),
        generation_artifacts: records,
        editor_workspace.latest_generated: latest_generated_result(records) or gr.skip(),
        editor_workspace.source_input: "preview" if displays else gr.skip(),
        native_main_source: records[0] if records and records[0]["precision"] else None,
        shared.get_ctrl("image_source"): None,
        metadata_json: gr.update(value=metadata),
    }


def append_work(gen_data):
    if "_image_workflow_step" in gen_data:
        return worker.add_task(gen_data)
    from ruinedfooocus.features.image_processing.image_composition import handles

    if handles(ui_controlnet.tool_kind(gen_data.get("cn_selection"), gen_data.get("cn_type"))):
        return worker.add_task(
            dict(
                gen_data,
                task_type="process",
                prompt="",
                image_total=1,
                image_number=1,
                generate_forever=False,
            )
        )
    from ruinedfooocus.features.image_processing import latent_composition

    if latent_composition.handles(
        ui_controlnet.tool_kind(gen_data.get("cn_selection"), gen_data.get("cn_type"))
    ):
        return worker.add_task(dict(latent_composition.ui_request(gen_data), task_type="process"))
    from ruinedfooocus.features.image_processing.native_image_outputs import capture_original_source

    tmp_data = capture_original_source(gen_data.copy())
    from ruinedfooocus.features.image_processing.still_image_utilities import is_utility

    kind = (
        ui_controlnet.tool_kind(tmp_data["cn_selection"], tmp_data.get("cn_type"))
        if tmp_data.get("cn_selection")
        else (tmp_data.get("controlnet") or {}).get("type")
    )
    if kind in ("depth-map", "restore-seedvr2") or is_utility(kind):
        return worker.add_task(dict(tmp_data, task_type="process", prompt="", image_total=1))
    imported_prompt = tmp_data.pop("_imported_prompt", None)
    from ruinedfooocus.features.gallery.image_metadata import restore_prompt_provenance

    restore_prompt_provenance(tmp_data, imported_prompt)
    if use_random_prompt(tmp_data, tmp_data["prompt"]):
        prompts = []
        prompts.append("")
    else:
        prompts = get_promptlist(tmp_data)

    tmp_data["image_total"] = len(prompts) * tmp_data["image_number"]
    tmp_data["task_type"] = "process"
    if len(prompts) == 1:
        tmp_data["prompt"] = prompts[0]
    else:
        tmp_data["prompt"] = prompts

    task_id = worker.add_task(tmp_data.copy())
    return task_id


def generate_clicked(*args):
    *values, version = args
    yield from editor_workspace.run(dict(zip(state["ctrls_name"], values)), version)


def generate_request(values):
    """Stream one Create or Library request through the existing owned worker."""
    from copy import deepcopy
    from ruinedfooocus.features.image_processing.image_composition import handles

    task_id = None
    finished = False
    composition_utility = handles(
        ui_controlnet.tool_kind(values.get("cn_selection"), values.get("cn_type"))
    )
    gen_data = dict(values) if composition_utility else deepcopy(values)
    depth_map = ui_controlnet.tool_kind(gen_data.get("cn_selection"), gen_data.get("cn_type")) in (
        "depth-map",
        "restore-seedvr2",
    )
    input_upscale = (
        not (depth_map or composition_utility)
        and gen_data.get("automatic_upscale") != "Off"
        and gen_data.get("automatic_upscale_target") == "Input image only"
    )

    def display(updates):
        updates[generation_task] = None if finished else task_id
        if input_upscale:
            updates.pop(
                main_view, None
            )  # Preserve the generated image while using the upload view.
        return updates

    yield display(update_clicked())

    try:
        from ruinedfooocus.features.generation.ui_request import build_ui_request

        gen_data = build_ui_request(
            gen_data,
            models=shared.models,
            settings=shared.settings.default_settings,
            preset_mode=settings.get("preset_mode", 7),
            family_for=ui_controlnet.family_for_checkpoint,
        )
        task_id = append_work(gen_data)
    except Exception as error:
        from ruinedfooocus.core.task_errors import describe_error

        gr.Warning(describe_error(error))
        yield display(update_results([]))
        return
    shared.state["interrupted"] = False
    finished = False

    try:
        # Publish the exact browser-owned job before waiting for preparation/compute.
        yield {generation_task: task_id}
        while not finished:
            try:
                flag, product = worker.task_result(task_id)
            except Exception as error:
                from ruinedfooocus.core.task_errors import describe_error

                worker.interrupt_processing(task_id)
                finished = True
                gr.Warning(describe_error(error))
                yield display(update_results([]))
                break

            if flag == "preview":
                yield display(update_preview(product))

            elif flag == "error":
                gr.Warning(f"Generation failed: {product}")

            elif flag == "results":
                finished = True
                from ruinedfooocus.features.generation.image_output_batches import (
                    ImageBatchArtifacts,
                )

                if isinstance(product, ImageBatchArtifacts) and not product:
                    gr.Info(t("No matching instances found."))
                if isinstance(product, dict) and product.get("error"):
                    gr.Warning(product["error"])
                    product = []
                yield display(update_results(product))
            time.sleep(0.1)
    finally:
        if not finished:
            worker.interrupt_processing(task_id)
        shared.state["interrupted"] = False


def stop_clicked(task_id):
    if task_id is not None:
        worker.interrupt_processing(task_id)


settings = settings.default_settings

metadata_json = gr.Json()

shared.wildcards = get_wildcard_files()

shared.gradio_root = gr.Blocks()

with shared.gradio_root as block:
    block.load()
    run_event = gr.Number(visible="hidden", value=0)
    generation_task = gr.State(None)
    generation_artifacts = gr.State([])
    native_main_source = gr.State(None)
    add_ctrl("image_source", gr.State(None))
    add_ctrl("_native_main_source", native_main_source)
    add_ctrl("_generation_artifacts", generation_artifacts)
    add_ctrl("run_event", run_event)

    def get_cfg_timestamp():
        return shared.state["last_config"]

    cfg_timestamp = gr.Textbox(visible="hidden", value=get_cfg_timestamp())
    cfg_timer = gr.Timer(value=5)
    cfg_timer.tick(
        fn=get_cfg_timestamp,
        api_visibility="undocumented",
        outputs=[cfg_timestamp],
        show_progress="hidden",
    )

    with gr.Row(elem_id="generation-layout"):
        with gr.Column(scale=5, min_width=0, elem_id="generation-workspace"):
            main_view = gr.Image(
                label=t("Image"),
                elem_id="main_view",
                value=str(asset_path("init_image.png")),
                height="var(--rf-preview-height)",
                type="filepath",
                image_mode=None,
                visible=True,
                buttons=["download", "fullscreen"],
            )
            add_ctrl("main_view", main_view)
            generation_originals = gr.File(
                label=translate_ui("Original output files"),
                file_count="multiple",
                interactive=False,
                visible=False,
            )
            add_ctrl("_generation_originals", generation_originals)
            upscale_upload = gr.Image(
                label=t("Drop Image Here or Click to Upload"),
                type="pil",
                image_mode="RGBA",
                format="png",
                height="var(--rf-preview-height)",
                visible=False,
                elem_id="upscale-upload",
                buttons=["download", "fullscreen"],
            )
            add_ctrl("upscale_upload", upscale_upload)
            inpaint_view = gr.ImageEditor(
                format="png",
                height="var(--rf-preview-height)",
                elem_id="inpaint_view",
                type="numpy",
                visible="hidden",
                show_label=False,
                buttons=["download", "fullscreen"],
                layers=False,
                interactive=True,
                transforms=(),
                brush=gr.Brush(colors=[PAINT_COLOR], color_mode="fixed"),
            )
            add_ctrl("inpaint_view", inpaint_view)
            edit_mode = gr.State(True)

            progress_html = gr.HTML(
                value=html.make_progress_html(32, "Progress 32%"),
                visible="hidden",
                padding=False,
                elem_id="progress-bar",
                elem_classes="progress-bar",
            )

            gallery = gr.Gallery(
                label=None,
                show_label=False,
                object_fit="scale-down",
                height=60,
                allow_preview=False,
                preview=False,
                interactive=False,
                visible="hidden",
                buttons=["download", "fullscreen"],
                value=[str(asset_path("init_image.png"))],
            )
            add_ctrl("_generation_gallery", gallery)

            from ruinedfooocus.features.gallery.ui_workspace import EditorWorkspace

            editor_workspace = EditorWorkspace()
            add_ctrl("_editor_destination", editor_workspace.destination)
            add_ctrl("_editor_context", editor_workspace.context)
            add_ctrl("_editor_source_input", editor_workspace.source_input)
            add_ctrl("_editor_action", editor_workspace.action)
            add_ctrl("_editor_denoise", editor_workspace.denoise)
            add_ctrl("_editor_output_explicit", editor_workspace.output_explicit)
            add_ctrl("_editor_prompt_processing", editor_workspace.prompt_processing)
            add_ctrl("_editor_mask", editor_workspace.mask)
            add_ctrl("_editor_background", editor_workspace.background)
            add_ctrl("_editor_framing", editor_workspace.framing)
            add_ctrl("_editor_mask_choice", editor_workspace.mask_choice)
            add_ctrl("_editor_mask_index", editor_workspace.mask_index)
            add_ctrl("_editor_mask_count", editor_workspace.mask_count)
            add_ctrl("_editor_mask_invert", editor_workspace.invert)
            add_ctrl("_editor_guide_handoff", editor_workspace.guided_results.handoff)
            add_ctrl("control_guide_denoise", editor_workspace.guided_results.denoise)

            with gr.Row(elem_id="prompt-row"):
                with gr.Column(scale=5, min_width=0):
                    with gr.Group():
                        with gr.Group(), gr.Row():
                            prompt = gr.Textbox(
                                label=t("Prompt"),
                                show_label=False,
                                placeholder=t("Type prompt here."),
                                container=False,
                                autofocus=True,
                                lines=3,
                                value=settings["prompt"],
                                scale=5,
                            )
                            add_ctrl("prompt", prompt)
                            add_ctrl("_imported_prompt", gr.State(None))

                with gr.Column(scale=1, min_width=0, elem_id="prompt-actions"):
                    run_button = gr.Button(
                        value=t("Generate"), variant="primary", elem_id="generate"
                    )
                    stop_button = gr.Button(
                        value=t("Stop"), variant="stop", interactive=False, visible="hidden"
                    )
                    spellcheck = gr.Dropdown(
                        label=translate_ui("Wildcards"),
                        visible="hidden",
                        choices=[],
                        value=None,
                        scale=1,
                    )

            @gallery.select(
                api_visibility="undocumented",
                inputs=[gallery, generation_artifacts],
                outputs=[
                    main_view,
                    metadata_json,
                    native_main_source,
                    editor_workspace.latest_generated,
                ],
                show_progress="hidden",
            )
            def gallery_change(files, records, sd: gr.SelectData):
                if records:
                    from ruinedfooocus.features.image_processing.native_image_outputs import (
                        latest_generated_result,
                        selected_result,
                    )

                    display, metadata, original = selected_result(records, sd.index)
                    return {
                        main_view: gr.update(value=display),
                        metadata_json: gr.update(value=metadata),
                        native_main_source: original,
                        editor_workspace.latest_generated: latest_generated_result(
                            records, sd.index
                        )
                        or gr.skip(),
                    }
                name = sd.value["image"]["path"]
                with Image.open(name) as im:
                    if im.info.get("parameters"):
                        metadata = im.info["parameters"]
                    else:
                        metadata = {"Data": "Preview Grid"}
                return {
                    main_view: gr.update(value=name),
                    metadata_json: gr.update(value=metadata),
                    native_main_source: None,
                }

            @prompt.input(api_visibility="undocumented", inputs=prompt, outputs=spellcheck)
            def checkforwildcards(text):
                test = find_unclosed_markers(text)
                if test is not None:
                    filtered = [s for s in shared.wildcards if test in s]
                    filtered.append(" ")
                    return {
                        spellcheck: gr.update(
                            interactive=True,
                            visible=True,
                            choices=filtered,
                            value=" ",
                        )
                    }
                else:
                    return {spellcheck: gr.update(interactive=False, visible="hidden")}

            @spellcheck.select(
                api_visibility="undocumented", inputs=[prompt, spellcheck], outputs=prompt
            )
            def select_spellcheck(text, selection):
                last_idx = text.rindex("__")
                newtext = f"{text[:last_idx]}__{selection}__"
                return {prompt: gr.update(value=newtext)}

            @main_view.upload(
                api_visibility="undocumented",
                inputs=[main_view, prompt, editor_workspace.destination],
                outputs=[
                    prompt,
                    gallery,
                    generation_artifacts,
                    native_main_source,
                    generation_originals,
                    shared.get_ctrl("image_source"),
                ],
            )
            def load_images_handler(file, prompt, destination):
                params = gr.skip()
                if destination == "create" and not prompt:
                    with Image.open(file) as image:
                        params = metadata_prompt(image) or ""
                return params, [file], [], None, gr.update(value=[], visible=False), None

        with gr.Column(scale=2, min_width=0, elem_id="generation-controls"):
            with (
                gr.Tabs(elem_id="create-controls-tabs") as create_controls_tabs,
                gr.Tab(label=t("Generation"), id="generation") as generation_tab,
            ):
                initial_checkpoint = primary_checkpoint_selection(
                    settings["base_model"],
                    lambda name: shared.models.get_models_by_path("checkpoints", name, fetch=False),
                    names=shared.models.get_names("checkpoints"),
                )
                base_model = gr.Dropdown(
                    label=translate_ui("Image model"),
                    choices=checkpoint_choices(
                        shared.models.get_names("checkpoints"),
                        initial_checkpoint,
                        lambda name: shared.models.get_models_by_path(
                            "checkpoints", name, fetch=False
                        ),
                    ),
                    value=initial_checkpoint,
                    interactive=True,
                    elem_id="create-model",
                )
                with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
                    create_models_button = gr.Button(
                        t("Models"), size="sm", elem_id="create-open-models"
                    )
                preset_notice = gr.Markdown("", visible=False)
                clear_preset_btn = gr.Button(
                    translate_ui("Clear generation preset"), visible=False, size="sm"
                )
                with gr.Accordion(
                    translate_ui("Sampling"), open=True, elem_id="generation-sampling"
                ):
                    initial_performance = settings["performance"]
                    initial_model_base = shared.models.get_model_base(
                        shared.models.get_models_by_path("checkpoints", initial_checkpoint)
                    )
                    model_family = gr.State(initial_model_base)
                    effective_image_checkpoint = gr.Textbox(value=initial_checkpoint, visible=False)
                    model_metadata_refresh = gr.Number(value=0, precision=0, visible=False)
                    initial_performance_choices = performance_settings.choices_for_model(
                        initial_model_base, initial_checkpoint
                    )
                    initial_performance = performance_settings.selection_for_model(
                        initial_model_base, initial_checkpoint, initial_performance
                    )

                    def custom_visible(key):
                        return initial_performance == performance_settings.CUSTOM_PERFORMANCE

                    performance_selection = gr.Dropdown(
                        label=t("Performance"),
                        choices=initial_performance_choices,
                        value=initial_performance,
                        buttons=[],
                    )
                    add_ctrl("performance_selection", performance_selection)
                    performance_help = gr.Markdown(
                        performance_settings.describe(initial_performance),
                    )
                    with gr.Accordion(t("Workflow settings"), open=False):
                        initial_recipe = performance_settings.control_options(
                            initial_performance, initial_checkpoint, initial_model_base
                        )
                        workflow_help = gr.Markdown(
                            performance_settings.describe_workflow(
                                initial_model_base,
                                initial_checkpoint,
                                initial_performance,
                                initial_recipe["custom_steps"],
                                initial_recipe["cfg"],
                                initial_recipe["sampler_name"],
                                initial_recipe["scheduler"],
                            )
                        )

                    def describe_performance(name, request: gr.Request = None):
                        return performance_settings.describe(
                            name, locale=resolve_locale(request=request)
                        )

                    performance_selection.change(
                        fn=describe_performance,
                        inputs=[performance_selection],
                        outputs=[performance_help],
                        api_visibility="undocumented",
                    )
                    custom_default_values = performance_settings.control_options(
                        initial_performance, initial_checkpoint, initial_model_base
                    )
                    if initial_performance == performance_settings.CUSTOM_PERFORMANCE:
                        custom_default_values |= settings.get("custom_sampling", {})
                    custom_steps = gr.Slider(
                        label=t("Custom Steps"),
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=custom_default_values["custom_steps"],
                        visible=custom_visible("custom_steps"),
                    )
                    add_ctrl("custom_steps", custom_steps)

                    cfg = gr.Slider(
                        label=t("CFG"),
                        minimum=0.0,
                        maximum=20.0,
                        step=0.1,
                        value=custom_default_values["cfg"],
                        visible=custom_visible("cfg"),
                    )
                    add_ctrl("cfg", cfg)
                    sampler_name = gr.Dropdown(
                        label=t("Sampler"),
                        choices=sorted(KSampler.SAMPLERS),
                        value=custom_default_values["sampler_name"],
                        visible=custom_visible("sampler_name"),
                    )
                    add_ctrl("sampler_name", sampler_name)
                    from ruinedfooocus.features.generation.native_sampling import scheduler_choices

                    scheduler = gr.Dropdown(
                        label=t("Scheduler"),
                        choices=scheduler_choices(KSampler.SCHEDULERS),
                        value=custom_default_values["scheduler"],
                        visible=custom_visible("scheduler"),
                    )
                    add_ctrl("scheduler", scheduler)

                    clip_skip = gr.Slider(
                        label=t("Clip Skip"),
                        info=translate_ui(
                            "Used by ordinary text encoding. Native image-edit encoders choose their own layers."
                        ),
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=custom_default_values["clip_skip"],
                        visible=custom_visible("clip_skip"),
                    )

                    add_ctrl("clip_skip", clip_skip)

                    with gr.Accordion(translate_ui("Model Defaults"), open=False):
                        recommendation_status = gr.Markdown()
                        auto_recommendations = gr.Checkbox(
                            label=translate_ui("Automatically refresh recommendations"), value=True
                        )
                        with gr.Row():
                            recommendation_use = gr.Button(translate_ui("Use model defaults"))
                            recommendation_refresh = gr.Button(translate_ui("Refresh"))
                        with gr.Row():
                            recommendation_save = gr.Button(translate_ui("Save sampling defaults"))
                            recommendation_reset = gr.Button(
                                translate_ui("Remove sampling override")
                            )

                    performance_outputs = [
                        cfg,
                        sampler_name,
                        scheduler,
                        clip_skip,
                        custom_steps,
                    ]

                with gr.Accordion(
                    translate_ui("Size and upscale"), open=True, elem_id="generation-size"
                ) as size_section:
                    resolution_settings.set_profile(shared.settings.name)
                    size_mode = gr.State("target")
                    add_ctrl("size_mode", size_mode)
                    initial_choices, initial_resolution, initial_resolution_auto = (
                        resolution_settings.model_selection(
                            initial_model_base,
                            settings["resolution"],
                            settings.get("resolution_auto", False),
                            initial_checkpoint,
                            performance_settings,
                        )
                    )
                    custom_resolution = initial_resolution == resolution_settings.CUSTOM_RESOLUTION
                    resolution_auto = gr.Checkbox(
                        label=t("Follow model recommended size"),
                        value=initial_resolution_auto,
                        info=t("Choosing a size keeps your override until you enable this again."),
                    )
                    add_ctrl("resolution_auto", resolution_auto)
                    aspect_ratios_selection = ResolutionDropdown(
                        label=t("Output target"),
                        choices=initial_choices,
                        value=initial_resolution,
                    )
                    add_ctrl("aspect_ratios_selection", aspect_ratios_selection)
                    # The checkbox above already restores the recommended size.
                    restore_recommended_size = gr.Button(
                        t("Use model recommended size"), size="sm", visible=False
                    )
                    with gr.Accordion(t("Size details"), open=False):
                        resolution_recommendation = gr.Markdown(
                            resolution_settings.recommendation_text(
                                initial_model_base,
                                checkpoint=initial_checkpoint,
                                performance=performance_settings,
                                selection=initial_resolution,
                                width=settings.get("custom_width", 1024),
                                height=settings.get("custom_height", 1024),
                                automatic=initial_resolution_auto,
                            )
                        )
                    ratio_delete = gr.Button(
                        t("Remove saved custom size"),
                        variant="stop",
                        visible=resolution_settings.is_custom(initial_resolution),
                        size="sm",
                    )
                    ratio_name = gr.Textbox(
                        label=t("Saved size name"),
                        show_label=True,
                        placeholder=t("Name"),
                        interactive=True,
                        visible=custom_resolution,
                    )
                    default_resolution = resolution_settings.aspect_ratios.get(
                        initial_resolution,
                        (settings.get("custom_width", 1024), settings.get("custom_height", 1024)),
                    )
                    custom_width = gr.Number(
                        label=t("Target width"),
                        minimum=1,
                        maximum=32768,
                        precision=0,
                        visible=custom_resolution,
                        value=default_resolution[0],
                    )
                    add_ctrl("custom_width", custom_width)
                    custom_height = gr.Number(
                        label=t("Target height"),
                        minimum=1,
                        maximum=32768,
                        precision=0,
                        visible=custom_resolution,
                        value=default_resolution[1],
                    )
                    add_ctrl("custom_height", custom_height)
                    with gr.Row(visible=custom_resolution) as target_shortcuts:
                        target_1080p = gr.Button(t("1080p"), size="sm", min_width=80)
                        target_1440p = gr.Button(t("1440p"), size="sm", min_width=80)
                        target_4k = gr.Button(t("4K"), size="sm", min_width=80)
                    for button, dimensions in (
                        (target_1080p, (1920, 1080)),
                        (target_1440p, (2560, 1440)),
                        (target_4k, (3840, 2160)),
                    ):
                        button.click(
                            lambda dimensions=dimensions: (
                                dimensions[0],
                                dimensions[1],
                                False,
                                "target",
                            ),
                            outputs=[custom_width, custom_height, resolution_auto, size_mode],
                            api_visibility="undocumented",
                            show_progress="hidden",
                        )
                    from ruinedfooocus.features.generation.automatic_upscale import (
                        choices as upscale_choices,
                        describe_size,
                        PRESETS,
                    )

                    with gr.Group() as automatic_upscale_controls:
                        upscale_factor = gr.Dropdown(
                            label=t("Upscale factor"),
                            choices=[("2×", "2"), ("4×", "4"), (t("All native factors"), "all")],
                            value="2",
                            interactive=True,
                        )
                        automatic_upscale = gr.Dropdown(
                            label=t("Automatic Upscale"),
                            choices=upscale_choices(2, advanced=False),
                            value="Off",
                            info=t(
                                "Choose anime, general or another upscale model. Downloaded on first use."
                            ),
                        )
                        add_ctrl("automatic_upscale", automatic_upscale)
                        with gr.Group(visible="hidden") as upscale_options:
                            advanced_upscalers = gr.Checkbox(
                                label=t("Show advanced upscale models"),
                                value=False,
                                info=t(
                                    "Includes registered specialist and inspected local pixel upscalers."
                                ),
                            )
                            automatic_upscale_target = gr.Dropdown(
                                label=t("Upscale source"),
                                choices=[
                                    (translate_ui("Generated image"), "Generated image"),
                                    (translate_ui("Input image only"), "Input image only"),
                                ],
                                value="Generated image",
                                info=t(
                                    "Input image only switches the main preview to an upload area and skips generation."
                                ),
                            )
                            add_ctrl("automatic_upscale_target", automatic_upscale_target)
                            add_ctrl("advanced_upscalers", advanced_upscalers)
                            upscale_size = gr.Textbox(
                                label=t("Effective stages"),
                                interactive=False,
                                lines=2,
                                value=t("Add a source image in Image tools to see its output size.")
                                if ui_controlnet.checkpoint_features(base_model.value).get("sd4x")
                                else describe_size(
                                    "Off",
                                    initial_resolution,
                                    default_resolution[0],
                                    default_resolution[1],
                                    resolution_settings.aspect_ratios,
                                    target_mode=True,
                                ),
                                elem_id="upscale-output-size",
                            )

                        def filter_upscalers(enabled, factor, current):
                            choices = upscale_choices(
                                None if factor == "all" else int(factor),
                                advanced=enabled or factor == "all",
                            )
                            return gr.update(
                                choices=choices, value=current if current in choices else "Off"
                            )

                        gr.on(
                            triggers=[advanced_upscalers.change, upscale_factor.change],
                            fn=filter_upscalers,
                            inputs=[advanced_upscalers, upscale_factor, automatic_upscale],
                            outputs=automatic_upscale,
                            api_visibility="undocumented",
                        )
                    ratio_save = gr.Button(
                        value=t("Save size"),
                        visible=custom_resolution,
                    )

                    @ratio_save.click(
                        api_visibility="undocumented",
                        inputs=[ratio_name, custom_width, custom_height, model_family],
                        outputs=[aspect_ratios_selection, resolution_auto],
                    )
                    def ratio_save_click(ratio_name, custom_width, custom_height, family):
                        ratio_name = ratio_name.strip()
                        if ratio_name != "":
                            try:
                                new_ratio_name = resolution_settings.save_custom(
                                    ratio_name,
                                    custom_width,
                                    custom_height,
                                    family,
                                )
                            except ValueError as error:
                                raise gr.Error(str(error)) from None
                            choices = resolution_settings.choices_for_model(family)
                            return gr.update(choices=choices, value=new_ratio_name), False
                        else:
                            return gr.update(), gr.update()

                    @ratio_delete.click(
                        inputs=[aspect_ratios_selection, model_family],
                        outputs=aspect_ratios_selection,
                        api_visibility="undocumented",
                    )
                    def ratio_delete_click(selection, family):
                        resolution_settings.remove_custom(selection)
                        return gr.update(
                            choices=resolution_settings.choices_for_model(family),
                            value=resolution_settings.selection_for_model(family, None),
                        )

                with gr.Accordion(translate_ui("Styles"), open=False, elem_id="generation-style"):
                    style_controls = ui_styles.create_ui(settings)
                    style_selection = style_controls.selection
                    add_ctrl("style_selection", style_selection)
                image_number = gr.Slider(
                    label=t("Image Number"),
                    minimum=0,
                    maximum=settings.get("image_number_max", 50),
                    step=1,
                    value=settings.get("image_number", 1),
                )
                add_ctrl("image_number", image_number)
                with gr.Accordion(
                    translate_ui("Negative prompt and seed"), open=False, elem_id="generation-seed"
                ):
                    auto_negative_prompt = gr.Checkbox(
                        label=t("Auto Negative Prompt"),
                        show_label=True,
                        value=settings["auto_negative_prompt"],
                    )
                    add_ctrl("auto_negative", auto_negative_prompt)
                    negative_prompt = gr.Textbox(
                        label=translate_ui("Negative Prompt"),
                        show_label=True,
                        placeholder=translate_ui("Type prompt here."),
                        value=settings["negative_prompt"],
                    )
                    add_ctrl("negative", negative_prompt)
                    seed_random = gr.Checkbox(label=t("Random Seed"), value=settings["seed_random"])
                    add_ctrl("seed_random", seed_random)
                    image_seed = gr.Number(
                        label=t("Seed"),
                        value=settings["seed"],
                        precision=0,
                        visible="hidden" if settings["seed_random"] else True,
                    )
                    add_ctrl("seed", image_seed)
                preset_accordion = gr.Accordion(
                    label=t(PNG_PRESET_LABEL),
                    open=False,
                )
                with preset_accordion:
                    gr.Markdown(
                        translate_ui(
                            "Reuse model/LoRAs, sampling and size from a saved RuinedFooocus PNG while writing new prompts. These settings stay active until cleared. Choose which groups to reuse in **Settings → Generation defaults → Saved settings (PNG)**."
                        )
                    )
                    with gr.Group(), gr.Column():
                        preset_image = gr.Image(
                            buttons=["download", "fullscreen"],
                            placeholder=translate_ui("Select or drop preset image here"),
                            show_label=False,
                            type="filepath",
                            interactive=True,
                            sources=["upload"],
                        )
                        preset_gallery = gr.Gallery(
                            show_label=False,
                            height="auto",
                            allow_preview=False,
                            preview=False,
                            columns=2,
                            rows=3,
                            object_fit="contain",
                            visible=True,
                            buttons=["download", "fullscreen"],
                            min_width=60,
                            selected_index=None,
                            value=path_manager.get_presets(),
                        )
                preset_selection = gr.Text(
                    value=None,
                    visible="hidden",
                )
                add_ctrl("preset_selection", preset_selection)
                preset_selection.change(
                    lambda selected: (
                        gr.update(interactive=not bool(selected)),
                        gr.update(
                            value="Saved PNG settings are active. Clear the preset to use your current controls.",
                            visible=bool(selected),
                        ),
                        gr.update(visible=bool(selected)),
                    ),
                    inputs=preset_selection,
                    outputs=[base_model, preset_notice, clear_preset_btn],
                    api_visibility="undocumented",
                    queue=False,
                )

                save_defaults_btn = gr.Button(
                    translate_ui("Save current defaults"),
                    size="sm",
                    elem_id="save-generation-defaults",
                )

                @seed_random.change(
                    api_visibility="undocumented", inputs=[seed_random], outputs=[image_seed]
                )
                def random_checked(r):
                    from ruinedfooocus.features.generation.ui_seed import seed_visibility

                    return seed_visibility(r)

                def refresh_seed(r, s):
                    if r:
                        return -1
                    else:
                        return s

            # Everyday choices stay in view: upscaling folds into its own
            # section and rarely used file and layout options into one.
            with size_section:
                upscale_section = gr.Accordion(
                    t("Upscale"), open=False, elem_id="generation-upscale"
                )
            with upscale_section:
                automatic_upscale_controls.unrender().render()
            with create_controls_tabs:
                model_tab = gr.Tab(
                    label=t("Models"),
                    id="create-models",
                    visible=True,
                    elem_id="models-panel",
                )
            with create_controls_tabs, model_tab:
                with gr.Tab(label=t("Model")):
                    from ruinedfooocus.features.models.ui_browser import (
                        create_ui as create_model_browser,
                    )

                    model_catalogue_button = gr.Button(
                        translate_ui("Browse model catalogue"),
                        render=False,
                        size="sm",
                        elem_id="model-catalogue-open",
                    )
                    modelfilter, model_gallery, model_gallery_records, update_model_filter = (
                        create_model_browser(
                            base_model,
                            shared.models,
                            model_refresh=model_metadata_refresh,
                            catalogue_button=model_catalogue_button,
                        )
                    )
                    add_ctrl("base_model_name", base_model)

                    def model_image_controls(name, performance):
                        model_base = shared.models.get_model_base(
                            shared.models.get_models_by_path("checkpoints", name)
                        )
                        choices = performance_settings.choices_for_model(model_base, name)
                        selection = performance_settings.selection_for_model(
                            model_base, name, performance
                        )
                        return (
                            gr.update(choices=choices, value=selection),
                            model_base,
                        )

                    model_controls_event = gr.on(
                        triggers=[base_model.change, model_metadata_refresh.change],
                        fn=model_image_controls,
                        inputs=[base_model, performance_selection],
                        outputs=[
                            performance_selection,
                            model_family,
                        ],
                        api_visibility="undocumented",
                        show_progress="hidden",
                    )

                    def model_resolution(family, checkpoint, current, automatic, *, force=False):
                        choices, selection, automatic = resolution_settings.model_selection(
                            family,
                            current,
                            automatic,
                            checkpoint,
                            performance_settings,
                            force=force,
                        )
                        return gr.update(choices=choices, value=selection), automatic

                    resolution_auto.input(
                        lambda family, checkpoint, current, automatic: (
                            *model_resolution(family, checkpoint, current, automatic, force=True),
                            "target",
                        ),
                        inputs=[model_family, base_model, aspect_ratios_selection, resolution_auto],
                        outputs=[aspect_ratios_selection, resolution_auto, size_mode],
                        api_visibility="undocumented",
                        trigger_mode="always_last",
                        show_progress="hidden",
                        show_progress_on=[],
                    )
                    restore_recommended_size.click(
                        lambda family, checkpoint, current: (
                            *model_resolution(family, checkpoint, current, True, force=True),
                            "target",
                        ),
                        inputs=[model_family, base_model, aspect_ratios_selection],
                        outputs=[aspect_ratios_selection, resolution_auto, size_mode],
                        api_visibility="undocumented",
                        show_progress="hidden",
                        show_progress_on=[],
                    )
                    for size_control in (aspect_ratios_selection, custom_width, custom_height):
                        size_control.input(
                            lambda: (False, "target"),
                            outputs=[resolution_auto, size_mode],
                            api_visibility="undocumented",
                            show_progress="hidden",
                        )

                with gr.Tab(label=translate_ui("LoRAs & embeddings")):
                    from ruinedfooocus.features.models.ui_adapters import (
                        create_ui as create_adapter_ui,
                    )
                    from ruinedfooocus.features.generation.defaults import startup_loras

                    adapters = create_adapter_ui(
                        prompt,
                        negative_prompt,
                        models=shared.models,
                        directories=lambda: path_manager.model_paths["embedding_path"],
                        cache_dir=path_manager.model_paths["cache_path"],
                        defaults=startup_loras(settings),
                        thumbnail=get_lora_thumbnail,
                        keywords=load_keywords,
                        weight_min=settings.get("lora_min", 0),
                        weight_max=settings.get("lora_max", 2),
                    )
                    lora_active_gallery = adapters.transport
                    lorafilter = adapters.search
                    lora_keywords = adapters.keywords
                    add_ctrl("loras", lora_active_gallery)
                    add_ctrl("lora_keywords", lora_keywords)
                    ui_styles.add_lora_keywords_control(style_selection)

                with gr.Row():
                    model_refresh = gr.Button(
                        value=translate_ui("Refresh all files"),
                        variant="secondary",
                        elem_classes="refresh_button",
                    )

            @model_refresh.click(
                api_visibility="undocumented",
                inputs=[
                    lora_active_gallery,
                    base_model,
                    modelfilter,
                    lorafilter,
                    model_metadata_refresh,
                ],
                outputs=[
                    model_gallery,
                    model_gallery_records,
                    adapters.catalog,
                    adapters.status,
                    style_selection,
                    preset_gallery,
                    base_model,
                    model_metadata_refresh,
                ],
                queue=False,
                show_progress="hidden",
            )
            def model_refresh_clicked(
                lora_active_gallery, selected_model, model_filter, lora_filter, revision
            ):
                shared.models.update_all_models()

                gallery_update, records = update_model_filter(model_filter)
                adapter_records, adapter_status = adapters.discover()
                primary = primary_checkpoint_selection(
                    selected_model,
                    lambda name: shared.models.get_models_by_path("checkpoints", name, fetch=False),
                )
                results = {
                    model_gallery: gallery_update,
                    model_gallery_records: records,
                    adapters.catalog: adapter_records,
                    adapters.status: adapter_status,
                    style_selection: gr.update(choices=list(load_styles().keys())),
                    preset_gallery: gr.update(value=path_manager.get_presets()),
                    base_model: gr.update(
                        choices=checkpoint_choices(
                            shared.models.get_names("checkpoints"),
                            selected_model,
                            lambda name: shared.models.get_models_by_path(
                                "checkpoints", name, fetch=False
                            ),
                        ),
                        **({"value": primary} if primary != selected_model else {}),
                    ),
                    model_metadata_refresh: int(revision or 0) + 1,
                }

                return results

            # Refresh thumbnails when the background metadata scan finishes.
            model_revision = gr.State(-1)
            selected_metadata_identity = gr.State(None)

            def update_model_previews(
                revision,
                model_filter,
                lora_filter,
                active_loras,
                selected_model,
                metadata_revision,
                previous_identity,
            ):
                current = shared.models.revision
                if revision == current:
                    return (gr.skip(),) * 7
                gallery_update, records = update_model_filter(model_filter)
                metadata = (
                    shared.models.get_models_by_path("checkpoints", selected_model)
                    if selected_model
                    else {}
                )
                identity = [selected_model] + [
                    metadata.get(key)
                    for key in ("baseModel", "id", "hf_repo_id", "_rf_file_signature")
                ]
                primary = primary_checkpoint_selection(selected_model, lambda name: metadata)
                return (
                    current,
                    gallery_update,
                    records,
                    adapters.discover()[0],
                    gr.update(
                        choices=checkpoint_choices(
                            shared.models.get_names("checkpoints"),
                            selected_model,
                            lambda name: shared.models.get_models_by_path(
                                "checkpoints", name, fetch=False
                            ),
                        ),
                        **({"value": primary} if primary != selected_model else {}),
                    ),
                    int(metadata_revision or 0) + 1 if identity != previous_identity else gr.skip(),
                    identity,
                )

            cfg_timer.tick(
                fn=update_model_previews,
                inputs=[
                    model_revision,
                    modelfilter,
                    lorafilter,
                    lora_active_gallery,
                    base_model,
                    model_metadata_refresh,
                    selected_metadata_identity,
                ],
                outputs=[
                    model_revision,
                    model_gallery,
                    model_gallery_records,
                    adapters.catalog,
                    base_model,
                    model_metadata_refresh,
                    selected_metadata_identity,
                ],
                api_visibility="undocumented",
                queue=False,
                show_progress="hidden",
            )

            ui_styles.bind_actions(style_controls, prompt, negative_prompt, lora_keywords)
            with create_controls_tabs:
                bind_prompt_generation = ui_onebutton.ui_onebutton(
                    prompt,
                    run_event,
                    effective_image_checkpoint,
                    style_selection,
                    image_number=image_number,
                    qwen_rewrite=True,
                )

            # Image tools follow a preset's effective checkpoint without changing
            # the user's Main model selection or API-default identity.
            def update_effective_image_checkpoint(checkpoint, preset):
                from ruinedfooocus.features.image_editing.image_guidance import (
                    effective_preset_checkpoint,
                )

                selected = effective_preset_checkpoint(
                    checkpoint, preset, settings.get("preset_mode", 7)
                )
                return gr.update(value=selected) if selected is not None else gr.update()

            gr.on(
                triggers=[
                    base_model.change,
                    preset_selection.change,
                    model_metadata_refresh.change,
                ],
                fn=update_effective_image_checkpoint,
                inputs=[base_model, preset_selection],
                outputs=effective_image_checkpoint,
                api_visibility="undocumented",
            )
            with create_controls_tabs:
                inpaint_toggle = ui_controlnet.add_controlnet_tab(
                    main_view,
                    inpaint_view,
                    prompt,
                    image_number,
                    run_event,
                    effective_image_checkpoint,
                    preset_selection=preset_selection,
                    main_model=base_model,
                    upscale_scale=automatic_upscale,
                    upscale_target=automatic_upscale_target,
                    model_refresh=model_metadata_refresh,
                )
            bind_prompt_generation(shared.get_ctrl("obp_image_prompt"))
            from ruinedfooocus.features.prompting.ui_caption import add_describe_controls

            prompt_tools_tab = next(
                tab for tab in create_controls_tabs.children if tab.elem_id == "prompt-tools"
            )
            with prompt_tools_tab:
                add_describe_controls(
                    prompt,
                    main_view,
                    native_main_source,
                    input_source=shared.get_ctrl("input_image"),
                    source_kind=editor_workspace.source_input,
                )
            from ruinedfooocus.features.image_processing.ui_native_formats import (
                create_ui as create_output_format_ui,
            )

            with create_controls_tabs, generation_tab:
                create_output_format_ui()
            from ruinedfooocus.features.image_processing.ui_color_operations import (
                bind_output_controls as bind_adjustment_outputs,
            )

            bind_adjustment_outputs()
            from ruinedfooocus.features.generation.ui_native_sampling import (
                create_ui as create_native_sampling_ui,
            )

            with create_controls_tabs, generation_tab:
                create_native_sampling_ui(effective_image_checkpoint, model_metadata_refresh)
            from ruinedfooocus.features.image_editing.ui_native_auxiliary import (
                create_ui as create_auxiliary_ui,
            )

            with create_controls_tabs, generation_tab:
                create_auxiliary_ui(
                    effective_image_checkpoint, model_metadata_refresh, inpaint_toggle
                )
            from ruinedfooocus.features.generation.ui_layered import create_ui as create_layered_ui

            with create_controls_tabs, generation_tab:
                create_layered_ui(effective_image_checkpoint, model_metadata_refresh)
            from ruinedfooocus.features.generation.ui_native_tensor_io import (
                create_ui as create_tensor_io_ui,
            )

            with create_controls_tabs, generation_tab:
                create_tensor_io_ui()
            regroup_generation_options(generation_tab, image_number, save_defaults_btn)
            upscale_input = upscale_upload

            def describe_resolution(
                locale,
                family,
                checkpoint,
                selection,
                width,
                height,
                automatic,
                image,
                tool_selection,
                tool_type,
                framing,
                inpainting,
            ):
                kind = ui_controlnet.tool_kind(tool_selection, tool_type)
                from ruinedfooocus.features.image_processing.still_image_utilities import is_utility

                if kind == "image-transform":
                    return localize("Transform size follows its operation controls.", locale)
                if is_utility(kind):
                    return localize("This image tool keeps the source image size.", locale)
                if kind == "depth-map":
                    return localize("Depth map keeps the source image size.", locale)
                source_size = None
                if (
                    not inpainting
                    and image is not None
                    and kind
                    in (
                        "img2img",
                        "edit",
                        "canny",
                        "depth",
                        "sketch",
                        "recolour",
                    )
                ):
                    source_size = image.size
                return resolution_settings.recommendation_text(
                    family,
                    locale=locale,
                    checkpoint=checkpoint,
                    performance=performance_settings,
                    selection=selection,
                    width=width,
                    height=height,
                    automatic=automatic,
                    source_size=source_size,
                    framing=framing,
                )

            def resolution_recommendation_text(
                family,
                checkpoint,
                selection,
                width,
                height,
                automatic,
                image,
                tool_selection,
                tool_type,
                framing,
                inpainting,
                request: gr.Request = None,
            ):
                return describe_resolution(
                    resolve_locale(request=request),
                    family,
                    checkpoint,
                    selection,
                    width,
                    height,
                    automatic,
                    image,
                    tool_selection,
                    tool_type,
                    framing,
                    inpainting,
                )

            resolution_description_inputs = [
                model_family,
                base_model,
                aspect_ratios_selection,
                custom_width,
                custom_height,
                resolution_auto,
                shared.get_ctrl("input_image"),
                shared.get_ctrl("cn_selection"),
                shared.get_ctrl("cn_type"),
                shared.get_ctrl("image_resize_mode"),
                inpaint_toggle,
            ]
            localized_output(
                describe_resolution, resolution_description_inputs, [resolution_recommendation]
            )
            gr.on(
                triggers=[control.change for control in resolution_description_inputs]
                + [model_metadata_refresh.change],
                fn=resolution_recommendation_text,
                inputs=resolution_description_inputs,
                outputs=resolution_recommendation,
                api_visibility="undocumented",
                trigger_mode="always_last",
                show_progress="hidden",
                show_progress_on=[],
            )

            def update_upscale_size(
                scale,
                target,
                resolution,
                width,
                height,
                image,
                checkpoint=None,
                reference=None,
                selection=None,
                control_type=None,
                size_mode="legacy_base",
                automatic=True,
                family=None,
                framing="preserve",
                inpainting=False,
            ):
                kind = ui_controlnet.tool_kind(selection, control_type)
                from ruinedfooocus.features.image_processing.still_image_utilities import is_utility

                if kind == "image-transform":
                    return t("Transform size follows its operation controls.")
                if is_utility(kind):
                    return t("This image tool keeps the source image size.")
                if kind == "depth-map":
                    return t("Depth map keeps the source image size.")
                if inpainting and target != "Input image only":
                    return t("Effective size follows the source image and mask or canvas borders.")
                if target == "Input image only":
                    if image is None:
                        return "Drop an image in the main upload area to see its output size."
                    if automatic or size_mode != "target":
                        width, height = image.size
                        resolution = "Input image"
                        size_mode = "legacy_base"
                elif checkpoint and ui_controlnet.checkpoint_features(checkpoint).get("sd4x"):
                    if kind in ("rembg", "upscale"):
                        return "Output size follows the selected image utility."
                    if reference is None:
                        return "Add a source image in Image tools to see its output size."
                    from ruinedfooocus.features.image_editing.sd4x_conditioning import sd4x_size

                    width, height = sd4x_size(reference)
                    resolution = None
                elif (
                    reference is not None
                    and framing == "preserve"
                    and kind
                    in (
                        "img2img",
                        "edit",
                        "canny",
                        "depth",
                        "sketch",
                        "recolour",
                    )
                ):
                    from ruinedfooocus.features.image_editing.image_geometry import reference_plan

                    selected_size = resolution_settings.aspect_ratios.get(
                        resolution, (width, height)
                    )
                    frame = reference_plan(reference.size, selected_size)
                    width, height, resolution, size_mode = (
                        frame["width"],
                        frame["height"],
                        None,
                        "legacy_base",
                    )
                return describe_size(
                    scale,
                    resolution,
                    width,
                    height,
                    resolution_settings.aspect_ratios,
                    target_mode=size_mode == "target"
                    and not (
                        checkpoint and ui_controlnet.checkpoint_features(checkpoint).get("sd4x")
                    ),
                    alignment=32
                    if family in ("HiDreamO1", "HiDream O1", "HunyuanImage21")
                    else 16
                    if family in ("Flux2", "QwenImage21", "MAGE", "ChromaRadiance")
                    else 8,
                )

            upscale_size_inputs = [
                automatic_upscale,
                automatic_upscale_target,
                aspect_ratios_selection,
                custom_width,
                custom_height,
                upscale_input,
                effective_image_checkpoint,
                shared.get_ctrl("input_image"),
                shared.get_ctrl("cn_selection"),
                shared.get_ctrl("cn_type"),
                size_mode,
                resolution_auto,
                model_family,
                shared.get_ctrl("image_resize_mode"),
                inpaint_toggle,
            ]
            gr.on(
                triggers=[control.change for control in upscale_size_inputs]
                + [model_metadata_refresh.change],
                fn=update_upscale_size,
                inputs=upscale_size_inputs,
                outputs=upscale_size,
                trigger_mode="always_last",
                concurrency_id="upscale_size",
                concurrency_limit=1,
                api_visibility="undocumented",
                show_progress="hidden",
                show_progress_on=[],
            )

            def switch_upscale_view(scale, target, inpainting, editing):
                enabled = scale != "Off"
                upload = enabled and target == "Input image only"
                return [
                    gr.update(visible=True if enabled else "hidden"),
                    gr.update(visible=editing and not upload and not inpainting),
                    gr.update(visible=True if editing and inpainting and not upload else "hidden"),
                    gr.update(visible=editing and upload),
                    gr.update(value=t("Upscale") if upload else t("Generate")),
                    gr.update(value="Generated image")
                    if not enabled and target != "Generated image"
                    else gr.update(),
                ]

            image_view_inputs = [
                automatic_upscale,
                automatic_upscale_target,
                inpaint_toggle,
                edit_mode,
            ]
            gr.on(
                triggers=[component.change for component in image_view_inputs],
                fn=switch_upscale_view,
                inputs=image_view_inputs,
                outputs=[
                    upscale_options,
                    main_view,
                    inpaint_view,
                    upscale_upload,
                    run_button,
                    automatic_upscale_target,
                ],
                trigger_mode="always_last",
                concurrency_id="image_preview_views",
                concurrency_limit=1,
                api_visibility="undocumented",
                show_progress="hidden",
                show_progress_on=[],
            )
            with create_controls_tabs, gr.Tab(label=t("Details")) as details_tab:
                with gr.Row():
                    metadata_json.render()
                with gr.Row():

                    def details_links(locale=None):
                        return """
                        <nav aria-label="{nav}">
                          <ul id="details-links">
                            <li><a href="https://discord.gg/CvpAFya9Rr"><img src="gradio_api/file=ruinedfooocus/ui/assets/icon_clyde_white_RGB.svg" width="16" height="16" alt="">Discord</a></li>
                            <li><a href="https://github.com/runew0lf/RuinedFooocus"><img src="gradio_api/file=ruinedfooocus/ui/assets/github-mark-white.svg" width="16" height="16" alt="">GitHub</a></li>
                          </ul>
                        </nav>
                        """.format(
                            nav=localize("Community and viewer links", locale),
                        )

                    links = gr.HTML(value=details_links())
                    localized_output(lambda locale: details_links(locale), [], [links])

            try:
                with open("update_log.md", "r", encoding="utf-8") as update_log:
                    changelog_text = update_log.read()
            except:
                changelog_text = ""
            with create_controls_tabs, details_tab:
                with gr.Accordion(t("Changelog"), open=False, elem_id="changelog-panel"):
                    hint_text = gr.Markdown(
                        value=changelog_text,
                        elem_id="hint-container",
                        elem_classes="hint-container",
                    )

            def performance_control_updates(selection, family, checkpoint):
                custom = selection == performance_settings.CUSTOM_PERFORMANCE
                options = performance_settings.control_options(selection, checkpoint, family)
                updates = []
                for key in ("cfg", "sampler_name", "scheduler", "clip_skip", "custom_steps"):
                    update = {"visible": custom}
                    if selection != performance_settings.CUSTOM_PERFORMANCE and not options.get(
                        "_recipe_unavailable"
                    ):
                        update["value"] = options[key]
                    updates.append(gr.update(**update))
                return updates

            def refresh_recommendation(
                selection, family, checkpoint, automatic, request: gr.Request = None, force=False
            ):
                locale = resolve_locale(request=request)
                store = performance_settings.model_recommendations
                if force or (automatic and selection == performance_settings.MODEL_PERFORMANCE):
                    store.refresh(checkpoint, KSampler.SAMPLERS, KSampler.SCHEDULERS, force)
                from ruinedfooocus.features.models.model_recommendations import FAMILY_PRESETS

                options = performance_settings.control_options(
                    performance_settings.MODEL_PERFORMANCE, checkpoint, family
                )
                if options.get("_recipe_unavailable"):
                    return options["_recipe_unavailable"]
                return store.describe(
                    checkpoint,
                    options,
                    KSampler.SAMPLERS,
                    KSampler.SCHEDULERS,
                    family_default=FAMILY_PRESETS.get(family, family)
                    in performance_settings.performance_options,
                    locale=locale,
                )

            recommendation_inputs = [
                performance_selection,
                model_family,
                base_model,
                auto_recommendations,
            ]
            recommendation_use.click(
                lambda: performance_settings.MODEL_PERFORMANCE,
                outputs=[performance_selection],
                api_visibility="undocumented",
            )

            def force_recommendation(
                selection, family, checkpoint, automatic, request: gr.Request = None
            ):
                return refresh_recommendation(
                    selection, family, checkpoint, automatic, request, force=True
                )

            recommendation_refresh.click(
                force_recommendation,
                inputs=recommendation_inputs,
                outputs=[recommendation_status],
                api_visibility="undocumented",
            ).then(
                performance_control_updates,
                inputs=[performance_selection, model_family, base_model],
                outputs=performance_outputs,
                api_visibility="undocumented",
            ).then(
                model_resolution,
                inputs=[model_family, base_model, aspect_ratios_selection, resolution_auto],
                outputs=[aspect_ratios_selection, resolution_auto],
                api_visibility="undocumented",
                show_progress="hidden",
                show_progress_on=[],
            ).then(
                resolution_recommendation_text,
                inputs=resolution_description_inputs,
                outputs=resolution_recommendation,
                api_visibility="undocumented",
                show_progress="hidden",
                show_progress_on=[],
            )

            def save_checkpoint_override(
                checkpoint, family, steps, guidance, sampler, schedule, skip
            ):
                performance_settings.model_recommendations.save_override(
                    checkpoint,
                    dict(
                        custom_steps=steps,
                        cfg=guidance,
                        sampler_name=sampler,
                        scheduler=schedule,
                        clip_skip=skip,
                    ),
                    KSampler.SAMPLERS,
                    KSampler.SCHEDULERS,
                )
                return performance_settings.MODEL_PERFORMANCE

            recommendation_save.click(
                save_checkpoint_override,
                inputs=[
                    base_model,
                    model_family,
                    custom_steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    clip_skip,
                ],
                outputs=[performance_selection],
                api_visibility="undocumented",
            )

            def reset_checkpoint_override(checkpoint, family):
                performance_settings.model_recommendations.reset_override(checkpoint)
                return performance_settings.MODEL_PERFORMANCE

            recommendation_reset.click(
                reset_checkpoint_override,
                inputs=[base_model, model_family],
                outputs=[performance_selection],
                api_visibility="undocumented",
            ).then(
                performance_control_updates,
                inputs=[performance_selection, model_family, base_model],
                outputs=performance_outputs,
                api_visibility="undocumented",
            )
            performance_selection.change(
                refresh_recommendation,
                inputs=recommendation_inputs,
                outputs=[recommendation_status],
                api_visibility="undocumented",
            ).then(
                performance_control_updates,
                inputs=[performance_selection, model_family, base_model],
                outputs=performance_outputs,
                api_visibility="undocumented",
            )
            # Resolve sizes once, after recommendations can update cached dimensions.
            model_controls_event.then(
                refresh_recommendation,
                inputs=recommendation_inputs,
                outputs=[recommendation_status],
                api_visibility="undocumented",
            ).then(
                performance_control_updates,
                inputs=[performance_selection, model_family, base_model],
                outputs=performance_outputs,
                api_visibility="undocumented",
            ).then(
                model_resolution,
                inputs=[model_family, base_model, aspect_ratios_selection, resolution_auto],
                outputs=[aspect_ratios_selection, resolution_auto],
                api_visibility="undocumented",
                trigger_mode="always_last",
                show_progress="hidden",
                show_progress_on=[],
            ).then(
                resolution_recommendation_text,
                inputs=resolution_description_inputs,
                outputs=resolution_recommendation,
                api_visibility="undocumented",
                show_progress="hidden",
                show_progress_on=[],
            )
            block.load(
                refresh_recommendation,
                inputs=recommendation_inputs,
                outputs=[recommendation_status],
                api_visibility="undocumented",
            ).then(
                performance_control_updates,
                inputs=[performance_selection, model_family, base_model],
                outputs=performance_outputs,
                api_visibility="undocumented",
            ).then(
                model_resolution,
                inputs=[model_family, base_model, aspect_ratios_selection, resolution_auto],
                outputs=[aspect_ratios_selection, resolution_auto],
                api_visibility="undocumented",
                trigger_mode="always_last",
                show_progress="hidden",
                show_progress_on=[],
            ).then(
                resolution_recommendation_text,
                inputs=resolution_description_inputs,
                outputs=resolution_recommendation,
                api_visibility="undocumented",
                show_progress="hidden",
                show_progress_on=[],
            )

            workflow_inputs = [
                model_family,
                base_model,
                performance_selection,
                custom_steps,
                cfg,
                sampler_name,
                scheduler,
            ]

            def describe_workflow(
                family,
                checkpoint,
                selection,
                steps,
                cfg,
                sampler,
                scheduler,
                request: gr.Request = None,
            ):
                return performance_settings.describe_workflow(
                    family,
                    checkpoint,
                    selection,
                    steps,
                    cfg,
                    sampler,
                    scheduler,
                    locale=resolve_locale(request=request),
                )

            localized_output(
                lambda locale, *values: performance_settings.describe_workflow(
                    *values, locale=locale
                ),
                workflow_inputs,
                [workflow_help],
            )
            localized_output(
                lambda locale, name: performance_settings.describe(name, locale=locale),
                [performance_selection],
                [performance_help],
            )
            localized_output(
                lambda locale, selection, family, checkpoint, automatic: refresh_recommendation(
                    selection,
                    family,
                    checkpoint,
                    False,
                    SimpleNamespace(cookies={LOCALE_COOKIE: locale}),
                ),
                recommendation_inputs,
                [recommendation_status],
            )
            for component in workflow_inputs:
                if component is not model_family:
                    component.change(
                        describe_workflow,
                        inputs=workflow_inputs,
                        outputs=[workflow_help],
                        api_visibility="undocumented",
                    )
            model_controls_event.then(
                describe_workflow,
                inputs=workflow_inputs,
                outputs=[workflow_help],
                api_visibility="undocumented",
            )

            @aspect_ratios_selection.change(
                api_visibility="undocumented",
                inputs=[aspect_ratios_selection],
                outputs=[
                    ratio_name,
                    custom_width,
                    custom_height,
                    ratio_save,
                    ratio_delete,
                    resolution_auto,
                    target_shortcuts,
                ],
                show_progress="hidden",
                show_progress_on=[],
            )
            def aspect_ratios_changed(selection):
                # Show resolution controls when selecting Custom
                if selection == resolution_settings.CUSTOM_RESOLUTION:
                    return [gr.update(visible=True)] * 4 + [
                        gr.update(visible=False),
                        False,
                        gr.update(visible=True),
                    ]

                # Hide resolution controls and update with selected resolution
                selected_width, selected_height = resolution_settings.get_aspect_ratios(selection)
                return {
                    ratio_name: gr.update(visible="hidden"),
                    custom_width: gr.update(visible="hidden", value=selected_width),
                    custom_height: gr.update(visible="hidden", value=selected_height),
                    ratio_save: gr.update(visible="hidden"),
                    ratio_delete: gr.update(visible=resolution_settings.is_custom(selection)),
                    target_shortcuts: gr.update(visible=False),
                }

        run_event.change(
            fn=refresh_seed,
            api_visibility="undocumented",
            inputs=[seed_random, image_seed],
            outputs=image_seed,
        ).then(
            fn=generate_clicked,
            api_visibility="undocumented",
            show_progress="minimal",
            show_progress_on=progress_html,
            inputs=[*state["ctrls_obj"], editor_workspace.version],
            js=editor_workspace.run_version_js(state["ctrls_name"].index("_editor_destination")),
            outputs=[
                run_button,
                stop_button,
                progress_html,
                main_view,
                inpaint_view,
                inpaint_toggle,
                gallery,
                metadata_json,
                hint_text,
                generation_task,
                generation_originals,
                generation_artifacts,
                native_main_source,
                shared.get_ctrl("image_source"),
                shared.get_ctrl("input_image"),
                editor_workspace.context,
                editor_workspace.source_input,
                editor_workspace.notice,
                editor_workspace.mask,
                editor_workspace.latest_generated,
                *editor_workspace.reference_outputs_from(shared.get_ctrl),
            ],
        )

        def poke(number):
            return number + 1

        run_button.click(
            fn=poke, api_visibility="undocumented", inputs=run_event, outputs=run_event
        )

        stop_button.click(
            fn=stop_clicked, inputs=[generation_task], api_visibility="undocumented", queue=False
        )

        modetoggle_btn = gr.Button(
            value=translate_ui("edit mode"), elem_id="edit_mode", visible="hidden"
        )

        def toggle_edit_mode(editing):
            return not editing

        modetoggle_btn.click(
            api_visibility="undocumented",
            fn=toggle_edit_mode,
            inputs=[edit_mode],
            outputs=[edit_mode],
        ).then(
            lambda: [gr.update(visible="hidden"), gr.update(visible="hidden")],
            outputs=[progress_html, gallery],
            api_visibility="undocumented",
        )

        def update_cfg(family, checkpoint, current_resolution, automatic):
            # Update ui components
            # Only refresh things like minimum, maximum and choices. Assume the user already
            # have options selected and don't overwrite them. (They should restart if they want that)
            resolution_settings.set_profile(shared.settings.name)
            resolution_update, automatic = model_resolution(
                family, checkpoint, current_resolution, automatic
            )
            return {
                image_number: gr.update(maximum=settings.get("image_number_max", 50)),
                performance_selection: gr.update(
                    choices=performance_settings.choices_for_model(family, checkpoint)
                ),
                aspect_ratios_selection: resolution_update,
                resolution_auto: automatic,
                cfg_timestamp: gr.update(value=shared.state["last_config"]),
            }

        # If cfg_timestamp has a new value, trigger an update
        cfg_timestamp.change(
            fn=update_cfg,
            inputs=[model_family, base_model, aspect_ratios_selection, resolution_auto],
            api_visibility="undocumented",
            outputs=[
                cfg_timestamp,
                image_number,
                performance_selection,
                aspect_ratios_selection,
                resolution_auto,
            ],
            show_progress="hidden",
        )

        # Preset functions
        def preset_select(preset_gallery, evt: gr.SelectData):
            path = evt.value["image"]["path"]
            preset = Path(path).with_suffix("").name
            show_perf = gr.update(visible=False)
            show_size = (
                gr.update(visible="hidden")
                if (0b001 & settings.get("preset_mode", 7))
                else gr.update()
            )
            return {
                preset_image: gr.update(value=path),
                preset_selection: gr.update(value=path),
                preset_accordion: gr.update(label=t(PNG_PRESET_LABEL) + ": " + preset),
                performance_selection: show_perf,
                performance_help: show_perf,
                cfg: show_perf,
                sampler_name: show_perf,
                scheduler: show_perf,
                clip_skip: show_perf,
                custom_steps: show_perf,
                aspect_ratios_selection: show_size,
                resolution_auto: show_size,
                ratio_name: show_size,
                custom_width: show_size,
                custom_height: show_size,
                ratio_save: show_size,
                target_shortcuts: show_size,
                model_tab: gr.update(visible=True),
            }

        def preset_unselect(performance_selection_val, aspect_ratios_selection_val):
            show_perf = performance_selection_val == performance_settings.CUSTOM_PERFORMANCE
            show_size = (
                True
                if (aspect_ratios_selection_val == resolution_settings.CUSTOM_RESOLUTION)
                else "hidden"
            )

            return {
                preset_selection: gr.update(value=""),
                preset_accordion: gr.update(label=t(PNG_PRESET_LABEL)),
                performance_selection: gr.update(visible=True),
                performance_help: gr.update(visible=True),
                cfg: gr.update(visible=show_perf),
                sampler_name: gr.update(visible=show_perf),
                scheduler: gr.update(visible=show_perf),
                clip_skip: gr.update(visible=show_perf),
                custom_steps: gr.update(visible=show_perf),
                aspect_ratios_selection: gr.update(visible=True),
                resolution_auto: gr.update(visible=True),
                ratio_name: gr.update(visible=show_size),
                custom_width: gr.update(visible=show_size),
                custom_height: gr.update(visible=show_size),
                ratio_save: gr.update(visible=show_size),
                target_shortcuts: gr.update(visible=show_size),
                model_tab: gr.update(visible=True),
            }

        def preset_image_upload(preset_image):
            path = preset_image
            preset = Path(path).with_suffix("").name
            return {
                preset_selection: gr.update(value=path),
                preset_accordion: gr.update(label=t(PNG_PRESET_LABEL) + ": " + preset),
                performance_selection: gr.update(visible="hidden"),
                performance_help: gr.update(visible="hidden"),
                cfg: gr.update(visible="hidden"),
                sampler_name: gr.update(visible="hidden"),
                scheduler: gr.update(visible="hidden"),
                clip_skip: gr.update(visible="hidden"),
                custom_steps: gr.update(visible="hidden"),
                aspect_ratios_selection: gr.update(visible="hidden"),
                resolution_auto: gr.update(visible="hidden"),
                ratio_name: gr.update(visible="hidden"),
                custom_width: gr.update(visible="hidden"),
                custom_height: gr.update(visible="hidden"),
                ratio_save: gr.update(visible="hidden"),
                target_shortcuts: gr.update(visible="hidden"),
                model_tab: gr.update(visible=True),
            }

        preset_reset_outputs = [
            preset_selection,
            preset_accordion,
            performance_selection,
            performance_help,
            cfg,
            sampler_name,
            scheduler,
            clip_skip,
            custom_steps,
            aspect_ratios_selection,
            resolution_auto,
            ratio_name,
            custom_width,
            custom_height,
            ratio_save,
            target_shortcuts,
            model_tab,
        ]
        gr.on(
            triggers=[preset_image.clear, clear_preset_btn.click],
            fn=preset_unselect,
            api_visibility="undocumented",
            inputs=[
                performance_selection,
                aspect_ratios_selection,
            ],
            outputs=preset_reset_outputs,
        )
        preset_image.upload(
            fn=preset_image_upload,
            api_visibility="undocumented",
            inputs=[
                preset_image,
            ],
            outputs=preset_reset_outputs,
        )
        preset_gallery.select(
            fn=preset_select,
            api_visibility="undocumented",
            inputs=[preset_gallery],
            outputs=[preset_image, *preset_reset_outputs],
        )

    from ruinedfooocus.features.generation.defaults import (
        DEFAULT_CONTROLS,
        SAMPLING_KEYS,
        generation_defaults,
    )

    default_keys = (
        set(DEFAULT_CONTROLS.values()) | set(SAMPLING_KEYS) | {"loras", "preset_selection"}
    )
    default_controls = {
        key: component
        for key, component in zip(state["ctrls_name"], state["ctrls_obj"])
        if key in default_keys
    }

    def save_current_defaults(*values):
        try:
            defaults = generation_defaults(dict(zip(default_controls, values)))
        except ValueError as error:
            raise gr.Error(str(error)) from error
        shared.settings.default_settings.update(defaults)
        shared.settings.save_settings()
        gr.Info("Saved generation defaults to this profile for the next launch.")

    save_defaults_btn.click(
        save_current_defaults, inputs=list(default_controls.values()), api_visibility="undocumented"
    )
    add_api()

if args.api:
    from ruinedfooocus.integrations.openai_api import install, validate_options

    api_key = validate_options(args)
launch_app(args)
add_fastapi()
if args.api:
    install(shared.server_app, api_key)
    print(
        f"API enabled: {shared.local_url}v1/docs | Image tool: {shared.local_url}tools/openapi.json"
    )

# Wait...
shared.models.start_inbox_sorting()
while True:
    time.sleep(100)  # Waiting to exit
