from ruinedfooocus.core.translation_manager import translate_ui, resolve_locale
from ruinedfooocus.features.chat.assistants import (
    assistant_choices,
    assistant_info,
    assistant_identity,
    assistant_edit_snapshot,
    save_assistant,
    delete_assistant,
)
from ruinedfooocus.core.paths import ASSET_ROOT, asset_path
import gradio as gr
from ruinedfooocus.app.state import path_manager, settings
import ruinedfooocus.app.async_worker as worker
from pathlib import Path
import re
from functools import wraps
from ruinedfooocus.core.task_errors import describe_error
from ruinedfooocus.features.chat.llama_models import (
    DEFAULT_MODEL,
    ModelCatalogueCache,
    model_catalogue,
    model_label,
    preferred_quant,
    saved_selection,
    selection_choices,
    import_model,
    local_models,
    remove_local_model,
)
from ruinedfooocus.features.chat.llama_file_picker import select_gguf
from ruinedfooocus.features.chat.ui_reasoning_settings import (
    SAVED_PREFERENCE,
    conversation_reasoning,
)
from ruinedfooocus.features.chat.model_preparation import (
    ChatModelSession,
    ModelTransfer,
    capture_model,
    validate_message,
    runtime_selection_key,
    initialized_runtime_status,
)
from ruinedfooocus.features.chat.ui_load_state import (
    load_operation,
    progress_text,
    initialized_text,
    same_initialized_instance,
)
from ruinedfooocus.features.chat.attachment_ui import AttachmentControls, source_summary
import time
from concurrent.futures import CancelledError


def display_history(history, show_reasoning):
    """Keep full history in session state; thoughts are an optional display detail."""
    messages = []
    for message in history:
        if message["role"] == "assistant" and isinstance(message["content"], str):
            # Close partial streamed thoughts so Gradio keeps them collapsed, too.
            replacement = (lambda m: f"<{m[1]}>{m[2]}</{m[1]}>") if show_reasoning else ""
            text = re.sub(
                r"<(think|thinking)>(.*?)(?:</\1>|$)",
                replacement,
                message["content"],
                flags=re.DOTALL,
            ).strip()
            if not text:
                continue
            message = dict(message, content=text)
        messages.append(message)
    return messages


def reasoning_display_available(capability, history):
    """Output visibility is independent of whether computation can be configured."""
    return bool((capability or {}).get("control")) or display_history(
        history or [], True
    ) != display_history(history or [], False)


def assistant_avatar(info):
    """Materialize the portrait for Gradio without exposing its saved directory."""
    from PIL import Image

    if info.get("avatar") is None:
        return None
    with Image.open(info["avatar"]) as image:
        return image.copy()


def chat_model_catalogue():
    """Keep UI browsing snapshots separate from live preparation/resolution."""
    return ModelCatalogueCache(model_catalogue)


def create_chat(image_controls=None, *, editor_workspace=None, prompt_control=None):
    image_controls = image_controls or {}
    from itertools import count
    from ruinedfooocus.app.resource_lifecycle import latest_ui_requests, ui_request_owner

    image_versions = count(1)
    model_catalogue = chat_model_catalogue()
    from ruinedfooocus.features.chat.image_tool_requests import ChatPermissionSession

    def gr_llama_get_assistants(selected):
        choices = assistant_choices()
        if selected not in {value for _, value in choices}:
            selected = "__normal__"
        return gr.update(choices=choices, value=selected)

    def llama_select_assistant(dropdown):
        info = assistant_info(dropdown)
        return {
            llama_chat: gr.update(value=info["chatstart"]),
            llama_history: info["chatstart"],
            llama_msg: gr.update(value="", placeholder=info.get("input_hint", "Message")),
            llama_avatar: gr.update(
                value=assistant_avatar(info),
                label=info["name"],
                visible=info["avatar"] is not None,
            ),
            llama_system: gr.update(value=info["system"]),
            llama_embed: gr.update(value=info["embed"]),
        }

    selected_file = settings.default_settings.get("llama_localfile") or DEFAULT_MODEL
    custom_file = Path(selected_file).is_file() and Path(selected_file).resolve() not in {
        Path(path).resolve() for path in local_models().values()
    }
    default_model, default_quant = saved_selection(model_catalogue(), selected_file)
    saved_local = any(
        value == default_quant
        for _, value in model_catalogue("Local models").get(default_model, [])
    )
    default_source = "Local models" if custom_file or saved_local else "Downloadable"
    catalogue = model_catalogue(default_source)
    default_choices, default_quant, default_available = selection_choices(
        catalogue, default_model, default_quant
    )
    if default_model not in catalogue:
        default_model = None
    from ruinedfooocus.features.models.model_capabilities import (
        configured_runtime,
        selection_eligibility,
    )

    default_decision = selection_eligibility(
        default_quant,
        path_manager.DOWNLOADABLE_FILES,
        configured_runtime(settings.default_settings.get("llm_runtime", "llama.cpp")),
    )
    html_dir = ASSET_ROOT
    with gr.Blocks() as app_llama_chat:
        with gr.Row(elem_id="chat-layout"):
            with gr.Column(scale=3, min_width=0, elem_id="chat-conversation"), gr.Group():
                default_bot = assistant_identity(
                    settings.default_settings.get("llama_assistant", "__normal__")
                )
                if default_bot not in {value for _, value in assistant_choices()}:
                    default_bot = "__normal__"
                gr.HTML(
                    "<h1>" + str(translate_ui("Chat")) + "</h1>",
                    elem_id="chat-conversation-heading",
                    elem_classes=["rf-page-head"],
                )
                gr.HTML(
                    f"<p>{translate_ui('Talk to a language model running on this PC.')}</p>",
                    elem_classes=["rf-page-head"],
                )
                setup_model = gr.Button(
                    translate_ui("Choose / load model"),
                    min_width=0,
                    size="sm",
                    elem_id="chat-setup-model",
                )
                setup_model.click(
                    fn=None,
                    outputs=[],
                    js=asset_path("chat_setup.js").read_text(encoding="utf-8"),
                    queue=False,
                    api_visibility="undocumented",
                    show_progress="hidden",
                )
                llama_history = gr.State(assistant_info(default_bot)["chatstart"])
                llama_chat = gr.Chatbot(
                    label="",
                    show_label=False,
                    height="var(--rf-chat-height)",
                    resizable=True,
                    elem_id="chat-messages",
                    reasoning_tags=[("<think>", "</think>"), ("<thinking>", "</thinking>")],
                    buttons=["copy_all"],
                    sanitize_html=True,
                    value=assistant_info(default_bot)["chatstart"],
                )
                llama_msg = gr.Textbox(
                    label=translate_ui("Message"),
                    show_label=False,
                    placeholder=translate_ui("Message"),
                    elem_id="chat-message-input",
                    lines=2,
                    max_lines=8,
                    interactive=False,
                )
                with gr.Row(
                    elem_id="chat-message-actions", elem_classes=["rf-actions"], equal_height=True
                ):
                    send_message = gr.Button(
                        translate_ui("Send"),
                        variant="primary",
                        min_width=112,
                        interactive=False,
                    )
                    clear_chat = gr.Button(translate_ui("New conversation"), min_width=112)
                attachment_controls = AttachmentControls()
                image_intent = gr.Radio(
                    label=translate_ui("Image generation"),
                    choices=[
                        (translate_ui("Off"), "text"),
                        (translate_ui("On"), "run"),
                    ],
                    value="text",
                    info=translate_ui("Allow one image action with the current Create settings."),
                    elem_classes=["rf-chat-permission"],
                )
                rewrite_image_prompt = gr.Radio(
                    label=translate_ui("Rewrite image prompts"),
                    choices=[(translate_ui("Off"), False), (translate_ui("On"), True)],
                    value=False,
                    elem_classes=["rf-chat-permission"],
                )
                update_create_prompt = gr.Radio(
                    label=translate_ui("Update Create prompt"),
                    choices=[(translate_ui("Off"), False), (translate_ui("On"), True)],
                    value=False,
                    elem_classes=["rf-chat-permission"],
                    visible=prompt_control is not None,
                )
                supplied_image_prompt = gr.Textbox(
                    label=translate_ui("Image prompt to preserve"),
                    placeholder=translate_ui("Optional existing visual prompt"),
                    info=translate_ui(
                        "Leave empty for a new image request. Off preserves this prompt exactly."
                    ),
                    lines=2,
                    max_length=12000,
                    visible=False,
                )
                permission_session = gr.State(ChatPermissionSession())
                prompt_action = None
                prompt_action_group = gr.Group() if prompt_control is not None else None
                image_tool_result = gr.State(None)
                image_tool_status = gr.Markdown("", elem_id="chat-image-tool-status")
                open_image_result = gr.Button(
                    translate_ui("Open / Edit image"), interactive=False, size="sm"
                )
                editor_context_input = (
                    editor_workspace.context if editor_workspace is not None else gr.State(None)
                )
                if editor_workspace is not None:
                    editor_workspace.bind_chat_open(open_image_result, image_tool_result)
                image_channel = f"chat-image-{image_tool_result._id}"
                with gr.Accordion(
                    translate_ui("Image reference"),
                    open=False,
                    visible=False,
                    elem_id="chat-image-reference-group",
                ) as image_reference_group:
                    image_source = gr.Dropdown(
                        label=translate_ui("Image source"),
                        value="New image",
                        choices=[
                            (translate_ui("New image"), "New image"),
                            (translate_ui("Previous generated image"), "Previous generated image"),
                            (translate_ui("Uploaded image"), "Uploaded image"),
                        ],
                        info=translate_ui(
                            "Applies to this message. New image ignores retained sources."
                        ),
                    )
                    image_reference = gr.Image(
                        label=translate_ui("Chat image reference"),
                        type="pil",
                        image_mode="RGBA",
                        format="png",
                        visible=False,
                        height=128,
                        sources=["upload", "clipboard"],
                        elem_id="chat-image-reference",
                    )
                    edit_mode = gr.Markdown("", visible=False)
                    image_denoise = gr.Slider(
                        label=translate_ui("Image-to-image denoise"),
                        minimum=0.01,
                        maximum=1.0,
                        value=0.25,
                        step=0.01,
                        visible=False,
                        info=translate_ui(
                            "Lower values preserve the source; higher values redraw more."
                        ),
                    )
                    preview_image = gr.Button(translate_ui("Check image readiness"), size="sm")
                    checkpoint_control = image_controls.get("base_model_name")
                    format_control = image_controls.get("moss_image_edit_format")

                    def reference_controls(
                        intent,
                        vision,
                        active,
                        mode,
                        reference,
                        previous,
                        checkpoint=None,
                        moss_format=None,
                    ):
                        # Native edit models edit the reference directly; others use denoise.
                        from ruinedfooocus.features.image_editing.ui import checkpoint_features

                        native = (
                            bool(checkpoint)
                            and checkpoint_features(checkpoint, moss_format)["native_edit"]
                        )
                        relevant = intent != "text" or (vision and bool(active))
                        using = relevant and mode != "New image"
                        from ruinedfooocus.features.chat.llama_vision import reference_capabilities

                        capability = reference_capabilities(
                            image_generation=intent != "text",
                            vision_supported=bool(vision and active),
                            vision_initialized=bool(vision and model_has_vision(active)),
                            source=mode,
                            source_available=(
                                reference is not None
                                if mode == "Uploaded image"
                                else bool(previous and previous.get("status") == "completed")
                            ),
                            native_edit=native,
                        )
                        note = (
                            "Native edit: this checkpoint edits the reference directly."
                            if native
                            else "Image to image: denoise controls how much the reference changes."
                        )
                        return (
                            gr.update(visible=relevant),
                            gr.update(visible=relevant and mode == "Uploaded image"),
                            gr.update(visible=capability["denoise_visible"]),
                            gr.update(
                                visible=using,
                                value=capability["reason"] or note
                                if intent != "text"
                                else capability["reason"],
                            ),
                            gr.update(visible=intent != "text"),
                            gr.update(visible=intent != "text"),
                        )

            gr.HTML(
                '<div role="separator" tabindex="0" aria-label="Resize chat" '
                'aria-orientation="vertical" aria-valuemin="35" aria-valuemax="85" '
                'aria-valuenow="72"></div>',
                elem_id="chat-divider",
                min_width=12,
                scale=0,
                js_on_load=(ASSET_ROOT / "chat_splitter.js").read_text(encoding="utf-8"),
            )
            with gr.Column(scale=2, min_width=0, elem_id="chat-controls"):
                llama_avatar = gr.Image(
                    buttons=["download", "fullscreen"],
                    value=assistant_avatar(assistant_info(default_bot)),
                    label=assistant_info(default_bot)["name"],
                    height="clamp(96px, 18dvh, 160px)",
                    elem_id="chat-avatar",
                    show_label=True,
                    visible=assistant_info(default_bot)["avatar"] is not None,
                )
                llama_assistants = gr.Dropdown(
                    choices=assistant_choices(),
                    value=default_bot,
                    label=translate_ui("Assistant"),
                    show_label=True,
                    interactive=True,
                    min_width=0,
                )
                with gr.Row(
                    elem_id="chat-assistant-actions",
                    elem_classes=["rf-actions"],
                    equal_height=True,
                ):
                    assistant_new = gr.Button(translate_ui("New"), size="sm", min_width=80)
                    assistant_edit = gr.Button(translate_ui("Edit"), size="sm", min_width=80)
                    assistant_delete = gr.Button(
                        translate_ui("Delete"),
                        size="sm",
                        min_width=80,
                        interactive=default_bot != "__normal__",
                    )
                    llama_reload = gr.Button(translate_ui("Refresh"), size="sm", min_width=80)
                assistant_draft = gr.State(None)
                assistant_version = gr.State(None)
                with gr.Group(visible=False) as assistant_editor:
                    assistant_name = gr.Textbox(label=translate_ui("Assistant name"), max_lines=1)
                    assistant_prompt = gr.Textbox(label=translate_ui("System prompt"), lines=6)
                    assistant_image = gr.Image(
                        label=translate_ui("Assistant image (optional)"),
                        type="pil",
                        sources=["upload"],
                        buttons=[],
                        height=160,
                    )
                    with gr.Row(elem_classes=["rf-actions"], equal_height=True):
                        assistant_save = gr.Button(
                            translate_ui("Save"), variant="primary", size="sm", min_width=96
                        )
                        assistant_save_as = gr.Button(
                            translate_ui("Save as new"), size="sm", min_width=96
                        )
                        assistant_cancel = gr.Button(
                            translate_ui("Cancel"), size="sm", min_width=96
                        )
                with gr.Row(elem_classes=["rf-actions"], equal_height=True):
                    assistant_startup = gr.Button(
                        translate_ui("Use at startup"),
                        size="sm",
                        variant="secondary",
                        min_width=96,
                    )
                assistant_status = gr.Markdown("")
                pending_assistant_delete = gr.State(None)
                assistant_delete_dialog = gr.HTML(
                    "",
                    html_template=(ASSET_ROOT / "assistant_remove.html").read_text(
                        encoding="utf-8"
                    ),
                    js_on_load=(ASSET_ROOT / "assistant_remove.js").read_text(encoding="utf-8"),
                    css_template="""dialog { max-width: min(480px, 90vw); margin: auto; padding: 24px;
                      border-radius: 8px; background: var(--block-background-fill); color: var(--body-text-color);
                      border: 1px solid var(--border-color-primary); } dialog::backdrop { background: #0009; }
                      .assistant-name { overflow-wrap: anywhere; } .actions { display: flex; gap: 8px; margin-top: 16px; }
                      button { padding: 8px 12px; min-height: 44px; border-radius: 6px; cursor: pointer;
                        background: var(--button-secondary-background-fill); color: var(--body-text-color); }
                      button[data-action="delete"] { background: #b91c1c; color: white; }""",
                )
                llama_active_model = gr.State(None)
                model_session = gr.State(ChatModelSession())
                chat_reasoning_capability = gr.State(None)
                from ruinedfooocus.app.state import add_ctrl

                add_ctrl("chat_reasoning_capability", chat_reasoning_capability)
                llama_selection = gr.State(default_model)
                llama_selected_quant = gr.State(default_quant)

                def share_chat_model(model):
                    # Prompt rewrite and enhancement use the model loaded here for this session;
                    # Settings > Custom GGUF path stays the saved startup default.
                    if model:
                        settings.default_settings["llama_localfile"] = model

                llama_active_model.change(
                    share_chat_model,
                    inputs=llama_active_model,
                    queue=False,
                    api_visibility="undocumented",
                )
                with gr.Accordion(
                    translate_ui("Model library"), open=True, elem_id="chat-model-library"
                ) as model_library:
                    llama_source = gr.Radio(
                        [
                            (translate_ui("Installed"), "Local models"),
                            (translate_ui("Download"), "Downloadable"),
                        ],
                        value=default_source,
                        label=translate_ui("Model library"),
                        show_label=False,
                        elem_id="chat-model-source",
                    )
                    model_query = gr.Textbox(
                        label=translate_ui("Find a model or quantization"),
                        placeholder=translate_ui("Model name, quantization or filename"),
                    )
                    llama_model = gr.Dropdown(
                        label=translate_ui("Model"),
                        filterable=False,
                        choices=[(model_label(name), name) for name in catalogue],
                        value=default_model,
                        interactive=True,
                        visible=default_source == "Downloadable",
                    )
                    local_picker = gr.HTML(
                        {
                            "choices": [
                                (model_label(name), name)
                                for name in model_catalogue("Local models")
                            ],
                            "selected": default_model,
                        },
                        visible=default_source == "Local models",
                        elem_id="chat-local-models",
                        html_template=(html_dir / "chat_local_models.html").read_text(
                            encoding="utf-8"
                        ),
                        js_on_load=(html_dir / "chat_local_models.js").read_text(encoding="utf-8"),
                        css_template="""
                          label { display: block; margin: 0 0 8px; font-size: var(--block-label-text-size); }
                          details { margin: 0; padding: 0; }
                          summary { cursor: pointer; list-style: none; padding: 6px 12px; min-height: 44px;
                            margin: 0; box-sizing: border-box; line-height: 28px; overflow-wrap: anywhere; border-radius: var(--input-radius);
                            background: var(--input-background-fill); border: 1px solid var(--border-color-primary); }
                          summary::after { content: '▾'; float: right; }
                          .model-options { max-height: 280px; overflow-y: auto; margin-top: 4px;
                            border: 1px solid var(--border-color-primary); border-radius: var(--input-radius);
                            background: var(--input-background-fill); box-shadow: 0 4px 12px #0003; }
                          .model-options > div { display: flex; align-items: stretch; margin: 0;
                            border-bottom: 1px solid var(--border-color-primary); }
                          .model-options > div:last-child { border-bottom: 0; }
                          .model-options > div:has([aria-pressed="true"]) {
                            background: color-mix(in srgb, var(--color-accent) 20%, var(--input-background-fill)); }
                          .model-options > div:hover { background: var(--button-secondary-background-fill-hover); }
                          button { background: transparent; color: var(--body-text-color);
                            cursor: pointer; border: 0; padding: 9px; font: inherit; margin: 0; border-radius: 0; }
                          button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: -2px; }
                          .choose-model[aria-pressed="true"] { color: var(--color-accent); font-weight: 600; }
                          .choose-model { flex: 1; min-width: 0; text-align: left; overflow-wrap: anywhere; }
                          .remove-model { flex: 0 0 44px; min-height: 44px; font-size: 20px; }
                        """,
                    )
                    llama_quant = gr.Dropdown(
                        label=translate_ui("Quantization"),
                        interactive=len(default_choices) > 1,
                        visible=bool(default_quant),
                        choices=default_choices,
                        value=default_quant,
                    )
                    from ruinedfooocus.features.chat.model_browser import model_action_state

                    def model_roots():
                        roots = {}
                        for key in ("path_llm", "path_vision"):
                            folders = path_manager.paths.get(key, [])
                            if isinstance(folders, str):
                                folders = [folders]
                            roots[key] = [path_manager.get_abspath(folder) for folder in folders]
                        return roots

                    with gr.Row(
                        elem_id="chat-model-actions",
                        elem_classes=["rf-actions"],
                        equal_height=True,
                    ):
                        initial_actions = model_action_state(
                            default_quant, path_manager.DOWNLOADABLE_FILES, roots=model_roots()
                        )
                        llama_use = gr.Button(
                            translate_ui("Load"),
                            size="sm",
                            min_width=112,
                            interactive=default_available
                            and default_decision.allowed
                            and initial_actions["load_available"],
                        )
                        llama_import = gr.Button(
                            translate_ui("Import GGUF"), size="sm", min_width=112
                        )
                        llama_refresh = gr.Button(
                            translate_ui("Reindex models"),
                            size="sm",
                            min_width=112,
                            elem_id="chat-refresh-models",
                            visible=False,
                        )
                        download_candidate = gr.Button(
                            translate_ui("Download"),
                            size="sm",
                            min_width=112,
                            visible=initial_actions["download_visible"],
                        )
                        cancel_download = gr.Button(
                            translate_ui("Cancel download"),
                            size="sm",
                            min_width=112,
                            visible=False,
                        )
                        unload_active = gr.Button(
                            translate_ui("Unload"), size="sm", min_width=112, visible=False
                        )
                        cancel_load = gr.Button(
                            translate_ui("Cancel loading"), size="sm", min_width=112, visible=False
                        )
                    llama_model_status = gr.Markdown(
                        (default_decision.reason if not default_decision.allowed else "")
                        if default_available
                        else "Select an installed model or a downloadable candidate."
                    )
                    llama_download = gr.HTML(visible=False)
                    preparation_status = gr.Markdown("")
                active_status = gr.State("No active Chat model.")
                sampling_status = gr.State("No active Chat model sampling state.")
                add_ctrl("chat_selected_quant", llama_selected_quant)
                add_ctrl("chat_runtime_status", active_status)
                add_ctrl("chat_sampling_status", sampling_status)
                llama_active_model.change(
                    fn=lambda loaded: gr.update(interactive=bool(loaded)),
                    inputs=llama_active_model,
                    outputs=send_message,
                    queue=False,
                    show_progress="hidden",
                    api_visibility="undocumented",
                )
                show_reasoning = gr.Checkbox(
                    label=translate_ui("Show reasoning"),
                    value=False,
                    visible=False,
                    info=translate_ui(
                        "Display only. Hiding reasoning does not disable model computation."
                    ),
                )
                reasoning_mode = gr.Dropdown(
                    label=translate_ui("Reasoning computation"),
                    choices=[
                        (translate_ui("Use saved preference"), SAVED_PREFERENCE),
                        "Auto",
                        "Off",
                        "On",
                    ],
                    value=SAVED_PREFERENCE,
                    info=translate_ui(
                        "Saved preference follows your profile. Auto uses the model's template default."
                    ),
                    visible=False,
                    interactive=False,
                )
                reasoning_effort = gr.Dropdown(
                    label=translate_ui("Reasoning effort"),
                    choices=[
                        (translate_ui("Use saved preference"), SAVED_PREFERENCE),
                        "Auto",
                        "Low",
                        "Medium",
                        "High",
                    ],
                    value=SAVED_PREFERENCE,
                    visible=False,
                    interactive=False,
                )
                reasoning_status = gr.Markdown(
                    translate_ui("Load a model to inspect its reasoning controls."), visible=False
                )

                def reasoning_controls(capability, mode, effort):
                    capability = capability or {
                        "modes": ("Auto",),
                        "efforts": (),
                        "control": None,
                        "reason": "Load a model to inspect its reasoning controls.",
                    }
                    modes = (SAVED_PREFERENCE, *capability["modes"])
                    efforts = (SAVED_PREFERENCE, "Auto", *capability["efforts"])
                    selected = mode if mode in modes else SAVED_PREFERENCE

                    def choices(values):
                        return [
                            (translate_ui("Use saved preference"), value)
                            if value == SAVED_PREFERENCE
                            else value
                            for value in values
                        ]

                    return (
                        gr.update(
                            choices=choices(modes),
                            value=selected,
                            visible=bool(capability["control"]) and "Off" in capability["modes"],
                            interactive=bool(capability["control"])
                            and "Off" in capability["modes"],
                        ),
                        gr.update(
                            choices=choices(efforts),
                            value=effort if effort in efforts else SAVED_PREFERENCE,
                            visible=bool(capability["efforts"]),
                            interactive=selected != "Off",
                        ),
                        gr.update(value=capability["reason"], visible=bool(capability["control"])),
                    )

                gr.on(
                    triggers=[chat_reasoning_capability.change, reasoning_mode.input],
                    fn=reasoning_controls,
                    inputs=[chat_reasoning_capability, reasoning_mode, reasoning_effort],
                    outputs=[reasoning_mode, reasoning_effort, reasoning_status],
                    queue=False,
                    api_visibility="undocumented",
                    show_progress="hidden",
                )
                gr.on(
                    triggers=[chat_reasoning_capability.change, llama_history.change],
                    fn=lambda capability, history: gr.update(
                        visible=reasoning_display_available(capability, history)
                    ),
                    inputs=[chat_reasoning_capability, llama_history],
                    outputs=show_reasoning,
                    queue=False,
                    api_visibility="undocumented",
                    show_progress="hidden",
                )
                from ruinedfooocus.features.chat.llama_vision import (
                    model_has_vision,
                    vision_models,
                    IMAGE_DETAILS,
                )

                enable_vision = gr.Checkbox(
                    label=translate_ui("Model vision"), value=False, visible=False
                )
                llama_active_model.change(
                    lambda active: gr.update(visible=bool(active)),
                    inputs=llama_active_model,
                    outputs=enable_vision,
                    queue=False,
                    api_visibility="undocumented",
                )
                vision_include_system = gr.Checkbox(
                    label=translate_ui("Include system prompt"),
                    value=False,
                    visible=False,
                    info=translate_ui("Use the assistant's instructions when reviewing images."),
                )
                vision_detail = gr.Dropdown(
                    label=translate_ui("Vision image detail"),
                    choices=list(IMAGE_DETAILS),
                    value="standard",
                    visible=False,
                    info=translate_ui("High detail uses more memory. The model may resize images."),
                )
                gr.on(
                    triggers=[enable_vision.change, image_source.change],
                    fn=lambda enabled, mode: gr.update(visible=enabled and mode != "New image"),
                    inputs=[enable_vision, image_source],
                    outputs=vision_detail,
                    queue=False,
                    api_visibility="undocumented",
                )
                vision_model = gr.Dropdown(
                    label=translate_ui("Vision model"),
                    visible=False,
                    choices=[
                        (model_label(Path(name).stem), name) for name in vision_models(path_manager)
                    ],
                    value=None,
                    info=translate_ui("Used to inspect images; downloaded when needed."),
                )

                def vision_controls(enabled, model, source):
                    return gr.update(
                        visible=enabled
                        and bool(model)
                        and source != "New image"
                        and not model_has_vision(model)
                    )

                enable_vision.change(
                    vision_controls,
                    inputs=[enable_vision, llama_active_model, image_source],
                    outputs=vision_model,
                    api_visibility="undocumented",
                )
                enable_vision.change(
                    lambda enabled, source: gr.update(visible=enabled and source != "New image"),
                    inputs=[enable_vision, image_source],
                    outputs=vision_include_system,
                    api_visibility="undocumented",
                )
                llama_active_model.change(
                    vision_controls,
                    inputs=[enable_vision, llama_active_model, image_source],
                    outputs=vision_model,
                    api_visibility="undocumented",
                )
                image_source.change(
                    vision_controls,
                    inputs=[enable_vision, llama_active_model, image_source],
                    outputs=vision_model,
                    queue=False,
                    api_visibility="undocumented",
                )
                reference_inputs = [
                    image_intent,
                    enable_vision,
                    llama_active_model,
                    image_source,
                    image_reference,
                    image_tool_result,
                ]
                if checkpoint_control is not None:
                    reference_inputs += [checkpoint_control] + (
                        [format_control] if format_control is not None else []
                    )
                gr.on(
                    triggers=[control.change for control in reference_inputs],
                    fn=reference_controls,
                    inputs=reference_inputs,
                    outputs=[
                        image_reference_group,
                        image_reference,
                        image_denoise,
                        edit_mode,
                        supplied_image_prompt,
                        preview_image,
                    ],
                    show_progress_on=[
                        image_reference,
                        image_denoise,
                        edit_mode,
                        supplied_image_prompt,
                        preview_image,
                    ],
                    queue=False,
                    api_visibility="undocumented",
                )
                pending_remove = gr.State(None)
                remove_dialog = gr.HTML(
                    "",
                    elem_id="chat-model-confirm",
                    html_template=(html_dir / "chat_model_remove.html").read_text(encoding="utf-8"),
                    js_on_load=(html_dir / "chat_model_remove.js").read_text(encoding="utf-8"),
                    css_template="""
                      dialog { max-width: min(560px, 90vw); max-height: 85vh; overflow: auto;
                        margin: auto; padding: 24px; border-radius: 8px;
                        background: var(--block-background-fill); color: var(--body-text-color);
                        border: 1px solid var(--border-color-primary); }
                      dialog::backdrop { background: #0009; }
                      .model-path { overflow-wrap: anywhere; font-family: monospace; }
                      select { max-width: 100%; padding: 8px; background: var(--input-background-fill);
                        color: var(--body-text-color); border: 1px solid var(--border-color-primary); }
                      .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
                      button { padding: 8px 12px; border-radius: 6px; cursor: pointer;
                        background: var(--button-secondary-background-fill); color: var(--body-text-color); }
                      button[data-action="delete"] { background: #b91c1c; color: white; }
                    """,
                )
                llama_system = gr.Textbox(
                    visible="hidden",
                    value=assistant_info(default_bot)["system"],
                )
                llama_embed = gr.Textbox(
                    visible="hidden",
                    value=assistant_info(default_bot)["embed"],
                )

        def llama_get_text(message, force, active_model, selected_model):
            return validate_message(message, active_model), bool(force)

        def error_controls(message):
            gr.Warning(message)
            return {
                llama_model_status: message,
                cancel_load: gr.update(visible=False),
                llama_download: gr.update(visible=False),
            }

        def recover_controls(fn):
            @wraps(fn)
            def guarded(*args, **kwargs):
                try:
                    yield from fn(*args, **kwargs)
                except Exception as error:
                    updates = error_controls(describe_error(error))
                    session = next(
                        (value for value in args if isinstance(value, ChatModelSession)), None
                    )
                    if session is not None:
                        updates.update(action_updates(session.selected, session))
                    yield updates

            return guarded

        @recover_controls
        def llama_respond(
            message,
            system,
            embed,
            chat_history,
            model,
            show_thinking,
            vision_enabled,
            review_model,
            include_system,
            force,
            source_mode,
            source_image,
            source_denoise,
            allow_images,
            detail,
            *image_values,
            reasoning=None,
            attachments=None,
            image_intent="text",
            editor_context=None,
            ui_turn=None,
            request=None,
            permissions=None,
            draft_guard=None,
        ):
            chat_history = list(chat_history)
            chat_history.append({"role": "user", "content": message})

            gen_data = {
                "task_type": "llama",
                "force_image": force,
                "enable_image_tools": allow_images,
                "image_tool_intent": image_intent,
                "image_source": source_mode,
                "image_reference": source_image if source_mode == "Uploaded image" else None,
                "image_denoise": source_denoise,
                "vision_enabled": vision_enabled,
                "vision_include_system": include_system,
                "image_detail": detail,
                "vision_model": review_model,
                "model": model,
                "system": system,
                "embed": embed,
                "history": chat_history,
                "image_settings": dict(zip(image_controls, image_values)),
            }
            if permissions is not None:
                gen_data.update(
                    _chat_tool_permissions=permissions,
                    enable_image_generation=permissions.enable_image_generation,
                    rewrite_image_prompt=permissions.rewrite_image_prompt,
                    update_create_prompt=permissions.update_create_prompt,
                )
            if editor_context is not None:
                gen_data["editor_context"] = editor_context
            if attachments is not None:
                gen_data["_chat_attachments"] = attachments
            if reasoning is not None:
                from ruinedfooocus.features.chat.model_profiles import validate_reasoning_request

                gen_data["reasoning"] = validate_reasoning_request(reasoning)

            # Add work
            if ui_turn is not None and (
                not latest_ui_requests.current(ui_turn)
                or ui_request_owner(request) != ui_turn.owner
            ):
                yield {
                    image_tool_status: translate_ui(
                        "Image request cancelled. Send the current selection again."
                    )
                }
                return
            try:
                task_id = worker.add_task(gen_data.copy())
                if ui_turn is not None:
                    ui_turn.source = task_id
                    if (
                        not latest_ui_requests.current(ui_turn)
                        or ui_request_owner(request) != ui_turn.owner
                    ):
                        worker.interrupt_processing(task_id)
                        yield {
                            image_tool_status: translate_ui(
                                "Image request cancelled. Send the current selection again."
                            )
                        }
                        return
            except Exception as error:
                if attachments is None:
                    raise
                if attachments.same_conversation():
                    yield {
                        **error_controls(describe_error(error)),
                        llama_msg: gr.update(value=message),
                    }
                return
            if attachments is not None or ui_turn is not None:
                yield {attachment_controls.stop_reply: gr.update(visible=True)}

            # Wait for result
            finished = False
            sources = None
            stale_image = False
            while not finished:
                flag, product = worker.task_result(task_id)
                if ui_turn is not None and (
                    not latest_ui_requests.current(ui_turn)
                    or ui_request_owner(request) != ui_turn.owner
                ):
                    if flag == "results":
                        yield {
                            image_tool_status: translate_ui(
                                "Image request cancelled or replaced. Send the current selection again."
                            )
                        }
                        return
                    continue
                if attachments is not None and not attachments.current() and flag != "results":
                    continue
                if flag == "preview":
                    yield {
                        llama_chat: display_history(product, show_thinking),
                        llama_history: product,
                        llama_model_status: "Thinking…",
                    }
                elif flag == "download":
                    yield {
                        llama_model_status: "Preparing this message's dependencies…",
                        **(
                            {
                                image_tool_status: translate_ui(
                                    "Preparing the selected image resources…"
                                )
                            }
                            if ui_turn is not None
                            else {}
                        ),
                    }
                elif flag == "reasoning":
                    yield {reasoning_status: product["status"]}
                elif flag == "sampling":
                    from ruinedfooocus.features.chat.ui_sampling import active_sampling_status

                    yield {sampling_status: active_sampling_status(product)}
                elif flag == "attachment_sources":
                    sources = product
                elif flag == "chat_prompt_proposal" and prompt_action is not None:
                    try:
                        ticket = prompt_action.stage(product, permissions, draft_guard)
                    except (ValueError, PermissionError) as error:
                        yield {image_tool_status: str(error)[:4000]}
                    else:
                        yield {prompt_action.ticket: ticket}
                elif flag == "image_tool_result" and ui_turn is not None:
                    from ruinedfooocus.features.chat.chat_tools import public_image_tool_result

                    terminal = public_image_tool_result(product)
                    if prompt_action is not None and terminal.get("prompt_proposal"):
                        try:
                            ticket = prompt_action.stage(terminal, permissions, draft_guard)
                        except (ValueError, PermissionError) as error:
                            yield {image_tool_status: str(error)[:4000]}
                        else:
                            yield {prompt_action.ticket: ticket}
                    status = terminal.get("status")
                    can_open = False
                    if (
                        status == "completed"
                        and terminal.get("editor")
                        and editor_workspace is not None
                    ):
                        try:
                            editor_workspace.result_context(terminal, request=request)
                            can_open = True
                        except (ValueError, PermissionError, KeyError):
                            stale_image = True
                            terminal = dict(
                                status="stale",
                                error={
                                    "message": "This image result was replaced. Open the current image history."
                                },
                            )
                    error = terminal.get("error") or {}
                    notice = error.get("message", "") if isinstance(error, dict) else str(error)
                    if not notice:
                        notice = (
                            "Preview only. No image was run."
                            if status == "preview"
                            else "Image action completed."
                            if status == "completed"
                            else "Image action cancelled."
                        )
                    readiness = terminal.get("readiness") or {}
                    if isinstance(readiness, dict):
                        missing = readiness.get("missing") or readiness.get("requirements") or []
                        if isinstance(missing, list) and missing:
                            names = [
                                str(
                                    row.get("name")
                                    or row.get("role")
                                    or row.get("key")
                                    or "Required resource"
                                )
                                if isinstance(row, dict)
                                else str(row)
                                for row in missing[:12]
                            ]
                            notice += "\n\nRequired resources: " + ", ".join(names)
                    if (
                        not latest_ui_requests.current(ui_turn)
                        or ui_request_owner(request) != ui_turn.owner
                    ):
                        continue
                    yield {
                        image_tool_result: terminal,
                        image_tool_status: notice[:4000],
                        open_image_result: gr.update(interactive=can_open),
                    }
                elif flag == "results":
                    finished = True

            if stale_image:
                return
            if attachments is not None and not attachments.same_conversation():
                return
            if attachments is not None and not attachments.current():
                yield {
                    **error_controls(translate_ui("Attachments changed. Send the message again.")),
                    llama_chat: display_history(chat_history[:-1], show_thinking),
                    llama_history: chat_history[:-1],
                    llama_msg: gr.update(value=message),
                }
                return
            if isinstance(product, dict) and product.get("error"):
                updates = {
                    **error_controls(product["error"]),
                    llama_chat: display_history(chat_history, show_thinking),
                    llama_history: chat_history,
                }
                if attachments is not None:
                    updates[llama_msg] = gr.update(value=message)
                    updates[llama_history] = chat_history[:-1]
                    updates[llama_chat] = display_history(chat_history[:-1], show_thinking)
                if "active_model" in product:
                    updates[llama_active_model] = product["active_model"]
                    updates[llama_msg] = gr.update(interactive=bool(product["active_model"]))
                    updates[sampling_status] = (
                        "Reload to inspect active sampling."
                        if product["active_model"]
                        else "No active Chat model sampling state."
                    )
                yield updates
                return
            if ui_turn is not None and (
                not latest_ui_requests.current(ui_turn)
                or ui_request_owner(request) != ui_turn.owner
            ):
                return
            chat_history.append({"role": "assistant", "content": product})
            yield {
                llama_chat: display_history(chat_history, show_thinking),
                llama_history: chat_history,
                llama_model_status: "",
                **(
                    {attachment_controls.sources: source_summary(sources)}
                    if attachments is not None and sources is not None
                    else {}
                ),
            }

        def action_updates(value, session=None, decision=None):
            state = model_action_state(
                value, path_manager.DOWNLOADABLE_FILES, session, roots=model_roots()
            )
            eligible = (
                bool(value)
                and (
                    decision
                    or selection_eligibility(
                        value,
                        path_manager.DOWNLOADABLE_FILES,
                        configured_runtime(
                            settings.default_settings.get("llm_runtime", "llama.cpp")
                        ),
                    )
                ).allowed
            )
            initialized = initialized_runtime_status()["runtime_snapshot"]
            operation = getattr(session, "load_operation", None)
            loaded = False
            if initialized and value:
                from ruinedfooocus.features.chat.model_browser import catalogue_local_file

                selected = (
                    Path(value)
                    if Path(value).is_absolute()
                    else catalogue_local_file(value, path_manager.DOWNLOADABLE_FILES, model_roots())
                )
                loaded = (
                    selected is not None
                    and session is not None
                    and bool(session.active)
                    and same_initialized_instance(
                        getattr(session, "initialized_runtime", None), initialized
                    )
                    and Path(session.active).resolve() == selected.resolve()
                    and initialized.get("selection_key")
                    == runtime_selection_key(selected, settings.default_settings)
                )
            loading = operation is not None and operation.current_selection()
            stopping = loading and operation.cancelled.is_set()
            return {
                download_candidate: gr.update(visible=state["download_visible"]),
                cancel_download: gr.update(visible=state["cancel_visible"]),
                cancel_load: gr.update(visible=loading and not stopping),
                llama_use: gr.update(
                    value="Stopping…"
                    if stopping
                    else "Loading…"
                    if loading
                    else "Loaded"
                    if loaded
                    else "Load",
                    interactive=state["load_available"] and eligible and not (loading or loaded),
                ),
            }

        def clear_model_selection(session=None):
            if session is not None:
                session.select(None)
            return {
                llama_selection: None,
                llama_selected_quant: None,
                llama_quant: gr.update(choices=[], value=None, visible=False),
                llama_model_status: "",
                **action_updates(None, session),
            }

        def download_update(progress, model=None, session=None):
            received, total, speed = progress
            if received is None:
                return {
                    **action_updates(model, session),
                    llama_download: gr.update(visible=False),
                    preparation_status: "Verifying model…",
                }
            amount = f"{received / 1000000:.1f} MB"
            value = ""
            if total:
                percent = min(100, received * 100 / total)
                value = f'value="{percent:.1f}"'
                amount = f"{percent:.0f}%"
            return {
                **action_updates(model, session),
                preparation_status: f"Downloading: {amount} · Speed: {speed / 1000000:.1f} MB/s",
                llama_download: gr.update(
                    visible=True,
                    value=f'<progress max="100" {value} style="width:100%;accent-color:var(--color-accent)"></progress>',
                ),
            }

        def select_model(
            model, source, selected_quant=None, session=None, *, preserve=False, catalogue=None
        ):
            from ruinedfooocus.features.models.model_capabilities import (
                configured_runtime,
                selection_eligibility,
            )

            # Dismissing/clearing a dropdown can emit an empty input. Only an
            # explicit valid choice changes the canonical session selection.
            if not model:
                return gr.skip()
            choices, value, available = selection_choices(
                model_catalogue(source) if catalogue is None else catalogue,
                model,
                selected_quant,
                preserve=preserve or selected_quant is not None,
            )
            decision = (
                selection_eligibility(
                    value,
                    path_manager.DOWNLOADABLE_FILES,
                    configured_runtime(settings.default_settings.get("llm_runtime", "llama.cpp")),
                )
                if value
                else None
            )
            message = (
                (decision.reason if not decision.allowed else "")
                if decision
                else "Choose a model or import a local GGUF."
            )
            if session is not None:
                session.select(value)
            actions = action_updates(value, session, decision)
            return {
                **actions,
                llama_selection: model,
                llama_selected_quant: value,
                llama_quant: gr.update(
                    choices=choices, value=value, visible=bool(value), interactive=len(choices) > 1
                ),
                llama_model_status: message,
            }

        def select_source(source, model=None, selected_quant=None, session=None, *, catalogue=None):
            catalogue = model_catalogue(source) if catalogue is None else catalogue
            choices = [(model_label(name), name) for name in catalogue]
            if model not in catalogue:
                model = None
            return {
                llama_model: gr.update(
                    choices=choices, value=model, visible=source == "Downloadable"
                ),
                local_picker: gr.update(
                    value={"choices": choices, "selected": model}, visible=source == "Local models"
                ),
                **(
                    select_model(
                        model, source, selected_quant, session, preserve=True, catalogue=catalogue
                    )
                    if model
                    else clear_model_selection(session)
                ),
            }

        def import_gguf(session):
            try:
                path = select_gguf()
                if not path:
                    return {llama_model_status: gr.skip()}
                model, quant = import_model(path)
            except (OSError, ValueError, RuntimeError) as error:
                return {llama_model_status: str(error)}
            return {
                llama_source: "Local models",
                **select_source("Local models", model, quant, session),
            }

        def settle_superseded_load(session, operation, revision):
            current = initialized_runtime_status()
            previous = session.active
            # A late native completion is resident evidence, never permission to
            # activate a candidate whose load this browser already cancelled.
            accepted = getattr(session, "initialized_runtime", None)
            unchanged = current["active_model"] == previous and same_initialized_instance(
                accepted, current["runtime_snapshot"]
            )
            session.active = previous if unchanged else None
            session.initialized_runtime = current["runtime_snapshot"] if session.active else None
            with session.lock:
                if getattr(session, "load_operation", None) is operation:
                    session.load_operation = None
            return {
                llama_active_model: session.active,
                llama_msg: gr.update(interactive=bool(session.active)),
                **({chat_reasoning_capability: None} if session.active != previous else {}),
                **action_updates(session.selected, session),
                **(
                    {
                        preparation_status: "Loading cancelled. You can retry.",
                        llama_download: gr.update(visible=False),
                    }
                    if operation.cancelled.is_set() and session.current(revision)
                    else {}
                ),
            }

        @recover_controls
        def use_model(model, selection, source, action, session, request: gr.Request = None):
            if action == "Unload":
                yield {
                    llama_model_status: "Unloading model…",
                    llama_use: gr.update(interactive=False),
                }
                task_id = worker.add_task({"task_type": "llama", "unload": True})
                while True:
                    flag, result = worker.task_result(task_id)
                    if flag == "results":
                        break
                if isinstance(result, dict) and result.get("unloaded"):
                    session.active = None
                    session.initialized_runtime = None
                    yield {
                        llama_active_model: None,
                        chat_reasoning_capability: None,
                        active_status: "No active Chat model.",
                        sampling_status: "No active Chat model sampling state.",
                        llama_model_status: "Model unloaded.",
                        llama_msg: gr.update(interactive=False),
                        llama_use: gr.update(value="Load", interactive=True),
                    }
                else:
                    message = (
                        result.get("error", str(result))
                        if isinstance(result, dict)
                        else str(result)
                    )
                    yield {**error_controls(message), **action_updates(session.selected, session)}
                return
            if not model:
                yield {llama_model_status: "Choose a model and quantization first."}
                return
            if Path(model).is_absolute() and not Path(model).is_file():
                yield {
                    llama_model_status: "The model file is missing. Import it again from its current location."
                }
                return
            if model not in [value for _, value in model_catalogue(source).get(selection, [])]:
                yield {llama_model_status: "Choose an available quantization, then click Load."}
                return
            prepared = capture_model(
                model, settings.default_settings, path_manager, local=local_models()
            )
            if not prepared.present:
                yield {llama_model_status: "Download this candidate first, then click Load."}
                return
            revision = session.select(model)
            expected_key = runtime_selection_key(prepared.model, prepared.settings)
            with load_operation(
                session,
                lambda: (
                    session.current(revision)
                    and runtime_selection_key(prepared.model, settings.default_settings)
                    == expected_key
                ),
            ) as operation:
                yield {
                    llama_model_status: "Verifying candidate…",
                    llama_use: gr.update(value="Loading…", interactive=False),
                    cancel_load: gr.update(visible=True),
                }
                # Verification is I/O work. A failure here cannot unload active B.
                ModelTransfer(prepared).result()
                if not operation.current():
                    yield settle_superseded_load(session, operation, revision)
                    return
                yield {llama_model_status: "Loading model…"}
                if not operation.current():
                    yield settle_superseded_load(session, operation, revision)
                    return
                task_id = worker.add_task(
                    {
                        "task_type": "llama",
                        "model": str(prepared.model),
                        "load_only": True,
                        "_chat_preparation": prepared,
                        "_selection_current": lambda: operation.current(),
                        "_runtime_load_id": operation.load_id,
                    }
                )
                operation.task_id = task_id
                while True:
                    flag, result = worker.task_result(task_id)
                    if (
                        flag == "runtime_progress"
                        and operation.current()
                        and result.get("load_id") == operation.load_id
                    ):
                        message, progress = progress_text(
                            result, locale=resolve_locale(request=request)
                        )
                        yield {
                            preparation_status: message,
                            llama_download: gr.update(visible=True, value=progress),
                        }
                    elif flag == "download" and operation.current():
                        yield {llama_model_status: "Loading model…"}
                    if flag == "results":
                        break
                if not operation.current():
                    yield settle_superseded_load(session, operation, revision)
                    return
                if isinstance(result, dict) and result.get("skipped"):
                    return
                if isinstance(result, dict) and result.get("ready"):
                    from ruinedfooocus.features.chat.ui_sampling import (
                        active_sampling_status,
                        active_runtime_status,
                    )

                    session.active = str(prepared.model)
                    session.initialized_runtime = result.get("runtime_snapshot")
                    yield {
                        llama_active_model: str(prepared.model),
                        chat_reasoning_capability: result.get("reasoning_capability"),
                        active_status: active_runtime_status(prepared.model.name, result),
                        sampling_status: active_sampling_status(result.get("sampling_state")),
                        llama_model_status: initialized_text(
                            session.initialized_runtime, locale=resolve_locale(request=request)
                        )
                        if session.initialized_runtime
                        else f"Ready: {Path(model).name}",
                        preparation_status: "",
                        llama_msg: gr.update(interactive=True),
                        llama_use: gr.update(
                            value="Loaded" if session.initialized_runtime else "Load",
                            interactive=not bool(session.initialized_runtime),
                        ),
                    }
                else:
                    message = (
                        result.get("error", str(result))
                        if isinstance(result, dict)
                        else str(result)
                    )
                    updates = error_controls(message)
                    if not isinstance(result, dict) or "active_model" not in result:
                        result = initialized_runtime_status()
                    if isinstance(result, dict) and "active_model" in result:
                        session.initialized_runtime = result.get("runtime_snapshot")
                        session.active = result["active_model"]
                        updates.update(
                            {
                                llama_active_model: session.active,
                                chat_reasoning_capability: None,
                                sampling_status: "Reload to inspect active sampling."
                                if session.active
                                else "No active Chat model sampling state.",
                                active_status: f"Active: {Path(session.active).name}"
                                if session.active
                                else "No active Chat model.",
                                llama_msg: gr.update(interactive=bool(session.active)),
                            }
                        )
                    yield updates
            yield {
                **action_updates(session.selected, session),
                llama_download: gr.update(visible=False),
            }

        def download_model(model, session):
            if not model:
                yield {preparation_status: "Choose a candidate first."}
                return
            try:
                revision = session.select(model)
                request = capture_model(
                    model, settings.default_settings, path_manager, local=local_models()
                )
                revision, transfer = session.start(request)
                while not transfer.done():
                    if session.current(revision):
                        yield download_update(transfer.progress, model, session)
                    time.sleep(0.1)
                transfer.result()
                if session.current(revision):
                    yield {
                        **action_updates(model, session),
                        preparation_status: f"Downloaded: {request.model.name}. Click Load to use it.",
                        llama_download: gr.update(visible=False),
                    }
            except CancelledError:
                if session.current(revision):
                    yield {
                        **action_updates(model, session),
                        preparation_status: "Download cancelled.",
                        llama_download: gr.update(visible=False),
                    }
            except Exception as error:
                if session.current(revision):
                    yield {
                        **action_updates(model, session),
                        preparation_status: describe_error(error),
                        llama_download: gr.update(visible=False),
                    }
            finally:
                from ruinedfooocus.features.chat.llama_models import invalidate_model_catalogue

                invalidate_model_catalogue()
            # Terminal jobs refresh actions for today's selection, even when
            # their old candidate no longer owns status/message updates.
            yield action_updates(session.selected, session)

        model_outputs = [
            llama_selection,
            llama_selected_quant,
            llama_model,
            local_picker,
            llama_quant,
            llama_model_status,
            llama_msg,
            llama_use,
            download_candidate,
            cancel_download,
            cancel_load,
        ]

        def search_models(source, model, quant, query, session=None):
            if session is not None:
                with session.lock:
                    version = getattr(session, "model_search_version", 0) + 1
                    session.model_search_version = version
                    selection_revision = session.revision
            terms = (query or "").casefold().split()
            inventory = model_catalogue(source)
            catalogue = {
                name: choices
                for name, choices in inventory.items()
                if all(
                    term
                    in (
                        model_label(name)
                        + " "
                        + name
                        + " "
                        + " ".join(label + " " + value for label, value in choices)
                    ).casefold()
                    for term in terms
                )
            }
            choices = [(model_label(name), name) for name in catalogue]
            if model and (
                model not in inventory or quant not in [value for _, value in inventory[model]]
            ):
                model = None

            def publish():
                if session is not None and (
                    session.model_search_version != version
                    or session.revision != selection_revision
                ):
                    return gr.skip()
                return {
                    **(clear_model_selection(session) if model is None else {}),
                    llama_model: gr.update(
                        choices=choices,
                        value=model if model in catalogue else None,
                        visible=source == "Downloadable",
                    ),
                    local_picker: gr.update(
                        value={
                            "choices": choices,
                            "selected": model,
                            "selected_label": model_label(model) if model else None,
                        },
                        visible=source == "Local models",
                    ),
                }

            if session is not None:
                with session.lock:
                    return publish()
            return publish()

        def refresh_model_browser(source, model, quant, session):
            if session is not None:
                with session.lock:
                    selection_revision = session.revision
                    version = getattr(session, "model_search_version", 0) + 1
                    session.model_search_version = version
            model_catalogue.invalidate()
            catalogue = model_catalogue(source)

            def publish():
                if session is not None and (
                    session.revision != selection_revision
                    or session.model_search_version != version
                ):
                    return gr.skip()
                return select_source(source, model, quant, session, catalogue=catalogue)

            if session is not None:
                with session.lock:
                    return publish()
            return publish()

        search_inputs = [
            llama_source,
            llama_selection,
            llama_selected_quant,
            model_query,
            model_session,
        ]
        search_outputs = list(model_outputs)
        llama_refresh.click(
            refresh_model_browser,
            inputs=[llama_source, llama_selection, llama_selected_quant, model_session],
            outputs=model_outputs,
            queue=False,
            api_visibility="undocumented",
        ).then(
            search_models,
            inputs=search_inputs,
            outputs=search_outputs,
            api_visibility="undocumented",
        )
        model_library.expand(
            refresh_model_browser,
            inputs=[llama_source, llama_selection, llama_selected_quant, model_session],
            outputs=model_outputs,
            queue=False,
            api_visibility="undocumented",
        ).then(
            search_models,
            inputs=search_inputs,
            outputs=search_outputs,
            api_visibility="undocumented",
        )
        gr.on(
            triggers=[
                model_query.input,
                model_query.change,
                model_query.submit,
                llama_selected_quant.change,
            ],
            fn=search_models,
            inputs=search_inputs,
            outputs=search_outputs,
            trigger_mode="always_last",
            queue=False,
            api_visibility="undocumented",
        )

        def local_model_action(source, current_quant, session, event: gr.EventData):
            data = event._data or {}
            model = data.get("model")
            choices = model_catalogue("Local models").get(model, [])
            if source != "Local models" or not choices:
                return {llama_model_status: "This model is no longer available. Refresh the list."}
            if data.get("action") == "select":
                return select_source(
                    "Local models",
                    model,
                    current_quant
                    if any(value == current_quant for _, value in choices)
                    else preferred_quant(choices),
                    session,
                )
            if data.get("action") != "remove":
                return {llama_model_status: gr.skip()}
            files = local_models()
            entries = [
                (label, str(Path(files.get(value, value)).resolve())) for label, value in choices
            ]
            paths = [path for _, path in entries]
            current_path = (
                str(Path(files.get(current_quant, current_quant)).resolve())
                if current_quant
                else None
            )
            return {
                pending_remove: paths,
                remove_dialog: {
                    "files": entries,
                    "selected": current_path if current_path in paths else paths[0],
                },
            }

        def confirm_remove(paths, selection, quant, session, event: gr.EventData):
            data = event._data or {}
            action, index = data.get("action"), data.get("file_index")
            result = {pending_remove: None, remove_dialog: ""}
            if (
                action not in ("hide", "delete")
                or type(index) is not int
                or not 0 <= index < len(paths or [])
            ):
                return result
            path = paths[index]
            try:
                remove_local_model(path, delete_file=action == "delete")
            except (OSError, ValueError) as error:
                return {**result, llama_model_status: str(error)}
            updates = select_source("Local models", selection, quant, session)
            choices = model_catalogue("Local models").get(selection, [])
            if quant in [value for _, value in choices]:
                updates[llama_quant] = gr.update(
                    choices=choices, value=quant, visible=bool(quant), interactive=len(choices) > 1
                )
                updates[llama_msg] = gr.skip()
                updates[llama_use] = gr.skip()
            return {
                **result,
                **updates,
                llama_model_status: "File deleted."
                if action == "delete"
                else "Removed from list. The file was kept.",
            }

        local_picker.click(
            local_model_action,
            inputs=[llama_source, llama_selected_quant, model_session],
            outputs=[pending_remove, remove_dialog, *model_outputs],
            api_visibility="undocumented",
        )
        remove_dialog.click(
            confirm_remove,
            inputs=[pending_remove, llama_selection, llama_selected_quant, model_session],
            outputs=[pending_remove, remove_dialog, *model_outputs],
            api_visibility="undocumented",
        )
        llama_import.click(
            import_gguf,
            inputs=[model_session],
            outputs=[llama_source, *model_outputs],
            api_visibility="undocumented",
        )
        llama_source.input(
            select_source,
            inputs=[llama_source, llama_selection, llama_selected_quant, model_session],
            outputs=model_outputs,
            queue=False,
            api_visibility="undocumented",
        ).then(
            search_models,
            inputs=search_inputs,
            outputs=search_outputs,
            api_visibility="undocumented",
        )

        def choose_model(model, source, previous_model, quant, session):
            if not model:
                return gr.skip()
            # Gradio also emits the current choice when dismissing its menu.
            # Only choosing a different family may select its preferred quant.
            same_model = model == previous_model
            return select_model(
                model, source, quant if same_model else None, session, preserve=same_model
            )

        llama_model.input(
            choose_model,
            inputs=[
                llama_model,
                llama_source,
                llama_selection,
                llama_selected_quant,
                model_session,
            ],
            outputs=model_outputs,
            queue=False,
            api_visibility="undocumented",
        ).then(
            search_models,
            inputs=search_inputs,
            outputs=search_outputs,
            api_visibility="undocumented",
        )
        llama_quant.input(
            lambda quant, model, source, session: select_model(
                model, source, quant, session, preserve=True
            ),
            inputs=[llama_quant, llama_selection, llama_source, model_session],
            outputs=model_outputs,
            queue=False,
            api_visibility="undocumented",
        )
        llama_use.click(
            use_model,
            inputs=[llama_selected_quant, llama_selection, llama_source, llama_use, model_session],
            outputs=[
                llama_active_model,
                active_status,
                sampling_status,
                chat_reasoning_capability,
                llama_source,
                preparation_status,
                llama_download,
                *model_outputs,
            ],
            api_visibility="undocumented",
            show_progress="hidden",
            concurrency_id="chat-model-load",
        )
        download_candidate.click(
            download_model,
            inputs=[llama_selected_quant, model_session],
            outputs=[
                preparation_status,
                llama_download,
                download_candidate,
                cancel_download,
                llama_use,
                cancel_load,
            ],
            concurrency_id="chat-download",
            show_progress="hidden",
            api_visibility="undocumented",
        ).then(
            refresh_model_browser,
            inputs=[llama_source, llama_selection, llama_selected_quant, model_session],
            outputs=model_outputs,
            api_visibility="undocumented",
        ).then(
            search_models,
            inputs=search_inputs,
            outputs=search_outputs,
            api_visibility="undocumented",
        )

        def cancel_model_download(value, session):
            session.cancel()
            return action_updates(value, session)

        cancel_download.click(
            cancel_model_download,
            inputs=[llama_selected_quant, model_session],
            outputs=[download_candidate, cancel_download, llama_use, cancel_load],
            queue=False,
            api_visibility="undocumented",
        )

        def cancel_model_load(value, session):
            operation = getattr(session, "load_operation", None)
            if operation is not None:
                operation.cancelled.set()
                if operation.task_id is not None:
                    worker.interrupt_processing(operation.task_id)
            return {
                **action_updates(value, session),
                preparation_status: "Stopping model loading…",
                llama_download: gr.update(visible=False),
            }

        cancel_load.click(
            cancel_model_load,
            inputs=[llama_selected_quant, model_session],
            outputs=[
                cancel_load,
                llama_use,
                cancel_download,
                download_candidate,
                preparation_status,
                llama_download,
            ],
            queue=False,
            api_visibility="undocumented",
            show_progress="hidden",
        )

        def unload_model(session):
            yield from use_model(None, None, None, "Unload", session)

        unload_active.click(
            unload_model,
            inputs=[model_session],
            outputs=[
                llama_active_model,
                active_status,
                sampling_status,
                chat_reasoning_capability,
                llama_model_status,
                llama_msg,
                llama_use,
                cancel_load,
                llama_download,
            ],
            concurrency_id="chat-model-load",
            show_progress="hidden",
            api_visibility="undocumented",
        )
        llama_active_model.change(
            lambda active: gr.update(visible=bool(active)),
            inputs=llama_active_model,
            outputs=unload_active,
            queue=False,
            api_visibility="undocumented",
        )

        def submit_message(
            message,
            force,
            selected_model,
            system,
            embed,
            history,
            model,
            show_thinking,
            vision_enabled,
            review_model,
            include_system,
            source_mode="New image",
            source_image=None,
            source_denoise=0.25,
            allow_images=False,
            detail="standard",
            context=None,
            previous_result=None,
            *image_values,
            reasoning=None,
            attachments=None,
            request=None,
            permissions=None,
            draft_guard=None,
        ):
            try:
                intent = (
                    force
                    if force in ("text", "preview", "run")
                    else "run"
                    if force is True
                    else "text"
                )
                message, force = llama_get_text(message, False, model, selected_model)
                if intent == "text" and not vision_enabled:
                    source_mode, source_image, context = "New image", None, None
                if intent != "text":
                    if editor_workspace is None and source_mode != "New image":
                        raise ValueError(
                            "Open the shared image workspace before choosing an image reference."
                        )
                    if editor_workspace is not None:
                        if (
                            source_mode == "Previous generated image"
                            and previous_result is not None
                        ):
                            context = editor_workspace.result_context(
                                previous_result, request=request
                            )
                        context = editor_workspace.image_context(
                            context, source_mode, source_image, request=request
                        )
                else:
                    from ruinedfooocus.features.chat.llama_vision import image_reference_source

                    image_reference_source(source_mode, source_image, history)
                    context = None
                ui_turn = (
                    latest_ui_requests.begin(
                        ui_request_owner(request), image_channel, next(image_versions)
                    )
                    if intent != "text"
                    or (permissions is not None and permissions.update_create_prompt)
                    else None
                )
            except (ValueError, PermissionError, KeyError) as error:
                gr.Warning(str(error))
                yield {llama_msg: gr.skip(), image_tool_status: str(error)[:4000]}
                return
            yield {
                llama_msg: "",
                image_intent: "text",
                image_source: "New image",
                send_message: gr.update(interactive=False),
                clear_chat: gr.update(interactive=attachments is not None),
                **({attachment_controls.sources: ""} if attachments is not None else {}),
                **(
                    {
                        image_tool_result: None,
                        open_image_result: gr.update(interactive=False),
                        image_tool_status: translate_ui("Waiting for an image-tool response…"),
                        attachment_controls.stop_reply: gr.update(visible=True),
                    }
                    if ui_turn is not None
                    else {}
                ),
            }
            yield from llama_respond(
                message,
                system,
                embed,
                history,
                model,
                show_thinking,
                vision_enabled,
                review_model,
                include_system,
                force,
                source_mode,
                source_image,
                source_denoise,
                intent != "text",
                detail,
                *image_values,
                **({"reasoning": reasoning} if reasoning is not None else {}),
                **({"attachments": attachments} if attachments is not None else {}),
                image_intent=intent,
                editor_context=context,
                ui_turn=ui_turn,
                request=request,
                permissions=permissions,
                draft_guard=draft_guard,
            )
            if ui_turn is not None:
                latest_ui_requests.finish(ui_turn)
            yield {
                send_message: gr.update(interactive=True),
                clear_chat: gr.update(interactive=True),
                **(
                    {attachment_controls.stop_reply: gr.update(visible=False)}
                    if attachments is not None or ui_turn is not None
                    else {}
                ),
            }

        def submit_with_reasoning(
            *values, attachments=None, request=None, permissions=None, draft_guard=None
        ):
            # Gradio captures these values for this browser event before queuing;
            # model changes or another client's controls cannot rewrite the turn.
            reasoning = conversation_reasoning(
                values[-3], values[-2], settings.default_settings, values[-1]
            )
            yield from submit_message(
                *values[:-3],
                **({"reasoning": reasoning} if reasoning is not None else {}),
                **({"attachments": attachments} if attachments is not None else {}),
                **({"request": request} if request is not None else {}),
                **(
                    {"permissions": permissions, "draft_guard": draft_guard}
                    if permissions is not None
                    else {}
                ),
            )

        def submit_with_permissions(*values, attachments=None, request=None):
            draft_guard = values[-1]
            arguments = values[:-4]
            (
                *arguments,
                permission_owner,
                rewrite_allowed,
                update_allowed,
                original_prompt,
                assistant_context,
                draft_context,
            ) = arguments
            try:
                policy = permission_owner.capture(
                    ui_request_owner(request),
                    enable_image_generation=arguments[1] in ("preview", "run")
                    or arguments[1] is True,
                    rewrite_image_prompt=bool(rewrite_allowed),
                    update_create_prompt=bool(update_allowed) and prompt_action is not None,
                    prompt_target="supplied"
                    if original_prompt and original_prompt.strip()
                    else "new",
                    original_prompt=original_prompt
                    if original_prompt and original_prompt.strip()
                    else None,
                )
            except ValueError as error:
                yield {image_tool_status: str(error)[:4000]}
                return
            if prompt_action is not None:
                yield {prompt_action.permissions: policy}
            yield from submit_with_reasoning(
                *arguments,
                *values[-4:-1],
                **({"attachments": attachments} if attachments is not None else {}),
                **({"request": request} if request is not None else {}),
                permissions=policy,
                draft_guard=draft_guard,
            )

        def submit_with_attachments(key, *values, request: gr.Request = None):
            from gradio.context import LocalContext

            request = request or LocalContext.request.get(None)
            turn = attachment_controls.session(key).turn() if key and values[1] == "text" else None
            if turn is None or not turn.files:
                yield {attachment_controls.sources: ""}
                yield from submit_with_permissions(*values, request=request)
            else:
                yield from submit_with_permissions(*values, attachments=turn, request=request)

        if prompt_control is not None:
            from ruinedfooocus.features.prompting.prompt_preview import create_chat_prompt_action

            with prompt_action_group:
                prompt_action = create_chat_prompt_action(
                    prompt_control,
                    owner_for_request=ui_request_owner,
                    context_inputs=[
                        image_intent,
                        rewrite_image_prompt,
                        update_create_prompt,
                        llama_assistants,
                        llama_quant,
                        llama_model,
                        llama_source,
                        supplied_image_prompt,
                        image_source,
                    ],
                )

        submission_inputs = [
            attachment_controls.picker,
            llama_msg,
            image_intent,
            llama_selected_quant,
            llama_system,
            llama_embed,
            llama_history,
            llama_active_model,
            show_reasoning,
            enable_vision,
            vision_model,
            vision_include_system,
            image_source,
            image_reference,
            image_denoise,
            gr.State(False),
            vision_detail,
            editor_context_input,
            image_tool_result,
            *image_controls.values(),
            permission_session,
            rewrite_image_prompt,
            update_create_prompt,
            supplied_image_prompt,
            llama_assistants,
            prompt_control if prompt_control is not None else gr.State(None),
            reasoning_mode,
            reasoning_effort,
            chat_reasoning_capability,
        ]
        submission_js = attachment_controls.capture_key_js
        draft_capture = gr.State(None)
        if prompt_action is not None:
            draft_capture, begin = prompt_action.capture(
                submission_inputs, f"chat-prompt-{image_tool_result._id}"
            )
            # Message clearing and the per-turn generation/source reset are
            # automatic. Explicit input revokes the immutable server policy.
            excluded = [llama_msg.elem_id, image_intent.elem_id, image_source.elem_id]
            import json

            submission_js = f"""(...values) => {{
                const captured = ({begin})(...values);
                const guard = captured[captured.length - 1];
                if (guard) guard.contexts = guard.contexts.filter(context => !{json.dumps(excluded)}.includes(context.id));
                captured[0] = document.querySelector('#chat-conversation')?.dataset.attachmentKey || '';
                return captured;
            }}"""
        submission_outputs = [
            llama_chat,
            llama_history,
            llama_model_status,
            llama_download,
            llama_active_model,
            llama_msg,
            llama_use,
            image_intent,
            image_source,
            send_message,
            clear_chat,
            reasoning_status,
            sampling_status,
            chat_reasoning_capability,
            attachment_controls.sources,
            attachment_controls.stop_reply,
            image_tool_result,
            image_tool_status,
            open_image_result,
            cancel_load,
            *(
                [prompt_action.permissions, prompt_action.ticket]
                if prompt_action is not None
                else []
            ),
        ]
        gr.on(
            triggers=[llama_msg.submit, send_message.click],
            fn=submit_with_attachments,
            js=submission_js,
            api_visibility="undocumented",
            inputs=[*submission_inputs, draft_capture],
            outputs=submission_outputs,
            show_progress="hidden",
            concurrency_id="chat-response",
        )

        def preview_with_attachments(key, *values, request: gr.Request = None):
            values = list(values)
            values[1] = "preview"
            yield from submit_with_attachments(key, *values, request=request)

        preview_image.click(
            preview_with_attachments,
            js=submission_js,
            inputs=[*submission_inputs, draft_capture],
            outputs=submission_outputs,
            show_progress="hidden",
            concurrency_id="chat-response",
            api_visibility="undocumented",
        )

        def cancel_image_request(request: gr.Request = None):
            owner = ui_request_owner(request)
            try:
                turn = latest_ui_requests.peek(owner, image_channel)
            except CancelledError:
                return {component: gr.skip() for component in image_result_outputs}
            if turn.finished:
                return {component: gr.skip() for component in image_result_outputs}
            latest_ui_requests.cancel(owner, image_channel)
            if turn.source is not None:
                worker.interrupt_processing(turn.source)
            return {
                image_tool_result: None,
                open_image_result: gr.update(interactive=False),
                image_tool_status: translate_ui(
                    "Image request cancelled. Send the current selection again."
                ),
            }

        def revoke_permissions(session, request: gr.Request = None):
            session.revoke(ui_request_owner(request))
            return cancel_image_request(request)

        invalidate_prompt_js = prompt_action.invalidate_js if prompt_action is not None else None
        gr.on(
            triggers=[
                image_intent.input,
                rewrite_image_prompt.input,
                update_create_prompt.input,
                supplied_image_prompt.input,
            ],
            fn=revoke_permissions,
            inputs=permission_session,
            js=invalidate_prompt_js,
            outputs=[image_tool_result, image_tool_status, open_image_result],
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )

        image_result_outputs = [image_tool_result, image_tool_status, open_image_result]
        attachment_controls.stop_reply.click(
            cancel_image_request,
            outputs=image_result_outputs,
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )
        image_inputs = [image_source, image_reference, image_denoise, *image_controls.values()]
        gr.on(
            triggers=[control.input for control in image_inputs if hasattr(control, "input")],
            fn=revoke_permissions,
            inputs=permission_session,
            js=invalidate_prompt_js,
            outputs=image_result_outputs,
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )
        gr.on(
            triggers=[
                llama_selected_quant.change,
                llama_active_model.change,
                llama_use.click,
                unload_active.click,
            ],
            fn=revoke_permissions,
            inputs=permission_session,
            js=invalidate_prompt_js,
            outputs=image_result_outputs,
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )
        if editor_workspace is not None:
            editor_workspace.context.change(
                revoke_permissions,
                inputs=permission_session,
                js=invalidate_prompt_js,
                outputs=image_result_outputs,
                queue=False,
                api_visibility="private",
                show_progress="hidden",
            )

        def new_conversation(key, bot, request: gr.Request = None):
            cancel_image_request(request)
            history = assistant_info(bot)["chatstart"]
            return (
                history,
                history,
                "",
                "text",
                "New image",
                None,
                "",
                *attachment_controls.clear(key),
                None,
                "",
                gr.update(interactive=False),
            )

        clear_chat.click(
            new_conversation,
            inputs=[attachment_controls.picker, llama_assistants],
            js=attachment_controls.capture_key_js,
            outputs=[
                llama_history,
                llama_chat,
                llama_msg,
                image_intent,
                image_source,
                image_reference,
                supplied_image_prompt,
                attachment_controls.files,
                attachment_controls.sources,
                *image_result_outputs,
            ],
            api_visibility="undocumented",
            queue=False,
        )
        gr.on(
            triggers=[
                clear_chat.click,
                llama_assistants.select,
                attachment_controls.stop_reply.click,
            ],
            fn=revoke_permissions,
            inputs=permission_session,
            js=invalidate_prompt_js,
            outputs=image_result_outputs,
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )

        def select_assistant_and_clear(key, assistant, request: gr.Request = None):
            files, sources = attachment_controls.clear(key)
            return {
                **cancel_image_request(request),
                **llama_select_assistant(assistant),
                attachment_controls.files: files,
                attachment_controls.sources: sources,
                image_tool_result: None,
                image_tool_status: "",
                open_image_result: gr.update(interactive=False),
                image_intent: "text",
                image_source: "New image",
                image_reference: None,
                supplied_image_prompt: "",
            }

        llama_assistants.select(
            fn=select_assistant_and_clear,
            api_visibility="undocumented",
            inputs=[attachment_controls.picker, llama_assistants],
            js=attachment_controls.capture_key_js,
            outputs=[
                llama_chat,
                llama_msg,
                llama_avatar,
                llama_system,
                llama_embed,
                llama_history,
                attachment_controls.files,
                attachment_controls.sources,
                image_intent,
                image_source,
                image_reference,
                supplied_image_prompt,
                *image_result_outputs,
            ],
            queue=False,
        )
        show_reasoning.change(
            display_history,
            inputs=[llama_history, show_reasoning],
            outputs=[llama_chat],
            api_visibility="undocumented",
        )
        llama_chat.clear(
            lambda key: ([], *attachment_controls.clear(key)),
            inputs=attachment_controls.picker,
            js=attachment_controls.capture_key_js,
            outputs=[llama_history, attachment_controls.files, attachment_controls.sources],
            api_visibility="undocumented",
            queue=False,
        )

        def refresh_assistant_catalogue(selected):
            picker = gr_llama_get_assistants(selected)
            current = picker["value"]
            info = assistant_info(current)
            updates = {
                llama_assistants: picker,
                assistant_delete: gr.update(interactive=current != "__normal__"),
                llama_avatar: gr.update(
                    value=assistant_avatar(info), label=info["name"], visible=True
                ),
            }
            if current != selected:
                updates.update({llama_system: info["system"], llama_embed: info["embed"]})
            return updates

        def load_assistant_session():
            from ruinedfooocus.core.profile_store import read_profile

            values = read_profile(settings.settings_path)
            selected = values.get("llama_assistant", "__normal__")
            updates = refresh_assistant_catalogue(selected)
            current = updates[llama_assistants]["value"]
            updates.update(llama_select_assistant(current))
            return updates

        app_llama_chat.load(
            load_assistant_session,
            outputs=[
                llama_assistants,
                assistant_delete,
                llama_avatar,
                llama_system,
                llama_embed,
                llama_chat,
                llama_history,
                llama_msg,
            ],
            api_visibility="undocumented",
            show_progress="hidden",
        )
        llama_reload.click(
            fn=refresh_assistant_catalogue,
            inputs=[llama_assistants],
            api_visibility="undocumented",
            outputs=[llama_assistants, assistant_delete, llama_avatar, llama_system, llama_embed],
            show_progress="hidden",
        )

        def open_assistant_editor(selected=None):
            try:
                info, identity, revision = (
                    assistant_edit_snapshot(selected)
                    if selected
                    else ({"name": "", "system": ""}, None, None)
                )
            except (ValueError, OSError) as error:
                return {assistant_status: str(error)}
            portrait = assistant_info(selected) if selected else None
            avatar = (
                assistant_avatar(portrait)
                if portrait and portrait["avatar"] != ASSET_ROOT / "chat_normal.png"
                else None
            )
            return {
                assistant_editor: gr.update(visible=True),
                assistant_name: info["name"],
                assistant_prompt: info.get("system", ""),
                assistant_image: avatar,
                assistant_draft: identity,
                assistant_version: revision,
                assistant_status: ""
                if identity or not selected
                else translate_ui("Save creates your own copy of this assistant."),
            }

        def save_assistant_draft(
            name, prompt, identity, revision, selected, avatar=None, save_as=False
        ):
            from ruinedfooocus.app.arguments import args

            try:
                identity, revision = save_assistant(
                    name,
                    prompt,
                    None if save_as else identity,
                    None if save_as else revision,
                    guest=args.guest,
                    avatar=avatar,
                )
            except (ValueError, OSError) as error:
                return {assistant_status: str(error)}
            updates = {
                llama_assistants: gr_llama_get_assistants(selected),
                assistant_draft: identity,
                assistant_version: revision,
                assistant_status: translate_ui(
                    "Assistant saved. Select it to start a conversation with the saved instructions."
                ),
            }
            if identity == selected:
                info = assistant_info(identity)
                updates[llama_avatar] = gr.update(
                    value=assistant_avatar(info), label=info["name"], visible=True
                )
            return updates

        def request_assistant_delete(selected):
            from ruinedfooocus.app.arguments import args

            if args.guest or selected == "__normal__":
                return {
                    assistant_status: translate_ui("General chat cannot be deleted.")
                    if selected == "__normal__"
                    else translate_ui("Guest sessions cannot delete assistants.")
                }
            try:
                info, _, revision = assistant_edit_snapshot(selected)
            except (ValueError, OSError) as error:
                return {assistant_status: str(error)}
            return {
                pending_assistant_delete: {"id": selected, "revision": revision},
                assistant_delete_dialog: {"name": info["name"]},
            }

        def confirm_assistant_delete(pending, selected, evt: gr.EventData):
            from ruinedfooocus.app.arguments import args

            updates = {pending_assistant_delete: None, assistant_delete_dialog: ""}
            if not pending or evt._data.get("action") != "delete":
                return updates
            identity = pending["id"]

            def repair_startup():
                if settings.default_settings.get("llama_assistant") == identity:
                    values = dict(settings.default_settings, llama_assistant="__normal__")
                    settings.save_profile(values, path_manager.paths, profile=settings.name)
                    settings.default_settings["llama_assistant"] = "__normal__"

            try:
                delete_assistant(
                    identity, pending["revision"], guest=args.guest, before_delete=repair_startup
                )
            except (ValueError, OSError) as error:
                return {**updates, assistant_status: str(error)}
            current = "__normal__" if selected == identity else selected
            updates.update(
                {
                    llama_assistants: gr_llama_get_assistants(current),
                    assistant_status: translate_ui("Assistant deleted."),
                }
            )
            if selected == identity:
                info = assistant_info("__normal__")
                updates.update(
                    {
                        llama_avatar: gr.update(
                            value=assistant_avatar(info), label=info["name"], visible=True
                        ),
                        llama_system: info["system"],
                        llama_embed: info["embed"],
                    }
                )
            return updates

        def save_startup_assistant(selected):
            from ruinedfooocus.app.arguments import args

            if args.guest:
                return translate_ui("Guest sessions cannot change startup settings.")
            if selected not in {value for _, value in assistant_choices()}:
                return translate_ui("Select an available assistant first.")
            try:
                values = dict(settings.default_settings, llama_assistant=selected)
                settings.save_profile(values, path_manager.paths, profile=settings.name)
            except (ValueError, OSError) as error:
                return str(error)
            return translate_ui(
                "Startup assistant saved for this profile. Open conversations keep their selected assistant."
            )

        editor_outputs = [
            assistant_editor,
            assistant_name,
            assistant_prompt,
            assistant_draft,
            assistant_version,
            assistant_status,
            assistant_image,
        ]
        assistant_new.click(
            open_assistant_editor,
            outputs=editor_outputs,
            api_visibility="undocumented",
            show_progress="hidden",
        )
        assistant_edit.click(
            open_assistant_editor,
            inputs=llama_assistants,
            outputs=editor_outputs,
            api_visibility="undocumented",
            show_progress="hidden",
        )
        draft_inputs = [
            assistant_name,
            assistant_prompt,
            assistant_draft,
            assistant_version,
            llama_assistants,
            assistant_image,
        ]
        draft_outputs = [
            llama_assistants,
            assistant_draft,
            assistant_version,
            assistant_status,
            llama_avatar,
        ]
        assistant_save.click(
            save_assistant_draft,
            inputs=draft_inputs,
            outputs=draft_outputs,
            api_visibility="undocumented",
            show_progress="hidden",
        )
        assistant_save_as.click(
            lambda *values: save_assistant_draft(*values, save_as=True),
            inputs=draft_inputs,
            outputs=draft_outputs,
            api_visibility="undocumented",
            show_progress="hidden",
        )
        assistant_cancel.click(
            lambda: (gr.update(visible=False), "", "", None, None, "", None),
            outputs=editor_outputs,
            api_visibility="undocumented",
            show_progress="hidden",
        )
        assistant_startup.click(
            save_startup_assistant,
            inputs=llama_assistants,
            outputs=assistant_status,
            api_visibility="undocumented",
            show_progress="hidden",
        )
        llama_assistants.change(
            lambda selected: gr.update(interactive=selected != "__normal__"),
            inputs=llama_assistants,
            outputs=assistant_delete,
            api_visibility="undocumented",
        )
        assistant_delete.click(
            request_assistant_delete,
            inputs=llama_assistants,
            outputs=[pending_assistant_delete, assistant_delete_dialog, assistant_status],
            api_visibility="undocumented",
        )
        assistant_delete_dialog.click(
            confirm_assistant_delete,
            inputs=[pending_assistant_delete, llama_assistants],
            outputs=[
                pending_assistant_delete,
                assistant_delete_dialog,
                llama_assistants,
                assistant_status,
                llama_avatar,
                llama_system,
                llama_embed,
            ],
            api_visibility="undocumented",
        )

    def select_catalogue_file(filename, session=None):
        """Select an exact local Hub candidate; loading remains an explicit action."""
        from ruinedfooocus.features.chat.llama_models import invalidate_model_catalogue

        invalidate_model_catalogue()
        target = Path(filename).resolve(strict=True)
        files = local_models()
        for family, choices in model_catalogue("Local models").items():
            for _, quant in choices:
                try:
                    candidate = Path(files.get(quant, quant)).resolve(strict=True)
                except (OSError, ValueError):
                    continue
                if candidate == target and candidate.is_file():
                    return {
                        llama_source: "Local models",
                        model_query: "",
                        **select_source("Local models", family, quant, session),
                    }
        raise ValueError(
            "The installed Chat file is not in the local model list. Refresh and select it again."
        )

    app_llama_chat.select_catalogue_file = select_catalogue_file
    app_llama_chat.catalogue_selection_inputs = [model_session]
    app_llama_chat.catalogue_selection_outputs = [llama_source, model_query, *model_outputs]
    return app_llama_chat
