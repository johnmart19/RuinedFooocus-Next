from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui

"""Checkpoint-scoped component editing without loading native weights."""

import gradio as gr

from ruinedfooocus.app.state import path_manager, settings


def role_choices(name, key, overrides, scope, advanced=False, locale=None):
    if not key:
        # Unknown/uninspected checkpoints have no selectable component role.
        return gr.update(choices=[], value=None, interactive=False)
    from ruinedfooocus.features.models.component_candidates import installed_candidates
    from ruinedfooocus.features.models.component_profiles import (
        component_inventory,
        session_choices,
    )
    from ruinedfooocus.features.models.ui_model_components import selected

    try:
        path, family = selected(name)
        choices, identity = session_choices(path, settings.default_settings, overrides, scope)
        state = component_inventory(path, family, settings.default_settings, path_manager, choices)
        row = next((row for row in state["rows"] if row["key"] == key), None)
        if row is None:
            return gr.update(choices=[], value=None, interactive=False)
        if row.get("locked"):
            return gr.update(
                choices=[("Bundled local file", "")],
                value="",
                interactive=False,
                info=localize(
                    "Effective: {filename}. The local bundle fixes this role.",
                    locale,
                    filename=row["filename"],
                ),
            )
        from ruinedfooocus.features.models.approximate_codecs import ALIASES, is_choice

        embedded = row["status"] == "Embedded in checkpoint" or row.get("embedded_fallback", False)
        if embedded and key not in ALIASES.values():
            return gr.update(
                choices=[("Built into checkpoint", "")],
                value="",
                interactive=False,
                info=localize(
                    "The checkpoint provides this component. Profile defaults do not replace it.",
                    locale,
                ),
            )
        candidates = installed_candidates(key, path_manager, companion_rows=state["rows"])
        if embedded:
            candidates = [item for item in candidates if is_choice(item["name"])]
        options = [("Automatic", "")]
        options += [
            (
                (
                    localize(
                        "Approximate codec: {alias} (encoder + decoder)",
                        locale,
                        alias=candidate["name"].split(":", 1)[1],
                    )
                    if is_choice(candidate["name"])
                    else candidate["name"]
                )
                + (" · Unverified" if candidate["compatibility"] == "unknown" else ""),
                candidate["name"],
            )
            for candidate in candidates
            if candidate["compatibility"] == "compatible"
            or advanced
            and candidate["compatibility"] == "unknown"
        ]
        value = (
            ""
            if row["status"] == "Embedded in checkpoint"
            else choices.get(key, settings.default_settings.get(key)) or ""
        )
        if value not in {option[1] for option in options}:
            candidate = next((item for item in candidates if item["name"] == value), None)
            status = (
                "Incompatible"
                if candidate and candidate["compatibility"] == "incompatible"
                else "Missing"
                if row["status"] == "Missing"
                else "Unverified"
            )
            # Retain the actual choice until an explicit save. Validation still
            # rejects missing/incompatible files and requires consent for unknowns.
            options.append((f"Current: {value} · {status}", value))
        return gr.update(
            choices=options,
            value=value,
            interactive=True,
            info=localize(
                "Effective: {filename}. Alternatives match this role's native reader and component contract.",
                locale,
                filename=row["filename"],
            ),
        )
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from None


def save_choice(name, key, value, advanced, overrides, scope):
    from ruinedfooocus.features.models.component_candidates import validate_local_choice
    from ruinedfooocus.features.models.component_profiles import (
        component_inventory,
        saved_choices,
        session_choices,
        update_choices,
    )
    from ruinedfooocus.features.models.ui_model_components import selected

    try:
        path, family = selected(name)
        choices, identity = session_choices(path, settings.default_settings, overrides, scope)
        if scope != identity:
            raise ValueError("The checkpoint changed. Refresh components before saving.")
        state = component_inventory(path, family, settings.default_settings, path_manager, choices)
        row = next(
            (row for row in state["rows"] if row["key"] == key),
            None,
        )
        from ruinedfooocus.features.models.approximate_codecs import choice_alias

        if (
            row is None
            or row.get("locked")
            or (row["status"] == "Embedded in checkpoint" or row.get("embedded_fallback"))
            and value
            and not choice_alias(key, value)
        ):
            raise ValueError(
                "Select an external component role. Built-in components cannot be replaced."
            )
        choices[key] = validate_local_choice(
            key, value, path_manager, allow_unknown=advanced, companion_rows=state["rows"]
        )
        saved = saved_choices(settings.default_settings, path)
        saved[key] = choices[key]
        updated = update_choices(settings.default_settings, path, identity, saved)
        settings.save_profile(updated, path_manager.paths, profile=settings.name)
        return choices, identity
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from None


def create_ui(base_model, overrides, scope, refresh_callback, refresh_outputs):
    with gr.Accordion(translate_ui("Component choices for this checkpoint"), open=False):
        gr.Markdown(
            translate_ui(
                "Applies on the next load. Automatic keeps the normal VAE; a full approximate codec requires its encoder and decoder."
            )
        )
        role = gr.Dropdown(label=translate_ui("Component role"), choices=[], interactive=True)
        alternative = gr.Dropdown(
            label=translate_ui("Installed alternative"), choices=[], interactive=False
        )
        advanced = gr.Checkbox(
            label=translate_ui("Include unverified installed files (advanced)"),
            value=False,
            info=translate_ui("Unverified files may fail to load."),
        )
        save = gr.Button(translate_ui("Save this component choice"), size="sm", interactive=False)

    def update_choice(name, key, overrides, scope, advanced, request: gr.Request = None):
        choice = role_choices(
            name, key, overrides, scope, advanced, resolve_locale(request=request)
        )
        return choice, gr.update(interactive=choice.get("interactive", False))

    def refresh_choice(name, overrides, scope, selected_role):
        result = list(refresh_callback(name, overrides, scope))
        roles = result[-1]
        if selected_role in {value for label, value in roles.get("choices", ())}:
            result[-1] = {**roles, "value": selected_role}
        return tuple(result)

    for control in (role, advanced):
        control.change(
            update_choice,
            inputs=[base_model, role, overrides, scope, advanced],
            outputs=[alternative, save],
            show_progress="hidden",
            api_visibility="undocumented",
        )
    save.click(
        save_choice,
        inputs=[base_model, role, alternative, advanced, overrides, scope],
        outputs=[overrides, scope],
        show_progress="hidden",
        api_visibility="undocumented",
    ).then(
        refresh_choice,
        inputs=[base_model, overrides, scope, role],
        outputs=[*refresh_outputs, role],
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return role, alternative, advanced
