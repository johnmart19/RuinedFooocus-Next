"""Contextual controls for native multi-stage and source-image workflows."""

import gradio as gr
from ruinedfooocus.features.generation.sdxl_refiner import FIELDS as REFINER_FIELDS
from ruinedfooocus.features.generation.bernini_image import FIELDS as BERNINI_FIELDS
from ruinedfooocus.features.generation.cascade_guidance import (
    FIELDS as GUIDANCE_FIELDS,
    IMAGE_FIELD,
)

from ruinedfooocus.core.translation_manager import translate_ui as t


FIELDS = (
    "cascade_stage_b",
    "cascade_b_steps",
    "cascade_b_cfg",
    "cascade_b_sampler",
    "cascade_b_scheduler",
    "cascade_compression",
    "cascade_source_denoise",
    "refiner_noise_augmentation",
    "pid_input_vae",
    "pid_latent_format",
    "pid_degrade_sigma",
    "model_recipe_path",
    "clip_layer_source",
    "model_recipe_sha256",
    "model_recipe_overrides",
    *REFINER_FIELDS,
    *GUIDANCE_FIELDS,
    IMAGE_FIELD,
    *BERNINI_FIELDS,
)


def workflow_visibility(family):
    return (
        family in ("Stable_Cascade_C", "StableCascade_C"),
        family == "HunyuanImage21Refiner",
        family == "PiD",
    )


def input_vae_choices(manager, latent_format, current=None):
    from ruinedfooocus.features.models.component_candidates import installed_candidates
    from ruinedfooocus.features.image_editing.pid_conditioning import pid_vae_roles

    names = (
        list(
            dict.fromkeys(
                candidate["name"]
                for role in pid_vae_roles(latent_format)
                for candidate in installed_candidates(role, manager)
                if candidate["compatibility"] == "compatible"
            )
        )
        if latent_format
        else []
    )
    return gr.update(choices=local_choices(names, current), value=current)


def stage_b_status(name, models):
    from ruinedfooocus.features.generation.cascade import stage_b_defaults

    recipe = stage_b_defaults(name, models)
    if not recipe:
        return t("No verified Stage B recipe. Enter all four sampling values explicitly.")
    return t(
        "Verified official Stage B recipe: {steps} steps · CFG {cfg} · {sampler} · {scheduler}. Explicit values override it."
    ).format(
        steps=recipe["cascade_b_steps"],
        cfg=recipe["cascade_b_cfg"],
        sampler=recipe["cascade_b_sampler"],
        scheduler=recipe["cascade_b_scheduler"],
    )


def local_choices(names, current):
    choices = list(names)
    if current and current not in choices:
        choices.append((t("Saved: {name} · Unavailable").format(name=current), current))
    return choices


def stage_b_choices(models):
    from ruinedfooocus.features.generation.cascade import require_bundle

    names = []
    for name in models.get_names("checkpoints"):
        path = models.get_file("checkpoints", name)
        if path is None:
            continue
        try:
            if models.detect_checkpoint_base(path) not in ("Stable_Cascade_B", "StableCascade_B"):
                continue
            require_bundle(path, "b")
        except (OSError, ValueError):
            continue
        names.append(name)
    return names


def create_ui(base_model, *, preset_selection=None, main_model=None, model_refresh=None):
    import ruinedfooocus.app.state as shared
    from comfy.samplers import KSampler
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint
    from ruinedfooocus.features.image_editing.image_guidance import effective_preset_checkpoint
    from ruinedfooocus.features.models.yaml_recipes import FAMILIES, discover_local_recipes

    defaults = shared.settings.default_settings
    family = family_for_checkpoint(base_model.value)
    initial = workflow_visibility(family)
    controls = {}
    with gr.Accordion(t("Stable Cascade · Stage B"), open=False, visible=initial[0]) as cascade:
        gr.Markdown(
            t(
                "Choose an installed bundled Stage B checkpoint. Blank sampling fields use its verified recipe when available; otherwise enter them explicitly. Native image edit and Image to image use the source image's native Cascade grids. Masks are unavailable."
            )
        )
        controls["cascade_stage_b"] = gr.Dropdown(
            label=t("Stage B checkpoint"),
            choices=local_choices(stage_b_choices(shared.models), defaults.get("cascade_stage_b")),
            value=defaults.get("cascade_stage_b"),
        )
        stage_b_recipe = gr.Markdown(
            stage_b_status(controls["cascade_stage_b"].value, shared.models)
        )
        with gr.Row():
            controls["cascade_b_steps"] = gr.Number(
                label=t("Stage B steps"),
                minimum=1,
                maximum=200,
                precision=0,
                value=defaults.get("cascade_b_steps"),
                info=t("Blank: Auto when a verified Stage B recipe is available."),
                min_width=180,
            )
            controls["cascade_b_cfg"] = gr.Number(
                label=t("Stage B CFG"),
                minimum=0,
                maximum=20,
                value=defaults.get("cascade_b_cfg"),
                min_width=180,
                info=t("Blank: Auto when a verified Stage B recipe is available."),
            )
        with gr.Row():
            controls["cascade_b_sampler"] = gr.Dropdown(
                label=t("Stage B sampler"),
                choices=sorted(KSampler.SAMPLERS),
                value=defaults.get("cascade_b_sampler"),
                min_width=180,
            )
            controls["cascade_b_scheduler"] = gr.Dropdown(
                label=t("Stage B scheduler"),
                choices=sorted(KSampler.SCHEDULERS),
                value=defaults.get("cascade_b_scheduler"),
                min_width=180,
            )
        with gr.Row():
            controls["cascade_compression"] = gr.Number(
                label=t("Cascade compression"),
                minimum=4,
                maximum=128,
                precision=0,
                value=defaults.get("cascade_compression", 42),
                min_width=180,
            )
            controls["cascade_source_denoise"] = gr.Number(
                label=t("Cascade source denoise"),
                minimum=0,
                maximum=1,
                value=defaults.get("cascade_source_denoise", 0.75),
                info=t("Used only with a source image. Zero preserves the encoded Stage C prior."),
                min_width=180,
            )
    with gr.Accordion(t("Hunyuan image refiner"), open=False, visible=initial[1]) as refiner:
        gr.Markdown(t("Choose Native image edit and add the image to refine."))
        controls["refiner_noise_augmentation"] = gr.Number(
            label=t("Noise augmentation"),
            minimum=0,
            maximum=1,
            value=defaults.get("refiner_noise_augmentation"),
            info=t("Blank uses the native value, 0.10. Zero is an explicit override."),
        )
    with gr.Accordion(t("Pixel diffusion input"), open=False, visible=initial[2]) as pid:
        gr.Markdown(
            t(
                "Choose Native image edit and add a source image. Select the VAE and latent format that encoded it."
            )
        )
        controls["pid_latent_format"] = gr.Dropdown(
            label=t("Input latent format"),
            choices=["flux", "sd3", "sdxl", "qwenimage"],
            value=defaults.get("pid_latent_format"),
        )
        saved_vae = defaults.get("pid_input_vae")
        vaes = input_vae_choices(shared.path_manager, controls["pid_latent_format"].value)[
            "choices"
        ]
        controls["pid_input_vae"] = gr.Dropdown(
            label=t("Input VAE"), choices=local_choices(vaes, saved_vae), value=saved_vae
        )
        controls["pid_degrade_sigma"] = gr.Number(
            label=t("Input degradation"),
            minimum=0,
            maximum=1,
            value=defaults.get("pid_degrade_sigma"),
            info=t("Blank uses the native value, 0."),
        )

    recipe_choices = discover_local_recipes(shared.path_manager)
    saved_recipe = (defaults.get("model_recipe_path") or "") if family in FAMILIES else ""
    if saved_recipe and saved_recipe not in {value for _, value in recipe_choices}:
        recipe_choices.append((t("Saved recipe · Unavailable"), saved_recipe))
    with gr.Accordion(
        t("Local checkpoint recipe"),
        open=False,
        visible=family in FAMILIES,
    ) as recipe:
        controls["model_recipe_path"] = gr.Dropdown(
            label=t("YAML recipe"),
            choices=[(t("None"), "")] + recipe_choices,
            value=saved_recipe,
            info=t(
                "Optional prediction and CLIP-layer settings for an ordinary Stable Diffusion checkpoint."
                " To accept an edited file, choose None and select it again."
            ),
        )
        controls["clip_layer_source"] = gr.Dropdown(
            label=t("CLIP layer settings"),
            choices=[(t("Current Clip Skip"), "request"), (t("Selected recipe"), "recipe")],
            value=defaults.get("clip_layer_source", "request") if saved_recipe else "request",
            interactive=bool(saved_recipe),
            info=t("Editing Clip Skip or choosing a PNG preset restores Current Clip Skip."),
        )
        refresh_recipes = gr.Button(t("Refresh local recipes"), size="sm")
    controls["model_recipe_sha256"] = gr.State(
        defaults.get("model_recipe_sha256") if saved_recipe else None
    )
    controls["model_recipe_overrides"] = gr.State(
        defaults.get("model_recipe_overrides") if saved_recipe else None
    )

    from ruinedfooocus.features.generation.ui_sdxl_refiner import create_ui as create_refiner_ui
    from ruinedfooocus.features.generation.ui_cascade_guidance import (
        create_ui as create_cascade_guidance_ui,
    )

    controls.update(
        create_cascade_guidance_ui(
            base_model,
            preset_selection=preset_selection,
            main_model=main_model,
            model_refresh=model_refresh,
        )
    )

    controls.update(
        create_refiner_ui(
            base_model,
            preset_selection=preset_selection,
            main_model=main_model,
            model_refresh=model_refresh,
        )
    )
    from ruinedfooocus.features.generation.ui_bernini_image import create_ui as create_bernini_ui

    controls.update(
        create_bernini_ui(
            base_model,
            preset_selection=preset_selection,
            main_model=main_model,
            model_refresh=model_refresh,
        )
    )
    for key, component in controls.items():
        shared.add_ctrl(key, component)

    controls["cascade_stage_b"].change(
        lambda name: stage_b_status(name, shared.models),
        inputs=controls["cascade_stage_b"],
        outputs=stage_b_recipe,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    identity = [controls["model_recipe_sha256"], controls["model_recipe_overrides"]]
    controls["model_recipe_path"].input(
        lambda: (None, None),
        outputs=identity,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    controls["model_recipe_path"].change(
        lambda path: gr.update(interactive=bool(path), **({} if path else {"value": "request"})),
        inputs=controls["model_recipe_path"],
        outputs=controls["clip_layer_source"],
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def refresh_recipe_choices(current):
        choices = [(t("None"), "")] + discover_local_recipes(shared.path_manager)
        if current and current not in {value for _, value in choices}:
            choices.append((t("Selected recipe · Unavailable"), current))
        return gr.update(choices=choices, value=current), gr.skip(), gr.skip(), gr.skip()

    refresh_recipes.click(
        refresh_recipe_choices,
        inputs=controls["model_recipe_path"],
        outputs=[controls["model_recipe_path"], *identity, controls["clip_layer_source"]],
        show_progress="hidden",
        api_visibility="undocumented",
    )
    clip_skip = shared.get_ctrl("clip_skip")
    if clip_skip is not None:
        clip_skip.input(
            lambda: "request",
            outputs=controls["clip_layer_source"],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
    if preset_selection is not None:
        preset_selection.change(
            lambda preset: "request" if preset else gr.skip(),
            inputs=preset_selection,
            outputs=controls["clip_layer_source"],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    def refresh_input_vae(latent_format, current):
        return input_vae_choices(shared.path_manager, latent_format, current)

    controls["pid_latent_format"].change(
        refresh_input_vae,
        inputs=[controls["pid_latent_format"], controls["pid_input_vae"]],
        outputs=controls["pid_input_vae"],
        show_progress="hidden",
        api_visibility="undocumented",
    )

    model = main_model if main_model is not None else base_model

    def context(name, preset=None):
        checkpoint = (
            effective_preset_checkpoint(
                name, preset, shared.settings.default_settings.get("preset_mode", 7)
            )
            if preset_selection is not None
            else name
        )
        if checkpoint is None:
            return [gr.update() for _ in range(8)]
        family = family_for_checkpoint(checkpoint)
        updates = [
            gr.update(visible=active)
            for active in (*workflow_visibility(family), family in FAMILIES)
        ]
        return updates + ([gr.skip()] * 4 if family in FAMILIES else ["", "request", None, None])

    gr.on(
        triggers=[model.change]
        + ([preset_selection.change] if preset_selection is not None else [])
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=context,
        inputs=[model, preset_selection] if preset_selection is not None else [model],
        outputs=[
            cascade,
            refiner,
            pid,
            recipe,
            controls["model_recipe_path"],
            controls["clip_layer_source"],
            *identity,
        ],
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    if model_refresh is not None:

        def refresh_choices(stage_b, latent_format, input_vae):
            names = stage_b_choices(shared.models)
            return gr.update(
                choices=local_choices(names, stage_b), value=stage_b
            ), refresh_input_vae(latent_format, input_vae)

        model_refresh.change(
            refresh_choices,
            inputs=[
                controls["cascade_stage_b"],
                controls["pid_latent_format"],
                controls["pid_input_vae"],
            ],
            outputs=[controls["cascade_stage_b"], controls["pid_input_vae"]],
            show_progress="hidden",
            api_visibility="undocumented",
        )
    return controls
