import re
import random

from ruinedfooocus.features.prompting.sdxl_styles import (
    apply_style,
    load_styles,
)
from ruinedfooocus.features.prompting.prompt_scheduling import prompt_switch_per_step
from ruinedfooocus.features.prompting.one_button.build_dynamic_prompt import (
    build_dynamic_prompt,
    build_dynamic_negative,
)
from ruinedfooocus.app.state import path_manager


def process_metadata(gen_data):
    if gen_data.get("_metadata_prompt"):
        return gen_data
    from ruinedfooocus.features.gallery.image_metadata import read_metadata

    if read_metadata(gen_data.get("prompt")) is None:
        return gen_data
    from ruinedfooocus.app.state import models
    from ruinedfooocus.features.generation.request_metadata import capture_sampling_replay

    gen_data.update(capture_sampling_replay(gen_data, models, _literal_boundary=False))
    return gen_data


def get_promptlist(gen_data):
    from ruinedfooocus.features.gallery.image_metadata import read_metadata
    from ruinedfooocus.features.prompting.prompt_batches import split_prompt_batch

    if gen_data.get("_metadata_prompt") or read_metadata(gen_data["prompt"]) is not None:
        return [gen_data["prompt"]]
    return split_prompt_batch(gen_data["prompt"])


def process_wildcards(wildcard_text, directory=None, *, _stack=()):
    from ruinedfooocus.features.prompting.wildcards import wildcard_files, wildcard_choices

    if directory is None:
        directory = path_manager.paths.get("path_wildcards")
    placeholders = re.findall(r"__([\w-]+?(?::(?:(?!__).)+?)?)__", wildcard_text)
    if not placeholders:
        return wildcard_text
    for placeholder in placeholders:
        if placeholder in _stack:
            raise ValueError("Recursive wildcard: " + " -> ".join((*_stack, placeholder)))

    placeholder_choices = {}  # Store random choices for each placeholder
    files = wildcard_files(directory)

    for placeholder in placeholders:
        # Skip onebuttonprompt wildcards for now; handled below
        if placeholder.startswith("onebutton"):
            continue

        elif placeholder not in placeholder_choices:
            file_path = files.get(placeholder)
            if file_path is not None:
                words = wildcard_choices(file_path)
                if not words:
                    raise ValueError(f"Wildcard file has no choices: {file_path}")
                placeholder_choices[placeholder] = words
            else:
                print(
                    f"Error: Could not find file {placeholder}.txt in {directory} or its subdirectories."
                )
                placeholder_choices[placeholder] = None

    for placeholder in placeholders:
        random_choice = ""

        # Some one button prompt specials
        if placeholder.startswith("onebutton"):
            subjectoverride = ""
            placeholdersplit = placeholder.split(":", 1)
            if len(placeholdersplit) > 1:
                subjectoverride = placeholdersplit[1]

            if placeholder.startswith("onebuttonprompt"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonsubject"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonhumanoid"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonmale"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="male",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonfemale"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="female",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonanimal"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="animal - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonobject"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="object",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonlandscape"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="landscape - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonconcept"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="concept - all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebuttonartist"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    onlyartists=True,
                    artists=subjectoverride or "all",
                    advancedprompting=False,
                    base_model="SDXL",
                )
            elif placeholder.startswith("onebutton1girl"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="female",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            elif placeholder.startswith("onebutton1boy"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="human - all",
                    gender="male",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            elif placeholder.startswith("onebuttonfurry"):
                random_choice = build_dynamic_prompt(
                    insanitylevel=5,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    forcesubject="animal - all",
                    advancedprompting=False,
                    base_model="Anime Model",
                )
            # failover
            else:
                random_choice = build_dynamic_prompt(
                    insanitylevel=3,
                    imagetype="subject only mode",
                    givensubject=subjectoverride,
                    advancedprompting=False,
                    base_model="SDXL",
                )

        # Regular wildcards
        else:
            if placeholder_choices[placeholder] is None:
                continue
            random_choice = random.choice(placeholder_choices[placeholder])

        if "__" in random_choice:
            random_choice = process_wildcards(
                random_choice, directory, _stack=(*_stack, placeholder)
            )
        wildcard_text = wildcard_text.replace(f"__{placeholder}__", random_choice, 1)

    return wildcard_text


def use_random_prompt(gen_data, prompt):
    from ruinedfooocus.features.gallery.image_metadata import read_metadata

    if gen_data.get("_metadata_prompt") or read_metadata(prompt) is not None:
        return False
    if "obp_assume_direct_control" not in gen_data:
        return False
    # An empty image-conditioned prompt is intentional, not a request for OBP.
    control = gen_data.get("controlnet")
    tool = control.get("type") if control is not None else gen_data.get("cn_selection")
    if control is None and tool == "Custom...":
        tool = gen_data.get("cn_type")
    image_conditioned = (
        gen_data.get("input_image") is not None and tool not in (None, "", "None")
    ) or ("inpaint_toggle" in gen_data and gen_data["inpaint_toggle"])
    if image_conditioned:
        return gen_data.get("obp_image_prompt", False)
    return gen_data["obp_assume_direct_control"] or not prompt.strip()


def process_prompt(style, prompt, negative, gen_data=None):
    if gen_data is None:
        gen_data = {}
    from ruinedfooocus.features.image_editing.edit_prompt import render_prompt

    rendered = render_prompt(prompt, gen_data)
    if gen_data.get("_metadata_prompt") and not gen_data.get("_apply_prompt_processing"):
        return rendered, negative
    gen_data["raw_prompt"] = prompt
    gen_data["raw_negative"] = negative
    prompt = rendered
    enhanced = False

    def enhance_once(text, model_type="", preset_focus=""):
        nonlocal enhanced
        if enhanced:
            return text
        from ruinedfooocus.features.prompting.prompt_enhancement import enhance_prompt

        def check_cancelled():
            from ruinedfooocus.app.async_worker import check_interrupt

            check_interrupt(gen_data)

        result = enhance_prompt(text, model_type, preset_focus, check_cancelled=check_cancelled)
        enhanced = True
        return result

    if use_random_prompt(gen_data, prompt):
        from ruinedfooocus.features.prompting.one_button.prompt_formats import (
            format_for,
            engine_mode,
            finalize,
        )

        prompt_format = format_for(
            gen_data.get("OBP_modeltype", "Auto"), gen_data.get("base_model_name")
        )
        prompt = build_dynamic_prompt(
            insanitylevel=gen_data["obp_insanitylevel"],
            forcesubject=gen_data["obp_subject"],
            artists=gen_data["obp_artist"],
            subtypeobject=gen_data["obp_chosensubjectsubtypeobject"],
            subtypehumanoid=gen_data["obp_chosensubjectsubtypehumanoid"],
            subtypeconcept=gen_data["obp_chosensubjectsubtypeconcept"],
            gender=gen_data["obp_chosengender"],
            imagetype=gen_data["obp_imagetype"],
            imagemodechance=gen_data["obp_imagemodechance"],
            givensubject=gen_data["obp_givensubject"],
            smartsubject=gen_data["obp_smartsubject"],
            overrideoutfit=gen_data["obp_givenoutfit"],
            prefixprompt=gen_data["obp_prefixprompt"],
            suffixprompt=gen_data["obp_suffixprompt"],
            giventypeofimage=gen_data["obp_giventypeofimage"],
            antivalues=gen_data["obp_antistring"],
            OBP_preset=gen_data["OBP_preset"],
            advancedprompting=False,
            base_model=engine_mode(prompt_format),
            prompt_enhancer=gen_data["OBP_promptenhance"],
            enhancer=enhance_once,
        )
        prompt = finalize(prompt, prompt_format)

    # styles
    available_styles = load_styles()
    from ruinedfooocus.features.prompting.style_catalog import resolve_inline_styles

    styles, prompt = resolve_inline_styles(prompt, style, available_styles)
    if "lora_keywords" in gen_data:
        keywords = gen_data["lora_keywords"]
    else:
        keywords = ""
    provenance = {}
    p_txt, n_txt = apply_style(
        styles,
        prompt,
        negative,
        keywords,
        enhancer=enhance_once,
        provenance=provenance,
    )
    gen_data["style_provenance"] = provenance

    # wildcards
    if "__" in p_txt:
        p_txt = process_wildcards(p_txt)

    # apply auto negative prompt if enabled
    if "auto_negative" in gen_data and gen_data["auto_negative"] == True:
        n_txt = build_dynamic_negative(
            positive_prompt=p_txt, existing_negative_prompt=n_txt, base_model="SDXL"
        )
    return p_txt, n_txt


def parse_loras(prompt, negative):
    """Extract <lora:name> and <lora:name:weight> tags; names are resolved later.

    The weight is optional (1.0). A malformed weight leaves the tag as text.
    """
    from ruinedfooocus.features.models.lora_names import parse_loras as parse_inline_loras

    return parse_inline_loras(prompt, negative)
