from ruinedfooocus.core.paths import data_path, resource_path
import random
import re
from ruinedfooocus.features.prompting.style_catalog import STYLE_ALIASES
from ruinedfooocus.features.prompting.prompt_enhancement import enhance_prompt
from ruinedfooocus.features.prompting.one_button import build_dynamic_prompt

DEFAULT_STYLES_FILE = resource_path("defaults", "styles.default")
STYLES_FILE = data_path("settings", "styles.csv")


def load_styles():
    from ruinedfooocus.features.prompting.style_catalog import style_templates

    return style_templates(DEFAULT_STYLES_FILE, STYLES_FILE)


def apply_style(style, prompt, negative_prompt, lora_keywords, *, enhancer=None, provenance=None):
    from ruinedfooocus.features.prompting.style_catalog import normalize_selection

    style = normalize_selection(style)
    styles = load_styles()
    allstyles = [
        name for name in styles if name.startswith("Style:") and name != "Style: Pick Random"
    ]
    output_prompt = ""
    negative_parts = []
    temp_style_prompt = prompt
    bLocalChat = False
    artifylist = []
    artifystylelist = []
    index = 0

    while "Style: Pick Random" in style:
        style[style.index("Style: Pick Random")] = random.choice(allstyles) if allstyles else "None"

    if provenance is not None:
        provenance.update(
            version=1,
            selected=style.copy(),
            templates={name: styles[name] for name in style if name in styles},
        )
    if not style:
        return prompt, negative_prompt

    for s in style.copy():
        _s = s.upper().strip()
        if _s in map(str.upper, ["Local chat model", "Style: Local chat model", *STYLE_ALIASES]):
            bLocalChat = True
            del style[style.index(s)]

        if _s in map(str.upper, ["LoRA keywords", "Style: LoRA keywords"]):
            style[style.index(s)] = "LoRA keywords"  # Make sure it has the correct name

    while index < len(style):
        if style[index].startswith("Artify"):
            artifylist.append(style[index].replace("Artify: ", ""))
            artifystylelist.append(style[index])
        index += 1

    style = [x for x in style if x not in artifystylelist]

    for s in style:
        if s not in styles:
            continue
        p, n = styles[s]
        if (
            s == "Style: sai-cinematic"
            and p
            and p.startswith("cinematic film still {prompt}")
            and re.search(r"\banime\b", prompt, re.IGNORECASE)
        ):
            p = p.replace("cinematic film still", "cinematic anime illustration", 1)
            # Do not make the style reject the medium explicitly requested in the prompt.
            n = ", ".join(
                tag.strip()
                for tag in (n or "").split(",")
                if tag.strip().lower()
                not in {"", "anime", "cartoon", "graphic", "painting", "crayon"}
            )
        if p and s != "None":
            # A style adds to the user's text; legacy empty/header rows cannot replace it.
            if "{prompt}" not in p:
                p = "{prompt}, " + p
            template = (p + ", ").replace(", ,", ", ")

            def substitute(match):
                if match.group(0) == "{prompt}":
                    return temp_style_prompt
                return "{lora_keywords}" if lora_keywords is None else lora_keywords

            temp_style_prompt = re.sub(r"\{prompt\}|\{lora_keywords\}", substitute, template)
            output_prompt = temp_style_prompt
        if n:
            negative_parts.append(n)

    # Retain the original when no visual template was selected.
    if output_prompt == "":
        output_prompt = prompt

    for artist in artifylist:
        output_prompt = build_dynamic_prompt.artify_prompt(prompt=output_prompt, artists=artist)

    if bLocalChat:
        output_prompt = (enhancer or enhance_prompt)(output_prompt)

    return output_prompt, ", ".join(negative_parts + ([negative_prompt] if negative_prompt else []))
