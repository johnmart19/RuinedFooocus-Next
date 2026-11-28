"""Shape One Button prompts for the model family that will read them.

The One Button engine writes SD-era keyword prompts. Modern models differ:
Illustrious/NoobAI/Pony expect booru tags led by quality tags, Anima reads
lowercase tags mixed with prose through an LLM encoder (no weight syntax), and
Flux/Qwen/Krea/MOSS-style models want plain descriptions without weights, tags
or emojis. Values are stored in settings, so the original keys stay valid.
"""

import re

AUTO, TAGS, ANIME, ANIMA, NATURAL = "Auto", "SDXL", "Anime Model", "Anima", "Stable Cascade"
FORMAT_CHOICES = [
    ("Auto (match checkpoint)", AUTO),
    ("Tags: SD 1.5 / SDXL", TAGS),
    ("Anime tags: Illustrious / NoobAI / Pony", ANIME),
    ("Anima: tags + description", ANIMA),
    ("Natural language: Flux, Qwen, Krea, MOSS, Z-Image…", NATURAL),
]
FORMATS = {value for _, value in FORMAT_CHOICES}

# Checkpoint family substrings (Civitai base model or detected architecture), first match wins.
_FAMILY_FORMATS = (
    (("anima",), ANIMA),
    (("illustrious", "noobai", "pony", "animagine"), ANIME),
    (("sd 1", "sd1", "sd 2", "sd2", "sdxl", "ssd", "playground", "segmind"), TAGS),
)
ANIME_QUALITY = "masterpiece, best quality, very aesthetic, absurdres"
ANIMA_QUALITY = "masterpiece, best quality"

_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️‍]")
_WEIGHT = re.compile(r"\(([^()]*?)(?::\s*-?\d+(?:\.\d+)?)?\)")
_SD_BOILERPLATE = re.compile(
    r"\b(?:8k|4k|uhd|hdr|highres|absurdres|masterpiece|best quality|high quality|trending on artstation|"
    r"unreal engine|octane render|award ?winning|awardwinning)\b",
    re.I,
)
# The engine writes e.g. "gorgeous 1girl", so bare nouns read naturally in place.
_BOORU_COUNT = {
    "1girl": "girl",
    "1boy": "boy",
    "2girls": "two girls",
    "2boys": "two boys",
    "1other": "person",
    "solo": "",
}


def family_format(family):
    """Prompt format for a checkpoint family; unknown modern families use natural language."""
    name = str(family or "").lower()
    if not name or name == "unknown":
        return TAGS
    for keys, value in _FAMILY_FORMATS:
        if any(key in name for key in keys):
            return value
    return NATURAL


def resolve(value, family=None):
    """Concrete format for a stored choice; Auto follows the checkpoint family."""
    if value == AUTO:
        return family_format(family)
    return value if value in FORMATS else TAGS


def checkpoint_family(checkpoint):
    """Civitai base model, or the architecture detected from the file when unknown."""
    if not checkpoint:
        return None
    try:
        import ruinedfooocus.app.state as shared

        family = shared.models.get_model_base(
            shared.models.get_models_by_path("checkpoints", checkpoint, fetch=False)
        )
        path = shared.models.get_file("checkpoints", checkpoint)
        detected = shared.models.detect_checkpoint_base(path) if path is not None else None
        return detected or family
    except Exception:
        return None


def format_for(value, checkpoint=None):
    return resolve(value, checkpoint_family(checkpoint) if value == AUTO else None)


def engine_mode(fmt):
    """One Button engine mode: Anima starts from anime tags; natural language drops weights."""
    return {ANIMA: ANIME}.get(fmt, fmt)


def _tidy(prompt):
    prompt = _EMOJI.sub("", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"\s*,\s*(?:,\s*)*", ", ", prompt)
    return prompt.strip(" ,")


def _without_weights(prompt):
    while _WEIGHT.search(prompt):
        prompt = _WEIGHT.sub(r"\1", prompt)
    return re.sub(r"[\[\]{}]", "", prompt)


def _with_prefix(prefix, prompt):
    tags = {tag.strip().lower() for tag in prompt.split(",")}
    missing = [tag for tag in prefix.split(", ") if tag not in tags]
    return ", ".join(missing + [prompt]) if missing else prompt


def finalize(prompt, fmt):
    """Clean the engine output for the target format; LoRA/style tags pass through unchanged."""
    if not prompt:
        return prompt
    protected = re.findall(r"<[^<>]+>", prompt)
    text = re.sub(r"<[^<>]+>", " ", prompt)
    # The local enhancer appends its addition on a new line; join it as another phrase.
    text = re.sub(r"\s*\n+\s*", ", ", text.strip())
    text = re.sub(r"\bawardwinning\b", "award-winning", text, flags=re.I)
    if fmt in (ANIMA, NATURAL):
        text = _without_weights(text)
    if fmt == ANIMA:
        # Anima's captions are lowercase tags with spaces; only score_* keeps underscores.
        text = re.sub(r"(?<!score)_", " ", text.lower())
        text = _with_prefix(ANIMA_QUALITY, _tidy(text))
    elif fmt == ANIME:
        text = _with_prefix(ANIME_QUALITY, _tidy(text))
    elif fmt == NATURAL:
        text = _SD_BOILERPLATE.sub("", text)
        text = re.sub(
            r"\b(" + "|".join(_BOORU_COUNT) + r")\b",
            lambda match: _BOORU_COUNT[match.group(1).lower()],
            text,
            flags=re.I,
        )
        text = _tidy(text)
        text = text[:1].upper() + text[1:]
    else:
        text = _tidy(text)
    return " ".join([text, *protected]).strip()
