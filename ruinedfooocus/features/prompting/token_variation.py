"""Opt-in token mutation using an already cached image tokenizer, never a loader."""

import random
import re
import sys


def cached_tokenizer(checkpoint):
    """Resolve only an unchanged, selected, single-stream tokenizer already in RAM.

    Multi-encoder models need an explicit stream policy and are deliberately not
    approximated with a different model's tokenizer or a downloaded GPT-2 copy.
    """
    shared = sys.modules.get("ruinedfooocus.app.state")
    pipeline = shared.state.get("pipeline") if shared is not None else None
    if not checkpoint or pipeline is None:
        raise ValueError(
            "Tokens needs the selected image model's already cached tokenizer. No model is loaded by this tool."
        )
    from ruinedfooocus.features.models.model_identity import file_signature, files_unchanged

    path = shared.models.get_file("checkpoints", checkpoint)
    try:
        selected = file_signature(path) if path is not None else None
    except OSError:
        selected = None
    if selected is None or selected != getattr(pipeline, "base_file_signature", None):
        raise ValueError(
            "Tokens is unavailable: the selected checkpoint does not match the cached image model."
        )
    if not files_unchanged(getattr(pipeline, "component_file_signatures", None)):
        raise ValueError("Tokens is unavailable: a cached image-model component changed.")
    from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

    components = {
        key: value
        for key, value in shared.settings.default_settings.items()
        if key in RESOURCE_DEFAULTS
    }
    if components != getattr(pipeline, "component_settings", None):
        raise ValueError(
            "Tokens is unavailable: component overrides changed since the cached model was loaded."
        )
    clip = getattr(getattr(pipeline, "xl_base", None), "clip", None)
    wrapper = getattr(clip, "tokenizer", None)
    # Pinned ComfyUI SD1Tokenizer declares its sole stream in .clip. SDXL,
    # Flux and other multi-stream tokenizers intentionally have no such contract.
    stream = getattr(wrapper, "clip", None)
    tokenizer = (
        getattr(getattr(wrapper, stream, None), "tokenizer", None)
        if isinstance(stream, str)
        else None
    )
    if not all(
        callable(getattr(tokenizer, name, None))
        for name in ("get_vocab", "tokenize", "convert_tokens_to_string")
    ):
        raise ValueError(
            "Tokens is unsupported for this cached tokenizer (including multi-encoder models). Words and One Button remain available."
        )
    return tokenizer


# Match syntax/literals before editable words. Only the word matches below can
# change; their original separators, punctuation and whitespace remain untouched.
LITERAL_PATTERN = (
    r'<[^<>]*>|__[^\n]*?__|"(?:\\.|[^"\\])*"|'
    r"“[^”]*”|«[^»]*»|'(?:\\.|[^'\\])*'"
)
PROMPT_PARTS = re.compile(LITERAL_PATTERN + r"|\[[^\[\]]*\]|\([^()]*\)|\{[^{}]*\}|\b[\w'-]+\b")
EDITABLE_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
LITERALS = re.compile(LITERAL_PATTERN)


def map_prompt_words(prompt, replace):
    """Keep nested prompt groups intact as well as literals and directives."""
    groups = {"(": ")", "[": "]", "{": "}"}
    parts = []
    start = 0
    cursor = 0
    while cursor < len(prompt):
        literal = LITERALS.match(prompt, cursor)
        if literal:
            cursor = literal.end()
            continue
        if prompt[cursor] not in groups:
            cursor += 1
            continue
        parts.append(PROMPT_PARTS.sub(replace, prompt[start:cursor]))
        group_start = cursor
        stack = [groups[prompt[cursor]]]
        cursor += 1
        while cursor < len(prompt) and stack:
            literal = LITERALS.match(prompt, cursor)
            if literal:
                cursor = literal.end()
                continue
            char = prompt[cursor]
            if char in groups:
                stack.append(groups[char])
            elif char == stack[-1]:
                stack.pop()
            cursor += 1
        parts.append(prompt[group_start:cursor])
        start = cursor
    parts.append(PROMPT_PARTS.sub(replace, prompt[start:]))
    return "".join(parts)


def safe_token_vocabulary(tokenizer):
    """Build once per preview; do not retain an encoder in a global cache."""
    special = set(getattr(tokenizer, "all_special_tokens", ()))
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    candidates = []
    # Some installed tokenizer backends return hash-map iteration order anew on
    # each call. Stable token IDs make a fixed RNG seed reproducible.
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        if token in special or token_id in special_ids:
            continue
        # A replacement must not introduce new directives, numbers or delimiters.
        decoded = tokenizer.convert_tokens_to_string([token]).strip()
        if EDITABLE_WORD.fullmatch(decoded) and decoded not in ("AND", "BREAK"):
            candidates.append(token)
    if not candidates:
        raise ValueError("Tokens is unsupported: this tokenizer has no safe text-piece vocabulary.")
    return candidates


def randomize_tokens(prompt, strength, tokenizer, *, rng=None, candidates=None):
    """Mutate actual tokenizer pieces within unprotected words, not dictionary words."""
    rng = rng or random
    if not 0 <= float(strength) <= 100:
        raise ValueError("Variation strength must be between 0 and 100.")
    if not strength:
        return prompt
    candidates = safe_token_vocabulary(tokenizer) if candidates is None else candidates
    special = set(getattr(tokenizer, "all_special_tokens", ()))

    def mutate(match):
        word = match.group()
        if not EDITABLE_WORD.fullmatch(word) or word in special or word in ("AND", "BREAK"):
            return word
        pieces = tokenizer.tokenize(word)
        changed = False
        result = []
        for piece in pieces:
            if piece not in special and rng.random() < float(strength) / 100:
                result.append(rng.choice(candidates))
                changed = True
            else:
                result.append(piece)
        if not changed:
            return word
        value = tokenizer.convert_tokens_to_string(result).strip()
        # Byte-level or custom decoders may emit unsafe sequences even from safe
        # isolated tokens. Reject that mutation rather than alter prompt syntax.
        return (
            value
            if value
            and all(
                EDITABLE_WORD.fullmatch(part) and part not in ("AND", "BREAK")
                for part in value.split()
            )
            else word
        )

    return map_prompt_words(prompt, mutate)
