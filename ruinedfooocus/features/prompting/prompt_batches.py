"""Explicit prompt batches without splitting literal lettering or escaped text."""

import re

BATCH_SEPARATOR = "\n---\n"
# Keep complete quoted literals and escaped characters ahead of separator lines.
# Apostrophes inside words are prose, including inside single-quoted lettering.
_TOKENS = re.compile(
    r'\\.|"(?:\\.|[^"\\])*"|“[^”]*”|«[^»]*»|'
    r"(?<!\w)'(?:\\.|[^'\\]|(?<=\w)'(?=\w))*'(?!\w)|"
    r"(?P<separator>(?:^|\r?\n)[ \t]*---[ \t]*(?:\r?\n|$))",
    re.MULTILINE,
)


def split_prompt_batch(text):
    """Only an unquoted standalone --- line separates prompts; retain other bytes."""
    result, start = [], 0
    for match in _TOKENS.finditer(text):
        if match.lastgroup == "separator":
            result.append(text[start : match.start()])
            start = match.end()
    return [*result, text[start:]]


def join_prompt_batch(prompts):
    prompts = tuple(prompts)
    joined = BATCH_SEPARATOR.join(prompts)
    expected = [part for prompt in prompts for part in split_prompt_batch(prompt)]
    if split_prompt_batch(joined) != expected:
        raise ValueError("Quoted text crosses a batch separator. Close or escape the quote first.")
    return joined
