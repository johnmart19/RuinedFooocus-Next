from ruinedfooocus.core.paths import resource_path
import random


def get_hint():
    hintfiles = resource_path("hints").glob("*.txt")
    hints = []
    for hintfile in hintfiles:
        lines = hintfile.read_text(encoding="utf8").splitlines()
        hints += lines

    hint = f"**LPT:** *{random.choice(hints)}*"
    return hint
