"""Shared wildcard discovery, with user roots preceding shipped defaults."""

import os
from pathlib import Path
from functools import lru_cache

from ruinedfooocus.core.paths import APP_ROOT, data_path, resource_path


def wildcard_roots(directories=None):
    if directories is None:
        directories = ["wildcards"]
    elif isinstance(directories, (str, os.PathLike)):
        directories = [directories]
    roots = []
    for directory in directories:
        root = Path(directory).expanduser()
        root = root.resolve() if root.is_absolute() else data_path(root)
        if root not in roots:
            roots.append(root)
    # Retain user files left in the former bundled location without migrating them.
    for official in (APP_ROOT / "wildcards_official", resource_path("wildcards")):
        if official not in roots:
            roots.append(official)
    return roots


def wildcard_files(directories=None):
    files = {}
    for root in wildcard_roots(directories):
        for file in sorted(root.rglob("*.txt")):
            files.setdefault(file.stem, file)
    return files


@lru_cache(maxsize=32)
def _read_choices(filename, mtime_ns, size):
    return tuple(
        word.strip()
        for word in Path(filename).read_text(encoding="utf-8-sig").splitlines()
        if not word.startswith("#")
    )


def wildcard_choices(path):
    """Cache file contents by cheap filesystem identity, not generation-time hashes."""
    path = Path(path).resolve()
    stat = path.stat()
    return _read_choices(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _word_index(filename, mtime_ns, size):
    choices = tuple(word.lower() for word in _read_choices(filename, mtime_ns, size) if word)
    return choices, frozenset(choices)


def wildcard_word_index():
    path = wildcard_files().get("words")
    if path is None:
        raise ValueError("Prompt variation needs a words.txt wildcard file.")
    stat = path.stat()
    return _word_index(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
