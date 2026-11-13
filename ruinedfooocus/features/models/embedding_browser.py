"""Discover textual inversions without loading weights or contacting model services."""

from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re

from ruinedfooocus.features.models.model_identity import file_signature

# Match the pinned native weight formats. Explicit extensions also let the
# native loader open .pth/.ckpt files without its extension fallback.
EXTENSIONS = frozenset((".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"))
_WEIGHT = r":[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)"
_WORD = re.compile(r"[\w-]+")
_PROMPT_TOKENS = re.compile(
    r""""(?:\\.|[^"\\])*"|(?<!\w)'(?:\\.|[^'\\])*'|`[^`\n]*`|"""
    r"(?<!\S)\#[^\n]*|<[^>\n]*>|\[[^\]\n]*\]|\\.|(?P<word>[\w-]+)"
)


@dataclass(frozen=True)
class Embedding:
    name: str
    signature: tuple
    aliases: tuple[str, ...]

    @property
    def token(self):
        return "embedding:" + self.name


def _safe_name(name):
    # Native SDTokenizer splits embedding names at whitespace; prompt syntax is
    # parsed before lookup. Never transform a filename into a different file.
    return not re.search(r"[\s<>\[\](){},:|\\\"'`]|__|---", name)


def _aliases(name):
    path = PurePosixPath(name)
    return {name, str(path.with_suffix("")), path.name, path.stem}


def scan_embeddings(directories):
    """Return unique native prompt names and a concise discovery status.

    Extensions are explicit so two formats with the same stem remain distinct.
    Alias roots and nested configured roots never produce duplicate entries.
    Identical prompt names pointing to different files are hidden: Comfy's
    recursive directory expansion does not promise which root wins.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    files, aliases = {}, defaultdict(set)
    roots, unsupported, unavailable = set(), set(), 0
    for directory in directories:
        try:
            root = Path(directory).resolve()
            root_key = os.path.normcase(str(root))
            if root_key in roots or not root.is_dir():
                continue
        except OSError:
            unavailable += 1
            continue
        roots.add(root_key)
        try:
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                    continue
                try:
                    signature = tuple(file_signature(path))
                except OSError:
                    unavailable += 1
                    continue
                identity = os.path.normcase(signature[0])
                name = path.relative_to(root).as_posix()
                # Include unsafe names in alias collision checks even though
                # they cannot be inserted as native prompt references.
                for alias in _aliases(name):
                    aliases[os.path.normcase(alias)].add(identity)
                if not _safe_name(name):
                    unsupported.add(identity)
                    continue
                files.setdefault(identity, (name, signature))
        except OSError:
            unavailable += 1
    records = []
    ambiguous = 0
    for identity, (name, signature) in files.items():
        if aliases[os.path.normcase(name)] != {identity}:
            ambiguous += 1
            continue
        unique_aliases = tuple(
            sorted(
                alias for alias in _aliases(name) if aliases[os.path.normcase(alias)] == {identity}
            )
        )
        records.append(Embedding(name, signature, unique_aliases))
    records.sort(key=lambda entry: (entry.name.casefold(), entry.name))
    status = f"{len(records)} embedding{'s' if len(records) != 1 else ''} available."
    if unsupported:
        subject = "filename contains" if len(unsupported) == 1 else "filenames contain"
        status += (
            f" {len(unsupported)} {subject} whitespace or prompt syntax and cannot be inserted."
        )
    if ambiguous:
        subject = "prompt name is" if ambiguous == 1 else "prompt names are"
        status += f" {ambiguous} conflicting {subject} hidden."
    if unavailable:
        subject = "file or folder" if unavailable == 1 else "files or folders"
        status += f" {unavailable} {subject} could not be read."
    return records, status


def selected_embedding(records, name):
    entry = next((entry for entry in records if entry.name == name), None)
    if entry is None:
        raise ValueError("Choose an installed embedding first.")
    try:
        unchanged = tuple(file_signature(entry.signature[0])) == entry.signature
    except OSError:
        unchanged = False
    if not unchanged:
        raise ValueError(
            "The embedding file changed or was removed. Refresh embeddings and select again."
        )
    return entry


def embedding_trigger_map(entries, cache_dirs=None):
    """Map standalone publisher triggers and unique file stems to installed entries."""
    from ruinedfooocus.features.models.resource_cards import resource_metadata

    triggers, conflicts = {}, set()
    for entry in entries:
        words = resource_metadata(entry.signature[0], cache_dirs).get("trainedWords", [])
        words = words if isinstance(words, list) else []
        stems = [
            PurePosixPath(alias).stem
            for alias in entry.aliases
            if PurePosixPath(alias).name == alias
            and Path(alias).suffix.lower() in EXTENSIONS
            and PurePosixPath(alias).stem in entry.aliases
        ]
        for word in [*words, *stems]:
            if not isinstance(word, str):
                continue
            word = word.strip()
            if not _WORD.fullmatch(word) or not _safe_name(word):
                continue
            key = word.casefold()
            if key in triggers and triggers[key][1] != entry:
                conflicts.add(key)
            else:
                triggers.setdefault(key, (word, entry))
    return {word: entry for key, (word, entry) in triggers.items() if key not in conflicts}


def embedding_trigger(entry, trigger_map):
    """Prefer the publisher's unambiguous trigger, then fall back to native syntax."""
    return next(
        (word for word, resource in (trigger_map or {}).items() if resource == entry), entry.token
    )


def _bare_embedding_matches(prompt, trigger_map):
    if not trigger_map or prompt.lstrip().startswith("{"):
        return
    lookup = {word.casefold(): entry for word, entry in trigger_map.items()}
    for match in _PROMPT_TOKENS.finditer(prompt):
        word = match["word"]
        if word is None or (entry := lookup.get(word.casefold())) is None:
            continue
        start, end = match.span()
        before, after = prompt[start - 1 : start] if start else "", prompt[end:]
        if before and not (before.isspace() or before in "(,;"):
            continue
        if after and not (
            after[0].isspace()
            or after[0] in "),;!?"
            or (after.startswith(".") and (len(after) == 1 or after[1].isspace()))
            or (before == "(" and re.match(_WEIGHT, after))
        ):
            continue
        yield start, end, entry


def expand_embedding_triggers(prompt, trigger_map):
    """Resolve bare triggers for native encoding, preserving the supplied prompt."""
    parts, signatures, offset = [], [], 0
    for start, end, entry in _bare_embedding_matches(prompt, trigger_map):
        parts.append(prompt[offset:start])
        if start and prompt[start - 1] in ",;":
            parts.append(" ")
        parts.append(entry.token)
        # The native reader handles commas only as a suffix on a whitespace-
        # delimited token. Separate other prose punctuation before it gets there.
        if end < len(prompt) and prompt[end] in ",;.!?":
            parts.append(" ")
        signatures.append(entry.signature)
        offset = end
    if not parts:
        return prompt, ()
    return "".join(parts) + prompt[offset:], tuple(signatures)


_NEGATIVE_NAME = re.compile(
    r"(?:lazyneg|lazyhand|lazydn|easynegative|bad[-_]?prompt|bad[-_]?hands?|"
    r"negative[-_]?hands?|neg[-_]?hands?|verybadimagenegative|negative|neg|unaesthetic)"
    r"(?:[-_]?v?\d+(?:\.\d+)?)?|ng_deepnegative_v1_75t",
    re.IGNORECASE,
)


def embedding_target(entry, metadata=None, default="Positive"):
    """Choose the prompt an activated embedding belongs in.

    Negative embeddings are recognised by name/trigger (``lazyneg``,
    ``lazyhand``, ``EasyNegative``, ``bad_prompt``) or a publisher description
    that recommends only the negative prompt. Otherwise ``default`` is kept.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    words = metadata.get("trainedWords") if isinstance(metadata.get("trainedWords"), list) else []
    names = [PurePosixPath(entry.name).stem, *[w for w in words if isinstance(w, str)]]
    explicit = metadata.get("prompt_target")
    if explicit in ("Positive", "Negative"):
        return explicit
    if any(_NEGATIVE_NAME.fullmatch(name.strip()) for name in names):
        return "Negative"
    description = re.sub(r"<[^>]+>", " ", str(metadata.get("description") or "")).casefold()
    if re.search(r"\bin (?:the )?negative\b", description) and "positive" not in description:
        return "Negative"
    return default


def embedding_spans(prompt, entry, trigger_map=None):
    """Find native references and bare triggers mapped to this installed entry."""
    names = "|".join(re.escape(name) for name in sorted(entry.aliases, key=len, reverse=True))
    if os.name == "nt":
        names = "(?i:" + names + ")"
    pattern = r"(?:^|(?<=[\s(]))embedding:(?:" + names + r")(?=,*($|[\s)])|" + _WEIGHT + ")"
    spans = [match.span() for match in re.finditer(pattern, prompt)]
    spans.extend(
        (start, end)
        for start, end, resource in _bare_embedding_matches(prompt, trigger_map)
        if resource == entry
    )
    return sorted(spans)


def insert_embedding(prompt, entry, trigger_map=None):
    """Prefix one recognized trigger while retaining exact existing prompt text."""
    if embedding_spans(prompt, entry, trigger_map):
        return prompt
    return embedding_trigger(entry, trigger_map) + (" " if prompt else "") + prompt


def remove_embedding(prompt, entry, trigger_map=None):
    """Remove only this embedding's exact tokens; retain surrounding prompt text.

    A complete single-token attention wrapper is removed with its reference.
    Whitespace, commas, other references and later user edits are untouched.
    """
    spans = []
    for start, end in embedding_spans(prompt, entry, trigger_map):
        if start and prompt[start - 1] == "(":
            weight = re.match(_WEIGHT, prompt[end:])
            if weight:
                start, end = start - 1, end + weight.end()
            elif prompt[end : end + 1] == ")":
                start, end = start - 1, end + 1
        spans.append((start, end))
    for start, end in reversed(spans):
        prompt = prompt[:start] + prompt[end:]
    return prompt


@lru_cache(maxsize=4)
def cached_trigger_map(roots, file_signatures, cache_directory):
    entries, _ = scan_embeddings(list(roots))
    return embedding_trigger_map(entries, cache_directory)


def _tree_signature(directory, suffixes):
    rows = []
    try:
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in suffixes:
                stat = path.stat()
                rows.append((str(path), stat.st_mtime_ns, stat.st_size))
    except OSError:
        pass
    return rows


def native_embedding_text(prompt, directories, cache_directory):
    """Expand bare embedding triggers (e.g. ``lazyneg``) to native references.

    The cache key covers every embedding file and its trigger metadata, so added,
    nested, removed or newly described embeddings apply without a restart.
    """
    roots = tuple(str(root) for root in directories if Path(root).is_dir())
    signatures = tuple(
        sorted(row for root in roots for row in _tree_signature(root, EXTENSIONS))
        + sorted(_tree_signature(cache_directory, {".json"}))
    )
    mapping = cached_trigger_map(roots, signatures, str(cache_directory))
    return expand_embedding_triggers(prompt, mapping)[0]
