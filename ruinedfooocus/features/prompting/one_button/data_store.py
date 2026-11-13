"""Read-only prompt sources and bounded, lazy immutable CSV data.

Raw rows retain order and duplicates. Header handling and request filtering belong
to the caller; no random selection or mutable caller result is cached here.
"""

from collections import OrderedDict
from collections.abc import Mapping
import csv
from pathlib import Path
import sys
from threading import RLock
from types import MappingProxyType
import warnings

from ruinedfooocus.core.paths import APP_ROOT, data_path

PACKAGE = Path(__file__).resolve().parent
CACHE_MAX_ENTRIES = 192
CACHE_MAX_BYTES = 96 * 1024 * 1024
_cache = OrderedDict()
_cache_bytes = 0
_lock = RLock()


def user_roots():
    """Writable data wins over legacy clone and former package user files."""
    return tuple(
        dict.fromkeys(
            (
                data_path("random_prompt", "userfiles"),
                APP_ROOT / "random_prompt/userfiles",
                PACKAGE / "userfiles",
            )
        )
    )


def user_file(relative):
    return next((root / relative for root in user_roots() if (root / relative).is_file()), None)


def source_file(directory, name):
    directory = Path(directory)
    parts = directory.parts
    if parts and parts[0] == "userfiles":
        return user_file(Path(*parts[1:]) / name)
    path = directory / name if directory.is_absolute() else PACKAGE / directory / name
    # tokenwriter was the same ordered dictionary with a different final newline.
    if path == PACKAGE / "csvfiles/special_lists/tokenwriter.csv":
        path = PACKAGE / "csvfiles/tokens.csv"
    return path if path.is_file() else None


def clear_cache():
    global _cache_bytes
    with _lock:
        _cache.clear()
        _cache_bytes = 0


def cache_info():
    with _lock:
        return {
            "entries": len(_cache),
            "bytes": _cache_bytes,
            "max_entries": CACHE_MAX_ENTRIES,
            "max_bytes": CACHE_MAX_BYTES,
        }


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino


def _deep_size(value, seen=None):
    """Account for index-owned strings even when its source rows are evicted.

    Shared objects are counted once per entry, conservatively again across
    entries. This is retained Python object size, not a process/RSS limit.
    """
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        if isinstance(value, MappingProxyType):
            size += sys.getsizeof(dict(value))
        size += sum(_deep_size(key, seen) + _deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(_deep_size(item, seen) for item in value)
    return size


def _cached(key, make):
    global _cache_bytes
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key][0]
        value, size = make()
        # Drop obsolete versions of this file/parser rather than retaining them.
        for old in list(_cache):
            if old[0:3] == key[0:3]:
                _cache_bytes -= _cache.pop(old)[1]
        if size <= CACHE_MAX_BYTES:
            while _cache and (
                len(_cache) >= CACHE_MAX_ENTRIES or _cache_bytes + size > CACHE_MAX_BYTES
            ):
                _cache_bytes -= _cache.popitem(last=False)[1][1]
            _cache[key] = value, size
            _cache_bytes += size
        return value


def read_rows(path, delimiter=";", strict=None):
    """UTF-8 (optional BOM) CSV, preserving physical rows and duplicates.

    Shipped legacy text uses permissive CSV quoting (e.g. card_names); user
    inputs are strict so malformed new files fail with a source/line diagnostic.
    """
    if path is None:
        return ()
    identity = _identity(path)
    if strict is None:
        strict = not Path(identity[0]).is_relative_to((PACKAGE / "csvfiles").resolve())

    def parse():
        with open(identity[0], newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file, delimiter=delimiter, strict=strict)
            try:
                rows = tuple(tuple(row) for row in reader)
            except csv.Error as error:
                raise ValueError(
                    f"{identity[0]}:{reader.line_num}: malformed CSV: {error}"
                ) from error
        size = _deep_size(rows)
        return rows, size

    return _cached(("rows", (delimiter, strict), *identity), parse)


def table_rows(path, columns, delimiter=";"):
    """Validate named tables without losing paired-column correspondence."""
    rows = tuple(
        (number, row)
        for number, row in enumerate(read_rows(path, delimiter), 1)
        if row and any(row)
    )
    if not rows:
        return (), ()
    header = rows[0][1]
    if any(column not in header for column in columns):
        raise ValueError(f"{path}: missing columns {columns!r}; got {header!r}")
    result = []
    for number, row in rows[1:]:
        if len(row) != len(header):
            warnings.warn(
                f"{path}:record {number}: expected {len(header)} columns, got {len(row)}; skipped",
                RuntimeWarning,
            )
            continue
        result.append(row)
    return header, tuple(result)


def artist_index(path):
    """Ordered projections share one parse/index for each artist source identity."""
    identity = _identity(path)

    def build():
        header, rows = table_rows(path, ("Artist", "Tags", "Medium", "Description"), ",")
        columns = {name: index for index, name in enumerate(header)}
        categories = {
            name: tuple(row[columns["Artist"]] for row in rows if row[index] == "1")
            for name, index in columns.items()
        }
        artists = {}
        for row in rows:
            artists.setdefault(row[columns["Artist"]], []).append(
                tuple(row[columns[name]] for name in ("Tags", "Medium", "Description"))
            )
        artists = {name: tuple(values) for name, values in artists.items()}
        descriptions = tuple(row[columns["Description"]] for row in rows)
        names = tuple(row[columns["Artist"]] for row in rows)
        tags = tuple(row[columns["Tags"]] for row in rows)
        value = MappingProxyType(categories), MappingProxyType(artists), descriptions, names, tags
        return value, _deep_size(value)

    return _cached(("artists", ",", *identity), build)
