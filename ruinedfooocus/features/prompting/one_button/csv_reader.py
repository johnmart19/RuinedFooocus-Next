"""One Button's logical CSV API; selection stays outside the parsed-row cache."""

import os
from pathlib import Path
import random
import warnings

from ruinedfooocus.core.paths import data_path
from .data_store import PACKAGE, artist_index, read_rows, source_file, table_rows, user_file


def userfiles_path():
    """Return the writable destination without creating it during reads."""
    return str(data_path("random_prompt", "userfiles")) + os.sep


def random_read_from_csv(filename):
    rows = tuple(
        row
        for row in read_rows(source_file("./csvfiles/", filename + ".csv"), ",")
        if row and any(row)
    )
    if not rows:
        raise ValueError(f"Prompt dataset {filename!r} has no choices")
    return random.choice([row[0] for row in rows])


def add_from_csv(completeprompt, csvfilename, addcomma, prefix, suffix):
    addition = prefix + " " + random_read_from_csv(csvfilename) + " " + suffix
    return (", " if addcomma == 1 else " ").join([completeprompt, addition])


def csv_to_list(
    csvfilename,
    antilist=None,
    directory="./csvfiles/",
    lowerandstrip=0,
    delimiter=";",
    listoflistmode=False,
    skipheader=False,
    gender="all",
    insanitylevel=-1,
):
    directory = str(directory).replace("\\", "/")
    normalized = Path(directory).as_posix()
    packaged = normalized in ("csvfiles", "csvfiles/special_lists", "csvfiles/templates")
    selected = csvfilename
    replacement = user_file(csvfilename + "_replace.csv") if packaged else None
    if replacement is not None:
        selected += "_replace"
    elif packaged and insanitylevel > 0:
        # Keep the original RNG call order, including its draw after light selection.
        if insanitylevel < 4 and source_file(directory, csvfilename + "_light.csv"):
            selected += "_light"
        if insanitylevel < 7 and random.randint(0, 13) < 12 and selected == csvfilename:
            if source_file(directory, csvfilename + "_medium.csv"):
                selected += "_medium"

    blocked = set(antilist or ())

    def project(path):
        rows = read_rows(path, delimiter)
        if skipheader:
            rows = rows[1:]
        if listoflistmode:
            return [list(row) for row in rows if row and any(row)]
        result = []
        for number, row in enumerate(rows, 2 if skipheader else 1):
            if not row or not any(row):
                continue
            if gender != "all":
                if len(row) < 2:
                    warnings.warn(
                        f"{path}:{number}: missing gender column; skipped", RuntimeWarning
                    )
                    continue
                if row[1] not in (gender, "genderless", "both"):
                    continue
            if row[0].lower().strip() not in blocked:
                result.append(row[0].lower().strip() if lowerandstrip == 1 else row[0])
        return result

    csv_path = replacement or source_file(directory, selected + ".csv")
    txt_path = (
        replacement.with_suffix(".txt")
        if replacement
        else source_file(directory, selected + ".txt")
    )
    if txt_path is not None and not txt_path.is_file():
        txt_path = None
    result = project(csv_path)
    if txt_path is not None:
        txt = project(txt_path)
        result = txt if listoflistmode else result + txt
    if normalized in ("csvfiles", "csvfiles/special_lists"):
        addon = user_file(selected + "_addon.csv")
        if addon is not None:
            extra = project(addon)
            # Legacy list-of-lists addon shape is intentionally retained.
            result.append(extra) if listoflistmode else result.extend(extra)
    if listoflistmode:
        return result
    seen = set()
    unique = []
    for value in result:
        if value.lower() not in seen:
            seen.add(value.lower())
            unique.append(value)
    return unique


def _artists(csvfilename="artists_and_category"):
    path = source_file("./csvfiles/", csvfilename + ".csv")
    if path is None:
        raise ValueError(f"Artist dataset {csvfilename!r} is missing")
    return artist_index(path)


def artist_category_by_category_csv_to_list(csvfilename, artist):
    rows = _artists(csvfilename)[1].get(artist, ())
    return tuple([row[column] for row in rows] for column in range(3))


def artist_category_csv_to_list(csvfilename, category):
    return list(_artists(csvfilename)[0][category])


def artist_descriptions_csv_to_list(csvfilename):
    return list(_artists(csvfilename)[2])


def load_all_artist_and_category():
    index = _artists()
    return list(index[3]), list(index[4])


def load_config_csv(suffix=""):
    name = "config" + ("_" + suffix if suffix else "") + ".csv"
    path = user_file(name) or source_file("./csvfiles/config/", "default_" + name)
    if path is None:
        raise ValueError(f"Prompt configuration {name!r} is missing")
    result = []
    for number, row in enumerate(read_rows(path)[1:], 2):
        if not row or not any(row) or any(value.startswith("#") for value in row):
            continue
        if len(row) != 2:
            warnings.warn(
                f"{path}:{number}: expected 2 config columns, got {len(row)}; skipped",
                RuntimeWarning,
            )
            continue
        result.append(list(row))
    return result


def load_negative_list():
    path = user_file("negativewords_replace.csv") or source_file(
        "./csvfiles/special_lists/", "negativewords.csv"
    )
    primers, negatives = [], []
    for source in (path, user_file("negativewords_addon.csv")):
        if source is None:
            continue
        header, rows = table_rows(source, ("primer", "negative"))
        if not header:
            continue
        primer, negative = header.index("primer"), header.index("negative")
        primers.extend(row[primer] for row in rows)
        negatives.extend(row[negative] for row in rows)
    return primers, negatives


def export_token_dictionary(output_path, *, sort_and_dedupe=False):
    """Write an explicit UTF-8 copy, never the shared bundled source or its alias."""
    output_path = Path(output_path).resolve()
    if output_path.is_relative_to(PACKAGE.resolve()):
        raise ValueError("Token exports must be outside the read-only One Button package")
    rows = read_rows(source_file("./csvfiles/", "tokens.csv"))
    values = [row[0] for row in rows]
    if sort_and_dedupe:
        values = sorted(
            {value.lower() for value in values if not all(c == value[0] for c in value)}
        )
    with output_path.open("w", encoding="utf-8", newline="") as file:
        for value in values:
            file.write(value + "\n")
    return output_path


def sort_and_dedupe_csv_file(output_path):
    return export_token_dictionary(output_path, sort_and_dedupe=True)
