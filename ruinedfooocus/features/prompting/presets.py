"""Rewrite instructions; inference belongs to the shared local chat runtime."""

from ruinedfooocus.core.paths import content_roots


def rewrite_presets():
    presets = {}
    for folder in content_roots("prompt_presets", "llamas"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.txt")):
            try:
                with path.open(encoding="utf-8") as stream:
                    name = stream.readline().strip()
                if name:
                    presets[path.relative_to(folder).as_posix()] = (name, str(path))
            except (OSError, UnicodeError) as error:
                print(f"Cannot read prompt preset {path}: {error}")
    return sorted(presets.values(), key=lambda item: (item[0].casefold(), item[1]))
