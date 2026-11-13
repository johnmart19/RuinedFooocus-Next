"""Validate static UI extraction and shipped catalogues without loading the app.

Run ``venv/Scripts/python tools/check_translations.py``. --extract adds new English
messages; it never changes existing IDs or overwrites translated strings.
"""

import argparse
import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ruinedfooocus.core.translation_manager import TranslationManager, read_json


def ui_files():
    files = set((ROOT / "ruinedfooocus/ui").glob("*.py"))
    files.update((ROOT / "ruinedfooocus/features").glob("*/ui*.py"))
    return sorted(files)


def localized_files():
    """Non-UI modules with server text or static properties marked for translation."""
    ui = set(ui_files())
    return sorted(
        path
        for path in (ROOT / "ruinedfooocus").rglob("*.py")
        if "vendor" not in path.parts
        and path not in ui
        and re.search(r"\b(?:localize|translate_ui)\s*\(", path.read_text(encoding="utf-8-sig"))
    )


def localize_nodes(tree):
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in ("localize", "translate_ui")
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            yield call.args[0]


def schema_nodes(tree):
    """Presentation columns of declarative controls; keys/defaults stay opaque."""
    for node in ast.walk(tree):
        rows, columns = None, ()
        # Category labels/reasons reach the request-local UI localize() through
        # variables; extract their presentation fields without importing caches.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Category"
        ):
            for index in (1, 4):
                if (
                    index < len(node.args)
                    and isinstance(node.args[index], ast.Constant)
                    and isinstance(node.args[index].value, str)
                ):
                    yield node.args[index]
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("STARTUP_CONTROLS", "ADVANCED_CONTROLS")
            for target in node.targets
        ):
            rows = node.value
            columns = (1, 3) if node.targets[0].id == "STARTUP_CONTROLS" else (1, 2)
        elif (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and isinstance(node.iter, (ast.Tuple, ast.List))
        ):
            rows = node.iter
            columns = tuple(
                i
                for i, name in enumerate(node.target.elts)
                if isinstance(name, ast.Name) and name.id in ("label", "info", "help_text")
            )
        if isinstance(rows, (ast.Tuple, ast.List)):
            for row in rows.elts:
                if isinstance(row, (ast.Tuple, ast.List)):
                    for index in columns:
                        if (
                            index < len(row.elts)
                            and isinstance(row.elts[index], ast.Constant)
                            and isinstance(row.elts[index].value, str)
                        ):
                            yield row.elts[index]


def adjustment_schema():
    """Read only the new adjustment panels' declared presentation enums."""
    constants, labels = {}, []
    for alias, filename, names in (
        ("transform", "image_geometry_tools.py", ("INTERPOLATIONS",)),
        ("mask", "image_mask_tools.py", ("OPERATIONS", "COMPOSITE_OPERATIONS", "CHANNELS")),
        (
            "color",
            "image_color_tools.py",
            (
                "OPERATIONS",
                "BLEND_MODES",
                "TRANSFER_METHODS",
                "DITHERS",
                "MORPHOLOGY_OPERATIONS",
                "TEXT_POSITIONS",
                "TEXT_ALIGNMENTS",
            ),
        ),
    ):
        path = ROOT / "ruinedfooocus/features/image_processing" / filename
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id in names:
                values = ast.literal_eval(node.value)
                if not isinstance(values, (tuple, list)) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise ValueError(f"Invalid adjustment choice schema: {path}:{node.lineno}")
                constants[(alias, target.id)] = values
            elif target.id == "LABELS" and isinstance(node.value, ast.Dict):
                labels.extend((path, value) for value in node.value.values)
    return constants, labels


def adjustment_nodes(node, constants):
    """Resolve the finite field helper, its simple label loops and displayed choices."""
    if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
        if isinstance(node.iter, (ast.Tuple, ast.List)) and all(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in node.iter.elts
        ):
            for statement in node.body:
                call = statement.value if isinstance(statement, ast.Expr) else None
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "field"
                    and len(call.args) > 1
                ):
                    label = call.args[1]
                    if (
                        isinstance(label, ast.BinOp)
                        and isinstance(label.op, ast.Add)
                        and isinstance(label.left, ast.Constant)
                        and isinstance(label.left.value, str)
                        and isinstance(label.right, ast.Name)
                        and label.right.id == node.target.id
                    ):
                        for value in node.iter.elts:
                            yield ast.copy_location(
                                ast.Constant(label.left.value + value.value), label
                            )
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
        and len(node.args) > 1
    ):
        return
    if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
        yield node.args[1]
    for keyword in node.keywords:
        value = keyword.value
        if (
            keyword.arg == "info"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            yield value
        if keyword.arg != "choices":
            continue
        options = ()
        if isinstance(value, (ast.Tuple, ast.List)):
            options = ast.literal_eval(value)
        elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            options = constants.get((value.value.id, value.attr), ())
        for option in options:
            text = (
                {
                    "histogram": "Histogram (256-bin)",
                    "sdr-to-hdr": "SDR to HDR",
                    "crop-to-mask": "Crop to selection",
                }[option]
                if option in ("histogram", "sdr-to-hdr", "crop-to-mask")
                else option.replace("-", " ").replace("_", " ").capitalize()
            )
            yield ast.copy_location(ast.Constant(text), value)


def static_nodes(tree, *, adjustment_constants=None):
    """Presentation strings only; prompt text and protocol values stay opaque."""
    for call in ast.walk(tree):
        if adjustment_constants is not None:
            yield from adjustment_nodes(call, adjustment_constants)
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Name) and call.func.id in ("t", "translate_ui", "localize"):
            if (
                call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                yield call.args[0]
        if (
            not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "gr"
        ):
            continue
        for kw in call.keywords:
            if kw.arg in ("label", "info", "placeholder", "description", "title") or (
                kw.arg == "value" and call.func.attr in ("Button", "Markdown")
            ):
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    yield kw.value
        if (
            call.func.attr
            in ("Button", "Tab", "TabItem", "Accordion", "Markdown", "Info", "Warning", "Error")
            and call.args
        ):
            if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                yield call.args[0]


def messages():
    schema_files = [
        ROOT / "ruinedfooocus/core/settings_schema.py",
        ROOT / "ruinedfooocus/backends/comfy/image_compute.py",
    ]
    cache_schema = ROOT / "ruinedfooocus/core/cache_maintenance.py"
    if cache_schema.is_file():
        schema_files.append(cache_schema)
    adjustment_constants, adjustment_labels = adjustment_schema()
    for path in ui_files() + schema_files:
        feature = path.parent.name if path.parent.parent.name == "features" else "shell"
        if path.name in ("ui_settings.py", "ui_hardware.py") or path in schema_files:
            feature = "settings"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        constants = (
            adjustment_constants
            if path.name in ("ui_image_adjustments.py", "ui_color_operations.py")
            else None
        )
        for node in [*static_nodes(tree, adjustment_constants=constants), *schema_nodes(tree)]:
            text = node.value
            if text.strip() and re.search(r"[a-zA-Z]", text):
                yield feature, text, str(path.relative_to(ROOT)), node.lineno
    for path, node in adjustment_labels:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield "image_processing", node.value, str(path.relative_to(ROOT)), node.lineno
    for path in localized_files():
        feature = path.parent.name if path.parent.parent.name == "features" else "shell"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in localize_nodes(tree):
            if node.value.strip() and re.search(r"[a-zA-Z]", node.value):
                yield feature, node.value, str(path.relative_to(ROOT)), node.lineno


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    root = ROOT / "ruinedfooocus/resources/translations"
    aliases = read_json(root / "legacy-keys.json")
    english = read_json(root / "en.json")
    missing = []
    for feature, text, path, line in messages():
        if text not in aliases:
            key = feature + "." + (re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "text")
            if key in english and english[key].strip() != text.strip():
                raise ValueError(f"ID collision for {text!r}; choose a stable explicit ID")
            aliases[text] = key
            english.setdefault(key, text)
            missing.append(f"{path}:{line}: {text}")
    if args.extract:
        for name, data in (("en.json", english), ("legacy-keys.json", aliases)):
            (root / name).write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    elif missing:
        raise SystemExit("Unregistered UI strings:\n" + "\n".join(missing))
    manager = TranslationManager(legacy_root=root / "no-custom-catalogues")
    if manager.errors:
        raise SystemExit("\n".join(manager.errors))
    for locale, values in manager.translations.items():
        translated = sum(value != english[key] for key, value in values.items() if key in english)
        print(
            f"{locale}: {len(values)} entries, {translated} differ from English; {len(english) - len(values)} missing entries use English"
        )


if __name__ == "__main__":
    main()
