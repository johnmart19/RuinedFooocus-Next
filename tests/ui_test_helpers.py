"""Extract UI callbacks for CPU checks without importing the live app shell."""

import ast
from pathlib import Path
from ruinedfooocus.core.translation_manager import translate_ui

ROOT = Path(__file__).resolve().parents[1]


def function_from(path, name, namespace):
    namespace.setdefault("translate_ui", translate_ui)
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]
