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


def raise_refusal(result):
    """Turn a fenced prompt action's refusal result back into the gr.Error it reports.

    Fenced actions return {"updates": None, "error": message} so the browser step
    can show the message; other results pass through unchanged.
    """
    if isinstance(result, dict) and result.get("updates") is None and result.get("error"):
        import gradio as gr

        raise gr.Error(result["error"])
    return result
