"""Enumerate UI event bindings/API routes without importing application or native runtimes.

Rows are source sites, not a runtime action count: a loop/helper can bind multiple
controls. Defaults are recorded explicitly rather than invented ownership evidence.
"""

import argparse
import ast
import hashlib
import json
from fnmatch import fnmatchcase
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EVENTS = frozenset(
    {
        "click",
        "change",
        "input",
        "submit",
        "select",
        "navigate",
        "paginate",
        "load",
        "tick",
        "release",
        "clear",
        "upload",
        "stop",
        "play",
        "pause",
        "stream",
        "then",
        "success",
        "failure",
        "key_up",
        "focus",
        "blur",
        "like",
        "unload",
    }
)
ROUTES = frozenset({"get", "post", "put", "patch", "delete", "websocket", "api_route", "tool"})
EVENT_OPTIONS = frozenset(
    {"fn", "inputs", "outputs", "js", "queue", "api_visibility", "cancels", "show_progress"}
)


def expression(value):
    return ast.unparse(value) if value is not None else None


def _scope(node, parents):
    node = parents.get(node)
    while node is not None and not isinstance(
        node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        node = parents.get(node)
    return node


def _qualified(node, parents):
    names = []
    while node is not None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        node = parents.get(node)
    return ".".join(reversed(names))


def _callback(symbol, node, parents, definitions):
    """Resolve the lexical definition, rather than another nested callback of the same name."""
    if not isinstance(symbol, ast.Name):
        return symbol if isinstance(symbol, ast.Lambda) else None
    scope = _scope(node, parents)
    while scope is not None:
        found = definitions.get((id(scope), symbol.id))
        if found is not None:
            return found
        scope = _scope(scope, parents)
    return None


def _group(relative, callback, groups):
    for group in groups:
        if any(fnmatchcase(relative, pattern) for pattern in group["sources"]) and re.search(
            group.get("callbacks", ".*"), callback or ""
        ):
            return group
    return None


def _options(node, parents, assignments):
    values, spreads = {}, []
    for keyword in node.keywords:
        if keyword.arg is not None:
            values[keyword.arg] = keyword.value
            continue
        spread = keyword.value
        spreads.append(expression(spread))
        if isinstance(spread, ast.Name):
            scope = _scope(node, parents)
            while scope is not None:
                candidates = assignments.get((id(scope), spread.id), [])
                preceding = [value for line, value in candidates if line < node.lineno]
                if preceding:
                    spread = preceding[-1]
                    break
                scope = _scope(scope, parents)
        if isinstance(spread, ast.Call) and expression(spread.func) == "dict":
            values.update({key.arg: key.value for key in spread.keywords if key.arg})
        elif isinstance(spread, ast.Dict):
            values.update(
                {
                    key.value: value
                    for key, value in zip(spread.keys, spread.values)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
            )
    return values, spreads


def inventory(root=ROOT, *, groups=None):
    root = Path(root)
    contract_path = ROOT / "tools/action_contracts.json"
    groups = (
        groups
        if groups is not None
        else json.loads(contract_path.read_text(encoding="utf-8"))["groups"]
    )
    rows = []
    excluded = []
    for path in sorted((root / "ruinedfooocus").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if "/vendor/" in relative:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        definitions = {(id(_scope(node, parents)), node.name): node for node in functions}
        assignments = {}
        for assignment in ast.walk(tree):
            if isinstance(assignment, ast.Assign):
                for target in assignment.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(
                            (id(_scope(assignment, parents)), target.id), []
                        ).append((assignment.lineno, assignment.value))
        for candidates in assignments.values():
            candidates.sort(key=lambda candidate: candidate[0])
        decorators = {
            id(decorator): function
            for function in functions
            for decorator in function.decorator_list
        }
        gradio = any(
            isinstance(node, ast.Import)
            and any(alias.name == "gradio" for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "gradio"
            for node in ast.walk(tree)
        )
        fastapi = any(
            isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "fastapi" for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "fastapi"
            for node in ast.walk(tree)
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = expression(node.func)
            values, spreads = _options(node, parents, assignments)
            attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
            imperative_route = (
                fastapi
                and attribute == "add_api_route"
                and (len(node.args) >= 2 or "endpoint" in values)
                and (bool(node.args) or "path" in values)
            )
            route = attribute in ROUTES and id(node) in decorators or imperative_route
            api_registration = gradio and name == "gr.api"
            candidate = attribute in EVENTS or name in {"gr.on", "trigger", "localized_output"}
            helper = name == "localized_output"
            # ThreadPoolExecutor.submit, session.select, dictionary.clear,
            # platform.release and model.load are ordinary method calls.
            ui = (
                candidate
                and gradio
                and (
                    name in {"gr.on", "trigger"}
                    or bool(EVENT_OPTIONS.intersection(values))
                    or attribute == "load"
                    and name in {"block.load", "main_tabs.load"}
                )
            )
            if not (route or ui or helper or api_registration):
                if candidate:
                    excluded.append({"source": relative, "line": node.lineno, "call": name})
                continue
            callback = decorators.get(id(node))
            callback_value = (
                values.get("endpoint", node.args[1] if len(node.args) > 1 else None)
                if imperative_route
                else values.get("fn", node.args[0] if node.args else None)
            )
            symbol = callback.name if callback is not None else expression(callback_value)
            declaration = callback or _callback(callback_value, node, parents, definitions)
            qualified = _qualified(declaration, parents) if declaration is not None else symbol
            if isinstance(declaration, ast.Lambda):
                qualified = f"{qualified}.<lambda>: {symbol}"
            group = _group(relative, qualified, groups)
            loops, ancestor = [], parents.get(node)
            while ancestor is not None:
                if isinstance(ancestor, (ast.For, ast.AsyncFor)):
                    loops.append(f"{expression(ancestor.target)} in {expression(ancestor.iter)}")
                ancestor = parents.get(ancestor)
            source = (
                expression(declaration) or symbol or "client JavaScript / empty load registration"
            )
            inputs = expression(values.get("inputs"))
            outputs = expression(values.get("outputs"))
            if helper:
                inputs = inputs or (expression(node.args[1]) if len(node.args) > 1 else None)
                outputs = outputs or (expression(node.args[2]) if len(node.args) > 2 else None)
            start = f"{node.lineno}:{node.col_offset}"
            end = f"{node.end_lineno}:{node.end_col_offset}"
            rows.append(
                {
                    "id": f"{relative}:{start}-{end}",
                    "kind": "route"
                    if route
                    else "gradio-api"
                    if api_registration
                    else "localized-output-helper"
                    if helper
                    else "event-site",
                    "owner": relative,
                    "group": group["id"] if group else None,
                    "binding": name,
                    "callback": symbol,
                    "callback_qualified": qualified,
                    "callback_line": declaration.lineno if declaration else None,
                    "callback_resolution": "lexical"
                    if declaration is not None
                    else "external/factory or client JavaScript; follow declaring source",
                    "callback_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "inputs": inputs,
                    "outputs": outputs,
                    "triggers": expression(values.get("triggers")),
                    "option_spreads": spreads,
                    "route": expression(values.get("path", node.args[0] if node.args else None))
                    if route
                    else None,
                    "api_name": expression(values.get("api_name")),
                    "http_methods": expression(values.get("methods"))
                    if route and "methods" in values
                    else attribute.upper()
                    if route and attribute in {"get", "post", "put", "patch", "delete"}
                    else None,
                    "enclosing_loops": list(reversed(loops)),
                    "work": group["work"] if group else "UNREVIEWED",
                    "resource_owner": group["resource_owner"] if group else "UNREVIEWED",
                    "queue": expression(values.get("queue"))
                    or (
                        "HTTP request dispatcher"
                        if route
                        else "Gradio API default"
                        if api_registration
                        else "Gradio default"
                    ),
                    "concurrency_id": expression(values.get("concurrency_id"))
                    or (
                        "Route admission in resource_owner" if route else "Gradio callback default"
                    ),
                    "concurrency_limit": expression(values.get("concurrency_limit"))
                    or "Gradio default",
                    "trigger_mode": expression(values.get("trigger_mode"))
                    or "Gradio event default",
                    "progress": expression(values.get("show_progress"))
                    or ("HTTP response/stream" if route else "Gradio default"),
                    "progress_targets": expression(values.get("show_progress_on")) or outputs,
                    "cancels": expression(values.get("cancels")),
                    "persistence": group["persistence"] if group else "UNREVIEWED",
                    "cancellation_contract": group["cancellation"] if group else "UNREVIEWED",
                    "acceptance": group["acceptance"] if group else "UNREVIEWED",
                    "scope": "Source registration/helper site, including conditional and loop bindings; runtime expansion and browser delivery require live evidence.",
                }
            )
    identities = {
        row["owner"]: hashlib.sha256((root / row["owner"]).read_bytes()).hexdigest() for row in rows
    }
    return {
        "schema_version": 3,
        "gradio_version": "6.31.0",
        "scope": "Non-vendored Python Gradio event/API registrations, localized-output helpers and decorated or FastAPI add_api_route HTTP/MCP routes. Ordinary same-named methods are excluded. Counts are source sites, never runtime actions.",
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "groups": groups,
        "source_sha256": identities,
        "actions": sorted(rows, key=lambda row: row["id"]),
        "unassigned": [row["id"] for row in rows if row["group"] is None],
        "excluded_method_calls": excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strict", action="store_true", help="Fail when a source site has no reviewed group"
    )
    args = parser.parse_args()
    document = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{len(document['actions'])} source registrations/helpers/routes; {len(document['unassigned'])} unassigned: {args.output}"
    )
    return int(args.strict and bool(document["unassigned"]))


if __name__ == "__main__":
    raise SystemExit(main())
