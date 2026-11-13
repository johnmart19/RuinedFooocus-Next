"""Build/review a local native still-image facts candidate; never run a graph.

This explicit maintainer tool reuses the native audit's pinned readers and AST
registration helpers. Runtime code consumes the generated envelope only.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_native_model_coverage import (
    assignments,
    class_definitions,
    inherited,
    literal_sequence,
    module_registrations,
    read_checkout,
    read_official_templates,
    read_pin_sources,
)
from tools.native_capability_inventory import ROLE_CONTRACTS, recipe_graphs, recipe_inventory
from ruinedfooocus.features.models.native_snapshot import (
    ACCEPTED_COMFY_PIN,
    ACCEPTED_GGUF_PIN,
    PARSER_VERSION,
    SCHEMA_VERSION,
    content_digest,
    load_native_snapshot,
    native_snapshot_delta,
    parse_native_snapshot,
    refresh_native_snapshot,
)

REQUIRED_SCHEMAS = (
    "CheckpointLoaderSimple",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "UnetLoaderGGUF",
    "CLIPLoaderGGUF",
    "KSampler",
    "VAEDecode",
    "SaveImage",
)
PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
V3_TYPES = {
    "Int": "INT",
    "Float": "FLOAT",
    "String": "STRING",
    "Boolean": "BOOLEAN",
    "Combo": "COMBO",
}


def _literal(node, fallback=None):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return fallback


def _return(method):
    if method is None:
        return None
    values = [node.value for node in ast.walk(method) if isinstance(node, ast.Return)]
    return values[0] if len(values) == 1 else None


def _dict(node):
    return (
        {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if isinstance(node, ast.Dict)
        else {}
    )


def _literal_passthrough(classes, name):
    method, _ = inherited(classes, name, "execute", True)
    returned = _return(method)
    if (
        isinstance(returned, ast.Call)
        and ast.unparse(returned.func) in {"io.NodeOutput", "IO.NodeOutput"}
        and len(returned.args) == 1
        and isinstance(returned.args[0], ast.Name)
        and len(method.body) == 1
    ):
        return returned.args[0].id
    return None


def _static_choice(classes, name):
    """Attest the pinned two-input lazy Boolean selector's syntax, not its body."""
    method, _ = inherited(classes, name, "execute", True)
    if method is None or len(method.body) != 2:
        return None
    assignment, returned = method.body
    if (
        not isinstance(assignment, ast.Assign)
        or len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or not isinstance(assignment.value, ast.IfExp)
    ):
        return None
    choice = assignment.value
    if not all(isinstance(item, ast.Name) for item in (choice.test, choice.body, choice.orelse)):
        return None
    selected = assignment.targets[0].id
    if (
        not isinstance(returned, ast.Return)
        or not isinstance(returned.value, ast.Call)
        or ast.unparse(returned.value.func) not in {"io.NodeOutput", "IO.NodeOutput"}
        or len(returned.value.args) != 1
    ):
        return None
    result = returned.value.args[0]
    if (
        not isinstance(result, ast.IfExp)
        or ast.unparse(result.test) != f"{selected} is MISSING"
        or not isinstance(result.body, ast.Constant)
        or result.body.value is not None
        or not isinstance(result.orelse, ast.Name)
        or result.orelse.id != selected
    ):
        return None
    lazy, _ = inherited(classes, name, "check_lazy_status", True)
    if lazy is None:
        return None
    return {
        "control": choice.test.id,
        "on_true": choice.body.id,
        "on_false": choice.orelse.id,
        "source_sha256": content_digest(
            {
                "execute": ast.dump(method, include_attributes=False),
                "lazy": ast.dump(lazy, include_attributes=False),
            }
        ),
    }


def io_type_facts(source):
    """Read native V3 type names/widget traits and enum constants, not classes."""
    classes = class_definitions(ast.parse(source))
    types, constants = {}, {}
    for name, node in classes.items():
        for key, value in assignments(node.body).items():
            constants[f"io.{name}.{key}"] = value
            constants[f"IO.{name}.{key}"] = value
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or ast.unparse(decorator.func) != "comfytype":
                continue
            type_name = next(
                (_literal(item.value) for item in decorator.keywords if item.arg == "io_type"), None
            )
            if not isinstance(type_name, str):
                continue
            input_class = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.ClassDef) and item.name == "Input"
                ),
                None,
            )
            bases = [ast.unparse(base) for base in input_class.bases] if input_class else []
            options = {}
            constructor = (
                next(
                    (
                        item
                        for item in input_class.body
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
                    ),
                    None,
                )
                if input_class
                else None
            )
            if constructor:
                options = {
                    arg.arg: _literal(value)
                    for arg, value in zip(
                        constructor.args.args[-len(constructor.args.defaults) :],
                        constructor.args.defaults,
                    )
                }
            types[name] = {
                "type": type_name,
                "widget": type_name != "RESOLUTION_PREVIEW"
                and any(
                    base == "WidgetInput"
                    or base.endswith(("Combo.Input", "Int.Input", "Float.Input", "String.Input"))
                    for base in bases
                ),
                "optional": options.get("optional", False),
            }
    return types, constants


def node_schema(classes, name, mode, type_constants=None, io_types=None, constants=None):
    """Project V1/V3 port/widget declarations without importing native code."""
    method_name = "INPUT_TYPES" if mode == "v1" else "define_schema"
    method, owner = inherited(classes, name, method_name, True)
    returned = _return(method)
    inputs, outputs, unresolved = [], [], []
    constants, io_types = constants or {}, io_types or {}

    def literal(node, depth=0):
        value = _literal(node)
        if value is not None or node is None or depth > 8:
            return value
        if ast.unparse(node) in constants:
            return _literal(constants[ast.unparse(node)])
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner_name = name if node.value.id in {"cls", "s"} else node.value.id
            if owner_name in classes:
                return literal(inherited(classes, owner_name, node.attr)[0], depth + 1)
        return None

    def v3_port_type(node):
        if not isinstance(node, ast.Attribute):
            return None
        if isinstance(node.value, ast.Attribute):
            key = node.value.attr
            return io_types.get(key, {}).get("type", V3_TYPES.get(key, key.upper()))
        if (
            isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) in {"io.Custom", "IO.Custom"}
            and len(node.value.args) == 1
        ):
            return literal(node.value.args[0])
        return None

    if mode == "v1":
        returned_fields = _dict(returned)
        if not returned_fields:
            unresolved.append("dynamic-input-schema")
        for group in ("required", "optional"):
            declaration = returned_fields.get(group)
            fields = _dict(declaration)
            if declaration is not None and not fields and not isinstance(declaration, ast.Dict):
                unresolved.append(f"dynamic-{group}-inputs")
            for port_name, descriptor in fields.items():
                if not isinstance(descriptor, (ast.List, ast.Tuple)) or not descriptor.elts:
                    unresolved.append(f"dynamic-input:{port_name}")
                    continue
                type_node = descriptor.elts[0]
                port_type = _literal(type_node)
                if isinstance(type_node, ast.Attribute) and ast.unparse(type_node.value) == "IO":
                    port_type = (type_constants or {}).get(type_node.attr)
                dynamic = not isinstance(port_type, str)
                if dynamic:
                    port_type = "COMBO"
                options = (
                    {key: literal(value) for key, value in _dict(descriptor.elts[1]).items()}
                    if len(descriptor.elts) > 1
                    else {}
                )
                inputs.append(
                    {
                        "name": port_name,
                        "type": port_type,
                        "required": group == "required",
                        "widget": port_type in PRIMITIVES and not options.get("forceInput", False),
                        "options": options,
                        "dynamic_choices": dynamic,
                    }
                )
        output_node, _ = inherited(classes, name, "RETURN_TYPES")
        output_types = _literal(output_node)
        if isinstance(output_node, (ast.List, ast.Tuple)):
            output_types = [
                (type_constants or {}).get(item.attr)
                if isinstance(item, ast.Attribute) and ast.unparse(item.value) == "IO"
                else _literal(item)
                for item in output_node.elts
            ]
        if isinstance(output_types, (list, tuple)) and all(
            isinstance(item, str) for item in output_types
        ):
            outputs = list(output_types)
        else:
            unresolved.append("dynamic-output-schema")
    else:
        if not isinstance(returned, ast.Call) or ast.unparse(returned.func) not in {
            "io.Schema",
            "IO.Schema",
        }:
            unresolved.append("dynamic-v3-schema")
        else:
            keywords = {item.arg: item.value for item in returned.keywords}
            declared_inputs = keywords.get("inputs")
            if not isinstance(declared_inputs, (ast.List, ast.Tuple)):
                unresolved.append("dynamic-v3-inputs")
            else:
                for port in declared_inputs.elts:
                    if not isinstance(port, ast.Call):
                        unresolved.append("dynamic-v3-input")
                        continue
                    port_keywords = {item.arg: item.value for item in port.keywords}
                    port_type = v3_port_type(port.func)
                    port_name = literal(port.args[0] if port.args else port_keywords.get("id"))
                    if (
                        not isinstance(port_name, str)
                        or not isinstance(port.func, ast.Attribute)
                        or port.func.attr != "Input"
                        or port_type is None
                    ):
                        unresolved.append("dynamic-v3-input")
                        continue
                    options = {item.arg: literal(item.value) for item in port.keywords}
                    io_name = (
                        port.func.value.attr if isinstance(port.func.value, ast.Attribute) else None
                    )
                    traits = io_types.get(io_name, {})
                    if io_name == "MultiType":
                        allowed = port_keywords.get(
                            "types", port.args[1] if len(port.args) > 1 else None
                        )
                        if isinstance(allowed, (ast.List, ast.Tuple)):
                            type_names = [
                                io_types.get(item.attr, {}).get("type")
                                if isinstance(item, ast.Attribute)
                                else None
                                for item in allowed.elts
                            ]
                            if type_names and all(type_names):
                                port_type = ",".join(type_names)
                            else:
                                unresolved.append("dynamic-v3-multitype")
                        else:
                            unresolved.append("dynamic-v3-multitype")
                    elif port_type.startswith(
                        ("COMFY_DYNAMIC", "COMFY_AUTOGROW", "COMFY_MATCHTYPE")
                    ):
                        unresolved.append("dynamic-v3-control")
                    inputs.append(
                        {
                            "name": port_name,
                            "type": port_type,
                            "required": not options.get("optional", traits.get("optional", False)),
                            "widget": (port_type in PRIMITIVES or traits.get("widget", False))
                            and not options.get("force_input", False),
                            "options": options,
                            "dynamic_choices": "options" in options and options["options"] is None,
                        }
                    )
            declared_outputs = keywords.get("outputs")
            if isinstance(declared_outputs, (ast.List, ast.Tuple)):
                for port in declared_outputs.elts:
                    port_type = v3_port_type(port.func) if isinstance(port, ast.Call) else None
                    if (
                        port_type is not None
                        and isinstance(port.func, ast.Attribute)
                        and port.func.attr == "Output"
                    ):
                        outputs.append(port_type)
                    else:
                        unresolved.append("dynamic-v3-output")
            else:
                unresolved.append("dynamic-v3-outputs")
    returned_types, _ = inherited(classes, name, "RETURN_TYPES")
    # Bind the whole declaration, including dynamic expressions, so a source
    # change cannot hide behind an unchanged partial/static port projection.
    semantic = {
        "input_schema": ast.dump(method, include_attributes=False) if method else None,
        "return_types": ast.dump(returned_types, include_attributes=False)
        if returned_types
        else None,
        "resolved_inputs": inputs,
        "resolved_outputs": outputs,
        "literal_passthrough": _literal_passthrough(classes, name),
        "static_choice": _static_choice(classes, name),
    }
    return {
        "registration": mode,
        "class": name,
        "owner": owner,
        "inputs": inputs,
        "outputs": outputs,
        "unresolved": sorted(set(unresolved)),
        "contract_sha256": content_digest(semantic),
        "literal_passthrough": _literal_passthrough(classes, name),
        "static_choice": _static_choice(classes, name),
    }


def sampler_choice_facts(source):
    """Resolve only the pinned public native enum declarations without imports."""
    tree = ast.parse(source)
    constants, classes = assignments(tree.body), class_definitions(tree)
    if "KSampler" not in classes:
        raise ValueError("Native KSampler choice declarations are unavailable.")
    public = assignments(classes["KSampler"].body)
    watched = {}

    def sequence(node, pending=()):
        if isinstance(node, ast.Name):
            if node.id in pending or node.id not in constants:
                raise ValueError("Native sampling choices require a literal declaration.")
            watched[node.id] = constants[node.id]
            return sequence(constants[node.id], (*pending, node.id))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return sequence(node.left, pending) + sequence(node.right, pending)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "list"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
        ):
            name = node.args[0].id
            mapping = constants.get(name)
            if not isinstance(mapping, ast.Dict):
                raise ValueError("Native scheduler mapping needs a static review.")
            watched[name] = mapping
            return literal_sequence(ast.List(elts=mapping.keys, ctx=ast.Load()), {})
        if isinstance(node, (ast.List, ast.Tuple)):
            choices = []
            for item in node.elts:
                choices.extend(
                    sequence(item.value, pending)
                    if isinstance(item, ast.Starred)
                    else literal_sequence(ast.List(elts=[item], ctx=ast.Load()), {})
                )
            return choices
        raise ValueError("Native sampling choices are dynamic; keep the last good snapshot.")

    values = {
        name: sequence(public.get(attribute))
        for name, attribute in (("samplers", "SAMPLERS"), ("schedulers", "SCHEDULERS"))
    }
    for choices in values.values():
        if not choices or len(choices) != len(set(choices)):
            raise ValueError("Native sampling choices must be nonempty unique literals.")
    # Reject import-time mutation/reassignment instead of trusting the initial
    # list. Native function bodies are not evaluated and remain outside this
    # narrow declaration projection.
    declarations = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ],
        type_ignores=[],
    )
    nodes = tuple(ast.walk(declarations))
    for name in watched:
        writes = [
            node
            for node in nodes
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
        ]
        if len(writes) != 1:
            raise ValueError("Native sampling declaration mutation needs explicit review.")
        for node in nodes:
            target = (
                node.func.value
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                else node.value
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                else None
            )
            if isinstance(target, ast.Name) and target.id == name:
                raise ValueError("Native sampling declaration mutation needs explicit review.")
    class_declarations = ast.Module(
        body=[
            node
            for node in classes["KSampler"].body
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ],
        type_ignores=[],
    )
    for attribute in ("SAMPLERS", "SCHEDULERS"):
        if (
            sum(
                isinstance(node, ast.Name)
                and node.id == attribute
                and isinstance(node.ctx, ast.Store)
                for node in ast.walk(class_declarations)
            )
            != 1
        ):
            raise ValueError("Native public sampling alias needs explicit review.")
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "KSampler"
            and node.attr == attribute
            and isinstance(node.ctx, ast.Store)
            for node in nodes
        ):
            raise ValueError("Native public sampling alias mutation needs explicit review.")
        for node in nodes:
            target = (
                node.func.value
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                else node.value
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del))
                else None
            )
            if target is not None and ast.unparse(target) == f"KSampler.{attribute}":
                raise ValueError("Native public sampling alias mutation needs explicit review.")
    contract = {
        "bindings": {
            name: ast.dump(public[attribute], include_attributes=False)
            for name, attribute in (("samplers", "SAMPLERS"), ("schedulers", "SCHEDULERS"))
        },
        "declarations": {
            name: ast.dump(node, include_attributes=False) for name, node in sorted(watched.items())
        },
    }
    return {
        **values,
        "source": {
            "path": "comfy/samplers.py",
            "bindings": {"samplers": "KSampler.SAMPLERS", "schedulers": "KSampler.SCHEDULERS"},
            "contract_sha256": content_digest(contract),
        },
    }


def _widget_values(node, schema):
    values = node.get("widgets_values", [])
    if isinstance(values, dict):
        return values, []
    if not isinstance(values, list) or schema is None:
        return {}, ["widget-schema-unavailable"] if values else []
    result, offset = {}, 0
    for port in schema["inputs"]:
        if not port["widget"]:
            continue
        if offset >= len(values):
            if not port["required"] and port["options"].get("default") is not None:
                result[port["name"]] = port["options"]["default"]
                continue
            return result, ["widget-values-incomplete"]
        result[port["name"]] = values[offset]
        offset += 1
        if port["options"].get("control_after_generate"):
            if offset >= len(values) or values[offset] not in {
                "fixed",
                "increment",
                "decrement",
                "randomize",
            }:
                return result, ["seed-control-widget-unresolved"]
            offset += 1
    # Extra UI-only widgets are retained as unresolved rather than shifting the
    # interpretation of later values or treating them as native defaults.
    return result, (["extra-widget-values"] if offset != len(values) else [])


def _links(graph):
    result = {}
    for link in graph.get("links", []):
        if isinstance(link, list) and len(link) >= 5:
            key, origin, slot, target, target_slot = link[:5]
        elif isinstance(link, dict):
            key, origin, slot, target, target_slot = (
                link.get(name)
                for name in ("id", "origin_id", "origin_slot", "target_id", "target_slot")
            )
        else:
            continue
        result[key] = (origin, slot, target, target_slot)
    return result


def connected_graph_facts(document, schemas, *, open_image_outputs=False):
    """Traverse data links to still-image outputs; never evaluate/execute nodes.

    Subgraph output/input slots resolve back to each calling wrapper. Muted,
    bypassed, recursive or externally controlled branches remain unresolved.
    """
    graph_rows = list(recipe_graphs(document))
    definitions = {
        graph["id"]: (location, graph) for location, graph in graph_rows if "id" in graph
    }
    active, unresolved, visited = {}, set(), set()

    def external_input(graph, slot, caller):
        if caller is None or type(slot) is not int or not 0 <= slot < len(graph.get("inputs", [])):
            return "unresolved", None
        parent, parent_location, wrapper, parent_caller = caller
        name = graph["inputs"][slot]["name"]
        parent_port = next(
            (port for port in wrapper.get("inputs", []) if port.get("name") == name), {}
        )
        if parent_port.get("link") is not None:
            return "link", (parent, parent_location, parent_port["link"], parent_caller)
        overrides = proxy_overrides(wrapper)
        if ("-1", name) in overrides:
            return "value", overrides[("-1", name)]
        return "default", None

    def proxy_overrides(wrapper):
        values = wrapper.get("widgets_values", [])
        if not values:
            return {}
        proxies = wrapper.get("properties", {}).get("proxyWidgets", [])
        if (
            not isinstance(values, list)
            or len(values) != len(proxies)
            or any(not isinstance(proxy, list) or len(proxy) != 2 for proxy in proxies)
        ):
            unresolved.add("subgraph-widget-overrides-unresolved")
            return {}
        return {(str(proxy[0]), proxy[1]): value for proxy, value in zip(proxies, values)}

    def literal_link(graph, location, link_id, caller, seen=None):
        # Only native primitives whose entire execute body returns one input are
        # followed. This reads a literal; arbitrary node functions never run.
        seen = set() if seen is None else seen
        marker = location, link_id
        if marker in seen or len(seen) > 64:
            return False, None
        seen.add(marker)
        link = _links(graph).get(link_id)
        if link is None:
            return False, None
        if link[0] == graph.get("inputNode", {}).get("id", -10):
            kind, value = external_input(graph, link[1], caller)
            if kind == "value":
                return True, value
            if kind == "link":
                return literal_link(*value, seen)
            return False, None
        node = next((row for row in graph["nodes"] if row.get("id") == link[0]), {})
        schema = schemas.get(node.get("type"), {})
        name = schema.get("literal_passthrough")
        if name is None or link[1] != 0 or node.get("mode", 0) != 0:
            return False, None
        widgets, _ = _widget_values(node, schema)
        if caller:
            overrides = proxy_overrides(caller[2])
            if (str(node["id"]), name) in overrides:
                widgets[name] = overrides[(str(node["id"]), name)]
        port = next((row for row in node.get("inputs", []) if row.get("name") == name), {})
        if port.get("link") is not None:
            reference = _links(graph).get(port["link"])
            if reference and reference[0] == graph.get("inputNode", {}).get("id", -10):
                kind, value = external_input(graph, reference[1], caller)
                if kind == "value":
                    return True, value
                if kind == "default":
                    return (name in widgets), widgets.get(name)
            return literal_link(graph, location, port["link"], caller, seen)
        return (name in widgets), widgets.get(name)

    def node_widgets(graph, location, node, caller):
        widgets, problems = _widget_values(node, schemas.get(node.get("type")))
        if caller:
            overrides = proxy_overrides(caller[2])
            widgets.update(
                {
                    name: value
                    for (node_id, name), value in overrides.items()
                    if node_id == str(node["id"])
                }
            )
        for port in node.get("inputs", []):
            if port.get("link") is None or port.get("name") not in widgets:
                continue
            link = _links(graph).get(port["link"])
            if link and link[0] == graph.get("inputNode", {}).get("id", -10):
                kind, value = external_input(graph, link[1], caller)
                if kind == "value":
                    widgets[port["name"]] = value
                    continue
                if kind == "default":
                    continue
            resolved, value = literal_link(graph, location, port["link"], caller)
            if resolved:
                widgets[port["name"]] = value
            else:
                unresolved.add(f"linked-widget-value-unresolved:{node['type']}:{port['name']}")
        return widgets, problems

    def visit(graph, location, node_id, output_slot=0, caller=None, depth=0):
        if depth > 64:
            unresolved.add("recursive-or-deep-subgraph")
            return
        marker = (location, node_id, output_slot)
        if marker in visited:
            return
        visited.add(marker)
        node = next((item for item in graph["nodes"] if item.get("id") == node_id), None)
        if node is None:
            unresolved.add("missing-linked-node")
            return
        if node.get("mode", 0) != 0:
            unresolved.add("muted-or-bypassed-active-node")
            return
        if node.get("type") in definitions:
            _, nested = definitions[node["type"]]
            outputs = nested.get("outputs", [])
            if type(output_slot) is not int or not 0 <= output_slot < len(outputs):
                unresolved.add("subgraph-output-unresolved")
                return
            links = outputs[output_slot].get("linkIds", [])
            if not links:
                unresolved.add("subgraph-output-link-unresolved")
            nested_location = f"{location}/call:{node_id}"
            nested_caller = (graph, location, node, caller)
            for link_id in links:
                follow(nested, nested_location, link_id, nested_caller, depth + 1)
            proxy_overrides(node)
            return
        active[f"{location}:{node_id}"] = (location, graph, node, caller)
        choice = schemas.get(node["type"], {}).get("static_choice")
        selected_port = None
        if choice:
            controls, _ = node_widgets(graph, location, node, caller)
            selected = controls.get(choice["control"])
            if type(selected) is bool:
                selected_port = choice["on_true"] if selected else choice["on_false"]
            else:
                unresolved.add(f"dynamic-branch-selection-unresolved:{node['type']}")
        for port in node.get("inputs", []):
            if (
                choice
                and port.get("name") in {choice["on_true"], choice["on_false"]}
                and port.get("name") != selected_port
            ):
                continue
            if port.get("link") is not None:
                follow(graph, location, port["link"], caller, depth + 1)

    def follow(graph, location, link_id, caller, depth):
        link = _links(graph).get(link_id)
        if link is None:
            unresolved.add("missing-graph-link")
        else:
            if link[0] == graph.get("inputNode", {}).get("id", -10):
                kind, value = external_input(graph, link[1], caller)
                if kind == "link":
                    follow(*value, depth + 1)
                    return
                if kind == "value":
                    return
                target_node = next((row for row in graph["nodes"] if row.get("id") == link[2]), {})
                ports = target_node.get("inputs", [])
                port_name = (
                    ports[link[3]].get("name")
                    if type(link[3]) is int and 0 <= link[3] < len(ports)
                    else None
                )
                widgets, _ = _widget_values(target_node, schemas.get(target_node.get("type")))
                if kind == "default" and port_name in widgets:
                    return
                unresolved.add("subgraph-input-or-widget-override-unresolved")
                return
            source_node = next((item for item in graph["nodes"] if item.get("id") == link[0]), {})
            target_node = next((item for item in graph["nodes"] if item.get("id") == link[2]), {})
            source_schema = schemas.get(source_node.get("type"))
            target_schema = schemas.get(target_node.get("type"))
            if source_schema is not None and target_schema is not None:
                target_inputs = target_node.get("inputs", [])
                target_name = (
                    target_inputs[link[3]].get("name")
                    if type(link[3]) is int and 0 <= link[3] < len(target_inputs)
                    else None
                )
                target_port = next(
                    (port for port in target_schema["inputs"] if port["name"] == target_name), None
                )
                source_outputs = source_schema["outputs"]
                if (
                    type(link[1]) is not int
                    or not 0 <= link[1] < len(source_outputs)
                    or target_port is None
                ):
                    unresolved.add("native-link-ports-unresolved")
                elif source_outputs[link[1]].startswith("COMFY_") or target_port["type"].startswith(
                    "COMFY_"
                ):
                    unresolved.add("dynamic-native-link-types")
                elif not set(source_outputs[link[1]].split(",")).intersection(
                    target_port["type"].split(",")
                ) and not {
                    source_outputs[link[1]],
                    target_port["type"],
                } & {"*", "ANY"}:
                    unresolved.add("native-link-type-mismatch")
            visit(graph, location, link[0], link[1], caller, depth)

    def visit_sinks(graph, location, caller=None, ancestry=()):
        count = 0
        for node in graph["nodes"]:
            if node.get("mode", 0) != 0:
                continue
            if node.get("type") in {"SaveImage", "SaveImageAdvanced", "PreviewImage"}:
                visit(graph, location, node["id"], caller=caller)
                count += 1
            elif node.get("type") in definitions:
                if node["type"] in ancestry or len(ancestry) > 32:
                    unresolved.add("recursive-or-deep-subgraph")
                    continue
                nested = definitions[node["type"]][1]
                nested_count = visit_sinks(
                    nested,
                    f"{location}/call:{node['id']}",
                    (graph, location, node, caller),
                    (*ancestry, node["type"]),
                )
                if nested_count:
                    proxy_overrides(node)
                    count += nested_count
        return count

    sink_count = visit_sinks(document, "root")
    if open_image_outputs:
        for node in document.get("nodes", []):
            if node.get("type") in definitions and node.get("mode", 0) == 0:
                for slot, port in enumerate(node.get("outputs", [])):
                    if port.get("type") == "IMAGE":
                        visit(document, "root", node["id"], slot)
                        sink_count += 1
    if not sink_count:
        unresolved.add("no-connected-root-still-image-output")
    selected, optional = [], []
    for location, graph in graph_rows:
        for node in graph["nodes"]:
            if node.get("type") not in definitions:
                for model in node.get("properties", {}).get("models", []):
                    optional.append(
                        {
                            "graph": location,
                            "definition": graph.get("id"),
                            "node_instance": node.get("id"),
                            "node_id": node.get("type"),
                            **model,
                            "activity": "inactive-or-alternative",
                        }
                    )
    values = []
    for location, graph, node, caller in active.values():
        definition = graph.get("id")
        schema = schemas.get(node["type"])
        widgets, problems = node_widgets(graph, location, node, caller)
        if schema is None:
            unresolved.add(f"unregistered-native-node:{node['type']}")
        # Unknown non-model frontend display helpers do not provide defaults.
        if problems:
            unresolved.update(f"{node['type']}:{problem}" for problem in problems)
        if schema and schema["unresolved"]:
            unresolved.update(f"{node['type']}:{problem}" for problem in schema["unresolved"])
        values.append(
            {
                "node_id": node["type"],
                "graph": location,
                "node_instance": node["id"],
                "widgets": widgets,
            }
        )
        for model in node.get("properties", {}).get("models", []):
            selected_names = {
                value.replace("\\", "/").rsplit("/", 1)[-1]
                for value in widgets.values()
                if isinstance(value, str)
            }
            if model["name"] in selected_names:
                selected.append(
                    {
                        "graph": location,
                        "definition": definition,
                        "node_instance": node["id"],
                        "node_id": node["type"],
                        **model,
                        "activity": "active-required",
                    }
                )
            elif not widgets or problems:
                unresolved.add(f"{node['type']}:model-selection-unresolved")
    active_annotation_keys = {
        (row["definition"], row["node_id"], row["node_instance"], row["name"], row["url"])
        for row in selected
    }
    optional = [
        row
        for row in optional
        if (row["definition"], row["node_id"], row["node_instance"], row["name"], row["url"])
        not in active_annotation_keys
    ]
    return {
        "active_nodes": values,
        "required_models": selected,
        "optional_models": optional,
        "unresolved": sorted(unresolved),
    }


def _url_identity(url):
    parts = urlsplit(url)
    path = parts.path.split("/")
    if parts.hostname == "huggingface.co" and len(path) >= 6 and path[3] in {"resolve", "blob"}:
        return (parts.hostname, *path[1:3], *path[5:])
    return parts.hostname, parts.path


def _model_annotations(root, rows):
    from ruinedfooocus.features.models.model_registry import registry_document

    registry = registry_document(root / "ruinedfooocus/resources/model_metadata")
    artifacts = {row["id"]: row for row in registry["artifacts"]}
    sources = {}
    for row in rows:
        for identity in row.get("available_artifact_ids", []):
            artifact = artifacts.get(identity, {})
            url = artifact.get("artifact", {}).get("pinned_url")
            if url:
                sources.setdefault(_url_identity(url), []).append(
                    {
                        "native_class": row["id"],
                        "family_id": row["component_family"],
                        "behavioral_variant": artifact.get("behavioral_variant"),
                    }
                )
    return sources


def recipe_facts(recipe_id, raw, schemas, model_annotations, *, source_kind):
    document = json.loads(raw)
    graph = connected_graph_facts(
        document, schemas, open_image_outputs=source_kind == "pinned-comfy-blueprint"
    )
    unresolved = list(graph["unresolved"])
    models = []
    families, variants, classes = set(), set(), set()
    for model in graph["required_models"] + graph["optional_models"]:
        role = model["directory"]
        if role not in ROLE_CONTRACTS:
            unresolved.append(f"unreviewed-template-model-role:{role}")
        annotations = (
            model_annotations.get(_url_identity(model["url"]), [])
            if role in {"checkpoints", "diffusion_models", "unet"}
            else []
        )
        for row in annotations if model["activity"] == "active-required" else []:
            families.add(row["family_id"])
            classes.add(row["native_class"])
            variants.add(row["behavioral_variant"])
        models.append(
            {
                "recipe_id": recipe_id,
                "role": role,
                "name": model["name"],
                "url": model["url"],
                "hashes": {
                    key: model[key] for key in ("hash", "hash_type", "sha256") if key in model
                },
                "graph": model["graph"],
                "node_id": model["node_id"],
                "node_instance": model["node_instance"],
                "activity": model["activity"],
            }
        )
    if len(families) != 1:
        unresolved.append(
            "primary-family-annotation-unresolved" if not families else "multiple-primary-families"
        )
    node_ids = sorted({row["node_id"] for row in graph["active_nodes"]})
    if (
        sum(
            row["node_id"]
            in {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
            for row in graph["active_nodes"]
        )
        > 1
    ):
        unresolved.append("multiple-sampling-stages")
    has_image_source = "LoadImage" in node_ids or any("VAEEncode" in name for name in node_ids)
    has_empty_latent = any(name.startswith("Empty") and "Latent" in name for name in node_ids)
    operation = "edit" if has_image_source else "generate" if has_empty_latent else "unresolved"
    sampling, sizes = {}, set()
    for row in graph["active_nodes"]:
        widgets = row["widgets"]
        if row["node_id"].startswith("Empty") and "Latent" in row["node_id"]:
            width, height = widgets.get("width"), widgets.get("height")
            if type(width) is int and type(height) is int and width > 0 and height > 0:
                sizes.add((width, height))
        for key in ("steps", "cfg", "sampler_name", "scheduler", "denoise", "guidance"):
            if key in widgets:
                sampling_key = "custom_steps" if key == "steps" else key
                if sampling_key in sampling and sampling[sampling_key] != widgets[key]:
                    unresolved.append(f"conflicting-sampling:{key}")
                sampling[sampling_key] = widgets[key]
        if row["node_id"] == "KSamplerSelect" and "sampler_name" in widgets:
            sampling["sampler_name"] = widgets["sampler_name"]
        if row["node_id"] == "KSamplerAdvanced":
            unresolved.append("advanced-sampler-stage-needs-overlay")
    if len(sizes) != 1:
        unresolved.append("example-canvas-unresolved" if not sizes else "multiple-example-canvases")
    required = sorted({row["role"] for row in models if row["activity"] == "active-required"})
    optional = sorted({row["role"] for row in models if row["activity"] != "active-required"})
    return {
        "recipe_id": recipe_id,
        "source": {
            "kind": "native-workflow-example",
            "bundle": source_kind,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "family_ids": sorted(families),
        "native_classes": sorted(classes),
        "behavioral_variant": next(iter(variants)) if len(variants) == 1 else None,
        "operation": operation,
        "sampling": sampling,
        "default_size": list(next(iter(sizes))) if len(sizes) == 1 else None,
        "node_ids": node_ids,
        "required_roles": required,
        "optional_roles": optional,
        "models": models,
        "unresolved": sorted(set(unresolved)),
    }


def derive_snapshot_facts(
    report, sources, blueprints, templates, catalogue_rows, model_annotations
):
    """Pure projection interface for fixture tests and explicit source rebuilds."""
    schemas = {}
    recipes = {}
    type_classes = class_definitions(ast.parse(sources.get("native/node_typing.py", b"")))
    type_constants = (
        {key: _literal(value) for key, value in assignments(type_classes["IO"].body).items()}
        if "IO" in type_classes
        else {}
    )
    io_types, io_constants = io_type_facts(sources.get("native/io.py", b""))
    for source, raw in sorted(sources.items()):
        if source == "native/nodes.py" or "/comfy_extras/" in source or source == "gguf/nodes.py":
            tree = ast.parse(raw)
            mode, classes, registered = module_registrations(tree, source)
            constants = {**assignments(tree.body), **io_constants}
            for node_id, node, _ in registered:
                schemas[node_id] = {
                    **node_schema(classes, node.id, mode, type_constants, io_types, constants),
                    "source": source,
                }
    if "native/samplers.py" in sources:
        choices = sampler_choice_facts(sources["native/samplers.py"])
        schema = schemas["KSampler"]
        for port in schema["inputs"]:
            name = {"sampler_name": "samplers", "scheduler": "schedulers"}.get(port["name"])
            if name is not None:
                port["options"]["choices"] = choices[name]
                port["dynamic_choices"] = False
        schema["sampling_choices_source"] = choices["source"]
        schema["contract_sha256"] = content_digest(
            {"schema": schema["contract_sha256"], "sampling_choices": choices}
        )
    for bundle, documents in (
        ("pinned-comfy-blueprint", blueprints),
        ("installed-official-templates", templates),
    ):
        inventory = recipe_inventory(
            documents, report["native_nodes"], official=bundle == "installed-official-templates"
        )
        for row in inventory["recipes"]:
            if row["scope"] == "still-image-or-utility":
                key = bundle + ":" + row["id"]
                recipes[key] = recipe_facts(
                    key, documents[row["id"]], schemas, model_annotations, source_kind=bundle
                )
    required_node_ids = set(REQUIRED_SCHEMAS)
    for row in recipes.values():
        required_node_ids.update(row["node_ids"])
    schemas = {key: row for key, row in schemas.items() if key in required_node_ids}
    classes = {}
    for key in ("native/supported_models.py", "native/supported_models_base.py"):
        classes.update(class_definitions(ast.parse(sources[key])))
    latent_classes = class_definitions(ast.parse(sources["native/latent_formats.py"]))
    configs = {row["class"]: row for row in report["native_config_classes"]}
    families, aliases = {}, {}
    for row in catalogue_rows:
        name, family = row["id"], row["component_family"]
        config = configs.get(name)
        if config is None:
            continue
        latent_node, _ = inherited(classes, name, "latent_format")
        latent_name = ast.unparse(latent_node).split(".")[-1] if latent_node is not None else None
        latent = {"class": latent_name}
        for attribute in (
            "latent_channels",
            "latent_dimensions",
            "spacial_downscale_ratio",
            "temporal_downscale_ratio",
        ):
            value, _ = inherited(latent_classes, latent_name, attribute)
            latent[attribute] = _literal(value)
        family_row = families.setdefault(
            family,
            {
                "family_id": family,
                "native_classes": [],
                "configurations": [],
                "component_roles": [],
                "template_models": [],
                "examples": [],
                "example_recipe": None,
                "unresolved": [],
            },
        )
        family_row["native_classes"].append(name)
        family_row["configurations"].append(
            {
                "native_class": name,
                "unet_match": config["unet_match"],
                "latent": latent,
                "runtime_factories": config["runtime_factories"],
                "contract_sha256": config["contract_sha256"],
                "source": {"path": "comfy/supported_models.py", "line": config["line"]},
                "component_prefixes": {
                    key: _literal(inherited(classes, name, key)[0])
                    for key in ("vae_key_prefix", "text_encoder_key_prefix", "clip_vision_prefix")
                },
            }
        )
        components = row.get("component_roles") or {}
        roles = list(components.get("text", []))
        if components.get("vae") and components["vae"] != "pixel_space":
            roles.append(components["vae"])
        family_row["component_roles"] = [
            {
                "role": role,
                "provenance": "existing-rf-component-overlay",
                "activity": "selected-loader-dependent",
            }
            for role in roles
        ]
        for alias in (name, family, row.get("native_family")):
            if alias:
                aliases[alias] = family
    for family, row in sorted(families.items()):
        examples = [recipe for recipe in recipes.values() if recipe["family_ids"] == [family]]
        row["native_classes"].sort()
        row["configurations"].sort(key=lambda item: item["native_class"])
        row["examples"] = sorted(examples, key=lambda item: item["recipe_id"])
        row["template_models"] = [model for recipe in row["examples"] for model in recipe["models"]]
        candidates = [
            recipe
            for recipe in examples
            if recipe["operation"] == "generate"
            and not recipe["unresolved"]
            and recipe["default_size"]
            and {"custom_steps", "cfg", "sampler_name", "scheduler"} <= recipe["sampling"].keys()
        ]
        semantics = {
            content_digest(
                {
                    key: recipe[key]
                    for key in (
                        "native_classes",
                        "behavioral_variant",
                        "sampling",
                        "default_size",
                        "required_roles",
                    )
                }
            )
            for recipe in candidates
        }
        if len(semantics) == 1:
            row["example_recipe"] = sorted(candidates, key=lambda item: item["recipe_id"])[0]
        elif len(semantics) > 1:
            row["unresolved"].append("conflicting-connected-generation-examples")
        if not examples:
            row["unresolved"].append("no-exact-family-template-annotation")
    return {"families": families, "aliases": aliases, "schemas": schemas, "recipes": recipes}


def build_candidate(root, native_root, gguf_root, templates_root=None):
    report = read_checkout(root, native_root, gguf_root, templates_root)
    templates, template_identity = read_official_templates(templates_root)
    if not template_identity.get("recipe_assets_match_manifest"):
        raise ValueError(
            "Official recipe manifest is unavailable or differs; preserve the last good snapshot."
        )
    inputs, sources = {}, {}
    for prefix, checkout, pin in (
        ("native", native_root, ACCEPTED_COMFY_PIN),
        ("gguf", gguf_root, ACCEPTED_GGUF_PIN),
    ):
        paths = {
            row["source_key"].split("/", 1)[1]
            for row in report["native_node_modules"]
            if row["source_key"].startswith(prefix + "/")
        }
        if prefix == "native":
            paths.update(
                (
                    "comfy/supported_models.py",
                    "comfy/supported_models_base.py",
                    "comfy/model_detection.py",
                    "comfy/latent_formats.py",
                    "comfy/sd.py",
                    "comfy/samplers.py",
                    "comfy/comfy_types/node_typing.py",
                    "comfy_api/latest/_io.py",
                )
            )
        read = read_pin_sources(checkout, prefix, pin, sorted(paths))
        for path, raw in read.items():
            name = path.removeprefix("comfy/") if path.startswith("comfy/") else path
            if path == "comfy/comfy_types/node_typing.py":
                name = "node_typing.py"
            elif path == "comfy_api/latest/_io.py":
                name = "io.py"
            sources[f"{prefix}/{name}"] = raw
            inputs[f"{prefix}/{path}"] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    blueprint_paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(native_root),
            "ls-tree",
            "-r",
            "--name-only",
            ACCEPTED_COMFY_PIN,
            "--",
            "blueprints",
        ],
        text=True,
        timeout=20,
    ).splitlines()
    blueprints = read_pin_sources(
        native_root,
        "ComfyUI blueprints",
        ACCEPTED_COMFY_PIN,
        [path for path in blueprint_paths if path.endswith(".json")],
    )
    resource_root = root / "ruinedfooocus/resources/model_metadata"
    catalogue_rows = []
    for name in ("configurations-image.json", "configurations-utility.json"):
        catalogue_rows.extend(
            json.loads((resource_root / "native_coverage" / name).read_bytes())["rows"]
        )
    annotations = _model_annotations(root, catalogue_rows)
    facts = derive_snapshot_facts(
        report, sources, blueprints, templates, catalogue_rows, annotations
    )
    if missing := set(REQUIRED_SCHEMAS) - facts["schemas"].keys():
        raise ValueError(f"Required native schemas unavailable: {sorted(missing)}")
    overlay_paths = [
        resource_root / "index.json",
        *sorted((resource_root / "artifacts").glob("*.json")),
        *sorted((resource_root / "native_coverage").glob("configurations-*.json")),
    ]
    overlay_paths.extend(
        (
            root / "ruinedfooocus/features/models/image_components.py",
            root / "tools/update_native_snapshot.py",
            root / "ruinedfooocus/features/models/native_snapshot.py",
        )
    )
    inputs.update(
        {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for path in overlay_paths
        }
    )
    source_manifest = {
        "native_pin": ACCEPTED_COMFY_PIN,
        "gguf_pin": ACCEPTED_GGUF_PIN,
        "sources": dict(sorted(inputs.items())),
        "blueprints": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(blueprints.items())
        },
        "template_identity": {
            **template_identity,
            "observed_json_manifest_sha256": content_digest(
                {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(templates.items())}
            ),
            "observed_json_sha256": {
                name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(templates.items())
            },
        },
        "scope": "Static source facts and connected still-image examples; package artwork, model bytes, device/runtime execution and publisher recommendations are unqualified.",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "source_manifest": source_manifest,
        "source_manifest_sha256": content_digest(source_manifest),
        "snapshot_sha256": content_digest(facts),
        "required_contracts": {
            "families": sorted(facts["families"]),
            "schemas": sorted(REQUIRED_SCHEMAS),
            "native_classes": sorted(
                {name for family in facts["families"].values() for name in family["native_classes"]}
            ),
        },
    }
    manifest["snapshot_id"] = content_digest(manifest)
    document = {"manifest": manifest, "facts": facts}
    parse_native_snapshot(document)
    return document


def write_candidate(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_candidate_paths(candidate, *, compare=None, promote=None):
    if any(
        path is not None and path.resolve() == candidate.resolve() for path in (compare, promote)
    ):
        raise ValueError(
            "Candidate path must differ from comparison/promotion paths; preserve the last good envelope."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--gguf-root", type=Path, required=True)
    parser.add_argument("--templates-root", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--promote", type=Path)
    parser.add_argument("--reviewed-semantics", action="store_true")
    args = parser.parse_args()
    validate_candidate_paths(args.candidate, compare=args.compare, promote=args.promote)
    document = build_candidate(ROOT, args.native_root, args.gguf_root, args.templates_root)
    write_candidate(args.candidate, document)
    summary = {
        "candidate": str(args.candidate),
        "snapshot_id": document["manifest"]["snapshot_id"],
        "families": len(document["facts"]["families"]),
        "schemas": len(document["facts"]["schemas"]),
        "recipes": len(document["facts"]["recipes"]),
    }
    if args.compare:
        previous = load_native_snapshot(args.compare)
        if previous is None:
            raise ValueError("Comparison snapshot is unavailable; candidate remains unpromoted.")
        summary["delta"] = native_snapshot_delta(previous, parse_native_snapshot(document))
    if args.promote:
        result = refresh_native_snapshot(
            args.candidate, cache_path=args.promote, reviewed_semantics=args.reviewed_semantics
        )
        summary.update(promoted=result.promoted, error=result.error)
        if not result.promoted:
            print(json.dumps(summary))
            return 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
