"""Native progress registration must not retain metadata layout containers."""

import ast
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import unittest

import gradio as gr
from gradio.state_holder import SessionState


ROOT = Path(__file__).resolve().parents[1]
# Positive owners captured by the real native tracker, with output layout positions.
# These positions describe transport ownership; panel callback behavior has its own tests.
CASES = (
    ("image_editing/ui_control_model.py", "context", 3, (0,)),
    ("image_editing/ui_qwen_union.py", "context", 3, (0,)),
    ("image_editing/ui_guidance.py", "context_changed", 37, (0, 2, 31, 32, 33, 34, 35)),
    ("generation/ui_sdxl_refiner.py", "refiner_context", 2, (0,)),
    ("generation/ui_bernini_image.py", "bernini_context", 1, (0,)),
    ("image_editing/ui_native_workflows.py", "context", 8, (0, 1, 2, 3)),
    ("image_editing/ui_reference_adapters.py", "context", 5, (0,)),
    ("image_editing/ui_ordered_references.py", "context", 2, (0,)),
    ("image_editing/ui_model_patches.py", "context", 10, (0, 4, 5)),
    ("image_editing/ui_lora_hooks.py", "context", 9, (0, 1, 3, 4, 5, 7, 8)),
    ("image_processing/ui_native_vision.py", "context", 6, (0, 1, 2)),
    ("image_processing/ui_depth_utilities.py", "utility_context", 9, (0, 6, 7, 8)),
    ("image_processing/ui_wevisone.py", "context", 3, (0,)),
    ("image_editing/ui.py", "inpaint_context", 9, (1, 3, 5, 8)),
    ("generation/ui_native_sampling.py", "context", 7, (0,)),
    ("image_editing/ui_native_auxiliary.py", "context", 4, (0, 1)),
    ("generation/ui_layered.py", "context", 3, (0,)),
    ("chat/ui.py", "reference_controls", 6, (0,)),
    ("../ui/app.py", "aspect_ratios_changed", 7, (6,)),
)
ADDITIONAL_VIEW_CASES = (
    # Preserve the extra safe view scope included in the original patch; it was
    # not one of the nineteen callback owners identified by the runtime graph.
    ("image_processing/ui_image_composition.py", "context", 3, (0,)),
    ("gallery/ui_guided_results.py", "project", 14, ()),
    ("../ui/app.py", "switch_upscale_view", 6, (0,)),
)


def declarations(path, callback):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(
            keyword.arg == "outputs" for keyword in node.keywords
        ):
            function = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "fn"), None
            )
            if function is None and node.args:
                function = node.args[0]
            if (
                isinstance(function, ast.Name)
                and function.id == callback
                or (
                    isinstance(function, ast.Lambda)
                    and any(
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id == callback
                        for child in ast.walk(function)
                    )
                )
            ):
                calls.append(deepcopy(node))
        if isinstance(node, ast.FunctionDef) and node.name == callback and node.decorator_list:
            calls.extend(deepcopy(node.decorator_list))
    if not calls:
        raise AssertionError(f"Missing actual event declaration: {path}: {callback}")
    return tree, sorted(calls, key=lambda call: call.lineno)


class LayoutProgressTargetsTests(unittest.TestCase):
    def registered(self, case, binding=0):
        relative, callback, count, layouts = case
        path = ROOT / "ruinedfooocus/features" / relative
        tree, calls = declarations(path, callback)
        call = calls[binding]
        output = next(keyword.value for keyword in call.keywords if keyword.arg == "outputs")
        with gr.Blocks(analytics_enabled=False) as app:
            trigger = gr.Button()
            outputs = [
                gr.Group() if i in layouts else gr.Textbox(value="retained draft")
                for i in range(count)
            ]
            namespace = {"gr": gr, "trigger": trigger}
            cursor = iter(outputs)

            def bind(expression, value):
                if isinstance(expression, ast.Name):
                    namespace[expression.id] = value
                elif isinstance(expression, ast.Subscript) and isinstance(
                    expression.value, ast.Name
                ):
                    namespace.setdefault(expression.value.id, {})[
                        ast.literal_eval(expression.slice)
                    ] = value
                else:
                    raise AssertionError(ast.dump(expression))

            if isinstance(output, ast.List):
                expanded = count - sum(not isinstance(item, ast.Starred) for item in output.elts)
                for item in output.elts:
                    if isinstance(item, ast.Starred):
                        bind(item.value, [next(cursor) for _ in range(expanded)])
                    else:
                        bind(item, next(cursor))
            else:
                bind(output, outputs if count > 1 else outputs[0])
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "quiet" for target in node.targets
                ):
                    namespace["quiet"] = eval(
                        compile(ast.Expression(node.value), str(path), "eval"), namespace
                    )
            # Keep the actual production output/progress/options expressions. Isolate only
            # the unrelated triggers, inputs and callback implementation in this caller test.
            call.func = ast.Attribute(ast.Name("trigger", ast.Load()), "click", ast.Load())
            call.args = []
            call.keywords = [
                keyword
                for keyword in call.keywords
                if keyword.arg not in ("triggers", "fn", "inputs")
            ]
            namespace["publish"] = lambda: {
                component: gr.update(visible=True) for component in outputs
            }
            call.keywords.append(ast.keyword("fn", ast.Name("publish", ast.Load())))
            ast.fix_missing_locations(call)
            eval(compile(ast.Expression(call), str(path), "eval"), namespace)
        config = app.get_config_file()["dependencies"][-1]
        return app, config, outputs

    def test_actual_event_scopes_preserve_modes_outputs_and_native_data_publication(self):
        for case in (*CASES, *ADDITIONAL_VIEW_CASES):
            path = ROOT / "ruinedfooocus/features" / case[0]
            _, calls = declarations(path, case[1])
            for binding in range(len(calls)):
                with self.subTest(owner=case[:2], line=calls[binding].lineno):
                    self.check_registration(case, binding)

    def check_registration(self, case, binding):
        app, config, outputs = self.registered(case, binding)
        full = case[1] == "reference_controls"
        self.assertEqual(config["show_progress"], "full" if full else "hidden")
        self.assertEqual(config["outputs"], [component._id for component in outputs])
        self.assertEqual(
            config["show_progress_on"],
            [component._id for index, component in enumerate(outputs) if index not in case[3]]
            if full
            else [],
        )
        result = asyncio.run(app.process_api(config["id"], [], state=SessionState(app)))
        self.assertEqual(result["data"], [dict(__type__="update", visible=True)] * len(outputs))

    def test_native_completed_layout_status_is_not_replayed_by_unrelated_work(self):
        cases = []
        for case in (*CASES, *ADDITIONAL_VIEW_CASES):
            path = ROOT / "ruinedfooocus/features" / case[0]
            _, calls = declarations(path, case[1])
            for binding in range(len(calls)):
                _, config, outputs = self.registered(case, binding)
                cases.append(config | {"layout_ids": [outputs[index]._id for index in case[3]]})
        assets = Path(gr.__file__).parent / "templates/frontend/assets"
        result = subprocess.run(
            [
                "node",
                str(ROOT / "tests/test_layout_progress_targets.js"),
                str(assets),
                gr.__version__,
            ],
            input=json.dumps(cases),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        proof = json.loads(result.stdout)
        self.assertGreater(proof["positiveControlReads"], 0)
        self.assertEqual(proof["metadataLayoutReads"], 0)
        self.assertEqual(proof["metadataLayoutIds"], [])
        self.assertGreater(proof["fullLeafProgressIds"], 0)


if __name__ == "__main__":
    unittest.main()
