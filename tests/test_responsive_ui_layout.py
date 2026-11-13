"""Static layout contracts and native Gradio structure; no rendered acceptance."""

import ast
from pathlib import Path
import re
import subprocess
import unittest

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "ruinedfooocus/ui/assets/ui.css").read_text(encoding="utf-8")
SHELL_CSS = (ROOT / "ruinedfooocus/ui/assets/shell.css").read_text(encoding="utf-8")


def split_selectors(value):
    depth, start = 0, 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            yield value[start:index].strip()
            start = index + 1
    yield value[start:].strip()


def rules(selector, css=CSS):
    return [
        declarations
        for selectors, declarations in re.findall(
            r"([^{}]+)\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        )
        if selector in split_selectors(selectors)
    ]


def layout_node(node, component_id):
    if node["id"] == component_id:
        return node
    return next(
        (
            found
            for child in node.get("children", [])
            if (found := layout_node(child, component_id))
        ),
        None,
    )


class ResponsiveUILayoutTests(unittest.TestCase):
    def test_slider_header_wraps_before_value_controls_squeeze_scaled_help(self):
        selector = ':is(#app-tabs, #image-stage) .block:has(input[type="range"]) .head'
        header = rules(selector)[0]
        self.assertIn("display: flex", header)
        self.assertIn("flex-wrap: wrap", header)
        label = rules(f"{selector} > label")[0]
        basis = re.search(r"flex:\s*1\s+1\s+([\d.]+)rem", label)
        self.assertIsNotNone(basis)
        for available, root_font, value_width, gap, wraps in (
            (300, 32, 170, 16, True),
            (740, 32, 170, 16, False),
            (300, 16, 85, 8, False),
        ):
            with self.subTest(available=available, root_font=root_font):
                self.assertEqual(float(basis[1]) * root_font + value_width + gap > available, wraps)
        value = rules(f"{selector} > .tab-like-container")[0]
        self.assertIn("flex: 0 1 auto", value)
        self.assertIn("margin-inline-start: auto", value)
        for declarations in (label, value):
            self.assertIn("max-width: 100%", declarations)
            self.assertNotRegex(declarations, r"font-size|overflow:\s*hidden")

    def test_chat_fit_uses_available_space_and_scales_with_the_user_font(self):
        native_assets = Path(gr.__file__).resolve().parent / "templates/frontend/assets"
        result = subprocess.run(
            ["node", "tests/test_chat_splitter.js", str(native_assets)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_chat_setup_action_reserves_space_after_the_native_heading_and_description(self):
        tree = ast.parse((ROOT / "ruinedfooocus/features/chat/ui.py").read_text(encoding="utf-8"))
        conversation = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and any(
                    key.arg == "elem_id"
                    and isinstance(key.value, ast.Constant)
                    and key.value.value == "chat-conversation"
                    for key in item.context_expr.keywords
                )
                for item in node.items
            )
        )
        header = [
            node
            for node in conversation.body
            if isinstance(node, (ast.Expr, ast.Assign))
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) in ("gr.HTML", "gr.Button")
        ][:3]
        native = ast.fix_missing_locations(
            ast.Module(
                body=[ast.With(items=conversation.items, body=header, type_comment=None)],
                type_ignores=[],
            )
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            exec(compile(native, "native Chat header", "exec"), {"gr": gr, "translate_ui": str})
        header_components = [
            entry for entry in demo.config["components"] if entry["type"] in ("html", "button")
        ]
        self.assertEqual([entry["type"] for entry in header_components], ["html", "html", "button"])
        self.assertIn("<h1>", header_components[0]["props"]["value"])
        self.assertIn("<p>", header_components[1]["props"]["value"])
        self.assertEqual(header_components[2]["props"]["elem_id"], "chat-setup-model")
        action = rules("#chat-setup-model", SHELL_CSS)
        self.assertEqual(len(action), 1)
        self.assertIn("position: static", action[0])
        self.assertIn("max-inline-size: 100%", action[0])
        self.assertIn("white-space: normal", action[0])
        self.assertNotRegex(action[0], r"font-size|inset|overflow:\s*hidden")

    def test_tabs_and_sibling_image_stage_share_tokens_and_scoped_controls(self):
        scope = ":is(#app-tabs, #image-stage)"
        self.assertNotIn("#app-tabs", CSS.replace(scope, "shared-root"))
        self.assertIn("--rf-preview-height:", rules(scope)[0])
        self.assertIn("--rf-action-height:", rules(scope)[0])
        for selector in (
            f"{scope} .rf-actions > button",
            f"{scope} #prompt-actions",
            f"{scope} #prompt-row",
            f"{scope} textarea",
            f"{scope} button:focus-visible",
        ):
            self.assertTrue(rules(selector), selector)
        # Navigation owns whether this sibling stage is shown. Scoping must not
        # introduce a root display/visibility rule that defeats Gradio updates.
        for root in rules(scope):
            self.assertNotRegex(root, r"(?:^|;)\s*(?:display|visibility|opacity)\s*:")

    def test_common_actions_and_legacy_aliases_share_one_wrapping_rule(self):
        common = rules(":is(#app-tabs, #image-stage) .rf-actions")
        self.assertEqual(len(common), 1)
        self.assertRegex(common[0], r"flex-wrap:\s*wrap")
        self.assertRegex(common[0], r"align-items:\s*stretch")
        for alias in (
            "#chat-controls .rf-chat-actions",
            "#chat-message-actions",
            "#prompt-tools .rf-prompt-actions",
            "#browser-actions",
            "#one-button-actions",
        ):
            self.assertEqual(rules(alias), common, alias)
        buttons = rules(":is(#app-tabs, #image-stage) .rf-actions > button")[0]
        self.assertIn("min-height: var(--rf-action-height)", buttons)
        self.assertIn("max-inline-size: 100%", buttons)
        self.assertIn("white-space: normal", buttons)
        self.assertIn("word-break: normal", buttons)
        self.assertNotRegex(buttons, r"font-size|max-height|overflow:\s*hidden")
        compact = rules(":is(#app-tabs, #image-stage) .rf-actions-compact > button")[0]
        self.assertRegex(compact, r"flex:\s*0\s+1\s+auto\s*!important")

    def test_native_action_buttons_are_direct_children_and_filters_have_a_form(self):
        with gr.Blocks(analytics_enabled=False) as demo:
            with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]) as actions:
                gr.Button("Refresh")
                gr.Button("Завантажити вибрані файли")
            with gr.Row(elem_classes=["rf-filters"]) as filters:
                for label in ("Category", "Workflow", "Quantization"):
                    gr.Dropdown(["All"], value="All", label=label)
        components = {entry["id"]: entry for entry in demo.config["components"]}
        action_children = layout_node(demo.config["layout"], actions._id)["children"]
        self.assertEqual(
            [components[item["id"]]["type"] for item in action_children], ["button"] * 2
        )
        self.assertEqual(
            components[actions._id]["props"]["elem_classes"], ["rf-actions", "rf-actions-compact"]
        )
        filter_children = layout_node(demo.config["layout"], filters._id)["children"]
        self.assertEqual(len(filter_children), 1)
        form = filter_children[0]
        self.assertEqual(components[form["id"]]["type"], "form")
        self.assertEqual(
            [components[item["id"]]["type"] for item in form["children"]], ["dropdown"] * 3
        )
        for selector in (
            ":is(#app-tabs, #image-stage) .rf-filters > *",
            ":is(#app-tabs, #image-stage) .rf-filters > .form > *",
        ):
            cell = rules(selector)[0]
            self.assertIn("min-inline-size: min(100%, 12rem) !important", cell)
            self.assertIn("max-inline-size: 100%", cell)
        wrapper = rules(":is(#app-tabs, #image-stage) .rf-filters > .form")[0]
        self.assertIn("display: flex", wrapper)
        self.assertIn("flex-wrap: wrap", wrapper)

    def test_phone_pagination_rules_match_the_actual_native_gallery_row(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        row = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and any(
                    key.arg == "elem_id"
                    and isinstance(key.value, ast.Constant)
                    and key.value.value == "browser-pagination"
                    for key in item.context_expr.keywords
                )
                for item in node.items
            )
        )
        scope = {"gr": gr, "t": lambda text: text, "pages": 3}
        with gr.Blocks(analytics_enabled=False) as demo:
            exec(
                compile(ast.Module(body=[row], type_ignores=[]), "native gallery pager", "exec"),
                scope,
            )
        components = {entry["id"]: entry for entry in demo.config["components"]}
        children = layout_node(demo.config["layout"], scope["pagination"]._id)["children"]
        # Gradio uses a Form only when the Number requests a container. Both
        # native layouts must put that same editable field in the middle cell.
        page = scope["ib_page"]
        page_type = "form" if page.container else "number"
        self.assertEqual(
            [components[item["id"]]["type"] for item in children],
            ["button", page_type, "markdown", "button"],
        )
        page_node = children[1]["children"][0] if page.container else children[1]
        self.assertEqual(page_node["id"], page._id)
        self.assertEqual(page.label, "Page")
        self.assertTrue(page.interactive is not False)
        self.assertEqual(children[-1]["id"], scope["next_page"]._id)
        self.assertIn("grid-column: 2", rules("#browser-pagination > #browser-page")[0])
        self.assertIn("grid-column: 1 / -1", rules("#browser-page-total")[-1])
        self.assertIn("grid-column: 3", rules("#browser-pagination > button:last-child")[0])
        phone = rules("#browser-pagination")[-1]
        self.assertIn("minmax(0, 4.5rem)", phone)
        start = CSS.index("#browser-pagination {", CSS.index("@media (max-width: 600px)"))
        self.assertGreater(CSS.rfind("@media (max-width: 600px)", 0, start), 0)

    def test_narrow_tabs_do_not_keep_an_unbounded_minimum_or_duplicate_slider_rules(self):
        tabs = rules("#generation-controls .tab-container.wrap")[0]
        self.assertIn("minmax(min(100%, 7.5rem), 1fr)", tabs)
        self.assertEqual(len(rules(":is(#app-tabs, #image-stage) .tab-like-container")), 1)
        self.assertEqual(len(rules("#chat-message-input textarea")), 1)
        self.assertIn(
            "min-height: var(--rf-inline-control-height)",
            rules(":is(#app-tabs, #image-stage) .tab-like-container")[0],
        )
        self.assertIn("outline:", rules(":is(#app-tabs, #image-stage) button:focus-visible")[0])
        self.assertIn("resize: vertical", rules("textarea")[0])


if __name__ == "__main__":
    unittest.main()
