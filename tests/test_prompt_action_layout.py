"""Source contracts for intrinsic Generate sizing and readable narrow layouts."""

from pathlib import Path
import ast
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "ruinedfooocus/ui/assets/ui.css").read_text(encoding="utf-8")


def rules(selector):
    return re.findall(re.escape(selector) + r"\s*\{([^{}]*)\}", CSS)


class PromptActionLayoutTests(unittest.TestCase):
    def test_action_intrinsic_size_overrides_the_native_scaled_column(self):
        actions = rules(":is(#app-tabs, #image-stage) #prompt-actions")[0]
        self.assertRegex(actions, r"flex:\s*0\s+1\s+max-content\s*!important")
        self.assertRegex(actions, r"min-inline-size:\s*min\(100%,\s*7\.5rem\)\s*!important")
        self.assertRegex(actions, r"max-inline-size:\s*100%")

    def test_long_localized_labels_wrap_without_clipping_or_smaller_fonts(self):
        button = rules(":is(#app-tabs, #image-stage) #prompt-actions button")[0]
        self.assertRegex(button, r"white-space:\s*normal")
        self.assertRegex(button, r"word-break:\s*normal")
        self.assertRegex(button, r"overflow-wrap:\s*break-word")
        self.assertNotRegex(button, r"font-size|text-overflow|overflow:\s*hidden|max-height")
        # Phones retain a complete action row instead of squeezing the label.
        phone = rules(":is(#app-tabs, #image-stage) #prompt-actions")[-1]
        self.assertRegex(phone, r"width:\s*100%")
        self.assertRegex(phone, r"flex:\s*1\s+1\s+100%\s*!important")
        self.assertRegex(phone, r"min-inline-size:\s*0\s*!important")

    def test_prompt_retains_editable_space_before_the_action_wraps(self):
        row = rules("#prompt-row")[0]
        self.assertRegex(row, r"flex-wrap:\s*wrap")
        prompt = rules("#prompt-row > :first-child")[0]
        self.assertRegex(prompt, r"min-inline-size:\s*min\(100%,\s*12rem\)\s*!important")

    def test_phone_grid_overrides_native_row_without_squeezing_either_child(self):
        phone = rules(":is(#app-tabs, #image-stage) #prompt-row")[0]
        self.assertRegex(phone, r"display:\s*grid\s*!important")
        self.assertRegex(phone, r"grid-template-columns:\s*minmax\(0,\s*1fr\)")
        prompt = rules(":is(#app-tabs, #image-stage) #prompt-row > :first-child")[0]
        self.assertRegex(prompt, r"min-inline-size:\s*0\s*!important")
        # This selector belongs inside the phone media query, not desktop layout.
        start = CSS.index(":is(#app-tabs, #image-stage) #prompt-row {")
        self.assertGreater(CSS.rfind("@media (max-width: 600px)", 0, start), 0)

    def test_production_gradio_prompt_row_places_both_columns_in_the_phone_grid(self):
        import gradio as gr

        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        row = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and any(
                    key.arg == "elem_id"
                    and isinstance(key.value, ast.Constant)
                    and key.value.value == "prompt-row"
                    for key in item.context_expr.keywords
                )
                for item in node.items
            )
        )
        label = "Згенерувати зображення з докладним описом"
        scope = dict(
            gr=gr,
            t=lambda value: label if value == "Generate" else value,
            translate_ui=lambda value: value,
            settings={"prompt": "preserved draft"},
            add_ctrl=lambda *args: None,
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            with gr.Tabs(elem_id="app-tabs") as tabs:
                with gr.Tab("Create"):
                    pass
            with gr.Group(elem_id="image-stage") as image_stage:
                exec(
                    compile(
                        ast.Module(body=[row], type_ignores=[]), "production_prompt_row", "exec"
                    ),
                    scope,
                )
        self.assertEqual(
            [node["id"] for node in demo.config["layout"]["children"]],
            [tabs._id, image_stage._id],
        )
        components = {
            entry["props"].get("elem_id"): entry
            for entry in demo.config["components"]
            if entry.get("props", {}).get("elem_id")
        }
        row_id, action_id = (components[key]["id"] for key in ("prompt-row", "prompt-actions"))

        def layout(node):
            if node["id"] == row_id:
                return node
            return next(
                (found for child in node.get("children", []) if (found := layout(child))), None
            )

        children = layout(demo.config["layout"])["children"]
        self.assertEqual(len(children), 2)
        self.assertEqual(children[1]["id"], action_id)
        self.assertEqual(components["generate"]["props"]["value"], label)
        self.assertEqual(scope["prompt"].value, "preserved draft")
        self.assertEqual(demo.blocks[children[0]["id"]].min_width, 0)
        self.assertEqual(demo.blocks[action_id].min_width, 0)


if __name__ == "__main__":
    unittest.main()
