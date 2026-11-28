from ruinedfooocus.core.paths import asset_path, resource_path

"""Exercise the pinned ControlNet conditioning node; no network weights/GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from ruinedfooocus.features.image_editing import native_control_recipes

ROOT = Path(__file__).resolve().parents[1]


class Control:
    def __init__(self, copies):
        self.copies = copies
        self.hint = None
        self.previous = None

    def copy(self):
        other = Control(self.copies)
        self.copies.append(other)
        return other

    def set_cond_hint(self, hint, strength, limits, **options):
        self.hint, self.strength, self.limits, self.options = hint, strength, limits, options
        return self

    def set_previous_controlnet(self, previous):
        self.previous = previous


class ControlNetSwitchTests(unittest.TestCase):
    def setUp(self):
        native = ast.parse((ROOT / "repositories/ComfyUI/nodes.py").read_text(encoding="utf-8"))
        node = next(
            n
            for n in native.body
            if isinstance(n, ast.ClassDef) and n.name == "ControlNetApplyAdvanced"
        )
        node.body = [
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "apply_controlnet"
        ]
        namespace = {
            "asset_path": asset_path,
            "resource_path": resource_path,
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "native_controlnet", "exec"),
            namespace,
        )
        self.native = namespace[node.name]
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        self.branch = next(
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "self.xl_controlnet"
        )
        self.select = next(
            n
            for n in process.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "positive_cond" for t in n.targets)
        )
        self.control_image = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.IfExp)
            and any(
                isinstance(target, ast.Name) and target.id == "control_image"
                for target in node.targets
            )
        )

    def run_route(self, switched=True, strength=0.7):
        copies = []
        control = Control(copies)
        cached_positive = [[torch.zeros(1, device="cpu"), {"cached": "positive"}]]
        cached_negative = [[torch.zeros(1, device="cpu"), {"cached": "negative"}]]
        positive = [[torch.zeros(1, device="cpu"), {"request": "positive"}]]
        negative = [[torch.zeros(1, device="cpu"), {"request": "negative"}]]
        intervals = (
            [
                [
                    torch.full((1,), float(i), device="cpu"),
                    {"start_percent": i * 0.5, "end_percent": (i + 1) * 0.5},
                ]
                for i in range(2)
            ]
            if switched
            else []
        )
        vae = object()
        owner = SimpleNamespace(
            xl_controlnet=control,
            xl_base_patched=SimpleNamespace(vae=vae),
            conditions={"+": {"cache": cached_positive}, "-": {"cache": cached_negative}},
        )
        image = torch.linspace(0, 1, 32 * 64 * 3, device="cpu").reshape(1, 32, 64, 3)
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            self=owner,
            positive_conditioning=positive,
            negative_conditioning=negative,
            switched_prompt=intervals,
            input_image=image,
            controlnet=dict(strength=strength, start=0.1, stop=0.9),
            ControlNetApplyAdvanced=self.native,
            native_control_recipes=native_control_recipes,
            gen_data={},
            named_source_mask=None,
            worker=SimpleNamespace(check_interrupt=lambda data: None),
            guided_edit=False,
            control_guide=None,
        )
        exec(
            compile(
                ast.Module(body=[self.control_image, self.branch, self.select], type_ignores=[]),
                "controlnet_route",
                "exec",
            ),
            scope,
        )
        return scope, copies, control, positive, negative, intervals

    def test_every_switched_interval_reaches_sampling_with_native_controlnet(self):
        scope, copies, base, positive, negative, intervals = self.run_route()
        selected = scope["positive_cond"]
        self.assertEqual(len(selected), 2)
        for row, index in zip(selected, range(2)):
            metadata = row[1]
            self.assertEqual(
                (metadata["start_percent"], metadata["end_percent"]),
                (index * 0.5, (index + 1) * 0.5),
            )
            native = metadata["control"]
            self.assertEqual(tuple(native.hint.shape), (1, 3, 32, 64))
            self.assertEqual((native.strength, native.limits), (0.7, (0.1, 0.9)))
            self.assertIs(native.options["vae"], scope["self"].xl_base_patched.vae)
            self.assertFalse(metadata["control_apply_to_uncond"])
        self.assertEqual(
            len(copies), 1
        )  # Native node shares the control across matching CFG branches.
        self.assertIs(scope["negative_conditioning"][0][1]["control"], copies[0])
        self.assertIsNone(base.hint)
        self.assertTrue(all("control" not in row[1] for row in positive + negative + intervals))
        self.assertNotIn("control", scope["self"].conditions["+"]["cache"][0][1])

    def test_current_request_conditioning_is_not_replaced_by_cached_text(self):
        scope, *_ = self.run_route(switched=False)
        self.assertEqual(scope["positive_cond"][0][1]["request"], "positive")
        self.assertEqual(scope["negative_conditioning"][0][1]["request"], "negative")
        self.assertNotIn("cached", scope["positive_cond"][0][1])

    def test_zero_strength_retains_switch_intervals_without_loading_control(self):
        scope, copies, _, _, _, intervals = self.run_route(strength=0)
        self.assertIs(scope["positive_cond"], intervals)
        self.assertEqual(copies, [])
        self.assertTrue(all("control" not in row[1] for row in scope["positive_cond"]))

    def test_different_requests_do_not_reuse_mutated_control_hints(self):
        first, *_ = self.run_route()
        second, *_ = self.run_route()
        self.assertIsNot(
            first["positive_cond"][0][1]["control"], second["positive_cond"][0][1]["control"]
        )


if __name__ == "__main__":
    unittest.main()
