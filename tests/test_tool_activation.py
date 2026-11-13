"""Tool/resource pairs are one explicit, source-free activation transaction."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.image_editing.tool_activation import activation_values, installed_tool
from ruinedfooocus.features.models.model_identity import file_signature


class ToolActivationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.path = self.root / "resource.safetensors"
        self.path.write_bytes(b"inert header fixture")
        self.manager = SimpleNamespace(get_model_folders=lambda _: [self.root])

    def row(self, category):
        return dict(
            category=category,
            local=str(self.path),
            local_signature=file_signature(self.path),
            size_matches=True,
        )

    def test_registered_installed_upscaler_uses_exact_local_identity(self):
        with patch(
            "ruinedfooocus.features.generation.automatic_upscale.native_metadata",
            return_value=dict(native_scale=4, input_channels=3),
        ):
            resource = installed_tool(self.row("upscalers"), self.manager)
            values = activation_values(
                "upscale",
                resource,
                presets={"Upscale Fast": dict(type="upscale", upscaler="old.pth")},
                custom="Custom...",
            )
        self.assertEqual(values["cn_selection"], "Custom...")
        self.assertEqual(values["cn_type"], "Upscale")
        self.assertEqual(values["cn_upscale"], self.path.name)
        self.assertNotIn("input_image", values)
        self.assertNotIn("prompt", values)

    def test_birefnet_and_da3_use_native_validated_resource_contracts(self):
        for category, module, returned, kind, field in (
            (
                "background_removal",
                "ruinedfooocus.features.image_processing.native_background.local_models",
                {self.path.name: str(self.path)},
                "rembg",
                "rembg_native_model",
            ),
            (
                "geometry_estimation",
                "ruinedfooocus.features.image_processing.native_vision.native_model_choices",
                [("DA3", str(self.path))],
                "depth-map",
                "native_vision_model",
            ),
        ):
            with self.subTest(category=category), patch(module, return_value=returned):
                target = installed_tool(self.row(category), self.manager)
                values = activation_values(
                    kind, target, presets={"Tool": dict(type=kind)}, custom="Custom..."
                )
                self.assertEqual(values["cn_selection"], "Tool")
                self.assertEqual(values[field], target["value"])
                if kind == "depth-map":
                    self.assertNotIn("cn_type", values)

    def test_unsupported_incomplete_replaced_and_priority_shadowed_files_do_not_offer_use(self):
        self.assertIsNone(installed_tool(self.row("latent_upscalers"), self.manager))
        with patch(
            "ruinedfooocus.features.generation.automatic_upscale.native_metadata", return_value=None
        ):
            self.assertIsNone(installed_tool(self.row("upscalers"), self.manager))
        shadow = self.root / "priority"
        shadow.mkdir()
        (shadow / self.path.name).write_bytes(b"same-name different file")
        self.manager.get_model_folders = lambda _: [shadow, self.root]
        with patch(
            "ruinedfooocus.features.generation.automatic_upscale.native_metadata",
            return_value=dict(native_scale=4, input_channels=3),
        ):
            self.assertIsNone(installed_tool(self.row("upscalers"), self.manager))
        row = self.row("upscalers")
        self.path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed or is incomplete"):
            installed_tool(row, self.manager)
        with self.assertRaisesRegex(ValueError, "does not support"):
            activation_values(
                "rembg",
                dict(kind="upscale", resource="cn_upscale", value="wrong"),
                presets={},
                custom="Custom...",
            )

    def test_programmatic_preset_and_type_changes_do_not_overwrite_atomic_resource_or_denoise(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/image_editing/ui.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in ("cn_changed", "cn_type_changed")
        ]
        scope = dict(
            gr=SimpleNamespace(update=lambda **value: value),
            checkpoint_features=lambda *_: {"sdxl": True},
            controlnet=SimpleNamespace(
                cn_options={"Pixel": {"type": "upscale", "upscaler": "old.pth"}}
            ),
            NEWCN="Custom...",
            cn_outputs=list(range(3)),
            cn_sliders=list(range(6)),
            SDXL_CONTROLS=set(),
            t=lambda text: text,
            translate_ui=lambda text: text,
            control_types=lambda _: ["Img2img", "Upscale"],
        )
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "tool update callbacks", "exec"), scope
        )
        saved = scope["cn_changed"]("Pixel", "Upscale", "model", apply_values=False)
        self.assertNotIn("value", saved[-1])
        custom = scope["cn_type_changed"]("Upscale", "Custom...", "model")
        self.assertTrue(all("value" not in update for update in custom))
        img2img = scope["cn_type_changed"]("Img2img", "Custom...", "model")
        self.assertNotIn("value", img2img[2])
        deliberate = scope["cn_type_changed"]("Img2img", "Custom...", "model", apply_values=True)
        self.assertEqual(deliberate[2]["value"], 0.25)

    def test_ui_utility_adapter_updates_preset_and_resource_without_checkpoint_lookup(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/image_editing/ui.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "activate_tool"
        )
        controls = {
            name: SimpleNamespace(choices=[("Old", "old.pth")])
            for name in ("cn_selection", "cn_type", "cn_upscale", "automatic_upscale_target")
        }

        # Real Gradio controls are hashable; this fixture also verifies one output per control.
        class Control:
            choices = [("Old", "old.pth")]

        controls = {name: Control() for name in controls}
        lookup = Mock(side_effect=AssertionError("generation checkpoint lookup"))
        scope = dict(
            shared=SimpleNamespace(get_ctrl=controls.__getitem__),
            gr=SimpleNamespace(update=lambda **value: value),
            checkpoint_features=lookup,
            controlnet=SimpleNamespace(cn_options={}),
            NEWCN="Custom...",
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "UI tool adapter", "exec"), scope)
        result = scope["activate_tool"](
            "upscale",
            dict(kind="upscale", resource="cn_upscale", value="exact.pth"),
            checkpoint="irrelevant",
        )
        self.assertEqual(result[controls["cn_selection"]]["value"], "Custom...")
        self.assertEqual(result[controls["cn_upscale"]]["value"], "exact.pth")
        self.assertIn(("exact.pth", "exact.pth"), result[controls["cn_upscale"]]["choices"])
        lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
