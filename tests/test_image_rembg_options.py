from ruinedfooocus.core.paths import asset_path, resource_path

"""Background-removal controls and saved options without model inference."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

from test_powerup_family_controls import functions

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ruinedfooocus/features/image_editing/ui.py"


def callback(name, scope):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[name]


class BackgroundOptionTests(unittest.TestCase):
    def test_force_control_only_appears_for_active_background_removal(self):
        scope, _ = functions()
        for selection, kind, scale, target, visible in (
            ("Remove background", "Rembg", "Off", "Generated image", True),
            ("New...", "Rembg", "Off", "Generated image", True),
            ("Canny", "Rembg", "Off", "Generated image", False),
            ("None", "Rembg", "Off", "Generated image", False),
            ("New...", "Img2img", "Off", "Generated image", False),
            ("Remove background", "Rembg", "2× General (RealESRGAN)", "Input image only", False),
        ):
            with self.subTest(selection=selection, kind=kind, target=target):
                updates = scope["input_context"](
                    False, selection, kind, "selected", None, scale, target
                )
                self.assertEqual(updates[4], {"visible": True if visible else "hidden"})
                self.assertNotIn("value", updates[4])

    def test_force_is_saved_with_custom_background_preset_and_reloaded(self):
        scope, _ = functions()
        scope["load_cnsettings"] = lambda: dict(scope["controlnet"].cn_options)
        scope["save_cnsettings"] = Mock(
            side_effect=lambda data: scope["controlnet"].cn_options.update(data)
        )
        run = callback("cn_save", scope)
        run(
            "Transparent redo",
            None,
            "Rembg",
            0.0,
            1.0,
            1.0,
            0.2,
            0.8,
            "None",
            "selected",
            None,
            True,
        )
        self.assertIs(scope["save_cnsettings"].call_args.args[0]["Transparent redo"]["force"], True)
        load = callback("rembg_preset_changed", scope)
        self.assertEqual(load("Transparent redo"), {"value": True})
        self.assertEqual(load("Remove background"), {"value": False})
        self.assertEqual(load("New..."), {"value": False})


if __name__ == "__main__":
    unittest.main()
