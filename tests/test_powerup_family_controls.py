from ruinedfooocus.core.paths import asset_path, resource_path

"""Exercise production PowerUp family policy without starting Gradio or Torch."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch
import ruinedfooocus.app  # Initialize the parent before replacing its state submodule.
from ruinedfooocus.features.models.image_scope import require_image_family

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ruinedfooocus/features/image_editing/ui.py"


def functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "checkpoint_features",
        "control_types",
        "input_label",
        "image_tool_help",
        "sd4x_output_info",
        "tool_kind",
        "input_context",
        "powerup_choices",
        "preset_for",
        "edit_tool",
        "fallback_selection",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SDXL_CONTROLS" for target in node.targets
        )
    ]
    models = SimpleNamespace(
        get_models_by_path=Mock(return_value={}),
        get_model_base=Mock(return_value="Pony"),
        get_file=Mock(return_value=Path("model.safetensors")),
        detect_checkpoint_base=Mock(return_value="SDXL 1.0"),
    )
    controls = SimpleNamespace(
        controlnet_models=dict.fromkeys(
            ("canny", "depth", "sketch", "recolour", "img2img", "upscale", "rembg")
        ),
        cn_options={
            "Img2Img": {"type": "img2img"},
            "Canny": {"type": "canny"},
            "Depth": {"type": "depth"},
            "Sketch": {"type": "sketch"},
            "Recolour": {"type": "recolour"},
            "Remove background": {"type": "rembg"},
        },
    )
    namespace = {
        "asset_path": asset_path,
        "resource_path": resource_path,
        "shared": SimpleNamespace(models=models, settings=SimpleNamespace(default_settings={})),
        "controlnet": controls,
        "require_image_family": require_image_family,
        "NEWCN": "New...",
        "t": lambda value: value,
        "translate_ui": lambda value: value,
        "gr": SimpleNamespace(update=lambda **kwargs: kwargs),
    }
    guidance = ast.parse(
        (ROOT / "ruinedfooocus/features/image_editing/ui_guidance.py").read_text(encoding="utf-8")
    )
    family = next(
        n
        for n in guidance.body
        if isinstance(n, ast.FunctionDef) and n.name == "family_for_checkpoint"
    )
    family_scope = {}
    exec(compile(ast.Module(body=[family], type_ignores=[]), "family_policy", "exec"), family_scope)

    def resolve(name):
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": namespace["shared"]}),
            patch.object(ruinedfooocus.app, "state", namespace["shared"], create=True),
        ):
            return family_scope["family_for_checkpoint"](name)

    namespace["family_for_checkpoint"] = resolve
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace, models


class PowerUpFamilyTests(unittest.TestCase):
    def test_instruct_help_tracks_the_selected_checkpoint_without_claiming_fresh_generation(self):
        scope, models = functions()
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "model_changed"
        )
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
        for family in ("SD15_instructpix2pix", "SDXL_instructpix2pix"):
            models.detect_checkpoint_base.return_value = family
            info = scope["model_changed"]("selected", "None")[0]["info"]
            self.assertIn("requires Native image edit and a source image", info)
            self.assertNotIn("fresh image", info)
        models.detect_checkpoint_base.return_value = "SDXL 1.0"
        self.assertIn(
            "None generates a fresh image", scope["model_changed"]("selected", "None")[0]["info"]
        )

    def test_local_family_aliases_keep_native_edits_and_sdxl_controls_without_fetch(self):
        scope, models = functions()
        models.detect_checkpoint_base.return_value = None
        for family, expected_sdxl, expected_edit in [
            ("Qwen Image", False, True),
            ("Mage-Flow", False, True),
            ("MOSS Image", False, True),
            ("HiDream O1", False, True),
            ("Pony XL", True, False),
            ("Animagine XL 3.1", True, False),
        ]:
            with self.subTest(family=family):
                models.get_model_base.return_value = family
                features = scope["checkpoint_features"]("selected", "ai-toolkit")
                self.assertEqual(features["sdxl"], expected_sdxl)
                self.assertEqual(features["native_edit"], expected_edit)
        models.get_models_by_path.assert_called_with("checkpoints", "selected", fetch=False)

    def test_saved_upscaler_is_editable_and_rembg_has_no_ignored_sliders(self):
        scope, _ = functions()
        scope["controlnet"].cn_options["Custom upscale"] = {
            "type": "upscale",
            "upscaler": "old.pth",
        }
        scope.update(
            gr=SimpleNamespace(update=lambda **kwargs: kwargs),
            cn_sliders=[None] * 6,
            cn_outputs=[None] * 3,
        )
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for name in ("cn_changed", "cn_type_changed"):
            callback = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            callback.decorator_list = []
            exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
        self.assertEqual(
            scope["cn_changed"]("Custom upscale", "Upscale", "model")[-1],
            {"visible": True, "value": "old.pth"},
        )
        self.assertTrue(
            all(
                update["visible"] in (False, "hidden")
                for update in scope["cn_type_changed"]("Rembg", "New...", "model")
            )
        )
        function = next(
            node
            for node in ast.parse(
                (ROOT / "ruinedfooocus/features/image_editing/controlnet.py").read_text(
                    encoding="utf-8"
                )
            ).body
            if isinstance(node, ast.FunctionDef) and node.name == "get_settings"
        )
        preset = {"type": "upscale", "upscaler": "old.pth"}
        namespace = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            "controlnet_settings": {"Custom upscale": preset},
            "NEWCN": "New...",
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "control_settings", "exec"),
            namespace,
        )
        result = namespace["get_settings"](
            {"cn_selection": "Custom upscale", "cn_upscale": "new.pth"}
        )
        self.assertEqual(result["upscaler"], "new.pth")
        self.assertEqual(preset["upscaler"], "old.pth")

    def test_sdxl_aliases_use_backend_architecture_and_keep_controls(self):
        scope, models = functions()
        for alias in ("Pony", "Illustrious", "NoobAI", "SDXL 1.0"):
            models.get_model_base.return_value = alias
            features = scope["checkpoint_features"]("selected")
            self.assertTrue(features["sdxl"])
            choices = dict(scope["powerup_choices"](**features))
            self.assertIn("Canny", choices)
            self.assertIn("Depth", choices)
        models.detect_checkpoint_base.assert_called_with(Path("model.safetensors"))
        models.get_models_by_path.assert_called_with("checkpoints", "selected", fetch=False)
        # Real tensor architecture takes precedence over misleading cached metadata.
        models.detect_checkpoint_base.return_value = "QwenImage"
        self.assertFalse(scope["checkpoint_features"]("selected")["sdxl"])

    def test_native_edits_keep_upload_route_and_only_native_control_families_offer_processors(self):
        scope, models = functions()
        for family in ("MageFlow", "Fibo2", "QwenImage", "MOSSImage", "HiDreamO1"):
            models.get_model_base.return_value = family
            models.detect_checkpoint_base.return_value = family
            features = scope["checkpoint_features"]("selected", "ai-toolkit")
            self.assertFalse(features["sdxl"])
            self.assertEqual(scope["input_label"](features), "Image edit reference")
            choices = scope["powerup_choices"](**features)
            self.assertIn(("Image to image", "Img2Img"), choices)
            self.assertEqual(
                bool({"Canny", "Depth", "Sketch", "Recolour"} & {value for _, value in choices}),
                family == "QwenImage",
            )
            self.assertIn("Img2img", scope["control_types"](features))
            self.assertEqual("Canny" in scope["control_types"](features), family == "QwenImage")

    def test_model_change_filters_selection_and_context_disables_unsupported_inpainting(self):
        scope, models = functions()
        models.get_model_base.return_value = "QwenImage"
        models.detect_checkpoint_base.return_value = "QwenImage"
        scope["gr"] = SimpleNamespace(update=lambda **kwargs: kwargs)
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "model_changed"
        )
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
        result = scope["model_changed"]("selected", "Canny")
        self.assertEqual(result[0]["value"], "Canny")
        self.assertIn("Canny", result[1]["choices"])
        self.assertEqual(
            scope["input_context"](True, "None", "Img2img", "selected")[2],
            {"visible": False, "value": False},
        )
        self.assertEqual(
            scope["input_context"](False, "None", "Img2img", "selected")[0]["label"],
            "Image edit reference",
        )

    def test_uploads_pick_the_best_edit_tool_and_edits_survive_checkpoint_changes(self):
        scope, models = functions()
        scope["controlnet"].cn_options["Native image edit"] = {"type": "edit"}
        scope["controlnet"].controlnet_models["edit"] = None
        models.detect_checkpoint_base.return_value = "QwenImage"
        qwen = scope["checkpoint_features"]("selected")
        models.detect_checkpoint_base.return_value = "SDXL 1.0"
        sdxl = scope["checkpoint_features"]("selected")
        self.assertEqual(scope["edit_tool"](qwen), "Native image edit")
        self.assertEqual(scope["edit_tool"](sdxl), "Img2Img")
        fallback = scope["fallback_selection"]
        # Native edit falls back to image to image, and returns on an edit-capable checkpoint.
        self.assertEqual(fallback(sdxl, "Native image edit", False), ("Img2Img", True))
        self.assertEqual(fallback(qwen, "Img2Img", True), ("Native image edit", False))
        # A deliberate image-to-image choice and other tools are left alone.
        self.assertEqual(fallback(qwen, "Img2Img", False), ("Img2Img", False))
        self.assertEqual(fallback(sdxl, "Canny", False), ("Canny", False))
        self.assertEqual(fallback(qwen, "Canny", False), ("Canny", False))

    def test_moss_format_toggle_updates_presentation_without_clearing_inputs(self):
        scope, models = functions()
        models.get_model_base.return_value = "MOSSImage"
        models.detect_checkpoint_base.return_value = "MOSSImage"
        scope.update(
            gr=SimpleNamespace(update=lambda **kwargs: kwargs),
            cn_sliders=[None] * 6,
            cn_outputs=[None] * 3,
        )
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for name in ("cn_changed", "edit_format_changed"):
            callback = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            callback.decorator_list = []
            exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
        disabled = scope["checkpoint_features"]("selected", "Disabled")
        self.assertFalse(disabled["native_edit"])
        self.assertEqual(scope["input_label"](disabled), "Image to image")
        for edit_format in ("ai-toolkit", "Moss-Image-Edit", "Disabled"):
            with self.subTest(edit_format=edit_format):
                updates = scope["edit_format_changed"](
                    edit_format, "selected", "Img2Img", "Img2img"
                )
                self.assertNotIn("value", updates[0])  # selected saved preset remains
                self.assertNotIn("value", updates[1])  # current denoise remains
                self.assertTrue(updates[1]["visible"])
                context = scope["input_context"](
                    False, "Img2Img", "Img2img", "selected", edit_format
                )[0]
                self.assertNotIn("value", context)  # uploaded pixels remain
                self.assertEqual(
                    context["label"],
                    "Image to image" if edit_format == "Disabled" else "Image edit reference",
                )
                self.assertIn(("Image to image", "Img2Img"), updates[0]["choices"])
                changed = scope["cn_changed"]("Img2Img", "Img2img", "selected", edit_format)
                self.assertTrue(changed[5]["visible"])

    def test_inpainting_hides_unused_uploads_and_keeps_all_control_guides(self):
        scope, _ = functions()
        for selection, kind in (("None", "Canny"), ("Img2Img", "Img2img"), ("New...", "Img2img")):
            with self.subTest(selection=selection):
                update, strength = scope["input_context"](True, selection, kind, "selected")[:2]
                self.assertFalse(update["visible"])
                self.assertEqual(strength["visible"], "hidden")
                self.assertNotIn("value", update)
                normal, normal_strength = scope["input_context"](
                    False, selection, kind, "selected"
                )[:2]
                self.assertTrue(normal["visible"])
                self.assertEqual(
                    normal_strength["visible"], True if selection != "None" else "hidden"
                )
        for selection in ("Canny", "Depth", "Sketch", "Recolour"):
            for preset in (selection, "New..."):
                update, strength = scope["input_context"](True, preset, selection, "selected")[:2]
                self.assertEqual(update, {"visible": True, "label": "Control guide"})
                self.assertTrue(strength["visible"])

    def test_other_image_operations_keep_their_upload_and_native_edits_recover_after_model_change(
        self,
    ):
        scope, models = functions()
        self.assertTrue(
            scope["input_context"](True, "Remove background", "Rembg", "selected")[0]["visible"]
        )
        for family in ("MageFlow", "Fibo2", "QwenImage"):
            with self.subTest(family=family):
                models.detect_checkpoint_base.return_value = family
                update = scope["input_context"](False, "None", "Img2img", "selected")[0]
                self.assertEqual(update, {"visible": True, "label": "Image edit reference"})

    def test_custom_strength_hints_follow_the_type_instead_of_the_previous_preset(self):
        scope, _ = functions()
        scope.update(cn_sliders=[None] * 6, cn_outputs=[None] * 3)
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for name in ("cn_changed", "cn_type_changed"):
            callback = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            callback.decorator_list = []
            exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
        scope["cn_changed"]("Img2Img", "Img2img", "selected")
        guide = scope["cn_type_changed"]("Canny", "New...", "selected")[2]
        self.assertIn("composition", guide["info"])
        self.assertNotIn("preserve", guide["info"])
        img2img = scope["cn_type_changed"]("Img2img", "New...", "selected")[2]
        self.assertEqual(img2img["label"], "Denoise")
        self.assertIn("preserve the input", img2img["info"])

    def test_checkpoint_tools_are_not_an_image_operation(self):
        scope, _ = functions()
        choices = scope["powerup_choices"](False)
        self.assertNotIn("Checkpoint tools", {value for _, value in choices})

    def test_saved_canny_exposes_thresholds_and_generation_does_not_mutate_preset(self):
        scope, _ = functions()
        scope["controlnet"].cn_options["Canny"].update(edge_low=0.15, edge_high=0.75)
        scope.update(
            gr=SimpleNamespace(update=lambda **kwargs: kwargs),
            cn_sliders=[None] * 6,
            cn_outputs=[None] * 3,
        )
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "cn_changed"
        )
        callback.decorator_list = []
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
        values = scope["cn_changed"]("Canny", "Canny", "selected")
        self.assertEqual(values[6], {"visible": True, "value": 0.15})
        self.assertEqual(values[7], {"visible": True, "value": 0.75})
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/image_editing/controlnet.py").read_text(
                encoding="utf-8"
            )
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_settings"
        )
        preset = {"type": "canny", "edge_low": 0.15, "edge_high": 0.75, "strength": 1.0}
        namespace = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            "controlnet_settings": {"Canny": preset},
            "NEWCN": "New...",
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "control_settings", "exec"),
            namespace,
        )
        result = namespace["get_settings"](
            {"cn_selection": "Canny", "cn_edge_low": 0.0, "cn_edge_high": 0.9}
        )
        self.assertEqual(result["edge_low"], 0.0)
        self.assertEqual(result["edge_high"], 0.9)
        self.assertEqual(preset["edge_low"], 0.15)
        self.assertEqual(preset["edge_high"], 0.75)

    def test_native_img2img_exposes_the_effective_denoise(self):
        scope, models = functions()
        models.get_model_base.return_value = "MageFlow"
        models.detect_checkpoint_base.return_value = "MageFlow"
        scope.update(
            gr=SimpleNamespace(update=lambda **kwargs: kwargs),
            cn_sliders=[None] * 6,
            cn_outputs=[None] * 3,
        )
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "cn_changed"
        )
        callback.decorator_list = []
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
        result = scope["cn_changed"]("Img2Img", "Img2img", "selected")
        self.assertTrue(result[5]["visible"])
        self.assertEqual(result[5]["label"], "Denoise")


if __name__ == "__main__":
    unittest.main()
