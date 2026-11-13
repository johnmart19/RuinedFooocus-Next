from ruinedfooocus.core.paths import asset_path, resource_path, data_path

"""Retired image-tool requests must not silently generate an unrelated image."""
import ast
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ruinedfooocus.features.generation.image_preparation import CONTROL_MODELS
from ruinedfooocus.features.image_processing.still_image_utilities import KINDS as DEPTH_UTILITIES

ROOT = Path(__file__).resolve().parents[1]


def control_scope():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/image_editing/controlnet.py").read_text(encoding="utf-8")
    )
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("load_cnsettings", "get_settings")
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id
            in ("DEFAULT_CNSETTINGS_FILE", "CNSETTINGS_FILE", "NEWCN", "controlnet_models")
            for target in node.targets
        )
    ]
    scope = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        data_path=data_path,
        os=os,
        shutil=shutil,
        json=json,
        CONTROL_MODELS=CONTROL_MODELS,
        DEPTH_UTILITIES=DEPTH_UTILITIES,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "control_settings", "exec"), scope)
    scope["controlnet_settings"] = json.loads(
        (ROOT / "ruinedfooocus/resources/defaults/powerup.default").read_text()
    )
    return scope


def route(scope, data):
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/generation/pipelines.py").read_text(encoding="utf-8")
    )
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "update")
    state = {"pipeline": None}
    factories = {
        name: SimpleNamespace(pipeline=Mock(return_value=SimpleNamespace(pipeline_type=[kind])))
        for name, kind in (
            ("image_pipeline", "sdxl"),
            ("upscale_pipeline", "upscale"),
            ("rembg_pipeline", "rembg"),
            ("llama_pipeline", "llama"),
        )
    }
    models = SimpleNamespace(
        get_file=lambda *args: Path("model.safetensors"),
        get_models_by_path=lambda *args: {},
        get_model_base=lambda *args: "SDXL",
    )
    namespace = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        state=state,
        shared=SimpleNamespace(models=models),
        Path=Path,
        VIDEO_PIPELINES={},
        controlnet=SimpleNamespace(get_settings=scope["get_settings"]),
        selected_pipeline=lambda: state["pipeline"],
        NoPipeLine=lambda: SimpleNamespace(pipeline_type=[]),
        **factories,
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "select_image_pipeline", "exec"),
        namespace,
    )
    return (
        lambda: namespace["update"](dict(prompt="", base_model_name="model.safetensors", **data)),
        state,
        factories,
    )


class ImageToolRegistryTests(unittest.TestCase):
    def test_custom_tool_choices_only_offer_implemented_types(self):
        scope = control_scope()
        self.assertNotIn("faceswap", scope["controlnet_models"])
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/image_editing/ui.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "control_types"
        )
        ui = dict(
            controlnet=SimpleNamespace(controlnet_models=scope["controlnet_models"]),
            SDXL_CONTROLS=frozenset(("canny", "depth", "sketch", "recolour")),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "custom_choices", "exec"), ui)
        choices = ui["control_types"]({"sdxl": True})
        self.assertNotIn("Faceswap", choices)
        self.assertTrue({"Img2img", "Upscale", "Rembg", "Canny", "Depth"} <= set(choices))

    def test_stale_direct_saved_and_custom_face_swap_requests_fail_clearly(self):
        scope = control_scope()
        scope["controlnet_settings"]["Legacy swap"] = {"type": "faceswap"}
        requests = [
            {"controlnet": {"type": "faceswap"}},
            {"controlnet": {"type": "FaceSwap"}},
            {"cn_selection": "Legacy swap"},
            dict(
                cn_selection=scope["NEWCN"],
                cn_type="Faceswap",
                cn_edge_low=0.2,
                cn_edge_high=0.8,
                cn_start=0,
                cn_stop=1,
                cn_strength=1,
                cn_upscale="None",
            ),
        ]
        for request in requests:
            run, state, factories = route(scope, request)
            with (
                self.subTest(request=request),
                self.assertRaisesRegex(RuntimeError, "Face swapping.*not available"),
            ):
                run()
            self.assertIsNone(state["pipeline"])
            for factory in factories.values():
                factory.pipeline.assert_not_called()

    def test_legacy_saved_swap_is_not_offered_and_the_user_file_is_preserved(self):
        scope = control_scope()
        defaults = scope["controlnet_settings"]
        legacy = dict(defaults, **{"Legacy swap": {"type": "faceswap"}})
        with tempfile.TemporaryDirectory() as folder:
            source, saved = Path(folder) / "defaults.json", Path(folder) / "saved.json"
            source.write_text(json.dumps(defaults))
            original = json.dumps(legacy)
            saved.write_text(original)
            scope.update(
                DEFAULT_CNSETTINGS_FILE=str(source), CNSETTINGS_FILE=str(saved), save_json=Mock()
            )
            loaded = scope["load_cnsettings"]()
            self.assertNotIn("Legacy swap", loaded)
            self.assertEqual(loaded["Restore cropped face"], defaults["Restore cropped face"])
            scope["save_json"].assert_not_called()
            self.assertEqual(saved.read_text(), original)

    def test_face_restoration_remains_an_upscale_preset_and_direct_settings_are_copied(self):
        scope = control_scope()
        expected = {"type": "upscale", "upscaler": "faceswap/GFPGANv1.4.pth"}
        self.assertEqual(scope["get_settings"]({"cn_selection": "Restore cropped face"}), expected)
        run, _, _ = route(scope, {"cn_selection": "Restore cropped face"})
        self.assertEqual(run().pipeline_type, ["upscale"])
        direct = {"type": "img2img", "denoise": 0.25}
        result = scope["get_settings"]({"controlnet": direct})
        result["denoise"] = 0.9
        self.assertEqual(direct["denoise"], 0.25)
        self.assertEqual(scope["get_settings"]({}), {})
        self.assertEqual(scope["get_settings"]({"cn_selection": "None"}), {})

    def test_explicit_depth_request_is_not_replaced_by_stale_ui_options(self):
        scope = control_scope()
        direct = {
            "type": "depth",
            "strength": 0.7,
            "depth_preprocess": "none",
            "depth_resolution": 280,
        }
        result = scope["get_settings"](
            {"controlnet": direct, "cn_depth_preprocess": "auto", "cn_depth_resolution": 504}
        )
        self.assertEqual(result, direct)
        self.assertIsNot(result, direct)

    def test_retired_image_settings_do_not_change_chat_routing(self):
        run, _, factories = route(
            control_scope(), {"task_type": "llama", "controlnet": {"type": "faceswap"}}
        )
        self.assertEqual(run().pipeline_type, ["llama"])
        factories["image_pipeline"].pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
