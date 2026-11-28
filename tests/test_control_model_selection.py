"""Native control-file selection is header bounded, explicit and signature bound."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.image_editing import control_candidates as controls


def header_file(path, tensors):
    header = {
        name: {"dtype": "F16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    data = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(data).to_bytes(8, "little") + data)
    return path


class ControlModelSelection(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.roots = [Path(temporary).resolve() / "first", Path(temporary).resolve() / "second"]
        for root in self.roots:
            root.mkdir()
        self.manager = SimpleNamespace(
            get_model_folders=lambda category: self.roots,
            preparation_request=Mock(
                side_effect=AssertionError("Discovery must not prepare/download")
            ),
        )

    def test_native_family_branches_and_t2i_layouts_do_not_use_filenames(self):
        fixtures = [
            (
                {
                    "zero_convs.0.0.weight": [320, 320, 1, 1],
                    "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": [320, 768],
                },
                "SD 1.5",
            ),
            (
                {
                    "lora_controlnet": [],
                    "input_blocks.1.1.transformer_blocks.0.attn2.to_k.down": [128, 2048],
                },
                "SDXL",
            ),
            (
                {
                    "controlnet_blocks.0.weight": [3072, 3072],
                    "controlnet_x_embedder.weight": [3072, 64],
                },
                "Flux",
            ),
            (
                {
                    "controlnet_blocks.0.weight": [1536, 1536],
                    "pos_embed_input.proj.weight": [1536, 16, 2, 2],
                },
                "SD3",
            ),
            (
                {
                    "control_blocks.0.after_proj.weight": [3072, 3072],
                    "control_blocks.0.img_mod.1.weight": [18432, 3072],
                    "control_img_in.weight": [3072, 64],
                },
                "QwenImage",
            ),
            (
                {
                    "controlnet_blocks.0.weight": [3072, 3072],
                    "controlnet_x_embedder.weight": [3072, 68],
                    "transformer_blocks.0.img_mlp.net.0.proj.weight": [24576, 3072],
                },
                "QwenImage",
            ),
            ({"after_proj_list.18.bias": [1408]}, "HunyuanDiT"),
            (
                {"conv_in.weight": [320, 768, 3, 3], "body.0.block2.weight": [320, 320, 3, 3]},
                "SDXL",
            ),
        ]
        for index, (shape, family) in enumerate(fixtures):
            with self.subTest(family=family):
                path = header_file(self.roots[0] / f"renamed-{index}.safetensors", shape)
                info = controls.inspect_control(path)
                self.assertEqual(controls.compatibility(info, family), "compatible")
                self.assertEqual(controls.compatibility(info, "PiD"), "incompatible")
        unknown = controls.classify_header(
            {
                "conv_in.weight": {"shape": [320, 999, 3, 3]},
                "body.0.block2.weight": {"shape": [320, 320, 3, 3]},
            }
        )
        self.assertEqual(controls.compatibility(unknown, "SDXL"), "unknown")

    def test_pinned_native_loader_routes_the_recognized_modern_headers_and_passes_primary(self):
        from test_native_model_coverage import NATIVE

        source = NATIVE / "comfy/controlnet.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {
            "load_controlnet_hunyuandit": "HunyuanDiT",
            "load_controlnet_flux_xlabs_mistoline": "Flux",
            "load_controlnet_sd35": "SD3",
            "load_controlnet_mmdit": "SD3",
            "load_controlnet_qwen_instantx": "QwenImage",
            "load_controlnet_flux_instantx": "Flux",
            "load_controlnet_qwen_fun": "QwenImage",
        }
        scope = {name: (lambda *a, family=family, **kw: family) for name, family in names.items()}
        scope["convert_mistoline"] = lambda data: data
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_controlnet_state_dict"
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        for shape, family in (
            ({"after_proj_list.18.bias": [1408]}, "HunyuanDiT"),
            ({"controlnet_blocks.0.linear.weight": [3072, 3072]}, "Flux"),
            (
                {
                    "controlnet_blocks.0.weight": [3072, 3072],
                    "controlnet_x_embedder.weight": [3072, 64],
                },
                "Flux",
            ),
            (
                {
                    "controlnet_blocks.0.weight": [1536, 1536],
                    "pos_embed_input.proj.weight": [1536, 16, 2, 2],
                },
                "SD3",
            ),
            (
                {
                    "control_blocks.0.after_proj.weight": [3072, 3072],
                    "control_blocks.0.img_mod.1.weight": [18432, 3072],
                    "control_img_in.weight": [3072, 64],
                },
                "QwenImage",
            ),
        ):
            native = scope["load_controlnet_state_dict"](
                {key: SimpleNamespace(shape=dimensions) for key, dimensions in shape.items()}
            )
            self.assertEqual(native, family)
            self.assertEqual(
                controls.classify_header(
                    {key: {"shape": dimensions} for key, dimensions in shape.items()}
                )["families"],
                (family,),
            )
        load = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_controlnet"
        )
        read = Mock(return_value={"owned": "weights"})
        forward = Mock(return_value=object())
        scope = dict(
            os=os,
            comfy=SimpleNamespace(utils=SimpleNamespace(load_torch_file=read)),
            load_controlnet_state_dict=forward,
        )
        exec(compile(ast.Module(body=[load], type_ignores=[]), str(source), "exec"), scope)
        primary = object()
        scope["load_controlnet"]("local.safetensors", model=primary)
        read.assert_called_once_with("local.safetensors", safe_load=True)
        self.assertIs(forward.call_args.kwargs["model"], primary)

    def test_context_dimensions_match_exact_pinned_native_sd_configs(self):
        from test_native_model_coverage import NATIVE

        tree = ast.parse((NATIVE / "comfy/supported_models.py").read_text(encoding="utf-8"))
        for class_name, family in (("SD15", "SD15"), ("SD20", "SD20"), ("SDXL", "SDXL")):
            cls = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            config = ast.literal_eval(
                next(
                    node.value
                    for node in cls.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "unet_config"
                        for target in node.targets
                    )
                )
            )
            header = {
                "zero_convs.0.0.weight": {"shape": [320, 320, 1, 1]},
                "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": {
                    "shape": [320, config["context_dim"]]
                },
            }
            self.assertEqual(controls.classify_header(header)["families"], (family,))

    def test_filter_keeps_unknown_explicit_and_excludes_known_incompatible(self):
        header_file(
            self.roots[0] / "xl.safetensors",
            {
                "lora_controlnet": [],
                "input_blocks.1.1.transformer_blocks.0.attn2.to_k.down": [128, 2048],
            },
        )
        header_file(
            self.roots[0] / "other.safetensors", {"controlnet_blocks.0.linear.weight": [3072, 3072]}
        )
        (self.roots[0] / "imported.pth").write_bytes(b"not-read")
        rows = controls.local_candidates(self.manager, "SDXL", refresh=True)
        self.assertEqual(
            {row["name"]: row["compatibility"] for row in rows},
            {"xl.safetensors": "compatible", "imported.pth": "unknown"},
        )
        self.manager.preparation_request.assert_not_called()

    def test_capture_rejects_replacement_root_priority_change_and_escaped_names(self):
        name = "imported.safetensors"
        path = header_file(self.roots[0] / name, {"unknown": [1]})
        rows = controls.local_candidates(self.manager, "SDXL", refresh=True)
        proof = rows[0]["signature"]
        captured, signature, info = controls.selected_control(self.manager, name, "SDXL", proof)
        self.assertEqual(captured, path.resolve())
        self.assertEqual(signature, proof)
        self.assertEqual(info["families"], ())
        header_file(self.roots[1] / name, {"unknown": [1]})
        path.unlink()
        with self.assertRaisesRegex(ValueError, "changed"):
            controls.selected_control(self.manager, name, "SDXL", proof)
        for invalid in ("../escaped.safetensors", str(self.roots[1] / name), "missing.safetensors"):
            with self.assertRaises(ValueError):
                controls.selected_control(self.manager, invalid, "SDXL")
        self.manager.preparation_request.assert_not_called()

    def test_inventory_refresh_and_identity_do_not_read_weight_payload(self):
        path = header_file(self.roots[0] / "first.safetensors", {"unknown": [1]})
        rows = controls.local_candidates(self.manager, "SDXL", refresh=True)
        self.assertEqual(len(rows), 1)
        header_file(self.roots[0] / "new.safetensors", {"unknown": [2]})
        self.assertEqual(len(controls.local_candidates(self.manager, "SDXL")), 1)
        self.assertEqual(len(controls.local_candidates(self.manager, "SDXL", refresh=True)), 2)
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            controls.selected_control(self.manager, rows[0]["name"], "SDXL", rows[0]["signature"])

    def test_actual_asset_capture_keeps_selected_file_after_roots_change_and_rejects_replacement(
        self,
    ):
        from test_image_preparation import Manager
        from ruinedfooocus.features.generation.image_preparation import (
            capture_image_assets,
            prepared_asset,
            KEY,
        )

        root = self.roots[0]
        manager = Manager(root)
        selected = header_file(
            root / "controlnet/custom.safetensors",
            {"controlnet_blocks.0.linear.weight": [3072, 3072]},
        )
        primary = header_file(root / "checkpoints/image.safetensors", {"unknown": [1]})
        models = SimpleNamespace(
            get_file=lambda category, name: primary,
            detect_checkpoint_base=lambda path: "Flux",
            get_model_base=lambda metadata: "Flux",
            get_models_by_path=lambda *a, **kw: {},
        )
        data = dict(base_model_name=primary.name, input_image="owned source", image_engine="native")
        settings = dict(
            type="canny",
            model=selected.name,
            model_signature=controls.file_signature(selected),
            start=0,
            stop=1,
            strength=0.7,
        )
        with patch(
            "ruinedfooocus.features.models.component_profiles.component_inventory",
            return_value={"rows": []},
        ):
            captured = capture_image_assets(data, manager, settings, models=models)
        item = next(item for item in captured[KEY].artifacts if item.role == "controlnet")
        self.assertEqual(item.name, "canny")
        self.assertEqual(item.request.destination, selected.resolve())
        settings["model"] = "different.safetensors"
        manager.root = root / "changed"
        with prepared_asset(captured, "controlnet", "canny", manager, "controlnet") as path:
            self.assertEqual(path, selected.resolve())
        selected.write_bytes(selected.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            with prepared_asset(captured, "controlnet", "canny", manager, "controlnet"):
                self.fail("Changed selected model reached the native loader")

    def test_explicit_selection_rejects_unknown_primary_and_stock_automatic_on_other_family(self):
        from test_image_preparation import Manager
        from ruinedfooocus.features.generation.image_preparation import capture_image_assets

        manager = Manager(self.roots[0])
        model = header_file(self.roots[0] / "checkpoints/image.safetensors", {"unknown": [1]})
        selected = header_file(self.roots[0] / "controlnet/imported.safetensors", {"unknown": [1]})
        family = ["Flux"]
        models = SimpleNamespace(
            get_file=lambda category, name: model,
            detect_checkpoint_base=lambda path: family[0],
            get_model_base=lambda metadata: family[0],
            get_models_by_path=lambda *a, **kw: {},
        )
        data = dict(base_model_name=model.name, input_image="source", image_engine="native")
        with self.assertRaisesRegex(ValueError, "Choose a local control model"):
            capture_image_assets(data, manager, {"type": "canny"}, models=models)
        family[0] = "Unknown"
        with self.assertRaisesRegex(ValueError, "incompatible"):
            capture_image_assets(
                data, manager, {"type": "control", "model": selected.name}, models=models
            )
        with self.assertRaisesRegex(ValueError, "native image engine"):
            capture_image_assets(
                dict(data, image_engine="sdcpp-cpu"),
                manager,
                {"type": "control", "model": selected.name},
                models=models,
            )

    def test_api_control_defaults_are_checkpoint_scoped_and_never_applied_to_utilities(self):
        from test_api_image_edits import queued_data

        with patch(
            "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
            side_effect=lambda data, *a, **kw: data,
        ):
            selected = queued_data(
                "SDXL",
                controlnet={"type": "canny"},
                defaults_options={
                    "cn_model": "custom.safetensors",
                    "cn_model_signature": ["bound", 1, 2, 3],
                },
            )
            self.assertEqual(selected["controlnet"]["model"], "custom.safetensors")
            explicit_auto = queued_data(
                "SDXL",
                controlnet={"type": "canny", "model": ""},
                defaults_options={"cn_model": "custom.safetensors"},
            )
            self.assertEqual(explicit_auto["controlnet"]["model"], "")
            other = queued_data(
                "SDXL",
                controlnet={"type": "canny"},
                defaults_options={
                    "base_model_name": "different.safetensors",
                    "cn_model": "custom.safetensors",
                },
            )
            self.assertNotIn("model", other["controlnet"])
            utility = queued_data(
                "SDXL",
                controlnet={"type": "rembg"},
                defaults_options={"cn_model": "custom.safetensors"},
            )
            self.assertNotIn("model", utility["controlnet"])


if __name__ == "__main__":
    unittest.main()
