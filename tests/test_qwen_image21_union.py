"""Qwen Image 2.1 Union asset routing and pinned ComfyUI invocation on CPU."""

from contextlib import nullcontext
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import gradio as gr
import torch

from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_editing import qwen_image21_union as union
from ruinedfooocus.features.image_editing.ui_qwen_union import create_ui
from ruinedfooocus.integrations.openai_api import ImageRequest
from test_api_image_edits import queued_data
from test_image_preparation import Manager
from test_gallery_actions import Models, functions as gallery_functions
from test_qwen_image21_conditioning import native_class, native_function


def fixture_header():
    return {
        "control_img_in.weight": {"shape": [4096, 129]},
        "control_blocks.0.img_mlp.out.weight": {"shape": [4096, 12288]},
        "control_blocks.0.attn.norm_q.weight": {"shape": [128]},
        **{f"control_blocks.{i}.after_proj.weight": {"shape": [4096, 4096]} for i in range(16)},
    }


class UnionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.manager = Manager(Path(temp.name).resolve())
        self.manager.add("union", union.CATEGORY, union.DEFAULT_MODEL)
        self.path = self.manager.root / union.CATEGORY / union.DEFAULT_MODEL
        self.path.parent.mkdir()
        self.guide = Image.new("RGB", (64, 48), (70, 140, 0))
        self.source = Image.new("RGB", (64, 48), (255, 0, 0))
        self.mask = Image.new("L", (64, 48), 0)
        self.mask.paste(255, (0, 0, 32, 48))

    def write_header(self, header):
        encoded = json.dumps(header).encode("utf-8")
        self.path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)

    def capture(self, mode, **data):
        return assets.capture_image_assets(
            dict(qwen_union_guide=self.guide, **data),
            self.manager,
            {"type": union.KIND, "mode": mode},
        )

    def test_all_guide_modes_capture_a_distinct_typed_asset(self):
        for mode in union.GUIDE_MODES:
            with self.subTest(mode=mode):
                data = self.capture(mode, reference_images=[self.guide])
                self.assertIsNone(data["reference_images"])
                self.assertEqual(data["controlnet"]["mode"], mode)
                self.assertIsNot(data["qwen_union_guide"], self.guide)
                pixel = data["qwen_union_guide"].getpixel((0, 0))
                self.assertEqual(len(set(pixel)), 1 if mode == "grayscale" else 3)
                (item,) = data[assets.KEY].artifacts
                self.assertEqual((item.role, item.name), (union.ROLE, union.DEFAULT_MODEL))
                self.assertEqual(item.request.destination, self.path)
                self.assertIsNone(item.signature)

    def test_local_patch_is_typed_and_replaced_files_are_rejected(self):
        self.write_header(fixture_header())
        data = self.capture("pose")
        self.assertIsNotNone(data[assets.KEY].artifacts[0].signature)
        self.assertFalse(assets.needs_image_preparation(data))
        self.write_header({"not-a-patch": {"shape": [1]}})
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            with assets.prepared_asset(
                data, union.ROLE, union.DEFAULT_MODEL, self.manager, union.CATEGORY
            ):
                self.fail("Native loading opened a replaced file.")
        with self.assertRaisesRegex(ValueError, "safetensors patch"):
            self.capture("pose")
        with self.assertRaisesRegex(ValueError, "configured model patches"):
            union.settings({"model": "../patch.safetensors"})

    def test_inpaint_and_combined_requests_use_black_keep_white_redraw(self):
        data = self.capture("inpaint", qwen_union_source=self.source, qwen_union_mask=self.mask)
        self.assertIsNone(data["qwen_union_guide"])
        combined = self.capture("pose", qwen_union_source=self.source, qwen_union_mask=self.mask)
        self.assertEqual(combined["qwen_union_guide"].size, self.source.size)
        for x, expected in ((0, 255), (63, 0)):
            self.assertEqual(combined["qwen_union_mask"].getpixel((x, 0)), expected)
        with self.assertRaisesRegex(ValueError, "both an inpaint"):
            self.capture("pose", qwen_union_source=self.source)
        with self.assertRaisesRegex(ValueError, "mark the edit area"):
            self.capture(
                "inpaint",
                qwen_union_source=self.source,
                qwen_union_mask=Image.new("L", self.source.size, 0),
            )
        with self.assertRaisesRegex(ValueError, "native image engine"):
            self.capture("pose", image_engine="sdcpp-cpu")

    def test_pinned_native_node_registers_a_request_clone_with_the_right_sigma_window(self):
        self.write_header(fixture_header())
        data = self.capture("pose", qwen_union_source=self.source, qwen_union_mask=self.mask)
        data["controlnet"].update(start=0.2, stop=0.8, strength=0.65)

        class Transformer:
            transformer_blocks = list(range(32))

        class Control:
            control_blocks = list(range(16))

        native_patch = SimpleNamespace(model=Control())
        loader = Mock(return_value=(native_patch,))
        clone = Mock(model_options={})
        sampling = SimpleNamespace(percent_to_sigma=lambda pct: 10 * (1 - pct))
        model = SimpleNamespace(
            get_model_object={
                "diffusion_model": Transformer(),
                "model_sampling": sampling,
            }.__getitem__,
            clone=Mock(return_value=clone),
        )
        clone.get_model_object = model.get_model_object
        vae = SimpleNamespace(
            latent_channels=64,
            output_channels=4,
            spacial_compression_encode=lambda: 16,
            encode=Mock(
                side_effect=lambda image: torch.zeros(
                    1, 64, image.shape[1] // 16, image.shape[2] // 16
                )
            ),
        )
        comfy = SimpleNamespace(
            ldm=SimpleNamespace(
                qwen_image21=SimpleNamespace(model=SimpleNamespace(QwenImage21FunControl=Control))
            ),
            latent_formats=SimpleNamespace(
                QwenImage21=lambda: SimpleNamespace(latent_channels=64, process_in=lambda x: x)
            ),
            model_management=SimpleNamespace(loaded_models=lambda **_: [], load_models_gpu=Mock()),
            patcher_extension=SimpleNamespace(
                WrappersMP=SimpleNamespace(DIFFUSION_MODEL="diffusion")
            ),
            utils=SimpleNamespace(
                common_upscale=native_function("comfy/utils.py", "common_upscale", {"torch": torch})
            ),
        )
        scope = {"torch": torch, "comfy": comfy}
        native_file = "comfy_extras/nodes_model_patch.py"
        native_class(
            native_file,
            "QwenImage21FunControlPatch",
            scope,
            methods=("__init__", "register", "prepare", "diffusion_model_wrapper"),
        )
        native_class(native_file, "QwenImage21FunControlBlockPatch", scope, methods=("__init__",))
        applier = native_class(
            native_file, "QwenImageDiffsynthControlnet", scope, methods=("diffsynth_controlnet",)
        )
        modules = {
            "folder_paths": SimpleNamespace(
                get_full_path_or_raise=Mock(return_value=str(self.path))
            ),
            "comfy.ldm.qwen_image21.model": SimpleNamespace(
                QwenImage21FunControl=Control, QwenImage21Transformer2DModel=Transformer
            ),
            "comfy_extras.nodes_model_patch": SimpleNamespace(
                ModelPatchLoader=lambda: SimpleNamespace(load_model_patch=loader),
                ZImageFunControlnet=applier,
            ),
        }
        with (
            patch.dict("sys.modules", modules),
            patch.object(assets, "prepared_asset", return_value=nullcontext(self.path)),
        ):
            result = union.apply_union(model, vae, data, self.manager)
            self.assertIs(union.apply_union(model, vae, data, self.manager), result)
        self.assertEqual(data["controlnet"]["model_signature"], union.file_signature(self.path))
        self.assertIs(result, clone)
        loader.assert_called_once_with(union.DEFAULT_MODEL)
        patch_fn = clone.add_wrapper.call_args.args[1].__self__
        self.assertEqual(patch_fn.injection_layers, list(range(0, 32, 2)))
        self.assertAlmostEqual(patch_fn.sigma_range[0], 8.0)
        self.assertAlmostEqual(patch_fn.sigma_range[1], 2.0)
        self.assertEqual(patch_fn.strength, 0.65)
        patch_fn.prepare(3, 4)
        self.assertEqual(patch_fn.control.shape, (1, 129, 3, 4))
        self.assertTrue(torch.equal(patch_fn.control[0, 64, 0], torch.tensor([0, 0, 1, 1])))
        masked = vae.encode.call_args.args[0]
        self.assertTrue(torch.equal(masked[0, 0, 0], torch.tensor([0.5, 0.5, 0.5])))
        self.assertTrue(torch.equal(masked[0, 0, -1], torch.tensor([1.0, 0.0, 0.0])))

    def test_http_schema_distinguishes_union_from_the_local_mask_edit(self):
        payload = {"prompt": "red dress", "controlnet": {"type": "qwen-union", "mode": "inpaint"}}
        with self.assertRaisesRegex(ValueError, "inpaint source"):
            ImageRequest.model_validate(payload)
        request = ImageRequest.model_validate(
            dict(payload, qwen_union_source="source", qwen_union_mask="mask")
        )
        self.assertEqual(request.controlnet.mode, "inpaint")
        with self.assertRaisesRegex(ValueError, "own image operation"):
            ImageRequest.model_validate(
                dict(
                    payload,
                    qwen_union_source="source",
                    qwen_union_mask="mask",
                    input_image="source",
                    mask="other",
                )
            )

    def test_api_runtime_decodes_union_sources_without_ordinary_edit_references(self):
        with (
            patch(
                "ruinedfooocus.features.models.component_profiles.snapshot_request",
                side_effect=lambda data, *_: data,
            ),
            patch.object(
                assets, "capture_image_assets", side_effect=lambda data, *_args, **_kw: data
            ),
        ):
            data = queued_data(
                "QwenImage21",
                controlnet={"type": union.KIND, "mode": "pose"},
                payload_options={"qwen_union_source": "source", "qwen_union_mask": "mask"},
            )
        self.assertIsNone(data["input_image"])
        self.assertEqual(data["controlnet"]["model"], union.DEFAULT_MODEL)
        self.assertTrue(
            all(
                isinstance(data[key], Image.Image)
                for key in ("qwen_union_guide", "qwen_union_source", "qwen_union_mask")
            )
        )

    def test_gradio_build_exposes_all_modes_and_upload_fields(self):
        fields = {}
        self.manager.get_folder_list = lambda _category: [union.DEFAULT_MODEL]
        shared = SimpleNamespace(add_ctrl=fields.__setitem__, path_manager=self.manager)
        with patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}), gr.Blocks() as app:
            create_ui(
                gr.Dropdown(["None", "Union"], value="None"),
                gr.Dropdown(["Edit"], value="Edit"),
                gr.Dropdown(["QwenImage21"], value="QwenImage21"),
                None,
                lambda preset, _kind: union.KIND if preset == "Union" else "",
                lambda name: {"qwen_union": name == "QwenImage21"},
            )
        self.assertEqual(
            {key.removeprefix("qwen_union_") for key in fields},
            {"model", "model_signature", "mode", "guide", "source", "mask"},
        )
        self.assertEqual(
            [value for _label, value in fields["qwen_union_mode"].choices], list(union.MODES)
        )
        self.assertEqual(fields["qwen_union_mask"].image_mode, "L")
        functions = [entry.fn for entry in app.fns.values()]
        context = next(fn for fn in functions if fn.__name__ == "context")
        mode_change = next(fn for fn in functions if fn.__name__ == "<lambda>")
        self.assertFalse(context("Union", "Edit", "SDXL", None)[0]["visible"])
        self.assertEqual(
            context("Union", "Edit", "QwenImage21", None)[1]["value"], union.DEFAULT_MODEL
        )
        self.assertFalse(mode_change("inpaint")[0]["visible"])
        self.assertTrue(mode_change("inpaint")[1]["open"])

    def test_png_writer_keeps_applied_recipe_and_replay_retains_the_original_uploads(self):
        self.write_header(fixture_header())
        captured = self.capture("lineart")
        recipe = dict(captured["controlnet"], strength=0.6, start=0.15, stop=0.8)
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        block = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "(gen_data.get('controlnet') or {}).get('type') == 'qwen-union'"
        )
        scope = {
            "gen_data": dict(captured, controlnet=dict(recipe, image=self.guide)),
            "prompt": {"Prompt": "red dress"},
        }
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"), scope)
        self.assertEqual(scope["prompt"]["controlnet"], recipe)
        info = PngInfo()
        info.add_text("parameters", json.dumps(scope["prompt"]))
        output = self.manager.root / "result.png"
        self.guide.save(output, pnginfo=info)
        with Image.open(output) as image:
            metadata = json.loads(image.info["parameters"])
        api_request = ImageRequest.model_validate(
            dict(prompt="red dress", controlnet=metadata["controlnet"], qwen_union_guide="guide")
        )
        self.assertEqual(list(api_request.controlnet.model_signature), recipe["model_signature"])
        replayed = assets.image_control_data(
            dict(prompt=json.dumps(metadata), qwen_union_guide=self.guide)
        )
        self.assertIs(replayed["qwen_union_guide"], self.guide)
        self.assertEqual(replayed["controlnet"], recipe)
        self.assertEqual(replayed["qwen_union_model_signature"], recipe["model_signature"])
        self.assertEqual(
            (replayed["cn_strength"], replayed["cn_start"], replayed["cn_stop"]), (0.6, 0.15, 0.8)
        )
        self.assertEqual(replayed["qwen_union_mode"], "lineart")
        self.path.write_bytes(self.path.read_bytes() + b"replaced")
        with self.assertRaisesRegex(ValueError, "selected Union patch changed"):
            assets.capture_image_assets(replayed, self.manager, recipe)

    def test_library_setup_restores_union_controls_and_clears_separate_uploads(self):
        self.write_header(fixture_header())
        recipe = self.capture("scribble")["controlnet"]
        models = Models(self.manager.root)
        scope = gallery_functions()
        scope["apply_metadata"] = lambda *_: {}
        metadata = {
            "prompt": "red dress",
            "base_model_name": "new/model.safetensors",
            "loras": [],
            "controlnet": recipe,
        }
        modules = {
            "ruinedfooocus.app": SimpleNamespace(
                state=SimpleNamespace(models=models, settings=SimpleNamespace(default_settings={}))
            ),
            "ruinedfooocus.core.util": SimpleNamespace(
                get_lora_thumbnail=lambda _name: "thumbnail", load_keywords=lambda _name: ""
            ),
        }
        with patch.dict("sys.modules", modules):
            updates = scope["reuse_setup"](metadata)
        self.assertEqual(updates["cn_selection"]["value"], "Qwen Image 2.1 Union")
        self.assertEqual(updates["qwen_union_mode"]["value"], "scribble")
        self.assertEqual(updates["qwen_union_model_signature"]["value"], recipe["model_signature"])
        self.assertTrue(
            all(
                updates[key]["value"] is None
                for key in ("qwen_union_guide", "qwen_union_source", "qwen_union_mask")
            )
        )
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/gallery/ui.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        binding = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "metadata_names"
                for target in node.targets
            )
        )
        # The field lists are imported just above the binding in the same function.
        owner = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and binding in node.body
        )
        imports = [
            node
            for node in owner.body[: owner.body.index(binding)]
            if isinstance(node, ast.ImportFrom)
        ]
        namespace = {"METADATA_CONTROLS": {}, "FIELD_SCHEMA": {}}
        exec(
            compile(ast.Module(body=[*imports, binding], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self.assertTrue(set(union.replay_values(metadata)) <= set(namespace["metadata_names"]))


if __name__ == "__main__":
    unittest.main()
