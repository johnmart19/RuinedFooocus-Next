"""Production Library setup callbacks serialize every restored field through Gradio."""

import ast
import asyncio
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.features.gallery.actions import resolve_setup, source_path, source_record
from ruinedfooocus.features.gallery.image_metadata import read_metadata
from ruinedfooocus.features.generation.cascade_guidance import (
    FIELDS as CASCADE_FIELDS,
    IMAGE_FIELD,
    replay_values as replay_cascade,
)
from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    capabilities,
    resolve_guidance,
    restore_guidance_metadata,
)
from ruinedfooocus.features.image_editing.model_patches import FIELDS as PATCH_FIELDS
from ruinedfooocus.features.image_editing.native_control_recipes import FIELDS as CONTROL_FIELDS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ruinedfooocus/features/gallery/ui.py"


def setup_fixture(random_seed=True, *, progress_owner="gallery"):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("METADATA_CONTROLS", "REFERENCE_RESETS")
    }
    names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_ctrl"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    names.update(constants["METADATA_CONTROLS"].values())
    names.update(constants["REFERENCE_RESETS"])
    names.update(FIELD_SCHEMA)
    names.update(PATCH_FIELDS)
    names.update(CONTROL_FIELDS)
    names.update(CASCADE_FIELDS)
    names.add(IMAGE_FIELD)

    def make_control(name):
        if name == "seed":
            return gr.Number(
                label="Seed",
                value=37,
                precision=0,
                visible="hidden" if random_seed else True,
                interactive=True,
            )
        if name == "seed_random":
            return gr.Checkbox(label="Random Seed", value=random_seed)
        if name in {
            "custom_steps",
            "cfg",
            "clip_skip",
            "custom_width",
            "custom_height",
            "cascade_control_strength",
            "cascade_control_start",
            "cascade_control_end",
            "cascade_control_denoise",
        }:
            return gr.Number(label=name)
        if name in {"cascade_super_resolution", "cascade_vision_enabled"}:
            return gr.Checkbox(label=name)
        if name in {
            "automatic_upscale",
            "output_bit_depth",
            "output_color_space",
            "native_edit_recipe",
        }:
            return gr.Dropdown(label=name, choices=[], allow_custom_value=True)
        return gr.Textbox(label=name)

    def get_ctrl(name):
        if name not in controls:
            controls[name] = make_control(name)
        return controls[name]

    with gr.Blocks(analytics_enabled=False) as app:
        with gr.Tabs() as top:
            with gr.Tab("Create", id="create"):
                with gr.Tabs() as create:
                    with gr.Tab("Generation", id="generation"):
                        controls = {name: make_control(name) for name in sorted(names)}
                        for name, choices in (
                            ("sampler_name", ["euler_ancestral", "dpmpp_2m"]),
                            ("scheduler", ["normal", "karras"]),
                        ):
                            controls[name] = gr.Dropdown(choices=choices, value=choices[-1])
                        controls[IMAGE_FIELD] = gr.Gallery(value=[])
                with gr.Tabs() as tools:
                    with gr.Tab("Image tools", id="image-tools"):
                        pass
            gr.Tab("Chat", id="chat")
            with gr.Tab("Library", id="library"):
                scope = {
                    name: gr.Button(name)
                    for name in (
                        "prompt_btn",
                        "setup_btn",
                        "edit_btn",
                        "inpaint_btn",
                        "upscale_btn",
                        "rembg_btn",
                        "grid_btn",
                        "composite_btn",
                        "continue_btn",
                    )
                }
                scope.update(
                    selected=gr.JSON(), metadata_json=gr.JSON(), prompt_version=gr.Textbox()
                )
        scope.update(
            constants,
            gr=gr,
            get_ctrl=get_ctrl,
            FIELD_SCHEMA=FIELD_SCHEMA,
            Image=Image,
            json=json,
            read_metadata=read_metadata,
            source_path=source_path,
            resolve_setup=resolve_setup,
            capabilities=capabilities,
            resolve_guidance=resolve_guidance,
            restore_guidance_metadata=restore_guidance_metadata,
            lock_selection=Mock(),
            t=lambda text: text,
            status_output=gr.Markdown(),
        )
        callbacks = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in ("apply_metadata", "reuse_setup", "reuse_prompt", "bind_actions")
        ]
        exec(compile(ast.Module(body=callbacks, type_ignores=[]), str(SOURCE), "exec"), scope)
        binding_options = {}
        scope["progress_status"] = scope["status_output"]
        if progress_owner == "workspace":
            scope.update(browser=object(), open_editor_btn=gr.Button())
            workspace = SimpleNamespace(
                context=gr.State(None),
                action=gr.State("generate"),
                notice=gr.Markdown(),
                bind_picker=Mock(),
                bind_library=Mock(),
            )
            binding_options["editor_workspace"] = workspace
        if progress_owner == "local":
            scope["browser_controls"] = gr.Column()
            local = SimpleNamespace(notice=gr.Markdown(), operation=gr.State("edit"), bind=Mock())
            scope["progress_status"] = local.notice
            binding_options.update(run_request=Mock(), run_controls={"task": gr.State(None)})
            with patch(
                "ruinedfooocus.features.gallery.ui_editing.LibraryEditor", return_value=local
            ):
                scope["bind_actions"](
                    top, create, lambda *_: {}, [], image_tools_tabs=tools, **binding_options
                )
        else:
            scope["bind_actions"](
                top, create, lambda *_: {}, [], image_tools_tabs=tools, **binding_options
            )
        app_tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        seed_callback = next(
            node
            for node in ast.walk(app_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "random_checked"
        )
        scope.update(seed_random=controls["seed_random"], image_seed=controls["seed"])
        exec(
            compile(ast.Module(body=[seed_callback], type_ignores=[]), "seed visibility", "exec"),
            scope,
        )
    event = next(
        (
            event
            for event in app.config["dependencies"]
            if event["targets"] == [(scope["setup_btn"]._id, "click")]
        ),
        None,
    )
    if progress_owner != "local":
        assert event is not None
    return app, event, controls, scope, top, create


class GallerySetupOutputTests(unittest.TestCase):
    def test_prepare_full_progress_uses_existing_owner_status_and_keeps_transport(self):
        for owner in ("workspace", "local", "gallery"):
            with self.subTest(owner=owner):
                app, _, controls, scope, *_ = setup_fixture(progress_owner=owner)
                functions = [
                    fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "prepare"
                ]
                self.assertEqual(len(functions), 2 if owner == "local" else 6)
                config = {row["id"]: row for row in app.config["dependencies"]}
                for function in functions:
                    self.assertEqual(config[function._id]["show_progress"], "full")
                    self.assertEqual(
                        config[function._id]["show_progress_on"], [scope["progress_status"]._id]
                    )
                prepare = scope["prepare_image"] = Mock(
                    side_effect=lambda *_: {
                        controls["main_view"]: gr.update(value="prepared source")
                    }
                )
                for button in ("grid_btn", "composite_btn"):
                    function = next(
                        fn for fn in functions if fn.targets == [(scope[button]._id, "click")]
                    )
                    result = asyncio.run(
                        app.process_api(
                            function._id,
                            [
                                {"path": "inert-source"},
                                "model",
                                "Disabled",
                                "Off",
                                "None",
                                "",
                                0.37,
                                None,
                            ],
                            state=SessionState(app),
                        )
                    )
                    updates = dict(zip(config[function._id]["outputs"], result["data"]))
                    self.assertTrue(prepare.called, result)
                    self.assertIn("value", updates[controls["main_view"]._id], prepare.call_args)
                    self.assertEqual(updates[controls["main_view"]._id]["value"], "prepared source")
                    self.assertEqual(
                        prepare.call_args.args[1],
                        "image-grid" if button == "grid_btn" else "image-composite",
                    )

    def test_metadata_without_seed_preserves_current_random_choice_and_seed_visibility(self):
        for random_seed in (True, False):
            with self.subTest(random_seed=random_seed):
                self.assert_full_setup({}, saved_seed=None, random_seed=random_seed)

    def test_zero_and_random_sentinel_seed_are_restored_without_replacement(self):
        for saved_seed in (0, -1):
            with self.subTest(seed=saved_seed):
                self.assert_full_setup({}, saved_seed=saved_seed)

    def test_manual_random_checkbox_only_updates_seed_visibility(self):
        app, _, controls, _, _, _ = setup_fixture()
        event = next(
            event
            for event in app.config["dependencies"]
            if app.fns[event["id"]].fn.__name__ == "random_checked"
        )
        state = SessionState(app)
        for random_seed in (True, False, True):
            result = asyncio.run(app.process_api(event["id"], [random_seed], state=state))
            update = result["data"][0]
            self.assertEqual(update["visible"], "hidden" if random_seed else True)
            self.assertNotIn("value", update)
            self.assertEqual(controls["seed"].value, 37)

    def test_current_native_replay_mappings_only_return_declared_setup_components(self):
        app, event, controls, _, _, _ = setup_fixture()
        declared = set(event["outputs"])
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        replay_modules = {
            node.module
            for function in tree.body
            if isinstance(function, ast.FunctionDef)
            and function.name in ("apply_metadata", "reuse_setup")
            for node in ast.walk(function)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "replay_values" for alias in node.names)
        }
        self.assertIn("ruinedfooocus.features.generation.cascade_guidance", replay_modules)
        for module_name in sorted(replay_modules):
            module = importlib.import_module(module_name)
            samples = [{}]
            if module_name.endswith("qwen_image21_union"):
                samples += [{"controlnet": {"type": module.KIND}}]
            elif module_name.endswith("qwen_image21_local_edit"):
                samples += [{"qwen_local_edit_mode": mode} for mode in module.LOCAL_EDIT_MODES]
            elif module_name.endswith("model_patches"):
                samples += [{"model_patch_recipe": recipe} for recipe in module.RECIPES]
            elif module_name.endswith("native_control_recipes"):
                samples += [{"control_recipe": recipe} for recipe in ("union", "alimama-inpaint")]
            elif module_name.endswith("native_tensor_io"):
                samples += [
                    {
                        "native_tensor_source": {"family": "SDXL", "inputs": [], "workflow": True},
                        "native_latent_denoise": 0.4,
                        "native_export_latent": True,
                        "native_export_conditioning": True,
                        "native_export_handoff": True,
                    }
                ]
            for metadata in samples:
                with self.subTest(module=module_name, metadata=metadata):
                    values = (
                        module.replay_values(metadata, {})
                        if module_name.endswith("native_tensor_io")
                        else module.replay_values(metadata)
                    )
                    missing = {
                        name
                        for name in values
                        if name not in controls or controls[name]._id not in declared
                    }
                    self.assertEqual(missing, set())
        self.assertIn(controls[IMAGE_FIELD]._id, declared)
        self.assertEqual(app.fns[event["id"]].fn.__name__, "use_setup")

    def test_plain_image_full_setup_serializes_sampling_resets_and_navigation(self):
        self.assert_full_setup({})

    def test_cascade_full_setup_serializes_guidance_and_clears_reference_images(self):
        self.assert_full_setup(
            dict(
                cascade_super_resolution=True,
                cascade_super_resolution_model="sr.safetensors",
                cascade_control_strength=0.7,
                cascade_vision_enabled=True,
                cascade_vision_model="vision.safetensors",
                cascade_vision_strengths=[0.5, 1.25],
                cascade_vision_crop="center",
            )
        )

    def assert_full_setup(self, extra, *, saved_seed=20261117, random_seed=True):
        app, event, controls, _, top, create = setup_fixture(random_seed=random_seed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            Image.new("RGB", (8, 12), "blue").save(original)
            original_bytes = original.read_bytes()
            model = root / "fixture.safetensors"
            model.write_bytes(b"inert identity fixture, no model load")
            metadata = dict(
                prompt="A saved literal prompt",
                negative="blur",
                base_model_name=model.name,
                base_model_hash="a" * 64,
                steps=30,
                cfg=3,
                sampler_name="euler_ancestral",
                scheduler="normal",
                clip_skip=2,
                width=832,
                height=1216,
                **extra,
            )
            if saved_seed is not None:
                metadata["seed"] = saved_seed
            shared = SimpleNamespace(
                models=SimpleNamespace(
                    get_names=lambda kind: [model.name] if kind == "checkpoints" else [],
                    get_file=lambda kind, name: model if name == model.name else None,
                    get_models_by_path=lambda *args, **kwargs: {"_rf_local_sha256": "a" * 64},
                ),
                settings=SimpleNamespace(default_settings={}),
            )
            modules = {
                "ruinedfooocus.app": SimpleNamespace(state=shared),
                "ruinedfooocus.core.util": SimpleNamespace(
                    get_lora_thumbnail=lambda *_: None, load_keywords=lambda *_: ""
                ),
            }
            with (
                patch.dict("sys.modules", modules),
                patch(
                    "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                    return_value="SDXL",
                ),
                patch(
                    "ruinedfooocus.features.models.component_profiles.replay_choices",
                    return_value=({}, None),
                ),
            ):
                state = SessionState(app)
                result = asyncio.run(
                    app.process_api(
                        event["id"],
                        [source_record(original), metadata, "1024×1024", None, None, None],
                        state=state,
                    )
                )
            projected = dict(zip(event["outputs"], result["data"]))
            if saved_seed is not None:
                self.assertEqual(projected[controls["seed"]._id]["value"], saved_seed)
                self.assertIs(projected[controls["seed"]._id]["visible"], True)
                self.assertFalse(projected[controls["seed_random"]._id]["value"])
                self.assertIs(state[controls["seed"]._id].visible, True)
                self.assertTrue(state[controls["seed"]._id].interactive)
            else:
                self.assertEqual(projected[controls["seed"]._id], gr.skip())
                self.assertEqual(projected[controls["seed_random"]._id], gr.skip())
                self.assertEqual(controls["seed"].value, 37)
                self.assertEqual(controls["seed_random"].value, random_seed)
                self.assertEqual(
                    state[controls["seed"]._id].visible, "hidden" if random_seed else True
                )
            for name, expected in dict(
                prompt=metadata["prompt"],
                custom_steps=30,
                cfg=3,
                sampler_name="euler_ancestral",
                scheduler="normal",
                clip_skip=2,
                custom_width=832,
                custom_height=1216,
            ).items():
                self.assertEqual(projected[controls[name]._id]["value"], expected)
            for name, value in replay_cascade(extra).items():
                self.assertIn(controls[name]._id, event["outputs"])
                expected = str(value) if name == "cascade_vision_strengths" else value
                self.assertEqual(projected[controls[name]._id]["value"], expected)
            self.assertEqual(projected[controls[IMAGE_FIELD]._id]["value"], [])
            self.assertEqual(projected[top._id]["selected"], "create")
            self.assertEqual(projected[create._id]["selected"], "generation")
            self.assertEqual(original.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
