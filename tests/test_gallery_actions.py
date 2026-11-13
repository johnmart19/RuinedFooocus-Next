"""Library handoffs configure shared tools without loading weights or changing originals."""

import ast
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image
from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui

from ruinedfooocus.features.gallery.actions import (
    resolve_resource,
    resolve_setup,
    select_source,
    source_path,
    source_record,
)
from ruinedfooocus.features.gallery.image_metadata import read_metadata, restore_prompt_provenance

ROOT = Path(__file__).resolve().parents[1]


class Models:
    def __init__(self, root):
        self.files = {
            "checkpoints": {"new/model.safetensors": "a" * 64},
            "loras": {"detail.safetensors": "b" * 64},
        }
        self.root = root
        for files in self.files.values():
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"inert fixture, not model weights")

    def get_names(self, kind):
        return list(self.files[kind])

    def get_file(self, kind, name):
        path = self.root / name
        return path if name in self.files[kind] and path.is_file() else None

    def get_models_by_path(self, kind, name, fetch=False):
        assert not fetch
        return {"_rf_local_sha256": self.files[kind][name]}


def functions():
    tree = ast.parse((ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("METADATA_CONTROLS", "REFERENCE_RESETS")
    }

    class Control(str):
        @property
        def choices(self):
            return [(value, value) for value in ("euler", "normal")]

    namespace = dict(
        gr=gr,
        get_ctrl=Control,
        read_metadata=read_metadata,
        source_path=source_path,
        resolve_setup=resolve_setup,
        Image=Image,
        json=json,
        **constants,
    )
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("reuse_prompt", "reuse_setup", "prepare_image")
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "gallery actions", "exec"), namespace)
    return namespace


class GalleryActionsTests(unittest.TestCase):
    def test_preset_callbacks_share_complete_declared_outputs(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        outputs = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "preset_reset_outputs"
                for target in node.targets
            )
        )
        declared = {node.id for node in outputs.elts}
        for name in ("preset_unselect", "preset_image_upload", "preset_select"):
            callback = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            returned = next(
                node.value for node in ast.walk(callback) if isinstance(node, ast.Return)
            )
            expected = {node.id for node in returned.keys}
            self.assertEqual(
                expected, declared | ({"preset_image"} if name == "preset_select" else set())
            )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = {item.arg: item.value for item in node.keywords}
            callback = keywords.get("fn")
            if isinstance(callback, ast.Name) and callback.id in (
                "preset_image_upload",
                "preset_select",
            ):
                self.assertIn("preset_reset_outputs", ast.unparse(keywords["outputs"]))

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.path = self.root / "source.png"
        Image.new("RGBA", (73, 121), (20, 40, 60, 127)).save(self.path)
        self.record = source_record(self.path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.models = Models(self.root)

    def test_original_selection_is_immutable_and_rejects_replaced_deleted_or_stale_files(self):
        self.assertEqual(select_source([self.record], 0, self.record["caption"]), self.record)
        other = self.root / "other.png"
        other.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, "list changed"):
            select_source([source_record(other)], 0, self.record["caption"])
        self.path.write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "changed or was removed"):
            source_path(self.record)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "changed or was removed"):
            source_path(self.record)

    def test_resource_resolution_uses_cached_identity_and_reports_missing_or_mismatched_weights(
        self,
    ):
        self.assertEqual(
            resolve_resource(self.models, "checkpoints", "old/model.safetensors", "A" * 64),
            "new/model.safetensors",
        )
        self.assertEqual(
            resolve_resource(self.models, "checkpoints", "model.safetensors"),
            "new/model.safetensors",
        )
        for name, digest in (("missing.safetensors", None), ("new/model.safetensors", "c" * 64)):
            with self.assertRaisesRegex(ValueError, "Cannot uniquely match"):
                resolve_resource(self.models, "checkpoints", name, digest)

    def test_setup_restores_recorded_checkpoint_and_lora_weights_without_substitution(self):
        data = {
            "Prompt": "Two apples.",
            "base_model_name": "old/model.safetensors",
            "base_model_hash": "a" * 64,
            "loras": [["b" * 64, "-0.25 - detail.safetensors"]],
        }
        restored, checkpoint, loras = resolve_setup(data, self.models)
        self.assertEqual(restored["negative"], "")
        self.assertEqual(checkpoint, "new/model.safetensors")
        self.assertEqual(loras, [("detail.safetensors", -0.25)])
        data["loras"] = [["", "nan - detail.safetensors"]]
        with self.assertRaisesRegex(ValueError, "finite"):
            resolve_setup(data, self.models)

    def test_setup_replay_ignores_missing_lora_keywords_and_preserves_real_triggers(self):
        namespace = functions()
        namespace["apply_metadata"] = lambda *_: {}
        self.models.files["loras"]["other.safetensors"] = "c" * 64
        (self.root / "other.safetensors").write_bytes(b"inert fixture")
        data = {
            "Prompt": "two apples",
            "base_model_name": "new/model.safetensors",
            "loras": [["", "0.5 - detail.safetensors"], ["", "0.7 - other.safetensors"]],
        }
        keywords = Mock()
        modules = {
            "ruinedfooocus.app": SimpleNamespace(
                state=SimpleNamespace(
                    models=self.models, settings=SimpleNamespace(default_settings={})
                )
            ),
            "ruinedfooocus.core.util": SimpleNamespace(
                get_lora_thumbnail=lambda name: "thumbnail", load_keywords=keywords
            ),
        }
        # Avoid importing app.state: importing it binds app.state on the parent
        # package beyond patch cleanup and bypasses later isolated module fixtures.
        with patch.dict("sys.modules", modules):
            for values, expected in (
                ([" ", "\n\t"], ""),
                (["", "real trigger, detail"], "real trigger, detail"),
                (["first trigger", "last trigger"], "first trigger, last trigger"),
            ):
                with self.subTest(values=values):
                    keywords.side_effect = values
                    result = namespace["reuse_setup"](data)
                    self.assertEqual(result["lora_keywords"]["value"], expected)
                    self.assertEqual(
                        [item[1] for item in result["loras"]["value"]],
                        ["0.5 - detail.safetensors", "0.7 - other.safetensors"],
                    )

    def test_prompt_only_reuse_is_literal_and_clears_stale_image_conditioning(self):
        run = functions()["reuse_prompt"]
        data = {
            "Prompt": "first --- second __literal__",
            "Negative": "no blur",
            "raw_prompt": "two apples",
            "raw_negative": "no text",
        }
        result = run(data)
        self.assertEqual(result["prompt"]["value"], data["Prompt"])
        self.assertEqual(result["_imported_prompt"]["prompt"], data["Prompt"])
        self.assertIsNone(result["input_image"]["value"])
        self.assertFalse(result["inpaint_toggle"]["value"])
        self.assertEqual(result["style_selection"]["value"], [])
        self.assertEqual(run(data, True)["prompt"]["value"], "two apples")
        with self.assertRaisesRegex(gr.Error, "no original prompt"):
            run({"Prompt": "legacy"}, True)

    def test_full_setup_wires_resolved_model_loras_and_clears_old_references_atomically(self):
        scope = functions()
        apply = Mock(return_value={"prompt": gr.update(value="Two apples.")})
        scope["apply_metadata"] = apply
        metadata = {
            "Prompt": "Two apples.",
            "Negative": "",
            "base_model_name": "old/model.safetensors",
            "base_model_hash": "a" * 64,
            "loras": [["b" * 64, "1.25 - detail.safetensors"]],
            "steps": 30,
            "cfg": 3,
            "width": 512,
            "height": 768,
            "sampler_name": "euler",
            "scheduler": "normal",
            "image_component_overrides": {"clip_l": "recorded.safetensors", "vae_sdxl": None},
        }
        modules = {
            "ruinedfooocus.app": SimpleNamespace(
                state=SimpleNamespace(
                    models=self.models, settings=SimpleNamespace(default_settings={})
                )
            ),
            "ruinedfooocus.core.util": SimpleNamespace(
                get_lora_thumbnail=lambda name: "thumbnail",
                load_keywords=lambda name: "detail trigger",
            ),
        }
        with patch.dict("sys.modules", modules):
            result = scope["reuse_setup"](metadata)
        self.assertEqual(apply.call_args.args[1:], ("new/model.safetensors", None))
        self.assertEqual(result["base_model_name"]["value"], "new/model.safetensors")
        self.assertEqual(result["image_component_overrides"], metadata["image_component_overrides"])
        from ruinedfooocus.features.models.component_profiles import session_choices

        self.assertEqual(
            session_choices(
                self.models.get_file("checkpoints", "new/model.safetensors"),
                {},
                result["image_component_overrides"],
                result["_image_component_scope"],
            )[0],
            metadata["image_component_overrides"],
        )
        self.assertEqual(result["loras"]["value"], [("thumbnail", "1.25 - detail.safetensors")])
        self.assertEqual(result["lora_keywords"]["value"], "detail trigger")
        self.assertIsNone(result["input_image"]["value"])
        self.assertIsNone(result["preset_selection"]["value"])

    def test_full_setup_restores_qwen_local_mode_with_brush_and_reference_inputs_cleared(self):
        from test_image_metadata_boundary import gallery

        scope = functions()
        scope["apply_metadata"] = lambda metadata, *_: gallery(metadata)
        metadata = {
            "prompt": "Recolor the painted area.",
            "base_model_name": "old/model.safetensors",
            "base_model_hash": "a" * 64,
            "image_geometry": {"qwen_local_edit_mode": "mask"},
            "main_view": "private-source.png",
            "inpaint_view": {"layers": ["private-mask.png"]},
            "reference_images": ["private-reference.png"],
        }
        modules = {
            "ruinedfooocus.app": SimpleNamespace(
                state=SimpleNamespace(
                    models=self.models, settings=SimpleNamespace(default_settings={})
                )
            ),
            "ruinedfooocus.core.util": SimpleNamespace(
                get_lora_thumbnail=lambda name: "thumbnail", load_keywords=lambda _: ""
            ),
        }
        with patch.dict("sys.modules", modules):
            result = scope["reuse_setup"](metadata)
        self.assertEqual(result["qwen_local_edit_mode"]["value"], "mask")
        self.assertFalse(result["inpaint_toggle"]["value"])
        self.assertIsNone(result["inpaint_view"]["value"])
        self.assertIsNone(result["input_image"]["value"])
        self.assertEqual(result["reference_images"]["value"], [])
        self.assertNotIn("main_view", result)

    def test_replay_retains_original_provenance_until_prompt_or_processing_changes(self):
        imported = {
            "prompt": "styled apples",
            "negative": "noise",
            "raw_prompt": "apples",
            "raw_negative": "",
            "style_provenance": {"selected": ["Style: sai-anime"]},
        }
        for changed in (
            {},
            {"prompt": "new prompt"},
            {"negative": "new negative"},
            {"style_selection": ["Style: sai-cinematic"]},
            {"auto_negative": True},
        ):
            values = {"prompt": "styled apples", "negative": "noise", **changed}
            restore_prompt_provenance(values, imported)
            if not changed:
                self.assertTrue(values["_metadata_prompt"])
                self.assertEqual(values["raw_prompt"], "apples")
            else:
                self.assertNotIn("_metadata_prompt", values)

    def test_image_actions_keep_source_bytes_dimensions_and_transparency_and_never_start_work(self):
        run = functions()["prepare_image"]
        policy = SimpleNamespace(
            checkpoint_features=lambda *_: {
                "sdxl": False,
                "inpaint": True,
                "native_edit": False,
            },
            edit_tool=lambda _: "Img2Img",
            preset_for=lambda kind: "Remove background",
            tool_kind=lambda selection, _: "img2img" if selection == "Img2Img" else None,
            controlnet=SimpleNamespace(
                cn_options={"Img2Img": {"type": "img2img", "denoise": 0.25}}
            ),
        )
        with patch.dict("sys.modules", {"ruinedfooocus.features.image_editing.ui": policy}):
            for action in ("edit", "inpaint", "upscale", "rembg"):
                with self.subTest(action=action):
                    result = run(self.record, action, "model", "Disabled", "Off")
                    self.assertEqual(result["main_view"]["value"].size, (73, 121))
                    self.assertEqual(
                        result["main_view"]["value"].getpixel((0, 0)), (20, 40, 60, 127)
                    )
                    self.assertIsNone(result["_native_main_source"]["value"])
                    self.assertEqual(result["_generation_artifacts"]["value"], [])
                    self.assertNotIn("run_event", result)
                    if action == "edit":
                        self.assertEqual(result["cn_strength"]["value"], 0.25)
                    else:
                        self.assertNotIn("cn_strength", result)
                    self.assertNotIn("custom_width", result)
                    key = "upscale_upload" if action == "upscale" else "input_image"
                    if action != "inpaint":
                        self.assertEqual(
                            result[key]["value"].tobytes(), result["main_view"]["value"].tobytes()
                        )
            preserved = run(
                self.record, "edit", "model", "Disabled", "Off", "Img2Img", "Img2Img", 0.31
            )
            self.assertEqual(preserved["cn_strength"]["value"], 0.31)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.digest)
        with Image.open(self.path) as image:
            self.assertEqual(
                (image.size, image.mode, image.getpixel((0, 0))),
                ((73, 121), "RGBA", (20, 40, 60, 127)),
            )

    def test_inpaint_reports_unsupported_checkpoint_without_changing_controls(self):
        run = functions()["prepare_image"]
        policy = SimpleNamespace(
            checkpoint_features=lambda *_: {"sdxl": False, "inpaint": False},
            edit_tool=lambda _: "Img2Img",
            preset_for=lambda _: None,
            tool_kind=lambda *_: None,
            controlnet=SimpleNamespace(cn_options={}),
        )
        with patch.dict("sys.modules", {"ruinedfooocus.features.image_editing.ui": policy}):
            with self.assertRaisesRegex(gr.Error, "supported inpainting"):
                run(self.record, "inpaint", "anima", "Disabled", "Off")

    def test_page_filters_and_original_source_snapshots_do_not_leak_between_sessions(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/imagebrowser.py").read_text(encoding="utf-8")
        )
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ImageBrowser"
        )
        nodes = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("num_images_pages", "load_images")
        ]
        namespace = dict(
            Path=Path,
            save_file_to_cache=lambda path, _: path,
            get_upload_folder=lambda: str(self.root),
        )
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "imagebrowser pages", "exec"),
            namespace,
        )
        database = sqlite3.connect(":memory:")
        self.addCleanup(database.close)
        database.execute("CREATE TABLE images(fullpath text, path text, json text)")
        from ruinedfooocus.features.gallery.index_store import ensure_schema

        ensure_schema(database)
        for index in range(6):
            path = self.root / f"original{index}.png"
            path.write_bytes(self.path.read_bytes())
            database.execute(
                "INSERT INTO images(fullpath,path,json) VALUES (?,?,?)",
                (str(path), path.name, json.dumps({"prompt": f"subject{index}"})),
            )
        database.commit()
        browser = SimpleNamespace(sql_conn=database, images_per_page=2)
        browser.num_images_pages = lambda search="": namespace["num_images_pages"](browser, search)
        first, _, snapshot1 = namespace["load_images"](browser, 1, "subject0", include_paths=True)
        _, _, snapshot2 = namespace["load_images"](browser, 1, "subject1", include_paths=True)
        self.assertNotEqual(snapshot1[0]["path"], snapshot2[0]["path"])
        self.assertEqual(select_source(snapshot1, 0, first[0][1])["path"], snapshot1[0]["path"])
        self.assertEqual(browser.num_images_pages(), (6, 3))
        self.assertEqual(namespace["load_images"](browser, 2)[1], "Images 3–4 of 6")
        self.assertEqual(namespace["load_images"](browser, 1, "subject0")[1], "Images: 1")
        self.assertEqual(namespace["load_images"](browser, 1, "no matches")[1], "No images found")

    def test_pagination_controls_follow_filtered_page_boundaries(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "render_page"
        )
        browser = SimpleNamespace(
            load_images=lambda page, term, **kwargs: (
                [],
                "summary",
                [],
                {
                    "count": 1 if term else 6,
                    "pages": 1 if term else 3,
                    "page": min(max(1, int(page)), 1 if term else 3),
                },
            ),
        )
        scope = dict(
            browser=browser,
            gr=SimpleNamespace(update=lambda **values: values, Gallery=lambda **values: values),
            availability=lambda *_: {},
            localize=localize,
            resolve_locale=resolve_locale,
        )
        for name in (
            "gallery",
            "ib_page",
            "status_output",
            "page_records",
            "ib_range",
            "pagination",
            "page_total",
            "previous_page",
            "next_page",
            "selected",
            "metadata_json",
            "metadata_output",
        ):
            scope[name] = name
        exec(compile(ast.Module(body=[node], type_ignores=[]), "pagination", "exec"), scope)
        render = scope["render_page"]
        first = render(0, "")
        self.assertEqual(first["ib_page"]["value"], 1)
        self.assertFalse(first["previous_page"]["interactive"])
        self.assertTrue(first["next_page"]["interactive"])
        last = render(99, "")
        self.assertEqual(last["ib_page"]["value"], 3)
        self.assertFalse(last["next_page"]["interactive"])
        self.assertTrue(last["previous_page"]["interactive"])
        filtered = render(3, "one match")
        self.assertEqual(filtered["ib_page"]["value"], 1)
        self.assertTrue(filtered["pagination"]["visible"])
        self.assertFalse(filtered["previous_page"]["interactive"])
        self.assertFalse(filtered["next_page"]["interactive"])

    def test_refresh_clears_preview_through_real_gradio_update_serialization(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "render_page"
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "source.png"
            Image.new("RGB", (3, 4), "blue").save(image)
            values = [(str(image), "original caption")]
            with gr.Blocks() as app:
                gallery = gr.Gallery(value=values, selected_index=0, preview=True)
                button = gr.Button()
                button.click(lambda: None, outputs=gallery, api_visibility="undocumented")
            scope = dict(
                gr=gr,
                gallery=gallery,
                browser=SimpleNamespace(
                    load_images=lambda *args, **kwargs: (
                        values,
                        "summary",
                        [{"path": str(image)}],
                        {"count": 1, "pages": 1, "page": 1},
                    ),
                ),
                availability=lambda *_: {"actions_enabled": False},
                localize=localize,
                resolve_locale=resolve_locale,
            )
            for name in (
                "ib_page",
                "status_output",
                "page_records",
                "ib_range",
                "pagination",
                "page_total",
                "previous_page",
                "next_page",
                "selected",
                "metadata_json",
                "metadata_output",
            ):
                scope[name] = name
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "library_refresh", "exec"), scope
            )
            updates = scope["render_page"](1, "")
            state = SessionState(app)
            state[gallery._id] = gallery
            serialized = asyncio.run(
                app.postprocess_data(next(iter(app.fns.values())), updates[gallery], state)
            )[0]
            self.assertIn("selected_index", serialized)
            self.assertIsNone(serialized["selected_index"])
            self.assertFalse(serialized["preview"])
            self.assertEqual(serialized["value"][0]["caption"], "original caption")
            self.assertEqual(serialized["elem_classes"], [])
            self.assertIsNone(updates["selected"])
            self.assertEqual(updates["metadata_json"], {})
            self.assertEqual(updates["metadata_output"], "")
            self.assertFalse(updates["actions_enabled"])

    def test_pending_refresh_clears_client_preview_before_server_result(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "lock_selection"
        )
        namespace = {
            "json": json,
            "gallery": "gallery",
            "action_buttons": ["use"],
            "navigation_controls": ["refresh"],
            "previous_page": "previous",
            "next_page": "next",
            "metadata_output": "metadata",
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "pending_library_refresh", "exec"),
            namespace,
        )
        trigger = Mock()
        namespace["lock_selection"](trigger, reset=True)
        script = trigger.call_args.kwargs["js"]
        gallery_update = json.loads(script[script.index("[") + 1 : script.index(", ...Array")])
        self.assertNotIn("value", gallery_update)
        self.assertIsNone(gallery_update["selected_index"])
        self.assertFalse(gallery_update["preview"])
        namespace["lock_selection"](trigger)
        self.assertNotIn("selected_index", trigger.call_args.kwargs["js"])

    def test_programmatic_tool_handoff_preserves_denoise_but_user_preset_loads_it(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/image_editing/ui.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cn_changed"
        )
        namespace = dict(
            gr=SimpleNamespace(update=lambda **values: values),
            checkpoint_features=lambda *_: {"sdxl": True},
            controlnet=SimpleNamespace(
                cn_options={"Img2Img": {"type": "img2img", "denoise": 0.25}}
            ),
            NEWCN="Custom",
            cn_outputs=[1, 2],
            cn_sliders=list(range(6)),
            t=lambda text: text,
            translate_ui=translate_ui,
            SDXL_CONTROLS=set(),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "cn_changed", "exec"), namespace)
        programmatic = namespace["cn_changed"]("Img2Img", "Img2Img", "model", apply_values=False)
        deliberate = namespace["cn_changed"]("Img2Img", "Img2Img", "model")
        self.assertNotIn("value", programmatic[4])
        self.assertEqual(deliberate[4]["value"], 0.25)

    def test_failed_selection_unlocks_library_and_clears_previous_action_source(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        namespace = dict(
            gr=SimpleNamespace(
                update=lambda **values: values, SelectData=object, Request=object, Warning=Mock()
            ),
            select_source=select_source,
            source_path=source_path,
            read_metadata=read_metadata,
            action_buttons=[
                "prompt_btn",
                "setup_btn",
                "edit_btn",
                "inpaint_btn",
                "upscale_btn",
                "rembg_btn",
            ],
            navigation_controls=["refresh", "page", "search", "term"],
            selected="selected",
            metadata_json="metadata",
            metadata_output="details",
            gallery="gallery",
        )
        for name in ("availability", "select"):
            node = next(
                n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
            )
            exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
        deselected = namespace["select"](
            [self.record],
            SimpleNamespace(index=0, value={"caption": self.record["caption"]}, selected=False),
        )
        self.assertIsNone(deselected["selected"])
        self.assertFalse(deselected["edit_btn"]["interactive"])
        self.path.unlink()
        result = namespace["select"](
            [self.record], SimpleNamespace(index=0, value={"caption": self.record["caption"]})
        )
        self.assertIsNone(result["selected"])
        self.assertEqual(result["metadata"], {})
        self.assertEqual(result["gallery"], {"elem_classes": []})
        self.assertFalse(result["edit_btn"]["interactive"])
        self.assertTrue(result["refresh"]["interactive"])


if __name__ == "__main__":
    unittest.main()
