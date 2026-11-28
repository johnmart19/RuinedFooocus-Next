"""The image picker and public API share the local model inventory."""

import os
from pathlib import Path
import queue
import runpy
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.features.models.component_profiles import snapshot_request
from ruinedfooocus.features.models.local_diffusers import BUNDLE_KEY, DIFFUSION_FILES, TEXT_FILES
from ruinedfooocus.features.models.selection import checkpoint_records
from ruinedfooocus.integrations.openai_api import create_apps


ROOT = Path(__file__).resolve().parents[1]


class ApiModelDiscoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="rf-api-private-library-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.write("models/upscale_models/nested/pixel.SFT")
        self.write("models/upscale_models/4x-latent.GGUF")
        self.write("models/loras/nested/style.PT")
        self.write("models/loras/not-a-native-lora.GGUF")
        with (
            patch.dict(os.environ, {"RF_DATA_DIR": str(self.root)}),
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.arguments": SimpleNamespace(
                        args=SimpleNamespace(settings=None)
                    )
                },
            ),
            patch.object(PathManager, "DOWNLOADABLE_FILES", {}),
        ):
            self.manager = PathManager()
            self.manager.DOWNLOADABLE_FILES = self.manager.DOWNLOADABLE_FILES.copy()
        self.bundle_roots = [
            self.root / name for name in ("private-library-one", "private-library-two")
        ]
        self.manager.paths["path_diffusers"] = self.bundle_roots.copy()
        self.state = SimpleNamespace(
            path_manager=self.manager,
            settings=SimpleNamespace(default_settings={}),
            translate=lambda text, **_: text,
        )
        chat_path = self.write("models/llm/local-chat.gguf")
        modules = patch.dict(
            sys.modules,
            {
                "ruinedfooocus.app.state": self.state,
                "ruinedfooocus.features.chat.llama_models": SimpleNamespace(
                    DEFAULT_MODEL=chat_path.name, local_models=lambda: {chat_path.name: chat_path}
                ),
                "ruinedfooocus.features.chat.llama_vision": SimpleNamespace(
                    vision_models=lambda _: []
                ),
            },
        )
        state = patch.object(sys.modules["ruinedfooocus.app"], "state", self.state, create=True)
        defaults = patch.object(api_runtime, "image_defaults", {})
        for patcher in (modules, state, defaults):
            patcher.start()
            self.addCleanup(patcher.stop)
        models = runpy.run_path(str(ROOT / "ruinedfooocus/features/models/model_handler.py"))[
            "Models"
        ]
        self.state.models = models(offline=True)
        self.state.models.detect_checkpoint_base = Mock(return_value=None)

    def write(self, path, content=b"local candidate fixture"):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def bundle(self, folder):
        index = self.write(
            folder / "model_index.json", b'{"_class_name":"arbitrary.external.code"}'
        )
        self.write(folder / "unet" / DIFFUSION_FILES[0])
        self.write(folder / "vae" / DIFFUSION_FILES[0])
        self.write(folder / "text_encoder" / TEXT_FILES[0])
        return index

    def picker_records(self):
        models = self.state.models
        return checkpoint_records(
            models.get_names("checkpoints"),
            lambda name: models.get_models_by_path("checkpoints", name),
        )

    def test_registry_candidates_keep_existing_file_ids_and_model_roles(self):
        names = [
            "nested/weights" + suffix
            for suffix in (
                ".safetensors",
                ".SFT",
                ".ckpt",
                ".PTH",
                ".bin",
                ".PT",
                ".pt2",
                ".PKL",
                ".gguf",
            )
        ]
        for name in (*names, "nested/notes.json"):
            self.write("models/checkpoints/" + name)
        self.state.models.update_all_models()

        catalogue = api_runtime.models()
        self.assertEqual(
            {name for name in catalogue if name.startswith("image/")},
            {"image/" + name for name in names},
        )
        self.assertEqual(
            {api_runtime.model_id(record["name"]) for record in self.picker_records()}, set(names)
        )
        for name in names:
            public_id, selected = api_runtime.resolve_model(
                ("image/" + name).replace("/", "\\"), "image"
            )
            self.assertEqual(public_id, "image/" + name)
            self.assertEqual(selected["name"], str(Path(name)))
        self.assertEqual(self.state.models.get_names("loras"), [str(Path("nested/style.PT"))])
        self.assertIn("upscaler/nested/pixel.SFT", catalogue)
        self.assertIn("upscaler/4x-UltraSharp.pth", catalogue)
        self.assertNotIn("upscaler/4x-latent.GGUF", catalogue)
        self.assertIn("chat/local-chat.gguf", catalogue)
        with self.assertRaisesRegex(ValueError, "Unknown local image model"):
            api_runtime.resolve_model("upscaler/nested/pixel.SFT", "image")

    def test_http_resolves_raw_image_aliases_before_retired_namespace_admission(self):
        from ruinedfooocus.features.models.image_scope import require_image_operation

        for name in ("video/foo.safetensors", "3d/foo.safetensors"):
            self.write("models/checkpoints/" + name)
        self.state.models.update_all_models()
        captured = []

        def images(payload, selected):
            require_image_operation(payload)
            captured.append(payload["model"])
            events = queue.Queue()
            events.put(("result", []))
            return SimpleNamespace(events=events, cancelled=threading.Event())

        backend = SimpleNamespace(
            models=api_runtime.models, resolve_model=api_runtime.resolve_model, images=images
        )
        api, _ = create_apps(backend, "secret")
        app = FastAPI()
        app.mount("/v1", api)
        with TestClient(app, headers={"Authorization": "Bearer secret"}) as client:
            for name in ("video/foo.safetensors", "3d/foo.safetensors"):
                for alias in (name, name.replace("/", "\\"), "image/" + name):
                    response = client.post(
                        "/v1/images/generations", json={"model": alias, "prompt": "fixture"}
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(captured[-1], "image/" + name)
            before = len(captured)
            response = client.post(
                "/v1/images/generations", json={"model": "video/default", "prompt": "fixture"}
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(len(captured), before)

    def test_typed_gguf_lora_reaches_actual_browser_capabilities_and_api_admission(self):
        import gguf
        import gradio as gr
        import numpy as np
        from ruinedfooocus.features.models import ui_adapters

        path = self.write("models/loras/nested/custom.gGuF")
        writer = gguf.GGUFWriter(path, "custom-image-adapter")
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_tensor("linear.weight.lora_a", np.zeros((2, 4), dtype=np.float32))
        writer.add_tensor("linear.weight.lora_b", np.zeros((3, 2), dtype=np.float16))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self.state.models.update_all_models()
        self.state.performance_settings = SimpleNamespace(performance_options=[])
        with (
            patch.dict(
                sys.modules,
                {
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"])
                    ),
                    "ruinedfooocus.features.prompting.sdxl_styles": SimpleNamespace(
                        load_styles=dict
                    ),
                    "ruinedfooocus.core.util": SimpleNamespace(
                        _get_model_thumbnail=lambda *_, **__: None
                    ),
                },
            ),
            patch.object(api_runtime, "models", return_value={}),
            patch.object(ui_adapters, "t", side_effect=lambda text: text),
        ):
            api, _ = create_apps(api_runtime, "secret")
            app = FastAPI()
            app.mount("/v1", api)
            with TestClient(app, headers={"Authorization": "Bearer secret"}) as client:
                response = client.get("/v1/capabilities")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                set(response.json()["loras"]), {"nested/style.PT", "nested/custom.gGuF"}
            )
            with gr.Blocks():
                controls = ui_adapters.create_ui(
                    gr.Textbox(),
                    gr.Textbox(),
                    models=self.state.models,
                    directories=[],
                    cache_dir=self.root / "cards",
                    defaults=[],
                    thumbnail=lambda _: None,
                    keywords=lambda _: "",
                )
            cards = controls.catalog.value
            self.assertEqual(
                {api_runtime.model_id(card["name"]) for card in cards},
                {"nested/style.PT", "nested/custom.gGuF"},
            )
        selected = api_runtime.resolve_image_loras(
            [
                {"name": "nested/custom.gGuF", "strength": -0.5},
            ]
        )
        self.assertEqual(selected[0][1], f"-0.5 - {Path('nested/custom.gGuF')}")
        with self.assertRaisesRegex(ValueError, "Unknown local LoRA"):
            api_runtime.resolve_image_loras(
                [
                    {"name": "not-a-native-lora.GGUF", "strength": 1},
                ]
            )
        # The API rechecks a once-admitted file when it has changed since Refresh.
        path.write_bytes(b"replaced file")
        with self.assertRaises(ValueError):
            api_runtime.resolve_image_loras(
                [
                    {"name": "nested/custom.gGuF", "strength": 1},
                ]
            )

    def test_named_sensenova_index_picker_api_snapshot_and_preparation_share_all_shards(self):
        from test_sensenova_bundle import write_bundle
        from ruinedfooocus.features.models import sensenova_bundle as sense
        from ruinedfooocus.features.models.component_profiles import component_inventory
        from ruinedfooocus.features.generation.image_preparation import (
            KEY as ASSETS_KEY,
            capture_image_assets,
            prepare_image_assets,
        )

        index = write_bundle(self.root / "models/diffusion_models/SenseNova")
        self.state.models.update_all_models()
        name = str(Path("SenseNova") / sense.INDEX_NAME)
        self.assertEqual(self.state.models.get_names("checkpoints"), [name])
        self.assertEqual({row["name"] for row in self.picker_records()}, {name})
        public = "image/SenseNova/" + sense.INDEX_NAME
        _, selected = api_runtime.resolve_model(public, "image")
        self.assertEqual(selected["family"], "SenseNovaU15")
        data = snapshot_request(
            {"base_model_name": selected["name"], "_api_job": True},
            self.state.models,
            {},
            self.manager,
        )
        bundle = data[sense.KEY]
        self.assertEqual(len(bundle.files), 9)
        inventory = component_inventory(index, "SenseNovaU15", {}, self.manager)
        self.assertEqual(inventory["bundle_identity"], bundle.identity)
        self.assertEqual(inventory["rows"], [])
        data = capture_image_assets(data, self.manager, {}, models=self.state.models, offline=True)
        sources = [item for item in data[ASSETS_KEY].artifacts if item.role == "sensenova-bundle"]
        self.assertEqual(
            {item.signature for item in sources}, {item.signature for item in bundle.files}
        )
        prepared = prepare_image_assets(data, check_cancelled=lambda: None)
        self.assertEqual(prepared[sense.KEY].identity, bundle.identity)
        # Index selection remains fixed when a shard changes behind the queued request.
        bundle.shards[-1].path.write_bytes(b"replaced shard")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_image_assets(data, check_cancelled=lambda: None)
        self.assertIn("unavailable_reason", api_runtime.models()[public])
        with self.assertRaises(ValueError):
            api_runtime.resolve_model(public, "image")

    def test_incomplete_and_foreign_eight_shard_indexes_stay_visible_without_partial_models(self):
        import json
        import torch
        from safetensors.torch import save_file
        from test_sensenova_bundle import write_bundle
        from ruinedfooocus.features.models import sensenova_bundle as sense

        for case in ("incomplete", "foreign"):
            with self.subTest(case=case):
                index = write_bundle(self.root / "models/diffusion_models" / case)
                if case == "incomplete":
                    (index.parent / sense.SHARD_NAMES[-1]).unlink()
                else:
                    first = index.parent / sense.SHARD_NAMES[0]
                    save_file(
                        {
                            "foreign.vision.weight": torch.zeros(
                                (1024, 1, 1, 1), dtype=torch.bfloat16
                            )
                        },
                        first,
                        metadata={"format": "pt"},
                    )
                    document = json.loads(index.read_text())
                    document["weight_map"]["foreign.vision.weight"] = document["weight_map"].pop(
                        sense.VISION
                    )
                    index.write_text(json.dumps(document))
                with patch.object(sense, "capture_bundle", wraps=sense.capture_bundle) as capture:
                    self.state.models.update_all_models()
                    catalogue = api_runtime.models()
                    self.picker_records()
                    self.state.models.get_models_by_path("checkpoints", index)
                    self.assertEqual(capture.call_count, 1, capture.call_args_list)
                name = str(Path(case) / sense.INDEX_NAME)
                self.assertIn(name, self.state.models.get_names("checkpoints"))
                self.assertFalse(
                    any(
                        Path(item).name in sense.SHARD_NAMES
                        for item in self.state.models.get_names("checkpoints")
                    )
                )
                public = "image/" + case + "/" + sense.INDEX_NAME
                self.assertEqual(catalogue[public]["family"], "Unknown")
                self.assertTrue(catalogue[public]["unavailable_reason"])
                with self.assertRaises(ValueError):
                    api_runtime.resolve_model(public, "image")
                api, _ = create_apps(api_runtime, "secret")
                app = FastAPI()
                app.mount("/v1", api)
                with (
                    TestClient(app, headers={"Authorization": "Bearer secret"}) as client,
                    patch.object(api_runtime, "images") as generate,
                ):
                    response = client.get("/v1/models")
                    shown = next(row for row in response.json()["data"] if row["id"] == public)
                    self.assertEqual(shown["family"], "Unknown")
                    self.assertFalse(shown["generation_available"])
                    self.assertTrue(shown["unavailable_reason"])
                    self.assertNotIn(str(self.root), response.text)
                    rejected = client.post(
                        "/v1/images/generations", json={"model": public, "prompt": "fixture"}
                    )
                    self.assertEqual(rejected.status_code, 404, rejected.text)
                    generate.assert_not_called()

    def test_public_bundle_selection_freezes_the_pickers_private_native_identity(self):
        index = self.bundle(self.bundle_roots[0] / "nested" / "Example bundle")
        malformed = self.write(self.bundle_roots[0] / "malformed" / index.name, b"[]")
        incomplete = self.write(self.bundle_roots[0] / "incomplete" / index.name, b"{}")
        self.write(self.bundle_roots[0] / "wrong-case" / "MODEL_INDEX.JSON", b"{}")
        self.state.models.update_all_models()
        records = self.picker_records()
        self.assertEqual(
            {row["name"] for row in records}, {str(index), str(malformed), str(incomplete)}
        )

        output = self.write("result.png", b"fixture response")
        captured = []

        def submit(_payload, selected):
            captured.append(
                snapshot_request(
                    {"base_model_name": selected["name"]}, self.state.models, {}, self.manager
                )
            )
            job = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event())
            job.events.put(("result", [output]))
            return job

        api, _ = create_apps(api_runtime, "secret")
        app = FastAPI()
        app.mount("/v1", api)
        with TestClient(app, headers={"Authorization": "Bearer secret"}) as client:
            response = client.get("/v1/models")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(self.root.name, response.text)
            self.assertNotIn(self.bundle_roots[0].name, response.text)
            image_ids = [
                model["id"] for model in response.json()["data"] if model["type"] == "image"
            ]
            self.assertEqual(len(image_ids), 1)
            public_id = image_ids[0]
            self.assertRegex(
                public_id, r"^image/diffusers/[0-9a-f]{24}/Example bundle/model_index\.json$"
            )

            self.state.settings.default_settings["base_model"] = str(index)
            with patch.object(api_runtime, "images", side_effect=submit) as images:
                for model in (public_id, "image/default"):
                    response = client.post(
                        "/v1/images/generations", json={"prompt": "fixture", "model": model}
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                response = client.post(
                    "/v1/images/generations", json={"prompt": "fixture", "model": str(index)}
                )
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(self.root.name, response.text)
                self.assertEqual(images.call_count, 2)

        for data in captured:
            self.assertEqual(data["base_model_name"], str(index))
            self.assertEqual(data[BUNDLE_KEY].index.source, index)
            self.assertEqual(
                {item.role for item in data[BUNDLE_KEY].files},
                {"index", "diffusion", "text_encoder", "vae"},
            )

    def test_same_named_bundles_keep_ids_across_root_reordering(self):
        indexes = [self.bundle(root / "same") for root in self.bundle_roots]
        direct = self.bundle(self.bundle_roots[0])
        self.state.models.update_all_models()
        initial = {
            item["name"]: model
            for model, item in api_runtime.models().items()
            if item["kind"] == "image"
        }
        self.assertEqual(set(initial), {str(index) for index in (*indexes, direct)})
        self.assertEqual(len(set(initial.values())), 3)
        self.assertRegex(initial[str(direct)], r"^image/diffusers/[0-9a-f]{24}/model_index\.json$")

        self.manager.paths["path_diffusers"].reverse()
        self.state.models.update_all_models()
        for name, model in initial.items():
            self.assertEqual(api_runtime.resolve_model(model, "image")[1]["name"], name)

        self.manager.paths["path_diffusers"].remove(self.bundle_roots[0])
        # The inventory can still contain the old root until the picker refreshes.
        with self.assertRaisesRegex(ValueError, "Unknown local image model"):
            api_runtime.resolve_model(initial[str(indexes[0])], "image")
        self.assertEqual(
            api_runtime.resolve_model(initial[str(indexes[1])], "image")[1]["name"], str(indexes[1])
        )
