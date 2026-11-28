"""Explicit full codec pairs travel through the existing UI/API and replay paths."""

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import ruinedfooocus.app
from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.models import approximate_codecs as codecs
from ruinedfooocus.features.models import component_profiles as profiles
from ruinedfooocus.features.models import ui_model_components
from ruinedfooocus.integrations.openai_api import create_apps
from test_approximate_codecs import ROOT, write_component
from test_image_preparation import Manager


class ApproximateCodecRouteTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.checkpoint = self.root / "model.safetensors"
        header = json.dumps({"conditioner.embedders.0.x": {}, "first_stage_model.x": {}}).encode()
        self.checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        self.manager = Manager(self.root)
        self.manager.paths["path_vae_approx"] = str(self.root / "approx")
        self.manager.EXTENSIONS = [".safetensors", ".pt", ".gguf"]
        for role in ("encoder", "decoder"):
            write_component(
                self.root / "approx" / f"taesdxl_{role}.safetensors",
                codecs.expected_shapes("taesdxl", role),
            )
        self.choices = {"vae_sdxl": "approx:taesdxl"}
        self.identity = profiles.checkpoint_identity(self.checkpoint)
        self.models = SimpleNamespace(
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
            get_model_base=lambda row: row["baseModel"],
            get_file=lambda *args: self.checkpoint,
            get_names=lambda kind: [self.checkpoint.name] if kind == "checkpoints" else [],
            offline=True,
        )
        self.settings = SimpleNamespace(
            default_settings={"performance": "Own settings", "resolution": "default"},
            name="owned-test",
            save_profile=Mock(),
        )
        self.shared = SimpleNamespace(
            settings=self.settings,
            models=self.models,
            path_manager=self.manager,
            performance_settings=SimpleNamespace(
                CUSTOM_PERFORMANCE="Own settings",
                performance_options=[],
                choices_for_model=lambda *args: ["Own settings"],
                selection_for_model=lambda *args: "Own settings",
                get_perf_options=lambda *args: dict(
                    custom_steps=2, cfg=4, sampler_name="euler", scheduler="normal", clip_skip=1
                ),
            ),
            resolution_settings=SimpleNamespace(
                CUSTOM_RESOLUTION="Custom...",
                selection_for_model=lambda *args: "default",
                choices_for_model=lambda *args: ["default"],
                aspect_ratios={"default": (512, 512)},
            ),
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": self.shared,
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"])
                    ),
                    "ruinedfooocus.features.prompting.sdxl_styles": SimpleNamespace(
                        load_styles=dict
                    ),
                },
            )
        )
        self.enterContext(patch.object(ruinedfooocus.app, "state", self.shared, create=True))

    def test_embedded_vae_editor_offers_pair_saves_and_restores_automatic(self):
        self.enterContext(
            patch.object(ui_model_components, "selected", return_value=(self.checkpoint, "SDXL"))
        )
        editor = runpy.run_path(
            str(ROOT / "ruinedfooocus/features/models/ui_component_overrides.py")
        )
        ordinary = self.root / "vae" / "ordinary.safetensors"
        ordinary.parent.mkdir()
        ordinary.write_bytes(b"local custom file")
        choice = editor["role_choices"](self.checkpoint.name, "vae_sdxl", {}, self.identity)
        self.assertTrue(choice["interactive"])
        self.assertEqual({value for _, value in choice["choices"]}, {"", "approx:taesdxl"})
        self.assertEqual(choice["value"], "")
        saved, scope = editor["save_choice"](
            self.checkpoint.name, "vae_sdxl", "approx:taesdxl", False, {}, self.identity
        )
        self.assertEqual(saved, self.choices)
        profile = self.settings.save_profile.call_args.args[0]
        self.settings.default_settings = profile
        self.assertEqual(profiles.saved_choices(profile, self.checkpoint), self.choices)
        selected = editor["role_choices"](self.checkpoint.name, "vae_sdxl", saved, scope)
        self.assertEqual(selected["value"], "approx:taesdxl")
        with self.assertRaisesRegex(editor["gr"].Error, "Built-in"):
            editor["save_choice"](
                self.checkpoint.name, "vae_sdxl", "ordinary.safetensors", True, saved, scope
            )
        reset, _ = editor["save_choice"](self.checkpoint.name, "vae_sdxl", "", False, saved, scope)
        self.assertIsNone(reset["vae_sdxl"])
        inventory = profiles.component_inventory(
            self.checkpoint, "SDXL", {}, self.manager, self.choices
        )
        row = next(row for row in inventory["rows"] if row["key"] == "vae_sdxl")
        self.assertEqual(row["compatibility"], "compatible")
        self.assertEqual(len(row["files"]), 2)
        self.assertEqual(inventory["missing_payload_bytes"], 0)
        refreshed = ui_model_components.refresh(self.checkpoint.name, self.choices, scope)
        self.assertEqual(len(refreshed[1]["vae_sdxl"]["codec_files"]), 2)

    def test_public_capabilities_offer_installed_pair_only_for_exact_model_contract(self):
        with patch.object(
            api_runtime,
            "models",
            return_value={
                "image/sdxl": dict(kind="image", name=self.checkpoint.name, family="SDXL"),
                "image/pixel": dict(
                    kind="image", name="pixel.safetensors", family="ZImagePixelSpace"
                ),
                "image/layered": dict(
                    kind="image", name="layered.safetensors", family="QwenImageLayered"
                ),
            },
        ):
            api, _ = create_apps(api_runtime, "secret")
            app = FastAPI()
            app.mount("/v1", api)
            with TestClient(app, headers={"Authorization": "Bearer secret"}) as client:
                response = client.get("/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        pair = models["image/sdxl"]["approximate_codecs"][0]
        self.assertEqual(pair["choice"], "approx:taesdxl")
        self.assertEqual(pair["slot"], "vae_sdxl")
        self.assertEqual(pair["local_bytes"], sum(item["bytes"] for item in pair["files"]))
        self.assertEqual(len(pair["identity"]), 64)
        self.assertEqual(models["image/pixel"]["approximate_codecs"], [])
        self.assertEqual(models["image/layered"]["approximate_codecs"], [])

    def test_actual_api_captures_pair_before_queue_and_rejects_foreign_or_missing_pair(self):
        with (
            patch.object(
                api_runtime,
                "image_defaults",
                dict(base_model_name=self.checkpoint.name, performance_selection="Own settings"),
            ),
            patch.object(api_runtime, "submit", side_effect=lambda run: run) as submit,
        ):
            operation = api_runtime.images(
                dict(prompt="Pattern", size="512x512", image_component_overrides=self.choices),
                dict(name=self.checkpoint.name, family="SDXL"),
            )
            captured = dict(
                zip(
                    operation.__code__.co_freevars,
                    (cell.cell_contents for cell in operation.__closure__),
                )
            )["data"]
            pair = captured[codecs.KEY]
            self.assertEqual(pair.alias, "taesdxl")
            items = captured[assets.KEY].artifacts
            self.assertEqual(
                {item.name for item in items}, {"vae_sdxl:encoder", "vae_sdxl:decoder"}
            )
            self.assertTrue(all(not item.request.url for item in items))
            self.assertEqual(captured[profiles.REQUEST_KEY]["vae_sdxl"], "approx:taesdxl")
            submit.reset_mock()
            with self.assertRaisesRegex(ValueError, "VAE contract"):
                api_runtime.images(
                    dict(
                        prompt="Pattern", image_component_overrides={"vae_sdxl": "approx:taeqi2_1"}
                    ),
                    dict(name=self.checkpoint.name, family="SDXL"),
                )
            pair.encoder.path.unlink()
            with self.assertRaisesRegex(ValueError, "both local"):
                api_runtime.images(
                    dict(prompt="Pattern", image_component_overrides=self.choices),
                    dict(name=self.checkpoint.name, family="SDXL"),
                )
            submit.assert_not_called()

    def test_preparation_keeps_both_file_signatures_and_does_not_recapture_a_changed_pair(self):
        request = profiles.snapshot_request(
            dict(
                base_model_name=self.checkpoint.name,
                prompt="Pattern",
                image_component_overrides=self.choices,
            ),
            self.models,
            {},
            self.manager,
        )
        pair = request[codecs.KEY]
        repeated = profiles.snapshot_request(request, self.models, {}, self.manager)
        self.assertIs(repeated[codecs.KEY], pair)
        captured = assets.capture_image_assets(
            request, self.manager, {}, models=self.models, values={}
        )
        self.assertIs(captured[codecs.KEY], pair)
        prepared = assets.prepare_image_assets(captured, check_cancelled=lambda: None)
        self.assertFalse(assets.needs_image_preparation(prepared))
        self.assertEqual(len(prepared[assets.KEY].artifacts), 2)
        stamp = pair.decoder.path.stat()
        os.utime(pair.decoder.path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10000000))
        with self.assertRaisesRegex(ValueError, "changed"):
            assets.capture_image_assets(request, self.manager, {}, models=self.models, values={})
        with self.assertRaisesRegex(ValueError, "changed"):
            profiles.snapshot_request(request, self.models, {}, self.manager)
        with self.assertRaisesRegex(ValueError, "changed"):
            assets.prepare_image_assets(captured, check_cancelled=lambda: None)

    def test_library_and_preset_replay_preserve_alias_but_capture_current_pair(self):
        from test_gallery_actions import functions
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        from ruinedfooocus.features.generation.ui_request import build_ui_request

        scope = functions()
        scope["apply_metadata"] = Mock(return_value={})
        metadata = dict(
            software="RuinedFooocus",
            prompt="Pattern",
            base_model_name=self.checkpoint.name,
            loras=[],
            image_component_overrides=self.choices,
        )
        with patch.dict(
            sys.modules,
            {
                "ruinedfooocus.core.util": SimpleNamespace(
                    get_lora_thumbnail=lambda _: None, load_keywords=lambda _: ""
                )
            },
        ):
            replay = scope["reuse_setup"](metadata)
        self.assertEqual(replay[profiles.REQUEST_KEY], self.choices)
        self.assertEqual(
            profiles.session_choices(
                self.checkpoint, {}, replay[profiles.REQUEST_KEY], replay[profiles.SCOPE_KEY]
            )[0],
            self.choices,
        )
        request = dict(metadata, **{profiles.SCOPE_KEY: replay[profiles.SCOPE_KEY]})
        captured = profiles.snapshot_request(request, self.models, {}, self.manager)
        self.assertIsInstance(captured[codecs.KEY], codecs.ApproximateCodec)
        info = PngInfo()
        info.add_text("parameters", json.dumps(metadata))
        preset = self.root / "preset.png"
        Image.new("RGB", (1, 1)).save(preset, pnginfo=info)
        with patch.dict(
            sys.modules,
            {
                "ruinedfooocus.features.image_editing.controlnet": SimpleNamespace(
                    get_settings=lambda _: {}
                )
            },
        ):
            submitted = build_ui_request(
                dict(preset_selection=str(preset)), models=self.models, settings={}, preset_mode=4
            )
        self.assertEqual(submitted[profiles.REQUEST_KEY], self.choices)
        self.assertEqual(submitted[profiles.SCOPE_KEY], self.identity)

    def test_non_native_engine_and_pixel_or_layered_slots_do_not_admit_full_aliases(self):
        with self.assertRaisesRegex(ValueError, "native image engine"):
            assets.capture_image_assets(
                dict(
                    base_model_name=self.checkpoint.name,
                    image_engine="sdcpp-vulkan",
                    image_component_overrides=self.choices,
                ),
                self.manager,
                {},
            )
        for family, choices in (
            ("ZImagePixelSpace", self.choices),
            ("QwenImageLayered", {"vae_qwen_image_layered": "approx:taef1"}),
        ):
            with self.subTest(family=family), self.assertRaises(ValueError):
                profiles.freeze_choices(self.checkpoint, family, {}, self.manager, choices)
        hidden = {codecs.KEY: "irrelevant utility capture"}
        self.assertNotIn(codecs.KEY, assets.omit_generation_conditioning(hidden))

    def test_startup_preload_forwards_the_saved_pair_and_both_prepared_sources(self):
        from test_image_preparation import function

        self.settings.default_settings = profiles.update_choices(
            dict(preload_image_model=True, base_model=self.checkpoint.name),
            self.checkpoint,
            self.identity,
            self.choices,
        )
        pipeline = SimpleNamespace(pipeline_type=["sdxl"], load_base_model=Mock())
        preload = function(
            "preload_startup_model",
            dict(
                shared=self.shared,
                settings=self.settings,
                pipelines=SimpleNamespace(update=Mock(return_value=pipeline)),
                traceback=Mock(),
                describe_error=str,
            ),
        )
        preload()
        pipeline.load_base_model.assert_called_once()
        options = pipeline.load_base_model.call_args.kwargs
        self.assertEqual(options["component_overrides"]["vae_sdxl"], "approx:taesdxl")
        request = options["prepared_data"]
        self.assertIsInstance(request[codecs.KEY], codecs.ApproximateCodec)
        self.assertEqual(len(request[assets.KEY].artifacts), 2)

    def test_shared_startup_capture_keeps_named_sensenova_sources(self):
        from test_image_preparation import function
        from test_sensenova_bundle import write_bundle
        from ruinedfooocus.features.models.sensenova_bundle import KEY as SENSE_KEY

        index = write_bundle(self.root / "sense")
        self.shared.models = SimpleNamespace(
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SenseNovaU15"},
            get_model_base=lambda row: row["baseModel"],
            get_file=lambda *args: index,
        )
        self.settings.default_settings = dict(preload_image_model=True, base_model=index.name)
        pipeline = SimpleNamespace(pipeline_type=["sdxl"], load_base_model=Mock())
        preload = function(
            "preload_startup_model",
            dict(
                shared=self.shared,
                settings=self.settings,
                pipelines=SimpleNamespace(update=Mock(return_value=pipeline)),
                traceback=Mock(),
                describe_error=str,
            ),
        )
        preload()
        pipeline.load_base_model.assert_called_once()
        data = pipeline.load_base_model.call_args.kwargs["prepared_data"]
        self.assertEqual(len(data[SENSE_KEY].files), 9)
        self.assertEqual(len(data[assets.KEY].artifacts), 9)


if __name__ == "__main__":
    unittest.main()
