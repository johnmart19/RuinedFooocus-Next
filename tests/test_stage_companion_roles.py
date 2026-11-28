"""Cascade Stage B stays available for its prior-conditioned companion route."""

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient
import torch

from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.models import stage_companions
from ruinedfooocus.features.models.acquisition import Catalogue, checkpoint_name
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_registry import ModelRegistry, download_identity
from ruinedfooocus.features.models.selection import (
    checkpoint_choices,
    checkpoint_records,
    primary_checkpoint_selection,
)
from ruinedfooocus.integrations.openai_api import create_apps
import test_api_model_discovery as api_fixtures
from test_image_preparation import Manager
from test_native_utility_admission import metadata_lookup
from test_native_variant_defaults import write_header
from test_qwen_image21_conditioning import native_class

ROOT = Path(__file__).resolve().parents[1]


def stage_header(stage="b", width=640):
    prefix = "model.diffusion_model."
    return {
        prefix + "clf.1.weight": [16, 320, 1, 1],
        prefix + ("clip_mapper.weight" if stage == "b" else "clip_txt_mapper.weight"): (
            [5120, 1280] if stage == "b" else [2048, 1280]
        ),
        prefix + "down_blocks.1.0.channelwise.0.weight": [width * 4, width],
        **{prefix + f"fixture{index}": [1] for index in range(3)},
        "text_encoder.clip_g.text_projection": [1280, 1280],
        "vae.vquantizer.codebook.weight": [8192, 4],
    }


class StageCompanionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.path = write_header(self.root / "renamed.safetensors", stage_header())
        self.cache = self.root / "metadata"
        self.cache.mkdir()
        self.lookup = metadata_lookup()
        self.owner = NS(
            cache_paths={"checkpoints": self.cache},
            offline=True,
            detect_checkpoint_base=Mock(return_value=None),
        )

    def metadata(self, path=None):
        return self.lookup(self.owner, "checkpoints", path or self.path, fetch=False)

    def test_native_headers_override_stale_publisher_role_without_filename_rules_or_writes(self):
        cached = (self.cache / self.path.name).with_suffix(".json")
        stale = dict(baseModel="SDXL 1.0", modelId=42, _rf_file_signature=file_signature(self.path))
        cached.write_text(json.dumps(stale), encoding="utf-8")
        before = cached.read_bytes()
        for width in (640, 576):
            write_header(self.path, stage_header(width=width))
            metadata = self.metadata()
            self.assertTrue(stage_companions.is_stage_companion(metadata))
            self.assertEqual(metadata["baseModel"], "Stable_Cascade_B")
            self.assertEqual(
                metadata["_rf_intrinsic_companion"]["signature"], file_signature(self.path)
            )
        self.assertEqual(cached.read_bytes(), before)
        self.owner.detect_checkpoint_base.assert_not_called()
        write_header(self.path, stage_header("c"))
        self.assertFalse(stage_companions.is_stage_companion(self.metadata()))
        named = write_header(self.root / "stable_cascade_stage_b.safetensors", {"unknown": [1]})
        self.assertIsNone(stage_companions.passive_companion_file(named))
        self.assertEqual(
            checkpoint_choices([named.name], metadata=lambda _: self.metadata(named)),
            [("stable cascade stage b", named.name)],
        )

    def test_primary_picker_and_default_clear_only_intrinsic_companions_and_admission_is_actionable(
        self,
    ):
        from ruinedfooocus.features.generation.image_preparation import capture_image_assets
        from ruinedfooocus.features.image_editing.ui_native_workflows import stage_b_choices

        metadata = self.metadata()
        models = NS(
            get_names=lambda _: [self.path.name],
            get_file=lambda *args: self.path,
            detect_checkpoint_base=lambda path: metadata["baseModel"],
            get_models_by_path=lambda *args, **kwargs: metadata,
            get_model_base=lambda info: info["baseModel"],
        )
        self.assertEqual(checkpoint_records([self.path.name], lambda _: metadata), [])
        self.assertEqual(
            checkpoint_choices([self.path.name], self.path.name, lambda _: metadata), []
        )
        self.assertIsNone(primary_checkpoint_selection(self.path.name, lambda _: metadata))
        self.assertEqual(
            primary_checkpoint_selection("missing.safetensors", lambda _: {}), "missing.safetensors"
        )
        self.assertEqual(stage_b_choices(models), [self.path.name])
        with self.assertRaisesRegex(ValueError, "Select Stage C as the primary"):
            capture_image_assets(
                dict(base_model_name=self.path.name, prompt="ordinary request"),
                Manager(self.root),
                {},
                models=models,
            )

    def test_native_conditioning_supplies_the_real_prior_and_missing_prior_is_zero(self):
        node = native_class(
            "comfy_extras/nodes_stable_cascade.py",
            "StableCascade_StageB_Conditioning",
            dict(io=NS(NodeOutput=lambda value: (value,))),
            methods=("execute",),
        )
        model = native_class(
            "comfy/model_base.py",
            "StableCascade_B",
            dict(torch=torch, comfy=NS(conds=NS(CONDRegular=lambda value: value))),
            methods=("extra_conds",),
        )
        prior = torch.ones(1, 16, 2, 3)
        original = [["literal prompt", dict(pooled_output=torch.zeros(1, 1280))]]
        conditioned = node.execute(original, {"samples": prior})[0]
        self.assertNotIn("stable_cascade_prior", original[0][1])
        noise = torch.zeros(1, 4, 16, 16)
        self.assertIs(model().extra_conds(noise=noise, **conditioned[0][1])["effnet"], prior)
        self.assertFalse(
            torch.count_nonzero(model().extra_conds(noise=noise, **original[0][1])["effnet"])
        )

    def test_catalogue_exposes_expected_companion_without_rewriting_historical_registry_role(self):
        key = "checkpoints/" + self.path.name
        source = dict(
            path="path_checkpoints", filename=self.path.name, url="https://example.invalid/owned"
        )
        entry = dict(
            id="existing:" + key,
            catalogue_key=key,
            state="existing",
            profile_id="cascade",
            role="image_diffusion_checkpoint",
            architecture="Stable_Cascade_B",
            native_family="Stable_Cascade_B",
            chat_weight=False,
            download_identity=download_identity(source),
            artifact=dict(bytes=self.path.stat().st_size, sha256=None),
            companions=dict(text=[], image=[]),
        )
        original = copy.deepcopy(entry)
        registry = ModelRegistry(
            dict(
                schema_version=1,
                profiles=dict(cascade=dict(consumers=["Stable_Cascade_B"])),
                artifacts=[entry],
                benchmarks=[],
            )
        )
        manager = NS(DOWNLOADABLE_FILES={key: source}, get_model_folders=lambda _: [self.root])
        record = Catalogue(manager, registry).record(key)
        self.assertEqual(record["task"], "Component")
        self.assertEqual(record["category_label"], "Component")
        self.assertIn("Stage B companion", record["support"])
        self.assertFalse(record["can_activate"])
        self.assertEqual(entry, original)
        with self.assertRaises(ValueError):
            checkpoint_name(record, object())

    def test_real_primary_dropdown_default_uses_role_filter_and_preserves_saved_preference(self):
        self.assertEqual(self.lookup(self.owner, "checkpoints", None), {})
        self.assertEqual(self.lookup(self.owner, "checkpoints", ""), {})
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        assignments = []
        for name in ("initial_checkpoint", "base_model"):
            assignments.append(
                next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == name
                        for target in node.targets
                    )
                )
            )
        settings = dict(base_model=self.path.name)
        dropdown = Mock(side_effect=lambda **values: NS(**values))
        scope = dict(
            settings=settings,
            shared=NS(
                models=NS(
                    get_names=lambda _: [self.path.name],
                    get_models_by_path=lambda *args, **kwargs: self.metadata(),
                )
            ),
            primary_checkpoint_selection=primary_checkpoint_selection,
            checkpoint_choices=checkpoint_choices,
            gr=NS(Dropdown=dropdown),
            translate_ui=lambda value: value,
        )
        exec(
            compile(ast.Module(body=assignments, type_ignores=[]), "real primary selector", "exec"),
            scope,
        )
        self.assertIsNone(scope["base_model"].value)
        self.assertFalse(scope["base_model"].choices)
        self.assertEqual(settings["base_model"], self.path.name)
        scope.update(
            gr=gr,
            base_model="base",
            model_gallery="gallery",
            model_gallery_records="records",
            model_metadata_refresh="revision",
            style_selection="styles",
            preset_gallery="presets",
            update_model_filter=lambda _: ("gallery update", []),
            adapters=NS(catalog="adapters", status="adapter status", discover=lambda: ([], "")),
            load_styles=dict,
            path_manager=NS(get_presets=list),
        )
        scope["shared"].models.update_all_models = Mock()
        scope["shared"].models.revision = 1
        for name in ("model_refresh_clicked", "update_model_previews"):
            node = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            node.decorator_list = []
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "real picker refresh", "exec"),
                scope,
            )
        update = scope["model_refresh_clicked"]([], self.path.name, "", "", 0)["base"]
        self.assertIsNone(update["value"])
        update = scope["update_model_previews"](-1, "", "", [], self.path.name, 0, None)[4]
        self.assertIsNone(update["value"])
        self.assertEqual(settings["base_model"], self.path.name)

    def test_public_models_and_default_resolution_share_current_intrinsic_role(self):
        fixture = api_fixtures.ApiModelDiscoveryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        path = write_header(fixture.root / "models/checkpoints/renamed.safetensors", stage_header())
        other = fixture.write("models/checkpoints/unknown.safetensors")
        fixture.state.models.update_all_models()
        self.assertNotIn("image/" + path.name, api_runtime.models())
        self.assertIn("image/" + other.name, api_runtime.models())
        fixture.state.settings.default_settings["base_model"] = path.name
        with self.assertRaisesRegex(ValueError, "Choose an ID from /v1/models"):
            api_runtime.resolve_model("default", "image")
        api, _ = create_apps(api_runtime, "secret")
        app = FastAPI()
        app.mount("/v1", api)
        with TestClient(app, headers={"Authorization": "Bearer secret"}) as client:
            response = client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("image/" + path.name, {row["id"] for row in response.json()["data"]})
        self.assertEqual(fixture.state.settings.default_settings["base_model"], path.name)

    def test_cold_header_role_does_not_import_a_tensor_runtime(self):
        code = r"""
import ast,builtins,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','comfy','comfy_extras'):
        raise AssertionError('Companion browsing imported a tensor runtime: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.stage_companions import passive_companion_file,is_stage_companion
from ruinedfooocus.features.models.selection import checkpoint_choices,primary_checkpoint_selection
path=Path(sys.argv[1]);cache=path.parent/'cold-metadata';cache.mkdir()
assert passive_companion_file(path)=='Stable_Cascade_B'
source=Path('ruinedfooocus/features/models/model_handler.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='get_models_by_path')
scope=dict(Path=Path,json=json,sys=sys,file_signature=file_signature)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
def forbidden(*args):raise AssertionError('Companion inspection called runtime classifier')
owner=NS(cache_paths={'checkpoints':cache},offline=True,detect_checkpoint_base=forbidden)
info=scope['get_models_by_path'](owner,'checkpoints',path)
assert is_stage_companion(info)
assert checkpoint_choices([path.name],path.name,lambda _:info)==[]
assert primary_checkpoint_selection(path.name,lambda _:info)is None
assert 'torch'not in sys.modules and not any(name=='comfy'or name.startswith('comfy.')for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            cwd=ROOT,
            env=dict(os.environ, RF_DATA_DIR=str(self.root / "cold-data")),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
