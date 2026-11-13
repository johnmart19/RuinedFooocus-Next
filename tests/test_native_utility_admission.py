"""Native utility headers cannot become ordinary Create checkpoints or stale cards."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.features.image_editing.tool_activation import activation_values, installed_tool
from ruinedfooocus.features.models.acquisition import Catalogue, checkpoint_name
from ruinedfooocus.features.models.model_architecture import inspect_model_file
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_registry import ModelRegistry, download_identity
from ruinedfooocus.features.models.selection import checkpoint_choices, checkpoint_records
from ruinedfooocus.features.models.ui_acquisition import create_ui
from ruinedfooocus.features.models.utility_models import is_image_utility
from ruinedfooocus.features.models import utility_models
from test_native_variant_defaults import SHAPES, write_header
import test_native_vision as fixtures

ROOT = Path(__file__).resolve().parents[1]


def metadata_lookup():
    source = ROOT / "ruinedfooocus/features/models/model_handler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_models_by_path"
    )
    scope = dict(Path=Path, json=json, sys=sys, file_signature=file_signature)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    return scope["get_models_by_path"]


class UtilityAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeVisionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.manager = self.fixture.root, self.fixture.manager
        self.cache = self.root / "metadata"
        self.cache.mkdir()
        self.lookup = metadata_lookup()
        self.owner = NS(
            cache_paths={"checkpoints": self.cache},
            offline=True,
            detect_checkpoint_base=lambda path: inspect_model_file(path).get("architecture"),
        )

    def metadata(self, path):
        return self.lookup(self.owner, "checkpoints", path, fetch=False)

    def test_intrinsic_rt_sam_headers_override_stale_sdxl_and_leave_unknowns_visible(self):
        paths = [self.fixture.files[name] for name in ("RT_DETR_v4", "SAM3", "SAM31", "wrong")]
        for path in paths:
            cached = (self.cache / path.name).with_suffix(".json")
            original = dict(
                baseModel="SDXL 1.0",
                modelId=42,
                rf_source="publisher",
                _rf_file_signature=file_signature(path),
            )
            cached.write_text(json.dumps(original), encoding="utf-8")
            before = cached.read_bytes()
            result = self.metadata(path)
            self.assertEqual(cached.read_bytes(), before)
            self.assertEqual(result["modelId"], 42)
            if path.stem != "wrong":
                self.assertTrue(is_image_utility(result))
                self.assertEqual(result["baseModel"], path.stem)
                self.assertEqual(result["_rf_intrinsic_utility"]["signature"], file_signature(path))
            else:
                self.assertFalse(is_image_utility(result))
        values = {str(path): self.metadata(path) for path in paths}
        metadata = values.__getitem__
        self.assertEqual(
            [row["name"] for row in checkpoint_records(list(values), metadata)], [str(paths[-1])]
        )
        self.assertEqual(
            checkpoint_choices(list(values), str(paths[1]), metadata), [("wrong", str(paths[-1]))]
        )

    def test_changed_header_recomputes_role_and_capture_mutation_never_binds_a_stale_class(self):
        path = self.fixture.files["SAM3"]
        original = self.metadata(path)
        self.assertTrue(is_image_utility(original))
        cached = (self.cache / path.name).with_suffix(".json")
        cached.write_text(json.dumps(original), encoding="utf-8")
        before = cached.read_bytes()
        write_header(path, SHAPES["Omnigen2"])
        changed = self.metadata(path)
        self.assertFalse(is_image_utility(changed))
        self.assertEqual(changed["baseModel"], "Omnigen2")
        self.assertNotIn("rf_model_role", changed)
        self.assertEqual(cached.read_bytes(), before)

        def mutate(current):
            write_header(current, {"unknown": [1]})
            return "SAM3"

        self.owner.detect_checkpoint_base = mutate
        result = self.metadata(path)
        self.assertFalse(is_image_utility(result))
        self.assertNotIn("_rf_intrinsic_utility", result)

    def catalogue(self):
        rows, downloads = [], {}
        for native in ("RT_DETR_v4", "SAM31"):
            path = self.fixture.files[native]
            key = "diffusion_models/" + path.name
            source = dict(
                path="path_diffusion_models",
                filename=path.name,
                url="https://example.invalid/" + native,
            )
            downloads[key] = source
            rows.append(
                dict(
                    id="existing:" + key,
                    catalogue_key=key,
                    state="existing",
                    profile_id="utility",
                    role="image_utility_model",
                    label=native,
                    chat_weight=False,
                    native_class=native,
                    native_family=native,
                    auto_recipe_supported=False,
                    download_identity=download_identity(source),
                    artifact=dict(
                        bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    ),
                    companions=dict(text=[], image=[]),
                )
            )
        self.manager.DOWNLOADABLE_FILES = downloads
        self.manager.model_paths = {"cache_path": self.root / "cache"}
        registry = ModelRegistry(
            dict(
                schema_version=1,
                profiles={"utility": {"consumers": ["Image tools"]}},
                artifacts=rows,
                benchmarks=[],
            )
        )
        return Catalogue(self.manager, registry), registry

    def test_registered_utility_cards_and_atomic_tool_use_require_current_native_headers(self):
        catalogue, registry = self.catalogue()
        utilities = catalogue.records(
            category="Image tools", task="Image tools", status="Installed"
        )
        self.assertEqual(len(utilities), 2)
        for row in utilities:
            self.assertFalse(row["can_activate"])
            with self.assertRaises(ValueError):
                checkpoint_name(row, object())
            target = installed_tool(row, self.manager)
            self.assertEqual(target["kind"], "segment" if "SAM31" in row["key"] else "detect")
            values = activation_values(
                target["kind"],
                target,
                presets={"Named utility": {"type": target["kind"]}},
                custom="Custom...",
            )
            self.assertEqual(values["cn_selection"], "Named utility")
            self.assertEqual(values["native_vision_model"], row["local"])
            self.assertNotIn("cn_type", values)
        self.assertFalse(
            {row["key"] for row in utilities}
            & {row["key"] for row in catalogue.records(category="Image models")}
        )
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                create_ui(self.manager, NS(update_all_models=Mock()), offline=lambda: True)
            details = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "selected_details")
            _, _, _, action = details(utilities[0]["key"], utilities, locale="en")
        self.assertTrue(action["interactive"])
        self.assertEqual(action["value"], "Use installed file in Image tools")
        row = utilities[0]
        write_header(Path(row["local"]), {"unknown": [1]})
        with self.assertRaisesRegex(ValueError, "changed or is incomplete"):
            installed_tool(row, self.manager)

    def test_unknown_headers_stay_unclassified_and_known_tools_work_without_native_runtime(self):
        path = self.fixture.files["wrong"]
        row = dict(
            category="diffusion_models",
            local=str(path),
            local_signature=file_signature(path),
            size_matches=True,
        )
        self.assertIsNone(installed_tool(row, self.manager))
        path = self.fixture.files["SAM31"]
        row = dict(
            category="diffusion_models",
            local=str(path),
            local_signature=file_signature(path),
            size_matches=True,
        )
        with patch.dict(sys.modules, {"comfy.model_detection": None}):
            self.assertEqual(installed_tool(row, self.manager)["kind"], "segment")

    def test_native_source_changes_invalidate_positive_header_and_predicate_caches(self):
        source_root = ROOT / "repositories/ComfyUI/comfy"
        copied = self.root / "native-source"
        copied.mkdir()
        sources = []
        for name in utility_models._PREDICATE_FILES:
            target = copied / name
            shutil.copyfile(source_root / name, target)
            sources.append(target)

        def signatures():
            return tuple(tuple(file_signature(path)) for path in sources)

        path = self.fixture.files["RT_DETR_v4"]
        with patch.object(utility_models, "_native_source_signatures", signatures):
            self.assertEqual(utility_models.passive_utility_file(path), "RT_DETR_v4")
            original = signatures()
            detector = copied / "model_detection.py"
            detector.write_text(
                detector.read_text(encoding="utf-8").replace(
                    'dit_config["image_model"] = "RT_DETR_v4"',
                    'dit_config["image_model"] = "unknown"',
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(signatures(), original)
            self.assertIsNone(utility_models.passive_utility_file(path))

    def test_cold_create_hub_callers_use_native_predicates_without_importing_torch_or_comfy(self):
        code = r"""
import ast, builtins, io, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
assert 'torch' not in sys.modules and 'comfy' not in sys.modules
original_import=builtins.__import__
def no_runtime(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','comfy','comfy_extras'):
        raise AssertionError('Utility browsing started a tensor/model runtime: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=no_runtime
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.utility_models import passive_utility_file,is_image_utility
from ruinedfooocus.features.models.selection import checkpoint_records
from ruinedfooocus.features.generation.image_preparation import capture_image_assets
from ruinedfooocus.features.image_processing.native_vision import native_model_choices
from ruinedfooocus.features.image_editing.tool_activation import installed_tool,activation_values
source=Path('ruinedfooocus/features/models/model_handler.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='get_models_by_path')
scope=dict(Path=Path,json=json,sys=sys,file_signature=file_signature)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
paths=[Path(value).resolve()for value in sys.argv[1:]]
original_open=io.open
reads={str(path):[] for path in paths}
class HeaderOnly:
    def __init__(self,stream,path):self.stream,self.path,self.limit=stream,path,None
    def __enter__(self):return self
    def __exit__(self,*args):return self.stream.__exit__(*args)
    def __getattr__(self,name):return getattr(self.stream,name)
    def read(self,size=-1):
        offset=self.stream.tell()
        assert size>=0 and (offset==0 and size==8 or self.limit is not None and offset+size<=self.limit),'Tensor payload read'
        data=self.stream.read(size)
        if offset==0:self.limit=8+int.from_bytes(data,'little')
        reads[self.path].append(dict(offset=offset,bytes=len(data)))
        return data
def header_open(file,mode='r',*args,**kwargs):
    path=str(Path(file).resolve()) if isinstance(file,(str,Path)) else None
    if path in reads:
        assert mode=='rb','Model write or unsupported read'
        return HeaderOnly(original_open(file,mode,*args,**kwargs),path)
    return original_open(file,mode,*args,**kwargs)
io.open=builtins.open=header_open
root=paths[0].parent.parent
cache=root/'cold-metadata';cache.mkdir()
def forbidden(*args,**kwargs):raise AssertionError('Passive positive utility called runtime classifier')
owner=NS(cache_paths={'checkpoints':cache},offline=True,detect_checkpoint_base=forbidden)
manager=NS(get_model_folders=lambda category:[root/category])
metadata={}
for path,expected in zip(paths,('RT_DETR_v4','SAM31')):
    assert passive_utility_file(path)==expected
    cached=(cache/path.name).with_suffix('.json')
    cached.write_text(json.dumps(dict(baseModel='SDXL 1.0',_rf_file_signature=file_signature(path))),encoding='utf-8')
    before=cached.read_bytes()
    values=scope['get_models_by_path'](owner,'checkpoints',path)
    assert is_image_utility(values)and values['baseModel']==expected
    assert cached.read_bytes()==before
    metadata[str(path)]=values
    models=NS(get_models_by_path=lambda *args,**kwargs:values,get_model_base=lambda value:value.get('baseModel',''))
    try:
        capture_image_assets(dict(base_model_name=str(path),prompt='ordinary generation'),manager,{},models=models)
    except ValueError as error:
        assert 'matching Image tool' in str(error),str(error)
    else:raise AssertionError('Cold ordinary diffusion admission accepted a utility')
    row=dict(category='diffusion_models',role='image_utility_model',local=str(path),local_signature=file_signature(path),size_matches=True)
    target=installed_tool(row,manager)
    kind='detect'if expected=='RT_DETR_v4'else'segment'
    assert target['kind']==kind and target['value']==str(path)
    assert activation_values(kind,target,presets={'Named utility':dict(type=kind)},custom='custom')==dict(cn_selection='Named utility',automatic_upscale_target='Generated image',native_vision_model=str(path))
assert checkpoint_records(list(metadata),metadata.__getitem__)==[]
assert str(paths[0]) in {value for _,value in native_model_choices(manager,'detect')}
assert str(paths[1]) in {value for _,value in native_model_choices(manager,'segment')}
assert 'torch'not in sys.modules and not any(name=='comfy'or name.startswith('comfy.')for name in sys.modules)
assert all(len(value)==2 and value[0]==dict(offset=0,bytes=8) for value in reads.values()),reads
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(self.fixture.files["RT_DETR_v4"]),
                str(self.fixture.files["SAM31"]),
            ],
            cwd=ROOT,
            env=dict(os.environ, RF_DATA_DIR=str(self.root / "cold-data")),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_create_capture_rejects_a_utility_before_component_or_weight_loading(self):
        from ruinedfooocus.features.generation.image_preparation import capture_image_assets

        path = self.fixture.files["SAM31"]
        models = NS(
            get_models_by_path=lambda *args, **kwargs: self.metadata(path),
            get_model_base=lambda metadata: metadata.get("baseModel", ""),
        )
        forbidden = Mock(side_effect=AssertionError("generation components loaded"))
        with patch(
            "ruinedfooocus.features.models.component_profiles.component_inventory", forbidden
        ):
            with self.assertRaisesRegex(ValueError, "matching Image tool"):
                capture_image_assets(
                    dict(base_model_name=str(path), prompt="ordinary generation"),
                    self.manager,
                    {},
                    models=models,
                )
        forbidden.assert_not_called()

    def test_actual_api_model_discovery_excludes_utilities_and_preserves_unknown_files(self):
        import ruinedfooocus.app
        from ruinedfooocus.app import api_runtime as api

        paths = {
            self.fixture.files[key].name: self.fixture.files[key]
            for key in ("RT_DETR_v4", "SAM31", "wrong")
        }
        models = NS(
            get_names=lambda kind: list(paths),
            get_file=lambda kind, name: paths.get(name),
            get_models_by_path=lambda kind, path: self.metadata(path),
            get_model_base=lambda metadata: metadata.get("baseModel", ""),
        )
        shared = NS(
            models=models,
            settings=NS(default_settings={}),
            path_manager=NS(upscaler_filenames=[], DOWNLOADABLE_FILES={}),
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "ruinedfooocus.features.chat.llama_models": NS(
                        local_models=lambda: {}, DEFAULT_MODEL="unused"
                    ),
                    "ruinedfooocus.features.chat.llama_vision": NS(
                        vision_models=lambda manager: []
                    ),
                },
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            self.assertEqual(set(api.models()), {"image/wrong.safetensors"})
            with self.assertRaisesRegex(ValueError, "Unknown local image"):
                api.resolve_model("image/SAM31.safetensors", "image")


if __name__ == "__main__":
    unittest.main()
