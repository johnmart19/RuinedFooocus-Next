"""New native 0.45 roles stay distinct from existing diffusion workflows."""

from contextlib import contextmanager
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from safetensors.numpy import save_file

from ruinedfooocus.features.models import component_candidates as components
from ruinedfooocus.features.models import native_text_identity as text
from ruinedfooocus.features.models import native_workflow_admission as admission
from ruinedfooocus.features.models import utility_models
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_inbox import _local_category
from ruinedfooocus.features.generation.image_preparation import capture_image_assets

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


@contextmanager
def native_sources():
    signatures = lambda: tuple(
        tuple(file_signature(NATIVE / "comfy" / name))
        for name in ("model_detection.py", "supported_models.py", "supported_models_base.py")
    )
    with (
        patch.object(text, "_source", return_value=NATIVE / "comfy/sd.py"),
        patch.object(admission, "_native_source_signatures", signatures),
        patch.object(utility_models, "_native_source_signatures", signatures),
    ):
        yield


def encoder_shapes(width=2048, *, wevis=True):
    values = {
        "model.visual.merger.linear_fc2.weight": (width, 1),
        "model.visual.deepstack_merger_list.0.norm.weight": (1,),
    }
    if wevis:
        values["mask_decoder.query_builder.content_proj.weight"] = (1, 1)
    return values


def minimax_shapes(*, audio=False):
    values = {
        "video_patch_proj.weight": (4, 96),
        "condition_proj.weight": (4, 8),
        "final_layer.video_out.weight": (96, 4),
        "blocks.0.attn.q_norm.weight": (2,),
        "blocks.0.attn.qkv_proj.weight": (12, 4),
        "blocks.0.mlp.fc1.weight": (16, 4),
        "time_embedder.proj_in.weight": (4, 2),
        "time_embedder.proj_out.weight": (4, 4),
        "rope.inv_freq": (1,),
    }
    if audio:
        values.update({"audio_patch_proj.weight": (4, 16), "final_layer.audio_out.weight": (16, 4)})
    return values


class Native045RoleTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(native_sources())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def file(self, shapes, name="unrelated.safetensors"):
        path = self.root / name
        save_file({key: np.zeros(shape, dtype=np.float16) for key, shape in shapes.items()}, path)
        return path

    def test_cold_native_encoder_enum_recognizes_wevis_two_and_four_without_runtime_imports(self):
        with patch.dict(sys.modules, {"comfy.sd": None}):
            for width, expected in ((2048, "WEVISONE_2B"), (2560, "WEVISONE_4B")):
                with self.subTest(width=width):
                    classification = components.classify_shapes(encoder_shapes(width))
                    self.assertEqual(classification["text_type"], expected)
                    self.assertEqual(classification["text_role"], "image_utility_encoder")

    def test_qwen_three_vl_two_b_is_distinct_from_wevis_and_existing_four_eight_roles(self):
        value = components.classify_shapes(encoder_shapes(wevis=False))
        self.assertEqual(value["text_type"], "QWEN3VL_2B")
        self.assertNotIn("text_role", value)
        self.assertEqual(components.compatibility("clip_qwen3vl_4b", value), "incompatible")
        self.assertEqual(components.compatibility("clip_qwen3vl_8b", value), "incompatible")

    def test_wevis_is_never_an_ordinary_or_unknown_generation_clip_slot(self):
        value = components.classify_shapes(encoder_shapes())
        for key in (*components.TEXT_TYPES, "clip_unregistered"):
            with self.subTest(key=key):
                self.assertEqual(components.compatibility(key, value), "incompatible")
        path = self.file(encoder_shapes())
        result = components.inspect_local_component("clip_qwen3vl_4b", path)
        self.assertEqual(result["compatibility"], "incompatible")
        self.assertIn("utility encoder", result["reason"])

    def test_inbox_uses_existing_text_encoder_root_for_native_wevis_role(self):
        with patch.dict(sys.modules):
            sys.modules.pop("comfy.model_detection", None)
            sys.modules.pop("comfy.sd", None)
            self.assertEqual(_local_category(self.file(encoder_shapes())), "text_encoders")

    def test_new_minimax_image_is_a_separate_pending_image_family_not_legacy_video(self):
        image = self.file(minimax_shapes())
        video = self.file(minimax_shapes(audio=True), "legacy-video.safetensors")
        self.assertEqual(admission.planned_image_family(image), "MiniMaxH3Image")
        self.assertIsNone(admission.planned_image_family(video))
        with self.assertRaisesRegex(ValueError, "Capture.*MiniMax"):
            admission.require_available_image_workflow(image)
        admission.require_available_image_workflow(video)

    def test_inbox_keeps_new_image_model_in_diffusion_root_without_claiming_a_connected_workflow(
        self,
    ):
        self.assertEqual(_local_category(self.file(minimax_shapes())), "diffusion_models")

    def test_actual_submit_asset_capture_refuses_unconnected_image_graph_before_components(self):
        path = self.file(minimax_shapes())
        models = NS(
            get_models_by_path=Mock(return_value={}),
            get_model_base=Mock(return_value="MiniMaxH3Image"),
            get_file=Mock(return_value=path),
        )
        with self.assertRaisesRegex(ValueError, "PDD"):
            capture_image_assets(
                dict(base_model_name=path.name, task_type="tool_call", prompt="test"),
                NS(),
                {},
                models=models,
            )

    def test_actual_direct_loader_refuses_unconnected_image_graph_before_native_load(self):
        path = self.file(minimax_shapes())
        source = ROOT / "ruinedfooocus/features/generation/image_pipeline.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "load_base_model"
        )
        node.decorator_list = []
        native = Mock()
        scope = dict(
            settings=NS(default_settings={}),
            shared=NS(models=NS(get_model_path=Mock(return_value=path))),
            Path=Path,
            file_signature=file_signature,
            load_checkpoint_guess_config=native,
            load_diffusion_model=native,
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        with self.assertRaisesRegex(ValueError, "Capture.*MiniMax"):
            scope["load_base_model"](NS(), path.name)
        native.assert_not_called()

    def test_malformed_family_metadata_does_not_promote_an_unknown_header(self):
        for family in ({}, [], False):
            with self.subTest(family=family):
                admission.require_available_image_workflow(family=family)

    def test_unknown_and_malformed_encoder_headers_remain_unknown(self):
        for value in ({}, {"unrelated.weight": (2, 2)}, {"bad": (True,)}, {"bad": (-1,)}):
            with self.subTest(value=value):
                self.assertIsNone(text.passive_native_text_type(value))

    def test_cold_subprocess_blocks_torch_and_comfy_imports_for_both_native_role_readers(self):
        code = """
import importlib.abc,sys
from pathlib import Path
class NoRuntime(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','comfy'}: raise RuntimeError('runtime import:'+name)
sys.meta_path.insert(0,NoRuntime())
from ruinedfooocus.features.models import native_text_identity as text
from ruinedfooocus.features.models import native_workflow_admission as admission
from ruinedfooocus.features.models import utility_models as utility
from ruinedfooocus.features.models.model_identity import file_signature
text._source=lambda:Path(sys.argv[1])/'comfy/sd.py'
signatures=lambda:tuple(tuple(file_signature(Path(sys.argv[1])/'comfy'/n)) for n in ('model_detection.py','supported_models.py','supported_models_base.py'))
admission._native_source_signatures=signatures
utility._native_source_signatures=signatures
assert text.passive_native_text_type({'mask_decoder.query_builder.content_proj.weight':(1,1),'model.visual.merger.linear_fc2.weight':(2048,1)})=='WEVISONE_2B'
assert admission.planned_image_family(sys.argv[2])=='MiniMaxH3Image'
assert 'torch' not in sys.modules and 'comfy' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(NATIVE), str(self.file(minimax_shapes()))],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
