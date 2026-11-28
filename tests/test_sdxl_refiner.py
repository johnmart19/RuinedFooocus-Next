"""Standalone native refiner components, ADM inputs and request-local cache context."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.image_editing.native_conditioning import refiner_text_conditioning
from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
from test_model_cache import method
from test_native_model_coverage import route_info
import test_model_recommendations as recommendation_fixtures

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
PIPELINE = "ruinedfooocus/features/generation/image_pipeline.py"


def native_refiner_node():
    source = NATIVE / "comfy_extras/nodes_clip_sdxl.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CLIPTextEncodeSDXLRefiner"
    )
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
    node.body[0].returns = None
    scope = dict(io=SimpleNamespace(NodeOutput=lambda value: (value,)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(CLIPTextEncodeSDXLRefiner=scope[node.name])


def clip_fixture():
    def encode(tokens, add_dict=None):
        return [[tokens, {"pooled_output": "native-G-pooled", **(add_dict or {})}]]

    return SimpleNamespace(
        tokenize=Mock(side_effect=lambda text: {"g": text}),
        encode_from_tokens_scheduled=Mock(side_effect=encode),
    )


def text_encoder(clip):
    warm = Mock()
    generic = Mock()
    generic.encode.return_value = ([["generic", {}]],)
    run = method(
        PIPELINE,
        "textencode",
        dict(
            CLIPTextEncode=lambda: generic,
            warm_patched_text_encoder=warm,
            path_manager=SimpleNamespace(
                model_paths={"embedding_path": [], "cache_path": ROOT / "tmp"}
            ),
        ),
    )
    owner = SimpleNamespace(
        xl_base_patched=SimpleNamespace(clip=clip),
        conditions={key: {"text": None, "cache": None} for key in ("+", "-", "switch")},
    )
    return run, owner, warm, generic


class SdxlRefinerTests(unittest.TestCase):
    def test_native_refiner_corrects_stale_sdxl_label_for_request_recipe_without_rewriting_metadata(
        self,
    ):
        lookup = method(
            "ruinedfooocus/features/models/model_handler.py",
            "get_models_by_path",
            dict(json=json, Path=Path),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "arbitrary.safetensors"
            checkpoint.write_bytes(b"identity-only fixture")
            cached = root / "arbitrary.json"
            publisher = dict(
                baseModel="SDXL 1.0",
                name="Publisher title",
                modelId=42,
                rf_source="original publisher",
            )
            cached.write_text(json.dumps(publisher))
            before = cached.read_bytes()
            owner = SimpleNamespace(
                cache_paths={"checkpoints": root},
                offline=True,
                detect_checkpoint_base=lambda path: "SDXLRefiner",
            )
            self.assertEqual(
                lookup(owner, "checkpoints", checkpoint), publisher | dict(baseModel="SDXLRefiner")
            )
            self.assertEqual(cached.read_bytes(), before)

    def test_standalone_refiner_has_only_native_g_encoder_and_sdxl_vae(self):
        get_info = method(
            PIPELINE, "get_clip_and_vae", dict(settings=SimpleNamespace(default_settings={}))
        )
        model = SimpleNamespace(
            model=type("SDXLRefiner", (), {"model_config": SimpleNamespace(unet_config={})})()
        )
        info = get_info(SimpleNamespace(known_model_info=route_info()), model)
        self.assertEqual(info["clip_type"], "STABLE_DIFFUSION")
        self.assertEqual(info["clip_keys"], ["clip_g"])
        self.assertEqual(info["vae_key"], "vae_sdxl")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "arbitrary.safetensors"
            raw = json.dumps({"weight": {"shape": [1]}}).encode()
            checkpoint.write_bytes(len(raw).to_bytes(8, "little") + raw)
            paths = SimpleNamespace(
                paths={"path_clip": [str(root)], "path_vae": str(root)},
                get_abspath=Path,
                DOWNLOADABLE_FILES={},
            )
            state = inventory(checkpoint, "SDXLRefiner", {}, paths)
            self.assertEqual([row["key"] for row in state["rows"]], ["clip_g", "vae_sdxl"])

    def test_single_g_native_loader_selects_refiner_text_model(self):
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "te_model == TEModel.CLIP_G"
        )
        target = SimpleNamespace()
        scope = dict(
            te_model="G",
            TEModel=SimpleNamespace(CLIP_G="G"),
            clip_type="STABLE_DIFFUSION",
            CLIPType=SimpleNamespace(STABLE_CASCADE="CASCADE", SD3="SD3", HIDREAM="HIDREAM"),
            clip_target=target,
            sdxl_clip=SimpleNamespace(
                SDXLRefinerClipModel="refiner-G", SDXLTokenizer="native-SDXL"
            ),
        )
        exec(
            compile(ast.Module(body=[branch], type_ignores=[]), "native_single_G_loader", "exec"),
            scope,
        )
        self.assertEqual((target.clip, target.tokenizer), ("refiner-G", "native-SDXL"))

    def test_native_refiner_node_preserves_prompt_and_adds_request_geometry_and_aesthetic(self):
        clip = clip_fixture()
        text = 'LEFT---RIGHT: a sign reads "OPEN 24/7"'
        with patch.dict("sys.modules", {"comfy_extras.nodes_clip_sdxl": native_refiner_node()}):
            positive = refiner_text_conditioning(clip, text, 1152, 896)
            negative = refiner_text_conditioning(clip, "blur", 1152, 896, negative=True)
        self.assertEqual(positive[0][0], {"g": text})
        self.assertEqual(
            positive[0][1],
            dict(pooled_output="native-G-pooled", aesthetic_score=6.0, width=1152, height=896),
        )
        self.assertEqual(negative[0][1]["aesthetic_score"], 2.5)

    def test_actual_text_cache_reencodes_same_prompt_when_refiner_geometry_changes(self):
        clip = clip_fixture()
        run, owner, warm, generic = text_encoder(clip)
        with (
            patch.dict("sys.modules", {"comfy_extras.nodes_clip_sdxl": native_refiner_node()}),
            patch(
                "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
                side_effect=lambda text, *args: text,
            ),
        ):
            run(owner, "+", "unchanged words", 1, refiner_size=(1024, 768))
            first = owner.conditions["+"]["cache"]
            run(owner, "+", "unchanged words", 1, refiner_size=(1024, 768))
            self.assertEqual(clip.tokenize.call_count, 1)
            self.assertIs(owner.conditions["+"]["cache"], first)
            run(owner, "+", "unchanged words", 1, refiner_size=(768, 1024))
            self.assertEqual(clip.tokenize.call_count, 2)
            self.assertEqual(first[0][1]["width"], 1024)
            self.assertEqual(owner.conditions["+"]["cache"][0][1]["width"], 768)
            self.assertEqual(warm.call_count, 2)
            run(owner, "+", "unchanged words", 1)
        generic.encode.assert_called_once_with(clip=clip, text="unchanged words")
        self.assertNotIn("width", owner.conditions["+"]["cache"][0][1])

    def test_actual_process_passes_geometry_to_positive_negative_and_prompt_switches(self):
        tree = ast.parse((ROOT / PIPELINE).read_text(encoding="utf-8"))
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "image_edit and input_images > 0"
        )
        clip = clip_fixture()
        run, owner, _, _ = text_encoder(clip)
        owner.textencode = MethodType(run, owner)
        scope = dict(
            self=owner,
            callback=None,
            architecture="SDXLRefiner",
            positive_prompt="early and late",
            negative_prompt="blur",
            clip_skip=1,
            gen_data=dict(width=768, height=1152, steps=2),
            pp=SimpleNamespace(prompt_switch_per_step=lambda text, steps: ["early", "late"]),
            switched_prompt=[],
            tensor_conditions={},
            conditioning_set_values=lambda cond, values: [
                [text, meta | values] for text, meta in cond
            ],
        )
        with (
            patch.dict("sys.modules", {"comfy_extras.nodes_clip_sdxl": native_refiner_node()}),
            patch(
                "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
                side_effect=lambda text, *args: text,
            ),
        ):
            exec(compile(ast.Module(body=branch.orelse, type_ignores=[]), PIPELINE, "exec"), scope)
        self.assertEqual(owner.conditions["-"]["cache"][0][1]["aesthetic_score"], 2.5)
        for entry in owner.conditions["+"]["cache"] + scope["switched_prompt"]:
            self.assertEqual(
                (entry[1]["width"], entry[1]["height"], entry[1]["aesthetic_score"]),
                (768, 1152, 6.0),
            )
        self.assertEqual(
            [entry[1]["_rf_prompt_step"] for entry in scope["switched_prompt"]], [0, 1]
        )

    def test_unqualified_refiner_auto_remains_strict_and_controls_remain_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            perf = recommendation_fixtures.RecommendationTests()._performance(Path(folder))
            perf.model_recommendations.offline = True
            with patch.dict(
                "sys.modules",
                {
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
                    )
                },
            ):
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", "local.safetensors", "SDXLRefiner")
                self.assertIn(
                    "Choose Own settings",
                    perf.control_options("Model Defaults", "local.safetensors", "SDXLRefiner")[
                        "_recipe_unavailable"
                    ],
                )
            manual = dict(
                performance_selection="Own settings", cfg=3.0, custom_steps=12, clip_skip=1
            )
            self.assertEqual(perf.apply(manual, "SDXLRefiner"), manual)


if __name__ == "__main__":
    unittest.main()
