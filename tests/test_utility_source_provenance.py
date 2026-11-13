"""Processing PNGs retain inert source recipes and truthful current output facts."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.gallery.editor_session import EditorSessionStore
from ruinedfooocus.features.gallery.image_metadata import read_metadata
from ruinedfooocus.features.generation.request_metadata import (
    SOURCE_GENERATION,
    capture_source_generation,
)
from ruinedfooocus.features.generation.image_output_batches import ImageOutputBatch
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]
OWNER = ("utility-provenance-fixture",)


def png(path, size, metadata, *, alpha=255, mode="RGBA"):
    info = PngInfo()
    info.add_text("parameters", json.dumps(metadata))
    Image.new(mode, size, (20, 40, 60, alpha) if mode == "RGBA" else (20, 40, 60)).save(
        path, pnginfo=info
    )
    return path


def worker_metadata(data, size):
    tree = ast.parse((ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process")
    loop = next(
        n
        for n in ast.walk(process)
        if isinstance(n, ast.For) and ast.unparse(n.iter) == "enumerate(imgs)"
    )
    start = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "PngInfo"
    )
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    scope = dict(
        sampling=False,
        gen_data=dict(data, width=size[0], height=size[1]),
        width=size[0],
        height=size[1],
        control_tools=SimpleNamespace(get_settings=lambda d: d["controlnet"]),
        pipeline=SimpleNamespace(pipeline_type=["rembg"]),
        imgs=(pixels,),
        x=pixels,
        ImageOutputBatch=ImageOutputBatch,
        output_index=0,
        deepcopy=deepcopy,
    )
    exec(
        compile(
            ast.Module(body=loop.body[start:end], type_ignores=[]),
            "worker-utility-metadata",
            "exec",
        ),
        scope,
    )
    return scope["prompt"]


class UtilitySourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rf-utility-provenance-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = EditorSessionStore(self.root / "sessions")
        self.root_recipe = dict(
            prompt="J0", base_model_name="old.safetensors", seed=0, width=12, height=10
        )
        self.recipe = dict(
            prompt="J1 rendered",
            raw_prompt="J1 original",
            negative="none",
            raw_negative="raw none",
            base_model_name="Janku.safetensors",
            seed=20261112,
            steps=7,
            width=8,
            height=14,
            loras=[],
        )
        self.original = png(self.root / "J0.png", (12, 10), self.root_recipe)
        self.snapshot = self.store.start(self.original, provenance=self.root_recipe, owner=OWNER)

    def ticket(self, operation="rembg"):
        return self.store.capture(
            session_id=self.snapshot.session_id,
            revision=self.snapshot.revision,
            source_id=self.snapshot.current.artifact_id,
            operation=operation,
            parameters={},
            owner=OWNER,
        )

    def generation(self, *, metadata=None):
        ticket = self.ticket("img2img")
        path = png(self.root / "J1.png", (8, 14), metadata or self.recipe)
        self.snapshot = self.store.complete(
            ticket,
            path,
            execution=dict(
                metadata or self.recipe,
                kind="generation",
                geometry={"policy": "fixture", "source": [12, 10], "result": [8, 14]},
            ),
            owner=OWNER,
        )
        return path

    def test_real_worker_png_keeps_latest_source_generation_separate_and_originals_unchanged(self):
        source = self.generation()
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.original, source)}
        ticket = self.ticket()
        captured = capture_source_generation(
            {
                "_editor_operation": ticket,
                "controlnet": {"type": "rembg"},
                "rembg_native_model": "BiRefNet-fixture",
            },
            store=self.store,
            owner=OWNER,
        )
        self.assertEqual(captured[SOURCE_GENERATION]["seed"], 20261112)
        metadata = worker_metadata(captured, (16, 28))
        output = png(self.root / "F.png", (16, 28), metadata)
        with Image.open(output) as image:
            actual = json.loads(image.info["parameters"])
            self.assertEqual(image.size, (16, 28))
        self.assertEqual((actual["width"], actual["height"]), (16, 28))
        self.assertEqual(actual["kind"], "processing")
        self.assertEqual(actual["controlnet"], {"type": "rembg"})
        self.assertNotIn("Prompt", actual)
        self.assertNotIn("base_model_name", actual)
        self.assertNotIn("seed", actual)
        replay = read_metadata(actual)
        self.assertEqual(
            (replay["prompt"], replay["raw_prompt"], replay["base_model_name"]),
            ("J1 rendered", "J1 original", "Janku.safetensors"),
        )
        self.assertEqual((replay["width"], replay["height"]), (16, 28))
        self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes}, hashes)
        self.snapshot = self.store.complete(
            ticket,
            output,
            execution=dict(
                actual, geometry={"policy": "fixture", "source": [8, 14], "result": [16, 28]}
            ),
            owner=OWNER,
        )
        again = capture_source_generation(
            {"_editor_operation": self.ticket()}, store=self.store, owner=OWNER
        )
        self.assertEqual(again[SOURCE_GENERATION]["prompt"], "J1 rendered")
        self.assertNotIn("source_generation", again[SOURCE_GENERATION])

    def test_undo_uses_current_head_and_missing_new_generation_never_substitutes_old_root(self):
        self.generation()
        self.snapshot = self.store.undo(
            self.snapshot.session_id, self.snapshot.revision, owner=OWNER
        )
        captured = capture_source_generation(
            {"_editor_operation": self.ticket()}, store=self.store, owner=OWNER
        )
        self.assertEqual(captured[SOURCE_GENERATION]["prompt"], "J0")
        self.store.cancel(captured["_editor_operation"], owner=OWNER)
        self.snapshot = self.store.redo(
            self.snapshot.session_id, self.snapshot.revision, owner=OWNER
        )
        self.snapshot = self.store.undo(
            self.snapshot.session_id, self.snapshot.revision, owner=OWNER
        )
        self.generation(metadata={"kind": "generation"})
        missing = capture_source_generation(
            {"_editor_operation": self.ticket()}, store=self.store, owner=OWNER
        )
        self.assertNotIn(SOURCE_GENERATION, missing)

    def test_untrusted_fallback_is_bounded_inert_and_never_opens_source_or_model_paths(self):
        from ruinedfooocus.features.gallery.actions import resolve_setup

        hostile = dict(
            self.recipe,
            base_model_name="C:/untrusted/foreign.safetensors",
            image_source={"filename": "../../private.png"},
            task_type="tool_call",
            _api_job={"owner": "other"},
            editor_context={"source_id": "foreign"},
            model_recipe_path="../../private.json",
            native_composition_source={"path": "../../private.png"},
            native_tensor_source={"source_id": "foreign"},
            image_utility_source={"source_id": "foreign"},
        )
        with patch.object(
            Path, "open", side_effect=AssertionError("Metadata followed an untrusted path")
        ):
            replay = read_metadata(
                {"kind": "processing", "width": 16, "height": 28, "source_generation": hostile}
            )
            self.assertNotIn("_api_job", replay)
            self.assertNotIn("task_type", replay)
            self.assertNotIn("editor_context", replay)
            for key in (
                "image_source",
                "model_recipe_path",
                "native_composition_source",
                "native_tensor_source",
                "image_utility_source",
            ):
                self.assertNotIn(key, replay)
            models = SimpleNamespace(
                get_names=lambda _: ["installed.safetensors"],
                get_file=Mock(side_effect=AssertionError("Uninstalled path opened")),
            )
            with self.assertRaisesRegex(ValueError, "Cannot uniquely match"):
                resolve_setup({"kind": "processing", "source_generation": hostile}, models)
            models.get_file.assert_not_called()
            captured = capture_source_generation(
                {SOURCE_GENERATION: hostile, "image_source": {"filename": "../../private.png"}}
            )
            self.assertNotIn(SOURCE_GENERATION, captured)
        for value in (
            None,
            {},
            {"prompt": 7},
            {"prompt": "x" * 131073},
            {"prompt": "ok", "cfg": float("nan")},
        ):
            with self.subTest(value_type=type(value).__name__):
                self.assertIsNone(read_metadata({"kind": "processing", "source_generation": value}))
        self.assertIsNone(
            read_metadata({"kind": "processing", "image_source": {"filename": "J1.png"}})
        )

    def test_processing_replay_uses_current_target_without_mutating_source_size_recipe(self):
        from dataclasses import asdict
        from ruinedfooocus.features.generation.size_plan import SizePlan, replay_size
        from ruinedfooocus.features.generation.request_metadata import attach_source_generation

        source = dict(
            self.recipe,
            width=832,
            height=1216,
            generation_width=416,
            generation_height=608,
            size_plan=asdict(SizePlan.create((832, 1216), scale=2)),
            automatic_upscale="2× fixture",
            automatic_upscale_target="Input image only",
        )
        before = deepcopy(source)
        raw = attach_source_generation(
            {SOURCE_GENERATION: source}, {"kind": "processing", "width": 1664, "height": 2432}
        )
        self.assertEqual(raw["source_generation"], before)
        replay = read_metadata(raw)
        self.assertEqual(replay_size(replay), (1664, 2432))
        self.assertNotIn("size_plan", replay)
        self.assertNotIn("generation_width", replay)
        self.assertEqual(replay["automatic_upscale"], "Off")
        self.assertEqual(raw["source_generation"], before)
        for dimensions in (
            (None, None),
            (True, 2),
            (-1, 14),
            (8.5, 14),
            (float("nan"), 14),
            (16777217, 1),
        ):
            malformed = read_metadata(dict(raw, width=dimensions[0], height=dimensions[1]))
            self.assertNotIn("width", malformed)
            self.assertNotIn("height", malformed)
            self.assertNotIn("size_plan", malformed)


class UtilityOpaqueAlphaTests(unittest.TestCase):
    def test_stage_and_commit_admit_only_redundant_opaque_alpha_without_changing_frame(self):
        for alpha in (255, 128):
            with (
                self.subTest(alpha=alpha),
                tempfile.TemporaryDirectory(prefix="rf-utility-alpha-") as folder,
            ):
                root = Path(folder)
                source = png(root / "source.png", (12, 10), {"prompt": "source"}, alpha=alpha)
                output = png(root / "depth.png", (12, 10), {"kind": "processing"}, mode="RGB")
                before = source.read_bytes()
                store = EditorSessionStore(root / "sessions")
                snapshot = store.start(source, provenance={"prompt": "source"}, owner=OWNER)
                ticket = store.capture(
                    session_id=snapshot.session_id,
                    revision=snapshot.revision,
                    source_id=snapshot.current.artifact_id,
                    operation="depth-map",
                    parameters={},
                    owner=OWNER,
                )
                job = SimpleNamespace(state={}, owner=OWNER)
                scope = dict(
                    json=json,
                    current_job=lambda: job,
                    check_interrupt=lambda _: None,
                    validate_editor_data=lambda data: store.validate(
                        data["_editor_operation"], owner=OWNER
                    ),
                )
                function_from(
                    "ruinedfooocus/app/async_worker.py", "_source_has_opaque_alpha", scope
                )
                stage = function_from(
                    "ruinedfooocus/app/async_worker.py", "stage_editor_result", scope
                )
                data = {"_editor_operation": ticket, "_sampling_captured": False, "_job": job}
                if alpha < 255:
                    with self.assertRaisesRegex(ValueError, "did not preserve its source alpha"):
                        stage(data, [str(output)])
                    self.assertNotIn("_editor_publication", job.state)
                else:
                    stage(data, [str(output)])
                    path, execution = job.state["_editor_publication"]
                    self.assertEqual(execution["geometry"]["source"], [12, 10])
                    self.assertEqual(execution["geometry"]["result"], [12, 10])
                    self.assertIn("fully opaque", execution["alpha_policy"])
                    completed = store.complete(ticket, path, execution=execution, owner=OWNER)
                    self.assertEqual(completed.current.frame.size, (12, 10))
                    self.assertEqual(completed.current.frame.alpha, "none")
                self.assertEqual(source.read_bytes(), before)
