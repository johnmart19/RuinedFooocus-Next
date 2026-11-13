"""Create prompt batches reach the native adapter with their captured child LoRAs."""

from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import random
import re
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.model_folders import model_folders
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.gallery.image_metadata import read_metadata
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.models.effective_image_plan import get_effective_image_plan
from ruinedfooocus.features.models.lora_names import merge_loras
import test_lora_source_capture as lora_fixtures
from ui_test_helpers import function_from

WORKER = "ruinedfooocus/app/async_worker.py"
PROMPTS = "ruinedfooocus/features/prompting/prompt_processing.py"


class BatchPaths:
    get_abspath = staticmethod(Path)

    def __init__(self, root, lora_root):
        self.paths = {
            "path_checkpoints": [str(root)],
            "path_loras": [str(lora_root)],
        }
        self.DOWNLOADABLE_FILES = {}
        self.model_paths = {"cache_path": root / "cache"}

    def get_model_folders(self, category):
        return model_folders(self.paths, category, self.get_abspath)


class WorkerPromptBatchTests(unittest.TestCase):
    def setUp(self):
        self.files = lora_fixtures.LoraSourceCaptureTests()
        self.files.setUp()
        self.addCleanup(self.files.doCleanups)
        self.checkpoint = self.files.root / "checkpoint.safetensors"
        header = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
        self.checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        self.files.models.offline = True
        self.files.models.get_file.side_effect = lambda category, name: (
            self.checkpoint if category == "checkpoints" else self.files.find(category, name)
        )
        self.files.models.get_models_by_path = lambda *args, **kwargs: {"baseModel": "SDXL"}
        self.files.models.get_model_base = lambda row: row["baseModel"]
        self.manager = BatchPaths(self.files.root, self.files.first)
        self.shared = SimpleNamespace(
            models=self.files.models,
            path_manager=self.manager,
            settings=SimpleNamespace(default_settings={"image_engine": "native"}),
            performance_settings=SimpleNamespace(apply=lambda data, family: dict(data)),
            resolution_settings=SimpleNamespace(
                CUSTOM_RESOLUTION="Custom...", aspect_ratios={"square": (64, 64)}
            ),
            state={"interrupted": False},
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": self.shared}))
        app = importlib.import_module("ruinedfooocus.app")
        self.enterContext(patch.object(app, "state", self.shared, create=True))
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"])
                    )
                },
            )
        )
        self.events = Mock()
        self.scope = dict(
            shared=self.shared,
            settings=self.shared.settings,
            control_tools=SimpleNamespace(get_settings=lambda data: data.get("controlnet") or {}),
            check_interrupt=Mock(),
            add_result=self.events,
            describe_error=str,
            resource_path=resource_path,
            random=random,
            Path=Path,
            re=re,
            merge_loras=merge_loras,
            process_prompt=Mock(
                side_effect=lambda styles, prompt, negative, data: (prompt, negative)
            ),
            shift_attention=lambda text, distance: text,
            TimeIt=lambda *args: nullcontext(),
            model_management=SimpleNamespace(InterruptProcessingException=InterruptedError),
            traceback=SimpleNamespace(print_exc=Mock()),
        )
        for path, name in (
            (PROMPTS, "get_promptlist"),
            (PROMPTS, "process_metadata"),
            (PROMPTS, "parse_loras"),
            (WORKER, "is_sha256_hash"),
            (WORKER, "capture_image_data"),
            (WORKER, "reset_preview"),
            (WORKER, "_process"),
            (WORKER, "process"),
        ):
            function_from(path, name, self.scope)

    def test_create_batch_dispatches_exact_original_child_loras_after_roots_change(self):
        style = self.files.file("style.safetensors")
        other = self.files.file("other.safetensors")
        self.files.file(style.name, self.files.second).write_bytes(b"later style source")
        self.files.file(other.name, self.files.second).write_bytes(b"later other source")
        originals = [
            "first <lora:style:0.7>",
            "second <lora:style:-0.2> <lora:other:1>",
            "third <lora:style:0>",
        ]
        submitted = self.scope["get_promptlist"]({"prompt": "\n---\n".join(originals)})
        self.assertEqual(submitted, originals)
        captured = self.scope["capture_image_data"](
            dict(
                task_type="process",
                task_id=1,
                base_model_name=self.checkpoint.name,
                base_model_hash="inert checkpoint hash",
                prompt=submitted,
                negative="",
                loras=[["thumbnail", "0.4 - style.safetensors"]],
                style_selection=[],
                performance_selection="Own settings",
                custom_steps=9,
                cfg=3.5,
                sampler_name="euler",
                scheduler="normal",
                clip_skip=1,
                aspect_ratios_selection="Custom...",
                custom_width=64,
                custom_height=64,
                image_number=1,
                image_total=3,
                seed=17,
                controlnet={},
                automatic_upscale="Off",
                generate_forever=False,
                silent=True,
            )
        )
        plan = get_effective_image_plan(captured)
        self.assertIsNotNone(plan)
        self.assertEqual(captured["prompt"], originals)
        self.files.models.model_dirs["loras"] = [self.files.second]
        self.manager.paths["path_loras"][0] = str(self.files.second)
        loader, owner, reader, native = self.files.loader()
        actual_loader = MethodType(loader, owner)
        packets, executed = [], []

        def load_loras(rows, *, captured_loras):
            packets.append(tuple((item.path, item.weight) for item in captured_loras))
            return actual_loader(rows, captured_loras=captured_loras)

        def process_image(*, gen_data, callback):
            executed.append((gen_data["prompt"], gen_data["positive_prompt"]))
            return []

        owner.pipeline_type = ["sdxl"]
        owner.parse_gen_data = lambda data: data
        owner.load_base_model = Mock()
        owner.load_loras = Mock(side_effect=load_loras)
        owner.process = Mock(side_effect=process_image)
        self.scope["pipelines"] = SimpleNamespace(update=Mock(return_value=owner))
        self.scope["process"](captured)
        self.assertEqual(
            packets,
            [((style, 0.7),), ((other, 1), (style, -0.2)), ()],
        )
        self.assertEqual([row[0] for row in executed], originals)
        self.assertTrue(all("<lora:" not in row[1] for row in executed))
        self.assertEqual([call.args[3] for call in native.call_args_list], [0.7, 1, -0.2])
        self.assertEqual({call.args[0] for call in reader.call_args_list}, {str(style), str(other)})
        self.assertIs(owner.xl_base_patched, owner.xl_base)
        self.files.models.get_model_path.assert_not_called()
        self.assertFalse(any(call.args[1] == "error" for call in self.events.call_args_list))

        owner.load_loras.reset_mock()
        owner.load_base_model.reset_mock()
        owner.process.reset_mock()
        self.scope["pipelines"].update.reset_mock()
        reader.reset_mock()
        native.reset_mock()
        self.events.reset_mock()
        with self.assertRaisesRegex(ValueError, "original image prompt was not captured"):
            self.scope["process"](dict(captured, prompt=["uncaptured child <lora:style:0.7>"]))
        self.scope["pipelines"].update.assert_not_called()
        owner.load_base_model.assert_not_called()
        owner.load_loras.assert_not_called()
        owner.process.assert_not_called()
        reader.assert_not_called()
        native.assert_not_called()
        self.events.assert_not_called()

    def test_mixed_saved_settings_batch_rejects_before_image_resources_or_processing(self):
        saved = json.dumps({"Prompt": "saved image prompt", "Negative": "", "steps": 7})
        prompts = self.scope["get_promptlist"]({"prompt": saved + "\n---\nordinary prompt"})
        self.assertEqual(len(prompts), 2)
        self.assertIsNotNone(read_metadata(prompts[0]))
        request = {"task_type": "process", "prompt": prompts}
        self.scope["pipelines"] = SimpleNamespace(update=Mock())
        state_before = dict(self.shared.state)
        with (
            patch.object(assets, "capture_pending_image_selection") as pending,
            patch.object(assets, "capture_image_assets") as capture,
            patch.object(assets, "prepare_image_assets") as prepare,
        ):
            with self.assertRaisesRegex(
                ValueError, "Apply saved image settings before starting a prompt batch"
            ):
                self.scope["capture_image_data"](request)
            pending.assert_not_called()
            capture.assert_not_called()
            prepare.assert_not_called()
        self.scope["pipelines"].update.assert_not_called()
        self.scope["process_prompt"].assert_not_called()
        self.files.models.get_file.assert_not_called()
        self.files.models.get_model_path.assert_not_called()
        self.events.assert_not_called()
        self.assertEqual(self.shared.state, state_before)
        self.assertEqual(request, {"task_type": "process", "prompt": prompts})


if __name__ == "__main__":
    unittest.main()
