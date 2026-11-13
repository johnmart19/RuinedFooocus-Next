"""Public API, Library replay and production process hook boundaries."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.settings_schema import validate_known
from ruinedfooocus.features.generation import image_preparation
from ruinedfooocus.features.generation.defaults import generation_defaults, STARTUP_DEFAULTS
from ruinedfooocus.features.gallery.image_metadata import render_values
from ruinedfooocus.features.image_editing import lora_hooks
from ruinedfooocus.integrations.openai_api import ImageRequest
import test_api_native_inpaint as api_fixture
import test_gallery_actions as gallery_fixture
import test_library_editing_workflow as library_fixture
import test_native_lora_hooks as hook_fixture

ROOT = Path(__file__).resolve().parents[1]


class LoraHookCallerTests(unittest.TestCase):
    def fixture(self, cls):
        instance = cls()
        instance.setUp()
        self.addCleanup(instance.doCleanups)
        return instance

    def api(self):
        api = self.fixture(api_fixture.NativeInpaintApiTests)
        root = Path(api.selected["path"]).parent
        (root / "loras").mkdir()
        (root / "loras/schedule.safetensors").write_bytes(b"inert local capture fixture")
        state = sys.modules["ruinedfooocus.app.state"]
        state.path_manager.get_model_folders = lambda category: [root / category]
        return api

    def test_public_api_captures_exact_hook_plan_artifact_and_prompt_static_loras(self):
        api = self.api()
        response = api.client.post(
            "/images/generations",
            json=dict(
                prompt="unchanged literal prompt",
                size="512x512",
                lora_hook_mode="interpolated",
                lora_hook_model="schedule.safetensors",
                lora_hook_model_strength=-0.5,
                lora_hook_clip_strength=0.25,
                lora_hook_schedule_clip=True,
                lora_hook_interpolation="ease_in",
                lora_hook_keyframe_count=11,
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = api.prepared[-1]
        plan = data[lora_hooks.KEY]
        self.assertEqual(plan.family, "SDXL")
        self.assertEqual(plan.settings()["lora_hook_keyframe_count"], 11)
        self.assertEqual(plan.artifact.name, "schedule.safetensors")
        self.assertIn(plan.artifact, data[image_preparation.KEY].artifacts)
        self.assertEqual(data["prompt"], "unchanged literal prompt")
        self.assertEqual(data["image_engine"], "native")

    def test_public_api_malformed_schedule_is_rejected_before_job_submission(self):
        api = self.api()
        for values in (
            dict(lora_hook_keyframes='[{"start":0.2,"strength":1}]'),
            dict(lora_hook_keyframe_count=1000),
            dict(lora_hook_model="../outside.pt"),
            dict(lora_hook_start=0.8, lora_hook_end=0.2),
        ):
            response = api.client.post(
                "/images/generations",
                json=dict(
                    prompt="owned",
                    lora_hook_mode="keyframes",
                    lora_hook_model="schedule.safetensors",
                )
                | values,
            )
            self.assertIn(response.status_code, (400, 422), response.text)
        api.submit.assert_not_called()

    def test_schema_defaults_settings_and_metadata_round_trip_without_private_objects(self):
        values = dict(
            lora_hook_mode="keyframes",
            lora_hook_model="nested/owned.pt",
            lora_hook_clip_strength=-0.75,
            lora_hook_schedule_clip=True,
        )
        model = ImageRequest(prompt="owned", **values)
        self.assertEqual(model.lora_hook_clip_strength, -0.75)
        normalized = validate_known(values)
        self.assertEqual(normalized, values)
        defaults = generation_defaults(dict(STARTUP_DEFAULTS, lora_hook_model="nested/owned.pt"))
        self.assertEqual(defaults["lora_hook_mode"], "off")
        self.assertEqual(defaults["lora_hook_model"], "nested/owned.pt")
        metadata = render_values(
            dict(
                prompt="saved prompt",
                base_model_name="owned",
                **values,
                native_lora_hooks={"target": "primary-checkpoint"},
                _lora_hook_plan=object(),
            )
        )
        self.assertEqual(metadata["lora_hook_clip_strength"], -0.75)
        self.assertNotIn(lora_hooks.KEY, metadata)
        self.assertEqual(metadata["native_lora_hooks"]["target"], "primary-checkpoint")

    def test_plain_library_edit_resets_only_scheduled_lora_and_preserves_static_loras_and_source(
        self,
    ):
        library = self.fixture(library_fixture.LibraryRequestTests)
        library.controls.update(
            lora_hook_mode="keyframes", lora_hook_model="stale.pt", lora_hook_clip_strength=0.4
        )
        request = library.build(strength=0.37)
        self.assertEqual(request["lora_hook_mode"], "off")
        self.assertEqual(request["lora_hook_model"], "")
        self.assertEqual(request["loras"], library.controls["loras"])
        self.assertEqual(request["input_image"].getpixel((0, 0)), (20, 40, 60, 127))
        self.assertEqual(request["custom_width"], library.controls["custom_width"])

    def test_explicit_library_metadata_replay_restores_fields_and_keeps_plan_uncaptured(self):
        functions = gallery_fixture.functions()
        values = dict(
            prompt="owned",
            base_model_name="model.safetensors",
            lora_hook_mode="interpolated",
            lora_hook_model="schedule.safetensors",
            lora_hook_keyframe_count=8,
            lora_hook_model_strength=-1.25,
        )
        from ruinedfooocus.features.image_editing.image_guidance import (
            restore_guidance_metadata,
            capabilities,
            resolve_guidance,
        )

        functions.update(
            restore_guidance_metadata=restore_guidance_metadata,
            capabilities=capabilities,
            resolve_guidance=resolve_guidance,
        )
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_metadata"
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "production metadata replay", "exec"),
            functions,
        )
        with patch(
            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
            return_value="SDXL",
        ):
            updates = functions["apply_metadata"](values, "model.safetensors", None)
        for key in (
            "lora_hook_mode",
            "lora_hook_model",
            "lora_hook_keyframe_count",
            "lora_hook_model_strength",
        ):
            self.assertEqual(updates[key]["value"], values[key])
        gallery = self.fixture(gallery_fixture.GalleryActionsTests)
        values.update(base_model_name="new/model.safetensors", loras=[])
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app": NS(
                        state=NS(models=gallery.models, settings=NS(default_settings={}))
                    ),
                    "ruinedfooocus.core.util": NS(
                        get_lora_thumbnail=lambda *_: "thumbnail", load_keywords=lambda *_: ""
                    ),
                },
            ),
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                return_value="SDXL",
            ),
        ):
            replayed = functions["reuse_setup"](values)
        self.assertEqual(replayed["lora_hook_mode"]["value"], "interpolated")
        self.assertEqual(replayed["lora_hook_model_strength"]["value"], -1.25)
        reset = functions["reuse_prompt"](values)
        self.assertEqual(reset["lora_hook_mode"]["value"], "off")
        captured = image_preparation.image_control_data(
            dict(prompt=json.dumps(values), controlnet={})
        )
        self.assertEqual(captured["lora_hook_keyframe_count"], 8)
        self.assertNotIn(lora_hooks.KEY, captured)

    def test_actual_process_decorator_and_final_conditioning_call_use_request_scope(self):
        files = self.fixture(hook_fixture.NativeLoraHookTests)
        data = files.capture(lora_hook_clip_strength=0.4, lora_hook_schedule_clip=True)
        owner, original = files.owner()
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        self.assertTrue(
            any(isinstance(n, ast.Attribute) and n.attr == "scoped" for n in process.decorator_list)
        )
        attach = next(
            n
            for n in process.body
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "conditioning"
        )
        positive, negative = [["positive", {"start_percent": 0.3}]], [["negative", {}]]

        def encode_and_sample(request_owner, gen_data=None, callback=None):
            self.assertIsNot(request_owner.xl_base_patched, original)
            self.assertTrue(request_owner.xl_base_patched.clip.use_clip_schedule)
            scope = dict(
                self=request_owner,
                positive_cond=positive,
                negative_conditioning=negative,
                lora_hooks=lora_hooks,
            )
            exec(
                compile(
                    ast.Module(body=[attach], type_ignores=[]),
                    "production conditioning hook call",
                    "exec",
                ),
                scope,
            )
            self.assertIn("hooks", scope["positive_cond"][0][1])
            self.assertIn("hooks", scope["negative_conditioning"][0][1])
            self.assertEqual(scope["positive_cond"][0][1]["start_percent"], 0.3)
            return "owned-sampler-result"

        worker = NS(check_interrupt=Mock())
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.async_worker": worker,
                    "ruinedfooocus.app.state": NS(path_manager=files.manager),
                },
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "async_worker", worker, create=True),
        ):
            self.assertEqual(
                lora_hooks.scoped(encode_and_sample)(owner, data), "owned-sampler-result"
            )
        self.assertIs(owner.xl_base_patched, original)


if __name__ == "__main__":
    unittest.main()
