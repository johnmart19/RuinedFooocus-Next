"""Captured local identities cannot become acquisition requests after removal."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_editing import lora_hooks
from ruinedfooocus.features.models.effective_image_plan import KEY as PLAN_KEY
import test_api_native_inpaint as api_fixture

ROOT = Path(__file__).resolve().parents[1]


def worker_function(name, scope):
    tree = ast.parse((ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "production captured-source worker", "exec"
        ),
        scope,
    )
    return scope[name]


class NativeSourceRetentionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"RF_OFFLINE": "0", "HF_HUB_OFFLINE": "0"}))
        self.api = api_fixture.NativeInpaintApiTests()
        self.api.setUp()
        self.addCleanup(self.api.doCleanups)
        self.root = Path(self.api.selected["path"]).parent
        (self.root / "loras").mkdir()
        self.source = self.root / "loras/registered.safetensors"
        self.source.write_bytes(b"inert owned registered local hook")
        self.shared = sys.modules["ruinedfooocus.app.state"]
        self.manager = self.shared.path_manager
        self.manager.get_model_folders = lambda category: [self.root / category]
        self.manager.DOWNLOADABLE_FILES["registered-hook"] = dict(
            path="path_loras", filename=self.source.name
        )
        self.manager.preparation_request = lambda key, destination=None: PreparationRequest(
            "https://example.invalid/registered-local-hook", destination or self.source
        )
        self.service = Mock()
        self.service.submit.side_effect = FileNotFoundError("inert no-network sentinel")
        self.prepare_assets = assets.prepare_image_assets
        self.optional_prepare = Mock(side_effect=lambda data, **_: data)
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.features.generation.sdcpp_pipeline": NS(
                        prepare_image_request=self.optional_prepare,
                        capture_image_request=lambda data, *_, **__: data,
                    )
                },
            )
        )
        self.worker_prepare = worker_function("prepare_image_data", {})

    def payload(self):
        return dict(
            prompt="literal source request",
            size="512x512",
            lora_hook_mode="keyframes",
            lora_hook_model=self.source.name,
        )

    def api_capture(self):
        response = self.api.client.post("/images/generations", json=self.payload())
        self.assertEqual(response.status_code, 200, response.text)
        data = self.api.prepared[-1]
        item = data[lora_hooks.KEY].artifact
        self.assertIsNotNone(item.signature)
        self.assertEqual(item.request.url, "https://example.invalid/registered-local-hook")
        self.assertFalse(data[assets.KEY].offline)
        return data

    def prepare(self, data, **kwargs):
        return self.prepare_assets(data, service=self.service, **kwargs)

    def test_api_prepare_of_deleted_registered_local_hook_never_submits_its_url(self):
        worker = sys.modules["ruinedfooocus.app.async_worker"]
        captured = []

        def remove_before_prepare(data, **kwargs):
            item = data[lora_hooks.KEY].artifact
            self.assertIsNotNone(item.signature)
            self.assertTrue(item.request.url)
            captured.append(data)
            self.source.unlink()
            return self.worker_prepare(data, **kwargs)

        worker.prepare_image_data = remove_before_prepare
        worker._process = Mock(side_effect=AssertionError("no compute admission"))
        with (
            patch.object(assets, "prepare_image_assets", side_effect=self.prepare),
            self.assertRaises(FileNotFoundError),
        ):
            self.api.client.post("/images/generations", json=self.payload())
        self.assertEqual(len(captured), 1)
        self.service.submit.assert_not_called()
        worker._process.assert_not_called()
        self.optional_prepare.assert_not_called()

    def test_actual_ui_capture_and_worker_prepare_refuse_deleted_signed_catalogue_file(self):
        data = self.api_capture()
        api_plan = data[PLAN_KEY]
        api_hook = data[lora_hooks.KEY].artifact
        scope = dict(
            shared=self.shared,
            settings=self.shared.settings,
            control_tools=NS(get_settings=lambda *_: {}),
        )
        capture = worker_function("capture_image_data", scope)
        stale = dict(data, task_type="process", _sampling_captured=True)
        stale.pop("_api_job", None)
        stale.pop(assets.KEY, None)
        with self.assertRaisesRegex(ValueError, "image request changed after plan capture"):
            capture(stale)
        # A new UI form has public values, not the prior API request's frozen captures.
        form = {key: value for key, value in data.items() if not key.startswith("_")}
        form["task_type"] = "process"
        apply = Mock(side_effect=lambda values, family: dict(values))
        self.shared.performance_settings.apply = apply
        captured = capture(form)
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[1], "SDXL")
        self.assertTrue(captured["_sampling_captured"])
        self.assertIsNot(captured[PLAN_KEY], api_plan)
        self.assertIs(data[PLAN_KEY], api_plan)
        self.assertEqual(captured[lora_hooks.KEY].artifact.request.url, api_hook.request.url)
        self.assertEqual(captured[lora_hooks.KEY].artifact.signature, api_hook.signature)
        self.assertNotIn(PLAN_KEY, form)
        self.source.unlink()
        with (
            patch.object(assets, "prepare_image_assets", side_effect=self.prepare),
            self.assertRaises(FileNotFoundError),
        ):
            self.worker_prepare(captured, check_cancelled=Mock())
        self.service.submit.assert_not_called()
        self.optional_prepare.assert_not_called()

    def test_removed_signed_source_preflight_precedes_other_intended_acquisitions(self):
        data = self.api_capture()
        captured = data[assets.KEY]
        missing_first_use = assets.ImageArtifact(
            "upscaler",
            "first-use.pth",
            PreparationRequest(
                "https://example.invalid/intended-first-use", self.root / "first-use.pth"
            ),
            None,
        )
        data[assets.KEY] = assets.ImageAssets(
            (missing_first_use, *captured.artifacts), offline=False
        )
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare_assets(data, check_cancelled=Mock(), service=self.service)
        self.service.submit.assert_not_called()

    def test_removal_between_preflight_and_transfer_dispatch_cannot_submit_the_url(self):
        data = self.api_capture()
        calls = []

        def check_cancelled():
            calls.append(True)
            if len(calls) == len(data[assets.KEY].artifacts) + 1:
                self.source.unlink()

        with self.assertRaises(FileNotFoundError):
            self.prepare_assets(data, check_cancelled=check_cancelled, service=self.service)
        self.service.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
