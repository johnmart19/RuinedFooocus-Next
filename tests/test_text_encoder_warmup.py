"""LoRA-patched text encoders are warmed only when DynamicVRAM would patch them lazily."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ruinedfooocus.backends.comfy import text_encoder_warmup as warm  # noqa: E402


class WarmupTests(unittest.TestCase):
    def run_with(self, clip, *, aimdo=True, loaded=()):
        encode = mock.Mock(return_value=("cond",))
        fake = {
            "comfy": types.ModuleType("comfy"),
            "comfy.memory_management": SimpleNamespace(aimdo_enabled=aimdo),
            "comfy.model_management": SimpleNamespace(
                current_loaded_models=[SimpleNamespace(model=m) for m in loaded]
            ),
            "nodes": SimpleNamespace(CLIPTextEncode=lambda: SimpleNamespace(encode=encode)),
        }
        with mock.patch.dict(sys.modules, fake):
            return warm.warm_patched_text_encoder(clip), encode

    def test_unloaded_patched_encoder_gets_one_empty_encode(self):
        clip = SimpleNamespace(patcher=SimpleNamespace(patches={"k": ["lora"]}))
        warmed, encode = self.run_with(clip)
        self.assertTrue(warmed)
        encode.assert_called_once_with(clip=clip, text="")

    def test_warmup_failure_is_logged_not_raised(self):
        clip = SimpleNamespace(patcher=SimpleNamespace(patches={"k": ["lora"]}))
        fake = {
            "comfy": types.ModuleType("comfy"),
            "comfy.memory_management": SimpleNamespace(aimdo_enabled=True),
            "comfy.model_management": SimpleNamespace(current_loaded_models=[]),
            "nodes": SimpleNamespace(
                CLIPTextEncode=lambda: SimpleNamespace(
                    encode=mock.Mock(side_effect=RuntimeError("needs images"))
                )
            ),
        }
        with mock.patch.dict(sys.modules, fake), self.assertLogs(level="WARNING"):
            self.assertFalse(warm.warm_patched_text_encoder(clip))

    def test_no_warmup_when_loaded_unpatched_or_legacy_loading(self):
        patched = SimpleNamespace(patches={"k": ["lora"]})
        clip = SimpleNamespace(patcher=patched)
        for case in (
            dict(clip=clip, loaded=(patched,)),
            dict(clip=SimpleNamespace(patcher=SimpleNamespace(patches={}))),
            dict(clip=clip, aimdo=False),
        ):
            warmed, encode = self.run_with(**case)
            self.assertFalse(warmed)
            encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
