"""Positive native TE identities and safe installed-file selection."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import safetensors.torch
import torch

from ruinedfooocus.features.image_processing import wevisone
import test_wevisone as fixtures


class WeVisOneIdentity(unittest.TestCase):
    def setUp(self):
        from ruinedfooocus.features.models import native_text_identity

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.manager = NS(get_model_folders=lambda key: [self.root] if key == "clip" else [])
        self.enterContext(
            patch.object(
                native_text_identity, "_source", return_value=fixtures.NATIVE / "comfy/sd.py"
            )
        )

    def model(self, name, *, width=2048, decoder=True):
        tensors = {
            "model.visual.merger.linear_fc2.weight": torch.zeros(width, 1),
            "model.visual.deepstack_merger_list.0.norm.weight": torch.zeros(1),
        }
        if decoder:
            tensors["mask_decoder.query_builder.content_proj.weight"] = torch.zeros(1, 1)
        path = self.root / name
        safetensors.torch.save_file(tensors, str(path), metadata={"variant": "wevisone"})
        return path

    def test_actual_native_two_and_four_b_identity_admit_renamed_files_only_with_decoder(self):
        two = self.model("renamed.safetensors")
        four = self.model("alternative.sft", width=2560)
        generic = self.model("wevisone-in-name.safetensors", decoder=False)
        self.assertEqual(wevisone.validate_model(two), "WEVISONE_2B")
        self.assertEqual(wevisone.validate_model(four), "WEVISONE_4B")
        with self.assertRaisesRegex(ValueError, "mask decoder"):
            wevisone.validate_model(generic)
        choices = wevisone.model_choices(self.manager)
        self.assertEqual({value for _, value in choices}, {str(two), str(four)})

    def test_recursive_header_is_skipped_by_actual_passive_picker(self):
        good = self.model("good.safetensors")
        nested = b'{"__metadata__":' + b"[" * 20000 + b"0" + b"]" * 20000 + b"}"
        (self.root / "bad.safetensors").write_bytes(len(nested).to_bytes(8, "little") + nested)
        with self.assertRaises(ValueError):
            wevisone.validate_model(self.root / "bad.safetensors")
        self.assertEqual([value for _, value in wevisone.model_choices(self.manager)], [str(good)])

    def test_capture_uses_clip_root_role_and_rejects_outside_model_before_native_loading(self):
        model = self.model("good.safetensors")
        data = dict(
            input_image=fixtures.Image.new("RGBA", (6, 4), (1, 2, 3, 128)),
            controlnet={"type": "wevisone-matte"},
            wevisone_model=str(model),
        )
        signatures = tuple(
            tuple(fixtures.file_signature(fixtures.NATIVE / name)) for name in wevisone.NATIVE_FILES
        )
        with patch.object(wevisone, "native_source_signatures", return_value=signatures):
            captured, artifacts = wevisone.capture_assets(data, self.manager)
            self.assertEqual(artifacts[0].role, "wevisone-clip")
            self.assertEqual(captured[wevisone.KEY].native_type, "WEVISONE_2B")
            with self.assertRaises(ValueError):
                wevisone.capture_assets(
                    data | {"wevisone_model": "../other.safetensors"}, self.manager
                )

    def test_native_loader_default_delegation_and_incompatible_pair_cleanup(self):
        class NativeEncoder:
            pass

        class NativeTokenizer:
            pass

        clip = NS(cond_stage_model=NativeEncoder(), tokenizer=NativeTokenizer(), patcher=object())
        sd = NS(CLIPType=NS(STABLE_DIFFUSION=object()), load_clip=Mock(return_value=clip))
        management = NS(unload_model_and_clones=Mock())
        modules = {
            "comfy": NS(sd=sd),
            "comfy.sd": sd,
            "comfy.model_management": management,
            "comfy.text_encoders.wevisone": NS(
                WeVisOneTEModel=NativeEncoder, WeVisOneTokenizer=NativeTokenizer
            ),
            "folder_paths": NS(get_folder_paths=lambda category: ["owned embeddings"]),
        }
        model = self.model("owned.safetensors")
        with patch.dict(sys.modules, modules):
            result, patchers = wevisone._load_clip(model)
            self.assertIs(result, clip)
            self.assertEqual(patchers, (clip.patcher,))
            self.assertEqual(
                sd.load_clip.call_args.kwargs,
                dict(
                    ckpt_paths=[str(model)],
                    embedding_directory=["owned embeddings"],
                    clip_type=sd.CLIPType.STABLE_DIFFUSION,
                ),
            )
            clip.tokenizer = object()
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                wevisone._load_clip(model)
        management.unload_model_and_clones.assert_called_once_with(clip.patcher)


if __name__ == "__main__":
    unittest.main()
