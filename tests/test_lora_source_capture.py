"""Freeze static LoRA files and exercise the existing loader with CPU fixtures."""

import ast
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gguf
import numpy as np

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.generation.image_preparation import (
    KEY as ASSETS_KEY,
    ImageArtifact,
    ImageAssets,
)
from ruinedfooocus.features.models import lora_sources as sources
from ruinedfooocus.features.models.model_identity import file_signature

ROOT = Path(__file__).resolve().parents[1]


class LoraSourceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.first, self.second = self.root / "first", self.root / "second"
        self.first.mkdir()
        self.second.mkdir()
        self.models = SimpleNamespace(
            model_dirs={"loras": [self.first, self.second]},
            get_names=Mock(return_value=["style.safetensors", "other.safetensors"]),
            get_file=Mock(side_effect=self.find),
            get_file_from_hash=Mock(return_value=None),
            get_model_path=Mock(side_effect=AssertionError("No late resolver or acquisition")),
        )

    def find(self, category, name):
        self.assertEqual(category, "loras")
        return next(
            (root / name for root in self.models.model_dirs[category] if (root / name).is_file()),
            None,
        )

    def file(self, name, root=None):
        path = (root or self.first) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original local weights")
        return path

    def gguf(self, name, root=None):
        path = (root or self.first) / name
        writer = gguf.GGUFWriter(path, "sdxl")
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_tensor("linear.weight.lora_a", np.zeros((1, 4), dtype=np.float32))
        writer.add_tensor("linear.weight.lora_b", np.zeros((3, 1), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path

    def packet(self, rows):
        return sources.capture_request({}, self.models, loras=rows)[sources.KEY]

    def loader(self):
        source = ROOT / "ruinedfooocus/features/generation/image_pipeline.py"
        node = next(
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "load_loras"
        )
        reader = Mock(side_effect=lambda path, **_: Path(path).read_bytes())
        native = Mock(
            side_effect=lambda unet, clip, weights, model_weight, clip_weight: (
                SimpleNamespace(
                    patches={
                        "weight": [
                            *getattr(unet, "patches", {}).get("weight", []),
                            weights,
                            model_weight,
                        ]
                    }
                ),
                SimpleNamespace(patcher=SimpleNamespace(patches={})),
            )
        )
        scope = dict(
            shared=SimpleNamespace(models=self.models),
            file_signature=file_signature,
            comfy=SimpleNamespace(
                utils=SimpleNamespace(load_torch_file=reader),
                sd=SimpleNamespace(load_lora_for_models=native),
            ),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        base = SimpleNamespace(unet=None, clip=None, vae=None, clip_vision=None)
        owner = SimpleNamespace(
            xl_base=base, xl_base_patched_hash=None, StableDiffusionModel=SimpleNamespace
        )
        return scope["load_loras"], owner, reader, native

    def test_original_prompt_cards_and_explicit_weights_share_existing_merge_rules(self):
        style = self.file("style.safetensors")
        other = self.file("other.safetensors")
        data = dict(
            loras=[["thumbnail", "0.4 - style.safetensors"]],
            prompt="literal <lora:style:-.75>",
            negative="<lora:other:0.25> <lora:malformed:1..2>",
        )
        captured = sources.capture_request(data, self.models)
        self.assertEqual(captured["prompt"], data["prompt"])
        self.assertEqual(captured["loras"], data["loras"])
        self.assertEqual(
            [(item.path, item.weight) for item in captured[sources.KEY]],
            [(other, 0.25), (style, -0.75)],
        )
        self.models.get_names.assert_called_once_with("loras")
        self.models.get_model_path.assert_not_called()

    def test_loader_uses_captured_file_across_roots_drift_and_keeps_cache_identity(self):
        selected = self.file("style.safetensors")
        self.file("style.safetensors", self.second).write_bytes(b"different source")
        request = [dict(name=selected.name, weight=-0.25)]
        packet = self.packet(request)
        self.models.model_dirs["loras"] = [self.second]
        run, owner, reader, native = self.loader()
        run(owner, request, captured_loras=packet)
        reader.assert_called_once_with(str(selected), safe_load=True)
        self.assertEqual(native.call_args.args, (None, None, selected.read_bytes(), -0.25, -0.25))
        first = owner.xl_base_patched
        run(owner, request, captured_loras=packet)
        self.assertIs(owner.xl_base_patched, first)
        self.assertEqual(reader.call_count, 1)
        self.models.get_model_path.assert_not_called()
        selected.write_bytes(b"changed after capture")
        with self.assertRaisesRegex(ValueError, "captured image LoRA changed"):
            run(owner, request, captured_loras=packet)
        self.assertEqual(reader.call_count, 1)

    def test_new_capture_supports_add_replace_remove_and_never_patches_previous_result(self):
        style = self.file("style.safetensors")
        other = self.file("other.safetensors")
        run, owner, _, native = self.loader()
        for name, weight in ((style.name, 0.5), (other.name, -0.4), (style.name, 1.0)):
            request = [dict(name=name, weight=weight)]
            run(owner, request, captured_loras=self.packet(request))
        self.assertEqual([call.args[:2] for call in native.call_args_list], [(None, None)] * 3)
        run(owner, [], captured_loras=self.packet([]))
        self.assertIs(owner.xl_base_patched, owner.xl_base)

    def test_runtime_add_remove_or_weight_change_is_refused_before_native_loading(self):
        self.file("style.safetensors")
        original = [dict(name="style.safetensors", weight=0.5)]
        packet = self.packet(original)
        run, owner, reader, _ = self.loader()
        for request in (
            [],
            [dict(name="style.safetensors", weight=0.8)],
            original + [dict(name="late.safetensors", weight=1)],
        ):
            with (
                self.subTest(request=request),
                self.assertRaisesRegex(ValueError, "selection changed"),
            ):
                run(owner, request, captured_loras=packet)
        reader.assert_not_called()

    def test_original_prompt_batch_keeps_per_prompt_weights_and_file_identity(self):
        style = self.file("style.safetensors")
        other = self.file("other.safetensors")
        data = dict(
            loras=[["thumbnail", "0.4 - style.safetensors"]],
            prompt=[
                "first <lora:style:0.7>",
                "second <lora:style:-0.2> <lora:other:1>",
                "third <lora:style:0>",
            ],
            negative="",
        )
        captured = sources.capture_request(data, self.models)
        packet = captured[sources.KEY]
        self.assertEqual(
            [(item.name, item.weight) for item in packet],
            [(other.name, 1), (style.name, 0.7), (style.name, -0.2)],
        )
        self.assertEqual(self.models.get_file.call_count, 2)
        self.models.model_dirs["loras"] = [self.second]
        run, owner, reader, native = self.loader()
        for prompt, request in zip(
            data["prompt"],
            (
                [dict(name=style.name, weight=0.7)],
                [dict(name=style.name, weight=-0.2), dict(name=other.name, weight=1)],
                [],
            ),
        ):
            child = sources.sources_for_prompt(prompt, packet, captured[sources.PROMPTS_KEY])
            run(owner, request, captured_loras=child)
        self.assertEqual([call.args[3] for call in native.call_args_list], [0.7, 1, -0.2])
        self.assertIs(owner.xl_base_patched, owner.xl_base)
        self.assertEqual({call.args[0] for call in reader.call_args_list}, {str(style), str(other)})
        first = sources.sources_for_prompt(data["prompt"][0], packet, captured[sources.PROMPTS_KEY])
        with self.assertRaisesRegex(ValueError, "selection changed"):
            run(owner, [dict(name=style.name, weight=-0.2)], captured_loras=first)
        with self.assertRaisesRegex(ValueError, "original image prompt was not captured"):
            sources.sources_for_prompt("different child", packet, captured[sources.PROMPTS_KEY])
        metadata = captured[sources.PROMPTS_KEY][0].metadata()
        metadata["loras"].clear()
        self.assertEqual(first[0].weight, 0.7)

    def test_changed_during_native_read_fails_and_does_not_publish_cache(self):
        source = self.file("style.safetensors")
        request = [dict(name=source.name, weight=1)]
        packet = self.packet(request)
        run, owner, reader, native = self.loader()

        def replace_source(*_, **__):
            source.write_bytes(b"replacement during native read")
            return "old weights"

        reader.side_effect = replace_source
        with self.assertRaisesRegex(RuntimeError, "captured image LoRA changed"):
            run(owner, request, captured_loras=packet)
        native.assert_not_called()
        self.assertIsNone(owner.xl_base_patched_hash)

    def test_public_paths_untyped_packets_and_outside_hash_results_are_refused(self):
        source = self.file("style.safetensors")
        for name in (
            str(source),
            "../private.safetensors",
            r"C:\private.safetensors",
            "local:stream.pt",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "relative filename"):
                self.packet([dict(name=name, weight=1)])
        with self.assertRaisesRegex(ValueError, "typed image LoRA"):
            sources.captured({sources.KEY: ({"path": str(source)},)})
        outside = self.file("private.safetensors", self.root)
        self.models.get_file_from_hash.return_value = outside
        with self.assertRaisesRegex(ValueError, "outside its configured"):
            self.packet([dict(name="missing.safetensors", weight=1, hash="a" * 64)])

    def pending(self, name="trusted.gguf"):
        request = PreparationRequest("https://catalogue.invalid/trusted", self.first / name)
        manager = SimpleNamespace(
            DOWNLOADABLE_FILES={"trusted": dict(path="path_loras", filename=name)},
            preparation_request=Mock(return_value=request),
        )
        data = sources.capture_request(
            {}, self.models, loras=[dict(name=name, weight=-0.5)], manager=manager
        )
        artifact = ImageArtifact("loras", "lora:" + name, request, None)
        data[ASSETS_KEY] = ImageAssets((artifact,), False)
        return data, artifact

    def test_missing_trusted_gguf_binds_only_typed_matching_prepared_file(self):
        original, artifact = self.pending()
        packet = original[sources.KEY]
        self.assertIsNone(packet[0].signature)
        with self.assertRaisesRegex(ValueError, "Prepare the captured"):
            sources.for_loras([dict(name="trusted.gguf", weight=-0.5)], packet)
        path = self.gguf("trusted.gguf")
        after = replace(artifact, signature=tuple(file_signature(path)))
        prepared = dict(original, **{ASSETS_KEY: ImageAssets((after,), False)})
        bound = sources.bind_prepared(original, prepared)
        self.assertEqual(bound[0].signature, tuple(file_signature(path)))
        self.assertEqual(bound[0].request, artifact.request)
        self.models.model_dirs["loras"] = [self.second]
        run, owner, _, _ = self.loader()
        native_result = (
            SimpleNamespace(patches={"native": [object()]}),
            SimpleNamespace(patcher=SimpleNamespace(patches={})),
        )
        with patch(
            "ruinedfooocus.backends.comfy.gguf_lora.apply_gguf_lora", return_value=native_result
        ) as apply:
            run(owner, [dict(name="trusted.gguf", weight=-0.5)], captured_loras=bound)
        apply.assert_called_once_with(path, file_signature(path), None, None, -0.5)
        self.models.get_model_path.assert_not_called()

    def test_prepared_wrong_request_identity_or_forged_artifact_is_refused(self):
        original, artifact = self.pending("trusted.safetensors")
        path = self.file("trusted.safetensors")
        after = replace(artifact, signature=tuple(file_signature(path)))
        for changed in (
            replace(after, request=replace(after.request, sha256="b" * 64)),
            replace(after, name="lora:other.safetensors"),
            replace(after, signature=(str(path), 1, 1, 1)),
            SimpleNamespace(**vars(after)),
        ):
            prepared = dict(original, **{ASSETS_KEY: ImageAssets((changed,), False)})
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                sources.bind_prepared(original, prepared)

    def test_missing_custom_and_zero_weight_are_not_acquired(self):
        with self.assertRaisesRegex(FileNotFoundError, "installed or catalogue"):
            self.packet([dict(name="missing.safetensors", weight=1)])
        self.assertEqual(self.packet([dict(name="missing.safetensors", weight=0)]), ())
        self.models.get_model_path.assert_not_called()

    def test_missing_catalogue_filename_does_not_choose_between_different_sources(self):
        request = PreparationRequest("https://catalogue.invalid/first", self.first / "shared.pt")
        manager = SimpleNamespace(
            DOWNLOADABLE_FILES={
                key: dict(path="path_loras", filename="shared.pt") for key in ("first", "second")
            },
            preparation_request=lambda key: replace(
                request, url="https://catalogue.invalid/" + key
            ),
        )
        with self.assertRaisesRegex(ValueError, "several catalogue sources"):
            sources.capture_request(
                {}, self.models, loras=[dict(name="shared.pt", weight=1)], manager=manager
            )


if __name__ == "__main__":
    unittest.main()
