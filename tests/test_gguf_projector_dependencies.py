"""Tiny CPU regression for implicit native GGUF vision-projector sources."""

import ast
from copy import deepcopy
import logging
import os
from pathlib import Path
import re
import struct
from types import FunctionType
import unittest
from unittest.mock import Mock, patch

import test_model_cache as cache
from ruinedfooocus.features.models import gguf_projectors as projectors


ROOT = Path(__file__).resolve().parents[1]
NATIVE_GGUF = Path(os.environ.get("RF_TEST_GGUF_SOURCE", ROOT / "repositories/molbal_comfyui_gguf"))


def tiny_gguf(path, arch="qwen2vl", *, text_only=None):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    fields = string("general.architecture") + struct.pack("<I", 8) + string(arch)
    if text_only is not None:
        fields += string("comfy.gguf.qwen3vl.text_only") + struct.pack("<I?", 7, text_only)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1 + (text_only is not None)) + fields)
    return path


def native_projector_lookup():
    """Execute the pinned resolver prefix, stopping before its first weight read."""
    source = NATIVE_GGUF / "loader.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = {
        node.name: deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("strip_quant_suffix", "gguf_mmproj_loader")
    }
    lookup = functions["gguf_mmproj_loader"]
    boundary = next(
        index
        for index, node in enumerate(lookup.body)
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "gguf_sd_loader"
            for child in ast.walk(node)
        )
    )
    lookup.body = lookup.body[:boundary] + [ast.Return(value=ast.Name(id="target", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=list(functions.values()), type_ignores=[]))
    scope = dict(os=os, re=re, logging=logging)
    exec(compile(module, str(source), "exec"), scope)
    return scope["gguf_mmproj_loader"]


class GGUFProjectorDependencyTests(unittest.TestCase):
    file = cache.ModelCacheTests.file
    replace = cache.ModelCacheTests.replace
    base_loader = cache.ModelCacheTests.base_loader

    def setUp(self):
        cache.ModelCacheTests.setUp(self)
        source = patch.object(projectors, "_source", return_value=NATIVE_GGUF / "loader.py")
        source.start()
        self.addCleanup(source.stop)

    def test_production_cache_reloads_when_native_implicit_projector_is_replaced(self):
        checkpoint = self.file("checkpoint.safetensors")
        encoder = self.file("Qwen2.5-VL-Q4_K_M.gguf")
        tiny_gguf(encoder)
        projector = self.file("mmproj-Qwen2.5-VL-f16.gguf")
        self.file("vae.safetensors")
        resolve = native_projector_lookup()
        self.assertEqual(Path(resolve(str(encoder))), projector)
        load, owner, native, _, _ = self.base_loader(checkpoint, bare=True)
        owner.get_clip_and_vae.return_value["clip_names"] = [encoder.name]

        with patch(
            "ruinedfooocus.backends.comfy.text_encoder_loader.load_mixed_clip",
            side_effect=lambda paths, clip_type, **kwargs: (
                self.assertEqual(Path(resolve(paths[0])), projector) or cache.Clip()
            ),
        ) as clip:
            load()
            load()
            self.assertEqual((native.call_count, clip.call_count), (1, 1))
            self.replace(projector)
            load()
        self.assertEqual((native.call_count, clip.call_count), (2, 2))

    def test_native_roles_text_only_and_other_architectures_match_projector_capture(self):
        encoder = self.root / "Qwen-Q4_K_M.GGUF"
        target = self.file("mmproj-Qwen-f16.gguf")
        for arch, text_only, expected in (
            ("qwen2vl", None, True),
            ("qwen35", None, True),
            ("qwen3vl", False, True),
            ("qwen3vl", True, False),
            ("t5", None, False),
        ):
            with self.subTest(arch=arch, text_only=text_only):
                tiny_gguf(encoder, arch, text_only=text_only)
                items = projectors.capture([encoder])
                self.assertEqual(bool(items), expected)
                if expected:
                    self.assertEqual(items[0].projector[0], str(target.resolve()))

    def test_fresh_requests_invalidate_absence_and_removal_but_queued_request_stays_frozen(self):
        encoder = tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        absent = projectors.capture([encoder])
        self.assertIsNone(absent[0].projector)
        self.assertTrue(projectors.unchanged(absent))
        target = self.file("mmproj-Qwen-f16.gguf")
        self.assertFalse(projectors.unchanged(absent))
        self.assertEqual(projectors.for_paths([encoder], absent), absent)
        present = projectors.capture([encoder])
        target.unlink()
        self.assertFalse(projectors.unchanged(present))
        with self.assertRaises((ValueError, FileNotFoundError)):
            projectors.captured({projectors.KEY: present})

    def test_native_qwen3_alias_ambiguity_is_not_hidden(self):
        encoder = tiny_gguf(self.root / "Qwen3-VL-4B-Q4_K_M.gguf", "qwen3vl")
        self.file("mmproj-Qwen3VL-4B-f16.gguf")
        self.file("mmproj-Qwen3VL-4B-f32.gguf")
        with self.assertRaisesRegex(ValueError, "Ambiguous Qwen3-VL"):
            projectors.capture([encoder])

    def test_source_identity_and_typed_capture_fail_closed(self):
        encoder = tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        source = self.root / "loader.py"
        source.write_bytes((NATIVE_GGUF / "loader.py").read_bytes())
        with patch.object(projectors, "_source", return_value=source):
            items = projectors.capture([encoder])
            self.replace(source)
            with self.assertRaisesRegex(ValueError, "changed"):
                projectors.captured({projectors.KEY: items})
        with self.assertRaisesRegex(ValueError, "typed"):
            projectors.captured({projectors.KEY: ["untrusted"]})

    def test_bounded_metadata_rejects_malformed_fields_without_tensor_reader(self):
        encoder = self.file("malformed.gguf")
        with self.assertRaisesRegex(ValueError, "magic"):
            projectors.capture([encoder])
        with patch.object(projectors, "_directory_names", side_effect=AssertionError("no scan")):
            tiny_gguf(encoder, "qwen3vl", text_only=True)
            self.assertEqual(projectors.capture([encoder]), ())

    def test_preparation_freezes_and_verifies_projector_without_acquisition(self):
        from ruinedfooocus.features.generation import image_preparation as assets
        from ruinedfooocus.core.preparation import PreparationRequest

        encoder = tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        target = self.file("mmproj-Qwen-f16.gguf")
        item = assets.ImageArtifact(
            "clip",
            "clip_qwen25",
            PreparationRequest("", encoder),
            tuple(projectors.file_signature(encoder)),
        )
        data = projectors.capture_assets({assets.KEY: assets.ImageAssets((item,), True)}, [item])
        service = Mock()
        service.artifact_lease.side_effect = lambda path, **kwargs: __import__(
            "contextlib"
        ).nullcontext(path)
        prepared = assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        self.assertEqual(prepared[projectors.KEY], data[projectors.KEY])
        service.submit.assert_not_called()
        self.replace(target)
        with self.assertRaisesRegex(ValueError, "changed"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)

    def test_actual_submission_capture_preserves_projector_in_typed_request(self):
        from test_image_preparation import NativeImagePreparationTests
        from ruinedfooocus.features.generation import image_preparation as assets

        fixture = NativeImagePreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        models = fixture.models({}, "QwenImage")
        encoder = fixture.local("clip", "Qwen-Q4_K_M.GGUF")
        tiny_gguf(encoder)
        target = fixture.local("clip", "mmproj-Qwen-f16.gguf")
        codec = fixture.local("vae", "codec.safetensors", b"\x02\0\0\0\0\0\0\0{}")
        request = assets.capture_image_assets(
            {
                "base_model_name": "fixture",
                "image_component_overrides": {
                    "clip_qwen25": encoder.name,
                    "vae_qwen_image": codec.name,
                },
            },
            fixture.manager,
            {},
            models=models,
        )
        self.assertEqual(request[projectors.KEY][0].projector[0], str(target.resolve()))
        self.assertEqual({item.role for item in request[assets.KEY].artifacts}, {"clip", "vae"})

    def test_queued_source_change_is_refused_before_production_cache_reuse(self):
        checkpoint = self.file("checkpoint.safetensors")
        encoder = tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        target = self.file("mmproj-Qwen-f16.gguf")
        self.file("vae.safetensors")
        load, owner, native, _, _ = self.base_loader(checkpoint, bare=True)
        owner.get_clip_and_vae.return_value["clip_names"] = [encoder.name]
        data = {projectors.KEY: projectors.capture([encoder])}
        with patch(
            "ruinedfooocus.backends.comfy.text_encoder_loader.load_mixed_clip",
            return_value=cache.Clip(),
        ) as clip:
            owner.load_base_model(checkpoint.name, unet_only=True, prepared_data=data)
            self.replace(target)
            with self.assertRaisesRegex(ValueError, "changed"):
                owner.load_base_model(checkpoint.name, unet_only=True, prepared_data=data)
        self.assertEqual((native.call_count, clip.call_count), (1, 1))

    def test_native_reader_is_bound_to_capture_and_module_globals_are_unchanged(self):
        encoder = tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        target = self.file("mmproj-Qwen-f16.gguf")
        items = projectors.capture([encoder])
        tree = ast.parse((NATIVE_GGUF / "loader.py").read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("strip_quant_suffix", "gguf_mmproj_loader")
        ]
        read = Mock(return_value=({"marker": "captured"}, {}))
        scope = dict(
            os=os,
            re=re,
            logging=logging,
            gguf_sd_loader=read,
            sd_map_replace=lambda sd, mapping: sd,
            CLIP_VISION_SD_MAP={},
        )
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(projectors._source()), "exec"),
            scope,
        )

        def clip(path):
            return gguf_mmproj_loader(path)  # noqa: F821 -- native function's scoped global

        native = FunctionType(clip.__code__.replace(co_filename=str(projectors._source())), scope)
        bound = projectors.bind(native, encoder, items)
        self.file("mmproj-Qwen-new.gguf")
        self.assertEqual(bound(str(encoder)), {"marker": "captured"})
        self.assertEqual(Path(read.call_args.args[0]), target)
        self.assertIs(native.__globals__["os"], os)
        self.assertIsNot(bound.__globals__["gguf_mmproj_loader"], scope["gguf_mmproj_loader"])
        absent = (projectors.GGUFProjector(items[0].encoder, items[0].source, None, None),)
        read.reset_mock()
        self.assertEqual(projectors.bind(native, encoder, absent)(str(encoder)), {})
        read.assert_not_called()
        wrong_source = FunctionType(clip.__code__, scope)
        with self.assertRaisesRegex(ValueError, "imported native"):
            projectors.bind(wrong_source, encoder, items)

        from test_text_encoder_formats import native_loaders
        from ruinedfooocus.backends.comfy.text_encoder_loader import load_mixed_clip
        import sys

        def mixed(path, **kwargs):
            return gguf_mmproj_loader(path), {}  # noqa: F821 -- scoped native resolver

        with native_loaders() as loaders:
            sys.modules["molbal_comfyui_gguf.loader"].gguf_clip_loader = FunctionType(
                mixed.__code__.replace(co_filename=str(projectors._source())), scope
            )
            load_mixed_clip([str(encoder)], "QWEN_IMAGE", projectors=items)
            self.assertEqual(loaders.patcher.call_args.args[2], [{"marker": "captured"}])
            loaders.progress.complete_file.assert_called_once_with(str(encoder))

    def test_production_loader_holds_encoder_and_projector_once_during_native_read(self):
        from contextlib import contextmanager
        from ruinedfooocus.core.preparation import preparation_service

        checkpoint = self.file("checkpoint.safetensors")
        encoder = tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        target = self.file("mmproj-Qwen-f16.gguf")
        self.file("vae.safetensors")
        load, owner, _, _, _ = self.base_loader(checkpoint, bare=True)
        owner.get_clip_and_vae.return_value["clip_names"] = [encoder.name]
        active = set()

        @contextmanager
        def lease(path, **kwargs):
            path = Path(path).resolve()
            self.assertNotIn(path, active, "no nested read of a file while a writer may wait")
            active.add(path)
            try:
                yield path
            finally:
                active.remove(path)

        def read(paths, clip_type, **kwargs):
            with projectors.read_scope(kwargs["projectors"]):
                self.assertEqual(active, {encoder, target})
            return cache.Clip()

        with (
            patch.object(preparation_service, "artifact_lease", side_effect=lease),
            patch(
                "ruinedfooocus.backends.comfy.text_encoder_loader.load_mixed_clip", side_effect=read
            ),
        ):
            load()
        self.assertIsNone(owner.load_error)
        self.assertFalse(active)

    def test_projector_read_lease_does_not_reenter_the_callers_encoder_lease(self):
        from ruinedfooocus.core.preparation import preparation_service

        encoder = tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        items = projectors.capture([encoder])
        calls = []
        from contextlib import contextmanager

        @contextmanager
        def lease(path, **kwargs):
            calls.append(("enter", path))
            yield path
            calls.append(("exit", path))

        with patch.object(preparation_service, "artifact_lease", side_effect=lease):
            with projectors.read_scope(items):
                self.assertEqual(calls, [])
        target = self.file("mmproj-Qwen-f16.gguf")
        calls.clear()
        with patch.object(preparation_service, "artifact_lease", side_effect=lease):
            with projectors.read_scope(projectors.capture([encoder])):
                self.assertEqual(calls, [("enter", target)])


if __name__ == "__main__":
    unittest.main()
