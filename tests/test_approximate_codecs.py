"""Full approximate pairs reuse native layouts and retain both local identities."""

import ast
from concurrent.futures import CancelledError
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ruinedfooocus.features.models import approximate_codecs as codecs
from test_native_model_coverage import NATIVE


def native_classes():
    source = NATIVE / "comfy/taesd/taesd.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    scope = dict(
        torch=torch,
        nn=torch.nn,
        comfy=SimpleNamespace(ops=SimpleNamespace(disable_weight_init=torch.nn)),
    )
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def native_contracts():
    klass = native_classes()["TAESD"]
    with torch.device("meta"):
        return {
            str(channels): {
                role: {
                    name: list(value.shape)
                    for name, value in getattr(klass(latent_channels=channels), "taesd_" + role)
                    .state_dict()
                    .items()
                }
                for role in ("encoder", "decoder")
            }
            for channels in (4, 16, 64, 128)
        }


def write_component(path, shapes, *, dtype="F32", change=None):
    offset, header = 0, {}
    for name, shape in shapes.items():
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * {"F32": 4, "F16": 2, "BF16": 2}[dtype]
        header[name] = dict(shape=list(shape), dtype=dtype, data_offsets=[offset, offset + size])
        offset += size
    if change:
        change(header)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(len(raw).to_bytes(8, "little"))
        stream.write(raw)
        stream.truncate(8 + len(raw) + offset)
    return path


class ApproximateCodecTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.paths = SimpleNamespace(paths={"path_vae_approx": [str(self.root)]}, get_abspath=Path)

    def pair(self, alias="taesd", suffix=".safetensors"):
        for role in ("encoder", "decoder"):
            write_component(
                self.root / f"{alias}_{role}{suffix}", codecs.expected_shapes(alias, role)
            )
        return codecs.capture_pair(codecs.ALIASES[alias], codecs.PREFIX + alias, self.paths)

    def test_receipt_matches_actual_pinned_native_modules(self):
        self.assertEqual(codecs._contracts(), native_contracts())

    def test_all_six_aliases_need_both_complete_matching_native_sides(self):
        for alias, key in codecs.ALIASES.items():
            with self.subTest(alias=alias):
                pair = self.pair(alias)
                self.assertEqual(pair.alias, alias)
                row = codecs.inventory_row(key, codecs.PREFIX + alias, self.paths)
                self.assertEqual(row["compatibility"], "compatible")
                self.assertEqual(row["bytes"], sum(item.signature[1] for item in pair.files))
                self.assertEqual(len(row["files"]), 2)
                self.assertFalse(row["download_available"])
                self.assertEqual(len(codecs.installed_pairs(key, self.paths)), 1)
        with self.assertRaisesRegex(ValueError, "VAE contract"):
            codecs.capture_pair("vae_mage_flow", "approx:taef2", self.paths)

    def test_decoder_only_preview_is_never_a_full_codec(self):
        write_component(
            self.root / "taesd_decoder.safetensors", codecs.expected_shapes("taesd", "decoder")
        )
        self.assertEqual(codecs.installed_pairs("vae_sd", self.paths), [])
        with self.assertRaisesRegex(ValueError, "both local"):
            codecs.capture_pair("vae_sd", "approx:taesd", self.paths)

    def test_unknown_torch_pairs_are_visible_without_unpickling_or_weight_reads(self):
        for role in ("encoder", "decoder"):
            (self.root / f"taesd_{role}.pt").write_bytes(b"uninspected Torch candidate")
        with patch(
            "torch.load", side_effect=AssertionError("Browsing must not load Torch weights")
        ):
            rows = codecs.installed_pairs("vae_sd", self.paths)
            self.assertEqual(rows[0]["compatibility"], "unknown")
            pair = codecs.capture_pair("vae_sd", "approx:taesd", self.paths)
            self.assertTrue(all(item.header_sha256 is None for item in pair.files))

    def test_header_limits_duplicate_keys_and_recursive_metadata_are_refused(self):
        self.pair()
        path = self.root / "taesd_encoder.safetensors"
        for raw in (
            b'{"x":{},"x":{}}',
            b'{"__metadata__":' + b"[" * 5000 + b"0" + b"]" * 5000 + b"}",
            b"[]",
        ):
            path.write_bytes(len(raw).to_bytes(8, "little") + raw)
            with self.assertRaises(ValueError):
                codecs.capture_pair("vae_sd", "approx:taesd", self.paths)
        self.pair()
        with (
            patch.object(codecs, "MAX_HEADER_BYTES", 8),
            self.assertRaisesRegex(ValueError, "oversized"),
        ):
            codecs._header_contract.cache_clear()
            codecs.capture_pair("vae_sd", "approx:taesd", self.paths)

    def test_pair_signature_changes_and_private_client_capture_is_replaced(self):
        pair = self.pair()
        data = codecs.capture_request(
            {"image_component_overrides": {"vae_sd": "approx:taesd"}, codecs.KEY: "untrusted"},
            self.paths,
        )
        self.assertEqual(data[codecs.KEY], pair)
        stamp = pair.decoder.path.stat()
        os.utime(pair.decoder.path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10_000_000))
        with self.assertRaisesRegex(ValueError, "changed"):
            pair.verify()
        replacement = codecs.capture_pair("vae_sd", "approx:taesd", self.paths)
        self.assertNotEqual(pair.identity, replacement.identity)
        with self.assertRaisesRegex(ValueError, "Capture both"):
            codecs.captured_pair({}, {"vae_sd": "approx:taesd"})

    def test_malformed_pair_ranges_shapes_and_missing_keys_are_rejected(self):
        self.pair()
        for change in (
            lambda header: header.pop(next(iter(header))),
            lambda header: header[next(iter(header))].update(shape=[1]),
            lambda header: header[next(iter(header))].update(data_offsets=[1, 2]),
            lambda header: header[next(iter(header))].update(dtype="I64"),
        ):
            write_component(
                self.root / "taesd_encoder.safetensors",
                codecs.expected_shapes("taesd", "encoder"),
                change=change,
            )
            with self.assertRaises(ValueError):
                codecs.capture_pair("vae_sd", "approx:taesd", self.paths)

    def test_native_flux2_bodies_pack_and_unpack_a_synthetic_cpu_image(self):
        native = native_classes()
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.inference_mode():
                encoded = native["EncoderFlux2"]()(torch.zeros(1, 3, 16, 16))
                self.assertEqual(tuple(encoded.shape), (1, 128, 1, 1))
                decoded = native["DecoderFlux2"]()(torch.zeros(1, 128, 1, 1))
                self.assertEqual(tuple(decoded.shape), (1, 3, 16, 16))
        finally:
            torch.set_num_threads(previous)

    def test_actual_native_alias_readers_vae_encode_decode_and_reload(self):
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="-1",
            HIP_VISIBLE_DEVICES="-1",
            ROCR_VISIBLE_DEVICES="-1",
        )
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "native-check"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def check_native():
    sys.path.insert(0, str(NATIVE))
    sys.argv = [sys.argv[0], "--cpu"]
    from comfy.cli_args import args
    from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management

    def reject(*args, **kwargs):
        raise AssertionError("Accelerator initialization/probes are prohibited")

    for backend in (torch.cuda, torch.xpu):
        for name in (
            "_lazy_init",
            "init",
            "is_available",
            "device_count",
            "current_device",
            "get_device_properties",
            "get_device_name",
            "get_device_capability",
            "mem_get_info",
        ):
            if hasattr(backend, name):
                setattr(backend, name, reject)
    args.cpu = True
    initialize_cpu_model_management()
    import comfy.sd
    import comfy.utils
    from nodes import VAELoader
    from ruinedfooocus.backends.comfy import approximate_codecs as loader
    from ruinedfooocus.core.preparation import preparation_service

    check = unittest.TestCase()
    check.assertEqual(set(VAELoader.image_taes), set(codecs.ALIASES))
    torch.set_num_threads(1)
    held = set()
    actual_lease = preparation_service.artifact_lease

    @contextmanager
    def lease(path, **options):
        with actual_lease(path, **options) as path:
            held.add(Path(path))
            try:
                yield path
            finally:
                held.remove(Path(path))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = SimpleNamespace(paths={"path_vae_approx": [str(root)]}, get_abspath=Path)
        for alias, slot in codecs.ALIASES.items():
            for role in ("encoder", "decoder"):
                write_component(
                    root / f"{alias}_{role}.safetensors",
                    codecs.expected_shapes(alias, role),
                    dtype="F16" if alias == "taesd3" else "BF16" if alias == "taeqi2_1" else "F32",
                )
            pair = codecs.capture_pair(slot, codecs.PREFIX + alias, paths)
            actual_reader = comfy.utils.load_torch_file

            def read(filename, **options):
                check.assertEqual(held, {item.path for item in pair.files})
                check.assertTrue(options["safe_load"])
                return actual_reader(filename, **options)

            with (
                patch.object(preparation_service, "artifact_lease", side_effect=lease),
                patch.object(comfy.utils, "load_torch_file", side_effect=read) as reader,
            ):
                vae = loader.load_paired_vae(pair, device=torch.device("cpu"))
            check.assertEqual(reader.call_count, 2)
            check.assertEqual(held, set())
            factor = 16 if alias in ("taef2", "taeqi2_1") else 8
            channels = 4 if alias == "taeqi2_1" else 3
            check.assertEqual(vae.spacial_compression_encode(), factor)
            check.assertEqual(vae.spacial_compression_decode(), factor)
            check.assertEqual(vae.latent_channels, codecs.CHANNELS[alias])
            check.assertEqual(vae.output_channels, channels)
            if alias in ("taesd", "taesdxl", "taesd3", "taef1"):
                scale, shift = {
                    "taesd": (0.18215, 0),
                    "taesdxl": (0.13025, 0),
                    "taesd3": (1.5305, 0.0609),
                    "taef1": (0.3611, 0.1159),
                }[alias]
                check.assertAlmostEqual(float(vae.first_stage_model.vae_scale), scale, places=6)
                check.assertAlmostEqual(float(vae.first_stage_model.vae_shift), shift, places=6)
            elif alias == "taef2":
                check.assertEqual(float(vae.first_stage_model.vae_scale), 1.0)
                check.assertEqual(float(vae.first_stage_model.vae_shift), 0.0)
            else:
                native = comfy.latent_formats.QwenImage21()
                torch.testing.assert_close(
                    vae.first_stage_model.vae_scale, 1.0 / native.latents_std[0]
                )
                torch.testing.assert_close(vae.first_stage_model.vae_shift, native.latents_mean[0])
            # Execute the actual native wrapper and native zero-weight codec on a
            # tiny synthetic input; no published model or sampling is involved.
            with torch.inference_mode():
                encoded = vae.encode(torch.zeros(1, 16, 16, 3))
                check.assertEqual(
                    tuple(encoded.shape), (1, codecs.CHANNELS[alias], 16 // factor, 16 // factor)
                )
                decoded = vae.decode(encoded)
                check.assertEqual(tuple(decoded.shape), (1, 16, 16, channels))
            factory, arguments = vae.patcher.cached_patcher_init
            check.assertIs(factory, loader.load_pair_patcher)
            check.assertIs(arguments[0], pair)
            # Avoid retaining mmap storages in Windows fixture cleanup.
            del vae, encoded, decoded
            if alias == "taef2":
                reloaded = factory(*arguments, disable_dynamic=True)
                check.assertEqual(reloaded.model.taesd_encoder[-1].out_channels, 32)
                del reloaded
            import gc

            gc.collect()
            if alias == "taeqi2_1":
                stamp = pair.decoder.path.stat()
                os.utime(pair.decoder.path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10000000))
                with (
                    check.assertRaisesRegex(ValueError, "changed"),
                    patch.object(comfy.utils, "load_torch_file") as reader,
                ):
                    factory(*arguments)
                reader.assert_not_called()
        pair = codecs.capture_pair("vae_sd", "approx:taesd", paths)
        for role in ("encoder", "decoder"):
            state = {
                name: torch.zeros(shape)
                for name, shape in codecs.expected_shapes("taesd", role).items()
            }
            torch.save(state, root / f"taesd_{role}.pt")
        pair = codecs.capture_pair("vae_sd", "approx:taesd", paths)
        check.assertTrue(all(item.header_sha256 is None for item in pair.files))
        with patch.object(torch, "load", wraps=torch.load) as reader:
            vae = loader.load_paired_vae(pair, device=torch.device("cpu"))
        check.assertEqual(reader.call_count, 2)
        check.assertTrue(all(call.kwargs["weights_only"] for call in reader.call_args_list))
        del vae, state
        gc.collect()
        state = {
            name: torch.zeros(shape)
            for name, shape in codecs.expected_shapes("taesd", "encoder").items()
        }
        state[next(iter(state))] = torch.zeros(1)
        torch.save(state, root / "taesd_encoder.pt")
        malformed = codecs.capture_pair("vae_sd", "approx:taesd", paths)
        with patch.object(comfy.sd, "VAE") as construct:
            with check.assertRaisesRegex(ValueError, "native alias layout"):
                loader.load_paired_vae(malformed, device=torch.device("cpu"))
            construct.assert_not_called()
        del state
        gc.collect()
        with (
            patch.object(loader, "_cancelled", side_effect=CancelledError),
            patch.object(comfy.utils, "load_torch_file") as reader,
        ):
            with check.assertRaises(CancelledError):
                loader.load_paired_vae(pair)
            reader.assert_not_called()


if __name__ == "__main__":
    if "native-check" in sys.argv:
        check_native()
    else:
        unittest.main()
