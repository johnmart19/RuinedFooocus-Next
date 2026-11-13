"""Named shard admission, real native reader/classifier calls and source receipts."""

from concurrent.futures import CancelledError
import gzip
import hashlib
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
from safetensors.torch import save_file

from ruinedfooocus.features.models import sensenova_bundle as bundles

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
FIXTURE = ROOT / "tests/fixtures/sensenova-u15-headers.json.gz"


def write_bundle(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    weights, total = {}, 0
    for number, name in enumerate(bundles.SHARD_NAMES):
        key, shape = (
            (bundles.VISION, (1024, 1, 1, 1))
            if number == 0
            else ((bundles.QUERY, (4096, 1)) if number == 1 else (f"fixture.weight.{number}", (2,)))
        )
        tensor = torch.zeros(shape, dtype=torch.bfloat16)
        save_file({key: tensor}, root / name, metadata={"format": "pt"})
        total += tensor.numel() * tensor.element_size()
        weights[key] = name
    index = root / bundles.INDEX_NAME
    index.write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weights}), encoding="utf-8"
    )
    return index


class SenseNovaBundleTests(unittest.TestCase):
    def test_incomplete_bundle_error_is_visible_in_actual_component_status(self):
        import ruinedfooocus.app
        from ruinedfooocus.features.models import ui_model_components as ui

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = write_bundle(root / "bundle")
            (index.parent / bundles.SHARD_NAMES[-1]).unlink()
            expected = bundles.bundle_metadata(index)["_rf_bundle_error"]
            name = f"bundle/{bundles.INDEX_NAME}"
            manager = SimpleNamespace(prepare_file=Mock(side_effect=AssertionError("No download")))
            shared = SimpleNamespace(
                models=SimpleNamespace(
                    get_names=lambda category: [name],
                    get_file_from_name=lambda category, selected: index,
                    cache_paths={"checkpoints": root / "cache"},
                ),
                settings=SimpleNamespace(default_settings={}),
                path_manager=manager,
            )
            with (
                patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
                patch("torch.load", side_effect=AssertionError("No weight loading")),
            ):
                result = ui.refresh(name)
            self.assertEqual(len(result), 7)
            self.assertEqual(result[0], [])
            self.assertEqual(result[2], expected)
            self.assertFalse(result[3]["interactive"])
            manager.prepare_file.assert_not_called()

    def test_bounded_capture_preserves_files_and_freezes_all_nine_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_bundle(directory)
            originals = {path.name: path.read_bytes() for path in index.parent.iterdir()}
            with patch(
                "safetensors.safe_open",
                side_effect=AssertionError("Header capture never maps tensors"),
            ):
                captured = bundles.capture_bundle(index)
            self.assertEqual(len(captured.files), 9)
            self.assertEqual(len(captured.tensors), 8)
            self.assertEqual(bundles.bundle_inventory(captured)["rows"], [])
            self.assertEqual(list(bundles.discover_bundles([index.parent])), [index])
            self.assertEqual(
                {path.name: path.read_bytes() for path in index.parent.iterdir()}, originals
            )
            first = captured.identity
            with self.assertRaisesRegex(ValueError, "component overrides"):
                bundles.bundle_inventory(captured, {"vae_sd": "replace.safetensors"})
            shard = captured.shards[-1].path
            save_file(
                {"fixture.weight.7": torch.ones(2, dtype=torch.bfloat16)},
                shard,
                metadata={"format": "pt"},
            )
            with self.assertRaisesRegex(ValueError, "changed"):
                captured.verify()
            self.assertNotEqual(bundles.capture_bundle(index).identity, first)

    def test_index_duplicates_paths_types_counts_missing_keys_and_totals_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_bundle(directory)
            original = index.read_text()
            document = json.loads(original)
            for invalid in (
                {"metadata": {"total_size": True}, "weight_map": document["weight_map"]},
                {"metadata": document["metadata"], "weight_map": {}},
                {
                    "metadata": document["metadata"],
                    "weight_map": {
                        **document["weight_map"],
                        bundles.VISION: "../outside.safetensors",
                    },
                },
                {"metadata": {"total_size": 1}, "weight_map": document["weight_map"]},
                {
                    "metadata": document["metadata"],
                    "weight_map": {
                        **document["weight_map"],
                        "missing.weight": bundles.SHARD_NAMES[0],
                    },
                },
            ):
                index.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    bundles.capture_bundle(index)
            index.write_text(original[:-1] + ',"weight_map":{}}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                bundles.capture_bundle(index)
            index.write_text(original)
            with (
                patch.object(bundles, "MAX_INDEX_BYTES", 8),
                self.assertRaisesRegex(ValueError, "limit"),
            ):
                bundles.capture_bundle(index)
            with (
                patch.object(bundles, "MAX_HEADER_BYTES", 8),
                self.assertRaisesRegex(ValueError, "oversized"),
            ):
                bundles.capture_bundle(index)
            (index.parent / bundles.SHARD_NAMES[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "unavailable"):
                bundles.capture_bundle(index)
            self.assertEqual(list(bundles.discover_bundles([index.parent])), [index])
            self.assertEqual(bundles.bundle_metadata(index)["baseModel"], "Unknown")

    def test_tensor_duplicate_unindexed_overlapping_out_of_range_and_wrong_dtype_shapes(self):
        good = {"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
        for header, size in (
            ({**good, "extra": good["weight"]}, 4),
            (good, 6),
            ({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [1, 5]}}, 4),
            ({"weight": {"dtype": "BF16", "shape": [True], "data_offsets": [0, 2]}}, 2),
            ({"weight": {"dtype": [], "shape": [2], "data_offsets": [0, 4]}}, 4),
            ({"weight": {"dtype": "I64", "shape": [2], "data_offsets": [0, 16]}}, 16),
            ({**good, "__metadata__": {"format": 1}}, 4),
        ):
            with self.subTest(header=header), self.assertRaises(ValueError):
                bundles.validate_shard(header, size, bundles.SHARD_NAMES[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bundles._json(b'{"weight":{},"weight":{}}')

    def test_capture_cancellation_and_nested_read_scopes_keep_one_lease_per_source(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_bundle(directory)
            with self.assertRaises(CancelledError):
                bundles.capture_bundle(index, check_cancelled=Mock(side_effect=CancelledError))
            captured = bundles.capture_bundle(index)
            from ruinedfooocus.core.preparation import preparation_service

            lease = preparation_service.artifact_lease
            with patch.object(preparation_service, "artifact_lease", wraps=lease) as read:
                with bundles.read_scope(captured), bundles.read_scope(captured):
                    self.assertEqual(read.call_count, 9)
                with bundles.read_scope(captured):
                    self.assertEqual(read.call_count, 18)

    def test_cross_shard_duplicates_and_files_outside_the_captured_folder_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = write_bundle(root / "bundle")
            with self.assertRaisesRegex(ValueError, "one local folder"):
                bundles._capture(index, root.resolve())
            save_file(
                {bundles.VISION: torch.zeros((1024, 1, 1, 1), dtype=torch.bfloat16)},
                index.parent / bundles.SHARD_NAMES[2],
                metadata={"format": "pt"},
            )
            with self.assertRaisesRegex(ValueError, "duplicate or disagree"):
                bundles.capture_bundle(index)

    def test_inventory_inspection_reuses_one_capture_and_rechecks_repaired_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_bundle(directory)
            (index.parent / bundles.SHARD_NAMES[-1]).unlink()
            with patch.object(bundles, "capture_bundle", wraps=bundles.capture_bundle) as capture:
                self.assertEqual(bundles.bundle_metadata(index)["baseModel"], "Unknown")
                self.assertEqual(bundles.bundle_metadata(index)["baseModel"], "Unknown")
                self.assertEqual(capture.call_count, 1)
                write_bundle(directory)
                self.assertEqual(bundles.bundle_metadata(index)["baseModel"], "SenseNovaU15")
                self.assertEqual(capture.call_count, 2)

    def test_pinned_native_files_classifier_loader_reload_and_source_header_contract(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys; runpy.run_path(sys.argv[1])['check_native']()",
                str(Path(__file__).resolve()),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def check_native():
    sys.path.insert(0, str(COMFY))
    sys.argv = [sys.argv[0], "--cpu"]
    from comfy.cli_args import args
    from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management

    def reject(*_, **__):
        raise AssertionError("Unexpected accelerator initialization")

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
    import comfy.model_detection
    import comfy.sd
    import comfy.utils
    from ruinedfooocus.backends.comfy.sensenova_bundle import load_bundle_model
    from ruinedfooocus.backends.comfy.sensenova_clip import load_sensenova_clip

    with patch.dict(
        sys.modules,
        {
            "ruinedfooocus.backends.comfy.gguf_loader": SimpleNamespace(
                load_diffusion_model=Mock(side_effect=AssertionError("Unexpected GGUF dispatch"))
            ),
        },
    ):
        from ruinedfooocus.features.generation.pipeline_utils import load_diffusion_model

    check = unittest.TestCase()
    fixture = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    check.assertEqual(fixture["revision"], "a2462cc1b549bace2c0896143bbbcbb3fa217c76")
    check.assertEqual(
        hashlib.sha256(fixture["index_raw"].encode()).hexdigest(), fixture["index_sha256"]
    )
    index = bundles._json(fixture["index_raw"])
    total, tensors = 0, {}
    for shard in fixture["shards"]:
        raw = shard["raw_header"].encode()
        check.assertEqual(hashlib.sha256(raw).hexdigest(), shard["header_sha256"])
        name = shard["file"].removesuffix(".header.json")
        payload_bytes = shard["published_bytes"] - 8 - len(raw)
        rows, _ = bundles.validate_shard(bundles._json(raw), payload_bytes, name)
        for row in rows:
            check.assertNotIn(row.name, tensors)
            check.assertEqual(index["weight_map"][row.name], name)
            tensors[row.name] = torch.empty(row.shape, dtype=torch.bfloat16, device="meta")
        total += payload_bytes
    check.assertEqual(len(tensors), 1116)
    check.assertEqual(total, index["metadata"]["total_size"])
    config = comfy.model_detection.model_config_from_unet(tensors, "")
    check.assertEqual(type(config).__name__, "SenseNovaU15")
    del tensors

    with tempfile.TemporaryDirectory() as directory:
        captured = bundles.capture_bundle(write_bundle(directory))
        original = {item.path: item.path.read_bytes() for item in captured.files}

        def native(state, **options):
            check.assertEqual(len(state), 8)
            native_config = comfy.model_detection.model_config_from_unet(state, "")
            check.assertEqual(type(native_config).__name__, "SenseNovaU15")
            return SimpleNamespace(model=SimpleNamespace(model_config=native_config))

        with (
            patch.object(comfy.sd, "load_diffusion_model_state_dict", side_effect=native) as loader,
            patch.object(
                comfy.utils, "load_torch_file", wraps=comfy.utils.load_torch_file
            ) as files,
            patch.object(
                comfy.utils, "save_torch_file", side_effect=AssertionError("No merged weights")
            ),
        ):
            model = load_diffusion_model(str(captured.index.path), sensenova_bundle=captured)
            check.assertEqual(files.call_count, 8)
            check.assertEqual(loader.call_count, 1)
            reloader, arguments = model.cached_patcher_init
            reloader(*arguments, disable_dynamic=True)
            check.assertEqual(files.call_count, 16)
            sentinel_clip = SimpleNamespace(patcher=SimpleNamespace())
            with patch.object(comfy.sd, "CLIP", return_value=sentinel_clip):
                clip = load_sensenova_clip(model.model.model_config, captured)
                reload_clip, clip_arguments = clip.patcher.cached_patcher_init
                check.assertIs(reload_clip(*clip_arguments), sentinel_clip.patcher)
            check.assertEqual({path: path.read_bytes() for path in original}, original)
            changed = captured.shards[-1].path
            stamp = changed.stat()
            os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10_000_000))
            with check.assertRaisesRegex(ValueError, "changed"):
                reloader(*arguments)
            with check.assertRaisesRegex(ValueError, "changed"):
                reload_clip(*clip_arguments)
            check.assertEqual(files.call_count, 16)
