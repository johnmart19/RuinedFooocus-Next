from ruinedfooocus.core.paths import asset_path, resource_path

"""Model file handling: damaged files, header caching, publishing and load errors."""
import ast
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp").mkdir(exist_ok=True)

import ruinedfooocus.features.models.checkpoint_tools as checkpoint_tools  # noqa: E402
from ruinedfooocus.features.models.model_identity import cached_safetensors_header  # noqa: E402
from ruinedfooocus.core.task_errors import describe_error  # noqa: E402


def write_header(path, header, payload=b""):
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)


class DamagedFileTests(unittest.TestCase):
    def test_safetensors_and_archive_damage_gets_an_actionable_message(self):
        for message in (
            "Error while deserializing header: MetadataIncompleteBuffer",
            "Error while deserializing header: incomplete metadata, file not fully covered",
            "Error while deserializing header: header too large",
            "PytorchStreamReader failed reading zip archive: failed finding central directory",
        ):
            with self.subTest(message=message):
                text = describe_error(RuntimeError(message))
                self.assertIn("incomplete or damaged", text)
                self.assertIn("Checkpoint tools", text)
        self.assertNotIn("damaged", describe_error(RuntimeError("something else")))

    def test_unrecognized_checkpoints_explain_where_other_model_files_belong(self):
        text = describe_error(
            RuntimeError("ERROR: Could not detect model type of: x.safetensors\nhint")
        )
        self.assertIn("not a supported image checkpoint", text)
        self.assertIn("own model folders", text)
        self.assertNotIn("hint", text)


class HeaderCacheTests(unittest.TestCase):
    def test_unchanged_files_reuse_the_header_and_replaced_files_are_read_again(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            path = Path(directory) / "model.safetensors"
            write_header(path, {"a": {"shape": [1]}})
            first = cached_safetensors_header(path)
            with mock.patch(
                "ruinedfooocus.features.models.model_identity.read_safetensors_header"
            ) as reader:
                self.assertIs(cached_safetensors_header(path), first)
                reader.assert_not_called()
            time.sleep(0.01)
            write_header(path, {"a": {"shape": [1]}, "b": {"shape": [2, 2]}})
            self.assertIn("b", cached_safetensors_header(path))

    def test_detection_caches_known_families_but_not_unknown_results(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/models/model_handler.py").read_text(encoding="utf-8")
        )
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Models")
        node = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "detect_checkpoint_base"
        )
        node.decorator_list = []
        detector = mock.Mock(side_effect=[None, "SDXL 1.0", "Flux.1 D"])
        Models = type(
            "Models", (), {"_base_cache": {}, "_detect_checkpoint_base": staticmethod(detector)}
        )
        from ruinedfooocus.features.models.model_identity import file_signature

        scope = {
            "asset_path": asset_path,
            "resource_path": resource_path,
            "Path": Path,
            "Models": Models,
            "file_signature": file_signature,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "model_handler", "exec"), scope)
        detect = scope["detect_checkpoint_base"]
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            path = Path(directory) / "model.safetensors"
            write_header(path, {})
            self.assertIsNone(detect(path))  # unknown: not cached
            self.assertEqual(detect(path), "SDXL 1.0")
            self.assertEqual(detect(path), "SDXL 1.0")  # cached
            self.assertEqual(detector.call_count, 2)
            self.assertIsNone(detect(Path(directory) / "missing.safetensors"))


class ReportTests(unittest.TestCase):
    def test_sizes_use_readable_units_for_small_and_large_files(self):
        self.assertEqual(checkpoint_tools.format_size(900), "900 B")
        self.assertEqual(checkpoint_tools.format_size(131072), "128.00 KiB")
        self.assertEqual(checkpoint_tools.format_size(6_939_000_000), "6.46 GiB")


class PublishTests(unittest.TestCase):
    def test_fp16_conversion_publishes_valid_copies_without_touching_original(self):
        import numpy as np
        from safetensors.numpy import load_file, save_file

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source = Path(directory) / "tiny.safetensors"
            values = np.linspace(-10, 10, 4096, dtype=np.float32)
            save_file({"weight": values}, source)
            original = source.read_bytes()
            with mock.patch.object(checkpoint_tools, "os", self.no_link_posix()):
                result = checkpoint_tools.create_fp16_copy(source, Path(directory) / "cache")
            target = Path(result["path"])
            np.testing.assert_array_equal(load_file(target)["weight"], values.astype(np.float16))
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(
                (Path(directory) / "cache" / target.with_suffix(".json").name).is_file()
            )
            with self.assertRaises(FileExistsError):
                checkpoint_tools.create_fp16_copy(source, Path(directory) / "cache")
            self.assertEqual(source.read_bytes(), original)

    def no_link_posix(self):
        platform = mock.Mock(wraps=os)
        platform.name = "posix"
        platform.link.side_effect = OSError("no hard links")
        platform.rename = os.replace  # POSIX rename semantics, even on a Windows test host.
        return platform

    def test_no_link_fallback_cannot_overwrite_a_target_after_a_stale_existence_check(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.write_bytes(b"converted")
            target.write_bytes(b"other writer")
            exists = Path.exists
            with (
                mock.patch.object(checkpoint_tools, "os", self.no_link_posix()),
                mock.patch.object(
                    Path,
                    "exists",
                    autospec=True,
                    side_effect=lambda path: False if path == target else exists(path),
                ),
            ):
                with self.assertRaises(FileExistsError):
                    checkpoint_tools._publish(source, target)
            self.assertEqual(target.read_bytes(), b"other writer")
            self.assertEqual(source.read_bytes(), b"converted")

    def test_no_link_copy_failure_removes_only_its_partial_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.write_bytes(b"converted")
            with (
                mock.patch.object(checkpoint_tools, "os", self.no_link_posix()),
                mock.patch.object(
                    checkpoint_tools.shutil, "copyfileobj", side_effect=OSError("disk full")
                ),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    checkpoint_tools._publish(source, target)
            self.assertFalse(target.exists())
            self.assertEqual(source.read_bytes(), b"converted")
            with mock.patch.object(checkpoint_tools, "os", self.no_link_posix()):
                checkpoint_tools._publish(source, target)
            self.assertEqual(target.read_bytes(), b"converted")
            self.assertFalse(source.exists())

    def test_drives_without_hard_links_still_publish_without_overwriting(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.write_text("new")
            with mock.patch.object(os, "link", side_effect=OSError("hard links unsupported")):
                checkpoint_tools._publish(source, target)
                self.assertEqual(target.read_text(), "new")
                self.assertFalse(source.exists())
                source.write_text("newer")
                with self.assertRaises(FileExistsError):
                    checkpoint_tools._publish(source, target)
            self.assertEqual(target.read_text(), "new")


class LoadErrorTests(unittest.TestCase):
    def test_a_failed_load_reports_its_reason_instead_of_a_generic_error(self):
        source = (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        guard = ast.Module(
            body=[
                ast.FunctionDef(
                    name="guard",
                    args=process.args,
                    body=process.body[:1] + [ast.Return(None)],
                    decorator_list=[],
                    returns=None,
                    type_params=[],
                )
            ],
            type_ignores=[],
        )
        scope = {
            "asset_path": asset_path,
            "resource_path": resource_path,
        }
        exec(compile(ast.fix_missing_locations(guard), "image_pipeline", "exec"), scope)
        owner = type(
            "Owner",
            (),
            {
                "xl_base_patched": None,
                "load_error": "x.safetensors needs the VAE file a.safetensors",
            },
        )()
        with self.assertRaisesRegex(ValueError, "needs the VAE file"):
            scope["guard"](owner, {})
        owner.load_error = None
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            scope["guard"](owner, {})
        # Every load failure path records a reason for the user.
        load = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_base_model"
        )
        text = ast.get_source_segment(source, load)
        for reason in (
            "Checkpoint not found",
            "text encoder file(s) that are not installed",
            "needs the VAE file",
            "not recognized",
            "not supported as an image model",
        ):
            self.assertIn(reason, text)


if __name__ == "__main__":
    unittest.main()
