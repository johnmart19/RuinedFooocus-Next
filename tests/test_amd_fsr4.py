"""AMD FSR 2, 3 and 4 adapter: runtime discovery, helper protocol and the Transform caller.

The native helper and AMD's DLLs are replaced by a stand-in process boundary, so
these checks need no GPU. They do not show that the upscalers themselves run or looks good.
"""

from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from ruinedfooocus.core.bounded_subprocess import BoundedProcessError
from ruinedfooocus.features.image_processing import amd_fsr, amd_fsr4
from ruinedfooocus.features.image_processing import image_geometry_tools as transform


class AmdFsr4Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.enterContext(patch.dict("os.environ", {"RF_AMD_FSR_DIR": str(self.runtime)}))
        self.enterContext(patch.object(amd_fsr4.os, "name", "nt"))
        self.calls = []

    def install(self):
        for name in amd_fsr4.RUNTIME_FILES:
            (self.runtime / name).write_bytes(b"stand-in")

    def helper(self, command, **options):
        """Stand-in for fsr_still.exe: nearest-neighbour enlargement, same file protocol."""
        self.calls.append((command, options))
        source, target = Path(command[1]), Path(command[2])
        width, height = int(command[3]), int(command[4])
        data = source.read_bytes()
        source_width, source_height = struct.unpack("<II", data[4:12])
        image = Image.frombytes("RGBA", (source_width, source_height), data[12:])
        result = image.resize((width, height), Image.NEAREST)
        target.write_bytes(amd_fsr4.MAGIC + struct.pack("<II", width, height) + result.tobytes())
        return b"", b""

    def test_runtime_is_reported_missing_by_name_and_only_windows_is_supported(self):
        self.assertEqual(amd_fsr4.runtime_directory(), self.runtime)
        reason = amd_fsr4.unavailable_reason()
        self.assertIn("tools/native/amd_fsr/build.py", reason)
        for name in amd_fsr4.RUNTIME_FILES:
            self.assertIn(name, reason)
        (self.runtime / amd_fsr4.HELPER).write_bytes(b"x")
        self.assertNotIn(amd_fsr4.HELPER + ",", amd_fsr4.unavailable_reason())
        self.install()
        self.assertIsNone(amd_fsr4.unavailable_reason())
        with patch.object(amd_fsr4.os, "name", "posix"):
            self.assertIn("Windows", amd_fsr4.unavailable_reason())
        with self.assertRaisesRegex(ValueError, "runtime"):
            (self.runtime / amd_fsr4.HELPER).unlink()
            amd_fsr4.upscale(torch.zeros(1, 2, 2, 3), 4, 4, work=self.root / "work")

    def test_passes_are_at_most_two_times_and_reductions_are_refused(self):
        self.assertEqual(amd_fsr4.pass_sizes((100, 50), (200, 100)), [(200, 100)])
        self.assertEqual(amd_fsr4.pass_sizes((100, 50), (400, 200)), [(200, 100), (400, 200)])
        previous = (100, 100)
        for step in amd_fsr4.pass_sizes((100, 100), (300, 250)):
            self.assertLessEqual(step[0] / previous[0], 2.0 + 1e-6)
            self.assertLessEqual(step[1] / previous[1], 2.0 + 1e-6)
            previous = step
        self.assertEqual(previous, (300, 250))
        for target in ((99, 200), (200, 49)):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "only enlarges"):
                amd_fsr4.pass_sizes((100, 50), target)

    def test_colour_goes_through_the_helper_and_alpha_stays_separate(self):
        self.install()
        pixels = torch.zeros(1, 3, 2, 4)
        pixels[..., 0], pixels[..., 1], pixels[..., 3] = 1.0, 0.5, 0.25
        work = self.root / "work"
        with patch("ruinedfooocus.core.bounded_subprocess.run_bounded", self.helper):
            result = amd_fsr4.upscale(pixels, 8, 12, sharpness=0.5, work=work)
        self.assertEqual(tuple(result.shape), (1, 12, 8, 4))
        self.assertTrue(torch.allclose(result[..., 0], torch.ones(1, 12, 8)))
        self.assertTrue(torch.allclose(result[..., 1], torch.full((1, 12, 8), 128 / 255)))
        self.assertTrue(torch.allclose(result[..., 3], torch.full((1, 12, 8), 0.25), atol=1e-5))
        # 2x3 -> 8x12 is a factor of four: two passes, each asking for the FSR 4 provider.
        self.assertEqual(len(self.calls), 2)
        for command, options in self.calls:
            self.assertEqual(Path(command[0]), self.runtime / amd_fsr4.HELPER)
            self.assertEqual(command[5:], ["0", "1", "4.", "0.5"])
            self.assertEqual(options["timeout"], amd_fsr4.TIMEOUT_SECONDS)
        self.assertEqual(list(work.iterdir()), [])
        gray = torch.full((1, 2, 2, 1), 0.2)
        with patch("ruinedfooocus.core.bounded_subprocess.run_bounded", self.helper):
            self.assertEqual(tuple(amd_fsr4.upscale(gray, 4, 4, work=work).shape), (1, 4, 4, 1))
        for bad in (torch.zeros(1, 2, 2, 2), torch.zeros(2, 2, 2, 3)):
            with self.assertRaises(ValueError):
                amd_fsr4.upscale(bad, 4, 4, work=work)
        with self.assertRaises(ValueError):
            amd_fsr4.upscale(gray, 4, 4, sharpness=2, work=work)

    def test_helper_failures_become_plain_messages_and_leave_no_files(self):
        self.install()
        work = self.root / "work"
        pixels = torch.zeros(1, 2, 2, 3)
        cases = (
            (
                BoundedProcessError(2, b"", b"FAIL requested upscaler version is not available"),
                "not offered by this GPU or driver",
            ),
            (BoundedProcessError(2, b"", b"FAIL device removed (0x887a0005)"), "device removed"),
            (subprocess.TimeoutExpired("fsr_still", 1), "did not finish in time"),
        )
        for error, expected in cases:
            with (
                self.subTest(expected=expected),
                patch("ruinedfooocus.core.bounded_subprocess.run_bounded", side_effect=error),
                self.assertRaisesRegex(ValueError, expected),
            ):
                amd_fsr4.upscale(pixels, 4, 4, work=work)
            self.assertEqual(list(work.iterdir()), [])

        def truncated(command, **options):
            Path(command[2]).write_bytes(amd_fsr4.MAGIC + struct.pack("<II", 4, 4) + b"short")
            return b"", b""

        with (
            patch("ruinedfooocus.core.bounded_subprocess.run_bounded", truncated),
            self.assertRaisesRegex(ValueError, "incomplete"),
        ):
            amd_fsr4.upscale(pixels, 4, 4, work=work)

    def test_transform_resize_refuses_unsuitable_requests_before_queueing(self):
        self.assertIn(amd_fsr.INTERPOLATION, transform.INTERPOLATIONS)
        options = dict(
            input_image=Image.new("RGBA", (8, 6), (10, 20, 30, 255)),
            controlnet={"type": "image-transform"},
            transform_operation="resize",
            transform_interpolation="amd-fsr",
            transform_width=16,
            transform_height=12,
        )
        gpu = patch(
            "ruinedfooocus.features.image_processing.image_shaders.cpu_mode", return_value=False
        )
        with (
            patch(
                "ruinedfooocus.features.image_processing.image_shaders.cpu_mode", return_value=True
            ),
            self.assertRaisesRegex(ValueError, "CPU mode"),
        ):
            transform.capture_assets(dict(options), None)
        with gpu, self.assertRaisesRegex(ValueError, "runtime"):
            transform.capture_assets(dict(options), None)
        self.install()
        with gpu:
            captured, _ = transform.capture_assets(dict(options), None)
            self.assertEqual(captured[transform.KEY].output_size, (16, 12))
            with self.assertRaisesRegex(ValueError, "only enlarges"):
                transform.capture_assets(dict(options, transform_width=4, transform_height=3), None)
        self.assertEqual(
            transform._fsr_cover((8, 6), (16, 16), dict(transform_fit="center-crop")), (22, 16)
        )

    def test_each_version_asks_the_helper_for_its_own_provider_and_never_another(self):
        self.install()
        work = self.root / "work"
        pixels = torch.zeros(1, 2, 2, 3)
        self.assertEqual(amd_fsr.VERSIONS, ("fsr4", "fsr3", "fsr2", "fsr1"))
        self.assertEqual(amd_fsr.DEFAULT_VERSION, "fsr4")
        self.assertEqual(transform.DEFAULTS["transform_fsr_version"], "fsr4")
        for version, provider in (("fsr4", "4."), ("fsr3", "3."), ("fsr2", "2.")):
            self.calls.clear()
            with (
                self.subTest(version=version),
                patch("ruinedfooocus.core.bounded_subprocess.run_bounded", self.helper),
            ):
                result = amd_fsr.upscale(version, pixels, 4, 4)
                self.assertEqual(tuple(result.shape), (1, 4, 4, 3))
                self.assertEqual([command[7] for command, _ in self.calls], [provider])
                self.assertIn(version[-1], amd_fsr.source(version))
        self.calls.clear()
        with patch("ruinedfooocus.core.bounded_subprocess.run_bounded", self.helper):
            self.assertEqual(tuple(amd_fsr.upscale("fsr1", pixels, 4, 4).shape), (1, 4, 4, 3))
        self.assertEqual(self.calls, [])
        self.assertIn("FSR 1", amd_fsr.source("fsr1"))
        error = BoundedProcessError(2, b"", b"FAIL requested upscaler version is not available")
        with (
            patch("ruinedfooocus.core.bounded_subprocess.run_bounded", side_effect=error),
            self.assertRaisesRegex(ValueError, "AMD FSR 3 is not offered"),
        ):
            amd_fsr4.upscale(pixels, 4, 4, version="fsr3", work=work)
        # After the first pass FSR 2 and 3 sharpen less, so a 4x result has no stipple.
        for version, later in (("fsr4", "1.0"), ("fsr3", "0.8"), ("fsr2", "0.65")):
            self.calls.clear()
            with patch("ruinedfooocus.core.bounded_subprocess.run_bounded", self.helper):
                amd_fsr4.upscale(pixels, 16, 16, version=version, work=work)
            self.assertEqual([command[8] for command, _ in self.calls], ["1.0", later, later])
        self.calls.clear()
        with patch("ruinedfooocus.core.bounded_subprocess.run_bounded", self.helper):
            amd_fsr4.upscale(pixels, 8, 8, version="fsr2", sharpness=0.3, work=work)
        self.assertEqual([command[8] for command, _ in self.calls], ["0.3", "0.3"])
        with self.assertRaisesRegex(ValueError, "FSR 2, 3 or 4"):
            amd_fsr4.upscale(pixels, 4, 4, version="fsr1", work=work)

    def test_checks_name_the_version_and_point_at_fsr_1_when_the_gpu_path_is_closed(self):
        common = dict(color_space="sRGB", source_size=(8, 6), target_size=(16, 12))
        with self.assertRaisesRegex(ValueError, "AMD FSR 3 runs on the GPU.*FSR 1"):
            amd_fsr.check("fsr3", dtype="uint8", cpu=True, **common)
        with self.assertRaisesRegex(ValueError, "AMD FSR 2 works on 8-bit"):
            amd_fsr.check("fsr2", dtype="uint16", cpu=False, **common)
        with self.assertRaisesRegex(ValueError, "runtime.*FSR 1 needs no runtime"):
            amd_fsr.check("fsr4", dtype="uint8", cpu=False, **common)
        amd_fsr.check("fsr1", dtype="uint16", cpu=True, **common)
        with self.assertRaisesRegex(ValueError, "sRGB"):
            amd_fsr.check("fsr1", dtype="uint8", cpu=True, **dict(common, color_space="linear"))
        with self.assertRaisesRegex(ValueError, "8-bit or 16-bit"):
            amd_fsr.check("fsr1", dtype="float32", cpu=True, **common)


if __name__ == "__main__":
    unittest.main()
