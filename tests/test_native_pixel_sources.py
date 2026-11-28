"""Primary header evidence separates native Zeta from the official L2P decoder."""

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

from ruinedfooocus.features.models.native_pixel_sources import (
    require_native_pixel_source,
    source_error,
)
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.resource_defaults import load_family_recipe

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
FIXTURE = ROOT / "tests/fixtures/zeta-l2p-source-headers.json.gz"
ARTIFACT = "existing:diffusion_models/zeta-chroma-base-x0-pixel-no-dino-1024.safetensors"


def write_l2p_fixture(path):
    import torch
    from safetensors.torch import save_file

    save_file(
        {
            "local_decoder.enc1.0.weight": torch.zeros((2, 3, 3, 3)),
            "local_decoder.bottleneck.0.weight": torch.zeros((2, 2, 1, 1)),
            "local_decoder.out_conv.weight": torch.zeros((3, 2, 1, 1)),
        },
        path,
    )


class NativePixelSourceTests(unittest.TestCase):
    def test_named_guard_checks_real_tensor_topology_and_cached_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.safetensors"
            write_l2p_fixture(path)
            with self.assertRaisesRegex(ValueError, "convolutional pixel decoder"):
                require_native_pixel_source(path)
            self.assertIsNone(source_error({"dec_net.cond_embed.weight": {"shape": [3840, 3840]}}))
            path.write_bytes(b"another native container candidate")
            require_native_pixel_source(path)
            for suffix in (".pt", ".bin", ".json"):
                require_native_pixel_source(path.with_suffix(suffix))

    def test_exact_zeta_tuple_keeps_auto_disabled_and_no_sampling_or_size_guess(self):
        row = load_registry().artifact(ARTIFACT)
        self.assertEqual(row["native_class"], "ZImagePixelSpace")
        self.assertFalse(row["auto_recipe_supported"])
        self.assertEqual(row["artifact"]["bytes"], 12997743288)
        self.assertEqual(
            row["artifact"]["sha256"],
            "c652596db0e8b8211ae19b527295184ddfe99e60677fe1b10920ebc0d8572964",
        )
        recipe = load_family_recipe(row["evidence"]["default_recipe_id"])
        self.assertFalse(recipe["applicability"]["auto_recipe_supported"])
        self.assertTrue(all(value is None for value in recipe["sampling"].values()))
        self.assertIsNone(recipe["default_size"])
        self.assertEqual(recipe["applicability"]["component_contract"]["vae_role"], "pixel_space")

    def test_actual_native_classifiers_dispatcher_components_and_defaults(self):
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
    sys.path.insert(0, str(ROOT / "tests"))
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
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
    from ruinedfooocus.features.models.model_architecture import (
        inspect_tensor_model,
        inspect_model_file,
    )
    from ruinedfooocus.features.models.component_profiles import component_inventory
    from ruinedfooocus.features.models.native_recipes import compatible_native_recipe
    from ruinedfooocus.features.models.model_recommendations import ModelRecommendations
    from ruinedfooocus.features.models.performance import (
        PerformanceSettings,
        UnavailableModelRecipe,
    )
    from test_image_preparation import Manager

    check = unittest.TestCase()
    fixture = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    selected = None
    for source in fixture["sources"]:
        raw = source["raw_header"].encode()
        check.assertEqual(hashlib.sha256(raw).hexdigest(), source["header_sha256"])
        header = json.loads(raw)
        tensors = {
            name: torch.empty(row["shape"], device="meta")
            for name, row in header.items()
            if name != "__metadata__"
        }
        if "/zhen-nan/L2P/" in source["url"]:
            converted = comfy.model_detection.convert_diffusers_mmdit(dict(tensors))
            native = comfy.model_detection.model_config_from_unet(converted, "")
            check.assertEqual(type(native).__name__, "ZImage")
            check.assertIsNone(inspect_tensor_model(header))
            check.assertIn("convolutional pixel decoder", source_error(header))
        else:
            native = comfy.model_detection.model_config_from_unet(tensors, "")
            if native is None:
                converted = comfy.model_detection.convert_diffusers_mmdit(dict(tensors))
                native = comfy.model_detection.model_config_from_unet(converted, "")
            check.assertEqual(type(native).__name__, "ZImagePixelSpace")
            contract = inspect_tensor_model(header, include_config=True)
            check.assertEqual(contract["native_class"], "ZImagePixelSpace")
            check.assertEqual(contract["native_config"]["in_channels"], 3)
            check.assertTrue(contract["native_config"]["use_x0"])
            check.assertEqual(contract["native_config"]["decoder_num_res_blocks"], 4)
            if "no-dino-1024" in source["url"]:
                selected = source
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "renamed-native.safetensors"
        raw = selected["raw_header"].encode()
        path.write_bytes(len(raw).to_bytes(8, "little") + raw)
        require_native_pixel_source(path)
        contract = inspect_model_file(path)
        check.assertEqual(contract["native_class"], "ZImagePixelSpace")
        inventory = component_inventory(path, "ZImagePixelSpace", {}, Manager(root))
        check.assertEqual([row["key"] for row in inventory["rows"]], ["clip_qwen3_4b"])
        check.assertFalse(any(row["category"] == "vae" for row in inventory["rows"]))
        recipe = compatible_native_recipe(path, {})
        check.assertFalse(recipe["auto_recipe_supported"])
        check.assertTrue(all(value is None for value in recipe["sampling"].values()))
        performance = object.__new__(PerformanceSettings)
        performance.default_settings = dict(
            custom_steps=20, cfg=1, sampler_name="euler", scheduler="simple", clip_skip=1
        )
        performance.performance_options, performance.legacy_options = {}, {}
        performance.model_recommendations = ModelRecommendations(
            cache_dir=root / "cache", path=root / "settings.json", offline=True
        )
        with check.assertRaisesRegex(UnavailableModelRecipe, "Own settings"):
            performance.get_perf_options("Model Defaults", str(path), "ZImagePixelSpace")
        check.assertEqual(
            performance.get_perf_options("Own settings"), performance.default_settings
        )
        with patch.dict(
            sys.modules,
            {
                "ruinedfooocus.backends.comfy.gguf_loader": SimpleNamespace(
                    load_diffusion_model=Mock()
                ),
            },
        ):
            from ruinedfooocus.features.generation.pipeline_utils import load_diffusion_model
        wrong = root / "actual-l2p.safetensors"
        write_l2p_fixture(wrong)
        with patch.object(comfy.sd, "load_diffusion_model") as load:
            with check.assertRaisesRegex(ValueError, "convolutional pixel decoder"):
                load_diffusion_model(str(wrong))
            load.assert_not_called()
            load_diffusion_model(str(path))
            load.assert_called_once_with(str(path), model_options={})
