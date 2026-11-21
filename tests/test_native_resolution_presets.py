"""Execute the pinned native selector; only its result container is a CPU shim."""

import json
import math
import os
from pathlib import Path
import runpy
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models import native_resolutions
from ruinedfooocus.features.models.resolution_dropdown import ResolutionDropdown, dimension_value
from ruinedfooocus.features.models.resolutions import ResolutionSettings
from ruinedfooocus.features.models.resource_defaults import load_family_recipe

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


class NativeResolutionPresetTests(unittest.TestCase):
    def setUp(self):
        native_resolutions._aspect_presets.cache_clear()
        self.addCleanup(native_resolutions._aspect_presets.cache_clear)
        io = SimpleNamespace(
            ComfyNode=object, NodeOutput=lambda *values: SimpleNamespace(result=values)
        )
        with patch.dict(
            "sys.modules",
            {
                "comfy_api": ModuleType("comfy_api"),
                "comfy_api.latest": SimpleNamespace(ComfyExtension=object, io=io),
            },
        ):
            module = runpy.run_path(str(NATIVE / "comfy_extras/nodes_resolution.py"))
        native = ModuleType("comfy_extras.nodes_resolution")
        native.__dict__.update(module)
        self.native = native
        self.modules = patch.dict(
            "sys.modules",
            {"comfy_extras": ModuleType("comfy_extras"), "comfy_extras.nodes_resolution": native},
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_native_ratio_grid_scales_with_the_model_canvas_instead_of_copying_sdxl(self):
        first = native_resolutions.aspect_presets(1024, 1024)
        second = native_resolutions.aspect_presets(1536, 1536)
        expected = tuple(
            tuple(self.native.ResolutionSelector.execute(ratio, 1.0, 8).result)
            for ratio in self.native.ASPECT_RATIOS
        )
        self.assertEqual(first, expected)
        self.assertEqual(len(first), 8)
        self.assertIn((840, 1256), first)
        self.assertNotIn((832, 1216), first)
        self.assertNotEqual(first, second)
        self.assertTrue(all(w % 8 == h % 8 == 0 for w, h in second))
        self.assertTrue(all(abs(w * h - 1536**2) < 1536**2 * 0.02 for w, h in second))
        flux = load_family_recipe("native-flux2-dev")
        self.assertEqual(native_resolutions.canvas_multiple(flux), 16)
        aligned = native_resolutions.aspect_presets(
            1024, 1024, native_resolutions.canvas_multiple(flux)
        )
        self.assertTrue(all(w % 16 == h % 16 == 0 for w, h in aligned))
        self.assertNotEqual(first, aligned)
        self.assertEqual(native_resolutions.canvas_multiple(load_family_recipe("FLUX 3 Klein")), 16)
        self.assertEqual(native_resolutions.canvas_multiple(load_family_recipe("Anima")), 8)
        self.assertEqual(native_resolutions.canvas_multiple({}, "Flux3"), 16)

    def test_anima_follows_native_sampling_and_preset_geometry_and_fixed_models_do_not_expand(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(ResolutionSettings, "RESOLUTIONS_FILE", Path(folder) / "resolutions.json"),
        ):
            settings = ResolutionSettings()
            choices = settings.choices_for_model("Anima")
            self.assertEqual(len(choices), 9)
            self.assertTrue(all("ComfyUI preset" in c for c in choices[:-1]))
            self.assertEqual(settings.aspect_ratios[choices[1]], (840, 1256))
            self.assertEqual(dimension_value(choices[1]), "840x1256")
            dropdown = ResolutionDropdown(choices, value=choices[1])
            self.assertEqual(dropdown.preprocess(choices[1]), "840x1256")
            recipe = load_family_recipe("Anima")
            self.assertEqual(
                (recipe["sampling"]["sampler_name"], recipe["sampling"]["scheduler"]),
                ("er_sde", "simple"),
            )
            for identifier in ("native-qwenimage-layered", "native-pid-flux1-1024-to-4096-4step"):
                performance = SimpleNamespace(
                    family_preset=lambda *args: identifier,
                    model_recommendations=SimpleNamespace(
                        cached_resolutions=lambda *_: ([], None), metadata=lambda *_: {}
                    ),
                )
                choices = settings.choices_for_model("QwenImage", "inert.gguf", performance)
                self.assertEqual(len(choices), 2)

    def test_explicit_publisher_list_never_inherits_an_unrelated_family_list(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(ResolutionSettings, "RESOLUTIONS_FILE", Path(folder) / "resolutions.json"),
        ):
            settings = ResolutionSettings()
            publisher = SimpleNamespace(
                cached_resolutions=lambda *_: ([(812, 1216), (1152, 1536)], "Civitai version 42"),
                metadata=lambda *_: {},
            )
            performance = SimpleNamespace(
                model_recommendations=publisher, family_preset=lambda *_: "Anima"
            )
            choices = settings.choices_for_model("Anima", "inert.safetensors", performance)
            self.assertEqual(
                [settings.aspect_ratios[c] for c in choices[:-1] if "recommended" in c],
                [(812, 1216), (1152, 1536)],
            )
            self.assertTrue(
                any(
                    settings.aspect_ratios[c][0] > settings.aspect_ratios[c][1]
                    for c in choices[:-1]
                )
            )
            self.assertNotIn((832, 1216), [settings.aspect_ratios[c] for c in choices[:-1]])

    def test_aspect_ratio_labels_and_math_fallback_without_comfy_extras(self):
        self.assertEqual(native_resolutions.aspect_ratio_label(1024, 1024), "1:1")
        self.assertEqual(native_resolutions.aspect_ratio_label(840, 1256), "2:3")
        self.assertEqual(native_resolutions.aspect_ratio_label(1368, 768), "16:9")
        native_resolutions._aspect_presets.cache_clear()
        with patch.dict("sys.modules", {"comfy_extras.nodes_resolution": None}):
            fallback = native_resolutions.aspect_presets(1024, 1024, 8)
        self.assertEqual(len(fallback), 8)
        self.assertIn((840, 1256), fallback)
        self.assertIn((1368, 768), fallback)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(ResolutionSettings, "RESOLUTIONS_FILE", Path(folder) / "resolutions.json"),
        ):
            settings = ResolutionSettings()
            choices = settings.choices_for_model("Anima")
            self.assertIn("1024x1024 (1:1 · ComfyUI preset)", choices)
            self.assertIn("840x1256 (2:3 · ComfyUI preset)", choices)


if __name__ == "__main__":
    unittest.main()
