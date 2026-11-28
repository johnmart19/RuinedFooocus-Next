"""Data-only YAML parsing and local file identity; never initialize a model."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.features.models import yaml_recipes as recipes

TEXT = """model:
  target: inert.module.DoesNotExist
  params:
    parameterization: v
    cond_stage_config:
      target: also.inert.TextEncoder
      params:
        layer_idx: -2
"""


class YamlParserTests(unittest.TestCase):
    def test_extracts_only_native_fields_without_executing_declared_targets(self):
        with (
            patch("os.system", side_effect=AssertionError("YAML executed code")),
            patch("importlib.import_module", side_effect=AssertionError("YAML imported code")),
        ):
            parsed = recipes.parse_recipe(TEXT)
        self.assertEqual(parsed, recipes.RecipeOverrides("v_prediction", -2))
        for index in (-8, -1, 0, 2):
            with self.subTest(index=index):
                parsed = recipes.parse_recipe(TEXT.replace("layer_idx: -2", f"layer_idx: {index}"))
                self.assertEqual(parsed.clip_layer_idx, index)

    def test_eps_declaration_keeps_detected_prediction_and_layer_is_independent(self):
        self.assertIsNone(
            recipes.parse_recipe(
                TEXT.replace("parameterization: v", "parameterization: eps")
            ).prediction
        )
        parsed = recipes.parse_recipe(TEXT.replace("    parameterization: v\n", ""))
        self.assertEqual(parsed, recipes.RecipeOverrides(None, -2))
        parsed = recipes.parse_recipe("model: {params: {parameterization: v}}")
        self.assertEqual(parsed, recipes.RecipeOverrides("v_prediction", None))

    def test_rejects_code_tags_aliases_duplicate_or_nontext_keys_and_multiple_documents(self):
        invalid = [
            "!!python/object/apply:os.system ['echo forbidden']",
            "model: !Custom {params: {parameterization: v}}",
            "model: &m {params: {parameterization: v}}\ncopy: *m",
            "model: {params: {parameterization: v, parameterization: eps}}",
            "model: {params: {parameterization: v}, params: {}}",
            "model: {params: {parameterization: v}, 1: ignored}",
            "model: {params: {parameterization: v}, <<: {other: value}}",
            "model: {params: {parameterization: v}, timestamp: 2026-11-09}",
            TEXT + "---\nmodel: {params: {parameterization: eps}}",
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                recipes.parse_recipe(payload)

    def test_rejects_unsupported_fields_types_empty_or_malformed_recipe(self):
        invalid = [
            "",
            "[]",
            "model: []",
            "model: {params: []}",
            "model: {params: {}}",
            "model: {params: {cond_stage_config: null}}",
            "model: {params: {parameterization: true}}",
            TEXT.replace("parameterization: v", "parameterization: epsilon"),
            TEXT.replace("parameterization: v", "parameterization: null"),
            TEXT.replace("layer_idx: -2", "layer_idx: true"),
            TEXT.replace("layer_idx: -2", "layer_idx: '-2'"),
            TEXT.replace("layer_idx: -2", "layer_idx: -2.0"),
            b"\xff",
            object(),
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                recipes.parse_recipe(payload)

    def test_bounds_actual_bytes_node_count_and_nesting_before_construction(self):
        for payload in (
            TEXT + "#" * recipes.MAX_BYTES,
            TEXT + "#" + "é" * (recipes.MAX_BYTES // 2),
            TEXT + "padding: [" + ",".join("x" for _ in range(recipes.MAX_NODES)) + "]",
            TEXT + "padding: " + "[" * recipes.MAX_DEPTH + "0" + "]" * recipes.MAX_DEPTH,
        ):
            with self.subTest(size=len(payload)), self.assertRaises(ValueError):
                recipes.parse_recipe(payload)


class YamlCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.recipe = self.root / "chosen.yaml"
        self.recipe.write_text(TEXT, encoding="utf-8")
        self.checkpoint = self.root / "chosen.safetensors"
        self.checkpoint.write_bytes(b"opaque weight fixture must never be read")
        self.service = PreparationService(
            transport=Mock(side_effect=AssertionError("No acquisition"))
        )
        self.addCleanup(self.service._executor.shutdown, wait=True)

    def capture(self, **options):
        return recipes.capture_recipe(
            self.recipe, self.checkpoint, family="SD 2.0", service=self.service, **options
        )

    def test_frozen_files_request_layer_default_and_explicit_recipe_opt_in(self):
        original_open = Path.open

        def recipe_only_open(path, *args, **kwargs):
            self.assertNotEqual(path, self.checkpoint, "Capture read checkpoint weights")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", recipe_only_open):
            request = self.capture()
            explicit = self.capture(clip_layer_source="recipe")
            with recipes.recipe_read_scope(request, service=self.service) as prepared:
                self.assertIs(prepared, request)
        self.assertEqual(request.clip_layer_source, "request")
        self.assertIsNone(request.recipe_clip_layer)  # Includes explicit/legacy Clip Skip 1.
        self.assertEqual(explicit.recipe_clip_layer, -2)
        self.assertNotEqual(request.identity, explicit.identity)
        self.assertEqual(request.identity, self.capture().identity)
        self.assertEqual(self.service._artifacts, {})

    def test_refuses_other_engines_families_policy_or_missing_companions(self):
        for options in ({"engine": "sdcpp-vulkan"}, {"clip_layer_source": "automatic"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture(**options)
        for family in ("Flux", "SD3", "Chroma", "Omnigen2", "Unknown", "SDXL_instructpix2pix"):
            with self.subTest(family=family), self.assertRaises(ValueError):
                recipes.capture_recipe(
                    self.recipe, self.checkpoint, family=family, service=self.service
                )
        for name in ("model.gguf", "model_index.json", "weights.zip", "chosen.yaml"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "single-file"):
                recipes.capture_recipe(
                    self.recipe, self.root / name, family="SDXL", service=self.service
                )
        self.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.capture()
        self.assertEqual(self.service._artifacts, {})

    def test_changed_or_deleted_yaml_checkpoint_and_content_digest_are_refused(self):
        captured = self.capture()
        self.recipe.write_text(TEXT.replace("-2", "-1"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            captured.verify()
        # Even a matching stat signature cannot hide changed YAML bytes.
        with (
            patch.object(recipes.RecipeFile, "verify", return_value=None),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            captured.verify()
        captured = self.capture()
        self.checkpoint.write_bytes(b"new checkpoint identity")
        with self.assertRaisesRegex(ValueError, "changed"):
            captured.verify()
        captured = self.capture()
        self.recipe.unlink()
        with self.assertRaisesRegex(ValueError, "changed"):
            captured.verify()

    def test_resolved_path_retarget_and_mutation_during_load_are_refused(self):
        captured = self.capture()
        other = self.root / "other.yaml"
        other.write_text(TEXT, encoding="utf-8")
        # Reproduce the observable result of a symlink/junction retarget without
        # requiring Windows symlink creation privileges in a CPU fixture.
        retargeted = replace(captured, recipe=replace(captured.recipe, source=other))
        with self.assertRaisesRegex(ValueError, "changed"):
            retargeted.verify()
        with self.assertRaisesRegex(ValueError, "changed"):
            with recipes.recipe_read_scope(captured, service=self.service):
                self.recipe.write_text(TEXT + "# changed during load", encoding="utf-8")
        self.assertEqual(self.service._artifacts, {})

    def test_actual_read_bound_survives_growth_after_stat_and_cancellation_releases_leases(self):
        original_open = Path.open

        def oversized_open(path, *args, **kwargs):
            if path == self.recipe:
                return io.BytesIO(b"x" * (recipes.MAX_BYTES + 1))
            return original_open(path, *args, **kwargs)

        with (
            patch.object(Path, "open", oversized_open),
            self.assertRaisesRegex(ValueError, "64 KiB"),
        ):
            self.capture()
        captured = self.capture()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(CancelledError):
            self.capture(cancelled=cancelled)
        with self.assertRaises(CancelledError):
            with recipes.recipe_read_scope(captured, cancelled=cancelled, service=self.service):
                self.fail("Cancelled recipe reached loader")
        self.assertEqual(self.service._artifacts, {})

    def test_existing_read_lease_covers_both_exact_files_without_replacement_service(self):
        captured = self.capture()
        held = []

        @contextmanager
        def lease(path, *, cancelled=None):
            held.append(path)
            try:
                yield path
            finally:
                held.remove(path)

        with patch.object(self.service, "artifact_lease", side_effect=lease):
            with recipes.recipe_read_scope(captured, service=self.service):
                self.assertEqual(held, [self.recipe, self.checkpoint])
        self.assertFalse(held)


if __name__ == "__main__":
    unittest.main()
