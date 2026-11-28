from ruinedfooocus.core.paths import asset_path, resource_path

"""CPU-only replay of the production metadata importer and guidance policy."""
import ast
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ruinedfooocus.features.image_editing.image_guidance import resolve_guidance

ROOT = Path(__file__).resolve().parents[1]


def replay(parameters, family="QwenImage"):
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/prompting/prompt_processing.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "process_metadata"
    )
    scope = {"asset_path": asset_path, "resource_path": resource_path, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "metadata_replay", "exec"), scope)
    models = SimpleNamespace(
        get_models_by_path=lambda *args, **kwargs: {},
        get_model_base=lambda info: family,
        get_file=lambda *args: Path("model.safetensors"),
        detect_checkpoint_base=lambda path: family,
    )
    with patch.dict("sys.modules", {"ruinedfooocus.app.state": SimpleNamespace(models=models)}):
        return scope["process_metadata"](
            {
                "base_model_name": "model.safetensors",
                "prompt": json.dumps({"prompt": "edit", **parameters}),
            }
        )


class EffectiveShiftReplayTests(unittest.TestCase):
    def test_qwen_legacy_turbo_sigmas_restore_a_visible_schedule_choice(self):
        from ruinedfooocus.features.models.model_recommendations import QWEN_IMAGE21_TURBO_SIGMAS
        from ruinedfooocus.features.image_editing.image_guidance import restore_guidance_metadata

        metadata = {"effective_sampling_sigmas": list(QWEN_IMAGE21_TURBO_SIGMAS)}
        self.assertEqual(
            restore_guidance_metadata(metadata, family="QwenImage21"),
            {"qwen_image21_schedule": "turbo_8"},
        )
        imported = replay({"image_guidance": metadata}, family="QwenImage21")
        self.assertEqual(imported["qwen_image21_schedule"], "turbo_8")
        self.assertEqual(
            resolve_guidance(imported, {}, "QwenImage21")["qwen_image21_schedule"], "turbo_8"
        )
        self.assertEqual(restore_guidance_metadata(metadata, family="SDXL"), {})
        for explicit in ("standard", "turbo_8"):
            self.assertEqual(
                restore_guidance_metadata(
                    metadata | {"qwen_image21_schedule": explicit}, family="QwenImage21"
                )["qwen_image21_schedule"],
                explicit,
            )
        self.assertEqual(
            restore_guidance_metadata(
                metadata | {"qwen_image21_schedule": "auto"}, family="QwenImage21"
            )["qwen_image21_schedule"],
            "turbo_8",
        )
        for sigmas in ([1, 0], [True, *QWEN_IMAGE21_TURBO_SIGMAS[1:]], "1,0"):
            self.assertEqual(
                restore_guidance_metadata(
                    {"effective_sampling_sigmas": sigmas}, family="QwenImage21"
                ),
                {},
            )

    def test_recorded_auto_shift_replays_when_current_model_default_changes(self):
        imported = replay(
            {
                "sampling_shift": None,
                "effective_sampling_shift": 3.1,
                "effective_sampling_shift_mode": "flow",
            }
        )
        self.assertEqual(resolve_guidance(imported, {}, "QwenImage")["sampling_shift"], 3.1)
        self.assertEqual(imported["effective_sampling_shift"], 3.1)

    def test_explicit_requested_shift_has_precedence(self):
        imported = replay({"sampling_shift": 2.2, "effective_sampling_shift": 3.1})
        self.assertEqual(imported["sampling_shift"], 2.2)

    def test_grouped_metadata_can_restore_recorded_effective_shift(self):
        imported = replay(
            {
                "image_guidance": {
                    "sampling_shift": None,
                    "effective_sampling_shift": 4.0,
                    "effective_sampling_shift_mode": "flow",
                }
            }
        )
        self.assertEqual(imported["sampling_shift"], 4.0)

    def test_grouped_moss_guidance_replays_all_fields_and_keeps_unrelated_metadata(self):
        imported = replay(
            {
                "cfg": 4.5,
                "seed": 71,
                "image_guidance": {
                    "moss_image_edit_format": "Moss-Image-Edit",
                    "moss_image_edit_strength": 2.5,
                    "moss_image_slg": True,
                    "sampling_shift": None,
                    "moss_image_slg_double_layers": "1, 3",
                    "moss_image_slg_single_layers": "8, 9",
                    "moss_image_slg_start": 0.2,
                    "moss_image_slg_end": 0.8,
                    "cfg_norm_pre_cfg": True,
                    "effective_sampling_shift": 5.0,
                    "cfg_norm_strength": 1.25,
                    "image_resize_mode": "crop",
                    "vae_tiled": True,
                    "vae_tile_size": 256,
                    "vae_overlap": 32,
                },
            },
            family="MOSSImage",
        )
        values = resolve_guidance(imported, {}, "MOSSImage")
        from ruinedfooocus.features.image_editing.image_guidance import FIELD_SCHEMA

        self.assertEqual(
            values,
            {
                **{key: spec["default"] for key, spec in FIELD_SCHEMA.items()},
                "hidream_o1_seam_smoothing": False,
                "moss_image_edit_format": "Moss-Image-Edit",
                "moss_image_edit_strength": 2.5,
                "moss_image_slg": True,
                "moss_image_slg_double_layers": "1, 3",
                "moss_image_slg_single_layers": "8, 9",
                "moss_image_slg_start": 0.2,
                "moss_image_slg_end": 0.8,
                "cfg_norm_pre_cfg": True,
                "sampling_shift": 5.0,
                "cfg_norm_strength": 1.25,
                "image_resize_mode": "crop",
                "vae_tiled": True,
                "vae_tile_size": 256,
                "vae_overlap": 32,
            },
        )
        self.assertEqual(imported["cfg"], 4.5)
        self.assertEqual(imported["seed"], 71)
        self.assertEqual(imported["prompt"], "edit")
        self.assertEqual(imported["performance_selection"], "Own settings")

    def test_invalid_recorded_effective_shift_is_not_promoted(self):
        for value in (True, float("nan"), float("inf"), -1, 0, 101, "not a number"):
            with self.subTest(value=value):
                imported = replay({"sampling_shift": None, "effective_sampling_shift": value})
                self.assertIsNone(imported["sampling_shift"])

    def test_native_flux_shift_is_not_promoted_to_unsupported_override(self):
        imported = replay(
            {"sampling_shift": None, "effective_sampling_shift": 1.15}, family="Flux.1 D"
        )
        self.assertIsNone(imported["sampling_shift"])
        self.assertIsNone(resolve_guidance(imported, {}, "Flux.1 D")["sampling_shift"])

    def test_older_auto_metadata_without_effective_shift_stays_auto(self):
        self.assertIsNone(replay({"sampling_shift": None})["sampling_shift"])


if __name__ == "__main__":
    unittest.main()
