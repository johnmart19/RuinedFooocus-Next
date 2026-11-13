"""Submission-time joins retain existing component, recipe and geometry owners."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan
from ruinedfooocus.features.models import effective_image_plan as plans
from ruinedfooocus.features.models import lora_sources
from ruinedfooocus.features.models.component_profiles import snapshot_request
from ruinedfooocus.features.models.model_identity import file_signature


def header(path, dtype="F16", metadata=None):
    value = {"weight": dict(shape=[1], dtype=dtype, data_offsets=[0, 0])}
    if metadata is not None:
        value["__metadata__"] = metadata
    raw = json.dumps(value).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)
    return path


class EffectiveImagePlanTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.path = header(self.root / "custom-renamed-turbo.safetensors")
        self.lora = header(self.root / "style.safetensors")
        self.family = "SDXL"
        self.models = SimpleNamespace(
            get_file=lambda category, name: self.lora if category == "loras" else self.path,
            model_dirs={"loras": [str(self.root)]},
            get_names=lambda category: [self.lora.name],
            get_models_by_path=lambda *args, **kwargs: {"baseModel": self.family},
            get_model_base=lambda info: info["baseModel"],
        )
        self.paths = SimpleNamespace(
            paths={"path_clip": [], "path_vae": []}, get_abspath=Path, DOWNLOADABLE_FILES={}
        )
        self.values = {"clip_l": "custom-encoder.safetensors"}
        self.request = dict(
            base_model_name=self.path.name,
            prompt="literal prompt",
            negative_prompt="retain this",
            negative="actual worker negative",
            denoise=0.37,
            loras=[["custom", "0.5 - style.safetensors"]],
            performance_selection="Own settings",
            custom_steps=17,
            cfg=4.5,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=2,
            custom_width=1001,
            custom_height=503,
            resolution_auto=False,
            size_mode="target",
            controlnet={"type": "img2img", "denoise": 0.37},
            _sampling_captured=True,
            **{SIZE_KEY: SizePlan.create((1001, 503), 2)},
        )
        self.real_native_facts = plans._native_facts
        self.enterContext(
            patch.object(
                plans,
                "_native_facts",
                return_value={
                    "snapshot": {"native_pin": "accepted-pin", "snapshot_sha256": "fixture"},
                    "family": {"family_id": "SDXL", "unresolved": []},
                },
            )
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                side_effect=AssertionError("No weight hashing during capture"),
            )
        )

    def capture(self, data=None):
        request = snapshot_request(data or self.request, self.models, self.values, self.paths)
        with patch(
            "ruinedfooocus.features.models.component_profiles.component_inventory",
            side_effect=AssertionError("Reuse the already captured inventory"),
        ):
            return plans.capture_effective_image_plan(request, self.models, self.values, self.paths)

    def test_actual_component_capture_is_joined_once_and_remains_immutable(self):
        captured = self.capture()
        plan = plans.get_effective_image_plan(captured)
        self.assertIs(plan.components, captured[plans.COMPONENTS_KEY])
        self.assertIs(plan.size.native_plan, self.request[SIZE_KEY])
        self.assertEqual(plan.size.native_plan.target, (1001, 503))
        self.assertEqual(plan.size.native_plan.canvas, (504, 256))
        self.assertEqual(plan.recipe.sampling["custom_steps"], 17)
        self.assertEqual(plan.size.ownership, "manual")
        self.assertEqual(plan.variant, None)  # A filename cannot declare Turbo.
        self.assertEqual(plan.quantization, ("F16",))
        self.values["clip_l"] = "later-global.safetensors"
        captured["image_component_overrides"]["clip_l"] = "later-request.safetensors"
        self.assertEqual(plan.components[0].filename, "custom-encoder.safetensors")
        detached = plan.metadata()
        detached["recipe"]["sampling"]["cfg"] = 99
        self.assertEqual(plan.recipe.sampling["cfg"], 4.5)
        with self.assertRaises(FrozenInstanceError):
            plan.family = "Flux"
        for key in ("negative_prompt", "denoise", "loras"):
            self.assertEqual(captured[key], self.request[key])

    def test_same_submission_retains_object_and_metadata_is_deterministic(self):
        first = self.capture()
        again = plans.capture_effective_image_plan(first, self.models, {}, self.paths)
        self.assertIs(again[plans.KEY], first[plans.KEY])
        self.assertEqual(self.capture()[plans.KEY].metadata(), first[plans.KEY].metadata())

    def test_stale_settings_or_model_replacement_cannot_reuse_plan(self):
        captured = self.capture()
        for changed in (
            {"cfg": 9},
            {"negative_prompt": "changed"},
            {"negative": "changed actual worker negative"},
            {"loras": []},
            {"custom_width": 999},
            {"native_negative_model": "new.safetensors"},
            {"reference_adapter": "different"},
            {"inpaint_denoise": 0.8},
            {"cn_strength": 0.9},
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "request changed"),
            ):
                plans.capture_effective_image_plan(captured | changed, self.models, {}, self.paths)
        header(self.path, "BF16")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            plans.capture_effective_image_plan(captured, self.models, {}, self.paths)

    def test_client_metadata_cannot_claim_a_captured_plan(self):
        with self.assertRaisesRegex(ValueError, "request service"):
            plans.capture_effective_image_plan(
                self.request | {plans.KEY: {}}, self.models, {}, self.paths
            )

    def test_replay_and_source_geometry_keep_ownership(self):
        replay = self.capture(self.request | {"_metadata_prompt": True, "resolution_auto": True})[
            plans.KEY
        ]
        self.assertEqual(replay.size.ownership, "replay")
        self.assertEqual(replay.recipe.ownership, "replay")
        source = self.request | {
            "input_image": Image.new("RGBA", (333, 221)),
            "image_resize_mode": "preserve",
        }
        source.pop(SIZE_KEY)
        captured = self.capture(source)[plans.KEY]
        self.assertEqual(captured.size.ownership, "source")
        self.assertEqual(captured.size.framing, "source_proportions")
        self.assertIsNone(captured.size.native_plan)

    def test_processing_plan_has_owned_source_geometry_and_no_inherited_sampling(self):
        request = self.request | dict(
            task_type="rembg",
            _sampling_captured=False,
            input_image=Image.new("RGBA", (333, 221)),
            controlnet={"type": "rembg"},
        )
        request.pop(SIZE_KEY)
        captured = plans.capture_effective_image_plan(request, None, {}, None)[plans.KEY]
        self.assertEqual(captured.operation, "rembg")
        self.assertEqual((captured.size.width, captured.size.height), (333, 221))
        self.assertEqual(captured.recipe.sampling, {})
        self.assertEqual(captured.required_components, ())
        self.assertEqual(captured.checkpoint_name, "")

    def test_only_active_required_inventory_rows_are_required(self):
        state = {
            "rows": [
                dict(
                    key=key,
                    category="clip",
                    location="",
                    filename=key,
                    status="Missing",
                    active=active,
                    required=required,
                )
                for key, active, required in (
                    ("main", True, True),
                    ("optional", True, False),
                    ("inactive", False, True),
                )
            ]
        }
        rows = plans.capture_component_inventory(state)
        captured = self.capture()[plans.KEY]
        from dataclasses import replace

        self.assertEqual(
            [row.key for row in replace(captured, components=rows).required_components], ["main"]
        )

    def test_alternate_storage_does_not_declare_behavior_or_require_curated_sha(self):
        for dtype in ("F16", "BF16", "F8_E4M3"):
            with self.subTest(dtype=dtype):
                header(self.path, dtype)
                plan = self.capture()[plans.KEY]
                self.assertEqual(plan.family, "SDXL")
                self.assertIsNone(plan.variant)
                self.assertEqual(plan.quantization, (dtype,))
                self.assertEqual(plan.recipe.ownership, "manual")

    def test_retired_operation_is_rejected_before_inventory(self):
        with self.assertRaisesRegex(ValueError, "still images"):
            plans.capture_effective_image_plan(
                self.request | {"operation": "video"}, None, {}, None
            )

    def test_installed_component_replacement_invalidates_the_capture(self):
        encoder = header(self.root / "custom-encoder.safetensors")
        self.paths.paths["path_clip"] = [str(self.root)]
        captured = self.capture()
        self.assertIsNotNone(captured[plans.KEY].components[0].signature)
        header(encoder, "BF16")
        with self.assertRaisesRegex(ValueError, "component clip_l changed"):
            plans.capture_effective_image_plan(captured, self.models, {}, self.paths)

    def test_gguf_quantization_uses_logical_header_storage_not_filename(self):
        import gguf
        import numpy as np

        self.path = self.root / "unregistered-any-name.gguf"
        writer = gguf.GGUFWriter(str(self.path), "sdxl")
        writer.add_string("modelspec.variant", "base")
        writer.add_tensor(
            "weight",
            gguf.quantize(np.zeros((1, 32), dtype=np.float32), gguf.GGMLQuantizationType.Q8_0),
            raw_dtype=gguf.GGMLQuantizationType.Q8_0,
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        captured = self.capture(self.request | {"base_model_name": self.path.name})[plans.KEY]
        self.assertEqual(captured.storage, "gguf")
        self.assertEqual(captured.quantization, ("Q8_0",))
        self.assertEqual(captured.family, "SDXL")
        self.path.unlink()  # Bounded header inspection releases the file on Windows.

    def test_current_native_header_cannot_inherit_a_different_cached_family(self):
        with (
            patch.dict("sys.modules", {"comfy.model_detection": SimpleNamespace()}),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_tensor_model",
                return_value={"architecture": "Flux.1 D"},
            ),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value={"architecture": "Flux.1 D", "native_class": "Flux"},
            ),
            self.assertRaisesRegex(ValueError, "header differs"),
        ):
            plans.capture_effective_image_plan(self.request, self.models, {}, self.paths)

    def test_plan_captures_real_bundled_snapshot_identity(self):
        with patch.object(plans, "_native_facts", side_effect=self.real_native_facts):
            facts = self.capture()[plans.KEY].metadata()["native_facts"]
        self.assertEqual(
            facts["snapshot"]["native_pin"], "989d93d0d968e704c589ce0146bb18c77f6b50cc"
        )
        self.assertEqual(facts["snapshot"]["gguf_pin"], "da5f435cd9a9e54ad27eed417ed72b1f2543ef41")
        self.assertEqual(len(facts["snapshot"]["snapshot_sha256"]), 64)
        self.assertEqual(facts["family"]["family_id"], "SDXL")

    def test_real_native_header_records_its_selected_config_and_variant(self):
        from test_native_variant_defaults import SHAPES, native_backend, write_header

        self.family = "QwenImage21"
        write_header(self.path, SHAPES["QwenImage21"], "F8_E4M3", {"variant": "base"})
        with native_backend():
            plan = self.capture()[plans.KEY]
        selection = plan.metadata()["native_facts"]["selection"]
        self.assertEqual(plan.native_class, "QwenImage21")
        self.assertEqual(plan.variant, "base")
        self.assertEqual(selection["native_class"], "QwenImage21")
        self.assertTrue(selection["native_config"])
        self.assertEqual(selection["variant_declarations"], ["base"])

    def test_bundle_primary_replacement_invalidates_unchanged_index(self):
        from test_local_diffusers import bundle_fixture
        from ruinedfooocus.features.models.local_diffusers import capture_bundle

        self.path = bundle_fixture(self.root / "bundle")
        captured = self.capture(self.request | {"base_model_name": str(self.path)})
        primary = capture_bundle(self.path).diffusion.path
        primary.write_bytes(b"replaced bundle primary fixture")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            captured[plans.KEY].verify_files()

    def test_selected_workflow_example_keeps_its_recipe_and_source_identity(self):
        facts = {
            "snapshot": {"snapshot_sha256": "fixture"},
            "family": {
                "example_recipe": {
                    "recipe_id": "connected-example",
                    "source": {
                        "kind": "native-workflow-example",
                        "sha256": "source-fixture-digest",
                    },
                }
            },
        }
        request = self.request | dict(
            performance_selection="Model Defaults",
            sampling_defaults_source="Native workflow example for SDXL; no publisher recommendation",
        )
        with patch.object(plans, "_native_facts", return_value=facts):
            recipe = self.capture(request)[plans.KEY].recipe.metadata()
        self.assertEqual(recipe["source_kind"], "native-workflow-example")
        example = recipe["provenance"]["template_example"]
        self.assertEqual(example["recipe_id"], "connected-example")
        self.assertEqual(example["source"]["sha256"], "source-fixture-digest")

    def test_component_snapshot_cannot_move_to_another_primary(self):
        request = snapshot_request(self.request, self.models, self.values, self.paths)
        self.path = header(self.root / "second.safetensors")
        request["base_model_name"] = self.path.name
        with self.assertRaisesRegex(ValueError, "components were captured for another checkpoint"):
            plans.capture_effective_image_plan(request, self.models, self.values, self.paths)

    def projector(self, encoder, target=None):
        from ruinedfooocus.features.models.gguf_projectors import GGUFProjector

        reader = self.root / "native-reader.py"
        if not reader.exists():
            reader.write_bytes(b"owned native reader source fixture")
        return GGUFProjector(
            tuple(file_signature(encoder)),
            tuple(file_signature(reader)),
            tuple(file_signature(target)) if target is not None else None,
            str(target) if target is not None else None,
        )

    def test_replay_owner_only_change_rejects_the_old_plan(self):
        captured = self.capture()
        with self.assertRaisesRegex(ValueError, "request changed"):
            plans.capture_effective_image_plan(
                captured | {"_metadata_prompt": True}, self.models, self.values, self.paths
            )
        replay = self.capture(self.request | {"_metadata_prompt": True})[plans.KEY]
        self.assertEqual(replay.recipe.ownership, "replay")
        self.assertEqual(replay.size.ownership, "replay")

    def test_projector_signature_and_captured_absence_share_the_effective_plan(self):
        from ruinedfooocus.features.models import gguf_projectors

        encoder = self.root / "paired.gguf"
        encoder.write_bytes(b"owned encoder metadata fixture")
        absent = (self.projector(encoder),)
        original = self.capture(self.request | {gguf_projectors.KEY: absent})
        plan = original[plans.KEY]
        self.assertIs(plan.projectors, absent)
        self.assertEqual(plan.metadata()["gguf_projectors"][0]["state"], "captured-absent")
        target = self.root / "projector.gguf"
        target.write_bytes(b"new sidecar does not retarget the queued absence")
        plan.verify_request(original)
        self.assertIsNone(plan.projectors[0].projector)
        present = (self.projector(encoder, target),)
        captured = self.capture(self.request | {gguf_projectors.KEY: present})
        self.assertEqual(
            captured[plans.KEY].metadata()["gguf_projectors"][0]["selected_path"], str(target)
        )
        target.write_bytes(b"replaced sidecar fixture")
        with self.assertRaisesRegex(ValueError, "changed"):
            captured[plans.KEY].verify_request(captured)

    def test_projector_packet_cannot_retarget_a_captured_request(self):
        from ruinedfooocus.features.models import gguf_projectors

        encoder = self.root / "paired.gguf"
        encoder.write_bytes(b"owned encoder metadata fixture")
        captured = self.capture(self.request | {gguf_projectors.KEY: (self.projector(encoder),)})
        target = self.root / "new-projector.gguf"
        target.write_bytes(b"different selected sidecar fixture")
        changed = captured | {gguf_projectors.KEY: (self.projector(encoder, target),)}
        with self.assertRaisesRegex(ValueError, "request changed"):
            captured[plans.KEY].verify_request(changed)
        with self.assertRaisesRegex(ValueError, "selection changed"):
            plans.finalize_prepared_image_plan(captured, changed)

    def prepared_projector(self):
        from ruinedfooocus.core.preparation import PreparationRequest
        from ruinedfooocus.features.generation.image_preparation import (
            KEY as ASSET_KEY,
            ImageArtifact,
            ImageAssets,
        )
        from ruinedfooocus.features.models import gguf_projectors

        encoder = self.root / "acquired-encoder.gguf"
        self.paths.paths["path_clip"] = [str(self.root)]
        self.values["clip_l"] = encoder.name
        original = self.capture()
        request = PreparationRequest("https://example.invalid/encoder.gguf", encoder)
        original[ASSET_KEY] = ImageAssets((ImageArtifact("clip", "clip_l", request, None),), True)
        encoder.write_bytes(b"owned prepared encoder fixture")
        signature = tuple(file_signature(encoder))
        prepared = original | {
            ASSET_KEY: ImageAssets((ImageArtifact("clip", "clip_l", request, signature),), True),
            gguf_projectors.KEY: (self.projector(encoder),),
        }
        return original, prepared

    def test_preparation_can_bind_only_its_new_encoder_projector_receipt(self):
        original, prepared = self.prepared_projector()
        result = plans.finalize_prepared_image_plan(original, prepared)
        before, after = original[plans.KEY], result[plans.KEY]
        for field in ("components", "recipe", "size"):
            self.assertIs(getattr(before, field), getattr(after, field))
        self.assertEqual(before.checkpoint_identity, after.checkpoint_identity)
        self.assertEqual(len(after.projectors), 1)
        self.assertIsNone(after.projectors[0].projector)
        after.verify_request(result)

    def test_preparation_without_new_dependencies_retains_the_plan_object(self):
        original = self.capture()
        prepared = plans.finalize_prepared_image_plan(original, dict(original))
        self.assertIs(prepared[plans.KEY], original[plans.KEY])

    def test_preparation_cannot_change_recipe_or_invent_a_projector_receipt(self):
        from ruinedfooocus.features.generation.image_preparation import KEY as ASSET_KEY

        original, prepared = self.prepared_projector()
        with self.assertRaisesRegex(ValueError, "request changed"):
            plans.finalize_prepared_image_plan(original, prepared | {"cfg": 19})
        with self.assertRaisesRegex(ValueError, "prepared receipt"):
            plans.finalize_prepared_image_plan(
                original, prepared | {ASSET_KEY: original[ASSET_KEY]}
            )

    def test_client_json_cannot_claim_a_native_projector_packet(self):
        from ruinedfooocus.features.models import gguf_projectors

        with self.assertRaisesRegex(ValueError, "typed GGUF"):
            self.capture(self.request | {gguf_projectors.KEY: [{}]})

    def test_installed_lora_source_stays_frozen_when_model_roots_change(self):
        captured = self.capture()
        plan = captured[plans.KEY]
        self.assertIs(plan.lora_sources, captured[lora_sources.KEY])
        self.assertEqual(plan.lora_sources[0].path, self.lora)
        replacement = header(self.root / "replacement.safetensors")
        self.models.get_file = lambda *args: replacement
        plan.verify_request(captured)
        self.assertEqual(plan.metadata()["lora_sources"][0]["signature"][0], str(self.lora))
        header(self.lora, "BF16")
        with self.assertRaisesRegex(ValueError, "LoRA changed"):
            plan.verify_request(captured)

    def test_pending_catalogue_lora_uses_the_exact_preparation_receipt(self):
        from ruinedfooocus.core.preparation import PreparationRequest
        from ruinedfooocus.features.generation.image_preparation import (
            KEY as ASSET_KEY,
            ImageArtifact,
            ImageAssets,
        )

        pending = self.root / "acquired-style.safetensors"
        request = PreparationRequest("https://example.invalid/style.safetensors", pending)
        self.models.get_file = lambda category, name: None if category == "loras" else self.path
        self.paths.DOWNLOADABLE_FILES = {
            "trusted-style": {"path": "path_loras", "filename": "style.safetensors"}
        }
        self.paths.preparation_request = lambda key: request
        original = self.capture(
            self.request | {"prompt": ["one <lora:style:0.5>", "two <lora:style:0.8>"]}
        )
        plan = original[plans.KEY]
        rows = [item for item in plan.required_components if item.category == "loras"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(
            (row.key, row.path, row.download_key),
            ("lora:style.safetensors", str(pending), "trusted-style"),
        )
        missing = ImageArtifact("loras", row.key, request, None)
        original[ASSET_KEY] = ImageAssets((missing,), True)
        header(pending)
        prepared = original | {
            ASSET_KEY: ImageAssets(
                (ImageArtifact("loras", row.key, request, tuple(file_signature(pending))),), True
            )
        }
        result = plans.finalize_prepared_image_plan(original, prepared)
        finalized = result[plans.KEY]
        self.assertIs(finalized.components, plan.components)
        self.assertIs(finalized.recipe, plan.recipe)
        self.assertIs(finalized.size, plan.size)
        self.assertEqual(finalized.lora_sources[0].signature, tuple(file_signature(pending)))
        self.assertEqual([item.weight for item in finalized.lora_sources], [0.5, 0.8])
        finalized.verify_request(result)
        with self.assertRaisesRegex(ValueError, "preparation receipt"):
            plans.finalize_prepared_image_plan(original, original)

    def test_missing_custom_lora_cannot_claim_catalogue_preparation(self):
        self.models.get_file = lambda category, name: None if category == "loras" else self.path
        with self.assertRaisesRegex(FileNotFoundError, "installed or catalogue"):
            self.capture()

    def test_original_prompt_children_keep_exact_adapter_selection_and_weights(self):
        prompts = ["one <lora:style:0.7>", "two <lora:style:-0.2>", "none <lora:style:0>"]
        captured = self.capture(self.request | {"prompt": prompts, "loras": []})
        plan = captured[plans.KEY]
        for prompt, weights in zip(prompts, ([0.7], [-0.2], [])):
            self.assertEqual([item.weight for item in plan.loras_for_prompt(prompt)], weights)
        self.assertEqual({item.path for item in plan.lora_sources}, {self.lora})
        with self.assertRaisesRegex(ValueError, "prompt"):
            plan.loras_for_prompt("not an original child")


if __name__ == "__main__":
    unittest.main()
