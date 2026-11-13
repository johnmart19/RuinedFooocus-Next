"""Finite inventories preserve independent denominators and unfinished work."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_native_model_coverage import read_official_templates
from tools.native_capability_inventory import (
    config_assignment,
    node_assignment,
    recipe_inventory,
    reconcile_inventory,
)


def native_node(name="LoadImage", **values):
    return {
        "id": name,
        "native_domain": "image",
        "role": "image-processing",
        "RF_source_references": [],
        **values,
    }


class NativeCapabilityInventoryTests(unittest.TestCase):
    def test_shared_still_image_operations_are_not_excluded_by_video_class_or_domain(self):
        for name, packet in (
            ("MiniMaxH3", "G"),
            ("SeedVR2", "U"),
            ("HunyuanVideo15", "G"),
            ("WAN21_T2V", "G"),
            ("WAN21_I2V", "G"),
        ):
            row = config_assignment({"class": name, "RF_runtime_classes": []})
            self.assertEqual(
                (row["packet"], row["disposition"]), (packet, "planned-operation-review")
            )
        for name in (
            "SeedVR2Preprocess",
            "SeedVR2PostProcessing",
            "SeedVR2Conditioning",
            "LTXVSeparateAVLatent",
        ):
            self.assertEqual(
                node_assignment(native_node(name, native_domain="video"))["packet"], "U"
            )
        for name in ("HunyuanRefinerLatent", "MiniMaxH3ImageConditioning"):
            row = node_assignment(
                native_node(
                    name, native_domain="video", RF_source_references=[{"path": "still.py"}]
                )
            )
            self.assertEqual(row["packet"], "G")
            self.assertEqual(row["disposition"], "source-consumer-needs-acceptance")
        self.assertEqual(
            node_assignment(native_node("SeedVR2TemporalChunk", native_domain="video"))[
                "disposition"
            ],
            "scoped-exclusion",
        )
        report = {
            "native_clip_types": {"COSMOS": 11, "MINIMAX": 35, "WAN": 13, "MOCHI": 7},
            "RF_image_clip_types": ["COSMOS"],
            "native_folder_roles": [],
            "native_nodes": [],
            "native_config_classes": [],
        }
        rows = {
            row["id"]: row for row in reconcile_inventory(report, {}, {}, {})["native_clip_types"]
        }
        self.assertEqual(rows["COSMOS"]["packet"], "F")
        self.assertEqual(rows["MINIMAX"]["packet"], "F")
        self.assertEqual(rows["WAN"]["packet"], "F")
        self.assertEqual(rows["MOCHI"]["disposition"], "scoped-exclusion")

    def test_temporal_and_mesh_only_nodes_do_not_inherit_generic_model_image_scope(self):
        for name, domain in (
            ("ReplaceVideoLatentFrames", "model"),
            ("Kandinsky5ImageToVideo", "model"),
            ("LatentUpscaleModelLoader", "model"),
            ("UNetTemporalAttentionMultiply", "unclassified"),
            ("MoGePointMapToMesh", "image"),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    node_assignment(native_node(name, native_domain=domain))["disposition"],
                    "scoped-exclusion",
                )
        self.assertEqual(
            node_assignment(native_node("FutureVideoNode", native_domain="video"))["disposition"],
            "scoped-exclusion",
        )

    def test_pure_image_dataset_helpers_do_not_admit_training_or_video_operations(self):
        for name, category in (
            ("ResizeImagesByLongerEdge", "image/transform"),
            ("NormalizeImages", "image/color"),
            ("ImageGrid", "image/batch"),
            ("TruncateText", "text"),
        ):
            self.assertEqual(
                node_assignment(
                    native_node(name, native_domain="training", native_category=category)
                )["disposition"],
                "planned-operation-review",
            )
        for name, category in (
            ("LoadTrainingDataset", "model/training"),
            ("VideoTemporalCrop", "video/transform"),
        ):
            self.assertEqual(
                node_assignment(
                    native_node(name, native_domain="training", native_category=category)
                )["disposition"],
                "scoped-exclusion",
            )

    def test_shared_image_blueprints_keep_scope_without_admitting_temporal_nodes(self):
        def document(name):
            return json.dumps(
                {"nodes": [{"type": name}], "outputs": [{"name": "IMAGE", "type": "IMAGE"}]}
            ).encode()

        result = recipe_inventory(
            {
                "refine.json": document("HunyuanRefinerLatent"),
                "minimax.json": document("MiniMaxH3ImageConditioning"),
                "temporal.json": document("ReplaceVideoLatentFrames"),
            },
            [
                native_node("HunyuanRefinerLatent", native_domain="video"),
                native_node("MiniMaxH3ImageConditioning", native_domain="video"),
                native_node("ReplaceVideoLatentFrames", native_domain="model"),
            ],
        )
        rows = {row["id"]: row for row in result["recipes"]}
        for name in ("refine.json", "minimax.json"):
            self.assertEqual(rows[name]["scope"], "still-image-or-utility")
            self.assertEqual(len(rows[name]["shared_image_node_ids"]), 1)
            self.assertEqual(rows[name]["excluded_node_ids"], [])
        self.assertEqual(rows["temporal.json"]["scope"], "video")
        self.assertEqual(rows["temporal.json"]["excluded_node_ids"], ["ReplaceVideoLatentFrames"])

    def test_mixed_moge_recipe_retains_image_outputs_and_marks_mesh_branch_excluded(self):
        documents = {}
        for name, outputs in (("mixed", ["IMAGE", "MESH"]), ("mesh", ["MESH"])):
            documents[name + ".json"] = json.dumps(
                {
                    "nodes": [{"type": "MoGePointMapToMesh"}, {"type": "MoGeRender"}],
                    "outputs": [{"name": output, "type": output} for output in outputs],
                }
            ).encode()
        rows = {
            row["id"]: row
            for row in recipe_inventory(
                documents,
                [native_node("MoGePointMapToMesh"), native_node("MoGeRender")],
            )["recipes"]
        }
        self.assertEqual(rows["mixed.json"]["scope"], "still-image-or-utility")
        self.assertEqual(rows["mixed.json"]["excluded_node_ids"], ["MoGePointMapToMesh"])
        self.assertEqual(rows["mixed.json"]["declared_output_types"], ["IMAGE", "MESH"])
        self.assertEqual(rows["mesh.json"]["scope"], "3d")

    def test_gemini_image_blueprint_is_hosted_even_without_api_filename(self):
        row = recipe_inventory(
            {"Image Captioning.json": json.dumps({"nodes": [{"type": "GeminiNode"}]}).encode()},
            [],
        )["recipes"][0]
        self.assertEqual(row["scope"], "hosted-partner")
        self.assertEqual(row["disposition"], "scoped-exclusion")

    def test_nested_image_preprocessing_does_not_turn_splat_recipe_into_image_tool(self):
        row = recipe_inventory(
            {
                "fixture.json": json.dumps(
                    {
                        "nodes": [{"type": "splat"}],
                        "definitions": {
                            "subgraphs": [
                                {
                                    "id": "splat",
                                    "nodes": [{"type": "VAEDecodeTripoSplat"}, {"type": "prep"}],
                                    "outputs": [{"type": "SPLAT"}],
                                },
                                {"id": "prep", "nodes": [], "outputs": [{"type": "IMAGE"}]},
                            ]
                        },
                    }
                ).encode()
            },
            [],
        )["recipes"][0]
        self.assertEqual(row["scope"], "3d")
        self.assertEqual(row["declared_output_types"], ["SPLAT"])

    def test_explicit_root_mesh_exports_override_direct_image_preprocessing_wrapper(self):
        for root_exports in ([{"type": "MESH"}], []):
            with self.subTest(root_exports=root_exports):
                row = recipe_inventory(
                    {
                        "fixture.json": json.dumps(
                            {
                                "nodes": [{"type": "mesh"}, {"type": "prep"}],
                                "outputs": root_exports,
                                "definitions": {
                                    "subgraphs": [
                                        {
                                            "id": "mesh",
                                            "nodes": [{"type": "MoGePointMapToMesh"}],
                                            "outputs": [{"type": "MESH"}],
                                        },
                                        {
                                            "id": "prep",
                                            "nodes": [],
                                            "outputs": [{"type": "IMAGE"}],
                                        },
                                    ]
                                },
                            }
                        ).encode()
                    },
                    [native_node("MoGePointMapToMesh")],
                )["recipes"][0]
                self.assertEqual(row["scope"], "3d")
                self.assertEqual(
                    row["declared_output_types"], [port["type"] for port in root_exports]
                )

    def test_graph_output_media_precedes_filename_or_shared_model_domain(self):
        documents = {
            "video_bernini_r_image_editing.json": json.dumps(
                {"nodes": [{"id": 1, "type": "SaveImage"}]}
            ).encode(),
            "template_ingredients.json": json.dumps(
                {"nodes": [{"id": 2, "type": "SaveVideo"}]}
            ).encode(),
            "Image to Gaussian Splat.json": json.dumps(
                {"nodes": [{"id": 3, "type": "VAEDecodeTripoSplat"}]}
            ).encode(),
            "Text to Music.json": json.dumps(
                {"nodes": [{"id": 4, "type": "VAEDecodeAudio"}]}
            ).encode(),
        }
        recipes = recipe_inventory(documents, [], official=True)["recipes"]
        scopes = {row["id"]: row["scope"] for row in recipes}
        self.assertEqual(scopes["video_bernini_r_image_editing.json"], "still-image-or-utility")
        self.assertEqual(scopes["template_ingredients.json"], "video")
        self.assertEqual(scopes["Image to Gaussian Splat.json"], "3d")
        self.assertEqual(scopes["Text to Music.json"], "audio")

    def test_nested_graph_calls_are_not_native_nodes_or_duplicate_model_slots(self):
        graph = {
            "id": "child",
            "nodes": [
                {
                    "id": 3,
                    "type": "LoadImage",
                    "properties": {
                        "models": [
                            {
                                "name": "encoder.sft",
                                "directory": "text_encoders",
                                "url": "https://example.invalid/encoder.sft",
                            }
                        ]
                    },
                }
            ],
        }
        document = {
            "id": "parent",
            "nodes": [{"id": 1, "type": "child"}],
            "definitions": {"subgraphs": [graph]},
        }
        inventory = recipe_inventory(
            {"fixture.json": json.dumps(document).encode()}, [native_node()]
        )
        self.assertEqual(inventory["counts"]["nested_graphs"], 1)
        self.assertEqual(inventory["counts"]["subgraph_wrappers"], 1)
        self.assertEqual(inventory["counts"]["leaf_node_instances"], 1)
        self.assertEqual(inventory["counts"]["node_ids"], 1)
        self.assertEqual(inventory["counts"]["model_slots"], 1)
        self.assertEqual(inventory["model_slots"][0]["node_instance"], 3)
        self.assertEqual(inventory["model_slots"][0]["graph"], "root/subgraphs/0")
        self.assertEqual(inventory["model_slots"][0]["evidence"]["real_runtime"], "pending")

    def test_graph_indices_are_separate_from_recipes_and_foreign_nodes_stay_reviewable(self):
        raw = json.dumps({"nodes": [{"id": 2, "type": "FrontendHelper"}]}).encode()
        result = recipe_inventory({"index.json": b"[]", "fixture.json": raw}, [])
        self.assertEqual(result["counts"]["recipes"], 1)
        self.assertEqual(result["node_ids"][0]["disposition"], "external-or-frontend-node-review")
        self.assertEqual(result["recipes"][0]["sha256"], hashlib.sha256(raw).hexdigest())

    def test_packet_assignments_do_not_claim_an_absent_operation_is_unsupported(self):
        for name, packet in (
            ("SaveImageAdvanced", "E"),
            ("TextEncodeQwenImageEditPlus", "R"),
            ("AnimaLLLiteApply", "P"),
            ("EmptyQwenImageLayeredLatentImage", "G"),
            ("MoGeInference", "U"),
        ):
            row = node_assignment(native_node(name))
            self.assertEqual(row["packet"], packet)
            self.assertEqual(row["disposition"], "planned-operation-review")
        for domain in ("audio", "video", "3d", "training"):
            self.assertEqual(
                node_assignment(native_node(native_domain=domain))["disposition"],
                "scoped-exclusion",
            )
        self.assertEqual(
            node_assignment(native_node(RF_source_references=[{"path": "caller.py"}]))[
                "disposition"
            ],
            "source-consumer-needs-acceptance",
        )

    def test_specialized_auxiliary_configs_are_not_missing_ordinary_generators(self):
        row = config_assignment({"class": "LotusD", "RF_runtime_classes": []})
        self.assertEqual(row["packet"], "U")
        self.assertEqual(row["disposition"], "planned-operation-review")
        self.assertEqual(
            config_assignment(
                {"class": "CosmosI2VPredict2", "RF_runtime_classes": ["CosmosPredict2"]}
            )["disposition"],
            "scoped-exclusion",
        )
        self.assertEqual(
            config_assignment(
                {"class": "CosmosT2IPredict2", "RF_runtime_classes": ["CosmosPredict2"]}
            )["disposition"],
            "source-consumer-needs-acceptance",
        )

    def test_new_native_role_requires_review_instead_of_disappearing_from_ledger(self):
        report = {"native_folder_roles": [{"role": "future_role"}]}
        with self.assertRaisesRegex(ValueError, "no reviewed packet contract"):
            reconcile_inventory(report, {}, {}, {})

    def test_missing_template_installation_is_an_explicit_separate_blocker(self):
        with tempfile.TemporaryDirectory() as folder:
            documents, identity = read_official_templates(Path(folder) / "missing/templates")
        self.assertEqual(documents, {})
        self.assertEqual(identity["disposition"], "blocked")

    def test_template_manifest_mismatch_preserves_observed_bytes_and_exact_disposition(self):
        with tempfile.TemporaryDirectory() as folder:
            packages = Path(folder)
            root = packages / "comfyui_workflow_templates_json/templates"
            root.mkdir(parents=True)
            manifest = packages / "comfyui_workflow_templates_core/manifest.json"
            manifest.parent.mkdir()
            recipe = b'{"nodes": []}'
            (root / "recipe.json").write_bytes(recipe)
            (root / "index.json").write_bytes(b"[]")
            manifest.write_text(
                json.dumps(
                    {
                        "templates": [
                            {
                                "assets": [
                                    {
                                        "filename": "recipe.json",
                                        "sha256": hashlib.sha256(recipe).hexdigest(),
                                    },
                                    {"filename": "index.json", "sha256": "outdated-index"},
                                ]
                            }
                        ]
                    }
                )
            )
            documents, identity = read_official_templates(root)
            self.assertEqual(documents["recipe.json"], recipe)
            self.assertFalse(identity["all_json_assets_match_manifest"])
            self.assertTrue(identity["recipe_assets_match_manifest"])
            self.assertEqual(identity["manifest_mismatches"][0]["filename"], "index.json")
            (root / "unexpected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "JSON set differs"):
                read_official_templates(root)

    def test_template_manifest_paths_cannot_escape_the_json_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            packages = Path(folder)
            root = packages / "comfyui_workflow_templates_json/templates"
            root.mkdir(parents=True)
            manifest = packages / "comfyui_workflow_templates_core/manifest.json"
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "templates": [
                            {"assets": [{"filename": "../outside.json", "sha256": "digest"}]}
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "must be a basename"):
                read_official_templates(root)


if __name__ == "__main__":
    unittest.main()
