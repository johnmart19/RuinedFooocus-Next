"""Maintainer coverage snapshots inspect syntax; they never import native code."""

import copy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.audit_native_model_coverage import (
    compare_baseline,
    derive_node_inventory,
    derive_sources,
    read_checkout,
    read_pin_sources,
)


def fixture_sources():
    return {
        "native/supported_models_base.py": """
class BASE:
    noise_aug_config = None
    def get_model(self):
        return model_base.BaseModel(self)
""",
        "native/supported_models.py": """
import torch
raise RuntimeError('Native source must never execute')
class SDXL(BASE):
    unet_config = {'image_model': 'sdxl'}
    def get_model(self):
        return model_base.SDXL(self)
    def clip_target(self):
        return ClipTarget(NativeTokenizer, NativeTE)
class Variant(SDXL):
    unet_config = {'image_model': 'sdxl', 'depth': 4}
class Unknown(BASE):
    unet_config = {'image_model': 'audio_or_new_task'}
models = [SDXL, Variant, Unknown]
""",
        "native/sd.py": """
class CLIPType:
    STABLE_DIFFUSION = 1
    FUTURE_TASK = 2
def load_clip(path):
    return native_loader(path)
def load_checkpoint_guess_config(path):
    return native_loader(path)
def load_diffusion_model(path):
    return native_loader(path)
""",
        "native/utils.py": """
def load_torch_file(path):
    if path.endswith(('.safetensors', '.sft')):
        return read_safe(path)
    return torch.load(path, weights_only=True)
""",
        "rf/image_components.py": "_COMPONENT_KEYS = {'SDXL': (('clip_l', 'clip_g'), 'vae_sdxl')}\nALIASES = {'Variant': 'SDXL'}",
        "rf/image_pipeline.py": """
class pipeline:
    known_model_info = {'SDXL': {'clip_type': comfy.sd.CLIPType.STABLE_DIFFUSION}}
    def load_base_model(self, path):
        return native_loader(path)
""",
        "rf/model_architecture.py": "def detect():\n    return {'SD15': 'SD 1.5'}.get(label, label)",
        "rf/model_formats.py": "SAFETENSORS_SUFFIXES = ('.safetensors', '.sft')",
        "rf/model_handler.py": "class Models:\n    def __init__(self):\n        self.EXTENSIONS = [*SAFETENSORS_SUFFIXES, '.gguf']",
    }


def node_sources():
    return {
        "native/nodes.py": """
raise RuntimeError('This source must not execute')
class Helper:
    CATEGORY = 'utilities/primitive'
NODE_CLASS_MAPPINGS = {'GraphHelper': Helper}
def init_builtin_extra_nodes():
    extras_files = ['nodes_hooks.py', 'nodes_qwen.py']
    for node_file in extras_files:
        load_custom_node(node_file)
""",
        "native/comfy_extras/nodes_hooks.py": """
class Hook:
    NodeId = 'RegisteredHook'
    CATEGORY = 'advanced/hooks'
node_list = [Hook]
NODE_CLASS_MAPPINGS = {}
for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
""",
        "native/comfy_extras/nodes_qwen.py": """
class SchemaBase:
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id=cls.native_id, category='model/conditioning/qwen image')
class TextEncodeQwenImage21(SchemaBase):
    native_id: str = 'TextEncodeQwenImage21'
class VideoFixture(TextEncodeQwenImage21):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = 'VideoFixture'
        schema.category = 'model/latent/hunyuan video'
        return schema
class Unregistered(TextEncodeQwenImage21):
    native_id = 'Ghost'
class Extension(ComfyExtension):
    async def get_node_list(self):
        return [TextEncodeQwenImage21, VideoFixture]
async def comfy_entrypoint():
    return Extension()
""",
        "native/comfy_extras/nodes_unlisted.py": """
class Unlisted:
    CATEGORY = 'image'
NODE_CLASS_MAPPINGS = {'Unlisted': Unlisted}
""",
        "native/folder_paths.py": """
supported_pt_extensions: set[str] = {'.ckpt', '.safetensors'}
folder_names_and_paths['checkpoints'] = (['ignored-path'], supported_pt_extensions)
folder_names_and_paths['diffusers'] = (['ignored-path'], ['folder'])
""",
        "gguf/__init__.py": "from .nodes import NODE_CLASS_MAPPINGS",
        "gguf/nodes.py": """
class VAELoaderGGUF:
    CATEGORY = 'bootleg'
NODE_CLASS_MAPPINGS = {'VAELoaderGGUF': VAELoaderGGUF}
""",
        "rf-callers/features/image_editing/qwen_image21_conditioning.py": """
from comfy_extras.nodes_qwen import TextEncodeQwenImage21 as Encoder, VideoFixture
def text_to_image_conditioning():
    return Encoder.execute()
class QwenImage21References:
    def conditioning(self):
        return Encoder.execute()
""",
        "rf-callers/features/generation/image_pipeline.py": """
class pipeline:
    def process(self):
        return self.textencode_pair()
    def textencode_pair(self):
        return None
""",
    }


class NativeCoverageAuditTests(unittest.TestCase):
    def test_native_inheritance_aliases_and_unknown_tasks_are_not_executed_or_promoted(self):
        result = derive_sources(fixture_sources())
        records = {item["class"]: item for item in result["native_config_classes"]}
        self.assertEqual(records["Variant"]["runtime_factories"], ["SDXL"])
        self.assertEqual(records["Variant"]["factory_owner"], "SDXL")
        self.assertEqual(records["Variant"]["RF_component_canonical"], "SDXL")
        self.assertEqual(records["Unknown"]["route_status"], "native-task-route-needs-review")
        self.assertEqual(records["Unknown"]["RF_runtime_classes"], [])
        self.assertEqual(
            result["loader_contract"]["RF_discovery_extensions"], [".gguf", ".safetensors", ".sft"]
        )
        reader = result["loader_contract"]["native_weight_reader"]
        self.assertEqual(reader["explicit_suffix_branches"], [".safetensors", ".sft"])
        self.assertIn("torch.load", reader["calls"])

    def test_pin_drift_and_possible_renames_remain_human_review(self):
        before = derive_sources(fixture_sources())
        self.assertEqual(
            compare_baseline(json.loads(json.dumps(before)), before)["RF_changed_component_roles"],
            [],
        )
        after = copy.deepcopy(before)
        renamed = after["native_config_classes"][1]
        renamed["class"] = "NewVariantLabel"
        after["native_clip_types"]["NEW_ENCODER"] = 3
        drift = compare_baseline(before, after)
        self.assertEqual(drift["possible_renames_require_review"], [["Variant", "NewVariantLabel"]])
        self.assertEqual(drift["added_CLIP_types"], ["NEW_ENCODER"])
        self.assertEqual(before["native_config_classes"][1]["class"], "Variant")
        after["RF_runtime_configurations"].append("NewExplicitRFRoute")
        self.assertEqual(
            compare_baseline(before, after)["RF_added_runtime_routes"], ["NewExplicitRFRoute"]
        )
        after["native_config_classes"][0]["contract_sha256"] = "changed"
        self.assertEqual(compare_baseline(before, after)["changed_native_contracts"], ["SDXL"])

    def test_wrong_native_pin_refuses_before_reading_or_importing_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ruinedfooocus/app").mkdir(parents=True)
            (root / "ruinedfooocus/app/bootstrap.py").write_text(
                "git_repos = [{'name': 'ComfyUI', 'hash': 'expected'}]"
            )
            with patch(
                "tools.audit_native_model_coverage.subprocess.check_output", return_value="other\n"
            ):
                with self.assertRaisesRegex(ValueError, "differs from declared pin"):
                    read_checkout(root, root / "missing-native")

    def test_gguf_pin_is_checked_before_reading_native_registries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ruinedfooocus/app").mkdir(parents=True)
            (root / "ruinedfooocus/app/bootstrap.py").write_text(
                "git_repos = ["
                "{'name': 'ComfyUI', 'hash': 'core-pin'},"
                "{'name': 'molbal/ComfyUI-GGUF', 'path': 'gguf', 'hash': 'gguf-pin'}]"
            )
            with patch(
                "tools.audit_native_model_coverage.subprocess.check_output",
                side_effect=["core-pin\n", "wrong-gguf\n"],
            ) as git:
                with self.assertRaisesRegex(ValueError, "ComfyUI-GGUF checkout .* differs"):
                    read_checkout(root, root / "ComfyUI")
            self.assertEqual(git.call_count, 2)
            self.assertEqual(git.call_args.args[0][2], str(root / "repositories/gguf"))

    def test_native_source_bytes_are_verified_against_blob_even_if_git_diff_is_clean(self):
        pinned = b"print('source')\n"
        oid = hashlib.sha1(f"blob {len(pinned)}\0".encode("ascii") + pinned).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "nodes.py"
            with (
                patch(
                    "tools.audit_native_model_coverage.subprocess.check_output",
                    return_value=f"100644 blob {oid}\tnodes.py\0".encode(),
                ),
                patch("tools.audit_native_model_coverage.subprocess.run") as git_diff,
            ):
                git_diff.return_value.returncode = 0
                path.write_bytes(pinned.replace(b"\n", b"\r\n"))
                self.assertEqual(
                    read_pin_sources(root, "GGUF", "pin", ["nodes.py"])["nodes.py"],
                    path.read_bytes(),
                )
                path.write_bytes(pinned + b"raise RuntimeError()\n")
                with self.assertRaisesRegex(ValueError, "nodes.py differs from its pinned blob"):
                    read_pin_sources(root, "GGUF", "pin", ["nodes.py"])

    def test_static_registration_is_not_a_count_of_classes_or_app_workflows(self):
        report = derive_node_inventory(node_sources())
        nodes = {node["id"]: node for node in report["native_nodes"]}
        self.assertEqual(
            set(nodes),
            {
                "GraphHelper",
                "RegisteredHook",
                "TextEncodeQwenImage21",
                "VideoFixture",
                "VAELoaderGGUF",
            },
        )
        self.assertEqual(nodes["GraphHelper"]["disposition"], "internal-primitive")
        self.assertEqual(nodes["RegisteredHook"]["registration"], "v1")
        qwen = nodes["TextEncodeQwenImage21"]
        self.assertEqual(qwen["registration"], "v3")
        self.assertEqual(qwen["native_category"], "model/conditioning/qwen image")
        self.assertEqual(qwen["disposition"], "specialized-consumer")
        self.assertEqual(qwen["source"]["source_key"], "native/comfy_extras/nodes_qwen.py")
        self.assertEqual(len(qwen["RF_source_references"]), 1)
        self.assertEqual(nodes["VideoFixture"]["native_domain"], "video")
        self.assertEqual(nodes["VideoFixture"]["RF_source_references"], [])
        self.assertEqual(nodes["VAELoaderGGUF"]["disposition"], "excluded-from-image-audit")
        ops = {row["operation"]: row for row in report["operation_families"]}
        self.assertEqual(
            ops["Qwen Image 2.1 text conditioning"]["disposition"], "source-route-present"
        )
        self.assertEqual(
            ops["Qwen Image 2.1 reference edit"]["disposition"], "source-route-present"
        )
        self.assertEqual(ops["Qwen Image 2.1 reference edit"]["unresolved_anchors"], [])
        self.assertEqual(ops["Joy Image Edit"]["disposition"], "needs-review")
        self.assertEqual(
            ops["Joy Image Edit"]["unresolved_anchors"],
            [
                "TextEncodeJoyImageEdit",
                "features/generation/image_pipeline.py:joy_reference_conditioning",
            ],
        )
        self.assertEqual(report["native_folder_roles"][1]["native_discovery_tokens"], ["folder"])

    def test_unknown_registration_forms_and_collisions_require_review(self):
        sources = node_sources()
        sources["native/nodes.py"] += "    extras_files.append('nodes_unlisted.py')\n"
        with self.assertRaisesRegex(ValueError, "Built-in node loader paths need review"):
            derive_node_inventory(sources)
        sources = node_sources()
        sources["native/comfy_extras/nodes_hooks.py"] += (
            "\nNODE_CLASS_MAPPINGS.update(load_external())\n"
        )
        with self.assertRaisesRegex(ValueError, "dynamic V1 registration needs review"):
            derive_node_inventory(sources)
        sources = node_sources()
        sources["native/comfy_extras/nodes_qwen.py"] = sources[
            "native/comfy_extras/nodes_qwen.py"
        ].replace("return [TextEncodeQwenImage21, VideoFixture]", "return dynamically_register()")
        with self.assertRaisesRegex(ValueError, "literal list of local node classes"):
            derive_node_inventory(sources)
        sources = node_sources()
        sources["gguf/nodes.py"] = sources["gguf/nodes.py"].replace(
            "'VAELoaderGGUF':", "'GraphHelper':"
        )
        with self.assertRaisesRegex(ValueError, "duplicate node ID 'GraphHelper'"):
            derive_node_inventory(sources)

    def test_schema_one_baseline_retains_config_drift_and_reports_node_baseline_boundary(self):
        previous = derive_sources(fixture_sources())
        current = previous | derive_node_inventory(node_sources())
        drift = compare_baseline(previous, current)
        self.assertEqual(drift["added_config_classes"], [])
        self.assertTrue(drift["inventory_baseline_needs_review"])
        changed = copy.deepcopy(current)
        changed["native_nodes"][0]["disposition"] = "needs-review"
        changed["native_nodes"] = changed["native_nodes"][:-1]
        changed["operation_families"][0]["disposition"] = "missing-app-route"
        drift = compare_baseline(current, changed)
        self.assertEqual(drift["changed_native_node_rows"], [current["native_nodes"][0]["id"]])
        self.assertEqual(drift["removed_native_node_ids"], [current["native_nodes"][-1]["id"]])
        self.assertEqual(drift["changed_operation_dispositions"], ["text-to-image"])

    def test_finite_recipe_drift_keeps_a_separate_reviewable_denominator(self):
        previous = derive_sources(fixture_sources())
        current = copy.deepcopy(previous)
        current["finite_inventory"] = {
            "schema_version": 1,
            "blueprints": {"recipes": [{"id": "recipe.json", "sha256": "original"}]},
            "official_templates": {"recipes": []},
        }
        self.assertTrue(
            compare_baseline(previous, current)["finite_inventory_baseline_needs_review"]
        )
        before = copy.deepcopy(current)
        current["finite_inventory"]["blueprints"]["recipes"][0]["sha256"] = "changed"
        current["finite_inventory"]["official_templates"]["recipes"].append(
            {"id": "new.json", "sha256": "new"}
        )
        drift = compare_baseline(before, current)
        self.assertEqual(drift["changed_blueprints"], ["recipe.json"])
        self.assertEqual(drift["added_official_templates"], ["new.json"])

    def test_gguf_lora_operation_distinguishes_native_registration_and_connected_rf_caller(self):
        sources = node_sources()
        sources["gguf/nodes.py"] += """
class GGUFLoraImport:
    CATEGORY = 'bootleg'
"""
        sources["gguf/nodes.py"] = (
            sources["gguf/nodes.py"].replace(
                "NODE_CLASS_MAPPINGS = {'VAELoaderGGUF': VAELoaderGGUF}", ""
            )
            + "NODE_CLASS_MAPPINGS = {'VAELoaderGGUF': VAELoaderGGUF, 'GGUFLoraImport': GGUFLoraImport}\n"
        )
        sources["gguf/lora.py"] = "def load_gguf_lora(): pass"
        caller = "rf-callers/features/generation/image_pipeline.py"
        sources[caller] = "class pipeline:\n    def load_loras(self): pass"

        def disposition():
            return next(
                row["disposition"]
                for row in derive_node_inventory(sources)["operation_families"]
                if row["operation"] == "GGUF LoRA"
            )

        self.assertEqual(disposition(), "missing-app-route")
        sources["rf-callers/backends/comfy/gguf_lora.py"] = """
def apply_gguf_lora():
    from molbal_comfyui_gguf.lora import load_gguf_lora
    return load_gguf_lora()
"""
        self.assertEqual(disposition(), "needs-review")
        sources[caller] = """
class pipeline:
    def load_loras(self):
        from ruinedfooocus.backends.comfy.gguf_lora import apply_gguf_lora as apply
        return apply()
"""
        self.assertEqual(disposition(), "source-route-present")

    def test_gguf_vae_ordinary_exclusion_tracks_both_pinned_video_guards(self):
        sources = node_sources()
        sources["gguf/nodes.py"] = sources["gguf/nodes.py"].replace(
            "    CATEGORY = 'bootleg'",
            """    CATEGORY = 'bootleg'
    def load_vae(self, extra):
        if extra['arch_str'] != 'minimax_h3_vae':
            raise ValueError('unsupported architecture')
        if extra.get('gguf_quant_mode') != 'int8_convrot':
            raise ValueError('unsupported quantization')""",
        )
        sources["gguf/loader.py"] = "def gguf_sd_loader(): pass"
        sources["rf-callers/features/models/image_components.py"] = "def inventory(): pass"
        sources["rf-callers/features/generation/image_pipeline.py"] = """
class pipeline:
    def load_base_model(self, path):
        from ruinedfooocus.core.model_formats import require_image_vae_format
        require_image_vae_format(path)
"""

        def disposition():
            return next(
                row["disposition"]
                for row in derive_node_inventory(sources)["operation_families"]
                if row["operation"] == "GGUF VAE"
            )

        self.assertEqual(disposition(), "unsupported")
        sources["gguf/nodes.py"] = sources["gguf/nodes.py"].replace("int8_convrot", "other_quant")
        self.assertEqual(disposition(), "needs-review")


if __name__ == "__main__":
    unittest.main()
