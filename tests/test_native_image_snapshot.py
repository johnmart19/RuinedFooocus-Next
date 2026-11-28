"""Native facts are sourced data; runtime never imports/scans native code."""

import ast
import builtins
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models import native_snapshot as runtime
from tools.update_native_snapshot import (
    connected_graph_facts,
    derive_snapshot_facts,
    node_schema,
    recipe_facts,
    sampler_choice_facts,
    write_candidate,
)
from tools.audit_native_model_coverage import module_registrations


URL = "https://huggingface.co/Example/Image/resolve/main/model.sft"


def source_fixture():
    return {
        "native/samplers.py": """
raise RuntimeError('native samplers must not execute')
KSAMPLER_NAMES = ['euler', 'heun']
SAMPLER_NAMES = KSAMPLER_NAMES + ['ddim']
SCHEDULER_HANDLERS = {'normal': SchedulerHandler(probe_device()), 'karras': SchedulerHandler(other_callback)}
SCHEDULER_NAMES = list(SCHEDULER_HANDLERS)
class KSampler:
    SAMPLERS = SAMPLER_NAMES
    SCHEDULERS = SCHEDULER_NAMES
""",
        "native/supported_models_base.py": "class BASE:\n    latent_format = latent_formats.Image\n    vae_key_prefix = ['vae.']\n    text_encoder_key_prefix = ['text.']\n    clip_vision_prefix = None\n",
        "native/supported_models.py": "raise RuntimeError('do not import source')\nclass Image(BASE):\n    pass\n",
        "native/latent_formats.py": "class Image:\n    latent_channels = 4\n    latent_dimensions = 2\n    spacial_downscale_ratio = 8\n    temporal_downscale_ratio = 1\n",
        "native/nodes.py": """
raise RuntimeError('do not import nodes')
class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'unet_name': (['model.sft', 'alternative.sft'],)}}
    RETURN_TYPES = ('MODEL',)
class Empty:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'width': ('INT', {'default': 512}), 'height': ('INT', {'default': 512})}}
    RETURN_TYPES = ('LATENT',)
class Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'latent_image': ('LATENT',), 'seed': ('INT', {'control_after_generate': True}), 'steps': ('INT', {}), 'cfg': ('FLOAT', {}), 'sampler_name': (dynamic.samplers, {}), 'scheduler': (dynamic.schedulers, {}), 'denoise': ('FLOAT', {})}}
    RETURN_TYPES = ('LATENT',)
class Decode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',)}}
    RETURN_TYPES = ('IMAGE',)
class Save:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',)}}
    RETURN_TYPES = ()
NODE_CLASS_MAPPINGS = {'UNETLoader': Loader, 'EmptyLatentImage': Empty, 'KSampler': Sampler, 'VAEDecode': Decode, 'SaveImage': Save}
""",
    }


def schemas_fixture():
    mode, classes, rows = module_registrations(
        ast.parse(source_fixture()["native/nodes.py"]), "fixture"
    )
    return {key: node_schema(classes, node.id, mode) for key, node, _ in rows}


def graph_fixture():
    model = {"name": "model.sft", "directory": "diffusion_models", "url": URL}
    alternative = {
        **model,
        "name": "alternative.sft",
        "url": URL.replace("model.sft", "alternative.sft"),
    }
    return {
        "nodes": [
            {
                "id": 1,
                "type": "UNETLoader",
                "widgets_values": ["model.sft"],
                "properties": {"models": [model, alternative]},
            },
            {"id": 2, "type": "EmptyLatentImage", "widgets_values": [768, 512]},
            {
                "id": 3,
                "type": "KSampler",
                "inputs": [{"name": "model", "link": 1}, {"name": "latent_image", "link": 2}],
                "widgets_values": [0, "fixed", 28, 4.5, "euler", "normal", 1.0],
            },
            {"id": 4, "type": "VAEDecode", "inputs": [{"name": "samples", "link": 3}]},
            {"id": 5, "type": "SaveImage", "inputs": [{"name": "images", "link": 4}]},
            {
                "id": 6,
                "type": "UNETLoader",
                "widgets_values": ["unused.sft"],
                "properties": {
                    "models": [
                        {
                            **model,
                            "name": "unused.sft",
                            "url": URL.replace("model.sft", "unused.sft"),
                        }
                    ]
                },
                "mode": 0,
            },
        ],
        "links": [
            [1, 1, 0, 3, 0, "MODEL"],
            [2, 2, 0, 3, 1, "LATENT"],
            [3, 3, 0, 4, 0, "LATENT"],
            [4, 4, 0, 5, 0, "IMAGE"],
        ],
    }


def facts_fixture(documents=None, sources=None):
    report = {
        "native_nodes": [
            {"id": key, "native_domain": "image", "role": "sampling", "RF_source_references": []}
            for key in schemas_fixture()
        ],
        "native_config_classes": [
            {
                "class": "Image",
                "unet_match": {"kind": "image"},
                "runtime_factories": ["Image"],
                "contract_sha256": "config-contract",
                "line": 2,
            }
        ],
    }
    from tools.update_native_snapshot import _url_identity

    annotations = {
        _url_identity(URL): [
            {"family_id": "Image", "native_class": "Image", "behavioral_variant": None}
        ]
    }
    catalogue = [
        {
            "id": "Image",
            "component_family": "Image",
            "native_family": "Image alias",
            "component_roles": {"text": ["clip_image"], "vae": "vae_image"},
        }
    ]
    return derive_snapshot_facts(
        report,
        sources or source_fixture(),
        {},
        documents or {"image.json": json.dumps(graph_fixture()).encode()},
        catalogue,
        annotations,
    )


def document_fixture(facts=None):
    facts = copy.deepcopy(facts or facts_fixture())
    source = {
        "native_pin": runtime.ACCEPTED_COMFY_PIN,
        "gguf_pin": runtime.ACCEPTED_GGUF_PIN,
        "template_identity": {
            "manifest_sha256": "template-manifest",
            "recipe_assets_match_manifest": True,
        },
        "sources": {"native/source.py": "source-sha"},
    }
    manifest = {
        "schema_version": runtime.SCHEMA_VERSION,
        "parser_version": runtime.PARSER_VERSION,
        "source_manifest": source,
        "source_manifest_sha256": runtime.content_digest(source),
        "snapshot_sha256": runtime.content_digest(facts),
        "required_contracts": {"families": list(facts["families"]), "schemas": ["KSampler"]},
    }
    manifest["snapshot_id"] = runtime.content_digest(manifest)
    return {"manifest": manifest, "facts": facts}


class NativeSnapshotExtractionTests(unittest.TestCase):
    def test_public_sampler_aliases_concatenations_and_scheduler_keys_are_static(self):
        choices = sampler_choice_facts(source_fixture()["native/samplers.py"])
        self.assertEqual(choices["samplers"], ["euler", "heun", "ddim"])
        self.assertEqual(choices["schedulers"], ["normal", "karras"])
        self.assertEqual(
            choices["source"]["bindings"],
            {"samplers": "KSampler.SAMPLERS", "schedulers": "KSampler.SCHEDULERS"},
        )
        before = choices["source"]["contract_sha256"]
        changed = source_fixture()["native/samplers.py"].replace(
            "['ddim']", "['ddim', 'future_sampler']"
        )
        self.assertNotEqual(before, sampler_choice_facts(changed)["source"]["contract_sha256"])

    def test_dynamic_or_mutated_sampler_declarations_require_source_review(self):
        source = source_fixture()["native/samplers.py"]
        changes = (
            source.replace(
                "SAMPLER_NAMES = KSAMPLER_NAMES + ['ddim']", "SAMPLER_NAMES = runtime_registry()"
            ),
            source + "\nSAMPLER_NAMES.append('unreviewed')\n",
            source + "\nSCHEDULER_HANDLERS['dynamic'] = callback\n",
            source + "\nKSampler.SAMPLERS = runtime_registry()\n",
            source + "\nKSampler.SAMPLERS.append('unreviewed')\n",
            source + "\nKSampler.SCHEDULERS[0] = 'unreviewed'\n",
        )
        for changed in changes:
            with self.subTest(source=changed[-70:]), self.assertRaises(ValueError):
                sampler_choice_facts(changed)

    def test_deterministic_source_projection_and_examples_are_separate_from_advice(self):
        before, after = facts_fixture(), facts_fixture()
        self.assertEqual(before, after)
        family = before["families"]["Image"]
        self.assertEqual(family["configurations"][0]["latent"]["latent_channels"], 4)
        example = family["example_recipe"]
        self.assertEqual(example["source"]["kind"], "native-workflow-example")
        self.assertEqual(example["operation"], "generate")
        self.assertEqual(example["default_size"], [768, 512])
        self.assertEqual(
            example["sampling"],
            {
                "custom_steps": 28,
                "cfg": 4.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        )
        self.assertNotIn("publisher_recommendation", family)
        self.assertEqual(before["aliases"]["Image alias"], "Image")

    def test_active_models_exclude_disconnected_and_same_node_alternatives(self):
        facts = connected_graph_facts(graph_fixture(), schemas_fixture())
        self.assertEqual([row["name"] for row in facts["required_models"]], ["model.sft"])
        self.assertEqual(
            {row["name"] for row in facts["optional_models"]}, {"alternative.sft", "unused.sft"}
        )
        self.assertEqual(facts["unresolved"], [])

    def test_conflicting_connected_examples_do_not_choose_by_sort_order(self):
        second = graph_fixture()
        second["nodes"][2]["widgets_values"][2] = 40
        facts = facts_fixture(
            {
                "first.json": json.dumps(graph_fixture()).encode(),
                "second.json": json.dumps(second).encode(),
            }
        )
        self.assertIsNone(facts["families"]["Image"]["example_recipe"])
        self.assertIn(
            "conflicting-connected-generation-examples", facts["families"]["Image"]["unresolved"]
        )

    def test_link_type_change_is_visible_and_cannot_become_an_example(self):
        graph = graph_fixture()
        graph["links"][2][1] = 1  # MODEL reaches the decoder's LATENT input.
        facts = facts_fixture({"changed.json": json.dumps(graph).encode()})
        example = facts["families"]["Image"]["examples"][0]
        self.assertIn("native-link-type-mismatch", example["unresolved"])
        self.assertIsNone(facts["families"]["Image"]["example_recipe"])

    def test_unknown_loader_and_muted_active_branch_stay_unresolved(self):
        graph = graph_fixture()
        graph["nodes"][0]["type"] = "ForeignLoader"
        facts = connected_graph_facts(graph, schemas_fixture())
        self.assertIn("unregistered-native-node:ForeignLoader", facts["unresolved"])
        graph["nodes"][0]["mode"] = 4
        facts = connected_graph_facts(graph, schemas_fixture())
        self.assertIn("muted-or-bypassed-active-node", facts["unresolved"])
        self.assertEqual(facts["required_models"], [])

    def test_selected_subgraph_resolves_output_to_callers_input(self):
        graph = graph_fixture()
        graph["nodes"][3] = {
            "id": 4,
            "type": "decode-graph",
            "inputs": [{"name": "samples", "link": 3}],
        }
        graph["definitions"] = {
            "subgraphs": [
                {
                    "id": "decode-graph",
                    "inputNode": {"id": -10},
                    "outputNode": {"id": -20},
                    "inputs": [{"name": "samples", "type": "LATENT", "linkIds": [10]}],
                    "outputs": [{"name": "images", "type": "IMAGE", "linkIds": [11]}],
                    "nodes": [
                        {"id": 1, "type": "VAEDecode", "inputs": [{"name": "samples", "link": 10}]}
                    ],
                    "links": [
                        {
                            "id": 10,
                            "origin_id": -10,
                            "origin_slot": 0,
                            "target_id": 1,
                            "target_slot": 0,
                        },
                        {
                            "id": 11,
                            "origin_id": 1,
                            "origin_slot": 0,
                            "target_id": -20,
                            "target_slot": 0,
                        },
                    ],
                }
            ]
        }
        facts = connected_graph_facts(graph, schemas_fixture())
        self.assertEqual(facts["unresolved"], [])
        self.assertIn("VAEDecode", [row["node_id"] for row in facts["active_nodes"]])
        self.assertEqual([row["name"] for row in facts["required_models"]], ["model.sft"])
        graph["nodes"][3]["widgets_values"] = [1024]
        self.assertIn(
            "subgraph-widget-overrides-unresolved",
            connected_graph_facts(graph, schemas_fixture())["unresolved"],
        )

    def test_subgraph_owned_sink_and_default_loader_widgets_are_connected(self):
        child = graph_fixture()
        child.update(id="whole-image", inputs=[{"name": "unet_name", "linkIds": [9]}])
        child["nodes"][0]["inputs"] = [
            {"name": "unet_name", "widget": {"name": "unet_name"}, "link": 9}
        ]
        child["links"].append([9, -10, 0, 1, 0, "COMBO"])
        graph = {
            "nodes": [{"id": 10, "type": "whole-image", "widgets_values": []}],
            "definitions": {"subgraphs": [child]},
        }
        facts = connected_graph_facts(graph, schemas_fixture())
        self.assertEqual(facts["unresolved"], [])
        self.assertEqual([row["name"] for row in facts["required_models"]], ["model.sft"])
        graph["nodes"][0].update(
            widgets_values=["alternative.sft", 1024],
            properties={"proxyWidgets": [["1", "unet_name"], ["2", "width"]]},
        )
        facts = connected_graph_facts(graph, schemas_fixture())
        self.assertEqual(facts["unresolved"], [])
        self.assertEqual([row["name"] for row in facts["required_models"]], ["alternative.sft"])
        self.assertEqual(
            next(
                row["widgets"]["width"]
                for row in facts["active_nodes"]
                if row["node_id"] == "EmptyLatentImage"
            ),
            1024,
        )

    def test_native_literal_primitive_binds_connected_canvas_without_execution(self):
        schemas = schemas_fixture()
        schemas["PrimitiveInt"] = {
            "inputs": [
                {
                    "name": "value",
                    "type": "INT",
                    "widget": True,
                    "required": True,
                    "options": {"control_after_generate": "fixed"},
                }
            ],
            "outputs": ["INT"],
            "literal_passthrough": "value",
            "unresolved": [],
        }
        graph = graph_fixture()
        graph["nodes"][1]["inputs"] = [{"name": "width", "widget": {"name": "width"}, "link": 8}]
        graph["nodes"].append({"id": 7, "type": "PrimitiveInt", "widgets_values": [1024, "fixed"]})
        graph["links"].append([8, 7, 0, 2, 0, "INT"])
        facts = connected_graph_facts(graph, schemas)
        self.assertEqual(facts["unresolved"], [])
        self.assertEqual(
            next(
                row["widgets"]["width"]
                for row in facts["active_nodes"]
                if row["node_id"] == "EmptyLatentImage"
            ),
            1024,
        )

    def test_blueprint_open_image_interface_is_a_finite_source_closure(self):
        child = graph_fixture()
        child["nodes"] = [row for row in child["nodes"] if row["id"] != 5]
        child.update(id="image-return", outputs=[{"type": "IMAGE", "linkIds": [4]}])
        child["links"][-1][3] = -20
        graph = {
            "nodes": [
                {"id": 10, "type": "image-return", "outputs": [{"type": "IMAGE", "links": []}]}
            ],
            "definitions": {"subgraphs": [child]},
        }
        facts = connected_graph_facts(graph, schemas_fixture(), open_image_outputs=True)
        self.assertEqual(facts["unresolved"], [])
        self.assertIn("VAEDecode", [row["node_id"] for row in facts["active_nodes"]])
        self.assertIn(
            "no-connected-root-still-image-output",
            connected_graph_facts(graph, schemas_fixture())["unresolved"],
        )

    def test_literal_native_lazy_switch_does_not_require_the_other_model_branch(self):
        graph, schemas = graph_fixture(), schemas_fixture()
        schemas["Switch"] = {
            "inputs": [
                {
                    "name": "switch",
                    "type": "BOOLEAN",
                    "widget": True,
                    "required": True,
                    "options": {},
                },
                {
                    "name": "on_false",
                    "type": "MODEL",
                    "widget": False,
                    "required": False,
                    "options": {},
                },
                {
                    "name": "on_true",
                    "type": "MODEL",
                    "widget": False,
                    "required": False,
                    "options": {},
                },
            ],
            "outputs": ["MODEL"],
            "static_choice": {"control": "switch", "on_false": "on_false", "on_true": "on_true"},
            "unresolved": [],
        }
        graph["nodes"].append(
            {
                "id": 7,
                "type": "Switch",
                "widgets_values": [False],
                "inputs": [{"name": "on_false", "link": 5}, {"name": "on_true", "link": 6}],
            }
        )
        graph["links"][0][1] = 7
        graph["links"].extend([[5, 1, 0, 7, 0, "MODEL"], [6, 6, 0, 7, 1, "MODEL"]])
        facts = connected_graph_facts(graph, schemas)
        self.assertEqual(facts["unresolved"], [])
        self.assertEqual([row["name"] for row in facts["required_models"]], ["model.sft"])
        self.assertIn("unused.sft", [row["name"] for row in facts["optional_models"]])

    def test_equal_serial_sampler_widgets_remain_an_unresolved_staged_recipe(self):
        graph = graph_fixture()
        second = copy.deepcopy(graph["nodes"][2])
        second.update(
            id=7, inputs=[{"name": "model", "link": 5}, {"name": "latent_image", "link": 6}]
        )
        graph["nodes"].append(second)
        graph["links"][2][1] = 7
        graph["links"].extend([[5, 1, 0, 7, 0, "MODEL"], [6, 3, 0, 7, 1, "LATENT"]])
        facts = facts_fixture({"staged.json": json.dumps(graph).encode()})
        self.assertIsNone(facts["families"]["Image"]["example_recipe"])
        self.assertIn(
            "multiple-sampling-stages", facts["families"]["Image"]["examples"][0]["unresolved"]
        )

    def test_v1_io_constants_and_v3_optional_schema_are_static_facts(self):
        tree = ast.parse("""
class Legacy:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': (IO.IMAGE,), 'strength': (IO.FLOAT, {'default': 1.0})}}
    RETURN_TYPES = (IO.IMAGE,)
class Modern:
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Modern', inputs=[io.Image.Input('image'), io.Mask.Input('mask', optional=True), io.Int.Input('width', default=512)], outputs=[io.Image.Output()])
class Dynamic:
    @classmethod
    def define_schema(cls):
        return custom_schema()
""")
        from tools.audit_native_model_coverage import class_definitions

        classes = class_definitions(tree)
        legacy = node_schema(classes, "Legacy", "v1", {"IMAGE": "IMAGE", "FLOAT": "FLOAT"})
        self.assertEqual(legacy["outputs"], ["IMAGE"])
        self.assertFalse(legacy["inputs"][0]["widget"])
        modern = node_schema(classes, "Modern", "v3")
        self.assertFalse(modern["inputs"][1]["required"])
        self.assertEqual(modern["outputs"], ["IMAGE"])
        self.assertEqual(modern["unresolved"], [])
        self.assertEqual(node_schema(classes, "Dynamic", "v3")["unresolved"], ["dynamic-v3-schema"])

    def test_v3_keyword_ids_enum_controls_multitypes_and_optional_defaults(self):
        from tools.update_native_snapshot import io_type_facts, _widget_values
        from tools.audit_native_model_coverage import class_definitions

        source = """
class ControlAfterGenerate:
    fixed = 'fixed'
@comfytype(io_type='INT')
class Int:
    class Input(WidgetInput):
        pass
@comfytype(io_type='CLIP_VISION_OUTPUT')
class ClipVisionOutput:
    pass
@comfytype(io_type='IMAGE')
class Image:
    pass
@comfytype(io_type='MASK')
class Mask:
    pass
@comfytype(io_type='COMFY_MULTITYPED_V3')
class MultiType:
    pass
"""
        types, constants = io_type_facts(source)
        classes = class_definitions(
            ast.parse("""
class Native:
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Native', inputs=[io.Int.Input(id='seed', control_after_generate=io.ControlAfterGenerate.fixed), io.MultiType.Input('source', types=[io.Image, io.Mask]), io.Int.Input(id='optional', optional=True, default=16)], outputs=[io.ClipVisionOutput.Output()])
""")
        )
        schema = node_schema(classes, "Native", "v3", io_types=types, constants=constants)
        self.assertEqual(schema["outputs"], ["CLIP_VISION_OUTPUT"])
        self.assertEqual(schema["inputs"][1]["type"], "IMAGE,MASK")
        widgets, unresolved = _widget_values({"widgets_values": [42, "fixed"]}, schema)
        self.assertEqual(widgets, {"seed": 42, "optional": 16})
        self.assertEqual(unresolved, [])


class NativeSnapshotRuntimeTests(unittest.TestCase):
    def test_cold_published_choices_need_no_native_import_scan_or_user_cache(self):
        original_import = builtins.__import__

        def no_native_import(name, *args, **kwargs):
            if name == "comfy" or name.startswith("comfy."):
                raise AssertionError("Cold facts must not import a native runtime")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.dict(os.environ, {"RF_DATA_DIR": folder}),
                patch.dict(runtime._loaded, {}, clear=True),
                patch.dict(runtime._last_good, {}, clear=True),
                patch.object(runtime, "_read_snapshot", wraps=runtime._read_snapshot) as read,
                patch("builtins.__import__", side_effect=no_native_import),
            ):
                samplers, schedulers = runtime.native_sampler_choices()
                self.assertIsInstance(samplers, tuple)
                self.assertIsInstance(schedulers, tuple)
                self.assertIn("euler", samplers)
                self.assertIn("normal", schedulers)
                self.assertEqual((samplers, schedulers), runtime.native_sampler_choices())
                read.assert_called_once_with(runtime.DEFAULT_SNAPSHOT)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_enum_addition_requires_review_and_removal_keeps_the_last_good_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            destination, candidate = Path(folder) / "accepted.json", Path(folder) / "candidate.json"
            write_candidate(destination, document_fixture())
            sources = source_fixture()
            sources["native/samplers.py"] = sources["native/samplers.py"].replace(
                "['ddim']", "['ddim', 'future_sampler']"
            )
            write_candidate(candidate, document_fixture(facts_fixture(sources=sources)))
            failed = runtime.refresh_native_snapshot(candidate, cache_path=destination)
            self.assertFalse(failed.promoted)
            self.assertEqual(
                failed.delta["sampling_choices"]["samplers"]["added"], ("future_sampler",)
            )
            accepted = runtime.refresh_native_snapshot(
                candidate, cache_path=destination, reviewed_semantics=True
            )
            self.assertTrue(accepted.promoted)
            before = destination.read_bytes()
            sources["native/samplers.py"] = sources["native/samplers.py"].replace(
                "['euler', 'heun']", "['heun']"
            )
            write_candidate(candidate, document_fixture(facts_fixture(sources=sources)))
            failed = runtime.refresh_native_snapshot(
                candidate, cache_path=destination, reviewed_semantics=True
            )
            self.assertFalse(failed.promoted)
            self.assertIn("KSampler", failed.delta["required_breaking"])
            self.assertEqual(failed.delta["sampling_choices"]["samplers"]["removed"], ("euler",))
            self.assertIs(failed.snapshot, accepted.snapshot)
            self.assertEqual(destination.read_bytes(), before)

    def test_cold_clone_without_user_cache_reads_bundled_snapshot_once(self):
        with tempfile.TemporaryDirectory() as folder:
            empty_data = Path(folder)
            with (
                patch.dict(os.environ, {"RF_DATA_DIR": str(empty_data)}),
                patch.dict(runtime._loaded, {}, clear=True),
                patch.dict(runtime._last_good, {}, clear=True),
                patch.object(runtime, "_read_snapshot", wraps=runtime._read_snapshot) as read,
            ):
                snapshot = runtime.load_native_snapshot()
                self.assertIsNotNone(snapshot)
                self.assertIs(runtime.load_native_snapshot(), snapshot)
                self.assertIsNotNone(runtime.get_image_family_facts("SDXL", snapshot=snapshot))
                read.assert_called_once_with(runtime.DEFAULT_SNAPSHOT)
            self.assertEqual(list(empty_data.iterdir()), [])

    def test_bundled_snapshot_is_valid_and_matches_bootstrap_pins(self):
        snapshot = runtime.load_native_snapshot()
        self.assertIsNotNone(snapshot)
        bootstrap = ast.parse(
            (Path(runtime.__file__).resolve().parents[2] / "app/bootstrap.py").read_bytes()
        )
        from tools.audit_native_model_coverage import assignments

        pins = {
            row["name"]: row["hash"]
            for row in ast.literal_eval(assignments(bootstrap.body)["git_repos"])
        }
        self.assertEqual(snapshot.metadata()["native_pin"], pins["ComfyUI"])
        self.assertEqual(snapshot.metadata()["gguf_pin"], pins["molbal/ComfyUI-GGUF"])
        self.assertLess(len(json.dumps(snapshot.metadata())), 2048)

    def test_frozen_cached_snapshot_metadata_is_owned_and_no_runtime_tools_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.json"
            write_candidate(path, document_fixture())
            with patch.object(runtime, "_read_snapshot", wraps=runtime._read_snapshot) as read:
                first = runtime.load_native_snapshot(path)
                self.assertIs(first, runtime.load_native_snapshot(path))
                self.assertEqual(read.call_count, 1)
            family = runtime.get_image_family_facts("Image alias", snapshot=first)
            with self.assertRaises(TypeError):
                family.configurations[0]["unet_match"]["kind"] = "other"
            owned = family.metadata()
            owned["configurations"][0]["unet_match"]["kind"] = "other"
            self.assertEqual(family.configurations[0]["unet_match"]["kind"], "image")
            self.assertEqual(first.metadata()["native_pin"], runtime.ACCEPTED_COMFY_PIN)
            self.assertIsNone(runtime.get_image_family_facts("new architecture", snapshot=first))
        tree = ast.parse(Path(runtime.__file__).read_bytes())
        self.assertFalse(
            any(
                isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tools")
                for node in ast.walk(tree)
            )
        )

    def test_changed_parser_pin_source_and_payload_are_rejected(self):
        for kind in ("parser", "pin", "source", "facts"):
            with self.subTest(kind=kind):
                document = document_fixture()
                if kind == "parser":
                    document["manifest"]["parser_version"] = "new-unreviewed-parser"
                elif kind == "pin":
                    document["manifest"]["source_manifest"]["native_pin"] = "0" * 40
                elif kind == "source":
                    document["manifest"]["source_manifest"]["sources"]["native/source.py"] = (
                        "changed"
                    )
                else:
                    document["facts"]["families"]["Image"]["native_classes"] = ["Different"]
                with self.assertRaises(ValueError):
                    runtime.parse_native_snapshot(document)

    def test_atomic_promotion_invalidates_cache_and_failure_retains_last_good(self):
        with tempfile.TemporaryDirectory() as folder:
            destination, candidate = Path(folder) / "accepted.json", Path(folder) / "candidate.json"
            write_candidate(destination, document_fixture())
            previous = runtime.load_native_snapshot(destination)
            changed = facts_fixture()
            changed["families"]["Image"]["component_roles"][0]["role"] = "new_component"
            write_candidate(candidate, document_fixture(changed))
            promoted = runtime.refresh_native_snapshot(candidate, cache_path=destination)
            self.assertTrue(promoted.promoted)
            self.assertIs(promoted.snapshot, runtime.load_native_snapshot(destination))
            self.assertNotEqual(previous.snapshot_id, promoted.snapshot.snapshot_id)
            accepted_bytes = destination.read_bytes()
            candidate.write_text("invalid", encoding="utf-8")
            failed = runtime.refresh_native_snapshot(candidate, cache_path=destination)
            self.assertFalse(failed.promoted)
            self.assertIs(failed.snapshot, promoted.snapshot)
            self.assertEqual(destination.read_bytes(), accepted_bytes)
            write_candidate(candidate, document_fixture())
            with patch.object(runtime.os, "replace", side_effect=OSError("interrupted promotion")):
                failed = runtime.refresh_native_snapshot(candidate, cache_path=destination)
            self.assertFalse(failed.promoted)
            self.assertEqual(destination.read_bytes(), accepted_bytes)
            self.assertFalse(list(Path(folder).glob("*.candidate")))

    def test_semantic_delta_requires_review_and_required_removal_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            destination, candidate = Path(folder) / "accepted.json", Path(folder) / "candidate.json"
            write_candidate(destination, document_fixture())
            changed = facts_fixture()
            changed["schemas"]["KSampler"]["contract_sha256"] = "new-required-input-semantics"
            write_candidate(candidate, document_fixture(changed))
            failed = runtime.refresh_native_snapshot(candidate, cache_path=destination)
            self.assertFalse(failed.promoted)
            self.assertEqual(failed.delta["required_changed"], ("schemas:KSampler",))
            self.assertIn("Image", failed.delta["affected_families"])
            self.assertTrue(
                runtime.refresh_native_snapshot(
                    candidate, cache_path=destination, reviewed_semantics=True
                ).promoted
            )
            removed = facts_fixture()
            del removed["schemas"]["KSampler"]
            document = document_fixture(removed)
            document["manifest"]["required_contracts"]["schemas"] = []
            document["manifest"]["snapshot_id"] = runtime.content_digest(
                {key: value for key, value in document["manifest"].items() if key != "snapshot_id"}
            )
            write_candidate(candidate, document)
            failed = runtime.refresh_native_snapshot(
                candidate, cache_path=destination, reviewed_semantics=True
            )
            self.assertFalse(failed.promoted)
            self.assertEqual(failed.delta["required_removed"], ("schemas:KSampler",))

    def test_optional_becoming_required_blocks_even_reviewed_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            destination, candidate = Path(folder) / "accepted.json", Path(folder) / "candidate.json"
            old = facts_fixture()
            old["schemas"]["KSampler"]["inputs"][0]["required"] = False
            old["schemas"]["KSampler"]["contract_sha256"] = "optional-port"
            write_candidate(destination, document_fixture(old))
            previous = runtime.load_native_snapshot(destination)
            accepted_bytes = destination.read_bytes()
            write_candidate(candidate, document_fixture())
            failed = runtime.refresh_native_snapshot(
                candidate, cache_path=destination, reviewed_semantics=True
            )
            self.assertFalse(failed.promoted)
            self.assertEqual(failed.delta["required_breaking"], ("KSampler",))
            self.assertIs(failed.snapshot, previous)
            self.assertEqual(destination.read_bytes(), accepted_bytes)

    def test_template_and_source_identity_changes_are_visible_without_network(self):
        previous = document_fixture()
        changed = document_fixture()
        source = changed["manifest"]["source_manifest"]
        source["template_identity"]["manifest_sha256"] = "new-template"
        changed["manifest"]["source_manifest_sha256"] = runtime.content_digest(source)
        changed["manifest"]["snapshot_id"] = runtime.content_digest(
            {key: value for key, value in changed["manifest"].items() if key != "snapshot_id"}
        )
        delta = runtime.native_snapshot_delta(
            runtime.parse_native_snapshot(previous), runtime.parse_native_snapshot(changed)
        )
        self.assertTrue(delta["identity_changed"])
        self.assertEqual(delta["families"]["changed"], [])

    def test_graph_roles_and_configuration_semantics_have_explicit_upgrade_deltas(self):
        with tempfile.TemporaryDirectory() as folder:
            destination, candidate = Path(folder) / "accepted.json", Path(folder) / "candidate.json"
            write_candidate(destination, document_fixture())
            changed = facts_fixture()
            changed["families"]["Image"]["configurations"][0]["contract_sha256"] = (
                "changed-native-predicate"
            )
            next(iter(changed["recipes"].values()))["required_roles"].append("clip_vision")
            write_candidate(candidate, document_fixture(changed))
            failed = runtime.refresh_native_snapshot(candidate, cache_path=destination)
            self.assertFalse(failed.promoted)
            self.assertEqual(failed.delta["configuration_semantics_changed"], ("Image",))
            self.assertEqual(
                failed.delta["graph_roles"]["changed"], ("installed-official-templates:image.json",)
            )
            self.assertTrue(
                runtime.refresh_native_snapshot(
                    candidate, cache_path=destination, reviewed_semantics=True
                ).promoted
            )

    def test_native_class_removal_cannot_hide_in_a_retained_canonical_family(self):
        old = facts_fixture()
        old["families"]["Image"]["native_classes"].append("ImageVariant")
        previous, candidate = (
            runtime.parse_native_snapshot(document_fixture(old)),
            runtime.parse_native_snapshot(document_fixture()),
        )
        self.assertIn(
            "native_classes:ImageVariant",
            runtime.native_snapshot_delta(previous, candidate)["required_removed"],
        )

    def test_failed_candidate_serialization_preserves_previous_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / "candidate.json"
            write_candidate(candidate, document_fixture())
            accepted = candidate.read_bytes()
            with patch("tools.update_native_snapshot.json.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_candidate(candidate, document_fixture())
            self.assertEqual(candidate.read_bytes(), accepted)
            self.assertFalse(list(Path(folder).glob("*.tmp")))

    def test_candidate_cannot_overwrite_the_comparison_or_promotion_envelope(self):
        from tools.update_native_snapshot import validate_candidate_paths

        with tempfile.TemporaryDirectory() as folder:
            accepted = Path(folder) / "accepted.json"
            write_candidate(accepted, document_fixture())
            before = accepted.read_bytes()
            for keyword in ("compare", "promote"):
                with self.assertRaises(ValueError):
                    validate_candidate_paths(accepted, **{keyword: accepted})
            self.assertEqual(before, accepted.read_bytes())


if __name__ == "__main__":
    unittest.main()
