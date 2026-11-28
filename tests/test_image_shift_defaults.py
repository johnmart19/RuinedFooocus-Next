from ruinedfooocus.core.paths import asset_path, resource_path

"""Native shift defaults and explicit startup/per-image overrides; no weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ruinedfooocus.features.image_editing.image_guidance import (
    STARTUP_SHIFT_LABELS,
    resolve_guidance,
    request_model,
    effective_guidance,
)
from ruinedfooocus.core.settings_validation import validate_settings

ROOT = Path(__file__).resolve().parents[1]


class Sampling:
    def __init__(self, config):
        self.shift = config.sampling_settings.get("shift", 1.0)
        self.multiplier = config.sampling_settings.get("multiplier", 1000)
        self.noise_scale = 2.7

    def set_parameters(self, shift, multiplier):
        self.shift, self.multiplier = shift, multiplier

    def set_noise_scale(self, value):
        self.noise_scale = value


class Patcher:
    def __init__(self, shift):
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(sampling_settings={"shift": shift})
        )
        self.sampling = Sampling(self.model.model_config)

    def clone(self):
        other = Patcher(self.sampling.shift)
        other.model, other.sampling = self.model, self.sampling
        return other

    def get_model_object(self, name):
        return self.sampling

    def add_object_patch(self, name, value):
        self.sampling = value


def native_nodes():
    tree = ast.parse(
        (ROOT / "repositories/ComfyUI/comfy_extras/nodes_model_advanced.py").read_text(
            encoding="utf-8"
        )
    )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in ("ModelSamplingSD3", "ModelSamplingAuraFlow")
    ]
    for node in nodes:
        node.body = [
            n
            for n in node.body
            if isinstance(n, ast.FunctionDef) and n.name in ("patch", "patch_aura")
        ]
    scope = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        comfy=SimpleNamespace(
            model_sampling=SimpleNamespace(
                ModelSamplingDiscreteFlow=Sampling, CONST=type("CONST", (), {})
            )
        ),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "native_shift_nodes", "exec"), scope)
    return dict(
        AuraFlow=scope["ModelSamplingAuraFlow"]().patch_aura, SD3=scope["ModelSamplingSD3"]().patch
    )


class ImageShiftDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        cls = next(
            n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline"
        )
        self.registry = next(
            n
            for n in cls.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "known_model_info" for t in n.targets)
        )
        self.resolver = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_clip_and_vae"
        )
        process = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        # These shift contracts execute the actual guidance capture and
        # request-clone call. Geometry, conditioning and optional patch stages
        # between them have their own callers and must not enter this fixture.
        route = [
            next(
                n
                for n in process.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            )
            for name in ("guidance", "sampling_model")
        ]
        fallback = [
            n
            for n in process.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test).startswith("gen_data.get('sampling_shift'")
        ]
        self.route = compile(
            ast.Module(body=fallback + route, type_ignores=[]),
            "production_shift_route",
            "exec",
        )
        native = ast.parse(
            (ROOT / "repositories/ComfyUI/comfy/supported_models.py").read_text(encoding="utf-8")
        )
        self.native_settings = {}
        for node in native.body:
            if isinstance(node, ast.ClassDef) and node.name in (
                "Lumina2",
                "ZImage",
                "QwenImage",
                "Anima",
                "AuraFlow",
            ):
                assignment = next(
                    n
                    for n in node.body
                    if isinstance(n, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "sampling_settings" for t in n.targets
                    )
                )
                self.native_settings[node.name] = ast.literal_eval(assignment.value)

    def info(self, family, defaults):
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            comfy=Mock(),
            settings=SimpleNamespace(default_settings=defaults),
        )
        exec(
            compile(
                ast.Module(body=[self.registry, self.resolver], type_ignores=[]),
                "production_component_info",
                "exec",
            ),
            scope,
        )
        config = (
            {"dim": 3840}
            if family == "ZImage"
            else {"clip_text_dim": 1024}
            if family == "NewBieImage"
            else {}
        )
        model = type("Lumina2" if family in ("ZImage", "NewBieImage") else family, (), {})()
        model.model_config = SimpleNamespace(unet_config=config)
        return scope["get_clip_and_vae"](
            SimpleNamespace(known_model_info=scope["known_model_info"]),
            SimpleNamespace(model=model),
        )

    def run_request(self, family, defaults, request=None):
        shift = self.native_settings.get(family, {"shift": 3})["shift"]
        base = Patcher(shift)
        info = self.info(family, defaults)
        nodes = native_nodes()
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            resolve_guidance=resolve_guidance,
            settings=SimpleNamespace(default_settings=defaults),
            gen_data=request or {},
            architecture=family,
            checkpoint="checkpoint.safetensors",
            self=SimpleNamespace(model_info=info, xl_base_patched=SimpleNamespace(unet=base)),
            request_model=lambda model, values, family, **kwargs: request_model(
                model, values, family, nodes
            ),
        )
        exec(self.route, scope)
        return base, scope["sampling_model"], scope["guidance"]

    def test_blank_missing_and_null_startup_shifts_preserve_native_variant_defaults(self):
        for family, key in (
            ("Anima", "anima_image_shift"),
            ("AuraFlow", "auraflow_shift"),
            ("HiDream", "hidream_shift"),
            ("Lumina2", "lumina2_shift"),
            ("ZImage", "lumina2_shift"),
            ("NewBieImage", "newbieimage_shift"),
            ("QwenImage", "qwen_image_shift"),
            ("SD3", "sd3_shift"),
        ):
            for defaults in ({}, {key: None}, {key: ""}):
                with self.subTest(family=family, defaults=defaults):
                    info = self.info(family, defaults)
                    self.assertIsNone(info.get("model_sampling"))
                    base, patched, values = self.run_request(family, defaults)
                    self.assertIs(patched.sampling, base.sampling)
                    self.assertIsNone(values["sampling_shift"])

    def test_explicit_startup_shift_applies_only_to_a_request_clone(self):
        for family, key in (
            ("Anima", "anima_image_shift"),
            ("Lumina2", "lumina2_shift"),
            ("ZImage", "lumina2_shift"),
            ("NewBieImage", "newbieimage_shift"),
            ("QwenImage", "qwen_image_shift"),
            ("HiDream", "hidream_shift"),
            ("SD3", "sd3_shift"),
        ):
            with self.subTest(family=family):
                base, patched, values = self.run_request(family, {key: "4.25"})
                self.assertEqual(patched.sampling.shift, 4.25)
                self.assertIsNot(patched.sampling, base.sampling)
                self.assertEqual(
                    base.sampling.shift, self.native_settings.get(family, {"shift": 3})["shift"]
                )
                self.assertEqual(patched.sampling.noise_scale, 2.7)
                self.assertEqual(
                    effective_guidance(values, patched, family)["effective_sampling_shift"], 4.25
                )

    def test_per_image_shift_wins_and_auto_returns_to_the_saved_startup_override(self):
        for value, expected in ((7, 7), (None, 4.25), ("", 4.25)):
            with self.subTest(value=value):
                _, patched, values = self.run_request(
                    "Lumina2", {"lumina2_shift": 4.25}, {"sampling_shift": value}
                )
                self.assertEqual(patched.sampling.shift, expected)
                self.assertEqual(values["sampling_shift"], expected)
        base, restored, _ = self.run_request("Lumina2", {})
        self.assertIs(restored.sampling, base.sampling)
        self.assertEqual(restored.sampling.shift, self.native_settings["Lumina2"]["shift"])

    def test_settings_reject_out_of_range_image_startup_shifts_before_saving(self):
        for field in STARTUP_SHIFT_LABELS:
            for value in (0.01, 4.25, 100):
                self.assertEqual(validate_settings({field: value})[field], value)
            for value in (0, 100.01, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_settings({field: value})


if __name__ == "__main__":
    unittest.main()
