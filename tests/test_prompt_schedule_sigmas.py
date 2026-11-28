from ruinedfooocus.core.paths import asset_path, resource_path

"""Select switched conditioning at native scheduler sigmas, entirely on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.prompting.prompt_scheduling import bind_prompt_steps

NATIVE = Path(__file__).resolve().parents[1] / "repositories/ComfyUI"


def native_function(path, name, scope, prefix=None):
    tree = ast.parse((NATIVE / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if prefix is not None:
        node.body = node.body[:prefix] + [ast.Return(value=ast.Constant(True))]
        ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


class PromptSigmaTests(unittest.TestCase):
    def conditioning(self, count):
        return [
            [
                str(i),
                {"_rf_prompt_step": i, "start_percent": i / count, "end_percent": (i + 1) / count},
            ]
            for i in range(count)
        ]

    def test_native_nodes_and_midpoints_select_one_interval_for_normal_karras_flow_and_denoise(
        self,
    ):
        scope = {"asset_path": asset_path, "resource_path": resource_path, "torch": torch}
        native_function("comfy/k_diffusion/sampling.py", "append_zero", scope)
        karras = native_function("comfy/k_diffusion/sampling.py", "get_sigmas_karras", scope)
        eligible = native_function("comfy/samplers.py", "get_area_and_mult", {}, prefix=5)
        schedules = [
            torch.cat([torch.linspace(14, 0.03, 8), torch.zeros(1)]),
            karras(8, 0.03, 14),
            torch.linspace(1, 0, 9),
            karras(16, 0.03, 14)[-9:],
        ]
        model = SimpleNamespace(get_model_object=lambda name: object())
        for sigmas in schedules:
            for sampler in ("euler", "lcm"):
                with self.subTest(sigmas=sigmas.tolist(), sampler=sampler):
                    native = Mock(return_value=SimpleNamespace(sigmas=sigmas))
                    basic = Mock(return_value=[sigmas])
                    modules = {
                        "comfy.samplers": SimpleNamespace(KSampler=native),
                        "comfy_extras.nodes_custom_sampler": SimpleNamespace(
                            BasicScheduler=SimpleNamespace(execute=basic)
                        ),
                    }
                    original = self.conditioning(8)
                    with patch.dict("sys.modules", modules):
                        bound = bind_prompt_steps(original, model, 8, sampler, "karras", 0.5)
                    if sampler == "lcm":
                        basic.assert_called_once_with(
                            model=model, scheduler="karras", steps=8, denoise=0.5
                        )
                    else:
                        native.assert_called_once_with(
                            model, 8, device="cpu", sampler="euler", scheduler="karras", denoise=0.5
                        )
                    x = torch.zeros((1, 1, 1, 1), device="cpu")
                    for index, sigma in enumerate(sigmas[:-1]):
                        probes = [sigma, (sigma + sigmas[index + 1]) / 2]
                        if index == 0:
                            probes.append(
                                sigma * 1.05
                            )  # Churn above the first node remains covered.
                        if index == 7:
                            probes.append(torch.tensor(0.0))
                        for probe in probes:
                            selected = [
                                text
                                for text, values in bound
                                if eligible(values, x, probe.reshape(1))
                            ]
                            self.assertEqual(selected, [str(index)])
                    self.assertIn("_rf_prompt_step", original[0][1])
                    self.assertNotIn("_rf_prompt_step", bound[0][1])
                    self.assertNotIn("start_percent", bound[0][1])

    def test_native_clip_limits_are_intersected_without_overriding_sigma_bounds(self):
        sigmas = torch.tensor([10.0, 5.0, 0.0])
        sampling = SimpleNamespace(percent_to_sigma=lambda p: 10 * (1 - p))
        model = SimpleNamespace(get_model_object=lambda name: sampling)
        original = self.conditioning(2)
        original[0][1].update(clip_start_percent=0.2, clip_end_percent=0.6)
        with patch.dict(
            "sys.modules",
            {
                "comfy.samplers": SimpleNamespace(
                    KSampler=lambda *a, **k: SimpleNamespace(sigmas=sigmas)
                )
            },
        ):
            result = bind_prompt_steps(original, model, 2, "euler", "normal", 1)
        self.assertEqual(result[0][1]["timestep_start"], 8)
        self.assertGreater(result[0][1]["timestep_end"], 5)
        self.assertNotIn("clip_start_percent", result[0][1])
        self.assertNotIn("clip_end_percent", result[0][1])

    def test_plain_prompts_do_not_construct_a_scheduler_and_repeated_sigmas_are_explicit(self):
        plain = [["cat", {}]]
        self.assertIs(bind_prompt_steps(plain, None, 4, "euler", "normal", 1), plain)
        with patch.dict(
            "sys.modules",
            {
                "comfy.samplers": SimpleNamespace(
                    KSampler=lambda *a, **k: SimpleNamespace(sigmas=torch.tensor([1.0, 1.0, 0.0]))
                )
            },
        ):
            with self.assertRaisesRegex(ValueError, "distinct descending sigmas"):
                bind_prompt_steps(self.conditioning(2), None, 2, "euler", "normal", 1)


if __name__ == "__main__":
    unittest.main()
