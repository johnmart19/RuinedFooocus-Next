"""Local #! commands must not leak into shared generation defaults."""

import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import ruinedfooocus.app


class HashbangTests(unittest.TestCase):
    def test_short_and_invalid_commands_do_not_crash_or_save_partial_presets(self):
        defaults = dict(
            custom_steps=30, cfg=8, sampler_name="uni_pc", scheduler="simple", clip_skip=1
        )
        for prompt, expected in [
            ("#!add performance", None),
            ("#!add performance\nname:", None),
            ("#!add performance\nname: Foo\nsteps: bad", None),
            ("#!add performance\nname: Foo", "Foo"),
            ("#!add performance\nname: Foo: bar\n", "Foo: bar"),
        ]:
            with self.subTest(prompt=prompt):
                save = Mock()
                state = SimpleNamespace(
                    performance_settings=SimpleNamespace(
                        default_settings=defaults,
                        load_performance=lambda: {},
                        save_performance=save,
                    ),
                    update_cfg=Mock(),
                )
                # "import ruinedfooocus.app.state as shared" reads the package
                # attribute, so patch it as well as sys.modules.
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "ruinedfooocus.app.state": state,
                            "ruinedfooocus.app.async_worker": Mock(),
                            "comfy.samplers": SimpleNamespace(
                                KSampler=SimpleNamespace(SAMPLERS=["uni_pc"], SCHEDULERS=["simple"])
                            ),
                        },
                    ),
                    patch.object(ruinedfooocus.app, "state", state, create=True),
                ):
                    sys.modules.pop("ruinedfooocus.features.automation.hashbang_pipeline", None)
                    module = importlib.import_module(
                        "ruinedfooocus.features.automation.hashbang_pipeline"
                    )
                    # Real packaged images: a missing resource must fail this closure test.
                    module.pipeline().process(dict(task_id=1, prompt=prompt))
                sys.modules.pop("ruinedfooocus.features.automation.hashbang_pipeline", None)
                if expected is None:
                    save.assert_not_called()
                    state.update_cfg.assert_not_called()
                else:
                    save.assert_called_once_with({expected: defaults})

    def test_add_performance_saves_preset_without_changing_defaults(self):
        defaults = dict(
            custom_steps=30, cfg=8, sampler_name="dpmpp_2m_sde_gpu", scheduler="karras", clip_skip=1
        )
        saved = {}
        settings = SimpleNamespace(
            default_settings=defaults,
            load_performance=lambda: {},
            save_performance=lambda options: saved.update(options),
        )
        shared = SimpleNamespace(performance_settings=settings, update_cfg=Mock())
        samplers = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=["uni_pc"], SCHEDULERS=["simple"])
        )
        stubs = {
            "ruinedfooocus.app.state": shared,
            "ruinedfooocus.app.async_worker": Mock(),
            "comfy.samplers": samplers,
        }
        with (
            patch.dict(sys.modules, stubs),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            sys.modules.pop("ruinedfooocus.features.automation.hashbang_pipeline", None)
            module = importlib.import_module("ruinedfooocus.features.automation.hashbang_pipeline")
            prompt = "#!add performance\nname: WAN\nsteps: 12\ncfg: 4\nscheduler: simple\nsampler: uni_pc"
            with patch.object(module.Image, "open", return_value=Mock()):
                module.pipeline().process(dict(task_id=1, prompt=prompt))
        sys.modules.pop("ruinedfooocus.features.automation.hashbang_pipeline", None)
        self.assertEqual(
            saved["WAN"],
            dict(custom_steps=12, cfg=4.0, sampler_name="uni_pc", scheduler="simple", clip_skip=1),
        )
        self.assertEqual(
            defaults,
            dict(
                custom_steps=30,
                cfg=8,
                sampler_name="dpmpp_2m_sde_gpu",
                scheduler="karras",
                clip_skip=1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
