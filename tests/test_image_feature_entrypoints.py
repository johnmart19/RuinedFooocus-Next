"""Real image feature callers must work in their declared prerequisite prefix."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRELUDE = """
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

root = Path(os.environ['RF_DATA_DIR']).resolve()
profile = root / 'settings' / 'entrypoint'
profile.mkdir(parents=True)
sys.argv = ['entrypoint', '--cpu', '--offline', '--skip-dependency-install',
            '--settings', 'entrypoint']
import torch
assert not torch.cuda.is_initialized()
def no_gpu(*args, **kwargs):
    raise AssertionError('Image entrypoint attempted GPU discovery or initialization')
torch.cuda._lazy_init = no_gpu
for name in ('init', 'is_available', 'device_count', 'current_device',
             'get_device_properties', 'get_device_name', 'get_device_capability', 'mem_get_info'):
    setattr(torch.cuda, name, no_gpu)
import requests
def no_external_io(*args, **kwargs):
    raise AssertionError('Image entrypoint attempted network or weight loading')
requests.sessions.Session.request = no_external_io
torch.load = no_external_io
"""


class ImageFeatureEntrypointsTests(unittest.TestCase):
    def run_entrypoint(self, script):
        # Each real app.state singleton belongs to a fresh, private profile.
        with tempfile.TemporaryDirectory(prefix="rf-image-entrypoint-") as folder:
            result = subprocess.run(
                [sys.executable, "-c", PRELUDE + textwrap.dedent(script)],
                cwd=ROOT,
                env=dict(
                    os.environ,
                    RF_DATA_DIR=folder,
                    RF_OFFLINE="1",
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    GRADIO_ANALYTICS_ENABLED="False",
                    PYTHONDONTWRITEBYTECODE="1",
                    CUDA_VISIBLE_DEVICES="-1",
                    HIP_VISIBLE_DEVICES="-1",
                    ROCR_VISIBLE_DEVICES="-1",
                ),
                text=True,
                capture_output=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_p01_path_manager_uses_real_arguments_and_owned_profile(self):
        self.run_entrypoint("""
            output = root / 'chosen-outputs'
            (profile / 'paths.json').write_text(json.dumps({'path_outputs': str(output)}))
            from ruinedfooocus.app.arguments import args
            from ruinedfooocus.core.path import PathManager
            manager = PathManager()
            assert args.settings == manager.name == 'entrypoint'
            assert manager.settings_path == profile / 'paths.json'
            assert Path(manager.model_paths['temp_outputs_path']).is_relative_to(root)
            assert manager.paths['path_outputs'] == str(output)
            assert manager.get_abspath(manager.paths['path_outputs']) == output
            assert manager.DOWNLOADABLE_FILES
            saved = json.loads(manager.settings_path.read_text())
            assert saved['path_outputs'] == str(output)
            assert saved['path_checkpoints'] == manager.DEFAULT_PATHS['path_checkpoints']
            for value in manager.model_paths.values():
                for item in value if isinstance(value, list) else [value]:
                    assert Path(item).resolve().is_relative_to(root), item
        """)

    def test_p02_validate_known_reaches_actual_image_validators(self):
        self.run_entrypoint("""
            from ruinedfooocus.core.settings_schema import validate_known
            from ruinedfooocus.features.image_processing.image_composition import DEFAULTS
            assert validate_known({}) == {}
            values = dict(DEFAULTS, grid_columns=3.0, grid_cell_width=513,
                          image_component_profiles={}, image_resize_mode='preserve',
                          model_recipe_sha256='a' * 64, retained_extension={'enabled': True})
            result = validate_known(values)
            assert result['grid_columns'] == 3 and type(result['grid_columns']) is int
            assert result['grid_cell_width'] == 513
            assert result['image_component_profiles'] == {}
            assert result['retained_extension'] == {'enabled': True}
            assert values['grid_columns'] == 3.0 and type(values['grid_columns']) is float
            for changes in ({'grid_columns': 0}, {'grid_cell_width': 32.5},
                            {'composition_mode': 'invented'}, {'image_component_profiles': []}):
                try:
                    validate_known(values | changes)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Image validator accepted ' + repr(changes))
        """)

    def test_p04_wildcard_helper_uses_real_catalogue_and_settings(self):
        self.run_entrypoint("""
            custom = root / 'owned-wildcards'
            custom.mkdir()
            selected = custom / 'entrypoint_custom.txt'
            selected.write_text('owned value\\n', encoding='utf-8')
            (profile / 'paths.json').write_text(json.dumps({'path_wildcards': [str(custom)]}))
            (profile / 'settings.json').write_text(json.dumps({'language': 'en', 'grid_columns': 3}))
            from ruinedfooocus.core.util import get_wildcard_files
            from ruinedfooocus.app.state import models, path_manager, settings
            from ruinedfooocus.features.prompting.wildcards import wildcard_files
            names = get_wildcard_files()
            assert 'entrypoint_custom' in names and 'words' in names
            assert 'onebuttonprompt' in names and len(names) == len(set(names))
            assert wildcard_files(path_manager.paths['path_wildcards'])['entrypoint_custom'] == selected
            assert settings.default_settings['grid_columns'] == 3
            assert settings.settings_path == profile / 'settings.json'
            assert models.offline is True
            assert all(Path(path).resolve().is_relative_to(root)
                       for paths in models.model_dirs.values() for path in paths)
        """)

    def test_p11_inventory_inspection_capture_and_resolution_use_real_owners(self):
        self.run_entrypoint("""
            source = Path(os.environ.get('RF_TEST_COMFY_SOURCE',
                                        Path.cwd() / 'repositories' / 'ComfyUI'))
            assert (source / 'comfy' / 'model_detection.py').is_file(), source
            sys.path.insert(0, str(source))
            from comfy.cli_args import args as native_args
            native_args.cpu = True
            from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management
            initialize_cpu_model_management()
            from comfy import model_detection

            checkpoint = root / 'models' / 'checkpoints' / 'entrypoint.safetensors'
            checkpoint.parent.mkdir(parents=True)
            header = {'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
            raw = json.dumps(header).encode('utf-8')
            checkpoint.write_bytes(len(raw).to_bytes(8, 'little') + raw + bytes(4))

            from ruinedfooocus.app.state import models, path_manager, resolution_settings
            from ruinedfooocus.features.models.image_components import inventory
            from ruinedfooocus.features.models.model_architecture import inspect_model_file
            from ruinedfooocus.features.models.component_profiles import snapshot_request
            from ruinedfooocus.features.models import effective_image_plan as plans
            from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan

            # Only native shape classification is substituted. Every RF owner,
            # header reader, catalogue, component resolver and capture is real.
            supported = type('SDXL', (), {'unet_config': {}, 'vae_key_prefix': (),
                                         'text_encoder_key_prefix': ()})()
            with patch.object(model_detection, 'unet_prefix_from_state_dict', return_value=''), \\
                 patch.object(model_detection, 'model_config_from_unet', return_value=supported):
                details = inspect_model_file(checkpoint)
                assert details['architecture'] == 'SDXL 1.0'
                assert details['native_class'] == 'SDXL'
                rows = inventory(checkpoint, 'SDXL', {}, path_manager)['rows']
                assert [row['key'] for row in rows] == ['clip_l', 'clip_g', 'vae_sdxl']
                assert all(row['status'] == 'Missing' for row in rows)
                request = dict(base_model_name=checkpoint.name, prompt='owned prompt',
                               negative_prompt='owned negative', loras=[], controlnet={},
                               performance_selection='Own settings', custom_steps=17, cfg=4.5,
                               sampler_name='euler', scheduler='simple', clip_skip=2,
                               custom_width=513, custom_height=257, resolution_auto=False,
                               size_mode='target', _sampling_captured=True,
                               **{SIZE_KEY: SizePlan.create((513, 257))})
                captured = snapshot_request(request, models, {}, path_manager)
                captured = plans.capture_effective_image_plan(captured, models, {}, path_manager)
                plan = plans.get_effective_image_plan(captured)
                assert plan.family == 'SDXL 1.0' and plan.native_class == 'SDXL'
                assert plan.size.native_plan.target == (513, 257)
                assert plan.recipe.sampling['custom_steps'] == 17
                assert len(plan.components) == 3
                assert plans.capture_effective_image_plan(captured, models, {}, path_manager)[plans.KEY] is plan
                request['prompt'] = 'later request'
                assert captured['prompt'] == 'owned prompt'

            resolution_settings.set_profile('entrypoint')
            label = resolution_settings.save_custom('Entry pixels', 513, 257, 'SDXL')
            assert resolution_settings.aspect_ratios[label] == (513, 257)
            saved = json.loads((profile / 'image_sizes.json').read_text())
            assert saved['sizes'] == [dict(name='Entry pixels', family='SDXL', width=513, height=257)]
            assert not torch.cuda.is_initialized()
        """)


if __name__ == "__main__":
    unittest.main()
