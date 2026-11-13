"""API image edits pick the checkpoint's native edit workflow when no tool is named."""

import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.models.image_components import native_edit_supported


def queued_data(
    family,
    name="local.safetensors",
    controlnet=None,
    moss_format="Disabled",
    preferences=None,
    recipe_resolver=None,
    payload_options=None,
    selection="Own settings",
    source_image=None,
    decode_calls=None,
    defaults_options=None,
):
    import ruinedfooocus.app.api_runtime as api_runtime

    perf = recipe_resolver or SimpleNamespace(
        CUSTOM_PERFORMANCE="Own settings",
        choices_for_model=lambda *args: ["Own settings"],
        selection_for_model=lambda *args: "Own settings",
        get_perf_options=lambda *args: dict(
            custom_steps=8, cfg=4, sampler_name="euler", scheduler="simple", clip_skip=1
        ),
    )
    shared = SimpleNamespace(
        settings=SimpleNamespace(
            default_settings={
                "performance": selection,
                "moss_image_edit_format": moss_format,
                "resolution": "default",
            }
        ),
        performance_settings=perf,
        models=SimpleNamespace(
            get_models_by_path=lambda *args, **kwargs: {"baseModel": family},
            get_model_base=lambda value: value["baseModel"],
            get_file=lambda *args: checkpoint,
        ),
        path_manager=SimpleNamespace(paths={}, DOWNLOADABLE_FILES={}, get_abspath=Path),
        resolution_settings=SimpleNamespace(
            CUSTOM_RESOLUTION="Custom...",
            selection_for_model=lambda *args: "default",
            aspect_ratios={"default": (1024, 1024)},
        ),
    )
    if preferences:
        shared.settings.default_settings.update(preferences)
    samplers = SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"]))
    defaults = dict(
        base_model_name=name,
        performance_selection=selection,
        moss_image_edit_format=moss_format,
    )
    defaults.update(defaults_options or {})
    payload = dict(prompt="make it blue", size="512x512", input_image="upload")
    payload.update(payload_options or {})

    def decode(encoded, **options):
        if decode_calls is not None:
            decode_calls.append(options)
        return source_image or Image.new("RGB", (64, 64))

    if controlnet:
        payload["controlnet"] = controlnet
    with (
        tempfile.TemporaryDirectory() as temporary,
        patch.dict(sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": samplers}),
        patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
        patch.object(api_runtime, "image_defaults", defaults),
        patch.object(api_runtime, "decode_image", side_effect=decode),
        patch.object(api_runtime, "submit", side_effect=lambda run: run),
    ):
        checkpoint = Path(temporary) / "fixture.safetensors"
        shared.path_manager.get_model_folders = lambda category: [Path(temporary)]
        for upscale_name in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"):
            (Path(temporary) / upscale_name).write_bytes(b"owned inert native upscale fixture")
        header = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
        checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        operation = api_runtime.images(payload, dict(family=family, name=name))
        shared.settings.default_settings.update(
            image_engine="sdcpp-vulkan", sdcpp_vulkan_device="Vulkan9"
        )
    cells = dict(
        zip(operation.__code__.co_freevars, (cell.cell_contents for cell in operation.__closure__))
    )
    return cells["data"]


def queued_control(*args, **kwargs):
    return queued_data(*args, **kwargs)["controlnet"]


class ApiExactTargetTests(unittest.TestCase):
    def test_2x_and_4x_capture_exact_fullhd_target_before_queueing(self):
        from ruinedfooocus.features.generation.size_plan import KEY

        for factor in (2, 4):
            captured = queued_data(
                "SDXL",
                payload_options={
                    "input_image": None,
                    "size": "1920x1080",
                    "prompt": "Patterned framed artwork",
                    "automatic_upscale": f"{factor}× General (RealESRGAN)",
                    "automatic_upscale_factor": factor,
                },
            )
            self.assertEqual(captured[KEY].target, (1920, 1080))
            self.assertEqual(captured[KEY].content, (1920 // factor, 1080 // factor))
            self.assertEqual(captured["size_mode"], "target")

    def test_mismatched_requested_upscale_factor_fails_before_queueing(self):
        with self.assertRaisesRegex(ValueError, "native factor"):
            queued_data(
                "SDXL",
                payload_options={
                    "input_image": None,
                    "size": "1920x1080",
                    "prompt": "A framed picture",
                    "automatic_upscale": "2× General (RealESRGAN)",
                    "automatic_upscale_factor": 4,
                },
            )

    def test_odd_target_is_captured_and_native_canvas_is_separate(self):
        from ruinedfooocus.features.generation.size_plan import KEY

        captured = queued_data(
            "SDXL",
            payload_options={
                "input_image": None,
                "size": "1919x1079",
                "prompt": "A framed picture",
                "automatic_upscale": "2× General (RealESRGAN)",
            },
        )
        self.assertEqual(captured[KEY].target, (1919, 1079))
        self.assertEqual(captured[KEY].restoration, "uniform_fit_pad")

    def test_4k_target_is_permitted_when_4x_base_is_within_native_limits(self):
        from ruinedfooocus.features.generation.size_plan import KEY

        captured = queued_data(
            "SDXL",
            payload_options={
                "input_image": None,
                "size": "3840x2160",
                "prompt": "A framed picture",
                "automatic_upscale": "4× General (RealESRGAN)",
            },
        )
        self.assertEqual(captured[KEY].content, (960, 540))


class ApiImageEditTests(unittest.TestCase):
    def test_sd4x_default_size_follows_source_but_explicit_incompatible_target_refuses(self):
        source = Image.new("RGBA", (117, 101), (80, 120, 220, 128))
        original = source.tobytes()
        decode_calls = []
        for size in (None, "auto"):
            data = queued_data(
                "SD_X4Upscaler",
                source_image=source,
                payload_options={"size": size},
                decode_calls=decode_calls,
            )
            self.assertEqual((data["custom_width"], data["custom_height"]), (468, 404))
            self.assertNotIn("_sd4x_requested_size", data)
            self.assertEqual(data["input_image"].tobytes(), source.tobytes())
            self.assertEqual(data["input_image"].info, source.info)
            self.assertEqual(data["input_image"].mode, "RGBA")
        with self.assertRaisesRegex(ValueError, "SD4x output must be 468×404"):
            queued_data("SD_X4Upscaler", source_image=source, payload_options={"size": "512x512"})
        data = queued_data(
            "SD_X4Upscaler", source_image=source, payload_options={"size": "468x404"}
        )
        self.assertEqual(data["_sd4x_requested_size"], (468, 404))
        self.assertEqual(source.tobytes(), original)
        self.assertTrue(all(call["preserve_alpha"] is True for call in decode_calls))
        with self.assertRaisesRegex(ValueError, "padding or cropping"):
            queued_data(
                "SD_X4Upscaler",
                source_image=source,
                payload_options={"size": "auto", "image_resize_mode": "crop"},
            )

    def test_engine_and_device_are_captured_before_preparation_can_wait(self):
        from ruinedfooocus.features.generation.sdcpp_pipeline import SNAPSHOT_KEY

        data = queued_data(
            "SDXL 1.0", preferences={"image_engine": "native", "sdcpp_vulkan_device": "Vulkan1"}
        )
        snapshot = data[SNAPSHOT_KEY]
        self.assertEqual((snapshot.engine, snapshot.device), ("native", "Vulkan1"))

    def test_image_operation_snapshots_all_actual_component_roles(self):
        self.assertEqual(
            queued_data("SDXL 1.0")["image_component_overrides"],
            {"clip_l": None, "clip_g": None, "vae_sdxl": None},
        )

    def test_edit_capable_checkpoints_use_native_editing_by_default(self):
        self.assertEqual(queued_control("Qwen"), {"type": "edit"})
        self.assertEqual(queued_control("MageFlow"), {"type": "edit"})
        self.assertEqual(
            queued_control("Flux.1 D", name="flux1-kontext-dev.safetensors"), {"type": "edit"}
        )

    def test_other_checkpoints_keep_gentle_image_to_image(self):
        self.assertEqual(queued_control("SDXL 1.0"), {"type": "img2img", "denoise": 0.25})
        self.assertEqual(queued_control("Flux.1 D"), {"type": "img2img", "denoise": 0.25})
        self.assertEqual(queued_control("MOSSImage"), {"type": "img2img", "denoise": 0.25})

    def test_an_explicit_tool_is_never_replaced(self):
        control = {"type": "img2img", "denoise": 0.6}
        self.assertEqual(queued_control("Qwen", controlnet=control), control)

    def test_ui_and_api_share_one_capability_rule(self):
        self.assertTrue(native_edit_supported("MOSS Image", moss_format="ai-toolkit"))
        self.assertFalse(native_edit_supported("MOSS Image"))
        self.assertFalse(native_edit_supported(None))


if __name__ == "__main__":
    unittest.main()
