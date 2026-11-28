"""Native Krea2 reference conditioning, model-only scope and request transport."""

from contextlib import contextmanager
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.features.image_editing import krea_style
from ruinedfooocus.features.generation import native_tensor_context
from ruinedfooocus.features.image_editing.image_guidance import resolve_guidance, request_model
from ruinedfooocus.features.image_editing.ordered_native_references import (
    OrderedNativeReferences,
    recipe_for,
    omit_inactive_recipe,
)
from test_ordered_native_references import runtime, vae_fixture
from test_qwen_image21_conditioning import native_class


@contextmanager
def native_modules(modules):
    missing = object()
    original = {key: sys.modules.get(key, missing) for key in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def clip_fixture():
    return NS(
        tokenize=Mock(side_effect=lambda text, **kwargs: dict(text=text, **kwargs)),
        encode_from_tokens_scheduled=Mock(
            side_effect=lambda tokens: [
                [torch.full((1, 2, 4), float(len(tokens["text"]))), {"text": tokens["text"]}]
            ]
        ),
    )


class KreaStyleReferenceTests(unittest.TestCase):
    def setUp(self):
        modules = runtime()
        zero = native_class(
            "nodes.py", "ConditioningZeroOut", dict(torch=torch), methods=("zero_out",)
        )
        modules["nodes"] = NS(ConditioningZeroOut=zero)
        self.enterContext(native_modules(modules))
        self.enterContext(
            patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder")
        )

    def test_actual_native_conditioning_method_zero_negative_and_independent_references(self):
        clip, vae = clip_fixture(), vae_fixture()
        refs = OrderedNativeReferences(
            "krea2-style",
            vae,
            (Image.new("RGB", (24, 40), "red"), Image.new("RGB", (40, 24), "blue")),
        )
        positive, negative = refs.conditioning(clip, "portrait", "")
        self.assertEqual(recipe_for("Krea2"), "krea2-style")
        self.assertEqual(positive[0][1]["reference_latents_method"], "index_timestep_zero")
        self.assertEqual(len(positive[0][1]["reference_latents"]), 2)
        self.assertTrue(torch.count_nonzero(negative[0][0]) == 0)
        self.assertIs(negative[0][1]["reference_latents"], positive[0][1]["reference_latents"])
        self.assertEqual(vae.encode.call_count, 2)
        _, explicit = refs.conditioning(clip, "portrait", "blur")
        self.assertEqual(
            explicit[0][1]["text"],
            "Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: <|vision_start|><|image_pad|><|vision_end|>blur",
        )
        self.assertEqual(vae.encode.call_count, 2)

    def test_native_flux_canvas_patch_uses_actual_dimensions_and_keeps_base_unmodified(self):
        guidance = resolve_guidance(
            dict(_native_krea_style=object(), width=768, height=1280), {}, "Krea2"
        )
        base = NS(clone=Mock(return_value="clone"))
        native = Mock(return_value=("patched",))
        self.assertEqual(request_model(base, guidance, "Krea2", nodes={"Flux": native}), "patched")
        native.assert_called_once_with(
            model="clone", max_shift=1.15, base_shift=0.5, width=768, height=1280
        )
        self.assertNotIn(
            "_model_sampling_flux", resolve_guidance(dict(width=768, height=1280), {}, "Krea2")
        )

    def test_capture_uses_shared_preparation_without_download_and_requires_recipe(self):
        from ruinedfooocus.features.generation.image_preparation import ImageArtifact
        from ruinedfooocus.core.preparation import PreparationRequest

        item = ImageArtifact(
            "krea-style-lora",
            krea_style.DEFAULT_LORA,
            PreparationRequest("https://example.invalid/model", Path("missing.safetensors")),
            None,
        )
        data = dict(
            base_model_name="krea",
            input_image=Image.new("RGB", (32, 40)),
            native_edit_recipe="krea2-style",
        )
        with (
            patch(
                "ruinedfooocus.features.generation.image_preparation._control_checkpoint_family",
                return_value="Krea 2",
            ),
            patch(
                "ruinedfooocus.features.generation.image_preparation._artifact", return_value=item
            ) as capture,
        ):
            captured, assets = krea_style.capture_assets(data, object(), object(), {"type": "edit"})
        self.assertEqual(assets, (item,))
        self.assertEqual(captured[krea_style.KEY], krea_style.KreaStyle(krea_style.DEFAULT_LORA, 1))
        capture.assert_called_once()
        self.assertNotIn(krea_style.KEY, data)
        self.assertNotIn(
            "krea_style_lora", omit_inactive_recipe(captured, {"type": "img2img"}, "Krea2")
        )
        for invalid in (dict(krea_style_lora="../x"), dict(krea_style_strength=float("nan"))):
            with self.assertRaises(ValueError):
                krea_style.values(invalid)

    def test_canonical_defaults_save_and_startup_reload_preserve_custom_style_settings(self):
        from ruinedfooocus.core.settings import SettingsManager
        from ruinedfooocus.features.generation.defaults import STARTUP_DEFAULTS, generation_defaults

        controls = dict(
            krea_style_lora="nested\\chosen.safetensors",
            krea_style_strength=-0.75,
            input_image=object(),
            reference_images=[object()],
        )
        saved = generation_defaults(controls)
        self.assertEqual(saved["krea_style_lora"], "nested/chosen.safetensors")
        self.assertEqual(saved["krea_style_strength"], -0.75)
        self.assertNotIn("input_image", saved)
        self.assertNotIn("reference_images", saved)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with (
                patch(
                    "ruinedfooocus.core.settings.settings_path",
                    return_value=folder / "settings.json",
                ),
                patch("ruinedfooocus.core.settings.profile_folder", return_value=folder),
            ):
                manager = SettingsManager(defaults=STARTUP_DEFAULTS)
                self.assertEqual(
                    {key: manager.default_settings[key] for key in krea_style.FIELDS},
                    krea_style.DEFAULTS,
                )
                self.assertNotIn(krea_style.KEY, manager.default_settings)
                manager.save_profile(manager.default_settings | saved, {}, profile="owned")
                restored = SettingsManager("owned", defaults=STARTUP_DEFAULTS)
                self.assertEqual(
                    restored.default_settings["krea_style_lora"], saved["krea_style_lora"]
                )
                self.assertEqual(restored.default_settings["krea_style_strength"], -0.75)
                self.assertNotIn(krea_style.KEY, restored.default_settings)

    def test_startup_reader_validates_style_filename_and_finite_strength_before_ui(self):
        from ruinedfooocus.core.profile_store import read_profile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for values in (
                dict(krea_style_lora="../outside.safetensors"),
                dict(krea_style_lora=4),
                dict(krea_style_strength=True),
                dict(krea_style_strength=float("nan")),
                dict(krea_style_strength=10.01),
            ):
                raw = json.dumps(values)
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_profile(path)
                self.assertEqual(path.read_text(encoding="utf-8"), raw)
            path.write_text(
                json.dumps(
                    dict(
                        krea_style_lora="nested\\style.sft",
                        krea_style_strength="0.25",
                        unrelated={"keep": True},
                    )
                ),
                encoding="utf-8",
            )
            restored = read_profile(path)
            self.assertEqual(restored["krea_style_lora"], "nested/style.sft")
            self.assertEqual(restored["krea_style_strength"], 0.25)
            self.assertEqual(restored["unrelated"], {"keep": True})

    def test_model_only_native_loader_scope_restores_model_on_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / krea_style.DEFAULT_LORA
            path.write_bytes(b"fixture")
            original = NS(
                unet=NS(model=type("Krea2", (), {})(), patches={}),
                clip=object(),
                vae=object(),
                clip_vision=None,
            )
            owner = NS(xl_base_patched=original, StableDiffusionModel=lambda **kwargs: NS(**kwargs))
            events = []

            @contextmanager
            def lease(*args):
                events.append("enter")
                yield path
                events.append("exit")

            patched_model = NS(patches={"key": [object()]})
            load = Mock(return_value=(patched_model,))
            modules = dict(
                nodes=NS(LoraLoaderModelOnly=lambda: NS(load_lora_model_only=load)),
                folder_paths=NS(get_full_path_or_raise=lambda *_: str(path)),
            )
            modules["ruinedfooocus.app.async_worker"] = NS(check_interrupt=Mock())
            with (
                native_modules(modules),
                patch("ruinedfooocus.app.resource_lifecycle.require_compute_owner"),
                patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease),
            ):
                data = {
                    krea_style.KEY: krea_style.KreaStyle(krea_style.DEFAULT_LORA, 1),
                    "native_edit_recipe": "krea2-style",
                }

                @krea_style.scoped
                def operation(owner, data):
                    self.assertIs(owner.xl_base_patched.unet, patched_model)
                    self.assertIs(owner.xl_base_patched.clip, original.clip)
                    self.assertEqual(events[-1], "enter")
                    return "done"

                self.assertEqual(operation(owner, data), "done")
                self.assertIs(owner.xl_base_patched, original)
                load.assert_called_once_with(original.unet, krea_style.DEFAULT_LORA, 1)

                @krea_style.scoped
                def ordinary(owner, data):
                    self.assertIs(owner.xl_base_patched, original)
                    return "ordinary"

                self.assertEqual(ordinary(owner, {}), "ordinary")
                self.assertEqual(load.call_count, 1)

                @krea_style.scoped
                def fail(owner, data):
                    raise RuntimeError("cancelled")

                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    fail(owner, data)
                self.assertIs(owner.xl_base_patched, original)

    def test_public_api_transports_recipe_style_fields_and_enforces_reference_limit(self):
        from test_api_image_edits import queued_data
        from ruinedfooocus.integrations.openai_api import ImageRequest

        with self.assertRaises(ValueError):
            ImageRequest(
                input_image="fixture", native_edit_recipe="krea2-style", reference_images=["r"] * 3
            )
        with self.assertRaises(ValueError):
            ImageRequest(
                input_image="fixture", krea_style_strength=1, controlnet=dict(type="img2img")
            )
        from ruinedfooocus.features.generation.image_preparation import ImageArtifact
        from ruinedfooocus.core.preparation import PreparationRequest

        item = ImageArtifact(
            "krea-style-lora",
            "custom.safetensors",
            PreparationRequest("", Path("custom.safetensors")),
            None,
        )
        with patch(
            "ruinedfooocus.features.generation.image_preparation._artifact", return_value=item
        ):
            captured = queued_data(
                "Krea2",
                payload_options=dict(
                    native_edit_recipe="krea2-style",
                    krea_style_lora="custom.safetensors",
                    krea_style_strength=0.75,
                ),
            )
        self.assertEqual(captured["controlnet"], {"type": "edit"})
        self.assertEqual(captured["krea_style_strength"], 0.75)
        self.assertEqual(captured["krea_style_lora"], "custom.safetensors")

    def test_actual_gradio_context_shows_style_controls_for_effective_krea_checkpoint(self):
        import ruinedfooocus.app as app_package
        from ruinedfooocus.features.image_editing import ui_ordered_references as ui

        registered = {}
        shared = NS(
            settings=NS(
                default_settings=dict(
                    krea_style_lora="saved-style.safetensors", krea_style_strength=0.4
                )
            ),
            add_ctrl=lambda key, obj: registered.update({key: obj}),
        )
        with (
            native_modules({"ruinedfooocus.app.state": shared}),
            patch.object(app_package, "state", shared, create=True),
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda name: "Krea2" if name == "krea" else "Boogu",
            ),
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                model, tool, kind = (
                    gr.Textbox(value="krea"),
                    gr.Textbox(value="Native image edit"),
                    gr.Textbox(value="edit"),
                )
                ui.create_ui(model, tool, kind, None, lambda tool, kind: kind)
            self.assertEqual(registered["krea_style_lora"].value, "saved-style.safetensors")
            self.assertEqual(registered["krea_style_strength"].value, 0.4)
            self.assertIn(
                "krea2-style", [value for _, value in registered["native_edit_recipe"].choices]
            )
            index = next(key for key, fn in app.fns.items() if fn.fn.__name__ == "style_context")
            for checkpoint, operation, expected in (
                ("krea", "edit", True),
                ("krea", "img2img", False),
                ("boogu", "edit", False),
            ):
                response = asyncio.run(
                    app.process_api(
                        index,
                        [checkpoint, "Native image edit", operation, "auto"],
                        state=SessionState(app),
                    )
                )
                self.assertEqual(response["data"][0]["visible"], expected)

    def test_tensor_context_binds_style_strength_and_prepared_adapter_file_identity(self):
        from ruinedfooocus.features.generation.image_preparation import ImageArtifact, ImageAssets
        from ruinedfooocus.core.preparation import PreparationRequest
        from ruinedfooocus.features.models.model_identity import file_signature

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / krea_style.DEFAULT_LORA
            path.write_bytes(b"original style adapter")
            item = ImageArtifact(
                "krea-style-lora",
                path.name,
                PreparationRequest("", path),
                tuple(file_signature(path)),
            )
            data = dict(
                krea_style_lora=path.name, krea_style_strength=1, native_edit_recipe="krea2-style"
            )
            capture = native_tensor_context.capture_context(data, None)
            plan = NS(family="Krea2", context=native_tensor_context.canonical(capture), inputs=())
            data.update(_native_tensor_plan=plan, _native_image_assets=ImageAssets((item,), True))
            vae = NS(
                first_stage_model=object(),
                latent_channels=16,
                latent_dim=3,
                spacial_compression_decode=lambda: 8,
            )
            model = NS(
                model_config=type("Krea2", (), dict(unet_config=dict(image_model="krea2")))()
            )
            owner = NS(
                xl_base_patched=NS(
                    vae=vae, unet=NS(model=model), clip=NS(cond_stage_model=object())
                )
            )
            runtime = native_tensor_context.runtime_context(owner, data)
            source = NS(
                context=dict(
                    version=native_tensor_context.VERSION, runtime=runtime, capture=capture
                )
            )
            plan.inputs = (source,)
            native_tensor_context.validate_runtime(owner, data)
            changed = dict(data, krea_style_strength=0.5)
            plan.context = native_tensor_context.canonical(
                native_tensor_context.capture_context(changed, None)
            )
            with self.assertRaisesRegex(ValueError, "exact original"):
                native_tensor_context.validate_runtime(owner, changed)
            plan.context = native_tensor_context.canonical(capture)
            path.write_bytes(b"replaced style adapter payload")
            with self.assertRaisesRegex(ValueError, "exact original"):
                native_tensor_context.validate_runtime(owner, data)
