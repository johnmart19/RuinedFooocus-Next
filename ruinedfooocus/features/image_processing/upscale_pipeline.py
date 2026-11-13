import numpy as np
import torch
import folder_paths
import ruinedfooocus.app.async_worker as worker
import ruinedfooocus.features.image_editing.controlnet as controlnet
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.app.state import path_manager


class InterruptibleUpscaler:
    """Check the current job between tiles without changing ComfyUI globals."""

    def __init__(self, model, gen_data):
        self.model = model
        self.gen_data = gen_data

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, image):
        worker.check_interrupt(self.gen_data)
        return self.model(image)


class pipeline:
    pipeline_type = ["upscale"]
    model_hash = ""

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_upscaler_model(self, model_name, gen_data=None):
        from ruinedfooocus.features.generation.image_preparation import prepared_asset

        with prepared_asset(
            gen_data, "upscaler", model_name, path_manager, "upscalers"
        ) as model_path:
            signature = file_signature(model_path)
            if signature == getattr(self, "upscaler_file_signature", None):
                return self.upscaler_model
            from comfy_extras.nodes_upscale_model import UpscaleModelLoader

            folder_paths.add_model_folder_path(
                "upscale_models", str(model_path.parent), is_default=True
            )
            out = UpscaleModelLoader.execute(model_path.name)[0]
            self.upscaler_model = out
            self.upscaler_file_signature = signature
            from ruinedfooocus.features.generation.automatic_upscale import remember_native_metadata

            remember_native_metadata(model_path, out, path_manager)
            return out

    def load_base_model(self, name, hash=None):
        # Check if model is already loaded
        if self.model_hash == name:
            return
        print(f"Loading model: {name}")
        self.model_hash = name
        return

    def load_loras(self, loras):
        return

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        input_image = gen_data["input_image"]
        if input_image is None:
            raise ValueError("Choose an input image for upscaling.")
        from PIL import ImageOps
        from ruinedfooocus.features.generation.size_plan import KEY, SizePlan

        source = ImageOps.exif_transpose(input_image)
        source_size = source.size
        plan = gen_data.get(KEY)
        if isinstance(plan, SizePlan):
            from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

            input_image, _ = prepare_reference(
                source,
                plan.content,
                "pad",
                multiple=1,
                preserve_alpha=source.has_transparency_data,
            )
        else:
            source = source.convert("RGBA" if source.has_transparency_data else "RGB")
            input_image = torch.from_numpy(np.array(source).astype(np.float32) / 255.0)[None,]

        worker.add_result(gen_data["task_id"], "preview", (-1, f"Load upscaling model ...", None))

        cn_settings = controlnet.get_settings(gen_data)
        upscaler_name = cn_settings.get("upscaler")
        if not upscaler_name or upscaler_name == "None":
            raise ValueError("Choose an upscaler in Image tools.")

        upscaler_model = self.load_upscaler_model(upscaler_name, gen_data)
        if isinstance(plan, SizePlan) and upscaler_model.scale != plan.scale:
            raise ValueError("Loaded upscaler scale disagrees with the captured output target.")
        if isinstance(plan, SizePlan) and any(
            getattr(upscaler_model, key, 3) != 3 for key in ("input_channels", "output_channels")
        ):
            raise ValueError(
                "Exact image targets require a native RGB upscaler; source alpha is preserved separately."
            )
        worker.add_result(gen_data["task_id"], "preview", (-1, "Upscaling image ...", None))
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        decoded = ImageUpscaleWithModel.execute(
            InterruptibleUpscaler(upscaler_model, gen_data), input_image
        )[0]
        if isinstance(plan, SizePlan):
            decoded = plan.finish(decoded)
            gen_data["size_plan"] = plan.metadata()
            gen_data["upscale_source_size"] = list(source_size)
            import hashlib

            gen_data["upscale_source"] = dict(
                width=source_size[0],
                height=source_size[1],
                mode=source.mode,
                pixel_mode="RGBA" if source.has_transparency_data else "RGB",
                pixels_sha256=hashlib.sha256(
                    source.convert("RGBA" if source.has_transparency_data else "RGB").tobytes()
                ).hexdigest(),
            )
        worker.check_interrupt(gen_data)
        from ruinedfooocus.features.image_processing.native_image_formats import pipeline_images

        return pipeline_images(decoded, gen_data)
