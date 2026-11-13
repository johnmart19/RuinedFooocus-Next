from ruinedfooocus.packaging.backend_preferences import force_cpu_for_auxiliary
import ruinedfooocus.app.async_worker as worker
from ruinedfooocus.app.arguments import args
from ruinedfooocus.backends.onnx.onnx_runtime import new_session, session_request_identity
from ruinedfooocus.features.image_editing.controlnet import get_settings

import rembg
import PIL.Image
from typing import Any


class pipeline:
    def remove_background(
        self,
        image: PIL.Image.Image,
        rembg_session: Any = None,
        force: bool = False,
        check_cancelled=None,
        **rembg_kwargs,
    ) -> PIL.Image.Image:
        do_remove = force or not (
            image.has_transparency_data and image.convert("RGBA").getextrema()[3][0] < 255
        )
        if do_remove:
            if rembg_session is None:
                identity = session_request_identity(cpu=force_cpu_for_auxiliary(args))
                if (
                    getattr(self, "_rembg_session", None) is not None
                    and getattr(self, "_rembg_identity", None) != identity
                ):
                    close = getattr(self._rembg_session, "close", None)
                    if close:
                        close()
                    self._rembg_session = None
                if getattr(self, "_rembg_session", None) is None:
                    self._rembg_session = new_session(cpu=force_cpu_for_auxiliary(args))
                    self._rembg_identity = identity
                rembg_session = self._rembg_session
            if hasattr(rembg_session, "check_cancelled"):
                rembg_session.check_cancelled = check_cancelled
            try:
                if not any(
                    rembg_kwargs.get(option)
                    for option in ("alpha_matting", "vitmatte", "decontaminate")
                ):
                    # PNG expects straight RGB plus alpha, not RGB blended with
                    # black and then attenuated a second time by its alpha.
                    rembg_kwargs.setdefault("putalpha", True)
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
            finally:
                if hasattr(rembg_session, "check_cancelled"):
                    rembg_session.check_cancelled = None
        return image

    pipeline_type = ["rembg"]
    model_hash = ""

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        if gen_data.get("input_image") is not None:
            gen_data["width"], gen_data["height"] = gen_data["input_image"].size
        return gen_data

    def load_base_model(self, name, hash=None):
        return

    def load_loras(self, loras):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        image = gen_data.get("input_image")
        if image is None:
            raise ValueError("Add an image in Image tools to remove its background.")
        worker.check_interrupt(gen_data)
        options = get_settings(gen_data)
        force = options.get("force", False)
        if gen_data.get("controlnet") is None:
            force = gen_data.get("rembg_force", force)

        worker.add_result(gen_data["task_id"], "preview", (-1, f"Removing background ...", None))

        if gen_data.get("rembg_native_model"):
            from ruinedfooocus.app.state import path_manager
            import torch
            from ruinedfooocus.features.image_processing.native_background import (
                clear_model,
                remove_background,
            )

            if getattr(self, "_rembg_session", None) is not None:
                close = getattr(self._rembg_session, "close", None)
                if close:
                    close()
                self._rembg_session = None
            gen_data["rembg_force"] = force
            try:
                image = remove_background(
                    self, image, gen_data, path_manager, lambda: worker.check_interrupt(gen_data)
                )
            except torch.OutOfMemoryError:
                from ruinedfooocus.core.device_fallback import warn_cpu_fallback

                clear_model(self)
                warn_cpu_fallback("background removal", "native model exceeded device memory")
                message = "CPU recovery: native background model exceeded device memory."
                gen_data["rembg_runtime_status"] = message
                gen_data.pop("native_background_source", None)
                worker.add_result(gen_data["task_id"], "preview", (-1, message, None))
                worker.check_interrupt(gen_data)
                self._rembg_session = new_session(cpu=True)
                self._rembg_identity = session_request_identity(cpu=True)
                image = self.remove_background(
                    image,
                    rembg_session=self._rembg_session,
                    force=force,
                    check_cancelled=lambda: worker.check_interrupt(gen_data),
                )
        else:
            from ruinedfooocus.features.image_processing.native_background import clear_model

            clear_model(self)
            image = self.remove_background(
                image, force=force, check_cancelled=lambda: worker.check_interrupt(gen_data)
            )
        runtime_status = getattr(getattr(self, "_rembg_session", None), "runtime_status", "")
        if runtime_status.startswith("CPU recovery:"):
            gen_data["rembg_runtime_status"] = runtime_status
            worker.add_result(gen_data["task_id"], "preview", (-1, runtime_status, None))
        worker.check_interrupt(gen_data)

        # Return finished image to preview
        if callback is not None:
            callback(1, 0, 0, 1, image)

        return [image]
