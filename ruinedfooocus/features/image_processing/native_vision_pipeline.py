"""Single-image native vision utility on the existing image compute lane."""


class pipeline:
    pipeline_type = ["native_vision"]
    model_hash = ""

    def parse_gen_data(self, data):
        data["original_image_number"] = data.get("image_number", 1)
        data["image_number"] = 1
        data["show_preview"] = False
        if data["controlnet"].get("type") == "restore-seedvr2":
            from ruinedfooocus.features.image_processing.native_seedvr2 import geometry

            _, size, _ = geometry(data["input_image"].size, data["seedvr2_scale"])
        else:
            size = data["input_image"].size
        data["width"], data["height"] = size
        return data

    def load_base_model(self, name, hash=None):
        pass

    def load_loras(self, loras):
        pass

    def process(self, gen_data, callback=None):
        from ruinedfooocus.app import async_worker as worker
        from ruinedfooocus.app.state import path_manager
        from ruinedfooocus.features.image_processing import native_seedvr2, native_vision

        kind = str(gen_data["controlnet"]["type"]).lower()
        message = {
            "detect": "Detecting objects ...",
            "segment": "Segmenting the image ...",
            "depth-map": "Estimating depth ...",
            "restore-seedvr2": "Restoring still image ...",
        }[kind]
        worker.add_result(gen_data["task_id"], "preview", (-1, message, None))
        if kind == native_seedvr2.KIND:
            native_vision.clear_model(self)
            process = native_seedvr2.process
        else:
            native_seedvr2.clear_model(self)
            process = native_vision.process
        result = process(self, gen_data, path_manager, lambda: worker.check_interrupt(gen_data))
        from ruinedfooocus.features.generation.image_output_batches import ImageOutputBatch

        if isinstance(result, ImageOutputBatch):
            return result
        if callback:
            import numpy as np
            from PIL import Image

            preview = (
                result
                if isinstance(result, Image.Image)
                else (np.clip(result, 0, 1) * 255).astype(np.uint8)
            )
            callback(1, 0, 0, 1, preview)
        return [result]
