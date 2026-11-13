"""One bounded CPU image operation on the existing image execution lane."""


class pipeline:
    pipeline_type = ["image_composition"]
    model_hash = ""

    def parse_gen_data(self, data):
        from ruinedfooocus.features.image_processing.image_composition import worker_data

        result = worker_data(data)
        result["original_image_number"] = 1
        result["show_preview"] = False
        return result

    def load_base_model(self, name, hash=None):
        pass

    def load_loras(self, loras):
        pass

    def process(self, gen_data, callback=None):
        from ruinedfooocus.app import async_worker as worker
        from ruinedfooocus.features.image_processing.image_composition import process

        worker.add_result(gen_data["task_id"], "preview", (-1, "Composing the images ...", None))
        result = process(gen_data, lambda: worker.check_interrupt(gen_data))
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
