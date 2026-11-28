"""Still utilities on the existing serial image compute lane."""

from ruinedfooocus.features.image_processing.native_vision_pipeline import (
    pipeline as VisionPipeline,
)


class pipeline(VisionPipeline):
    pipeline_type = ["depth_utility"]

    def process(self, gen_data, callback=None):
        from ruinedfooocus.app import async_worker as worker
        from ruinedfooocus.app.state import path_manager
        from ruinedfooocus.features.image_processing.still_image_utilities import process

        worker.add_result(gen_data["task_id"], "preview", (-1, "Processing image ...", None))
        results = process(self, gen_data, path_manager, lambda: worker.check_interrupt(gen_data))
        if len(results) != 1:
            raise ValueError("A still-image utility must return exactly one image.")
        if callback:
            import numpy as np
            from PIL import Image

            result = results[0]
            preview = (
                result
                if isinstance(result, Image.Image)
                else (np.clip(result, 0, 1) * 255).astype(np.uint8)
            )
            callback(1, 0, 0, 1, preview)
        return results
