from ruinedfooocus.core.paths import asset_path
import ruinedfooocus.app.async_worker as worker
from PIL import Image

# Copy this file, add suitable code and add logic to ruinedfooocus/features/generation/pipelines.py to select it


class pipeline:
    pipeline_type = ["template"]

    model_hash = ""

    def load_base_model(self, name, hash=None):
        # Check if model is already loaded
        if self.model_hash == name:
            return
        print(f"Loading model: {name}")
        self.model_hash = name
        return

    def load_loras(self, loras):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(gen_data["task_id"], "preview", (-1, f"Generating ...", None))

        image = Image.open(str(asset_path("logo.png")))

        # Return finished image to preview
        if callback is not None:
            callback(gen_data["steps"], 0, 0, gen_data["steps"], image)

        return [image]
