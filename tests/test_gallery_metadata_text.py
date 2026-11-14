import json
import unittest

from ruinedfooocus.features.gallery.imagebrowser import format_metadata_string


class GalleryMetadataText(unittest.TestCase):
    def text(self, **params):
        return format_metadata_string(
            {"file_path": "out.png", "parameters": json.dumps({"Prompt": "cat", **params})}
        )

    def test_empty_optional_rows_are_omitted(self):
        text = self.text(base_model_name="model.safetensors", base_model_hash="", software="")
        self.assertIn("Model: model.safetensors", text)
        self.assertNotIn("Hash:", text)
        self.assertNotIn("Software", text)

    def test_recorded_hash_is_shown(self):
        text = self.text(base_model_name="model.safetensors", base_model_hash="abc123")
        self.assertIn("Hash: abc123", text)


if __name__ == "__main__":
    unittest.main()
