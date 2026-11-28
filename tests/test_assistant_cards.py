"""Malformed character cards cannot interrupt the passive persona catalogue."""

import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.chat import assistants


class CharacterCardTests(unittest.TestCase):
    def test_malformed_character_cards_do_not_hide_valid_assistants(self):
        fields = {
            "name": "Valid character",
            "first_mes": "Hello",
            "personality": "Helpful",
            "scenario": "A conversation",
        }
        malformed = [[], {"data": []}, *({**fields, key: 42} for key in fields)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, payload in [
                *((f"invalid-{index}", value) for index, value in enumerate(malformed)),
                ("valid-v1", fields),
                ("valid-v2", {"data": {**fields, "name": "Valid nested character"}}),
            ]:
                metadata = PngInfo()
                metadata.add_text("chara", base64.b64encode(json.dumps(payload).encode()).decode())
                Image.new("RGB", (2, 2)).save(folder / f"{name}.png", pnginfo=metadata)

            with (
                patch.object(assistants, "content_roots", return_value=(folder,)),
                patch.object(assistants, "data_path", side_effect=folder.joinpath),
            ):
                choices = assistants.assistant_choices()
                self.assertIn(("Valid character", str(folder / "valid-v1.png")), choices)
                self.assertIn(("Valid nested character", str(folder / "valid-v2.png")), choices)
                for index in range(len(malformed)):
                    path = folder / f"invalid-{index}.png"
                    self.assertNotIn(str(path), {identity for _, identity in choices})
                    with self.assertRaisesRegex(ValueError, "Unsupported character card"):
                        assistants.character_card(path)


if __name__ == "__main__":
    unittest.main()
