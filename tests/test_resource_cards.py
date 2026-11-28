"""Adapter cards read local presentation data without inspecting or loading weights."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.models.resource_cards import (
    DEFAULT_PREVIEW_ROLES,
    default_preview,
    resource_card,
    resource_records,
)
from ruinedfooocus.features.models.model_identity import file_signature


class ResourceCardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "opaque_file.safetensors"
        self.path.write_bytes(b"No tensor parsing or hashing is needed for a card")
        self.cache = self.root / "cache"
        self.cache.mkdir()

    def metadata(self, path, title="Publisher title", **extra):
        path.write_text(
            json.dumps({"model": {"name": title}, "name": "v1", **extra}), encoding="utf-8"
        )

    def test_local_metadata_and_preview_keep_publisher_title_separate_from_identity(self):
        self.metadata(self.path.with_suffix(".civitai.info"))
        preview = self.path.with_suffix(".preview.png")
        Image.new("RGB", (8, 8)).save(preview)
        with patch.object(Path, "open", autospec=True, wraps=Path.open) as opened:
            # This guard allows only presentation files, never the weight file.
            opened.side_effect = lambda path, *a, **kw: (
                self.fail("card opened weights") if path == self.path else open(path, *a, **kw)
            )
            card = resource_card(self.path.name, self.path, "Embedding", self.cache)
        self.assertEqual(card["label"], "Publisher title v1")
        self.assertEqual(card["preview"], str(preview))
        self.assertEqual(card["name"], self.path.name)
        self.assertNotIn(self.path.name, card["caption"])
        self.assertEqual(card["kind"], "Embedding")

    def test_cached_metadata_and_preview_work_for_new_embedding_without_network(self):
        self.metadata((self.cache / self.path.name).with_suffix(".json"))
        preview = (self.cache / self.path.name).with_suffix(".png")
        Image.new("RGB", (8, 8)).save(preview)
        card = resource_card(self.path.name, self.path, "Embedding", self.cache)
        self.assertEqual(card["label"], "Publisher title v1")
        self.assertEqual(card["preview"], str(preview))

    def test_meaningful_embedding_variant_name_is_retained_but_generic_versions_keep_title(self):
        metadata = self.path.with_suffix(".civitai.info")
        self.metadata(metadata, name="LazyPositive")
        self.assertEqual(
            resource_card(self.path.name, self.path, "Embedding")["label"], "LazyPositive"
        )
        for version, expected in (
            ("v1", "Publisher title v1"),
            ("1.2", "Publisher title · 1.2"),
            ("V3.0a", "Publisher title V3.0a"),
        ):
            self.metadata(metadata, name=version)
            self.assertEqual(
                resource_card(self.path.name, self.path, "Embedding")["label"],
                expected,
            )

    def test_replaced_file_rejects_signed_cache_and_malformed_sidecar(self):
        signature = file_signature(self.path)
        self.metadata(
            (self.cache / self.path.name).with_suffix(".json"), _rf_file_signature=signature
        )
        self.path.write_bytes(b"replacement")
        self.path.with_suffix(".civitai.info").write_text("[]", encoding="utf-8")
        card = resource_card(self.path.name, self.path, "LoRA", self.cache)
        self.assertEqual(card["metadata"], {})
        self.assertEqual(card["label"], "opaque file")
        self.assertIsInstance(card["preview"], Image.Image)

    def test_duplicate_labels_and_search_retain_native_file_identity(self):
        other = self.root / "different_name.pt"
        other.write_bytes(b"fixture")
        for path in (self.path, other):
            self.metadata(path.with_suffix(".civitai.info"))
        items = [(self.path.name, self.path), (other.name, other)]
        cards = resource_records(items, "Embedding")
        self.assertEqual(
            [card["label"] for card in cards], ["Publisher title v1 · 1", "Publisher title v1 · 2"]
        )
        self.assertNotEqual(cards[0]["caption"], cards[1]["caption"])
        self.assertEqual(resource_records(items, "Embedding", query="different_name"), [cards[1]])

    def test_kind_is_explicit_and_restricted_and_publisher_preview_can_be_disabled(self):
        with self.assertRaises(ValueError):
            resource_card("embedding_name.safetensors", self.path, "guess")
        with patch(
            "ruinedfooocus.features.models.resource_cards.resource_preview",
            side_effect=AssertionError("preview must not be inspected"),
        ):
            card = resource_card(self.path.name, self.path, "Embedding", preview_allowed=False)
        self.assertIsInstance(card["preview"], Image.Image)

    def test_exact_user_sidecar_precedes_legacy_and_cached_art(self):
        legacy = self.path.with_suffix(".preview.png")
        exact = self.path.with_name(self.path.name + ".preview.png")
        for path, color in ((legacy, "red"), (exact, "blue")):
            Image.new("RGB", (8, 8), color).save(path)
        self.assertEqual(
            resource_card(self.path.name, self.path, "LoRA", self.cache)["preview"], str(exact)
        )

    def test_category_fallback_is_a_reusable_static_local_raster_separate_from_publisher_art(self):
        path = Path(default_preview("Image models", self.cache))
        with Image.open(path) as image:
            self.assertEqual(image.size, (512, 512))
            self.assertEqual(getattr(image, "n_frames", 1), 1)
        self.assertEqual(default_preview("Image models", self.cache), str(path))
        self.assertEqual(path.parent.name, "default-previews")
        self.assertFalse(path.with_suffix(".source.json").exists())
        path.write_bytes(b"invalid old default")
        self.assertEqual(default_preview("Image models", self.cache), str(path))
        with Image.open(path) as image:
            image.verify()

    def test_every_catalogue_category_has_distinct_static_artwork(self):
        from ruinedfooocus.features.models.acquisition import CATEGORIES

        paths = {
            label: Path(default_preview(label, self.cache)) for label in set(CATEGORIES.values())
        }
        # Every category has bundled artwork; only Detection shares its role's image.
        shared = {"Detection": "Vision encoders"}
        self.assertEqual(len(set(paths.values())), len(set(CATEGORIES.values())) - len(shared))
        for label, owner in shared.items():
            self.assertEqual(paths[label], paths[owner])
        generic = Path(default_preview("Component", self.cache))
        self.assertNotIn(generic, paths.values())
        self.assertEqual(default_preview("unrecognized category", self.cache), str(generic))
        self.assertEqual(len(DEFAULT_PREVIEW_ROLES), 16)
        for path in [*paths.values(), generic]:
            with self.subTest(path=path), Image.open(path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (512, 512))
                self.assertEqual(getattr(image, "n_frames", 1), 1)
            self.assertFalse(path.with_suffix(".source.json").exists())

    def test_artwork_revision_changes_default_path_without_removing_prior_or_publisher_art(self):
        legacy = self.cache / "default-previews/image-model.jpeg"
        legacy.parent.mkdir()
        Image.new("RGB", (320, 240), "gray").save(legacy)
        publisher = (self.cache / self.path.name).with_suffix(".jpeg")
        Image.new("RGB", (8, 8), "blue").save(publisher)
        before = {path: path.read_bytes() for path in (legacy, publisher)}
        first = Path(default_preview("Image model", self.cache))
        before[first] = first.read_bytes()
        replacement = self.root / "new-bundled-art.jpeg"
        Image.new("RGB", (512, 512), "orange").save(replacement)
        with patch(
            "ruinedfooocus.features.models.resource_cards.asset_path", return_value=replacement
        ):
            second = Path(default_preview("Image model", self.cache))
        self.assertNotEqual(first, second)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(
            resource_card(self.path.name, self.path, "LoRA", self.cache)["preview"], str(publisher)
        )

    def test_explicit_fallback_root_preserves_real_preview_and_marks_only_defaults(self):
        cards = resource_records(
            [(self.path.name, self.path)], "LoRA", self.cache, fallback_cache_dir=self.cache
        )
        self.assertTrue(cards[0]["preview_is_default"])
        self.assertEqual(cards[0]["preview"], cards[0]["fallback"])
        self.assertEqual(Path(cards[0]["fallback"]).parent, self.cache / "default-previews")
        exact = self.path.with_name(self.path.name + ".preview.png")
        Image.new("RGB", (8, 8), "blue").save(exact)
        real = resource_card(
            self.path.name, self.path, "LoRA", self.cache, fallback_cache_dir=self.cache
        )
        self.assertFalse(real["preview_is_default"])
        self.assertEqual(real["preview"], str(exact))
        self.assertEqual(real["fallback"], cards[0]["fallback"])
        disabled = resource_card(
            self.path.name, self.path, "LoRA", self.cache, False, fallback_cache_dir=self.cache
        )
        self.assertTrue(disabled["preview_is_default"])
        self.assertEqual(disabled["preview"], disabled["fallback"])


if __name__ == "__main__":
    unittest.main()
