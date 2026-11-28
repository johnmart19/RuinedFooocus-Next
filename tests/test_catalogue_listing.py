"""Download catalogue listing: category order and the caption printed on placeholder artwork.

Pure functions and static asset checks. They do not fetch publisher artwork or
render the page in a browser.
"""

from pathlib import Path
import unittest

from ruinedfooocus.features.models import ui_acquisition as catalogue

ROOT = Path(__file__).resolve().parents[1]


def row(key, category, **extra):
    return dict(key=key, category_label=category, **extra)


class CatalogueListingTests(unittest.TestCase):
    def test_image_models_lead_and_each_category_keeps_its_own_order(self):
        rows = [
            row("2x-AniScale.pth", "Upscalers"),
            row("chat-b.gguf", "Chat models"),
            row("zeta.safetensors", "Image models"),
            row("hidden.safetensors", "Image models", _details_only=True),
            row("style.safetensors", "LoRAs"),
            row("alpha.safetensors", "Image models"),
            row("new-kind.bin", "Something new"),
            row("vae.safetensors", "VAEs"),
            row("chat-a.gguf", "Chat models"),
        ]
        listed = catalogue.listed_records(rows)
        self.assertEqual(
            [item["key"] for item in listed],
            [
                "zeta.safetensors",
                "alpha.safetensors",
                "style.safetensors",
                "2x-AniScale.pth",
                "vae.safetensors",
                "chat-b.gguf",
                "chat-a.gguf",
                "new-kind.bin",
            ],
        )
        self.assertEqual(catalogue.CATEGORY_ORDER[0], "Image models")
        self.assertEqual(len(set(catalogue.CATEGORY_ORDER)), len(catalogue.CATEGORY_ORDER))
        # Every category the catalogue defines has a deliberate place.
        from ruinedfooocus.features.models.acquisition import CATEGORIES

        self.assertTrue(set(CATEGORIES.values()) <= set(catalogue.CATEGORY_ORDER))

    def test_placeholder_caption_states_a_known_fact_or_the_category(self):
        caption = catalogue.placeholder_caption
        self.assertEqual(
            caption(row("flux.safetensors", "Image models", annotation={"architecture": "Flux2"})),
            "Flux2",
        )
        # Unknown or unusable family: the category, never a guess from the filename.
        for annotation in (None, {}, {"architecture": None}, {"architecture": "x" * 40}):
            self.assertEqual(
                caption(row("flux-dev.safetensors", "Image models", annotation=annotation)),
                "Image models",
            )
        self.assertEqual(caption(row("4x-UltraSharp.pth", "Upscalers")), "4× · Upscalers")
        self.assertEqual(caption(row("2X_Anime.pth", "Upscalers")), "2× · Upscalers")
        self.assertEqual(caption(row("RealESRGAN.pth", "Upscalers")), "Upscalers")
        # A family name on another category is not what that entry is.
        self.assertEqual(
            caption(row("te.safetensors", "Text encoders", annotation={"architecture": "Flux2"})),
            "Text encoders",
        )

    def test_cards_name_placeholders_and_clear_the_name_when_artwork_arrives(self):
        script = (ROOT / "ruinedfooocus/ui/assets/adapter_cards.js").read_text(encoding="utf-8")
        styles = (ROOT / "ruinedfooocus/ui/assets/adapter_cards.css").read_text(encoding="utf-8")
        self.assertIn("media.dataset.caption = caption", script)
        self.assertIn("delete media.dataset.caption", script)
        # A later artwork patch reuses the caption stored on the card.
        self.assertIn('button.dataset.caption || ""', script)
        self.assertIn(".adapter-card-media[data-caption]::after", styles)
        self.assertIn("content: attr(data-caption)", styles)


if __name__ == "__main__":
    unittest.main()
