"""Popular-model lists: what is kept from a listing, caching, and where a download goes.

Listings and downloads are stubbed. These checks do not contact Civitai or
Hugging Face and do not prove that either service still answers in this shape.
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ruinedfooocus.features.models import popular_models as feed
from ruinedfooocus.features.models import ui_popular

DIGEST = "ab" * 32


def civitai_item(identity=1, **changes):
    version = dict(
        id=identity * 10,
        name="v1",
        baseModel="Anima",
        publishedAt="2026-11-01T10:00:00Z",
        availability="Public",
        files=[
            dict(
                name="model one.safetensors",
                sizeKB=2048,
                primary=True,
                downloadUrl=f"https://civitai.com/api/download/models/{identity * 10}",
                hashes=dict(SHA256=DIGEST.upper()),
            )
        ],
        images=[
            dict(
                url="https://image.civitai.com/a/b/original=true/1.jpeg", nsfwLevel=4, type="image"
            ),
            dict(
                url="https://image.civitai.com/a/b/original=true/2.mp4", nsfwLevel=1, type="video"
            ),
            dict(
                url="https://image.civitai.com/a/b/original=true/3.jpeg", nsfwLevel=1, type="image"
            ),
        ],
    )
    version.update(changes.pop("version", {}))
    item = dict(
        id=identity,
        name=f"Model {identity}",
        type="Checkpoint",
        nsfw=False,
        stats=dict(downloadCount=1500, thumbsUpCount=20),
        modelVersions=[version],
    )
    item.update(changes)
    return item


class PopularModelTests(unittest.TestCase):
    def test_only_supported_base_models_are_recognised(self):
        family = feed.supported_family
        self.assertEqual(family("Anima"), "Anima")
        self.assertEqual(family("Illustrious"), "SDXL")
        self.assertEqual(family("Krea 2 Turbo"), "Krea2")
        self.assertEqual(family("ZImageTurbo"), "ZImage")
        self.assertEqual(family("Flux.2 Klein 9B"), "Flux2Klein9B")
        for unknown in ("Some Future Model", "", None, 7):
            self.assertIsNone(family(unknown))

    def test_listing_keeps_supported_public_verifiable_files(self):
        items = [
            civitai_item(1),
            civitai_item(2, version=dict(baseModel="Some Future Model")),
            civitai_item(3, version=dict(availability="EarlyAccess")),
            civitai_item(4, nsfw=True),
            civitai_item(5, type="LORA"),
            civitai_item(
                6,
                version=dict(
                    files=[
                        dict(
                            name="m.safetensors",
                            primary=True,
                            downloadUrl="https://civitai.com/api/download/models/60",
                            hashes={},
                        )
                    ]
                ),
            ),
            civitai_item(
                7,
                version=dict(
                    files=[
                        dict(
                            name="m.ckpt",
                            primary=True,
                            downloadUrl="https://civitai.com/api/download/models/70",
                            hashes=dict(SHA256=DIGEST),
                        )
                    ]
                ),
            ),
            civitai_item(
                8,
                version=dict(
                    files=[
                        dict(
                            name="m.safetensors",
                            primary=True,
                            downloadUrl="https://example.com/m.safetensors",
                            hashes=dict(SHA256=DIGEST),
                        )
                    ]
                ),
            ),
            "not a model",
        ]
        entries = feed.parse_civitai(items, "Image models")
        self.assertEqual([entry["key"] for entry in entries], ["civitai:10"])
        entry = entries[0]
        self.assertEqual(entry["sha256"], DIGEST)
        self.assertEqual(entry["bytes"], 2048 * 1024)
        self.assertEqual(entry["published"], "2026-11-01")
        # The thumbnail is the first still picture rated for everyone, in a small size.
        self.assertEqual(entry["image"], "https://image.civitai.com/a/b/width=320/3.jpeg")
        self.assertEqual(
            [e["key"] for e in feed.parse_civitai(items, "Image models", include_mature=True)],
            ["civitai:10", "civitai:40"],
        )
        self.assertEqual([e["key"] for e in feed.parse_civitai(items, "LoRAs")], ["civitai:50"])
        self.assertEqual(feed.parse_civitai(None, "LoRAs"), [])

    def test_file_names_from_a_listing_never_leave_the_folder(self):
        self.assertEqual(feed.safe_filename("..\\..\\evil/../x y.safetensors"), "x y.safetensors")
        self.assertEqual(feed.safe_filename("a<b>|c?.gguf"), "a_b__c_.gguf")
        self.assertEqual(feed.safe_filename("..."), "")

    def test_chat_listing_and_file_choice(self):
        items = [
            dict(
                id="org/Model-GGUF",
                downloads=10,
                likes=2,
                tags=["gguf"],
                createdAt="2026-10-05T00:00:00Z",
            ),
            dict(id="org/Private-GGUF", private=True, tags=["gguf"]),
            dict(id="org/Gated-GGUF", gated="auto", tags=["gguf"]),
            dict(id="org/NotGguf", tags=["safetensors"]),
            dict(id="../escape", tags=["gguf"]),
        ]
        entries = feed.parse_huggingface(items)
        self.assertEqual([entry["repository"] for entry in entries], ["org/Model-GGUF"])

        def sibling(name, size=4 * 1024**3, digest=DIGEST):
            return dict(rfilename=name, lfs=dict(sha256=digest, size=size))

        manifest = dict(
            sha="f" * 40,
            siblings=[
                sibling("model-Q8_0.gguf"),
                sibling("model-q5_k_m.gguf"),
                sibling("model-Q4_K_M.gguf", digest="cd" * 32),
                sibling("mmproj-model-Q4_K_M.gguf"),
                sibling("sub/model-Q4_K_M.gguf"),
                sibling("model-Q4_K_M-00001-of-00002.gguf"),
                dict(rfilename="README.md"),
            ],
        )
        self.assertEqual(feed.choose_chat_file(manifest)["file"], "model-Q4_K_M.gguf")
        resolved = feed.resolve_file(entries[0], manifest=manifest)
        self.assertEqual(resolved["sha256"], "cd" * 32)
        self.assertEqual(
            resolved["url"],
            f"https://huggingface.co/org/Model-GGUF/resolve/{'f' * 40}/model-Q4_K_M.gguf",
        )
        # Too large, no hash, or an unrecognised file: nothing is guessed.
        for siblings in (
            [sibling("model-Q4_K_M.gguf", size=feed.MAX_CHAT_BYTES + 1)],
            [dict(rfilename="model-Q4_K_M.gguf")],
            [sibling("model-skeleton.gguf")],
        ):
            self.assertIsNone(feed.choose_chat_file(dict(siblings=siblings)))
        with self.assertRaises(ValueError):
            feed.resolve_file(entries[0], manifest=dict(sha="f" * 40, siblings=[]))

    def test_list_is_cached_refreshed_when_old_and_survives_an_outage(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []

            def items(*selection):
                calls.append(selection)
                return [civitai_item(len(calls))]

            def load(now, **options):
                return feed.load(
                    "Civitai",
                    "Image models",
                    "Most downloaded",
                    "This week",
                    cache_dir=folder,
                    is_offline=options.pop("is_offline", lambda: False),
                    fetch_items=options.pop("fetch_items", items),
                    now=now,
                    **options,
                )

            first = load(1000.0)
            self.assertEqual(
                (len(calls), first["entries"][0]["key"], first["error"]), (1, "civitai:10", "")
            )
            self.assertEqual(
                load(1000.0 + feed.MAX_AGE_SECONDS - 1)["entries"][0]["key"], "civitai:10"
            )
            self.assertEqual(len(calls), 1)
            # Old enough, or asked for: fetched again without anyone editing a list.
            self.assertEqual(
                load(1000.0 + feed.MAX_AGE_SECONDS + 1)["entries"][0]["key"], "civitai:20"
            )
            self.assertEqual(
                load(1000.0 + feed.MAX_AGE_SECONDS + 2, force=True)["entries"][0]["key"],
                "civitai:30",
            )
            later = 1000.0 + 3 * feed.MAX_AGE_SECONDS

            def broken(*selection):
                raise OSError("no route")

            failed = load(later, fetch_items=broken)
            self.assertEqual(failed["entries"][0]["key"], "civitai:30")
            self.assertTrue(failed["stale"])
            self.assertIn("could not be refreshed", failed["error"])
            offline = load(later, is_offline=lambda: True, fetch_items=broken)
            self.assertTrue(offline["stale"])
            self.assertIn("Offline mode", offline["error"])
            self.assertEqual(len(calls), 3)
            # A different selection has its own list.
            other = feed.load(
                "Civitai",
                "LoRAs",
                "Newest",
                "This week",
                cache_dir=folder,
                is_offline=lambda: True,
                now=later,
            )
            self.assertEqual((other["entries"], other["stale"]), ([], False))
        with self.assertRaises(ValueError):
            feed.load("Civitai", "Chat models", "Newest", cache_dir=folder)

    def test_saved_list_is_shown_at_once_and_artwork_is_fetched_separately(self):
        import io

        from PIL import Image

        picture = io.BytesIO()
        Image.new("RGB", (640, 480), "navy").save(picture, "PNG")
        with tempfile.TemporaryDirectory() as folder:
            chosen = ("Civitai", "Image models", "Most downloaded", "This week")
            self.assertIsNone(feed.peek(*chosen, cache_dir=folder))
            items = [civitai_item(1), civitai_item(2, version=dict(images=[]))]
            result = feed.load(
                *chosen,
                cache_dir=folder,
                is_offline=lambda: False,
                fetch_items=lambda *selection: items,
                now=1000.0,
            )
            first, second = result["entries"]
            target = feed.thumbnail_path(first, folder)
            self.assertFalse(target.exists())
            self.assertIsNone(feed.thumbnail_path(second, folder))
            self.assertFalse(feed.peek(*chosen, cache_dir=folder, now=1001.0)["stale"])
            self.assertTrue(
                feed.peek(*chosen, cache_dir=folder, now=1000.0 + feed.MAX_AGE_SECONDS + 1)["stale"]
            )
            asked = []

            def fetch(url):
                asked.append(url)
                return picture.getvalue()

            original, feed.offline = feed.offline, lambda: False
            try:
                self.assertEqual(feed.fetch_artwork(result["entries"], folder, fetch), 1)
                # Cached pictures are not requested again; a failed one keeps its placeholder.
                self.assertEqual(feed.fetch_artwork(result["entries"], folder, fetch), 0)
                self.assertEqual(asked, [first["image"]])
                other = dict(first, image="https://image.civitai.com/a/b/width=320/9.jpeg")
                self.assertEqual(
                    feed.fetch_artwork([other], folder, lambda url: b"not a picture"), 0
                )
            finally:
                feed.offline = original
            with Image.open(target) as saved:
                self.assertLessEqual(max(saved.size), 512)
            manager = SimpleNamespace(
                model_paths=dict(
                    modelfile_path=[folder],
                    lorafile_path=[folder],
                    llm_path=folder,
                    cache_path=folder,
                )
            )
            cards = ui_popular.card_data(result["entries"], "", manager)["cards"]
            self.assertEqual([card["preview_is_default"] for card in cards], [False, True])

    def test_download_goes_to_the_category_folder_with_the_listed_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manager = SimpleNamespace(
                model_paths=dict(
                    modelfile_path=[str(root / "checkpoints"), str(root / "other")],
                    lorafile_path=[str(root / "loras")],
                    llm_path=str(root / "llm"),
                    cache_path=str(root / "cache"),
                )
            )
            entry = feed.parse_civitai([civitai_item(1)], "Image models")[0]
            submitted = []

            class Service:
                def submit(self, request, progress=None):
                    submitted.append(request)
                    return SimpleNamespace(result=lambda timeout=None: None, done=lambda: True)

            original, feed.offline = feed.offline, lambda: False
            try:
                path = feed.download(entry, manager, service=Service())
                self.assertEqual(path, root / "checkpoints" / "model one.safetensors")
                request = submitted[0]
                self.assertEqual(request.sha256, DIGEST)
                self.assertEqual(Path(request.destination), path.resolve())
                self.assertFalse(ui_popular.is_installed(entry, manager))
                path.write_bytes(b"x")
                # Present already: reported as installed and never fetched again.
                self.assertTrue(ui_popular.is_installed(entry, manager))
                self.assertEqual(feed.download(entry, manager, service=Service()), path)
                self.assertEqual(len(submitted), 1)
                lora = dict(entry, category="LoRAs")
                self.assertEqual(feed.installed_path(lora, manager).parent, root / "loras")
                feed.offline = lambda: True
                with self.assertRaises(ValueError):
                    feed.download(lora, manager, service=Service())
            finally:
                feed.offline = original

    def test_sign_in_failures_say_what_to_do(self):
        entry = dict(source="Civitai", page="https://civitai.com/models/1")
        error = OSError("401")
        error.response = SimpleNamespace(status_code=401)
        self.assertIn("signed-in", feed.explain_failure(error, entry))
        self.assertIn("https://civitai.com/models/1", feed.explain_failure(error, entry))
        self.assertIn("did not finish", feed.explain_failure(OSError("reset"), entry))

    def test_cards_show_base_model_downloads_and_installed_state(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = SimpleNamespace(
                model_paths=dict(
                    modelfile_path=[folder],
                    lorafile_path=[folder],
                    llm_path=folder,
                    cache_path=folder,
                )
            )
            entries = feed.parse_civitai(
                [civitai_item(1), civitai_item(2, version=dict(baseModel="Illustrious"))],
                "Image models",
            )
            self.assertEqual(ui_popular.families(entries), ["Anima", "Illustrious"])
            value = ui_popular.card_data(entries, "civitai:20", manager, "Illustrious")
            self.assertEqual([card["key"] for card in value["cards"]], ["civitai:20"])
            card = value["cards"][0]
            self.assertEqual((card["kind"], card["caption"]), ("Illustrious", "Illustrious"))
            self.assertIn("↓ 1.5k", card["state"])
            self.assertTrue(card["preview_is_default"])
            self.assertEqual(value["selected"], "civitai:20")
            # A selection that the filter hides is dropped.
            self.assertEqual(
                ui_popular.card_data(entries, "civitai:10", manager, "Illustrious")["selected"], ""
            )
            self.assertIn("Base model: Anima", ui_popular.details_text(entries[0], manager))
        self.assertEqual(ui_popular.short_count(1_250_000), "1.2M")
        self.assertEqual(ui_popular.size_text(3 * 1024**3), "3.0 GB")


if __name__ == "__main__":
    unittest.main()
