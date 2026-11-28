"""Exercise automatic Inbox sorting with temporary folders and real tiny containers."""

import errno
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gguf import GGUFWriter
import numpy as np
from safetensors.numpy import save_file

from ruinedfooocus.features.models import model_inbox


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.inbox = self.root / "inbox"
        self.inbox.mkdir()
        self.manager = SimpleNamespace(
            model_paths={"cache_path": self.root / "cache"},
            DOWNLOADABLE_FILES={},
            get_model_download_folder=lambda category: self.root / category,
        )

    def lora(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "lora_unet_test.lora_down.weight": np.zeros((2, 3), dtype=np.float32),
                "lora_unet_test.lora_up.weight": np.zeros((4, 2), dtype=np.float32),
            },
            path,
        )

    def gguf(self, path):
        writer = GGUFWriter(path, "qwen3")
        writer.add_string("tokenizer.chat_template", "{{ messages }}")
        for name in ("token_embd.weight", "blk.0.attn_q.weight"):
            writer.add_tensor(name, np.zeros((2, 2), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def scan(self, sorter):
        return sorter.scan([self.inbox])

    def test_stable_models_and_sidecars_sort_offline_after_arrival(self):
        sorter = model_inbox.InboxSorter(self.manager, offline=True)
        self.assertEqual(self.scan(sorter), 0)
        lora = self.inbox / "collection" / "detail.SFT"
        self.lora(lora)
        artwork = lora.with_suffix(".preview.webp")
        artwork.write_bytes(b"preview fixture")
        embedding = self.inbox / "concept.safetensors"
        save_file({"clip_l": np.zeros((2, 768), dtype=np.float32)}, embedding)
        chat = self.inbox / "chat.gguf"
        self.gguf(chat)
        chat_artwork = self.inbox / "chat.gguf.preview.png"
        chat_artwork.write_bytes(b"exact preview fixture")
        before = [path.read_bytes() for path in (lora, embedding, chat)]
        with patch("ruinedfooocus.features.models.model_sources.civitai_metadata") as remote:
            self.assertEqual(self.scan(sorter), 0)
            self.assertEqual(self.scan(sorter), 3)
            remote.assert_not_called()
        moved = [
            self.root / "loras" / "collection" / lora.name,
            self.root / "embeddings" / embedding.name,
            self.root / "llm" / chat.name,
        ]
        self.assertEqual([path.read_bytes() for path in moved], before)
        self.assertEqual(moved[0].with_suffix(".preview.webp").read_bytes(), b"preview fixture")
        self.assertEqual(
            (moved[2].parent / chat_artwork.name).read_bytes(), b"exact preview fixture"
        )
        self.assertTrue(
            all(not path.exists() for path in (lora, artwork, embedding, chat, chat_artwork))
        )

    def test_catalogue_hash_resolves_auxiliary_role_without_name_guess(self):
        path = self.inbox / "renamed.pth"
        path.write_bytes(b"fixture")
        self.manager.DOWNLOADABLE_FILES["upscalers/upscale.pth"] = {
            "filename": "upscale.pth",
            "path": "path_upscalers",
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        sorter = model_inbox.InboxSorter(self.manager, offline=True)
        self.assertEqual(self.scan(sorter), 0)
        self.assertEqual(self.scan(sorter), 1)
        self.assertEqual((self.root / "upscalers" / path.name).read_bytes(), b"fixture")

    def test_gguf_adapter_is_not_an_image_diffusion_weight(self):
        path = self.inbox / "detail.gguf"
        writer = GGUFWriter(path, "sdxl")
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        for name in ("layer.lora_a", "layer.lora_b"):
            writer.add_tensor(name, np.zeros((2, 2), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        sorter = model_inbox.InboxSorter(self.manager, offline=True)
        self.assertEqual(self.scan(sorter), 0)
        self.assertEqual(self.scan(sorter), 1)
        self.assertTrue((self.root / "loras" / path.name).is_file())
        self.assertFalse((self.root / "diffusion_models" / path.name).exists())

    def test_online_fallback_uses_only_exact_publisher_hash(self):
        known = self.inbox / "known.pt"
        unknown = self.inbox / "different.pt"
        known.write_bytes(b"known")
        unknown.write_bytes(b"other")
        digest = hashlib.sha256(known.read_bytes()).hexdigest()
        sorter = model_inbox.InboxSorter(self.manager, offline=False)
        with patch(
            "ruinedfooocus.features.models.model_sources.civitai_metadata",
            return_value={
                "model": {"type": "Checkpoint"},
                "files": [{"hashes": {"SHA256": digest}}],
            },
        ) as remote:
            self.assertEqual(self.scan(sorter), 0)
            self.assertEqual(self.scan(sorter), 1)
            remote.assert_any_call(f"model-versions/by-hash/{digest}")
        self.assertTrue(unknown.is_file())
        self.assertEqual((self.root / "checkpoints" / known.name).read_bytes(), b"known")

    def test_bundle_shard_and_partial_files_stay_while_other_models_sort(self):
        bundle = self.inbox / "bundle"
        bundle.mkdir()
        (bundle / "model_index.json").write_text("{}", encoding="utf-8")
        nested = bundle / "unet" / "candidate.safetensors"
        self.lora(nested)
        partial = self.inbox / "partial.safetensors"
        partial.write_bytes(b"incomplete")
        shard = self.inbox / "shared-00001-of-00002.safetensors"
        self.lora(shard)
        ready = self.inbox / "ready.safetensors"
        self.lora(ready)
        sorter = model_inbox.InboxSorter(self.manager, offline=True)
        self.assertEqual(self.scan(sorter), 0)
        self.assertEqual(self.scan(sorter), 1)
        self.assertTrue(all(path.is_file() for path in (nested, partial, shard)))
        self.lora(partial)
        self.assertEqual(self.scan(sorter), 0)
        self.assertEqual(self.scan(sorter), 1)
        self.assertTrue((self.root / "loras" / partial.name).is_file())

    def test_sidecar_conflict_keeps_both_models_and_artwork(self):
        lora = self.inbox / "existing.safetensors"
        self.lora(lora)
        artwork = lora.with_suffix(".png")
        artwork.write_bytes(b"incoming preview")
        target = self.root / "loras"
        target.mkdir()
        (target / artwork.name).write_bytes(b"existing preview")
        sorter = model_inbox.InboxSorter(self.manager, offline=True)
        self.scan(sorter)
        self.assertEqual(self.scan(sorter), 0)
        self.assertTrue(lora.is_file())
        self.assertEqual(artwork.read_bytes(), b"incoming preview")
        self.assertEqual((target / artwork.name).read_bytes(), b"existing preview")
        self.assertFalse((target / lora.name).exists())

    def test_racing_destination_keeps_sources_and_reverts_own_sidecar(self):
        source, sidecar = self.inbox / "file.sft", self.inbox / "file.png"
        source.write_bytes(b"incoming")
        sidecar.write_bytes(b"preview")
        target = self.root / "loras"
        target.mkdir()
        destination = target / source.name
        publish = model_inbox._publish

        def collision(src, dest):
            if dest == destination:
                dest.write_bytes(b"already placed")
            publish(src, dest)

        with patch.object(model_inbox, "_publish", side_effect=collision):
            with self.assertRaises(FileExistsError):
                model_inbox._move_group([(sidecar, target / sidecar.name), (source, destination)])
        self.assertEqual(destination.read_bytes(), b"already placed")
        self.assertFalse((target / sidecar.name).exists())
        self.assertEqual(source.read_bytes(), b"incoming")
        self.assertEqual(sidecar.read_bytes(), b"preview")

    def test_cross_volume_publish_keeps_existing_destination(self):
        source, target = self.inbox / "source.sft", self.root / "target.sft"
        source.write_bytes(b"new payload")
        link = model_inbox.os.link

        def cross_volume(src, dest):
            if src == source:
                raise OSError(errno.EXDEV, "cross-volume fixture")
            link(src, dest)

        with patch.object(model_inbox.os, "link", side_effect=cross_volume):
            model_inbox._publish(source, target)
            self.assertEqual(target.read_bytes(), b"new payload")
            source.write_bytes(b"second payload")
            with self.assertRaises(FileExistsError):
                model_inbox._publish(source, target)
        self.assertEqual(target.read_bytes(), b"new payload")
        self.assertEqual(source.read_bytes(), b"second payload")
