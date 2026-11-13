"""Portable Chat manifests contain only bounded inline source names and text."""

from base64 import b64encode
import json
import unittest

from ruinedfooocus.features.chat.attachments import (
    AttachmentSession,
    MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
)


def upload(name, content):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return {"name": name, "data": b64encode(content).decode("ascii")}


def manifest(*sources, version=1):
    return json.dumps({"version": version, "sources": list(sources)}, ensure_ascii=False)


class RagManifestTests(unittest.TestCase):
    def setUp(self):
        self.session = AttachmentSession(("ui", "", "local", "fixture"))

    def test_roundtrip_preserves_text_order_and_duplicate_names_and_regenerates_ids(self):
        contents = [
            {"name": "café.md", "text": "# Café\r\nFirst source."},
            {
                "name": "café.md",
                "text": "Look up file://example.invalid/note and https://example.invalid as text.",
            },
        ]
        self.session.attach(0, [upload("collection.RAG", manifest(*contents))])
        exported = self.session.export_manifest(1)
        contents[0]["text"] = contents[0]["text"].replace("\r\n", "\n")
        self.assertEqual(json.loads(exported), {"version": 1, "sources": contents})
        self.session.clear()
        self.session.attach(2, [upload("previous.txt", "Already attached")])
        restored = self.session.attach(3, [upload("collection.rag", exported)])
        self.assertEqual(
            restored["files"],
            [
                {"id": "D1", "name": "previous.txt"},
                {"id": "D2", "name": "café.md"},
                {"id": "D3", "name": "café.md"},
            ],
        )

    def test_invalid_manifest_and_mixed_selection_are_atomic(self):
        self.session.attach(0, [upload("existing.txt", "Keep this source")])
        before = self.session.turn()
        good = {"name": "one.txt", "text": "A valid source"}
        invalid = [
            (b"\xff\xfe", "UTF-8"),
            ("{", "valid UTF-8 JSON"),
            ("[" * 1200 + "]" * 1200, "manifest"),
            ('{"version":1,"version":1,"sources":[]}', "unique fields"),
            ('{"version":1,"sources":[{"name":"one.txt","text":"a","text":"b"}]}', "unique fields"),
            ('{"version":1,"sources":[],"vectors":[]}', "only version and sources"),
            (manifest(good, version=True), "unsupported"),
            (manifest(good, version=2), "unsupported"),
            (manifest(), "one to four"),
            (manifest(*([good] * 5)), "one to four"),
            (manifest(good, {"name": "two.txt", "url": "https://example.invalid"}), "inline text"),
            (
                manifest(good, {"name": "two.txt", "text": "Inline", "path": "notes.txt"}),
                "inline text",
            ),
            (manifest(good, {"name": "../escape.txt", "text": "Inert"}), "plain filename"),
            (manifest(good, {"name": "C:\\escape.txt", "text": "Inert"}), "plain filename"),
            (manifest(good, {"name": "nested.rag", "text": "Inert"}), "plain filename"),
            ('{"version":1,"sources":[{"name":"one.txt","text":"\\ud800"}]}', "invalid Unicode"),
            (manifest(good, {"name": "two.txt", "text": "\x00"}), "binary control"),
            (
                manifest(good, {"name": "two.txt", "text": "é" * (MAX_FILE_BYTES // 2 + 1)}),
                "256 KiB",
            ),
            (manifest(*([good] * 4)), "more than four"),
        ]
        for content, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.session.attach(1, [upload("collection.rag", content)])
            self.assertEqual(self.session.turn(), before)
        with self.assertRaisesRegex(ValueError, "by itself"):
            self.session.attach(
                1, [upload("collection.rag", manifest(good)), upload("two.txt", "Another source")]
            )
        with self.assertRaisesRegex(ValueError, "plain filename"):
            self.session.attach(1, [upload("../collection.rag", manifest(good))])
        with self.assertRaisesRegex(ValueError, "too large"):
            self.session.attach(1, [upload("collection.rag", b" " * (MAX_MANIFEST_BYTES + 1))])
        self.assertEqual(self.session.turn(), before)

    def test_csv_and_json_roundtrip_in_version_one_as_inline_text(self):
        contents = [
            {"name": "rows.csv", "text": "name,value\nCafé,7"},
            {"name": "metadata.JSON", "text": '{"$ref":"file:///unread.json",'},
        ]
        self.session.attach(0, [upload(item["name"], item["text"]) for item in contents])
        exported = self.session.export_manifest(1)
        self.assertEqual(json.loads(exported), {"version": 1, "sources": contents})
        self.session.clear()
        self.session.attach(2, [upload("collection.rag", exported)])
        self.assertEqual(
            [(file.source, file.name, file.text) for file in self.session.turn().files],
            [(f"D{i}", item["name"], item["text"]) for i, item in enumerate(contents, 1)],
        )

    def test_full_escaped_corpus_roundtrips_inside_the_separate_manifest_limit(self):
        text = '"\\' * (MAX_FILE_BYTES // 2)
        selected = [upload(f"file{number}.txt", text) for number in range(4)]
        self.session.attach(0, selected)
        result = self.session.export_manifest(1)
        self.assertGreater(len(result.encode("utf-8")), MAX_FILE_BYTES)
        self.assertLessEqual(len(result.encode("utf-8")), MAX_MANIFEST_BYTES)
        self.session.clear()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.session.export_manifest(1)
        with self.assertRaisesRegex(ValueError, "before exporting"):
            self.session.export_manifest(2)
        self.session.attach(2, [upload("export.rag", result)])
        self.assertEqual(len(self.session.turn().files), 4)
        self.assertTrue(all(item.text == text for item in self.session.turn().files))

    def test_http_import_export_stays_owner_tab_and_revision_bound(self):
        from fastapi.testclient import TestClient
        import gradio as gr
        from gradio.routes import App
        from ruinedfooocus.features.chat.attachment_ui import AttachmentControls

        with gr.Blocks() as app:
            controls = AttachmentControls()
            relabel = gr.Button().click(
                lambda: gr.update(export_label="Exporter"), outputs=controls.files, queue=False
            )
        app.auth = [("fixture", "password")]
        app.auth_message = None
        server = App.create_app(app)
        first, other = TestClient(server), TestClient(server)
        key, session = "9" * 32, "manifest-fixture-tab"
        marker = "These private facts stay in the explicitly exported text."
        download = {
            "component_id": controls.files._id,
            "fn_name": "export_manifest",
            "session_hash": session,
            "data": {"key": key, "revision": 1},
        }
        route = "/gradio_api/component_server/"
        self.assertEqual(first.post(route, json=download).status_code, 401)
        for client in (first, other):
            self.assertEqual(
                client.post(
                    "/login", data={"username": "fixture", "password": "password"}
                ).status_code,
                200,
            )
        source = {"name": "private.txt", "text": marker}
        upload_response = first.post(
            "/gradio_api/run/attach",
            json={
                "data": [],
                "fn_index": 0,
                "session_hash": session,
                "trigger_id": controls.picker._id,
                "event_data": {
                    "key": key,
                    "revision": 0,
                    "files": [upload("private.rag", manifest(source))],
                },
            },
        )
        self.assertEqual(upload_response.status_code, 200, upload_response.text)
        self.assertNotIn(marker, upload_response.text)
        # The normal localization update replaces the component's session config.
        response = first.post(
            "/gradio_api/run/relabel",
            json={
                "data": [],
                "fn_index": relabel["id"],
                "session_hash": session,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        response = first.post(route, json=download)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            json.loads(response.json()["manifest"]), {"version": 1, "sources": [source]}
        )
        self.assertNotIn(marker, repr(list(server.state_holder[session].components)))
        for client, request in (
            (other, download),
            (first, {**download, "session_hash": "different-tab"}),
            (first, {**download, "data": {"key": "1" * 32, "revision": 1}}),
            (first, {**download, "data": {"key": key, "revision": 0}}),
        ):
            response = client.post(route, json=request)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("error", response.json())
            self.assertNotIn(marker, response.text)


if __name__ == "__main__":
    unittest.main()
