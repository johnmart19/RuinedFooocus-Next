"""Actual bounded PDF/OOXML extraction, inert content, citations and owner fences."""

from base64 import b64encode
from concurrent.futures import CancelledError
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape
import zipfile
import zlib

from pypdf import PdfReader, PdfWriter

from ruinedfooocus.features.chat.attachments import AttachmentSession
from ruinedfooocus.features.chat.attachment_retrieval import (
    InlineAttachmentTurn,
    add_references,
    prepare_passages,
)
from ruinedfooocus.features.chat.document_extraction import extract_document


OWNER = ("ui", "", "local", "documents-fixture")


def upload(name, data):
    return {"name": name, "data": b64encode(data).decode("ascii")}


def docx(text="Код Café: 0007, сума 0.000123456789.", *, xml=None, extras=None):
    source = (
        xml
        if xml is not None
        else (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p></w:body></w:document>"
        ).encode("utf-8")
    )
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", source)
        for name, content in (extras or {}).items():
            archive.writestr(name, content)
    return output.getvalue()


def pdf(*texts, compressed=False, raw_stream=None):
    """Tiny valid PDF fixture generator; production parsing always uses pypdf."""
    font = 3 + 2 * len(texts)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Count {len(texts)} /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(len(texts)))}] >>".encode(),
    ]
    for index, text in enumerate(texts):
        content = (
            raw_stream
            if raw_stream is not None
            else b"BT /F1 12 Tf 50 700 Td ("
            + text.encode("cp1252")
            .replace(b"\\", b"\\\\")
            .replace(b"(", b"\\(")
            .replace(b")", b"\\)")
            + b") Tj ET"
        )
        if compressed:
            content = zlib.compress(content)
        objects.extend(
            [
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {4 + 2 * index} 0 R >>".encode(),
                f"<< /Length {len(content)}{' /Filter /FlateDecode' if compressed else ''} >>\nstream\n".encode()
                + content
                + b"\nendstream",
            ]
        )
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    output = bytearray(b"%PDF-1.7\n")
    offsets = [0]
    for number, value in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode() + value + b"\nendobj\n")
    start = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    return bytes(output)


class DocumentAttachmentTests(unittest.TestCase):
    def test_actual_pdf_and_docx_preserve_unicode_numeric_precision_and_pages(self):
        session = AttachmentSession(OWNER)
        session.attach(
            0,
            [
                upload("report.PDF", pdf("Café code 0007", "Budget 0.000123456789")),
                upload("notes.docx", docx()),
            ],
        )
        report, notes = session.turn().files
        self.assertEqual(report.text, "[Page 1]\nCafé code 0007\n[Page 2]\nBudget 0.000123456789")
        self.assertEqual(report.pages, ((1, 2, 1), (3, 4, 2)))
        self.assertEqual(notes.text, "Код Café: 0007, сума 0.000123456789.")
        candidates = prepare_passages(session.turn(), "Café code and Budget", lambda: None)
        refs = [passage.reference() for passage in candidates if passage.name == report.name]
        self.assertEqual({ref["page"] for ref in refs}, {1, 2})
        self.assertTrue(all(ref["end"] - ref["start"] <= 1 for ref in refs))
        messages = [{"role": "user", "content": "Café code and Budget"}]
        admitted = add_references(messages, candidates, {"llm_n_ctx": 4096, "llm_n_predict": 256})
        self.assertIn({"id": "D1", "name": "report.PDF", "start": 1, "end": 2, "page": 1}, admitted)
        self.assertNotIn("0007", repr(session.turn()))

    def test_docx_links_fields_and_embeds_are_never_resolved_or_executed(self):
        xml = b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:hyperlink><w:r><w:t>Visible link</w:t></w:r></w:hyperlink><w:r><w:instrText>INCLUDETEXT file:///secret</w:instrText><w:t> 0007</w:t></w:r></w:p></w:body></w:document>'
        text, _ = extract_document(
            docx(
                xml=xml,
                extras={
                    "word/_rels/document.xml.rels": '<Relationship Target="https://example.invalid/private" TargetMode="External"/>',
                    "word/embeddings/ignored.bin": b"pickle is not loaded",
                },
            ),
            ".docx",
        )
        self.assertEqual(text, "Visible link 0007")

    def test_invalid_encrypted_empty_and_oversized_documents_are_atomic(self):
        encrypted = BytesIO()
        writer = PdfWriter()
        writer.add_page(PdfReader(BytesIO(pdf("secret"))).pages[0])
        writer.encrypt("private")
        writer.write(encrypted)
        session = AttachmentSession(OWNER)
        session.attach(0, [upload("keep.txt", b"Keep")])
        before = session.turn()
        invalid = [
            ("bad.pdf", b"not PDF", "PDF"),
            ("locked.pdf", encrypted.getvalue(), "Encrypted"),
            ("scan.pdf", pdf(""), "no extractable text"),
            ("pages.pdf", pdf(*["text"] * 33), "32 pages"),
            ("bomb.pdf", pdf("", compressed=True, raw_stream=b" " * (2 * 1024 * 1024 + 1)), "PDF"),
            ("empty.docx", docx(""), "no extractable"),
            ("bad.docx", b"not ZIP", "DOCX"),
            ("bomb.docx", docx(extras={"huge.txt": "x" * (2 * 1024 * 1024)}), "archive limits"),
            ("escape.docx", docx(extras={"../outside": "no"}), "DOCX"),
            ("text.docx", docx("é" * 140000), "256 KiB"),
        ]
        for name, raw, message in invalid:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                session.attach(1, [upload("partial.txt", b"Must not publish"), upload(name, raw)])
            self.assertEqual(session.turn(), before)

    def test_entity_and_dtd_inputs_are_rejected_in_utf8_and_utf16(self):
        xml = '<!DOCTYPE w:document [<!ENTITY x SYSTEM "file:///private">]><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>&x;</w:t></w:r></w:p></w:body></w:document>'
        for encoding in ("utf-8", "utf-16"):
            with self.subTest(encoding=encoding), self.assertRaisesRegex(ValueError, "DOCX"):
                extract_document(docx(xml=xml.encode(encoding)), ".docx")

    def test_api_and_manifest_keep_inline_extracted_text_without_parser_state(self):
        source = upload("notes.docx", docx("Literal 0007"))
        turn = InlineAttachmentTurn.capture(
            [source], [{"role": "user", "content": "What is literal?"}]
        )
        self.assertEqual(turn.files[0].text, "Literal 0007")
        session = AttachmentSession(OWNER)
        session.attach(0, [upload("report.pdf", pdf("Literal 0007"))])
        exported = session.export_manifest(1)
        self.assertEqual(set(json.loads(exported)["sources"][0]), {"name", "text"})
        session.clear()
        with patch(
            "ruinedfooocus.features.chat.attachments.extract_document",
            side_effect=AssertionError("manifest must not reparse files"),
        ):
            session.attach(2, [upload("saved.rag", exported.encode())])
        self.assertEqual(session.files[0].pages, ())
        self.assertIn("0007", session.files[0].text)

    def test_reset_during_extraction_rejects_stale_publish_and_cancel_starts_no_child(self):
        session = AttachmentSession(OWNER)

        def extraction(raw, suffix, check):
            session.clear()
            check()

        with patch(
            "ruinedfooocus.features.chat.attachments.extract_document", side_effect=extraction
        ):
            with self.assertRaisesRegex(ValueError, "Attachments changed"):
                session.attach(0, [upload("notes.docx", docx())])
        self.assertEqual(session.files, ())

        def cancel():
            raise CancelledError("Stopped")

        with self.assertRaises(CancelledError):
            extract_document(docx(), ".docx", cancel)


if __name__ == "__main__":
    unittest.main()
