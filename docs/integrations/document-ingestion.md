# Private document ingestion

## PDF dependency contract

The text-only PDF adapter uses pypdf 7.1.0, declared in `pip/modules.txt` and
`resources/document_parsers.json`. The official
[PyPI version metadata](https://pypi.org/pypi/pypdf/7.1.0/json), checked
2026-11-13, declares Python >=3.10, BSD-3-Clause and one platform-independent
pure-Python wheel, 418,698 bytes, SHA-256
`47cca6b24bdcf4c4d65b5f7b8b972535c61710a061261e737688777cfd9a7bfb`.
Python 3.10 additionally uses the already-declared typing-extensions dependency.
No crypto, image, OCR, Brotli or font extras are requested.

The existing unified package plan owns installation and its resolved hash closure.
Offline and `--skip-dependency-install` retain their normal acquisition refusal.
Attaching a file never installs a package. A missing or different parser version
produces an explicit preparation error; encrypted PDFs and OCR are unsupported.
Do not change application Python or replace unrelated packages for this adapter.

The [official extraction guide](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
warns that text extraction parses entire content streams and can consume excessive
memory. The adapter must therefore combine input/page/decompression/output limits
with a cancellable isolated process; file size alone is insufficient.
Context7 ID `/py-pdf/pypdf/6.21.4` provided the narrow API lookup; the actual
7.1.0 [reader](https://github.com/py-pdf/pypdf/blob/7.1.0/pypdf/_reader.py) and
[configuration](https://github.com/py-pdf/pypdf/blob/7.1.0/pypdf/_configuration.py)
sources establish the shipped version's public configuration API.

## Capture, extraction and lifecycle

The existing owner-only native file picker and the optional `rf_attachments`
API field accept `.pdf` and `.docx` through the same attachment decoder. There is
no new upload cache, filesystem/path resolver, remote fetch, pickle or document
execution. `chat/document_extraction.py` uses the bounded I/O preparation lane;
`_document_worker.py` receives bytes on stdin and returns small UTF-8 JSON.
Source IDs, owner/tab capability, conversation and revision fences remain with
`attachments.py`; a failed batch publishes no partial sources.

Each original and its extracted text must fit 256 KiB; the conversation retains
the larger byte count for each source within the existing 1 MiB/four-file bound.
The worker has a five-second deadline and 512 MiB process-memory cap (Windows
Job Object / POSIX address space), and the canonical
[child owner](bounded-child-operations.md) bounds output and terminates descendants
on failure or cancellation. The child denies external file opens, sockets and
process execution after loading its trusted parser. This is an additional Python
guard, not an OS security sandbox. Parsing never grants shared compute admission.

PDF uses `PdfReader(BytesIO(...), strict=True)` and public
`apply_configuration` limits for declared/decoded streams, arrays/dictionaries,
filters, parsing work, page trees and form invocations. Decoded streams cap at
2 MiB, pages at 32 and external JBIG2 decoding is disabled. Text extraction
retains Unicode and numeric strings, inserts trusted page boundaries and supplies
page metadata to passage/citation records. It does not repair complex reading
order, recover scanned text or decrypt a PDF.

DOCX uses standard-library ZIP/Expat, with at most 256 entries, 2 MiB total
declared expanded size, 1 MiB `word/document.xml`, 64 XML levels and 100,000
elements. It reads only body text, paragraph/table separators and explicit breaks.
DTD/entities, encrypted ZIP entries, unsafe/duplicate member names and unsupported
compression are rejected. External relationships, field instructions, macros,
embedded objects, headers, footnotes and page layout are never interpreted.

Original binary bytes leave scope after extraction. Text remains transient owner
conversation/request data, outside model settings, saved chat history and activity
summaries. `.rag` v1 exports inline extracted text with the original plain filename;
it contains no binary payload or parser/page state. Import never reparses or fetches
a document, and citations then refer to extracted text lines only.

## CPU evidence and remaining real-model cases

`test_document_attachments.py` uses actual pypdf 7.1.0 and OOXML through the real
isolated child. It covers Cyrillic/Latin Unicode, exact numeric strings, PDF page
references, archive/decompression limits, DTD/entity denial including UTF-16,
encrypted/empty/scanned/corrupt input, ignored relationships/embeds, atomic batches,
manifest round trips and conversation reset/cancel. The Chat API suite exercises
the real parsers through both JSON and SSE with a fake completion/embedding stage.
Those mocks certify transport and privacy, not grounding semantics or a GPU.

```powershell
python tools/check_cpu.py test_document_parser_dependency.py test_document_attachments.py test_chat_attachments.py test_rag_manifest.py test_chat_api_attachments.py
```

The 7.1.0 update passed these bounded extraction cases against its digest-verified
private wheel extraction before application-environment installation. Parser
version eligibility remains exact; an environment still on 7.0.1 does not
advertise PDF support until its normal dependency update completes.
Model-based answer quality, supported embedded/direct-vision paths and GPU
embeddings remain in [the acceptance inventory](remaining-acceptance.md).
