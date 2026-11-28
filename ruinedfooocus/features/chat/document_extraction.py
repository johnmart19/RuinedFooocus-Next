"""Prepare bounded document text outside compute; never install or resolve source paths."""

from concurrent.futures import CancelledError
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys
import time

from ruinedfooocus.core.bounded_subprocess import run_bounded
from ruinedfooocus.core.paths import resource_path


DOCUMENT_SUFFIXES = (".pdf", ".docx")
MAX_DOCUMENT_BYTES = 256 * 1024
_MEMORY_LIMIT = 512 * 1024 * 1024
_ERRORS = {
    "parser-unavailable": "The declared PDF parser is unavailable. Restart online without --skip-dependency-install to prepare application dependencies.",
    "encrypted-pdf": "Encrypted PDFs are unsupported. Attach an unencrypted text PDF.",
    "scanned-pdf": "This PDF has no extractable text. OCR and scanned PDFs are unsupported.",
    "page-limit": "Attach a PDF with at most 32 pages.",
    "pdf-limit": "This PDF exceeds the safe text-extraction limits.",
    "archive-limit": "This DOCX exceeds the safe XML or archive limits.",
    "text-limit": "Extracted document text must be 256 KiB or smaller.",
    "input-limit": "Each attachment must be 256 KiB or smaller.",
    "empty-docx": "This DOCX has no extractable document text.",
    "invalid-pdf": "The PDF could not be read as a supported text document.",
    "invalid-docx": "The DOCX could not be read as a supported text document.",
}


def document_capabilities():
    """Read parser eligibility without launching or probing the parser child."""
    expected = json.loads(resource_path("document_parsers.json").read_text(encoding="utf-8"))["pdf"]
    try:
        pdf_available = version(expected["package"]) == expected["version"]
    except PackageNotFoundError:
        pdf_available = False
    return {
        "suffixes": ((".pdf",) if pdf_available else ()) + (".docx",),
        "pdf_available": pdf_available,
        "pdf_reason": "" if pdf_available else _ERRORS["parser-unavailable"],
        "notes": "Text PDF only, at most 32 pages; scanned/encrypted PDFs and OCR are unsupported. DOCX document text only.",
    }


def extract_document(raw, suffix, check_cancelled=None):
    if (
        suffix not in DOCUMENT_SUFFIXES
        or not isinstance(raw, bytes)
        or len(raw) > MAX_DOCUMENT_BYTES
    ):
        raise ValueError(_ERRORS["input-limit"])
    from ruinedfooocus.app.preparation_admission import preparation_admission

    deadline = time.monotonic() + 5

    def prepare():
        contract = json.loads(resource_path("document_parsers.json").read_text(encoding="utf-8"))
        try:
            output, _ = run_bounded(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("_document_worker.py")),
                    suffix,
                    contract["pdf"]["version"],
                ],
                input_data=raw,
                timeout=5,
                deadline=deadline,
                max_output=2 * MAX_DOCUMENT_BYTES + 8192,
                check_cancelled=check_cancelled,
                memory_limit=_MEMORY_LIMIT,
            )
            result = json.loads(output)
        except CancelledError:
            raise
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            raise ValueError(
                "Document extraction stopped or exceeded its time or resource limit."
            ) from error
        if not isinstance(result, dict) or set(result) not in ({"error"}, {"text", "pages"}):
            raise ValueError("The document parser returned an invalid result.")
        if "error" in result:
            raise ValueError(_ERRORS.get(result["error"], "The document could not be read safely."))
        text, pages = result["text"], result["pages"]
        if (
            not isinstance(text, str)
            or len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES
            or not isinstance(pages, list)
            or len(pages) > 32
        ):
            raise ValueError("The document parser returned an invalid result.")
        previous = 0
        for start, end, page in pages:
            if (
                any(type(value) is not int for value in (start, end, page))
                or not previous < start <= end <= text.count("\n") + 1
                or not 1 <= page <= 32
            ):
                raise ValueError("The document parser returned invalid page references.")
            previous = end
        return text, tuple(tuple(location) for location in pages)

    return preparation_admission.submit_work(prepare).result()
