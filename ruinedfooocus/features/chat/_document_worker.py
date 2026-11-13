"""Private text-only parser child. Bytes enter and leave only through inherited pipes."""

from io import BytesIO
import json
import logging
from pathlib import PurePosixPath
import sys
from xml.parsers import expat
import zipfile


MAX_INPUT = MAX_TEXT = 256 * 1024
MAX_XML = 1024 * 1024
MAX_EXPANDED = 2 * 1024 * 1024
MAX_PAGES = 32
MEMORY_LIMIT = 512 * 1024 * 1024
_WORD_NS = (
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main}",
    "http://purl.oclc.org/ooxml/wordprocessingml/main}",
)


class DocumentError(ValueError):
    pass


def _text_limit(text):
    if len(text.encode("utf-8")) > MAX_TEXT:
        raise DocumentError("text-limit")
    return text


def _docx(raw):
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > 256 or sum(info.file_size for info in entries) > MAX_EXPANDED:
            raise DocumentError("archive-limit")
        names = set()
        for info in entries:
            path = PurePosixPath(info.filename)
            if (
                info.filename in names
                or path.is_absolute()
                or ".." in path.parts
                or any(char in info.filename for char in ("\\", ":", "\x00"))
                or info.flag_bits & 1
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            ):
                raise DocumentError("invalid-docx")
            names.add(info.filename)
        if "word/document.xml" not in names or "[Content_Types].xml" not in names:
            raise DocumentError("invalid-docx")
        document = archive.getinfo("word/document.xml")
        if document.file_size > MAX_XML:
            raise DocumentError("archive-limit")
        with archive.open(document) as stream:
            xml = stream.read(MAX_XML + 1)
        if len(xml) > MAX_XML:
            raise DocumentError("archive-limit")

    parser = expat.ParserCreate(namespace_separator="}")
    pieces = []
    used = elements = 0
    stack = []

    def add(text):
        nonlocal used
        used += len(text.encode("utf-8"))
        if used > MAX_TEXT:
            raise DocumentError("text-limit")
        pieces.append(text)

    def local(name):
        return next((name[len(ns) :] for ns in _WORD_NS if name.startswith(ns)), "")

    def start(name, attrs):
        nonlocal elements
        elements += 1
        stack.append(local(name))
        if len(stack) > 64 or elements > 100000:
            raise DocumentError("archive-limit")
        if len(stack) == 1 and stack[0] != "document":
            raise DocumentError("invalid-docx")
        if stack[-1] == "tab":
            add("\t")
        elif stack[-1] in ("br", "cr"):
            add("\n")

    def end(name):
        tag = stack.pop()
        if tag in ("p", "tr"):
            add("\n")
        elif tag == "tc":
            add("\t")

    def data(text):
        if stack and stack[-1] == "t" and "del" not in stack:
            add(text)

    def reject(*args):
        raise DocumentError("invalid-docx")

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = data
    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ExternalEntityRefHandler = reject
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.Parse(xml, True)
    text = "".join(pieces).strip("\n")
    if not text.strip():
        raise DocumentError("empty-docx")
    return _text_limit(text), []


def _pdf(raw, pypdf):
    if not raw.startswith(b"%PDF-"):
        raise DocumentError("invalid-pdf")
    # Public per-context limits, verified against the exact declared release.
    limits = dict(
        maximum_declared_stream_length=MAX_EXPANDED,
        array_based_stream_maximum_output_length=MAX_EXPANDED,
        array_maximum_element_count=20000,
        dictionary_maximum_element_count=20000,
        jbig2_maximum_output_length=MAX_EXPANDED,
        lzw_maximum_output_length=MAX_EXPANDED,
        run_length_maximum_output_length=MAX_EXPANDED,
        zlib_maximum_output_length=MAX_EXPANDED,
        zlib_maximum_recovery_input_length=MAX_INPUT,
        brotli_maximum_output_length=MAX_EXPANDED,
        flate_maximum_columns=4096,
        flate_maximum_row_length=MAX_EXPANDED,
        image_maximum_buffer_size=MAX_EXPANDED,
        stream_filters_maximum_length=4,
        stream_decoding_work_maximum_length=4 * MAX_EXPANDED,
        stream_parsing_work_maximum_count=100000,
        page_tree_maximum_entries=MAX_PAGES * 4,
        page_tree_maximum_depth=16,
        page_tree_maximum_page_entries=MAX_PAGES * 128,
        xform_maximum_invocations_per_extraction=64,
        jbig2dec_binary=None,
        disable_legacy_handling=True,
    )
    with pypdf.apply_configuration(**limits):
        reader = pypdf.PdfReader(BytesIO(raw), strict=True, xref_stream_maximum_entries=20000)
        if reader.is_encrypted:
            raise DocumentError("encrypted-pdf")
        count = len(reader.pages)
        if not 1 <= count <= MAX_PAGES:
            raise DocumentError("page-limit")
        parts, locations = [], []
        used = 0
        line = 1
        visible = False
        for number, page in enumerate(reader.pages, 1):
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > MAX_EXPANDED:
                raise DocumentError("pdf-limit")
            text = (
                (page.extract_text() or "")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\f", "\n")
                .strip("\n")
            )
            visible |= bool(text.strip())
            block = f"[Page {number}]\n{text}\n"
            used += len(block.encode("utf-8"))
            if used > MAX_TEXT:
                raise DocumentError("text-limit")
            parts.append(block)
            end = line + block.count("\n") - 1
            locations.append([line, end, number])
            line = end + 1
        if not visible:
            raise DocumentError("scanned-pdf")
        return "".join(parts).rstrip("\n"), locations


def _no_external_io(event, args):
    if (
        event == "open"
        or event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn"))
        or event == "os.system"
    ):
        raise PermissionError("External I/O is disabled in document extraction")


def main():
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    logging.disable(logging.CRITICAL)
    pypdf = None
    kind, expected_version = sys.argv[1:]
    if kind == ".pdf":
        try:
            import pypdf

            if pypdf.__version__ != expected_version:
                raise ImportError("Unsupported parser version")
        except ImportError:
            return {"error": "parser-unavailable"}
    raw = sys.stdin.buffer.read(MAX_INPUT + 1)
    if len(raw) > MAX_INPUT:
        return {"error": "input-limit"}
    # Preload the encodings used by text PDFs before closing filesystem access.
    for encoding in ("utf-8", "utf-16", "utf-16-be", "utf-16-le", "latin-1", "cp1252", "cp437"):
        "".encode(encoding)
    sys.addaudithook(_no_external_io)
    try:
        text, pages = _pdf(raw, pypdf) if kind == ".pdf" else _docx(raw)
        return {"text": text, "pages": pages}
    except DocumentError as error:
        return {"error": str(error)}
    except (MemoryError, RecursionError):
        return {"error": "pdf-limit" if kind == ".pdf" else "archive-limit"}
    except Exception as error:
        if pypdf is not None and isinstance(error, pypdf.errors.LimitReachedError):
            return {"error": "pdf-limit"}
        # Parser diagnostics can include document content. Never echo them to UI,
        # API errors, activity summaries or logs.
        return {"error": "invalid-pdf" if kind == ".pdf" else "invalid-docx"}


if __name__ == "__main__":
    sys.stdout.buffer.write(
        json.dumps(main(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
