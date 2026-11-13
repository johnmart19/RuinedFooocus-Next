"""Bounded, conversation-owned text attachments. No document paths or URLs."""

from base64 import b64decode
from binascii import Error as Base64Error
from collections import OrderedDict
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
import json
from pathlib import PurePath
import re
from threading import RLock
from time import monotonic
from unicodedata import category
from weakref import WeakSet

from ruinedfooocus.features.chat.document_extraction import DOCUMENT_SUFFIXES, extract_document


MAX_FILES = 4
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 2 * MAX_TOTAL_BYTES + 4096
MANIFEST_VERSION = 1
TEXT_FILE_SUFFIXES = (".txt", ".md", ".csv", ".json")
ATTACHMENT_SUFFIXES = TEXT_FILE_SUFFIXES + DOCUMENT_SUFFIXES
_TEXT_FILENAME_ERROR = (
    "Attach a .txt, .md, .csv, .json, text PDF or DOCX file with a plain filename."
)
_KEY = re.compile(r"[0-9a-f]{32}")


def attachment_capabilities():
    """One compact picker/help projection from the actual bounded parsers."""
    from ruinedfooocus.features.chat.document_extraction import document_capabilities

    documents = document_capabilities()
    suffixes = TEXT_FILE_SUFFIXES + documents["suffixes"] + (".rag",)
    return {
        "suffixes": suffixes,
        "accept": ",".join(suffixes),
        "label": ", ".join(suffixes),
        "limits": {
            "files": MAX_FILES,
            "file_bytes": MAX_FILE_BYTES,
            "total_bytes": MAX_TOTAL_BYTES,
        },
        "notes": documents["notes"] + " Select a .rag manifest by itself.",
        "pdf_available": documents["pdf_available"],
        "pdf_reason": documents["pdf_reason"],
    }


@dataclass(frozen=True)
class Attachment:
    source: str
    name: str
    text: str = field(repr=False)
    size: int
    pages: tuple = ()


def plain_filename(name, suffixes):
    return (
        isinstance(name, str)
        and 1 <= len(name) <= 120
        and name == name.strip()
        and not any(char in "/\\:" or category(char).startswith("C") for char in name)
        and PurePath(name).suffix.lower() in suffixes
    )


def decode_upload_bytes(payload, maximum, limit_error):
    if not isinstance(payload, str) or len(payload) > 4 * ((maximum + 2) // 3):
        raise ValueError(limit_error)
    try:
        raw = b64decode(payload, validate=True)
    except (Base64Error, ValueError) as error:
        raise ValueError("The attachment could not be read as UTF-8 text.") from error
    if len(raw) > maximum:
        raise ValueError(limit_error)
    return raw


def decode_upload(payload, maximum, limit_error):
    raw = decode_upload_bytes(payload, maximum, limit_error)
    try:
        return raw.decode("utf-8-sig"), len(raw)
    except UnicodeDecodeError as error:
        raise ValueError("The attachment could not be read as UTF-8 text.") from error


def text_source(name, text, size, pages=()):
    if not plain_filename(name, ATTACHMENT_SUFFIXES):
        raise ValueError(_TEXT_FILENAME_ERROR)
    if size > MAX_FILE_BYTES:
        raise ValueError("Each attachment must be 256 KiB or smaller.")
    if not text.strip() or any(category(char) == "Cc" and char not in "\n\r\t" for char in text):
        raise ValueError("Attach a nonempty UTF-8 text file without binary control characters.")
    return name, text.replace("\r\n", "\n").replace("\r", "\n"), size, pages


def _unique_fields(pairs):
    fields = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("A .rag manifest cannot contain duplicate fields.")
        fields[name] = value
    return fields


def _manifest_integer(token):
    # The version is the only integer in v1. Bound parsing on older Python too.
    return int(token) if len(token) <= 16 else None


def decode_manifest(file):
    if not plain_filename(file["name"], (".rag",)):
        raise ValueError("Use a plain filename for the .rag manifest.")
    encoded, _ = decode_upload(
        file.get("data"), MAX_MANIFEST_BYTES, "The .rag manifest is too large."
    )
    try:
        manifest = json.loads(
            encoded, object_pairs_hook=_unique_fields, parse_int=_manifest_integer
        )
    except (ValueError, RecursionError) as error:
        raise ValueError(
            "The .rag manifest must be valid UTF-8 JSON with unique fields."
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != {"version", "sources"}:
        raise ValueError("A .rag manifest must contain only version and sources.")
    version = manifest["version"]
    if type(version) is not int or version != MANIFEST_VERSION:
        raise ValueError("This .rag version is unsupported. Use version 1.")
    sources = manifest["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_FILES:
        raise ValueError("A .rag manifest must contain one to four text sources.")
    decoded = []
    for source in sources:
        if (
            not isinstance(source, dict)
            or set(source) != {"name", "text"}
            or not isinstance(source["text"], str)
        ):
            raise ValueError("Each .rag source must contain only a filename and inline text.")
        text = source["text"]
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError("The .rag source contains invalid Unicode text.") from error
        decoded.append(text_source(source["name"], text, size))
    return decoded


def decode_files(files, check_cancelled=None):
    if (
        not isinstance(files, list)
        or not 1 <= len(files) <= MAX_FILES
        or not all(isinstance(file, dict) for file in files)
    ):
        raise ValueError("Select one to four text attachments.")
    manifests = [
        file
        for file in files
        if isinstance(file.get("name"), str) and file["name"].lower().endswith(".rag")
    ]
    if manifests:
        if len(files) != 1:
            raise ValueError("Select one .rag manifest by itself.")
        return decode_manifest(manifests[0])
    decoded = []
    for file in files:
        name = file.get("name")
        if check_cancelled is not None:
            check_cancelled()
        if not plain_filename(name, ATTACHMENT_SUFFIXES):
            raise ValueError(_TEXT_FILENAME_ERROR)
        suffix = PurePath(name).suffix.lower()
        if suffix in DOCUMENT_SUFFIXES:
            raw = decode_upload_bytes(
                file.get("data"), MAX_FILE_BYTES, "Each attachment must be 256 KiB or smaller."
            )
            text, pages = extract_document(raw, suffix, check_cancelled)
            decoded.append(text_source(name, text, max(len(raw), len(text.encode("utf-8"))), pages))
        else:
            text, size = decode_upload(
                file.get("data"), MAX_FILE_BYTES, "Each attachment must be 256 KiB or smaller."
            )
            decoded.append(text_source(name, text, size))
    return decoded


@dataclass(frozen=True)
class AttachmentTurn:
    session: "AttachmentSession" = field(repr=False)
    conversation: int
    revision: int
    files: tuple[Attachment, ...] = field(repr=False)

    def __deepcopy__(self, memo):
        # The immutable request shares its server-side revocation fence.
        return self

    def check(self, owner, cancelled=None):
        self.session.check_turn(self, owner, cancelled)

    def current(self):
        return self.session.is_current(self)

    def same_conversation(self):
        return self.session.is_current(self, conversation_only=True)


class AttachmentSession:
    def __init__(self, owner):
        self.owner = owner
        self.lock = RLock()
        self.conversation = self.revision = self.sequence = 0
        self.files = ()
        self.active = WeakSet()

    def view(self, status=""):
        with self.lock:
            return {
                "revision": self.revision,
                "files": [{"id": file.source, "name": file.name} for file in self.files],
                "status": status,
            }

    def _advance(self):
        self.revision += 1
        for event in self.active:
            event.set()
        self.active.clear()

    def clear(self):
        with self.lock:
            self._advance()
            self.conversation += 1
            self.sequence = 0
            self.files = ()
            return self.view()

    def _require_revision(self, revision):
        if type(revision) is not int or revision != self.revision:
            raise ValueError("Attachments changed. Repeat the action in the current conversation.")

    def attach(self, revision, files):
        with self.lock:
            self._require_revision(revision)

        def check():
            with self.lock:
                self._require_revision(revision)

        decoded = decode_files(files, check)
        with self.lock:
            self._require_revision(revision)
            if len(self.files) + len(decoded) > MAX_FILES:
                raise ValueError("Remove an attachment before adding more than four files.")
            if (
                sum(file.size for file in self.files) + sum(file[2] for file in decoded)
                > MAX_TOTAL_BYTES
            ):
                raise ValueError("Attachments can use at most 1 MiB per conversation.")
            start = self.sequence + 1
            self.files += tuple(
                Attachment(f"D{number}", name, text, size, pages)
                for number, (name, text, size, pages) in enumerate(decoded, start)
            )
            self.sequence += len(decoded)
            self._advance()
            return self.view()

    def remove(self, revision, source):
        with self.lock:
            self._require_revision(revision)
            if not isinstance(source, str) or not any(file.source == source for file in self.files):
                raise ValueError("This attachment is no longer in the conversation.")
            self.files = tuple(file for file in self.files if file.source != source)
            self._advance()
            return self.view()

    def export_manifest(self, revision):
        with self.lock:
            self._require_revision(revision)
            if not self.files:
                raise ValueError("Attach text files before exporting a .rag manifest.")
            sources = [{"name": file.name, "text": file.text} for file in self.files]
            return json.dumps(
                {"version": MANIFEST_VERSION, "sources": sources},
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def stop(self):
        with self.lock:
            for event in self.active:
                event.set()

    def turn(self):
        with self.lock:
            return AttachmentTurn(self, self.conversation, self.revision, self.files)

    def is_current(self, turn, *, conversation_only=False):
        with self.lock:
            return self.conversation == turn.conversation and (
                conversation_only or self.revision == turn.revision
            )

    def check_turn(self, turn, owner, cancelled):
        with self.lock:
            if owner != self.owner:
                raise PermissionError("These attachments belong to a different Chat session.")
            if not self.is_current(turn):
                raise CancelledError("Chat attachments changed. Send the message again.")
            if cancelled is not None:
                if cancelled.is_set():
                    raise CancelledError("The Chat reply was stopped.")
                self.active.add(cancelled)


class AttachmentSessions:
    """Bound memory even when a browser closes without delivering an unload event."""

    def __init__(self, capacity=64, idle_seconds=3600):
        self.capacity = capacity
        self.idle_seconds = idle_seconds
        self.lock = RLock()
        self.entries = OrderedDict()

    def get(self, owner, key, *, create=True):
        if (
            not owner
            or owner[0] not in ("ui", "ui-login")
            or not isinstance(key, str)
            or not _KEY.fullmatch(key)
        ):
            raise PermissionError("Reload the Chat tab to start a private attachment session.")
        identity = owner, key
        with self.lock:
            now = monotonic()
            while self.entries:
                _, last_used = next(iter(self.entries.values()))
                if now - last_used <= self.idle_seconds:
                    break
                _, (expired, _) = self.entries.popitem(last=False)
                expired.clear()
            session, _ = self.entries.pop(identity, (None, None))
            if session is None:
                if not create:
                    raise PermissionError("The attachment conversation is no longer available.")
                session = AttachmentSession(owner)
            self.entries[identity] = session, now
            if len(self.entries) > self.capacity:
                _, (evicted, _) = self.entries.popitem(last=False)
                evicted.clear()
            return session
