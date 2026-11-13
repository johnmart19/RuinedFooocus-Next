"""Incremental SQLite publication: readers keep the last complete Library snapshot."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

from ruinedfooocus.core.preparation import preparation_service
from ruinedfooocus.core.cache_contract import canonical_path_key

SCHEMA_VERSION = 2
MEDIA_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".exr", ".avif"})
IMAGE_ROWS = (
    "("
    + " OR ".join(f"lower(fullpath) LIKE '%{suffix}'" for suffix in sorted(MEDIA_SUFFIXES))
    + ")"
)
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class MediaMetadataError(ValueError):
    """One stable file was rejected by a media decoder, not by index storage."""


def _writer_lock(key):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(key), threading.RLock())


def check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise CancelledError("Library refresh was cancelled")


def ensure_schema(connection):
    """Keep old rows; their missing signatures make the first refresh a cold scan."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    if version != 0:
        raise ValueError("This Library index schema needs its matching application version")
    with connection:
        connection.execute("CREATE TABLE IF NOT EXISTS images(fullpath text, path text, json text)")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
        for column, kind in (
            ("identity", "text"),
            ("size", "integer"),
            ("mtime_ns", "integer"),
            ("ctime_ns", "integer"),
        ):
            if column not in columns:
                connection.execute(f"ALTER TABLE images ADD COLUMN {column} {kind}")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS image_identity ON images(identity)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS library_snapshot(id integer PRIMARY KEY CHECK(id=1), "
            "revision integer NOT NULL, roots text NOT NULL)"
        )
        connection.execute("INSERT OR IGNORE INTO library_snapshot VALUES(1,0,'[]')")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def connect_index(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(canonical_path_key(path)):
        connection = sqlite3.connect(path, timeout=5, check_same_thread=False)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            ensure_schema(connection)
        except BaseException:
            connection.close()
            raise
    return connection


@contextmanager
def index_connection(browser):
    """A callback owns its connection; legacy injected in-memory fixtures stay usable."""
    path = getattr(browser, "_database_path", None)
    if path is None:
        # No application caller shares this path: it supports isolated caller fixtures.
        yield browser.sql_conn
        return
    with preparation_service.artifact_lease(path):
        # Schema initialization happens at construction, never in a page read.
        connection = sqlite3.connect(path, timeout=5)
        try:
            yield connection
        finally:
            connection.close()


@dataclass(frozen=True)
class ScanResult:
    count: int
    parsed: int
    reused: int
    removed: int
    revision: int
    seconds: float
    skipped: int = 0
    warnings: tuple[str, ...] = ()


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _walk_error(error):
    raise error


def incremental_scan(
    connection,
    folders,
    metadata_reader,
    *,
    cancelled=None,
    progress=None,
    publish=None,
    only_if_uninitialized=False,
):
    """Publish one transaction only after every root and changed source is stable.

    WAL readers remain on the previous committed snapshot throughout this scan.
    Metadata is parsed only for new/replaced files. Explicit decoder rejections
    skip one stable file and report it. Other failures/cancel roll back all
    upserts/removals. A missing or unreadable configured root also preserves it.
    Initial-only callers return None once a snapshot exists, under the same lock.
    """
    started = time.monotonic()
    roots = {}
    for folder in folders:
        folder = Path(folder)
        roots.setdefault(canonical_path_key(folder), folder)
    database = connection.execute("PRAGMA database_list").fetchone()[2]
    key = canonical_path_key(database) if database else id(connection)
    count = parsed = reused = 0
    skipped = rejected_rows = 0
    warnings = []
    with _writer_lock(key):
        ensure_schema(connection)
        check_cancelled(cancelled)
        if (
            only_if_uninitialized
            and connection.execute("SELECT revision FROM library_snapshot WHERE id=1").fetchone()[0]
        ):
            return None
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("CREATE TEMP TABLE rf_seen(identity text PRIMARY KEY)")
            for folder in roots.values():
                check_cancelled(cancelled)
                if not folder.is_dir():
                    raise OSError(f"Library folder is unavailable: {folder}")
                for directory, _, filenames in os.walk(folder, onerror=_walk_error):
                    check_cancelled(cancelled)
                    for filename in filenames:
                        path = Path(directory) / filename
                        if path.suffix.lower() not in MEDIA_SUFFIXES:
                            continue
                        check_cancelled(cancelled)
                        identity = canonical_path_key(path)
                        seen = connection.execute(
                            "INSERT OR IGNORE INTO rf_seen VALUES(?)", (identity,)
                        )
                        if not seen.rowcount:
                            continue
                        signature = _signature(path)
                        if not path.is_file():
                            raise OSError(f"Library source is no longer a file: {path}")
                        relative = str(path.relative_to(folder))
                        old = connection.execute(
                            "SELECT size,mtime_ns,ctime_ns,json FROM images WHERE identity=?",
                            (identity,),
                        ).fetchone()
                        if old is not None and tuple(old[:3]) == signature:
                            metadata = old[3]
                            # Root ordering can change the relative path without rereading a file.
                            try:
                                value = json.loads(metadata)
                            except (ValueError, TypeError):
                                old = None
                            else:
                                if not isinstance(value, dict):
                                    old = None
                                    value = {}
                                value["file_path"] = relative
                                metadata = json.dumps(value)
                        if old is None or tuple(old[:3]) != signature:
                            try:
                                value = metadata_reader(path)
                            except MediaMetadataError as error:
                                check_cancelled(cancelled)
                                if _signature(path) != signature:
                                    raise OSError(
                                        f"Library source changed while scanning: {path}"
                                    ) from error
                                rejected_rows += connection.execute(
                                    "DELETE FROM images WHERE identity=?", (identity,)
                                ).rowcount
                                skipped += 1
                                if len(warnings) < 20:
                                    warnings.append(f"{relative}: {error}"[:1024])
                                continue
                            if not isinstance(value, dict):
                                raise ValueError(f"Invalid Library metadata: {path}")
                            check_cancelled(cancelled)
                            if _signature(path) != signature:
                                raise OSError(f"Library source changed while scanning: {path}")
                            metadata = json.dumps(value | {"file_path": relative})
                            parsed += 1
                        else:
                            reused += 1
                        connection.execute(
                            "INSERT INTO images(fullpath,path,json,identity,size,mtime_ns,ctime_ns) "
                            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(identity) DO UPDATE SET "
                            "fullpath=excluded.fullpath,path=excluded.path,json=excluded.json,"
                            "size=excluded.size,mtime_ns=excluded.mtime_ns,ctime_ns=excluded.ctime_ns",
                            (str(path), relative, metadata, identity, *signature),
                        )
                        count += 1
                        if progress is not None and (count == 1 or count % 32 == 0):
                            progress(count, parsed, reused)
            check_cancelled(cancelled)
            removed = (
                rejected_rows
                + connection.execute(
                    "DELETE FROM images WHERE identity IS NULL OR identity NOT IN (SELECT identity FROM rf_seen)"
                ).rowcount
            )
            connection.execute(
                "UPDATE library_snapshot SET revision=revision+1,roots=? WHERE id=1",
                (json.dumps(list(roots)),),
            )
            revision = connection.execute(
                "SELECT revision FROM library_snapshot WHERE id=1"
            ).fetchone()[0]
            connection.execute("DROP TABLE rf_seen")
            # The caller's cancellation event and transaction commit are the final boundary.
            check_cancelled(cancelled)
            if publish is None:
                connection.commit()
            else:
                publish(connection.commit)
        except BaseException:
            connection.rollback()
            raise
    return ScanResult(
        count,
        parsed,
        reused,
        removed,
        revision,
        time.monotonic() - started,
        skipped,
        tuple(warnings),
    )


def page_snapshot(connection, page, search, per_page):
    """Count and page come from one stable read transaction; no full-index stat pass."""
    connection.execute("BEGIN")
    try:
        query = (f"%{search or ''}%",)
        count = connection.execute(
            f"SELECT count(*) FROM images WHERE json LIKE ? AND {IMAGE_ROWS}", query
        ).fetchone()[0]
        pages = max(1, (count + per_page - 1) // per_page)
        try:
            page = int(page or 1)
        except (TypeError, ValueError, OverflowError):
            page = 1
        page = min(max(1, page), pages)
        rows = connection.execute(
            f"SELECT fullpath,path,size,mtime_ns,ctime_ns FROM images WHERE json LIKE ? AND {IMAGE_ROWS} "
            "ORDER BY CASE WHEN path GLOB '[0-9]*' THEN 0 ELSE 1 END, "
            "CASE WHEN path GLOB '[0-9]*' THEN path END DESC, "
            "CASE WHEN path NOT GLOB '[0-9]*' THEN Upper(path) END ASC, fullpath ASC "
            "LIMIT ? OFFSET ?",
            (*query, per_page, (page - 1) * per_page),
        ).fetchall()
        connection.commit()
        return count, pages, page, rows
    except BaseException:
        connection.rollback()
        raise
