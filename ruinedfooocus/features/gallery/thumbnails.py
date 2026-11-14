"""Bounded, disposable grid thumbnails; Library actions keep using originals."""

from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading

from PIL import Image, ImageOps

from ruinedfooocus.core.cache_contract import (
    LIBRARY_THUMBNAIL_MAX_BYTES,
    LIBRARY_THUMBNAIL_MAX_FILES,
    canonical_path_key,
    file_identity,
    library_thumbnail_root,
    owned_library_thumbnail,
    record_key,
)
from ruinedfooocus.core.preparation import PreparationBusy, preparation_service

GRID_THUMBNAIL_SIDE = 1024
_maintenance_lock = threading.Lock()
_dirty_roots = {}


def _linked_directory(directory):
    try:
        status = directory.lstat()
        return directory.is_symlink() or bool(getattr(status, "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


def _prune(directory):
    """Trim oldest access timestamps only after all cooperating copies finish."""
    key = canonical_path_key(directory)
    with _maintenance_lock:
        if not _dirty_roots.get(key, True):
            return
    try:
        with preparation_service.artifact_lease(directory, write=True, blocking=False):
            if _linked_directory(directory):
                return
            entries = []
            if directory.is_dir():
                for path in directory.iterdir():
                    if owned_library_thumbnail(path):
                        status = path.stat()
                        entries.append((status.st_mtime_ns, status.st_size, path))
            count, total = len(entries), sum(item[1] for item in entries)
            for _, size, path in sorted(entries):
                if count <= LIBRARY_THUMBNAIL_MAX_FILES and total <= LIBRARY_THUMBNAIL_MAX_BYTES:
                    break
                path.unlink()
                count -= 1
                total -= size
            with _maintenance_lock:
                _dirty_roots[key] = False
    except (OSError, PreparationBusy):
        # Active copies defer cleanup to the last reader; storage failures are retryable.
        pass


def _thumbnail(path, directory):
    identity = file_identity(path)
    key = record_key(
        "library-grid-thumbnail", identity | {"side": GRID_THUMBNAIL_SIDE, "transform_version": 1}
    )
    target = directory / f"{key}.webp"
    with preparation_service.artifact_lease(target, write=True):
        if owned_library_thumbnail(target) and target.stat().st_size:
            os.utime(target, None)
            return target
        with Image.open(path) as image:
            if max(image.size) <= GRID_THUMBNAIL_SIDE and identity["size"] <= 256 * 1024:
                return path
            image.draft("RGB", (GRID_THUMBNAIL_SIDE, GRID_THUMBNAIL_SIDE))
            ImageOps.exif_transpose(image, in_place=True)
            alpha = "A" in image.getbands() or "transparency" in image.info
            with image.convert("RGBA" if alpha else "RGB") as thumbnail:
                thumbnail.thumbnail((GRID_THUMBNAIL_SIDE, GRID_THUMBNAIL_SIDE))
                directory.mkdir(parents=True, exist_ok=True)
                partial = None
                try:
                    with NamedTemporaryFile(
                        dir=directory, prefix=f"{key}.", suffix=".tmp", delete=False
                    ) as temporary:
                        partial = Path(temporary.name)
                    thumbnail.save(partial, "WEBP", quality=82, method=0)
                    if (
                        partial.stat().st_size > LIBRARY_THUMBNAIL_MAX_BYTES
                        or file_identity(path) != identity
                    ):
                        return path
                    os.replace(partial, target)
                    with _maintenance_lock:
                        _dirty_roots[canonical_path_key(directory)] = True
                finally:
                    if partial is not None:
                        partial.unlink(missing_ok=True)
        return target


@contextmanager
def grid_thumbnail_batch(cache_root):
    """Lease a page's copies together and trim retention once after delivery."""
    directory = library_thumbnail_root(cache_root)
    key = canonical_path_key(directory)
    with _maintenance_lock:
        _dirty_roots.setdefault(key, True)

    def thumbnail(path):
        path = Path(path)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            return path
        try:
            return path if _linked_directory(directory) else _thumbnail(path, directory)
        except (OSError, ValueError):
            return path

    try:
        with preparation_service.artifact_lease(directory):
            yield thumbnail
    finally:
        _prune(directory)


@contextmanager
def grid_thumbnail_path(path, cache_root):
    """Keep a derived copy leased until its caller finishes copying/reading it."""
    path = Path(path)
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        yield path
        return
    with grid_thumbnail_batch(cache_root) as thumbnail:
        yield thumbnail(path)


def grid_thumbnail(path, cache_root):
    """Return a grid copy; callers reading the file should use grid_thumbnail_path."""
    with grid_thumbnail_path(path, cache_root) as display:
        return display
