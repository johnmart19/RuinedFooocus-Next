"""Bounded publisher artwork for exact catalogue identities, without weight reads."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote, urlsplit

from PIL import Image

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.models.model_previews import (
    SUFFIXES,
    is_warning,
    local_preview,
    valid_preview,
    write_preview,
)
from ruinedfooocus.features.models import model_sources


def local_checkpoint_preview(name, models):
    """Show author sidecars immediately, before background cache copying runs."""
    filename = models.get_file("checkpoints", name)
    if not filename:
        return None
    path = Path(filename)
    ambiguous = (
        sum(
            Path(candidate).with_suffix("").as_posix().casefold()
            == Path(name).with_suffix("").as_posix().casefold()
            for candidate in models.get_names("checkpoints")
        )
        > 1
    )
    return local_preview(path, ambiguous=ambiguous)


def preview_identity(row):
    annotation = row.get("annotation") or {}
    artifact = annotation.get("artifact") or {}
    source, digest = artifact.get("pinned_url"), artifact.get("sha256")
    if not isinstance(source, str) or not isinstance(digest, str):
        return None
    parsed = urlsplit(source)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.query
        or parsed.fragment
        or len(parts) < 5
        or parts[2] != "resolve"
        or not re.fullmatch(r"[a-fA-F0-9]{40}", parts[3])
        or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)
        or not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) for name in parts[:2])
    ):
        return None
    return dict(
        source=source,
        sha256=digest.lower(),
        repository="/".join(parts[:2]),
        revision=parts[3],
        key=hashlib.sha256((source + "\0" + digest.lower()).encode()).hexdigest(),
    )


def _read(url, limit):
    return model_sources.hf_metadata.read(url, limit)


def fetch_registered_preview(identity, destination, *, offline=False):
    """Two bounded reads: pinned metadata, then one declared raster image.

    An artifact digest must match that revision's published manifest. No guessed
    repository search, social thumbnail, remote video, model bytes or model hash.
    """
    repository, revision = identity["repository"], identity["revision"]
    is_offline = offline if callable(offline) else lambda: offline
    metadata = model_sources.huggingface_manifest(repository, revision, offline=is_offline())
    if not isinstance(metadata, dict) or metadata.get("sha") != revision:
        return None
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list) or not any(
        isinstance(row, dict)
        and isinstance(row.get("lfs"), dict)
        and isinstance(row["lfs"].get("sha256"), str)
        and row["lfs"]["sha256"].lower() == identity["sha256"]
        for row in siblings
    ):
        return None
    images = [
        row["rfilename"]
        for row in siblings
        if isinstance(row, dict)
        and isinstance(row.get("rfilename"), str)
        and Path(row["rfilename"]).suffix.lower() in SUFFIXES
        and not row["rfilename"].startswith("/")
        and ".." not in Path(row["rfilename"]).parts
        and "\\" not in row["rfilename"]
    ]
    images.sort(key=lambda name: (Path(name).suffix.lower() == ".gif", name))
    if not images:
        return None
    image_url = (
        f"https://huggingface.co/{repository}/resolve/{revision}/{quote(images[0], safe='/')}"
    )
    if is_offline():
        return None
    body = _read(image_url, 12 * 1024 * 1024)
    with Image.open(io.BytesIO(body)) as image:
        if image.width * image.height > 16 * 1024 * 1024:
            raise ValueError("Publisher artwork dimensions exceed size limit")
    published = write_preview(body, destination)
    save_json(
        destination.with_suffix(".source.json"),
        dict(
            source=identity["source"],
            sha256=identity["sha256"],
            revision=revision,
            image_url=image_url,
            checked_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return published


class CataloguePreviews:
    """Process-owned bounded workers; session filters/selections never enter a job."""

    def __init__(self, directory, *, workers=2, pending=4, retry_seconds=600, fetcher=None):
        if not 1 <= workers <= pending <= 8:
            raise ValueError("Require 1 <= artwork workers <= pending <= 8")
        self.directory = Path(directory)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model-artwork")
        self.limit = pending
        self.retry_seconds = retry_seconds
        self.fetcher = fetcher or fetch_registered_preview
        self.lock = threading.RLock()
        self.jobs, self.checked = {}, {}
        self.revision = 0
        self.closed = False
        self.offline = False

    def __deepcopy__(self, memo):
        return self

    def cached(self, row):
        identity = preview_identity(row)
        if identity is None:
            return None
        base = self.directory / identity["key"]
        return next(
            (
                str(path)
                for suffix in SUFFIXES
                if (path := base.with_suffix(suffix)).is_file()
                and valid_preview(path)
                and not is_warning(path)
            ),
            None,
        )

    def request(self, rows, *, offline=False):
        with self.lock:
            self.offline = offline
            if offline:
                for future in list(self.jobs.values()):
                    if future is not None:
                        future.cancel()
                return False
            if self.closed:
                return False
            for row in rows:
                identity = preview_identity(row)
                if identity is None or self.cached(row):
                    continue
                key = identity["key"]
                if len(self.jobs) >= self.limit:
                    break
                if (
                    key in self.jobs
                    or time.monotonic() - self.checked.get(key, -float("inf")) < self.retry_seconds
                ):
                    continue
                # Reserve before submitting: a fast cache miss may finish before
                # add_done_callback, and repeated clients must still deduplicate.
                self.jobs[key] = None
                future = self.executor.submit(self._fetch, identity, self.directory / key)
                self.jobs[key] = future
                future.add_done_callback(lambda result, key=key: self._complete(key, result))
            return bool(self.jobs)

    def _fetch(self, identity, destination):
        def is_offline():
            with self.lock:
                return self.offline

        if is_offline():
            return None
        if self.fetcher is fetch_registered_preview:
            return self.fetcher(identity, destination, offline=is_offline)
        return self.fetcher(identity, destination)

    def _complete(self, key, future):
        try:
            future.result()
        except Exception:
            # Remote metadata/artwork errors retain the local placeholder and
            # backoff. A failed background callback must still release its slot.
            pass
        with self.lock:
            self.jobs.pop(key, None)
            self.checked[key] = time.monotonic()
            if len(self.checked) > 512:
                self.checked.pop(next(iter(self.checked)))
            self.revision += 1

    @property
    def active(self):
        with self.lock:
            return bool(self.jobs)

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
