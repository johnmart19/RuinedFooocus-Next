"""Hub downloads with the application's byte-progress callback."""

import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from uuid import uuid4

from filelock import FileLock
from huggingface_hub import constants, hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import build_hf_headers, get_session
from tqdm import tqdm

from ruinedfooocus.core.config_io import save_json


_HUB = "https://huggingface.co"
_HTTP_TIMEOUT = 15


def download_hub_file(url, destination, progress=None):
    """Synchronous compatibility entry point using shared preparation ownership."""
    from ruinedfooocus.core.preparation import PreparationRequest, preparation_service

    return preparation_service.submit(PreparationRequest(url, destination), progress).result()


def _transfer_hub_file(url, staging, progress, cancelled, *, expected_size=None, sha256=None):
    """Keep downloads in destination-owned staging."""
    from ruinedfooocus.core.preparation import _check_cancelled

    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname != "huggingface.co" or len(parts) < 5 or parts[2] != "resolve":
        raise ValueError("Expected a Hugging Face model resolve URL")
    repository = "/".join(parts[:2])
    revision = unquote(parts[3])
    filename = unquote("/".join(parts[4:]))
    relative = Path(filename)
    if relative.anchor or ".." in relative.parts or ":" in filename or "\\" in filename:
        raise ValueError("Invalid Hugging Face filename")
    _check_cancelled(cancelled)

    class DownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            self.started = self.last_update = time.monotonic()
            super().__init__(*args, **kwargs)
            self.initial_bytes = self.n
            if progress:
                progress(self.n, self.total or 0, 0)

        def update(self, amount=1):
            _check_cancelled(cancelled)
            # tqdm skips its counter when terminal bars are disabled; UI still needs it.
            if self.disable:
                self.n += amount
            result = super().update(amount)
            now = time.monotonic()
            if progress and now - self.last_update >= 0.5:
                progress(
                    self.n,
                    self.total or 0,
                    (self.n - self.initial_bytes) / max(now - self.started, 0.001),
                )
                self.last_update = now
            return result

    try:
        if (
            expected_size is not None
            and sha256 is not None
            and re.fullmatch("[0-9a-f]{40}", revision)
            and parsed.scheme == "https"
            and parsed.netloc.lower() in ("huggingface.co", "huggingface.co:443")
            and not (parsed.query or parsed.fragment)
            and constants.ENDPOINT == _HUB
            and not constants.HF_HUB_OFFLINE
            and not isinstance(try_to_load_from_cache(repository, filename, revision=revision), str)
        ):
            downloaded = _download_verified_http(
                url, revision, expected_size, sha256, staging, progress, cancelled
            )
            if downloaded is not None:
                return downloaded
        downloaded = Path(
            hf_hub_download(
                repo_id=repository,
                filename=filename,
                revision=revision,
                local_dir=staging,
                tqdm_class=DownloadProgress,
            )
        )
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in (401, 403):
            raise RuntimeError(
                f"Cannot download {relative.name}: Hugging Face access is required. "
                f"Request access at https://huggingface.co/{repository}, then use hf auth login "
                "or HF_TOKEN. Alternatively, place the file in its configured model folder."
            ) from error
        raise
    _check_cancelled(cancelled)
    return downloaded


def _download_verified_http(url, revision, size, sha256, staging, progress, cancelled):
    from ruinedfooocus.core.preparation import _check_cancelled

    part = staging / "hub-http.part"
    receipt = staging / "hub-http.json"
    etag = f'"{sha256}"'
    identity = {"url": url, "etag": etag, "size": size}
    headers = build_hf_headers()
    headers["accept-encoding"] = "identity"
    client = get_session()
    # Multiple RF processes can share a data root. The stable partial has one writer.
    with FileLock(staging / "hub-http.lock", timeout=0):
        _check_cancelled(cancelled)
        with client.stream(
            "HEAD", url, headers=headers, timeout=_HTTP_TIMEOUT, follow_redirects=False
        ) as response:
            _check_http_status(response)
            source = _ordinary_http_source(url, response, revision, etag, size)
        if source is None:
            part.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
            return None

        offset = _resume_offset(part, receipt, identity)
        if offset < size:
            _check_cancelled(cancelled)
            if urlparse(source).hostname != "huggingface.co":
                headers.pop("authorization", None)
            if offset:
                headers.update({"range": f"bytes={offset}-", "if-range": etag})
            with client.stream(
                "GET",
                source,
                headers=headers,
                timeout=_HTTP_TIMEOUT,
                follow_redirects=False,
                auth=None,
            ) as response:
                _check_http_status(response)
                if offset == 0 and (response.is_redirect or response.headers.get("etag") != etag):
                    return None
                offset = _response_offset(response, offset, size, etag)
                save_json(receipt, identity)
                _stream_verified_response(response, part, offset, size, progress, cancelled)
        _check_cancelled(cancelled)
        # Transfer ownership of the complete bytes to this caller before unlocking.
        completed = staging / f"hub-http-{uuid4().hex}.complete"
        receipt.unlink(missing_ok=True)
        part.replace(completed)
        return completed


def _check_http_status(response):
    if response.status_code in (401, 403):
        raise HfHubHTTPError("Hub access is required", response=response)
    if response.status_code >= 400:
        response.raise_for_status()


def _ordinary_http_source(url, response, revision, etag, size):
    headers = response.headers
    if (
        (response.status_code != 200 and not (response.is_redirect and "location" in headers))
        or "x-xet-hash" in headers
        or headers.get("x-repo-commit") != revision
        or (headers.get("x-linked-etag") or headers.get("etag")) != etag
        or (headers.get("x-linked-size") or headers.get("content-length")) != str(size)
    ):
        return None
    source = urljoin(url, headers["location"]) if response.is_redirect else url
    target = urlparse(source)
    host = target.hostname or ""
    if (
        target.scheme == "https"
        and target.netloc.lower() in (host, f"{host}:443")
        and (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co")))
    ):
        return source
    return None


def _validated_resume_offset(part, receipt, identity):
    """Read a bounded receipt without changing a partial or claiming verification."""
    try:
        with receipt.open("rb") as stream:
            data = stream.read(16 * 1024 + 1)
        if len(data) <= 16 * 1024 and json.loads(data) == identity:
            size = part.stat().st_size
            if size <= identity["size"]:
                return size
    except (OSError, ValueError, UnicodeError):
        pass
    return None


def _resume_offset(part, receipt, identity):
    offset = _validated_resume_offset(part, receipt, identity)
    if offset is not None:
        return offset
    part.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)
    return 0


def _response_offset(response, offset, size, etag):
    headers = response.headers
    if headers.get("etag") != etag or headers.get("content-encoding", "identity") != "identity":
        raise OSError("Hub response identity does not match the requested artifact")
    if response.status_code == 200:
        if "content-range" in headers:
            raise OSError("Unexpected Content-Range on a full Hub response")
        offset = 0
    elif (
        response.status_code != 206
        or not offset
        or headers.get("content-range") != f"bytes {offset}-{size - 1}/{size}"
    ):
        raise OSError("Invalid Content-Range for resumed Hub response")
    if "content-length" in headers and headers["content-length"] != str(size - offset):
        raise OSError("Hub response length does not match the requested artifact")
    return offset


def _stream_verified_response(response, part, offset, size, progress, cancelled):
    from ruinedfooocus.core.preparation import _check_cancelled

    received = offset
    started = last_update = time.monotonic()
    if progress:
        progress(received, size, 0)
    with part.open("ab" if offset else "wb") as output:
        for chunk in response.iter_raw(chunk_size=1024 * 1024):
            _check_cancelled(cancelled)
            if received + len(chunk) > size:
                raise OSError("Hub response exceeded the requested artifact size")
            received += output.write(chunk)
            now = time.monotonic()
            if progress and now - last_update >= 0.5:
                progress(received, size, (received - offset) / (now - started))
                last_update = now
    if received != size:
        raise OSError("Incomplete Hub response; retry to resume")
