"""Public metadata lookups; never download checkpoint weights."""

from collections import OrderedDict
from email.utils import parsedate_to_datetime
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlparse, urlunparse

import requests

_civitai_not_found = {}
_REPOSITORY = r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"


class HuggingFaceMetadata:
    """Process-owned bounded metadata cache and single-flight, shared by artwork callers.

    One attempt per cache window; a failed request never sleeps/retries here.
    Existing background workers own scheduling. Cooldown also covers raster reads.
    Cache state is deliberately memory-only: a restart permits one fresh attempt.
    """

    def __init__(self, *, clock=None, wall_clock=None):
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(2)
        self.cache = OrderedDict()
        self.inflight = {}
        self.not_before = 0

    @staticmethod
    def key(url, params=None):
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query)) | (params or {})
        return urlunparse(
            (
                "https",
                "huggingface.co",
                parsed.path.casefold(),
                "",
                urlencode(sorted(query.items())),
                "",
            )
        )

    def cooling_down(self):
        with self.lock:
            return self.clock() < self.not_before

    def _cooldown(self, response):
        if response.status_code != 429:
            return
        value = response.headers.get("Retry-After", "")
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                delay = parsedate_to_datetime(value).timestamp() - self.wall_clock()
            except (TypeError, ValueError, OverflowError):
                delay = 60
        # Bound malformed/very long headers; all callers share the same deadline.
        delay = max(1, min(delay, 86400)) if delay >= 0 else 60
        with self.lock:
            self.not_before = max(self.not_before, self.clock() + delay)

    def read(self, url, limit, *, params=None, offline=False):
        """Bounded HTTPS HF/CDN bytes only, never model payloads or retry loops."""
        if offline or self.cooling_down() or not self.slots.acquire(blocking=False):
            raise ValueError("Publisher artwork is offline, busy or cooling down")
        started = self.clock()
        try:
            for _ in range(4):
                parsed = urlparse(url)
                host = parsed.hostname or ""
                remaining = 20 - (self.clock() - started)
                if remaining <= 0:
                    raise ValueError("Publisher artwork request exceeded time limit")
                if (
                    parsed.scheme != "https"
                    or parsed.username
                    or not (
                        host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))
                    )
                ):
                    raise ValueError("Publisher artwork redirect is outside its source service")
                if self.cooling_down():
                    raise ValueError("Publisher artwork is cooling down")
                with requests.get(
                    url,
                    params=params,
                    stream=True,
                    timeout=(min(3, remaining), min(10, remaining)),
                    allow_redirects=False,
                ) as response:
                    self._cooldown(response)
                    response.raise_for_status()
                    if response.is_redirect:
                        url = urljoin(url, response.headers["location"])
                        params = None
                        continue
                    chunks, size = [], 0
                    for chunk in response.iter_content(16384):
                        if self.clock() - started > 20:
                            raise ValueError("Publisher artwork request exceeded time limit")
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("Publisher artwork response exceeds size limit")
                        chunks.append(chunk)
                    return b"".join(chunks)
            raise ValueError("Publisher artwork exceeded redirect limit")
        finally:
            self.slots.release()

    def _remember(self, key, body, ttl):
        self.cache[key] = (self.clock() + ttl, body)
        self.cache.move_to_end(key)
        while (
            len(self.cache) > 128
            or sum(len(item[1] or b"") for item in self.cache.values()) > 16 * 1024 * 1024
        ):
            self.cache.popitem(last=False)

    def metadata(self, url, *, params=None, offline=False):
        if offline:
            return None
        key = self.key(url, params)
        with self.lock:
            cached = self.cache.get(key)
            if cached and self.clock() < cached[0]:
                return json.loads(cached[1]) if cached[1] is not None else None
            if self.cooling_down():
                return None
            waiting = self.inflight.get(key)
            if waiting is None:
                if len(self.inflight) >= 8:
                    return None
                waiting = self.inflight[key] = threading.Event()
                owner = True
            else:
                owner = False
        if not owner:
            # Only background/explicit-refresh callers wait; card building is local.
            if not waiting.wait(20):
                return None
            with self.lock:
                cached = self.cache.get(key)
                return json.loads(cached[1]) if cached and cached[1] is not None else None
        body, ttl = None, 60
        try:
            body = self.read(url, 2 * 1024 * 1024, params=params)
            value = json.loads(body)
            ttl = 86400 if "/revision/" in url else 3600
            return value
        except (requests.RequestException, ValueError, UnicodeError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            ttl = 600 if status in (401, 403, 404) else 60
            body = None
            return None
        finally:
            with self.lock:
                if self.cooling_down():
                    ttl = max(ttl, self.not_before - self.clock())
                self._remember(key, body, ttl)
                # An unrevisioned result can satisfy an exact request for its SHA.
                if (
                    body is not None
                    and isinstance(value, dict)
                    and re.fullmatch(r"[a-fA-F0-9]{40}", str(value.get("sha", "")))
                ):
                    path = urlparse(url).path
                    if path.startswith("/api/models/") and "/revision/" not in path:
                        alias = url.split("?", 1)[0] + "/revision/" + value["sha"]
                        self._remember(self.key(alias, params), body, 86400)
                self.inflight.pop(key).set()


hf_metadata = HuggingFaceMetadata()


def huggingface_manifest(repository, revision="", *, offline=False):
    if not isinstance(repository, str) or not re.fullmatch(_REPOSITORY, repository):
        return None
    if revision and not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        return None
    url = f"https://huggingface.co/api/models/{repository}"
    if revision:
        url += f"/revision/{revision}"
    info = hf_metadata.metadata(url, params={"blobs": "true"}, offline=offline)
    if (
        not isinstance(info, dict)
        or (info.get("id") and str(info["id"]).casefold() != repository.casefold())
        or not re.fullmatch(r"[a-fA-F0-9]{40}", str(info.get("sha", "")))
        or (revision and info["sha"].lower() != revision.lower())
    ):
        return None
    return info


def huggingface_source(metadata, header=None, verified=None):
    """Prefer an already verified artifact URL, then cached repo/revision and header."""
    if verified:
        return verified["artifact"].get("pinned_url", "")
    if metadata.get("hf_repo_id"):
        return f"https://huggingface.co/{metadata['hf_repo_id']}" + (
            f"/tree/{metadata['hf_revision']}" if metadata.get("hf_revision") else ""
        )
    source = header.get("modelspec.source", "") if isinstance(header, dict) else ""
    return source if isinstance(source, str) else ""


def huggingface_recommendations(repository, sha256, *, revision="", offline=False):
    """Read bounded JSON/card data only after matching a cached checkpoint hash.

    The hash is compared locally with the public manifest, never sent to HF.
    Pin card/config reads to the matched repository revision. No weights/code.
    """
    if (
        not isinstance(repository, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository)
        or not isinstance(sha256, str)
        or not re.fullmatch(r"[a-fA-F0-9]{64}", sha256)
    ):
        return None

    def read(url, limit, as_json=False):
        body = hf_metadata.read(url, limit, offline=offline).decode("utf-8")
        return json.loads(body) if as_json else body

    try:
        info = huggingface_manifest(repository, revision, offline=offline)
        if info is None:
            return None
        repository = info.get("id", repository)
        siblings = info.get("siblings", [])
        if not isinstance(siblings, list) or not any(
            isinstance(item, dict)
            and isinstance(item.get("lfs"), dict)
            and item["lfs"].get("sha256", "").lower() == sha256.lower()
            for item in siblings
            if isinstance(item, dict)
        ):
            return None
        revision = info.get("sha")
        if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            return None
        names = {
            item.get("rfilename")
            for item in siblings
            if isinstance(item, dict) and isinstance(item.get("rfilename"), str)
        }
        result = {
            "description": "",
            "generationConfig": {},
            "source": f"Hugging Face {repository} @ {revision[:12]} (repository-wide settings)",
            "source_url": f"https://huggingface.co/{repository}/blob/{revision}/README.md",
        }
        card_data = info.get("cardData")
        parents = card_data.get("base_model", []) if isinstance(card_data, dict) else []
        parents = [parents] if isinstance(parents, str) else parents
        result["source_repositories"] = (
            [name for name in parents if isinstance(name, str)][:3]
            if isinstance(parents, list)
            else []
        )
        base = f"https://huggingface.co/{repository}/resolve/{revision}/"
        if "README.md" in names:
            result["description"] = read(base + "README.md", 200000)
        if "generation_config.json" in names:
            try:
                config = read(base + "generation_config.json", 64000, as_json=True)
                if isinstance(config, dict):
                    result["generationConfig"] = config
            except (requests.RequestException, ValueError, UnicodeError):
                pass  # A valid card remains useful without a generation config.
        return result
    except (requests.RequestException, ValueError, UnicodeError, AttributeError):
        return None


def civitai_metadata(endpoint):
    import time

    for host in ("civitai.com", "civitai.red"):
        key = (host, endpoint)
        if time.monotonic() - _civitai_not_found.get(key, -float("inf")) < 300:
            continue
        try:
            response = requests.get(f"https://{host}/api/v1/{endpoint}", timeout=(5, 20))
            if response.status_code == 404:
                # A model may exist only on Hugging Face. This is a normal miss.
                if len(_civitai_not_found) >= 256:
                    _civitai_not_found.pop(next(iter(_civitai_not_found)))
                _civitai_not_found[key] = time.monotonic()
                continue
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("id"):
                return data
        except (requests.RequestException, ValueError) as error:
            print(f"Model metadata lookup failed on {host}: {error}")
    return None


def huggingface_preview(path, sha256, source="", *, offline=False):
    """Use repository artwork only after matching a checkpoint's LFS SHA-256."""
    if offline or not isinstance(sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        return None
    path = Path(path)
    repositories = []
    revision = ""
    url = urlparse(source) if isinstance(source, str) else urlparse("")
    parts = url.path.strip("/").split("/")
    if url.scheme == "https" and url.hostname == "huggingface.co" and len(parts) >= 2:
        repository = "/".join(parts[:2])
        if re.fullmatch(_REPOSITORY, repository):
            repositories.append(repository)
            if (
                len(parts) >= 4
                and parts[2] in ("resolve", "blob", "tree")
                and re.fullmatch(r"[a-fA-F0-9]{40}", parts[3])
            ):
                revision = parts[3]
    if not repositories:
        # Keep the meaningful model name. A publisher token or GGUF format alone
        # must not broaden image-artwork discovery into language-model manifests.
        query = re.split(
            r"[-_](?:fp\d+|bf\d+|int\d+|q\d+)(?:[-_]|$)", path.stem, maxsplit=1, flags=re.I
        )[0]
        query = re.sub(r"[_-]+", " ", query).strip()[:100]
        if len(query) < 3:
            return None
        candidates = hf_metadata.metadata(
            "https://huggingface.co/api/models",
            params={"search": query, "limit": 5, "full": "true"},
        )
        if not isinstance(candidates, list):
            return None
        image_tasks = {"text-to-image", "image-to-image", "unconditional-image-generation"}

        def image_candidate(item):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                return False
            task, tags = item.get("pipeline_tag"), item.get("tags")
            if task:
                return isinstance(task, str) and task in image_tasks
            return item.get("library_name") == "diffusers" or (
                isinstance(tags, list)
                and any(isinstance(tag, str) and tag in image_tasks for tag in tags)
            )

        repositories.extend(item["id"] for item in candidates[:5] if image_candidate(item))
    for repository in dict.fromkeys(repositories):
        if hf_metadata.cooling_down():
            break
        try:
            data = huggingface_manifest(repository, revision)
            if data is None:
                continue
            files = data.get("siblings", [])
            if not isinstance(files, list) or not any(
                isinstance(item, dict)
                and isinstance(item.get("lfs"), dict)
                and str(item["lfs"].get("sha256", "")).lower() == sha256.lower()
                for item in files
            ):
                continue
            matched_revision = data["sha"]
            images = [
                item["rfilename"]
                for item in files
                if isinstance(item, dict)
                and isinstance(item.get("rfilename"), str)
                and Path(item["rfilename"]).suffix.lower()
                in (".png", ".jpg", ".jpeg", ".webp", ".gif")
                and not item["rfilename"].startswith("/")
                and ".." not in Path(item["rfilename"]).parts
                and "\\" not in item["rfilename"]
            ]
            images.sort(
                key=lambda name: (
                    Path(name).suffix.lower() == ".gif",
                    Path(name).stem != path.stem,
                    name,
                )
            )
            previews = [
                {
                    "url": f"https://huggingface.co/{repository}/resolve/{matched_revision}/{quote(name, safe='/')}",
                    "type": "image",
                }
                for name in images[:5]
            ]
            return {"hf_repo_id": repository, "hf_revision": matched_revision, "images": previews}
        except (ValueError, TypeError, KeyError):
            continue
    return None
