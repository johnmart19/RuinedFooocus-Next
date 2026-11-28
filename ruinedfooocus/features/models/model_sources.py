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
from requests.adapters import HTTPAdapter

_DEFAULT_REQUESTS_GET = requests.get
_civitai_not_found = {}
_REPOSITORY = r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"


def _pooled_session(pool_size=32):
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "RuinedFooocus"})
    return session


class HuggingFaceMetadata:
    """Process-owned bounded metadata cache and single-flight, shared by artwork callers.

    One attempt per cache window; a failed request never sleeps/retries here.
    Existing background workers own scheduling. Cooldown also covers raster reads.
    Cache state is deliberately memory-only: a restart permits one fresh attempt.
    """

    MAX_CONCURRENT_REQUESTS = 16

    def __init__(self, *, clock=None, wall_clock=None):
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(self.MAX_CONCURRENT_REQUESTS)
        self.session = _pooled_session(32)
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
                getter = self.session.get if requests.get is _DEFAULT_REQUESTS_GET else requests.get
                with getter(
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
                if len(self.inflight) >= 48:
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
            urls = extract_hf_artwork_urls(
                data,
                repository,
                matched_revision,
                rank=lambda name: (
                    Path(name).suffix.lower() == ".gif",
                    Path(name).stem != path.stem,
                    name,
                ),
            )
            previews = [{"url": url, "type": "image"} for url in urls[:5]]
            return {"hf_repo_id": repository, "hf_revision": matched_revision, "images": previews}
        except (ValueError, TypeError, KeyError):
            continue
    return None


_HF_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_HF_README_IMAGE_RE = re.compile(
    r'!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)|<img\b[^>]*\bsrc=["\']([^"\']+)["\']',
    re.I,
)
_HF_SKIP_IMAGE_TOKENS = (
    "shields.io",
    "badgen.net",
    "img.shields",
    "favicon",
    "social-thumbnails",
)


def is_hf_artwork_url(url):
    """True when `url` is a bounded HTTPS Hugging Face artwork URL (no SVG/social)."""
    if not isinstance(url, str):
        return False
    text = url.strip()
    if not text or any(token in text.lower() for token in _HF_SKIP_IMAGE_TOKENS):
        return False
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username
        or not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co")))
    ):
        return False
    suffix = Path(parsed.path).suffix.lower()
    if suffix == ".svg":
        return False
    return True


def _resolve_hf_image_url(raw, repository, revision):
    if not isinstance(raw, str):
        return None
    text = raw.strip().strip("<>").strip()
    if not text or text.startswith(("#", "data:", "mailto:")):
        return None
    lower = text.lower()
    if any(token in lower for token in _HF_SKIP_IMAGE_TOKENS):
        return None
    if lower.startswith("https://"):
        parsed = urlparse(text)
        path = parsed.path
        if (parsed.hostname or "").lower() == "huggingface.co" and "/blob/" in path:
            path = path.replace("/blob/", "/resolve/", 1)
            text = urlunparse(parsed._replace(path=path))
        if not is_hf_artwork_url(text):
            return None
        suffix = Path(urlparse(text).path).suffix.lower()
        host = (parsed.hostname or "").lower()
        if suffix and suffix not in _HF_IMAGE_SUFFIXES:
            return None
        if not suffix and host != "cdn-uploads.huggingface.co":
            return None
        return text
    if lower.startswith("http://") or "://" in text:
        return None
    rel = text.split("?", 1)[0].split("#", 1)[0].lstrip("/")
    while rel.startswith("./"):
        rel = rel[2:]
    if (
        not rel
        or "\\" in rel
        or ".." in Path(rel).parts
        or Path(rel).suffix.lower() not in _HF_IMAGE_SUFFIXES
    ):
        return None
    return f"https://huggingface.co/{repository}/resolve/{revision}/{quote(rel, safe='/')}"


def extract_hf_artwork_urls(
    metadata,
    repository,
    revision,
    *,
    offline=False,
    allow_readme=True,
    allow_base_model=True,
    rank=None,
):
    """Return ordered HTTPS artwork URLs: siblings -> cardData -> README -> base_model."""
    if not isinstance(metadata, dict) or not repository or not revision:
        return []
    siblings = metadata.get("siblings") if isinstance(metadata.get("siblings"), list) else []
    sibling_images = [
        item["rfilename"]
        for item in siblings
        if isinstance(item, dict)
        and isinstance(item.get("rfilename"), str)
        and Path(item["rfilename"]).suffix.lower() in _HF_IMAGE_SUFFIXES
        and not item["rfilename"].startswith("/")
        and ".." not in Path(item["rfilename"]).parts
        and "\\" not in item["rfilename"]
    ]
    rank_key = rank or (lambda name: (Path(name).suffix.lower() == ".gif", name))
    sibling_images.sort(key=rank_key)
    urls = [
        f"https://huggingface.co/{repository}/resolve/{revision}/{quote(name, safe='/')}"
        for name in sibling_images
    ]
    card_data = metadata.get("cardData")
    if isinstance(card_data, dict):
        thumb = _resolve_hf_image_url(card_data.get("thumbnail"), repository, revision)
        if thumb and thumb not in urls:
            urls.append(thumb)
    if not urls and allow_readme and not offline and not hf_metadata.cooling_down():
        readme_name = next(
            (
                item["rfilename"]
                for item in siblings
                if isinstance(item, dict)
                and isinstance(item.get("rfilename"), str)
                and item["rfilename"].casefold() == "readme.md"
            ),
            None,
        )
        if readme_name:
            readme_url = (
                f"https://huggingface.co/{repository}/resolve/{revision}/"
                f"{quote(readme_name, safe='/')}"
            )
            try:
                body = hf_metadata.read(readme_url, 64 * 1024, offline=offline)
                text = body.decode("utf-8", errors="ignore")
            except (OSError, ValueError):
                text = ""
            found = []
            for md_url, html_url in _HF_README_IMAGE_RE.findall(text):
                resolved = _resolve_hf_image_url(md_url or html_url, repository, revision)
                if resolved and resolved not in found:
                    found.append(resolved)
            found.sort(key=lambda u: rank_key(Path(urlparse(u).path).name or u))
            urls.extend(found)
    if (
        not urls
        and allow_base_model
        and not offline
        and not hf_metadata.cooling_down()
        and isinstance(card_data, dict)
    ):
        raw_bases = card_data.get("base_model")
        if isinstance(raw_bases, str):
            raw_bases = [raw_bases]
        elif not isinstance(raw_bases, list):
            raw_bases = []
        upstreams = []
        for candidate in raw_bases:
            if not isinstance(candidate, str):
                continue
            repo_id = candidate.strip()
            if (
                re.fullmatch(_REPOSITORY, repo_id)
                and repo_id.casefold() != repository.casefold()
                and repo_id not in upstreams
            ):
                upstreams.append(repo_id)
        for upstream in upstreams[:2]:
            if hf_metadata.cooling_down():
                break
            upstream_meta = huggingface_manifest(upstream, offline=offline)
            if not isinstance(upstream_meta, dict):
                continue
            upstream_rev = upstream_meta.get("sha") or "main"
            upstream_urls = extract_hf_artwork_urls(
                upstream_meta,
                upstream,
                upstream_rev,
                offline=offline,
                allow_readme=allow_readme,
                allow_base_model=False,
                rank=rank,
            )
            if upstream_urls:
                urls.extend(upstream_urls)
                break
    return list(dict.fromkeys(urls))
