"""Live lists of popular models from Civitai and Hugging Face, limited to what runs here.

The bundled catalogue is reviewed by hand and ages. This module asks the two public
listings what people download now: Civitai for image checkpoints and LoRAs,
Hugging Face for GGUF chat models. Only public listing endpoints are read, with no
account and no data about this installation beyond the query itself.

An entry is kept only when the application supports its base model and the
publisher states a SHA256 for the file, so a download can be verified. Lists are
cached in the data cache and refreshed when older than `MAX_AGE_SECONDS`; a
failed refresh keeps showing the last good list.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
import time

MAX_AGE_SECONDS = 6 * 3600
LIST_LIMIT = 100
THUMBNAIL_WIDTH = 320
MAX_THUMBNAIL_BYTES = 4 * 1024 * 1024
# Chat files above this are impractical on consumer hardware and are not offered.
MAX_CHAT_BYTES = 24 * 1024**3
CHAT_QUANTIZATIONS = ("Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q8_0", "Q4_0")

SOURCES = ("Civitai", "Hugging Face")
CATEGORIES = {
    "Civitai": ("Image models", "LoRAs"),
    "Hugging Face": ("Chat models",),
}
SORTS = {
    "Civitai": {
        "Most downloaded": "Most Downloaded",
        "Highest rated": "Highest Rated",
        "Newest": "Newest",
    },
    "Hugging Face": {
        "Trending": "trendingScore",
        "Most downloaded": "downloads",
        "Most liked": "likes",
    },
}
PERIODS = {"This week": "Week", "This month": "Month", "All time": "AllTime"}
_CIVITAI_TYPES = {"Image models": "Checkpoint", "LoRAs": "LORA"}
# Base-model names Civitai uses that the family table spells differently.
_EXTRA_FAMILIES = {"qwen21": "QwenImage21", "flux3": "Flux3", "flux2": "Flux2"}
_locks, _locks_guard = {}, threading.Lock()
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ ()+-]")


def offline():
    from ruinedfooocus.app.arguments import args

    return bool(getattr(args, "offline", False)) or any(
        os.environ.get(name, "").upper() in ("1", "TRUE", "YES", "ON")
        for name in ("RF_OFFLINE", "HF_HUB_OFFLINE")
    )


def _normal(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def supported_family(base):
    """Native family for a Civitai base-model name, or None when it does not run here."""
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS

    if not isinstance(base, str) or not base.strip():
        return None
    table = {_normal(name): name for name in IMAGE_COMPONENTS}
    table.update({_normal(alias): family for alias, family in ALIASES.items()})
    table.update(_EXTRA_FAMILIES)
    names = [base, re.sub(r"(?i)\s*turbo$", "", base), re.sub(r"(?i)turbo$", "", base)]
    for name in names:
        family = table.get(_normal(name))
        if family in IMAGE_COMPONENTS:
            return family
    return None


def safe_filename(name):
    """A plain file name for the models folder; never a path."""
    name = _SAFE_NAME.sub("_", Path(str(name).replace("\\", "/")).name).strip(" .")
    return name[:120] if name else ""


def _thumbnail_url(url):
    """Ask Civitai's image host for a small rendition of a listed picture."""
    if not isinstance(url, str) or not url.startswith("https://image.civitai.com/"):
        return None
    if re.search(r"/(?:original=true|width=\d+)[^/]*/", url):
        return re.sub(r"/(?:original=true|width=\d+)[^/]*/", f"/width={THUMBNAIL_WIDTH}/", url)
    head, _, tail = url.rpartition("/")
    return f"{head}/width={THUMBNAIL_WIDTH}/{tail}"


def parse_civitai(items, category, *, include_mature=False):
    """Listed models this application can run, each with one verifiable file."""
    entries = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("type") != _CIVITAI_TYPES[category]:
            continue
        if item.get("nsfw") and not include_mature:
            continue
        versions = [v for v in item.get("modelVersions") or [] if isinstance(v, dict)]
        version = next((v for v in versions if v.get("availability", "Public") == "Public"), None)
        if version is None:
            continue
        family = supported_family(version.get("baseModel"))
        if family is None:
            continue
        files = [f for f in version.get("files") or [] if isinstance(f, dict)]
        file = next((f for f in files if f.get("primary")), files[0] if files else None)
        if file is None:
            continue
        digest = str((file.get("hashes") or {}).get("SHA256", "")).lower()
        name = safe_filename(file.get("name", ""))
        url = file.get("downloadUrl")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not name.lower().endswith((".safetensors", ".gguf"))
            or not isinstance(url, str)
            or not url.startswith("https://civitai.com/api/download/models/")
        ):
            continue
        # Only a picture the site rates for everyone is shown by default;
        # when mature listings are explicitly enabled, fall back to the first non-video image.
        pictures = [
            picture
            for picture in version.get("images") or []
            if isinstance(picture, dict) and picture.get("type") in (None, "image")
        ]
        image = next(
            (picture.get("url") for picture in pictures if picture.get("nsfwLevel") == 1),
            next((picture.get("url") for picture in pictures), None) if include_mature else None,
        )
        stats = item.get("stats") or {}
        size = file.get("sizeKB")
        entries.append(
            dict(
                key=f"civitai:{version.get('id')}",
                source="Civitai",
                category=category,
                name=str(item.get("name", ""))[:120],
                version=str(version.get("name", ""))[:60],
                base=str(version.get("baseModel", "")),
                family=family,
                published=str(version.get("publishedAt") or "")[:10],
                downloads=int(stats.get("downloadCount") or 0),
                likes=int(stats.get("thumbsUpCount") or 0),
                file=name,
                bytes=int(size * 1024) if isinstance(size, (int, float)) else None,
                sha256=digest,
                url=url,
                image=_thumbnail_url(image),
                page=f"https://civitai.com/models/{item.get('id')}?modelVersionId={version.get('id')}",
                mature=bool(item.get("nsfw")),
            )
        )
    return entries


def parse_huggingface(items):
    """Public, ungated GGUF chat repositories; the file is chosen at download time."""
    from ruinedfooocus.features.models.model_sources import is_hf_artwork_url

    entries = []
    for item in items if isinstance(items, list) else []:
        identity = item.get("id") if isinstance(item, dict) else None
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", identity)
            or item.get("private")
            or item.get("gated")
            or "gguf" not in (item.get("tags") or [])
        ):
            continue
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        raw_thumb = card_data.get("thumbnail")
        image = raw_thumb.strip() if is_hf_artwork_url(raw_thumb) else None
        entries.append(
            dict(
                key=f"huggingface:{identity}",
                source="Hugging Face",
                category="Chat models",
                name=identity.split("/", 1)[1][:120],
                version=identity.split("/", 1)[0],
                base="GGUF",
                family="Chat",
                published=str(item.get("createdAt") or "")[:10],
                downloads=int(item.get("downloads") or 0),
                likes=int(item.get("likes") or 0),
                repository=identity,
                image=image,
                page=f"https://huggingface.co/{identity}",
                mature=False,
            )
        )
    return entries


def choose_chat_file(manifest):
    """One GGUF from a repository manifest: the first listed quantization that fits."""
    files = []
    for sibling in manifest.get("siblings") or [] if isinstance(manifest, dict) else []:
        name = sibling.get("rfilename") if isinstance(sibling, dict) else None
        lfs = sibling.get("lfs") if isinstance(sibling, dict) else None
        if (
            isinstance(name, str)
            and name.lower().endswith(".gguf")
            and "/" not in name
            and "mmproj" not in name.lower()
            and not re.search(r"-\d{5}-of-\d{5}", name)
            and isinstance(lfs, dict)
            and re.fullmatch(r"[0-9a-fA-F]{64}", str(lfs.get("sha256", "")))
            and isinstance(lfs.get("size"), int)
            and 0 < lfs["size"] <= MAX_CHAT_BYTES
        ):
            files.append((name, lfs["size"], lfs["sha256"].lower()))
    for wanted in CHAT_QUANTIZATIONS:
        for name, size, digest in sorted(files):
            if wanted.lower() in name.lower():
                return dict(file=name, bytes=size, sha256=digest)
    # No recognised quantization: do not guess which file is the model.
    return None


_HTTP_SESSION = None
_HTTP_SESSION_LOCK = threading.Lock()


def _http_session():
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        with _HTTP_SESSION_LOCK:
            if _HTTP_SESSION is None:
                import requests
                from requests.adapters import HTTPAdapter

                session = requests.Session()
                adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                session.headers.update({"User-Agent": "RuinedFooocus"})
                _HTTP_SESSION = session
    return _HTTP_SESSION


def _get_json(url, params):
    response = _http_session().get(url, params=params, timeout=(8, 40))
    response.raise_for_status()
    return response.json()


def query(source, category, sort, period):
    """One request to a public listing; returns its raw item list."""
    if source == "Civitai":
        data = _get_json(
            "https://civitai.com/api/v1/models",
            dict(
                types=_CIVITAI_TYPES[category],
                sort=SORTS[source][sort],
                period=PERIODS[period],
                limit=LIST_LIMIT,
            ),
        )
        return data.get("items") if isinstance(data, dict) else []
    return _get_json(
        "https://huggingface.co/api/models",
        dict(
            filter="gguf",
            pipeline_tag="text-generation",
            sort=SORTS[source][sort],
            direction=-1,
            limit=60,
            cardData="true",
        ),
    )


def _cache_file(cache_dir, source, category, sort, period, include_mature):
    name = hashlib.sha256(
        json.dumps([source, category, sort, period, bool(include_mature)]).encode()
    ).hexdigest()[:20]
    return Path(cache_dir) / "popular-models" / f"{name}.json"


def thumbnail_path(entry, cache_dir):
    """Where an entry's cached thumbnail is kept; the file may not exist yet."""
    url = entry.get("image")
    if isinstance(url, str) and url:
        key = url
    elif (
        entry.get("source") == "Hugging Face"
        and isinstance(entry.get("repository"), str)
        and entry["repository"]
    ):
        key = f"hf-repo:{entry['repository']}"
    else:
        return None
    name = hashlib.sha256(key.encode()).hexdigest()[:24] + ".jpeg"
    # Publisher artwork lives in the cache folder the interface already serves;
    # anywhere else Gradio copies every picture again on each update.
    return Path(cache_dir) / "catalogue-previews" / "popular" / name


def _resolve_popular_artwork_urls(entry):
    url = entry.get("image")
    if isinstance(url, str) and url:
        return [url]
    repo = entry.get("repository")
    if entry.get("source") != "Hugging Face" or not isinstance(repo, str) or not repo:
        return []
    from ruinedfooocus.features.models import model_sources
    from ruinedfooocus.features.models.catalogue_previews import _preview_rank

    manifest = model_sources.huggingface_manifest(repo, offline=offline())
    if not isinstance(manifest, dict):
        return []
    revision = manifest.get("sha") or "main"
    return model_sources.extract_hf_artwork_urls(
        manifest, repo, revision, offline=offline(), rank=_preview_rank
    )[:3]


def fetch_artwork_batches(entries, cache_dir, fetch_image=None, *, batch_size=6, max_wait=0.2):
    """Stream newly cached `(entry, target)` batches over pooled keep-alive sockets."""
    from ruinedfooocus.features.models.model_previews import write_preview

    fetch = fetch_image or _read_image
    missing = [
        (entry, target)
        for entry in entries
        if (target := thumbnail_path(entry, cache_dir)) is not None and not target.is_file()
    ]
    if not missing or offline():
        return

    def one(item):
        entry, target = item
        try:
            urls = _resolve_popular_artwork_urls(entry)
            for url in urls:
                try:
                    body = fetch(url)
                    if not body or len(body) > MAX_THUMBNAIL_BYTES:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_preview(body, target)
                    return entry, target
                except Exception:
                    continue
            return None
        except Exception:  # One missing picture leaves that card on its placeholder.
            return None

    step = max(1, int(batch_size or 1))
    pending_batch = []
    last_flush = time.monotonic()
    with ThreadPoolExecutor(max_workers=24, thread_name_prefix="popular-artwork") as pool:
        futures = [pool.submit(one, item) for item in missing]
        for future in as_completed(futures):
            try:
                saved = future.result()
                if saved is not None:
                    pending_batch.append(saved)
                    now = time.monotonic()
                    if len(pending_batch) >= step or (max_wait and now - last_flush >= max_wait):
                        yield pending_batch
                        pending_batch = []
                        last_flush = now
            except Exception:
                continue
    if pending_batch:
        yield pending_batch


def fetch_artwork_iter(entries, cache_dir, fetch_image=None, *, batch_size=4):
    """Stream cumulative arrival counts in small batches."""
    arrived = 0
    for batch in fetch_artwork_batches(
        entries, cache_dir, fetch_image=fetch_image, batch_size=batch_size, max_wait=0
    ):
        arrived += len(batch)
        yield arrived


def fetch_artwork(entries, cache_dir, fetch_image=None):
    """Download the thumbnails that are not cached yet. Returns how many arrived."""
    total = 0
    for total in fetch_artwork_iter(entries, cache_dir, fetch_image=fetch_image):
        pass
    return total


def _read_image(url):
    with _http_session().get(url, timeout=(6, 15), stream=True) as response:
        response.raise_for_status()
        body = io.BytesIO()
        for chunk in response.iter_content(64 * 1024):
            body.write(chunk)
            if body.tell() > MAX_THUMBNAIL_BYTES:
                return None
        return body.getvalue()


def _check_selection(source, category, sort, period):
    if (
        source not in SOURCES
        or category not in CATEGORIES[source]
        or sort not in SORTS[source]
        or period not in PERIODS
    ):
        raise ValueError("Choose a listed source, category, sort order and period.")


def _saved(path):
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(saved, dict) and isinstance(saved.get("entries"), list):
            return saved
    except (OSError, ValueError):
        pass
    return None


def _fresh(saved, now):
    age = now - saved["time"] if saved and isinstance(saved.get("time"), (int, float)) else None
    return age is not None and 0 <= age < MAX_AGE_SECONDS


def peek(source, category, sort, period="This month", *, cache_dir, include_mature=False, now=None):
    """The saved list for one selection without any request, or None.

    `stale` says it is older than `MAX_AGE_SECONDS` and due for a refresh.
    """
    _check_selection(source, category, sort, period)
    saved = _saved(_cache_file(cache_dir, source, category, sort, period, include_mature))
    if saved is None:
        return None
    return dict(
        entries=saved["entries"],
        fetched=saved.get("fetched"),
        stale=not _fresh(saved, time.time() if now is None else now),
        error="",
    )


def load(
    source,
    category,
    sort,
    period="This month",
    *,
    cache_dir,
    include_mature=False,
    force=False,
    is_offline=None,
    fetch_items=None,
    now=None,
):
    """The list for one selection: fresh when possible, otherwise the last good one.

    Returns `entries`, `fetched` (UTC text or None), `stale` and `error`. Artwork
    is fetched separately by `fetch_artwork`, so a list never waits for pictures.
    """
    _check_selection(source, category, sort, period)
    now = time.time() if now is None else now
    path = _cache_file(cache_dir, source, category, sort, period, include_mature)
    saved = _saved(path)
    blocked = (offline if is_offline is None else is_offline)()
    if saved and not force and _fresh(saved, now):
        return dict(entries=saved["entries"], fetched=saved.get("fetched"), stale=False, error="")
    if blocked:
        return dict(
            entries=saved["entries"] if saved else [],
            fetched=saved.get("fetched") if saved else None,
            stale=bool(saved),
            error="Offline mode: popular lists need a connection.",
        )
    with _locks_guard:
        lock = _locks.setdefault(str(path), threading.Lock())
    try:
        with lock:
            # Two controls can ask for the same list at once; the second reuses
            # what the first has just saved instead of asking the site again.
            other = _saved(path)
            if other and other.get("time") != (saved or {}).get("time") and _fresh(other, now):
                return dict(
                    entries=other["entries"], fetched=other.get("fetched"), stale=False, error=""
                )
            items = (fetch_items or query)(source, category, sort, period)
            entries = (
                parse_civitai(items, category, include_mature=include_mature)
                if source == "Civitai"
                else parse_huggingface(items)
            )
            fetched = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(dict(time=now, fetched=fetched, entries=entries)), encoding="utf-8"
            )
            os.replace(temporary, path)
            return dict(entries=entries, fetched=fetched, stale=False, error="")
    except Exception as error:  # A listing outage must not break the page.
        return dict(
            entries=saved["entries"] if saved else [],
            fetched=saved.get("fetched") if saved else None,
            stale=bool(saved),
            error=f"The {source} list could not be refreshed: {str(error)[:160]}",
        )


def installed_path(entry, manager):
    """Where an entry's file is, or would be, in the model folders."""
    folder = {
        "Image models": "modelfile_path",
        "LoRAs": "lorafile_path",
        "Chat models": "llm_path",
    }[entry["category"]]
    root = manager.model_paths[folder]
    root = root[0] if isinstance(root, (list, tuple)) else root
    name = safe_filename(entry.get("file", ""))
    return Path(root) / name if name else None


def resolve_file(entry, *, manifest=None):
    """Fill in the file of a Hugging Face entry from its repository manifest."""
    if entry.get("file") and entry.get("sha256") and entry.get("url"):
        return entry
    if entry.get("source") != "Hugging Face":
        raise ValueError("This entry has no downloadable file.")
    if manifest is None:
        from ruinedfooocus.features.models.model_sources import huggingface_manifest

        manifest = huggingface_manifest(entry["repository"], offline=offline())
    chosen = choose_chat_file(manifest)
    if chosen is None:
        raise ValueError(
            "This repository has no single GGUF file in a common quantization and usable size."
        )
    revision = manifest["sha"]
    return dict(
        entry,
        **chosen,
        url=f"https://huggingface.co/{entry['repository']}/resolve/{revision}/{chosen['file']}",
    )


def explain_failure(error, entry):
    """A download failure in words the reader can act on."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in (401, 403):
        return (
            f"{entry['source']} only gives this file to signed-in accounts. "
            f"Download it from {entry['page']} and put it in the models folder."
        )
    if status == 404:
        return "The publisher has removed this file. Refresh the list."
    return f"The download did not finish: {str(error)[:200] or type(error).__name__}"


def start_download(entry, manager, *, progress=None, service=None):
    """Queue one entry for its model folder; the publisher's SHA256 is verified.

    Returns the destination and a subscription, which is None when the file exists.
    """
    from ruinedfooocus.core.preparation import PreparationRequest, preparation_service

    if offline():
        raise ValueError("Offline mode: downloads need a connection.")
    entry = resolve_file(entry)
    destination = installed_path(entry, manager)
    if destination is None:
        raise ValueError("This entry has no usable file name.")
    if destination.exists():
        return destination, None
    destination.parent.mkdir(parents=True, exist_ok=True)
    service = preparation_service if service is None else service
    raw_bytes = entry.get("bytes")
    expected_size = (
        int(raw_bytes)
        if isinstance(raw_bytes, int) and not isinstance(raw_bytes, bool) and raw_bytes > 0
        else None
    )
    return destination, service.submit(
        PreparationRequest(
            entry["url"],
            destination,
            expected_size=expected_size,
            sha256=entry["sha256"],
            model_root=destination.parent,
        ),
        progress=progress,
    )


def download(entry, manager, *, progress=None, check_cancelled=None, service=None):
    """Download one entry and wait for it."""
    destination, subscription = start_download(entry, manager, progress=progress, service=service)
    if subscription is None:
        return destination
    while True:
        if check_cancelled is not None:
            check_cancelled()
        try:
            subscription.result(timeout=0.3)
            return destination
        except TimeoutError:
            if subscription.done():
                raise
