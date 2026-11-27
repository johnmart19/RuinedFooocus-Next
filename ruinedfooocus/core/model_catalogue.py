"""Authoritative model catalogue loader, validator, and ComfyUI template sync."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse

from ruinedfooocus.core.paths import resource_path

COMFYUI_DIRECTORY_MAP = {
    "checkpoints": ("checkpoints", "path_checkpoints"),
    "diffusion_models": ("diffusion_models", "path_diffusion_models"),
    "text_encoders": ("clip", "path_clip"),
    "clip": ("clip", "path_clip"),
    "clip_vision": ("clip_vision", "path_clip_vision"),
    "vae": ("vae", "path_vae"),
    "loras": ("loras", "path_loras"),
    "controlnet": ("controlnet", "path_controlnet"),
    "upscale_models": ("upscalers", "path_upscalers"),
    "embeddings": ("embeddings", "path_embeddings"),
    "model_patches": ("model_patches", "path_model_patches"),
}

_NON_IMAGE_TOKENS = (
    "video_",
    "audio_",
    "3d_",
    "api_",
    "wan2",
    "wan_",
    "wanmove",
    "ltxv",
    "ltx2",
    "ltx_",
    "ace_step",
    "acestep",
    "hunyuan_video",
    "hunyuanvideo",
    "mochi",
    "cosmos",
    "cogvideo",
    "svd",
    "tripo",
    "trellis",
    "yue2",
    "pan2",
    "stable_audio",
    "minimaxmusic",
)


def catalogue_directory() -> Path:
    return resource_path("model_catalogue")


def manifests_directory() -> Path:
    return resource_path("model_catalogue", "manifests")


def load_manifest(name: str) -> dict:
    """Load a multi-file downloadable manifest from resources/model_catalogue/manifests/."""
    candidate = manifests_directory() / name
    if not candidate.is_file():
        candidate = resource_path(name)
    return json.loads(candidate.read_text(encoding="utf-8"))


def load_catalogue(folder: Path | None = None) -> tuple[dict, dict[str, str]]:
    """Deterministically load all model_catalogue/*.json files and track source provenance."""
    root = Path(folder) if folder is not None else catalogue_directory()
    entries: dict = {}
    provenance: dict[str, str] = {}
    if not root.is_dir():
        return entries, provenance
    for path in sorted(root.glob("*.json"), key=lambda item: item.name.casefold()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"Error reading {path.name}: {error}")
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            entries[key] = value
            provenance[key] = path.name
    return entries, provenance


def validate_catalogue(entries: dict | None = None) -> list[str]:
    """Validate catalogue entries for safe filenames, valid URLs, sizes, and SHA256."""
    if entries is None:
        entries, _ = load_catalogue()
    issues: list[str] = []
    for key, info in sorted(entries.items()):
        if not isinstance(info, dict):
            issues.append(f"{key}: entry must be a JSON object")
            continue
        url = info.get("url")
        if not isinstance(url, str) or urlparse(url).scheme not in ("https", "http"):
            issues.append(f"{key}: invalid URL {url!r}")
        folder = info.get("path")
        if not isinstance(folder, str) or not folder.startswith("path_"):
            issues.append(f"{key}: invalid path key {folder!r}")
        filename = info.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
        ):
            issues.append(f"{key}: unsafe filename {filename!r}")
        size = info.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            issues.append(f"{key}: size must be a positive integer")
        sha256 = info.get("sha256")
        if sha256 is not None:
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(ch not in "0123456789abcdefABCDEF" for ch in sha256)
            ):
                issues.append(f"{key}: invalid sha256 digest")
    return issues


def _is_non_image_template(template_name: str, model_name: str, url: str) -> bool:
    haystack = f"{template_name} {model_name} {url}".casefold()
    return any(token in haystack for token in _NON_IMAGE_TOKENS)


def discover_comfyui_template_models(
    existing_entries: dict | None = None,
    *,
    templates_root: Path | None = None,
    include_non_image: bool = False,
) -> list[dict]:
    """Discover downloadable models from installed ComfyUI workflow templates not yet in RF."""
    if existing_entries is None:
        existing_entries, _ = load_catalogue()
    existing_urls = {
        str(item.get("url", "")).split("?", 1)[0]
        for item in existing_entries.values()
        if isinstance(item, dict)
    }
    existing_files = {
        str(item.get("filename", "")).casefold()
        for item in existing_entries.values()
        if isinstance(item, dict)
    }
    if templates_root is None:
        spec = importlib.util.find_spec("comfyui_workflow_templates_json")
        if spec is None or not spec.origin:
            return []
        templates_root = Path(spec.origin).parent
    templates_root = Path(templates_root)
    if not templates_root.is_dir():
        return []

    discovered: dict[str, dict] = {}
    for json_path in sorted(templates_root.rglob("*.json")):
        if not include_non_image and _is_non_image_template(json_path.name, "", ""):
            continue
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(obj):
            if isinstance(obj, dict):
                models = obj.get("models")
                if isinstance(models, list):
                    for item in models:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name")
                        url = item.get("url")
                        directory = item.get("directory", "")
                        if not isinstance(name, str) or not isinstance(url, str):
                            continue
                        if directory not in COMFYUI_DIRECTORY_MAP:
                            continue
                        if not include_non_image and _is_non_image_template(
                            json_path.name, name, url
                        ):
                            continue
                        clean_url = url.split("?", 1)[0]
                        if name.casefold() in existing_files or clean_url in existing_urls:
                            continue
                        category, path_key = COMFYUI_DIRECTORY_MAP[directory]
                        discovered[name] = {
                            "catalogue_key": f"{category}/{name}",
                            "filename": name,
                            "category": category,
                            "path": path_key,
                            "url": clean_url,
                            "directory": directory,
                            "template": json_path.name,
                        }
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        walk(doc)
    return [discovered[name] for name in sorted(discovered, key=str.casefold)]


def resolve_hf_lfs_metadata(url: str, *, timeout: int = 15) -> dict:
    """Resolve a Hugging Face /resolve/ URL to pinned commit SHA, exact bytes, and SHA256."""
    import requests

    clean = url.split("?", 1)[0]
    parsed = urlparse(clean)
    if parsed.hostname != "huggingface.co" or "/resolve/" not in parsed.path:
        raise ValueError(f"Not a Hugging Face resolve URL: {url}")
    repo_part, _, rest = parsed.path.lstrip("/").partition("/resolve/")
    _, _, rel_path = rest.partition("/")
    if not repo_part or not rel_path:
        raise ValueError(f"Incomplete Hugging Face resolve URL: {url}")
    head = requests.head(clean, allow_redirects=False, timeout=timeout)
    if head.status_code == 307 and head.headers.get("Location"):
        redirect = head.headers["Location"]
        if redirect.startswith("/"):
            redirect = f"https://huggingface.co{redirect}"
        clean = redirect.split("?", 1)[0]
        repo_part, _, rest = urlparse(clean).path.lstrip("/").partition("/resolve/")
        _, _, rel_path = rest.partition("/")
        head = requests.head(clean, allow_redirects=False, timeout=timeout)
    if head.status_code not in (200, 302):
        raise RuntimeError(f"HTTP {head.status_code} for {clean}")
    size_raw = head.headers.get("X-Linked-Size") or head.headers.get("Content-Length")
    size = int(size_raw) if size_raw and size_raw.isdigit() else None
    etag = (head.headers.get("X-Linked-Etag") or "").strip('"')
    sha256 = etag.lower() if len(etag) == 64 else None
    repo_resp = requests.get(f"https://huggingface.co/api/models/{repo_part}", timeout=timeout)
    commit_sha = repo_resp.json().get("sha") if repo_resp.status_code == 200 else None
    pinned_url = (
        f"https://huggingface.co/{repo_part}/resolve/{commit_sha}/{rel_path}"
        if commit_sha
        else clean
    )
    return {
        "repo": repo_part,
        "commit": commit_sha,
        "relative_path": rel_path,
        "pinned_url": pinned_url,
        "size": size,
        "sha256": sha256,
        "http_status": head.status_code,
    }
