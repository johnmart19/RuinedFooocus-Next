"""Inventory first-party acquisition sources and safely probe public download links.

Uses the standard library and the pure bundled model-registry reader. Does not
import tensor/application services, install packages, use credentials, or read
artifact bodies. See docs/download-audit.md for coverage.
"""

import argparse
import ast
from collections.abc import Mapping
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URL = re.compile(r"https?://[^\s<>\"']+")
REPORT_URL = re.compile(r"https?://[^\s\"']+")
MAX_REDIRECTS = 8
PACKAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[^\]]+\])?\s*(?:==|>=|<=|~=|!=|<|>).+")
ARTIFACT = re.compile(
    r"\.(?:gguf|safetensors|pth|pt|bin|onnx|zip|gz|bz2|whl|wasm|mjs|js)(?:$|\?)", re.I
)


def public_url(url):
    """Remove credentials, query values and fragments from report evidence."""
    parts = urlsplit(url)
    # Preserve symbolic ports in unresolved source templates.
    host = parts.netloc.rsplit("@", 1)[-1]
    query = "&".join(
        f"{item.split('=', 1)[0]}=<redacted>" for item in parts.query.split("&") if item
    )
    path = quote(parts.path, safe="/%:@!$&'()*+,;=-._~{}")
    return urlunsplit((parts.scheme, host, path, query, ""))


def sanitize_report_value(value):
    """Redact URL credentials/query values in nested informational fields too."""
    if isinstance(value, str):

        def redact(match):
            try:
                return public_url(match.group())
            except ValueError:
                return "<redacted-invalid-url>"

        return REPORT_URL.sub(redact, value)
    if isinstance(value, dict):
        return {
            sanitize_report_value(key): sanitize_report_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_report_value(item) for item in value]
    return value


def identity(url):
    parts = urlsplit(url)
    path = parts.path.strip("/").split("/")
    if parts.hostname == "huggingface.co" and len(path) >= 5 and path[2] == "resolve":
        return {
            "provider": "huggingface",
            "repository": "/".join(path[:2]),
            "revision": path[3],
            "file": "/".join(path[4:]),
        }
    if parts.hostname == "github.com" and len(path) >= 6 and path[2:4] == ["releases", "download"]:
        return {
            "provider": "github",
            "repository": "/".join(path[:2]),
            "version": path[4],
            "file": "/".join(path[5:]),
        }
    return {}


def literal(node, values):
    """Resolve only literal expressions; never execute source or call functions."""
    if isinstance(node, ast.Name):
        return values.get(node.id, "{" + node.id + "}")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(literal(part.value, values))
            if isinstance(part, ast.FormattedValue)
            else str(part.value)
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal(node.left, values), literal(node.right, values)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    return "{" + ast.unparse(node) + "}"


def plain(value):
    """Project immutable metadata without asking a lazy profile for qualification."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def annotate_registry(root, downloads, download_rows, source, add):
    """Join exact tuples through the public reader; history is outside this audit."""
    metadata = root / "ruinedfooocus/resources/model_metadata"
    index = metadata / "index.json"
    if not index.is_file():
        return
    from ruinedfooocus.features.models.model_registry import RegistryError, load_registry

    owner = source(index)
    try:
        registry = load_registry(metadata)
        entries = registry.entries(include_proposed=True)
        # Record only the metadata families/artifacts this audit may inspect.
        # Qualification/benchmarks stay lazy and are neither read nor hashed.
        for directory in ("families", "artifacts"):
            for path in sorted((metadata / directory).glob("*.json")):
                source(path)
        profiles = {}
        for entry in entries:
            key = entry["catalogue_key"]
            if entry["state"] == "existing":
                if key not in downloads:
                    raise RegistryError(f"Annotated download is absent from catalogue: {key}")
                registry.annotations_for(key, downloads[key])
                row = download_rows[key]
            else:
                candidate = entry.get("proposed_download", {})
                row = add(
                    owner,
                    entry["id"],
                    candidate.get("url"),
                    kind="resolver",
                    reason="Proposed annotation only; not a selectable catalogue download.",
                )
            row["registry"] = {
                field: plain(entry[field])
                for field in (
                    "id",
                    "profile_id",
                    "role",
                    "architecture",
                    "native_family",
                    "behavioral_variant",
                    "format",
                    "quantization",
                    "runtime",
                    "companions",
                    "evidence",
                )
                if field in entry
            }
            row["registry"]["download_tuple_matches"] = entry["state"] == "existing"
            row["expected_sha256"] = entry["artifact"].get("sha256")
            row["expected_bytes"] = entry["artifact"].get("bytes")
            identity = entry["profile_id"]
            if identity not in profiles:
                profile = registry.profile(identity)
                profiles[identity] = {
                    field: plain(profile[field])
                    for field in (
                        "identity",
                        "provenance",
                        "release",
                        "sources",
                        "dependencies",
                        "variants",
                    )
                    if field in profile
                }
        for identity, evidence in profiles.items():
            add(owner, identity, kind="reference", profile_evidence=evidence)

            def references(value, location):
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    add(owner, location, value, kind="reference", profile_id=identity)
                elif isinstance(value, dict):
                    for key, item in value.items():
                        references(item, f"{location}.{key}")
                elif isinstance(value, list):
                    for number, item in enumerate(value):
                        references(item, f"{location}[{number}]")

            references(evidence, identity)
    except RegistryError as error:
        add(owner, "registry-validation", kind="resolver", reason=str(error))


def inventory(root=ROOT):
    rows, sources = [], {}

    def source(path):
        name = path.relative_to(root).as_posix()
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return name

    def add(owner, key, url=None, kind="artifact", reason=None, **extra):
        row = {
            "owner": owner,
            "key": key,
            "kind": kind,
            "url": public_url(url) if url else None,
            "status": "not_checked",
            "reason": "Offline inventory; no network probe has run.",
            **sanitize_report_value(extra),
        }
        if url:
            row.update(identity(url))
        if kind in ("resolver", "template", "package", "repository"):
            row.update(
                status="unresolved",
                reason=reason
                or "Requires target OS, architecture, Python ABI, device and resolver metadata; no artifact URL was inferred.",
            )
        elif kind == "reference":
            row.update(
                status="reference_only",
                reason="Source/reference URL; not an artifact availability claim.",
            )
        if url and (urlsplit(url).username or urlsplit(url).query):
            row.update(
                status="unresolved",
                reason="Credential-bearing or query URL withheld from probing; resolve a public artifact identity explicitly.",
            )
        rows.append(row)
        return row

    # The catalogue, not registry annotations, is authoritative for downloads.
    catalog = root / "ruinedfooocus/resources/model_catalogue"
    downloads, download_rows = {}, {}
    for path in sorted(catalog.glob("*.json")):
        owner = source(path)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for key, entry in data.items():
            if not isinstance(entry, dict) or not entry.get("url"):
                add(
                    owner,
                    key,
                    kind="resolver",
                    reason="Catalogue entry has no explicit URL; inspect its schema.",
                )
                continue
            add(
                owner,
                key,
                entry["url"],
                destination=entry.get("filename"),
                model_path=entry.get("path"),
            )
            downloads[key], download_rows[key] = entry, rows[-1]
            for field, value in entry.items():
                if (
                    field != "url"
                    and isinstance(value, str)
                    and value.startswith(("http://", "https://"))
                ):
                    add(
                        owner,
                        f"{key}.{field}",
                        value,
                        kind="artifact" if ARTIFACT.search(value) else "reference",
                    )

    annotate_registry(root, downloads, download_rows, source, add)

    assets = root / "tools/validation_assets.json"
    if assets.exists():
        owner = source(assets)
        for index, entry in enumerate(json.loads(assets.read_text(encoding="utf-8-sig"))["assets"]):
            url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{quote(entry['file'], safe='/')}"
            add(
                owner,
                f"assets[{index}]",
                url,
                expected_sha256=entry.get("sha256"),
                expected_bytes=entry.get("bytes"),
            )

    # All first-party Python; explicitly prune vendor and resource/prompt data.
    paths, code_requirement_paths = [], []
    for directory, subdirectories, filenames in os.walk(root / "ruinedfooocus"):
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in {"vendor", "resources", "__pycache__"}
        )
        paths.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
        code_requirement_paths.extend(
            Path(directory) / name
            for name in filenames
            if name.startswith("requirements") and name.endswith(".txt")
        )
    paths += list(root.glob("*.py"))
    for path in sorted(paths):
        owner = source(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=owner)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                value = literal(node.value, values)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = value
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Constant, ast.JoinedStr)):
                continue
            parent = parents.get(node)
            if isinstance(parent, (ast.JoinedStr, ast.FormattedValue)):
                continue
            value = literal(node, values)
            if not isinstance(value, str):
                continue
            key = f"line:{node.lineno}"
            for match in URL.finditer(value):
                url = match.group().rstrip(",;)")
                if "{" in url:
                    kind = "template"
                elif ARTIFACT.search(url) or "/resolve/" in url:
                    kind = "artifact"
                elif any(
                    token in url
                    for token in ("/whl", "/simple", "/pypi/", "/api/", "api.github.com", "/full/")
                ):
                    kind = "endpoint"
                else:
                    kind = "reference"
                if url.endswith("/releases/tags/"):
                    kind = "resolver"
                add(owner, key, url, kind)
            if PACKAGE.match(value) and "\n" not in value:
                add(owner, key, kind="package", requirement=value)
            if (
                isinstance(node, ast.JoinedStr)
                and ("==" in value or ".whl" in value)
                and not URL.search(value)
            ):
                add(owner, key, kind="resolver", expression=value)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("RuntimeBundle", "NativeRuntime")
            ):
                args = [literal(arg, values) for arg in node.args]
                fields = (
                    ("torch", "torchvision", "index")
                    if node.func.id == "RuntimeBundle"
                    else ("package", "version", "provider", "systems")
                )
                declaration = dict(zip(fields, args))
                declaration.update(
                    {
                        keyword.arg: literal(keyword.value, values)
                        for keyword in node.keywords
                        if keyword.arg
                    }
                )
                add(
                    owner,
                    f"line:{node.lineno}:{node.func.id}",
                    kind="resolver",
                    declaration=declaration,
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr
                in ("from_pretrained", "hf_hub_download", "snapshot_download", "ensure_ready_async")
            ):
                target = literal(node.args[0], values) if node.args else None
                add(
                    owner,
                    f"line:{node.lineno}:{node.func.attr}",
                    kind="resolver",
                    expression=target,
                    reason="Dependency resolves additional files or vendor assets at runtime; repository/model ID is not a complete file manifest.",
                )
        if owner.endswith("app/bootstrap.py"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "git_repos" for t in node.targets
                ):
                    for repo in ast.literal_eval(node.value):
                        add(
                            owner,
                            f"git_repos:{repo['name']}",
                            repo["url"] + "/archive/" + repo["hash"] + ".tar.gz",
                            revision=repo["hash"],
                        )
        if owner.endswith("llama/llama_installer.py"):
            add(
                owner,
                "RELEASES+VERSION",
                values["RELEASES"] + values["VERSION"],
                kind="endpoint",
                version=values["VERSION"],
            )

    # pip inputs are inventories, never a claim that every target has a wheel.
    requirement_paths = list(root.glob("requirements*.txt"))
    if (root / "pip/modules.txt").exists():
        requirement_paths.append(root / "pip/modules.txt")
    requirement_paths.extend(code_requirement_paths)
    for path in sorted(requirement_paths):
        if "vendor" in path.parts:
            continue
        owner = source(path)
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if line and not line.startswith("#"):
                add(owner, f"line:{number}", kind="package", requirement=line)
    # No dependency graph, dynamic backend matrix or transitive CDN files are hidden.
    for owner, reason in (
        (
            "ruinedfooocus/packaging/runtime_bundles.py",
            "RuntimeBundle wheel matrix requires Python/OS/architecture/device extras and index resolution; static index health does not validate package versions.",
        ),
        (
            "ruinedfooocus/backends/llama/llama_installer.py",
            "Official release assets and companion libraries are selected by package_name/platform/backend and release metadata; enumerate the target context before validating the exact asset.",
        ),
        (
            "ruinedfooocus/backends/onnx/winml_catalog.py",
            "Windows ML certified provider catalogue selects vendor downloads for local hardware; requires a Windows target context.",
        ),
        (
            "ruinedfooocus/backends/onnx/model_contracts.py",
            "rembg resolves its selected model weights through its installed package/session registry; audit that dependency manifest for the selected model and version.",
        ),
        (
            "ruinedfooocus/features/image_processing/interrogate.py",
            "clip_interrogator/OpenCLIP and Transformers resolve transitive weights, tokenizers and configs; inspect pinned dependency manifests for complete file coverage.",
        ),
        (
            "ruinedfooocus/features/models/model_sources.py",
            "User-selected Hub/Civitai searches and arbitrary remote URLs are open-ended; audit a captured exact selection separately.",
        ),
    ):
        if (root / owner).exists():
            add(owner, "dynamic-acquisition-coverage", kind="resolver", reason=reason)
    return rows, sources


class NoAutomaticRedirect(HTTPRedirectHandler):
    """Return redirect streams untouched; urllib's default handler drains them."""

    def http_error_302(self, req, fp, code, msg, headers):
        return fp

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def classify(code):
    if 200 <= code < 300:
        return "reachable"
    if code == 401:
        return "authentication_required"
    if code == 403:
        return "access_denied"
    if code in (404, 410):
        return "broken"
    if code == 429:
        return "rate_limited"
    if code in (408, 425) or code >= 500:
        return "transient_failure"
    return "http_failure"


def normalize_endpoint_result(row):
    """Keep base/index HTTP health separate from downloadable artifact status.

    Also accepts a previously saved row, allowing reinterpretation without any
    network request. The original HTTP evidence and probe status are retained.
    """
    if row["kind"] != "endpoint" or row["status"] in ("not_checked", "unresolved"):
        return row
    status = row.get("probe_status", row["status"])
    if status.startswith("endpoint_"):
        return row
    row["probe_status"] = status
    row["probe_reason"] = row.get("probe_reason", row.get("reason"))
    # HTML is a normal representation for many package indexes.
    if status == "unexpected_content":
        status = "reachable"
    row["status"] = "endpoint_" + ("not_found" if status == "broken" else status)
    row["availability_scope"] = "base/index endpoint only; no resolved artifact guarantee"
    row["reason"] = (
        "Endpoint/base probe result does not establish whether an actual API resource or package artifact is available."
    )
    row["action"] = (
        "Resolve/use the actual API endpoint or package artifact for the target context; retain this base/index HTTP result as diagnostic evidence."
    )
    return row


class Probe:
    def __init__(self, timeout=12, retries=1, interval=0.2, opener=None):
        self.timeout, self.retries, self.interval = timeout, retries, interval
        self.opener = opener or build_opener(NoAutomaticRedirect())
        self.lock = threading.Lock()
        self.next_request = 0.0

    def rate_limit(self):
        # Shared by workers, retries, metadata calls and every redirect hop.
        with self.lock:
            pause = max(0, self.next_request - time.monotonic())
            if pause:
                time.sleep(pause)
            self.next_request = time.monotonic() + self.interval

    def open_response(self, url, method, headers):
        current, visited, redirects = url, {url}, []
        for hop in range(MAX_REDIRECTS + 1):
            self.rate_limit()
            request = Request(current, headers=headers, method=method)
            try:
                response = self.opener.open(request, timeout=self.timeout)
            except HTTPError as error:
                response = error
            code = response.status if hasattr(response, "status") else response.code
            if code not in (301, 302, 303, 307, 308):
                return response, redirects, None
            # Never drain a redirect response, including redirected metadata GETs.
            try:
                location = response.headers.get("Location")
                evidence = {
                    "method": method,
                    "http_status": code,
                    "final_url": public_url(response.geturl()),
                }
                if location:
                    target = urljoin(current, location)
                    parts = urlsplit(target)
                    evidence["location"] = public_url(target)
                redirects.append(evidence)
            finally:
                response.close()
            failure = None
            if not location:
                failure = "redirect_missing_location"
            elif (
                parts.scheme not in ("http", "https")
                or not parts.hostname
                or parts.username
                or (urlsplit(current).scheme == "https" and parts.scheme != "https")
            ):
                failure = "redirect_rejected"
            elif target in visited:
                failure = "redirect_loop"
            elif hop == MAX_REDIRECTS:
                failure = "redirect_limit"
            if failure:
                return (
                    None,
                    redirects,
                    {
                        "status": failure,
                        "reason": "Redirect could not be followed within the public URL, TLS and hop limits; response bodies were not read.",
                        "evidence": {**evidence, "redirects": redirects},
                    },
                )
            # Keep signed query values private in request state, never evidence.
            current = urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    quote(parts.path, safe="/%:@!$&'()*+,;=-._~"),
                    parts.query,
                    "",
                )
            )
            visited.add(target)
        raise AssertionError("Redirect loop exceeded its bounded iteration")

    def request(self, url, method="HEAD", metadata=False):
        headers = {"User-Agent": "RuinedFooocus-download-audit/1", "Accept-Encoding": "identity"}
        if method == "GET" and not metadata:
            headers["Range"] = "bytes=0-0"
        response, redirects, failure = self.open_response(url, method, headers)
        if failure:
            return failure
        with response:
            code = response.status if hasattr(response, "status") else response.code
            evidence = {
                "method": method,
                "http_status": code,
                "final_url": public_url(response.geturl()),
            }
            if redirects:
                evidence["redirects"] = redirects
            for key in (
                "Content-Length",
                "Content-Type",
                "Content-Range",
                "ETag",
                "X-Repo-Commit",
                "Retry-After",
            ):
                if response.headers.get(key):
                    evidence[key.lower()] = response.headers[key]
            # Artifact response bodies are never read, even if Range is ignored.
            if metadata and code == 200:
                body = response.read(1024 * 1024 + 1)
                if len(body) > 1024 * 1024:
                    return {
                        "status": "metadata_limit",
                        "evidence": evidence,
                        "reason": "Metadata exceeded 1 MiB cap.",
                    }
                return {"status": "reachable", "evidence": evidence, "data": json.loads(body)}
            status = classify(code)
            if (
                status == "reachable"
                and "text/html" in response.headers.get("Content-Type", "").lower()
            ):
                status = "unexpected_content"
            reasons = {
                "reachable": "Server accepted the artifact/header request; payload was not read.",
                "broken": "Server reports missing/gone (404/410); a private repository may also hide existence.",
                "authentication_required": "Server requires authentication; this does not establish that the artifact is missing.",
                "access_denied": "Public access denied; possible policy, CDN, anti-bot, authentication or gated-access restriction. The cause is unconfirmed.",
                "rate_limited": "Server rate limit; retry later before declaring an artifact missing.",
                "transient_failure": "Timeout or server failure; retry later before changing source.",
                "unexpected_content": "HTML response may be a landing/login page; inspect the artifact URL.",
                "http_failure": "Other HTTP failure; inspect response code and source context.",
            }
            return {"status": status, "evidence": evidence, "reason": reasons[status]}

    def __call__(self, url, metadata=False):
        last = None
        for attempt in range(self.retries + 1):
            try:
                last = self.request(url, "GET" if metadata else "HEAD", metadata)
                if not metadata and last["evidence"]["http_status"] in (405, 501):
                    last = self.request(url, "GET")
            except (URLError, OSError, ValueError) as error:
                # Exception strings can contain proxy credentials and signed URLs.
                last = {
                    "status": "network_failure"
                    if not isinstance(error, ValueError)
                    else "invalid_metadata",
                    "reason": type(error).__name__,
                }
            last["attempts"] = attempt + 1
            if (
                last["status"] not in ("rate_limited", "transient_failure", "network_failure")
                or attempt == self.retries
            ):
                return last
            time.sleep(min(2**attempt, 4))
        return last


def audit(rows, probe, workers=4, limit=None, compare_metadata=False, progress=None):
    def safe_probe(url, **options):
        try:
            return probe(url, **options)
        except Exception as error:
            # Keep the report actionable if one URL or custom worker fails.
            # Exception text may contain credentials, so retain only its class.
            return {"status": "probe_failure", "reason": type(error).__name__}

    urls = list(
        dict.fromkeys(
            row["url"] for row in rows if row["status"] == "not_checked" and row.get("url")
        )
    )
    selected = urls if limit is None else urls[:limit]
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(safe_probe, url): url for url in selected}
        for completed, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if progress and (completed % 25 == 0 or completed == len(selected)):
                progress({"stage": "links", "completed": completed, "total": len(selected)})
    # Metadata lookups are cached per repository for this run, not per file.
    metadata = {}
    metadata_total = len(
        {
            (row.get("provider"), row["repository"])
            for row in rows
            if compare_metadata
            and row.get("repository")
            and results.get(row.get("url"), {}).get("status") == "reachable"
        }
    )
    for row in rows:
        if row["status"] != "not_checked":
            continue
        result = results.get(row.get("url"))
        if result is None:
            row.update(status="not_checked", reason="Outside explicit live request limit.")
            continue
        row.update(result)
        row["availability_scope"] = (
            "endpoint only; no resolved artifact guarantee"
            if row["kind"] == "endpoint"
            else "HTTP headers only; no checksum, semantic, or runtime validation"
        )
        normalize_endpoint_result(row)
        if not compare_metadata or row["status"] != "reachable" or not row.get("repository"):
            continue
        provider, repo = row["provider"], row["repository"]
        cache_key = (provider, repo)
        if cache_key not in metadata:
            endpoint = (
                f"https://huggingface.co/api/models/{repo}/revision/main"
                if provider == "huggingface"
                else f"https://api.github.com/repos/{repo}/releases/latest"
            )
            metadata[cache_key] = safe_probe(endpoint, metadata=True)
            if progress and (len(metadata) % 25 == 0 or len(metadata) == metadata_total):
                progress({"stage": "metadata", "completed": len(metadata), "total": metadata_total})
        current = metadata[cache_key]
        row["upstream_metadata"] = {key: value for key, value in current.items() if key != "data"}
        data = current.get("data", {})
        if not isinstance(data, dict):
            row["upstream_metadata"].update(
                status="invalid_metadata", reason="Expected a metadata object."
            )
            continue
        revision = row.get("revision", "")
        candidate = data.get("sha") if provider == "huggingface" else data.get("tag_name")
        pinned = (
            bool(re.fullmatch(r"[0-9a-f]{40}", revision))
            if provider == "huggingface"
            else bool(row.get("version"))
        )
        if (
            pinned
            and candidate
            and candidate != (revision if provider == "huggingface" else row.get("version"))
        ):
            row["freshness"] = (
                "upstream_revision_differs"
                if provider == "huggingface"
                else "different_latest_release"
            )
            row["artifact_change"] = "unknown"
            row["update_candidate"] = {
                "revision_or_version": candidate,
                "validated": False,
                "reason": "Different upstream main/latest identity; the change may affect only documentation. Artifact payload change and replacement compatibility are unverified.",
            }
        elif provider == "huggingface" and not pinned:
            row["freshness"] = "mutable_revision"
            row["observed_main_revision"] = candidate
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Probe public headers; default is offline inventory"
    )
    parser.add_argument(
        "--compare-metadata",
        action="store_true",
        help="Compare reachable pins with current Hub main/GitHub latest metadata",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Global minimum request start interval in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum unique artifact/endpoint URLs to probe; omitted means whole inventory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.workers <= 8
        or not 0 < args.timeout <= 60
        or not 0 <= args.retries <= 2
        or not 0.05 <= args.interval <= 10
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error(
            "Bounds: workers 1..8, timeout (0,60], retries 0..2, interval .05..10, limit positive"
        )
    rows, sources = inventory()
    if args.live:
        audit(
            rows,
            Probe(args.timeout, args.retries, args.interval),
            args.workers,
            args.limit,
            args.compare_metadata,
            progress=lambda event: print(json.dumps(event), file=sys.stderr, flush=True),
        )
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "head": head,
        "audit_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mode": "live" if args.live else "offline_inventory",
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "source_sha256": sources,
        "coverage": {
            "catalogue": "all model_catalogue/*.json entries and URL fields; exact indexed-registry tuple/variant/runtime/companion annotations when available",
            "registry": "public load_registry/entries/annotations_for/profile interfaces; source references are reference_only; qualification and benchmark history excluded",
            "code": "all first-party ruinedfooocus/**/*.py and root *.py URL literals/templates, explicit package specs and supported acquisition calls",
            "requirements": "root/first-party requirements*.txt and pip/modules.txt; no transitive dependency expansion",
            "fixtures": "tools/validation_assets.json",
            "excluded": [
                "vendor and repositories",
                "weights, venvs, caches, user profiles, inputs, outputs",
                "prompt datasets and qualification/benchmark history",
                "developer tools other than validation_assets",
            ],
            "complete_artifact_resolution": False,
            "limit_counts": "--limit bounds unique source URLs; optional metadata adds at most one request group per selected provider/repository; retries and HEAD fallback are separately bounded",
        },
        "counts": dict(Counter(row["status"] for row in rows)),
        "artifact_counts": dict(
            Counter(row["status"] for row in rows if row["kind"] == "artifact")
        ),
        "endpoint_counts": dict(
            Counter(row["status"] for row in rows if row["kind"] == "endpoint")
        ),
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    report = sanitize_report_value(report)
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "items": len(rows), "counts": report["counts"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
