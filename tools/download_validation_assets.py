"""Reproduce pinned validation assets from tools/validation_assets.json.

Default: list only. Use --download explicitly, optionally --only BASENAME
(repeatable). Existing files must match size and SHA256; none are overwritten.
The manifest records public Hub metadata provenance; access controls still apply.
Use --report outputs/feature-validation-2026-10-25/validation-assets-results.json
for automatic discovery by summarize_feature_validation.py (*assets-results.json).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
RANGE_CHUNK = 64 * 1024 * 1024


def destination(asset):
    path = (ROOT / asset["destination"]).resolve()
    if not path.is_relative_to((ROOT / "models").resolve()):
        raise ValueError("Destination must remain inside ROOT/models")
    if not re.fullmatch(r"[0-9a-f]{40}", asset["revision"]) or not re.fullmatch(
        r"[0-9a-f]{64}", asset["sha256"]
    ):
        raise ValueError("Each asset needs a pinned 40-character revision and 64-character SHA256")
    if not isinstance(asset["bytes"], int) or asset["bytes"] <= 0:
        raise ValueError("Asset size must be a positive integer")
    return path


def verify(path, asset):
    if path.stat().st_size != asset["bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return digest.hexdigest()


def stage_folder(asset):
    destination(asset)
    staging = (ROOT / "models" / ".validation-downloads").resolve()
    folder = (staging / asset["sha256"]).resolve()
    if not staging.is_relative_to((ROOT / "models").resolve()) or not folder.is_relative_to(
        staging
    ):
        raise ValueError("Download staging must remain inside ROOT/models/.validation-downloads")
    return folder


def range_state(asset):
    """Validate only this immutable asset's contiguous, chunk-hashed prefix."""
    folder = stage_folder(asset)
    payload, record = folder / "range.incomplete", folder / "range.json"
    if payload.is_symlink() or record.is_symlink():
        raise ValueError("Ranged staging files must not be symbolic links")
    identity = {key: asset[key] for key in ("repo", "revision", "file", "sha256", "bytes")}
    if not record.exists():
        if payload.exists():
            raise ValueError("Unowned ranged payload has no identity record")
        return payload, record, dict(identity=identity, chunks=[]), 0
    if record.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Ranged state is oversized")
    state = json.loads(record.read_text(encoding="utf-8"))
    if state.get("identity") != identity or not isinstance(state.get("chunks"), list):
        raise ValueError("Ranged state identity mismatch")
    prefix = 0
    chunks = state["chunks"]
    if len(chunks) > (asset["bytes"] + RANGE_CHUNK - 1) // RANGE_CHUNK:
        raise ValueError("Invalid ranged chunk count")
    if not payload.exists():
        if chunks:
            raise ValueError("Recorded ranged prefix is missing")
        return payload, record, state, 0
    with payload.open("rb") as handle:
        for index, chunk in enumerate(chunks):
            expected = min(RANGE_CHUNK, asset["bytes"] - prefix)
            if (
                not isinstance(chunk, dict)
                or type(chunk.get("bytes")) is not int
                or chunk["bytes"] != expected
                or expected <= 0
            ):
                raise ValueError("Invalid ranged chunk boundary")
            data = handle.read(expected)
            if len(data) != expected or hashlib.sha256(data).hexdigest() != chunk.get("sha256"):
                raise ValueError("Ranged prefix chunk integrity mismatch")
            prefix += expected
    return payload, record, state, prefix


def write_range_state(record, state):
    record = Path(record)
    identity = state.get("identity", {})
    expected = stage_folder(dict(identity, destination="models/" + record.name))
    if record.name != "range.json" or record.parent.resolve() != expected or record.is_symlink():
        raise ValueError("Range state must remain in its identity-checked staging directory")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=expected,
            prefix="range-state-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, record)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ranged_download(asset, url):
    import requests
    from huggingface_hub.utils import build_hf_headers

    payload, record, state, prefix = range_state(asset)
    write_range_state(record, state)
    # Only our identity-checked payload can have an uncommitted tail removed.
    with payload.open("r+b" if payload.exists() else "w+b") as handle:
        handle.truncate(prefix)
        handle.seek(prefix)
        while prefix < asset["bytes"]:
            end = min(prefix + RANGE_CHUNK, asset["bytes"]) - 1
            expected = end - prefix + 1
            headers = build_hf_headers()
            headers.update({"Range": f"bytes={prefix}-{end}", "Accept-Encoding": "identity"})
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {prefix}-{end}/{asset['bytes']}"
                    or response.headers.get("Content-Encoding", "identity") != "identity"
                    or (
                        "Content-Length" in response.headers
                        and response.headers["Content-Length"] != str(expected)
                    )
                ):
                    raise ValueError("Server did not honor the exact identity-encoded range")
                data = bytearray()
                for block in response.iter_content(1024 * 1024):
                    if len(data) + len(block) > expected:
                        raise ValueError("Oversized ranged response")
                    data.extend(block)
                if len(data) != expected:
                    raise ValueError("Incomplete ranged response")
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            state["chunks"].append(dict(bytes=expected, sha256=hashlib.sha256(data).hexdigest()))
            write_range_state(record, state)
            prefix += expected
    return payload


def remaining_bytes(asset, ranged=False):
    if destination(asset).exists():
        return 0
    folder = stage_folder(asset)
    completed = folder / destination(asset).name
    if completed.exists():
        verify(completed, asset)
        return 0
    return asset["bytes"] - range_state(asset)[3] if ranged else asset["bytes"]


def acquire(asset, ranged=False):
    path = destination(asset)
    if path.exists():
        return verification_result(asset, "existing", verify(path, asset))
    folder = stage_folder(asset)
    folder.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock

    with FileLock(str(folder / "acquire.lock"), timeout=0):
        return acquire_locked(asset, path, folder, ranged)


def acquire_locked(asset, path, folder, ranged):
    # Hub 1.33 discards its own HTTP partials. Keep staging, but only --ranged
    # promises resumability; never credit arbitrary Hub incomplete files.
    temporary = folder / path.name
    url = f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/{quote(asset['file'], safe='/')}"
    if temporary.exists():
        verify(temporary, asset)
    elif ranged:
        temporary = ranged_download(asset, url)
    else:
        from ruinedfooocus.core.hub_download import download_hub_file

        download_hub_file(url, temporary)
    digest = verify(temporary, asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another downloader winning a race.
    # Failed promotion retains the verified staging payload for a later retry.
    if os.name == "nt":
        os.rename(temporary, path)  # Windows rename refuses an existing target.
    else:
        os.link(temporary, path)
        temporary.unlink()  # Remove only our payload; retain Hub cache metadata.
    if ranged:
        (folder / "range.json").unlink(missing_ok=True)
    return verification_result(asset, "downloaded", digest)


def verification_result(asset, action, digest):
    return dict(
        destination=asset["destination"],
        status="verified",
        action=action,
        bytes=asset["bytes"],
        sha256=asset["sha256"],
        actual_sha256=digest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list", action="store_true", help="List selected assets; never download")
    parser.add_argument("--only", action="append", default=[], metavar="BASENAME")
    parser.add_argument(
        "--http", action="store_true", help="Disable XET before importing Hub libraries"
    )
    parser.add_argument(
        "--ranged",
        action="store_true",
        help="Resume chunk-verified sequential HTTP ranges (recommended for large files)",
    )
    parser.add_argument("--report", type=Path, help="Write per-file results as JSON")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "tools/validation_assets.json").read_text(encoding="utf-8"))
    assets = [a for a in manifest["assets"] if not args.only or Path(a["file"]).name in args.only]
    unknown = set(args.only) - {Path(a["file"]).name for a in assets}
    if unknown:
        parser.error(f"Unknown basenames: {sorted(unknown)}")
    paths = [destination(a) for a in assets]
    if len(set(paths)) != len(paths):
        parser.error("Manifest contains duplicate destinations")
    if not args.download or args.list:
        for asset, path in zip(assets, paths):
            print(
                f"{asset['bytes']:>12}  {'present' if path.exists() else 'missing':7}  {path.relative_to(ROOT)}"
            )
        return 0
    missing_bytes = sum(remaining_bytes(a, args.ranged) for a in assets)
    if shutil.disk_usage(ROOT / "models").free < missing_bytes + 10 * 1024**3:
        parser.error("Insufficient free disk: missing asset bytes plus 10 GiB headroom required")
    if args.http:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    sys.path.insert(0, str(ROOT))

    def run(asset):
        try:
            return acquire(asset, ranged=args.ranged)
        except Exception as error:
            return dict(destination=asset["destination"], status="failed", error=str(error))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, assets))
    for result in results:
        print(
            f"{result['status']}: {result['destination']}"
            + (f" — {result['error']}" if "error" in result else "")
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"provenance": manifest["provenance"], "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    return int(any(r["status"] == "failed" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
