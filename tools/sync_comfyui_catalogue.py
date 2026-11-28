"""Discover and verify still-image models from ComfyUI workflow templates for model_catalogue."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Inspect installed ComfyUI workflow templates for downloadable still-image models "
            "missing from ruinedfooocus/resources/model_catalogue, with optional live HF metadata verification."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List new still-image models discovered in ComfyUI workflow templates.",
    )
    parser.add_argument(
        "--verify-live",
        action="store_true",
        help="Probe public Hugging Face HEAD/API endpoints to resolve pinned URLs, byte sizes, and SHA256 digests.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Filter to specific filenames when listing or verifying.",
    )
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=None,
        help="Optional path to ComfyUI workflow template JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the discovered/verified JSON report.",
    )
    args = parser.parse_args(argv)

    from ruinedfooocus.core.model_catalogue import (
        discover_comfyui_template_models,
        resolve_hf_lfs_metadata,
        validate_catalogue,
    )

    issues = validate_catalogue()
    if issues:
        print(f"Catalogue validation issues: {len(issues)}")
        for issue in issues[:20]:
            print("  -", issue)

    candidates = discover_comfyui_template_models(templates_root=args.templates_root)
    if args.only:
        wanted = {name.casefold() for name in args.only}
        candidates = [item for item in candidates if item["filename"].casefold() in wanted]

    results = []
    if args.verify_live:
        for item in candidates:
            entry = dict(item)
            try:
                entry["verified"] = resolve_hf_lfs_metadata(item["url"])
            except Exception as error:
                entry["error"] = str(error)
            results.append(entry)
    else:
        results = candidates

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(results)} entries to {args.output}")
    else:
        print(f"Discovered {len(results)} ComfyUI still-image models not yet in model_catalogue:")
        for item in results:
            verified = item.get("verified")
            if verified:
                print(
                    f"  [{item['category']}] {item['filename']} | "
                    f"size={verified['size']} | sha256={verified['sha256']} | {verified['pinned_url']}"
                )
            elif item.get("error"):
                print(f"  [{item['category']}] {item['filename']} | ERROR: {item['error']}")
            else:
                print(
                    f"  [{item['category']}] {item['filename']} ({item['template']}) -> {item['url']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
