"""CPU-only image comparison report and contact sheet; no quality claim."""

import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"read_error": str(error)}


def summarize(root):
    summary = {
        "interim": True,
        "quality_claim": False,
        "reports": {},
        "assets": {},
        "images": [],
        "warnings": [],
    }
    tiles = []
    for path in sorted(root.glob("*-comparison.json")):
        name = path.name.removesuffix("-comparison.json")
        process_path = root / (name + "-process.json")
        report, process = read(path), read(process_path)
        comparison = report.get("comparison", {}).get("passed") is True
        exit_code = process.get("exit_code")
        digest = process.get("comparison_sha256")
        correlated = (
            digest == hashlib.sha256(path.read_bytes()).hexdigest()
            if digest is not None
            else process_path.exists()
            and process_path.stat().st_mtime_ns >= path.stat().st_mtime_ns
        )
        status = (
            "failed"
            if not comparison
            else "exit unrecorded"
            if exit_code is None
            else "process record stale"
            if not correlated
            else "passed"
            if exit_code == 0
            else "failed"
        )
        summary["reports"][name] = dict(
            status=status,
            comparison_passed=comparison,
            exit_code=exit_code,
            process_record_correlated=correlated,
        )
    summary["report_counts"] = dict(Counter(r["status"] for r in summary["reports"].values()))
    for path in sorted(root.glob("*assets-results.json")):
        data = read(path)
        rows = data if isinstance(data, list) else data.get("results", [])
        verified = sum(
            bool(r.get("sha256"))
            and r.get("actual_sha256") == r["sha256"]
            and r.get("status") == "verified"
            for r in rows
        )
        summary["assets"][path.name] = dict(
            entries=len(rows),
            recorded_sha256_matches=verified,
            statuses=dict(Counter(r.get("status", "unknown") for r in rows)),
            rehashed_by_summary=False,
        )
    for path in sorted(root.glob("*.png")):
        try:
            with Image.open(path) as image:
                tiles.append((path.name, image.convert("RGB").copy()))
                summary["images"].append(path.name)
        except OSError as error:
            summary["warnings"].append(f"{path.name}: {error}")
    if tiles:
        sheet = Image.new("RGB", (960, 210 * ((len(tiles) + 2) // 3)), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, (label, image) in enumerate(tiles):
            x, y = index % 3 * 320, index // 3 * 210
            sheet.paste(ImageOps.contain(image, (312, 180)), (x + 4, y + 4))
            draw.text((x + 4, y + 186), label[:48], fill="white")
        sheet.save(root / "contact-sheet.jpg", quality=88)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("outputs/feature-validation-2026-10-25")
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.root.resolve())["report_counts"]))
