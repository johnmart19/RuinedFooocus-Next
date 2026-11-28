"""Run trusted comparison case JSON files sequentially in fresh subprocesses.

Each process record is correlated to its report by SHA256. A successful report
alone cannot pass: the subprocess must also exit with code zero. This runner
starts real inference when invoked; importing it or running --help does not.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_cases(cases, output_root):
    cases = [Path(path).resolve(strict=True) for path in cases]
    names = [path.stem for path in cases]
    if len(names) != len(set(names)):
        raise ValueError("Case basenames must be unique to preserve each report")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        report = output_root / (case.stem + "-comparison.json")
        process = output_root / (case.stem + "-process.json")
        # Clear stale success evidence before launching a new process.
        process.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
        started = time.monotonic()
        result = dict(case=str(case), exit_code=None, passed=False, comparison_sha256=None)
        print(f"START {case.stem}", flush=True)
        try:
            with (output_root / (case.stem + ".log")).open("w", encoding="utf-8") as log:
                run = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "tools/validate_comfy_workflows.py"),
                        str(case),
                        "--report",
                        str(report),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            result["exit_code"] = run.returncode
            if report.exists():
                content = report.read_bytes()
                result["comparison_sha256"] = hashlib.sha256(content).hexdigest()
                data = json.loads(content)
                comparison = data.get("comparison") if isinstance(data, dict) else None
                if not isinstance(comparison, dict):
                    raise ValueError("Invalid comparison report structure")
                result["passed"] = run.returncode == 0 and comparison.get("passed") is True
            else:
                result["error"] = "Subprocess did not write a comparison report"
        except (OSError, ValueError, TypeError) as error:
            result["error"] = str(error)
        finally:
            result["seconds"] = round(time.monotonic() - started, 1)
            process.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            results.append(result)
        print(
            f"END {case.stem}: passed={result['passed']} exit_code={result['exit_code']}",
            flush=True,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+", help="Trusted local comparison JSON paths")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Comparison reports, process records and logs",
    )
    args = parser.parse_args()
    results = run_cases(args.cases, args.output_root)
    return int(any(not result["passed"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
